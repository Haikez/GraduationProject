"""Mamba-TCN replacement for each STIDGCN temporal-convolution block."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import MambaTCNCAConfig


class CausalTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.left_padding = 2 * dilation
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        # x: [batch * nodes, time, channels]
        sequence = F.pad(x.transpose(1, 2), (self.left_padding, 0))
        sequence = self.conv(sequence).transpose(1, 2)
        return self.norm(x + self.dropout(F.gelu(sequence)))


class TorchMambaBlock(nn.Module):
    """Unfused Mamba-1 selective recurrence for short traffic windows."""

    def __init__(self, channels: int, config: MambaTCNCAConfig) -> None:
        super().__init__()
        inner = channels * config.mamba_expand
        state_dim = config.mamba_d_state
        dt_rank = math.ceil(channels / 16)
        self.in_proj = nn.Linear(channels, 2 * inner, bias=False)
        self.conv = nn.Conv1d(
            inner, inner, config.mamba_d_conv,
            padding=config.mamba_d_conv - 1, groups=inner,
        )
        self.x_proj = nn.Linear(inner, dt_rank + 2 * state_dim, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
            .expand(inner, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, channels, bias=False)
        self.state_dim = state_dim
        self.dt_rank = dt_rank
        with torch.no_grad():
            bound = dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -bound, bound)
            dt = torch.exp(torch.empty(inner).uniform_(math.log(0.001), math.log(0.1)))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, inputs: Tensor) -> Tensor:
        batch, steps, _ = inputs.shape
        x, z = self.in_proj(inputs).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :steps].transpose(1, 2))
        dt_raw, b, c = torch.split(
            self.x_proj(x), [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        a = -torch.exp(self.A_log).to(dtype=x.dtype)
        state = x.new_zeros(batch, x.shape[-1], self.state_dim)
        outputs = []
        for step in range(steps):
            delta_t = delta[:, step].unsqueeze(-1)
            state = (
                state * torch.exp(delta_t * a)
                + delta_t * x[:, step].unsqueeze(-1) * b[:, step].unsqueeze(1)
            )
            y = (state * c[:, step].unsqueeze(1)).sum(dim=-1)
            y = (y + self.D * x[:, step]) * F.silu(z[:, step])
            outputs.append(y)
        return self.out_proj(torch.stack(outputs, dim=1))


class MambaTCNTemporal(nn.Module):
    """Map STIDGCN [B,C,N,T] features to the same shape."""

    def __init__(self, channels: int, config: MambaTCNCAConfig) -> None:
        super().__init__()
        if not (config.use_tcn or config.use_mamba):
            raise ValueError("Enable at least one temporal branch")
        self.tcn = (
            nn.Sequential(*[
                CausalTCNBlock(channels, 2**index, config.dropout)
                for index in range(config.num_tcn_layers)
            ]) if config.use_tcn else None
        )
        if config.use_mamba:
            if config.mamba_backend == "torch":
                self.mamba = TorchMambaBlock(channels, config)
            elif config.mamba_backend == "official":
                try:
                    from mamba_ssm import Mamba
                except ImportError as exc:
                    raise ImportError("Install mamba-ssm for the official backend") from exc
                self.mamba = Mamba(
                    d_model=channels,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                )
            else:
                raise ValueError("mamba_backend must be 'torch' or 'official'")
            self.mamba_norm = nn.LayerNorm(channels)
        else:
            self.mamba = None
        self.gate = (
            nn.Linear(2 * channels, channels)
            if self.tcn is not None and self.mamba is not None else None
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, nodes, steps = x.shape
        sequence = x.permute(0, 2, 3, 1).reshape(batch * nodes, steps, channels)
        local = self.tcn(sequence) if self.tcn is not None else None
        long = self.mamba_norm(sequence + self.mamba(sequence)) if self.mamba is not None else None
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(torch.cat((local, long), dim=-1)))
            sequence = gate * local + (1.0 - gate) * long
        else:
            sequence = local if local is not None else long
        # Original TSConv ends with Tanh before its DGCN call.
        return torch.tanh(sequence).reshape(batch, nodes, steps, channels).permute(0, 3, 1, 2)
