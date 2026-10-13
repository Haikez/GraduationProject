"""Unfused Mamba-1 block for running DST-Mamba without CUDA extensions."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TorchMamba(nn.Module):
    """Pure PyTorch selective recurrence with the Mamba-1 block interface."""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        inner = d_model * expand
        dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv = nn.Conv1d(inner, inner, d_conv, padding=d_conv - 1,
                              groups=inner)
        self.x_proj = nn.Linear(inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .expand(inner, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.d_state = d_state
        self.dt_rank = dt_rank
        with torch.no_grad():
            bound = dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -bound, bound)
            dt = torch.exp(torch.empty(inner).uniform_(math.log(0.001), math.log(0.1)))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = inputs.shape
        x, z = self.in_proj(inputs).chunk(2, dim=-1)
        x = F.silu(self.conv(x.transpose(1, 2))[..., :steps].transpose(1, 2))
        dt_raw, b, c = torch.split(
            self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt_raw))
        a = -torch.exp(self.A_log).to(dtype=x.dtype)
        state = x.new_zeros(batch, x.shape[-1], self.d_state)
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
