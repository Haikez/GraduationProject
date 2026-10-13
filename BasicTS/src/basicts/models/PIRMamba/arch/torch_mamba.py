"""Explicit CPU checking backend; conv1d keys match official Mamba-1 weights."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TorchMamba(nn.Module):
    """Pure PyTorch selective recurrence with the Mamba-1 block interface."""

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        hidden_dim = d_model * expand
        dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.conv1d = nn.Conv1d(hidden_dim, hidden_dim, d_conv, padding=d_conv - 1,
                              groups=hidden_dim)
        self.x_proj = nn.Linear(hidden_dim, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, hidden_dim)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .expand(hidden_dim, -1).clone()
        )
        self.D = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.d_state = d_state
        self.dt_rank = dt_rank
        with torch.no_grad():
            bound = dt_rank ** -0.5
            nn.init.uniform_(self.dt_proj.weight, -bound, bound)
            dt = torch.exp(torch.empty(hidden_dim).uniform_(math.log(0.001), math.log(0.1)))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_len, _ = inputs.shape
        signal, gate = self.in_proj(inputs).chunk(2, dim=-1)
        signal = F.silu(self.conv1d(signal.transpose(1, 2))[..., :sequence_len].transpose(1, 2))
        time_features, state_input, state_output = torch.split(
            self.x_proj(signal), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        time_steps = F.softplus(self.dt_proj(time_features))
        transition = -torch.exp(self.A_log).to(dtype=signal.dtype)
        state = signal.new_zeros(batch_size, signal.shape[-1], self.d_state)
        outputs = []
        for step in range(sequence_len):
            time_step = time_steps[:, step].unsqueeze(-1)
            state = (
                state * torch.exp(time_step * transition)
                + time_step * signal[:, step].unsqueeze(-1) * state_input[:, step].unsqueeze(1)
            )
            output = (state * state_output[:, step].unsqueeze(1)).sum(dim=-1)
            output = (output + self.D * signal[:, step]) * F.silu(gate[:, step])
            outputs.append(output)
        return self.out_proj(torch.stack(outputs, dim=1))
