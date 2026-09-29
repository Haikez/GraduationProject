"""Minimal Mamba-TCN-CA-DG prototype for traffic-flow forecasting.

The module intentionally uses only PyTorch.  Its selective state-space block
is Mamba-inspired, but it is *not* the official ``mamba_ssm.Mamba`` operator.
This makes the first experiment reproducible on the existing Windows setup;
results from this prototype must not be presented as an official Mamba result.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class CausalTemporalBlock(nn.Module):
    """A residual causal dilated-convolution block for local temporal trends."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = (3 - 1) * dilation
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, dilation=dilation, padding=self.padding
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply causal convolution to ``[batch, time, node, hidden]`` features."""
        batch_size, steps, num_nodes, hidden_dim = x.shape
        sequence = x.permute(0, 2, 3, 1).reshape(batch_size * num_nodes, hidden_dim, steps)
        convolved = self.conv(sequence)[..., :steps]
        convolved = F.gelu(convolved).transpose(1, 2)
        convolved = self.dropout(convolved).reshape(batch_size, num_nodes, steps, hidden_dim)
        convolved = convolved.permute(0, 2, 1, 3)
        return self.norm(x + convolved)


class SelectiveStateSpace(nn.Module):
    """A lightweight input-selective diagonal state-space layer.

    The input controls both the time step and the injected state.  The state
    update is linear in sequence length and gives the MVP a dependency path
    complementary to the local TCN branch.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.delta_proj = nn.Linear(hidden_dim, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(hidden_dim, hidden_dim)
        self.log_a = nn.Parameter(torch.zeros(hidden_dim))
        self.skip = nn.Parameter(torch.ones(hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Run the selective recurrence over the time axis."""
        batch_size, steps, num_nodes, hidden_dim = x.shape
        state = x.new_zeros(batch_size, num_nodes, hidden_dim)
        outputs = []
        # A is negative, ensuring an exponentially stable diagonal transition.
        a = -torch.exp(self.log_a).view(1, 1, hidden_dim)

        for step in range(steps):
            current = x[:, step]
            delta = F.softplus(self.delta_proj(current)) + 1e-4
            decay = torch.exp(a * delta)
            proposal = torch.tanh(self.input_proj(current))
            state = decay * state + (1.0 - decay) * proposal
            output = torch.sigmoid(self.output_gate(current)) * state + self.skip * current
            outputs.append(output)

        return self.norm(torch.stack(outputs, dim=1))


class CrossSequenceAttention(nn.Module):
    """Bidirectional attention between alternating temporal sub-sequences.

    Attention is calculated along time independently for each traffic sensor.
    This makes the CA-STI tensor semantics explicit and avoids accidentally
    treating nodes as a temporal axis.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.even_from_odd = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.odd_from_even = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.even_norm = nn.LayerNorm(hidden_dim)
        self.odd_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, num_nodes, hidden_dim = x.shape
        even = x[:, 0::2].permute(0, 2, 1, 3).reshape(batch_size * num_nodes, -1, hidden_dim)
        odd = x[:, 1::2].permute(0, 2, 1, 3).reshape(batch_size * num_nodes, -1, hidden_dim)
        if odd.shape[1] == 0:
            return x

        even_message = self.even_from_odd(even, odd, odd, need_weights=False)[0]
        odd_message = self.odd_from_even(odd, even, even, need_weights=False)[0]
        even = self.even_norm(even + self.dropout(even_message))
        odd = self.odd_norm(odd + self.dropout(odd_message))

        output = x.clone()
        output[:, 0::2] = even.reshape(batch_size, num_nodes, -1, hidden_dim).permute(0, 2, 1, 3)
        output[:, 1::2] = odd.reshape(batch_size, num_nodes, -1, hidden_dim).permute(0, 2, 1, 3)
        return output


class DynamicGraphFusion(nn.Module):
    """Fuse instance-driven and pattern-bank graph logits with a learned gate."""

    def __init__(self, num_nodes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pattern_bank = nn.Parameter(torch.empty(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.pattern_bank)
        self.gate = nn.Linear(2, 1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Use the full historical window to construct one graph per sample.
        context = x.mean(dim=1)
        query = self.query_proj(context)
        key = self.key_proj(context)
        dynamic_logits = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        pattern_logits = torch.matmul(query, self.pattern_bank.transpose(0, 1)) / self.scale

        gate_inputs = torch.stack((dynamic_logits, pattern_logits), dim=-1)
        gate = torch.sigmoid(self.gate(gate_inputs)).squeeze(-1)
        adjacency = torch.softmax(gate * dynamic_logits + (1.0 - gate) * pattern_logits, dim=-1)
        messages = torch.einsum("bij,btjd->btid", adjacency, x)
        return self.norm(x + self.dropout(self.output_proj(messages)))


class MambaTCNCADG(nn.Module):
    """Mamba-TCN temporal fusion, single-level CA-STI, and dynamic graph fusion.

    Boolean component switches are exposed so the experiment matrix can use
    the same parameter budget and training configuration for ablations.
    """

    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_tcn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_tcn: bool = True,
        use_ssm: bool = True,
        use_ca: bool = True,
        use_dynamic_graph: bool = True,
    ) -> None:
        super().__init__()
        if not use_tcn and not use_ssm:
            raise ValueError("At least one temporal branch must be enabled")
        self.input_len = input_len
        self.output_len = output_len
        self.use_tcn = use_tcn
        self.use_ssm = use_ssm
        self.use_ca = use_ca
        self.use_dynamic_graph = use_dynamic_graph

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tcn = nn.Sequential(
            *[CausalTemporalBlock(hidden_dim, 2**index, dropout) for index in range(num_tcn_layers)]
        )
        self.ssm = SelectiveStateSpace(hidden_dim)
        self.temporal_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cross_attention = CrossSequenceAttention(hidden_dim, num_heads, dropout)
        self.dynamic_graph = DynamicGraphFusion(num_nodes, hidden_dim, dropout)
        self.output_proj = nn.Linear(input_len * hidden_dim, output_len)

    def forward(
        self,
        history_data: Tensor,
        future_data: Tensor | None = None,
        batch_seen: int | None = None,
        epoch: int | None = None,
        train: bool = True,
        **kwargs,
    ) -> Tensor:
        """Forecast future traffic flow from ``[batch, input_len, node, feature]``."""
        del future_data, batch_seen, epoch, train, kwargs
        if history_data.shape[1] != self.input_len:
            raise ValueError(f"Expected input_len={self.input_len}, got {history_data.shape[1]}")

        features = self.input_proj(history_data[..., : self.input_proj.in_features])
        tcn_features = self.tcn(features) if self.use_tcn else features
        ssm_features = self.ssm(features) if self.use_ssm else features

        if self.use_tcn and self.use_ssm:
            gate = torch.sigmoid(self.temporal_gate(torch.cat((tcn_features, ssm_features), dim=-1)))
            features = gate * tcn_features + (1.0 - gate) * ssm_features
        elif self.use_tcn:
            features = tcn_features
        else:
            features = ssm_features

        if self.use_ca:
            features = self.cross_attention(features)
        if self.use_dynamic_graph:
            features = self.dynamic_graph(features)

        batch_size, _, num_nodes, hidden_dim = features.shape
        flattened = features.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.input_len * hidden_dim)
        prediction = self.output_proj(flattened).permute(0, 2, 1).unsqueeze(-1)
        return prediction
