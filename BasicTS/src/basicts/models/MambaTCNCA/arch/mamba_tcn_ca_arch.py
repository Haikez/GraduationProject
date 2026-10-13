"""STIDGCN tree with TSConv and Hadamard interaction replaced.

The temporal embedding, DGCN graph generator, three-stage interaction tree,
GLU and regression head follow the local STIDGCN implementation. Each of its
four temporal convolutions per tree node becomes a Mamba-TCN block. The two
odd/even Hadamard gates become cross-attention with residual query features.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from basicts.models.STIDGCN.architecture import DGCN, GLU, TemporalEmbedding

from ..config import MambaTCNCAConfig
from .temporal import MambaTCNTemporal


class CrossSequenceAttention(nn.Module):
    """Attend over the other sub-sequence's time axis for each sensor."""

    def __init__(self, channels: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, query: Tensor, source: Tensor) -> Tensor:
        batch, channels, nodes, query_steps = query.shape
        source_steps = source.shape[-1]
        q = query.permute(0, 2, 3, 1).reshape(batch * nodes, query_steps, channels)
        kv = source.permute(0, 2, 3, 1).reshape(batch * nodes, source_steps, channels)
        message = self.attention(q, kv, kv, need_weights=False)[0]
        updated = q + message
        return updated.reshape(batch, nodes, query_steps, channels).permute(0, 3, 1, 2)


class MambaTCNIDGCN(nn.Module):
    """One original IDGCN node with only its two proposed replacements."""

    def __init__(self, config: MambaTCNCAConfig, memory: nn.Parameter) -> None:
        super().__init__()
        channels = 2 * config.channels
        self.temporal1 = MambaTCNTemporal(channels, config)
        self.temporal2 = MambaTCNTemporal(channels, config)
        self.temporal3 = MambaTCNTemporal(channels, config)
        self.temporal4 = MambaTCNTemporal(channels, config)
        self.dgcn = DGCN(
            channels=channels,
            num_nodes=config.num_nodes,
            diffusion_step=config.diffusion_step,
            dropout=config.dropout,
            emb=memory,
        )
        self.cross_odd = (
            CrossSequenceAttention(channels, config.num_heads, config.dropout)
            if config.use_ca else None
        )
        self.cross_even = (
            CrossSequenceAttention(channels, config.num_heads, config.dropout)
            if config.use_ca else None
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        even, odd = x[..., ::2], x[..., 1::2]
        x1 = self.dgcn(self.temporal1(even))
        d = (
            self.cross_odd(odd, torch.tanh(x1))
            if self.cross_odd is not None else odd * torch.tanh(x1)
        )

        x2 = self.dgcn(self.temporal2(odd))
        c = (
            self.cross_even(even, torch.tanh(x2))
            if self.cross_even is not None else even * torch.tanh(x2)
        )

        x3 = self.dgcn(self.temporal3(c))
        odd_updated = d + x3
        x4 = self.dgcn(self.temporal4(d))
        even_updated = c + x4
        return even_updated, odd_updated


class MambaTCNIDGCNTree(nn.Module):
    """The original 12→6→3 three-node STIDGCN tree and realignment."""

    def __init__(self, config: MambaTCNCAConfig) -> None:
        super().__init__()
        channels = 2 * config.channels
        self.memory1 = nn.Parameter(torch.randn(channels, config.num_nodes, 6))
        self.memory2 = nn.Parameter(torch.randn(channels, config.num_nodes, 3))
        self.memory3 = nn.Parameter(torch.randn(channels, config.num_nodes, 3))
        self.stage1 = MambaTCNIDGCN(config, self.memory1)
        self.stage2 = MambaTCNIDGCN(config, self.memory2)
        # STIDGCN uses memory2 in both child nodes; memory3 is retained for
        # source-topology compatibility even though it is not consumed.
        self.stage3 = MambaTCNIDGCN(config, self.memory2)

    @staticmethod
    def interleave(even: Tensor, odd: Tensor) -> Tensor:
        if even.shape != odd.shape:
            raise ValueError("STIDGCN tree requires equal even/odd lengths")
        return torch.stack((even, odd), dim=-1).flatten(-2)

    def forward(self, x: Tensor) -> Tensor:
        even1, odd1 = self.stage1(x)
        even2, odd2 = self.stage2(even1)
        even3, odd3 = self.stage3(odd1)
        left = self.interleave(even2, odd2)
        right = self.interleave(even3, odd3)
        return self.interleave(left, right) + x


class MambaTCNCADG(nn.Module):
    """BasicTS wrapper retaining the STIDGCN encoder and regression head."""

    def __init__(self, config: MambaTCNCAConfig) -> None:
        super().__init__()
        if config.input_len != 12:
            raise ValueError("The STIDGCN tree requires input_len=12")
        if config.num_nodes < 2 or config.channels < 1:
            raise ValueError("num_nodes must be >= 2 and channels must be positive")
        if config.use_ca and (
            config.num_heads < 1 or (2 * config.channels) % config.num_heads
        ):
            raise ValueError("2 * channels must be divisible by num_heads")
        if config.num_tcn_layers < 1 and config.use_tcn:
            raise ValueError("num_tcn_layers must be positive when TCN is enabled")
        self.num_nodes = config.num_nodes
        self.input_len = config.input_len
        self.temporal_embedding = TemporalEmbedding(config.granularity, config.channels)
        self.start_conv = nn.Conv2d(3, config.channels, kernel_size=(1, 1))
        self.tree = MambaTCNIDGCNTree(config)
        self.glu = GLU(2 * config.channels, config.dropout)
        self.regression_layer = nn.Conv2d(
            2 * config.channels, config.output_len, kernel_size=(1, config.input_len)
        )

    def forward(self, inputs: Tensor, inputs_timestamps: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (self.input_len, self.num_nodes):
            raise ValueError("Expected inputs [batch, 12, num_nodes]")
        if (
            inputs_timestamps is None or inputs_timestamps.ndim != 3
            or inputs_timestamps.shape[:2] != inputs.shape[:2]
            or inputs_timestamps.shape[-1] < 2
        ):
            raise ValueError("Expected time-of-day and day-of-week timestamps")

        times = inputs_timestamps[..., :2].unsqueeze(2).expand(
            -1, -1, self.num_nodes, -1
        )
        history = torch.cat((inputs.unsqueeze(-1), times), dim=-1)
        start_input = history.permute(0, 3, 2, 1).clone()
        start_input[:, 2] = (start_input[:, 2] * 7).round()
        embedding = self.temporal_embedding(history)
        x = torch.cat((self.start_conv(start_input), embedding), dim=1)
        x = self.tree(x)
        x = self.glu(x) + x
        return self.regression_layer(F.relu(x)).squeeze(-1)
