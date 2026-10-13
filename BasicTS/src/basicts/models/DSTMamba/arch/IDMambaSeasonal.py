"""STIDGCN seasonal pipeline with DGCN replaced by spatial Mamba."""

import torch
from torch import nn
from torch.nn import functional as F


def _temporal_filter(channels: int, dropout: float) -> nn.Sequential:
    # STIDGCN IDGCN's temporal filter: replicated padding, 5- and 3-tap convs.
    return nn.Sequential(
        nn.ReplicationPad2d((3, 3, 0, 0)),
        nn.Conv2d(channels, channels, kernel_size=(1, 5)),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Dropout(dropout),
        nn.Conv2d(channels, channels, kernel_size=(1, 3)),
        nn.Tanh(),
    )


class TemporalEmbedding(nn.Module):
    """The time-of-day and day-of-week embeddings used by STIDGCN."""

    def __init__(self, granularity: int, channels: int):
        super().__init__()
        if granularity < 1:
            raise ValueError("granularity must be positive")
        self.granularity = granularity
        self.time_day = nn.Parameter(torch.empty(granularity, channels))
        self.time_week = nn.Parameter(torch.empty(7, channels))
        nn.init.xavier_uniform_(self.time_day)
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, timestamps: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if timestamps is None or timestamps.ndim != 3 or timestamps.size(-1) < 2:
            raise ValueError("ID-Mamba needs [batch, history, 2] timestamps")
        day_index = (timestamps[..., 0] * self.granularity).long()
        day_index = day_index.clamp(0, self.granularity - 1)
        week_index = (timestamps[..., 1] * 7).round().long().clamp(0, 6)
        values = self.time_day[day_index] + self.time_week[week_index]
        return values.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, num_nodes, -1)


class SpatialBiMamba(nn.Module):
    """A BCHT-to-BCHT replacement for STIDGCN's shared DGCN module."""

    def __init__(self, channels: int, num_nodes: int, mamba_cls,
                 d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.node_embedding = nn.Parameter(torch.empty(1, num_nodes, channels))
        nn.init.xavier_uniform_(self.node_embedding)
        self.forward_ssm = mamba_cls(channels, d_state, d_conv, expand)
        self.backward_ssm = mamba_cls(channels, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout)
        # Preserve the DGCN position's residual path at initialization.
        for ssm in (self.forward_ssm, self.backward_ssm):
            nn.init.zeros_(ssm.out_proj.weight)
            if ssm.out_proj.bias is not None:
                nn.init.zeros_(ssm.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, nodes, steps = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(batch * steps, nodes, channels)
        normed = self.norm(seq + self.node_embedding)
        mixed = self.forward_ssm(normed)
        mixed = mixed + self.backward_ssm(normed.flip(1)).flip(1)
        seq = seq + self.dropout(mixed)
        return seq.reshape(batch, steps, nodes, channels).permute(0, 3, 2, 1)


class IDMambaInteraction(nn.Module):
    """STIDGCN IDGCN interaction equations with Mamba at each DGCN call."""

    def __init__(self, channels: int, num_nodes: int, mamba_cls,
                 d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.conv1 = _temporal_filter(channels, dropout)
        self.conv2 = _temporal_filter(channels, dropout)
        self.conv3 = _temporal_filter(channels, dropout)
        self.conv4 = _temporal_filter(channels, dropout)
        # STIDGCN shares one DGCN across the four interaction calls.
        self.spatial = SpatialBiMamba(
            channels, num_nodes, mamba_cls, d_state, d_conv, expand, dropout
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        even, odd = x[..., ::2], x[..., 1::2]
        d = odd * torch.tanh(self.spatial(self.conv1(even)))
        c = even * torch.tanh(self.spatial(self.conv2(odd)))
        odd_updated = d + self.spatial(self.conv3(c))
        even_updated = c + self.spatial(self.conv4(d))
        return even_updated, odd_updated


class IDMambaTree(nn.Module):
    """The complete three-block, two-level STIDGCN interaction tree."""

    def __init__(self, channels: int, num_nodes: int, mamba_cls,
                 d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        def level():
            return IDMambaInteraction(
                channels, num_nodes, mamba_cls, d_state, d_conv, expand, dropout
            )
        self.root = level()
        self.even_branch = level()
        self.odd_branch = level()

    @staticmethod
    def interleave(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
        return torch.stack((even, odd), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        root_even, root_odd = self.root(x)
        even_even, even_odd = self.even_branch(root_even)
        odd_even, odd_odd = self.odd_branch(root_odd)
        reordered_even = self.interleave(even_even, even_odd)
        reordered_odd = self.interleave(odd_even, odd_odd)
        return self.interleave(reordered_even, reordered_odd) + x


class SeasonalGLU(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.output = nn.Conv2d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.value(x) * torch.sigmoid(self.gate(x))
        return self.output(self.dropout(value))


class IDMambaSeasonal(nn.Module):
    """STIDGCN's encoder/tree/decoder on the decomposed seasonal residual."""

    def __init__(self, history_len: int, horizon: int, num_nodes: int,
                 channels: int, granularity: int, mamba_cls, d_state: int,
                 d_conv: int, expand: int, dropout: float):
        super().__init__()
        if history_len < 4 or history_len % 4:
            raise ValueError("the two-level ID-Mamba tree needs history length divisible by 4")
        if channels < 1 or num_nodes < 1:
            raise ValueError("ID-Mamba channels and nodes must be positive")
        self.history_len = history_len
        self.num_nodes = num_nodes
        self.start_conv = nn.Conv2d(3, channels, kernel_size=1)
        self.time_embedding = TemporalEmbedding(granularity, channels)
        hidden = channels * 2
        self.tree = IDMambaTree(
            hidden, num_nodes, mamba_cls, d_state, d_conv, expand, dropout
        )
        self.glu = SeasonalGLU(hidden, dropout)
        self.regression = nn.Conv2d(
            hidden, horizon, kernel_size=(1, history_len)
        )

    def forward(self, seasonal: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
        if seasonal.ndim != 3 or seasonal.shape[1:] != (self.history_len, self.num_nodes):
            raise ValueError(
                f"seasonal must have shape [batch, {self.history_len}, {self.num_nodes}]"
            )
        if (timestamps is None or timestamps.ndim != 3
                or timestamps.shape[:2] != seasonal.shape[:2]
                or timestamps.size(-1) < 2):
            raise ValueError("ID-Mamba needs matching time-of-day and weekday timestamps")
        # STIDGCN feeds value, time of day and weekday to start_conv.
        # BasicTS timestamps store weekday as a fraction of seven.
        weekday = (timestamps[..., 1] * 7).round()
        time_of_day = timestamps[..., 0].unsqueeze(-1).expand_as(seasonal)
        weekday = weekday.unsqueeze(-1).expand_as(seasonal)
        features = torch.stack((seasonal, time_of_day, weekday), dim=1)
        flow = self.start_conv(features.permute(0, 1, 3, 2))
        time = self.time_embedding(timestamps, self.num_nodes)
        x = torch.cat((flow, time), dim=1)
        x = self.tree(x)
        x = F.relu(self.glu(x) + x)
        return self.regression(x).squeeze(-1)
