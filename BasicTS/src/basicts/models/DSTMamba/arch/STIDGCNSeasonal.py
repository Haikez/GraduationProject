"""Full STIDGCN seasonal branch adapted to DST-Mamba history lengths."""

import torch
from torch import nn
from torch.nn import functional as F

from basicts.models.STIDGCN.architecture import GLU, IDGCN, TemporalEmbedding


class STIDGCNTree(nn.Module):
    """Source STIDGCN's two-level IDGCN tree with length-aware memories."""

    def __init__(self, history_len: int, channels: int, num_nodes: int,
                 dropout: float):
        super().__init__()
        if history_len < 4 or history_len % 4:
            raise ValueError("STIDGCN tree needs history length divisible by 4")
        self.memory1 = nn.Parameter(torch.randn(channels, num_nodes, history_len // 2))
        self.memory2 = nn.Parameter(torch.randn(channels, num_nodes, history_len // 4))
        self.IDGCN1 = IDGCN(
            device=None, channels=channels, diffusion_step=1,
            num_nodes=num_nodes, dropout=dropout, emb=self.memory1,
        )
        self.IDGCN2 = IDGCN(
            device=None, channels=channels, diffusion_step=1,
            num_nodes=num_nodes, dropout=dropout, emb=self.memory2,
        )
        # The source tree shares memory2 across both second-level blocks.
        self.IDGCN3 = IDGCN(
            device=None, channels=channels, diffusion_step=1,
            num_nodes=num_nodes, dropout=dropout, emb=self.memory2,
        )

    @staticmethod
    def interleave(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
        return torch.stack((even, odd), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        even, odd = self.IDGCN1(x)
        even_even, even_odd = self.IDGCN2(even)
        odd_even, odd_odd = self.IDGCN3(odd)
        reordered_even = self.interleave(even_even, even_odd)
        reordered_odd = self.interleave(odd_even, odd_odd)
        return self.interleave(reordered_even, reordered_odd) + x


class STIDGCNSeasonal(nn.Module):
    """STIDGCN encoder/tree/decoder on the decomposed seasonal series."""

    def __init__(self, history_len: int, horizon: int, num_nodes: int,
                 channels: int = 48, granularity: int = 288,
                 dropout: float = 0.1):
        super().__init__()
        if history_len < 4 or history_len % 4:
            raise ValueError("STIDGCN seasonal history must be divisible by 4")
        if horizon < 1 or num_nodes < 1 or channels < 1 or granularity < 1:
            raise ValueError("horizon, nodes, channels and granularity must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.history_len = history_len
        self.num_nodes = num_nodes
        self.Temb = TemporalEmbedding(granularity, channels)
        self.start_conv = nn.Conv2d(3, channels, kernel_size=1)
        hidden = channels * 2
        self.tree = STIDGCNTree(history_len, hidden, num_nodes, dropout)
        self.glu = GLU(hidden, dropout)
        self.regression_layer = nn.Conv2d(
            hidden, horizon, kernel_size=(1, history_len)
        )

    def forward(self, seasonal: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
        if (seasonal.ndim != 3
                or seasonal.shape[1:] != (self.history_len, self.num_nodes)):
            raise ValueError("seasonal series has the wrong shape")
        if (timestamps is None or timestamps.ndim != 3
                or timestamps.shape[:2] != seasonal.shape[:2]
                or timestamps.size(-1) < 2):
            raise ValueError("STIDGCN needs matching time-of-day and weekday timestamps")

        time_of_day = timestamps[..., 0].unsqueeze(-1).expand_as(seasonal)
        weekday = timestamps[..., 1].unsqueeze(-1).expand_as(seasonal)
        history = torch.stack((seasonal, time_of_day, weekday), dim=-1)
        features = history.permute(0, 3, 2, 1).clone()
        # The source start_conv receives a weekday index, while Temb receives
        # the original weekday fraction.
        features[:, 2] = (features[:, 2] * 7).round()
        x = torch.cat((self.start_conv(features), self.Temb(history)), dim=1)
        x = self.tree(x)
        x = self.glu(x) + x
        return self.regression_layer(F.relu(x)).squeeze(-1)
