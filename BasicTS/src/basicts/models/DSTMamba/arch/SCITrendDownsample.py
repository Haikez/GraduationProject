"""SCINet-style odd/even interaction for a lossless trend-scale split."""

import torch
from torch import nn


def _temporal_filter(hidden_channels: int, kernel_size: int, dropout: float) -> nn.Sequential:
    # Match the convolution structure of SCINet's Interactor. Zeroing the
    # output convolution makes the initial interaction an identity transform.
    pad = (kernel_size - 1) // 2 + 1
    layers = nn.Sequential(
        nn.ReplicationPad1d((pad, pad)),
        nn.Conv1d(1, hidden_channels, kernel_size),
        nn.LeakyReLU(negative_slope=0.01, inplace=True),
        nn.Dropout(dropout),
        nn.Conv1d(hidden_channels, 1, 3),
        nn.Tanh(),
    )
    nn.init.zeros_(layers[4].weight)
    nn.init.zeros_(layers[4].bias)
    return layers


class SCITrendDownsample(nn.Module):
    """Return half-length coarse and detail streams after SCI interaction.

    Filters are shared across nodes. The coarse stream is fed to the next
    scale; the detail stream retains the odd/even difference for prediction.
    At initialization, coarse is ordinary two-point average pooling.
    """

    def __init__(self, hidden_channels: int = 8, kernel_size: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd number")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.phi = _temporal_filter(hidden_channels, kernel_size, dropout)
        self.psi = _temporal_filter(hidden_channels, kernel_size, dropout)
        self.P = _temporal_filter(hidden_channels, kernel_size, dropout)
        self.U = _temporal_filter(hidden_channels, kernel_size, dropout)

    def interact(self, trend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if trend.ndim != 3 or trend.size(1) < 2 or trend.size(1) % 2:
            raise ValueError("trend must have shape [batch, even history length, nodes]")

        batch, length, nodes = trend.shape
        node_series = trend.transpose(1, 2).reshape(batch * nodes, 1, length)
        even, odd = node_series[..., ::2], node_series[..., 1::2]

        d = odd * torch.exp(self.phi(even))
        c = even * torch.exp(self.psi(odd))
        updated_even = c + self.U(d)
        updated_odd = d - self.P(c)
        return updated_even, updated_odd

    def forward(self, trend: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, nodes = trend.shape
        even, odd = self.interact(trend)
        coarse = (even + odd) * 0.5
        detail = (even - odd) * 0.5
        coarse = coarse.reshape(batch, nodes, length // 2).transpose(1, 2)
        detail = detail.reshape(batch, nodes, length // 2).transpose(1, 2)
        return coarse, detail
