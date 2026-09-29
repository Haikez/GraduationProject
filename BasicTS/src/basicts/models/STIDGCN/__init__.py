import torch
from torch import nn

from .architecture import STIDGCN as OriginalSTIDGCN


class STIDGCN(nn.Module):
    """Adapt the local STIDGCN to BasicTS 1.x input and timestamp tensors."""

    def __init__(self, config):
        super().__init__()
        self.backbone = OriginalSTIDGCN(
            input_dim=3, num_nodes=config.num_nodes, channels=config.channels,
            granularity=288, input_len=12, output_len=12, dropout=config.dropout,
        )
        source_mean = getattr(config, "source_mean", None)
        source_std = getattr(config, "source_std", None)
        self.source_scale = source_mean is not None and source_std is not None
        if self.source_scale:
            self.register_buffer("source_mean", torch.tensor(float(source_mean)))
            self.register_buffer("source_std", torch.tensor(float(source_std)))

    def forward(self, inputs, inputs_timestamps):
        if inputs_timestamps is None or inputs_timestamps.shape[-1] < 2:
            raise ValueError("STIDGCN requires time-of-day and day-of-week timestamps")
        times = inputs_timestamps.unsqueeze(2).expand(-1, -1, inputs.shape[2], -1)
        history = torch.cat((inputs.unsqueeze(-1), times[..., :2]), dim=-1)
        prediction = self.backbone(history).squeeze(-1)
        if self.source_scale:
            prediction = prediction * self.source_std + self.source_mean
        return prediction
