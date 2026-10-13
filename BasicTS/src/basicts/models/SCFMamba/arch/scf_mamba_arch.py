"""Split-stream Cross-attention Fusion Mamba."""
import torch
from torch import nn

from ..config import SCFMambaConfig
from .cross_attention import CrossSequenceAttention
from .embedding import CalendarEmbedding
from .sci import SCIBackbone


class SCFMamba(nn.Module):
    """Map normalized [batch, 96, nodes] and timestamps to [batch, horizon, nodes]."""

    def __init__(self, config: SCFMambaConfig):
        super().__init__()
        if config.input_len != 96 or config.output_len < 1 or config.num_features < 16:
            raise ValueError("SCF-Mamba requires input_len=96, positive output_len and at least 16 nodes")
        if config.mamba_backend not in ("official", "torch"):
            raise ValueError("mamba_backend must be official or torch")
        self.input_len = config.input_len
        self.num_features = config.num_features
        self.time_embedding = CalendarEmbedding()
        self.backbone = SCIBackbone(config)
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(54321)
            self.fusion = CrossSequenceAttention(config.num_features, config.input_len)
        for parameter in self.fusion.parameters():
            parameter._scf_attention = True

    def forward(self, inputs: torch.Tensor, inputs_timestamps: torch.Tensor):
        if inputs.ndim != 3 or inputs.shape[1:] != (self.input_len, self.num_features):
            raise ValueError(f"inputs must be [batch, {self.input_len}, {self.num_features}]")
        if inputs_timestamps is None or inputs_timestamps.shape != (*inputs.shape[:2], 2):
            raise ValueError("inputs_timestamps must be [batch, history, 2]")
        calendar = self.time_embedding(inputs_timestamps.to(inputs))
        return self.backbone(inputs, calendar, self.fusion)
