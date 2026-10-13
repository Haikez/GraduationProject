"""PIR-Mamba: periodic identity and pattern retrieval enhanced SCI/Mamba."""
from pathlib import Path

import torch
from torch import nn

from .attention import PatternRetrieval, TemporalCrossAttention
from .embedding import PeriodicEmbedding
from .sci import SCIBackbone


class PIRMamba(nn.Module):
    """Normalized [B,96,170] flow + [B,96,2] calendar -> [B,12,170] flow."""
    def __init__(self, config):
        super().__init__()
        if (config.input_len, config.output_len, config.num_nodes) != (96, 12, 170):
            raise ValueError("This trained architecture requires input96/output12/nodes170")
        if config.mamba_backend not in ("official", "torch"):
            raise ValueError("mamba_backend must be official or torch")
        self.input_len = config.input_len
        self.num_nodes = config.num_nodes
        self.time_embedding = PeriodicEmbedding()
        self.backbone = SCIBackbone(config)
        pattern_path = config.pattern_path or Path(__file__).resolve().parents[1] / "assets/pattern_keys.npy"
        self.pattern_retrieval = PatternRetrieval(pattern_path, config.activation_checkpoint)
        self.cross_attention = TemporalCrossAttention(config)

    def forward(self, inputs, inputs_timestamps):
        if inputs.ndim != 3 or inputs.shape[1:] != (self.input_len, self.num_nodes):
            raise ValueError("inputs must be [B,96,170]")
        if inputs_timestamps is None or inputs_timestamps.shape != (*inputs.shape[:2], 2):
            raise ValueError("inputs_timestamps must be [B,96,2]")
        calendar = self.time_embedding(inputs_timestamps.to(inputs))
        patterns = self.pattern_retrieval(inputs)
        forecast = self.backbone(inputs, calendar)
        return self.cross_attention(inputs, forecast, patterns)

