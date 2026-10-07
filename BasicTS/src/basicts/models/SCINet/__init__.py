"""BasicTS adapter for the original SCINet implementation."""

import torch
from torch import nn

from .arch import SCINetBackbone
from .config import SCINetConfig


class SCINet(nn.Module):
    """Map BasicTS [batch, history, features] to [batch, horizon, features]."""

    def __init__(self, config: SCINetConfig):
        super().__init__()
        if config.num_stacks not in (1, 2):
            raise ValueError("num_stacks must be 1 or 2")
        if config.num_levels < 1 or config.input_len % (2 ** config.num_levels):
            raise ValueError("input_len must be divisible by 2 ** num_levels")
        if config.num_stacks == 2:
            second_len = (config.concat_len or config.input_len) + config.output_len
            if second_len % (2 ** config.num_levels):
                raise ValueError("second stack length must be divisible by 2 ** num_levels")
        if not 0 <= config.concat_len <= config.input_len:
            raise ValueError("concat_len must be between 0 and input_len")
        if config.num_features < 1 or config.hidden_size <= 0:
            raise ValueError("num_features and hidden_size must be positive")
        hidden_channels = max(1, int(config.num_features * config.hidden_size))
        if config.groups < 1 or config.num_features % config.groups or hidden_channels % config.groups:
            raise ValueError("groups must divide both feature and hidden channel counts")
        self.backbone = SCINetBackbone(
            output_len=config.output_len,
            input_len=config.input_len,
            input_dim=config.num_features,
            hid_size=config.hidden_size,
            num_stacks=config.num_stacks,
            num_levels=config.num_levels,
            num_decoder_layer=config.num_decoder_layers,
            concat_len=config.concat_len,
            groups=config.groups,
            kernel=config.kernel_size,
            dropout=config.dropout,
            positionalE=config.positional_encoding,
            modified=config.modified,
            RIN=config.rin,
        )

    def forward(self, inputs: torch.Tensor):
        output = self.backbone(inputs)
        if isinstance(output, tuple):
            prediction, intermediate = output
            return {"prediction": prediction, "intermediate": intermediate}
        return output


__all__ = ["SCINet", "SCINetConfig"]
