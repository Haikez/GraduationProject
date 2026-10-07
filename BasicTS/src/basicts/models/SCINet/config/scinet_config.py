"""SCINet settings for BasicTS forecasting."""

from dataclasses import dataclass

from basicts.configs import BasicTSModelConfig


@dataclass
class SCINetConfig(BasicTSModelConfig):
    input_len: int = 12
    output_len: int = 12
    num_features: int = 170
    hidden_size: float = 0.0625
    num_stacks: int = 1
    num_levels: int = 2
    num_decoder_layers: int = 1
    concat_len: int = 0
    groups: int = 1
    kernel_size: int = 5
    dropout: float = 0.5
    positional_encoding: bool = True
    modified: bool = True
    rin: bool = False
