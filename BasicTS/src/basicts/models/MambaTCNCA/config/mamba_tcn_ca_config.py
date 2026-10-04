"""Model configuration following BasicTSModelConfig conventions."""

from dataclasses import dataclass

from basicts.configs import BasicTSModelConfig


@dataclass
class MambaTCNCAConfig(BasicTSModelConfig):
    num_nodes: int = 170
    input_len: int = 12
    output_len: int = 12
    channels: int = 32
    granularity: int = 288
    diffusion_step: int = 1
    num_tcn_layers: int = 2
    num_heads: int = 4
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_backend: str = "torch"
    dropout: float = 0.1
    use_tcn: bool = True
    use_mamba: bool = True
    use_ca: bool = True
