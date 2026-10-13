"""Configuration for the source DST-Mamba forecasting architecture."""

from dataclasses import dataclass

from basicts.configs import BasicTSModelConfig


@dataclass
class DSTMambaConfig(BasicTSModelConfig):
    history_seq_len: int = 96
    future_seq_len: int = 12
    num_channels: int = 170
    d_model: int = 512
    use_norm: bool = True
    emb_dropout: float = 0.1
    decom_type: str = "STD"
    std_kernel: int = 25
    rank: int = 32
    node_dim: int = 64
    e_layers: int = 2
    d_state: int = 32
    d_conv: int = 2
    expand: int = 1
    d_ff: int = 512
    ffn_dropout: float = 0.1
    ffn_activation: str = "gelu"
    ds_type: str = "avg"
    ds_layers: int = 3
    ds_window: int = 2
    trend_downsampling: str = "source"
    sci_hidden_channels: int = 8
    sci_kernel_size: int = 3
    sci_dropout: float = 0.0
    seasonal_encoder: str = "dst_mamba"
    idmamba_channels: int = 48
    idmamba_granularity: int = 288
    idmamba_dropout: float = 0.1
    stidgcn_channels: int = 48
    stidgcn_granularity: int = 288
    stidgcn_dropout: float = 0.1
    initial_tre_w: float = 0.9
    mamba_backend: str = "official"
