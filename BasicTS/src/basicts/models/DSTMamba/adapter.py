from dataclasses import dataclass
import torch
from torch import nn
from basicts.configs import BasicTSModelConfig
from .DSTMamba_Arch import DSTMamba as DSTMambaBackbone

@dataclass
class DSTMambaConfig(BasicTSModelConfig):
    history_seq_len: int = 96
    future_seq_len: int = 12
    num_channels: int = 170
    d_model: int = 512
    use_norm: bool = True
    emb_dropout: float = 0.1
    decom_type: str = 'STD'
    std_kernel: int = 25
    rank: int = 32
    node_dim: int = 64
    e_layers: int = 2
    d_state: int = 32
    d_conv: int = 2
    expand: int = 1
    d_ff: int = 512
    ffn_dropout: float = 0.1
    ffn_activation: str = 'gelu'
    ds_type: str = 'avg'
    ds_layers: int = 3
    ds_window: int = 2
    initial_tre_w: float = 0.9
    mamba_backend: str = 'torch'

class DSTMamba(nn.Module):
    def __init__(self, config: DSTMambaConfig):
        super().__init__()
        self.backbone = DSTMambaBackbone(**dict(config))

    def forward(self, inputs: torch.Tensor, inputs_timestamps=None) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError('Expected BasicTS inputs shaped [batch, history, nodes]')
        return self.backbone(inputs.unsqueeze(-1)).squeeze(-1)
