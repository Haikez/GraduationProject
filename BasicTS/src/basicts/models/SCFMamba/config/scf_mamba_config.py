"""SCF-Mamba forecasting settings."""
from dataclasses import dataclass, field

from basicts.configs import BasicTSModelConfig


@dataclass
class SCFMambaConfig(BasicTSModelConfig):
    input_len: int = field(default=96, metadata={"help": "History length; this architecture uses 96."})
    output_len: int = field(default=12, metadata={"help": "Number of forecast steps."})
    num_features: int = field(default=170, metadata={"help": "Number of traffic nodes."})
    mamba_backend: str = field(
        default="official", metadata={"help": "official for CUDA; torch for CPU checks."})
    activation_checkpoint: bool = field(
        default=True, metadata={"help": "Recompute node Mamba activations during training."})
