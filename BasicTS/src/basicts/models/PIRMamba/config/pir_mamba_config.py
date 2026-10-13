"""PIR-Mamba settings following BasicTS model-config conventions."""
from dataclasses import dataclass
from typing import Optional

from basicts.configs import BasicTSModelConfig


@dataclass
class PIRMambaConfig(BasicTSModelConfig):
    input_len: int = 96
    output_len: int = 12
    num_nodes: int = 170
    mamba_backend: str = "official"
    pattern_path: Optional[str] = None
    activation_checkpoint: bool = True

