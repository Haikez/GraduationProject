from torch import nn

from .architecture import MambaTCNCADG as OriginalMambaTCNCADG


class MambaTCNCADG(nn.Module):
    """Adapt the local Mamba-TCN-CA-DG prototype to BasicTS 1.x."""

    def __init__(self, config):
        super().__init__()
        self.backbone = OriginalMambaTCNCADG(**dict(config))

    def forward(self, inputs):
        return self.backbone(inputs.unsqueeze(-1)).squeeze(-1)
