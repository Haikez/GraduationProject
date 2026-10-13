"""Bidirectional node modeling with calendar conditioning."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class BidirectionalMamba(nn.Module):
    def __init__(self, backend):
        super().__init__()
        if backend == "official":
            from mamba_ssm import Mamba
        elif backend == "torch":
            from .torch_mamba import TorchMamba as Mamba
        else:
            raise ValueError("mamba_backend must be official or torch")
        self.forward_mamba = Mamba(d_model=128, d_state=32, d_conv=2, expand=1)
        self.backward_mamba = Mamba(d_model=128, d_state=32, d_conv=2, expand=1)
        self.ffn_in = nn.Conv1d(128, 128, 1)
        self.ffn_out = nn.Conv1d(128, 128, 1)
        self.mamba_norm = nn.LayerNorm(128)
        self.ffn_norm = nn.LayerNorm(128)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        forward = self.forward_mamba(features)
        backward = self.backward_mamba(features.flip(1)).flip(1)
        residual = self.mamba_norm(forward + backward)
        hidden = self.dropout(F.gelu(self.ffn_in(residual.transpose(1, 2))))
        hidden = self.dropout(self.ffn_out(hidden).transpose(1, 2))
        return self.ffn_norm(hidden + residual)


class NodeMamba(nn.Module):
    """Refine [batch, nodes, local_history] using 64 calendar features."""

    def __init__(self, history_len, num_nodes, backend, activation_checkpoint):
        super().__init__()
        self.input_proj = nn.Linear(history_len, 128)
        self.dropout = nn.Dropout(0.1)
        self.node_embedding = nn.Parameter(torch.empty(1, num_nodes, 128))
        nn.init.xavier_uniform_(self.node_embedding)
        self.encoder = BidirectionalMamba(backend)
        self.output_norm = nn.LayerNorm(128)
        self.output_proj = nn.Linear(128, history_len)
        self.gate = nn.Parameter(torch.zeros(()))
        self.time_proj = nn.Linear(192, 128)
        self.activation_checkpoint = activation_checkpoint

    def _encode(self, features, calendar):
        devices = [features.device.index] if features.is_cuda else []
        # Keep node dropout from consuming the surrounding SCI dropout stream.
        with torch.random.fork_rng(devices=devices):
            embedded = self.dropout(self.input_proj(features)) + self.node_embedding
            calendar = calendar.unsqueeze(1).expand(-1, features.shape[1], -1)
            embedded = self.time_proj(torch.cat((embedded, calendar), dim=-1))
            return self.output_proj(self.output_norm(self.encoder(embedded)))

    def forward(self, features, calendar):
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            correction = checkpoint(self._encode, features, calendar,
                                    use_reentrant=False, preserve_rng_state=True)
        else:
            correction = self._encode(features, calendar)
        return features + self.gate.tanh() * correction
