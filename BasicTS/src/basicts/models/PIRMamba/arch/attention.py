"""Causal pattern retrieval and per-node temporal cross-attention."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class PatternRetrieval(nn.Module):
    def __init__(self, pattern_path, activation_checkpoint):
        super().__init__()
        patterns = np.load(pattern_path)
        if patterns.shape != (16, 3, 1):
            raise ValueError("Expected the frozen pattern library [16,3,1]")
        self.window_projection = nn.Linear(3, 32)
        self.query_projection = nn.Linear(32, 32)
        self.key_projection = nn.Linear(32, 32)
        self.value_projection = nn.Linear(32, 32)
        self.register_buffer("patterns", torch.as_tensor(patterns[:, :, 0].copy(), dtype=torch.float32))
        self.activation_checkpoint = activation_checkpoint

    def _retrieve(self, windows):
        query = self.query_projection(self.window_projection(windows))
        memory = self.window_projection(self.patterns)
        scores = query @ self.key_projection(memory).transpose(-2, -1) * (32 ** -.5)
        return scores.softmax(-1) @ self.value_projection(memory)

    def forward(self, history):
        # [B,T,N] -> [B,N,T,3], each short window ends at its historical position.
        windows = F.pad(history.transpose(1, 2), (2, 0), mode="replicate").unfold(-1, 3, 1)
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(self._retrieve, windows, use_reentrant=False, preserve_rng_state=True)
        return self._retrieve(windows)


class TemporalCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_projection = nn.Linear(1, 32)
        self.history_projection = nn.Linear(1, 32)
        self.query_position = nn.Parameter(torch.empty(config.output_len, 32))
        self.history_position = nn.Parameter(torch.empty(config.input_len, 32))
        nn.init.xavier_uniform_(self.query_position)
        nn.init.xavier_uniform_(self.history_position)
        self.query_norm = nn.LayerNorm(32)
        self.history_norm = nn.LayerNorm(32)
        self.attention = nn.MultiheadAttention(32, 4, dropout=.1, batch_first=True)
        self.output_projection = nn.Linear(32, 1)
        self.gate = nn.Parameter(torch.zeros(()))
        self.activation_checkpoint = config.activation_checkpoint

    def _correct(self, history, forecast, patterns):
        batch_size, input_len, num_nodes = history.shape
        output_len = forecast.shape[1]
        devices = [history.device.index] if history.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            query = forecast.transpose(1, 2).reshape(batch_size * num_nodes, output_len, 1)
            memory = history.transpose(1, 2).reshape(batch_size * num_nodes, input_len, 1)
            query = self.query_norm(self.query_projection(query) + self.query_position)
            key = self.history_norm(self.history_projection(memory) + self.history_position)
            value = self.history_norm(patterns.reshape(batch_size * num_nodes, input_len, 32)
                                      + self.history_position)
            attended = self.attention(query, key, value, need_weights=False)[0]
            return self.output_projection(attended).reshape(batch_size, num_nodes, output_len).transpose(1, 2)

    def forward(self, history, forecast, patterns):
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            correction = checkpoint(self._correct, history, forecast, patterns,
                                    use_reentrant=False, preserve_rng_state=True)
        else:
            correction = self._correct(history, forecast, patterns)
        return forecast + torch.tanh(self.gate) * correction

