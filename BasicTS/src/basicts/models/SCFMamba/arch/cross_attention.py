"""Shared bidirectional attention before each SCI stream merge."""
from contextlib import contextmanager
import math

import torch
from torch import nn


class CrossSequenceAttention(nn.Module):
    def __init__(self, num_nodes, history_len, seed=54321):
        super().__init__()
        width = 64
        self.input_proj = nn.Linear(num_nodes, width)
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0.1, batch_first=True)
        self.output_proj = nn.Linear(width, num_nodes)
        self.even_gate = nn.Parameter(torch.zeros(()))
        self.odd_gate = nn.Parameter(torch.zeros(()))
        positions = torch.arange(history_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        encoding = torch.zeros(history_len, width)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("position_encoding", encoding.unsqueeze(0))
        self.seed = seed
        self._dropout_states = {}

    @contextmanager
    def _dropout_rng(self, device):
        devices = [device.index] if device.type == "cuda" else []
        key = str(device)
        with torch.random.fork_rng(devices=devices):
            state = self._dropout_states.get(key)
            if state is None:
                state = torch.Generator(device=device).manual_seed(self.seed).get_state()
            if devices:
                torch.cuda.set_rng_state(state, device)
            else:
                torch.set_rng_state(state)
            try:
                yield
            finally:
                self._dropout_states[key] = (torch.cuda.get_rng_state(device)
                                            if devices else torch.get_rng_state()).clone()

    def get_extra_state(self):
        """Include the independent attention dropout stream in native checkpoints."""
        return {"seed": self.seed,
                "dropout_states": {key: state.cpu() for key, state in self._dropout_states.items()}}

    def set_extra_state(self, state):
        self.seed = state["seed"]
        self._dropout_states = {key: value.cpu() for key, value in state["dropout_states"].items()}

    def _embed(self, stream, positions):
        return self.norm(self.input_proj(stream) + self.position_encoding[:, list(positions)])

    def forward(self, even, odd, even_positions, odd_positions):
        with self._dropout_rng(even.device):
            even_features = self._embed(even, even_positions)
            odd_features = self._embed(odd, odd_positions)
            even_update = self.output_proj(self.attention(
                even_features, odd_features, odd_features, need_weights=False)[0])
            odd_update = self.output_proj(self.attention(
                odd_features, even_features, even_features, need_weights=False)[0])
        return (even + self.even_gate.tanh() * even_update,
                odd + self.odd_gate.tanh() * odd_update)
