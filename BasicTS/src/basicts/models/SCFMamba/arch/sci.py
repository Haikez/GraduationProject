"""Two-level SCI interaction tree, stream fusion and forecast projection."""
import math

import torch
from torch import nn

from .node_mamba import NodeMamba


class SCIBlock(nn.Module):
    def __init__(self, history_len, num_nodes, backend, activation_checkpoint):
        super().__init__()
        hidden_size = int(num_nodes * 0.0625)

        def convolution():
            return nn.Sequential(
                nn.ReplicationPad1d((3, 3)),
                nn.Conv1d(num_nodes, hidden_size, 5),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(0.5),
                nn.Conv1d(hidden_size, num_nodes, 3),
                nn.Tanh(),
            )

        # Keep module construction order consistent with the trained B model.
        self.scale_odd = convolution()
        self.scale_even = convolution()
        self.update_odd = convolution()
        self.update_even = convolution()
        self.node_mamba = NodeMamba(history_len // 2, num_nodes, backend,
                                    activation_checkpoint)

    def forward(self, history, calendar):
        even = history[:, ::2].transpose(1, 2)
        odd = history[:, 1::2].transpose(1, 2)
        odd_scaled = odd * torch.exp(self.node_mamba(self.scale_odd(even), calendar))
        even_scaled = even * torch.exp(self.node_mamba(self.scale_even(odd), calendar))
        even_update = self.node_mamba(self.update_even(odd_scaled), calendar)
        odd_update = self.node_mamba(self.update_odd(even_scaled), calendar)
        return ((even_scaled + even_update).transpose(1, 2),
                (odd_scaled - odd_update).transpose(1, 2))


class SCITree(nn.Module):
    def __init__(self, history_len, num_nodes, depth, backend, activation_checkpoint):
        super().__init__()
        self.interaction = SCIBlock(history_len, num_nodes, backend, activation_checkpoint)
        self.depth = depth
        if depth:
            self.odd = SCITree(history_len // 2, num_nodes, depth - 1,
                               backend, activation_checkpoint)
            self.even = SCITree(history_len // 2, num_nodes, depth - 1,
                                backend, activation_checkpoint)

    def forward(self, history, calendar, fusion, positions):
        even, odd = self.interaction(history, calendar)
        even_positions, odd_positions = positions[::2], positions[1::2]
        if self.depth:
            even = self.even(even, calendar, fusion, even_positions)
            odd = self.odd(odd, calendar, fusion, odd_positions)
        even, odd = fusion(even, odd, even_positions, odd_positions)
        return torch.stack((even, odd), dim=2).flatten(1, 2)


class SCIBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_nodes = config.num_features
        self.tree = SCITree(config.input_len, self.num_nodes, 1,
                            config.mamba_backend, config.activation_checkpoint)
        self.head = nn.Conv1d(config.input_len, config.output_len, 1, bias=False)
        scales = (self.num_nodes + 1) // 2
        frequencies = torch.exp(torch.arange(scales, dtype=torch.float32)
                                * -(math.log(10000.0) / max(scales - 1, 1)))
        self.register_buffer("position_frequencies", frequencies)

    def forward(self, history, calendar, fusion):
        positions = torch.arange(history.shape[1], dtype=torch.float32, device=history.device)
        angles = positions.unsqueeze(1) * self.position_frequencies.unsqueeze(0)
        encoding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
        encoded = history.clone()
        encoded += encoding[:, :self.num_nodes].unsqueeze(0)
        features = self.tree(encoded, calendar, fusion, tuple(range(history.shape[1])))
        return self.head(features + encoded)
