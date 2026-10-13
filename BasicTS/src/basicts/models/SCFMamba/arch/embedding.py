"""Calendar features from the last observed timestamp."""
import torch
from torch import nn


class CalendarEmbedding(nn.Module):
    """Embed time-of-day/288 and cyclic-day/7 as 64 calendar features."""

    def __init__(self):
        super().__init__()
        self.time_of_day = nn.Parameter(torch.empty(288, 32))
        self.day_of_cycle = nn.Parameter(torch.empty(7, 32))
        nn.init.xavier_uniform_(self.time_of_day)
        nn.init.xavier_uniform_(self.day_of_cycle)

    def forward(self, timestamps):
        indices = torch.round(timestamps[:, -1] * timestamps.new_tensor([288, 7])).long()
        valid = ((indices[:, 0] >= 0) & (indices[:, 0] < 288)
                 & (indices[:, 1] >= 0) & (indices[:, 1] < 7))
        if not bool(valid.all()):
            raise ValueError("Expected normalized time-of-day/288 and cyclic-day/7 timestamps")
        return torch.cat((self.time_of_day[indices[:, 0]],
                          self.day_of_cycle[indices[:, 1]]), dim=-1)
