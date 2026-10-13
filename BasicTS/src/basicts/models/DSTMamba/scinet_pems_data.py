"""SCINet's PEMS windowing and split-specific z-score in BasicTS."""

import numpy as np
import pandas as pd
import torch

from basicts.data import BasicTSForecastingDataset
from basicts.runners.taskflow import BasicTSForecastingTaskFlow
from basicts.utils.constants import BasicTSMode


class SCINetPEMSDataset(BasicTSForecastingDataset):
    """Use the existing chronological 6:2:2 files with SCINet's windows.

    Each split computes its own per-node mean and population standard
    deviation before missing values are forward/backward filled. Training
    and validation slide by one step; test slides by twelve steps.
    """

    def __init__(self, dataset_name: str, input_len: int, output_len: int,
                 mode: BasicTSMode, use_timestamps: bool = True,
                 local: bool = True, data_file_path: str = None,
                 memmap: bool = False):
        super().__init__(dataset_name, input_len, output_len, mode,
                         use_timestamps, local, data_file_path, memmap)
        raw = self._data
        mean = np.mean(raw, axis=0)
        std = np.std(raw, axis=0)
        stats = {"mean": mean.tolist(),
                 "std": [1 if value == 0 else value for value in std]}
        filled = pd.DataFrame(raw).ffill(limit=len(raw))
        filled = filled.bfill(limit=len(raw)).values
        self._data = ((filled - stats["mean"]) / stats["std"]).astype(np.float32)
        self.scinet_mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.scinet_std = torch.tensor(stats["std"], dtype=torch.float32)
        self.stride = 12 if mode == BasicTSMode.TEST else 1

    def __len__(self) -> int:
        count = super().__len__()
        return count // self.stride

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index * self.stride)
        item["scinet_mean"] = self.scinet_mean
        item["scinet_std"] = self.scinet_std
        return item


class SCINetPEMSTaskFlow(BasicTSForecastingTaskFlow):
    """Keep normalized MAE loss and report raw-scale metrics."""

    def postprocess(self, runner, forward_return: dict) -> dict:
        mean = forward_return["scinet_mean"].unsqueeze(1)
        std = forward_return["scinet_std"].unsqueeze(1)
        forward_return["prediction"] = forward_return["prediction"] * std + mean
        forward_return["targets"] = forward_return["targets"] * std + mean
        return forward_return


def scinet_mape(prediction: torch.Tensor, targets: torch.Tensor,
                targets_mask: torch.Tensor = None) -> torch.Tensor:
    """PEMS MAPE as implemented by SCINet (including its >5 cutoff)."""

    values = torch.abs(prediction - targets) / torch.abs(targets) + 1e-5
    values = torch.where(values > 5, torch.zeros_like(values), values)
    return values.mean()
