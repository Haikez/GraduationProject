"""Train-only normalization and exact raw-label masks for traffic forecasting."""
from pathlib import Path

import numpy as np
import torch

from basicts.data import BasicTSForecastingDataset
from basicts.runners.taskflow import BasicTSForecastingTaskFlow


class SCFMambaDataset(BasicTSForecastingDataset):
    def __init__(self, dataset_name, input_len, output_len, mode,
                 use_timestamps=True, local=True, data_file_path=None, memmap=False):
        if not use_timestamps:
            raise ValueError("SCF-Mamba requires history timestamps")
        super().__init__(dataset_name, input_len, output_len, mode,
                         use_timestamps, local, data_file_path, memmap)
        folder = Path(data_file_path or f"datasets/{dataset_name}")
        train = np.load(folder / "train_data.npy", mmap_mode="r")
        if self._data.ndim != 2 or train.ndim != 2 or self._data.shape[1] != train.shape[1]:
            raise ValueError("Expected matching [time, nodes] arrays")
        if not np.isfinite(self._data).all() or not np.isfinite(train).all():
            raise ValueError("Traffic arrays must contain finite values")
        # Match the existing experiments' float32 statistics/float64 normalization.
        mean = np.asarray(train.mean(axis=0).tolist(), dtype=np.float64)
        std = np.asarray(train.std(axis=0).tolist(), dtype=np.float64)
        std[std == 0] = 1
        self._raw_data = self._data
        self._data = ((self._data - mean) / std).astype(np.float32)
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        if self.timestamps.shape != (len(self._data), 2):
            raise ValueError("Expected time-of-day and cyclic-day timestamps")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        raw_targets = self._raw_data[index + self.input_len:
                                     index + self.input_len + self.output_len]
        item.update(raw_targets=raw_targets.copy(), targets_mask=(raw_targets != 0).copy(),
                    scaler_mean=self.mean, scaler_std=self.std)
        return item


class SCFMambaTaskFlow(BasicTSForecastingTaskFlow):
    def preprocess(self, runner, data):
        data["inputs"] = data["inputs"].float()
        data["targets"] = data["targets"].float()
        data["targets_mask"] = data["targets_mask"].bool()
        return data

    def postprocess(self, runner, forward_return):
        mean = forward_return["scaler_mean"].unsqueeze(1)
        std = forward_return["scaler_std"].unsqueeze(1)
        forward_return["prediction"] = forward_return["prediction"].double() * std + mean
        forward_return["targets"] = forward_return["raw_targets"].double()
        return forward_return


def masked_mae(prediction, targets, targets_mask):
    error = (prediction - targets).abs()
    return torch.where(targets_mask, error, 0).sum() / targets_mask.sum().clamp_min(1)


def masked_mse(prediction, targets, targets_mask):
    error = (prediction - targets).square()
    return torch.where(targets_mask, error, 0).sum() / targets_mask.sum().clamp_min(1)


def masked_rmse(prediction, targets, targets_mask):
    return masked_mse(prediction, targets, targets_mask).sqrt()


def masked_mape(prediction, targets, targets_mask):
    denominator = torch.where(targets_mask, targets.abs(), 1)
    error = (prediction - targets).abs() / denominator
    return 100 * torch.where(targets_mask, error, 0).sum() / targets_mask.sum().clamp_min(1)
