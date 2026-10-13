"""Split-local stride-1 windows and train-only normalization of the V experiment."""
from pathlib import Path
import numpy as np
import torch
from basicts.data import BasicTSForecastingDataset
from basicts.runners.taskflow import BasicTSForecastingTaskFlow


class PIRMambaDataset(BasicTSForecastingDataset):
    def __init__(self, dataset_name, input_len, output_len, mode,
                 use_timestamps=True, local=True, data_file_path=None, memmap=False):
        if not use_timestamps:
            raise ValueError("PIR-Mamba requires history timestamps")
        super().__init__(dataset_name, input_len, output_len, mode,
                         use_timestamps, local, data_file_path, memmap)
        folder = Path(data_file_path or f"datasets/{dataset_name}")
        train = np.load(folder / "train_data.npy", mmap_mode="r")
        if self._data.ndim != 2 or self._data.shape[1] != 170:
            raise ValueError("Expected prepared [time,170] PEMS08 arrays")
        if not np.isfinite(self._data).all() or not np.isfinite(train).all():
            raise ValueError("Frozen V protocol expects finite data")
        # Original code converts float32 training statistics to Python lists,
        # then performs NumPy float64 normalization before float32 tensor casts.
        mean = np.asarray(train.mean(axis=0).tolist(), dtype=np.float64)
        std = np.asarray(train.std(axis=0).tolist(), dtype=np.float64)
        std[std == 0] = 1
        self._data = ((self._data - mean) / std).astype(np.float32)
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)
        if self.timestamps.shape != (len(self._data), 2):
            raise ValueError("Expected time-of-day and cyclic day timestamps")

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["scaler_mean"] = self.mean
        item["scaler_std"] = self.std
        return item


class PIRMambaTaskFlow(BasicTSForecastingTaskFlow):
    def preprocess(self, runner, data):
        # Dataset is already normalized. Include original zero observations in
        # L1 loss, matching V; no framework rescaling/missing-value replacement.
        data["inputs"] = data["inputs"].float()
        data["targets"] = data["targets"].float()
        data["targets_mask"] = torch.ones_like(data["targets"], dtype=torch.bool)
        return data

    def postprocess(self, runner, forward_return):
        mean = forward_return["scaler_mean"].unsqueeze(1)
        std = forward_return["scaler_std"].unsqueeze(1)
        forward_return["prediction"] = forward_return["prediction"].double() * std + mean
        forward_return["targets"] = forward_return["targets"].double() * std + mean
        return forward_return


def unmasked_mae(prediction, targets):
    """Raw-scale unmasked MAE: original checkpoint selection criterion."""
    return (prediction - targets).abs().mean()

