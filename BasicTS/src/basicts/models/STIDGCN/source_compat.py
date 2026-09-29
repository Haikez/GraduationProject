"""Use the original STIDGCN windows and loss scale in the BasicTS runner."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from basicts.runners.callback import BasicTSCallback
from basicts.runners.taskflow import BasicTSTaskFlow


class SourceWindows(Dataset):
    def __init__(self, dataset_name, mode, data_dir, batch_size, train_size):
        if dataset_name != "PEMS08":
            raise ValueError("This source data configuration is for PEMS08")
        self.mode = str(mode)
        with np.load(Path(data_dir) / f"{self.mode}.npz") as archive:
            x = archive["x"]
            y = archive["y"]
            self.flow = np.asarray(x[..., 0], dtype=np.float32)
            self.timestamps = np.asarray(x[:, :, 0, 1:3], dtype=np.float32)
            self.timestamps[:, :, 1] /= 7.0
            self.targets = np.asarray(y[..., 0], dtype=np.float32)
        if self.flow.shape[1:] != (12, 170) or self.targets.shape[1:] != (12, 170):
            raise ValueError("Expected 12-step PEMS08 windows with 170 nodes")

        self.indices = torch.arange(len(self.flow))
        if self.mode in ("train", "val"):
            generator = torch.Generator().manual_seed(6666)
            train_order = torch.randperm(train_size, generator=generator)
            self.indices = (train_order if self.mode == "train"
                            else torch.randperm(len(self.flow), generator=generator))
        self.padded_size = ((len(self.flow) + batch_size - 1) // batch_size) * batch_size

    def __len__(self):
        return self.padded_size if self.mode in ("train", "val") else len(self.flow)

    def __getitem__(self, index):
        index = int(self.indices[min(index, len(self.indices) - 1)])
        return {
            "inputs": self.flow[index],
            "targets": self.targets[index],
            "inputs_timestamps": self.timestamps[index],
        }


class SourceTaskFlow(BasicTSTaskFlow):
    def preprocess(self, runner, data):
        data["inputs"] = ((data["inputs"] - runner.cfg.model_config.source_mean)
                          / runner.cfg.model_config.source_std)
        data["targets_mask"] = data["targets"] > 0.0
        return data

    def postprocess(self, runner, forward_return):
        return forward_return

    def get_weight(self, forward_return):
        # The source trainer averages the batch metrics, including its padded batch.
        return 1.0


class SourceTrainingRules(BasicTSCallback):
    def __init__(self, patience=100, min_epochs=300):
        self.patience = patience
        self.min_epochs = min_epochs
        self.best = float("inf")
        self.since_best = 0

    def on_train_start(self, runner, **kwargs):
        torch.backends.cudnn.allow_tf32 = True
        runner.logger.info(
            "Source STIDGCN data: %s; cuDNN TF32: %s; seed: %s",
            runner.cfg.dataset_params.data_dir,
            torch.backends.cudnn.allow_tf32,
            runner.cfg.seed,
        )

    def on_validate_end(self, runner, train_step=None, train_epoch=None):
        value = runner.meter_pool.get_value("val/MAE")
        if value < self.best:
            self.best = value
            self.since_best = 0
        else:
            self.since_best += 1
        if train_epoch >= self.min_epochs and self.since_best >= self.patience:
            runner.should_training_stop = True
            runner.logger.info("Source-style early stopping at epoch %s", train_epoch)


def source_wmape(prediction, targets, targets_mask):
    error = torch.abs(prediction - targets)[targets_mask]
    actual = torch.abs(targets[targets_mask])
    return error.sum() / actual.sum()
