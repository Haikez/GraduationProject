"""Shared BasicTS configuration builder for the Mamba-TCN-CA-DG MVP."""

from __future__ import annotations

import os

from easydict import EasyDict

from basicts.data import TimeSeriesForecastingDataset
from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings

from .arch import MambaTCNCADG


VARIANTS = {
    "tcn": {"use_tcn": True, "use_ssm": False, "use_ca": False, "use_dynamic_graph": False},
    "tcn_ssm": {"use_tcn": True, "use_ssm": True, "use_ca": False, "use_dynamic_graph": False},
    "tcn_ssm_ca": {"use_tcn": True, "use_ssm": True, "use_ca": True, "use_dynamic_graph": False},
    "full": {"use_tcn": True, "use_ssm": True, "use_ca": True, "use_dynamic_graph": True},
}


def build_config(
    dataset_name: str,
    num_nodes: int,
    *,
    variant: str = "full",
    input_len: int = 12,
    output_len: int = 12,
    num_epochs: int = 100,
    batch_size: int = 16,
    seed: int = 0,
) -> EasyDict:
    """Build a compact, reproducible BasicTS configuration for one experiment."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose one of {sorted(VARIANTS)}")
    regular_settings = get_regular_settings(dataset_name)
    switches = VARIANTS[variant]

    cfg = EasyDict()
    cfg.DESCRIPTION = (
        "Mamba-TCN-CA-DG MVP. The SSM branch is a local PyTorch selective "
        "state-space layer, not the official mamba_ssm implementation."
    )
    cfg.GPU_NUM = 1
    cfg.ENV = EasyDict({"SEED": seed})
    cfg.RUNNER = SimpleTimeSeriesForecastingRunner

    cfg.DATASET = EasyDict()
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.TYPE = TimeSeriesForecastingDataset
    cfg.DATASET.PARAM = EasyDict(
        {
            "dataset_name": dataset_name,
            "train_val_test_ratio": regular_settings["TRAIN_VAL_TEST_RATIO"],
            "input_len": input_len,
            "output_len": output_len,
        }
    )

    cfg.SCALER = EasyDict()
    cfg.SCALER.TYPE = ZScoreScaler
    cfg.SCALER.PARAM = EasyDict(
        {
            "dataset_name": dataset_name,
            "train_ratio": regular_settings["TRAIN_VAL_TEST_RATIO"][0],
            "norm_each_channel": regular_settings["NORM_EACH_CHANNEL"],
            "rescale": regular_settings["RESCALE"],
        }
    )

    cfg.MODEL = EasyDict()
    cfg.MODEL.NAME = MambaTCNCADG.__name__
    cfg.MODEL.ARCH = MambaTCNCADG
    cfg.MODEL.PARAM = {
        "num_nodes": num_nodes,
        "input_len": input_len,
        "output_len": output_len,
        "input_dim": 1,
        "hidden_dim": 32,
        "num_tcn_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
        **switches,
    }
    cfg.MODEL.FORWARD_FEATURES = [0, 1, 2]
    cfg.MODEL.TARGET_FEATURES = [0]

    cfg.METRICS = EasyDict()
    cfg.METRICS.FUNCS = EasyDict({"MAE": masked_mae, "MAPE": masked_mape, "RMSE": masked_rmse})
    cfg.METRICS.TARGET = "MAE"
    cfg.METRICS.NULL_VAL = regular_settings["NULL_VAL"]

    cfg.TRAIN = EasyDict()
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.TRAIN.CKPT_SAVE_DIR = os.path.join(
        "checkpoints", MambaTCNCADG.__name__, f"{dataset_name}_{variant}_{num_epochs}_{input_len}_{output_len}_s{seed}"
    )
    cfg.TRAIN.LOSS = masked_mae
    cfg.TRAIN.OPTIM = EasyDict({"TYPE": "Adam", "PARAM": {"lr": 0.001, "weight_decay": 1e-4}})
    cfg.TRAIN.LR_SCHEDULER = EasyDict({"TYPE": "MultiStepLR", "PARAM": {"milestones": [50, 80], "gamma": 0.3}})
    cfg.TRAIN.CLIP_GRAD_PARAM = {"max_norm": 5.0}
    cfg.TRAIN.DATA = EasyDict({"BATCH_SIZE": batch_size, "SHUFFLE": True})

    cfg.VAL = EasyDict({"INTERVAL": 1, "DATA": EasyDict({"BATCH_SIZE": batch_size})})
    cfg.TEST = EasyDict({"INTERVAL": 1, "DATA": EasyDict({"BATCH_SIZE": batch_size})})
    cfg.EVAL = EasyDict({"HORIZONS": [3, 6, 12], "USE_GPU": True})
    return cfg
