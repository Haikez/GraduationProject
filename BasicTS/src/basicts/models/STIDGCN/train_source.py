"""Train BasicTS STIDGCN using the original STIDGCN PEMS08 windows."""

import argparse
from pathlib import Path

import numpy as np
import torch

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig, BasicTSModelConfig
from basicts.models.STIDGCN import STIDGCN
from basicts.models.STIDGCN.ranger21 import Ranger
from basicts.models.STIDGCN.source_compat import (
    SourceTaskFlow,
    SourceTrainingRules,
    SourceWindows,
    source_wmape,
)
from basicts.runners.callback import GradientClipping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/root/autodl-tmp/STIDGCN/data/PEMS08")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-dir", default="/root/autodl-tmp/STIDGCN/basicts-source-aligned")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    with np.load(data_dir / "train.npz") as archive:
        train_flow = archive["x"][..., 0]
        train_size = len(train_flow)
        source_mean = float(train_flow.mean())
        source_std = float(train_flow.std())
    if source_std <= 0:
        raise ValueError("Training flow has zero variance")

    # EasyTorch's `tf32=True` keeps PyTorch's default in a fresh process.
    # Set it explicitly so the requested cuDNN behavior is guaranteed.
    torch.backends.cudnn.allow_tf32 = True
    config = BasicTSForecastingConfig(
        model=STIDGCN,
        model_config=BasicTSModelConfig(
            num_nodes=170, channels=48, dropout=0.1,
            source_mean=source_mean, source_std=source_std,
        ),
        dataset_name="PEMS08",
        dataset_type=SourceWindows,
        dataset_params={
            "dataset_name": "PEMS08",
            "data_dir": str(data_dir),
            "batch_size": args.batch_size,
            "train_size": train_size,
        },
        taskflow=SourceTaskFlow(),
        scaler=None,
        gpus=args.gpu,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=6666,
        tf32=True,
        cudnn_benchmark=False,
        cudnn_determinstic=True,
        rescale=False,
        train_data_shuffle=False,
        val_data_shuffle=False,
        optimizer=Ranger,
        optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
        callbacks=[GradientClipping(5.0), SourceTrainingRules(100, 300)],
        metrics=["MAE", "RMSE", "MAPE", ("WMAPE", source_wmape)],
        eval_horizons=list(range(1, 13)),
        val_interval=1,
        test_interval=1000000,
        eval_after_train=True,
        ckpt_save_dir=args.checkpoint_dir,
    )
    BasicTSLauncher.launch_training(config)


if __name__ == "__main__":
    main()
