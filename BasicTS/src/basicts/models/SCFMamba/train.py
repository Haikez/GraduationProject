"""BasicTS forecasting configuration and command-line entry for SCF-Mamba."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import BasicTSCallback

from . import SCFMamba, SCFMambaConfig
from .data import (SCFMambaDataset, SCFMambaTaskFlow, masked_mae, masked_mape,
                   masked_mse, masked_rmse)


class SCFMambaAdam(Adam):
    """Build both parameter groups before BasicTS restores optimizer state."""

    def __init__(self, params, lr=0.001, weight_decay=1e-5,
                 betas=(0.9, 0.999), eps=1e-8):
        params = list(params)
        backbone = [p for p in params if not getattr(p, "_scf_attention", False)]
        attention = [p for p in params if getattr(p, "_scf_attention", False)]
        groups = [{"params": backbone, "name": "backbone", "weight_decay": weight_decay},
                  {"params": attention, "name": "attention", "weight_decay": 0.0}]
        super().__init__(groups, lr=lr, betas=betas, eps=eps)


class MaskedMetricsCallback(BasicTSCallback):
    def on_compute_loss(self, runner, forward_return, data=None, **kwargs):
        if data is not None:
            runner.should_optimizer_step = bool(forward_return["targets_mask"].any())

    @staticmethod
    def _fix_rmse(runner, phase, suffix=""):
        mse = runner.meter_pool.get_value(f"{phase}/MSE_masked{suffix}")
        meter = runner.meter_pool._pool[f"{phase}/RMSE_masked{suffix}"]["meter"]
        meter.reset()
        runner.update_meter(f"{phase}/RMSE_masked{suffix}", float(np.sqrt(mse)), 1)

    def on_epoch_end(self, runner, **kwargs):
        self._fix_rmse(runner, "train")

    def on_validate_end(self, runner, **kwargs):
        self._fix_rmse(runner, "val")

    def on_test_end(self, runner, **kwargs):
        self._fix_rmse(runner, "test")
        for horizon in runner.eval_horizons or []:
            self._fix_rmse(runner, "test", f"@h{horizon + 1}")


def build_config(dataset="PEMS08", data_dir=None, horizon=12, epochs=80,
                 batch_size=64, gpus="0", backend="official", seed=4321,
                 checkpoint_dir=None, num_workers=0, tf32=True, amp="off",
                 activation_checkpoint=True):
    if amp not in ("off", "bfloat16", "float16"):
        raise ValueError("amp must be off, bfloat16 or float16")
    if gpus is None and amp != "off":
        raise ValueError("AMP requires CUDA; use amp='off' for CPU checks")
    folder = Path(data_dir or f"datasets/{dataset}").resolve()
    train_file = folder / "train_data.npy"
    num_nodes = (np.load(train_file, mmap_mode="r").shape[1] if train_file.exists()
                 else {"PEMS08": 170, "PEMS04": 307}[dataset])
    if gpus is None and backend != "torch":
        raise ValueError("CPU execution requires backend='torch'")
    return BasicTSForecastingConfig(
        model=SCFMamba,
        model_config=SCFMambaConfig(output_len=horizon, num_features=num_nodes,
                                   mamba_backend=backend,
                                   activation_checkpoint=activation_checkpoint),
        dataset_name=dataset, input_len=96, output_len=horizon, use_timestamps=True,
        dataset_type=SCFMambaDataset, dataset_params={"data_file_path": str(folder)},
        scaler=None, rescale=False, taskflow=SCFMambaTaskFlow(),
        callbacks=[MaskedMetricsCallback()],
        gpus=gpus, batch_size=batch_size, num_epochs=epochs, seed=seed,
        loss=masked_mae, optimizer=SCFMambaAdam,
        optimizer_params={"lr": 0.001, "weight_decay": 1e-5, "betas": (0.9, 0.999)},
        lr_scheduler=ExponentialLR, lr_scheduler_params={"gamma": 0.95},
        metrics=[("MAE_masked", masked_mae), ("MSE_masked", masked_mse),
                 ("RMSE_masked", masked_rmse), ("MAPE_percent_masked", masked_mape)],
        target_metric="MAE_masked", eval_horizons=[h for h in (3, 6, 12, 24, 48, 96)
                                                 if h <= horizon],
        tf32=tf32, model_dtype="float32" if amp == "off" else amp, compile_model=False,
        cudnn_benchmark=False, cudnn_determinstic=True,
        ckpt_save_dir=checkpoint_dir, ckpt_save_strategy=None,
        test_interval=epochs + 1, eval_after_train=True, save_results=False,
        train_data_num_workers=num_workers, val_data_num_workers=num_workers,
        test_data_num_workers=num_workers, train_data_pin_memory=gpus is not None,
        val_data_pin_memory=gpus is not None, test_data_pin_memory=gpus is not None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("PEMS08", "PEMS04"), default="PEMS08")
    parser.add_argument("--data-dir")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", default="0", help="GPU IDs, or cpu for local checks")
    parser.add_argument("--backend", choices=("official", "torch"), default="official")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", choices=("off", "bfloat16", "float16"), default="off",
                        help="Autocast precision; parameters remain FP32")
    parser.add_argument("--activation-checkpoint", action=argparse.BooleanOptionalAction,
                        default=True, help="Recompute Mamba activations to save memory")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if min(args.horizon, args.epochs, args.batch_size) < 1 or args.num_workers < 0:
        parser.error("horizon, epochs and batch-size must be positive; workers cannot be negative")
    gpus = None if args.gpus == "cpu" else args.gpus
    if gpus is None and args.amp != "off":
        parser.error("AMP requires CUDA; CPU checks use --amp off")
    if gpus is not None and not args.print_config and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; CPU checks require --gpus cpu --backend torch")
    if (gpus is not None and not args.print_config and args.amp == "bfloat16"
            and not torch.cuda.is_bf16_supported()):
        parser.error("This GPU does not support BF16; choose --amp float16 or off")
    config = build_config(args.dataset, args.data_dir, args.horizon, args.epochs,
                          args.batch_size, gpus, args.backend, args.seed,
                          args.checkpoint_dir, args.num_workers, amp=args.amp,
                          activation_checkpoint=args.activation_checkpoint)
    if args.print_config:
        print(config)
    else:
        torch.set_num_threads(2)
        BasicTSLauncher.launch_training(config)


if __name__ == "__main__":
    main()
