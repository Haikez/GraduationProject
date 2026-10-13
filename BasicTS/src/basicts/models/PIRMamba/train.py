"""BasicTS training entry point. Importing this module never starts training."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from . import PIRMamba, PIRMambaConfig
from .data import PIRMambaDataset, PIRMambaTaskFlow, unmasked_mae


def build_config(data_dir="datasets/PEMS08", epochs=10, batch_size=64,
                 gpus="0", backend="official", checkpoint_dir=None):
    return BasicTSForecastingConfig(
        model=PIRMamba, model_config=PIRMambaConfig(mamba_backend=backend),
        dataset_name="PEMS08", input_len=96, output_len=12, use_timestamps=True,
        dataset_type=PIRMambaDataset,
        dataset_params={"data_file_path": str(Path(data_dir).resolve())},
        scaler=None, rescale=False, taskflow=PIRMambaTaskFlow(),
        batch_size=batch_size, gpus=None if gpus == "cpu" else gpus,
        num_epochs=epochs, loss=nn.L1Loss(), optimizer=Adam,
        optimizer_params={"lr": .001, "weight_decay": 1e-5, "betas": (.9, .999)},
        lr_scheduler=ExponentialLR, lr_scheduler_params={"gamma": .95},
        metrics=[("MAE_unmasked", unmasked_mae)], target_metric="MAE_unmasked",
        eval_horizons=[3, 6, 12], seed=4321, tf32=False,
        deterministic=False, cudnn_benchmark=False, cudnn_determinstic=True,
        ckpt_save_dir=checkpoint_dir or f"checkpoints/PIRMamba/PEMS08_{epochs}_96_12",
        compile_model=False, model_dtype="float32", save_results=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/PEMS08")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--backend", choices=["official", "torch"], default="official")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not args.print_config and args.gpus != "cpu" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --gpus cpu --backend torch for CPU inspection")
    if args.gpus == "cpu" and args.backend != "torch":
        parser.error("CPU requires the explicit torch backend")
    torch.set_num_threads(4)
    config = build_config(args.data_dir, args.epochs, args.batch_size,
                          args.gpus, args.backend, args.checkpoint_dir)
    if args.print_config:
        print(config)
    else:
        BasicTSLauncher.launch_training(config)


if __name__ == "__main__":
    main()
