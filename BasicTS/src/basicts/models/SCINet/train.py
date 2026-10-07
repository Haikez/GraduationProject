"""Train the source SCINet architecture on BasicTS PEMS datasets."""

import argparse
from pathlib import Path

import numpy as np
import torch

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.metrics.mae import masked_mae
from basicts.runners.callback import GradientClipping

from . import SCINet, SCINetConfig


def two_stack_mae(prediction, intermediate, targets, targets_mask=None):
    """Match the source code's sum of final and intermediate MAE losses."""
    return (masked_mae(prediction, targets, targets_mask)
            + masked_mae(intermediate, targets, targets_mask))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="PEMS08", help="Prepared BasicTS forecasting dataset")
    parser.add_argument("--data-dir", help="Prepared BasicTS dataset directory")
    parser.add_argument("--num-features", type=int, help="Override the number of series in the dataset")
    parser.add_argument("--gpus", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps", type=int, help="Short training run without evaluation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=12)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--stacks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--hidden-size", type=float, default=0.0625)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--concat-len", type=int, default=0)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--rin", action="store_true")
    parser.add_argument("--no-positional-encoding", action="store_true")
    parser.add_argument("--seed", type=int, default=4321)
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "input_len", "output_len", "levels", "kernel", "groups", "decoder_layers"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_features is not None and args.num_features < 1:
        parser.error("--num-features must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if not 0 <= args.dropout < 1 or args.hidden_size <= 0:
        parser.error("dropout must be in [0, 1) and hidden-size must be positive")
    return args


def build_config(args):
    data_dir = Path(args.data_dir or f"datasets/{args.dataset}")
    train_data_path = data_dir / "train_data.npy"
    if args.num_features is None:
        if not train_data_path.is_file():
            raise FileNotFoundError(f"Cannot infer num_features; missing {train_data_path}")
        num_features = np.load(train_data_path, mmap_mode="r").shape[-1]
    else:
        num_features = args.num_features
    model_config = SCINetConfig(
        input_len=args.input_len,
        output_len=args.output_len,
        num_features=num_features,
        hidden_size=args.hidden_size,
        num_stacks=args.stacks,
        num_levels=args.levels,
        num_decoder_layers=args.decoder_layers,
        concat_len=args.concat_len,
        groups=args.groups,
        kernel_size=args.kernel,
        dropout=args.dropout,
        positional_encoding=not args.no_positional_encoding,
        rin=args.rin,
    )
    dataset_params = {
        "dataset_name": args.dataset,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "use_timestamps": False,
        "memmap": False,
    }
    if args.data_dir:
        dataset_params["data_file_path"] = args.data_dir
    return BasicTSForecastingConfig(
        model=SCINet,
        model_config=model_config,
        dataset_name=args.dataset,
        dataset_params=dataset_params,
        input_len=args.input_len,
        output_len=args.output_len,
        use_timestamps=False,
        gpus=None if args.gpus.lower() == "cpu" else args.gpus,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=None if args.steps is not None else args.epochs,
        num_steps=args.steps,
        loss=two_stack_mae if args.stacks == 2 else "MAE",
        metrics=["MAE", "RMSE", "MAPE"],
        target_metric="MAE",
        optimizer=torch.optim.Adam,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        callbacks=[GradientClipping(5.0)],
        norm_each_channel=False,
        rescale=True,
        null_val=0.0,
        eval_horizons=sorted({h for h in (3, 6, 12, args.output_len) if h <= args.output_len}),
        eval_after_train=args.steps is None,
        save_results=args.steps is None,
        val_interval=args.steps + 1 if args.steps is not None else 1,
        test_interval=args.steps + 1 if args.steps is not None else args.epochs + 1,
        ckpt_save_dir=args.checkpoint_dir or f"checkpoints/SCINet/{args.dataset}_L{args.input_len}_H{args.output_len}_s{args.stacks}_seed{args.seed}",
    )


def main():
    BasicTSLauncher.launch_training(build_config(parse_args()))


if __name__ == "__main__":
    main()
