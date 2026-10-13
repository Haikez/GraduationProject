"""Train the STIDGCN-derived Mamba-TCN cross-attention model in BasicTS."""

import argparse

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.STIDGCN.ranger21 import Ranger
from basicts.runners.callback import GradientClipping

from . import MambaTCNCAConfig, MambaTCNCADG


# All variants retain the original STIDGCN embedding, DGCN tree, and head.
# without_ca restores the original Hadamard interaction.
VARIANTS = {
    "full": (True, True, True),
    "without_mamba": (True, False, True),
    "without_tcn": (False, True, True),
    "without_ca": (True, True, False),
}
DATASET_NODES = {"PEMS04": 307, "PEMS08": 170}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_NODES, default="PEMS08")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--gpus", default="0", help="CUDA device IDs, or 'cpu'")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps", type=int, help="Short training check without evaluation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--tcn-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    parser.add_argument("--mamba-backend", choices=("torch", "official"), default="torch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", help="Prepared BasicTS dataset directory")
    parser.add_argument("--checkpoint-dir", help="Override checkpoint directory")
    args = parser.parse_args()
    for name in (
        "epochs", "batch_size", "output_len", "channels", "tcn_layers",
        "attention_heads", "mamba_d_state", "mamba_d_conv", "mamba_expand",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    return args


def build_config(args: argparse.Namespace) -> BasicTSForecastingConfig:
    use_tcn, use_mamba, use_ca = VARIANTS[args.variant]
    model_config = MambaTCNCAConfig(
        num_nodes=DATASET_NODES[args.dataset],
        input_len=12,
        output_len=args.output_len,
        channels=args.channels,
        num_tcn_layers=args.tcn_layers,
        num_heads=args.attention_heads,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        mamba_backend=args.mamba_backend,
        use_tcn=use_tcn,
        use_mamba=use_mamba,
        use_ca=use_ca,
    )
    checkpoint_dir = args.checkpoint_dir or (
        f"checkpoints/MambaTCNCADG/{args.dataset}_{args.variant}"
        f"_{args.mamba_backend}_c{args.channels}_s{args.seed}"
    )
    dataset_params = {
        "dataset_name": args.dataset,
        "input_len": 12,
        "output_len": args.output_len,
        "use_timestamps": True,
        "memmap": False,
    }
    if args.data_dir:
        dataset_params["data_file_path"] = args.data_dir
    return BasicTSForecastingConfig(
        model=MambaTCNCADG,
        model_config=model_config,
        dataset_name=args.dataset,
        dataset_params=dataset_params,
        input_len=12,
        output_len=args.output_len,
        use_timestamps=True,
        gpus=None if args.gpus.lower() == "cpu" else args.gpus,
        num_epochs=None if args.steps is not None else args.epochs,
        num_steps=args.steps,
        batch_size=args.batch_size,
        norm_each_channel=False,
        rescale=True,
        null_val=0.0,
        seed=args.seed,
        optimizer=Ranger,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
        callbacks=[GradientClipping(5.0)],
        metrics=["MAE", "RMSE", "MAPE"],
        eval_horizons=sorted({
            horizon for horizon in (3, 6, 12, args.output_len)
            if horizon <= args.output_len
        }),
        eval_after_train=args.steps is None,
        save_results=args.steps is None,
        val_interval=args.steps + 1 if args.steps is not None else 1,
        test_interval=args.steps + 1 if args.steps is not None else args.epochs + 1,
        ckpt_save_dir=checkpoint_dir,
    )


def main() -> None:
    BasicTSLauncher.launch_training(build_config(parse_args()))


if __name__ == "__main__":
    main()
