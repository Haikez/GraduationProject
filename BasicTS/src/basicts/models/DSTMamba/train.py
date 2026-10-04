"""Train the DST-Mamba source presets with BasicTS prepared datasets."""

import argparse

import torch
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, GradientClipping

from basicts.models.DSTMamba import DSTMamba, DSTMambaConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("LTSF", "STF"), default="LTSF")
    parser.add_argument("--dataset", help="BasicTS dataset name; defaults to PEMS08 or METR-LA")
    parser.add_argument("--data-dir", help="Override prepared BasicTS dataset directory")
    parser.add_argument("--gpus", default="0" if torch.cuda.is_available() else "cpu",
                        help="CUDA device IDs or cpu")
    parser.add_argument("--epochs", type=int, help="Override source preset epoch count")
    parser.add_argument("--steps", type=int, help="Run a short training check")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mamba-backend", choices=("official", "torch"),
                        default="official" if torch.cuda.is_available() else "torch")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--e-layers", type=int)
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for name in ("d_model", "d_ff", "node_dim", "rank", "e_layers"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.d_model <= args.node_dim:
        parser.error("--d-model must exceed --node-dim")
    return args


def build_config(args: argparse.Namespace) -> BasicTSForecastingConfig:
    long_term = args.task == "LTSF"
    dataset = args.dataset or ("PEMS08" if long_term else "METR-LA")
    nodes = {"PEMS08": 170, "METR-LA": 207}
    if dataset not in nodes:
        raise ValueError(f"Unknown node count for {dataset}; add it to the dataset map")
    input_len = 96 if long_term else 12
    output_len = 12
    epochs = args.epochs or (10 if long_term else 100)
    model_config = DSTMambaConfig(
        history_seq_len=input_len,
        future_seq_len=output_len,
        num_channels=nodes[dataset],
        e_layers=2 if long_term else 3,
        d_model=args.d_model,
        d_ff=args.d_ff,
        node_dim=args.node_dim,
        rank=args.rank,
        mamba_backend=args.mamba_backend,
    )
    if args.e_layers is not None:
        model_config.e_layers = args.e_layers
    dataset_params = {
        "dataset_name": dataset,
        "input_len": input_len,
        "output_len": output_len,
        "use_timestamps": False,
        "memmap": False,
    }
    if args.data_dir:
        dataset_params["data_file_path"] = args.data_dir
    scheduler = ExponentialLR if long_term else MultiStepLR
    scheduler_params = {"gamma": 0.5} if long_term else {"milestones": [20, 40], "gamma": 0.1}
    return BasicTSForecastingConfig(
        model=DSTMamba,
        model_config=model_config,
        dataset_name=dataset,
        dataset_params=dataset_params,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=False,
        gpus=None if args.gpus.lower() == "cpu" else args.gpus,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=None if args.steps is not None else epochs,
        num_steps=args.steps,
        loss="MSE" if long_term else "HUBER",
        metrics=["MSE", "MAE"] if long_term else ["MAE", "RMSE", "MAPE"],
        target_metric="MSE" if long_term else "MAE",
        optimizer_params={"lr": 1e-3, "weight_decay": 0.0},
        lr_scheduler=scheduler,
        lr_scheduler_params=scheduler_params,
        callbacks=[EarlyStopping(3 if long_term else 10), GradientClipping(5.0)],
        norm_each_channel=long_term,
        rescale=not long_term,
        null_val=0.0,
        eval_horizons=[3, 6, 12],
        eval_after_train=args.steps is None,
        val_interval=args.steps + 1 if args.steps is not None else 1,
        test_interval=args.steps + 1 if args.steps is not None else epochs + 1,
        ckpt_save_dir=(
            f"checkpoints/DSTMamba/{dataset}_{args.task}_{args.mamba_backend}"
            f"_d{args.d_model}_s{args.seed}"
        ),
    )


def main() -> None:
    BasicTSLauncher.launch_training(build_config(parse_args()))


if __name__ == "__main__":
    main()
