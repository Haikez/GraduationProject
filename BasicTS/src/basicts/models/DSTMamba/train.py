"""Train the DST-Mamba source presets with BasicTS prepared datasets."""

import argparse

import torch
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.runners.callback import EarlyStopping, GradientClipping

from basicts.models.DSTMamba import DSTMamba, DSTMambaConfig
from basicts.models.STIDGCN.ranger21 import Ranger
from basicts.models.DSTMamba.scinet_pems_data import (
    SCINetPEMSDataset, SCINetPEMSTaskFlow, scinet_mape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("LTSF", "STF"), default="LTSF")
    parser.add_argument("--dataset", help="BasicTS dataset name; defaults to PEMS08 or METR-LA")
    parser.add_argument("--data-dir", help="Override prepared BasicTS dataset directory")
    parser.add_argument("--gpus", default="0" if torch.cuda.is_available() else "cpu",
                        help="CUDA device IDs or cpu")
    parser.add_argument("--epochs", type=int, help="Override source preset epoch count")
    parser.add_argument("--steps", type=int, help="Run a short training check")
    parser.add_argument("--training-profile", choices=("source", "scinet_pems", "stidgcn_basicts"),
                        default="source",
                        help="Use the DST-Mamba, SCINet PEMS, or BasicTS STIDGCN training recipe")
    parser.add_argument("--batch-size", type=int,
                        help="Override profile batch size (source: 32; scinet_pems: 8; stidgcn_basicts: 64)")
    parser.add_argument("--cudnn-tf32", action="store_true",
                        help="Enable cuDNN TF32 for convolutions; keep CUDA matmul TF32 off")
    parser.add_argument("--no-tf32", action="store_true",
                        help="Disable both cuDNN and CUDA matmul TF32 for legacy runs")
    parser.add_argument("--tf32", action="store_true",
                        help="Enable both cuDNN and CUDA matmul TF32, including matched STIDGCN-profile runs")
    parser.add_argument("--seed", type=int,
                        help="Override profile seed (source: 2024; scinet_pems: 4321; stidgcn_basicts: 42)")
    parser.add_argument("--input-len", type=int, help="Override the history length")
    parser.add_argument("--output-len", type=int, default=12)
    parser.add_argument("--trend-downsampling", choices=("source", "avg", "sci_low", "sci"),
                        default="source",
                        help="source: original path; avg/sci_low/sci: decompose first, then downsample trend")
    parser.add_argument("--ds-layers", type=int, default=3)
    parser.add_argument("--sci-hidden-channels", type=int, default=8)
    parser.add_argument("--sci-kernel-size", type=int, default=3)
    parser.add_argument("--sci-dropout", type=float, default=0.0)
    parser.add_argument("--seasonal-encoder", choices=("dst_mamba", "idmamba", "stidgcn"),
                        default="dst_mamba")
    parser.add_argument("--idmamba-channels", type=int, default=48)
    parser.add_argument("--idmamba-granularity", type=int, default=288)
    parser.add_argument("--idmamba-dropout", type=float, default=0.1)
    parser.add_argument("--stidgcn-channels", type=int, default=48)
    parser.add_argument("--stidgcn-granularity", type=int, default=288)
    parser.add_argument("--stidgcn-dropout", type=float, default=0.1)
    parser.add_argument("--mamba-backend", choices=("official", "torch"),
                        default="official" if torch.cuda.is_available() else "torch")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--node-dim", type=int, default=64)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--e-layers", type=int)
    args = parser.parse_args()
    if args.tf32 and (args.cudnn_tf32 or args.no_tf32):
        parser.error("--tf32 cannot be combined with --cudnn-tf32 or --no-tf32")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.input_len is not None and args.input_len < 1:
        parser.error("--input-len must be positive")
    if args.output_len < 1 or args.ds_layers < 1 or args.sci_hidden_channels < 1:
        parser.error("--output-len, --ds-layers and --sci-hidden-channels must be positive")
    if args.sci_kernel_size < 1 or args.sci_kernel_size % 2 != 1:
        parser.error("--sci-kernel-size must be a positive odd number")
    if not 0.0 <= args.sci_dropout < 1.0:
        parser.error("--sci-dropout must be in [0, 1)")
    if (args.idmamba_channels < 1 or args.idmamba_granularity < 1
            or not 0.0 <= args.idmamba_dropout < 1.0):
        parser.error("ID-Mamba channels/granularity must be positive and dropout in [0, 1)")
    if (args.stidgcn_channels < 1 or args.stidgcn_granularity < 1
            or not 0.0 <= args.stidgcn_dropout < 1.0):
        parser.error("STIDGCN channels/granularity must be positive and dropout in [0, 1)")
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
    scinet_pems = args.training_profile == "scinet_pems"
    stidgcn_basicts = args.training_profile == "stidgcn_basicts"
    if (scinet_pems or stidgcn_basicts) and (
            not long_term or dataset != "PEMS08" or args.seasonal_encoder != "stidgcn"):
        raise ValueError("This profile is for PEMS08 LTSF with the STIDGCN seasonal encoder")
    batch_size = args.batch_size if args.batch_size is not None else (
        64 if stidgcn_basicts else (8 if scinet_pems else 32))
    seed = args.seed if args.seed is not None else (
        42 if stidgcn_basicts else (4321 if scinet_pems else 2024))
    nodes = {"PEMS08": 170, "METR-LA": 207}
    if dataset not in nodes:
        raise ValueError(f"Unknown node count for {dataset}; add it to the dataset map")
    input_len = args.input_len or (12 if (scinet_pems or stidgcn_basicts) else (96 if long_term else 12))
    output_len = args.output_len
    if args.trend_downsampling in ("sci", "sci_low") and input_len % (2 ** args.ds_layers):
        raise ValueError("SCI downsampling needs input length divisible by 2**ds_layers")
    if args.seasonal_encoder == "idmamba" and input_len % 4:
        raise ValueError("ID-Mamba seasonal tree needs input length divisible by 4")
    if args.seasonal_encoder == "stidgcn" and input_len % 4:
        raise ValueError("STIDGCN seasonal tree needs input length divisible by 4")
    use_timestamps = args.seasonal_encoder in ("idmamba", "stidgcn")
    epochs = args.epochs or (300 if stidgcn_basicts else (80 if scinet_pems else (10 if long_term else 100)))
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
        trend_downsampling=args.trend_downsampling,
        ds_layers=args.ds_layers,
        sci_hidden_channels=args.sci_hidden_channels,
        sci_kernel_size=args.sci_kernel_size,
        sci_dropout=args.sci_dropout,
        seasonal_encoder=args.seasonal_encoder,
        idmamba_channels=args.idmamba_channels,
        idmamba_granularity=args.idmamba_granularity,
        idmamba_dropout=args.idmamba_dropout,
        stidgcn_channels=args.stidgcn_channels,
        stidgcn_granularity=args.stidgcn_granularity,
        stidgcn_dropout=args.stidgcn_dropout,
    )
    if args.e_layers is not None:
        model_config.e_layers = args.e_layers
    dataset_params = {
        "dataset_name": dataset,
        "input_len": input_len,
        "output_len": output_len,
        "use_timestamps": use_timestamps,
        "memmap": False,
    }
    if args.data_dir:
        dataset_params["data_file_path"] = args.data_dir
    scheduler = None if stidgcn_basicts else (ExponentialLR if long_term else MultiStepLR)
    # SCINet's default lradj=1 resets the LR to initial_lr * 0.95**epoch
    # at every epoch start; its additional five-epoch decay is overwritten.
    scheduler_params = (None if stidgcn_basicts else
                        ({"gamma": 0.95} if scinet_pems else
                         ({"gamma": 0.5} if long_term else {"milestones": [20, 40], "gamma": 0.1})))
    checkpoint_dir = (
        f"checkpoints/DSTMamba/{dataset}_{args.task}_{args.mamba_backend}"
        f"_d{args.d_model}_s{seed}"
    )
    if args.trend_downsampling != "source" or args.input_len is not None or output_len != 12:
        checkpoint_dir += f"_{args.trend_downsampling}_i{input_len}_o{output_len}"
    if args.trend_downsampling == "sci":
        checkpoint_dir += "_low_detail_zero"
    elif args.trend_downsampling == "sci_low":
        checkpoint_dir += "_zero"
    if args.seasonal_encoder == "idmamba":
        checkpoint_dir += f"_seasonal_idmamba_full_c{args.idmamba_channels}"
    elif args.seasonal_encoder == "stidgcn":
        checkpoint_dir += f"_seasonal_stidgcn_full_c{args.stidgcn_channels}"
    if scinet_pems:
        checkpoint_dir += "_scinet_pems_v2"
    elif stidgcn_basicts:
        checkpoint_dir += "_stidgcn_basicts"
    if args.cudnn_tf32:
        checkpoint_dir += "_cudnn_tf32"
    if args.tf32:
        checkpoint_dir += "_full_tf32"
    cfg = BasicTSForecastingConfig(
        model=DSTMamba,
        model_config=model_config,
        dataset_name=dataset,
        **({"dataset_type": SCINetPEMSDataset,
            "taskflow": SCINetPEMSTaskFlow(),
            "scaler": None} if scinet_pems else {}),
        dataset_params=dataset_params,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=use_timestamps,
        gpus=None if args.gpus.lower() == "cpu" else args.gpus,
        seed=seed,
        batch_size=batch_size,
        **({"cudnn_tf32": True} if args.cudnn_tf32 else {}),
        num_epochs=None if args.steps is not None else epochs,
        num_steps=args.steps,
        loss="MAE" if (scinet_pems or stidgcn_basicts) else ("MSE" if long_term else "HUBER"),
        metrics=["MAE", "RMSE", ("MAPE", scinet_mape)] if scinet_pems else
                (["MAE", "RMSE", "MAPE"] if stidgcn_basicts else
                 (["MSE", "MAE"] if long_term else ["MAE", "RMSE", "MAPE"])),
        target_metric="MAE" if (scinet_pems or stidgcn_basicts) else ("MSE" if long_term else "MAE"),
        optimizer=Ranger if stidgcn_basicts else torch.optim.Adam,
        optimizer_params=({"lr": 1e-3, "weight_decay": 1e-4} if stidgcn_basicts else
                          {"lr": 1e-3, "betas": (0.9, 0.999),
                           "weight_decay": 1e-5 if scinet_pems else 0.0}),
        lr_scheduler=scheduler,
        lr_scheduler_params=scheduler_params,
        callbacks=([GradientClipping(5.0)] if stidgcn_basicts else
                   ([] if scinet_pems else [EarlyStopping(3 if long_term else 10), GradientClipping(5.0)])),
        norm_each_channel=False if stidgcn_basicts else long_term,
        # SCINetPEMSTaskFlow leaves training loss normalized and rescales metrics.
        rescale=True if stidgcn_basicts else (scinet_pems or not long_term),
        null_val=float("nan") if scinet_pems else 0.0,
        eval_horizons=[3, 6, 12],
        eval_after_train=args.steps is None,
        val_interval=args.steps + 1 if args.steps is not None else 1,
        test_interval=args.steps + 1 if args.steps is not None else (
            1 if (scinet_pems or stidgcn_basicts) else epochs + 1),
        train_data_num_workers=1 if scinet_pems else 0,
        val_data_num_workers=1 if scinet_pems else 0,
        test_data_num_workers=1 if scinet_pems else 0,
        cudnn_benchmark=not scinet_pems,
        cudnn_determinstic=scinet_pems,
        ckpt_save_dir=checkpoint_dir,
    )
    # Explicit modes preserve old experiment hashes; unflagged runs use the
    # BasicTS framework default, which enables both TF32 backends on GPU.
    if args.tf32:
        cfg.tf32 = True
    elif stidgcn_basicts or args.cudnn_tf32 or args.no_tf32:
        cfg.tf32 = False
    return cfg


def main() -> None:
    BasicTSLauncher.launch_training(build_config(parse_args()))


if __name__ == "__main__":
    main()
