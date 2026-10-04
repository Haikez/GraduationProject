"""Run PEMS08 with the DST-Mamba source training protocol.

The data loader and runner come from the author's DST-Mamba checkout. Only the
model is loaded from BasicTS, with the official Mamba kernel on GPU.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from basicts.models.DSTMamba import DSTMamba, DSTMambaConfig


class SourceShapeModel(nn.Module):
    """Adapt BasicTS [B,T,N] tensors to the source runner's [B,T,N,1]."""

    def __init__(self, config):
        super().__init__()
        self.model = DSTMamba(config)

    def forward(self, history):
        return self.model(history.squeeze(-1)).unsqueeze(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("official", "torch"), default="official")
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root))
    from data.get_dataloader import build_LTSF_dataloader
    from runners.LTSFRunner import LTSFRunner

    config_path = args.source_root / "baselines/DSTMamba/LTSFConfig/PEMS08_IN96_OUT12.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["GENERAL"]["max_epochs"] = args.epochs
    config["MODEL_PARAM"]["mamba_backend"] = args.backend
    if args.output_dir is None:
        run_id = hashlib.md5(
            json.dumps({"config": config, "seed": args.seed,
                        "started": datetime.now().isoformat()}, sort_keys=True).encode()
        ).hexdigest()
        args.output_dir = (Path("checkpoints/DSTMamba") /
                           f"PEMS08_{args.epochs}_96_12" / run_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cfg.json").write_text(json.dumps(config, indent=2))
    (args.output_dir / "RUN_INFO.txt").write_text(
        "Training uses the original DST-Mamba LTSFRunner and data loader.\n"
        "DSTMamba_source_best_val_MSE.pt is its raw SourceShapeModel state_dict; "
        "it is not a BasicTS runner checkpoint.\n"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available")

    log_path = args.output_dir / f"training_log_{datetime.now():%Y%m%d%H%M%S}.log"
    with log_path.open("w", buffering=1) as log:
        print(f"Started {datetime.now().isoformat()} on {device}; seed={args.seed}", file=log)
        train_loader, val_loader, test_loader, scaler = build_LTSF_dataloader(
            str(args.source_root / "data/datasets_zoo/PEMS08"),
            batch_size=config["GENERAL"]["batch_size"],
            in_steps=config["DATA"]["in_steps"],
            out_steps=config["DATA"]["out_steps"],
            log=log,
        )
        model = SourceShapeModel(DSTMambaConfig(**config["MODEL_PARAM"])).to(device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}", file=log)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["OPTIM"]["initial_lr"],
            weight_decay=config["OPTIM"].get("weight_decay", 0),
            eps=config["OPTIM"].get("eps", 1e-8),
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config["OPTIM"]["lr_scheduler_gamma"]
        )
        runner = LTSFRunner(config, device=device, scaler=scaler, log=log)
        model = runner.train(
            model, train_loader, val_loader, optimizer, scheduler,
            nn.MSELoss(), max_epochs=args.epochs,
            early_stop_patience=config["GENERAL"]["early_stop_patience"],
            save=str(args.output_dir / "DSTMamba_source_best_val_MSE.pt"),
        )
        runner.test_model(model, test_loader)
        print(f"Finished {datetime.now().isoformat()}", file=log)

    match = re.search(r"All Steps \(1-\d+\) MSE = ([0-9.]+), MAE = ([0-9.]+)",
                      log_path.read_text())
    if match:
        (args.output_dir / "test_metrics.json").write_text(json.dumps({
            "overall": {"MSE": float(match.group(1)), "MAE": float(match.group(2))},
            "protocol": "original DST-Mamba LTSFRunner",
        }, indent=4))


if __name__ == "__main__":
    main()
