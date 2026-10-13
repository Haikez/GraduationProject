"""Run one full PEMS08 epoch with the DST-Mamba source training protocol on CPU.

The data loader and runner come from the author's DST-Mamba checkout. Only the
Mamba kernel uses BasicTS's unfused PyTorch implementation for CPU support.
"""

import argparse
import json
import random
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root))
    from data.get_dataloader import build_LTSF_dataloader
    from runners.LTSFRunner import LTSFRunner

    config_path = args.source_root / "baselines/DSTMamba/LTSFConfig/PEMS08_IN96_OUT12.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["GENERAL"]["max_epochs"] = 1
    config["MODEL_PARAM"]["mamba_backend"] = "torch"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    with (args.output_dir / "train.log").open("w", buffering=1) as log:
        print(f"Started {datetime.now().isoformat()} on {device}; seed={args.seed}", file=log)
        train_loader, val_loader, test_loader, scaler = build_LTSF_dataloader(
            str(args.source_root / "data/datasets_zoo/PEMS08"),
            batch_size=config["GENERAL"]["batch_size"],
            in_steps=config["DATA"]["in_steps"],
            out_steps=config["DATA"]["out_steps"],
            log=log,
        )
        model = SourceShapeModel(DSTMambaConfig(**config["MODEL_PARAM"]))
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
            nn.MSELoss(), max_epochs=1,
            early_stop_patience=config["GENERAL"]["early_stop_patience"],
            save=str(args.output_dir / "best.pt"),
        )
        runner.test_model(model, test_loader)
        print(f"Finished {datetime.now().isoformat()}", file=log)


if __name__ == "__main__":
    main()
