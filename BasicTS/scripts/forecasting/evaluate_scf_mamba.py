"""Evaluate SCF-Mamba on every split-local test window in original traffic units."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from basicts.models.SCFMamba import SCFMamba, SCFMambaConfig
from basicts.models.SCFMamba.data import SCFMambaDataset
from basicts.models.SCFMamba.utils import load_checkpoint


@torch.no_grad()
def evaluate(model, dataset, batch_size=64, device="cpu"):
    model.eval()
    totals = torch.zeros(dataset.output_len, 4, dtype=torch.float64)
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        prediction = model(batch["inputs"].to(device), batch["inputs_timestamps"].to(device))
        prediction = prediction.cpu().double() * batch["scaler_std"].unsqueeze(1)
        prediction += batch["scaler_mean"].unsqueeze(1)
        targets = batch["raw_targets"].double()
        mask = batch["targets_mask"]
        if prediction.shape != targets.shape or not prediction.isfinite().all():
            raise ValueError("Invalid model prediction")
        error = (prediction - targets).abs()
        totals[:, 0] += torch.where(mask, error, 0).sum((0, 2))
        totals[:, 1] += torch.where(mask, error.square(), 0).sum((0, 2))
        totals[:, 2] += torch.where(mask, error / torch.where(mask, targets.abs(), 1), 0).sum((0, 2))
        totals[:, 3] += mask.sum((0, 2))

    def metrics(row):
        absolute, squared, percentage, count = row.tolist()
        if not count:
            raise ValueError("No nonzero test targets")
        return {"MAE": absolute / count, "MSE": squared / count,
                "RMSE": (squared / count) ** 0.5,
                "MAPE_percent": percentage / count * 100, "valid_elements": int(count)}

    return {"windows": len(dataset), "all_H_steps": metrics(totals.sum(0)),
            "all_steps": {str(step + 1): metrics(row) for step, row in enumerate(totals)},
            "metric_protocol": {"mask": "exact raw target != 0", "scale": "original traffic flow",
                                "accumulation_dtype": "float64", "MAPE_unit": "percent"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("PEMS08", "PEMS04"), default="PEMS08")
    parser.add_argument("--data-dir")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("official", "torch"), default="official")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output")
    args = parser.parse_args()
    if min(args.horizon, args.batch_size) < 1:
        parser.error("horizon and batch-size must be positive")
    if args.device == "cpu" and args.backend != "torch":
        parser.error("CPU checks require --backend torch")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.output and Path(args.output).exists():
        parser.error("Output already exists; choose a new path")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    folder = Path(args.data_dir or f"datasets/{args.dataset}")
    num_nodes = np.load(folder / "train_data.npy", mmap_mode="r").shape[1]
    model = SCFMamba(SCFMambaConfig(output_len=args.horizon, num_features=num_nodes,
                                   mamba_backend=args.backend)).to(args.device)
    checkpoint = load_checkpoint(model, args.checkpoint)
    dataset = SCFMambaDataset(args.dataset, 96, args.horizon, "test", data_file_path=str(folder))
    report = evaluate(model, dataset, args.batch_size, args.device)
    report.update(model="SCF-Mamba", dataset=args.dataset, input_len=96, output_len=args.horizon,
                  checkpoint=checkpoint, precision={"model_dtype": "float32", "tf32": True, "amp": False})
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["all_H_steps"], indent=2))


if __name__ == "__main__":
    main()
