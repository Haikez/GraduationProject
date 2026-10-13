"""Exact streaming zero-masked evaluation; never trains or overwrites source runs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from basicts.models.PIRMamba import PIRMamba, PIRMambaConfig
from basicts.models.PIRMamba.utils import DEFAULT_CHECKPOINT, load_checkpoint
from basicts.models.PIRMamba.data import PIRMambaDataset


@torch.no_grad()
def evaluate(model, dataset, batch_size=64, device="cuda:0"):
    model.eval()
    error_totals = np.zeros((12, 4), dtype=np.float64)
    mean = dataset.mean.numpy()
    std = dataset.std.numpy()
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        prediction = model(batch["inputs"].to(device),
                           batch["inputs_timestamps"].to(device)).cpu().numpy() * std + mean
        target = batch["targets"].numpy() * std + mean
        if not np.isfinite(prediction).all():
            raise RuntimeError("Non-finite prediction")
        error = np.abs(prediction - target)
        valid = ~np.isclose(target, 0., atol=5e-5)
        error_totals[:, 0] += np.where(valid, error, 0).sum(axis=(0, 2))
        error_totals[:, 1] += np.where(valid, error**2, 0).sum(axis=(0, 2))
        error_totals[:, 2] += np.where(valid, error / np.where(valid, np.abs(target), 1.), 0).sum(axis=(0, 2))
        error_totals[:, 3] += valid.sum(axis=(0, 2))

    def metrics(totals):
        return {"MAE": float(totals[0]/totals[3]), "RMSE": float(np.sqrt(totals[1]/totals[3])),
                "MAPE_percent": float(totals[2]/totals[3]*100), "valid_elements": int(totals[3])}
    return {"windows": len(dataset), "all_12_steps": metrics(error_totals.sum(axis=0)),
            "horizons": {str(h): metrics(error_totals[h-1]) for h in [3, 6, 12]},
            "all_steps": {str(h): metrics(error_totals[h-1]) for h in range(1, 13)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/PEMS08")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=["official", "torch"], default="official")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--output", help="New JSON path; existing files are never overwritten")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.max_windows is not None and args.max_windows < 1):
        parser.error("batch-size and max-windows must be positive")
    if args.output and Path(args.output).exists():
        parser.error("output already exists; use a new path")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; CPU checks require --device cpu --backend torch")
    if args.device == "cpu" and args.backend != "torch":
        parser.error("CPU requires explicit --backend torch")
    torch.set_num_threads(4)
    # Match the frozen V experiment and BasicTS training precision settings.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = PIRMamba(PIRMambaConfig(mamba_backend=args.backend)).to(args.device)
    loaded = load_checkpoint(model, args.checkpoint)
    dataset = PIRMambaDataset("PEMS08", 96, 12, "test", data_file_path=args.data_dir)
    full_windows = len(dataset)
    if args.max_windows is not None:
        subset = Subset(dataset, range(min(args.max_windows, full_windows)))
        subset.mean, subset.std = dataset.mean, dataset.std
        dataset = subset
    report = evaluate(model, dataset, args.batch_size, args.device)
    report.update({"model": "PIR-Mamba", "backend": args.backend, "device": args.device,
                   "checkpoint": loaded, "full_test_windows": full_windows,
                   "partial_probe": len(dataset) != full_windows,
                   "new_training": False})
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with Path(args.output).open("x") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
