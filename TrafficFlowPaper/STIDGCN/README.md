# STIDGCN in BasicTS

The architecture is adapted from the local STIDGCN `model.py` for BasicTS forecasting. It expects 12 history steps and three features: flow, normalized time of day, and normalized day of week. It predicts 12 flow values per node. Available configs: PEMS03, PEMS04, PEMS07, and PEMS08-1 (the PEMS08 dataset directory in this BasicTS checkout is named `PEMS08-1`).

Run from the BasicTS root, for example:

```bash
python experiments/train.py -c baselines/STIDGCN/PEMS08-1.py -g 0
```

The configs use the original project's Ranger optimizer with the same learning rate and weight decay. The BasicTS PEMS08-1 data corresponds to the original PEMS08 windows, with day-of-week normalized by 7. The adapter restores the 0-6 weekday value for the original start convolution and uses the normalized value for temporal embedding. The BasicTS dataset splits, normalization, checkpoint selection, and metrics are handled by its standard runner.

For a comparison against the original PEMS08 test windows after training, pass the best validation checkpoint to the comparison script:

```bash
python -m baselines.STIDGCN.compare_source --checkpoint checkpoints/STIDGCN/PEMS08-1/<run-id>/STIDGCN_best_val_MAE.pt
```

The script checks prediction equality after loading the same trained weights into both implementations, then evaluates the BasicTS model on the original test windows and compares its metrics with the original run log. The original run's trained checkpoint is not present, so independently trained per-sample predictions cannot be compared.

To check only implementation parity, add `--parity-only`. This avoids reporting accuracy for an incomplete training checkpoint.
