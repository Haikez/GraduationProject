# DST-Mamba in BasicTS

This directory adapts the local `DST-Mamba/baselines/DSTMamba` source model to
the BasicTS 1.x forecasting interface. The decomposition, bidirectional Mamba
encoder, spatial adapter, RevIN, and multi-scale trend branch follow the source.
The model takes `inputs` shaped `[batch, history, nodes]` and returns
`[batch, horizon, nodes]`. Timestamps are accepted but unused.

The default `official` backend uses the source model's `mamba-ssm` package and
needs a supported Linux GPU environment. On macOS or CPU, select
`--mamba-backend torch` to use the slower, unfused PyTorch Mamba-1 recurrence.
This backend has its own weights and should not be used as a drop-in loader for
checkpoints from `mamba-ssm`.

From the BasicTS repository root, with the prepared datasets present:

```bash
python -m basicts.models.DSTMamba.train --task LTSF --dataset PEMS08 --gpus 0
python -m basicts.models.DSTMamba.train --task STF --dataset METR-LA --gpus 0
```

For a short local CPU check with a smaller model:

```bash
python -m basicts.models.DSTMamba.train --task LTSF --dataset PEMS08 \
  --mamba-backend torch --gpus cpu --steps 1 --batch-size 2 \
  --d-model 32 --d-ff 64 --node-dim 8 --rank 4 --e-layers 1
```

The two presets carry over the local source settings: PEMS08 96-to-12 with
MSE and two encoder layers; METR-LA 12-to-12 with Huber and three layers.
BasicTS handles data preparation, scaling, masking, metrics, callbacks, and
checkpoints. Its split and loss scaling conventions can differ from the source
runner, so the resulting scores are not automatically comparable.
