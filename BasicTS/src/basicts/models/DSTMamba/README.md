# DST-Mamba in BasicTS

This directory adapts the local `DST-Mamba/baselines/DSTMamba` source model to
the BasicTS 1.x forecasting interface. The decomposition, bidirectional Mamba
encoder, spatial adapter, RevIN, and multi-scale trend branch follow the source.
The model takes `inputs` shaped `[batch, history, nodes]` and returns
`[batch, horizon, nodes]`. The default seasonal encoder ignores timestamps; `idmamba` and `stidgcn`
use BasicTS time-of-day and weekday timestamps.

The default `official` Mamba backend uses the source model's `mamba-ssm`
package and needs a supported Linux GPU environment. `stidgcn` does not use a
Mamba backend. On macOS or CPU, select
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

## STIDGCN-inspired seasonal ID-Mamba

`--seasonal-encoder idmamba` replaces the DST-Mamba seasonal embedding and
encoder with the full STIDGCN seasonal-processing path, applied to the
moving-average decomposition's seasonal residual:

1. A 1x1 convolution embeds the seasonal residual, time of day, and weekday
   together. Separate time-of-day and day-of-week embeddings from BasicTS
   timestamps are concatenated with those features.
2. The three-block IDGCN tree splits even/odd time steps twice (96 to 48 to
   24 for PEMS08), applies four temporal interaction filters in each block,
   reorders the four subseries, and adds the original input as a residual.
3. A GLU and full-history regression convolution produce the seasonal
   forecast. The original DST-Mamba trend branch can use `avg`.

Each source DGCN call inside an interaction block is replaced by a shared,
bidirectional Mamba scan over the **node axis** at each time step. A learned
node embedding supplies sensor identity. This preserves the tree's shapes and
interaction equations, but Mamba does not reproduce the original dynamic
adjacency or diffusion operation. The two-level tree needs an input length
divisible by 4. The default base width is 48 channels, giving 96 channels
inside the tree, as in the local BasicTS STIDGCN preset. PEMS08 uses 288
samples per day. Full ID-Mamba checkpoints use a distinct `_full` path from
the earlier one-level pilot.

```bash
python -m basicts.models.DSTMamba.train --task LTSF --dataset PEMS08 --gpus 0 \
  --trend-downsampling avg --seasonal-encoder idmamba --seed 2024
```

The original DST-Mamba seasonal encoder remains the default (`dst_mamba`).
The ID-Mamba mode enables BasicTS timestamps; source and other modes retain
their existing timestamp settings.

## Full STIDGCN seasonal replacement

`--seasonal-encoder stidgcn --trend-downsampling avg` replaces the entire
seasonal Mamba path with STIDGCN on the decomposed seasonal residual. It
reuses the BasicTS STIDGCN temporal embedding, IDGCN interaction blocks,
dynamic graph generator, diffusion GCN, GLU and regression head. The original
STIDGCN tree fixes its input to 12 steps; this adapter sizes its two tree
memories for the full 96-step DST-Mamba history (96 to 48 to 24). The trend
path remains the existing `avg` path, and the final per-node `tre_w` still
combines trend and seasonal forecasts. Time-of-day and weekday timestamps
are required. This is a 96-step hybrid experiment, not an evaluation of the
standalone 12-step STIDGCN model.

```bash
python -m basicts.models.DSTMamba.train --task LTSF --dataset PEMS08 --gpus 0 \
  --trend-downsampling avg --seasonal-encoder stidgcn --seed 2024
```

The `--stidgcn-channels`, `--stidgcn-granularity` and `--stidgcn-dropout`
options default to 48, 288 and 0.1, matching the local BasicTS STIDGCN
preset. A history length divisible by four is required.

### SCINet PEMS training profile for the STIDGCN hybrid

Use `--training-profile scinet_pems` to train this 12-to-12 hybrid with the
PEMS08 SCINet data and optimization recipe. It selects Adam (lr 0.001,
betas 0.9/0.999, weight decay 1e-5), normalized MAE loss, batch size 8,
80 epochs, seed 4321, and an epoch-wise 0.95 learning-rate multiplier.
SCINet's separate five-epoch 0.5 scheduler does not change its effective
default learning rate because `lradj=1` resets the LR at each epoch start.
This profile removes early stopping and gradient clipping; validation and
test run each epoch, and the best checkpoint is chosen by raw-scale
validation MAE. The checkpoint directory ends in `_scinet_pems_v2`.

```bash
PYTHONPATH=src python -m basicts.models.DSTMamba.train \
  --task LTSF --dataset PEMS08 --gpus 0 \
  --trend-downsampling avg --seasonal-encoder stidgcn \
  --training-profile scinet_pems
```

The profile uses SCINet's chronological 60/20/20 split, input/output length
12/12, one-step train/validation windows, twelve-step test windows, and
`normtype=0` z-score: each split computes its own per-node mean and standard
deviation. Statistics are computed before forward/backward filling missing
values. The model still receives the BasicTS time-of-day and weekday inputs
needed by its STIDGCN seasonal branch. The 12-to-12 checkpoints cannot be
reused by the 96-to-12 experiment. Split-specific validation/test statistics
also mean these metrics should not be compared directly with train-only
normalization experiments.

## Match the BasicTS STIDGCN training recipe

The `stidgcn_basicts` profile uses the exact data and optimization settings
saved in the standalone BasicTS STIDGCN PEMS08 run. It keeps the DST-Mamba
hybrid architecture (average-pooled trend plus STIDGCN seasonal branch):

```bash
PYTHONPATH=src python -m basicts.models.DSTMamba.train \
  --task LTSF --dataset PEMS08 --gpus 0 \
  --trend-downsampling avg --seasonal-encoder stidgcn \
  --training-profile stidgcn_basicts --input-len 12 --output-len 12
```

The profile uses the BasicTS PEMS08 chronological splits with stride-one
windows in all splits, a scalar Z-score fitted on the training split, and
zero-value masking. It trains for 300 epochs with batch size 64 and seed 42,
using Ranger (learning rate 0.001, weight decay 0.0001), MAE loss, gradient
clipping at norm 5, no LR scheduler, and validation/test every epoch.
It selects the best checkpoint by validation MAE. The profile explicitly
disables TF32 to match the saved STIDGCN configuration, regardless of the
framework default. `--cudnn-tf32` is an optional faster precision variant
with a separate checkpoint directory; it is not an exact match to the
standalone run. Use `--tf32` to enable both GPU TF32 backends from the
first epoch in a separate `_full_tf32` checkpoint directory.

## SCI trend downsampling

`--trend-downsampling` selects the trend path:

- `source` (default) keeps the original implementation: downsample the raw
  series, then decompose each scale.
- `avg` decomposes the input once and repeatedly average-pools the trend.
- `sci_low` decomposes the input once, then applies the node-shared SCINet
  Interactor to the trend. At each level, the updated even and odd streams
  produce `coarse=(even+odd)/2` and `detail=(even-odd)/2`. Only the coarse
  stream goes into the multiscale trend mixer; `detail` is discarded.
- `sci` follows `sci_low` but predicts from each level's detail stream with
  a separate linear time mapping. Its three detail forecasts are added to the
  multiscale coarse forecast before the original per-node `tre_w` combines
  trend and seasonal forecasts. There is no average-pooling blend or gate.

The four SCI interaction filters use SCINet's convolution layout and affine
updates. Their last convolutions are initialized to zero, so the initial
coarse stream is exactly the two-point average of the input trend. The detail
prediction mappings in `sci` are also initialized to zero. This gives both
SCI modes the same initial predictions as `avg` when all shared weights are
identical, while allowing the SCI filters and detail heads to learn afterward.
The seasonal Mamba branch is unchanged.

These modes are SCI-inspired adaptations rather than the full SCINet tree,
which retains both streams at every split. Filter defaults are 8 hidden
channels, kernel size 3, and zero dropout, as in the earlier DST-Mamba SCI
comparison. SCI modes require `ds_window=2` and an input length divisible by
`2**ds_layers`. Earlier SCI checkpoint variants use separate paths and are
not load-compatible with these modes.

For a matched PEMS08 comparison from the BasicTS repository root:

```bash
python -m basicts.models.DSTMamba.train --task LTSF --dataset PEMS08 --gpus 0 \
  --trend-downsampling avg --seed 2024
python -m basicts.models.DSTMamba.train --task LTSF --dataset PEMS08 --gpus 0 \
  --trend-downsampling sci_low --seed 2024
python -m basicts.models.DSTMamba.train --task LTSF --dataset PEMS08 --gpus 0 \
  --trend-downsampling sci --seed 2024
```

Use `--output-len` to evaluate other horizons. For a CPU smoke check, append
`--mamba-backend torch --gpus cpu --steps 1 --batch-size 2 --d-model 32
--d-ff 64 --node-dim 8 --rank 4 --e-layers 1` to either command. Reported
performance gains require full matched training runs; a smoke check only
confirms that the model can execute and backpropagate.
