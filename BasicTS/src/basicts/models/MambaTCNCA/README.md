# STIDGCN with Mamba-TCN and cross-attention

This model preserves the local STIDGCN encoder and decoder:

1. flow, time-of-day and day-of-week input with the original temporal embedding;
2. a three-node, two-level IDGCN tree with the original DGCN graph generator,
   diffusion step, memory weighting and even/odd realignment;
3. the original GLU residual and convolutional regression head.

Inside each IDGCN node, its four temporal-convolution blocks are replaced by
parallel causal TCN and Mamba branches with a learned fusion gate. The two
Hadamard interactions are replaced by cross-attention: the odd sub-sequence
queries features derived from the even sub-sequence, and vice versa. The
remaining IDGCN update equations and DGCN calls keep their STIDGCN order.

The model is implemented as a BasicTS 1.x model/config package under
`src/basicts/models/MambaTCNCA`. It expects 12 historical steps, BasicTS
`inputs` with shape `[batch, 12, nodes]`, and `inputs_timestamps` with two
channels. It returns `[batch, output_len, nodes]`.

## Run

Run from the BasicTS root with the `basicts` environment:

```bash
PYTHONPATH=src python -m basicts.models.MambaTCNCA.train --dataset PEMS08 --variant full --gpus 0
PYTHONPATH=src python -m basicts.models.MambaTCNCA.train --dataset PEMS04 --variant full --gpus 0
```

The default uses 300 epochs, Ranger, batch size 16, 12 input/output steps,
BasicTS normalization, and MAE/RMSE/MAPE. This model now defaults to 32 base
channels (64 channels in the tree); the local STIDGCN source uses 48 base
channels. Reduce `--batch-size` if the
three tree nodes exceed available GPU memory.

For a brief training check, add `--steps 2 --batch-size 2`. The `--steps`
mode skips validation, testing, and checkpoint selection.

The variants keep the STIDGCN graph, embedding and regression head:

- `full`: Mamba-TCN and cross-attention;
- `without_mamba`: TCN and cross-attention;
- `without_tcn`: Mamba and cross-attention;
- `without_ca`: Mamba-TCN and original Hadamard interactions.

The default `--mamba-backend torch` runs an unfused PyTorch Mamba-1 selective
recurrence. `--mamba-backend official` uses `mamba_ssm.Mamba` when the official
[Mamba package](https://github.com/state-spaces/mamba) is installed. Record
the backend in experiment reports; they have different performance and
checkpoint parameters. The earlier standalone prototype and previous BasicTS
model used a different architecture, so their checkpoints are incompatible.

The source STIDGCN tree's third memory tensor is retained, including its
original behavior of sharing the second memory tensor with both child nodes.
This preserves the source topology while the two requested operators change.
