# Mamba-TCN-CA-DG MVP

This folder contains the first executable prototype for the thesis model:

1. a causal TCN branch for local temporal trends;
2. an input-selective diagonal state-space branch for a linear-time long-context path;
3. bidirectional cross-attention between alternating temporal sub-sequences; and
4. a gated fusion of instance-driven and pattern-bank dynamic graphs.

The state-space branch is deliberately implemented with stock PyTorch. It is
Mamba-inspired, but it is not the official `mamba_ssm.Mamba` kernel. Use it to
validate shapes, ablation logic, and the proposed interaction mechanism. Do not
label its outcomes as Mamba results in the dissertation.

## Quick start

Use the project virtual environment from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\train.py -c baselines/MambaTCNCA/PEMS04_smoke.py -g 0
```

For the complete 100-epoch experiment, select the ablation and seed through
environment variables:

```powershell
$env:MTCNCA_VARIANT = 'full' # tcn, tcn_ssm, tcn_ssm_ca, or full
$env:MTCNCA_EPOCHS = '100'
$env:MTCNCA_BATCH_SIZE = '16'
$env:MTCNCA_SEED = '0'
.\.venv\Scripts\python.exe experiments\train.py -c baselines/MambaTCNCA/PEMS04.py -g 0
```

`PEMS08.py` targets the repository's local `datasets/PEMS08-1` snapshot, whose
metadata identifies the data as standard PEMS08.
