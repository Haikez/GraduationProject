# SCINet for BasicTS

This package adapts the SCINet source in `/Users/haikez/Desktop/SCINet/models/SCINet.py` to BasicTS 1.x. The tree, even/odd interaction blocks, temporal projections, optional positional encoding and reversible normalization retain the source implementation. The adapter receives and returns tensors shaped `[batch, time, nodes]`. With two stacks it returns the final forecast and an intermediate forecast; the training entry point applies MAE to both as in the source experiment.

Run from the BasicTS repository root in the `basicts` environment:

```bash
PYTHONPATH=src python -m basicts.models.SCINet.train --dataset PEMS08 --gpus 0
PYTHONPATH=src python -m basicts.models.SCINet.train --dataset PEMS04 --gpus 0 --stacks 2
PYTHONPATH=src python -m basicts.models.SCINet.train --dataset ETTh1 --gpus 0 --input-len 96 --output-len 96
```

For a quick training check use `--steps 2 --batch-size 2`. The defaults follow the source PEMS experiment: 12 history and forecast steps, two tree levels, one stack, hidden ratio 0.0625, kernel 5, dropout 0.5, batch size 8, 80 epochs and Adam at 0.001. BasicTS prepares the PEMS datasets and handles scaling, masking, metrics and checkpoints. Its train-fitted scaler and dataset windows differ from the source experiment's per-split normalization, so reported scores are not directly comparable.

The entry point infers the number of series from the prepared dataset's `train_data.npy`. Set `--data-dir` when it is outside `datasets/<name>`. The hidden width has a minimum of one channel, so the PEMS default ratio also works with small multivariate datasets.

Source: original SCINet implementation, Apache-2.0 licensed; this BasicTS repository also carries an Apache-2.0 `LICENSE`. Changes here remove the stand-alone command-line demo, make the multi-layer decoder device independent, and provide shape/config validation.
