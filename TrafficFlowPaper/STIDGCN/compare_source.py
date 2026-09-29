"""Compare a trained BasicTS STIDGCN checkpoint with the original PEMS08 run."""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import torch

from baselines.STIDGCN.arch import STIDGCN
from basicts.scaler import ZScoreScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--source-root', type=Path, default=Path('../STIDGCN'))
    parser.add_argument('--output', type=Path, default=Path('checkpoints/STIDGCN/comparison.json'))
    parser.add_argument('--parity-only', action='store_true', help='Check the implementations without reporting training metrics')
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['model_state_dict']
    basic = STIDGCN(3, 170, 48, 288).eval()
    basic.load_state_dict(state)

    spec = importlib.util.spec_from_file_location('original_stidgcn', source_root / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.STIDGCN('cpu', 3, 170, 48, 288).eval()
    original.load_state_dict(state)

    test = np.load(source_root / 'data/PEMS08/test.npz')
    source_x = test['x']
    basic_x = source_x.copy()
    basic_x[..., 2] /= 7
    scaler = ZScoreScaler('PEMS08-1', .6, False, True)
    mean, std = scaler.mean.item(), scaler.std.item()
    predictions = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    basic.to(device)
    original.to(device)
    max_abs = 0.0
    abs_sum = 0.0
    value_count = 0
    with torch.no_grad():
        for start in range(0, len(basic_x), 32):
            source_batch = torch.from_numpy(source_x[start:start + 32].copy()).float().to(device)
            batch = torch.from_numpy(basic_x[start:start + 32].copy()).float().to(device)
            source_batch[..., 0] = (source_batch[..., 0] - mean) / std
            batch[..., 0] = (batch[..., 0] - mean) / std
            original_prediction = original(source_batch.permute(0, 3, 2, 1))
            basic_prediction = basic(batch)
            difference = (original_prediction - basic_prediction).abs()
            max_abs = max(max_abs, float(difference.max()))
            abs_sum += float(difference.sum())
            value_count += difference.numel()
            if not args.parity_only:
                predictions.append((basic_prediction.squeeze(-1) * std + mean).cpu().numpy())
    same_weights = {'max_abs': max_abs, 'mean_abs': abs_sum / value_count,
                    'test_windows': len(basic_x)}
    if args.parity_only:
        report = {'checkpoint': str(args.checkpoint.resolve()),
                  'training_epoch': checkpoint.get('epoch'),
                  'same_checkpoint_prediction_difference': same_weights,
                  'note': 'Parity only. This checkpoint was produced before the weekday input fix; its accuracy does not represent the corrected model.'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return
    prediction = np.concatenate(predictions)
    target = test['y'][..., 0]
    per_horizon = []
    for h in range(12):
        mask = target[:, h] > 0
        error = prediction[:, h][mask] - target[:, h][mask]
        per_horizon.append({'horizon': h + 1, 'MAE': float(np.abs(error).mean()),
                            'RMSE': float(np.sqrt(np.square(error).mean())),
                            'MAPE': float((np.abs(error) / target[:, h][mask]).mean())})
    log = (source_root / 'data/PEMS08.log').read_text(encoding='utf-8', errors='replace')
    match = re.findall(r'On average over 12 horizons, Test MAE: ([0-9.]+), Test RMSE: ([0-9.]+), Test MAPE: ([0-9.]+), Test WMAPE: ([0-9.]+)', log)
    report = {'checkpoint': str(args.checkpoint.resolve()),
              'training_epoch': checkpoint.get('epoch'),
              'same_checkpoint_prediction_difference': same_weights,
              'basicts_on_source_test': {'MAE': float(np.mean([r['MAE'] for r in per_horizon])),
                                         'RMSE': float(np.mean([r['RMSE'] for r in per_horizon])),
                                         'MAPE': float(np.mean([r['MAPE'] for r in per_horizon])),
                                         'per_horizon': per_horizon},
              'original_logged_test': dict(zip(('MAE', 'RMSE', 'MAPE', 'WMAPE'), map(float, match[-1]))) if match else None,
              'note': 'Original trained checkpoint is unavailable; exact prediction parity is checked using the BasicTS checkpoint loaded into both implementations.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'basicts_on_source_test'}, indent=2))
    print('BasicTS on source test:', {key: value for key, value in report['basicts_on_source_test'].items() if key != 'per_horizon'})


if __name__ == '__main__':
    main()
