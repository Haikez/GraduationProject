"""Run STIDGCN on a PEMS dataset from the BasicTS repository root."""
import argparse

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig, BasicTSModelConfig
from basicts.models.STIDGCN import STIDGCN
from basicts.models.STIDGCN.ranger21 import Ranger
from basicts.runners.callback import GradientClipping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"), default="PEMS04")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    nodes = {"PEMS03": 358, "PEMS04": 307, "PEMS07": 883, "PEMS08": 170}[args.dataset]
    config = BasicTSForecastingConfig(
        model=STIDGCN,
        model_config=BasicTSModelConfig(num_nodes=nodes, channels=48, dropout=0.1),
        dataset_name=args.dataset, input_len=12, output_len=12,
        use_timestamps=True, gpus=args.gpu, num_epochs=args.epochs,
        norm_each_channel=False, rescale=True, null_val=0.0,
        batch_size=args.batch_size, optimizer=Ranger,
        optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
        callbacks=[GradientClipping(5.0)],
        metrics=["MAE", "RMSE", "MAPE"], eval_horizons=[3, 6, 12],
    )
    BasicTSLauncher.launch_training(config)


if __name__ == "__main__":
    main()
