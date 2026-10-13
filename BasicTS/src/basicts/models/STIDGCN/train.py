"""Run STIDGCN on a PEMS dataset from the BasicTS repository root."""
import argparse

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig, BasicTSModelConfig
from basicts.models.STIDGCN import STIDGCN
from basicts.models.STIDGCN.ranger21 import Ranger
from basicts.runners.callback import GradientClipping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PEMS03", "PEMS04", "PEMS07", "PEMS08"), default="PEMS08")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--eval-horizons", type=int, nargs="+", default=[3, 6, 12, 24])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--model-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--compile-model", action="store_true")
    args = parser.parse_args()
    if args.input_len < 4 or args.input_len % 4:
        parser.error("--input-len must be positive and divisible by 4")
    if args.output_len < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("--output-len, --epochs and --batch-size must be positive")
    if any(h < 1 or h > args.output_len for h in args.eval_horizons):
        parser.error("--eval-horizons must be between 1 and --output-len")
    nodes = {"PEMS03": 358, "PEMS04": 307, "PEMS07": 883, "PEMS08": 170}[args.dataset]
    config = BasicTSForecastingConfig(
        model=STIDGCN,
        model_config=BasicTSModelConfig(
            num_nodes=nodes, channels=48, dropout=0.1,
            input_len=args.input_len, output_len=args.output_len,
        ),
        dataset_name=args.dataset, input_len=args.input_len, output_len=args.output_len,
        use_timestamps=True, gpus=args.gpu, num_epochs=args.epochs,
        norm_each_channel=False, rescale=True, null_val=0.0,
        batch_size=args.batch_size, optimizer=Ranger,
        optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
        callbacks=[GradientClipping(5.0)],
        metrics=["MAE", "RMSE", "MAPE"], eval_horizons=args.eval_horizons,
        seed=args.seed, ckpt_save_dir=args.checkpoint_dir, tf32=True,
        **({"model_dtype": args.model_dtype} if args.model_dtype != "float32" else {}),
        compile_model=args.compile_model,
    )
    BasicTSLauncher.launch_training(config)


if __name__ == "__main__":
    main()
