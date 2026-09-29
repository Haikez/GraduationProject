"""Run the local Mamba-inspired TCN/CA/DG prototype in BasicTS."""
import argparse

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig, BasicTSModelConfig
from basicts.models.MambaTCNCA import MambaTCNCADG
from basicts.runners.callback import GradientClipping

VARIANTS = {
    "tcn": (True, False, False, False),
    "tcn_ssm": (True, True, False, False),
    "tcn_ssm_ca": (True, True, True, False),
    "full": (True, True, True, True),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("PEMS04", "PEMS08"), default="PEMS04")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    switches = dict(zip(("use_tcn", "use_ssm", "use_ca", "use_dynamic_graph"), VARIANTS[args.variant]))
    config = BasicTSForecastingConfig(
        model=MambaTCNCADG,
        model_config=BasicTSModelConfig(
            num_nodes={"PEMS04": 307, "PEMS08": 170}[args.dataset],
            input_len=12, output_len=12, input_dim=1, hidden_dim=32,
            num_tcn_layers=2, num_heads=4, dropout=0.1, **switches,
        ),
        dataset_name=args.dataset, input_len=12, output_len=12,
        gpus=args.gpu, num_epochs=args.epochs, batch_size=args.batch_size,
        norm_each_channel=False, rescale=True, null_val=0.0,
        seed=args.seed, optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
        callbacks=[GradientClipping(5.0)], metrics=["MAE", "RMSE", "MAPE"],
        eval_horizons=[3, 6, 12],
        ckpt_save_dir=f"checkpoints/MambaTCNCADG/{args.dataset}_{args.variant}_s{args.seed}",
    )
    BasicTSLauncher.launch_training(config)


if __name__ == "__main__":
    main()
