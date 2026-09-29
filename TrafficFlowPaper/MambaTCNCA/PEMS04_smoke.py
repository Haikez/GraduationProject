"""One-epoch PEMS04 smoke configuration for validating the training pipeline."""

from .config import build_config


CFG = build_config("PEMS04", 307, variant="full", num_epochs=1, batch_size=8, seed=0)
