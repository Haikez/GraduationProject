"""PEMS08 experiment configuration for the local PEMS08-1 dataset snapshot."""

import os

from .config import build_config


CFG = build_config(
    "PEMS08-1",
    170,
    variant=os.getenv("MTCNCA_VARIANT", "full"),
    num_epochs=int(os.getenv("MTCNCA_EPOCHS", "100")),
    batch_size=int(os.getenv("MTCNCA_BATCH_SIZE", "16")),
    seed=int(os.getenv("MTCNCA_SEED", "0")),
)
