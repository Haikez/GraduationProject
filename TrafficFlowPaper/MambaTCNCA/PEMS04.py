"""PEMS04 experiment configuration.

Set ``MTCNCA_VARIANT`` to one of tcn, tcn_ssm, tcn_ssm_ca, or full before
launching BasicTS. Set ``MTCNCA_EPOCHS``, ``MTCNCA_BATCH_SIZE``, and
``MTCNCA_SEED`` for reproducible pilot or full runs.
"""

import os

from .config import build_config


CFG = build_config(
    "PEMS04",
    307,
    variant=os.getenv("MTCNCA_VARIANT", "full"),
    num_epochs=int(os.getenv("MTCNCA_EPOCHS", "100")),
    batch_size=int(os.getenv("MTCNCA_BATCH_SIZE", "16")),
    seed=int(os.getenv("MTCNCA_SEED", "0")),
)
