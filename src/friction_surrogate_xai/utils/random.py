"""Random seed management for reproducible small-data experiments."""

from __future__ import annotations

import os
import random

import numpy as np

from friction_surrogate_xai.constants import DEFAULT_RANDOM_SEED


def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Seed common random number generators and return the seed used."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed

