"""Project-wide constants that are safe to import from any module."""

from pathlib import Path

PACKAGE_NAME = "friction_surrogate_xai"
PROJECT_NAME = "friction-surrogate-xai"
DEFAULT_RANDOM_SEED = 42
DEFAULT_REPEATED_SEEDS = (7, 21, 42, 84, 168)

CONFIG_DIRNAME = "configs"
RAW_DATA_DIR = Path("data/raw")

