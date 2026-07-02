"""Logging configuration helpers."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

from friction_surrogate_xai.config.loader import load_yaml


def setup_logging(config_path: str | Path = "configs/logging.yaml") -> None:
    """Configure Python logging from the project YAML file."""
    config = load_yaml(config_path)
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Return a standard library logger."""
    return logging.getLogger(name)

