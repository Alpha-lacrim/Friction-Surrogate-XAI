"""Small utilities for EDA output generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def sanitize_filename(value: str) -> str:
    """Return a filesystem-friendly name derived from a column or dataset label."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return sanitized.strip("_").lower() or "unnamed"


def ensure_directory(path: Path) -> Path:
    """Create a directory and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    """Write a CSV file with UTF-8 encoding."""
    ensure_directory(path.parent)
    dataframe.to_csv(path, index=False, encoding="utf-8")
    return path


def write_matrix_csv(dataframe: pd.DataFrame, path: Path) -> Path:
    """Write a matrix-shaped CSV file with row labels."""
    ensure_directory(path.parent)
    dataframe.to_csv(path, encoding="utf-8")
    return path


def to_serializable(value: Any) -> Any:
    """Convert pandas/numpy scalar values to plain Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value

