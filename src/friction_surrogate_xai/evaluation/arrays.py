"""Array normalization helpers for evaluation inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def as_2d_array(values: Any, name: str) -> np.ndarray:
    """Return values as a numeric 2D numpy array."""
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional, got shape {array.shape}.")

    try:
        return array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc


def infer_target_names(y_true: Any, target_names: Sequence[str] | None = None) -> tuple[str, ...]:
    """Infer target names from explicit names or pandas objects."""
    target_count = as_2d_array(y_true, "y_true").shape[1]

    if target_names is not None:
        names = tuple(str(name) for name in target_names)
        if len(names) != target_count:
            raise ValueError(
                f"target_names has {len(names)} names, but y_true has {target_count} target(s)."
            )
        return names

    if isinstance(y_true, pd.DataFrame):
        return tuple(str(column) for column in y_true.columns)
    if isinstance(y_true, pd.Series) and y_true.name is not None:
        return (str(y_true.name),)
    return tuple(f"target_{index}" for index in range(target_count))


def validate_prediction_arrays(
    y_true: Any,
    y_pred: Any,
    target_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Validate y_true/y_pred compatibility and return normalized arrays."""
    true_array = as_2d_array(y_true, "y_true")
    pred_array = as_2d_array(y_pred, "y_pred")
    if true_array.shape != pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape; "
            f"got {true_array.shape} and {pred_array.shape}."
        )
    names = infer_target_names(y_true, target_names=target_names)
    return true_array, pred_array, names
