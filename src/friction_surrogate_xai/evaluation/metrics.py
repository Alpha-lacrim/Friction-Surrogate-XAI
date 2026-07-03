"""Regression metric computation and cross-validation summaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from friction_surrogate_xai.evaluation.arrays import validate_prediction_arrays
from friction_surrogate_xai.evaluation.statistics import confidence_interval

REGRESSION_METRICS = ("r2", "rmse", "nrmse", "mae")
ERROR_METRICS = ("rmse", "nrmse", "mae")


class RegressionMetricCalculator:
    """Compute reusable regression metrics for one or more targets."""

    def __init__(
        self,
        metrics: Sequence[str] = REGRESSION_METRICS,
        nrmse_denominator: str = "range",
        aggregate_targets: bool = True,
        confidence_level: float = 0.95,
        relative_gap_epsilon: float = 1.0e-12,
        stability_index_epsilon: float = 1.0e-12,
    ) -> None:
        unsupported = sorted(set(metrics).difference(REGRESSION_METRICS))
        if unsupported:
            raise ValueError(f"Unsupported regression metric(s): {unsupported}")
        self.metrics = tuple(metrics)
        self.nrmse_denominator = nrmse_denominator
        self.aggregate_targets = aggregate_targets
        self.confidence_level = confidence_level
        self.relative_gap_epsilon = relative_gap_epsilon
        self.stability_index_epsilon = stability_index_epsilon

    def evaluate_predictions(
        self,
        y_true: Any,
        y_pred: Any,
        *,
        target_names: Sequence[str] | None = None,
        dataset_key: str = "",
        model_name: str = "",
        split: str = "test",
    ) -> pd.DataFrame:
        """Evaluate predictions and return one wide row per target."""
        true_array, pred_array, names = validate_prediction_arrays(
            y_true,
            y_pred,
            target_names=target_names,
        )

        rows: list[dict[str, Any]] = []
        for target_index, target_name in enumerate(names):
            rows.append(
                self._row(
                    y_true=true_array[:, target_index],
                    y_pred=pred_array[:, target_index],
                    target_name=target_name,
                    dataset_key=dataset_key,
                    model_name=model_name,
                    split=split,
                )
            )

        if self.aggregate_targets and len(names) > 1:
            rows.append(
                self._row(
                    y_true=true_array.ravel(),
                    y_pred=pred_array.ravel(),
                    target_name="__aggregate__",
                    dataset_key=dataset_key,
                    model_name=model_name,
                    split=split,
                )
            )

        return pd.DataFrame(rows)

    def to_long_format(self, metrics_table: pd.DataFrame) -> pd.DataFrame:
        """Convert a wide metric table to canonical long format."""
        id_columns = [
            column
            for column in ("dataset_key", "model_name", "split", "target", "sample_count")
            if column in metrics_table.columns
        ]
        metric_columns = [metric for metric in self.metrics if metric in metrics_table.columns]
        if metrics_table.empty:
            return pd.DataFrame(columns=[*id_columns, "metric", "value"])
        return metrics_table.melt(
            id_vars=id_columns,
            value_vars=metric_columns,
            var_name="metric",
            value_name="value",
        )

    def train_test_gap(
        self,
        train_metrics: pd.DataFrame,
        test_metrics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute train/test gaps, using a positive gap as possible overfitting risk."""
        join_columns = [
            column for column in ("dataset_key", "model_name", "target") if column in test_metrics.columns
        ]
        merged = train_metrics.merge(
            test_metrics,
            on=join_columns,
            suffixes=("_train", "_test"),
        )

        rows: list[dict[str, Any]] = []
        for _, row in merged.iterrows():
            for metric in self.metrics:
                train_value = row.get(f"{metric}_train")
                test_value = row.get(f"{metric}_test")
                direction = ""
                if not _is_number(train_value) or not _is_number(test_value):
                    gap = np.nan
                    relative_gap = np.nan
                else:
                    if metric in ERROR_METRICS:
                        gap = float(test_value) - float(train_value)
                        direction = "test_minus_train"
                    else:
                        gap = float(train_value) - float(test_value)
                        direction = "train_minus_test"
                    denominator = max(abs(float(train_value)), self.relative_gap_epsilon)
                    relative_gap = gap / denominator
                rows.append(
                    {
                        "dataset_key": row.get("dataset_key", ""),
                        "model_name": row.get("model_name", ""),
                        "target": row.get("target", ""),
                        "metric": metric,
                        "train_value": train_value,
                        "test_value": test_value,
                        "gap": gap,
                        "relative_gap": relative_gap,
                        "gap_direction": direction,
                        "positive_gap_indicates": "possible_overfitting",
                    }
                )
        return pd.DataFrame(rows)

    def summarize_cross_validation(self, fold_metrics: pd.DataFrame) -> pd.DataFrame:
        """Summarize fold metrics with mean, standard deviation, CI, and stability."""
        long_metrics = self._fold_metrics_to_long(fold_metrics)
        if long_metrics.empty:
            return pd.DataFrame(
                columns=[
                    "dataset_key",
                    "model_name",
                    "target",
                    "metric",
                    "fold_count",
                    "mean",
                    "std",
                    "confidence_level",
                    "ci_lower",
                    "ci_upper",
                    "ci_margin",
                    "min",
                    "max",
                    "stability_index",
                ]
            )

        group_columns = [
            column
            for column in ("dataset_key", "model_name", "target", "metric")
            if column in long_metrics.columns
        ]
        rows: list[dict[str, Any]] = []
        for group_values, group in long_metrics.groupby(group_columns, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy()
            interval = confidence_interval(values, confidence_level=self.confidence_level)
            mean = interval.mean
            std = interval.std
            if np.isfinite(mean):
                denominator = max(abs(mean), self.stability_index_epsilon)
                stability_index = std / denominator if np.isfinite(std) else np.nan
            else:
                stability_index = np.nan

            row = dict(zip(group_columns, group_values, strict=False))
            row.update(
                {
                    "fold_count": interval.count,
                    "mean": mean,
                    "std": std,
                    "confidence_level": interval.confidence_level,
                    "ci_lower": interval.lower,
                    "ci_upper": interval.upper,
                    "ci_margin": interval.margin_of_error,
                    "min": float(np.min(values)) if values.size else np.nan,
                    "max": float(np.max(values)) if values.size else np.nan,
                    "stability_index": stability_index,
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def _row(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_name: str,
        dataset_key: str,
        model_name: str,
        split: str,
    ) -> dict[str, Any]:
        metric_values = self._single_target_metrics(y_true, y_pred)
        return {
            "dataset_key": dataset_key,
            "model_name": model_name,
            "split": split,
            "target": target_name,
            "sample_count": int(y_true.size),
            **metric_values,
        }

    def _single_target_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        values: dict[str, float] = {}
        if "r2" in self.metrics:
            values["r2"] = float(r2_score(y_true, y_pred))
        if "rmse" in self.metrics or "nrmse" in self.metrics:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if "rmse" in self.metrics:
                values["rmse"] = rmse
            if "nrmse" in self.metrics:
                denominator = self._nrmse_denominator(y_true)
                values["nrmse"] = rmse / denominator if denominator > 0 else np.nan
        if "mae" in self.metrics:
            values["mae"] = float(mean_absolute_error(y_true, y_pred))
        return values

    def _nrmse_denominator(self, y_true: np.ndarray) -> float:
        if self.nrmse_denominator == "range":
            return float(np.nanmax(y_true) - np.nanmin(y_true))
        if self.nrmse_denominator == "std":
            return float(np.nanstd(y_true, ddof=1))
        if self.nrmse_denominator == "mean_abs":
            return float(abs(np.nanmean(y_true)))
        if self.nrmse_denominator == "iqr":
            return float(np.nanpercentile(y_true, 75) - np.nanpercentile(y_true, 25))
        raise ValueError(f"Unsupported NRMSE denominator: {self.nrmse_denominator}")

    def _fold_metrics_to_long(self, fold_metrics: pd.DataFrame) -> pd.DataFrame:
        if fold_metrics.empty:
            return pd.DataFrame()
        if {"metric", "value"}.issubset(fold_metrics.columns):
            return fold_metrics.copy()

        id_columns = [
            column
            for column in ("dataset_key", "model_name", "target", "fold", "fold_id", "seed", "split")
            if column in fold_metrics.columns
        ]
        metric_columns = [metric for metric in self.metrics if metric in fold_metrics.columns]
        if not metric_columns:
            raise ValueError(
                "fold_metrics must contain either metric/value columns or wide metric columns."
            )
        return fold_metrics.melt(
            id_vars=id_columns,
            value_vars=metric_columns,
            var_name="metric",
            value_name="value",
        )


def _is_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
