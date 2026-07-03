"""Configurable integer-bin discretization of continuous input features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DiscretizedDataset:
    """Discretized dataframe plus metadata tables."""

    dataframe: pd.DataFrame
    metadata: pd.DataFrame
    comparison: pd.DataFrame


class DatasetDiscretizer:
    """Convert configured continuous input variables into integer bins."""

    def __init__(
        self,
        *,
        continuous_features: tuple[str, ...],
        method: str = "quantile",
        n_bins: int = 3,
        labels_start_at: int = 0,
        include_lowest: bool = True,
        duplicates: str = "drop",
        preserve_low_cardinality_integer_values: bool = True,
        low_cardinality_unique_threshold: int = 3,
        constant_bin_value: int = 0,
    ) -> None:
        self.continuous_features = continuous_features
        self.method = method
        self.n_bins = int(n_bins)
        self.labels_start_at = int(labels_start_at)
        self.include_lowest = include_lowest
        self.duplicates = duplicates
        self.preserve_low_cardinality_integer_values = preserve_low_cardinality_integer_values
        self.low_cardinality_unique_threshold = int(low_cardinality_unique_threshold)
        self.constant_bin_value = int(constant_bin_value)

    def transform(self, dataframe: pd.DataFrame, dataset_key: str) -> DiscretizedDataset:
        """Discretize configured features while preserving all other columns."""
        discrete = dataframe.copy()
        metadata_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []

        for feature in self.continuous_features:
            if feature not in dataframe.columns:
                continue
            if not pd.api.types.is_numeric_dtype(dataframe[feature]):
                metadata_rows.append(self._skipped_metadata(dataset_key, feature, "non_numeric"))
                continue

            original = pd.to_numeric(dataframe[feature], errors="coerce")
            binned, metadata = self._bin_series(original)
            discrete[feature] = binned.astype("Int64").astype(int)
            metadata_rows.append(
                {
                    "dataset_key": dataset_key,
                    "feature": feature,
                    **metadata,
                }
            )
            comparison_rows.append(
                {
                    "dataset_key": dataset_key,
                    "feature": feature,
                    "original_min": _safe_float(original.min()),
                    "original_max": _safe_float(original.max()),
                    "original_unique_count": int(original.nunique(dropna=False)),
                    "discrete_min": int(discrete[feature].min()),
                    "discrete_max": int(discrete[feature].max()),
                    "discrete_unique_count": int(discrete[feature].nunique(dropna=False)),
                    "changed_values": int((dataframe[feature] != discrete[feature]).sum()),
                }
            )

        return DiscretizedDataset(
            dataframe=discrete,
            metadata=pd.DataFrame(metadata_rows),
            comparison=pd.DataFrame(comparison_rows),
        )

    def _bin_series(self, series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
        non_null = series.dropna()
        unique_values = sorted(non_null.unique().tolist())
        unique_count = len(unique_values)

        if unique_count <= 1:
            binned = pd.Series(self.constant_bin_value, index=series.index)
            return binned, self._metadata(
                method="constant",
                unique_count=unique_count,
                actual_bins=1,
                bin_edges=[],
                mapping={str(value): self.constant_bin_value for value in unique_values},
                is_constant=True,
            )

        if (
            self.preserve_low_cardinality_integer_values
            and unique_count <= self.low_cardinality_unique_threshold
            and _all_integer_like(unique_values)
        ):
            mapping = {
                value: self.labels_start_at + index
                for index, value in enumerate(unique_values)
            }
            binned = series.map(mapping)
            return binned, self._metadata(
                method="unique_value_mapping",
                unique_count=unique_count,
                actual_bins=unique_count,
                bin_edges=[],
                mapping={str(key): int(value) for key, value in mapping.items()},
                is_constant=False,
            )

        if self.method == "quantile":
            binned, edges = pd.qcut(
                series,
                q=self.n_bins,
                labels=False,
                retbins=True,
                duplicates=self.duplicates,
            )
        elif self.method == "uniform":
            binned, edges = pd.cut(
                series,
                bins=self.n_bins,
                labels=False,
                retbins=True,
                include_lowest=self.include_lowest,
                duplicates=self.duplicates,
            )
        else:
            raise ValueError(f"Unsupported discretization method: {self.method}")

        binned = binned.astype("Int64") + self.labels_start_at
        binned = binned.fillna(self.constant_bin_value)
        actual_bins = int(pd.Series(binned).nunique(dropna=False))
        return binned, self._metadata(
            method=self.method,
            unique_count=unique_count,
            actual_bins=actual_bins,
            bin_edges=[_safe_float(edge) for edge in edges],
            mapping={},
            is_constant=False,
        )

    def _metadata(
        self,
        *,
        method: str,
        unique_count: int,
        actual_bins: int,
        bin_edges: list[float],
        mapping: dict[str, int],
        is_constant: bool,
    ) -> dict[str, Any]:
        return {
            "method": method,
            "requested_bins": self.n_bins,
            "actual_bins": actual_bins,
            "labels_start_at": self.labels_start_at,
            "unique_count_before": unique_count,
            "is_constant": is_constant,
            "bin_edges_json": json.dumps(bin_edges),
            "mapping_json": json.dumps(mapping, sort_keys=True),
        }

    @staticmethod
    def _skipped_metadata(dataset_key: str, feature: str, reason: str) -> dict[str, Any]:
        return {
            "dataset_key": dataset_key,
            "feature": feature,
            "method": "skipped",
            "requested_bins": None,
            "actual_bins": None,
            "labels_start_at": None,
            "unique_count_before": None,
            "is_constant": None,
            "bin_edges_json": "[]",
            "mapping_json": "{}",
            "skip_reason": reason,
        }


def _all_integer_like(values: list[Any]) -> bool:
    return all(float(value).is_integer() for value in values)


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric
