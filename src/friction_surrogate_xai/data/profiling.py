"""Dataset profiling and report generation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.data.contracts import (
    ConstantFeatureReport,
    DatasetMetadata,
    DatasetSchema,
    DescriptiveStatistics,
    DuplicateReport,
    MissingValueReport,
)


def _to_python(value: Any) -> Any:
    """Convert pandas/numpy scalar values into plain Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class MissingValueReporter:
    """Generate missing-value reports."""

    def report(self, dataframe: pd.DataFrame) -> MissingValueReport:
        """Return missing counts and ratios by column."""
        counts = {column: int(count) for column, count in dataframe.isna().sum().items()}
        row_count = max(len(dataframe), 1)
        ratios = {column: count / row_count for column, count in counts.items()}
        total_missing = int(sum(counts.values()))
        columns_with_missing = tuple(column for column, count in counts.items() if count > 0)
        return MissingValueReport(
            counts=counts,
            ratios=ratios,
            total_missing=total_missing,
            columns_with_missing=columns_with_missing,
        )


class DuplicateDetector:
    """Detect full-row and identifier duplicates."""

    def __init__(self, check_full_rows: bool = True, check_id_column: bool = True) -> None:
        self.check_full_rows = check_full_rows
        self.check_id_column = check_id_column

    def report(self, dataframe: pd.DataFrame, id_column: str) -> DuplicateReport:
        """Return duplicate diagnostics without modifying the dataframe."""
        full_duplicate_count = 0
        full_duplicate_indices: tuple[int, ...] = ()
        if self.check_full_rows:
            full_duplicate_mask = dataframe.duplicated(keep=False)
            full_duplicate_count = int(dataframe.duplicated().sum())
            full_duplicate_indices = tuple(int(index) for index in dataframe.index[full_duplicate_mask])

        id_duplicate_count = 0
        id_duplicate_values: tuple[Any, ...] = ()
        if self.check_id_column and id_column in dataframe.columns:
            id_duplicate_mask = dataframe[id_column].duplicated(keep=False)
            id_duplicate_count = int(dataframe[id_column].duplicated().sum())
            values = dataframe.loc[id_duplicate_mask, id_column].dropna().unique()
            id_duplicate_values = tuple(_to_python(value) for value in values)

        return DuplicateReport(
            full_duplicate_count=full_duplicate_count,
            full_duplicate_indices=full_duplicate_indices,
            id_duplicate_count=id_duplicate_count,
            id_duplicate_values=id_duplicate_values,
        )


class ConstantFeatureDetector:
    """Detect constant columns and configured constant features."""

    def __init__(self, dropna: bool = False) -> None:
        self.dropna = dropna

    def report(self, dataframe: pd.DataFrame, feature_columns: Iterable[str]) -> ConstantFeatureReport:
        """Return constant-column information for all columns and feature columns."""
        unique_counts = {
            column: int(dataframe[column].nunique(dropna=self.dropna)) for column in dataframe.columns
        }
        constant_columns = tuple(column for column, count in unique_counts.items() if count <= 1)
        configured_features = tuple(column for column in feature_columns if column in dataframe.columns)
        constant_feature_columns = tuple(
            column for column in configured_features if column in constant_columns
        )
        variable_feature_columns = tuple(
            column for column in configured_features if column not in constant_columns
        )
        constant_values = {
            column: self._constant_value(dataframe[column]) for column in constant_columns
        }
        return ConstantFeatureReport(
            constant_columns=constant_columns,
            constant_feature_columns=constant_feature_columns,
            variable_feature_columns=variable_feature_columns,
            unique_counts=unique_counts,
            constant_values=constant_values,
        )

    def _constant_value(self, series: pd.Series) -> Any:
        values = series.dropna().unique() if self.dropna else series.unique()
        if len(values) == 0:
            return None
        return _to_python(values[0])


class DescriptiveStatisticsGenerator:
    """Generate descriptive statistics for numeric and non-numeric columns."""

    def __init__(self, percentiles: Iterable[float] = (0.25, 0.5, 0.75)) -> None:
        self.percentiles = tuple(percentiles)

    def report(self, dataframe: pd.DataFrame) -> DescriptiveStatistics:
        """Return descriptive statistics without changing the dataframe."""
        numeric: dict[str, dict[str, Any]] = {}
        numeric_frame = dataframe.select_dtypes(include="number")
        for column in numeric_frame.columns:
            series = numeric_frame[column]
            mode = series.mode(dropna=False)
            numeric[column] = {
                "count": int(series.count()),
                "mean": _to_python(series.mean()),
                "median": _to_python(series.median()),
                "mode": _to_python(mode.iloc[0]) if not mode.empty else None,
                "variance": _to_python(series.var()),
                "std": _to_python(series.std()),
                "min": _to_python(series.min()),
                "q1": _to_python(series.quantile(0.25)),
                "q3": _to_python(series.quantile(0.75)),
                "max": _to_python(series.max()),
                "skewness": _to_python(series.skew()),
                "kurtosis": _to_python(series.kurt()),
                "percentiles": {
                    str(percentile): _to_python(series.quantile(percentile))
                    for percentile in self.percentiles
                },
            }

        non_numeric: dict[str, dict[str, Any]] = {}
        non_numeric_frame = dataframe.select_dtypes(exclude="number")
        for column in non_numeric_frame.columns:
            series = non_numeric_frame[column]
            mode = series.mode(dropna=False)
            non_numeric[column] = {
                "count": int(series.count()),
                "unique": int(series.nunique(dropna=False)),
                "mode": _to_python(mode.iloc[0]) if not mode.empty else None,
            }

        return DescriptiveStatistics(numeric=numeric, non_numeric=non_numeric)


class MetadataGenerator:
    """Generate lightweight metadata for loaded datasets."""

    def report(
        self,
        dataframe: pd.DataFrame,
        dataset_key: str,
        path: Path,
        sheet_name: str,
        schema: DatasetSchema,
    ) -> DatasetMetadata:
        """Return generated metadata for one dataframe."""
        return DatasetMetadata(
            key=dataset_key,
            path=path,
            sheet_name=sheet_name,
            rows=int(dataframe.shape[0]),
            columns=int(dataframe.shape[1]),
            column_names=tuple(dataframe.columns),
            dtypes={column: str(dtype) for column, dtype in dataframe.dtypes.items()},
            id_column=schema.id_column,
            feature_columns=schema.feature_columns,
            target_columns=schema.target_columns_for(dataset_key),
        )
