"""Typed data-layer contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetSchema:
    """Project-wide tabular schema loaded from configuration."""

    id_column: str
    feature_columns: tuple[str, ...]
    common_target_columns: tuple[str, ...]
    special_target_columns: dict[str, tuple[str, ...]]
    column_types: dict[str, str]
    value_constraints: dict[str, tuple[Any, ...]]

    def target_columns_for(self, dataset_key: str) -> tuple[str, ...]:
        """Return common plus dataset-specific target columns."""
        return self.common_target_columns + self.special_target_columns.get(dataset_key, ())

    def expected_columns_for(self, dataset_key: str) -> tuple[str, ...]:
        """Return the exact expected column contract for a dataset."""
        return (self.id_column,) + self.feature_columns + self.target_columns_for(dataset_key)


@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset file and shape metadata loaded from configuration."""

    key: str
    filename: str
    sheet_name: str
    raw_path: Path
    rows: int
    columns: int
    has_missing_values: bool
    constant_features: tuple[str, ...] = ()
    special_targets: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def resolved_path(self, root: Path) -> Path:
        """Resolve the configured raw path relative to a project root."""
        return self.raw_path if self.raw_path.is_absolute() else root / self.raw_path


@dataclass(frozen=True)
class ValidationResult:
    """Validation errors and warnings collected during loading."""

    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Return true when no blocking validation errors were detected."""
        return not self.errors

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Combine two validation results."""
        return ValidationResult(
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


@dataclass(frozen=True)
class MissingValueReport:
    """Missing-value counts and ratios for one dataset."""

    counts: dict[str, int]
    ratios: dict[str, float]
    total_missing: int
    columns_with_missing: tuple[str, ...]

    @property
    def has_missing_values(self) -> bool:
        """Return true when any missing value was found."""
        return self.total_missing > 0


@dataclass(frozen=True)
class DuplicateReport:
    """Duplicate-row and duplicate-identifier report."""

    full_duplicate_count: int
    full_duplicate_indices: tuple[int, ...]
    id_duplicate_count: int
    id_duplicate_values: tuple[Any, ...]

    @property
    def has_duplicates(self) -> bool:
        """Return true when row-level or ID-level duplicates were detected."""
        return self.full_duplicate_count > 0 or self.id_duplicate_count > 0


@dataclass(frozen=True)
class ConstantFeatureReport:
    """Detected constant columns and configured constant features."""

    constant_columns: tuple[str, ...]
    constant_feature_columns: tuple[str, ...]
    variable_feature_columns: tuple[str, ...]
    unique_counts: dict[str, int]
    constant_values: dict[str, Any]


@dataclass(frozen=True)
class DescriptiveStatistics:
    """Reusable descriptive statistics for reporting and EDA."""

    numeric: dict[str, dict[str, Any]]
    non_numeric: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class DatasetMetadata:
    """Generated metadata for one loaded dataset."""

    key: str
    path: Path
    sheet_name: str
    rows: int
    columns: int
    column_names: tuple[str, ...]
    dtypes: dict[str, str]
    id_column: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class DataQualityReport:
    """Full data quality report generated at load time."""

    metadata: DatasetMetadata
    validation: ValidationResult
    missing_values: MissingValueReport
    duplicates: DuplicateReport
    constants: ConstantFeatureReport
    descriptive_statistics: DescriptiveStatistics


@dataclass(frozen=True)
class LoadedDataset:
    """A loaded dataframe plus validated config and generated reports."""

    config: DatasetConfig
    dataframe: pd.DataFrame = field(repr=False)
    report: DataQualityReport

    @property
    def key(self) -> str:
        """Return the configured dataset key."""
        return self.config.key

