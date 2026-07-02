"""Dataset schema and datatype validation."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api import types as pandas_types

from friction_surrogate_xai.data.contracts import DatasetConfig, DatasetSchema, ValidationResult


class SchemaValidator:
    """Validate dataframe shape and columns against configuration."""

    def __init__(self, require_exact_columns: bool = True, require_configured_shape: bool = True) -> None:
        self.require_exact_columns = require_exact_columns
        self.require_configured_shape = require_configured_shape

    def validate(
        self,
        dataframe: pd.DataFrame,
        dataset_config: DatasetConfig,
        schema: DatasetSchema,
    ) -> ValidationResult:
        """Validate column names, order, and configured shape."""
        errors: list[str] = []
        warnings: list[str] = []
        expected_columns = schema.expected_columns_for(dataset_config.key)
        actual_columns = tuple(dataframe.columns)

        missing_columns = tuple(column for column in expected_columns if column not in actual_columns)
        unexpected_columns = tuple(column for column in actual_columns if column not in expected_columns)
        if missing_columns:
            errors.append(f"Missing expected columns: {missing_columns}")
        if unexpected_columns and self.require_exact_columns:
            errors.append(f"Unexpected columns: {unexpected_columns}")
        if not missing_columns and not unexpected_columns and actual_columns != expected_columns:
            warnings.append("Columns match the schema but are not in the configured order.")

        if self.require_configured_shape:
            expected_shape = (dataset_config.rows, dataset_config.columns)
            actual_shape = dataframe.shape
            if actual_shape != expected_shape:
                errors.append(f"Expected shape {expected_shape}, found {actual_shape}")

        return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))


class DataTypeValidator:
    """Validate configured column datatype aliases."""

    def validate(self, dataframe: pd.DataFrame, schema: DatasetSchema) -> ValidationResult:
        """Validate pandas dtypes and configured value constraints."""
        errors: list[str] = []
        warnings: list[str] = []

        for column, expected_kind in schema.column_types.items():
            if column not in dataframe.columns:
                continue
            if not self._matches_kind(dataframe[column], expected_kind):
                errors.append(
                    f"Column '{column}' expected datatype '{expected_kind}', "
                    f"found '{dataframe[column].dtype}'"
                )

        for column, allowed_values in schema.value_constraints.items():
            if column not in dataframe.columns or not allowed_values:
                continue
            observed_values = set(self._to_python(value) for value in dataframe[column].dropna().unique())
            unexpected_values = tuple(sorted(observed_values - set(allowed_values)))
            if unexpected_values:
                errors.append(
                    f"Column '{column}' contains values outside configured allowed values: "
                    f"{unexpected_values}"
                )

        return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))

    @staticmethod
    def _matches_kind(series: pd.Series, expected_kind: str) -> bool:
        kind = expected_kind.lower()
        if kind == "integer":
            return pandas_types.is_integer_dtype(series)
        if kind == "numeric":
            return pandas_types.is_numeric_dtype(series)
        if kind == "float":
            return pandas_types.is_float_dtype(series)
        if kind == "string":
            return pandas_types.is_string_dtype(series)
        if kind == "boolean":
            return pandas_types.is_bool_dtype(series)
        raise ValueError(f"Unsupported configured datatype alias: {expected_kind}")

    @staticmethod
    def _to_python(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        return value

