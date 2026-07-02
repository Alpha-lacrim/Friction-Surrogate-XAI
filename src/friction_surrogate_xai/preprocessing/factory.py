"""Factories for leakage-safe sklearn preprocessing pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from friction_surrogate_xai.data import DataCatalog, LoadedDataset
from friction_surrogate_xai.preprocessing.config import (
    PreprocessingConfig,
    load_preprocessing_config,
)
from friction_surrogate_xai.preprocessing.transformers import (
    ConfiguredColumnPreprocessor,
    ConstantFeatureRemover,
    FeatureValidator,
)


@dataclass(frozen=True)
class FeatureSchema:
    """Feature schema used to build a preprocessing pipeline."""

    dataset_key: str
    required_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]


class PreprocessingPipelineFactory:
    """Build unfitted sklearn pipelines that are safe to use inside CV."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        data_catalog: DataCatalog | None = None,
    ) -> None:
        self.config = config or load_preprocessing_config()
        self.data_catalog = data_catalog or DataCatalog.from_config()

    def feature_schema_for_dataset(self, dataset_key: str) -> FeatureSchema:
        """Return configured feature schema for a dataset."""
        feature_columns = self.data_catalog.schema.feature_columns
        categorical = tuple(
            column
            for column in self.config.features.get("categorical_features", ())
            if column in feature_columns
        )
        configured_numeric = tuple(self.config.features.get("numeric_features", ()))
        numeric = tuple(
            column
            for column in configured_numeric
            if column in feature_columns and column not in categorical
        )
        if not numeric:
            numeric = tuple(column for column in feature_columns if column not in categorical)

        return FeatureSchema(
            dataset_key=dataset_key,
            required_features=feature_columns,
            numeric_features=numeric,
            categorical_features=categorical,
        )

    def build_for_dataset(self, dataset_key: str, scaler: str | None = None) -> Pipeline:
        """Build an unfitted sklearn preprocessing pipeline for a configured dataset."""
        schema = self.feature_schema_for_dataset(dataset_key)
        validation_config = self.config.feature_validation
        constant_config = self.config.constant_feature_removal
        one_hot_config = dict(self.config.encoding.get("one_hot", {}))
        selected_scaler = scaler or self.config.scaling.get("default", "standard")

        pipeline = Pipeline(
            steps=[
                (
                    "feature_validation",
                    FeatureValidator(
                        required_features=schema.required_features,
                        numeric_features=schema.numeric_features,
                        categorical_features=schema.categorical_features,
                        enabled=bool(validation_config.get("enabled", True)),
                        allow_extra_columns=bool(validation_config.get("allow_extra_columns", False)),
                        strict_numeric_dtype=bool(validation_config.get("strict_numeric_dtype", True)),
                    ),
                ),
                (
                    "constant_feature_removal",
                    ConstantFeatureRemover(
                        enabled=bool(constant_config.get("enabled", True)),
                        dropna=bool(constant_config.get("dropna", False)),
                        tolerance=float(constant_config.get("tolerance", 0.0)),
                        protected_features=tuple(constant_config.get("protected_features", ())),
                    ),
                ),
                (
                    "column_preprocessor",
                    ConfiguredColumnPreprocessor(
                        numeric_features=schema.numeric_features,
                        categorical_features=schema.categorical_features,
                        scaler=selected_scaler,
                        scaler_params=self._scaler_params(selected_scaler),
                        one_hot_enabled=bool(one_hot_config.get("enabled", True)),
                        one_hot_params=self._one_hot_params(one_hot_config),
                        output="pandas",
                    ),
                ),
            ]
        )
        return pipeline

    def build_all(self, scaler: str | None = None) -> dict[str, Pipeline]:
        """Build unfitted pipelines for all configured datasets."""
        return {
            dataset_key: self.build_for_dataset(dataset_key, scaler=scaler)
            for dataset_key in self.data_catalog.dataset_keys()
        }

    def get_feature_frame(self, loaded_dataset: LoadedDataset) -> pd.DataFrame:
        """Return feature-only dataframe for a loaded dataset."""
        schema = self.feature_schema_for_dataset(loaded_dataset.key)
        return loaded_dataset.dataframe.loc[:, schema.required_features].copy()

    def _scaler_params(self, scaler: str) -> dict[str, Any]:
        if scaler == "standard":
            return dict(self.config.scaling.get("standard", {}))
        if scaler == "minmax":
            params = dict(self.config.scaling.get("minmax", {}))
            if "feature_range" in params:
                params["feature_range"] = tuple(params["feature_range"])
            return params
        if scaler == "robust":
            params = dict(self.config.scaling.get("robust", {}))
            if "quantile_range" in params:
                params["quantile_range"] = tuple(params["quantile_range"])
            return params
        if scaler == "none":
            return {}
        supported = ", ".join(self.config.scaling.get("supported", ()))
        raise ValueError(f"Unsupported scaler '{scaler}'. Supported scalers: {supported}")

    @staticmethod
    def _one_hot_params(one_hot_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "handle_unknown": one_hot_config.get("handle_unknown", "ignore"),
            "sparse_output": bool(one_hot_config.get("sparse_output", False)),
            "drop": one_hot_config.get("drop"),
        }

