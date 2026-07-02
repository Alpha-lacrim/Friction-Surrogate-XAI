"""Generate preprocessing pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from friction_surrogate_xai.preprocessing.artifacts import (
    PreprocessingArtifactManager,
    PreprocessingArtifacts,
    build_feature_validation_report,
)
from friction_surrogate_xai.preprocessing.config import (
    PreprocessingConfig,
    load_preprocessing_config,
)
from friction_surrogate_xai.preprocessing.factory import PreprocessingPipelineFactory
from friction_surrogate_xai.preprocessing.mlflow_logging import PreprocessingMLflowLogger


@dataclass(frozen=True)
class PreprocessingRunResult:
    """Result for one preprocessing artifact generation run."""

    dataset_key: str
    artifacts: PreprocessingArtifacts


class PreprocessingArtifactGenerator:
    """Build, save, and log unfitted preprocessing pipelines."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        factory: PreprocessingPipelineFactory | None = None,
    ) -> None:
        self.config = config or load_preprocessing_config()
        self.factory = factory or PreprocessingPipelineFactory(config=self.config)
        self.artifact_manager = PreprocessingArtifactManager(config=self.config)
        self.mlflow_logger = PreprocessingMLflowLogger(config=self.config.mlflow)

    def run_all(self, log_to_mlflow: bool | None = None) -> dict[str, PreprocessingRunResult]:
        """Generate preprocessing artifacts for all configured datasets."""
        return {
            dataset_key: self.run_dataset(dataset_key, log_to_mlflow=log_to_mlflow)
            for dataset_key in self.factory.data_catalog.dataset_keys()
        }

    def run_dataset(
        self,
        dataset_key: str,
        log_to_mlflow: bool | None = None,
    ) -> PreprocessingRunResult:
        """Generate preprocessing artifacts for one dataset."""
        feature_schema = self.factory.feature_schema_for_dataset(dataset_key)
        pipeline = self.factory.build_for_dataset(dataset_key)
        validation_report = build_feature_validation_report(
            dataset_key=dataset_key,
            feature_schema=feature_schema,
        )
        artifacts = self.artifact_manager.save(
            dataset_key=dataset_key,
            pipeline=pipeline,
            feature_schema=feature_schema,
            validation_report=validation_report,
        )

        should_log = self.config.mlflow.get("enabled", True) if log_to_mlflow is None else log_to_mlflow
        if should_log:
            self.mlflow_logger.log_artifacts(
                dataset_key=dataset_key,
                artifacts=artifacts,
                params=self._mlflow_params(dataset_key, feature_schema),
                metrics=self._mlflow_metrics(validation_report),
            )

        return PreprocessingRunResult(dataset_key=dataset_key, artifacts=artifacts)

    def _mlflow_params(self, dataset_key: str, feature_schema: Any) -> dict[str, Any]:
        return {
            "dataset_key": dataset_key,
            "default_scaler": self.config.scaling.get("default", "standard"),
            "constant_feature_removal": self.config.constant_feature_removal.get("enabled", True),
            "one_hot_enabled": self.config.encoding.get("one_hot", {}).get("enabled", True),
            "numeric_feature_count": len(feature_schema.numeric_features),
            "categorical_feature_count": len(feature_schema.categorical_features),
            "leakage_policy": "fit_inside_cv_folds_only",
        }

    @staticmethod
    def _mlflow_metrics(validation_report: dict[str, Any]) -> dict[str, float | int]:
        missing_count = len(
            [value for value in validation_report.get("missing_features", "").split(";") if value]
        )
        local_constant_count = len(
            [
                value
                for value in validation_report.get(
                    "constant_features_detected_from_local_data",
                    "",
                ).split(";")
                if value
            ]
        )
        return {
            "raw_data_available": int(bool(validation_report.get("raw_data_available", False))),
            "missing_feature_count": missing_count,
            "local_constant_feature_count": local_constant_count,
        }

