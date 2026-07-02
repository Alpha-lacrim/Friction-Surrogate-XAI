"""Save preprocessing pipeline artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.data import DataLoader
from friction_surrogate_xai.preprocessing.config import PreprocessingConfig
from friction_surrogate_xai.preprocessing.factory import FeatureSchema


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Paths generated for one dataset preprocessing artifact bundle."""

    dataset_key: str
    root_dir: Path
    pipeline_path: Path
    config_path: Path
    schema_path: Path
    validation_report_path: Path
    leakage_policy_path: Path

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return all saved artifact paths."""
        return (
            self.pipeline_path,
            self.config_path,
            self.schema_path,
            self.validation_report_path,
            self.leakage_policy_path,
        )


class PreprocessingArtifactManager:
    """Persist preprocessing pipeline artifacts for reproducibility."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def save(
        self,
        dataset_key: str,
        pipeline: Pipeline,
        feature_schema: FeatureSchema,
        validation_report: dict[str, Any],
    ) -> PreprocessingArtifacts:
        """Save an unfitted pipeline plus config/schema/validation artifacts."""
        root_dir = self._dataset_root(dataset_key)
        pipeline_path = root_dir / "preprocessing_pipeline_unfitted.joblib"
        config_path = root_dir / "preprocessing_config.json"
        schema_path = root_dir / "feature_schema.json"
        validation_report_path = root_dir / "feature_validation_report.csv"
        leakage_policy_path = root_dir / "leakage_policy.md"

        root_dir.mkdir(parents=True, exist_ok=True)
        if self.config.artifacts.get("save_unfitted_pipeline", True):
            joblib.dump(pipeline, pipeline_path)
        if self.config.artifacts.get("save_config_snapshot", True):
            config_path.write_text(
                json.dumps(self._config_snapshot(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if self.config.artifacts.get("save_feature_schema", True):
            schema_path.write_text(
                json.dumps(asdict(feature_schema), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if self.config.artifacts.get("save_validation_report", True):
            pd.DataFrame([validation_report]).to_csv(
                validation_report_path,
                index=False,
                encoding="utf-8",
            )
        if self.config.artifacts.get("save_leakage_policy", True):
            leakage_policy_path.write_text(self._leakage_policy_text(), encoding="utf-8")

        return PreprocessingArtifacts(
            dataset_key=dataset_key,
            root_dir=root_dir,
            pipeline_path=pipeline_path,
            config_path=config_path,
            schema_path=schema_path,
            validation_report_path=validation_report_path,
            leakage_policy_path=leakage_policy_path,
        )

    def _dataset_root(self, dataset_key: str) -> Path:
        configured_root = Path(self.config.output["root_dir"])
        root = configured_root if configured_root.is_absolute() else project_root() / configured_root
        return root / dataset_key

    def _config_snapshot(self) -> dict[str, Any]:
        return {
            "output": self.config.output,
            "feature_validation": self.config.feature_validation,
            "features": self.config.features,
            "constant_feature_removal": self.config.constant_feature_removal,
            "scaling": self.config.scaling,
            "encoding": self.config.encoding,
            "artifacts": self.config.artifacts,
            "mlflow": self.config.mlflow,
        }

    @staticmethod
    def _leakage_policy_text() -> str:
        return "\n".join(
            [
                "# Leakage Policy",
                "",
                "- Saved preprocessing pipelines are unfitted.",
                "- `ConstantFeatureRemover.fit` learns constant columns only from the data passed to that fit call.",
                "- Scalers are sklearn transformers inside the pipeline and must be fitted inside CV folds.",
                "- Do not fit this preprocessing pipeline on the full dataset before model validation.",
            ]
        )


def build_feature_validation_report(
    dataset_key: str,
    feature_schema: FeatureSchema,
    data_loader: DataLoader | None = None,
) -> dict[str, Any]:
    """Build a validation report without fitting scalers or encoders."""
    loader = data_loader or DataLoader()
    dataset_config = loader.catalog.get(dataset_key)
    raw_path = dataset_config.resolved_path(loader.catalog.root)
    report: dict[str, Any] = {
        "dataset_key": dataset_key,
        "raw_data_available": raw_path.exists(),
        "required_features": ";".join(feature_schema.required_features),
        "numeric_features": ";".join(feature_schema.numeric_features),
        "categorical_features": ";".join(feature_schema.categorical_features),
        "missing_features": "",
        "raw_columns_not_used_as_features": "",
        "constant_features_detected_from_local_data": "",
        "note": "pipeline is saved unfitted; scaling and constant removal fit inside CV folds",
    }
    if not raw_path.exists():
        report["note"] = "raw data not available locally; config-only artifact generated"
        return report

    loaded_dataset = loader.load(dataset_key)
    observed_features = tuple(loaded_dataset.dataframe.columns)
    missing = tuple(
        feature for feature in feature_schema.required_features if feature not in observed_features
    )
    raw_columns_not_used = tuple(
        feature for feature in observed_features if feature not in feature_schema.required_features
    )
    report["missing_features"] = ";".join(missing)
    report["raw_columns_not_used_as_features"] = ";".join(raw_columns_not_used)
    report["constant_features_detected_from_local_data"] = ";".join(
        loaded_dataset.report.constants.constant_feature_columns
    )
    return report
