"""MLflow logging for discretization artifacts and comparisons."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.eda.utils import sanitize_filename
from friction_surrogate_xai.experiments.mlflow_config import load_mlflow_settings


class DiscretizationMLflowLogger:
    """Log generated datasets and original-vs-discrete comparisons to MLflow."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def enabled(self) -> bool:
        """Return whether MLflow logging is enabled."""
        return bool(self.config.get("enabled", True))

    def log_dataset_generation(
        self,
        *,
        dataset_key: str,
        artifact_dir: Path,
        metadata: pd.DataFrame,
    ) -> None:
        """Log one discrete dataset generation run."""
        if not self.enabled():
            return

        import mlflow

        self._configure(mlflow)
        with mlflow.start_run(run_name=f"discretize_{dataset_key}"):
            mlflow.set_tag("dataset", dataset_key)
            mlflow.set_tag("discretization_stage", "dataset_generation")
            for tag_key, tag_value in self.config.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_metrics(
                {
                    "discretized_feature_count": float(len(metadata)),
                    "constant_feature_count": float(metadata.get("is_constant", pd.Series()).sum())
                    if "is_constant" in metadata
                    else 0.0,
                }
            )
            prefix = self.config.get("artifact_path_prefix", "discretization")
            mlflow.log_artifacts(str(artifact_dir), artifact_path=f"{prefix}/{dataset_key}")

    def log_comparison(
        self,
        *,
        dataset_key: str,
        target_name: str,
        artifact_dir: Path,
        comparison_summary: pd.DataFrame,
    ) -> None:
        """Log original-vs-discrete comparison reports."""
        if not self.enabled():
            return

        import mlflow

        self._configure(mlflow)
        with mlflow.start_run(
            run_name=f"compare_original_discrete_{dataset_key}_{sanitize_filename(target_name)}"
        ):
            mlflow.set_tag("dataset", dataset_key)
            mlflow.set_tag("target", target_name)
            mlflow.set_tag("discretization_stage", "original_vs_discrete_comparison")
            for tag_key, tag_value in self.config.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_metrics(self._comparison_metrics(comparison_summary))
            prefix = self.config.get("artifact_path_prefix", "discretization")
            mlflow.log_artifacts(
                str(artifact_dir),
                artifact_path=f"{prefix}/comparison/{dataset_key}/{sanitize_filename(target_name)}",
            )

    def _configure(self, mlflow: Any) -> None:
        settings = load_mlflow_settings()
        tracking_uri = settings.tracking_uri
        if tracking_uri.startswith("file:./"):
            tracking_uri = f"file:{project_root() / tracking_uri.removeprefix('file:./')}"
        if tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        experiment_name = self.config.get("experiment_name") or settings.experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @staticmethod
    def _comparison_metrics(comparison_summary: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {
            "comparison_rows": float(len(comparison_summary)),
        }
        if "delta_objective_discrete_minus_original" in comparison_summary:
            deltas = pd.to_numeric(
                comparison_summary["delta_objective_discrete_minus_original"],
                errors="coerce",
            ).dropna()
            if not deltas.empty:
                metrics["mean_delta_objective_discrete_minus_original"] = float(deltas.mean())
                metrics["best_delta_objective_discrete_minus_original"] = float(deltas.max())
        return {key: value for key, value in metrics.items() if np.isfinite(value)}
