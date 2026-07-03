"""MLflow logging for uncertainty reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.eda.utils import sanitize_filename
from friction_surrogate_xai.experiments.mlflow_config import load_mlflow_settings


class UncertaintyMLflowLogger:
    """Log uncertainty artifacts and metrics to MLflow."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def enabled(self) -> bool:
        """Return whether MLflow logging is enabled."""
        return bool(self.config.get("enabled", True))

    def log_run(
        self,
        *,
        dataset_key: str,
        target_name: str,
        artifact_dir: Path,
        comparison: pd.DataFrame,
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Log one uncertainty report run."""
        if not self.enabled():
            return

        import mlflow

        settings = load_mlflow_settings()
        tracking_uri = settings.tracking_uri
        if tracking_uri.startswith("file:./"):
            tracking_uri = f"file:{project_root() / tracking_uri.removeprefix('file:./')}"
        if tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

        experiment_name = self.config.get("experiment_name") or settings.experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(
            run_name=f"uncertainty_{dataset_key}_{sanitize_filename(target_name)}"
        ):
            mlflow.set_tag("dataset", dataset_key)
            mlflow.set_tag("target", target_name)
            for tag_key, tag_value in self.config.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_metrics(self._metric_payload(comparison, extra_metrics or {}))
            artifact_prefix = self.config.get("artifact_path_prefix", "uncertainty")
            mlflow.log_artifacts(
                str(artifact_dir),
                artifact_path=f"{artifact_prefix}/{dataset_key}/{sanitize_filename(target_name)}",
            )

    def _metric_payload(
        self,
        comparison: pd.DataFrame,
        extra_metrics: dict[str, Any],
    ) -> dict[str, float]:
        payload = {
            str(key): float(value)
            for key, value in extra_metrics.items()
            if _is_finite(value)
        }
        for _, row in comparison.iterrows():
            model = sanitize_filename(str(row.get("model_key", "model")))
            target = sanitize_filename(str(row.get("target", "target")))
            prefix = f"{model}_{target}"
            for column in (
                "coverage_probability",
                "coverage_error",
                "mean_interval_width",
                "mean_predictive_variance",
                "uncertainty_rank_score",
            ):
                value = row.get(column)
                if _is_finite(value):
                    payload[f"{prefix}_{column}"] = float(value)
        return payload


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
