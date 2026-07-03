"""MLflow logging for hyperparameter optimization trials and artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.eda.utils import sanitize_filename
from friction_surrogate_xai.experiments.mlflow_config import load_mlflow_settings
from friction_surrogate_xai.optimization.config import OptimizationConfig


class OptimizationMLflowLogger:
    """Log every optimization trial and final artifacts to MLflow."""

    def __init__(self, config: OptimizationConfig) -> None:
        self.config = config

    def enabled(self) -> bool:
        """Return whether MLflow logging is enabled."""
        return bool(self.config.mlflow.get("enabled", True))

    def log_trial(
        self,
        *,
        dataset_key: str,
        target_name: str,
        stage: str,
        model_key: str,
        trial_number: int,
        params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> None:
        """Log one Random Search or Optuna trial as an MLflow run."""
        if not self.enabled():
            return

        import mlflow

        self._configure_mlflow(mlflow)
        run_name = f"{stage}_{dataset_key}_{sanitize_filename(target_name)}_{model_key}_{trial_number}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("dataset", dataset_key)
            mlflow.set_tag("target", target_name)
            mlflow.set_tag("model_key", model_key)
            mlflow.set_tag("optimization_stage", stage)
            mlflow.set_tag("grid_search_used", "false")
            for tag_key, tag_value in self.config.mlflow.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_params(
                {
                    "trial_number": trial_number,
                    "model_key": model_key,
                    "target_name": target_name,
                    **{
                        f"param_{key}": self._serializable_param(value)
                        for key, value in params.items()
                    },
                }
            )
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if _is_finite(value)
                }
            )

    def log_artifacts(
        self,
        *,
        dataset_key: str,
        target_name: str,
        artifact_dir: Path,
        summary_metrics: dict[str, Any],
    ) -> None:
        """Log final optimization artifacts after all stages finish."""
        if not self.enabled():
            return

        import mlflow

        self._configure_mlflow(mlflow)
        run_name = f"optimization_summary_{dataset_key}_{sanitize_filename(target_name)}"
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("dataset", dataset_key)
            mlflow.set_tag("target", target_name)
            mlflow.set_tag("optimization_stage", "summary")
            mlflow.set_tag("grid_search_used", "false")
            for tag_key, tag_value in self.config.mlflow.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_params(
                {
                    "stage1_random_search": True,
                    "stage2_top_n": self.config.stage2_selection.get("top_n_models", 3),
                    "stage3_optuna": self.config.stage3_optuna.get("enabled", True),
                }
            )
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in summary_metrics.items()
                    if _is_finite(value)
                }
            )
            artifact_prefix = self.config.mlflow.get("artifact_path_prefix", "optimization")
            mlflow.log_artifacts(
                str(artifact_dir),
                artifact_path=f"{artifact_prefix}/{dataset_key}/{sanitize_filename(target_name)}",
            )

    def _configure_mlflow(self, mlflow: Any) -> None:
        settings = load_mlflow_settings()
        tracking_uri = settings.tracking_uri
        if tracking_uri.startswith("file:./"):
            tracking_uri = f"file:{project_root() / tracking_uri.removeprefix('file:./')}"
        if tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        experiment_name = self.config.mlflow.get("experiment_name") or settings.experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @staticmethod
    def _serializable_param(value: Any) -> str | float | int | bool | None:
        if isinstance(value, tuple):
            return json.dumps(list(value))
        if isinstance(value, list):
            return json.dumps(value)
        if isinstance(value, (str, float, int, bool)) or value is None:
            return value
        return json.dumps(value)


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
