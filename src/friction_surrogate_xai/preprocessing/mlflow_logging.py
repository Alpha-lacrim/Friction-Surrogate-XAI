"""MLflow logging for preprocessing artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.experiments.mlflow_config import load_mlflow_settings
from friction_surrogate_xai.preprocessing.artifacts import PreprocessingArtifacts


class PreprocessingMLflowLogger:
    """Log preprocessing artifacts into MLflow."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def enabled(self) -> bool:
        """Return whether MLflow logging is enabled."""
        return bool(self.config.get("enabled", True))

    def log_artifacts(
        self,
        dataset_key: str,
        artifacts: PreprocessingArtifacts,
        params: dict[str, Any],
        metrics: dict[str, float | int],
    ) -> None:
        """Log preprocessing artifact bundle to MLflow."""
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
        with mlflow.start_run(run_name=f"preprocessing_{dataset_key}"):
            mlflow.set_tag("dataset", dataset_key)
            for tag_key, tag_value in self.config.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            artifact_prefix = self.config.get("artifact_path_prefix", "preprocessing")
            mlflow.log_artifacts(
                str(Path(artifacts.root_dir)),
                artifact_path=f"{artifact_prefix}/{dataset_key}",
            )

