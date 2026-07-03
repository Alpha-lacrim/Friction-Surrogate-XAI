"""MLflow logging for the final orchestration pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.experiments.mlflow_config import load_mlflow_settings


class FinalPipelineMLflowLogger:
    """Log final pipeline state and artifacts to MLflow."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def enabled(self) -> bool:
        """Return whether MLflow logging is enabled."""
        return bool(self.config.get("enabled", True))

    def log_run(
        self,
        *,
        run_id: str,
        root_dir: Path,
        stage_rows: list[dict[str, Any]],
    ) -> None:
        """Log one final pipeline run."""
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
        with mlflow.start_run(run_name=f"final_pipeline_{run_id}"):
            mlflow.set_tag("run_id", run_id)
            for tag_key, tag_value in self.config.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_metrics(_stage_metrics(stage_rows))
            mlflow.log_artifacts(
                str(root_dir),
                artifact_path=self.config.get("artifact_path_prefix", "final_pipeline"),
            )


def _stage_metrics(stage_rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "stage_count": float(len(stage_rows)),
        "completed_stage_count": float(
            sum(1 for row in stage_rows if row.get("status") == "completed")
        ),
        "failed_stage_count": float(
            sum(1 for row in stage_rows if row.get("status") == "failed")
        ),
        "skipped_stage_count": float(
            sum(1 for row in stage_rows if row.get("status") == "skipped")
        ),
    }
