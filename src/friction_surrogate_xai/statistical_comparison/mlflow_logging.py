"""MLflow logging for statistical comparison artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.experiments.mlflow_config import load_mlflow_settings


class StatisticalComparisonMLflowLogger:
    """Log statistical-comparison artifacts and metrics to MLflow."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def enabled(self) -> bool:
        """Return whether MLflow logging is enabled."""
        return bool(self.config.get("enabled", True))

    def log_run(
        self,
        *,
        artifact_dir: Path,
        wilcoxon: pd.DataFrame,
        friedman: pd.DataFrame,
        nemenyi: pd.DataFrame,
        score_table: pd.DataFrame,
    ) -> None:
        """Log one statistical-comparison run."""
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
        with mlflow.start_run(run_name="statistical_comparison"):
            for tag_key, tag_value in self.config.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_metrics(
                {
                    "score_rows": float(len(score_table)),
                    "wilcoxon_tests": float(len(wilcoxon)),
                    "friedman_tests": float(len(friedman)),
                    "nemenyi_tests": float(len(nemenyi)),
                    "significant_wilcoxon": _count_significant(wilcoxon),
                    "significant_friedman": _count_significant(friedman),
                    "significant_nemenyi": _count_significant(nemenyi),
                }
            )
            artifact_prefix = self.config.get(
                "artifact_path_prefix",
                "statistical_comparison",
            )
            mlflow.log_artifacts(str(artifact_dir), artifact_path=artifact_prefix)


def _count_significant(table: pd.DataFrame) -> float:
    if table.empty or "significant" not in table:
        return 0.0
    return float(table["significant"].sum())
