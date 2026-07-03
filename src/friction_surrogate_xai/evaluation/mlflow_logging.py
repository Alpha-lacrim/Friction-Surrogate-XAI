"""MLflow logging for evaluation reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.eda.utils import sanitize_filename
from friction_surrogate_xai.experiments.mlflow_config import load_mlflow_settings


class EvaluationMLflowLogger:
    """Log evaluation metrics and artifacts into MLflow."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def enabled(self) -> bool:
        """Return whether MLflow logging is enabled."""
        return bool(self.config.get("enabled", True))

    def log_evaluation(
        self,
        *,
        dataset_key: str,
        model_name: str,
        artifact_dir: Path,
        params: dict[str, Any],
        metrics: pd.DataFrame,
        train_test_gap: pd.DataFrame,
        cv_summary: pd.DataFrame,
    ) -> None:
        """Log one evaluation run into MLflow."""
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

        with mlflow.start_run(run_name=f"evaluation_{dataset_key}_{model_name}"):
            mlflow.set_tag("dataset", dataset_key)
            mlflow.set_tag("model", model_name)
            for tag_key, tag_value in self.config.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_params(params)
            mlflow.log_metrics(
                self._metric_payload(
                    metrics=metrics,
                    train_test_gap=train_test_gap,
                    cv_summary=cv_summary,
                )
            )
            artifact_prefix = self.config.get("artifact_path_prefix", "evaluation")
            mlflow.log_artifacts(
                str(artifact_dir),
                artifact_path=f"{artifact_prefix}/{dataset_key}/{sanitize_filename(model_name)}",
            )

    def _metric_payload(
        self,
        *,
        metrics: pd.DataFrame,
        train_test_gap: pd.DataFrame,
        cv_summary: pd.DataFrame,
    ) -> dict[str, float]:
        payload: dict[str, float] = {}

        for _, row in metrics.iterrows():
            split = str(row.get("split", "split"))
            target = str(row.get("target", "target"))
            for metric in ("r2", "rmse", "nrmse", "mae"):
                if metric in row and _is_finite(row[metric]):
                    payload[self._name("metric", split, target, metric)] = float(row[metric])

        for _, row in train_test_gap.iterrows():
            if _is_finite(row.get("gap")):
                payload[
                    self._name(
                        "gap",
                        str(row.get("target", "target")),
                        str(row.get("metric", "metric")),
                    )
                ] = float(row["gap"])
            if _is_finite(row.get("relative_gap")):
                payload[
                    self._name(
                        "relative_gap",
                        str(row.get("target", "target")),
                        str(row.get("metric", "metric")),
                    )
                ] = float(row["relative_gap"])

        for _, row in cv_summary.iterrows():
            target = str(row.get("target", "target"))
            metric = str(row.get("metric", "metric"))
            for statistic in ("mean", "std", "ci_lower", "ci_upper", "stability_index"):
                if statistic in row and _is_finite(row[statistic]):
                    payload[self._name("cv", target, metric, statistic)] = float(row[statistic])

        return payload

    @staticmethod
    def _name(*parts: str) -> str:
        return "_".join(sanitize_filename(part).strip("_") for part in parts if part)


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
