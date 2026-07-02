"""MLflow setup helpers for future experiment pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from friction_surrogate_xai.config.loader import load_yaml, project_root


@dataclass(frozen=True)
class MLflowSettings:
    """Configuration values required before starting an MLflow run."""

    tracking_uri: str
    experiment_name: str
    artifact_location: str


def load_mlflow_settings(config_path: str | Path = "configs/mlflow.yaml") -> MLflowSettings:
    """Load MLflow settings from YAML without starting any runs."""
    config = load_yaml(config_path)["mlflow"]
    return MLflowSettings(
        tracking_uri=config["tracking_uri"],
        experiment_name=config["experiment_name"],
        artifact_location=config["artifact_location"],
    )


def configure_mlflow(settings: MLflowSettings | None = None) -> MLflowSettings:
    """Apply MLflow tracking settings and return the active settings.

    This helper configures the tracking URI and experiment name only. It does not
    create model runs or log artifacts.
    """
    import mlflow

    active_settings = settings or load_mlflow_settings()
    tracking_uri = active_settings.tracking_uri
    if tracking_uri.startswith("file:./"):
        tracking_uri = f"file:{project_root() / tracking_uri.removeprefix('file:./')}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(active_settings.experiment_name)
    return active_settings

