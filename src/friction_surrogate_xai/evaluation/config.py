"""Evaluation configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration values for reusable evaluation reports."""

    output: dict[str, Any]
    metrics: dict[str, Any]
    plots: dict[str, Any]
    learning_curve: dict[str, Any]
    validation_curve: dict[str, Any]
    reports: dict[str, Any]
    mlflow: dict[str, Any]

    @property
    def output_root(self) -> Path:
        """Return the configured evaluation output root."""
        return Path(self.output["root_dir"])


def load_evaluation_config(
    config_path: str | Path = "configs/evaluation.yaml",
) -> EvaluationConfig:
    """Load evaluation configuration from YAML."""
    raw_config = load_yaml(config_path)["evaluation"]
    return EvaluationConfig(
        output=dict(raw_config.get("output", {})),
        metrics=dict(raw_config.get("metrics", {})),
        plots=dict(raw_config.get("plots", {})),
        learning_curve=dict(raw_config.get("learning_curve", {})),
        validation_curve=dict(raw_config.get("validation_curve", {})),
        reports=dict(raw_config.get("reports", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
    )


def with_overrides(config: EvaluationConfig, **overrides: Any) -> EvaluationConfig:
    """Return a shallowly overridden copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "metrics": dict(config.metrics),
        "plots": dict(config.plots),
        "learning_curve": dict(config.learning_curve),
        "validation_curve": dict(config.validation_curve),
        "reports": dict(config.reports),
        "mlflow": dict(config.mlflow),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return EvaluationConfig(**values)
