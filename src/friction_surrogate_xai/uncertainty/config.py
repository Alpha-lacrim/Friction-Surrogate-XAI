"""Uncertainty configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class UncertaintyConfig:
    """Configuration for uncertainty report generation."""

    output: dict[str, Any]
    models: dict[str, Any]
    intervals: dict[str, Any]
    gpr: dict[str, Any]
    bootstrap: dict[str, Any]
    reports: dict[str, Any]
    plots: dict[str, Any]
    mlflow: dict[str, Any]

    @property
    def output_root(self) -> Path:
        """Return the configured output root."""
        return Path(self.output["root_dir"])

    @property
    def confidence_level(self) -> float:
        """Return the configured interval confidence level."""
        return float(self.intervals.get("confidence_level", 0.95))

    @property
    def gpr_model_key(self) -> str:
        """Return the model key treated as native GPR uncertainty."""
        return str(self.models.get("gpr_model_key", "gaussian_process_regression"))

    @property
    def default_model_keys(self) -> tuple[str, ...]:
        """Return configured default model keys."""
        return tuple(self.models.get("default_model_keys", (self.gpr_model_key,)))


def load_uncertainty_config(
    config_path: str | Path = "configs/uncertainty.yaml",
) -> UncertaintyConfig:
    """Load uncertainty configuration from YAML."""
    raw_config = load_yaml(config_path)["uncertainty"]
    return UncertaintyConfig(
        output=dict(raw_config.get("output", {})),
        models=dict(raw_config.get("models", {})),
        intervals=dict(raw_config.get("intervals", {})),
        gpr=dict(raw_config.get("gpr", {})),
        bootstrap=dict(raw_config.get("bootstrap", {})),
        reports=dict(raw_config.get("reports", {})),
        plots=dict(raw_config.get("plots", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
    )


def with_overrides(config: UncertaintyConfig, **overrides: Any) -> UncertaintyConfig:
    """Return a shallowly overridden config copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "models": dict(config.models),
        "intervals": dict(config.intervals),
        "gpr": dict(config.gpr),
        "bootstrap": dict(config.bootstrap),
        "reports": dict(config.reports),
        "plots": dict(config.plots),
        "mlflow": dict(config.mlflow),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return UncertaintyConfig(**values)
