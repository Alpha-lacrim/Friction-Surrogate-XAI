"""EDA configuration loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class EDAConfig:
    """Configuration values for automated EDA generation."""

    output: dict[str, Any]
    columns: dict[str, Any]
    statistics: dict[str, Any]
    plots: dict[str, Any]
    outliers: dict[str, Any]
    mlflow: dict[str, Any]

    @property
    def output_root(self) -> Path:
        """Return the configured EDA output root."""
        return Path(self.output["root_dir"])


def load_eda_config(config_path: str | Path = "configs/eda.yaml") -> EDAConfig:
    """Load EDA configuration from YAML."""
    raw_config = load_yaml(config_path)["eda"]
    return EDAConfig(
        output=dict(raw_config.get("output", {})),
        columns=dict(raw_config.get("columns", {})),
        statistics=dict(raw_config.get("statistics", {})),
        plots=dict(raw_config.get("plots", {})),
        outliers=dict(raw_config.get("outliers", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
    )


def with_overrides(config: EDAConfig, **overrides: Any) -> EDAConfig:
    """Return a shallowly overridden copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "columns": dict(config.columns),
        "statistics": dict(config.statistics),
        "plots": dict(config.plots),
        "outliers": dict(config.outliers),
        "mlflow": dict(config.mlflow),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return EDAConfig(**values)

