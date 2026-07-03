"""Discretization configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class DiscretizationConfig:
    """Configuration for discrete-input dataset generation and comparison."""

    output: dict[str, Any]
    columns: dict[str, Any]
    binning: dict[str, Any]
    comparison: dict[str, Any]
    reports: dict[str, Any]
    mlflow: dict[str, Any]

    @property
    def dataset_root(self) -> Path:
        """Return configured processed dataset root."""
        return Path(self.output["dataset_root_dir"])

    @property
    def report_root(self) -> Path:
        """Return configured report root."""
        return Path(self.output["report_root_dir"])


def load_discretization_config(
    config_path: str | Path = "configs/discretization.yaml",
) -> DiscretizationConfig:
    """Load discretization configuration from YAML."""
    raw_config = load_yaml(config_path)["discretization"]
    return DiscretizationConfig(
        output=dict(raw_config.get("output", {})),
        columns=dict(raw_config.get("columns", {})),
        binning=dict(raw_config.get("binning", {})),
        comparison=dict(raw_config.get("comparison", {})),
        reports=dict(raw_config.get("reports", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
    )


def with_overrides(config: DiscretizationConfig, **overrides: Any) -> DiscretizationConfig:
    """Return a shallowly overridden copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "columns": dict(config.columns),
        "binning": dict(config.binning),
        "comparison": dict(config.comparison),
        "reports": dict(config.reports),
        "mlflow": dict(config.mlflow),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return DiscretizationConfig(**values)
