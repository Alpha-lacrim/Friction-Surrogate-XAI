"""Statistical-comparison configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class StatisticalComparisonConfig:
    """Configuration for statistical model comparison."""

    output: dict[str, Any]
    inputs: dict[str, Any]
    schema: dict[str, Any]
    comparisons: dict[str, Any]
    tests: dict[str, Any]
    reports: dict[str, Any]
    plots: dict[str, Any]
    mlflow: dict[str, Any]

    @property
    def output_root(self) -> Path:
        """Return the configured output root."""
        return Path(self.output["root_dir"])

    @property
    def alpha(self) -> float:
        """Return the configured significance level."""
        return float(self.tests.get("alpha", 0.05))


def load_statistical_comparison_config(
    config_path: str | Path = "configs/statistical_comparison.yaml",
) -> StatisticalComparisonConfig:
    """Load statistical-comparison configuration from YAML."""
    raw_config = load_yaml(config_path)["statistical_comparison"]
    return StatisticalComparisonConfig(
        output=dict(raw_config.get("output", {})),
        inputs=dict(raw_config.get("inputs", {})),
        schema=dict(raw_config.get("schema", {})),
        comparisons=dict(raw_config.get("comparisons", {})),
        tests=dict(raw_config.get("tests", {})),
        reports=dict(raw_config.get("reports", {})),
        plots=dict(raw_config.get("plots", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
    )


def with_overrides(
    config: StatisticalComparisonConfig,
    **overrides: Any,
) -> StatisticalComparisonConfig:
    """Return a shallowly overridden config copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "inputs": dict(config.inputs),
        "schema": dict(config.schema),
        "comparisons": dict(config.comparisons),
        "tests": dict(config.tests),
        "reports": dict(config.reports),
        "plots": dict(config.plots),
        "mlflow": dict(config.mlflow),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return StatisticalComparisonConfig(**values)
