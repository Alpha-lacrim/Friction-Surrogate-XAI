"""Hyperparameter optimization configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for staged hyperparameter optimization."""

    output: dict[str, Any]
    stage1_random_search: dict[str, Any]
    stage2_selection: dict[str, Any]
    stage3_optuna: dict[str, Any]
    scoring: dict[str, Any]
    reports: dict[str, Any]
    plots: dict[str, Any]
    mlflow: dict[str, Any]
    search_spaces: dict[str, Any]

    @property
    def output_root(self) -> Path:
        """Return the configured optimization output root."""
        return Path(self.output["root_dir"])

    @property
    def repeated_seeds(self) -> tuple[int, ...]:
        """Return configured random-search CV seeds."""
        seeds = self.stage1_random_search.get("repeated_seeds", (42,))
        return tuple(int(seed) for seed in seeds)


def load_optimization_config(
    config_path: str | Path = "configs/optimization.yaml",
) -> OptimizationConfig:
    """Load optimization configuration from YAML."""
    raw_config = load_yaml(config_path)["optimization"]
    return OptimizationConfig(
        output=dict(raw_config.get("output", {})),
        stage1_random_search=dict(raw_config.get("stage1_random_search", {})),
        stage2_selection=dict(raw_config.get("stage2_selection", {})),
        stage3_optuna=dict(raw_config.get("stage3_optuna", {})),
        scoring=dict(raw_config.get("scoring", {})),
        reports=dict(raw_config.get("reports", {})),
        plots=dict(raw_config.get("plots", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
        search_spaces=dict(raw_config.get("search_spaces", {})),
    )


def with_overrides(config: OptimizationConfig, **overrides: Any) -> OptimizationConfig:
    """Return a shallowly overridden copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "stage1_random_search": dict(config.stage1_random_search),
        "stage2_selection": dict(config.stage2_selection),
        "stage3_optuna": dict(config.stage3_optuna),
        "scoring": dict(config.scoring),
        "reports": dict(config.reports),
        "plots": dict(config.plots),
        "mlflow": dict(config.mlflow),
        "search_spaces": dict(config.search_spaces),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return OptimizationConfig(**values)
