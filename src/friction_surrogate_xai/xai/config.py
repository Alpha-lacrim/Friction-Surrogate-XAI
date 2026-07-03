"""Explainability configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class XAIConfig:
    """Configuration for explainability report generation."""

    output: dict[str, Any]
    data: dict[str, Any]
    shap: dict[str, Any]
    permutation_importance: dict[str, Any]
    tree_importance: dict[str, Any]
    tree_interpreter: dict[str, Any]
    lime: dict[str, Any]
    interpretation: dict[str, Any]
    reports: dict[str, Any]
    plots: dict[str, Any]
    mlflow: dict[str, Any]

    @property
    def output_root(self) -> Path:
        """Return the configured XAI output root."""
        return Path(self.output["root_dir"])


def load_xai_config(config_path: str | Path = "configs/xai.yaml") -> XAIConfig:
    """Load XAI configuration from YAML."""
    raw_config = load_yaml(config_path)["xai"]
    return XAIConfig(
        output=dict(raw_config.get("output", {})),
        data=dict(raw_config.get("data", {})),
        shap=dict(raw_config.get("shap", {})),
        permutation_importance=dict(raw_config.get("permutation_importance", {})),
        tree_importance=dict(raw_config.get("tree_importance", {})),
        tree_interpreter=dict(raw_config.get("tree_interpreter", {})),
        lime=dict(raw_config.get("lime", {})),
        interpretation=dict(raw_config.get("interpretation", {})),
        reports=dict(raw_config.get("reports", {})),
        plots=dict(raw_config.get("plots", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
    )


def with_overrides(config: XAIConfig, **overrides: Any) -> XAIConfig:
    """Return a shallowly overridden copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "data": dict(config.data),
        "shap": dict(config.shap),
        "permutation_importance": dict(config.permutation_importance),
        "tree_importance": dict(config.tree_importance),
        "tree_interpreter": dict(config.tree_interpreter),
        "lime": dict(config.lime),
        "interpretation": dict(config.interpretation),
        "reports": dict(config.reports),
        "plots": dict(config.plots),
        "mlflow": dict(config.mlflow),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return XAIConfig(**values)
