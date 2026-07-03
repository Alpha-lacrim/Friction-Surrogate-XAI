"""Modeling configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class ModelingConfig:
    """Configuration for model construction and overfitting audits."""

    output: dict[str, Any]
    randomness: dict[str, Any]
    validation: dict[str, Any]
    scoring: dict[str, Any]
    overfitting_detection: dict[str, Any]
    reports: dict[str, Any]
    mlflow: dict[str, Any]
    model_registry: dict[str, Any]
    models: dict[str, Any]

    @property
    def output_root(self) -> Path:
        """Return the configured modeling output root."""
        return Path(self.output["root_dir"])

    @property
    def repeated_seeds(self) -> tuple[int, ...]:
        """Return the configured repeated random seeds."""
        seeds = self.randomness.get("repeated_seeds")
        if not seeds:
            seeds = [self.randomness.get("default_seed", 42)]
        return tuple(int(seed) for seed in seeds)


def load_modeling_config(config_path: str | Path = "configs/modeling.yaml") -> ModelingConfig:
    """Load modeling configuration from YAML."""
    raw_config = load_yaml(config_path)["modeling"]
    return ModelingConfig(
        output=dict(raw_config.get("output", {})),
        randomness=dict(raw_config.get("randomness", {})),
        validation=dict(raw_config.get("validation", {})),
        scoring=dict(raw_config.get("scoring", {})),
        overfitting_detection=dict(raw_config.get("overfitting_detection", {})),
        reports=dict(raw_config.get("reports", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
        model_registry=dict(raw_config.get("model_registry", {})),
        models=dict(raw_config.get("models", {})),
    )


def with_overrides(config: ModelingConfig, **overrides: Any) -> ModelingConfig:
    """Return a shallowly overridden copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "randomness": dict(config.randomness),
        "validation": dict(config.validation),
        "scoring": dict(config.scoring),
        "overfitting_detection": dict(config.overfitting_detection),
        "reports": dict(config.reports),
        "mlflow": dict(config.mlflow),
        "model_registry": dict(config.model_registry),
        "models": dict(config.models),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return ModelingConfig(**values)
