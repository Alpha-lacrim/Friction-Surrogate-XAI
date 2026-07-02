"""Preprocessing configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for leakage-safe preprocessing pipelines."""

    output: dict[str, Any]
    feature_validation: dict[str, Any]
    features: dict[str, Any]
    constant_feature_removal: dict[str, Any]
    scaling: dict[str, Any]
    encoding: dict[str, Any]
    artifacts: dict[str, Any]
    mlflow: dict[str, Any]

    @property
    def output_root(self) -> Path:
        """Return the configured preprocessing artifact root."""
        return Path(self.output["root_dir"])


def load_preprocessing_config(
    config_path: str | Path = "configs/preprocessing.yaml",
) -> PreprocessingConfig:
    """Load preprocessing configuration from YAML."""
    raw_config = load_yaml(config_path)["preprocessing"]
    return PreprocessingConfig(
        output=dict(raw_config.get("output", {})),
        feature_validation=dict(raw_config.get("feature_validation", {})),
        features=dict(raw_config.get("features", {})),
        constant_feature_removal=dict(raw_config.get("constant_feature_removal", {})),
        scaling=dict(raw_config.get("scaling", {})),
        encoding=dict(raw_config.get("encoding", {})),
        artifacts=dict(raw_config.get("artifacts", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
    )


def with_overrides(config: PreprocessingConfig, **overrides: Any) -> PreprocessingConfig:
    """Return a shallowly overridden copy for tests and scripted runs."""
    values = {
        "output": dict(config.output),
        "feature_validation": dict(config.feature_validation),
        "features": dict(config.features),
        "constant_feature_removal": dict(config.constant_feature_removal),
        "scaling": dict(config.scaling),
        "encoding": dict(config.encoding),
        "artifacts": dict(config.artifacts),
        "mlflow": dict(config.mlflow),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return PreprocessingConfig(**values)

