"""Final pipeline configuration loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml


@dataclass(frozen=True)
class FinalPipelineConfig:
    """Configuration for the final resumable orchestration pipeline."""

    run: dict[str, Any]
    selection: dict[str, Any]
    stages: dict[str, Any]
    stage_options: dict[str, Any]
    component_overrides: dict[str, Any]
    mlflow: dict[str, Any]

    @property
    def run_id(self) -> str:
        """Return configured run identifier."""
        return str(self.run.get("run_id", "latest"))

    @property
    def root_dir(self) -> Path:
        """Return configured pipeline output root."""
        return Path(self.run.get("root_dir", "reports/final_pipeline"))

    @property
    def resume(self) -> bool:
        """Return whether completed stages should be reused."""
        return bool(self.run.get("resume", True))

    @property
    def force_rerun(self) -> bool:
        """Return whether completed stages should be rerun."""
        return bool(self.run.get("force_rerun", False))

    @property
    def continue_on_error(self) -> bool:
        """Return whether later stages should run after a stage failure."""
        return bool(self.run.get("continue_on_error", True))


def load_final_pipeline_config(
    config_path: str | Path = "configs/final_pipeline.yaml",
) -> FinalPipelineConfig:
    """Load the final pipeline configuration from YAML."""
    raw_config = load_yaml(config_path)["final_pipeline"]
    return FinalPipelineConfig(
        run=dict(raw_config.get("run", {})),
        selection=dict(raw_config.get("selection", {})),
        stages=dict(raw_config.get("stages", {})),
        stage_options=dict(raw_config.get("stage_options", {})),
        component_overrides=dict(raw_config.get("component_overrides", {})),
        mlflow=dict(raw_config.get("mlflow", {})),
    )


def with_overrides(config: FinalPipelineConfig, **overrides: Any) -> FinalPipelineConfig:
    """Return a shallowly overridden config copy for tests and scripted runs."""
    values = {
        "run": dict(config.run),
        "selection": dict(config.selection),
        "stages": dict(config.stages),
        "stage_options": dict(config.stage_options),
        "component_overrides": dict(config.component_overrides),
        "mlflow": dict(config.mlflow),
    }
    for section, section_overrides in overrides.items():
        values[section].update(section_overrides)
    return FinalPipelineConfig(**values)
