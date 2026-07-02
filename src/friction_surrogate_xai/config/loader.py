"""YAML configuration loading for reproducible experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from friction_surrogate_xai.constants import CONFIG_DIRNAME


def project_root() -> Path:
    """Return the repository root inferred from the installed source layout."""
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dictionary for empty files."""
    resolved_path = Path(path)
    if not resolved_path.is_absolute():
        resolved_path = project_root() / resolved_path
    with resolved_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_project_configs(config_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load all first-level YAML config files from the project config directory."""
    resolved_dir = Path(config_dir) if config_dir else project_root() / CONFIG_DIRNAME
    configs: dict[str, dict[str, Any]] = {}
    for config_path in sorted(resolved_dir.glob("*.yaml")):
        configs[config_path.stem] = load_yaml(config_path)
    return configs

