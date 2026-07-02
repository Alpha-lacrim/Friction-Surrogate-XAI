"""Skeleton integrity checks.

These tests intentionally avoid running any ML algorithms.
"""

from __future__ import annotations

from pathlib import Path

from friction_surrogate_xai.config.loader import load_project_configs, load_yaml, project_root
from friction_surrogate_xai.data.metadata import (
    COMMON_TARGET_COLUMNS,
    DATASET_SPECS,
    FEATURE_COLUMNS,
    ID_COLUMN,
)


def test_project_root_contains_expected_files() -> None:
    root = project_root()
    assert (root / "pyproject.toml").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "environment.yml").exists()
    assert (root / "README.md").exists()


def test_all_configs_load() -> None:
    configs = load_project_configs()
    assert {"datasets", "experiments", "logging", "mlflow", "pipelines", "project"} <= set(configs)


def test_dataset_config_matches_static_metadata() -> None:
    dataset_config = load_yaml("configs/datasets.yaml")
    configured_datasets = dataset_config["datasets"]

    assert ID_COLUMN == dataset_config["schema"]["id_column"]
    assert tuple(dataset_config["schema"]["feature_columns"]) == FEATURE_COLUMNS
    assert tuple(dataset_config["schema"]["common_target_columns"]) == COMMON_TARGET_COLUMNS
    assert set(configured_datasets) == set(DATASET_SPECS)

    for key, spec in DATASET_SPECS.items():
        configured = configured_datasets[key]
        assert configured["rows"] == spec.rows
        assert configured["columns"] == spec.columns
        assert configured["sheet_name"] == spec.sheet_name
        assert tuple(configured["constant_features"]) == spec.constant_features


def test_canonical_raw_assets_exist(require_raw_data) -> None:
    root = project_root()
    expected_files = [
        "Dataset 0136.xlsx",
        "Dataset 0172.xlsx",
        "Dataset 3772.xlsx",
        "Mini project 1405.v2.pdf",
    ]
    for filename in expected_files:
        path = root / "data" / "raw" / filename
        assert path.exists()
        assert path.stat().st_size > 0


def test_no_top_level_training_entrypoints_exist() -> None:
    root = project_root()
    disallowed_names = {"train.py", "run_training.py", "model_factory.py"}
    actual_names = {path.name for path in Path(root).glob("*.py")}
    assert actual_names.isdisjoint(disallowed_names)
