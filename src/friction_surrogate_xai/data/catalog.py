"""Configurable dataset catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from friction_surrogate_xai.config.loader import load_yaml, project_root
from friction_surrogate_xai.data.contracts import DatasetConfig, DatasetSchema


@dataclass(frozen=True)
class DataCatalog:
    """Dataset catalog built from configuration, not hard-coded paths."""

    root: Path
    schema: DatasetSchema
    datasets: dict[str, DatasetConfig]
    loading_options: dict[str, Any]
    validation_options: dict[str, Any]

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/datasets.yaml",
        root: str | Path | None = None,
    ) -> "DataCatalog":
        """Build a catalog from a YAML config file."""
        resolved_root = Path(root).resolve() if root else project_root()
        config = load_yaml(config_path)
        schema_config = config["schema"]

        special_targets = {
            key: tuple(value)
            for key, value in schema_config.get("special_target_columns", {}).items()
        }
        value_constraints = {
            column: tuple(rule.get("allowed_values", ()))
            for column, rule in schema_config.get("value_constraints", {}).items()
        }
        schema = DatasetSchema(
            id_column=schema_config["id_column"],
            feature_columns=tuple(schema_config["feature_columns"]),
            common_target_columns=tuple(schema_config["common_target_columns"]),
            special_target_columns=special_targets,
            column_types=dict(schema_config.get("column_types", {})),
            value_constraints=value_constraints,
        )

        datasets = {
            key: cls._build_dataset_config(key, dataset_config)
            for key, dataset_config in config["datasets"].items()
        }
        return cls(
            root=resolved_root,
            schema=schema,
            datasets=datasets,
            loading_options=dict(config.get("loading", {})),
            validation_options=dict(config.get("validation", {})),
        )

    @staticmethod
    def _build_dataset_config(key: str, dataset_config: dict[str, Any]) -> DatasetConfig:
        targets_config = dataset_config.get("targets", {})
        return DatasetConfig(
            key=key,
            filename=dataset_config["filename"],
            sheet_name=dataset_config["sheet_name"],
            raw_path=Path(dataset_config["raw_path"]),
            rows=int(dataset_config["rows"]),
            columns=int(dataset_config["columns"]),
            has_missing_values=bool(dataset_config.get("has_missing_values", False)),
            constant_features=tuple(dataset_config.get("constant_features", ())),
            special_targets=tuple(targets_config.get("special", ())),
            notes=tuple(dataset_config.get("notes", ())),
        )

    def dataset_keys(self) -> tuple[str, ...]:
        """Return configured dataset keys in file order."""
        return tuple(self.datasets)

    def get(self, dataset_key: str) -> DatasetConfig:
        """Return a dataset config by key."""
        try:
            return self.datasets[dataset_key]
        except KeyError as exc:
            available = ", ".join(self.dataset_keys())
            raise KeyError(f"Unknown dataset '{dataset_key}'. Available datasets: {available}") from exc

