"""Reusable dataset loader classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from friction_surrogate_xai.data.catalog import DataCatalog
from friction_surrogate_xai.data.contracts import DataQualityReport, LoadedDataset, ValidationResult
from friction_surrogate_xai.data.profiling import (
    ConstantFeatureDetector,
    DescriptiveStatisticsGenerator,
    DuplicateDetector,
    MetadataGenerator,
    MissingValueReporter,
)
from friction_surrogate_xai.data.validation import DataTypeValidator, SchemaValidator


class DatasetValidationError(ValueError):
    """Raised when strict loading encounters schema or datatype errors."""

    def __init__(self, dataset_key: str, validation: ValidationResult) -> None:
        self.dataset_key = dataset_key
        self.validation = validation
        super().__init__(f"Dataset '{dataset_key}' failed validation: {validation.errors}")


class BaseDatasetLoader(ABC):
    """Dataset loader interface."""

    @abstractmethod
    def load(self, dataset_key: str, strict: bool | None = None) -> LoadedDataset:
        """Load one dataset and return dataframe plus reports."""

    @abstractmethod
    def load_all(self, strict: bool | None = None) -> dict[str, LoadedDataset]:
        """Load all configured datasets."""


class PandasExcelDatasetLoader(BaseDatasetLoader):
    """Excel dataset loader backed by pandas and a configurable data catalog."""

    def __init__(self, catalog: DataCatalog | None = None) -> None:
        self.catalog = catalog or DataCatalog.from_config()
        validation_options = self.catalog.validation_options
        duplicate_options = validation_options.get("duplicate_detection", {})
        constant_options = validation_options.get("constant_detection", {})
        descriptive_options = validation_options.get("descriptive_statistics", {})

        self.schema_validator = SchemaValidator(
            require_exact_columns=bool(validation_options.get("require_exact_columns", True)),
            require_configured_shape=bool(validation_options.get("require_configured_shape", True)),
        )
        self.datatype_validator = DataTypeValidator()
        self.missing_reporter = MissingValueReporter()
        self.duplicate_detector = DuplicateDetector(
            check_full_rows=bool(duplicate_options.get("check_full_rows", True)),
            check_id_column=bool(duplicate_options.get("check_id_column", True)),
        )
        self.constant_detector = ConstantFeatureDetector(
            dropna=bool(constant_options.get("dropna", False)),
        )
        self.descriptive_statistics = DescriptiveStatisticsGenerator(
            percentiles=tuple(descriptive_options.get("percentiles", (0.25, 0.5, 0.75))),
        )
        self.metadata_generator = MetadataGenerator()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/datasets.yaml",
        root: str | Path | None = None,
    ) -> "PandasExcelDatasetLoader":
        """Create a loader from a config file."""
        return cls(catalog=DataCatalog.from_config(config_path=config_path, root=root))

    def load(self, dataset_key: str, strict: bool | None = None) -> LoadedDataset:
        """Load one configured Excel dataset and generate validation reports."""
        dataset_config = self.catalog.get(dataset_key)
        path = dataset_config.resolved_path(self.catalog.root)
        dataframe = self._read_excel(path=path, sheet_name=dataset_config.sheet_name)
        report = self._build_report(dataset_key=dataset_key, path=path, dataframe=dataframe)

        strict_mode = (
            bool(self.catalog.loading_options.get("strict_validation", True))
            if strict is None
            else strict
        )
        if strict_mode and not report.validation.is_valid:
            raise DatasetValidationError(dataset_key=dataset_key, validation=report.validation)

        return LoadedDataset(config=dataset_config, dataframe=dataframe, report=report)

    def load_all(self, strict: bool | None = None) -> dict[str, LoadedDataset]:
        """Load all configured datasets."""
        return {
            dataset_key: self.load(dataset_key=dataset_key, strict=strict)
            for dataset_key in self.catalog.dataset_keys()
        }

    def _read_excel(self, path: Path, sheet_name: str) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Configured dataset file does not exist: {path}")

        engine = self.catalog.loading_options.get("engine")
        return pd.read_excel(path, sheet_name=sheet_name, engine=engine)

    def _build_report(self, dataset_key: str, path: Path, dataframe: pd.DataFrame) -> DataQualityReport:
        dataset_config = self.catalog.get(dataset_key)
        schema = self.catalog.schema
        validation = self.schema_validator.validate(
            dataframe=dataframe,
            dataset_config=dataset_config,
            schema=schema,
        ).merge(self.datatype_validator.validate(dataframe=dataframe, schema=schema))

        metadata = self.metadata_generator.report(
            dataframe=dataframe,
            dataset_key=dataset_key,
            path=path,
            sheet_name=dataset_config.sheet_name,
            schema=schema,
        )
        return DataQualityReport(
            metadata=metadata,
            validation=validation,
            missing_values=self.missing_reporter.report(dataframe),
            duplicates=self.duplicate_detector.report(dataframe, schema.id_column),
            constants=self.constant_detector.report(dataframe, schema.feature_columns),
            descriptive_statistics=self.descriptive_statistics.report(dataframe),
        )


DataLoader = PandasExcelDatasetLoader

