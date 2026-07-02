"""Tests for the configurable data loading and validation layer."""

from __future__ import annotations

import pandas as pd

from friction_surrogate_xai.data import DataCatalog, DataLoader
from friction_surrogate_xai.data.profiling import DuplicateDetector, MissingValueReporter
from friction_surrogate_xai.data.validation import DataTypeValidator, SchemaValidator


def test_catalog_loads_dataset_contract_from_config() -> None:
    catalog = DataCatalog.from_config()

    assert catalog.dataset_keys() == ("dataset_0136", "dataset_0172", "dataset_3772")
    assert catalog.loading_options["engine"] == "openpyxl"
    assert catalog.schema.id_column == "No."
    assert "Temperature (°C)" in catalog.schema.target_columns_for("dataset_0136")
    assert "Temperature (°C)" not in catalog.schema.target_columns_for("dataset_0172")


def test_loader_loads_all_datasets_and_generates_metadata(require_raw_data) -> None:
    datasets = DataLoader().load_all()

    assert datasets["dataset_0136"].dataframe.shape == (36, 12)
    assert datasets["dataset_0172"].dataframe.shape == (72, 10)
    assert datasets["dataset_3772"].dataframe.shape == (36, 10)

    for key, loaded_dataset in datasets.items():
        report = loaded_dataset.report
        assert report.validation.is_valid, (key, report.validation.errors)
        assert report.metadata.key == key
        assert report.metadata.path.exists()
        assert report.missing_values.total_missing == 0
        assert not report.missing_values.has_missing_values
        assert not report.duplicates.has_duplicates
        assert "Hardness (HV)" in report.descriptive_statistics.numeric
        assert "mean" in report.descriptive_statistics.numeric["Hardness (HV)"]


def test_constant_feature_detection_matches_real_datasets(require_raw_data) -> None:
    datasets = DataLoader().load_all()
    composite = "Composite Volume Fraction (%)"

    report_0136 = datasets["dataset_0136"].report.constants
    report_0172 = datasets["dataset_0172"].report.constants
    report_3772 = datasets["dataset_3772"].report.constants

    assert report_0136.constant_feature_columns == (composite,)
    assert report_0136.constant_values[composite] == 0
    assert report_0136.unique_counts[composite] == 1

    assert report_0172.constant_feature_columns == ()
    assert report_0172.unique_counts[composite] == 2

    assert report_3772.constant_feature_columns == (composite,)
    assert report_3772.constant_values[composite] == 1
    assert report_3772.unique_counts[composite] == 1


def test_schema_validator_detects_missing_columns(require_raw_data) -> None:
    catalog = DataCatalog.from_config()
    loaded = DataLoader(catalog=catalog).load("dataset_0172")
    broken_frame = loaded.dataframe.drop(columns=["wear rate"])

    result = SchemaValidator().validate(
        dataframe=broken_frame,
        dataset_config=catalog.get("dataset_0172"),
        schema=catalog.schema,
    )

    assert not result.is_valid
    assert any("Missing expected columns" in error for error in result.errors)


def test_datatype_validator_detects_bad_dtype(require_raw_data) -> None:
    catalog = DataCatalog.from_config()
    loaded = DataLoader(catalog=catalog).load("dataset_0172")
    broken_frame = loaded.dataframe.copy()
    broken_frame["Tool Shape"] = broken_frame["Tool Shape"].astype(str)

    result = DataTypeValidator().validate(broken_frame, catalog.schema)

    assert not result.is_valid
    assert any("Tool Shape" in error for error in result.errors)


def test_duplicate_and_missing_reporters_detect_in_memory_issues() -> None:
    frame = pd.DataFrame(
        {
            "No.": [1, 1, 2],
            "Tool Shape": [1, 1, 2],
            "wear rate": [5.0, 5.0, None],
        }
    )

    duplicate_report = DuplicateDetector().report(frame, id_column="No.")
    missing_report = MissingValueReporter().report(frame)

    assert duplicate_report.full_duplicate_count == 1
    assert duplicate_report.id_duplicate_count == 1
    assert duplicate_report.id_duplicate_values == (1,)
    assert missing_report.total_missing == 1
    assert missing_report.columns_with_missing == ("wear rate",)
