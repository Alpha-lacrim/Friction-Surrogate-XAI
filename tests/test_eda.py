"""Tests for the research-grade EDA module."""

from __future__ import annotations

import pandas as pd

from friction_surrogate_xai.data import DataLoader
from friction_surrogate_xai.eda.config import load_eda_config, with_overrides
from friction_surrogate_xai.eda.outliers import OutlierDetector
from friction_surrogate_xai.eda.plots import EDAPlotter
from friction_surrogate_xai.eda.runner import EDAReportGenerator
from friction_surrogate_xai.eda.statistics import StatisticalAnalyzer


def test_eda_generator_writes_tables_and_markdown_without_row_removal(
    tmp_path,
    require_raw_data,
) -> None:
    config = with_overrides(
        load_eda_config(),
        output={"root_dir": str(tmp_path)},
        plots={"enabled": False},
        mlflow={"enabled": False},
    )

    artifacts = EDAReportGenerator(config=config).run_all(log_to_mlflow=False)

    assert set(artifacts) == {"dataset_0136", "dataset_0172", "dataset_3772"}
    for dataset_key, dataset_artifacts in artifacts.items():
        tables_dir = dataset_artifacts.root_dir / "tables"
        assert dataset_artifacts.markdown_path.exists(), dataset_key
        assert not dataset_artifacts.plot_paths
        assert (tables_dir / "descriptive_statistics.csv").exists()
        assert (tables_dir / "confidence_intervals.csv").exists()
        assert (tables_dir / "normality_tests.csv").exists()
        assert (tables_dir / "correlation_pearson.csv").exists()
        assert (tables_dir / "correlation_spearman.csv").exists()
        assert (tables_dir / "correlation_kendall.csv").exists()
        assert (tables_dir / "outlier_scores.csv").exists()
        assert (tables_dir / "iqr_outliers.csv").exists()

        outlier_scores = pd.read_csv(tables_dir / "outlier_scores.csv")
        original_rows = DataLoader().load(dataset_key).dataframe.shape[0]
        assert len(outlier_scores) == original_rows


def test_eda_reports_expected_constant_feature_status(tmp_path, require_raw_data) -> None:
    config = with_overrides(
        load_eda_config(),
        output={"root_dir": str(tmp_path)},
        plots={"enabled": False},
        mlflow={"enabled": False},
    )

    artifacts = EDAReportGenerator(config=config).run_all(log_to_mlflow=False)
    composite = "Composite Volume Fraction (%)"

    constants_0136 = pd.read_csv(
        artifacts["dataset_0136"].root_dir / "tables" / "constant_features.csv"
    )
    constants_0172 = pd.read_csv(
        artifacts["dataset_0172"].root_dir / "tables" / "constant_features.csv"
    )
    constants_3772 = pd.read_csv(
        artifacts["dataset_3772"].root_dir / "tables" / "constant_features.csv"
    )

    def is_constant_feature(table: pd.DataFrame) -> bool:
        row = table.loc[table["column"] == composite].iloc[0]
        return bool(row["is_constant_feature"])

    assert is_constant_feature(constants_0136)
    assert not is_constant_feature(constants_0172)
    assert is_constant_feature(constants_3772)


def test_plotter_generates_required_plot_types_for_small_subset(tmp_path, require_raw_data) -> None:
    loaded = DataLoader().load("dataset_0172")
    columns = ("Tool Shape", "wear rate")
    stats = StatisticalAnalyzer().analyze(loaded.dataframe, columns)
    plotter = EDAPlotter(load_eda_config().plots)

    artifacts = plotter.generate(
        dataframe=loaded.dataframe,
        columns=columns,
        correlations=stats.correlations,
        output_dir=tmp_path,
        dataset_label="dataset_0172",
    )

    suffixes = {path.name for path in artifacts.paths}
    assert "tool_shape_histogram.png" in suffixes
    assert "wear_rate_kde.png" in suffixes
    assert "tool_shape_qq_plot.png" in suffixes
    assert "wear_rate_boxplot.png" in suffixes
    assert "dataset_0172_pairplot.png" in suffixes
    assert "pearson_correlation_heatmap.png" in suffixes
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts.paths)


def test_outlier_detector_is_detect_only() -> None:
    frame = pd.DataFrame(
        {
            "No.": [1, 2, 3, 4, 5, 6],
            "x": [1.0, 1.1, 1.2, 1.1, 1.0, 100.0],
            "y": [2.0, 2.1, 2.2, 2.1, 2.0, 2.2],
        }
    )

    reports = OutlierDetector().detect(frame, columns=("x", "y"), id_column="No.")

    assert len(reports.row_scores) == len(frame)
    assert reports.row_scores["iqr_is_outlier"].any()
    assert (reports.iqr_outliers["column"] == "x").any()
    assert frame.shape == (6, 3)
