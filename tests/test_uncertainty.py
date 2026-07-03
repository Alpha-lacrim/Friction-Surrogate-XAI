"""Tests for uncertainty estimation reports."""

from __future__ import annotations

import pandas as pd

from friction_surrogate_xai.uncertainty import UncertaintyReportGenerator
from friction_surrogate_xai.uncertainty.config import load_uncertainty_config, with_overrides
from friction_surrogate_xai.uncertainty.reports import build_comparison_report


def _feature_frame(n_rows: int = 18) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tool Shape": [(index % 3) + 1 for index in range(n_rows)],
            "Rotational Speed": [100.0 + 10.0 * index for index in range(n_rows)],
            "Plunging Speed": [2.0 + 0.2 * (index % 5) for index in range(n_rows)],
            "Composite Volume Fraction (%)": [index % 2 for index in range(n_rows)],
        }
    )


def _target(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        0.003 * frame["Rotational Speed"]
        - 0.08 * frame["Plunging Speed"]
        + 0.22 * frame["Composite Volume Fraction (%)"],
        name="wear rate",
    )


def _fast_config(tmp_path):
    return with_overrides(
        load_uncertainty_config(),
        output={"root_dir": str(tmp_path)},
        models={
            "default_model_keys": [
                "gaussian_process_regression",
                "ridge",
            ]
        },
        gpr={"cv_splits": 3, "repeated_seeds": [5]},
        bootstrap={"n_iterations": 12, "min_predictions_per_sample": 1},
        intervals={"confidence_level": 0.90, "random_state": 19},
        mlflow={"enabled": False},
    )


def test_uncertainty_report_generator_writes_gpr_and_bootstrap_reports(tmp_path) -> None:
    frame = _feature_frame()
    artifacts = UncertaintyReportGenerator(config=_fast_config(tmp_path)).generate(
        dataset_key="dataset_0172",
        target_name="wear rate",
        X=frame,
        y=_target(frame),
        model_keys=("gaussian_process_regression", "ridge"),
        log_to_mlflow=False,
    )

    table_names = {path.name for path in artifacts.table_paths}
    figure_names = {path.name for path in artifacts.figure_paths}

    assert "prediction_intervals.csv" in table_names
    assert "confidence_bands.csv" in table_names
    assert "uncertainty_summary.csv" in table_names
    assert "comparison_report.csv" in table_names
    assert "coverage_probability.png" in figure_names
    assert "mean_interval_width.png" in figure_names
    assert any(path.parent.name == "confidence_bands" for path in artifacts.figure_paths)
    assert artifacts.markdown_paths[0].exists()

    methods = set(artifacts.prediction_intervals["method"])
    assert "gpr_predictive_distribution" in methods
    assert "bootstrap_oob_prediction_interval" in methods
    assert {"coverage_probability", "mean_interval_width", "mean_predictive_variance"}.issubset(
        artifacts.summary.columns
    )
    assert set(artifacts.comparison["model_key"]) == {"gaussian_process_regression", "ridge"}
    assert artifacts.prediction_intervals["interval_width"].dropna().ge(0).all()


def test_comparison_report_ranks_coverage_error_before_width() -> None:
    summary = pd.DataFrame(
        [
            {
                "dataset_key": "dataset_0172",
                "model_key": "ridge",
                "method": "bootstrap",
                "target": "wear rate",
                "interval_level": 0.9,
                "coverage_probability": 0.9,
                "coverage_error": 0.0,
                "mean_interval_width": 2.0,
            },
            {
                "dataset_key": "dataset_0172",
                "model_key": "random_forest",
                "method": "bootstrap",
                "target": "wear rate",
                "interval_level": 0.9,
                "coverage_probability": 0.7,
                "coverage_error": 0.2,
                "mean_interval_width": 1.0,
            },
        ]
    )

    comparison = build_comparison_report(summary)

    assert comparison.iloc[0]["model_key"] == "ridge"
    assert comparison.iloc[0]["coverage_rank"] == 1
    assert "interval" in comparison.iloc[0]["comparison_note"]
