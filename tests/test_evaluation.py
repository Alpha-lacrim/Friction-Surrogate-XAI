"""Tests for the reusable evaluation framework."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from friction_surrogate_xai.evaluation import (
    EvaluationReportGenerator,
    RegressionMetricCalculator,
    load_evaluation_config,
)
from friction_surrogate_xai.evaluation.config import with_overrides


def test_regression_metrics_and_train_test_gap() -> None:
    calculator = RegressionMetricCalculator()
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 5.0]
    train_pred = [1.0, 2.0, 3.0, 4.0]

    test_metrics = calculator.evaluate_predictions(
        y_true,
        y_pred,
        dataset_key="demo",
        model_name="model",
        split="test",
        target_names=("strength",),
    )
    train_metrics = calculator.evaluate_predictions(
        y_true,
        train_pred,
        dataset_key="demo",
        model_name="model",
        split="train",
        target_names=("strength",),
    )
    gap = calculator.train_test_gap(train_metrics=train_metrics, test_metrics=test_metrics)

    row = test_metrics.iloc[0]
    assert row["r2"] == pytest.approx(0.8)
    assert row["rmse"] == pytest.approx(0.5)
    assert row["nrmse"] == pytest.approx(0.5 / 3.0)
    assert row["mae"] == pytest.approx(0.25)

    rmse_gap = gap.loc[gap["metric"] == "rmse"].iloc[0]
    r2_gap = gap.loc[gap["metric"] == "r2"].iloc[0]
    assert rmse_gap["gap"] == pytest.approx(0.5)
    assert rmse_gap["gap_direction"] == "test_minus_train"
    assert r2_gap["gap"] == pytest.approx(0.2)
    assert r2_gap["gap_direction"] == "train_minus_test"


def test_cross_validation_summary_reports_stability_and_confidence_interval() -> None:
    calculator = RegressionMetricCalculator(confidence_level=0.95)
    fold_metrics = pd.DataFrame(
        {
            "dataset_key": ["demo", "demo", "demo"],
            "model_name": ["ridge", "ridge", "ridge"],
            "target": ["wear", "wear", "wear"],
            "fold_id": [0, 1, 2],
            "rmse": [1.0, 2.0, 3.0],
            "mae": [0.8, 1.0, 1.2],
        }
    )

    summary = calculator.summarize_cross_validation(fold_metrics)
    rmse = summary.loc[summary["metric"] == "rmse"].iloc[0]

    assert rmse["fold_count"] == 3
    assert rmse["mean"] == pytest.approx(2.0)
    assert rmse["std"] == pytest.approx(1.0)
    assert rmse["ci_lower"] < rmse["mean"] < rmse["ci_upper"]
    assert rmse["stability_index"] == pytest.approx(0.5)


def test_evaluation_generator_writes_csv_markdown_and_gap_reports(tmp_path) -> None:
    config = with_overrides(
        load_evaluation_config(),
        output={"root_dir": str(tmp_path)},
        plots={"enabled": False},
        mlflow={"enabled": False},
    )
    fold_metrics = pd.DataFrame(
        {
            "dataset_key": ["demo", "demo", "demo"],
            "model_name": ["ridge", "ridge", "ridge"],
            "target": ["strength", "strength", "strength"],
            "fold_id": [0, 1, 2],
            "r2": [0.8, 0.7, 0.9],
            "rmse": [1.0, 1.2, 0.9],
            "nrmse": [0.1, 0.12, 0.09],
            "mae": [0.7, 0.8, 0.6],
        }
    )

    artifacts = EvaluationReportGenerator(config=config).generate(
        dataset_key="demo",
        model_name="ridge",
        y_train_true=[1.0, 2.0, 3.0, 4.0],
        y_train_pred=[1.0, 2.0, 3.0, 4.0],
        y_test_true=[1.0, 2.0, 3.0, 4.0],
        y_test_pred=[1.0, 2.0, 3.0, 5.0],
        target_names=("strength",),
        fold_metrics=fold_metrics,
        log_to_mlflow=False,
    )

    assert not artifacts.plot_paths
    assert artifacts.summary_path is not None
    assert artifacts.summary_path.exists()
    assert (artifacts.root_dir / "tables" / "metrics.csv").exists()
    assert (artifacts.root_dir / "tables" / "metrics_long.csv").exists()
    assert (artifacts.root_dir / "tables" / "train_test_gap.csv").exists()
    assert (artifacts.root_dir / "tables" / "cross_validation_summary.csv").exists()
    assert (artifacts.root_dir / "markdown" / "metrics.md").exists()
    assert not artifacts.train_test_gap.empty
    assert not artifacts.cv_summary.empty


def test_evaluation_generator_creates_required_plots_and_curves(tmp_path) -> None:
    config = with_overrides(
        load_evaluation_config(),
        output={"root_dir": str(tmp_path)},
        learning_curve={"cv": 3, "train_sizes": [0.5, 1.0]},
        validation_curve={"cv": 3},
        mlflow={"enabled": False},
    )
    X = pd.DataFrame({"x1": range(18), "x2": [value % 3 for value in range(18)]})
    y = pd.Series([2.0 * value + 1.0 for value in range(18)], name="wear")
    estimator = Ridge(random_state=42)

    artifacts = EvaluationReportGenerator(config=config).generate(
        dataset_key="demo",
        model_name="ridge",
        y_test_true=y.iloc[-6:],
        y_test_pred=y.iloc[-6:] + 0.5,
        target_names=("wear",),
        estimator=estimator,
        X=X,
        y=y,
        validation_curve_params={"param_name": "alpha", "param_range": [0.1, 1.0, 10.0]},
        log_to_mlflow=False,
    )

    filenames = {path.name for path in artifacts.plot_paths}
    assert "test_wear_prediction_vs_actual.png" in filenames
    assert "test_wear_residuals.png" in filenames
    assert "wear_learning_curve.png" in filenames
    assert "wear_alpha_validation_curve.png" in filenames
    assert all(path.exists() and path.stat().st_size > 0 for path in artifacts.plot_paths)
