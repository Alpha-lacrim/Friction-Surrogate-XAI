"""Tests for model anti-overfitting policies and audit reports."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from friction_surrogate_xai.evaluation import OverfittingAuditRunner, OverfittingRiskAnalyzer
from friction_surrogate_xai.evaluation.validation import ValidationStrategyFactory
from friction_surrogate_xai.models import ModelFactory, load_modeling_config
from friction_surrogate_xai.models.config import with_overrides


def _small_feature_frame(n_rows: int = 18) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tool Shape": [(index % 3) + 1 for index in range(n_rows)],
            "Rotational Speed": [100.0 + 10.0 * index for index in range(n_rows)],
            "Plunging Speed": [5.0 + float(index % 5) for index in range(n_rows)],
            "Composite Volume Fraction (%)": [index % 2 for index in range(n_rows)],
        }
    )


def _small_target(n_rows: int = 18) -> pd.Series:
    return pd.Series(
        [0.4 * index + 0.2 * (index % 3) for index in range(n_rows)],
        name="wear rate",
    )


def _fast_config(tmp_path=None):
    config = with_overrides(
        load_modeling_config(),
        output={"root_dir": str(tmp_path) if tmp_path is not None else "reports/evaluation/tests"},
        randomness={"default_seed": 1, "repeated_seeds": [1]},
        validation={
            "primary_strategy": "repeated_kfold",
            "n_splits": 3,
            "shuffle": True,
            "repeated_kfold": {"enabled": True, "n_repeats_per_seed": 1},
            "loocv": {"enabled": True, "max_samples_for_auto_fallback": 6},
            "nested_cv": {"enabled": True, "outer_splits": 3, "inner_splits": 2, "shuffle": True},
            "bootstrap": {
                "enabled": True,
                "n_iterations": 3,
                "sample_fraction": 1.0,
                "require_oob_samples": True,
            },
        },
        mlflow={"enabled": False},
    )
    config.models["random_forest"]["params"]["n_estimators"] = 10
    return config


def test_model_factory_registers_required_anti_overfitting_defaults() -> None:
    factory = ModelFactory()
    expected = {
        "linear_regression",
        "ridge",
        "elasticnet",
        "svr_rbf",
        "gaussian_process_regression",
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "shallow_mlp_regressor",
    }

    assert set(factory.enabled_model_keys()) == expected
    assert isinstance(factory.build("ridge", random_state=42), Ridge)
    forest = factory.build("random_forest", random_state=42)
    assert isinstance(forest, RandomForestRegressor)
    assert forest.max_depth == 4
    assert forest.min_samples_leaf == 3
    assert forest.bootstrap is True

    policies = factory.policy_table()
    ridge_controls = policies.loc[policies["model_key"] == "ridge", "overfitting_controls"].iloc[0]
    xgboost_controls = policies.loc[
        policies["model_key"] == "xgboost",
        "overfitting_controls",
    ].iloc[0]
    assert "L2 regularization" in ridge_controls
    assert "inner-fold early stopping" in xgboost_controls


def test_validation_strategy_factory_supports_repeated_loocv_nested_and_bootstrap() -> None:
    config = _fast_config()
    factory = ValidationStrategyFactory(config)

    repeated = factory.repeated_kfold(n_samples=12)
    loocv = factory.loocv(n_samples=12)
    nested = factory.nested_cv(n_samples=12)
    bootstrap = factory.bootstrap(n_samples=12)

    assert len(repeated) == 3
    assert len(loocv) == 12
    assert len(nested) == 3
    assert all(len(split.inner_folds) == 2 for split in nested)
    assert len(bootstrap) == 3
    assert all(len(split.validation_indices) > 0 for split in bootstrap)


def test_overfitting_analyzer_flags_large_gap_and_fold_variance() -> None:
    config = _fast_config()
    analyzer = OverfittingRiskAnalyzer(config)
    rows = []
    for fold_id, validation_score in enumerate([0.15, 0.35, 0.55]):
        for split, score in (("train", 0.99), ("validation", validation_score)):
            rows.append(
                {
                    "dataset_key": "demo",
                    "model_key": "high_capacity",
                    "model_name": "High Capacity",
                    "target": "wear rate",
                    "validation_strategy": "repeated_kfold",
                    "fold_id": fold_id,
                    "seed": 1,
                    "repeat_id": 0,
                    "split": split,
                    "r2": score,
                    "rmse": 0.1 if split == "train" else 2.0,
                    "nrmse": 0.01 if split == "train" else 0.2,
                    "mae": 0.08 if split == "train" else 1.7,
                }
            )

    summary = analyzer.summarize(pd.DataFrame(rows))
    primary = summary.loc[summary["metric"] == "r2"].iloc[0]

    assert primary["likely_overfitting"] is True or bool(primary["likely_overfitting"])
    assert primary["risk_level"] == "high"
    assert primary["mean_generalization_gap"] > 0.3
    assert "generalization gap" in primary["risk_reasons"]


def test_overfitting_audit_runner_generates_report_without_data_leakage(tmp_path) -> None:
    config = _fast_config(tmp_path)
    artifacts = OverfittingAuditRunner(config=config).run(
        dataset_key="dataset_0172",
        X=_small_feature_frame(),
        y=_small_target(),
        model_keys=("ridge",),
        strategy="repeated_kfold",
        include_bootstrap=True,
        include_nested=False,
        log_to_mlflow=False,
    )

    assert artifacts.report_path is not None
    assert artifacts.report_path.exists()
    assert (artifacts.root_dir / "tables" / "fold_scores.csv").exists()
    assert (artifacts.root_dir / "tables" / "overfitting_summary.csv").exists()
    assert (artifacts.root_dir / "tables" / "model_policies.csv").exists()
    assert "fit_preprocessor_inside_each_fold" in set(artifacts.fold_scores["preprocessing_policy"])
    assert {"repeated_kfold", "bootstrap_oob"}.issubset(
        set(artifacts.fold_scores["validation_strategy"])
    )
    assert not artifacts.summary.empty
