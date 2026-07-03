"""Tests for staged hyperparameter optimization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from friction_surrogate_xai.optimization import HyperparameterOptimizationRunner
from friction_surrogate_xai.optimization.config import load_optimization_config, with_overrides
from friction_surrogate_xai.optimization.spaces import SearchSpaceSampler


def _feature_frame(n_rows: int = 18) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tool Shape": [(index % 3) + 1 for index in range(n_rows)],
            "Rotational Speed": [100.0 + 8.0 * index for index in range(n_rows)],
            "Plunging Speed": [5.0 + float(index % 4) for index in range(n_rows)],
            "Composite Volume Fraction (%)": [index % 2 for index in range(n_rows)],
        }
    )


def _target(n_rows: int = 18) -> pd.Series:
    return pd.Series(
        [0.3 * index + 0.1 * (index % 3) for index in range(n_rows)],
        name="wear rate",
    )


def _fast_config(tmp_path):
    return with_overrides(
        load_optimization_config(),
        output={"root_dir": str(tmp_path)},
        stage1_random_search={
            "enabled": True,
            "n_trials_per_model": 2,
            "cv_splits": 3,
            "repeated_seeds": [3],
        },
        stage2_selection={"top_n_models": 2, "ranking_metric": "r2"},
        stage3_optuna={
            "enabled": True,
            "n_trials_per_model": 2,
            "sampler": "tpe",
            "direction": "maximize",
            "timeout_seconds": None,
        },
        plots={"enabled": True},
        mlflow={"enabled": False},
    )


def test_search_sampler_produces_non_grid_random_samples() -> None:
    sampler = SearchSpaceSampler(
        {
            "ridge": {
                "alpha": {"type": "log_float", "low": 0.001, "high": 10.0},
                "fit_intercept": {"type": "categorical", "values": [True, False]},
            }
        }
    )
    import numpy as np

    first = sampler.random_sample("ridge", np.random.default_rng(1))
    second = sampler.random_sample("ridge", np.random.default_rng(2))

    assert 0.001 <= first["alpha"] <= 10.0
    assert isinstance(first["fit_intercept"], bool)
    assert first != second


def test_optimization_runner_random_searches_all_models_and_optuna_only_top_models(tmp_path) -> None:
    config = _fast_config(tmp_path)
    model_keys = ("linear_regression", "ridge", "elasticnet")

    artifacts = HyperparameterOptimizationRunner(config=config).run(
        dataset_key="dataset_0172",
        X=_feature_frame(),
        y=_target(),
        target_name="wear rate",
        model_keys=model_keys,
        log_to_mlflow=False,
    )

    stage1 = artifacts.history.loc[artifacts.history["stage"] == "stage1_random_search"]
    stage3 = artifacts.history.loc[artifacts.history["stage"] == "stage3_optuna"]

    assert set(stage1["model_key"]) == set(model_keys)
    assert len(artifacts.top_models) == 2
    assert set(stage3["model_key"]) == set(artifacts.top_models["model_key"])
    assert not set(model_keys).difference(stage1["model_key"])
    assert (artifacts.root_dir / "tables" / "optimization_history.csv").exists()
    assert (artifacts.root_dir / "tables" / "best_parameters.csv").exists()
    assert (artifacts.root_dir / "tables" / "parameter_importance.csv").exists()
    assert artifacts.summary_path is not None
    assert artifacts.summary_path.exists()
    assert any(path.name == "optimization_history.png" for path in artifacts.plot_paths)
    assert any(path.name == "stage2_top_models.png" for path in artifacts.plot_paths)


def test_best_parameters_can_come_from_random_or_optuna_stage(tmp_path) -> None:
    artifacts = HyperparameterOptimizationRunner(config=_fast_config(tmp_path)).run(
        dataset_key="dataset_0172",
        X=_feature_frame(),
        y=_target(),
        target_name="wear rate",
        model_keys=("ridge", "elasticnet"),
        log_to_mlflow=False,
    )

    assert set(artifacts.best_parameters["model_key"]) == {"ridge", "elasticnet"}
    assert set(artifacts.best_parameters["stage"]).issubset(
        {"stage1_random_search", "stage3_optuna"}
    )
    assert artifacts.best_parameters["params_json"].str.startswith("{").all()


def test_optimization_package_does_not_use_grid_search() -> None:
    optimization_root = Path("src/friction_surrogate_xai/optimization")
    source = "\n".join(path.read_text(encoding="utf-8") for path in optimization_root.glob("*.py"))

    assert "GridSearchCV" not in source
    assert "ParameterGrid" not in source
