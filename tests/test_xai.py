"""Tests for the explainability framework."""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from friction_surrogate_xai.xai import XAIReportGenerator
from friction_surrogate_xai.xai.config import load_xai_config, with_overrides
from friction_surrogate_xai.xai.interpretation import ScientificInterpreter


def _dependencies_available() -> bool:
    return importlib.util.find_spec("shap") is not None and importlib.util.find_spec("lime") is not None


def _feature_frame(n_rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tool Shape": [(index % 3) + 1 for index in range(n_rows)],
            "Rotational Speed": [100.0 + 12.0 * index for index in range(n_rows)],
            "Plunging Speed": [3.0 + 0.25 * (index % 6) + 0.05 * index for index in range(n_rows)],
            "Composite Volume Fraction (%)": [index % 2 for index in range(n_rows)],
        }
    )


def _target(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        0.002 * frame["Rotational Speed"]
        - 0.12 * frame["Plunging Speed"]
        + 0.35 * frame["Composite Volume Fraction (%)"]
        + 0.04 * frame["Tool Shape"],
        name="wear rate",
    )


def _fast_config(tmp_path):
    return with_overrides(
        load_xai_config(),
        output={"root_dir": str(tmp_path)},
        data={"background_sample_size": 12, "local_sample_count": 1, "random_state": 13},
        shap={
            "enabled": True,
            "max_display": 6,
            "dependence_top_n": 1,
            "interaction_top_n": 1,
            "waterfall_local_samples": 1,
        },
        permutation_importance={"enabled": True, "n_repeats": 2, "random_state": 13},
        tree_interpreter={"enabled": True, "max_local_samples": 1},
        lime={"enabled": True, "max_local_samples": 1, "num_features": 4, "random_state": 13},
        mlflow={"enabled": False},
    )


def test_xai_runner_writes_importance_only_reports_without_optional_packages(tmp_path) -> None:
    frame = _feature_frame()
    config = with_overrides(
        _fast_config(tmp_path),
        shap={"enabled": False},
        lime={"enabled": False},
        tree_interpreter={"enabled": False},
    )

    artifacts = XAIReportGenerator(config=config).generate(
        dataset_key="dataset_0172",
        target_name="wear rate",
        model_key="random_forest",
        X=frame,
        y=_target(frame),
        params_override={"n_estimators": 8, "max_depth": 3, "min_samples_leaf": 2, "n_jobs": 1},
        log_to_mlflow=False,
    )

    generated_table_names = {path.name for path in artifacts.table_paths}
    assert "permutation_importance.csv" in generated_table_names
    assert "tree_importance.csv" in generated_table_names
    assert "scientific_feature_interpretations.csv" in generated_table_names
    assert artifacts.markdown_paths[0].exists()
    assert artifacts.mlflow_metrics["n_processed_features"] > 0


@pytest.mark.skipif(not _dependencies_available(), reason="SHAP and LIME are not installed")
def test_xai_report_generator_writes_figures_tables_and_scientific_summary(tmp_path) -> None:
    frame = _feature_frame()
    artifacts = XAIReportGenerator(config=_fast_config(tmp_path)).generate(
        dataset_key="dataset_0172",
        target_name="wear rate",
        model_key="random_forest",
        X=frame,
        y=_target(frame),
        params_override={"n_estimators": 8, "max_depth": 3, "min_samples_leaf": 2, "n_jobs": 1},
        log_to_mlflow=False,
    )

    generated_table_names = {path.name for path in artifacts.table_paths}
    generated_figure_names = {path.name for path in artifacts.figure_paths}

    assert "global_shap_importance.csv" in generated_table_names
    assert "local_shap_values.csv" in generated_table_names
    assert "permutation_importance.csv" in generated_table_names
    assert "tree_importance.csv" in generated_table_names
    assert "lime_local_explanations.csv" in generated_table_names
    assert "scientific_feature_interpretations.csv" in generated_table_names
    assert "shap_summary.png" in generated_figure_names
    assert "permutation_importance.png" in generated_figure_names
    assert any(path.name.endswith("_lime.png") for path in artifacts.figure_paths)
    assert artifacts.markdown_paths[0].exists()

    summary = artifacts.markdown_paths[0].read_text(encoding="utf-8")
    assert "Most Important Variables" in summary
    assert "Positive And Negative Effects" in summary
    assert "Nonlinear Behavior" in summary
    assert "Feature Interactions" in summary
    assert "Possible Engineering Interpretation" in summary


def test_scientific_interpreter_turns_xai_tables_into_domain_statements() -> None:
    shap_global = pd.DataFrame(
        {
            "feature": ["Rotational Speed", "Plunging Speed"],
            "mean_abs_shap": [0.4, 0.2],
            "mean_signed_shap": [0.1, -0.05],
        }
    )
    shap_effects = pd.DataFrame(
        {
            "feature": ["Rotational Speed", "Plunging Speed"],
            "correlation_feature_value_shap": [0.8, -0.7],
            "direction": ["positive", "negative"],
            "nonlinear_signal": ["approximately_monotonic", "non_monotonic_effect"],
            "low_value_mean_shap": [-0.1, 0.2],
            "high_value_mean_shap": [0.3, -0.3],
        }
    )
    interactions = pd.DataFrame(
        {
            "feature_a": ["Rotational Speed"],
            "feature_b": ["Composite Volume Fraction (%)"],
            "mean_abs_shap_product": [0.12],
            "feature_a_value_vs_feature_b_shap_corr": [0.33],
        }
    )

    interpretation = ScientificInterpreter(
        load_xai_config().interpretation,
    ).interpret(
        shap_global=shap_global,
        shap_effects=shap_effects,
        shap_interactions=interactions,
        permutation=pd.DataFrame(),
        tree_importance=pd.DataFrame(),
        target_name="wear rate",
    )

    assert "Rotational Speed" in interpretation.markdown_sections["most_important_variables"]
    assert "Positive effects" in interpretation.markdown_sections["positive_negative_effects"]
    assert "Plunging Speed" in interpretation.markdown_sections["nonlinear_behavior"]
    assert "Composite Volume Fraction" in interpretation.markdown_sections["feature_interactions"]
    assert "heat generation" in interpretation.feature_interpretations["engineering_note"].iloc[0]
