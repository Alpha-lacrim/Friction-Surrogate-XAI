"""Tests for leakage-safe preprocessing pipelines."""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from friction_surrogate_xai.preprocessing import (
    FeatureValidationError,
    PreprocessingArtifactGenerator,
    PreprocessingPipelineFactory,
    load_preprocessing_config,
)
from friction_surrogate_xai.preprocessing.config import with_overrides


def _feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Tool Shape": [1, 2, 3, 1, 2, 3],
            "Rotational Speed": [100.0, 110.0, 120.0, 1000.0, 1100.0, 1200.0],
            "Plunging Speed": [10.0, 11.0, 12.0, 100.0, 110.0, 120.0],
            "Composite Volume Fraction (%)": [0, 0, 0, 1, 1, 1],
        }
    )


def test_factory_builds_unfitted_sklearn_pipeline() -> None:
    pipeline = PreprocessingPipelineFactory().build_for_dataset("dataset_0172")

    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == [
        "feature_validation",
        "constant_feature_removal",
        "column_preprocessor",
    ]
    assert not hasattr(pipeline.named_steps["column_preprocessor"], "scaler_")


def test_scaler_fits_only_on_data_passed_to_pipeline_fit() -> None:
    frame = _feature_frame()
    train = frame.iloc[:3]
    full_mean = frame[["Rotational Speed", "Plunging Speed"]].mean().to_numpy()

    pipeline = PreprocessingPipelineFactory().build_for_dataset("dataset_0172")
    pipeline.fit(train)

    preprocessor = pipeline.named_steps["column_preprocessor"]
    scaler_mean = preprocessor.scaler_.mean_
    train_mean = train[["Rotational Speed", "Plunging Speed"]].mean().to_numpy()

    assert scaler_mean.tolist() == pytest.approx(train_mean.tolist())
    assert scaler_mean.tolist() != pytest.approx(full_mean.tolist())


def test_constant_feature_removal_is_learned_inside_pipeline_fit() -> None:
    frame = _feature_frame()
    train = frame.iloc[:3]
    test = frame.iloc[3:]

    pipeline = PreprocessingPipelineFactory().build_for_dataset("dataset_0172")
    pipeline.fit(train)
    transformed_test = pipeline.transform(test)

    remover = pipeline.named_steps["constant_feature_removal"]
    assert "Composite Volume Fraction (%)" in remover.constant_features_
    assert "Composite Volume Fraction (%)" not in transformed_test.columns


def test_one_hot_encoder_is_configurable_and_handles_unknown_categories() -> None:
    train = _feature_frame().iloc[:3]
    test = pd.DataFrame(
        {
            "Tool Shape": [99],
            "Rotational Speed": [130.0],
            "Plunging Speed": [13.0],
            "Composite Volume Fraction (%)": [0],
        }
    )

    pipeline = PreprocessingPipelineFactory().build_for_dataset("dataset_0172")
    pipeline.fit(train)
    transformed = pipeline.transform(test)

    assert "Tool Shape_1" in transformed.columns
    assert "Tool Shape_2" in transformed.columns
    assert "Tool Shape_3" in transformed.columns
    assert transformed.filter(like="Tool Shape_").sum(axis=1).iloc[0] == 0


def test_feature_validator_rejects_missing_required_features() -> None:
    frame = _feature_frame().drop(columns=["Plunging Speed"])
    pipeline = PreprocessingPipelineFactory().build_for_dataset("dataset_0172")

    with pytest.raises(FeatureValidationError):
        pipeline.fit(frame)


def test_artifact_generator_saves_unfitted_pipeline_and_metadata(tmp_path) -> None:
    config = with_overrides(
        load_preprocessing_config(),
        output={"root_dir": str(tmp_path)},
        mlflow={"enabled": False},
    )

    result = PreprocessingArtifactGenerator(config=config).run_dataset(
        "dataset_0172",
        log_to_mlflow=False,
    )

    artifacts = result.artifacts
    assert artifacts.pipeline_path.exists()
    assert artifacts.config_path.exists()
    assert artifacts.schema_path.exists()
    assert artifacts.validation_report_path.exists()
    assert artifacts.leakage_policy_path.exists()

    loaded_pipeline = joblib.load(artifacts.pipeline_path)
    assert isinstance(loaded_pipeline, Pipeline)
    assert not hasattr(loaded_pipeline.named_steps["column_preprocessor"], "scaler_")

