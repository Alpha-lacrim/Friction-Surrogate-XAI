"""Uncertainty estimators for native GPR and bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import KFold

from friction_surrogate_xai.models import ModelFactory
from friction_surrogate_xai.preprocessing import PreprocessingPipelineFactory


@dataclass(frozen=True)
class UncertaintyResult:
    """Prediction interval output for one model."""

    model_key: str
    method: str
    prediction_intervals: pd.DataFrame
    confidence_bands: pd.DataFrame
    summary: pd.DataFrame


class GPRUncertaintyEstimator:
    """Estimate uncertainty from Gaussian Process predictive distributions."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        preprocessing_factory: PreprocessingPipelineFactory,
        config: dict[str, Any],
        confidence_level: float,
    ) -> None:
        self.model_factory = model_factory
        self.preprocessing_factory = preprocessing_factory
        self.config = config
        self.confidence_level = confidence_level
        self.z_value = float(norm.ppf((1.0 + confidence_level) / 2.0))

    def estimate(
        self,
        *,
        dataset_key: str,
        model_key: str,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        params_override: dict[str, Any] | None = None,
    ) -> UncertaintyResult:
        """Generate fold-local GPR prediction intervals."""
        target_frame = _as_target_frame(y)
        rows: list[dict[str, Any]] = []
        folds = self._folds(len(X))
        method = str(self.config.get("method_name", "gpr_predictive_distribution"))

        for fold_id, (train_indices, validation_indices, seed) in enumerate(folds):
            X_train = X.iloc[train_indices].copy()
            X_validation = X.iloc[validation_indices].copy()
            preprocessor = clone(self.preprocessing_factory.build_for_dataset(dataset_key))
            X_train_processed = preprocessor.fit_transform(
                X_train,
                target_frame.iloc[train_indices],
            )
            X_validation_processed = preprocessor.transform(X_validation)

            for target_name in target_frame.columns:
                y_train = target_frame.iloc[train_indices][target_name]
                y_validation = target_frame.iloc[validation_indices][target_name]
                estimator = self.model_factory.build(
                    model_key,
                    random_state=seed,
                    params_override=params_override,
                )
                estimator.fit(X_train_processed, y_train.to_numpy())
                predictive_mean, predictive_std = estimator.predict(
                    X_validation_processed,
                    return_std=True,
                )
                predictive_mean = np.asarray(predictive_mean, dtype=float).ravel()
                predictive_std = np.asarray(predictive_std, dtype=float).ravel()
                predictive_std = np.maximum(predictive_std, 0.0)
                lower = predictive_mean - self.z_value * predictive_std
                upper = predictive_mean + self.z_value * predictive_std

                for local_position, sample_index in enumerate(X_validation.index):
                    true_value = float(y_validation.iloc[local_position])
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "model_key": model_key,
                            "method": method,
                            "target": target_name,
                            "sample_index": sample_index,
                            "fold_id": fold_id,
                            "seed": seed,
                            "y_true": true_value,
                            "predictive_mean": predictive_mean[local_position],
                            "predictive_variance": predictive_std[local_position] ** 2,
                            "predictive_std": predictive_std[local_position],
                            "interval_lower": lower[local_position],
                            "interval_upper": upper[local_position],
                            "interval_width": upper[local_position] - lower[local_position],
                            "covered": bool(
                                lower[local_position] <= true_value <= upper[local_position]
                            ),
                            "prediction_count": 1,
                            "interval_level": self.confidence_level,
                            "preprocessing_policy": "fit_preprocessor_inside_each_fold",
                        }
                    )

        intervals = pd.DataFrame(rows)
        bands = confidence_bands(intervals)
        summary = summarize_intervals(intervals, confidence_level=self.confidence_level)
        return UncertaintyResult(model_key, method, intervals, bands, summary)

    def _folds(self, n_samples: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
        n_splits = max(2, min(int(self.config.get("cv_splits", 5)), n_samples))
        shuffle = bool(self.config.get("shuffle", True))
        seeds = tuple(self.config.get("repeated_seeds", (42,)))
        folds: list[tuple[np.ndarray, np.ndarray, int]] = []
        sample_positions = np.arange(n_samples)
        for seed in seeds:
            splitter = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=seed if shuffle else None,
            )
            for train_indices, validation_indices in splitter.split(sample_positions):
                folds.append((train_indices, validation_indices, int(seed)))
        return folds


class BootstrapUncertaintyEstimator:
    """Estimate prediction intervals from bootstrap out-of-bag predictions."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        preprocessing_factory: PreprocessingPipelineFactory,
        config: dict[str, Any],
        confidence_level: float,
        random_state: int,
    ) -> None:
        self.model_factory = model_factory
        self.preprocessing_factory = preprocessing_factory
        self.config = config
        self.confidence_level = confidence_level
        self.random_state = random_state

    def estimate(
        self,
        *,
        dataset_key: str,
        model_key: str,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        params_override: dict[str, Any] | None = None,
    ) -> UncertaintyResult:
        """Generate bootstrap OOB prediction intervals."""
        target_frame = _as_target_frame(y)
        raw_rows: list[dict[str, Any]] = []
        method = str(self.config.get("method_name", "bootstrap_oob_prediction_interval"))
        n_samples = len(X)
        all_positions = np.arange(n_samples)
        all_index = np.asarray(X.index)
        n_iterations = int(self.config.get("n_iterations", 200))
        sample_fraction = float(self.config.get("sample_fraction", 1.0))
        train_size = max(1, int(round(n_samples * sample_fraction)))
        require_oob = bool(self.config.get("require_oob_samples", True))

        for iteration in range(n_iterations):
            seed = self.random_state + iteration
            rng = np.random.default_rng(seed)
            train_indices = rng.choice(all_positions, size=train_size, replace=True)
            validation_indices = np.setdiff1d(
                all_positions,
                np.unique(train_indices),
                assume_unique=False,
            )
            if require_oob and len(validation_indices) == 0:
                continue
            if len(validation_indices) == 0:
                validation_indices = all_positions

            X_train = X.iloc[train_indices].copy()
            X_validation = X.iloc[validation_indices].copy()
            preprocessor = clone(self.preprocessing_factory.build_for_dataset(dataset_key))
            X_train_processed = preprocessor.fit_transform(
                X_train,
                target_frame.iloc[train_indices],
            )
            X_validation_processed = preprocessor.transform(X_validation)

            for target_name in target_frame.columns:
                y_train = target_frame.iloc[train_indices][target_name]
                estimator = self.model_factory.build(
                    model_key,
                    random_state=seed,
                    params_override=params_override,
                )
                estimator.fit(X_train_processed, y_train.to_numpy())
                predictions = np.asarray(
                    estimator.predict(X_validation_processed),
                    dtype=float,
                ).ravel()
                for local_position, sample_position in enumerate(validation_indices):
                    raw_rows.append(
                        {
                            "dataset_key": dataset_key,
                            "model_key": model_key,
                            "method": method,
                            "target": target_name,
                            "sample_index": all_index[sample_position],
                            "bootstrap_iteration": iteration,
                            "seed": seed,
                            "prediction": predictions[local_position],
                            "preprocessing_policy": "fit_preprocessor_inside_each_bootstrap_sample",
                        }
                    )

        raw_predictions = pd.DataFrame(raw_rows)
        intervals = self._aggregate_bootstrap_predictions(
            raw_predictions=raw_predictions,
            target_frame=target_frame,
            model_key=model_key,
            method=method,
            dataset_key=dataset_key,
        )
        bands = confidence_bands(intervals)
        summary = summarize_intervals(intervals, confidence_level=self.confidence_level)
        return UncertaintyResult(model_key, method, intervals, bands, summary)

    def _aggregate_bootstrap_predictions(
        self,
        *,
        raw_predictions: pd.DataFrame,
        target_frame: pd.DataFrame,
        model_key: str,
        method: str,
        dataset_key: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        lower_quantile = (1.0 - self.confidence_level) / 2.0
        upper_quantile = 1.0 - lower_quantile
        min_predictions = int(self.config.get("min_predictions_per_sample", 5))

        for target_name in target_frame.columns:
            target_values = target_frame[target_name]
            target_predictions = raw_predictions.loc[raw_predictions["target"] == target_name]
            for sample_index, true_value in target_values.items():
                sample_predictions = pd.to_numeric(
                    target_predictions.loc[
                        target_predictions["sample_index"] == sample_index,
                        "prediction",
                    ],
                    errors="coerce",
                ).dropna()
                prediction_count = int(len(sample_predictions))
                if prediction_count:
                    mean = float(sample_predictions.mean())
                    variance = (
                        float(sample_predictions.var(ddof=1)) if prediction_count > 1 else 0.0
                    )
                    lower = float(sample_predictions.quantile(lower_quantile))
                    upper = float(sample_predictions.quantile(upper_quantile))
                    covered = bool(lower <= float(true_value) <= upper)
                    status = (
                        "ok"
                        if prediction_count >= min_predictions
                        else "low_oob_prediction_count"
                    )
                else:
                    mean = variance = lower = upper = np.nan
                    covered = False
                    status = "no_oob_predictions"
                rows.append(
                    {
                        "dataset_key": dataset_key,
                        "model_key": model_key,
                        "method": method,
                        "target": target_name,
                        "sample_index": sample_index,
                        "fold_id": np.nan,
                        "seed": np.nan,
                        "y_true": float(true_value),
                        "predictive_mean": mean,
                        "predictive_variance": variance,
                        "predictive_std": np.sqrt(variance) if np.isfinite(variance) else np.nan,
                        "interval_lower": lower,
                        "interval_upper": upper,
                        "interval_width": upper - lower if np.isfinite(upper - lower) else np.nan,
                        "covered": covered,
                        "prediction_count": prediction_count,
                        "interval_level": self.confidence_level,
                        "preprocessing_policy": "fit_preprocessor_inside_each_bootstrap_sample",
                        "status": status,
                    }
                )
        return pd.DataFrame(rows)


def summarize_intervals(
    prediction_intervals: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    """Summarize interval coverage and width by dataset/model/target."""
    if prediction_intervals.empty:
        return pd.DataFrame()
    group_columns = ["dataset_key", "model_key", "method", "target"]
    rows: list[dict[str, Any]] = []
    for group_values, group in prediction_intervals.groupby(group_columns, dropna=False):
        valid = group.loc[
            group["interval_lower"].notna()
            & group["interval_upper"].notna()
            & group["y_true"].notna()
        ].copy()
        widths = pd.to_numeric(valid["interval_width"], errors="coerce").dropna()
        variances = pd.to_numeric(valid["predictive_variance"], errors="coerce").dropna()
        coverage = (
            float(valid["covered"].astype(bool).mean()) if not valid.empty else np.nan
        )
        row = dict(zip(group_columns, group_values, strict=False))
        row.update(
            {
                "interval_level": confidence_level,
                "sample_count": int(len(valid)),
                "coverage_probability": coverage,
                "coverage_error": (
                    abs(coverage - confidence_level) if np.isfinite(coverage) else np.nan
                ),
                "mean_interval_width": float(widths.mean()) if not widths.empty else np.nan,
                "median_interval_width": float(widths.median()) if not widths.empty else np.nan,
                "std_interval_width": float(widths.std(ddof=1)) if len(widths) > 1 else 0.0,
                "mean_predictive_variance": (
                    float(variances.mean()) if not variances.empty else np.nan
                ),
                "min_prediction_count": (
                    int(valid["prediction_count"].min()) if not valid.empty else 0
                ),
                "mean_prediction_count": (
                    float(valid["prediction_count"].mean()) if not valid.empty else 0.0
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_bands(prediction_intervals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated interval rows into one confidence-band row per sample."""
    if prediction_intervals.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset_key", "model_key", "method", "target", "sample_index"]
    for group_values, group in prediction_intervals.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group_values, strict=False))
        row.update(
            {
                "y_true": float(pd.to_numeric(group["y_true"], errors="coerce").mean()),
                "predictive_mean": float(
                    pd.to_numeric(group["predictive_mean"], errors="coerce").mean()
                ),
                "predictive_variance": float(
                    pd.to_numeric(group["predictive_variance"], errors="coerce").mean()
                ),
                "interval_lower": float(
                    pd.to_numeric(group["interval_lower"], errors="coerce").mean()
                ),
                "interval_upper": float(
                    pd.to_numeric(group["interval_upper"], errors="coerce").mean()
                ),
                "interval_width": float(
                    pd.to_numeric(group["interval_width"], errors="coerce").mean()
                ),
                "covered": bool(group["covered"].astype(bool).mean() >= 0.5),
                "prediction_count": int(
                    pd.to_numeric(group["prediction_count"], errors="coerce").sum()
                ),
                "band_count": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _as_target_frame(y: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        return y.copy()
    name = y.name or "target"
    return y.to_frame(name=name)
