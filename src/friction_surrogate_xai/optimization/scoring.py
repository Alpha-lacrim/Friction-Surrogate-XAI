"""Leakage-safe trial scoring for hyperparameter optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from friction_surrogate_xai.evaluation.arrays import infer_target_names
from friction_surrogate_xai.evaluation.metrics import RegressionMetricCalculator
from friction_surrogate_xai.models import ModelFactory
from friction_surrogate_xai.preprocessing import PreprocessingPipelineFactory


@dataclass(frozen=True)
class TrialScore:
    """Aggregated CV score for one optimization trial."""

    metrics: dict[str, Any]
    fold_metrics: pd.DataFrame


class OptimizationTrialEvaluator:
    """Evaluate sampled parameters with fold-local preprocessing."""

    def __init__(
        self,
        *,
        model_factory: ModelFactory,
        preprocessing_factory: PreprocessingPipelineFactory,
        metrics: tuple[str, ...],
        primary_metric: str,
        nrmse_denominator: str,
        higher_is_better: tuple[str, ...],
    ) -> None:
        self.model_factory = model_factory
        self.preprocessing_factory = preprocessing_factory
        self.primary_metric = primary_metric
        self.higher_is_better = set(higher_is_better)
        self.metric_calculator = RegressionMetricCalculator(
            metrics=metrics,
            nrmse_denominator=nrmse_denominator,
            aggregate_targets=True,
        )

    def evaluate(
        self,
        *,
        dataset_key: str,
        model_key: str,
        params: dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        cv_splits: int,
        seed: int,
    ) -> TrialScore:
        """Evaluate one parameter set with KFold CV."""
        target_names = infer_target_names(y)
        folds = self._folds(n_samples=len(X), cv_splits=cv_splits, seed=seed)
        fold_rows: list[dict[str, Any]] = []

        for fold_id, (train_indices, validation_indices) in enumerate(folds):
            fold_metrics = self._evaluate_fold(
                dataset_key=dataset_key,
                model_key=model_key,
                params=params,
                X=X,
                y=y,
                target_names=target_names,
                train_indices=train_indices,
                validation_indices=validation_indices,
                fold_id=fold_id,
                seed=seed,
            )
            fold_rows.extend(fold_metrics)

        fold_frame = pd.DataFrame(fold_rows)
        aggregate = self._aggregate_fold_metrics(fold_frame)
        return TrialScore(metrics=aggregate, fold_metrics=fold_frame)

    def _evaluate_fold(
        self,
        *,
        dataset_key: str,
        model_key: str,
        params: dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        target_names: tuple[str, ...],
        train_indices: np.ndarray,
        validation_indices: np.ndarray,
        fold_id: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        X_train = X.iloc[train_indices].copy()
        X_validation = X.iloc[validation_indices].copy()
        y_train = y.iloc[train_indices].copy()
        y_validation = y.iloc[validation_indices].copy()

        preprocessor = clone(self.preprocessing_factory.build_for_dataset(dataset_key))
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_validation_processed = preprocessor.transform(X_validation)

        estimator = self.model_factory.build(
            model_key,
            random_state=seed,
            params_override=params,
        )
        estimator = self._maybe_wrap_multi_output(model_key, estimator, y_train)
        estimator.fit(X_train_processed, _fit_target(y_train))

        rows: list[dict[str, Any]] = []
        for split, y_true, predictions in (
            ("train", y_train, estimator.predict(X_train_processed)),
            ("validation", y_validation, estimator.predict(X_validation_processed)),
        ):
            metrics = self.metric_calculator.evaluate_predictions(
                y_true,
                predictions,
                target_names=target_names,
                dataset_key=dataset_key,
                model_name=model_key,
                split=split,
            )
            selected = _select_primary_row(metrics)
            row = selected.to_dict()
            row["model_key"] = model_key
            row["fold_id"] = fold_id
            row["seed"] = seed
            row["train_size"] = len(train_indices)
            row["validation_size"] = len(validation_indices)
            row["preprocessing_policy"] = "fit_preprocessor_inside_each_fold"
            rows.append(row)
        return rows

    def _aggregate_fold_metrics(self, fold_frame: pd.DataFrame) -> dict[str, Any]:
        train = fold_frame.loc[fold_frame["split"] == "train"].copy()
        validation = fold_frame.loc[fold_frame["split"] == "validation"].copy()
        metrics: dict[str, Any] = {}
        metric_columns = [
            column
            for column in ("r2", "rmse", "nrmse", "mae")
            if column in validation.columns
        ]
        for metric in metric_columns:
            train_values = pd.to_numeric(train[metric], errors="coerce").dropna()
            validation_values = pd.to_numeric(validation[metric], errors="coerce").dropna()
            metrics[f"mean_train_{metric}"] = float(train_values.mean()) if not train_values.empty else np.nan
            metrics[f"mean_validation_{metric}"] = (
                float(validation_values.mean()) if not validation_values.empty else np.nan
            )
            metrics[f"std_validation_{metric}"] = (
                float(validation_values.std(ddof=1)) if len(validation_values) > 1 else 0.0
            )
            if metric in self.higher_is_better:
                gap = metrics[f"mean_train_{metric}"] - metrics[f"mean_validation_{metric}"]
            else:
                gap = metrics[f"mean_validation_{metric}"] - metrics[f"mean_train_{metric}"]
            metrics[f"generalization_gap_{metric}"] = float(gap) if np.isfinite(gap) else np.nan

        primary_value = metrics.get(f"mean_validation_{self.primary_metric}", np.nan)
        metrics["objective_value"] = float(primary_value) if np.isfinite(primary_value) else np.nan
        metrics["fold_count"] = int(validation["fold_id"].nunique())
        return metrics

    @staticmethod
    def _folds(
        *,
        n_samples: int,
        cv_splits: int,
        seed: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if n_samples < 2:
            raise ValueError("At least two samples are required for optimization.")
        n_splits = max(2, min(int(cv_splits), n_samples))
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.arange(n_samples)))

    @staticmethod
    def _maybe_wrap_multi_output(
        model_key: str,
        estimator: BaseEstimator,
        y_train: pd.Series | pd.DataFrame,
    ) -> BaseEstimator:
        native_multi_output = {
            "linear_regression",
            "ridge",
            "random_forest",
            "extra_trees",
            "gaussian_process_regression",
        }
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1 and model_key not in native_multi_output:
            return MultiOutputRegressor(estimator)
        return estimator


def _fit_target(values: pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.to_numpy()
    if values.shape[1] == 1:
        return values.iloc[:, 0].to_numpy()
    return values.to_numpy()


def _select_primary_row(metrics: pd.DataFrame) -> pd.Series:
    aggregate = metrics.loc[metrics["target"] == "__aggregate__"]
    if not aggregate.empty:
        return aggregate.iloc[0]
    return metrics.iloc[0]
