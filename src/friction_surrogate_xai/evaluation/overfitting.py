"""Overfitting audits for tiny-data regression models."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename, write_csv
from friction_surrogate_xai.evaluation.arrays import infer_target_names
from friction_surrogate_xai.evaluation.metrics import RegressionMetricCalculator
from friction_surrogate_xai.evaluation.reports import EvaluationReportWriter
from friction_surrogate_xai.evaluation.validation import FoldSplit, ValidationStrategyFactory
from friction_surrogate_xai.experiments.mlflow_config import load_mlflow_settings
from friction_surrogate_xai.models import ModelFactory, ModelingConfig, load_modeling_config
from friction_surrogate_xai.preprocessing import PreprocessingPipelineFactory


@dataclass(frozen=True)
class OverfittingAuditArtifacts:
    """Generated artifacts for one overfitting audit."""

    dataset_key: str
    target_name: str
    root_dir: Path
    table_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]
    fold_scores: pd.DataFrame
    summary: pd.DataFrame
    model_policies: pd.DataFrame
    split_summary: pd.DataFrame

    @property
    def report_path(self) -> Path | None:
        """Return the main Markdown report path if generated."""
        for path in self.markdown_paths:
            if path.name == "overfitting_report.md":
                return path
        return None


@dataclass(frozen=True)
class FitMetadata:
    """Metadata captured while fitting a model in a fold."""

    early_stopping_requested: bool
    early_stopping_applied: bool
    early_stopping_status: str


class OverfittingRiskAnalyzer:
    """Identify likely overfitting from train/validation fold scores."""

    def __init__(self, config: ModelingConfig | None = None) -> None:
        self.config = config or load_modeling_config()
        self.thresholds = self.config.overfitting_detection
        self.metrics = tuple(self.config.scoring.get("metrics", ("r2", "rmse", "nrmse", "mae")))
        self.higher_is_better = set(self.config.scoring.get("higher_is_better", ("r2",)))
        self.primary_metric = str(self.config.scoring.get("primary_metric", "r2"))

    def summarize(self, fold_scores: pd.DataFrame) -> pd.DataFrame:
        """Summarize fold scores and flag likely overfitting."""
        if fold_scores.empty:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        join_columns = [
            "dataset_key",
            "model_key",
            "model_name",
            "target",
            "validation_strategy",
            "fold_id",
            "seed",
            "repeat_id",
        ]
        train_scores = fold_scores.loc[fold_scores["split"] == "train"].copy()
        validation_scores = fold_scores.loc[fold_scores["split"] == "validation"].copy()
        merged = train_scores.merge(
            validation_scores,
            on=join_columns,
            suffixes=("_train", "_validation"),
        )

        for metric in self.metrics:
            if f"{metric}_train" not in merged.columns or f"{metric}_validation" not in merged.columns:
                continue
            metric_frame = merged.copy()
            metric_frame["train_value"] = pd.to_numeric(
                metric_frame[f"{metric}_train"],
                errors="coerce",
            )
            metric_frame["validation_value"] = pd.to_numeric(
                metric_frame[f"{metric}_validation"],
                errors="coerce",
            )
            if metric in self.higher_is_better:
                metric_frame["generalization_gap"] = (
                    metric_frame["train_value"] - metric_frame["validation_value"]
                )
                gap_direction = "train_minus_validation"
            else:
                metric_frame["generalization_gap"] = (
                    metric_frame["validation_value"] - metric_frame["train_value"]
                )
                gap_direction = "validation_minus_train"

            group_columns = [
                "dataset_key",
                "model_key",
                "model_name",
                "target",
                "validation_strategy",
            ]
            for group_values, group in metric_frame.groupby(group_columns, dropna=False):
                valid = group[["train_value", "validation_value", "generalization_gap"]].dropna()
                row = dict(zip(group_columns, group_values, strict=False))
                row.update(self._summary_values(valid, metric, gap_direction))
                rows.append(row)
        return pd.DataFrame(rows)

    def _summary_values(
        self,
        values: pd.DataFrame,
        metric: str,
        gap_direction: str,
    ) -> dict[str, Any]:
        fold_count = int(len(values))
        if fold_count == 0:
            return {
                "metric": metric,
                "fold_count": 0,
                "mean_train": np.nan,
                "mean_validation": np.nan,
                "mean_generalization_gap": np.nan,
                "std_validation": np.nan,
                "std_generalization_gap": np.nan,
                "gap_direction": gap_direction,
                "risk_level": "unknown",
                "likely_overfitting": False,
                "risk_reasons": "no finite fold scores",
            }

        mean_train = float(values["train_value"].mean())
        mean_validation = float(values["validation_value"].mean())
        mean_gap = float(values["generalization_gap"].mean())
        std_validation = float(values["validation_value"].std(ddof=1)) if fold_count > 1 else 0.0
        std_gap = float(values["generalization_gap"].std(ddof=1)) if fold_count > 1 else 0.0
        risk_level, likely_overfitting, reasons = self._risk_decision(
            metric=metric,
            fold_count=fold_count,
            mean_train=mean_train,
            mean_validation=mean_validation,
            mean_gap=mean_gap,
            std_validation=std_validation,
        )
        return {
            "metric": metric,
            "fold_count": fold_count,
            "mean_train": mean_train,
            "mean_validation": mean_validation,
            "mean_generalization_gap": mean_gap,
            "std_validation": std_validation,
            "std_generalization_gap": std_gap,
            "gap_direction": gap_direction,
            "risk_level": risk_level,
            "likely_overfitting": likely_overfitting,
            "risk_reasons": "; ".join(reasons) if reasons else "no configured overfitting signal exceeded",
        }

    def _risk_decision(
        self,
        *,
        metric: str,
        fold_count: int,
        mean_train: float,
        mean_validation: float,
        mean_gap: float,
        std_validation: float,
    ) -> tuple[str, bool, list[str]]:
        min_folds = int(self.thresholds.get("min_validation_folds", 3))
        if fold_count < min_folds:
            return "unknown", False, [f"fewer than {min_folds} validation folds"]

        gap_warning = float(self.thresholds.get("train_validation_gap_warning", 0.15))
        gap_high = float(self.thresholds.get("train_validation_gap_high", 0.30))
        std_warning = float(self.thresholds.get("validation_std_warning", 0.10))
        std_high = float(self.thresholds.get("validation_std_high", 0.20))
        high_train = float(self.thresholds.get("high_train_score_threshold", 0.90))
        low_validation = float(self.thresholds.get("low_validation_score_threshold", 0.60))

        reasons: list[str] = []
        severity = 0
        if np.isfinite(mean_gap) and mean_gap >= gap_high:
            severity = max(severity, 2)
            reasons.append(f"high generalization gap >= {gap_high}")
        elif np.isfinite(mean_gap) and mean_gap >= gap_warning:
            severity = max(severity, 1)
            reasons.append(f"generalization gap >= {gap_warning}")

        if np.isfinite(std_validation) and std_validation >= std_high:
            severity = max(severity, 2)
            reasons.append(f"high validation variance >= {std_high}")
        elif np.isfinite(std_validation) and std_validation >= std_warning:
            severity = max(severity, 1)
            reasons.append(f"validation variance >= {std_warning}")

        if (
            metric in self.higher_is_better
            and np.isfinite(mean_train)
            and np.isfinite(mean_validation)
            and mean_train >= high_train
            and mean_validation <= low_validation
        ):
            severity = max(severity, 2)
            reasons.append(
                f"train score >= {high_train} while validation score <= {low_validation}"
            )

        if severity >= 2:
            return "high", True, reasons
        if severity == 1:
            return "medium", True, reasons
        return "low", False, reasons


class OverfittingReportWriter:
    """Write overfitting audit CSV and Markdown reports."""

    def __init__(self, config: ModelingConfig | None = None) -> None:
        self.config = config or load_modeling_config()
        self.markdown = EvaluationReportWriter(self.config.reports)

    def write(
        self,
        *,
        dataset_key: str,
        target_name: str,
        root_dir: Path,
        fold_scores: pd.DataFrame,
        summary: pd.DataFrame,
        model_policies: pd.DataFrame,
        split_summary: pd.DataFrame,
    ) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        """Write all overfitting audit reports."""
        tables_dir = ensure_directory(root_dir / self.config.output.get("tables_dir_name", "tables"))
        markdown_dir = ensure_directory(
            root_dir / self.config.output.get("markdown_dir_name", "markdown")
        )
        table_paths: list[Path] = []
        markdown_paths: list[Path] = []

        if self.config.reports.get("write_csv", True):
            table_paths.extend(
                [
                    write_csv(fold_scores, tables_dir / "fold_scores.csv"),
                    write_csv(summary, tables_dir / "overfitting_summary.csv"),
                    write_csv(model_policies, tables_dir / "model_policies.csv"),
                    write_csv(split_summary, tables_dir / "split_summary.csv"),
                ]
            )

        if self.config.reports.get("write_markdown", True):
            markdown_paths.append(
                self._write_markdown_report(
                    dataset_key=dataset_key,
                    target_name=target_name,
                    summary=summary,
                    model_policies=model_policies,
                    output_path=markdown_dir / "overfitting_report.md",
                )
            )
            markdown_paths.append(
                self._write_table(summary, markdown_dir / "overfitting_summary.md")
            )
            markdown_paths.append(
                self._write_table(model_policies, markdown_dir / "model_policies.md")
            )
        return tuple(table_paths), tuple(markdown_paths)

    def _write_markdown_report(
        self,
        *,
        dataset_key: str,
        target_name: str,
        summary: pd.DataFrame,
        model_policies: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        primary_metric = str(self.config.scoring.get("primary_metric", "r2"))
        primary_summary = summary.loc[summary["metric"] == primary_metric].copy()
        flagged = primary_summary.loc[primary_summary["likely_overfitting"] == True]
        if flagged.empty:
            flagged_text = "_No models crossed the configured overfitting thresholds._"
        else:
            flagged_text = self.markdown._markdown_table(
                flagged[
                    [
                        "model_key",
                        "validation_strategy",
                        "mean_train",
                        "mean_validation",
                        "mean_generalization_gap",
                        "std_validation",
                        "risk_level",
                        "risk_reasons",
                    ]
                ]
            )

        lines = [
            f"# Overfitting Audit: {dataset_key} / {target_name}",
            "",
            "## Priority",
            "",
            "This audit prioritizes overfitting prevention over raw accuracy. Rows are never removed and preprocessing must be fit inside each validation fold.",
            "",
            "## Likely Overfitting Models",
            "",
            flagged_text,
            "",
            "## Model Anti-Overfitting Policies",
            "",
            self.markdown._markdown_table(model_policies),
            "",
            "## Generated Tables",
            "",
            "- `tables/fold_scores.csv`",
            "- `tables/overfitting_summary.csv`",
            "- `tables/model_policies.csv`",
            "- `tables/split_summary.csv`",
        ]
        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _write_table(self, dataframe: pd.DataFrame, output_path: Path) -> Path:
        ensure_directory(output_path.parent)
        output_path.write_text(self.markdown._markdown_table(dataframe), encoding="utf-8")
        return output_path


class OverfittingMLflowLogger:
    """Log overfitting audit metrics and artifacts to MLflow."""

    def __init__(self, config: ModelingConfig | None = None) -> None:
        self.config = config or load_modeling_config()

    def log(
        self,
        *,
        dataset_key: str,
        target_name: str,
        root_dir: Path,
        summary: pd.DataFrame,
        split_summary: pd.DataFrame,
    ) -> None:
        """Log one audit run into MLflow."""
        if not self.config.mlflow.get("enabled", True):
            return

        import mlflow

        settings = load_mlflow_settings()
        tracking_uri = settings.tracking_uri
        if tracking_uri.startswith("file:./"):
            tracking_uri = f"file:{project_root() / tracking_uri.removeprefix('file:./')}"
        if tracking_uri.startswith("file:"):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

        experiment_name = self.config.mlflow.get("experiment_name") or settings.experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"overfitting_{dataset_key}_{sanitize_filename(target_name)}"):
            mlflow.set_tag("dataset", dataset_key)
            mlflow.set_tag("target", target_name)
            for tag_key, tag_value in self.config.mlflow.get("tags", {}).items():
                mlflow.set_tag(tag_key, tag_value)
            mlflow.log_params(
                {
                    "dataset_key": dataset_key,
                    "target_name": target_name,
                    "primary_metric": self.config.scoring.get("primary_metric", "r2"),
                    "repeated_seeds": ";".join(map(str, self.config.repeated_seeds)),
                    "validation_config": json.dumps(self.config.validation, sort_keys=True),
                }
            )
            mlflow.log_metrics(self._metrics(summary, split_summary))
            artifact_prefix = self.config.mlflow.get("artifact_path_prefix", "overfitting")
            mlflow.log_artifacts(
                str(root_dir),
                artifact_path=f"{artifact_prefix}/{dataset_key}/{sanitize_filename(target_name)}",
            )

    @staticmethod
    def _metrics(summary: pd.DataFrame, split_summary: pd.DataFrame) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not summary.empty:
            metrics["likely_overfitting_count"] = float(summary["likely_overfitting"].sum())
            metrics["high_risk_count"] = float((summary["risk_level"] == "high").sum())
            metrics["medium_risk_count"] = float((summary["risk_level"] == "medium").sum())
        if not split_summary.empty:
            for _, row in split_summary.iterrows():
                value = row.get("value")
                if _is_finite(value):
                    metrics[f"split_{sanitize_filename(str(row.get('name', 'value')))}"] = float(value)
        return metrics


class OverfittingAuditRunner:
    """Run leakage-safe model audits and generate overfitting reports."""

    def __init__(
        self,
        config: ModelingConfig | None = None,
        model_factory: ModelFactory | None = None,
        preprocessing_factory: PreprocessingPipelineFactory | None = None,
    ) -> None:
        self.config = config or load_modeling_config()
        self.model_factory = model_factory or ModelFactory(self.config)
        self.preprocessing_factory = preprocessing_factory or PreprocessingPipelineFactory()
        self.validation_factory = ValidationStrategyFactory(self.config)
        self.metric_calculator = RegressionMetricCalculator(
            metrics=tuple(self.config.scoring.get("metrics", ("r2", "rmse", "nrmse", "mae"))),
            nrmse_denominator=str(self.config.scoring.get("nrmse_denominator", "range")),
            aggregate_targets=True,
        )
        self.analyzer = OverfittingRiskAnalyzer(self.config)
        self.writer = OverfittingReportWriter(self.config)
        self.mlflow_logger = OverfittingMLflowLogger(self.config)

    def run(
        self,
        *,
        dataset_key: str,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        target_name: str | None = None,
        model_keys: tuple[str, ...] | None = None,
        strategy: str = "repeated_kfold",
        include_bootstrap: bool = True,
        include_nested: bool = True,
        log_to_mlflow: bool | None = None,
    ) -> OverfittingAuditArtifacts:
        """Run the configured overfitting audit."""
        selected_models = model_keys or self.model_factory.enabled_model_keys()
        target_names = infer_target_names(y, (target_name,) if target_name else None)
        resolved_target = target_name or ";".join(target_names)
        n_samples = len(X)

        folds = list(self._folds_for_strategy(strategy, n_samples))
        if include_nested and self.config.validation.get("nested_cv", {}).get("enabled", True):
            folds.extend(nested.outer_fold for nested in self.validation_factory.nested_cv(n_samples))
        if include_bootstrap and self.config.validation.get("bootstrap", {}).get("enabled", True):
            folds.extend(self.validation_factory.bootstrap(n_samples))

        fold_scores = self._evaluate_folds(
            dataset_key=dataset_key,
            X=X,
            y=y,
            target_names=target_names,
            model_keys=selected_models,
            folds=tuple(folds),
        )
        summary = self.analyzer.summarize(fold_scores)
        model_policies = self.model_factory.policy_table(selected_models)
        split_summary = pd.DataFrame(
            [
                {"name": key, "value": value}
                for key, value in self.validation_factory.split_summary(n_samples).items()
            ]
        )
        root_dir = self._run_root(dataset_key, resolved_target)
        table_paths, markdown_paths = self.writer.write(
            dataset_key=dataset_key,
            target_name=resolved_target,
            root_dir=root_dir,
            fold_scores=fold_scores,
            summary=summary,
            model_policies=model_policies,
            split_summary=split_summary,
        )

        should_log = self.config.mlflow.get("enabled", True) if log_to_mlflow is None else log_to_mlflow
        if should_log:
            self.mlflow_logger.log(
                dataset_key=dataset_key,
                target_name=resolved_target,
                root_dir=root_dir,
                summary=summary,
                split_summary=split_summary,
            )

        return OverfittingAuditArtifacts(
            dataset_key=dataset_key,
            target_name=resolved_target,
            root_dir=root_dir,
            table_paths=table_paths,
            markdown_paths=markdown_paths,
            fold_scores=fold_scores,
            summary=summary,
            model_policies=model_policies,
            split_summary=split_summary,
        )

    def _evaluate_folds(
        self,
        *,
        dataset_key: str,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        target_names: tuple[str, ...],
        model_keys: tuple[str, ...],
        folds: tuple[FoldSplit, ...],
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for model_key in model_keys:
            for fold in folds:
                rows.append(
                    self._evaluate_one_fold(
                        dataset_key=dataset_key,
                        model_key=model_key,
                        X=X,
                        y=y,
                        target_names=target_names,
                        fold=fold,
                    )
                )
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _evaluate_one_fold(
        self,
        *,
        dataset_key: str,
        model_key: str,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        target_names: tuple[str, ...],
        fold: FoldSplit,
    ) -> pd.DataFrame:
        X_train = X.iloc[fold.train_indices].copy()
        X_validation = X.iloc[fold.validation_indices].copy()
        y_train = _select_rows(y, fold.train_indices)
        y_validation = _select_rows(y, fold.validation_indices)

        preprocessor = clone(self.preprocessing_factory.build_for_dataset(dataset_key))
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_validation_processed = preprocessor.transform(X_validation)

        estimator = self.model_factory.build(model_key, random_state=fold.seed)
        estimator = self._maybe_wrap_multi_output(model_key, estimator, y_train)
        fit_metadata = self._fit_estimator(
            model_key=model_key,
            estimator=estimator,
            X_train=X_train_processed,
            y_train=y_train,
            seed=fold.seed,
        )

        train_predictions = estimator.predict(X_train_processed)
        validation_predictions = estimator.predict(X_validation_processed)
        train_metrics = self.metric_calculator.evaluate_predictions(
            y_train,
            train_predictions,
            target_names=target_names,
            dataset_key=dataset_key,
            model_name=model_key,
            split="train",
        )
        validation_metrics = self.metric_calculator.evaluate_predictions(
            y_validation,
            validation_predictions,
            target_names=target_names,
            dataset_key=dataset_key,
            model_name=model_key,
            split="validation",
        )
        metrics = pd.concat([train_metrics, validation_metrics], ignore_index=True)
        spec = self.model_factory.spec(model_key)
        metrics.insert(1, "model_key", model_key)
        metrics["model_name"] = spec.display_name
        metrics["validation_strategy"] = fold.strategy
        metrics["fold_id"] = fold.fold_id
        metrics["seed"] = fold.seed
        metrics["repeat_id"] = fold.repeat_id
        metrics["train_size"] = len(fold.train_indices)
        metrics["validation_size"] = len(fold.validation_indices)
        metrics["early_stopping_requested"] = fit_metadata.early_stopping_requested
        metrics["early_stopping_applied"] = fit_metadata.early_stopping_applied
        metrics["early_stopping_status"] = fit_metadata.early_stopping_status
        metrics["preprocessing_policy"] = "fit_preprocessor_inside_each_fold"
        metrics["variant"] = "original"
        metrics["output_mode"] = "multi_output" if len(target_names) > 1 else "single_output"
        return metrics

    def _fit_estimator(
        self,
        *,
        model_key: str,
        estimator: BaseEstimator,
        X_train: Any,
        y_train: pd.Series | pd.DataFrame,
        seed: int | None,
    ) -> FitMetadata:
        early_config = self.model_factory.spec(model_key).early_stopping
        requested = bool(early_config.get("enabled", False))
        if isinstance(estimator, MultiOutputRegressor):
            estimator.fit(X_train, _fit_target(y_train))
            return FitMetadata(requested, False, "skipped_for_multi_output_wrapper")
        if model_key not in {"xgboost", "lightgbm"} or not requested:
            estimator.fit(X_train, _fit_target(y_train))
            return FitMetadata(requested, requested and model_key in {"gradient_boosting", "shallow_mlp_regressor"}, "native_or_not_requested")

        validation_fraction = float(early_config.get("validation_fraction", 0.2))
        stopping_rounds = int(early_config.get("stopping_rounds", 20))
        if len(X_train) < 8:
            estimator.fit(X_train, _fit_target(y_train))
            return FitMetadata(requested, False, "skipped_too_few_training_samples")

        X_fit, X_stop, y_fit, y_stop = train_test_split(
            X_train,
            y_train,
            test_size=validation_fraction,
            random_state=seed or self.config.randomness.get("default_seed", 42),
        )
        try:
            if model_key == "xgboost":
                estimator.fit(
                    X_fit,
                    _fit_target(y_fit),
                    eval_set=[(X_stop, _fit_target(y_stop))],
                    early_stopping_rounds=stopping_rounds,
                    verbose=False,
                )
            elif model_key == "lightgbm":
                from lightgbm import early_stopping

                estimator.fit(
                    X_fit,
                    _fit_target(y_fit),
                    eval_set=[(X_stop, _fit_target(y_stop))],
                    callbacks=[early_stopping(stopping_rounds, verbose=False)],
                )
            return FitMetadata(requested, True, "applied_inner_fold_early_stopping")
        except TypeError:
            estimator.fit(X_train, _fit_target(y_train))
            return FitMetadata(requested, False, "fallback_fit_without_early_stopping")

    def _folds_for_strategy(self, strategy: str, n_samples: int) -> tuple[FoldSplit, ...]:
        if strategy == "primary":
            return self.validation_factory.choose_primary(n_samples)
        if strategy == "repeated_kfold":
            return self.validation_factory.repeated_kfold(n_samples)
        if strategy == "loocv":
            return self.validation_factory.loocv(n_samples)
        if strategy == "nested_cv":
            return tuple(nested.outer_fold for nested in self.validation_factory.nested_cv(n_samples))
        if strategy == "bootstrap_oob":
            return self.validation_factory.bootstrap(n_samples)
        raise ValueError(f"Unsupported overfitting validation strategy: {strategy}")

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
        if _target_width(y_train) > 1 and model_key not in native_multi_output:
            return MultiOutputRegressor(estimator)
        return estimator

    def _run_root(self, dataset_key: str, target_name: str) -> Path:
        configured_root = Path(self.config.output["root_dir"])
        root = configured_root if configured_root.is_absolute() else project_root() / configured_root
        return ensure_directory(root / sanitize_filename(dataset_key) / sanitize_filename(target_name))


def _select_rows(values: pd.Series | pd.DataFrame, indices: np.ndarray) -> pd.Series | pd.DataFrame:
    return values.iloc[indices].copy()


def _target_width(values: pd.Series | pd.DataFrame) -> int:
    if isinstance(values, pd.DataFrame):
        return int(values.shape[1])
    return 1


def _fit_target(values: pd.Series | pd.DataFrame) -> np.ndarray:
    if isinstance(values, pd.Series):
        return values.to_numpy()
    if values.shape[1] == 1:
        return values.iloc[:, 0].to_numpy()
    return values.to_numpy()


def _is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
