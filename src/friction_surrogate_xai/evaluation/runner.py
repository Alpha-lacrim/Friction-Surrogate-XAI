"""Reusable evaluation report orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename
from friction_surrogate_xai.evaluation.arrays import validate_prediction_arrays
from friction_surrogate_xai.evaluation.config import EvaluationConfig, load_evaluation_config
from friction_surrogate_xai.evaluation.metrics import RegressionMetricCalculator
from friction_surrogate_xai.evaluation.mlflow_logging import EvaluationMLflowLogger
from friction_surrogate_xai.evaluation.plots import EvaluationPlotter
from friction_surrogate_xai.evaluation.reports import EvaluationReportPaths, EvaluationReportWriter


@dataclass(frozen=True)
class EvaluationArtifacts:
    """Artifacts generated for one evaluation run."""

    dataset_key: str
    model_name: str
    root_dir: Path
    table_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]
    plot_paths: tuple[Path, ...]
    metrics: pd.DataFrame
    train_test_gap: pd.DataFrame
    cv_summary: pd.DataFrame

    @property
    def summary_path(self) -> Path | None:
        """Return the main markdown summary path if it was generated."""
        for path in self.markdown_paths:
            if path.name == "summary.md":
                return path
        return None


class EvaluationReportGenerator:
    """Generate regression evaluation plots, tables, reports, and MLflow logs."""

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or load_evaluation_config()
        gap_config = dict(self.config.metrics.get("train_test_gap", {}))
        stability_config = dict(self.config.metrics.get("fold_stability", {}))
        self.metric_calculator = RegressionMetricCalculator(
            metrics=tuple(self.config.metrics.get("regression", ("r2", "rmse", "nrmse", "mae"))),
            nrmse_denominator=str(self.config.metrics.get("nrmse_denominator", "range")),
            aggregate_targets=bool(self.config.metrics.get("aggregate_targets", True)),
            confidence_level=float(self.config.metrics.get("confidence_level", 0.95)),
            relative_gap_epsilon=float(gap_config.get("relative_denominator_epsilon", 1.0e-12)),
            stability_index_epsilon=float(stability_config.get("stability_index_epsilon", 1.0e-12)),
        )
        self.plotter = EvaluationPlotter(self.config.plots)
        self.report_writer = EvaluationReportWriter(self.config.reports)
        self.mlflow_logger = EvaluationMLflowLogger(self.config.mlflow)

    def generate(
        self,
        *,
        dataset_key: str,
        model_name: str,
        y_test_true: Any,
        y_test_pred: Any,
        target_names: Sequence[str] | None = None,
        y_train_true: Any | None = None,
        y_train_pred: Any | None = None,
        fold_metrics: pd.DataFrame | None = None,
        estimator: Any | None = None,
        X: Any | None = None,
        y: Any | None = None,
        validation_curve_params: dict[str, Any] | None = None,
        log_to_mlflow: bool | None = None,
    ) -> EvaluationArtifacts:
        """Generate complete evaluation artifacts from predictions and optional curve inputs."""
        self._validate_optional_train_inputs(y_train_true, y_train_pred)
        _, _, resolved_target_names = validate_prediction_arrays(
            y_test_true,
            y_test_pred,
            target_names=target_names,
        )

        run_root = self._run_root(dataset_key=dataset_key, model_name=model_name)
        plots_dir = ensure_directory(run_root / self.config.output.get("plots_dir_name", "plots"))
        tables_dir = ensure_directory(run_root / self.config.output.get("tables_dir_name", "tables"))
        markdown_dir = ensure_directory(
            run_root / self.config.output.get("markdown_dir_name", "markdown")
        )

        test_metrics = self.metric_calculator.evaluate_predictions(
            y_test_true,
            y_test_pred,
            target_names=resolved_target_names,
            dataset_key=dataset_key,
            model_name=model_name,
            split="test",
        )
        metric_tables = [test_metrics]
        train_test_gap = pd.DataFrame()

        if y_train_true is not None and y_train_pred is not None:
            train_metrics = self.metric_calculator.evaluate_predictions(
                y_train_true,
                y_train_pred,
                target_names=resolved_target_names,
                dataset_key=dataset_key,
                model_name=model_name,
                split="train",
            )
            metric_tables.insert(0, train_metrics)
            if self.config.metrics.get("train_test_gap", {}).get("enabled", True):
                train_test_gap = self.metric_calculator.train_test_gap(
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                )

        metrics = pd.concat(metric_tables, ignore_index=True)
        metrics_long = self.metric_calculator.to_long_format(metrics)
        cv_summary = (
            self.metric_calculator.summarize_cross_validation(fold_metrics)
            if fold_metrics is not None
            and self.config.metrics.get("fold_stability", {}).get("enabled", True)
            else pd.DataFrame()
        )

        plot_paths = self._generate_plots(
            dataset_key=dataset_key,
            model_name=model_name,
            plots_dir=plots_dir,
            y_test_true=y_test_true,
            y_test_pred=y_test_pred,
            y_train_true=y_train_true,
            y_train_pred=y_train_pred,
            target_names=resolved_target_names,
            estimator=estimator,
            X=X,
            y=y,
            validation_curve_params=validation_curve_params,
        )

        report_paths = self.report_writer.write(
            dataset_key=dataset_key,
            model_name=model_name,
            metrics=metrics,
            metrics_long=metrics_long,
            train_test_gap=train_test_gap,
            cv_summary=cv_summary,
            plot_paths=plot_paths,
            tables_dir=tables_dir,
            markdown_dir=markdown_dir,
        )

        should_log = self.config.mlflow.get("enabled", True) if log_to_mlflow is None else log_to_mlflow
        if should_log:
            self.mlflow_logger.log_evaluation(
                dataset_key=dataset_key,
                model_name=model_name,
                artifact_dir=run_root,
                params=self._mlflow_params(
                    dataset_key=dataset_key,
                    model_name=model_name,
                    target_names=resolved_target_names,
                    test_metrics=test_metrics,
                    has_train_metrics=y_train_true is not None,
                    has_fold_metrics=fold_metrics is not None,
                    has_learning_curve=estimator is not None and X is not None and y is not None,
                    has_validation_curve=validation_curve_params is not None,
                ),
                metrics=metrics,
                train_test_gap=train_test_gap,
                cv_summary=cv_summary,
            )

        return self._artifacts(
            dataset_key=dataset_key,
            model_name=model_name,
            run_root=run_root,
            report_paths=report_paths,
            plot_paths=plot_paths,
            metrics=metrics,
            train_test_gap=train_test_gap,
            cv_summary=cv_summary,
        )

    def _generate_plots(
        self,
        *,
        dataset_key: str,
        model_name: str,
        plots_dir: Path,
        y_test_true: Any,
        y_test_pred: Any,
        y_train_true: Any | None,
        y_train_pred: Any | None,
        target_names: Sequence[str],
        estimator: Any | None,
        X: Any | None,
        y: Any | None,
        validation_curve_params: dict[str, Any] | None,
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        paths.extend(
            self.plotter.prediction_vs_actual(
                y_true=y_test_true,
                y_pred=y_test_pred,
                output_dir=plots_dir,
                dataset_label=dataset_key,
                model_name=model_name,
                split="test",
                target_names=target_names,
            )
        )
        paths.extend(
            self.plotter.residuals(
                y_true=y_test_true,
                y_pred=y_test_pred,
                output_dir=plots_dir,
                dataset_label=dataset_key,
                model_name=model_name,
                split="test",
                target_names=target_names,
            )
        )

        if y_train_true is not None and y_train_pred is not None:
            paths.extend(
                self.plotter.prediction_vs_actual(
                    y_true=y_train_true,
                    y_pred=y_train_pred,
                    output_dir=plots_dir,
                    dataset_label=dataset_key,
                    model_name=model_name,
                    split="train",
                    target_names=target_names,
                )
            )
            paths.extend(
                self.plotter.residuals(
                    y_true=y_train_true,
                    y_pred=y_train_pred,
                    output_dir=plots_dir,
                    dataset_label=dataset_key,
                    model_name=model_name,
                    split="train",
                    target_names=target_names,
                )
            )

        if estimator is not None and X is not None and y is not None:
            target_label = target_names[0] if len(target_names) == 1 else "multi_output"
            learning_curve_path = self.plotter.learning_curve(
                estimator=estimator,
                X=X,
                y=y,
                output_dir=plots_dir,
                dataset_label=dataset_key,
                model_name=model_name,
                curve_config=self.config.learning_curve,
                target_name=target_label,
            )
            if learning_curve_path is not None:
                paths.append(learning_curve_path)

            validation_curve_params = validation_curve_params or {}
            validation_curve_path = self.plotter.validation_curve(
                estimator=estimator,
                X=X,
                y=y,
                output_dir=plots_dir,
                dataset_label=dataset_key,
                model_name=model_name,
                curve_config=self.config.validation_curve,
                param_name=validation_curve_params.get("param_name"),
                param_range=validation_curve_params.get("param_range"),
                target_name=target_label,
            )
            if validation_curve_path is not None:
                paths.append(validation_curve_path)

        return tuple(paths)

    def _run_root(self, *, dataset_key: str, model_name: str) -> Path:
        configured_root = Path(self.config.output["root_dir"])
        root = configured_root if configured_root.is_absolute() else project_root() / configured_root
        return ensure_directory(root / sanitize_filename(dataset_key) / sanitize_filename(model_name))

    @staticmethod
    def _validate_optional_train_inputs(y_train_true: Any | None, y_train_pred: Any | None) -> None:
        if (y_train_true is None) != (y_train_pred is None):
            raise ValueError("y_train_true and y_train_pred must be provided together.")

    @staticmethod
    def _mlflow_params(
        *,
        dataset_key: str,
        model_name: str,
        target_names: Sequence[str],
        test_metrics: pd.DataFrame,
        has_train_metrics: bool,
        has_fold_metrics: bool,
        has_learning_curve: bool,
        has_validation_curve: bool,
    ) -> dict[str, Any]:
        sample_count = int(test_metrics["sample_count"].iloc[0]) if not test_metrics.empty else 0
        return {
            "dataset_key": dataset_key,
            "model_name": model_name,
            "target_count": len(target_names),
            "target_names": ";".join(target_names),
            "test_sample_count": sample_count,
            "has_train_metrics": has_train_metrics,
            "has_fold_metrics": has_fold_metrics,
            "has_learning_curve": has_learning_curve,
            "has_validation_curve": has_validation_curve,
            "evaluation_policy": "report_do_not_modify_predictions_or_data",
        }

    @staticmethod
    def _artifacts(
        *,
        dataset_key: str,
        model_name: str,
        run_root: Path,
        report_paths: EvaluationReportPaths,
        plot_paths: tuple[Path, ...],
        metrics: pd.DataFrame,
        train_test_gap: pd.DataFrame,
        cv_summary: pd.DataFrame,
    ) -> EvaluationArtifacts:
        return EvaluationArtifacts(
            dataset_key=dataset_key,
            model_name=model_name,
            root_dir=run_root,
            table_paths=report_paths.table_paths,
            markdown_paths=report_paths.markdown_paths,
            plot_paths=plot_paths,
            metrics=metrics,
            train_test_gap=train_test_gap,
            cv_summary=cv_summary,
        )
