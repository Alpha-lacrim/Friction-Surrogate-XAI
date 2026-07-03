"""Three-stage hyperparameter optimization runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename
from friction_surrogate_xai.evaluation.arrays import infer_target_names
from friction_surrogate_xai.models import ModelFactory
from friction_surrogate_xai.optimization.config import OptimizationConfig, load_optimization_config
from friction_surrogate_xai.optimization.mlflow_logging import OptimizationMLflowLogger
from friction_surrogate_xai.optimization.plots import OptimizationPlotter
from friction_surrogate_xai.optimization.reports import (
    OptimizationReportPaths,
    OptimizationReportWriter,
)
from friction_surrogate_xai.optimization.scoring import OptimizationTrialEvaluator
from friction_surrogate_xai.optimization.spaces import SearchSpaceSampler
from friction_surrogate_xai.preprocessing import PreprocessingPipelineFactory


@dataclass(frozen=True)
class OptimizationArtifacts:
    """Generated artifacts for one optimization run."""

    dataset_key: str
    target_name: str
    root_dir: Path
    table_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]
    plot_paths: tuple[Path, ...]
    history: pd.DataFrame
    top_models: pd.DataFrame
    best_parameters: pd.DataFrame
    parameter_importance: pd.DataFrame

    @property
    def summary_path(self) -> Path | None:
        """Return the summary markdown path if generated."""
        for path in self.markdown_paths:
            if path.name == "optimization_summary.md":
                return path
        return None


class HyperparameterOptimizationRunner:
    """Run Random Search, top-3 selection, and Optuna optimization."""

    def __init__(
        self,
        config: OptimizationConfig | None = None,
        model_factory: ModelFactory | None = None,
        preprocessing_factory: PreprocessingPipelineFactory | None = None,
    ) -> None:
        self.config = config or load_optimization_config()
        self.model_factory = model_factory or ModelFactory()
        self.preprocessing_factory = preprocessing_factory or PreprocessingPipelineFactory()
        self.search_sampler = SearchSpaceSampler(self.config.search_spaces)
        self.evaluator = OptimizationTrialEvaluator(
            model_factory=self.model_factory,
            preprocessing_factory=self.preprocessing_factory,
            metrics=tuple(self.config.scoring.get("metrics", ("r2", "rmse", "nrmse", "mae"))),
            primary_metric=str(self.config.scoring.get("primary_metric", "r2")),
            nrmse_denominator=str(self.config.scoring.get("nrmse_denominator", "range")),
            higher_is_better=tuple(self.config.scoring.get("higher_is_better", ("r2",))),
        )
        self.mlflow_logger = OptimizationMLflowLogger(self.config)
        self.report_writer = OptimizationReportWriter(self.config.reports, self.config.output)
        self.plotter = OptimizationPlotter(self.config.plots)

    def run(
        self,
        *,
        dataset_key: str,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        target_name: str | None = None,
        model_keys: tuple[str, ...] | None = None,
        log_to_mlflow: bool | None = None,
    ) -> OptimizationArtifacts:
        """Run all optimization stages for one dataset/target."""
        selected_models = model_keys or self.model_factory.enabled_model_keys()
        target_names = infer_target_names(y, (target_name,) if target_name else None)
        resolved_target = target_name or ";".join(target_names)
        root_dir = self._run_root(dataset_key, resolved_target)
        mlflow_enabled = self.config.mlflow.get("enabled", True) if log_to_mlflow is None else log_to_mlflow

        stage1_results = self._run_stage1_random_search(
            dataset_key=dataset_key,
            target_name=resolved_target,
            X=X,
            y=y,
            model_keys=selected_models,
            log_to_mlflow=mlflow_enabled,
        )
        top_models = self._select_top_models(stage1_results)
        optuna_results, parameter_importance = self._run_stage3_optuna(
            dataset_key=dataset_key,
            target_name=resolved_target,
            X=X,
            y=y,
            top_models=tuple(top_models["model_key"]),
            log_to_mlflow=mlflow_enabled,
        )
        history = pd.concat([stage1_results, optuna_results], ignore_index=True)
        best_parameters = self._best_parameters(history)

        plots_dir = ensure_directory(root_dir / self.config.output.get("plots_dir_name", "plots"))
        plot_paths = self.plotter.generate(
            history=history,
            top_models=top_models,
            parameter_importance=parameter_importance,
            output_dir=plots_dir,
        )
        report_paths = self.report_writer.write(
            dataset_key=dataset_key,
            target_name=resolved_target,
            history=history,
            stage1_results=stage1_results,
            top_models=top_models,
            optuna_results=optuna_results,
            best_parameters=best_parameters,
            parameter_importance=parameter_importance,
            root_dir=root_dir,
        )
        if mlflow_enabled:
            self.mlflow_logger.log_artifacts(
                dataset_key=dataset_key,
                target_name=resolved_target,
                artifact_dir=root_dir,
                summary_metrics=self._summary_metrics(history, top_models),
            )

        return OptimizationArtifacts(
            dataset_key=dataset_key,
            target_name=resolved_target,
            root_dir=root_dir,
            table_paths=report_paths.table_paths,
            markdown_paths=report_paths.markdown_paths,
            plot_paths=plot_paths,
            history=history,
            top_models=top_models,
            best_parameters=best_parameters,
            parameter_importance=parameter_importance,
        )

    def _run_stage1_random_search(
        self,
        *,
        dataset_key: str,
        target_name: str,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        model_keys: tuple[str, ...],
        log_to_mlflow: bool,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        n_trials = int(self.config.stage1_random_search.get("n_trials_per_model", 20))
        cv_splits = int(self.config.stage1_random_search.get("cv_splits", 5))
        for model_index, model_key in enumerate(model_keys):
            model_trial_count = n_trials if self.search_sampler.has_space(model_key) else 1
            for trial_number in range(model_trial_count):
                seed = self._trial_seed(model_index=model_index, trial_number=trial_number)
                rng = np.random.default_rng(seed)
                params = self.search_sampler.random_sample(model_key, rng)
                score = self.evaluator.evaluate(
                    dataset_key=dataset_key,
                    model_key=model_key,
                    params=params,
                    X=X,
                    y=y,
                    cv_splits=cv_splits,
                    seed=seed,
                )
                row = self._trial_row(
                    dataset_key=dataset_key,
                    target_name=target_name,
                    stage="stage1_random_search",
                    model_key=model_key,
                    trial_number=trial_number,
                    seed=seed,
                    params=params,
                    metrics=score.metrics,
                )
                rows.append(row)
                if log_to_mlflow:
                    self.mlflow_logger.log_trial(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        stage="stage1_random_search",
                        model_key=model_key,
                        trial_number=trial_number,
                        params=params,
                        metrics=score.metrics,
                    )
        return pd.DataFrame(rows)

    def _run_stage3_optuna(
        self,
        *,
        dataset_key: str,
        target_name: str,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        top_models: tuple[str, ...],
        log_to_mlflow: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.stage3_optuna.get("enabled", True) or not top_models:
            return pd.DataFrame(), pd.DataFrame(columns=["model_key", "parameter", "importance"])

        import optuna
        from optuna.importance import get_param_importances

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        rows: list[dict[str, Any]] = []
        importance_rows: list[dict[str, Any]] = []
        n_trials = int(self.config.stage3_optuna.get("n_trials_per_model", 50))
        cv_splits = int(self.config.stage1_random_search.get("cv_splits", 5))
        timeout = self.config.stage3_optuna.get("timeout_seconds")
        direction = str(self.config.stage3_optuna.get("direction", self._direction()))

        for model_index, model_key in enumerate(top_models):
            seed = self._trial_seed(model_index=model_index, trial_number=10_000)
            sampler = optuna.samplers.TPESampler(seed=seed)
            study = optuna.create_study(direction=direction, sampler=sampler)

            def objective(trial: Any) -> float:
                params = self.search_sampler.suggest_optuna(model_key, trial)
                trial_seed = self._trial_seed(
                    model_index=model_index,
                    trial_number=20_000 + trial.number,
                )
                score = self.evaluator.evaluate(
                    dataset_key=dataset_key,
                    model_key=model_key,
                    params=params,
                    X=X,
                    y=y,
                    cv_splits=cv_splits,
                    seed=trial_seed,
                )
                row = self._trial_row(
                    dataset_key=dataset_key,
                    target_name=target_name,
                    stage="stage3_optuna",
                    model_key=model_key,
                    trial_number=trial.number,
                    seed=trial_seed,
                    params=params,
                    metrics=score.metrics,
                )
                rows.append(row)
                for key, value in row.items():
                    if key not in {"params_json"}:
                        trial.set_user_attr(key, value)
                if log_to_mlflow:
                    self.mlflow_logger.log_trial(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        stage="stage3_optuna",
                        model_key=model_key,
                        trial_number=trial.number,
                        params=params,
                        metrics=score.metrics,
                    )
                return self._objective_for_optuna(score.metrics["objective_value"])

            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=int(timeout) if timeout is not None else None,
            )
            try:
                importances = get_param_importances(study)
            except (ValueError, RuntimeError):
                importances = {}
            for parameter, importance in importances.items():
                importance_rows.append(
                    {
                        "model_key": model_key,
                        "parameter": parameter,
                        "importance": float(importance),
                    }
                )

        return pd.DataFrame(rows), pd.DataFrame(importance_rows)

    def _select_top_models(self, stage1_results: pd.DataFrame) -> pd.DataFrame:
        if stage1_results.empty:
            return pd.DataFrame(columns=["model_key", "best_stage1_objective", "stage1_trial_number"])
        ascending = not self._higher_is_better()
        sorted_results = stage1_results.sort_values("objective_value", ascending=ascending)
        best_by_model = sorted_results.groupby("model_key", as_index=False).first()
        top_n = int(self.config.stage2_selection.get("top_n_models", 3))
        top = best_by_model.head(top_n).copy()
        return top.rename(
            columns={
                "objective_value": "best_stage1_objective",
                "trial_number": "stage1_trial_number",
            }
        )[
            [
                "model_key",
                "best_stage1_objective",
                "stage1_trial_number",
                "params_json",
                "mean_validation_r2",
                "mean_validation_rmse",
                "mean_validation_nrmse",
                "mean_validation_mae",
            ]
        ]

    def _best_parameters(self, history: pd.DataFrame) -> pd.DataFrame:
        if history.empty:
            return pd.DataFrame()
        ascending = not self._higher_is_better()
        sorted_history = history.sort_values("objective_value", ascending=ascending)
        best = sorted_history.groupby("model_key", as_index=False).first()
        return best[
            [
                "model_key",
                "stage",
                "trial_number",
                "objective_value",
                "params_json",
                "mean_train_r2",
                "mean_validation_r2",
                "generalization_gap_r2",
                "std_validation_r2",
            ]
        ]

    def _trial_row(
        self,
        *,
        dataset_key: str,
        target_name: str,
        stage: str,
        model_key: str,
        trial_number: int,
        seed: int,
        params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "dataset_key": dataset_key,
            "target_name": target_name,
            "stage": stage,
            "model_key": model_key,
            "trial_number": int(trial_number),
            "seed": int(seed),
            "params_json": json.dumps(_serializable_params(params), sort_keys=True),
            **metrics,
        }

    def _trial_seed(self, *, model_index: int, trial_number: int) -> int:
        seeds = self.config.repeated_seeds
        base_seed = seeds[trial_number % len(seeds)]
        return int(base_seed + model_index * 10_000 + trial_number)

    def _direction(self) -> str:
        return "maximize" if self._higher_is_better() else "minimize"

    def _objective_for_optuna(self, value: Any) -> float:
        try:
            objective = float(value)
        except (TypeError, ValueError):
            objective = np.nan
        if np.isfinite(objective):
            return objective
        return -1.0e12 if self._higher_is_better() else 1.0e12

    def _higher_is_better(self) -> bool:
        primary = str(self.config.scoring.get("primary_metric", "r2"))
        return primary in set(self.config.scoring.get("higher_is_better", ("r2",)))

    @staticmethod
    def _summary_metrics(history: pd.DataFrame, top_models: pd.DataFrame) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "total_trials": float(len(history)),
            "stage1_trials": float((history["stage"] == "stage1_random_search").sum())
            if not history.empty
            else 0.0,
            "stage3_trials": float((history["stage"] == "stage3_optuna").sum())
            if not history.empty
            else 0.0,
            "top_model_count": float(len(top_models)),
        }
        if not top_models.empty:
            metrics["best_stage1_objective"] = float(top_models["best_stage1_objective"].iloc[0])
        return metrics

    def _run_root(self, dataset_key: str, target_name: str) -> Path:
        configured_root = Path(self.config.output["root_dir"])
        root = configured_root if configured_root.is_absolute() else project_root() / configured_root
        return ensure_directory(root / sanitize_filename(dataset_key) / sanitize_filename(target_name))


def _serializable_params(params: dict[str, Any]) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            serializable[key] = list(value)
        elif isinstance(value, np.generic):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable
