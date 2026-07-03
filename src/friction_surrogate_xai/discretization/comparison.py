"""Original-vs-discrete model comparison for selected top models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.discretization.config import DiscretizationConfig, load_discretization_config
from friction_surrogate_xai.discretization.mlflow_logging import DiscretizationMLflowLogger
from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename, write_csv
from friction_surrogate_xai.evaluation.reports import EvaluationReportWriter
from friction_surrogate_xai.models import ModelFactory
from friction_surrogate_xai.optimization.scoring import OptimizationTrialEvaluator
from friction_surrogate_xai.preprocessing import PreprocessingPipelineFactory


@dataclass(frozen=True)
class SelectedModel:
    """Top-model selection result."""

    model_key: str
    params: dict[str, Any]
    source: str
    rank: int


@dataclass(frozen=True)
class DiscreteComparisonArtifacts:
    """Generated original-vs-discrete comparison artifacts."""

    dataset_key: str
    target_name: str
    root_dir: Path
    score_path: Path
    summary_path: Path
    selected_models_path: Path
    markdown_path: Path
    variant_scores: pd.DataFrame
    comparison_summary: pd.DataFrame
    selected_models: pd.DataFrame


class TopModelSelector:
    """Resolve Top 3 models from optimization artifacts, with configured fallback."""

    def __init__(self, config: DiscretizationConfig | None = None) -> None:
        self.config = config or load_discretization_config()

    def select(
        self,
        *,
        dataset_key: str,
        target_name: str,
        explicit_model_keys: tuple[str, ...] | None = None,
    ) -> tuple[SelectedModel, ...]:
        """Return selected Top N models and their best parameters when available."""
        if explicit_model_keys:
            return tuple(
                SelectedModel(model_key=model_key, params={}, source="explicit", rank=index + 1)
                for index, model_key in enumerate(explicit_model_keys)
            )

        from_artifacts = self._from_optimization_artifacts(dataset_key, target_name)
        if from_artifacts:
            return from_artifacts

        fallback = tuple(self.config.comparison.get("fallback_top_models", ()))[: self.top_n]
        return tuple(
            SelectedModel(model_key=model_key, params={}, source="configured_fallback", rank=index + 1)
            for index, model_key in enumerate(fallback)
        )

    @property
    def top_n(self) -> int:
        """Return configured number of top models."""
        return int(self.config.comparison.get("top_n_models", 3))

    def _from_optimization_artifacts(
        self,
        dataset_key: str,
        target_name: str,
    ) -> tuple[SelectedModel, ...]:
        root = self._optimization_root() / sanitize_filename(dataset_key) / sanitize_filename(target_name)
        top_path = root / "tables" / "stage2_top_models.csv"
        best_path = root / "tables" / "best_parameters.csv"
        if not top_path.exists():
            return ()

        top_models = pd.read_csv(top_path).head(self.top_n)
        best_params = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()
        selected: list[SelectedModel] = []
        for rank, (_, row) in enumerate(top_models.iterrows(), start=1):
            model_key = str(row["model_key"])
            params_json = row.get("params_json", "{}")
            if (
                self.config.comparison.get("use_best_parameters_when_available", True)
                and not best_params.empty
                and model_key in set(best_params["model_key"])
            ):
                params_json = best_params.loc[
                    best_params["model_key"] == model_key,
                    "params_json",
                ].iloc[0]
            selected.append(
                SelectedModel(
                    model_key=model_key,
                    params=_parse_params(params_json),
                    source="optimization_artifacts",
                    rank=rank,
                )
            )
        return tuple(selected)

    def _optimization_root(self) -> Path:
        configured = Path(self.config.comparison.get("optimization_root_dir", "reports/optimization"))
        return configured if configured.is_absolute() else project_root() / configured


class OriginalVsDiscreteComparator:
    """Evaluate Top 3 models on original and discrete inputs with identical CV."""

    def __init__(
        self,
        config: DiscretizationConfig | None = None,
        model_factory: ModelFactory | None = None,
        preprocessing_factory: PreprocessingPipelineFactory | None = None,
    ) -> None:
        self.config = config or load_discretization_config()
        self.model_factory = model_factory or ModelFactory()
        self.preprocessing_factory = preprocessing_factory or PreprocessingPipelineFactory()
        self.selector = TopModelSelector(self.config)
        self.markdown = EvaluationReportWriter(self.config.reports)
        self.mlflow_logger = DiscretizationMLflowLogger(self.config.mlflow)
        self.evaluator = OptimizationTrialEvaluator(
            model_factory=self.model_factory,
            preprocessing_factory=self.preprocessing_factory,
            metrics=("r2", "rmse", "nrmse", "mae"),
            primary_metric="r2",
            nrmse_denominator="range",
            higher_is_better=("r2",),
        )

    def compare(
        self,
        *,
        dataset_key: str,
        original_dataframe: pd.DataFrame,
        discrete_dataframe: pd.DataFrame,
        feature_columns: tuple[str, ...],
        target_name: str,
        explicit_model_keys: tuple[str, ...] | None = None,
        log_to_mlflow: bool | None = None,
    ) -> DiscreteComparisonArtifacts:
        """Compare original and discrete inputs with identical folds and selected models."""
        selected_models = self.selector.select(
            dataset_key=dataset_key,
            target_name=target_name,
            explicit_model_keys=explicit_model_keys,
        )
        variant_scores = self._evaluate_variants(
            dataset_key=dataset_key,
            original_dataframe=original_dataframe,
            discrete_dataframe=discrete_dataframe,
            feature_columns=feature_columns,
            target_name=target_name,
            selected_models=selected_models,
        )
        comparison_summary = self._comparison_summary(variant_scores)
        selected_table = pd.DataFrame(
            [
                {
                    "rank": model.rank,
                    "model_key": model.model_key,
                    "selection_source": model.source,
                    "params_json": json.dumps(_serializable_params(model.params), sort_keys=True),
                }
                for model in selected_models
            ]
        )
        root_dir = self._comparison_root(dataset_key, target_name)
        tables_dir = ensure_directory(root_dir / self.config.output.get("tables_dir_name", "tables"))
        markdown_dir = ensure_directory(
            root_dir / self.config.output.get("markdown_dir_name", "markdown")
        )
        score_path = write_csv(variant_scores, tables_dir / "variant_scores.csv")
        summary_path = write_csv(comparison_summary, tables_dir / "original_vs_discrete_summary.csv")
        selected_path = write_csv(selected_table, tables_dir / "selected_top_models.csv")
        markdown_path = self._write_markdown(
            dataset_key=dataset_key,
            target_name=target_name,
            selected_models=selected_table,
            comparison_summary=comparison_summary,
            output_path=markdown_dir / "original_vs_discrete_summary.md",
        )

        should_log = self.config.mlflow.get("enabled", True) if log_to_mlflow is None else log_to_mlflow
        if should_log:
            self.mlflow_logger.log_comparison(
                dataset_key=dataset_key,
                target_name=target_name,
                artifact_dir=root_dir,
                comparison_summary=comparison_summary,
            )

        return DiscreteComparisonArtifacts(
            dataset_key=dataset_key,
            target_name=target_name,
            root_dir=root_dir,
            score_path=score_path,
            summary_path=summary_path,
            selected_models_path=selected_path,
            markdown_path=markdown_path,
            variant_scores=variant_scores,
            comparison_summary=comparison_summary,
            selected_models=selected_table,
        )

    def _evaluate_variants(
        self,
        *,
        dataset_key: str,
        original_dataframe: pd.DataFrame,
        discrete_dataframe: pd.DataFrame,
        feature_columns: tuple[str, ...],
        target_name: str,
        selected_models: tuple[SelectedModel, ...],
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        seeds = tuple(int(seed) for seed in self.config.comparison.get("repeated_seeds", (42,)))
        cv_splits = int(self.config.comparison.get("cv_splits", 5))
        variants = {
            "original": original_dataframe,
            "discrete": discrete_dataframe,
        }
        for variant_name, dataframe in variants.items():
            X = dataframe.loc[:, feature_columns].copy()
            y = dataframe.loc[:, target_name].copy()
            for model in selected_models:
                for seed in seeds:
                    score = self.evaluator.evaluate(
                        dataset_key=dataset_key,
                        model_key=model.model_key,
                        params=model.params,
                        X=X,
                        y=y,
                        cv_splits=cv_splits,
                        seed=seed,
                    )
                    rows.append(
                        {
                            "dataset_key": dataset_key,
                            "target_name": target_name,
                            "variant": variant_name,
                            "model_key": model.model_key,
                            "model_rank": model.rank,
                            "model_selection_source": model.source,
                            "seed": seed,
                            "params_json": json.dumps(
                                _serializable_params(model.params),
                                sort_keys=True,
                            ),
                            **score.metrics,
                        }
                    )
        return pd.DataFrame(rows)

    def _comparison_summary(self, scores: pd.DataFrame) -> pd.DataFrame:
        if scores.empty:
            return pd.DataFrame()
        metric_columns = [
            column
            for column in scores.columns
            if column.startswith("mean_validation_")
            or column.startswith("std_validation_")
            or column.startswith("generalization_gap_")
            or column == "objective_value"
        ]
        grouped = (
            scores.groupby(["dataset_key", "target_name", "model_key", "variant"])[metric_columns]
            .mean(numeric_only=True)
            .reset_index()
        )
        original = grouped.loc[grouped["variant"] == "original"].copy()
        discrete = grouped.loc[grouped["variant"] == "discrete"].copy()
        merged = original.merge(
            discrete,
            on=["dataset_key", "target_name", "model_key"],
            suffixes=("_original", "_discrete"),
        )
        merged["delta_objective_discrete_minus_original"] = (
            merged["objective_value_discrete"] - merged["objective_value_original"]
        )
        merged["delta_r2_discrete_minus_original"] = (
            merged["mean_validation_r2_discrete"] - merged["mean_validation_r2_original"]
        )
        merged["delta_rmse_discrete_minus_original"] = (
            merged["mean_validation_rmse_discrete"] - merged["mean_validation_rmse_original"]
        )
        return merged

    def _write_markdown(
        self,
        *,
        dataset_key: str,
        target_name: str,
        selected_models: pd.DataFrame,
        comparison_summary: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        lines = [
            f"# Original vs Discrete Comparison: {dataset_key} / {target_name}",
            "",
            "## Evaluation Policy",
            "",
            "Original and discrete inputs are evaluated with identical model keys, hyperparameters, CV splits, seeds, metrics, and fold-local preprocessing.",
            "",
            "## Selected Top Models",
            "",
            self.markdown._markdown_table(selected_models),
            "",
            "## Comparison Summary",
            "",
            self.markdown._markdown_table(comparison_summary),
            "",
        ]
        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _comparison_root(self, dataset_key: str, target_name: str) -> Path:
        configured = Path(self.config.output["report_root_dir"])
        root = configured if configured.is_absolute() else project_root() / configured
        return ensure_directory(
            root / "comparisons" / sanitize_filename(dataset_key) / sanitize_filename(target_name)
        )


def _parse_params(params_json: Any) -> dict[str, Any]:
    if not isinstance(params_json, str) or not params_json:
        return {}
    try:
        parsed = json.loads(params_json)
    except json.JSONDecodeError:
        return {}
    return {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in parsed.items()
    }


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
