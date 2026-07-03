"""End-to-end explainability report orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.data import DataLoader
from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename
from friction_surrogate_xai.xai.config import XAIConfig, load_xai_config
from friction_surrogate_xai.xai.importance import ImportanceAnalyzer, ImportanceArtifacts
from friction_surrogate_xai.xai.interpretation import (
    ScientificInterpretation,
    ScientificInterpreter,
)
from friction_surrogate_xai.xai.lime_analysis import LIMEAnalyzer, LIMEArtifacts
from friction_surrogate_xai.xai.mlflow_logging import XAIMLflowLogger
from friction_surrogate_xai.xai.preparation import PreparedXAIModel, XAIModelPreparer
from friction_surrogate_xai.xai.reports import XAIReportPaths, XAIReportWriter
from friction_surrogate_xai.xai.shap_analysis import SHAPAnalyzer, SHAPArtifacts


@dataclass(frozen=True)
class XAIArtifacts:
    """All artifacts produced by one explainability run."""

    root_dir: Path
    prepared: PreparedXAIModel
    shap: SHAPArtifacts
    importance: ImportanceArtifacts
    lime: LIMEArtifacts
    interpretation: ScientificInterpretation
    reports: XAIReportPaths
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]
    mlflow_metrics: dict[str, float]


class XAIReportGenerator:
    """Generate publication-quality XAI artifacts for one fitted surrogate model."""

    def __init__(
        self,
        config: XAIConfig | None = None,
        data_loader: DataLoader | None = None,
        preparer: XAIModelPreparer | None = None,
    ) -> None:
        self.config = config or load_xai_config()
        self.data_loader = data_loader or DataLoader()
        self.preparer = preparer or XAIModelPreparer()
        self.mlflow_logger = XAIMLflowLogger(self.config.mlflow)

    def generate(
        self,
        *,
        dataset_key: str,
        target_name: str,
        model_key: str,
        X: pd.DataFrame | None = None,
        y: pd.Series | None = None,
        dataframe: pd.DataFrame | None = None,
        params_override: dict[str, Any] | None = None,
        random_state: int | None = None,
        log_to_mlflow: bool | None = None,
    ) -> XAIArtifacts:
        """Generate SHAP, LIME, importance, and interpretation artifacts."""
        X, y = self._resolve_data(
            dataset_key=dataset_key,
            target_name=target_name,
            X=X,
            y=y,
            dataframe=dataframe,
        )
        seed = int(random_state if random_state is not None else self.config.data.get("random_state", 42))
        prepared = self.preparer.prepare(
            dataset_key=dataset_key,
            model_key=model_key,
            X=X,
            y=y,
            target_name=target_name,
            params_override=params_override,
            random_state=seed,
        )

        root_dir = self._run_root(dataset_key, target_name, model_key)
        tables_dir = ensure_directory(root_dir / self.config.output.get("tables_dir_name", "tables"))
        figures_dir = ensure_directory(root_dir / self.config.output.get("figures_dir_name", "figures"))
        markdown_dir = ensure_directory(root_dir / self.config.output.get("markdown_dir_name", "markdown"))

        shap_config = dict(self.config.shap)
        shap_config.update(self.config.data)
        shap_artifacts = SHAPAnalyzer(
            shap_config=shap_config,
            plot_config=self.config.plots,
        ).analyze(prepared=prepared, tables_dir=tables_dir, figures_dir=figures_dir)

        importance_artifacts = ImportanceAnalyzer(
            permutation_config=self.config.permutation_importance,
            tree_importance_config=self.config.tree_importance,
            tree_interpreter_config=self.config.tree_interpreter,
            plot_config=self.config.plots,
        ).analyze(
            prepared=prepared,
            shap_values=shap_artifacts.shap_values if shap_artifacts.shap_values.size else None,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
        )

        lime_artifacts = LIMEAnalyzer(
            lime_config=self.config.lime,
            plot_config=self.config.plots,
        ).analyze(prepared=prepared, tables_dir=tables_dir, figures_dir=figures_dir)

        interpretation = ScientificInterpreter(self.config.interpretation).interpret(
            shap_global=shap_artifacts.global_importance,
            shap_effects=shap_artifacts.effect_summary,
            shap_interactions=shap_artifacts.interaction_summary,
            permutation=importance_artifacts.permutation_importance,
            tree_importance=importance_artifacts.tree_importance,
            target_name=target_name,
        )
        report_paths = XAIReportWriter(self.config.reports).write(
            dataset_key=dataset_key,
            model_key=model_key,
            target_name=target_name,
            interpretation=interpretation,
            figures=shap_artifacts.figure_paths
            + importance_artifacts.figure_paths
            + lime_artifacts.figure_paths,
            tables_dir=tables_dir,
            markdown_dir=markdown_dir,
        )

        table_paths = (
            shap_artifacts.table_paths
            + importance_artifacts.table_paths
            + lime_artifacts.table_paths
            + report_paths.table_paths
        )
        figure_paths = (
            shap_artifacts.figure_paths
            + importance_artifacts.figure_paths
            + lime_artifacts.figure_paths
        )
        markdown_paths = report_paths.markdown_paths
        metrics = self._mlflow_metrics(
            prepared=prepared,
            shap_artifacts=shap_artifacts,
            importance_artifacts=importance_artifacts,
            table_paths=table_paths,
            figure_paths=figure_paths,
            markdown_paths=markdown_paths,
        )

        should_log = self.mlflow_logger.enabled() if log_to_mlflow is None else log_to_mlflow
        if should_log:
            self.mlflow_logger.log_run(
                dataset_key=dataset_key,
                model_key=model_key,
                target_name=target_name,
                artifact_dir=root_dir,
                metrics=metrics,
            )

        return XAIArtifacts(
            root_dir=root_dir,
            prepared=prepared,
            shap=shap_artifacts,
            importance=importance_artifacts,
            lime=lime_artifacts,
            interpretation=interpretation,
            reports=report_paths,
            table_paths=table_paths,
            figure_paths=figure_paths,
            markdown_paths=markdown_paths,
            mlflow_metrics=metrics,
        )

    def _resolve_data(
        self,
        *,
        dataset_key: str,
        target_name: str,
        X: pd.DataFrame | None,
        y: pd.Series | None,
        dataframe: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if X is not None and y is not None:
            return X.copy(), y.copy()

        if dataframe is None:
            loaded = self.data_loader.load(dataset_key)
            dataframe = loaded.dataframe
            feature_columns = loaded.report.metadata.feature_columns
        else:
            feature_columns = self.data_loader.catalog.schema.feature_columns

        if target_name not in dataframe.columns:
            raise KeyError(f"Target column '{target_name}' is not present in dataset '{dataset_key}'.")
        missing_features = [column for column in feature_columns if column not in dataframe.columns]
        if missing_features:
            raise KeyError(
                f"Dataset '{dataset_key}' is missing required feature columns: {missing_features}"
            )
        return (
            dataframe.loc[:, list(feature_columns)].copy(),
            dataframe.loc[:, target_name].copy(),
        )

    def _run_root(self, dataset_key: str, target_name: str, model_key: str) -> Path:
        return ensure_directory(
            self.config.output_root
            / dataset_key
            / sanitize_filename(target_name)
            / sanitize_filename(model_key)
        )

    @staticmethod
    def _mlflow_metrics(
        *,
        prepared: PreparedXAIModel,
        shap_artifacts: SHAPArtifacts,
        importance_artifacts: ImportanceArtifacts,
        table_paths: tuple[Path, ...],
        figure_paths: tuple[Path, ...],
        markdown_paths: tuple[Path, ...],
    ) -> dict[str, float]:
        metrics: dict[str, float] = {
            "n_samples": float(len(prepared.raw_features)),
            "n_raw_features": float(prepared.raw_features.shape[1]),
            "n_processed_features": float(prepared.processed_features.shape[1]),
            "n_tables": float(len(table_paths)),
            "n_figures": float(len(figure_paths)),
            "n_markdown_reports": float(len(markdown_paths)),
        }
        if not shap_artifacts.global_importance.empty:
            metrics["top_mean_abs_shap"] = float(
                shap_artifacts.global_importance["mean_abs_shap"].iloc[0]
            )
        if not importance_artifacts.permutation_importance.empty:
            values = importance_artifacts.permutation_importance.get("importance_mean")
            if values is not None and values.notna().any():
                metrics["top_permutation_importance"] = float(values.max())
        return metrics
