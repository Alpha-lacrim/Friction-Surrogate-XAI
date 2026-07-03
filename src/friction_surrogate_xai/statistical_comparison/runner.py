"""Statistical-comparison report orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.eda.utils import ensure_directory
from friction_surrogate_xai.statistical_comparison.config import (
    StatisticalComparisonConfig,
    load_statistical_comparison_config,
)
from friction_surrogate_xai.statistical_comparison.data import ScoreTableLoader
from friction_surrogate_xai.statistical_comparison.mlflow_logging import (
    StatisticalComparisonMLflowLogger,
)
from friction_surrogate_xai.statistical_comparison.plotting import (
    StatisticalComparisonPlotter,
)
from friction_surrogate_xai.statistical_comparison.reports import (
    StatisticalComparisonReportPaths,
    StatisticalComparisonReportWriter,
)
from friction_surrogate_xai.statistical_comparison.tests import StatisticalComparator


@dataclass(frozen=True)
class StatisticalComparisonArtifacts:
    """Generated statistical-comparison artifacts."""

    root_dir: Path
    score_table: pd.DataFrame
    wilcoxon: pd.DataFrame
    friedman: pd.DataFrame
    nemenyi: pd.DataFrame
    average_ranks: pd.DataFrame
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]


class StatisticalComparisonRunner:
    """Run configured statistical comparisons and write reports."""

    def __init__(
        self,
        config: StatisticalComparisonConfig | None = None,
        score_loader: ScoreTableLoader | None = None,
    ) -> None:
        self.config = config or load_statistical_comparison_config()
        self.score_loader = score_loader or ScoreTableLoader(self.config.inputs)
        self.comparator = StatisticalComparator(self.config.tests, self.config.alpha)
        self.plotter = StatisticalComparisonPlotter(self.config.plots)
        self.report_writer = StatisticalComparisonReportWriter(self.config.reports)
        self.mlflow_logger = StatisticalComparisonMLflowLogger(self.config.mlflow)

    def run(
        self,
        *,
        score_table: pd.DataFrame | None = None,
        input_paths: tuple[str | Path, ...] | None = None,
        log_to_mlflow: bool | None = None,
    ) -> StatisticalComparisonArtifacts:
        """Run all configured statistical comparisons."""
        scores = (
            score_table.copy()
            if score_table is not None
            else self.score_loader.load(input_paths)
        )
        scores = scores.loc[pd.to_numeric(scores.get("score"), errors="coerce").notna()].copy()
        root_dir = self._root_dir()
        tables_dir = ensure_directory(
            root_dir / self.config.output.get("tables_dir_name", "tables")
        )
        figures_dir = ensure_directory(
            root_dir / self.config.output.get("figures_dir_name", "figures")
        )
        markdown_dir = ensure_directory(
            root_dir / self.config.output.get("markdown_dir_name", "markdown")
        )

        results = [
            result
            for result in (
                self._compare_top_models(scores),
                self._compare_original_vs_discrete(scores),
                self._compare_single_vs_multi_output(scores),
            )
            if result is not None
        ]
        wilcoxon = pd.concat([result.wilcoxon for result in results], ignore_index=True)
        friedman = pd.concat([result.friedman for result in results], ignore_index=True)
        nemenyi = pd.concat([result.nemenyi for result in results], ignore_index=True)
        average_ranks = pd.concat(
            [result.average_ranks for result in results],
            ignore_index=True,
        )

        figure_paths = self.plotter.write(
            wilcoxon=wilcoxon,
            nemenyi=nemenyi,
            average_ranks=average_ranks,
            figures_dir=figures_dir,
        )
        report_paths = self.report_writer.write(
            score_table=scores,
            wilcoxon=wilcoxon,
            friedman=friedman,
            nemenyi=nemenyi,
            average_ranks=average_ranks,
            figure_paths=figure_paths,
            tables_dir=tables_dir,
            markdown_dir=markdown_dir,
        )

        should_log = self.mlflow_logger.enabled() if log_to_mlflow is None else log_to_mlflow
        if should_log:
            self.mlflow_logger.log_run(
                artifact_dir=root_dir,
                wilcoxon=wilcoxon,
                friedman=friedman,
                nemenyi=nemenyi,
                score_table=scores,
            )

        return StatisticalComparisonArtifacts(
            root_dir=root_dir,
            score_table=scores,
            wilcoxon=wilcoxon,
            friedman=friedman,
            nemenyi=nemenyi,
            average_ranks=average_ranks,
            table_paths=report_paths.table_paths,
            figure_paths=figure_paths,
            markdown_paths=report_paths.markdown_paths,
        )

    def _compare_top_models(self, scores: pd.DataFrame):
        config = dict(self.config.comparisons.get("top_models", {}))
        if not config.get("enabled", True):
            return None
        group_column = str(config.get("group_column", "model_key"))
        context_columns = tuple(config.get("context_columns", ()))
        filtered = scores.loc[scores["model_key"].notna()].copy()
        return self.comparator.compare(
            scores=filtered,
            comparison_name="top_models",
            group_column=group_column,
            context_columns=context_columns,
        )

    def _compare_original_vs_discrete(self, scores: pd.DataFrame):
        config = dict(self.config.comparisons.get("original_vs_discrete", {}))
        if not config.get("enabled", True):
            return None
        group_column = str(config.get("group_column", "variant"))
        group_values = tuple(config.get("group_values", ("original", "discrete")))
        filtered = scores.loc[scores[group_column].isin(group_values)].copy()
        return self.comparator.compare(
            scores=filtered,
            comparison_name="original_vs_discrete",
            group_column=group_column,
            group_values=group_values,
            context_columns=tuple(config.get("context_columns", ())),
        )

    def _compare_single_vs_multi_output(self, scores: pd.DataFrame):
        config = dict(self.config.comparisons.get("single_vs_multi_output", {}))
        if not config.get("enabled", True):
            return None
        group_column = str(config.get("group_column", "output_mode"))
        group_values = tuple(config.get("group_values", ("single_output", "multi_output")))
        filtered = scores.loc[scores[group_column].isin(group_values)].copy()
        return self.comparator.compare(
            scores=filtered,
            comparison_name="single_vs_multi_output",
            group_column=group_column,
            group_values=group_values,
            context_columns=tuple(config.get("context_columns", ())),
        )

    def _root_dir(self) -> Path:
        configured = self.config.output_root
        root = configured if configured.is_absolute() else project_root() / configured
        return ensure_directory(root)
