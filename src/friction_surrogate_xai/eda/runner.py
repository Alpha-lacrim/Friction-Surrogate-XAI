"""Automated EDA report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.config.loader import project_root
from friction_surrogate_xai.data import DataLoader, LoadedDataset
from friction_surrogate_xai.eda.config import EDAConfig, load_eda_config
from friction_surrogate_xai.eda.markdown import MarkdownSummaryWriter
from friction_surrogate_xai.eda.mlflow_logging import EDAMLflowLogger
from friction_surrogate_xai.eda.outliers import OutlierDetector, OutlierReports
from friction_surrogate_xai.eda.plots import EDAPlotter, PlotArtifacts
from friction_surrogate_xai.eda.statistics import StatisticalAnalyzer, StatisticalTables
from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename, write_csv, write_matrix_csv


@dataclass(frozen=True)
class DatasetEDAArtifacts:
    """Generated paths for one dataset EDA run."""

    dataset_key: str
    root_dir: Path
    plot_paths: tuple[Path, ...]
    table_paths: tuple[Path, ...]
    markdown_path: Path


class EDAReportGenerator:
    """Generate EDA plots, tables, markdown summaries, CSV reports, and MLflow logs."""

    def __init__(
        self,
        config: EDAConfig | None = None,
        data_loader: DataLoader | None = None,
    ) -> None:
        self.config = config or load_eda_config()
        self.data_loader = data_loader or DataLoader()
        stats_config = self.config.statistics
        self.statistical_analyzer = StatisticalAnalyzer(
            confidence_level=float(stats_config.get("confidence_level", 0.95)),
            normality_alpha=float(stats_config.get("normality_alpha", 0.05)),
            correlation_methods=tuple(
                stats_config.get("correlation_methods", ("pearson", "spearman", "kendall"))
            ),
        )
        outlier_config = self.config.outliers
        self.outlier_detector = OutlierDetector(
            iqr_multiplier=float(outlier_config.get("iqr_multiplier", 1.5)),
            isolation_forest_config=dict(outlier_config.get("isolation_forest", {})),
            lof_config=dict(outlier_config.get("local_outlier_factor", {})),
        )
        self.plotter = EDAPlotter(self.config.plots)
        self.markdown_writer = MarkdownSummaryWriter()
        self.mlflow_logger = EDAMLflowLogger(self.config.mlflow)

    def run_all(self, log_to_mlflow: bool | None = None) -> dict[str, DatasetEDAArtifacts]:
        """Generate EDA outputs for all configured datasets."""
        loaded_datasets = self.data_loader.load_all()
        return {
            dataset_key: self.run_dataset(
                loaded_dataset=loaded_dataset,
                log_to_mlflow=log_to_mlflow,
            )
            for dataset_key, loaded_dataset in loaded_datasets.items()
        }

    def run_dataset(
        self,
        loaded_dataset: LoadedDataset,
        log_to_mlflow: bool | None = None,
    ) -> DatasetEDAArtifacts:
        """Generate EDA outputs for one loaded dataset."""
        dataset_key = loaded_dataset.key
        dataset_root = self._dataset_root(dataset_key)
        plots_dir = ensure_directory(dataset_root / self.config.output.get("plots_dir_name", "plots"))
        tables_dir = ensure_directory(dataset_root / self.config.output.get("tables_dir_name", "tables"))
        markdown_dir = ensure_directory(
            dataset_root / self.config.output.get("markdown_dir_name", "markdown")
        )

        analysis_columns = self._analysis_columns(loaded_dataset)
        statistical_tables = self.statistical_analyzer.analyze(
            dataframe=loaded_dataset.dataframe,
            columns=analysis_columns,
        )
        outlier_reports = self.outlier_detector.detect(
            dataframe=loaded_dataset.dataframe,
            columns=analysis_columns,
            id_column=loaded_dataset.report.metadata.id_column,
        )

        table_paths = self._write_tables(
            loaded_dataset=loaded_dataset,
            tables=statistical_tables,
            outliers=outlier_reports,
            tables_dir=tables_dir,
        )
        plot_artifacts = self.plotter.generate(
            dataframe=loaded_dataset.dataframe,
            columns=analysis_columns,
            correlations=statistical_tables.correlations,
            output_dir=plots_dir,
            dataset_label=dataset_key,
        )
        markdown_path = self.markdown_writer.write(
            loaded_dataset=loaded_dataset,
            analysis_columns=analysis_columns,
            outlier_summary=outlier_reports.summary,
            output_path=markdown_dir / "summary.md",
        )

        should_log = self.config.mlflow.get("enabled", True) if log_to_mlflow is None else log_to_mlflow
        if should_log:
            self.mlflow_logger.log_dataset_run(
                dataset_key=dataset_key,
                artifact_dir=dataset_root,
                params=self._mlflow_params(loaded_dataset, analysis_columns),
                metrics=self._mlflow_metrics(loaded_dataset, outlier_reports, plot_artifacts),
            )

        return DatasetEDAArtifacts(
            dataset_key=dataset_key,
            root_dir=dataset_root,
            plot_paths=plot_artifacts.paths,
            table_paths=table_paths,
            markdown_path=markdown_path,
        )

    def _dataset_root(self, dataset_key: str) -> Path:
        configured_root = Path(self.config.output["root_dir"])
        root = configured_root if configured_root.is_absolute() else project_root() / configured_root
        return ensure_directory(root / sanitize_filename(dataset_key))

    def _analysis_columns(self, loaded_dataset: LoadedDataset) -> tuple[str, ...]:
        metadata = loaded_dataset.report.metadata
        selected: list[str] = []
        if self.config.columns.get("include_id_column", False):
            selected.append(metadata.id_column)
        if self.config.columns.get("include_features", True):
            selected.extend(metadata.feature_columns)
        if self.config.columns.get("include_targets", True):
            selected.extend(metadata.target_columns)

        dataframe = loaded_dataset.dataframe
        return tuple(
            column
            for column in dict.fromkeys(selected)
            if column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column])
        )

    def _write_tables(
        self,
        loaded_dataset: LoadedDataset,
        tables: StatisticalTables,
        outliers: OutlierReports,
        tables_dir: Path,
    ) -> tuple[Path, ...]:
        paths = [
            write_csv(tables.descriptive, tables_dir / "descriptive_statistics.csv"),
            write_csv(tables.confidence_intervals, tables_dir / "confidence_intervals.csv"),
            write_csv(tables.normality_tests, tables_dir / "normality_tests.csv"),
            write_csv(outliers.row_scores, tables_dir / "outlier_scores.csv"),
            write_csv(outliers.iqr_outliers, tables_dir / "iqr_outliers.csv"),
            write_csv(outliers.summary, tables_dir / "outlier_summary.csv"),
            write_csv(self._missing_values_table(loaded_dataset), tables_dir / "missing_values.csv"),
            write_csv(self._duplicate_table(loaded_dataset), tables_dir / "duplicates.csv"),
            write_csv(self._constant_features_table(loaded_dataset), tables_dir / "constant_features.csv"),
            write_csv(self._metadata_table(loaded_dataset), tables_dir / "metadata.csv"),
        ]
        for method, matrix in tables.correlations.items():
            paths.append(write_matrix_csv(matrix, tables_dir / f"correlation_{method}.csv"))
        return tuple(paths)

    @staticmethod
    def _missing_values_table(loaded_dataset: LoadedDataset) -> pd.DataFrame:
        report = loaded_dataset.report.missing_values
        return pd.DataFrame(
            [
                {
                    "column": column,
                    "missing_count": report.counts[column],
                    "missing_ratio": report.ratios[column],
                }
                for column in report.counts
            ]
        )

    @staticmethod
    def _duplicate_table(loaded_dataset: LoadedDataset) -> pd.DataFrame:
        report = loaded_dataset.report.duplicates
        return pd.DataFrame(
            [
                {
                    "full_duplicate_count": report.full_duplicate_count,
                    "full_duplicate_indices": ";".join(map(str, report.full_duplicate_indices)),
                    "id_duplicate_count": report.id_duplicate_count,
                    "id_duplicate_values": ";".join(map(str, report.id_duplicate_values)),
                }
            ]
        )

    @staticmethod
    def _constant_features_table(loaded_dataset: LoadedDataset) -> pd.DataFrame:
        report = loaded_dataset.report.constants
        rows = []
        for column in report.unique_counts:
            rows.append(
                {
                    "column": column,
                    "unique_count": report.unique_counts[column],
                    "is_constant_column": column in report.constant_columns,
                    "is_constant_feature": column in report.constant_feature_columns,
                    "constant_value": report.constant_values.get(column),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _metadata_table(loaded_dataset: LoadedDataset) -> pd.DataFrame:
        metadata = loaded_dataset.report.metadata
        return pd.DataFrame(
            [
                {
                    "dataset_key": metadata.key,
                    "path": str(metadata.path),
                    "sheet_name": metadata.sheet_name,
                    "rows": metadata.rows,
                    "columns": metadata.columns,
                    "id_column": metadata.id_column,
                    "feature_columns": ";".join(metadata.feature_columns),
                    "target_columns": ";".join(metadata.target_columns),
                }
            ]
        )

    @staticmethod
    def _mlflow_params(loaded_dataset: LoadedDataset, analysis_columns: tuple[str, ...]) -> dict[str, Any]:
        metadata = loaded_dataset.report.metadata
        return {
            "dataset_key": metadata.key,
            "rows": metadata.rows,
            "columns": metadata.columns,
            "analysis_column_count": len(analysis_columns),
            "outlier_policy": "detect_only_never_remove",
        }

    @staticmethod
    def _mlflow_metrics(
        loaded_dataset: LoadedDataset,
        outliers: OutlierReports,
        plots: PlotArtifacts,
    ) -> dict[str, float | int]:
        report = loaded_dataset.report
        row_scores = outliers.row_scores
        metrics: dict[str, float | int] = {
            "missing_values_total": report.missing_values.total_missing,
            "full_duplicate_count": report.duplicates.full_duplicate_count,
            "id_duplicate_count": report.duplicates.id_duplicate_count,
            "constant_feature_count": len(report.constants.constant_feature_columns),
            "iqr_outlier_rows": int(row_scores["iqr_is_outlier"].sum()),
            "iqr_outlier_cells": int(len(outliers.iqr_outliers)),
            "plot_count": len(plots.paths),
        }
        if "isolation_forest_is_outlier" in row_scores.columns:
            metrics["isolation_forest_outlier_rows"] = int(
                row_scores["isolation_forest_is_outlier"].sum()
            )
        if "lof_is_outlier" in row_scores.columns:
            metrics["lof_outlier_rows"] = int(row_scores["lof_is_outlier"].sum())
        return metrics

