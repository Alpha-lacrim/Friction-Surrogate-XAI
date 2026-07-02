"""Markdown summary generation for EDA."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from friction_surrogate_xai.data.contracts import LoadedDataset
from friction_surrogate_xai.eda.utils import ensure_directory


class MarkdownSummaryWriter:
    """Write dataset-level EDA markdown summaries."""

    def write(
        self,
        loaded_dataset: LoadedDataset,
        analysis_columns: tuple[str, ...],
        outlier_summary: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        """Write a markdown summary for one dataset."""
        report = loaded_dataset.report
        constants = report.constants
        lines = [
            f"# EDA Summary: {loaded_dataset.key}",
            "",
            "## Dataset",
            "",
            f"- Rows: {report.metadata.rows}",
            f"- Columns: {report.metadata.columns}",
            f"- Sheet: `{report.metadata.sheet_name}`",
            f"- Analysis columns: {len(analysis_columns)}",
            f"- Missing values: {report.missing_values.total_missing}",
            f"- Duplicate rows or IDs detected: {report.duplicates.has_duplicates}",
            "",
            "## Constant Features",
            "",
        ]

        if constants.constant_feature_columns:
            for column in constants.constant_feature_columns:
                value = constants.constant_values.get(column)
                lines.append(f"- `{column}` is constant at `{value}`.")
        else:
            lines.append("- No configured feature columns are constant.")

        lines.extend(
            [
                "",
                "## Outlier Policy",
                "",
                "Outliers are detected only. No rows are removed, filtered, or modified.",
                "",
                "## Outlier Summary",
                "",
                self._markdown_table(outlier_summary),
                "",
                "## Generated Reports",
                "",
                "- `tables/descriptive_statistics.csv`",
                "- `tables/confidence_intervals.csv`",
                "- `tables/normality_tests.csv`",
                "- `tables/correlation_*.csv`",
                "- `tables/outlier_scores.csv`",
                "- `tables/iqr_outliers.csv`",
                "- `plots/`",
            ]
        )

        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    @staticmethod
    def _markdown_table(dataframe: pd.DataFrame) -> str:
        """Render a small dataframe as markdown without optional dependencies."""
        if dataframe.empty:
            return "_No rows._"
        columns = list(dataframe.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in dataframe.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
        return "\n".join(lines)
