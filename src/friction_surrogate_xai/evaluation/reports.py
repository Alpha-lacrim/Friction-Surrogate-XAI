"""CSV and Markdown report generation for evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from friction_surrogate_xai.eda.utils import ensure_directory, write_csv


@dataclass(frozen=True)
class EvaluationReportPaths:
    """Generated report artifact paths."""

    table_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]

    @property
    def summary_path(self) -> Path | None:
        """Return the main summary path if it exists."""
        for path in self.markdown_paths:
            if path.name == "summary.md":
                return path
        return None


class EvaluationReportWriter:
    """Write publication-ready CSV and Markdown evaluation reports."""

    def __init__(self, reports_config: dict[str, Any] | None = None) -> None:
        self.config = reports_config or {}
        self.float_format = str(self.config.get("float_format", ".6g"))

    def write(
        self,
        *,
        dataset_key: str,
        model_name: str,
        metrics: pd.DataFrame,
        metrics_long: pd.DataFrame,
        train_test_gap: pd.DataFrame,
        cv_summary: pd.DataFrame,
        plot_paths: tuple[Path, ...],
        tables_dir: Path,
        markdown_dir: Path,
    ) -> EvaluationReportPaths:
        """Write all configured evaluation reports."""
        table_paths: list[Path] = []
        markdown_paths: list[Path] = []

        if self.config.get("write_csv", True):
            table_paths.extend(
                [
                    write_csv(metrics, tables_dir / "metrics.csv"),
                    write_csv(metrics_long, tables_dir / "metrics_long.csv"),
                ]
            )
            if not train_test_gap.empty:
                table_paths.append(write_csv(train_test_gap, tables_dir / "train_test_gap.csv"))
            if not cv_summary.empty:
                table_paths.append(
                    write_csv(cv_summary, tables_dir / "cross_validation_summary.csv")
                )

        if self.config.get("write_markdown", True):
            markdown_paths.extend(
                [
                    self._write_markdown_table(metrics, markdown_dir / "metrics.md"),
                    self._write_markdown_table(metrics_long, markdown_dir / "metrics_long.md"),
                ]
            )
            if not train_test_gap.empty:
                markdown_paths.append(
                    self._write_markdown_table(train_test_gap, markdown_dir / "train_test_gap.md")
                )
            if not cv_summary.empty:
                markdown_paths.append(
                    self._write_markdown_table(
                        cv_summary,
                        markdown_dir / "cross_validation_summary.md",
                    )
                )
            markdown_paths.append(
                self._write_summary(
                    dataset_key=dataset_key,
                    model_name=model_name,
                    metrics=metrics,
                    train_test_gap=train_test_gap,
                    cv_summary=cv_summary,
                    plot_paths=plot_paths,
                    output_path=markdown_dir / "summary.md",
                )
            )

        return EvaluationReportPaths(
            table_paths=tuple(table_paths),
            markdown_paths=tuple(markdown_paths),
        )

    def _write_summary(
        self,
        *,
        dataset_key: str,
        model_name: str,
        metrics: pd.DataFrame,
        train_test_gap: pd.DataFrame,
        cv_summary: pd.DataFrame,
        plot_paths: tuple[Path, ...],
        output_path: Path,
    ) -> Path:
        if "split" in metrics.columns:
            test_metrics = metrics.loc[metrics["split"] == "test"].copy()
        else:
            test_metrics = metrics.copy()
        if test_metrics.empty:
            test_metrics = metrics.copy()

        lines = [
            f"# Evaluation Summary: {dataset_key} / {model_name}",
            "",
            "## Scope",
            "",
            f"- Dataset: `{dataset_key}`",
            f"- Model: `{model_name}`",
            f"- Targets: {self._target_count(metrics)}",
            f"- Splits evaluated: {self._split_list(metrics)}",
            f"- Plot artifacts: {len(plot_paths)}",
            "",
            "## Test Metrics",
            "",
            self._markdown_table(test_metrics),
            "",
        ]

        if not train_test_gap.empty:
            lines.extend(
                [
                    "## Train/Test Gap",
                    "",
                    "Positive gaps indicate possible overfitting under the metric-specific direction.",
                    "",
                    self._markdown_table(train_test_gap),
                    "",
                ]
            )

        if not cv_summary.empty:
            lines.extend(
                [
                    "## Cross-Validation Stability",
                    "",
                    self._markdown_table(cv_summary),
                    "",
                ]
            )

        lines.extend(
            [
                "## Artifact Index",
                "",
                "- `tables/metrics.csv`",
                "- `tables/metrics_long.csv`",
                "- `tables/train_test_gap.csv` when train metrics are provided",
                "- `tables/cross_validation_summary.csv` when fold metrics are provided",
                "- `plots/prediction_vs_actual/`",
                "- `plots/residuals/`",
                "- `plots/learning_curves/` when estimator data are provided",
                "- `plots/validation_curves/` when a validation parameter is provided",
            ]
        )

        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _write_markdown_table(self, dataframe: pd.DataFrame, output_path: Path) -> Path:
        ensure_directory(output_path.parent)
        output_path.write_text(self._markdown_table(dataframe), encoding="utf-8")
        return output_path

    def _markdown_table(self, dataframe: pd.DataFrame) -> str:
        if dataframe.empty:
            return "_No rows._"
        display_frame = dataframe.copy()
        for column in display_frame.columns:
            if pd.api.types.is_numeric_dtype(display_frame[column]):
                display_frame[column] = display_frame[column].map(self._format_value)

        columns = [str(column) for column in display_frame.columns]
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join("---" for _ in columns) + " |",
        ]
        for _, row in display_frame.iterrows():
            lines.append("| " + " | ".join(str(row[column]) for column in display_frame.columns) + " |")
        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return ""
        return format(numeric, self.float_format)

    @staticmethod
    def _target_count(metrics: pd.DataFrame) -> int:
        if "target" not in metrics.columns:
            return 0
        return len([target for target in metrics["target"].unique() if target != "__aggregate__"])

    @staticmethod
    def _split_list(metrics: pd.DataFrame) -> str:
        if "split" not in metrics.columns:
            return ""
        return ", ".join(f"`{split}`" for split in sorted(metrics["split"].astype(str).unique()))
