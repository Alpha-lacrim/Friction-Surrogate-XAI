"""CSV and Markdown report generation for uncertainty outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.eda.utils import ensure_directory, write_csv
from friction_surrogate_xai.evaluation.reports import EvaluationReportWriter


@dataclass(frozen=True)
class UncertaintyReportPaths:
    """Generated report paths."""

    table_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]


class UncertaintyReportWriter:
    """Write publication-ready uncertainty tables and summaries."""

    def __init__(self, reports_config: dict[str, Any]) -> None:
        self.config = reports_config
        self.markdown = EvaluationReportWriter(reports_config)

    def write(
        self,
        *,
        dataset_key: str,
        target_name: str,
        prediction_intervals: pd.DataFrame,
        confidence_bands: pd.DataFrame,
        summary: pd.DataFrame,
        comparison: pd.DataFrame,
        figure_paths: tuple[Path, ...],
        tables_dir: Path,
        markdown_dir: Path,
    ) -> UncertaintyReportPaths:
        """Write uncertainty CSV and Markdown artifacts."""
        table_paths: list[Path] = []
        markdown_paths: list[Path] = []
        if self.config.get("write_csv", True):
            table_paths.extend(
                [
                    write_csv(prediction_intervals, tables_dir / "prediction_intervals.csv"),
                    write_csv(confidence_bands, tables_dir / "confidence_bands.csv"),
                    write_csv(summary, tables_dir / "uncertainty_summary.csv"),
                    write_csv(comparison, tables_dir / "comparison_report.csv"),
                ]
            )

        if self.config.get("write_markdown", True):
            markdown_paths.append(
                self._summary_markdown(
                    dataset_key=dataset_key,
                    target_name=target_name,
                    summary=summary,
                    comparison=comparison,
                    figure_paths=figure_paths,
                    output_path=markdown_dir / "uncertainty_summary.md",
                )
            )
        return UncertaintyReportPaths(tuple(table_paths), tuple(markdown_paths))

    def _summary_markdown(
        self,
        *,
        dataset_key: str,
        target_name: str,
        summary: pd.DataFrame,
        comparison: pd.DataFrame,
        figure_paths: tuple[Path, ...],
        output_path: Path,
    ) -> Path:
        lines = [
            f"# Uncertainty Summary: {dataset_key} / {target_name}",
            "",
            "## Scope",
            "",
            (
                "This report compares native Gaussian Process predictive intervals with "
                "bootstrap out-of-bag prediction intervals for non-GPR models."
            ),
            "",
            "## Coverage And Width",
            "",
            self.markdown._markdown_table(summary),
            "",
            "## Model Comparison",
            "",
            (
                "Lower coverage error indicates intervals closer to the requested confidence "
                "level. Width should be interpreted together with coverage."
            ),
            "",
            self.markdown._markdown_table(comparison),
            "",
            "## Generated Figures",
            "",
        ]
        lines.extend(_relative_figure_lines(figure_paths, output_path))
        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


def build_comparison_report(summary: pd.DataFrame) -> pd.DataFrame:
    """Build model comparison report from interval summaries."""
    if summary.empty:
        return pd.DataFrame()
    comparison = summary.copy()
    comparison["coverage_rank"] = comparison.groupby("target")["coverage_error"].rank(
        method="min",
        ascending=True,
    )
    comparison["width_rank"] = comparison.groupby("target")["mean_interval_width"].rank(
        method="min",
        ascending=True,
    )
    comparison["uncertainty_rank_score"] = comparison["coverage_rank"] + comparison["width_rank"]
    comparison["comparison_note"] = comparison.apply(_comparison_note, axis=1)
    return comparison.sort_values(
        ["target", "coverage_rank", "width_rank", "model_key"],
    ).reset_index(drop=True)


def _comparison_note(row: pd.Series) -> str:
    coverage = row.get("coverage_probability")
    level = row.get("interval_level")
    width = row.get("mean_interval_width")
    if pd.isna(coverage):
        return "Coverage could not be estimated."
    if coverage < level:
        calibration = "under-covers relative to the requested interval level"
    elif coverage > level:
        calibration = "over-covers relative to the requested interval level"
    else:
        calibration = "matches the requested interval level"
    return f"The interval {calibration}; mean width is {width}."


def _relative_figure_lines(figure_paths: tuple[Path, ...], output_path: Path) -> list[str]:
    root = output_path.parents[1]
    lines: list[str] = []
    for path in figure_paths:
        try:
            display = path.relative_to(root)
        except ValueError:
            display = path
        lines.append(f"- `{display}`")
    return lines
