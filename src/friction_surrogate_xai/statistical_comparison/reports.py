"""CSV and Markdown reports for statistical comparison outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.eda.utils import ensure_directory, write_csv
from friction_surrogate_xai.evaluation.reports import EvaluationReportWriter


@dataclass(frozen=True)
class StatisticalComparisonReportPaths:
    """Generated report artifact paths."""

    table_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]


class StatisticalComparisonReportWriter:
    """Write publication-ready comparison tables and summaries."""

    def __init__(self, reports_config: dict[str, Any]) -> None:
        self.config = reports_config
        self.markdown = EvaluationReportWriter(reports_config)

    def write(
        self,
        *,
        score_table: pd.DataFrame,
        wilcoxon: pd.DataFrame,
        friedman: pd.DataFrame,
        nemenyi: pd.DataFrame,
        average_ranks: pd.DataFrame,
        figure_paths: tuple[Path, ...],
        tables_dir: Path,
        markdown_dir: Path,
    ) -> StatisticalComparisonReportPaths:
        """Write all configured statistical-comparison reports."""
        table_paths: list[Path] = []
        markdown_paths: list[Path] = []
        if self.config.get("write_csv", True):
            table_paths.extend(
                [
                    write_csv(score_table, tables_dir / "normalized_scores.csv"),
                    write_csv(wilcoxon, tables_dir / "wilcoxon_signed_rank.csv"),
                    write_csv(friedman, tables_dir / "friedman_test.csv"),
                    write_csv(nemenyi, tables_dir / "nemenyi_post_hoc.csv"),
                    write_csv(average_ranks, tables_dir / "average_ranks.csv"),
                    write_csv(
                        significant_findings(wilcoxon, nemenyi),
                        tables_dir / "significant_findings.csv",
                    ),
                ]
            )
        if self.config.get("write_markdown", True):
            markdown_paths.append(
                self._summary(
                    score_table=score_table,
                    wilcoxon=wilcoxon,
                    friedman=friedman,
                    nemenyi=nemenyi,
                    average_ranks=average_ranks,
                    figure_paths=figure_paths,
                    output_path=markdown_dir / "statistical_comparison_summary.md",
                )
            )
        return StatisticalComparisonReportPaths(tuple(table_paths), tuple(markdown_paths))

    def _summary(
        self,
        *,
        score_table: pd.DataFrame,
        wilcoxon: pd.DataFrame,
        friedman: pd.DataFrame,
        nemenyi: pd.DataFrame,
        average_ranks: pd.DataFrame,
        figure_paths: tuple[Path, ...],
        output_path: Path,
    ) -> Path:
        findings = significant_findings(wilcoxon, nemenyi)
        lines = [
            "# Statistical Model Comparison",
            "",
            "## Scope",
            "",
            (
                "This report uses paired nonparametric tests suitable for tiny repeated "
                "experimental score tables."
            ),
            "",
            "## Input Summary",
            "",
            f"- Score rows: `{len(score_table)}`",
            (
                "- Comparison families: "
                f"`{score_table['comparison_type'].nunique() if not score_table.empty else 0}`"
            ),
            (
                "- Models/groups: "
                f"`{score_table['model_key'].nunique() if not score_table.empty else 0}`"
            ),
            "",
            "## Friedman Omnibus Tests",
            "",
            self.markdown._markdown_table(friedman),
            "",
            "## Significant Pairwise Findings",
            "",
            self.markdown._markdown_table(findings),
            "",
            "## Average Ranks",
            "",
            self.markdown._markdown_table(average_ranks),
            "",
            "## Generated Figures",
            "",
        ]
        lines.extend(_relative_figure_lines(figure_paths, output_path))
        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path


def significant_findings(wilcoxon: pd.DataFrame, nemenyi: pd.DataFrame) -> pd.DataFrame:
    """Return significant pairwise test rows."""
    frames: list[pd.DataFrame] = []
    for table in (wilcoxon, nemenyi):
        if not table.empty and "significant" in table:
            frames.append(table.loc[table["significant"] == True].copy())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
