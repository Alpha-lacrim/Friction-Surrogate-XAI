"""XAI CSV and Markdown report writing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.eda.utils import ensure_directory, write_csv
from friction_surrogate_xai.evaluation.reports import EvaluationReportWriter
from friction_surrogate_xai.xai.interpretation import ScientificInterpretation


@dataclass(frozen=True)
class XAIReportPaths:
    """Generated report paths."""

    table_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]


class XAIReportWriter:
    """Write XAI scientific interpretation reports."""

    def __init__(self, reports_config: dict[str, Any]) -> None:
        self.config = reports_config
        self.markdown = EvaluationReportWriter(reports_config)

    def write(
        self,
        *,
        dataset_key: str,
        model_key: str,
        target_name: str,
        interpretation: ScientificInterpretation,
        figures: tuple[Path, ...],
        tables_dir: Path,
        markdown_dir: Path,
    ) -> XAIReportPaths:
        """Write interpretation tables and summary Markdown."""
        table_paths: list[Path] = []
        markdown_paths: list[Path] = []
        interpretation_dir = ensure_directory(tables_dir / "interpretation")
        if self.config.get("write_csv", True):
            table_paths.extend(
                [
                    write_csv(
                        interpretation.feature_interpretations,
                        interpretation_dir / "scientific_feature_interpretations.csv",
                    ),
                    write_csv(
                        interpretation.interaction_interpretations,
                        interpretation_dir / "scientific_interaction_interpretations.csv",
                    ),
                ]
            )
        if self.config.get("write_markdown", True):
            markdown_paths.append(
                self._summary(
                    dataset_key=dataset_key,
                    model_key=model_key,
                    target_name=target_name,
                    interpretation=interpretation,
                    figures=figures,
                    output_path=markdown_dir / "xai_scientific_interpretation.md",
                )
            )
        return XAIReportPaths(tuple(table_paths), tuple(markdown_paths))

    def _summary(
        self,
        *,
        dataset_key: str,
        model_key: str,
        target_name: str,
        interpretation: ScientificInterpretation,
        figures: tuple[Path, ...],
        output_path: Path,
    ) -> Path:
        sections = interpretation.markdown_sections
        lines = [
            f"# XAI Scientific Interpretation: {dataset_key} / {target_name} / {model_key}",
            "",
            "## Scope",
            "",
            "This report combines SHAP, permutation importance, tree-based importance, tree-interpreter style contributions, and LIME when available.",
            "",
            "## Most Important Variables",
            "",
            sections.get("most_important_variables", ""),
            "",
            self.markdown._markdown_table(interpretation.feature_interpretations),
            "",
            "## Positive And Negative Effects",
            "",
            sections.get("positive_negative_effects", ""),
            "",
            "## Nonlinear Behavior",
            "",
            sections.get("nonlinear_behavior", ""),
            "",
            "## Feature Interactions",
            "",
            sections.get("feature_interactions", ""),
            "",
            self.markdown._markdown_table(interpretation.interaction_interpretations),
            "",
            "## Possible Engineering Interpretation",
            "",
            sections.get("engineering_interpretation", ""),
            "",
            "## Generated Figures",
            "",
        ]
        lines.extend(f"- `{path.relative_to(output_path.parents[1])}`" for path in figures)
        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
