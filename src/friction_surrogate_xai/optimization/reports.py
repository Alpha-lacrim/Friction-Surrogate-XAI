"""CSV and Markdown reports for hyperparameter optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.eda.utils import ensure_directory, write_csv
from friction_surrogate_xai.evaluation.reports import EvaluationReportWriter


@dataclass(frozen=True)
class OptimizationReportPaths:
    """Generated optimization report paths."""

    table_paths: tuple[Path, ...]
    markdown_paths: tuple[Path, ...]


class OptimizationReportWriter:
    """Write optimization histories, best parameters, and summaries."""

    def __init__(
        self,
        reports_config: dict[str, Any],
        output_config: dict[str, Any] | None = None,
    ) -> None:
        self.config = reports_config
        self.output_config = output_config or {}
        self.markdown = EvaluationReportWriter(reports_config)

    def write(
        self,
        *,
        dataset_key: str,
        target_name: str,
        history: pd.DataFrame,
        stage1_results: pd.DataFrame,
        top_models: pd.DataFrame,
        optuna_results: pd.DataFrame,
        best_parameters: pd.DataFrame,
        parameter_importance: pd.DataFrame,
        root_dir: Path,
    ) -> OptimizationReportPaths:
        """Write all optimization report artifacts."""
        tables_dir = ensure_directory(
            root_dir / self.output_config.get("tables_dir_name", "tables")
        )
        markdown_dir = ensure_directory(
            root_dir / self.output_config.get("markdown_dir_name", "markdown")
        )
        table_paths: list[Path] = []
        markdown_paths: list[Path] = []

        if self.config.get("write_csv", True):
            table_paths.extend(
                [
                    write_csv(history, tables_dir / "optimization_history.csv"),
                    write_csv(stage1_results, tables_dir / "stage1_random_search_results.csv"),
                    write_csv(top_models, tables_dir / "stage2_top_models.csv"),
                    write_csv(optuna_results, tables_dir / "stage3_optuna_results.csv"),
                    write_csv(best_parameters, tables_dir / "best_parameters.csv"),
                    write_csv(parameter_importance, tables_dir / "parameter_importance.csv"),
                ]
            )
            self._write_best_parameters_json(best_parameters, tables_dir / "best_parameters.json")
            table_paths.append(tables_dir / "best_parameters.json")

        if self.config.get("write_markdown", True):
            markdown_paths.extend(
                [
                    self._write_table(best_parameters, markdown_dir / "best_parameters.md"),
                    self._write_table(top_models, markdown_dir / "stage2_top_models.md"),
                    self._write_summary(
                        dataset_key=dataset_key,
                        target_name=target_name,
                        history=history,
                        top_models=top_models,
                        best_parameters=best_parameters,
                        output_path=markdown_dir / "optimization_summary.md",
                    ),
                ]
            )
        return OptimizationReportPaths(
            table_paths=tuple(table_paths),
            markdown_paths=tuple(markdown_paths),
        )

    def _write_summary(
        self,
        *,
        dataset_key: str,
        target_name: str,
        history: pd.DataFrame,
        top_models: pd.DataFrame,
        best_parameters: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        lines = [
            f"# Hyperparameter Optimization Summary: {dataset_key} / {target_name}",
            "",
            "## Search Policy",
            "",
            "- Stage 1: Random Search for every selected model.",
            "- Stage 2: rank Stage 1 results and select the top 3 models.",
            "- Stage 3: Optuna Bayesian optimization only for the selected top models.",
            "- Grid Search is not used.",
            "",
            "## Trial Counts",
            "",
            self.markdown._markdown_table(
                history.groupby(["stage", "model_key"]).size().reset_index(name="trial_count")
                if not history.empty
                else pd.DataFrame()
            ),
            "",
            "## Stage 2 Top Models",
            "",
            self.markdown._markdown_table(top_models),
            "",
            "## Best Parameters",
            "",
            self.markdown._markdown_table(best_parameters),
            "",
            "## Generated Tables",
            "",
            "- `tables/optimization_history.csv`",
            "- `tables/stage1_random_search_results.csv`",
            "- `tables/stage2_top_models.csv`",
            "- `tables/stage3_optuna_results.csv`",
            "- `tables/best_parameters.csv`",
            "- `tables/best_parameters.json`",
            "- `tables/parameter_importance.csv`",
            "- `plots/`",
        ]
        ensure_directory(output_path.parent)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _write_table(self, dataframe: pd.DataFrame, output_path: Path) -> Path:
        ensure_directory(output_path.parent)
        output_path.write_text(self.markdown._markdown_table(dataframe), encoding="utf-8")
        return output_path

    @staticmethod
    def _write_best_parameters_json(dataframe: pd.DataFrame, output_path: Path) -> Path:
        ensure_directory(output_path.parent)
        records = []
        for _, row in dataframe.iterrows():
            params_json = row.get("params_json", "{}")
            records.append(
                {
                    "model_key": row.get("model_key"),
                    "stage": row.get("stage"),
                    "objective_value": _json_value(row.get("objective_value")),
                    "params": json.loads(params_json) if isinstance(params_json, str) else {},
                }
            )
        output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return output_path


def _json_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
