"""Permutation, tree importance, and tree-interpreter style analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename, write_csv
from friction_surrogate_xai.xai.plotting import XAIPlotStyle
from friction_surrogate_xai.xai.preparation import PreparedXAIModel


@dataclass(frozen=True)
class ImportanceArtifacts:
    """Generated non-SHAP importance artifacts."""

    permutation_importance: pd.DataFrame
    tree_importance: pd.DataFrame
    tree_interpreter: pd.DataFrame
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]


class ImportanceAnalyzer:
    """Generate permutation, tree, and tree-interpreter style reports."""

    def __init__(
        self,
        *,
        permutation_config: dict[str, Any],
        tree_importance_config: dict[str, Any],
        tree_interpreter_config: dict[str, Any],
        plot_config: dict[str, Any],
    ) -> None:
        self.permutation_config = permutation_config
        self.tree_importance_config = tree_importance_config
        self.tree_interpreter_config = tree_interpreter_config
        self.plot_style = XAIPlotStyle(plot_config)

    def analyze(
        self,
        *,
        prepared: PreparedXAIModel,
        shap_values: np.ndarray | None,
        tables_dir: Path,
        figures_dir: Path,
    ) -> ImportanceArtifacts:
        """Generate all non-SHAP importance reports."""
        table_paths: list[Path] = []
        figure_paths: list[Path] = []
        importance_dir = ensure_directory(tables_dir / "importance")
        figure_dir = ensure_directory(figures_dir / "importance")

        permutation = self._permutation(prepared)
        if not permutation.empty:
            table_paths.append(write_csv(permutation, importance_dir / "permutation_importance.csv"))
            figure_paths.append(
                self._bar_plot(
                    permutation,
                    value_column="importance_mean",
                    title="Permutation Importance",
                    output_path=figure_dir / "permutation_importance",
                )
            )

        tree = self._tree_importance(prepared)
        if not tree.empty:
            table_paths.append(write_csv(tree, importance_dir / "tree_importance.csv"))
            figure_paths.append(
                self._bar_plot(
                    tree,
                    value_column="importance",
                    title="Tree Feature Importance",
                    output_path=figure_dir / "tree_importance",
                )
            )

        tree_interpreter = self._tree_interpreter(prepared, shap_values)
        if not tree_interpreter.empty:
            table_paths.append(
                write_csv(tree_interpreter, importance_dir / "tree_interpreter_contributions.csv")
            )
            global_tree_interpreter = (
                tree_interpreter.groupby("feature", as_index=False)["abs_contribution"]
                .mean()
                .rename(columns={"abs_contribution": "mean_abs_contribution"})
                .sort_values("mean_abs_contribution", ascending=False)
            )
            figure_paths.append(
                self._bar_plot(
                    global_tree_interpreter,
                    value_column="mean_abs_contribution",
                    title="Tree Interpreter Mean Absolute Contributions",
                    output_path=figure_dir / "tree_interpreter",
                )
            )

        return ImportanceArtifacts(
            permutation_importance=permutation,
            tree_importance=tree,
            tree_interpreter=tree_interpreter,
            table_paths=tuple(table_paths),
            figure_paths=tuple(figure_paths),
        )

    def _permutation(self, prepared: PreparedXAIModel) -> pd.DataFrame:
        if not self.permutation_config.get("enabled", True):
            return pd.DataFrame()
        result = permutation_importance(
            prepared.estimator,
            prepared.processed_features,
            prepared.target,
            n_repeats=int(self.permutation_config.get("n_repeats", 20)),
            random_state=int(self.permutation_config.get("random_state", 42)),
            scoring=self.permutation_config.get("scoring", "r2"),
        )
        table = pd.DataFrame(
            {
                "feature": prepared.processed_features.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        table["rank"] = range(1, len(table) + 1)
        return table

    def _tree_importance(self, prepared: PreparedXAIModel) -> pd.DataFrame:
        if not self.tree_importance_config.get("enabled", True):
            return pd.DataFrame()
        if not hasattr(prepared.estimator, "feature_importances_"):
            return pd.DataFrame(
                [
                    {
                        "feature": "",
                        "importance": np.nan,
                        "rank": "",
                        "status": "model_has_no_feature_importances",
                    }
                ]
            )
        importance = prepared.estimator.feature_importances_
        table = pd.DataFrame(
            {
                "feature": prepared.processed_features.columns,
                "importance": importance,
                "status": "ok",
            }
        ).sort_values("importance", ascending=False)
        table["rank"] = range(1, len(table) + 1)
        return table

    def _tree_interpreter(
        self,
        prepared: PreparedXAIModel,
        shap_values: np.ndarray | None,
    ) -> pd.DataFrame:
        if not self.tree_interpreter_config.get("enabled", True):
            return pd.DataFrame()
        sample_count = min(
            int(self.tree_interpreter_config.get("max_local_samples", 3)),
            len(prepared.processed_features),
        )
        if sample_count <= 0:
            return pd.DataFrame()

        try:
            from treeinterpreter import treeinterpreter as ti

            predictions, bias, contributions = ti.predict(
                prepared.estimator,
                prepared.processed_features.iloc[:sample_count],
            )
            contribution_values = _normalize_contributions(contributions)
            method = "treeinterpreter"
        except Exception:
            if shap_values is None or not self.tree_interpreter_config.get(
                "fallback_to_tree_shap",
                True,
            ):
                return pd.DataFrame(
                    [
                        {
                            "sample_position": None,
                            "feature": "",
                            "contribution": np.nan,
                            "abs_contribution": np.nan,
                            "method": "unavailable",
                            "status": "treeinterpreter_unavailable_and_no_fallback",
                        }
                    ]
                )
            contribution_values = shap_values[:sample_count]
            method = "tree_shap_fallback"

        rows: list[dict[str, Any]] = []
        for sample_position in range(sample_count):
            for feature_index, feature in enumerate(prepared.processed_features.columns):
                contribution = float(contribution_values[sample_position, feature_index])
                rows.append(
                    {
                        "sample_position": sample_position,
                        "feature": feature,
                        "feature_value": prepared.processed_features.iloc[
                            sample_position,
                            feature_index,
                        ],
                        "contribution": contribution,
                        "abs_contribution": abs(contribution),
                        "method": method,
                        "status": "ok",
                    }
                )
        return pd.DataFrame(rows)

    def _bar_plot(
        self,
        table: pd.DataFrame,
        *,
        value_column: str,
        title: str,
        output_path: Path,
    ) -> Path:
        plot_table = table.loc[table[value_column].notna()].head(15).copy()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=plot_table, x=value_column, y="feature", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(value_column.replace("_", " ").title())
        ax.set_ylabel("Feature")
        return self.plot_style.save(fig, output_path)


def _normalize_contributions(contributions: Any) -> np.ndarray:
    array = np.asarray(contributions)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array.astype(float)
