"""Publication-style plots for statistical comparison results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename


class StatisticalComparisonPlotter:
    """Visualize statistically significant differences and average ranks."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        sns.set_theme(
            style=config.get("style", "whitegrid"),
            context=config.get("context", "paper"),
            palette=config.get("palette", "deep"),
        )
        plt.rcParams.update(
            {
                "figure.dpi": int(config.get("dpi", 300)),
                "savefig.dpi": int(config.get("dpi", 300)),
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
            }
        )

    def write(
        self,
        *,
        wilcoxon: pd.DataFrame,
        nemenyi: pd.DataFrame,
        average_ranks: pd.DataFrame,
        figures_dir: Path,
    ) -> tuple[Path, ...]:
        """Write all configured comparison plots."""
        if not self.config.get("enabled", True):
            return ()
        paths: list[Path] = []
        if not wilcoxon.empty:
            paths.extend(self._pairwise_heatmaps(wilcoxon, figures_dir / "wilcoxon"))
        if not nemenyi.empty:
            paths.extend(self._pairwise_heatmaps(nemenyi, figures_dir / "nemenyi"))
        if not average_ranks.empty:
            paths.extend(self._rank_plots(average_ranks, figures_dir / "average_ranks"))
        return tuple(paths)

    def _pairwise_heatmaps(self, table: pd.DataFrame, output_dir: Path) -> list[Path]:
        paths: list[Path] = []
        context_columns = [
            column
            for column in (
                "comparison_name",
                "dataset_key",
                "target_name",
                "model_key",
                "variant",
                "output_mode",
            )
            if column in table.columns
        ]
        for context_values, group in table.groupby(context_columns, dropna=False):
            context = _context_label(context_columns, context_values)
            matrix = self._p_value_matrix(group)
            if matrix.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 5.0))
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3g",
                cmap="mako_r",
                vmin=0.0,
                vmax=max(0.05, float(matrix.max().max())),
                linewidths=0.4,
                ax=ax,
            )
            ax.set_title(f"P-value Matrix: {context}")
            paths.append(self._save(fig, output_dir / sanitize_filename(context)))
        return paths

    @staticmethod
    def _p_value_matrix(group: pd.DataFrame) -> pd.DataFrame:
        groups = sorted(set(group["group_a"]).union(set(group["group_b"])))
        matrix = pd.DataFrame(1.0, index=groups, columns=groups)
        for _, row in group.iterrows():
            value = pd.to_numeric(pd.Series([row.get("p_value")]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            matrix.loc[row["group_a"], row["group_b"]] = float(value)
            matrix.loc[row["group_b"], row["group_a"]] = float(value)
        return matrix

    def _rank_plots(self, table: pd.DataFrame, output_dir: Path) -> list[Path]:
        paths: list[Path] = []
        context_columns = [
            column
            for column in (
                "comparison_name",
                "dataset_key",
                "target_name",
                "variant",
                "output_mode",
            )
            if column in table.columns
        ]
        for context_values, group in table.groupby(context_columns, dropna=False):
            context = _context_label(context_columns, context_values)
            plot_table = group.sort_values("average_rank", ascending=True).copy()
            fig, ax = plt.subplots(figsize=(7.0, 4.8))
            sns.barplot(data=plot_table, x="average_rank", y="group", ax=ax)
            ax.set_title(f"Average Ranks: {context}")
            ax.set_xlabel("Average rank; lower is better")
            ax.set_ylabel("Group")
            paths.append(self._save(fig, output_dir / sanitize_filename(context)))
        return paths

    def _save(self, fig: plt.Figure, path_without_suffix: Path) -> Path:
        figure_format = str(self.config.get("figure_format", "png")).lower()
        path = path_without_suffix.with_suffix(f".{figure_format}")
        ensure_directory(path.parent)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path


def _context_label(columns: list[str], values: Any) -> str:
    if not isinstance(values, tuple):
        values = (values,)
    parts = [f"{column}={value}" for column, value in zip(columns, values, strict=False)]
    return ", ".join(parts)
