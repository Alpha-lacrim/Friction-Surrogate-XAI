"""Publication-style plots for uncertainty reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename


class UncertaintyPlotter:
    """Generate confidence-band and comparison plots."""

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

    def write_all(
        self,
        *,
        confidence_bands: pd.DataFrame,
        comparison: pd.DataFrame,
        figures_dir: Path,
    ) -> tuple[Path, ...]:
        """Write all configured uncertainty figures."""
        if not self.config.get("enabled", True):
            return ()
        paths: list[Path] = []
        if not confidence_bands.empty:
            for _, group in confidence_bands.groupby(["model_key", "target"], dropna=False):
                paths.append(self._confidence_band_plot(group, figures_dir / "confidence_bands"))
        if not comparison.empty:
            paths.append(
                self._comparison_bar(
                    comparison,
                    value_column="coverage_probability",
                    title="Coverage Probability",
                    output_path=figures_dir / "comparison" / "coverage_probability",
                    reference_column="interval_level",
                )
            )
            paths.append(
                self._comparison_bar(
                    comparison,
                    value_column="mean_interval_width",
                    title="Mean Prediction Interval Width",
                    output_path=figures_dir / "comparison" / "mean_interval_width",
                )
            )
        return tuple(paths)

    def _confidence_band_plot(self, table: pd.DataFrame, output_dir: Path) -> Path:
        max_points = int(self.config.get("max_band_points", 120))
        plot_table = table.copy().head(max_points)
        plot_table = plot_table.sort_values("sample_index").reset_index(drop=True)
        x_values = range(len(plot_table))
        model_key = str(plot_table["model_key"].iloc[0])
        target = str(plot_table["target"].iloc[0])

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.fill_between(
            list(x_values),
            plot_table["interval_lower"].to_numpy(dtype=float),
            plot_table["interval_upper"].to_numpy(dtype=float),
            alpha=0.22,
            label="Prediction interval",
        )
        ax.plot(
            list(x_values),
            plot_table["predictive_mean"],
            linewidth=1.6,
            label="Predictive mean",
        )
        ax.scatter(list(x_values), plot_table["y_true"], s=24, color="black", label="Observed")
        ax.set_title(f"Confidence Bands: {model_key} / {target}")
        ax.set_xlabel("Sample order")
        ax.set_ylabel(target)
        ax.legend(loc="best")
        return self._save(
            fig,
            output_dir / f"{sanitize_filename(model_key)}_{sanitize_filename(target)}",
        )

    def _comparison_bar(
        self,
        table: pd.DataFrame,
        *,
        value_column: str,
        title: str,
        output_path: Path,
        reference_column: str | None = None,
    ) -> Path:
        plot_table = table.sort_values(value_column, ascending=False).copy()
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.barplot(data=plot_table, x=value_column, y="model_key", hue="target", ax=ax)
        if reference_column and reference_column in plot_table:
            reference = pd.to_numeric(plot_table[reference_column], errors="coerce").dropna()
            if not reference.empty:
                ax.axvline(float(reference.iloc[0]), color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel(value_column.replace("_", " ").title())
        ax.set_ylabel("Model")
        return self._save(fig, output_path)

    def _save(self, fig: plt.Figure, path_without_suffix: Path) -> Path:
        figure_format = str(self.config.get("figure_format", "png")).lower()
        path = path_without_suffix.with_suffix(f".{figure_format}")
        ensure_directory(path.parent)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
