"""Publication-quality EDA plot generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename


@dataclass(frozen=True)
class PlotArtifacts:
    """Generated plot artifact paths."""

    paths: tuple[Path, ...]


class EDAPlotter:
    """Generate publication-quality EDA figures."""

    def __init__(self, plot_config: dict) -> None:
        self.config = plot_config
        sns.set_theme(
            style=self.config.get("style", "whitegrid"),
            context=self.config.get("context", "paper"),
            palette=self.config.get("palette", "deep"),
        )
        plt.rcParams.update(
            {
                "figure.dpi": int(self.config.get("dpi", 300)),
                "savefig.dpi": int(self.config.get("dpi", 300)),
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "figure.autolayout": False,
            }
        )

    def generate(
        self,
        dataframe: pd.DataFrame,
        columns: tuple[str, ...],
        correlations: dict[str, pd.DataFrame],
        output_dir: Path,
        dataset_label: str,
    ) -> PlotArtifacts:
        """Generate all configured EDA plots."""
        if not self.config.get("enabled", True):
            return PlotArtifacts(paths=())

        plot_paths: list[Path] = []
        numeric_frame = dataframe.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        for column in columns:
            plot_paths.append(self._histogram(numeric_frame, column, output_dir, dataset_label))
            plot_paths.append(self._kde(numeric_frame, column, output_dir, dataset_label))
            plot_paths.append(self._qq_plot(numeric_frame, column, output_dir, dataset_label))
            plot_paths.append(self._boxplot(numeric_frame, column, output_dir, dataset_label))

        plot_paths.append(self._pairplot(numeric_frame, output_dir, dataset_label))
        for method, matrix in correlations.items():
            plot_paths.append(self._correlation_heatmap(matrix, output_dir, dataset_label, method))

        return PlotArtifacts(paths=tuple(plot_paths))

    def _histogram(
        self,
        dataframe: pd.DataFrame,
        column: str,
        output_dir: Path,
        dataset_label: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=tuple(self.config.get("histogram_figsize", (7, 4.5))))
        sns.histplot(
            data=dataframe,
            x=column,
            bins=self.config.get("histogram_bins", "auto"),
            kde=False,
            ax=ax,
            color=sns.color_palette()[0],
        )
        ax.set_title(f"{dataset_label}: Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
        return self._save(fig, output_dir / "histograms" / f"{sanitize_filename(column)}_histogram")

    def _kde(
        self,
        dataframe: pd.DataFrame,
        column: str,
        output_dir: Path,
        dataset_label: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=tuple(self.config.get("kde_figsize", (7, 4.5))))
        series = dataframe[column].dropna()
        if series.nunique() > 1:
            sns.kdeplot(
                data=dataframe,
                x=column,
                fill=False,
                ax=ax,
                color=sns.color_palette()[1],
                warn_singular=False,
            )
        else:
            ax.axvline(series.iloc[0] if not series.empty else 0, color=sns.color_palette()[1])
            ax.text(0.5, 0.5, "constant column", transform=ax.transAxes, ha="center")
        ax.set_title(f"{dataset_label}: KDE of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        return self._save(fig, output_dir / "kde" / f"{sanitize_filename(column)}_kde")

    def _qq_plot(
        self,
        dataframe: pd.DataFrame,
        column: str,
        output_dir: Path,
        dataset_label: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=tuple(self.config.get("qq_figsize", (6, 6))))
        series = dataframe[column].dropna()
        if len(series) >= 3 and series.nunique() > 1:
            stats.probplot(series, dist=self.config.get("qq_distribution", "norm"), plot=ax)
        else:
            ax.text(0.5, 0.5, "constant or too few values", transform=ax.transAxes, ha="center")
        ax.set_title(f"{dataset_label}: QQ Plot of {column}")
        return self._save(fig, output_dir / "qq_plots" / f"{sanitize_filename(column)}_qq_plot")

    def _boxplot(
        self,
        dataframe: pd.DataFrame,
        column: str,
        output_dir: Path,
        dataset_label: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=tuple(self.config.get("boxplot_figsize", (6, 4.5))))
        sns.boxplot(data=dataframe, y=column, ax=ax, color=sns.color_palette()[2])
        ax.set_title(f"{dataset_label}: Box Plot of {column}")
        ax.set_ylabel(column)
        return self._save(fig, output_dir / "boxplots" / f"{sanitize_filename(column)}_boxplot")

    def _pairplot(self, dataframe: pd.DataFrame, output_dir: Path, dataset_label: str) -> Path:
        pairgrid = sns.pairplot(
            dataframe,
            corner=bool(self.config.get("pairplot_corner", True)),
            diag_kind=self.config.get("pairplot_diag_kind", "kde"),
            diag_kws={"warn_singular": False}
            if self.config.get("pairplot_diag_kind", "kde") == "kde"
            else None,
            height=float(self.config.get("pairplot_height", 2.2)),
            plot_kws={"s": 24, "alpha": 0.8, "edgecolor": "none"},
        )
        pairgrid.fig.suptitle(f"{dataset_label}: Pair Plot", y=1.02)
        path = output_dir / "pairplots" / f"{sanitize_filename(dataset_label)}_pairplot"
        saved = self._save(pairgrid.fig, path)
        plt.close(pairgrid.fig)
        return saved

    def _correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        output_dir: Path,
        dataset_label: str,
        method: str,
    ) -> Path:
        fig, ax = plt.subplots(figsize=tuple(self.config.get("heatmap_figsize", (10, 8))))
        sns.heatmap(
            correlation_matrix,
            cmap="vlag",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.8, "label": f"{method.title()} correlation"},
            ax=ax,
        )
        ax.set_title(f"{dataset_label}: {method.title()} Correlation Heatmap")
        return self._save(
            fig,
            output_dir / "correlations" / f"{sanitize_filename(method)}_correlation_heatmap",
        )

    def _save(self, fig: plt.Figure, path_without_suffix: Path) -> Path:
        figure_format = self.config.get("figure_format", "png").lower()
        path = path_without_suffix.with_suffix(f".{figure_format}")
        ensure_directory(path.parent)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
