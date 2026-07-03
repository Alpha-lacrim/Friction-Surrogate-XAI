"""Publication-quality optimization plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename


class OptimizationPlotter:
    """Generate optimization history and parameter-importance plots."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
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
            }
        )

    def generate(
        self,
        *,
        history: pd.DataFrame,
        top_models: pd.DataFrame,
        parameter_importance: pd.DataFrame,
        output_dir: Path,
    ) -> tuple[Path, ...]:
        """Generate all configured plots."""
        if not self.config.get("enabled", True):
            return ()
        paths: list[Path] = []
        if not history.empty:
            paths.append(self._history_plot(history, output_dir))
        if not top_models.empty:
            paths.append(self._top_models_plot(top_models, output_dir))
        if not parameter_importance.empty:
            for model_key, group in parameter_importance.groupby("model_key"):
                paths.append(self._importance_plot(model_key, group, output_dir))
        return tuple(paths)

    def _history_plot(self, history: pd.DataFrame, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        plot_frame = history.sort_values(["stage", "model_key", "trial_number"])
        sns.lineplot(
            data=plot_frame,
            x="trial_number",
            y="objective_value",
            hue="model_key",
            style="stage",
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("Trial")
        ax.set_ylabel("Mean validation objective")
        ax.set_title("Optimization History")
        return self._save(fig, output_dir / "optimization_history")

    def _top_models_plot(self, top_models: pd.DataFrame, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        sns.barplot(
            data=top_models,
            x="model_key",
            y="best_stage1_objective",
            hue="model_key",
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel("Best Stage 1 validation objective")
        ax.set_title("Stage 2 Top Models")
        ax.tick_params(axis="x", rotation=25)
        return self._save(fig, output_dir / "stage2_top_models")

    def _importance_plot(self, model_key: str, importance: pd.DataFrame, output_dir: Path) -> Path:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ordered = importance.sort_values("importance", ascending=False)
        sns.barplot(
            data=ordered,
            x="importance",
            y="parameter",
            hue="parameter",
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Importance")
        ax.set_ylabel("Parameter")
        ax.set_title(f"Optuna Parameter Importance: {model_key}")
        return self._save(
            fig,
            output_dir / f"{sanitize_filename(model_key)}_parameter_importance",
        )

    def _save(self, fig: plt.Figure, path_without_suffix: Path) -> Path:
        figure_format = self.config.get("figure_format", "png").lower()
        path = path_without_suffix.with_suffix(f".{figure_format}")
        ensure_directory(path.parent)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
