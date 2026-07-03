"""Shared plotting helpers for XAI reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from friction_surrogate_xai.eda.utils import ensure_directory


class XAIPlotStyle:
    """Apply publication-style plotting defaults and save figures."""

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

    def save(self, fig: plt.Figure, path_without_suffix: Path) -> Path:
        """Save a Matplotlib figure and return the path."""
        figure_format = self.config.get("figure_format", "png").lower()
        path = path_without_suffix.with_suffix(f".{figure_format}")
        ensure_directory(path.parent)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
