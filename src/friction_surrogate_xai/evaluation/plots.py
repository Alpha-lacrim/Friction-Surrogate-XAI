"""Publication-quality evaluation plot generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve, validation_curve

from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename
from friction_surrogate_xai.evaluation.arrays import validate_prediction_arrays


@dataclass(frozen=True)
class PlotArtifacts:
    """Generated plot artifact paths."""

    paths: tuple[Path, ...]


class EvaluationPlotter:
    """Generate publication-quality evaluation figures."""

    def __init__(self, plot_config: dict[str, Any]) -> None:
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

    def prediction_vs_actual(
        self,
        *,
        y_true: Any,
        y_pred: Any,
        output_dir: Path,
        dataset_label: str,
        model_name: str,
        split: str,
        target_names: Sequence[str] | None = None,
    ) -> tuple[Path, ...]:
        """Generate prediction-vs-actual plots for each target."""
        if not self.config.get("enabled", True):
            return ()

        true_array, pred_array, names = validate_prediction_arrays(
            y_true,
            y_pred,
            target_names=target_names,
        )
        paths: list[Path] = []
        for target_index, target_name in enumerate(names):
            paths.append(
                self._prediction_vs_actual_single(
                    y_true=true_array[:, target_index],
                    y_pred=pred_array[:, target_index],
                    output_dir=output_dir,
                    dataset_label=dataset_label,
                    model_name=model_name,
                    split=split,
                    target_name=target_name,
                )
            )
        return tuple(paths)

    def residuals(
        self,
        *,
        y_true: Any,
        y_pred: Any,
        output_dir: Path,
        dataset_label: str,
        model_name: str,
        split: str,
        target_names: Sequence[str] | None = None,
    ) -> tuple[Path, ...]:
        """Generate residual plots for each target."""
        if not self.config.get("enabled", True):
            return ()

        true_array, pred_array, names = validate_prediction_arrays(
            y_true,
            y_pred,
            target_names=target_names,
        )
        paths: list[Path] = []
        for target_index, target_name in enumerate(names):
            paths.append(
                self._residual_single(
                    y_true=true_array[:, target_index],
                    y_pred=pred_array[:, target_index],
                    output_dir=output_dir,
                    dataset_label=dataset_label,
                    model_name=model_name,
                    split=split,
                    target_name=target_name,
                )
            )
        return tuple(paths)

    def learning_curve(
        self,
        *,
        estimator: Any,
        X: Any,
        y: Any,
        output_dir: Path,
        dataset_label: str,
        model_name: str,
        curve_config: dict[str, Any],
        target_name: str = "target",
    ) -> Path | None:
        """Generate a learning curve for an sklearn-compatible estimator."""
        if not self.config.get("enabled", True) or not curve_config.get("enabled", True):
            return None

        train_sizes, train_scores, validation_scores = learning_curve(
            estimator=estimator,
            X=X,
            y=y,
            train_sizes=np.asarray(curve_config.get("train_sizes", (0.2, 0.5, 1.0))),
            cv=curve_config.get("cv", 5),
            scoring=curve_config.get("scoring", "neg_root_mean_squared_error"),
            n_jobs=curve_config.get("n_jobs"),
            shuffle=bool(curve_config.get("shuffle", True)),
            random_state=curve_config.get("random_state", 42),
        )
        return self._score_curve_plot(
            x_values=train_sizes,
            train_scores=train_scores,
            validation_scores=validation_scores,
            output_dir=output_dir / "learning_curves",
            dataset_label=dataset_label,
            model_name=model_name,
            target_name=target_name,
            x_label="Training samples",
            title=f"{dataset_label}: Learning Curve ({model_name})",
            filename=f"{sanitize_filename(target_name)}_learning_curve",
            scoring=str(curve_config.get("scoring", "neg_root_mean_squared_error")),
        )

    def validation_curve(
        self,
        *,
        estimator: Any,
        X: Any,
        y: Any,
        output_dir: Path,
        dataset_label: str,
        model_name: str,
        curve_config: dict[str, Any],
        param_name: str | None,
        param_range: Sequence[Any] | None,
        target_name: str = "target",
    ) -> Path | None:
        """Generate a validation curve when a parameter grid is provided."""
        if (
            not self.config.get("enabled", True)
            or not curve_config.get("enabled", True)
            or not param_name
            or param_range is None
        ):
            return None

        train_scores, validation_scores = validation_curve(
            estimator=estimator,
            X=X,
            y=y,
            param_name=param_name,
            param_range=param_range,
            cv=curve_config.get("cv", 5),
            scoring=curve_config.get("scoring", "neg_root_mean_squared_error"),
            n_jobs=curve_config.get("n_jobs"),
        )
        return self._score_curve_plot(
            x_values=np.asarray(param_range),
            train_scores=train_scores,
            validation_scores=validation_scores,
            output_dir=output_dir / "validation_curves",
            dataset_label=dataset_label,
            model_name=model_name,
            target_name=target_name,
            x_label=param_name,
            title=f"{dataset_label}: Validation Curve ({model_name})",
            filename=f"{sanitize_filename(target_name)}_{sanitize_filename(param_name)}_validation_curve",
            scoring=str(curve_config.get("scoring", "neg_root_mean_squared_error")),
        )

    def _prediction_vs_actual_single(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Path,
        dataset_label: str,
        model_name: str,
        split: str,
        target_name: str,
    ) -> Path:
        fig, ax = plt.subplots(
            figsize=tuple(self.config.get("prediction_vs_actual_figsize", (6.5, 5.5)))
        )
        sns.scatterplot(x=y_true, y=y_pred, ax=ax, s=42, edgecolor="white", linewidth=0.5)
        lower, upper = self._axis_limits(y_true, y_pred)
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1.2, linestyle="--")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{dataset_label} | {model_name} | {split}: {target_name}")
        return self._save(
            fig,
            output_dir
            / "prediction_vs_actual"
            / f"{sanitize_filename(split)}_{sanitize_filename(target_name)}_prediction_vs_actual",
        )

    def _residual_single(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_dir: Path,
        dataset_label: str,
        model_name: str,
        split: str,
        target_name: str,
    ) -> Path:
        residuals = y_true - y_pred
        fig, ax = plt.subplots(figsize=tuple(self.config.get("residual_figsize", (6.5, 5.0))))
        sns.scatterplot(x=y_pred, y=residuals, ax=ax, s=42, edgecolor="white", linewidth=0.5)
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (actual - predicted)")
        ax.set_title(f"{dataset_label} | {model_name} | {split}: {target_name}")
        return self._save(
            fig,
            output_dir
            / "residuals"
            / f"{sanitize_filename(split)}_{sanitize_filename(target_name)}_residuals",
        )

    def _score_curve_plot(
        self,
        *,
        x_values: np.ndarray,
        train_scores: np.ndarray,
        validation_scores: np.ndarray,
        output_dir: Path,
        dataset_label: str,
        model_name: str,
        target_name: str,
        x_label: str,
        title: str,
        filename: str,
        scoring: str,
    ) -> Path:
        train_values = self._display_scores(train_scores, scoring)
        validation_values = self._display_scores(validation_scores, scoring)
        train_mean = np.mean(train_values, axis=1)
        train_std = np.std(train_values, axis=1, ddof=1)
        validation_mean = np.mean(validation_values, axis=1)
        validation_std = np.std(validation_values, axis=1, ddof=1)

        figsize_key = "learning_curve_figsize" if "learning" in filename else "validation_curve_figsize"
        fig, ax = plt.subplots(figsize=tuple(self.config.get(figsize_key, (7.0, 5.0))))
        ax.plot(x_values, train_mean, marker="o", label="Train")
        ax.fill_between(x_values, train_mean - train_std, train_mean + train_std, alpha=0.18)
        ax.plot(x_values, validation_mean, marker="o", label="Validation")
        ax.fill_between(
            x_values,
            validation_mean - validation_std,
            validation_mean + validation_std,
            alpha=0.18,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(self._score_label(scoring))
        ax.set_title(f"{title}: {target_name}")
        ax.legend()
        return self._save(fig, output_dir / filename)

    @staticmethod
    def _display_scores(scores: np.ndarray, scoring: str) -> np.ndarray:
        if scoring.startswith("neg_"):
            return -scores
        return scores

    @staticmethod
    def _score_label(scoring: str) -> str:
        label = scoring.removeprefix("neg_").replace("_", " ").title()
        return label

    @staticmethod
    def _axis_limits(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        values = np.concatenate([y_true, y_pred])
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
        if np.isclose(lower, upper):
            padding = max(abs(lower) * 0.05, 1.0)
            return lower - padding, upper + padding
        padding = (upper - lower) * 0.05
        return lower - padding, upper + padding

    def _save(self, fig: plt.Figure, path_without_suffix: Path) -> Path:
        figure_format = self.config.get("figure_format", "png").lower()
        path = path_without_suffix.with_suffix(f".{figure_format}")
        ensure_directory(path.parent)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
