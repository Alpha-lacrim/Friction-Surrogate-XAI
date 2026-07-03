"""SHAP explainability reports."""

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

from friction_surrogate_xai.eda.utils import ensure_directory, sanitize_filename, write_csv
from friction_surrogate_xai.xai.plotting import XAIPlotStyle
from friction_surrogate_xai.xai.preparation import PreparedXAIModel


@dataclass(frozen=True)
class SHAPArtifacts:
    """Generated SHAP artifacts."""

    global_importance: pd.DataFrame
    local_values: pd.DataFrame
    effect_summary: pd.DataFrame
    interaction_summary: pd.DataFrame
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]
    shap_values: np.ndarray


class SHAPAnalyzer:
    """Generate SHAP global/local explanations and plots."""

    def __init__(self, shap_config: dict[str, Any], plot_config: dict[str, Any]) -> None:
        self.config = shap_config
        self.plot_style = XAIPlotStyle(plot_config)

    def analyze(
        self,
        *,
        prepared: PreparedXAIModel,
        tables_dir: Path,
        figures_dir: Path,
    ) -> SHAPArtifacts:
        """Run SHAP analysis for a prepared model."""
        if not self.config.get("enabled", True):
            return SHAPArtifacts(
                global_importance=pd.DataFrame(),
                local_values=pd.DataFrame(),
                effect_summary=pd.DataFrame(),
                interaction_summary=pd.DataFrame(),
                table_paths=(),
                figure_paths=(),
                shap_values=np.empty((0, 0)),
            )

        import shap

        X = prepared.processed_features
        background_size = min(len(X), int(self.config.get("background_sample_size", len(X))))
        if background_size and background_size < len(X):
            background = X.sample(background_size, random_state=42)
        else:
            background = X

        explainer = self._make_explainer(shap, prepared.estimator, background)
        explanation = explainer(X)
        shap_values = _extract_shap_values(explanation.values)
        base_values = _extract_base_values(explanation.base_values, len(X))

        global_importance = self._global_importance(X, shap_values)
        local_values = self._local_values(X, shap_values, base_values)
        effect_summary = self._effect_summary(X, shap_values)
        interaction_summary = self._interaction_summary(X, shap_values)

        shap_dir = ensure_directory(tables_dir / "shap")
        table_paths = (
            write_csv(global_importance, shap_dir / "global_shap_importance.csv"),
            write_csv(local_values, shap_dir / "local_shap_values.csv"),
            write_csv(effect_summary, shap_dir / "shap_effect_summary.csv"),
            write_csv(interaction_summary, shap_dir / "shap_interaction_summary.csv"),
        )
        figure_paths = self._plots(
            shap=shap,
            explanation=explanation,
            X=X,
            shap_values=shap_values,
            global_importance=global_importance,
            interaction_summary=interaction_summary,
            figures_dir=figures_dir / "shap",
        )
        return SHAPArtifacts(
            global_importance=global_importance,
            local_values=local_values,
            effect_summary=effect_summary,
            interaction_summary=interaction_summary,
            table_paths=table_paths,
            figure_paths=figure_paths,
            shap_values=shap_values,
        )

    def _make_explainer(self, shap: Any, estimator: Any, background: pd.DataFrame) -> Any:
        explainer_kind = str(self.config.get("explainer", "auto"))
        if explainer_kind == "tree" or (
            explainer_kind == "auto" and hasattr(estimator, "feature_importances_")
        ):
            try:
                return shap.TreeExplainer(estimator)
            except Exception:
                pass
        if explainer_kind == "linear":
            try:
                return shap.LinearExplainer(estimator, background)
            except Exception:
                pass
        return shap.Explainer(estimator.predict, background)

    def _global_importance(self, X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_signed = shap_values.mean(axis=0)
        return (
            pd.DataFrame(
                {
                    "feature": X.columns,
                    "mean_abs_shap": mean_abs,
                    "mean_signed_shap": mean_signed,
                    "rank": pd.Series(mean_abs).rank(ascending=False, method="first").astype(int),
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def _local_values(
        X: pd.DataFrame,
        shap_values: np.ndarray,
        base_values: np.ndarray,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row_index, sample_id in enumerate(X.index):
            for feature_index, feature in enumerate(X.columns):
                rows.append(
                    {
                        "sample_index": sample_id,
                        "feature": feature,
                        "feature_value": X.iloc[row_index, feature_index],
                        "shap_value": shap_values[row_index, feature_index],
                        "base_value": base_values[row_index],
                    }
                )
        return pd.DataFrame(rows)

    def _effect_summary(self, X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for feature_index, feature in enumerate(X.columns):
            values = pd.to_numeric(X.iloc[:, feature_index], errors="coerce")
            effects = shap_values[:, feature_index]
            corr = _safe_corr(values.to_numpy(dtype=float), effects)
            low_mask = values <= values.median()
            high_mask = values > values.median()
            low_effect = float(np.nanmean(effects[low_mask])) if low_mask.any() else np.nan
            high_effect = float(np.nanmean(effects[high_mask])) if high_mask.any() else np.nan
            rows.append(
                {
                    "feature": feature,
                    "mean_shap": float(np.nanmean(effects)),
                    "mean_abs_shap": float(np.nanmean(np.abs(effects))),
                    "correlation_feature_value_shap": corr,
                    "low_value_mean_shap": low_effect,
                    "high_value_mean_shap": high_effect,
                    "direction": _direction_from_effects(corr, high_effect - low_effect),
                    "nonlinear_signal": _nonlinear_signal(values, effects),
                }
            )
        return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    @staticmethod
    def _interaction_summary(X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for i, feature_a in enumerate(X.columns):
            for j, feature_b in enumerate(X.columns):
                if i >= j:
                    continue
                effect_product = np.abs(shap_values[:, i] * shap_values[:, j])
                value_corr = _safe_corr(
                    pd.to_numeric(X.iloc[:, i], errors="coerce").to_numpy(dtype=float),
                    shap_values[:, j],
                )
                rows.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "mean_abs_shap_product": float(np.nanmean(effect_product)),
                        "feature_a_value_vs_feature_b_shap_corr": value_corr,
                    }
                )
        return (
            pd.DataFrame(rows)
            .sort_values("mean_abs_shap_product", ascending=False)
            .reset_index(drop=True)
            if rows
            else pd.DataFrame()
        )

    def _plots(
        self,
        *,
        shap: Any,
        explanation: Any,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        global_importance: pd.DataFrame,
        interaction_summary: pd.DataFrame,
        figures_dir: Path,
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        max_display = int(self.config.get("max_display", 12))

        if self.config.get("beeswarm", True):
            try:
                shap.plots.beeswarm(explanation, max_display=max_display, show=False)
                paths.append(self.plot_style.save(plt.gcf(), figures_dir / "shap_beeswarm"))
            except Exception:
                paths.append(self._fallback_summary_plot(X, shap_values, figures_dir / "shap_beeswarm"))

        if self.config.get("summary_plot", True):
            try:
                shap.summary_plot(shap_values, X, max_display=max_display, show=False)
                paths.append(self.plot_style.save(plt.gcf(), figures_dir / "shap_summary"))
            except Exception:
                paths.append(self._fallback_summary_plot(X, shap_values, figures_dir / "shap_summary"))

        if self.config.get("waterfall", True):
            local_count = min(int(self.config.get("waterfall_local_samples", 3)), len(X))
            for sample_position in range(local_count):
                try:
                    shap.plots.waterfall(explanation[sample_position], max_display=max_display, show=False)
                    paths.append(
                        self.plot_style.save(
                            plt.gcf(),
                            figures_dir / "waterfall" / f"sample_{sample_position}_waterfall",
                        )
                    )
                except Exception:
                    paths.append(
                        self._fallback_waterfall(
                            X,
                            shap_values,
                            sample_position,
                            figures_dir / "waterfall" / f"sample_{sample_position}_waterfall",
                        )
                    )

        if self.config.get("dependence_plot", True):
            for feature in global_importance["feature"].head(int(self.config.get("dependence_top_n", 3))):
                paths.append(self._dependence_plot(X, shap_values, feature, figures_dir / "dependence"))

        if self.config.get("interaction_plot", True) and not interaction_summary.empty:
            top_pairs = interaction_summary.head(int(self.config.get("interaction_top_n", 2)))
            for _, row in top_pairs.iterrows():
                paths.append(
                    self._interaction_plot(
                        X,
                        shap_values,
                        str(row["feature_a"]),
                        str(row["feature_b"]),
                        figures_dir / "interactions",
                    )
                )
        return tuple(paths)

    def _fallback_summary_plot(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        path_without_suffix: Path,
    ) -> Path:
        importance = pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=importance, x="mean_abs_shap", y="feature", ax=ax)
        ax.set_title("Global SHAP Importance")
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_ylabel("Feature")
        return self.plot_style.save(fig, path_without_suffix)

    def _fallback_waterfall(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        sample_position: int,
        path_without_suffix: Path,
    ) -> Path:
        row = pd.DataFrame(
            {
                "feature": X.columns,
                "shap_value": shap_values[sample_position],
            }
        ).sort_values("shap_value", key=lambda s: s.abs(), ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=row.head(12), x="shap_value", y="feature", ax=ax)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title(f"Local SHAP Contributions: sample {sample_position}")
        return self.plot_style.save(fig, path_without_suffix)

    def _dependence_plot(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        feature: str,
        output_dir: Path,
    ) -> Path:
        feature_index = list(X.columns).index(feature)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        sns.scatterplot(
            x=pd.to_numeric(X[feature], errors="coerce"),
            y=shap_values[:, feature_index],
            ax=ax,
            s=40,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(feature)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP Dependence: {feature}")
        return self.plot_style.save(fig, output_dir / f"{sanitize_filename(feature)}_dependence")

    def _interaction_plot(
        self,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        feature_a: str,
        feature_b: str,
        output_dir: Path,
    ) -> Path:
        feature_index = list(X.columns).index(feature_a)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        scatter = ax.scatter(
            pd.to_numeric(X[feature_a], errors="coerce"),
            shap_values[:, feature_index],
            c=pd.to_numeric(X[feature_b], errors="coerce"),
            cmap="viridis",
            edgecolors="white",
            linewidths=0.4,
        )
        fig.colorbar(scatter, ax=ax, label=feature_b)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel(feature_a)
        ax.set_ylabel(f"SHAP value for {feature_a}")
        ax.set_title(f"SHAP Interaction View: {feature_a} colored by {feature_b}")
        return self.plot_style.save(
            fig,
            output_dir / f"{sanitize_filename(feature_a)}_x_{sanitize_filename(feature_b)}",
        )


def _extract_shap_values(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array.astype(float)


def _extract_base_values(values: Any, n_samples: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        return np.full(n_samples, float(array))
    if array.ndim == 2:
        array = array[:, 0]
    if len(array) != n_samples:
        return np.full(n_samples, float(np.ravel(array)[0]))
    return array.astype(float)


def _safe_corr(values: np.ndarray, effects: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(effects)
    if mask.sum() < 3:
        return np.nan
    if np.isclose(np.nanstd(values[mask]), 0) or np.isclose(np.nanstd(effects[mask]), 0):
        return np.nan
    return float(np.corrcoef(values[mask], effects[mask])[0, 1])


def _direction_from_effects(correlation: float, high_minus_low: float) -> str:
    signal = correlation if np.isfinite(correlation) else high_minus_low
    if not np.isfinite(signal) or np.isclose(signal, 0.0):
        return "mixed_or_flat"
    return "positive" if signal > 0 else "negative"


def _nonlinear_signal(values: pd.Series, effects: np.ndarray) -> str:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.nunique(dropna=True) < 4:
        return "low_cardinality_or_constant"
    try:
        bins = pd.qcut(numeric, q=min(4, numeric.nunique()), duplicates="drop")
        grouped = pd.Series(effects, index=values.index).groupby(bins, observed=False).mean()
    except ValueError:
        return "insufficient_unique_values"
    signs = np.sign(grouped.dropna())
    if len(set(signs)) > 1:
        return "sign_changes_across_feature_range"
    first_diff = np.diff(grouped.to_numpy(dtype=float))
    if len(first_diff) >= 2 and len(set(np.sign(first_diff[np.isfinite(first_diff)]))) > 1:
        return "non_monotonic_effect"
    return "approximately_monotonic"
