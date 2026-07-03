"""LIME local explanation reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from friction_surrogate_xai.eda.utils import ensure_directory, write_csv
from friction_surrogate_xai.xai.plotting import XAIPlotStyle
from friction_surrogate_xai.xai.preparation import PreparedXAIModel


@dataclass(frozen=True)
class LIMEArtifacts:
    """Generated LIME artifacts."""

    local_explanations: pd.DataFrame
    table_paths: tuple[Path, ...]
    figure_paths: tuple[Path, ...]


class LIMEAnalyzer:
    """Generate local LIME explanations."""

    def __init__(self, lime_config: dict[str, Any], plot_config: dict[str, Any]) -> None:
        self.config = lime_config
        self.plot_style = XAIPlotStyle(plot_config)

    def analyze(
        self,
        *,
        prepared: PreparedXAIModel,
        tables_dir: Path,
        figures_dir: Path,
    ) -> LIMEArtifacts:
        """Run LIME local explanations."""
        if not self.config.get("enabled", True):
            return LIMEArtifacts(pd.DataFrame(), (), ())

        from lime.lime_tabular import LimeTabularExplainer

        X = prepared.processed_features
        explainer = LimeTabularExplainer(
            training_data=X.to_numpy(),
            feature_names=list(X.columns),
            mode="regression",
            discretize_continuous=bool(self.config.get("discretize_continuous", True)),
            random_state=int(self.config.get("random_state", 42)),
        )
        sample_count = min(int(self.config.get("max_local_samples", 3)), len(X))
        num_features = min(int(self.config.get("num_features", 8)), X.shape[1])
        rows: list[dict[str, Any]] = []
        figure_paths: list[Path] = []
        for sample_position in range(sample_count):
            explanation = explainer.explain_instance(
                X.iloc[sample_position].to_numpy(),
                prepared.estimator.predict,
                num_features=num_features,
            )
            for feature_rule, contribution in explanation.as_list():
                rows.append(
                    {
                        "sample_position": sample_position,
                        "feature_rule": feature_rule,
                        "lime_contribution": contribution,
                    }
                )
            figure = explanation.as_pyplot_figure()
            figure_paths.append(
                self.plot_style.save(
                    figure,
                    figures_dir / "lime" / f"sample_{sample_position}_lime",
                )
            )
        table = pd.DataFrame(rows)
        table_path = write_csv(table, ensure_directory(tables_dir / "lime") / "lime_local_explanations.csv")
        return LIMEArtifacts(
            local_explanations=table,
            table_paths=(table_path,),
            figure_paths=tuple(figure_paths),
        )
