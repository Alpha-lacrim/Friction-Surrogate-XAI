"""Scientific interpretation generation for XAI reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScientificInterpretation:
    """Structured scientific interpretation text and tables."""

    feature_interpretations: pd.DataFrame
    interaction_interpretations: pd.DataFrame
    markdown_sections: dict[str, str]


class ScientificInterpreter:
    """Translate XAI tables into scientific interpretation statements."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.top_n = int(config.get("top_n_features", 5))
        self.engineering_notes = dict(config.get("engineering_notes", {}))

    def interpret(
        self,
        *,
        shap_global: pd.DataFrame,
        shap_effects: pd.DataFrame,
        shap_interactions: pd.DataFrame,
        permutation: pd.DataFrame,
        tree_importance: pd.DataFrame,
        target_name: str,
    ) -> ScientificInterpretation:
        """Create scientific interpretation tables and Markdown sections."""
        feature_table = self._feature_table(
            shap_global=shap_global,
            shap_effects=shap_effects,
            permutation=permutation,
            tree_importance=tree_importance,
            target_name=target_name,
        )
        interaction_table = self._interaction_table(shap_interactions)
        sections = {
            "most_important_variables": self._important_variables_text(feature_table),
            "positive_negative_effects": self._effect_direction_text(feature_table, target_name),
            "nonlinear_behavior": self._nonlinear_text(feature_table),
            "feature_interactions": self._interaction_text(interaction_table),
            "engineering_interpretation": self._engineering_text(feature_table, target_name),
        }
        return ScientificInterpretation(
            feature_interpretations=feature_table,
            interaction_interpretations=interaction_table,
            markdown_sections=sections,
        )

    def _feature_table(
        self,
        *,
        shap_global: pd.DataFrame,
        shap_effects: pd.DataFrame,
        permutation: pd.DataFrame,
        tree_importance: pd.DataFrame,
        target_name: str,
    ) -> pd.DataFrame:
        if not shap_global.empty:
            table = shap_global.loc[:, ["feature", "mean_abs_shap", "mean_signed_shap"]].copy()
        elif not permutation.empty:
            table = permutation.loc[:, ["feature"]].copy()
            table["mean_abs_shap"] = np.nan
            table["mean_signed_shap"] = np.nan
        else:
            table = pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_signed_shap"])

        if not shap_effects.empty:
            table = table.merge(
                shap_effects[
                    [
                        "feature",
                        "correlation_feature_value_shap",
                        "direction",
                        "nonlinear_signal",
                        "low_value_mean_shap",
                        "high_value_mean_shap",
                    ]
                ],
                on="feature",
                how="left",
            )
        if not permutation.empty and "importance_mean" in permutation:
            table = table.merge(
                permutation[["feature", "importance_mean", "importance_std"]],
                on="feature",
                how="left",
            )
        if not tree_importance.empty and "importance" in tree_importance:
            table = table.merge(
                tree_importance[["feature", "importance"]].rename(
                    columns={"importance": "tree_importance"}
                ),
                on="feature",
                how="left",
            )
        table["engineering_note"] = table["feature"].map(self._engineering_note)
        table["interpretation"] = table.apply(
            lambda row: self._feature_sentence(row, target_name),
            axis=1,
        )
        return table.head(self.top_n).reset_index(drop=True)

    @staticmethod
    def _interaction_table(shap_interactions: pd.DataFrame) -> pd.DataFrame:
        if shap_interactions.empty:
            return pd.DataFrame(columns=["feature_a", "feature_b", "interaction_strength", "interpretation"])
        table = shap_interactions.head(5).copy()
        table = table.rename(columns={"mean_abs_shap_product": "interaction_strength"})
        table["interpretation"] = table.apply(
            lambda row: (
                f"`{row['feature_a']}` and `{row['feature_b']}` show a possible joint effect. "
                "This should be interpreted as a screening signal and checked against process physics."
            ),
            axis=1,
        )
        return table

    def _engineering_note(self, feature: str) -> str:
        for key, note in self.engineering_notes.items():
            if feature == key or feature.startswith(f"{key}_"):
                return note
        return "No domain-specific note is configured for this transformed feature."

    @staticmethod
    def _feature_sentence(row: pd.Series, target_name: str) -> str:
        direction = row.get("direction", "mixed_or_flat")
        nonlinear = row.get("nonlinear_signal", "unknown")
        if direction == "positive":
            effect = f"higher values tend to increase predicted `{target_name}`."
        elif direction == "negative":
            effect = f"higher values tend to decrease predicted `{target_name}`."
        else:
            effect = f"the effect on predicted `{target_name}` is mixed or weak."
        return f"{effect} Nonlinear signal: `{nonlinear}`."

    @staticmethod
    def _important_variables_text(table: pd.DataFrame) -> str:
        if table.empty:
            return "No feature-importance table was generated."
        feature_list = ", ".join(f"`{feature}`" for feature in table["feature"].head(5))
        return f"The highest-priority variables in this explanation are {feature_list}."

    @staticmethod
    def _effect_direction_text(table: pd.DataFrame, target_name: str) -> str:
        if table.empty or "direction" not in table:
            return "Directional effects could not be summarized."
        positives = table.loc[table["direction"] == "positive", "feature"].tolist()
        negatives = table.loc[table["direction"] == "negative", "feature"].tolist()
        parts: list[str] = []
        if positives:
            parts.append(
                "Positive effects: "
                + ", ".join(f"`{feature}`" for feature in positives)
                + f" tend to increase predicted `{target_name}`."
            )
        if negatives:
            parts.append(
                "Negative effects: "
                + ", ".join(f"`{feature}`" for feature in negatives)
                + f" tend to decrease predicted `{target_name}`."
            )
        return " ".join(parts) if parts else "Top effects are mixed, flat, or model-dependent."

    @staticmethod
    def _nonlinear_text(table: pd.DataFrame) -> str:
        if table.empty or "nonlinear_signal" not in table:
            return "Nonlinear behavior could not be summarized."
        nonlinear = table.loc[
            table["nonlinear_signal"].isin(
                ["sign_changes_across_feature_range", "non_monotonic_effect"]
            )
        ]
        if nonlinear.empty:
            return "The leading effects look approximately monotonic or low-cardinality in this run."
        return (
            "Potential nonlinear behavior appears for "
            + ", ".join(f"`{feature}`" for feature in nonlinear["feature"])
            + "."
        )

    @staticmethod
    def _interaction_text(table: pd.DataFrame) -> str:
        if table.empty:
            return "No strong pairwise interaction screening signal was generated."
        row = table.iloc[0]
        return (
            f"The strongest screened interaction is `{row['feature_a']}` with `{row['feature_b']}`. "
            "Treat this as a hypothesis for later mechanical-engineering validation."
        )

    @staticmethod
    def _engineering_text(table: pd.DataFrame, target_name: str) -> str:
        if table.empty:
            return "No engineering interpretation could be generated."
        sentences = []
        for _, row in table.iterrows():
            sentences.append(
                f"- `{row['feature']}`: {row['engineering_note']} "
                f"Observed model effect for `{target_name}`: {row['interpretation']}"
            )
        return "\n".join(sentences)
