"""Score-table loading and normalization for statistical comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from friction_surrogate_xai.config.loader import project_root


CANONICAL_COLUMNS = (
    "comparison_source",
    "comparison_type",
    "dataset_key",
    "target_name",
    "model_key",
    "variant",
    "output_mode",
    "block_id",
    "metric",
    "score",
)


class ScoreTableLoader:
    """Load configured score tables and normalize them to one schema."""

    def __init__(self, inputs_config: dict[str, Any]) -> None:
        self.config = inputs_config

    def load(
        self,
        explicit_paths: tuple[str | Path, ...] | None = None,
    ) -> pd.DataFrame:
        """Load explicit and auto-discovered score tables."""
        frames: list[pd.DataFrame] = []
        paths = tuple(explicit_paths or ())
        paths += tuple(self.config.get("explicit_score_paths", ()))
        for path in paths:
            frames.append(self._load_explicit(Path(path)))

        if self.config.get("auto_discover", True):
            frames.extend(self._load_discretization_variant_scores())
            frames.extend(self._load_overfitting_fold_scores())

        if not frames:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)
        return pd.concat(frames, ignore_index=True, sort=False)

    def _load_explicit(self, path: Path) -> pd.DataFrame:
        resolved = self._resolve(path)
        frame = pd.read_csv(resolved)
        return normalize_score_table(frame, comparison_source=str(resolved))

    def _load_discretization_variant_scores(self) -> list[pd.DataFrame]:
        pattern = str(self.config.get("discretization_variant_pattern", ""))
        if not pattern:
            return []
        frames: list[pd.DataFrame] = []
        for path in project_root().glob(pattern):
            raw = pd.read_csv(path)
            if raw.empty:
                continue
            frame = pd.DataFrame(
                {
                    "comparison_source": str(path),
                    "comparison_type": "original_vs_discrete",
                    "dataset_key": raw.get("dataset_key", ""),
                    "target_name": raw.get("target_name", ""),
                    "model_key": raw.get("model_key", ""),
                    "variant": raw.get("variant", ""),
                    "output_mode": raw.get("output_mode", "single_output"),
                    "block_id": raw.apply(_variant_block_id, axis=1),
                    "metric": "objective_value",
                    "score": pd.to_numeric(raw.get("objective_value"), errors="coerce"),
                }
            )
            frames.append(frame)
        return frames

    def _load_overfitting_fold_scores(self) -> list[pd.DataFrame]:
        pattern = str(self.config.get("overfitting_fold_pattern", ""))
        if not pattern:
            return []
        frames: list[pd.DataFrame] = []
        for path in project_root().glob(pattern):
            raw = pd.read_csv(path)
            validation = raw.loc[raw.get("split", "") == "validation"].copy()
            if validation.empty or "r2" not in validation:
                continue
            frame = pd.DataFrame(
                {
                    "comparison_source": str(path),
                    "comparison_type": "top_models",
                    "dataset_key": validation.get("dataset_key", ""),
                    "target_name": validation.get("target", ""),
                    "model_key": validation.get("model_key", validation.get("model_name", "")),
                    "variant": validation.get("variant", "original"),
                    "output_mode": validation.get("output_mode", "single_output"),
                    "block_id": validation.apply(_fold_block_id, axis=1),
                    "metric": "r2",
                    "score": pd.to_numeric(validation["r2"], errors="coerce"),
                }
            )
            frames.append(frame)
        return frames

    @staticmethod
    def _resolve(path: Path) -> Path:
        return path if path.is_absolute() else project_root() / path


def normalize_score_table(
    frame: pd.DataFrame,
    *,
    comparison_source: str = "provided",
) -> pd.DataFrame:
    """Normalize a user-provided score table to canonical columns."""
    normalized = frame.copy()
    if "score" not in normalized:
        for candidate in ("objective_value", "mean_validation_r2", "r2", "value"):
            if candidate in normalized:
                normalized["score"] = normalized[candidate]
                break
    if "score" not in normalized:
        raise ValueError("Score table must contain `score` or a known metric column.")

    defaults = {
        "comparison_source": comparison_source,
        "comparison_type": "provided",
        "dataset_key": "dataset",
        "target_name": "target",
        "model_key": "model",
        "variant": "original",
        "output_mode": "single_output",
        "block_id": normalized.index.astype(str),
        "metric": "score",
    }
    for column, value in defaults.items():
        if column not in normalized:
            normalized[column] = value
    normalized["score"] = pd.to_numeric(normalized["score"], errors="coerce")
    return normalized.loc[:, list(CANONICAL_COLUMNS)].copy()


def _variant_block_id(row: pd.Series) -> str:
    return "|".join(
        str(row.get(column, ""))
        for column in ("model_key", "seed")
        if column in row
    )


def _fold_block_id(row: pd.Series) -> str:
    parts = [
        row.get("validation_strategy", ""),
        row.get("fold_id", ""),
        row.get("seed", ""),
        row.get("repeat_id", ""),
        row.get("target", ""),
    ]
    return "|".join(str(part) for part in parts)
