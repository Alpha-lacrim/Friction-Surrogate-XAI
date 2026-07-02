"""Outlier detection for EDA.

Outlier methods in this module are detect-only. They never remove, filter, or
mutate rows from the original dataframe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


@dataclass(frozen=True)
class OutlierReports:
    """Outlier report tables."""

    row_scores: pd.DataFrame
    iqr_outliers: pd.DataFrame
    summary: pd.DataFrame


class OutlierDetector:
    """Detect IQR, Isolation Forest, and Local Outlier Factor outliers."""

    def __init__(
        self,
        iqr_multiplier: float = 1.5,
        isolation_forest_config: dict[str, Any] | None = None,
        lof_config: dict[str, Any] | None = None,
    ) -> None:
        self.iqr_multiplier = iqr_multiplier
        self.isolation_forest_config = isolation_forest_config or {}
        self.lof_config = lof_config or {}

    def detect(
        self,
        dataframe: pd.DataFrame,
        columns: tuple[str, ...],
        id_column: str | None = None,
    ) -> OutlierReports:
        """Detect outliers in selected columns without modifying the input dataframe."""
        numeric_frame = dataframe.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        row_scores = pd.DataFrame({"row_index": dataframe.index.astype(int)})
        if id_column and id_column in dataframe.columns:
            row_scores[id_column] = dataframe[id_column].values

        iqr_flags, iqr_outliers = self._iqr(numeric_frame)
        row_scores["iqr_outlier_count"] = iqr_flags.sum(axis=1).astype(int).values
        row_scores["iqr_outlier_columns"] = [
            ";".join(iqr_flags.columns[row.to_numpy()].tolist()) for _, row in iqr_flags.iterrows()
        ]
        row_scores["iqr_is_outlier"] = row_scores["iqr_outlier_count"] > 0

        if self.isolation_forest_config.get("enabled", True):
            row_scores = self._isolation_forest(numeric_frame, row_scores)

        if self.lof_config.get("enabled", True):
            row_scores = self._local_outlier_factor(numeric_frame, row_scores)

        summary = self._summary(row_scores=row_scores, iqr_outliers=iqr_outliers)
        return OutlierReports(row_scores=row_scores, iqr_outliers=iqr_outliers, summary=summary)

    def _iqr(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        flags = pd.DataFrame(False, index=dataframe.index, columns=dataframe.columns)
        rows: list[dict[str, Any]] = []
        for column in dataframe.columns:
            series = dataframe[column]
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            column_flags = (series < lower) | (series > upper)
            flags[column] = column_flags.fillna(False)
            for index, value in series[column_flags].items():
                rows.append(
                    {
                        "row_index": int(index),
                        "column": column,
                        "value": value,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "method": "iqr",
                    }
                )
        return flags, pd.DataFrame(rows)

    def _isolation_forest(self, dataframe: pd.DataFrame, row_scores: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepared_frame(dataframe)
        if prepared.empty or len(prepared) < 2:
            row_scores["isolation_forest_is_outlier"] = False
            row_scores["isolation_forest_score"] = pd.NA
            return row_scores

        model = IsolationForest(
            contamination=self.isolation_forest_config.get("contamination", "auto"),
            random_state=int(self.isolation_forest_config.get("random_state", 42)),
        )
        labels = model.fit_predict(prepared)
        scores = model.decision_function(prepared)
        row_scores["isolation_forest_is_outlier"] = labels == -1
        row_scores["isolation_forest_score"] = scores
        return row_scores

    def _local_outlier_factor(self, dataframe: pd.DataFrame, row_scores: pd.DataFrame) -> pd.DataFrame:
        prepared = self._prepared_frame(dataframe)
        if prepared.empty or len(prepared) < 3:
            row_scores["lof_is_outlier"] = False
            row_scores["lof_score"] = pd.NA
            return row_scores

        configured_neighbors = int(self.lof_config.get("n_neighbors", 20))
        n_neighbors = max(1, min(configured_neighbors, len(prepared) - 1))
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.lof_config.get("contamination", "auto"),
        )
        labels = model.fit_predict(prepared)
        row_scores["lof_is_outlier"] = labels == -1
        row_scores["lof_score"] = model.negative_outlier_factor_
        row_scores["lof_n_neighbors"] = n_neighbors
        return row_scores

    @staticmethod
    def _prepared_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.fillna(dataframe.median(numeric_only=True)).fillna(0.0)

    @staticmethod
    def _summary(row_scores: pd.DataFrame, iqr_outliers: pd.DataFrame) -> pd.DataFrame:
        rows = [
            {
                "method": "iqr",
                "outlier_rows": int(row_scores["iqr_is_outlier"].sum()),
                "outlier_cells": int(len(iqr_outliers)),
                "policy": "detect_only_never_remove",
            }
        ]
        if "isolation_forest_is_outlier" in row_scores.columns:
            rows.append(
                {
                    "method": "isolation_forest",
                    "outlier_rows": int(row_scores["isolation_forest_is_outlier"].sum()),
                    "outlier_cells": None,
                    "policy": "detect_only_never_remove",
                }
            )
        if "lof_is_outlier" in row_scores.columns:
            rows.append(
                {
                    "method": "local_outlier_factor",
                    "outlier_rows": int(row_scores["lof_is_outlier"].sum()),
                    "outlier_cells": None,
                    "policy": "detect_only_never_remove",
                }
            )
        return pd.DataFrame(rows)

