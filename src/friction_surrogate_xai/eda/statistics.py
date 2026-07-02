"""Statistical EDA computations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from friction_surrogate_xai.eda.utils import to_serializable


@dataclass(frozen=True)
class StatisticalTables:
    """Tabular statistical outputs for one dataset."""

    descriptive: pd.DataFrame
    confidence_intervals: pd.DataFrame
    normality_tests: pd.DataFrame
    correlations: dict[str, pd.DataFrame]


class StatisticalAnalyzer:
    """Compute descriptive, normality, confidence interval, and correlation reports."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        normality_alpha: float = 0.05,
        correlation_methods: tuple[str, ...] = ("pearson", "spearman", "kendall"),
    ) -> None:
        self.confidence_level = confidence_level
        self.normality_alpha = normality_alpha
        self.correlation_methods = correlation_methods

    def analyze(self, dataframe: pd.DataFrame, columns: tuple[str, ...]) -> StatisticalTables:
        """Compute all statistical tables for selected numeric columns."""
        numeric_frame = dataframe.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        return StatisticalTables(
            descriptive=self._descriptive_statistics(numeric_frame),
            confidence_intervals=self._confidence_intervals(numeric_frame),
            normality_tests=self._normality_tests(numeric_frame),
            correlations={
                method: numeric_frame.corr(method=method) for method in self.correlation_methods
            },
        )

    def _descriptive_statistics(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for column in dataframe.columns:
            series = dataframe[column].dropna()
            mode = series.mode(dropna=False)
            rows.append(
                {
                    "column": column,
                    "count": int(series.count()),
                    "mean": to_serializable(series.mean()),
                    "median": to_serializable(series.median()),
                    "mode": to_serializable(mode.iloc[0]) if not mode.empty else None,
                    "variance": to_serializable(series.var(ddof=1)),
                    "std": to_serializable(series.std(ddof=1)),
                    "min": to_serializable(series.min()),
                    "q1": to_serializable(series.quantile(0.25)),
                    "q3": to_serializable(series.quantile(0.75)),
                    "max": to_serializable(series.max()),
                    "skewness": to_serializable(series.skew()),
                    "kurtosis": to_serializable(series.kurt()),
                }
            )
        return pd.DataFrame(rows)

    def _confidence_intervals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        alpha = 1.0 - self.confidence_level
        for column in dataframe.columns:
            series = dataframe[column].dropna()
            count = int(series.count())
            mean = series.mean()
            std = series.std(ddof=1)
            if count < 2 or np.isclose(std, 0):
                margin = 0.0 if count > 0 else np.nan
                lower = mean if count > 0 else np.nan
                upper = mean if count > 0 else np.nan
            else:
                standard_error = stats.sem(series)
                critical = stats.t.ppf(1 - alpha / 2, df=count - 1)
                margin = critical * standard_error
                lower = mean - margin
                upper = mean + margin
            rows.append(
                {
                    "column": column,
                    "confidence_level": self.confidence_level,
                    "count": count,
                    "mean": to_serializable(mean),
                    "lower": to_serializable(lower),
                    "upper": to_serializable(upper),
                    "margin_of_error": to_serializable(margin),
                }
            )
        return pd.DataFrame(rows)

    def _normality_tests(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for column in dataframe.columns:
            series = dataframe[column].dropna()
            rows.extend(self._shapiro(column, series))
            rows.extend(self._anderson(column, series))
            rows.extend(self._kolmogorov_smirnov(column, series))
        return pd.DataFrame(rows)

    def _shapiro(self, column: str, series: pd.Series) -> list[dict[str, Any]]:
        if self._should_skip_normality(series):
            return [self._skipped_normality_row(column, "shapiro_wilk")]
        statistic, p_value = stats.shapiro(series)
        return [
            {
                "column": column,
                "test": "shapiro_wilk",
                "statistic": to_serializable(statistic),
                "p_value": to_serializable(p_value),
                "alpha": self.normality_alpha,
                "reject_normality": bool(p_value < self.normality_alpha),
                "status": "ok",
                "details": "",
            }
        ]

    def _anderson(self, column: str, series: pd.Series) -> list[dict[str, Any]]:
        if self._should_skip_normality(series):
            return [self._skipped_normality_row(column, "anderson_darling")]
        result = stats.anderson(series, dist="norm")
        significance_level = self.normality_alpha * 100
        index = int(np.argmin(np.abs(result.significance_level - significance_level)))
        critical_value = result.critical_values[index]
        return [
            {
                "column": column,
                "test": "anderson_darling",
                "statistic": to_serializable(result.statistic),
                "p_value": None,
                "alpha": self.normality_alpha,
                "reject_normality": bool(result.statistic > critical_value),
                "status": "ok",
                "details": json.dumps(
                    {
                        "critical_values": [to_serializable(v) for v in result.critical_values],
                        "significance_levels": [
                            to_serializable(v) for v in result.significance_level
                        ],
                        "selected_critical_value": to_serializable(critical_value),
                    }
                ),
            }
        ]

    def _kolmogorov_smirnov(self, column: str, series: pd.Series) -> list[dict[str, Any]]:
        if self._should_skip_normality(series):
            return [self._skipped_normality_row(column, "kolmogorov_smirnov")]
        standardized = (series - series.mean()) / series.std(ddof=1)
        statistic, p_value = stats.kstest(standardized, "norm")
        return [
            {
                "column": column,
                "test": "kolmogorov_smirnov",
                "statistic": to_serializable(statistic),
                "p_value": to_serializable(p_value),
                "alpha": self.normality_alpha,
                "reject_normality": bool(p_value < self.normality_alpha),
                "status": "ok",
                "details": "sample standardized with sample mean and standard deviation",
            }
        ]

    @staticmethod
    def _should_skip_normality(series: pd.Series) -> bool:
        return int(series.count()) < 3 or np.isclose(series.std(ddof=1), 0)

    def _skipped_normality_row(self, column: str, test: str) -> dict[str, Any]:
        return {
            "column": column,
            "test": test,
            "statistic": None,
            "p_value": None,
            "alpha": self.normality_alpha,
            "reject_normality": None,
            "status": "skipped_constant_or_too_few_values",
            "details": "normality test requires at least three non-constant values",
        }

