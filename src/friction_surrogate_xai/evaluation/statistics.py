"""Statistical helpers for model evaluation tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ConfidenceInterval:
    """Mean confidence interval summary."""

    count: int
    mean: float
    std: float
    confidence_level: float
    lower: float
    upper: float
    margin_of_error: float

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation for DataFrame construction."""
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "confidence_level": self.confidence_level,
            "ci_lower": self.lower,
            "ci_upper": self.upper,
            "ci_margin": self.margin_of_error,
        }


def confidence_interval(
    values: Any,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Compute a t-based confidence interval for finite values."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    count = int(array.size)
    if count == 0:
        return ConfidenceInterval(
            count=0,
            mean=np.nan,
            std=np.nan,
            confidence_level=confidence_level,
            lower=np.nan,
            upper=np.nan,
            margin_of_error=np.nan,
        )

    mean = float(np.mean(array))
    if count == 1:
        return ConfidenceInterval(
            count=count,
            mean=mean,
            std=0.0,
            confidence_level=confidence_level,
            lower=mean,
            upper=mean,
            margin_of_error=0.0,
        )

    std = float(np.std(array, ddof=1))
    if np.isclose(std, 0.0):
        return ConfidenceInterval(
            count=count,
            mean=mean,
            std=std,
            confidence_level=confidence_level,
            lower=mean,
            upper=mean,
            margin_of_error=0.0,
        )

    alpha = 1.0 - confidence_level
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=count - 1))
    margin = critical * std / float(np.sqrt(count))
    return ConfidenceInterval(
        count=count,
        mean=mean,
        std=std,
        confidence_level=confidence_level,
        lower=mean - margin,
        upper=mean + margin,
        margin_of_error=margin,
    )
