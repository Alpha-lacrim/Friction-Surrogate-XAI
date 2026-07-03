"""Statistical comparison framework."""

from friction_surrogate_xai.statistical_comparison.config import (
    StatisticalComparisonConfig,
    load_statistical_comparison_config,
    with_overrides,
)
from friction_surrogate_xai.statistical_comparison.data import (
    ScoreTableLoader,
    normalize_score_table,
)
from friction_surrogate_xai.statistical_comparison.runner import (
    StatisticalComparisonArtifacts,
    StatisticalComparisonRunner,
)
from friction_surrogate_xai.statistical_comparison.tests import (
    StatisticalComparator,
    StatisticalTestResult,
)

__all__ = [
    "ScoreTableLoader",
    "StatisticalComparator",
    "StatisticalComparisonArtifacts",
    "StatisticalComparisonConfig",
    "StatisticalComparisonRunner",
    "StatisticalTestResult",
    "load_statistical_comparison_config",
    "normalize_score_table",
    "with_overrides",
]
