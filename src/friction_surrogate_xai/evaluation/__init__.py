"""Reusable model evaluation framework."""

from friction_surrogate_xai.evaluation.config import (
    EvaluationConfig,
    load_evaluation_config,
)
from friction_surrogate_xai.evaluation.metrics import (
    REGRESSION_METRICS,
    RegressionMetricCalculator,
)
from friction_surrogate_xai.evaluation.plots import EvaluationPlotter
from friction_surrogate_xai.evaluation.reports import EvaluationReportWriter
from friction_surrogate_xai.evaluation.runner import (
    EvaluationArtifacts,
    EvaluationReportGenerator,
)
from friction_surrogate_xai.evaluation.statistics import ConfidenceInterval, confidence_interval

__all__ = [
    "ConfidenceInterval",
    "EvaluationArtifacts",
    "EvaluationConfig",
    "EvaluationPlotter",
    "EvaluationReportGenerator",
    "EvaluationReportWriter",
    "REGRESSION_METRICS",
    "RegressionMetricCalculator",
    "confidence_interval",
    "load_evaluation_config",
]
