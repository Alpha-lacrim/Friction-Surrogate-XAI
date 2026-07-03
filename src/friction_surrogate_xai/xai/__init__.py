"""Explainability framework for surrogate model interpretation."""

from friction_surrogate_xai.xai.config import XAIConfig, load_xai_config, with_overrides
from friction_surrogate_xai.xai.importance import ImportanceAnalyzer, ImportanceArtifacts
from friction_surrogate_xai.xai.interpretation import (
    ScientificInterpretation,
    ScientificInterpreter,
)
from friction_surrogate_xai.xai.lime_analysis import LIMEAnalyzer, LIMEArtifacts
from friction_surrogate_xai.xai.preparation import PreparedXAIModel, XAIModelPreparer
from friction_surrogate_xai.xai.runner import XAIArtifacts, XAIReportGenerator
from friction_surrogate_xai.xai.shap_analysis import SHAPAnalyzer, SHAPArtifacts

__all__ = [
    "ImportanceAnalyzer",
    "ImportanceArtifacts",
    "LIMEAnalyzer",
    "LIMEArtifacts",
    "PreparedXAIModel",
    "ScientificInterpretation",
    "ScientificInterpreter",
    "SHAPAnalyzer",
    "SHAPArtifacts",
    "XAIArtifacts",
    "XAIConfig",
    "XAIModelPreparer",
    "XAIReportGenerator",
    "load_xai_config",
    "with_overrides",
]
