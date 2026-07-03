"""Discrete-input dataset generation and comparison workflows."""

from friction_surrogate_xai.discretization.config import (
    DiscretizationConfig,
    load_discretization_config,
)
from friction_surrogate_xai.discretization.workflow import (
    DiscretizationWorkflow,
    DiscretizationWorkflowArtifacts,
)

__all__ = [
    "DiscretizationConfig",
    "DiscretizationWorkflow",
    "DiscretizationWorkflowArtifacts",
    "load_discretization_config",
]
