"""Prepare fitted models and processed feature matrices for XAI reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from friction_surrogate_xai.models import ModelFactory
from friction_surrogate_xai.preprocessing import PreprocessingPipelineFactory


@dataclass(frozen=True)
class PreparedXAIModel:
    """Fitted model artifacts used by explainability analyzers."""

    dataset_key: str
    target_name: str
    model_key: str
    raw_features: pd.DataFrame
    processed_features: pd.DataFrame
    target: pd.Series
    preprocessor: Any
    estimator: Any


class XAIModelPreparer:
    """Fit preprocessing and model artifacts for post-hoc explanation."""

    def __init__(
        self,
        model_factory: ModelFactory | None = None,
        preprocessing_factory: PreprocessingPipelineFactory | None = None,
    ) -> None:
        self.model_factory = model_factory or ModelFactory()
        self.preprocessing_factory = preprocessing_factory or PreprocessingPipelineFactory()

    def prepare(
        self,
        *,
        dataset_key: str,
        model_key: str,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str,
        params_override: dict[str, Any] | None = None,
        random_state: int | None = None,
    ) -> PreparedXAIModel:
        """Fit preprocessing and estimator on the supplied explanation dataset."""
        preprocessor = self.preprocessing_factory.build_for_dataset(dataset_key)
        processed = preprocessor.fit_transform(X, y)
        if not isinstance(processed, pd.DataFrame):
            processed = pd.DataFrame(processed, index=X.index)

        estimator = self.model_factory.build(
            model_key,
            random_state=random_state,
            params_override=params_override,
        )
        estimator.fit(processed, y.to_numpy())
        return PreparedXAIModel(
            dataset_key=dataset_key,
            target_name=target_name,
            model_key=model_key,
            raw_features=X.copy(),
            processed_features=processed.copy(),
            target=y.copy(),
            preprocessor=preprocessor,
            estimator=estimator,
        )
