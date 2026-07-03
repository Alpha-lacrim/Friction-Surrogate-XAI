"""Configurable model registry with conservative anti-overfitting defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from friction_surrogate_xai.models.config import ModelingConfig, load_modeling_config


@dataclass(frozen=True)
class ModelSpec:
    """Model metadata and construction policy."""

    key: str
    display_name: str
    family: str
    params: dict[str, Any]
    random_state_param: str | None
    overfitting_controls: tuple[str, ...]
    early_stopping: dict[str, Any]
    enabled: bool = True


class ModelFactory:
    """Build sklearn-compatible regressors from the configured model registry."""

    def __init__(self, config: ModelingConfig | None = None) -> None:
        self.config = config or load_modeling_config()

    def all_model_keys(self) -> tuple[str, ...]:
        """Return all configured model keys."""
        return tuple(self.config.models.keys())

    def enabled_model_keys(self) -> tuple[str, ...]:
        """Return enabled model keys in registry order."""
        return tuple(
            model_key
            for model_key, raw_spec in self.config.models.items()
            if bool(raw_spec.get("enabled", True))
        )

    def spec(self, model_key: str) -> ModelSpec:
        """Return a normalized model specification."""
        if model_key not in self.config.models:
            raise KeyError(f"Unknown model key: {model_key}")
        raw_spec = self.config.models[model_key]
        return ModelSpec(
            key=model_key,
            display_name=str(raw_spec.get("display_name", model_key)),
            family=str(raw_spec.get("family", "unknown")),
            params=dict(raw_spec.get("params", {})),
            random_state_param=raw_spec.get("random_state_param"),
            overfitting_controls=tuple(raw_spec.get("overfitting_controls", ())),
            early_stopping=dict(raw_spec.get("early_stopping", {})),
            enabled=bool(raw_spec.get("enabled", True)),
        )

    def build(
        self,
        model_key: str,
        random_state: int | None = None,
        params_override: dict[str, Any] | None = None,
    ) -> BaseEstimator:
        """Build an unfitted estimator with configured anti-overfitting controls."""
        spec = self.spec(model_key)
        params = self._params_with_seed(spec, random_state, params_override=params_override)

        if model_key == "linear_regression":
            return LinearRegression(**params)
        if model_key == "ridge":
            return Ridge(**params)
        if model_key == "elasticnet":
            return ElasticNet(**params)
        if model_key == "svr_rbf":
            return SVR(**params)
        if model_key == "gaussian_process_regression":
            return GaussianProcessRegressor(
                kernel=self._gpr_kernel(model_key),
                **params,
            )
        if model_key == "random_forest":
            return RandomForestRegressor(**params)
        if model_key == "extra_trees":
            return ExtraTreesRegressor(**params)
        if model_key == "gradient_boosting":
            return GradientBoostingRegressor(**params)
        if model_key == "xgboost":
            return self._xgboost_regressor(params)
        if model_key == "lightgbm":
            return self._lightgbm_regressor(params)
        if model_key == "shallow_mlp_regressor":
            return MLPRegressor(**params)
        raise KeyError(f"No estimator builder is registered for model key: {model_key}")

    def policy_table(self, model_keys: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Return a table documenting each model's anti-overfitting controls."""
        selected_keys = model_keys or self.enabled_model_keys()
        rows: list[dict[str, Any]] = []
        for model_key in selected_keys:
            spec = self.spec(model_key)
            rows.append(
                {
                    "model_key": spec.key,
                    "display_name": spec.display_name,
                    "family": spec.family,
                    "enabled": spec.enabled,
                    "random_state_param": spec.random_state_param or "",
                    "early_stopping_enabled": bool(spec.early_stopping.get("enabled", False)),
                    "overfitting_controls": "; ".join(spec.overfitting_controls),
                    "params_json": json.dumps(spec.params, sort_keys=True),
                }
            )
        return pd.DataFrame(rows)

    def _params_with_seed(
        self,
        spec: ModelSpec,
        random_state: int | None,
        params_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = dict(spec.params)
        params.update(params_override or {})
        if spec.random_state_param and random_state is not None:
            params[spec.random_state_param] = random_state
        if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
        for key, value in list(params.items()):
            if isinstance(value, list):
                params[key] = tuple(value)
        return params

    def _gpr_kernel(self, model_key: str) -> ConstantKernel:
        raw_kernel = dict(self.config.models[model_key].get("kernel", {}))
        constant_value = float(raw_kernel.get("constant_value", 1.0))
        length_scale = float(raw_kernel.get("rbf_length_scale", 1.0))
        noise_level = float(raw_kernel.get("white_noise_level", 0.01))
        return ConstantKernel(constant_value) * RBF(length_scale) + WhiteKernel(noise_level)

    @staticmethod
    def _xgboost_regressor(params: dict[str, Any]) -> BaseEstimator:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoost is required for model key 'xgboost'.") from exc
        return XGBRegressor(**params)

    @staticmethod
    def _lightgbm_regressor(params: dict[str, Any]) -> BaseEstimator:
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("LightGBM is required for model key 'lightgbm'.") from exc
        return LGBMRegressor(**params)
