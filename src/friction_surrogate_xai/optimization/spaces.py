"""Search-space sampling for Random Search and Optuna."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


class SearchSpaceSampler:
    """Sample model hyperparameters without Grid Search."""

    def __init__(self, search_spaces: dict[str, Any]) -> None:
        self.search_spaces = search_spaces

    def random_sample(
        self,
        model_key: str,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Sample one parameter set for Random Search."""
        space = self.search_spaces.get(model_key, {})
        return {
            param_name: self._random_value(param_spec, rng)
            for param_name, param_spec in space.items()
        }

    def suggest_optuna(self, model_key: str, trial: Any) -> dict[str, Any]:
        """Suggest one parameter set using Optuna's Bayesian sampler."""
        space = self.search_spaces.get(model_key, {})
        return {
            param_name: self._optuna_value(param_name, param_spec, trial)
            for param_name, param_spec in space.items()
        }

    def has_space(self, model_key: str) -> bool:
        """Return whether a model has at least one tunable parameter."""
        return bool(self.search_spaces.get(model_key, {}))

    def _random_value(self, param_spec: dict[str, Any], rng: np.random.Generator) -> Any:
        param_type = str(param_spec.get("type", "categorical"))
        if param_type == "categorical":
            values = list(param_spec.get("values", ()))
            if not values:
                raise ValueError("Categorical search parameter requires non-empty values.")
            return _normalize_value(values[int(rng.integers(0, len(values)))])
        if param_type == "int":
            low = int(param_spec["low"])
            high = int(param_spec["high"])
            return int(rng.integers(low, high + 1))
        if param_type == "float":
            return float(rng.uniform(float(param_spec["low"]), float(param_spec["high"])))
        if param_type == "log_float":
            low = np.log(float(param_spec["low"]))
            high = np.log(float(param_spec["high"]))
            return float(np.exp(rng.uniform(low, high)))
        if param_type == "log_int":
            low = np.log(float(param_spec["low"]))
            high = np.log(float(param_spec["high"]))
            return int(round(float(np.exp(rng.uniform(low, high)))))
        raise ValueError(f"Unsupported search parameter type: {param_type}")

    def _optuna_value(self, name: str, param_spec: dict[str, Any], trial: Any) -> Any:
        param_type = str(param_spec.get("type", "categorical"))
        if param_type == "categorical":
            values = [_encode_optuna_choice(value) for value in param_spec.get("values", ())]
            if not values:
                raise ValueError(f"Categorical search parameter '{name}' requires values.")
            return _decode_optuna_choice(trial.suggest_categorical(name, values))
        if param_type == "int":
            return trial.suggest_int(name, int(param_spec["low"]), int(param_spec["high"]))
        if param_type == "float":
            return trial.suggest_float(name, float(param_spec["low"]), float(param_spec["high"]))
        if param_type == "log_float":
            return trial.suggest_float(
                name,
                float(param_spec["low"]),
                float(param_spec["high"]),
                log=True,
            )
        if param_type == "log_int":
            return trial.suggest_int(
                name,
                int(param_spec["low"]),
                int(param_spec["high"]),
                log=True,
            )
        raise ValueError(f"Unsupported search parameter type: {param_type}")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _encode_optuna_choice(value: Any) -> Any:
    normalized = _normalize_value(value)
    if isinstance(normalized, tuple):
        return "__json__:" + json.dumps(list(normalized))
    return normalized


def _decode_optuna_choice(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("__json__:"):
        return tuple(json.loads(value.removeprefix("__json__:")))
    return value
