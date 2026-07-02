"""Custom sklearn transformers for leakage-safe preprocessing."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pandas_types
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


class FeatureValidationError(ValueError):
    """Raised when feature validation fails."""


def _as_dataframe(X: Any) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise FeatureValidationError("Preprocessing expects a pandas DataFrame input.")
    return X


class FeatureValidator(BaseEstimator, TransformerMixin):
    """Validate required feature columns before preprocessing."""

    def __init__(
        self,
        required_features: tuple[str, ...],
        numeric_features: tuple[str, ...] = (),
        categorical_features: tuple[str, ...] = (),
        enabled: bool = True,
        allow_extra_columns: bool = False,
        strict_numeric_dtype: bool = True,
    ) -> None:
        self.required_features = required_features
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.enabled = enabled
        self.allow_extra_columns = allow_extra_columns
        self.strict_numeric_dtype = strict_numeric_dtype

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureValidator":
        """Validate columns without learning data-dependent transformations."""
        if self.enabled:
            self._validate(X)
        self.feature_names_in_ = tuple(X.columns) if isinstance(X, pd.DataFrame) else ()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and return a defensive dataframe copy."""
        if self.enabled:
            self._validate(X)
        return _as_dataframe(X).copy()

    def _validate(self, X: pd.DataFrame) -> None:
        dataframe = _as_dataframe(X)
        missing = tuple(feature for feature in self.required_features if feature not in dataframe.columns)
        if missing:
            raise FeatureValidationError(f"Missing required features: {missing}")

        duplicate_columns = dataframe.columns[dataframe.columns.duplicated()].tolist()
        if duplicate_columns:
            raise FeatureValidationError(f"Duplicate feature columns: {tuple(duplicate_columns)}")

        if not self.allow_extra_columns:
            unexpected = tuple(
                column for column in dataframe.columns if column not in self.required_features
            )
            if unexpected:
                raise FeatureValidationError(f"Unexpected feature columns: {unexpected}")

        if self.strict_numeric_dtype:
            bad_numeric = tuple(
                column
                for column in self.numeric_features
                if column in dataframe.columns and not pandas_types.is_numeric_dtype(dataframe[column])
            )
            if bad_numeric:
                raise FeatureValidationError(
                    f"Expected numeric dtype for features: {bad_numeric}"
                )


class ConstantFeatureRemover(BaseEstimator, TransformerMixin):
    """Drop features that are constant in the training fold only."""

    def __init__(
        self,
        enabled: bool = True,
        dropna: bool = False,
        tolerance: float = 0.0,
        protected_features: tuple[str, ...] = (),
    ) -> None:
        self.enabled = enabled
        self.dropna = dropna
        self.tolerance = tolerance
        self.protected_features = protected_features

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ConstantFeatureRemover":
        """Detect constants from the data passed to this fit call."""
        dataframe = _as_dataframe(X)
        protected = set(self.protected_features)
        constant_features: list[str] = []
        for column in dataframe.columns:
            if column in protected:
                continue
            if self.enabled and self._is_constant(dataframe[column]):
                constant_features.append(column)

        self.constant_features_ = tuple(constant_features)
        self.kept_features_ = tuple(column for column in dataframe.columns if column not in constant_features)
        self.feature_names_in_ = tuple(dataframe.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features learned as constant during fit."""
        check_is_fitted(self, "kept_features_")
        dataframe = _as_dataframe(X)
        available_kept = [column for column in self.kept_features_ if column in dataframe.columns]
        return dataframe.loc[:, available_kept].copy()

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return kept feature names."""
        check_is_fitted(self, "kept_features_")
        return np.asarray(self.kept_features_, dtype=object)

    def _is_constant(self, series: pd.Series) -> bool:
        values = series.dropna() if self.dropna else series
        if len(values) == 0:
            return True
        if pandas_types.is_numeric_dtype(values) and self.tolerance > 0:
            return bool((values.max() - values.min()) <= self.tolerance)
        return int(values.nunique(dropna=self.dropna)) <= 1


class ConfiguredColumnPreprocessor(BaseEstimator, TransformerMixin):
    """Scale numeric columns and one-hot encode categorical columns inside an sklearn pipeline."""

    def __init__(
        self,
        numeric_features: tuple[str, ...],
        categorical_features: tuple[str, ...] = (),
        scaler: str = "standard",
        scaler_params: dict[str, Any] | None = None,
        one_hot_enabled: bool = True,
        one_hot_params: dict[str, Any] | None = None,
        output: str = "pandas",
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scaler = scaler
        self.scaler_params = scaler_params
        self.one_hot_enabled = one_hot_enabled
        self.one_hot_params = one_hot_params
        self.output = output

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ConfiguredColumnPreprocessor":
        """Fit scalers and encoders on the provided training fold."""
        dataframe = _as_dataframe(X)
        self.numeric_features_ = tuple(
            column for column in self.numeric_features if column in dataframe.columns
        )
        self.categorical_features_ = tuple(
            column for column in self.categorical_features if column in dataframe.columns
        )

        if self.numeric_features_ and self.scaler != "none":
            self.scaler_ = self._make_scaler()
            self.scaler_.fit(dataframe.loc[:, self.numeric_features_])
        else:
            self.scaler_ = None

        if self.categorical_features_ and self.one_hot_enabled:
            self.encoder_ = self._make_encoder()
            self.encoder_.fit(dataframe.loc[:, self.categorical_features_])
        else:
            self.encoder_ = None

        self.feature_names_in_ = tuple(dataframe.columns)
        self.output_features_ = self._build_output_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        """Transform data using only artifacts learned during fit."""
        check_is_fitted(self, "output_features_")
        dataframe = _as_dataframe(X)
        parts: list[pd.DataFrame] = []

        if self.numeric_features_:
            numeric = dataframe.loc[:, self.numeric_features_]
            values = self.scaler_.transform(numeric) if self.scaler_ is not None else numeric.to_numpy()
            parts.append(pd.DataFrame(values, columns=self.numeric_features_, index=dataframe.index))

        if self.categorical_features_:
            categorical = dataframe.loc[:, self.categorical_features_]
            if self.encoder_ is not None:
                encoded = self.encoder_.transform(categorical)
                if hasattr(encoded, "toarray"):
                    encoded = encoded.toarray()
                columns = self.encoder_.get_feature_names_out(self.categorical_features_)
                parts.append(pd.DataFrame(encoded, columns=columns, index=dataframe.index))
            else:
                parts.append(categorical.copy())

        output = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=dataframe.index)
        if self.output == "numpy":
            return output.to_numpy()
        return output

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Return output feature names."""
        check_is_fitted(self, "output_features_")
        return np.asarray(self.output_features_, dtype=object)

    def _make_scaler(self) -> StandardScaler | MinMaxScaler | RobustScaler:
        params = dict(self.scaler_params or {})
        if self.scaler == "standard":
            return StandardScaler(**params)
        if self.scaler == "minmax":
            return MinMaxScaler(**params)
        if self.scaler == "robust":
            return RobustScaler(**params)
        raise ValueError(f"Unsupported scaler: {self.scaler}")

    def _make_encoder(self) -> OneHotEncoder:
        params = dict(self.one_hot_params or {})
        return OneHotEncoder(**params)

    def _build_output_feature_names(self) -> tuple[str, ...]:
        names: list[str] = list(self.numeric_features_)
        if self.categorical_features_:
            if self.encoder_ is not None:
                names.extend(self.encoder_.get_feature_names_out(self.categorical_features_).tolist())
            else:
                names.extend(self.categorical_features_)
        return tuple(names)

