from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_economic_return_target(
    prices: pd.Series,
    *,
    horizon: int = 5,
    transaction_cost_threshold: float | pd.Series = 0.0,
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    values = pd.to_numeric(prices, errors="coerce")
    future_return = values.shift(-horizon) / values - 1.0
    costs = (
        pd.Series(float(transaction_cost_threshold), index=values.index)
        if np.isscalar(transaction_cost_threshold)
        else pd.to_numeric(transaction_cost_threshold, errors="coerce").reindex(values.index)
    )
    return pd.DataFrame(
        {
            "future_return": future_return,
            "cost_threshold": costs,
            "target": (future_return > costs).astype("Int64").where(future_return.notna() & costs.notna()),
        }
    )


@dataclass
class ReturnPredictionModel:
    model_type: str = "logistic"
    random_state: int = 42
    splits: int = 5

    def __post_init__(self) -> None:
        self.model: ClassifierMixin | None = None
        self.feature_names: list[str] = []
        self.validation: dict[str, float] = {}
        self.backend: str = self.model_type

    def fit(self, features: pd.DataFrame, target: pd.Series) -> ReturnPredictionModel:
        x, y = _training_data(features, target)
        if len(x) < max(60, self.splits * 10) or y.nunique() != 2:
            raise ValueError("training requires at least 60 rows and both target classes")
        self.feature_names = [str(column) for column in x.columns]
        base, self.backend = _classifier(self.model_type, self.random_state)
        cv = TimeSeriesSplit(n_splits=self.splits)
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", base),
            ]
        )
        validation_targets: list[np.ndarray] = []
        validation_probabilities: list[np.ndarray] = []
        for train_indices, test_indices in cv.split(x):
            fold_model = clone(pipeline).fit(x.iloc[train_indices], y.iloc[train_indices])
            if hasattr(fold_model, "predict_proba"):
                fold_probability = fold_model.predict_proba(x.iloc[test_indices])[:, 1]
            else:
                decision = fold_model.decision_function(x.iloc[test_indices])
                fold_probability = 1.0 / (1.0 + np.exp(-np.asarray(decision, dtype=float)))
            validation_targets.append(y.iloc[test_indices].to_numpy(dtype=int))
            validation_probabilities.append(np.asarray(fold_probability, dtype=float))
        validation_y = np.concatenate(validation_targets)
        validation_probability = np.concatenate(validation_probabilities)
        self.validation = {
            "brier_score": float(brier_score_loss(validation_y, validation_probability)),
            "roc_auc": float(roc_auc_score(validation_y, validation_probability)),
            "observations": float(len(x)),
            "out_of_sample_observations": float(len(validation_y)),
        }
        calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=TimeSeriesSplit(n_splits=3))
        calibrated.fit(x, y)
        self.model = calibrated
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise ValueError("return model is not fitted")
        values = features.loc[:, self.feature_names]
        probability = self.model.predict_proba(values)[:, 1]
        return pd.Series(probability, index=values.index, name="probability_positive")

    def report(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "backend": self.backend,
            "features": self.feature_names,
            "temporal_validation": self.validation,
            "probability_calibrated": True,
            "research_only": True,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


class MetaLabelingEngine:
    """Learns TAKE/SKIP labels without replacing the primary strategy."""

    def __init__(self, *, model_type: str = "random_forest", random_state: int = 42):
        self.model = ReturnPredictionModel(model_type=model_type, random_state=random_state)

    def fit(
        self,
        features: pd.DataFrame,
        primary_side: pd.Series,
        future_return: pd.Series,
        transaction_cost: float | pd.Series,
    ) -> MetaLabelingEngine:
        side = pd.to_numeric(primary_side, errors="coerce").reindex(features.index)
        realized = pd.to_numeric(future_return, errors="coerce").reindex(features.index)
        costs = (
            pd.Series(float(transaction_cost), index=features.index)
            if np.isscalar(transaction_cost)
            else pd.to_numeric(transaction_cost, errors="coerce").reindex(features.index)
        )
        active = side.ne(0) & side.notna() & realized.notna() & costs.notna()
        labels = (side[active] * realized[active] > costs[active]).astype(int)
        self.model.fit(features.loc[active], labels)
        return self

    def decide(self, features: pd.DataFrame, *, threshold: float = 0.5) -> pd.DataFrame:
        probability = self.model.predict_proba(features)
        return pd.DataFrame(
            {
                "take_probability": probability,
                "meta_label": np.where(probability >= threshold, "TAKE_TRADE", "SKIP_TRADE"),
            },
            index=features.index,
        )


class ProbabilityCalibratedSignalEngine:
    def __init__(self, *, random_state: int = 42):
        self.classifier = ReturnPredictionModel(model_type="logistic", random_state=random_state)
        self.regressor = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=random_state, n_jobs=1)),
            ]
        )
        self.features: list[str] = []
        self.residual_downside = 0.0

    def fit(self, features: pd.DataFrame, future_returns: pd.Series, *, cost_threshold: float = 0.0) -> ProbabilityCalibratedSignalEngine:
        target = pd.to_numeric(future_returns, errors="coerce").reindex(features.index)
        valid = target.notna()
        x = features.loc[valid]
        y = target.loc[valid]
        self.features = [str(column) for column in x.columns]
        self.classifier.fit(x, (y > cost_threshold).astype(int))
        self.regressor.fit(x, y)
        prediction = self.regressor.predict(x)
        self.residual_downside = float(np.quantile(y.to_numpy(dtype=float) - prediction, 0.05))
        return self

    def predict(
        self,
        features: pd.DataFrame,
        forecast_volatility: pd.Series,
        *,
        maximum_position: float = 1.0,
    ) -> pd.DataFrame:
        x = features.loc[:, self.features]
        probability = self.classifier.predict_proba(x)
        expected = self.regressor.predict(x)
        volatility = pd.to_numeric(forecast_volatility, errors="coerce").reindex(x.index)
        confidence = (probability - 0.5).abs() * 2
        uncertainty = -(probability * np.log(probability.clip(1e-12)) + (1 - probability) * np.log((1 - probability).clip(1e-12))) / math.log(2)
        strength = np.maximum(expected, 0.0)
        raw_size = strength * confidence / volatility.replace(0.0, np.nan)
        position_size = raw_size.clip(lower=0.0, upper=maximum_position).fillna(0.0)
        return pd.DataFrame(
            {
                "expected_return": expected,
                "probability_positive_after_costs": probability,
                "expected_downside": expected + self.residual_downside,
                "uncertainty": uncertainty,
                "forecast_volatility": volatility,
                "position_size": position_size,
            },
            index=x.index,
        )


@dataclass
class TemporalConvolutionalReturnModel:
    """Small dependency-light causal TCN classifier for shadow research.

    The convolution filters and classification head are trained jointly with
    Adam.  Only the last causal receptive field is used for each prediction;
    no padding can expose observations after the decision timestamp.
    """

    dilations: tuple[int, ...] = (1, 2, 4, 8)
    channels: int = 4
    kernel_size: int = 3
    epochs: int = 120
    learning_rate: float = 0.01
    l2: float = 1e-4
    random_state: int = 42
    splits: int = 3

    def __post_init__(self) -> None:
        if (
            not self.dilations
            or any(value < 1 for value in self.dilations)
            or self.channels < 1
            or self.kernel_size < 2
            or self.epochs < 1
            or self.learning_rate <= 0
            or self.splits < 2
        ):
            raise ValueError("invalid TCN configuration")
        self.model: _NumpyTCNBinaryCore | None = None
        self.validation: dict[str, float] = {}
        self.sequence_length: int | None = None
        self.feature_count: int | None = None

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * max(self.dilations)

    def fit(
        self,
        sequences: np.ndarray,
        target: pd.Series | np.ndarray,
    ) -> TemporalConvolutionalReturnModel:
        x, y = _sequence_training_data(sequences, target)
        if x.shape[1] < self.receptive_field:
            raise ValueError(
                f"sequence length must cover causal receptive field {self.receptive_field}"
            )
        if len(x) < max(80, self.splits * 20) or len(np.unique(y)) != 2:
            raise ValueError("TCN training requires at least 80 rows and both target classes")
        self.sequence_length = int(x.shape[1])
        self.feature_count = int(x.shape[2])
        validation_targets: list[np.ndarray] = []
        validation_probabilities: list[np.ndarray] = []
        for fold, (train_indices, test_indices) in enumerate(
            TimeSeriesSplit(n_splits=self.splits).split(x)
        ):
            fold_model = self._new_core(self.random_state + fold).fit(
                x[train_indices], y[train_indices]
            )
            validation_targets.append(y[test_indices])
            validation_probabilities.append(
                fold_model.predict_proba(x[test_indices])
            )
        validation_y = np.concatenate(validation_targets)
        validation_probability = np.concatenate(validation_probabilities)
        self.validation = {
            "brier_score": float(
                brier_score_loss(validation_y, validation_probability)
            ),
            "roc_auc": float(
                roc_auc_score(validation_y, validation_probability)
            ),
            "observations": float(len(x)),
            "out_of_sample_observations": float(len(validation_y)),
        }
        self.model = self._new_core(self.random_state).fit(x, y)
        return self

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("TCN return model is not fitted")
        values = _sequence_array(sequences)
        if values.shape[1:] != (self.sequence_length, self.feature_count):
            raise ValueError("TCN prediction shape differs from fitted sequence shape")
        return self.model.predict_proba(values)

    def report(self) -> dict[str, Any]:
        return {
            "model_type": "TEMPORAL_CONVOLUTIONAL_NETWORK",
            "backend": "NUMPY_TRAINED_CAUSAL_DILATED_CONVOLUTION",
            "dilations": list(self.dilations),
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "receptive_field": self.receptive_field,
            "sequence_length": self.sequence_length,
            "feature_count": self.feature_count,
            "temporal_validation": self.validation,
            "probability_calibrated": False,
            "research_only": True,
            "money_control": False,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }

    def _new_core(self, random_state: int) -> _NumpyTCNBinaryCore:
        return _NumpyTCNBinaryCore(
            dilations=self.dilations,
            channels=self.channels,
            kernel_size=self.kernel_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            l2=self.l2,
            random_state=random_state,
        )


class _NumpyTCNBinaryCore:
    def __init__(
        self,
        *,
        dilations: tuple[int, ...],
        channels: int,
        kernel_size: int,
        epochs: int,
        learning_rate: float,
        l2: float,
        random_state: int,
    ) -> None:
        self.dilations = dilations
        self.channels = channels
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.random_state = random_state
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.parameters: dict[str, np.ndarray] = {}

    def fit(self, sequences: np.ndarray, target: np.ndarray) -> _NumpyTCNBinaryCore:
        self.mean = sequences.mean(axis=(0, 1), keepdims=True)
        self.scale = sequences.std(axis=(0, 1), keepdims=True)
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)
        values = (sequences - self.mean) / self.scale
        feature_count = values.shape[2]
        rng = np.random.default_rng(self.random_state)
        self.parameters = {
            "filters": rng.normal(
                0.0,
                0.05,
                size=(
                    len(self.dilations),
                    self.channels,
                    self.kernel_size,
                    feature_count,
                ),
            ),
            "bias": np.zeros((len(self.dilations), self.channels)),
            "head": rng.normal(
                0.0, 0.05, size=(len(self.dilations), self.channels)
            ),
            "intercept": np.zeros(1),
        }
        first = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        second = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        labels = target.astype(float)
        for step in range(1, self.epochs + 1):
            probability, hidden, lagged = self._forward(values)
            error = (probability - labels) / len(labels)
            gradients: dict[str, np.ndarray] = {
                "head": np.einsum("ndc,n->dc", hidden, error)
                + self.l2 * self.parameters["head"],
                "intercept": np.asarray([error.sum()]),
                "filters": np.zeros_like(self.parameters["filters"]),
                "bias": np.zeros_like(self.parameters["bias"]),
            }
            hidden_gradient = error[:, None, None] * self.parameters["head"][
                None, :, :
            ]
            activation_gradient = hidden_gradient * (1.0 - hidden**2)
            for dilation_index, inputs in enumerate(lagged):
                gradients["filters"][dilation_index] = np.einsum(
                    "nc,nkf->ckf",
                    activation_gradient[:, dilation_index, :],
                    inputs,
                ) + self.l2 * self.parameters["filters"][dilation_index]
                gradients["bias"][dilation_index] = activation_gradient[
                    :, dilation_index, :
                ].sum(axis=0)
            for name, gradient in gradients.items():
                first[name] = beta1 * first[name] + (1 - beta1) * gradient
                second[name] = beta2 * second[name] + (1 - beta2) * gradient**2
                first_hat = first[name] / (1 - beta1**step)
                second_hat = second[name] / (1 - beta2**step)
                self.parameters[name] -= self.learning_rate * first_hat / (
                    np.sqrt(second_hat) + epsilon
                )
        return self

    def predict_proba(self, sequences: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or not self.parameters:
            raise ValueError("TCN core is not fitted")
        values = (sequences - self.mean) / self.scale
        probability, _, _ = self._forward(values)
        return probability

    def _forward(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        lagged: list[np.ndarray] = []
        hidden = np.empty(
            (len(values), len(self.dilations), self.channels), dtype=float
        )
        for dilation_index, dilation in enumerate(self.dilations):
            indices = [
                values.shape[1] - 1 - lag * dilation
                for lag in range(self.kernel_size)
            ]
            inputs = values[:, indices, :]
            lagged.append(inputs)
            activation = np.einsum(
                "nkf,ckf->nc",
                inputs,
                self.parameters["filters"][dilation_index],
            ) + self.parameters["bias"][dilation_index]
            hidden[:, dilation_index, :] = np.tanh(activation)
        logits = (
            np.einsum("ndc,dc->n", hidden, self.parameters["head"])
            + self.parameters["intercept"][0]
        )
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
        return probability, hidden, lagged


def _training_data(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    x = features.replace([np.inf, -np.inf], np.nan).copy()
    y = pd.to_numeric(target, errors="coerce").reindex(x.index)
    valid = y.notna()
    x, y = x.loc[valid], y.loc[valid].astype(int)
    if not x.index.is_monotonic_increasing:
        raise ValueError("training data must be ordered causally")
    return x, y


def _sequence_array(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=float)
    if values.ndim != 3 or values.shape[1] < 2 or values.shape[2] < 1:
        raise ValueError("TCN sequences must have shape rows x time x features")
    if not np.isfinite(values).all():
        raise ValueError("TCN sequences must be finite")
    return values


def _sequence_training_data(
    sequences: np.ndarray,
    target: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = _sequence_array(sequences)
    if isinstance(target, pd.Series) and not target.index.is_monotonic_increasing:
        raise ValueError("training data must be ordered causally")
    labels = np.asarray(target, dtype=float).reshape(-1)
    if len(labels) != len(values):
        raise ValueError("TCN target length must equal sequence rows")
    valid = np.isfinite(labels)
    labels = labels[valid].astype(int)
    values = values[valid]
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("TCN target must be binary")
    return values, labels


def _classifier(model_type: str, random_state: int) -> tuple[ClassifierMixin, str]:
    if model_type == "logistic":
        return LogisticRegression(max_iter=2_000, random_state=random_state), "sklearn_logistic"
    if model_type == "ridge":
        return RidgeClassifier(random_state=random_state), "sklearn_ridge"
    if model_type == "lasso":
        return SGDClassifier(loss="log_loss", penalty="l1", max_iter=5_000, random_state=random_state), "sklearn_lasso_logistic"
    if model_type == "elastic_net":
        return SGDClassifier(loss="log_loss", penalty="elasticnet", l1_ratio=0.5, max_iter=5_000, random_state=random_state), "sklearn_elastic_net_logistic"
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=random_state, n_jobs=1), "sklearn_random_forest"
    if model_type in {"xgboost", "lightgbm", "catboost"}:
        try:
            if model_type == "xgboost":
                from xgboost import XGBClassifier

                return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=random_state, n_jobs=1), "xgboost"
            if model_type == "lightgbm":
                from lightgbm import LGBMClassifier

                return LGBMClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=random_state, n_jobs=1, verbose=-1), "lightgbm"
            from catboost import CatBoostClassifier

            return CatBoostClassifier(iterations=300, depth=3, learning_rate=0.05, random_seed=random_state, verbose=False), "catboost"
        except ImportError:
            return HistGradientBoostingClassifier(max_iter=300, max_depth=3, learning_rate=0.05, random_state=random_state), f"sklearn_hist_gradient_boosting_fallback_for_{model_type}"
    raise ValueError("unsupported model_type")
