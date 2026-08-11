from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from stocks.regimes.filter import hamilton_filter
from stocks.regimes.model import FrozenHMM, fit_markov_regression


REGIME_LABELS = (
    "EXPANSION_INFLATION",
    "EXPANSION_DISINFLATION",
    "SLOWDOWN",
    "RECESSION",
    "LIQUIDITY_CRISIS",
    "RISK_ON",
    "RISK_OFF",
)


@dataclass(frozen=True)
class RuleBasedRegimeDetector:
    crisis_vix: float = 35.0
    risk_off_vix: float = 25.0
    recession_pmi: float = 45.0
    expansion_pmi: float = 52.0

    def classify(self, features: Mapping[str, float]) -> dict[str, Any]:
        values = {key: float(value) for key, value in features.items() if value is not None}
        vix = values.get("vix", 20.0)
        pmi = values.get("pmi", 50.0)
        momentum = values.get("spx_momentum", 0.0)
        curve = values.get("yield_curve_10y_2y", 0.0)
        credit = values.get("credit_spread", 0.0)
        inflation_trend = values.get("inflation_trend", 0.0)
        liquidity = values.get("liquidity_growth", 0.0)
        breadth = values.get("market_breadth", 0.5)
        if vix >= self.crisis_vix or (credit > 5.0 and liquidity < 0):
            regime = "LIQUIDITY_CRISIS"
        elif pmi < self.recession_pmi and curve < 0:
            regime = "RECESSION"
        elif pmi < 50 or momentum < -0.05:
            regime = "SLOWDOWN"
        elif pmi >= self.expansion_pmi and momentum > 0 and inflation_trend > 0:
            regime = "EXPANSION_INFLATION"
        elif pmi >= self.expansion_pmi and momentum > 0:
            regime = "EXPANSION_DISINFLATION"
        elif vix >= self.risk_off_vix or breadth < 0.4:
            regime = "RISK_OFF"
        else:
            regime = "RISK_ON"
        return {
            "regime": regime,
            "confidence": _rule_confidence(values, regime),
            "features_used": sorted(values),
            "research_only": True,
            "execution_authority": "NONE",
        }


class StatisticalRegimeDetector:
    """K-Means or Gaussian-mixture regime probabilities with frozen scaling."""

    def __init__(self, *, method: str = "gmm", n_regimes: int = 4, random_state: int = 42):
        if method not in {"kmeans", "gmm"}:
            raise ValueError("method must be kmeans or gmm")
        if n_regimes < 2:
            raise ValueError("n_regimes must be at least 2")
        self.method = method
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: KMeans | GaussianMixture | None = None
        self.feature_names: list[str] = []
        self.cluster_labels: dict[int, str] = {}

    def fit(self, features: pd.DataFrame) -> StatisticalRegimeDetector:
        clean = features.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < self.n_regimes * 10:
            raise ValueError("insufficient regime observations")
        self.feature_names = [str(column) for column in clean.columns]
        scaled = self.scaler.fit_transform(clean)
        if self.method == "kmeans":
            self.model = KMeans(n_clusters=self.n_regimes, n_init=20, random_state=self.random_state).fit(scaled)
            centers = self.model.cluster_centers_
        else:
            self.model = GaussianMixture(n_components=self.n_regimes, covariance_type="full", n_init=5, random_state=self.random_state).fit(scaled)
            centers = self.model.means_
        self.cluster_labels = _canonical_cluster_labels(centers, self.feature_names)
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("regime detector is not fitted")
        values = features.loc[:, self.feature_names].replace([np.inf, -np.inf], np.nan).dropna()
        scaled = self.scaler.transform(values)
        if isinstance(self.model, GaussianMixture):
            raw = self.model.predict_proba(scaled)
        else:
            distances = self.model.transform(scaled)
            inverse = np.exp(-distances)
            raw = inverse / inverse.sum(axis=1, keepdims=True)
        result = pd.DataFrame(index=values.index)
        for cluster in range(self.n_regimes):
            label = self.cluster_labels[cluster]
            result[label] = result.get(label, 0.0) + raw[:, cluster]
        result["regime"] = result.idxmax(axis=1)
        result["confidence"] = result.drop(columns="regime").max(axis=1)
        return result


class StrategyAllocationEngine:
    DEFAULT = {
        "RISK_ON": {"trend": 0.35, "breakout": 0.25, "momentum": 0.25, "mean_reversion": 0.10, "defensive": 0.05},
        "EXPANSION_INFLATION": {"trend": 0.30, "breakout": 0.25, "momentum": 0.15, "mean_reversion": 0.10, "defensive": 0.20},
        "EXPANSION_DISINFLATION": {"trend": 0.35, "breakout": 0.20, "momentum": 0.25, "mean_reversion": 0.10, "defensive": 0.10},
        "SLOWDOWN": {"trend": 0.15, "breakout": 0.10, "momentum": 0.10, "mean_reversion": 0.30, "defensive": 0.35},
        "RECESSION": {"trend": 0.10, "breakout": 0.05, "momentum": 0.05, "mean_reversion": 0.20, "defensive": 0.60},
        "RISK_OFF": {"trend": 0.10, "breakout": 0.10, "momentum": 0.10, "mean_reversion": 0.30, "defensive": 0.40},
        "LIQUIDITY_CRISIS": {"trend": 0.05, "breakout": 0.05, "momentum": 0.05, "mean_reversion": 0.05, "defensive": 0.80},
    }

    def allocate(self, regime_probabilities: Mapping[str, float]) -> dict[str, Any]:
        probabilities = {key: max(float(value), 0.0) for key, value in regime_probabilities.items() if key in self.DEFAULT}
        total = sum(probabilities.values())
        if total <= 0:
            raise ValueError("recognized regime probabilities must sum to a positive value")
        probabilities = {key: value / total for key, value in probabilities.items()}
        weights: dict[str, float] = {}
        for regime, probability in probabilities.items():
            for strategy, weight in self.DEFAULT[regime].items():
                weights[strategy] = weights.get(strategy, 0.0) + probability * weight
        return {
            "regime_probabilities": probabilities,
            "strategy_weights": weights,
            "cash_defensive_weight": weights.get("defensive", 0.0),
            "research_only": True,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


class HiddenMarkovRegimeDetector:
    """Adapter over the repository's causal Hamilton-filter HMM."""

    def __init__(self, model: FrozenHMM | None = None):
        self.model = model

    def fit(self, features: pd.DataFrame, *, n_regimes: int = 3) -> HiddenMarkovRegimeDetector:
        self.model = fit_markov_regression(features, n_regimes=n_regimes)
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("HMM regime detector is not fitted")
        return hamilton_filter(self.model, features)

    def report(self) -> dict[str, Any]:
        if self.model is None:
            raise ValueError("HMM regime detector is not fitted")
        return {
            "method": "HIDDEN_MARKOV_MODEL",
            "n_regimes": self.model.n_regimes,
            "transition": self.model.transition,
            "expected_durations": self.model.expected_durations,
            "converged": self.model.converged,
            "smoothed_probabilities_used": False,
            "causal_filter": "HAMILTON_FILTER",
            "execution_authority": "NONE",
        }


def _canonical_cluster_labels(centers: np.ndarray, names: list[str]) -> dict[int, str]:
    momentum_index = names.index("spx_momentum") if "spx_momentum" in names else 0
    volatility_index = names.index("vix") if "vix" in names else min(1, len(names) - 1)
    ordering = sorted(range(len(centers)), key=lambda index: (centers[index, momentum_index], -centers[index, volatility_index]))
    labels = ["LIQUIDITY_CRISIS", "RISK_OFF", "SLOWDOWN", "RISK_ON", "EXPANSION_DISINFLATION", "EXPANSION_INFLATION"]
    selected = labels[: len(ordering)]
    return {cluster: selected[position] for position, cluster in enumerate(ordering)}


def _rule_confidence(features: Mapping[str, float], regime: str) -> float:
    evidence = {
        "LIQUIDITY_CRISIS": abs(features.get("vix", 20.0) - 35.0) / 20.0,
        "RECESSION": abs(features.get("pmi", 50.0) - 45.0) / 15.0,
        "SLOWDOWN": abs(features.get("pmi", 50.0) - 50.0) / 15.0,
        "RISK_OFF": abs(features.get("vix", 20.0) - 25.0) / 20.0,
    }.get(regime, abs(features.get("spx_momentum", 0.0)) * 5)
    return float(np.clip(0.5 + evidence, 0.5, 0.99))
