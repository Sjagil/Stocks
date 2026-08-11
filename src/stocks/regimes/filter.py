from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from stocks.regimes.model import FrozenHMM


def hamilton_filter(
    model: FrozenHMM,
    features: pd.DataFrame,
    *,
    initial_probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    clean = features.replace([np.inf, -np.inf], np.nan).dropna()
    exog = clean.loc[:, model.feature_names].to_numpy(dtype=float)
    endog = clean["world_index_ret"].to_numpy(dtype=float)
    transition = np.asarray(model.transition, dtype=float)
    intercepts = np.asarray(model.intercepts, dtype=float)
    coefficients = np.asarray(model.coefficients, dtype=float)
    variances = np.asarray(model.variances, dtype=float)
    alpha = np.asarray(
        initial_probabilities
        if initial_probabilities is not None
        else model.initial_probabilities,
        dtype=float,
    )
    alpha = alpha / alpha.sum()
    rows = []
    for observed, regressors in zip(endog, exog, strict=True):
        predicted = np.maximum(alpha @ transition, 1e-300)
        means = intercepts + coefficients @ regressors
        log_emission = (
            -0.5 * np.log(2.0 * np.pi * variances)
            - 0.5 * np.square(observed - means) / variances
        )
        log_weights = np.log(predicted) + log_emission
        alpha = np.exp(log_weights - logsumexp(log_weights))
        rows.append(alpha.copy())
    probabilities = pd.DataFrame(
        rows,
        index=clean.index,
        columns=[model.raw_to_label[state] for state in range(model.n_regimes)],
    )
    return probabilities.reindex(
        columns=[
            label
            for label in (
                "RISK_ON_TREND",
                "NEUTRAL_CHOPPY",
                "STRESS_HIGH_VOL",
                "INFLATION_RATE_SHOCK",
            )
            if label in probabilities
        ]
    )


@dataclass
class HysteresisState:
    active_label: str = "NEUTRAL_CHOPPY"
    pending_label: str | None = None
    confirmations: int = 0


class IntradayHMMFilter:
    def __init__(
        self,
        *,
        activation_threshold: float,
        deactivation_threshold: float,
        minimum_confirmations: int,
    ):
        if not 0 <= deactivation_threshold < activation_threshold <= 1:
            raise ValueError("INVALID_HMM_HYSTERESIS_THRESHOLDS")
        self.activation_threshold = activation_threshold
        self.deactivation_threshold = deactivation_threshold
        self.minimum_confirmations = minimum_confirmations
        self.state = HysteresisState()

    def update(self, probabilities: dict[str, float]) -> dict[str, object]:
        candidate = max(probabilities, key=lambda key: probabilities[key])
        candidate_probability = float(probabilities[candidate])
        active_probability = float(
            probabilities.get(self.state.active_label, 0.0)
        )
        if (
            candidate != self.state.active_label
            and candidate_probability >= self.activation_threshold
            and active_probability <= self.deactivation_threshold
        ):
            if self.state.pending_label == candidate:
                self.state.confirmations += 1
            else:
                self.state.pending_label = candidate
                self.state.confirmations = 1
            if self.state.confirmations >= self.minimum_confirmations:
                self.state.active_label = candidate
                self.state.pending_label = None
                self.state.confirmations = 0
        else:
            self.state.pending_label = None
            self.state.confirmations = 0
        return {
            "active_state": self.state.active_label,
            "execution_hints": _execution_hints(self.state.active_label),
        }


def _execution_hints(state: str) -> dict[str, str]:
    return {
        "RISK_ON_TREND": {
            "order_type": "LIMIT",
            "aggressiveness": "NORMAL",
            "speed": "PATIENT",
        },
        "NEUTRAL_CHOPPY": {
            "order_type": "LIMIT",
            "aggressiveness": "LOW",
            "speed": "PATIENT",
        },
        "STRESS_HIGH_VOL": {
            "order_type": "NONE",
            "aggressiveness": "ZERO",
            "speed": "HALT",
        },
        "INFLATION_RATE_SHOCK": {
            "order_type": "LIMIT",
            "aggressiveness": "LOW",
            "speed": "PATIENT",
        },
    }[state]
