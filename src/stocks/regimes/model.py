from __future__ import annotations

import re
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression,
)

from stocks.regimes.canonicalize import (
    CanonicalStateMap,
    canonicalize_states,
)


@dataclass(frozen=True)
class FrozenHMM:
    n_regimes: int
    feature_names: tuple[str, ...]
    transition: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]
    variances: tuple[float, ...]
    initial_probabilities: tuple[float, ...]
    raw_to_label: dict[int, str]
    converged: bool
    log_likelihood: float
    expected_durations: tuple[float, ...]
    training_state_occupancy: tuple[float, ...]
    training_observations: int
    feature_means: dict[str, float]
    feature_scales: dict[str, float]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def fit_markov_regression(
    features: pd.DataFrame,
    *,
    n_regimes: int = 3,
    search_reps: int = 5,
    max_iterations: int = 200,
    variance_floor: float = 1e-8,
) -> FrozenHMM:
    if n_regimes not in {3, 4}:
        raise ValueError("HMM_REGIME_COUNT_MUST_BE_THREE_OR_FOUR")
    clean = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < max(100, n_regimes * 30):
        raise ValueError("HMM_INSUFFICIENT_TRAINING_OBSERVATIONS")
    endog = clean["world_index_ret"].to_numpy(dtype=float)
    exog_frame = clean.drop(columns=["world_index_ret"])
    exog = exog_frame.to_numpy(dtype=float)
    model = MarkovRegression(
        endog,
        k_regimes=n_regimes,
        trend="c",
        exog=exog,
        switching_trend=True,
        switching_exog=False,
        switching_variance=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results = model.fit(
            search_reps=search_reps,
            search_iter=10,
            maxiter=max_iterations,
            disp=False,
        )
    params = {
        name: float(value)
        for name, value in zip(
            results.model.param_names,
            results.params,
            strict=True,
        )
    }
    intercepts = np.array(
        [params[f"const[{state}]"] for state in range(n_regimes)],
        dtype=float,
    )
    variances = np.maximum(
        np.array(
            [params[f"sigma2[{state}]"] for state in range(n_regimes)],
            dtype=float,
        ),
        variance_floor,
    )
    common = np.zeros(exog.shape[1], dtype=float)
    for feature_index in range(exog.shape[1]):
        prefix = f"x{feature_index + 1}["
        matches = [
            value for name, value in params.items() if name.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError("HMM_COMMON_COEFFICIENT_EXTRACTION_FAILED")
        common[feature_index] = matches[0]
    coefficients = np.tile(common, (n_regimes, 1))
    transition = np.asarray(
        results.regime_transition[:, :, 0],
        dtype=float,
    ).T
    filtered = np.asarray(
        results.filtered_marginal_probabilities,
        dtype=float,
    )
    state_weight = filtered.sum(axis=0)
    conditional_returns = np.divide(
        filtered.T @ endog,
        state_weight,
        out=np.zeros(n_regimes),
        where=state_weight > 0,
    )
    inflation_signature = None
    if n_regimes == 4 and "commodity_return" in exog_frame:
        commodity = exog_frame["commodity_return"].to_numpy(dtype=float)
        inflation_signature = np.divide(
            filtered.T @ commodity,
            state_weight,
            out=np.zeros(n_regimes),
            where=state_weight > 0,
        )
    mapping: CanonicalStateMap = canonicalize_states(
        variances,
        conditional_returns,
        inflation_signature=inflation_signature,
    )
    expected_durations = np.divide(
        1.0,
        np.maximum(1.0 - np.diag(transition), 1e-12),
    )
    mle = getattr(results, "mle_retvals", {})
    return FrozenHMM(
        n_regimes=n_regimes,
        feature_names=tuple(exog_frame.columns),
        transition=tuple(tuple(float(v) for v in row) for row in transition),
        intercepts=tuple(float(value) for value in intercepts),
        coefficients=tuple(
            tuple(float(value) for value in row) for row in coefficients
        ),
        variances=tuple(float(value) for value in variances),
        initial_probabilities=tuple(
            float(value) for value in filtered[-1]
        ),
        raw_to_label=mapping.raw_to_label,
        converged=bool(mle.get("converged", True)),
        log_likelihood=float(results.llf),
        expected_durations=tuple(
            float(value) for value in expected_durations
        ),
        training_state_occupancy=tuple(
            float(value) for value in filtered.mean(axis=0)
        ),
        training_observations=len(clean),
        feature_means={column: 0.0 for column in clean.columns},
        feature_scales={column: 1.0 for column in clean.columns},
    )


def frozen_hmm_from_payload(payload: dict[str, Any]) -> FrozenHMM:
    return FrozenHMM(
        n_regimes=int(payload["n_regimes"]),
        feature_names=tuple(payload["feature_names"]),
        transition=tuple(tuple(row) for row in payload["transition"]),
        intercepts=tuple(payload["intercepts"]),
        coefficients=tuple(
            tuple(row) for row in payload["coefficients"]
        ),
        variances=tuple(payload["variances"]),
        initial_probabilities=tuple(payload["initial_probabilities"]),
        raw_to_label={
            int(key): str(value)
            for key, value in payload["raw_to_label"].items()
        },
        converged=bool(payload["converged"]),
        log_likelihood=float(payload["log_likelihood"]),
        expected_durations=tuple(payload["expected_durations"]),
        training_state_occupancy=tuple(
            payload["training_state_occupancy"]
        ),
        training_observations=int(payload["training_observations"]),
        feature_means={
            str(key): float(value)
            for key, value in payload["feature_means"].items()
        },
        feature_scales={
            str(key): float(value)
            for key, value in payload["feature_scales"].items()
        },
    )


def validate_parameter_names(names: list[str]) -> bool:
    patterns = (
        r"p\[\d+->\d+\]",
        r"const\[\d+\]",
        r"x\d+\[\d+\]",
        r"sigma2\[\d+\]",
    )
    return all(
        any(re.fullmatch(pattern, name) for pattern in patterns)
        for name in names
    )
