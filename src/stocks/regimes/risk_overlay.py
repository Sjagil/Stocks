from __future__ import annotations

from typing import Mapping

import pandas as pd


def regime_multiplier(
    probabilities: pd.DataFrame,
    state_multipliers: Mapping[str, float],
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> pd.Series:
    missing = sorted(set(probabilities.columns) - set(state_multipliers))
    if missing:
        raise ValueError(f"MISSING_HMM_STATE_MULTIPLIERS:{','.join(missing)}")
    values = sum(
        probabilities[column] * float(state_multipliers[column])
        for column in probabilities.columns
    )
    return values.clip(lower=minimum, upper=maximum).rename(
        "regime_multiplier"
    )


def allowed_trade_risk(
    base_risk: float,
    multiplier: float,
    drawdown_multiplier: float,
) -> float:
    if base_risk < 0 or not 0 <= multiplier <= 1:
        raise ValueError("INVALID_HMM_RISK_INPUT")
    if not 0 <= drawdown_multiplier <= 1:
        raise ValueError("INVALID_DRAWDOWN_MULTIPLIER")
    return base_risk * multiplier * drawdown_multiplier


def rotate_weights(
    current: Mapping[str, float],
    target: Mapping[str, float],
    *,
    absolute_buffer: float = 0.05,
) -> dict[str, float]:
    if absolute_buffer < 0:
        raise ValueError("INVALID_ROTATION_BUFFER")
    output = {}
    for category in sorted(set(current) | set(target)):
        old = float(current.get(category, 0.0))
        proposed = float(target.get(category, 0.0))
        output[category] = old if abs(proposed - old) < absolute_buffer else proposed
    return output
