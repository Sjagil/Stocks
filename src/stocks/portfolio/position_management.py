from __future__ import annotations

import math
from typing import Any


def evaluate_long_position(
    *,
    entry_price: float,
    current_price: float,
    initial_stop: float,
    previous_stop: float,
    peak_price: float,
    atr: float,
    structural_stop: float | None,
    trend_strength: float,
    volatility_regime: float,
    first_target_taken: bool = False,
    second_target_taken: bool = False,
) -> dict[str, Any]:
    values = (
        entry_price,
        current_price,
        initial_stop,
        previous_stop,
        peak_price,
        atr,
        trend_strength,
        volatility_regime,
    )
    if not all(math.isfinite(value) for value in values):
        return _blocked("NON_FINITE_POSITION_INPUT")
    if min(entry_price, current_price, atr) <= 0:
        return _blocked("POSITIVE_PRICE_AND_ATR_REQUIRED")
    if initial_stop <= 0 or initial_stop >= entry_price:
        return _blocked("VALID_INITIAL_LONG_STOP_REQUIRED")

    initial_risk = entry_price - initial_stop
    peak = max(entry_price, current_price, peak_price)
    current_r = (current_price - entry_price) / initial_risk
    peak_r = (peak - entry_price) / initial_risk
    peak_profit = max(0.0, peak - entry_price)
    current_profit = max(0.0, current_price - entry_price)
    giveback = (
        max(0.0, min(1.0, (peak_profit - current_profit) / peak_profit))
        if peak_profit > 0 and peak_r >= 1.5
        else 0.0
    )

    trend = _clip(trend_strength)
    volatility = _clip(volatility_regime)
    trailing_k = 2.0 + 0.5 * trend + 0.5 * volatility
    trailing_stop = peak - trailing_k * atr
    valid_structural = (
        structural_stop
        if structural_stop is not None
        and math.isfinite(structural_stop)
        and 0 < structural_stop < current_price
        else initial_stop
    )
    previous = max(initial_stop, previous_stop)
    proposed_stop = max(initial_stop, previous, valid_structural)
    trailing_active = peak_r >= 1.5
    if trailing_active:
        proposed_stop = max(proposed_stop, trailing_stop)

    target_1 = entry_price + 2.0 * initial_risk
    target_2 = entry_price + 3.5 * initial_risk
    action = "HOLD"
    reasons: list[str] = []
    next_first_target_taken = first_target_taken
    next_second_target_taken = second_target_taken

    if current_price <= previous:
        action = "EXIT"
        reasons.append("PROTECTIVE_STOP_BREACHED")
    elif peak_r >= 1.5 and giveback >= 0.55:
        action = "EXIT"
        reasons.append("PROFIT_GIVEBACK_55_PERCENT")
    elif peak_r >= 1.5 and giveback >= 0.40:
        action = "REDUCE_50"
        reasons.append("PROFIT_GIVEBACK_40_PERCENT")
    elif peak_r >= 1.5 and giveback >= 0.25 and trend < 0.50:
        action = "TAKE_PARTIAL_25"
        reasons.extend(
            ("PROFIT_GIVEBACK_25_PERCENT", "TREND_STRENGTH_WEAKENED")
        )
    elif current_price >= target_2 and not second_target_taken:
        action = "TAKE_PARTIAL_50"
        reasons.append("SECOND_PROFIT_OBJECTIVE_REACHED")
        next_first_target_taken = True
        next_second_target_taken = True
    elif current_price >= target_1 and not first_target_taken:
        action = "TAKE_PARTIAL_25"
        reasons.append("FIRST_PROFIT_OBJECTIVE_REACHED")
        next_first_target_taken = True
    elif trailing_active and proposed_stop > previous + 1e-12:
        action = "UPDATE_TRAILING_STOP"
        reasons.append("RISK_ONLY_TRAILING_STOP_RAISED")
    else:
        reasons.append("POSITION_WITHIN_MANAGEMENT_BAND")

    return {
        "schema": "long_position_management_decision_v1",
        "status": "GO",
        "action": action,
        "reason_codes": reasons,
        "current_r": round(current_r, 8),
        "peak_r": round(peak_r, 8),
        "profit_giveback": round(giveback, 8),
        "initial_risk_per_share": round(initial_risk, 8),
        "peak_price": round(peak, 8),
        "target_1": round(target_1, 8),
        "target_2": round(target_2, 8),
        "previous_stop": round(previous, 8),
        "proposed_stop": round(proposed_stop, 8),
        "trailing_stop": round(trailing_stop, 8),
        "trailing_atr_multiplier": round(trailing_k, 8),
        "trailing_active": trailing_active,
        "first_target_taken": next_first_target_taken,
        "second_target_taken": next_second_target_taken,
        "automatic_execution_allowed": False,
        "broker_write_calls": 0,
        "execution_authority": "NONE",
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "long_position_management_decision_v1",
        "status": "DATA_BLOCKED",
        "action": "NO_ACTION",
        "reason_codes": [reason],
        "automatic_execution_allowed": False,
        "broker_write_calls": 0,
        "execution_authority": "NONE",
    }


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
