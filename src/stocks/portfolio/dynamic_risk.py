from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class CapitalTier:
    name: str
    minimum: Decimal
    maximum: Decimal | None
    maximum_positions: int
    normal_positions: tuple[int, int]
    maximum_active_swings: int
    maximum_position_weight: float
    base_risk_per_trade: float
    maximum_risk_per_trade: float
    maximum_cluster_positions: int


CAPITAL_TIERS = (
    CapitalTier(
        "BELOW_ECONOMIC_MINIMUM",
        Decimal("0"),
        Decimal("1000"),
        1,
        (0, 1),
        1,
        0.30,
        0.006,
        0.009,
        1,
    ),
    CapitalTier(
        "EUR_1K_TO_10K",
        Decimal("1000"),
        Decimal("10000"),
        5,
        (3, 5),
        3,
        0.30,
        0.007,
        0.011,
        2,
    ),
    CapitalTier(
        "EUR_10K_TO_25K",
        Decimal("10000"),
        Decimal("25000"),
        7,
        (4, 7),
        5,
        0.25,
        0.0065,
        0.010,
        3,
    ),
    CapitalTier(
        "EUR_25K_TO_50K",
        Decimal("25000"),
        Decimal("50000"),
        10,
        (6, 10),
        7,
        0.20,
        0.006,
        0.009,
        3,
    ),
    CapitalTier(
        "EUR_50K_TO_100K",
        Decimal("50000"),
        Decimal("100000"),
        12,
        (8, 12),
        8,
        0.15,
        0.005,
        0.008,
        3,
    ),
    CapitalTier(
        "EUR_100K_TO_250K",
        Decimal("100000"),
        Decimal("250000"),
        15,
        (10, 15),
        10,
        0.125,
        0.0045,
        0.007,
        4,
    ),
    CapitalTier(
        "EUR_250K_TO_500K",
        Decimal("250000"),
        Decimal("500000"),
        18,
        (12, 18),
        12,
        0.10,
        0.004,
        0.0065,
        4,
    ),
    CapitalTier(
        "ABOVE_EUR_500K",
        Decimal("500000"),
        None,
        25,
        (15, 25),
        15,
        0.075,
        0.0035,
        0.006,
        5,
    ),
)


def level_one_canary_risk_budget(
    *,
    account_equity_eur: Decimal,
    normal_risk_budget_eur: Decimal,
    configured_canary_risk_pct: Decimal,
    configured_absolute_cap_eur: Decimal,
) -> Decimal:
    """Return a conservative Level-1 subset of the normal risk budget."""

    values = (
        account_equity_eur,
        normal_risk_budget_eur,
        configured_canary_risk_pct,
        configured_absolute_cap_eur,
    )
    if any(not value.is_finite() or value < 0 for value in values):
        return Decimal("0")
    if account_equity_eur <= 0 or configured_canary_risk_pct > 1:
        return Decimal("0")
    return max(
        Decimal("0"),
        min(
            normal_risk_budget_eur,
            account_equity_eur * configured_canary_risk_pct,
            configured_absolute_cap_eur,
        ),
    )


def build_dynamic_risk_state(
    *,
    equity_eur: Decimal | None,
    equity_history: list[tuple[datetime, Decimal]],
    daily_pnl: list[tuple[date, Decimal]],
    candidates: list[dict[str, Any]],
    policy: dict[str, Any],
    technical_regime: dict[str, Any],
    macro: dict[str, Any],
    capital_level: int,
) -> dict[str, Any]:
    equity_observed = equity_eur is not None and equity_eur > 0
    tier = _capital_tier(equity_eur or Decimal("0"))
    small_policy = policy.get("small_account_whole_share", {})
    small_account_mode = (
        equity_observed
        and bool(small_policy.get("enabled", False))
        and (equity_eur or Decimal("0"))
        <= Decimal(str(small_policy.get("maximum_equity_eur", 0)))
    )
    base_risk_per_trade = tier.base_risk_per_trade
    maximum_risk_per_trade = tier.maximum_risk_per_trade
    maximum_position_weight = tier.maximum_position_weight
    maximum_etf_position_weight = tier.maximum_position_weight
    maximum_positions = tier.maximum_positions
    minimum_meaningful_target_weight = 0.0
    maximum_heat = min(
        float(policy["portfolio"]["maximum_portfolio_heat"]),
        0.05,
    )
    if small_account_mode:
        base_risk_per_trade = min(
            0.02,
            max(
                tier.base_risk_per_trade,
                float(small_policy.get("base_risk_per_trade", 0)),
            ),
        )
        maximum_risk_per_trade = min(
            0.025,
            max(
                base_risk_per_trade,
                float(small_policy.get("maximum_risk_per_trade", 0)),
            ),
        )
        maximum_position_weight = min(
            0.50,
            max(
                tier.maximum_position_weight,
                float(small_policy.get("maximum_stock_weight", 0)),
            ),
        )
        maximum_etf_position_weight = min(
            0.50,
            max(
                maximum_position_weight,
                float(small_policy.get("maximum_etf_weight", 0)),
            ),
        )
        maximum_positions = min(
            tier.maximum_positions,
            int(
                small_policy.get(
                    "maximum_positions", tier.maximum_positions
                )
            ),
        )
        minimum_meaningful_target_weight = min(
            maximum_position_weight,
            max(
                0.0,
                float(
                    small_policy.get(
                        "minimum_meaningful_target_weight", 0
                    )
                ),
            ),
        )
        maximum_heat = min(
            0.08,
            max(
                maximum_heat,
                float(small_policy.get("maximum_portfolio_heat", 0)),
            ),
        )
    eligible = [
        row
        for row in candidates
        if not row.get("research_allocation_blockers")
        and float(row.get("opportunity_score", 0))
        >= float(policy["ranking"]["minimum_allocation_score"])
    ]
    signal_quality = _average(
        [float(row.get("opportunity_score", 0)) for row in eligible]
    )
    diversification = _diversification_score(eligible)
    regime_confidence = _regime_confidence(technical_regime, macro)
    risk_budget_positions = max(
        1,
        math.floor(maximum_heat / max(base_risk_per_trade, 1e-9)),
    )
    tier_limit = min(
        maximum_positions,
        int(policy["portfolio"]["maximum_positions"]),
        risk_budget_positions,
    )
    raw_dynamic_limit = round(
        tier_limit
        * regime_confidence
        * signal_quality
        * diversification
    )
    dynamic_limit = (
        min(len(eligible), tier_limit, max(1, raw_dynamic_limit))
        if eligible
        else 0
    )
    scarcity = min(1.0, len(eligible) / max(tier_limit, 1))

    drawdown = _drawdown_state(equity_history)
    velocity = _drawdown_velocity(equity_history)
    losses = _loss_streak(daily_pnl)
    loss_guard = _loss_guard_state(
        equity_eur=equity_eur,
        daily_pnl=daily_pnl,
    )
    recovery = _recovery_state(equity_history)
    data_quality = _data_quality_multiplier(
        equity_observed, technical_regime, macro
    )
    combined = math.prod(
        (
            drawdown["multiplier"],
            velocity["multiplier"],
            losses["multiplier"],
            loss_guard["multiplier"],
            recovery["multiplier"],
            data_quality,
        )
    )
    operational_limit = _operational_position_limit(
        capital_level, tier_limit
    )
    return {
        "schema": "dynamic_portfolio_risk_state_v1",
        "status": "GO" if equity_observed else "EQUITY_UNAVAILABLE",
        "equity_observed": equity_observed,
        "equity_band": tier.name,
        "exact_equity_public": False,
        "capital_level": capital_level,
        "tier_maximum_positions": tier.maximum_positions,
        "risk_budget_maximum_positions": risk_budget_positions,
        "dynamic_research_maximum_positions": dynamic_limit,
        "operational_maximum_positions": operational_limit,
        "normal_position_range": list(tier.normal_positions),
        "maximum_active_swings": tier.maximum_active_swings,
        "maximum_position_weight": maximum_position_weight,
        "maximum_etf_position_weight": maximum_etf_position_weight,
        "base_risk_per_trade": base_risk_per_trade,
        "maximum_risk_per_trade": maximum_risk_per_trade,
        "maximum_portfolio_heat": maximum_heat,
        "minimum_meaningful_target_weight": (
            minimum_meaningful_target_weight
        ),
        "small_account_whole_share_mode": small_account_mode,
        "maximum_cluster_positions": tier.maximum_cluster_positions,
        "eligible_candidate_count": len(eligible),
        "signal_quality": round(signal_quality, 8),
        "signal_scarcity_multiplier": round(scarcity, 8),
        "diversification_score": round(diversification, 8),
        "regime_confidence": round(regime_confidence, 8),
        "portfolio_drawdown_pct": drawdown["value"],
        "portfolio_drawdown_status": drawdown["status"],
        "drawdown_velocity_per_day": velocity["value"],
        "drawdown_velocity_status": velocity["status"],
        "recovery_ratio": recovery["value"],
        "recovery_status": recovery["status"],
        "consecutive_loss_sessions": losses["count"],
        "loss_streak_status": losses["status"],
        "loss_guard": loss_guard,
        "new_entries_allowed": loss_guard["new_entries_allowed"],
        "risk_reducing_actions_allowed": True,
        "multipliers": {
            "drawdown": drawdown["multiplier"],
            "drawdown_velocity": velocity["multiplier"],
            "consecutive_loss": losses["multiplier"],
            "loss_guard": loss_guard["multiplier"],
            "recovery": recovery["multiplier"],
            "data_quality": data_quality,
            "combined": round(combined, 8),
        },
        "high_water_mark_observed": drawdown["status"] == "GO",
        "high_water_mark_value_public": False,
        "cash_is_valid_allocation": True,
        "automatic_risk_increase": False,
        "automatic_risk_reduction": True,
        "execution_authority": "NONE",
    }


def _capital_tier(equity: Decimal) -> CapitalTier:
    for tier in CAPITAL_TIERS:
        if equity >= tier.minimum and (
            tier.maximum is None or equity < tier.maximum
        ):
            return tier
    return CAPITAL_TIERS[0]


def _drawdown_state(
    history: list[tuple[datetime, Decimal]],
) -> dict[str, Any]:
    daily = _daily_last(history)
    if len(daily) < 2:
        return {"status": "INSUFFICIENT_HISTORY", "value": None, "multiplier": 1.0}
    values = [float(value) for _, value in daily if value > 0]
    if len(values) < 2:
        return {"status": "INSUFFICIENT_HISTORY", "value": None, "multiplier": 1.0}
    high_water = max(values)
    drawdown = max(0.0, 1.0 - values[-1] / high_water)
    return {
        "status": "GO",
        "value": round(drawdown, 8),
        "multiplier": _drawdown_multiplier(drawdown),
    }


def _drawdown_multiplier(drawdown: float) -> float:
    if drawdown < 0.05:
        return 1.0
    if drawdown < 0.08:
        return 0.90
    if drawdown < 0.12:
        return 0.75
    if drawdown < 0.16:
        return 0.55
    if drawdown < 0.20:
        return 0.30
    return 0.0


def _drawdown_velocity(
    history: list[tuple[datetime, Decimal]],
) -> dict[str, Any]:
    daily = _daily_last(history)
    if len(daily) < 3:
        return {"status": "INSUFFICIENT_HISTORY", "value": None, "multiplier": 1.0}
    recent = daily[-5:]
    running_high = max(float(value) for _, value in daily)
    first_dd = max(0.0, 1.0 - float(recent[0][1]) / running_high)
    last_dd = max(0.0, 1.0 - float(recent[-1][1]) / running_high)
    span = max((recent[-1][0] - recent[0][0]).days, 1)
    velocity = (last_dd - first_dd) / span
    multiplier = 1.0
    if velocity >= 0.02:
        multiplier = 0.25
    elif velocity >= 0.01:
        multiplier = 0.50
    elif velocity >= 0.005:
        multiplier = 0.75
    return {
        "status": "GO",
        "value": round(velocity, 8),
        "multiplier": multiplier,
    }


def _loss_streak(
    history: list[tuple[date, Decimal]],
) -> dict[str, Any]:
    latest_by_day: dict[date, Decimal] = {}
    for session_date, value in sorted(history, key=lambda row: row[0]):
        latest_by_day[session_date] = value
    if len(latest_by_day) < 2:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "count": 0,
            "multiplier": 1.0,
        }
    count = 0
    for value in reversed(list(latest_by_day.values())):
        if value < 0:
            count += 1
        else:
            break
    return {
        "status": "GO",
        "count": count,
        "multiplier": round(max(0.5, 1.0 - 0.1 * count), 8),
    }


def _loss_guard_state(
    *,
    equity_eur: Decimal | None,
    daily_pnl: list[tuple[date, Decimal]],
) -> dict[str, Any]:
    if equity_eur is None or equity_eur <= 0 or not daily_pnl:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "daily_loss_ratio": None,
            "rolling_week_loss_ratio": None,
            "rolling_month_loss_ratio": None,
            "new_entries_allowed": False,
            "position_review_required": False,
            "healthy_position_forced_liquidation": False,
            "multiplier": 0.0,
            "reason_codes": ["LOSS_GUARD_HISTORY_REQUIRED"],
        }

    latest_by_day: dict[date, Decimal] = {}
    for session_date, value in sorted(daily_pnl, key=lambda row: row[0]):
        latest_by_day[session_date] = value
    values = list(latest_by_day.values())
    equity = float(equity_eur)
    daily_ratio = float(values[-1]) / equity
    week_ratio = float(sum(values[-5:], Decimal("0"))) / equity
    month_ratio = float(sum(values[-21:], Decimal("0"))) / equity

    reasons: list[str] = []
    multiplier = 1.0
    new_entries_allowed = True
    review_required = False
    if daily_ratio <= -0.025:
        reasons.append("DAILY_LOSS_RISK_REVIEW")
        multiplier = 0.0
        new_entries_allowed = False
        review_required = True
    elif daily_ratio <= -0.015:
        reasons.append("DAILY_NEW_ENTRY_LIMIT_REACHED")
        multiplier = 0.0
        new_entries_allowed = False
    if week_ratio <= -0.04:
        reasons.append("ROLLING_WEEK_LOSS_THROTTLE")
        multiplier = min(multiplier, 0.50)
    if month_ratio <= -0.08:
        reasons.append("ROLLING_MONTH_DEFENSIVE_THROTTLE")
        multiplier = min(multiplier, 0.25)
    if not reasons:
        reasons.append("LOSS_GUARDS_CLEAR")

    return {
        "status": "GO",
        "daily_loss_ratio": round(daily_ratio, 8),
        "rolling_week_loss_ratio": round(week_ratio, 8),
        "rolling_month_loss_ratio": round(month_ratio, 8),
        "new_entries_allowed": new_entries_allowed,
        "position_review_required": review_required,
        "healthy_position_forced_liquidation": False,
        "multiplier": multiplier,
        "reason_codes": reasons,
    }


def _recovery_state(
    history: list[tuple[datetime, Decimal]],
) -> dict[str, Any]:
    daily = _daily_last(history)
    if len(daily) < 3:
        return {"status": "INSUFFICIENT_HISTORY", "value": None, "multiplier": 1.0}
    values = [float(value) for _, value in daily]
    high_index = max(range(len(values)), key=values.__getitem__)
    if high_index == len(values) - 1:
        return {"status": "AT_HIGH_WATER_MARK", "value": 1.0, "multiplier": 1.0}
    post_high = values[high_index:]
    low = min(post_high)
    high = values[high_index]
    if high <= low:
        return {"status": "AT_HIGH_WATER_MARK", "value": 1.0, "multiplier": 1.0}
    ratio = min(1.0, max(0.0, (values[-1] - low) / (high - low)))
    if ratio < 0.25:
        multiplier = 0.50
    elif ratio < 0.50:
        multiplier = 0.65
    elif ratio < 0.75:
        multiplier = 0.80
    else:
        multiplier = 1.0
    return {"status": "GO", "value": round(ratio, 8), "multiplier": multiplier}


def _daily_last(
    history: list[tuple[datetime, Decimal]],
) -> list[tuple[date, Decimal]]:
    latest: dict[date, tuple[datetime, Decimal]] = {}
    for timestamp, value in sorted(history, key=lambda row: row[0]):
        if value <= 0:
            continue
        latest[timestamp.date()] = (timestamp, value)
    return [(day, item[1]) for day, item in sorted(latest.items())]


def _regime_confidence(
    technical_regime: dict[str, Any], macro: dict[str, Any]
) -> float:
    technical = 0.8 if technical_regime.get("status") == "GO" else 0.5
    macro_confidence = float(
        macro.get("regime", {}).get(
            "confidence",
            macro.get("cycle_clock", {}).get("confidence", 0.5),
        )
        or 0.5
    )
    return _clip(0.6 * technical + 0.4 * macro_confidence)


def _diversification_score(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    count = len(candidates)
    ratios = []
    for key in ("sector", "region", "sleeve"):
        values = {str(row.get(key) or "UNKNOWN") for row in candidates}
        ratios.append(min(1.0, len(values) / count))
    return _average(ratios)


def _data_quality_multiplier(
    equity_observed: bool,
    technical_regime: dict[str, Any],
    macro: dict[str, Any],
) -> float:
    if not equity_observed:
        return 0.0
    score = 1.0
    if technical_regime.get("status") != "GO":
        score *= 0.75
    macro_quality = str(macro.get("data_quality", {}).get("status", "UNKNOWN"))
    if macro_quality == "DATA_INCOMPLETE":
        score *= 0.85
    elif macro_quality not in {"GO", "COMPLETE"}:
        score *= 0.75
    return round(score, 8)


def _operational_position_limit(capital_level: int, tier_limit: int) -> int:
    level_limits = {0: 0, 1: 1, 2: 3, 3: 5, 4: 15, 5: 30}
    return min(tier_limit, level_limits.get(capital_level, 0))


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))
