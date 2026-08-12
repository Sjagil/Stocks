from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from stocks.ibkr.paper_execution.operator_completion import (
    load_operator_completion_evidence,
)
from stocks.portfolio.manager import active_portfolio_command


LEVELS_PATH = Path("config/capital_scaling/levels_v1.json")
PRIVATE_STATE_PATH = Path("data/capital/private/current-level.json")
APPROVAL_TEMPLATE = "PROMOTE CAPITAL LEVEL {level} WITH MANUAL APPROVAL"


def capital_command(
    project_root: Path,
    command: str,
    *,
    level: int | None = None,
    approval: str | None = None,
    reason: str | None = None,
    account_equity_eur: Decimal | None = None,
    net_daily_pnl_eur: Decimal | None = None,
    enforce_daily_target: bool = False,
) -> dict[str, Any]:
    policy = _policy(project_root)
    if command == "status":
        recommendation = _recommendation(project_root, policy)
        state = _state(project_root, policy)
        if state["current_level"] > recommendation["recommended_level"]:
            return _demote(
                project_root,
                policy,
                recommendation["recommended_level"],
                "AUTOMATIC_EVIDENCE_OR_RISK_DEMOTION",
                automatic=True,
            )
        return _publish_status(project_root, policy, state, recommendation)
    if command == "capacity":
        return _capacity(project_root, policy)
    if command == "recommend-level":
        report = _recommendation(project_root, policy)
        _write_json(
            project_root / "output/capital/scaling_recommendation.json",
            report,
        )
        return report
    if command == "daily-target":
        if account_equity_eur is None or net_daily_pnl_eur is None:
            return {
                "schema": "daily_profit_target_v1",
                "status": "DATA_BLOCKED",
                "blockers": [
                    "ACCOUNT_EQUITY_EUR_REQUIRED",
                    "NET_DAILY_PNL_EUR_REQUIRED",
                ],
                "target_type": "SOFT_RISK_THROTTLE",
                "execution_authority": "NONE",
            }
        report = daily_profit_target(
            account_equity_eur,
            net_daily_pnl_eur,
            policy["daily_profit_target"],
            enforcement_active=enforce_daily_target,
        )
        _write_json(
            project_root / "output/capital/daily_profit_target.json",
            report,
        )
        return report
    if command == "promote":
        if level is None:
            raise ValueError("CAPITAL_LEVEL_REQUIRED")
        return _promote(project_root, policy, level, approval)
    if command == "demote":
        if level is None:
            raise ValueError("CAPITAL_LEVEL_REQUIRED")
        return _demote(
            project_root,
            policy,
            level,
            reason or "OPERATOR_DEMOTION",
            automatic=False,
        )
    raise ValueError(f"UNKNOWN_CAPITAL_COMMAND:{command}")


def portfolio_management_command(
    project_root: Path, command: str
) -> dict[str, Any]:
    if command == "capacity":
        return _capacity(project_root, _policy(project_root))
    if command == "p1":
        from stocks.portfolio.p1 import run_p1_readiness

        return run_p1_readiness(project_root)
    return active_portfolio_command(project_root, command)


def allowed_trade_risk(
    account_equity_eur: Decimal,
    base_risk_pct: Decimal,
    *,
    strategy_multiplier: Decimal,
    regime_multiplier: Decimal,
    drawdown_multiplier: Decimal,
    liquidity_multiplier: Decimal,
    data_quality_multiplier: Decimal,
) -> Decimal:
    multipliers = (
        strategy_multiplier,
        regime_multiplier,
        drawdown_multiplier,
        liquidity_multiplier,
        data_quality_multiplier,
    )
    if account_equity_eur < 0 or base_risk_pct < 0:
        raise ValueError("NEGATIVE_CAPITAL_OR_RISK")
    if any(value < 0 or value > 1 for value in multipliers):
        raise ValueError("RISK_MULTIPLIER_OUT_OF_RANGE")
    return account_equity_eur * base_risk_pct * math.prod(multipliers)


def whole_share_quantity(
    allowed_risk_eur: Decimal,
    entry_price_eur: Decimal,
    stop_price_eur: Decimal,
    *,
    cash_cap_eur: Decimal,
    position_cap_eur: Decimal,
    liquidity_cap_eur: Decimal,
) -> int:
    stop_distance = abs(entry_price_eur - stop_price_eur)
    if min(entry_price_eur, stop_distance) <= 0:
        return 0
    risk_quantity = allowed_risk_eur / stop_distance
    notional_cap = min(cash_cap_eur, position_cap_eur, liquidity_cap_eur)
    cash_quantity = notional_cap / entry_price_eur
    return int(
        min(risk_quantity, cash_quantity).to_integral_value(
            rounding=ROUND_DOWN
        )
    )


def capital_level_limits(
    project_root: Path,
    *,
    level: int,
    account_equity_eur: Decimal,
) -> dict[str, Any]:
    """Resolve percentage and absolute capital limits conservatively.

    The smaller of every configured absolute and equity-relative cap wins.
    This is a limit calculation only and never grants execution authority.
    """
    policy = _policy(project_root)
    raw = policy.get("levels", {}).get(str(level))
    if not isinstance(raw, dict):
        raise ValueError("INVALID_CAPITAL_LEVEL")
    if not account_equity_eur.is_finite() or account_equity_eur <= 0:
        raise ValueError("POSITIVE_ACCOUNT_EQUITY_REQUIRED")
    if any(
        bool(policy.get(flag, False))
        for flag in ("margin_enabled", "leverage_enabled", "shorting_enabled")
    ):
        raise ValueError("FORBIDDEN_CAPITAL_POLICY_FLAG")

    exposure_pct = _bounded_decimal(
        raw.get("maximum_exposure_pct", 0),
        lower=Decimal("0"),
        upper=Decimal("1"),
        name="maximum_exposure_pct",
    )
    stock_pct = _bounded_decimal(
        raw.get("maximum_position_pct", exposure_pct),
        lower=Decimal("0"),
        upper=Decimal("0.50"),
        name="maximum_position_pct",
    )
    etf_pct = _bounded_decimal(
        raw.get("maximum_etf_position_pct", stock_pct),
        lower=Decimal("0"),
        upper=Decimal("0.50"),
        name="maximum_etf_position_pct",
    )
    base_risk_pct = _bounded_decimal(
        raw.get("base_risk_per_trade_pct", 0),
        lower=Decimal("0"),
        upper=Decimal("0.02"),
        name="base_risk_per_trade_pct",
    )
    maximum_risk_pct = _bounded_decimal(
        raw.get("maximum_risk_per_trade_pct", base_risk_pct),
        lower=base_risk_pct,
        upper=Decimal("0.025"),
        name="maximum_risk_per_trade_pct",
    )
    heat_pct = _bounded_decimal(
        raw.get("maximum_portfolio_risk_pct", maximum_risk_pct),
        lower=maximum_risk_pct,
        upper=Decimal("0.08"),
        name="maximum_portfolio_risk_pct",
    )
    maximum_positions = int(raw.get("maximum_positions", 0) or 0)
    if maximum_positions < 0 or maximum_positions > 10:
        raise ValueError("maximum_positions_OUT_OF_RANGE")

    maximum_total_exposure_eur = _minimum_cap(
        account_equity_eur * exposure_pct,
        raw.get("maximum_exposure_eur"),
    )
    maximum_stock_order_eur = _minimum_cap(
        account_equity_eur * stock_pct,
        raw.get("maximum_order_eur"),
    )
    maximum_etf_order_eur = _minimum_cap(
        account_equity_eur * etf_pct,
        raw.get("maximum_order_eur"),
    )
    maximum_risk_eur = _minimum_cap(
        account_equity_eur * maximum_risk_pct,
        raw.get("maximum_risk_eur"),
    )
    return {
        "schema": "resolved_capital_level_limits_v1",
        "status": "GO",
        "capital_level": level,
        "capital_level_name": str(raw.get("name") or "UNKNOWN"),
        "account_equity_eur": str(account_equity_eur),
        "maximum_total_exposure_pct": str(exposure_pct),
        "maximum_total_exposure_eur": str(maximum_total_exposure_eur),
        "maximum_stock_position_pct": str(stock_pct),
        "maximum_etf_position_pct": str(etf_pct),
        "maximum_stock_order_eur": str(maximum_stock_order_eur),
        "maximum_etf_order_eur": str(maximum_etf_order_eur),
        "base_risk_per_trade_pct": str(base_risk_pct),
        "maximum_risk_per_trade_pct": str(maximum_risk_pct),
        "maximum_risk_per_trade_eur": str(maximum_risk_eur),
        "maximum_portfolio_heat_pct": str(heat_pct),
        "maximum_portfolio_heat_eur": str(account_equity_eur * heat_pct),
        "maximum_positions": maximum_positions,
        "minimum_round_trips": int(raw.get("minimum_round_trips", 0) or 0),
        "automatic_orders": bool(raw.get("automatic_orders", False)),
        "primary_sizing_authority": str(
            raw.get("primary_sizing_authority") or "NOT_SPECIFIED"
        ),
        "notional_cap_role": str(
            raw.get("notional_cap_role") or "PRIMARY_LIMIT"
        ),
        "whole_share_required": bool(raw.get("whole_share_required", False)),
        "fractional_shares_allowed": bool(
            raw.get("fractional_shares_allowed", False)
        ),
        "margin_enabled": False,
        "leverage_enabled": False,
        "shorting_enabled": False,
        "execution_authority": "NONE",
    }


def _minimum_cap(relative: Decimal, absolute: Any) -> Decimal:
    if absolute is None:
        return relative
    parsed = Decimal(str(absolute))
    if not parsed.is_finite() or parsed < 0:
        raise ValueError("INVALID_ABSOLUTE_CAP")
    return min(relative, parsed)


def _bounded_decimal(
    value: Any,
    *,
    lower: Decimal,
    upper: Decimal,
    name: str,
) -> Decimal:
    parsed = Decimal(str(value))
    if not parsed.is_finite() or parsed < lower or parsed > upper:
        raise ValueError(f"{name}_OUT_OF_RANGE")
    return parsed


def implementation_shortfall(
    side: str,
    decision_price: Decimal,
    arrival_price: Decimal,
    fill_price: Decimal,
) -> dict[str, float]:
    direction = Decimal("1") if side.upper() == "BUY" else Decimal("-1")
    timing = direction * (arrival_price - decision_price)
    execution = direction * (fill_price - arrival_price)
    total = direction * (fill_price - decision_price)
    return {
        "timing_cost_per_share": float(timing),
        "execution_cost_per_share": float(execution),
        "total_implementation_shortfall_per_share": float(total),
    }


def daily_profit_target(
    account_equity_eur: Decimal,
    net_daily_pnl_eur: Decimal,
    config: dict[str, Any],
    *,
    enforcement_active: bool = False,
) -> dict[str, Any]:
    if account_equity_eur < 0:
        raise ValueError("NEGATIVE_ACCOUNT_EQUITY")
    target_pct = Decimal(str(config["target_pct_of_equity"]))
    if target_pct < 0 or target_pct > Decimal("0.05"):
        raise ValueError("DAILY_PROFIT_TARGET_PCT_OUT_OF_RANGE")
    target = account_equity_eur * target_pct
    minimum = Decimal(str(config.get("minimum_target_eur", 0)))
    target = max(target, minimum)
    maximum_raw = config.get("maximum_target_eur")
    if maximum_raw is not None:
        target = min(target, Decimal(str(maximum_raw)))
    reached = target > 0 and net_daily_pnl_eur >= target
    progress = (
        net_daily_pnl_eur / target if target > 0 else Decimal("0")
    )
    return {
        "schema": "daily_profit_target_v1",
        "status": "GO",
        "generated_at": _now(),
        "session_date": datetime.now(
            ZoneInfo(str(config.get("timezone", "Europe/Amsterdam")))
        ).date().isoformat(),
        "target_type": "SOFT_RISK_THROTTLE",
        "target_pct_of_equity": float(target_pct),
        "daily_profit_target_eur": float(target),
        "net_daily_pnl_eur": float(net_daily_pnl_eur),
        "target_progress_ratio": float(progress),
        "target_reached": reached,
        "enforcement_active": enforcement_active,
        "input_source": "OPERATOR_SUPPLIED",
        "new_entries_allowed": not (reached and enforcement_active),
        "risk_increasing_actions_allowed": not (
            reached and enforcement_active
        ),
        "risk_reducing_exits_allowed": True,
        "force_liquidation": False,
        "risk_chasing_allowed": False,
        "entry_risk_multiplier": (
            0.0 if reached and enforcement_active else 1.0
        ),
        "reset_timezone": str(
            config.get("timezone", "Europe/Amsterdam")
        ),
        "profit_target_is_guarantee": False,
        "execution_authority": "NONE",
    }


def _recommendation(
    project_root: Path, policy: dict[str, Any]
) -> dict[str, Any]:
    from stocks.live.evidence import live_level_two_evidence

    phase9 = _read(project_root / "output/ibkr/phase9/status.json")
    execution = _read(
        project_root / "output/operations/execution-status.json"
    )
    research = _read(project_root / "output/research/phase11_9/status.json")
    reconciliation = _read(
        project_root / "output/ibkr/phase9/reconciliation-audit.json"
    )
    operator_completion = load_operator_completion_evidence(project_root)
    phase9_checks = phase9.get("checks", {})
    fill_canary = bool(
        phase9_checks.get("fill_canary")
        and phase9_checks.get("closing_sell_canary")
    )
    reconciled = (
        bool(phase9_checks.get("reconciliation"))
        and reconciliation.get("reconciliation_status")
        == "PAPER_RECONCILED_EMPTY"
    )
    financial_finalist = bool(research.get("FINANCIAL_FINALIST_GO"))
    critical_incidents = int(
        execution.get("critical_execution_incidents", 0) or 0
    )
    level_one_evidence_pass = (
        fill_canary and reconciled and critical_incidents == 0
    )
    level_two_evidence = live_level_two_evidence(
        project_root,
        minimum_round_trips=int(
            policy["levels"]["2"].get("minimum_round_trips", 5)
        ),
    )
    level_two_evidence_pass = (
        level_one_evidence_pass
        and financial_finalist
        and level_two_evidence.get("status") == "GO"
        and critical_incidents == 0
    )
    recommended = (
        2
        if level_two_evidence_pass
        else 1
        if level_one_evidence_pass
        else 0
    )
    blockers: list[str] = []
    if not level_one_evidence_pass:
        blockers.append("EXECUTION_FILL_CLOSE_CANARY_NOT_PROVEN")
    normal_allocation_blockers = (
        [] if financial_finalist else ["FINANCIAL_FINALIST_NOT_PROVEN"]
    )
    level_two_blockers = list(level_two_evidence.get("blockers", []))
    if not financial_finalist:
        level_two_blockers.append("FINANCIAL_FINALIST_NOT_PROVEN")
    if critical_incidents:
        level_two_blockers.append("CRITICAL_EXECUTION_INCIDENTS_PRESENT")
    return {
        "schema": "capital_scaling_recommendation_v1",
        "status": "GO",
        "recommended_level": recommended,
        "recommended_level_name": policy["levels"][str(recommended)]["name"],
        "recommendation_scope": (
            "CONTROLLED_WHOLE_SHARE_PORTFOLIO"
            if recommended == 2
            else "WHOLE_SHARE_EXECUTION_CANARY_ONLY"
            if recommended == 1
            else "SIGNALS_AND_SHADOW"
        ),
        "promotion_allowed": level_one_evidence_pass,
        "manual_promotion_required": True,
        "blockers": blockers,
        "normal_allocation_eligible": financial_finalist,
        "normal_allocation_blockers": normal_allocation_blockers,
        "level_two_promotion_eligible": level_two_evidence_pass,
        "level_two_blockers": sorted(set(level_two_blockers)),
        "financial_finalist_required_for_level_one": False,
        "financial_finalist_required_for_level_two": True,
        "evidence": {
            "paper_fill_close_canary": fill_canary,
            "paper_reconciliation": reconciled,
            "operator_attested_manual_completion": (
                operator_completion is not None
            ),
            "operator_attestation_effect": (
                "BROKER_STATE_CONTEXT_ONLY_NO_PROMOTION_EFFECT"
                if operator_completion is not None
                else "NONE"
            ),
            "financial_finalist": financial_finalist,
            "critical_execution_incidents": critical_incidents,
            "verified_live_level_one_round_trips": int(
                level_two_evidence.get("verified_round_trip_count", 0) or 0
            ),
            "live_level_two_evidence_status": level_two_evidence.get(
                "status", "NO_GO"
            ),
            "live_level_two_evidence_hash": level_two_evidence.get(
                "content_hash"
            ),
        },
        "automatic_capital_promotion": False,
        "automatic_risk_demotion": True,
    }


def _promote(
    project_root: Path,
    policy: dict[str, Any],
    level: int,
    approval: str | None,
) -> dict[str, Any]:
    if str(level) not in policy["levels"]:
        return {"status": "INVALID_CAPITAL_LEVEL", "requested_level": level}
    state = _state(project_root, policy)
    recommendation = _recommendation(project_root, policy)
    expected = APPROVAL_TEMPLATE.format(level=level)
    blockers = []
    if level <= state["current_level"]:
        blockers.append("PROMOTION_MUST_INCREASE_LEVEL")
    if level != state["current_level"] + 1:
        blockers.append("SEQUENTIAL_CAPITAL_PROMOTION_REQUIRED")
    if level > recommendation["recommended_level"]:
        blockers.append("EVIDENCE_LEVEL_NOT_REACHED")
    if approval != expected:
        blockers.append("EXACT_MANUAL_APPROVAL_REQUIRED")
    if blockers:
        return {
            "schema": "capital_promotion_v1",
            "status": "BLOCKED",
            "requested_level": level,
            "approval_challenge": expected,
            "blockers": blockers,
            "automatic_capital_promotion": False,
        }
    next_state = _new_state(level, policy)
    _write_private_state(project_root, next_state)
    _append_history(project_root, "PROMOTED", next_state, "MANUAL_APPROVAL")
    return _publish_status(
        project_root, policy, next_state, recommendation
    )


def _demote(
    project_root: Path,
    policy: dict[str, Any],
    level: int,
    reason: str,
    *,
    automatic: bool,
) -> dict[str, Any]:
    state = _state(project_root, policy)
    if str(level) not in policy["levels"]:
        return {"status": "INVALID_CAPITAL_LEVEL", "requested_level": level}
    if level >= state["current_level"]:
        return {
            "status": "DEMOTION_MUST_REDUCE_LEVEL",
            "current_level": state["current_level"],
            "requested_level": level,
        }
    next_state = _new_state(level, policy)
    next_state["demotion_reason"] = reason
    next_state["automatic_demotion"] = automatic
    _write_private_state(project_root, next_state)
    _append_history(project_root, "DEMOTED", next_state, reason)
    return _publish_status(
        project_root,
        policy,
        next_state,
        _recommendation(project_root, policy),
    )


def _publish_status(
    project_root: Path,
    policy: dict[str, Any],
    state: dict[str, Any],
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    level = policy["levels"][str(state["current_level"])]
    report = {
        "schema": "capital_scaling_status_v1",
        "status": "GO",
        "CAPITAL_SCALING_ENGINE": True,
        "CURRENT_CAPITAL_LEVEL": state["current_level"],
        "CURRENT_CAPITAL_LEVEL_NAME": level["name"],
        "MAXIMUM_APPROVED_CAPITAL": level.get(
            "maximum_exposure_eur"
        ),
        "CURRENT_TARGET_EXPOSURE": 0.0,
        "DYNAMIC_POSITION_SIZING": True,
        "VOLATILITY_TARGETING": True,
        "CORRELATION_RISK_CONTROL": True,
        "DRAWDOWN_THROTTLE": True,
        "LIQUIDITY_CAPACITY_CONTROL": True,
        "IMPLEMENTATION_SHORTFALL_TRACKING": True,
        "PROFIT_COMPOUNDING": True,
        "AUTOMATIC_RISK_DEMOTION": True,
        "AUTOMATIC_CAPITAL_PROMOTION": False,
        "MARGIN_ENABLED": False,
        "LEVERAGE_ENABLED": False,
        "SHORTING_ENABLED": False,
        "WITHDRAWALS_AVAILABLE": False,
        "level_limits": level,
        "recommendation": recommendation,
        "execution_authority": "NONE",
    }
    _write_json(
        project_root / "output/capital/current_level.json", report
    )
    _write_json(
        project_root / "output/capital/scaling_recommendation.json",
        recommendation,
    )
    _implementation_shortfall_report(project_root)
    return report


def _capacity(
    project_root: Path, policy: dict[str, Any]
) -> dict[str, Any]:
    config = policy["capacity"]
    window = int(config["average_daily_value_window"])
    order_rate = float(config["maximum_order_participation_rate"])
    daily_rate = float(config["maximum_daily_participation_rate"])
    root = (
        project_root
        / "data/research/critical_trading/yfinance"
    )
    fx = _latest_eur_per_usd(project_root)
    rows = []
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if not {"close", "volume"}.issubset(frame):
            continue
        traded_value = (
            frame["close"].astype(float)
            * frame["volume"].astype(float)
            * fx
        ).replace([math.inf, -math.inf], pd.NA).dropna()
        if traded_value.empty:
            continue
        adv = float(traded_value.tail(window).mean())
        if not math.isfinite(adv) or adv <= 0:
            continue
        rows.append(
            {
                "symbol": path.stem,
                "average_daily_traded_value_eur": adv,
                "maximum_order_value_eur": adv * order_rate,
                "maximum_daily_value_eur": adv * daily_rate,
                "capacity_confidence": (
                    "HIGH" if len(traded_value) >= window else "LOW"
                ),
                "capacity_limiting_factor": "ADV_PARTICIPATION",
            }
        )
    rows.sort(
        key=lambda row: float(str(row["maximum_order_value_eur"])),
        reverse=True,
    )
    report = {
        "schema": "capital_capacity_report_v1",
        "status": "GO" if rows else "DATA_BLOCKED",
        "maximum_order_participation_rate": order_rate,
        "maximum_daily_participation_rate": daily_rate,
        "instrument_count": len(rows),
        "instruments": rows,
        "execution_authority": "NONE",
    }
    _write_json(
        project_root / "output/capital/capacity_report.json", report
    )
    _write_json(
        project_root / "output/capital/liquidity_report.json", report
    )
    return report


def _target_allocation(
    candidates: list[dict[str, Any]],
    state: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    level = policy["levels"][str(state["current_level"])]
    maximum = float(level.get("maximum_exposure_pct") or 0)
    executable = [
        row
        for row in candidates
        if row.get("proposal_status") == "VALID_SIGNAL_EXECUTABLE"
    ]
    score_total = sum(max(float(row.get("score", 0)), 0) for row in executable)
    allocations = []
    for row in executable:
        raw = (
            max(float(row.get("score", 0)), 0) / score_total
            if score_total
            else 0
        )
        cap = float(level.get("maximum_position_pct") or maximum)
        allocations.append(
            {
                "ticker": row.get("ticker"),
                "strategy_id": row.get("strategy_id"),
                "target_weight": min(raw * maximum, cap),
                "target_quantity": int(row.get("target_quantity", 0)),
            }
        )
    target_exposure = sum(
        float(str(row["target_weight"])) for row in allocations
    )
    return {
        "schema": "target_allocation_v1",
        "status": "GO",
        "capital_level": state["current_level"],
        "target_exposure_pct": target_exposure,
        "cash_target_pct": max(0.0, 1.0 - target_exposure),
        "allocations": allocations,
        "authority": "NONE",
    }


def _exposure_report(
    target: dict[str, Any],
    state: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    level = policy["levels"][str(state["current_level"])]
    return {
        "schema": "portfolio_exposure_report_v1",
        "status": "GO",
        "capital_level": state["current_level"],
        "target_gross_exposure_pct": target["target_exposure_pct"],
        "maximum_gross_exposure_pct": level.get(
            "maximum_exposure_pct", 0
        ),
        "maximum_position_pct": level.get("maximum_position_pct", 0),
        "maximum_strategy_weight": level.get(
            "maximum_strategy_weight", 0
        ),
        "maximum_family_weight": level.get(
            "maximum_strategy_family_weight", 0
        ),
        "maximum_sector_weight": level.get("maximum_sector_weight", 0),
        "maximum_region_weight": level.get("maximum_region_weight", 0),
        "margin_enabled": False,
        "leverage_enabled": False,
        "shorting_enabled": False,
    }


def _risk_report(
    project_root: Path, target: dict[str, Any]
) -> dict[str, Any]:
    weights = {
        str(row["ticker"]): float(row["target_weight"])
        for row in target["allocations"]
        if float(row["target_weight"]) > 0
    }
    return {
        "schema": "portfolio_risk_contributions_v1",
        "status": "GO" if weights else "NO_TARGET_POSITIONS",
        "portfolio_volatility": None,
        "risk_contributions": [
            {"ticker": ticker, "target_weight": weight}
            for ticker, weight in sorted(weights.items())
        ],
        "method": "TARGET_WEIGHT_PLACEHOLDER_UNTIL_CONFIRMED_ALLOCATION",
        "authority": "NONE",
    }


def _rebalance_preview(
    target: dict[str, Any],
    capacity: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    caps = {
        row["symbol"]: row["maximum_order_value_eur"]
        for row in capacity.get("instruments", [])
    }
    rows = []
    for allocation in target["allocations"]:
        rows.append(
            {
                **allocation,
                "maximum_capacity_order_eur": caps.get(
                    allocation["ticker"]
                ),
                "action": "OBSERVE_ONLY",
                "broker_submission": False,
            }
        )
    return {
        "schema": "portfolio_rebalance_preview_v1",
        "status": "GO",
        "capital_level": state["current_level"],
        "orders": rows,
        "automatic_submission": False,
        "execution_authority": "NONE",
    }


def _implementation_shortfall_report(
    project_root: Path,
) -> dict[str, Any]:
    observations = (
        project_root
        / "data/capital/private/execution-observations.jsonl"
    )
    rows = []
    if observations.exists():
        for line in observations.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "intent_hash": hashlib.sha256(
                        str(row.get("intent_id", "")).encode()
                    ).hexdigest()[:24],
                    **implementation_shortfall(
                        str(row["side"]),
                        Decimal(str(row["decision_price"])),
                        Decimal(str(row["arrival_price"])),
                        Decimal(str(row["fill_price"])),
                    ),
                }
            )
    report = {
        "schema": "implementation_shortfall_report_v1",
        "status": "GO" if rows else "NO_REAL_FILL_OBSERVATIONS",
        "observation_count": len(rows),
        "observations": rows,
    }
    _write_json(
        project_root / "output/capital/implementation_shortfall.json",
        report,
    )
    return report


def _latest_eur_per_usd(project_root: Path) -> float:
    path = (
        project_root
        / "data/research/phase11_4/private/eurusd.parquet"
    )
    if not path.exists():
        return 1.0
    frame = pd.read_parquet(path).sort_values("date")
    values = frame["usd_per_eur"].dropna()
    return 1.0 / float(values.iloc[-1]) if not values.empty else 1.0


def _policy(project_root: Path) -> dict[str, Any]:
    return _read(project_root / LEVELS_PATH)


def _state(
    project_root: Path, policy: dict[str, Any]
) -> dict[str, Any]:
    path = project_root / PRIVATE_STATE_PATH
    if not path.exists():
        state = _new_state(0, policy)
        _write_private_state(project_root, state)
        return state
    state = _read(path)
    level = int(state.get("current_level", 0))
    if str(level) not in policy["levels"]:
        return _new_state(0, policy)
    return state


def _new_state(level: int, policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "capital_level_private_state_v1",
        "current_level": level,
        "level_name": policy["levels"][str(level)]["name"],
        "approved_at": _now(),
        "automatic_capital_promotion": False,
    }


def _write_private_state(
    project_root: Path, state: dict[str, Any]
) -> None:
    _write_json(project_root / PRIVATE_STATE_PATH, state)


def _append_history(
    project_root: Path,
    event: str,
    state: dict[str, Any],
    reason: str,
) -> None:
    path = project_root / "output/capital/scaling_history.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "capital_level": state["current_level"],
        "level_name": state["level_name"],
        "reason": reason,
        "occurred_at": _now(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 19:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def _now() -> str:
    return datetime.now(UTC).isoformat()
