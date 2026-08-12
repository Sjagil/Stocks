from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stocks.research.autopilot.contracts import stable_hash
from stocks.signals.active_swing import generate_active_swing_candidates
from stocks.signals.contracts import (
    SignalAction,
    SignalLifecycle,
    SignalPlan,
)
from stocks.signals.freshness import (
    evaluate_signal_freshness,
    signal_is_current,
)
from stocks.signals.market_reference import (
    apply_market_reference,
    latest_market_reference,
)
from stocks.signals.storage import SignalStore
from stocks.signals.timeframe_contracts import (
    declared_research_signal_timeframe_contract,
)
from stocks.universe import broad_commodity_symbols, broad_etf_symbols


CANONICAL_SIGNAL_FAMILIES = frozenset(
    {
        "adx_trend",
        "asymmetric_ma",
        "asymmetric_ma_crossover",
        "atr_breakout",
        "bollinger_breakout",
        "breakout_consensus",
        "channel_consensus",
        "commodity_etf_trend",
        "donchian_breakout",
        "ema_pullback",
        "etf_commodity_trend",
        "etf_rotation",
        "keltner_breakout",
        "ma_channel",
        "ma_crossover",
        "macd_trend",
        "momentum_consensus",
        "pullback_consensus",
        "quality_trend_consensus",
        "range_expansion_breakout",
        "risk_adjusted_momentum",
        "robust_trend_consensus",
        "roc_trend",
        "rsi2_adx_pullback",
        "stochastic_trend_pullback",
        "time_series_momentum",
        "trend_consensus",
        "triple_ma_trend",
        "volatility_contraction_breakout",
        "volume_breakout",
    }
)
DEFAULT_SIGNAL_PUBLICATION_BUDGET = 5_000
ACTIVE_SWING_SIGNAL_PATH = Path("output/signals/active_swing_15m_signals.json")
TIMEFRAME_ALIASES = {
    "15min": "15m",
    "60m": "1h",
    "1hour": "1h",
    "2hour": "2h",
    "4hour": "4h",
    "daily": "1d",
    "weekly": "1w",
    "monthly": "1mo",
}
SIGNAL_VALIDITY = {
    "15m": timedelta(hours=4),
    "1h": timedelta(hours=12),
    "2h": timedelta(hours=24),
    "4h": timedelta(days=3),
    "1d": timedelta(days=10),
    "1w": timedelta(weeks=6),
    "1mo": timedelta(days=62),
}
INTRADAY_BAR_LENGTH = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
}


def promote_manual_signals(
    project_root: Path, *, strategy_id: str, approval: str
) -> dict[str, Any]:
    candidate = _candidate(project_root, strategy_id)
    expected = os.environ.get("SIGNAL_MANUAL_APPROVAL_PHRASE", "")
    if not expected:
        return _blocked("APPROVAL_PHRASE_NOT_CONFIGURED")
    if approval != expected:
        return _blocked("APPROVAL_PHRASE_MISMATCH")
    if candidate is None:
        return _blocked("STRATEGY_NOT_FOUND")
    if candidate.get("classification") not in {
        "FROZEN_SHADOW",
        "MANUAL_SIGNAL_CANDIDATE",
    }:
        return _blocked("MANUAL_SIGNAL_ELIGIBLE_CLASSIFICATION_REQUIRED")
    strategy_hash = stable_hash(
        {
            "candidate_id": candidate["candidate_id"],
            "strategy_name": candidate["strategy_name"],
            "parameters": candidate["parameters"],
            "timeframe": candidate["timeframe"],
        }
    )
    with SignalStore(project_root) as store:
        store.promote(strategy_id, strategy_hash, candidate)
    return {
        "status": "GO",
        "strategy_id": strategy_id,
        "strategy_dna_hash": strategy_hash,
        "signal_authority": "MANUAL_ACTIONABLE",
        "execution_authority": "NONE",
        "paper_strategy_authority": "NONE",
        "live_strategy_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def signal_scan(
    project_root: Path,
    *,
    universe: str = "all",
    minimum_confidence: float = 0.0,
    minimum_reward_risk: float = 1.5,
    strategy: str | None = None,
    timeframe: str | None = None,
    asset_class: str | None = None,
    maximum_signals: int = DEFAULT_SIGNAL_PUBLICATION_BUDGET,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    with SignalStore(project_root) as store:
        authorities = store.authorities()
        manual_ids = {row["strategy_id"] for row in authorities}
        if strategy:
            authorities = [row for row in authorities if row["strategy_id"] == strategy]
        candidates = {row["candidate_id"]: row for row in _candidates(project_root)}
        selected_candidates = [
            row
            for row in candidates.values()
            if row.get("classification") in {"FROZEN_SHADOW", "MANUAL_SIGNAL_CANDIDATE"}
            and (strategy is None or row["candidate_id"] == strategy)
        ]
        symbols = _universe_symbols(project_root, universe)
        plans: list[dict[str, Any]] = []
        market_references: dict[str, dict[str, Any]] = {}
        for candidate in selected_candidates:
            if timeframe and candidate.get("timeframe") != timeframe:
                continue
            for symbol in symbols:
                plan = _build_signal(
                    project_root,
                    candidate,
                    symbol,
                    now=now,
                    manual_authority=candidate["candidate_id"] in manual_ids,
                )
                if plan is None:
                    continue
                payload = plan.public_payload()
                if float(plan.confidence_score) < minimum_confidence:
                    continue
                if float(plan.reward_risk_1) < minimum_reward_risk:
                    continue
                if asset_class and plan.asset_class != asset_class:
                    continue
                if symbol not in market_references:
                    market_references[symbol] = latest_market_reference(
                        project_root,
                        symbol,
                        now=now,
                    )
                payload = apply_market_reference(
                    project_root,
                    payload,
                    now=now,
                    reference=market_references[symbol],
                )
                store.append_signal(payload)
                plans.append(payload)
        phase11_14_observer_plans = _phase11_14_observer_signals(
            project_root,
            now=now,
        )
        for payload in phase11_14_observer_plans:
            if timeframe and payload.get("timeframe") != timeframe:
                continue
            if float(payload.get("confidence_score", 0.0)) < minimum_confidence:
                continue
            if float(payload.get("reward_risk_1", 0.0)) < minimum_reward_risk:
                continue
            if asset_class and payload.get("asset_class") != asset_class:
                continue
            symbol = str(payload["ticker"])
            if symbol not in market_references:
                market_references[symbol] = latest_market_reference(
                    project_root,
                    symbol,
                    now=now,
                )
            payload = apply_market_reference(
                project_root,
                payload,
                now=now,
                reference=market_references[symbol],
            )
            store.append_signal(payload)
            plans.append(payload)
        active_swing_15m = generate_active_swing_candidates(
            project_root,
            symbols,
            observed_at=now,
        )
        active_swing_rows: list[dict[str, Any]] = []
        for payload in active_swing_15m.get("signals", []):
            if timeframe and payload.get("timeframe") != timeframe:
                continue
            if strategy and payload.get("strategy_id") != strategy:
                continue
            if float(payload.get("confidence_score", 0.0)) < minimum_confidence:
                continue
            if float(payload.get("reward_risk_1", 0.0)) < minimum_reward_risk:
                continue
            contract = _contract(project_root, str(payload["ticker"]))
            payload["contract_identity"] = contract
            payload["asset_class"] = str(contract.get("security_type") or "STK")
            payload["currency"] = str(contract.get("currency") or "USD")
            payload["exchange"] = str(
                contract.get("primary_exchange") or contract.get("exchange") or "UNRESOLVED"
            )
            if not contract:
                payload["risks"] = [
                    *payload.get("risks", []),
                    "CONTRACT_IDENTITY_UNAVAILABLE",
                ]
            if asset_class and payload.get("asset_class") != asset_class:
                continue
            symbol = str(payload["ticker"])
            if symbol not in market_references:
                market_references[symbol] = latest_market_reference(
                    project_root,
                    symbol,
                    now=now,
                )
            payload = apply_market_reference(
                project_root,
                payload,
                now=now,
                reference=market_references[symbol],
            )
            active_swing_rows.append(payload)
    _publish_active_swing_outputs(
        project_root,
        {**active_swing_15m, "signals": active_swing_rows},
    )
    plans.sort(
        key=lambda row: (
            -float(row.get("confidence_score", 0)),
            str(row.get("ticker", "")),
            str(row.get("strategy_id", "")),
        )
    )
    plans = _promote_consensus(plans)
    invalidated_signal_count = sum(row.get("lifecycle_status") == "INVALIDATED" for row in plans)
    active_signal_assets = {
        str(row.get("ticker")) for row in plans if row.get("data_freshness") == "FRESH"
    }
    plans = _fair_signal_limit(plans, max(1, maximum_signals))
    phase11_14_inventory = _phase11_14_observer_inventory(project_root)
    published_assets = {str(row.get("ticker")) for row in plans}
    universe_assets = {str(symbol) for symbol in symbols}
    result = {
        "schema": "manual_signal_scan_v1",
        "status": "GO",
        "generated_at": now.isoformat(),
        "universe": universe,
        "frozen_shadow_strategy_count": len(selected_candidates),
        "phase11_14_observer_strategy_count": phase11_14_inventory["observer_strategy_count"],
        "phase11_14_exploratory_strategy_count": phase11_14_inventory["exploratory_strategy_count"],
        "phase11_14_active_observer_strategy_count": len(
            {str(row.get("strategy_id")) for row in phase11_14_observer_plans}
        ),
        "phase11_14_active_exploratory_strategy_count": len(
            {
                str(row.get("strategy_id"))
                for row in phase11_14_observer_plans
                if row.get("observer_tier") == "EXPLORATORY_FORWARD_OBSERVER"
            }
        ),
        "active_swing_15m_status": active_swing_15m.get("status"),
        "active_swing_15m_hypothesis_count": int(active_swing_15m.get("hypothesis_count", 0)),
        "active_swing_15m_candidate_count": int(active_swing_15m.get("candidate_count", 0)),
        "active_swing_15m_native_ready_symbol_count": int(
            active_swing_15m.get("native_15m_ready_symbol_count", 0)
        ),
        "active_swing_15m_candidate_unit": active_swing_15m.get("candidate_unit"),
        "active_swing_15m_portfolio_eligible": False,
        "active_swing_15m_canonical_money_signal": False,
        "active_swing_15m_canonical_signal_store_appended": False,
        "active_swing_15m_manual_execution_eligible": False,
        "active_swing_15m_execution_authority": "NONE",
        "authorized_strategy_count": len(manual_ids),
        "followed_asset_count": len(symbols),
        "active_signal_asset_count": len(active_signal_assets),
        "asset_without_active_signal_count": len(universe_assets - active_signal_assets),
        "published_asset_count": len(published_assets),
        "published_active_asset_coverage_ratio": round(
            len(published_assets) / max(1, len(active_signal_assets)), 6
        ),
        "covered_asset_count": len(published_assets),
        "covered_strategy_count": len({str(row.get("strategy_id")) for row in plans}),
        "coverage_policy": ("ASSET_FIRST_THEN_STRATEGY_ROUND_ROBIN"),
        "publication_budget": maximum_signals,
        "signal_count": len(plans),
        "price_invalidated_signal_count": invalidated_signal_count,
        "market_reference_policy": {
            "status": "GO",
            "source": "QUALIFIED_INTRADAY_1H_REFERENCE",
            "maximum_age_minutes": 95,
            "executable_quote": False,
            "stop_breach_blocks": True,
            "target_reached_blocks": True,
            "entry_zone_drift_blocks": True,
        },
        "signals": plans,
        "signal_authority": (
            "MANUAL_ACTIONABLE"
            if manual_ids
            else "SHADOW"
            if selected_candidates
            else "SHADOW"
            if active_swing_15m.get("candidate_count", 0)
            else "NONE"
        ),
        "execution_authority": "NONE",
        "automatic_execution": False,
        "broker_calls": 0,
        "orders_generated": 0,
    }
    _publish_signal_outputs(project_root, result, plans)
    return result


def active_swing_scan(
    project_root: Path,
    *,
    symbols: Iterable[str] | None = None,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Refresh only natural 15m candidates without replacing money signals."""
    now = observed_at or datetime.now(UTC)
    selected_symbols = sorted(
        {
            str(symbol).strip().upper()
            for symbol in (symbols or _universe_symbols(project_root, "all"))
            if str(symbol).strip()
        }
    )
    report = generate_active_swing_candidates(
        project_root,
        selected_symbols,
        observed_at=now,
    )
    market_references: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for candidate in report.get("signals", []):
        payload = dict(candidate)
        symbol = str(payload["ticker"]).upper()
        contract = _contract(project_root, symbol)
        payload["contract_identity"] = contract
        payload["asset_class"] = str(contract.get("security_type") or "STK")
        payload["currency"] = str(contract.get("currency") or "USD")
        payload["exchange"] = str(
            contract.get("primary_exchange") or contract.get("exchange") or "UNRESOLVED"
        )
        if not contract:
            payload["risks"] = [
                *payload.get("risks", []),
                "CONTRACT_IDENTITY_UNAVAILABLE",
            ]
        if symbol not in market_references:
            market_references[symbol] = latest_market_reference(
                project_root,
                symbol,
                now=now,
            )
        payload = apply_market_reference(
            project_root,
            payload,
            now=now,
            reference=market_references[symbol],
        )
        rows.append(payload)
    result = {
        **report,
        "signals": rows,
        "dedicated_fast_path": True,
        "canonical_money_signals_replaced": False,
        "canonical_signal_store_appended": False,
        "manual_execution_eligible": False,
        "portfolio_eligible": False,
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "order_calls": 0,
    }
    _publish_active_swing_outputs(project_root, result)
    return result


def signal_asset(project_root: Path, symbol: str) -> dict[str, Any]:
    result = signal_scan(
        project_root,
        maximum_signals=DEFAULT_SIGNAL_PUBLICATION_BUDGET,
    )
    matches = [row for row in result["signals"] if row["ticker"].upper() == symbol.upper()]
    return {
        **{key: value for key, value in result.items() if key != "signals"},
        "symbol": symbol.upper(),
        "signals": matches,
        "signal_count": len(matches),
    }


def signal_list(project_root: Path, mode: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    with SignalStore(project_root) as store:
        history_rows = store.signals()
    rows = history_rows
    current_rows = _latest_scan_signals(project_root)
    if mode == "active":
        rows = [
            row
            for row in _latest_signal_versions(current_rows or rows)
            if row["lifecycle_status"] not in {"CLOSED", "EXPIRED", "CANCELLED", "INVALIDATED"}
            and _signal_not_expired(row, now)
        ]
    elif mode == "expired":
        rows = [
            row
            for row in rows
            if row["lifecycle_status"] == "EXPIRED"
            or datetime.fromisoformat(row["expiration_timestamp"]) < now
        ]
    elif mode == "watchlist":
        rows = [
            row
            for row in _latest_signal_versions(current_rows or rows)
            if row["action"] == "WATCHLIST"
            and row["lifecycle_status"] not in {"CLOSED", "EXPIRED", "CANCELLED", "INVALIDATED"}
            and _signal_not_expired(row, now)
        ]
    return {
        "status": "GO",
        "mode": mode,
        "count": len(rows),
        "signals": rows,
        "current_state_source": (
            "LATEST_SCAN_ARTIFACT"
            if mode in {"active", "watchlist"} and current_rows
            else "PRIVATE_SIGNAL_HISTORY"
        ),
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def signal_explain(project_root: Path, signal_id: str) -> dict[str, Any]:
    with SignalStore(project_root) as store:
        signal = store.signal(signal_id)
    if signal is None:
        return _blocked("SIGNAL_NOT_FOUND")
    return {
        "status": "GO",
        "signal": signal,
        "model_signal_not_guarantee": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def signal_order_plan(
    project_root: Path,
    *,
    signal_id: str,
    capital: Decimal,
    risk: Decimal,
) -> dict[str, Any]:
    if capital <= 0:
        return _blocked("CAPITAL_MUST_BE_POSITIVE")
    if risk <= 0 or risk > Decimal("0.05"):
        return _blocked("RISK_MUST_BE_BETWEEN_ZERO_AND_FIVE_PERCENT")
    with SignalStore(project_root) as store:
        signal = store.signal(signal_id)
    if signal is None:
        return _blocked("SIGNAL_NOT_FOUND")
    try:
        entry = Decimal(str(signal["preferred_entry"]))
        stop = Decimal(str(signal["stop_loss"]))
    except (KeyError, ArithmeticError):
        return _blocked("SIGNAL_ORDER_GEOMETRY_INCOMPLETE")
    if entry <= 0 or stop <= 0 or stop >= entry:
        return _blocked("SIGNAL_ORDER_GEOMETRY_INVALID")
    currency = str(signal.get("currency") or "USD")
    fx_to_eur = _fx_to_eur(project_root, currency)
    entry_eur = entry * fx_to_eur
    unit_risk_eur = (entry - stop) * fx_to_eur
    risk_budget_eur = capital * risk
    by_risk = risk_budget_eur / unit_risk_eur
    by_cash = capital / entry_eur
    quantity = min(by_risk, by_cash).quantize(Decimal("1"), rounding=ROUND_DOWN)
    if quantity <= 0:
        return _blocked("POSITION_SIZE_BELOW_ONE_WHOLE_SHARE")
    estimated_value = quantity * entry_eur
    maximum_loss = quantity * unit_risk_eur
    return {
        "schema": "manual_signal_order_plan_v1",
        "status": "GO",
        "signal_id": signal_id,
        "symbol": signal.get("ticker"),
        "side": "BUY",
        "order_type": "LIMIT",
        "limit_price": str(entry),
        "quantity": str(quantity),
        "estimated_value_eur": str(estimated_value.quantize(Decimal("0.01"))),
        "initial_stop": str(stop),
        "target_1": signal.get("take_profit_1"),
        "target_2": signal.get("take_profit_2"),
        "maximum_planned_loss_eur": str(maximum_loss.quantize(Decimal("0.01"))),
        "risk_budget_eur": str(risk_budget_eur.quantize(Decimal("0.01"))),
        "fx_to_eur": str(fx_to_eur),
        "remaining_cash_eur": str((capital - estimated_value).quantize(Decimal("0.01"))),
        "commission_estimate": "UNAVAILABLE_USE_BROKER_SCHEDULE",
        "sector_exposure_after_trade": ("UNAVAILABLE_WITHOUT_CURRENT_PRIVATE_ACCOUNT_EQUITY"),
        "manual_plan_only": True,
        "model_signal_not_guarantee": True,
        "automatic_submission": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def signal_mark_executed(
    project_root: Path,
    *,
    signal_id: str,
    quantity: Decimal,
    fill_price: Decimal,
) -> dict[str, Any]:
    if quantity <= 0 or fill_price <= 0:
        return _blocked("INVALID_MANUAL_EXECUTION")
    with SignalStore(project_root) as store:
        signal = store.signal(signal_id)
        if signal is None:
            return _blocked("SIGNAL_NOT_FOUND")
        suggested = Decimal(str(signal["preferred_entry"]))
        slippage = (fill_price - suggested) * quantity
        execution_id = store.append_manual_execution(
            signal_id=signal_id,
            event_type="MANUAL_ENTRY",
            quantity=str(quantity),
            fill_price=str(fill_price),
            reason=None,
            payload={"suggested_entry": str(suggested), "slippage": str(slippage)},
            new_status=SignalLifecycle.EXECUTED_MANUALLY.value,
        )
    return {
        "status": "GO",
        "execution_id": execution_id,
        "signal_id": signal_id,
        "manual_execution": True,
        "slippage_value": str(slippage),
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def signal_mark_closed(
    project_root: Path,
    *,
    signal_id: str,
    quantity: Decimal,
    fill_price: Decimal,
    reason: str,
) -> dict[str, Any]:
    if quantity <= 0 or fill_price <= 0 or not reason.strip():
        return _blocked("INVALID_MANUAL_CLOSE")
    with SignalStore(project_root) as store:
        signal = store.signal(signal_id)
        if signal is None:
            return _blocked("SIGNAL_NOT_FOUND")
        entry = Decimal(str(signal["preferred_entry"]))
        actual_pnl = (fill_price - entry) * quantity
        model_exit = Decimal(str(signal["take_profit_1"]))
        model_pnl = (model_exit - entry) * quantity
        execution_id = store.append_manual_execution(
            signal_id=signal_id,
            event_type="MANUAL_CLOSE",
            quantity=str(quantity),
            fill_price=str(fill_price),
            reason=reason,
            payload={
                "actual_pnl": str(actual_pnl),
                "model_pnl_at_tp1": str(model_pnl),
                "deviation": str(actual_pnl - model_pnl),
            },
            new_status=SignalLifecycle.CLOSED.value,
        )
    return {
        "status": "GO",
        "execution_id": execution_id,
        "signal_id": signal_id,
        "actual_pnl": str(actual_pnl),
        "model_pnl_at_tp1": str(model_pnl),
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def signal_export(project_root: Path) -> dict[str, Any]:
    with SignalStore(project_root) as store:
        rows = store.signals()
    result = {
        "status": "GO",
        "signal_count": len(rows),
        "signals": rows,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    _publish_signal_outputs(project_root, result, rows)
    return result


def signal_status(project_root: Path) -> dict[str, Any]:
    with SignalStore(project_root) as store:
        authorities = store.authorities()
        signals = store.signals()
    frozen = [
        row for row in _candidates(project_root) if row.get("classification") == "FROZEN_SHADOW"
    ]
    frozen_families = {str(row.get("family")) for row in frozen if row.get("family")}
    now = datetime.now(UTC)
    open_lifecycle = [
        row
        for row in signals
        if row["lifecycle_status"] not in {"CLOSED", "EXPIRED", "CANCELLED", "INVALIDATED"}
        and _signal_not_expired(row, now)
    ]
    current_rows = _latest_scan_signals(project_root)
    latest = _latest_signal_versions(current_rows or signals)
    active = [
        row
        for row in latest
        if row["lifecycle_status"] not in {"CLOSED", "EXPIRED", "CANCELLED", "INVALIDATED"}
        and _signal_not_expired(row, now)
    ]
    return {
        "schema": "signal_only_status_v1",
        "status": "GO",
        "followed_assets": len(_universe_symbols(project_root, "all")),
        "manual_actionable_strategies": len(authorities),
        "frozen_shadow_strategies": len(frozen),
        "frozen_candidate_family_count": len(frozen_families),
        "canonical_signal_family_count": len(CANONICAL_SIGNAL_FAMILIES),
        "canonical_signal_families": sorted(CANONICAL_SIGNAL_FAMILIES),
        "unimplemented_frozen_candidate_families": sorted(
            frozen_families - CANONICAL_SIGNAL_FAMILIES
        ),
        "active_signals": len(active),
        "signal_history_record_count": len(signals),
        "unexpired_open_history_record_count": len(open_lifecycle),
        "superseded_unexpired_signal_count": len(open_lifecycle) - len(active),
        "active_signal_semantics": ("LATEST_UNEXPIRED_PER_TICKER_STRATEGY_TIMEFRAME"),
        "active_signal_source": (
            "LATEST_SCAN_ARTIFACT" if current_rows else "PRIVATE_SIGNAL_HISTORY_FALLBACK"
        ),
        "SIGNALS_CAN_RUN_WITHOUT_BROKER": True,
        "SIGNALS_INCLUDE_STOP_LOSS": True,
        "SIGNALS_INCLUDE_TAKE_PROFIT": True,
        "MANUAL_EXECUTION_SUPPORTED": True,
        "SIGNAL_AUTHORITY_SEPARATE_FROM_EXECUTION": True,
        "signal_authority": (
            "MANUAL_ACTIONABLE" if authorities else "SHADOW" if frozen else "NONE"
        ),
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
        "private_database": str(project_root / "data" / "signals" / "private" / "signals.sqlite3"),
    }


def _build_signal(
    project_root: Path,
    candidate: dict[str, Any],
    symbol: str,
    *,
    now: datetime,
    manual_authority: bool,
) -> SignalPlan | None:
    timeframe = _normalize_timeframe(candidate.get("timeframe"))
    path = _signal_bar_path(project_root, symbol, timeframe)
    if path is None:
        return None
    frame = _closed_signal_frame(path)
    required = {"session_date", "open", "high", "low", "close"}
    params = _parameters(candidate.get("parameters"))
    minimum_bars = _minimum_signal_bars(params)
    if not required.issubset(frame.columns) or len(frame) < minimum_bars:
        return None
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    last = float(close.iloc[-1])
    atr = _atr(frame, 14)
    family = candidate["family"]
    raw_action, confidence, reasons = _strategy_signal(
        family,
        close,
        high,
        low,
        params,
        open_=frame["open"].astype(float),
        volume=(frame["volume"].astype(float) if "volume" in frame else None),
    )
    if raw_action not in {SignalAction.BUY, SignalAction.STRONG_BUY}:
        return None
    contract = _contract(project_root, symbol)
    currency = str(contract.get("currency") or "USD")
    exchange = str(
        contract.get("primary_exchange")
        or contract.get("primaryExchange")
        or contract.get("exchange")
        or "UNRESOLVED"
    )
    last_row = frame.iloc[-1]
    data_time = _bar_available_at(last_row, timeframe)
    declared_expiration = data_time + SIGNAL_VALIDITY.get(
        timeframe,
        SIGNAL_VALIDITY["1d"],
    )
    exchange_timezone = str(last_row.get("exchange_timezone") or "").strip()
    freshness_evaluation = evaluate_signal_freshness(
        {
            "timeframe": timeframe,
            "data_timestamp": data_time,
            "expiration_timestamp": declared_expiration,
            "exchange_timezone": exchange_timezone,
        },
        now=now,
    )
    expiration = freshness_evaluation["effective_expiration"]
    freshness = str(freshness_evaluation["status"])
    swing_low = float(low.tail(20).min())
    stop = max(swing_low, last - (2.0 * atr))
    if not 0 < stop < last:
        stop = last - (2.0 * atr)
    risk = last - stop
    if risk <= 0:
        return None
    entry_low = last - (0.25 * atr)
    entry_high = last + (0.15 * atr)
    fx = _fx_to_eur(project_root, currency)
    capital = Decimal(os.environ.get("MANUAL_SIGNAL_REFERENCE_CAPITAL_EUR", "10000"))
    risk_pct = Decimal(os.environ.get("MANUAL_SIGNAL_MAX_RISK_PCT", "0.25")) / Decimal("100")
    position_pct = Decimal(os.environ.get("MANUAL_SIGNAL_MAX_POSITION_PCT", "5")) / Decimal("100")
    allowed_risk = capital * risk_pct
    max_notional = capital * position_pct
    risk_eur = Decimal(str(risk)) * fx
    quantity_risk = allowed_risk / risk_eur
    quantity_notional = max_notional / (Decimal(str(last)) * fx)
    quantity = min(quantity_risk, quantity_notional)
    if os.environ.get("MANUAL_SIGNAL_FRACTIONAL_SHARES", "false").lower() != "true":
        quantity = quantity.quantize(Decimal("1"), rounding=ROUND_DOWN)
    else:
        quantity = quantity.quantize(Decimal("0.001"), rounding=ROUND_DOWN)
    risks = ["MODEL_SIGNAL_NOT_GUARANTEE", "GAP_RISK"]
    lifecycle = SignalLifecycle.MANUAL_ACTIONABLE
    action = raw_action
    if not manual_authority:
        risks.append("SHADOW_ONLY_OPERATOR_PROMOTION_REQUIRED")
        lifecycle = SignalLifecycle.WATCHLIST
        action = SignalAction.WATCHLIST
    if not contract:
        risks.append("CONTRACT_IDENTITY_UNAVAILABLE")
        lifecycle = SignalLifecycle.WATCHLIST
        action = SignalAction.WATCHLIST
    if freshness != "FRESH":
        risks.append("STALE_DATA")
        lifecycle = SignalLifecycle.WATCHLIST
        action = SignalAction.WATCHLIST
    if quantity <= 0:
        risks.append("POSITION_SIZE_BELOW_MINIMUM")
        lifecycle = SignalLifecycle.WATCHLIST
        action = SignalAction.WATCHLIST
    strategy_hash = stable_hash(
        {
            "candidate_id": candidate["candidate_id"],
            "strategy_name": candidate["strategy_name"],
            "parameters": candidate["parameters"],
            "timeframe": candidate["timeframe"],
        }
    )
    signal_id = (
        "SIG-"
        + stable_hash(
            {
                "strategy_hash": strategy_hash,
                "ticker": symbol.upper(),
                "data_timestamp": data_time.isoformat(),
                "action": action.value,
            }
        )[:24]
    )

    def q(value: float) -> Decimal:
        return Decimal(str(value)).quantize(Decimal("0.0001"))

    return SignalPlan(
        signal_id=signal_id,
        asset=symbol.upper(),
        ticker=symbol.upper(),
        contract_identity=contract,
        asset_class=str(contract.get("security_type") or "STK"),
        exchange=exchange,
        currency=currency,
        signal_timestamp=now,
        data_timestamp=data_time,
        data_freshness=freshness,
        strategy_id=candidate["candidate_id"],
        strategy_family=str(
            candidate.get("family") or candidate.get("strategy_name") or "UNDECLARED"
        ),
        strategy_dna_hash=strategy_hash,
        strategy_timeframe_contract=declared_research_signal_timeframe_contract(
            project_root,
            candidate,
        ),
        timeframe=timeframe,
        higher_timeframe_context=f"CLOSED_{timeframe.upper()}_BAR",
        action=action,
        current_market_price=q(last),
        preferred_entry=q(last),
        entry_zone_low=q(entry_low),
        entry_zone_high=q(entry_high),
        limit_entry_price=q(last),
        invalidation_level=q(stop),
        stop_loss=q(stop),
        stop_method="ATR_ADJUSTED_20_SESSION_STRUCTURE",
        stop_distance_pct=q(risk / last),
        stop_distance_atr=q(risk / atr),
        take_profit_1=q(last + 1.5 * risk),
        take_profit_2=q(last + 2.5 * risk),
        take_profit_mode="PARTIAL_TARGETS_WITH_TRAILING_EXIT",
        reward_risk_1=Decimal("1.5000"),
        reward_risk_2=Decimal("2.5000"),
        suggested_quantity=quantity,
        maximum_order_value_eur=max_notional.quantize(Decimal("0.01")),
        maximum_planned_loss_eur=(quantity * risk_eur).quantize(Decimal("0.01")),
        estimated_transaction_costs_eur=(
            quantity * Decimal(str(last)) * fx * Decimal("0.001")
        ).quantize(Decimal("0.01")),
        expected_holding_period=_holding_period(timeframe),
        confidence_score=q(confidence),
        regime="TREND_COMPATIBLE",
        reasons=tuple(reasons),
        risks=tuple(risks),
        expiration_timestamp=expiration,
        lifecycle_status=lifecycle,
        automatic_execution_allowed=False,
        source_provider=str(last_row.get("provider") or "YFINANCE"),
        source_interval=str(last_row.get("source_interval") or timeframe),
        bar_origin=str(last_row.get("bar_origin") or "LEGACY_DAILY_CACHE"),
        bar_closed=True,
        exchange_timezone=exchange_timezone,
        signal_freshness_basis=str(freshness_evaluation["freshness_basis"]),
    )


def _normalize_timeframe(value: Any) -> str:
    timeframe = str(value or "1d").strip().lower()
    return TIMEFRAME_ALIASES.get(timeframe, timeframe)


def _signal_bar_path(
    project_root: Path,
    symbol: str,
    timeframe: str,
) -> Path | None:
    interval_root = (
        project_root
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / f"symbol={symbol.upper()}"
        / f"interval={timeframe}"
    )
    preferred_sources = {
        "15m": ("15m",),
        "1h": ("1h",),
        "2h": ("2h", "1h"),
        "4h": ("4h", "1h"),
        "1d": ("1d",),
        "1w": ("1w", "1d"),
        "1mo": ("1mo", "1d"),
    }.get(timeframe, (timeframe,))
    for source in preferred_sources:
        candidate = interval_root / f"source_interval={source}" / "bars.parquet"
        if candidate.exists():
            return candidate
    candidates = sorted(interval_root.glob("source_interval=*/bars.parquet"))
    if candidates:
        return candidates[0]
    if timeframe == "1d":
        legacy = (
            project_root / "data/research/critical_trading/yfinance" / f"{symbol.upper()}.parquet"
        )
        if legacy.exists():
            return legacy
    return None


def _signal_exchange_timezone(
    project_root: Path,
    symbol: str,
    timeframe: str,
) -> str:
    path = _signal_bar_path(project_root, symbol, timeframe)
    if path is None:
        return ""
    try:
        frame = pd.read_parquet(path, columns=["exchange_timezone"])
    except (FileNotFoundError, OSError, ValueError):
        return ""
    values = frame["exchange_timezone"].dropna().astype(str)
    return values.iloc[-1].strip() if not values.empty else ""


def _closed_signal_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"session_date", "open", "high", "low", "close"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    if "quality_status" in frame.columns:
        quality = frame["quality_status"].astype(str).str.startswith("VALIDATED")
        frame = frame.loc[quality]
    for field in ("is_partial", "partial_bucket"):
        if field in frame.columns:
            partial = frame[field].astype("boolean").fillna(False)
            frame = frame.loc[~partial]
    sort_field = "timestamp_utc" if "timestamp_utc" in frame.columns else "session_date"
    frame = frame.dropna(subset=list(required)).sort_values(sort_field)
    return frame.drop_duplicates(subset=[sort_field], keep="last")


def _minimum_signal_bars(params: dict[str, Any]) -> int:
    period_fields = {
        "channel",
        "entry_fast",
        "entry_slow",
        "exit_fast",
        "exit_slow",
        "fast",
        "length",
        "lookback",
        "period",
        "slow",
        "trend_ma",
        "window",
    }
    periods = []
    for key, value in params.items():
        if key not in period_fields:
            continue
        try:
            period = int(value)
        except (TypeError, ValueError):
            continue
        if period > 0:
            periods.append(period)
    return max(60, max(periods, default=50) + 10)


def _bar_available_at(row: pd.Series, timeframe: str) -> datetime:
    field = "timestamp_utc" if row.get("timestamp_utc") is not None else "session_date"
    timestamp = pd.Timestamp(row[field])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime() + INTRADAY_BAR_LENGTH.get(
        timeframe,
        timedelta(0),
    )


def _holding_period(timeframe: str) -> str:
    return {
        "15m": "1-20 sessions; tactical trigger expires within 4 hours",
        "1h": "2-20 closed 1h bars",
        "2h": "2-20 closed 2h bars",
        "4h": "2-20 closed 4h bars",
        "1d": "5-60 sessions",
        "1w": "2-20 closed weeks",
        "1mo": "2-12 closed months",
    }.get(timeframe, "5-60 sessions")


def _signal_not_expired(row: dict[str, Any], now: datetime) -> bool:
    return signal_is_current(row, now=now)


def _latest_signal_versions(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("ticker") or row.get("asset") or "").upper(),
            str(row.get("strategy_id") or ""),
            _normalize_timeframe(row.get("timeframe")),
        )
        prior = latest.get(key)
        if prior is None or _signal_version_timestamp(row) > _signal_version_timestamp(prior):
            latest[key] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("ticker") or row.get("asset") or ""),
            str(row.get("strategy_id") or ""),
            _normalize_timeframe(row.get("timeframe")),
        ),
    )


def _latest_scan_signals(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "output" / "signals" / "latest_signals.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    rows = payload.get("signals", [])
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _signal_version_timestamp(row: dict[str, Any]) -> datetime:
    for field in ("data_timestamp", "signal_timestamp"):
        try:
            value = datetime.fromisoformat(str(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value
    return datetime.min.replace(tzinfo=UTC)


def _strategy_signal(
    family: str,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    params: dict[str, Any],
    *,
    open_: pd.Series | None = None,
    volume: pd.Series | None = None,
) -> tuple[SignalAction, float, list[str]]:
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)
    open_series = open_.astype(float) if open_ is not None else close.shift(1).fillna(close)
    volume_series = (
        volume.astype(float) if volume is not None else pd.Series(0.0, index=close.index)
    )
    if family == "time_series_momentum" and params.get("signal_variant") == "donchian_breakout":
        channel = int(params.get("channel", 20))
        if len(close) <= channel:
            return (
                SignalAction.NO_SIGNAL,
                0.0,
                ["INSUFFICIENT_HISTORY"],
            )
        prior_high = float(high.iloc[-channel - 1 : -1].max())
        breakout = float(close.iloc[-1]) > prior_high
        strength = max(
            0.0,
            min(
                1.0,
                float(close.iloc[-1]) / prior_high - 1.0,
            ),
        )
        return (
            SignalAction.BUY if breakout else SignalAction.NO_SIGNAL,
            min(0.90, 0.68 + strength * 5.0) if breakout else 0.0,
            [
                "CLOSE_ABOVE_PRIOR_20_BAR_CHANNEL_HIGH",
                "CLOSED_WEEKLY_BAR",
            ]
            if breakout
            else [],
        )
    fast = max(
        2,
        int(params.get("fast", params.get("entry_fast", 20))),
    )
    slow = max(
        fast + 1,
        int(params.get("slow", params.get("entry_slow", 52))),
    )
    channel = max(2, int(params.get("channel", 20)))
    atr_multiple = float(params.get("atr_mult", 2.0))
    roc_threshold = float(params.get("roc_threshold", 0.0))
    minimum = max(slow + 2, channel + 2, 30)
    if len(close) < minimum:
        return SignalAction.NO_SIGNAL, 0.0, ["INSUFFICIENT_HISTORY"]
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    trend = float(close.iloc[-1]) > float(ema_slow.iloc[-1]) and float(ema_fast.iloc[-1]) > float(
        ema_slow.iloc[-1]
    )
    prior_high = float(high.iloc[-channel - 1 : -1].max())
    atr_series = _atr_series(high, low, close, 14)
    atr_now = float(atr_series.iloc[-1])
    roc_period = min(max(4, channel), len(close) - 1)
    roc = float(close.iloc[-1] / close.iloc[-roc_period] - 1.0)
    trend_strength = max(
        0.0,
        min(
            1.0,
            float(ema_fast.iloc[-1] / ema_slow.iloc[-1] - 1.0) * 10.0,
        ),
    )

    if family in {
        "ma_crossover",
        "asymmetric_ma_crossover",
        "asymmetric_ma",
        "ma_channel",
    }:
        if len(close) <= slow:
            return SignalAction.NO_SIGNAL, 0.0, ["INSUFFICIENT_HISTORY"]
        fast_ma = float(close.tail(fast).mean())
        slow_ma = float(close.tail(slow).mean())
        active = fast_ma > slow_ma and float(close.iloc[-1]) > slow_ma
        strength = max(
            0.0,
            min(1.0, (fast_ma / slow_ma - 1.0) * 10.0),
        )
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            0.60 + 0.30 * strength if active else 0.0,
            ["FAST_MA_ABOVE_SLOW_MA", "CLOSE_ABOVE_SLOW_MA"] if active else [],
        )
    if family == "triple_ma_trend":
        middle = max(fast + 1, (fast + slow) // 2)
        ema_middle = close.ewm(
            span=middle,
            adjust=False,
            min_periods=middle,
        ).mean()
        active = trend and float(ema_fast.iloc[-1]) > float(ema_middle.iloc[-1]) > float(
            ema_slow.iloc[-1]
        )
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.64 + 0.25 * trend_strength) if active else 0.0,
            ["TRIPLE_EMA_ALIGNMENT", "CLOSE_ABOVE_SLOW_EMA"] if active else [],
        )
    if family == "macd_trend":
        macd = ema_fast - ema_slow
        signal = macd.ewm(
            span=9,
            adjust=False,
            min_periods=9,
        ).mean()
        active = (
            trend and float(macd.iloc[-1]) > float(signal.iloc[-1]) and float(macd.iloc[-1]) > 0
        )
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.65 + 0.20 * trend_strength) if active else 0.0,
            ["MACD_ABOVE_SIGNAL", "MACD_POSITIVE", "TREND_FILTER"] if active else [],
        )
    if family == "bollinger_breakout":
        period = min(
            len(close) - 1,
            int(params.get("period", max(channel, 20))),
        )
        sigma = float(params.get("entry_sigma", params.get("sigma", 2.0)))
        window = close.tail(period)
        upper = float(window.mean() + sigma * window.std(ddof=0))
        breakout = float(close.iloc[-1]) > upper
        return (
            SignalAction.BUY if breakout else SignalAction.NO_SIGNAL,
            0.72 if breakout else 0.0,
            ["CLOSE_ABOVE_BOLLINGER_BREAKOUT_LEVEL"] if breakout else [],
        )
    if family in {"donchian_breakout", "atr_breakout"}:
        range_confirmed = float(close.iloc[-1] - close.iloc[-2]) >= 0.20 * atr_now
        breakout = float(close.iloc[-1]) > prior_high
        active = breakout and (family == "donchian_breakout" or range_confirmed)
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(
                0.90,
                0.68
                + max(
                    0.0,
                    float(close.iloc[-1] / prior_high - 1.0) * 5.0,
                ),
            )
            if active
            else 0.0,
            [
                "CLOSE_ABOVE_PRIOR_CHANNEL_HIGH",
                ("ATR_EXPANSION_CONFIRMED" if family == "atr_breakout" else "DONCHIAN_BREAKOUT"),
            ]
            if active
            else [],
        )
    if family == "keltner_breakout":
        basis = close.ewm(
            span=slow,
            adjust=False,
            min_periods=slow,
        ).mean()
        upper = basis + atr_multiple * atr_series
        active = trend and float(close.iloc[-1]) > float(upper.iloc[-1])
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.68 + 0.20 * trend_strength) if active else 0.0,
            ["CLOSE_ABOVE_KELTNER_CHANNEL", "TREND_FILTER"] if active else [],
        )
    if family in {
        "trend_consensus",
        "robust_trend_consensus",
    }:
        adx, plus_di, minus_di = _adx_series(high, low, close)
        votes = [
            trend,
            roc > roc_threshold,
            float(adx.iloc[-1]) >= float(params.get("adx_min", 20)),
            float(plus_di.iloc[-1]) > float(minus_di.iloc[-1]),
        ]
        required = 3 if family == "robust_trend_consensus" else 2
        active = sum(votes) >= required
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.58 + 0.075 * sum(votes)) if active else 0.0,
            [
                f"TREND_CONSENSUS_{sum(votes)}_OF_4",
                "CLOSED_BAR_VOTING",
            ]
            if active
            else [],
        )
    if family == "roc_trend":
        active = trend and roc > roc_threshold
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.62 + min(0.25, max(0.0, roc))) if active else 0.0,
            ["POSITIVE_RATE_OF_CHANGE", "TREND_FILTER"] if active else [],
        )
    if family in {
        "risk_adjusted_momentum",
        "momentum_consensus",
        "etf_commodity_trend",
    }:
        returns = close.pct_change(fill_method=None)
        volatility = float(returns.tail(min(26, len(returns))).std(ddof=0))
        horizons = [
            min(4, len(close) - 1),
            min(13, len(close) - 1),
            min(26, len(close) - 1),
        ]
        momenta = [float(close.iloc[-1] / close.iloc[-period] - 1.0) for period in horizons]
        if family == "risk_adjusted_momentum":
            score = roc / max(volatility, 1e-9)
            active = trend and roc > 0 and score > 0
            reasons = [
                "POSITIVE_RISK_ADJUSTED_MOMENTUM",
                "TREND_FILTER",
            ]
        else:
            positive = sum(value > 0 for value in momenta)
            active = trend and positive >= 2
            score = float(np.mean(momenta)) / max(volatility, 1e-9)
            reasons = [
                f"POSITIVE_MOMENTUM_HORIZONS_{positive}_OF_3",
                "TREND_FILTER",
            ]
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.62 + min(0.25, max(0.0, score) * 0.02)) if active else 0.0,
            reasons if active else [],
        )
    if family == "adx_trend":
        adx, plus_di, minus_di = _adx_series(high, low, close)
        active = (
            trend
            and float(adx.iloc[-1]) >= float(params.get("adx_min", 20))
            and float(plus_di.iloc[-1]) > float(minus_di.iloc[-1])
        )
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.62 + float(adx.iloc[-1]) / 200.0) if active else 0.0,
            ["ADX_TREND_STRENGTH", "POSITIVE_DIRECTIONAL_MOVEMENT"] if active else [],
        )
    if family in {
        "ema_pullback",
        "stochastic_trend_pullback",
        "rsi2_adx_pullback",
        "pullback_consensus",
    }:
        rsi2 = _rsi_series(close, 2)
        rsi14 = _rsi_series(close, 14)
        lowest = low.rolling(14, min_periods=14).min()
        highest = high.rolling(14, min_periods=14).max()
        stochastic = 100.0 * (close - lowest) / (highest - lowest).replace(0, np.nan)
        distance = abs(float(close.iloc[-1] / ema_fast.iloc[-1] - 1.0))
        rebound = float(close.iloc[-1]) > float(open_series.iloc[-1])
        if family == "ema_pullback":
            active = trend and distance <= 0.03 and rebound
            reasons = ["EMA_PULLBACK_ZONE", "POSITIVE_CLOSE_RECLAIM"]
        elif family == "stochastic_trend_pullback":
            active = (
                trend
                and float(stochastic.iloc[-1]) <= 35.0
                and float(stochastic.iloc[-1]) > float(stochastic.iloc[-2])
            )
            reasons = ["STOCHASTIC_PULLBACK_TURN", "TREND_FILTER"]
        elif family == "rsi2_adx_pullback":
            adx, _, _ = _adx_series(high, low, close)
            active = (
                trend
                and float(rsi2.iloc[-1]) <= float(params.get("rsi_low", 15))
                and float(adx.iloc[-1]) >= float(params.get("adx_min", 18))
            )
            reasons = ["RSI2_OVERSOLD", "ADX_TREND_FILTER"]
        else:
            votes = [
                distance <= 0.04,
                float(rsi14.iloc[-1]) <= float(params.get("rsi14_low", 42)),
                float(stochastic.iloc[-1]) <= 35.0,
                rebound,
            ]
            active = trend and sum(votes) >= 2
            reasons = [
                f"PULLBACK_CONSENSUS_{sum(votes)}_OF_4",
                "TREND_FILTER",
            ]
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(
                0.90,
                0.62
                + 0.20
                * max(
                    0.0,
                    1.0 - distance / 0.10,
                ),
            )
            if active
            else 0.0,
            reasons if active else [],
        )
    if family in {
        "range_expansion_breakout",
        "volatility_contraction_breakout",
        "volume_breakout",
        "breakout_consensus",
        "channel_consensus",
    }:
        current_range = float(high.iloc[-1] - low.iloc[-1])
        median_range = float((high - low).iloc[-21:-1].median())
        relative_volume = float(volume_series.iloc[-1]) / max(
            float(volume_series.iloc[-21:-1].median()),
            1.0,
        )
        returns = close.pct_change(fill_method=None)
        short_vol = returns.rolling(5, min_periods=5).std()
        long_vol = returns.rolling(20, min_periods=20).std()
        prior_contraction = float(short_vol.iloc[-2] / max(long_vol.iloc[-2], 1e-9))
        donchian = float(close.iloc[-1]) > prior_high
        range_expansion = current_range > max(
            median_range * 1.25,
            atr_now * 0.9,
        )
        volume_confirmed = relative_volume >= float(params.get("volume_mult", 1.25))
        contraction = prior_contraction <= float(params.get("volatility_ratio", 0.75))
        bollinger_window = close.tail(max(20, channel))
        bollinger_upper = float(
            bollinger_window.mean() + float(params.get("sigma", 2.0)) * bollinger_window.std(ddof=0)
        )
        bollinger = float(close.iloc[-1]) > bollinger_upper
        if family == "range_expansion_breakout":
            votes = [donchian, range_expansion]
            required = 2
        elif family == "volatility_contraction_breakout":
            votes = [donchian, contraction, trend]
            required = 3
        elif family == "volume_breakout":
            votes = [donchian, volume_confirmed]
            required = 2
        elif family == "breakout_consensus":
            votes = [
                donchian,
                bollinger,
                range_expansion,
                volume_confirmed,
            ]
            required = 2
        else:
            keltner_upper = float(ema_slow.iloc[-1] + atr_multiple * atr_now)
            votes = [
                donchian,
                bollinger,
                float(close.iloc[-1]) > keltner_upper,
            ]
            required = 2
        active = sum(votes) >= required
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.60 + 0.075 * sum(votes)) if active else 0.0,
            [
                f"{family.upper()}_{sum(votes)}_OF_{len(votes)}",
                "CLOSED_BAR_BREAKOUT_CONFIRMATION",
            ]
            if active
            else [],
        )
    if family == "quality_trend_consensus":
        return (
            SignalAction.NO_SIGNAL,
            0.0,
            ["PIT_FUNDAMENTAL_QUALITY_REQUIRED"],
        )
    if family in {"etf_rotation", "commodity_etf_trend"}:
        lookback = int(params.get("lookback", 63))
        trend_period = int(params.get("trend_period", 100))
        momentum = float(close.iloc[-1] / close.iloc[-lookback] - 1.0)
        trend = float(close.iloc[-1]) > float(close.tail(trend_period).mean())
        active = momentum > 0 and trend
        return (
            SignalAction.BUY if active else SignalAction.NO_SIGNAL,
            min(0.90, 0.62 + max(0.0, momentum)),
            ["POSITIVE_ABSOLUTE_MOMENTUM", "CLOSE_ABOVE_TREND_FILTER"] if active else [],
        )
    return SignalAction.NO_SIGNAL, 0.0, ["FAMILY_SIGNAL_NOT_IMPLEMENTED"]


def _atr_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    previous = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous).abs(),
            (low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def _rsi_series(close: pd.Series, period: int) -> pd.Series:
    change = close.diff()
    gain = (
        change.clip(lower=0.0)
        .ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )
    loss = (
        (-change.clip(upper=0.0))
        .ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        )
        .mean()
    )
    relative = gain / loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + relative)
    result = result.mask((loss == 0) & (gain > 0), 100.0)
    return result.mask((loss == 0) & (gain == 0), 50.0)


def _adx_series(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    upward = high.diff()
    downward = -low.diff()
    plus_dm = upward.where(
        (upward > downward) & (upward > 0.0),
        0.0,
    )
    minus_dm = downward.where(
        (downward > upward) & (downward > 0.0),
        0.0,
    )
    atr = _atr_series(high, low, close, period)
    plus_di = (
        100.0
        * plus_dm.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        / atr.replace(0.0, np.nan)
    )
    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    directional = 100.0 * (plus_di - minus_di).abs() / denominator
    adx = directional.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    return adx, plus_di, minus_di


def _atr(frame: pd.DataFrame, period: int) -> float:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    value = float(true_range.tail(period).mean())
    return value if math.isfinite(value) and value > 0 else float(close.iloc[-1]) * 0.02


def _candidate(project_root: Path, strategy_id: str) -> dict[str, Any] | None:
    return next(
        (row for row in _candidates(project_root) if row.get("candidate_id") == strategy_id),
        None,
    )


def _phase11_14_observer_signals(
    project_root: Path,
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    path = project_root / "output" / "research" / "phase11_14" / "latest-forward-observation.json"
    try:
        observation = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    if observation.get("status") != "GO" or observation.get("schema") not in {
        "phase11_14_forward_observation_v2",
        "phase11_14_forward_observation_v3",
    }:
        return []
    qualification_hash = str(observation.get("qualification_hash") or "")
    if not qualification_hash:
        return []
    plans: list[dict[str, Any]] = []
    for candidate in observation.get("observations", []):
        if candidate.get("observation_status") != "OBSERVATION_COMPLETE":
            continue
        timeframe = _normalize_timeframe(candidate.get("timeframe"))
        closed_at = _utc_datetime(candidate.get("closed_bar_timestamp"))
        if closed_at is None:
            continue
        declared_expiration = closed_at + SIGNAL_VALIDITY.get(
            timeframe,
            SIGNAL_VALIDITY["1d"],
        )
        observer_tier = str(candidate.get("observer_tier") or "ROBUST_FORWARD_OBSERVER")
        portfolio_eligible = bool(candidate.get("portfolio_eligible", False))
        for raw in candidate.get("raw_active_signals", []):
            if raw.get("execution_envelope_status") != "GO" or str(raw.get("action")) != "BUY":
                continue
            symbol = str(raw.get("symbol") or "").upper()
            exchange_timezone = _signal_exchange_timezone(
                project_root,
                symbol,
                timeframe,
            )
            freshness_evaluation = evaluate_signal_freshness(
                {
                    "timeframe": timeframe,
                    "data_timestamp": closed_at,
                    "expiration_timestamp": declared_expiration,
                    "exchange_timezone": exchange_timezone,
                },
                now=now,
            )
            if not freshness_evaluation["is_current"]:
                continue
            expiration = freshness_evaluation["effective_expiration"]
            entry = _positive_decimal(raw.get("entry_reference"))
            stop = _positive_decimal(raw.get("stop_loss"))
            target_1 = _positive_decimal(raw.get("take_profit_1"))
            target_2 = _positive_decimal(raw.get("take_profit_2"))
            if (
                not symbol
                or entry is None
                or stop is None
                or target_1 is None
                or target_2 is None
                or not stop < entry < target_1 <= target_2
            ):
                continue
            risk = entry - stop
            zone_buffer = risk * Decimal("0.10")
            contract = _contract(project_root, symbol)
            currency = str(contract.get("currency") or "USD")
            exchange = str(
                contract.get("primary_exchange") or contract.get("exchange") or "UNRESOLVED"
            )
            confidence_cap = (
                Decimal("0.55")
                if observer_tier == "EXPLORATORY_FORWARD_OBSERVER"
                else Decimal("0.75")
            )
            confidence = min(
                confidence_cap,
                Decimal(str(raw.get("confidence_score", 0.0))),
            )
            reward_risk_1 = (target_1 - entry) / risk
            reward_risk_2 = (target_2 - entry) / risk
            risks = [
                "FORWARD_OBSERVATION_ONLY",
                "NOT_FINANCIAL_FINALIST",
                "EXECUTION_AUTHORITY_NONE",
            ]
            if observer_tier == "EXPLORATORY_FORWARD_OBSERVER":
                risks.append("EXPLORATORY_NOT_PORTFOLIO_ELIGIBLE")
            if not raw.get("currently_attested", False):
                risks.append("CURRENT_SHARIAH_ATTESTATION_REQUIRED")
            if not contract:
                risks.append("CONTRACT_IDENTITY_UNAVAILABLE")
            strategy_id = str(candidate.get("strategy_id") or "")
            signal_id = (
                "SIG-P1114-"
                + stable_hash(
                    {
                        "qualification_hash": qualification_hash,
                        "strategy_id": strategy_id,
                        "symbol": symbol,
                        "closed_bar_timestamp": closed_at.isoformat(),
                        "observer_tier": observer_tier,
                    }
                )[:24]
            )
            plans.append(
                {
                    "signal_id": signal_id,
                    "asset": symbol,
                    "ticker": symbol,
                    "contract_identity": contract,
                    "asset_class": str(
                        candidate.get("asset_class") or contract.get("security_type") or "STK"
                    ),
                    "exchange": exchange,
                    "currency": currency,
                    "signal_timestamp": now,
                    "data_timestamp": closed_at,
                    "data_freshness": "FRESH",
                    "exchange_timezone": exchange_timezone,
                    "signal_freshness_basis": freshness_evaluation["freshness_basis"],
                    "signal_freshness": {
                        key: (value.isoformat() if isinstance(value, datetime) else value)
                        for key, value in freshness_evaluation.items()
                        if key != "is_current"
                    },
                    "strategy_id": strategy_id,
                    "strategy_family": str(
                        candidate.get("family") or candidate.get("formula") or "UNDECLARED"
                    ),
                    "strategy_dna_hash": qualification_hash,
                    "strategy_timeframe_contract": (
                        declared_research_signal_timeframe_contract(
                            project_root,
                            candidate,
                        )
                    ),
                    "timeframe": timeframe,
                    "higher_timeframe_context": (f"CLOSED_{timeframe.upper()}_BAR_PHASE11_14"),
                    "action": "WATCHLIST",
                    "current_market_price": entry,
                    "preferred_entry": entry,
                    "entry_zone_low": entry - zone_buffer,
                    "entry_zone_high": entry + zone_buffer,
                    "limit_entry_price": entry,
                    "invalidation_level": stop,
                    "stop_loss": stop,
                    "stop_method": str(raw.get("stop_policy") or "ATR_ADJUSTED_STRUCTURE"),
                    "stop_distance_pct": risk / entry,
                    "stop_distance_atr": Decimal("2.0"),
                    "take_profit_1": target_1,
                    "take_profit_2": target_2,
                    "take_profit_mode": str(
                        raw.get("exit_policy") or "PRIMARY_TARGET_WITH_TRAILING_EXIT"
                    ),
                    "reward_risk_1": reward_risk_1,
                    "reward_risk_2": reward_risk_2,
                    "suggested_quantity": Decimal("0"),
                    "maximum_order_value_eur": Decimal("0"),
                    "maximum_planned_loss_eur": Decimal("0"),
                    "estimated_transaction_costs_eur": Decimal("0"),
                    "expected_holding_period": _holding_period(timeframe),
                    "confidence_score": confidence,
                    "regime": "FORWARD_OBSERVATION",
                    "reasons": [
                        "PHASE11_14_FROZEN_QUALIFICATION_BOUNDARY",
                        observer_tier,
                        "CLOSED_BAR_SIGNAL",
                    ],
                    "risks": risks,
                    "expiration_timestamp": expiration,
                    "lifecycle_status": "WATCHLIST",
                    "automatic_execution_allowed": False,
                    "source_provider": "YFINANCE",
                    "source_interval": timeframe,
                    "bar_origin": "PHASE11_14_FORWARD_OBSERVATION",
                    "bar_closed": True,
                    "observer_tier": observer_tier,
                    "portfolio_eligible": portfolio_eligible,
                    "deployment_eligible": False,
                    "execution_eligible": False,
                    "automatic_promotion": False,
                    "broker_calls": 0,
                    "orders_generated": 0,
                }
            )
    return plans


def _phase11_14_observer_inventory(
    project_root: Path,
) -> dict[str, int]:
    path = project_root / "output" / "research" / "phase11_14" / "latest-forward-observation.json"
    try:
        observation = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {
            "observer_strategy_count": 0,
            "exploratory_strategy_count": 0,
        }
    rows = [
        row
        for row in observation.get("observations", [])
        if row.get("observation_status") == "OBSERVATION_COMPLETE"
    ]
    return {
        "observer_strategy_count": len({str(row.get("strategy_id")) for row in rows}),
        "exploratory_strategy_count": len(
            {
                str(row.get("strategy_id"))
                for row in rows
                if row.get("observer_tier") == "EXPLORATORY_FORWARD_OBSERVER"
            }
        ),
    }


def _utc_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _positive_decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (TypeError, ValueError, ArithmeticError):
        return None
    return parsed if parsed.is_finite() and parsed > 0 else None


def _candidates(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "output" / "research" / "recovered_survivors.json"
    if not path.exists():
        return []
    rows = list(json.loads(path.read_text(encoding="utf-8")).get("survivors", []))
    dynamic = project_root / "config/dynamic/strategies_v1.json"
    if dynamic.exists():
        rows.extend(json.loads(dynamic.read_text(encoding="utf-8")).get("strategies", []))
    hmm_registry = (
        project_root / "output" / "research" / "phase11_11" / "frozen-shadow-registry.json"
    )
    if hmm_registry.exists():
        rows.extend(
            json.loads(hmm_registry.read_text(encoding="utf-8")).get(
                "candidates",
                [],
            )
        )
    return rows


def _universe_symbols(project_root: Path, universe: str) -> list[str]:
    config_path = project_root / "config" / "screener" / "daily_screener_v1.json"
    etfs = set()
    if config_path.exists():
        etfs = set(json.loads(config_path.read_text(encoding="utf-8")).get("etf_symbols", []))
    etfs.update(broad_etf_symbols(project_root))
    cache = project_root / "data" / "research" / "critical_trading" / "yfinance"
    symbols = sorted(path.stem.upper() for path in cache.glob("*.parquet"))
    if universe == "etfs":
        return [symbol for symbol in symbols if symbol in etfs]
    if universe == "commodities":
        commodity = broad_commodity_symbols(project_root) | {
            "DBA",
            "DBB",
            "DBC",
            "GLD",
            "SLV",
        }
        return [symbol for symbol in symbols if symbol in commodity]
    if universe == "stocks":
        return [symbol for symbol in symbols if symbol not in etfs]
    return symbols


def _contract(project_root: Path, symbol: str) -> dict[str, Any]:
    path = project_root / "output" / "ibkr" / "contracts" / "stocks.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    matches = frame[frame["symbol"].astype(str).str.upper().eq(symbol.upper())]
    if len(matches) != 1:
        return {}
    row = matches.iloc[0].to_dict()
    resolved_at = pd.to_datetime(row.get("resolved_at"), utc=True, errors="coerce")
    now = pd.Timestamp(datetime.now(UTC))
    if pd.isna(resolved_at) or resolved_at > now or now - resolved_at >= pd.Timedelta(days=7):
        return {}
    allowed = {
        "con_id",
        "symbol",
        "local_symbol",
        "security_type",
        "currency",
        "exchange",
        "primary_exchange",
        "trading_class",
        "min_tick",
        "contract_hash",
        "resolved_at",
        "server_version",
    }
    identity = {
        key: value
        for key, value in row.items()
        if key in allowed and not (isinstance(value, float) and np.isnan(value))
    }
    identity.update(
        {
            "resolved_at": resolved_at.isoformat(),
            "cache_expires_at": (resolved_at + pd.Timedelta(days=7)).isoformat(),
            "cache_status": "FRESH",
            "contract_source": "PHASE2_EXACT_STK_CACHE",
        }
    )
    return identity


def _fx_to_eur(project_root: Path, currency: str) -> Decimal:
    if currency.upper() == "EUR":
        return Decimal("1")
    path = project_root / "data" / "fx" / "fx_daily.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        for column in ("quote_currency", "currency"):
            if column in frame.columns:
                rows = frame[frame[column].astype(str).str.upper().eq(currency.upper())]
                if not rows.empty:
                    for value_column in ("fx_to_eur", "rate_to_eur", "eur_rate"):
                        if value_column in rows.columns:
                            return Decimal(str(float(rows[value_column].iloc[-1])))
    return Decimal("0.90") if currency.upper() == "USD" else Decimal("1")


def _parameters(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw))
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _promote_consensus(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_ticker: dict[str, set[str]] = {}
    for plan in plans:
        if plan["action"] == SignalAction.BUY:
            by_ticker.setdefault(plan["ticker"], set()).add(plan["strategy_id"])
    for plan in plans:
        if plan["action"] == SignalAction.BUY and len(by_ticker[plan["ticker"]]) >= 2:
            plan["action"] = SignalAction.STRONG_BUY
            plan["reasons"] = [*plan["reasons"], "MULTI_STRATEGY_CONFIRMATION"]
    return plans


def _fair_signal_limit(plans: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(plans) <= maximum:
        return plans
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        by_ticker.setdefault(str(plan["ticker"]), []).append(plan)
    ticker_heads = sorted(
        (rows[0] for rows in by_ticker.values()),
        key=lambda row: (
            -float(row.get("confidence_score", 0)),
            str(row.get("ticker", "")),
            str(row.get("strategy_id", "")),
        ),
    )
    for plan in ticker_heads[:maximum]:
        identity = str(plan.get("signal_id"))
        selected.append(plan)
        selected_ids.add(identity)
    if len(selected) >= maximum:
        return selected
    queues: dict[str, list[dict[str, Any]]] = {}
    for plan in plans:
        identity = str(plan.get("signal_id"))
        if identity in selected_ids:
            continue
        queues.setdefault(str(plan["strategy_id"]), []).append(plan)
    strategy_ids = sorted(queues)
    while len(selected) < maximum and strategy_ids:
        next_round: list[str] = []
        for strategy_id in strategy_ids:
            queue = queues[strategy_id]
            if not queue:
                continue
            plan = queue.pop(0)
            selected.append(plan)
            selected_ids.add(str(plan.get("signal_id")))
            if queue:
                next_round.append(strategy_id)
            if len(selected) >= maximum:
                break
        strategy_ids = next_round
    return selected


def _publish_signal_outputs(
    project_root: Path, report: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    root = project_root / "output" / "signals"
    root.mkdir(parents=True, exist_ok=True)
    (root / "latest_signals.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(rows).to_csv(root / "latest_signals.csv", index=False)
    parquet_rows = [
        {key: (_parquet_safe_value(value)) for key, value in row.items()} for row in rows
    ]
    pd.DataFrame(parquet_rows).to_parquet(root / "signal_history.parquet", index=False)
    columns = [
        "ticker",
        "action",
        "preferred_entry",
        "stop_loss",
        "take_profit_1",
        "take_profit_2",
        "reward_risk_1",
        "confidence_score",
        "strategy_id",
        "timeframe",
        "expiration_timestamp",
        "lifecycle_status",
    ]
    body = "\n".join(
        "<tr>" + "".join(f"<td>{row.get(column, '')}</td>" for column in columns) + "</tr>"
        for row in rows
    )
    (root / "latest_signals.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Signals</title>"
        "<style>body{font-family:Arial;margin:24px}table{border-collapse:collapse;"
        "width:100%}th,td{border:1px solid #ccd1d1;padding:6px}th{background:#eef2f3}"
        "</style></head><body><h1>Manual model signals</h1>"
        "<p>Model signals are not guaranteed outcomes. Automatic execution is disabled.</p>"
        "<table><thead><tr>"
        + "".join(f"<th>{column}</th>" for column in columns)
        + f"</tr></thead><tbody>{body}</tbody></table></body></html>",
        encoding="utf-8",
    )
    active = [
        row
        for row in rows
        if row.get("lifecycle_status") not in {"CLOSED", "EXPIRED", "CANCELLED", "INVALIDATED"}
    ]
    (root / "active_signals.json").write_text(
        json.dumps(active, indent=2, default=str), encoding="utf-8"
    )


def _publish_active_swing_outputs(project_root: Path, report: dict[str, Any]) -> None:
    path = project_root / ACTIVE_SWING_SIGNAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parquet_safe_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, set):
        value = sorted(value, key=str)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED",
        "reason": reason,
        "signal_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
