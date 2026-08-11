from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.costs import estimate_transaction_cost, load_shared_cost_model
from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash


POLICY_PATH = Path("config/portfolio/p2_2_execution_feasibility_v1.json")
PRIVATE_REPORT_PATH = Path("data/portfolio/private/p2-2-execution-feasibility.json")
PUBLIC_REPORT_PATH = Path("output/portfolio/p2-2-execution-feasibility.json")
FREEZE_POINTER = Path("output/portfolio/orchestrator/p2-2-execution-feasibility-freeze.json")
FREEZE_ROOT = Path("output/portfolio/orchestrator/freezes/p2_2")
P2_2_SOURCES = (
    "config/capital_scaling/levels_v1.json",
    "config/costs/shared_transaction_cost_v1.json",
    "config/portfolio/p2_2_execution_feasibility_v1.json",
    "config/portfolio/quant_capability_authority_v1.json",
    "config/portfolio/strategy_authority_registry_v1.json",
    "src/stocks/capital/canary.py",
    "src/stocks/portfolio/execution_feasibility.py",
    "src/stocks/portfolio/learning_integration.py",
    "src/stocks/portfolio/orchestrator.py",
    "src/stocks/quant_platform/ml.py",
    "src/stocks/quant_platform/professional.py",
    "src/stocks/quant_platform/regime.py",
    "tests/test_p2_2_execution_feasibility.py",
    "tests/test_quant_platform_ml.py",
    "tests/test_quant_platform_professional.py",
)


def build_execution_feasibility_report(
    project_root: Path,
    *,
    opportunities: Iterable[dict[str, Any]],
    funnel: dict[str, Any],
    account: dict[str, Any],
    live_authority: dict[str, Any],
    vectorized_stage0: dict[str, Any] | None = None,
    market_data_capabilities: dict[str, Any] | None = None,
    level_two_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate whole-share Level-1 feasibility without creating orders."""

    root = project_root.resolve()
    policy = _read_json(root / POLICY_PATH)
    _validate_policy(policy)
    rows = list(opportunities)
    watchlist = list(funnel.get("watchlist_candidates", []))
    candidates = _candidate_rows(rows, watchlist)
    instruments = _instrument_map(root)
    prices = _stage0_price_map(vectorized_stage0 or {})
    fx = _fx_map(root)
    equity = _decimal(account.get("net_liquidation_eur"))
    available_cash = _decimal(
        account.get("eur_available_for_new_longs", account.get("available_funds_eur"))
    )
    configured_fraction = _decimal(
        policy["risk_budget"]["risk_fraction_of_equity"]
    )
    configured_cap = _decimal(policy["risk_budget"]["absolute_cap_eur"])
    risk_budget = max(
        Decimal("0"), min(equity * configured_fraction, configured_cap)
    )
    active_notional_cap = _decimal(policy["notional"]["active_hard_cap_eur"])
    cost_model = load_shared_cost_model(root)
    market = market_data_capabilities or {}
    realtime_status = str(
        market.get("summary", {}).get("realtime_top_of_book") or "UNPROVEN"
    )
    live_quote_available = realtime_status == "AVAILABLE"
    maximum_event_risk = Decimal("0.75")
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        symbol = str(candidate.get("symbol") or "").upper()
        instrument = instruments.get(symbol, {})
        targets = dict(candidate.get("targets") or {})
        live_price = _first_positive(candidate.get("live_quote_price"))
        price_local = _first_positive(
            live_price,
            targets.get("entry"),
            candidate.get("entry_price"),
            prices.get(symbol, {}).get("last_price"),
        )
        price_timestamp = (
            candidate.get("live_quote_timestamp")
            or candidate.get("timestamp")
            or prices.get(symbol, {}).get("data_timestamp")
        )
        price_source = (
            "LIVE_REALTIME_TOP_OF_BOOK"
            if live_price is not None
            else "OPPORTUNITY_PLANNED_ENTRY"
            if _first_positive(targets.get("entry")) is not None
            else "VECTORIZED_STAGE0_RESEARCH_LAST_PRICE"
            if price_local is not None
            else "UNAVAILABLE"
        )
        price_live_actionable = bool(
            live_price is not None
            and live_quote_available
            and candidate.get("live_quote_status") == "GO"
        )
        currency = str(
            candidate.get("currency") or instrument.get("currency") or ""
        ).upper()
        fx_to_eur = fx.get(currency)
        price_eur = (
            price_local * fx_to_eur
            if price_local is not None and fx_to_eur is not None
            else None
        )
        stop_local = _first_positive(
            targets.get("stop"), candidate.get("stop_price")
        )
        take_profit_local = _first_positive(
            targets.get("take_profit"), candidate.get("take_profit_price")
        )
        risk_per_share = (
            (price_local - stop_local) * fx_to_eur
            if price_local is not None
            and stop_local is not None
            and fx_to_eur is not None
            and 0 < stop_local < price_local
            else None
        )
        risk_qty = _floor_div(risk_budget, risk_per_share)
        notional_qty = _floor_div(active_notional_cap, price_eur)
        cash_qty = _floor_div(available_cash, price_eur)
        position_weight_cap = (
            Decimal("0.20")
            if str(candidate.get("asset_class") or instrument.get("asset_type"))
            .upper()
            .endswith("ETF")
            else Decimal("0.15")
        )
        position_qty = _floor_div(equity * position_weight_cap, price_eur)
        quantity_limits = {
            "risk": risk_qty,
            "active_notional": notional_qty,
            "cash": cash_qty,
            "position_weight": position_qty,
            "maximum_execution_quantity": 100,
        }
        final_qty = min(quantity_limits.values()) if quantity_limits else 0
        actual_notional = (
            Decimal(final_qty) * price_eur if price_eur is not None else Decimal("0")
        )
        actual_risk = (
            Decimal(final_qty) * risk_per_share
            if risk_per_share is not None
            else Decimal("0")
        )
        economics_qty = max(1, final_qty) if price_eur is not None else 0
        economic_notional = (
            Decimal(economics_qty) * price_eur
            if price_eur is not None
            else Decimal("0")
        )
        costs = estimate_transaction_cost(
            economic_notional,
            currency=currency or "UNKNOWN",
            model=cost_model,
            round_trip=True,
        )
        expected_return = _optional_decimal(
            candidate.get("expected_gross_return")
        )
        expected_gross_edge = (
            economic_notional * expected_return
            if expected_return is not None and economics_qty > 0
            else None
        )
        expected_net_edge = (
            expected_gross_edge - Decimal(str(costs["total_cost_eur"]))
            if expected_gross_edge is not None
            else None
        )
        cost_to_edge = (
            Decimal(str(costs["total_cost_eur"])) / expected_gross_edge
            if expected_gross_edge is not None and expected_gross_edge > 0
            else None
        )
        expected_r = _optional_decimal(candidate.get("expected_r"))
        if (
            expected_r is None
            and take_profit_local is not None
            and price_local is not None
            and stop_local is not None
            and price_local > stop_local
        ):
            expected_r = (take_profit_local - price_local) / (
                price_local - stop_local
            )
        binding = dict(candidate.get("strategy_binding") or {})
        blockers = _candidate_blockers(
            candidate=candidate,
            account=account,
            live_authority=live_authority,
            binding=binding,
            price_local=price_local,
            price_eur=price_eur,
            stop_local=stop_local,
            risk_per_share=risk_per_share,
            final_qty=final_qty,
            live_quote_available=live_quote_available,
            price_live_actionable=price_live_actionable,
            expected_return=expected_return,
            expected_net_edge=expected_net_edge,
            cost_to_edge=cost_to_edge,
            maximum_cost_to_edge=_decimal(
                policy["economics"]["maximum_cost_to_expected_edge_ratio"]
            ),
            maximum_event_risk=maximum_event_risk,
        )
        sensitivity = {
            str(int(cap)): {
                "notional_cap_eur": int(cap),
                "notional_quantity": _floor_div(Decimal(cap), price_eur),
                "whole_share_notional_feasible": _floor_div(
                    Decimal(cap), price_eur
                )
                >= 1,
            }
            for cap in policy["notional"]["sensitivity_caps_eur"]
        }
        results.append(
            {
                "schema": "p2_2_candidate_feasibility_v1",
                "symbol": symbol,
                "instrument_id": candidate.get("instrument_id")
                or instrument.get("instrument_id"),
                "asset_class": candidate.get("asset_class")
                or instrument.get("asset_type"),
                "contributing_strategy_id": candidate.get(
                    "contributing_strategy_id"
                ),
                "strategy_timeframe": candidate.get("holding_horizon"),
                "strategy_evidence_status": candidate.get(
                    "strategy_evidence_status"
                ),
                "exact_strategy_binding_status": binding.get("status", "NO_GO"),
                "exact_strategy_binding": binding,
                "strategy_live_authorized": binding.get("status") == "GO",
                "shariah_status": candidate.get("shariah_status"),
                "signal_timestamp": candidate.get("timestamp"),
                "price_timestamp": price_timestamp,
                "price_source": price_source,
                "price_is_live_actionable": price_live_actionable,
                "realtime_level_one_status": realtime_status,
                "price_local": _number(price_local),
                "currency": currency or None,
                "fx_to_eur": _number(fx_to_eur),
                "whole_share_price_eur": _number(price_eur),
                "whole_share_only": True,
                "fractional_shares_allowed": False,
                "protective_stop_local": _number(stop_local),
                "take_profit_local": _number(take_profit_local),
                "risk_budget_eur": _number(risk_budget),
                "risk_budget_semantics": "MAXIMUM_LOSS_TO_PROTECTIVE_STOP",
                "risk_per_share_eur": _number(risk_per_share),
                "maximum_quantity_by_risk": risk_qty,
                "maximum_quantity_by_notional": notional_qty,
                "maximum_quantity_by_cash": cash_qty,
                "maximum_quantity_by_position_weight": position_qty,
                "final_maximum_whole_quantity": final_qty,
                "actual_notional_eur": _number(actual_notional),
                "actual_stop_risk_eur": _number(actual_risk),
                "notional_sensitivity": sensitivity,
                "expected_gross_return": _number(expected_return),
                "expected_existing_net_return": _number(
                    _optional_decimal(candidate.get("expected_net_return"))
                ),
                "expected_gross_edge_eur": _number(expected_gross_edge),
                "estimated_round_trip_cost_eur": _number(
                    Decimal(str(costs["total_cost_eur"]))
                ),
                "expected_net_edge_eur": _number(expected_net_edge),
                "cost_to_expected_edge_ratio": _number(cost_to_edge),
                "expected_r": _number(expected_r),
                "liquidity": candidate.get("liquidity"),
                "observed_spread_bps": candidate.get("spread_bps"),
                "configured_half_spread_bps": _number(
                    cost_model.half_spread_bps
                ),
                "estimated_slippage_bps": candidate.get(
                    "estimated_slippage_bps"
                ),
                "correlation_cluster": candidate.get("correlation_cluster"),
                "correlation_contribution": candidate.get(
                    "correlation_contribution"
                ),
                "event_risk": candidate.get("event_risk"),
                "regime_fit": candidate.get("regime_fit"),
                "learning_overlay": candidate.get("learning_overlay", {}),
                "learning_may_grant_feasibility": False,
                "feasibility_status": (
                    "FEASIBLE_NOW" if not blockers else "REJECTED"
                ),
                "exact_rejection_reasons": blockers,
                "primary_rejection_reason": blockers[0] if blockers else None,
                "execution_authority": "NONE",
                "broker_writes": 0,
            }
        )
    return _summarize_report(
        results,
        policy=policy,
        equity=equity,
        available_cash=available_cash,
        risk_budget=risk_budget,
        realtime_status=realtime_status,
        opportunity_count=len(rows),
        watchlist_count=len(watchlist),
        level_two_evidence=level_two_evidence or {},
    )


def publish_execution_feasibility_report(
    project_root: Path, report: dict[str, Any]
) -> None:
    _atomic_json(project_root / PRIVATE_REPORT_PATH, report)
    public = {
        **report,
        "account_equity_eur": None,
        "available_cash_eur": None,
        "financial_account_values_public": False,
    }
    _atomic_json(project_root / PUBLIC_REPORT_PATH, public)


def build_p2_2_freeze(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    missing = [relative for relative in P2_2_SOURCES if not (root / relative).is_file()]
    hashes = {
        relative: sha256_file(root / relative).upper()
        for relative in P2_2_SOURCES
        if (root / relative).is_file()
    }
    core: dict[str, Any] = {
        "schema": "p2_2_execution_feasibility_freeze_v1",
        "freeze_status": "P2_2_EXECUTION_FEASIBILITY_FROZEN_GO"
        if not missing
        else "NO_GO",
        "source_hashes": hashes,
        "risk_budget_absolute_cap_eur": 9,
        "whole_shares_only": True,
        "automatic_notional_change": False,
        "automatic_strategy_promotion": False,
        "learning_money_control": False,
        "blockers": [f"MISSING_FREEZE_SOURCE:{value}" for value in missing],
        "broker_writes": 0,
    }
    freeze_hash = stable_hash(core)
    report = {
        **core,
        "status": "GO" if not missing else "NO_GO",
        "created_at": _now(),
        "freeze_hash": freeze_hash,
    }
    if not missing:
        immutable = root / FREEZE_ROOT / f"{freeze_hash}.json"
        existing = _read_json(immutable)
        if existing:
            existing_core = {
                key: value
                for key, value in existing.items()
                if key not in {"status", "created_at", "freeze_hash"}
            }
            if (
                existing.get("freeze_hash") != freeze_hash
                or existing_core != core
            ):
                raise FileExistsError(
                    f"immutable freeze collision: {immutable}"
                )
            report = existing
        else:
            _write_immutable_json(immutable, report)
        _atomic_json(root / FREEZE_POINTER, report)
    return report


def verify_p2_2_freeze(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    frozen = _read_json(root / FREEZE_POINTER)
    changed = sorted(
        relative
        for relative, digest in frozen.get("source_hashes", {}).items()
        if not (root / relative).is_file()
        or sha256_file(root / relative).upper() != digest
    )
    blockers = [f"P2_2_FROZEN_SOURCE_CHANGED:{value}" for value in changed]
    if frozen.get("freeze_status") != "P2_2_EXECUTION_FEASIBILITY_FROZEN_GO":
        blockers.append("P2_2_FREEZE_MARKER_INVALID")
    core = {
        key: value
        for key, value in frozen.items()
        if key not in {"status", "created_at", "freeze_hash"}
    }
    if not frozen or frozen.get("freeze_hash") != stable_hash(core):
        blockers.append("P2_2_FREEZE_HASH_INVALID")
    immutable = root / FREEZE_ROOT / f"{frozen.get('freeze_hash')}.json"
    if not immutable.is_file() or _read_json(immutable) != frozen:
        blockers.append("P2_2_IMMUTABLE_FREEZE_COPY_MISMATCH")
    return {
        "schema": "p2_2_execution_feasibility_freeze_verification_v1",
        "status": "GO" if not blockers else "NO_GO",
        "freeze_hash": frozen.get("freeze_hash"),
        "changed_sources": changed,
        "blockers": sorted(set(blockers)),
        "broker_writes": 0,
    }


def _candidate_rows(
    opportunities: list[dict[str, Any]], watchlist: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in opportunities:
        strategies = list(row.get("strategy_ids") or [])
        if row.get("strategy_id") and row.get("strategy_id") not in strategies:
            strategies.append(row.get("strategy_id"))
        if not strategies and row.get("strategy_family"):
            strategies.append(row.get("strategy_family"))
        if not strategies:
            strategies = [None]
        for strategy in strategies:
            key = (
                str(row.get("symbol") or "").upper(),
                str(strategy or "UNIDENTIFIED"),
                str(row.get("holding_horizon") or ""),
            )
            if not key[0] or key in seen:
                continue
            seen.add(key)
            candidates.append({**row, "contributing_strategy_id": strategy})
    for row in watchlist:
        symbol = str(row.get("symbol") or "").upper()
        if any(key[0] == symbol for key in seen):
            continue
        candidates.append(
            {
                "symbol": symbol,
                "asset_class": row.get("asset_type"),
                "sector": row.get("sector"),
                "strategy_ids": [],
                "contributing_strategy_id": None,
                "strategy_evidence_status": "WATCHLIST_ONLY",
                "shariah_status": (
                    "SHARIAH_ALLOWED"
                    if str(row.get("sector") or "").startswith("SHARIAH_")
                    else "SHARIAH_DATA_UNAVAILABLE"
                ),
                "blockers": list(row.get("rejection_reasons", [])),
                "learning_overlay": {},
                "strategy_binding": {"status": "NO_GO", "blockers": ["NO_CONTRIBUTING_STRATEGY"]},
            }
        )
    return candidates


def _candidate_blockers(
    *,
    candidate: dict[str, Any],
    account: dict[str, Any],
    live_authority: dict[str, Any],
    binding: dict[str, Any],
    price_local: Decimal | None,
    price_eur: Decimal | None,
    stop_local: Decimal | None,
    risk_per_share: Decimal | None,
    final_qty: int,
    live_quote_available: bool,
    price_live_actionable: bool,
    expected_return: Decimal | None,
    expected_net_edge: Decimal | None,
    cost_to_edge: Decimal | None,
    maximum_cost_to_edge: Decimal,
    maximum_event_risk: Decimal,
) -> list[str]:
    blockers: list[str] = []
    if not candidate.get("contributing_strategy_id"):
        blockers.append("CONTRIBUTING_STRATEGY_UNIDENTIFIED")
    if binding.get("status") != "GO":
        blockers.append("EXACT_STRATEGY_LIVE_AUTHORITY_NOT_PROVEN")
    if candidate.get("shariah_status") != "SHARIAH_ALLOWED":
        blockers.append("SHARIAH_NOT_ALLOWED_OR_NOT_PROVEN")
    if account.get("status") != "GO" or not account.get("fresh_reconciliation"):
        blockers.append("FRESH_RECONCILED_ACCOUNT_UNAVAILABLE")
    if live_authority.get("execution_authority") not in {
        "AUTONOMOUS_LEVEL_ONE",
        "LIVE_LEVEL_ONE",
    } or live_authority.get("lifecycle_status") not in {None, "ACTIVE"}:
        blockers.append("LEVEL_ONE_AUTHORITY_NOT_ACTIVE")
    if not live_quote_available:
        blockers.append("REALTIME_LEVEL_ONE_QUOTE_ENTITLEMENT_UNAVAILABLE")
    if not price_live_actionable:
        blockers.append("LIVE_ACTIONABLE_ENTRY_PRICE_UNAVAILABLE")
    if price_local is None:
        blockers.append("ENTRY_PRICE_UNAVAILABLE")
    if price_eur is None:
        blockers.append("EUR_WHOLE_SHARE_PRICE_UNAVAILABLE")
    if stop_local is None or risk_per_share is None:
        blockers.append("VALID_PROTECTIVE_STOP_UNAVAILABLE")
    if risk_per_share is not None and risk_per_share > Decimal("9"):
        blockers.append("ONE_SHARE_STOP_RISK_EXCEEDS_EUR9")
    if final_qty < 1:
        blockers.append("NO_WHOLE_SHARE_WITHIN_ALL_LEVEL_ONE_LIMITS")
    if expected_return is None or expected_return <= 0:
        blockers.append("POSITIVE_EXPECTED_GROSS_EDGE_NOT_PROVEN")
    if expected_net_edge is None or expected_net_edge <= 0:
        blockers.append("POSITIVE_EXPECTED_NET_EDGE_NOT_PROVEN")
    if cost_to_edge is None or cost_to_edge > maximum_cost_to_edge:
        blockers.append("COST_TO_EDGE_RATIO_NOT_GO")
    if candidate.get("liquidity") is None:
        blockers.append("LIQUIDITY_EVIDENCE_UNAVAILABLE")
    if candidate.get("spread_bps") is None:
        blockers.append("OBSERVED_SPREAD_UNAVAILABLE")
    event_risk = _optional_decimal(candidate.get("event_risk"))
    if event_risk is None:
        blockers.append("EVENT_RISK_UNAVAILABLE")
    elif event_risk > maximum_event_risk:
        blockers.append("EVENT_RISK_EXCEEDS_LIMIT")
    blockers.extend(str(value) for value in candidate.get("blockers", []))
    return list(dict.fromkeys(blockers))


def _summarize_report(
    results: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    equity: Decimal,
    available_cash: Decimal,
    risk_budget: Decimal,
    realtime_status: str,
    opportunity_count: int,
    watchlist_count: int,
    level_two_evidence: dict[str, Any],
) -> dict[str, Any]:
    primary = Counter(
        row["primary_rejection_reason"]
        for row in results
        if row.get("primary_rejection_reason")
    )
    all_gates = Counter(
        blocker
        for row in results
        for blocker in row.get("exact_rejection_reasons", [])
    )
    marginal = {
        gate: sum(
            1
            for row in results
            if set(row.get("exact_rejection_reasons", [])) == {gate}
        )
        for gate in all_gates
    }
    biggest_gate = min(
        ((-count, gate) for gate, count in all_gates.items()),
        default=(0, None),
    )[1]
    sensitivity: dict[str, Any] = {}
    for cap in policy["notional"]["sensitivity_caps_eur"]:
        key = str(int(cap))
        sensitivity[key] = {
            "notional_cap_eur": int(cap),
            "whole_share_price_capacity_count": sum(
                row.get("notional_sensitivity", {})
                .get(key, {})
                .get("whole_share_notional_feasible", False)
                for row in results
            ),
            "policy_activated": int(cap)
            == int(policy["notional"]["active_hard_cap_eur"]),
        }
    feasible = [row for row in results if row["feasibility_status"] == "FEASIBLE_NOW"]
    body: dict[str, Any] = {
        "schema": "p2_2_execution_feasibility_report_v1",
        "status": "GO",
        "decision_status": "FEASIBLE_CANDIDATES_AVAILABLE"
        if feasible
        else "NO_FEASIBLE_CANDIDATE_RETAIN_CASH",
        "generated_at": _now(),
        "account_equity_eur": _number(equity),
        "available_cash_eur": _number(available_cash),
        "level_one_risk_budget_eur": _number(risk_budget),
        "risk_budget_semantics": "MAXIMUM_LOSS_TO_PROTECTIVE_STOP",
        "whole_share_only": True,
        "fractional_shares_allowed": False,
        "active_notional_cap_eur": policy["notional"]["active_hard_cap_eur"],
        "notional_cap_automatically_changed": False,
        "realtime_level_one_status": realtime_status,
        "source_opportunity_count": opportunity_count,
        "source_watchlist_count": watchlist_count,
        "evaluated_symbol_strategy_count": len(results),
        "feasible_now_count": len(feasible),
        "rejected_count": len(results) - len(feasible),
        "candidates": results,
        "primary_rejection_distribution": dict(primary.most_common()),
        "all_gate_rejection_distribution": dict(all_gates.most_common()),
        "single_gate_marginal_recovery_count": marginal,
        "biggest_marginal_loss_gate": biggest_gate,
        "notional_sensitivity": sensitivity,
        "newly_justified_strategy_bindings": [],
        "strategy_registry_mutated": False,
        "automatic_strategy_promotion": False,
        "learning_may_grant_authority": False,
        "level_two": {
            "status": level_two_evidence.get("status", "NO_GO"),
            "verified_round_trip_count": int(
                level_two_evidence.get("verified_round_trip_count", 0) or 0
            ),
            "minimum_round_trips": int(
                level_two_evidence.get("minimum_round_trips", 5) or 5
            ),
            "activated": False,
        },
        "cash_is_valid_outcome": True,
        "forced_signal_created": False,
        "orders_created": 0,
        "orders_submitted": 0,
        "broker_writes": 0,
    }
    body["content_hash"] = stable_hash(body)
    return body


def _instrument_map(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "output/universe/instruments.parquet"
    if not path.is_file():
        return {}
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return {}
    if "symbol" not in frame:
        return {}
    columns = [
        value
        for value in (
            "instrument_id",
            "symbol",
            "asset_type",
            "currency",
            "primary_exchange",
            "compliance_status",
        )
        if value in frame
    ]
    return {
        str(row["symbol"]).upper(): row.to_dict()
        for _, row in frame.loc[:, columns].drop_duplicates("symbol", keep="last").iterrows()
    }


def _fx_map(root: Path) -> dict[str, Decimal]:
    rates = {"EUR": Decimal("1")}
    path = root / "data/fx/fx_daily.parquet"
    if not path.is_file():
        return rates
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return rates
    if not {"base_currency", "quote_currency", "rate"}.issubset(frame.columns):
        return rates
    if "session_date" in frame:
        frame = frame.sort_values("session_date")
    for _, row in frame.iterrows():
        if str(row["quote_currency"]).upper() == "EUR":
            value = _optional_decimal(row["rate"])
            if value is not None and value > 0:
                rates[str(row["base_currency"]).upper()] = value
    return rates


def _stage0_price_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("symbol") or "").upper(): row
        for row in payload.get("survivors", [])
        if row.get("symbol")
    }


def _validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema") != "p2_2_execution_feasibility_policy_v1":
        raise ValueError("P2_2_POLICY_REQUIRED")
    if _decimal(policy.get("risk_budget", {}).get("absolute_cap_eur")) != 9:
        raise ValueError("P2_2_EUR9_RISK_CAP_REQUIRED")
    constraints = policy.get("constraints", {})
    if not constraints.get("whole_shares_only") or constraints.get(
        "fractional_shares_allowed", True
    ):
        raise ValueError("P2_2_WHOLE_SHARE_POLICY_REQUIRED")


def _floor_div(numerator: Decimal, denominator: Decimal | None) -> int:
    if denominator is None or denominator <= 0 or numerator <= 0:
        return 0
    return max(
        0,
        int((numerator / denominator).to_integral_value(rounding=ROUND_DOWN)),
    )


def _first_positive(*values: Any) -> Decimal | None:
    for value in values:
        parsed = _optional_decimal(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return parsed if parsed.is_finite() else None


def _decimal(value: Any) -> Decimal:
    return _optional_decimal(value) or Decimal("0")


def _number(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != payload:
            raise FileExistsError(f"immutable freeze collision: {path}")
        return
    _atomic_json(path, payload)


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "P2_2_SOURCES",
    "build_execution_feasibility_report",
    "build_p2_2_freeze",
    "publish_execution_feasibility_report",
    "verify_p2_2_freeze",
]
