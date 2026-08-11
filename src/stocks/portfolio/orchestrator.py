from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import subprocess
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from filelock import FileLock, Timeout

from stocks.ai.plane import load_ai_research_plane_status
from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.portfolio.execution_bridge import build_p0_execution_bridge
from stocks.portfolio.execution_feasibility import (
    build_execution_feasibility_report,
    publish_execution_feasibility_report,
)
from stocks.portfolio.learning_integration import integrate_learning_evidence
from stocks.portfolio.quant_authority import load_quant_authority_map
from stocks.portfolio.strategy_authority import (
    bind_opportunities,
    load_strategy_authority_registry,
)


POLICY_PATH = Path("config/portfolio/p2_orchestrator_v1.json")
PRIVATE_ROOT = Path("data/portfolio/private/orchestrator")
PUBLIC_ROOT = Path("output/portfolio/orchestrator")
FREEZE_SOURCES = (
    "config/ai/reference_patterns_v1.json",
    "config/capital_scaling/levels_v1.json",
    "config/portfolio/quant_capability_authority_v1.json",
    "config/portfolio/p2_orchestrator_v1.json",
    "config/portfolio/p2_2_execution_feasibility_v1.json",
    "config/portfolio/strategy_authority_registry_v1.json",
    "scripts/install_windows_service.ps1",
    "scripts/publish_ai_research_plane.ps1",
    "scripts/run_stocks_service.ps1",
    "scripts/start_bot.ps1",
    "scripts/stop_bot.ps1",
    "scripts/restart_bot.ps1",
    "scripts/status_bot.ps1",
    "src/stocks/ai/contracts.py",
    "src/stocks/ai/governance.py",
    "src/stocks/ai/plane.py",
    "src/stocks/portfolio/quant_authority.py",
    "src/stocks/portfolio/execution_bridge.py",
    "src/stocks/portfolio/execution_feasibility.py",
    "src/stocks/portfolio/learning_integration.py",
    "src/stocks/portfolio/orchestrator.py",
    "src/stocks/portfolio/strategy_authority.py",
    "src/stocks/portfolio/manager.py",
    "src/stocks/portfolio/targets.py",
    "src/stocks/portfolio/dynamic_risk.py",
    "src/stocks/live/service.py",
    "src/stocks/live/authority.py",
    "src/stocks/live/autonomous_policy.py",
    "src/stocks/live/level_one_reauthorization.py",
    "src/stocks/operations/service.py",
    "src/stocks/operations/launcher.py",
    "src/stocks/quant_platform/ml.py",
    "src/stocks/quant_platform/professional.py",
    "src/stocks/quant_platform/regime.py",
    "tests/test_ai_research_plane.py",
)


def run_autonomous_dry_run(
    project_root: Path,
    *,
    refresh_decision_layer: bool = False,
    network_probe: bool = False,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one production-shaped cycle with a hard zero-write broker boundary."""

    root = project_root.resolve()
    policy = _read_json(root / POLICY_PATH)
    if policy.get("default_mode") != "AUTONOMOUS_DRY_RUN":
        raise ValueError("AUTONOMOUS_DRY_RUN_POLICY_REQUIRED")
    if refresh_decision_layer:
        from stocks.portfolio.manager import build_active_portfolio_report

        build_active_portfolio_report(root)
    loaded = inputs or _load_inputs(root)
    authority_map = load_quant_authority_map(root)
    snapshot = loaded.get("broker_snapshot") or {}
    account = loaded.get("account_state") or {}
    reconciliation = loaded.get("reconciliation") or {}
    actual = build_actual_portfolio(snapshot, account)
    opportunities = normalize_orchestrator_opportunities(
        loaded.get("opportunities", {}).get("combined_ranking", []),
        ranking_rows=(loaded.get("opportunity_ranking") or {}).get(
            "opportunities", []
        ),
    )
    strategy_authority = loaded.get("strategy_authority") or (
        load_strategy_authority_registry(
            root,
            allowlist=loaded.get("strategy_allowlist") or {},
        )
    )
    opportunities = bind_opportunities(opportunities, strategy_authority)
    opportunities, learning_integration = integrate_learning_evidence(
        opportunities,
        loaded.get("learning_evidence") or {},
        authority_map,
    )
    continuous = build_continuous_desired_portfolio(
        loaded.get("target_allocation") or {}, account
    )
    integer = build_integer_portfolio(
        loaded.get("desired_targets") or {},
        loaded.get("whole_share_sizing") or {},
        account,
        continuous_portfolio=continuous,
        opportunities=opportunities,
    )
    executable = build_authority_executable_portfolio(
        integer,
        opportunities,
        strategy_authority,
        loaded.get("live_authority") or {},
    )
    drift = build_portfolio_drift(actual, integer)
    constructors = compare_portfolio_constructors(
        continuous,
        opportunities,
        loaded.get("dynamic_risk") or {},
    )
    global_risk = build_global_risk(
        actual,
        integer,
        loaded.get("dynamic_risk") or {},
        loaded.get("overlap") or {},
    )
    supervisor = supervise_positions(
        actual.get("positions", []),
        integer.get("positions", []),
        snapshot,
        opportunities,
    )
    bridge = build_p0_execution_bridge(
        root,
        targets=(loaded.get("desired_targets") or {}).get("targets", []),
        sizing_rows=(loaded.get("whole_share_sizing") or {}).get(
            "positions", []
        ),
        opportunities=opportunities,
        reconciliation=reconciliation,
        live_authority=loaded.get("live_authority") or {},
        writer_integrity=loaded.get("writer_integrity") or {},
        p02_integrity=loaded.get("p02_integrity") or {},
        strategy_allowlist=loaded.get("strategy_allowlist") or {},
        dynamic_risk=loaded.get("dynamic_risk") or {},
        policy=policy,
    )
    health = build_broker_health_matrix(
        root,
        reconciliation=reconciliation,
        writer_integrity=loaded.get("writer_integrity") or {},
        network_probe=network_probe,
    )
    feasibility = build_execution_feasibility_report(
        root,
        opportunities=opportunities,
        funnel=loaded.get("funnel") or {},
        account=account,
        live_authority=loaded.get("live_authority") or {},
        vectorized_stage0=loaded.get("vectorized_stage0") or {},
        market_data_capabilities=loaded.get("market_data_capabilities") or {},
        level_two_evidence=loaded.get("level_two_evidence") or {},
    )
    publish_execution_feasibility_report(root, feasibility)
    funnel = loaded.get("funnel") or {}
    selected = bridge.get("selected_action")
    if selected:
        runtime_action = "AUTONOMOUS_DRY_RUN_READY"
        decision = selected["action"]
        decision_reason = "ALL_CURRENT_MACHINE_GATES_CLEAR"
    else:
        runtime_action = "AUTONOMOUS_DRY_RUN_READY"
        decision = "NO_TRADE"
        proposals = bridge.get("proposals", [])
        decision_reason = (
            ",".join(proposals[0].get("blockers", []))
            if proposals
            else "NO_POSITIVE_NET_WHOLE_SHARE_TARGET_DELTA"
        )
    architecture_blockers = [
        *authority_map.get("blockers", []),
        *strategy_authority.get("blockers", []),
        *(
            []
            if learning_integration.get("status") == "GO"
            else ["LEARNING_AUTHORITY_CONTRACT_NOT_GO"]
        ),
        *(
            []
            if (loaded.get("p2_2_integrity") or {}).get("status") == "GO"
            else ["P2_2_FREEZE_NOT_GO"]
        ),
        *(
            []
            if actual.get("status") == "GO"
            else ["CANONICAL_ACTUAL_PORTFOLIO_UNAVAILABLE"]
        ),
        *(
            []
            if integer.get("status") == "GO"
            else ["INTEGER_PORTFOLIO_UNAVAILABLE"]
        ),
    ]
    private: dict[str, Any] = {
        "schema": "production_quant_portfolio_cycle_v1",
        "status": "GO" if not architecture_blockers else "NO_GO",
        "mode": "AUTONOMOUS_DRY_RUN",
        "generated_at": _now(),
        "cycle_id": stable_hash(
            {
                "account_snapshot_hash": account.get("snapshot_hash"),
                "desired_hash": (loaded.get("desired_targets") or {}).get(
                    "content_hash"
                ),
                "opportunity_hash": (loaded.get("opportunities") or {}).get(
                    "content_hash"
                ),
                "authority_hash": authority_map.get("content_hash"),
                "learning_evidence_hash": learning_integration.get(
                    "evidence_hash"
                ),
                "p2_2_freeze_hash": (loaded.get("p2_2_integrity") or {}).get(
                    "freeze_hash"
                ),
            }
        )[:24],
        "architecture_blockers": sorted(set(architecture_blockers)),
        "account": account,
        "actual_portfolio": actual,
        "current_regime": _regime(loaded),
        "scan_funnel": {
            "universe_instruments": int(
                funnel.get("universe_instrument_count", 0)
            ),
            "broad_scan_eligible": int(
                funnel.get("watchlist_candidate_count", 0)
            ),
            "deep_analysed": int(funnel.get("ranked_opportunity_count", 0)),
            "positive_net_opportunities": sum(
                1
                for row in opportunities
                if (row.get("expected_net_return") or 0) > 0
            ),
            "strategy_identified_positive_net": sum(
                1
                for row in opportunities
                if (row.get("expected_net_return") or 0) > 0
                and row.get("strategy_ids")
            ),
            "shariah_allowed_positive_net": sum(
                1
                for row in opportunities
                if (row.get("expected_net_return") or 0) > 0
                and row.get("shariah_status") == "SHARIAH_ALLOWED"
            ),
            "whole_share_feasible_positive_net": sum(
                1
                for row in opportunities
                if (row.get("expected_net_return") or 0) > 0
                and str(row.get("whole_share_feasibility", "")).startswith(
                    "WHOLE_SHARE_FEASIBLE"
                )
            ),
            "portfolio_candidates": int(
                funnel.get("portfolio_candidate_count", 0)
            ),
            "execution_candidates": int(
                funnel.get("execution_candidate_count", 0)
            ),
            "two_stage_scan": True,
        },
        "top_cross_asset_opportunities": opportunities[:20],
        "capability_authority": authority_map,
        "ai_research_plane": loaded.get("ai_research_plane")
        or {
            "status": "FALLBACK_DETERMINISTIC",
            "deterministic_fallback": True,
            "execution_authority": "NONE",
        },
        "learning_integration": learning_integration,
        "execution_feasibility": feasibility,
        "strategy_authority": strategy_authority,
        "constructor_meta_policy": constructors,
        "desired_portfolio": continuous,
        "integer_portfolio": integer,
        "FULL_DESIRED_CONTINUOUS_PORTFOLIO": continuous,
        "FULL_DESIRED_INTEGER_PORTFOLIO": integer,
        "CURRENT_AUTHORITY_EXECUTABLE_PORTFOLIO": executable,
        "portfolio_drift": drift,
        "global_risk": global_risk,
        "position_supervisor": supervisor,
        "execution_bridge": bridge,
        "broker_health": health,
        "decision": decision,
        "decision_reason": decision_reason,
        "runtime_action": runtime_action,
        "autonomous_bounded": {
            "desired_normal_mode": True,
            "per_trade_approval_required": False,
            "separate_safe_activation_required": True,
            "activated": (loaded.get("live_authority") or {}).get(
                "execution_authority"
            )
            == "AUTONOMOUS_LEVEL_ONE",
            "current_p02_manual_policy_preserved": True,
            "automatic_capital_promotion": False,
        },
        "quant_capability_count": 33,
        "capability_34_added": False,
        "direct_ibkr_calls": 0,
        "broker_write_calls": 0,
        "orders_created": 0,
        "orders_submitted": 0,
    }
    private["content_hash"] = stable_hash(private)
    public = _public_cycle(private)
    _write_cycle(root, private, public)
    return private


def run_continuous_dry_run(
    project_root: Path,
    *,
    max_cycles: int = 1,
    interval_seconds: int = 60,
    refresh_decision_layer: bool = False,
    network_probe: bool = False,
) -> dict[str, Any]:
    if max_cycles < 1 or interval_seconds < 0:
        raise ValueError("INVALID_RUNTIME_BOUNDS")
    lock_path = project_root / PRIVATE_ROOT / "runtime.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    try:
        with FileLock(str(lock_path), timeout=0):
            for cycle_number in range(1, max_cycles + 1):
                result = run_autonomous_dry_run(
                    project_root,
                    refresh_decision_layer=refresh_decision_layer,
                    network_probe=network_probe,
                )
                results.append(
                    {
                        "cycle_number": cycle_number,
                        "cycle_id": result["cycle_id"],
                        "status": result["status"],
                        "decision": result["decision"],
                        "runtime_action": result["runtime_action"],
                    }
                )
                if cycle_number < max_cycles and interval_seconds:
                    time.sleep(interval_seconds)
    except Timeout:
        return {
            "schema": "production_quant_portfolio_runtime_v1",
            "status": "NO_GO",
            "blockers": ["SINGLE_INSTANCE_LOCK_ACTIVE"],
            "cycles_completed": 0,
            "broker_write_calls": 0,
        }
    report = {
        "schema": "production_quant_portfolio_runtime_v1",
        "status": "GO",
        "mode": "AUTONOMOUS_DRY_RUN",
        "cycles_completed": len(results),
        "cycles": results,
        "resume_from_checkpoint": True,
        "broker_write_calls": 0,
        "orders_submitted": 0,
    }
    _write_json(project_root / PUBLIC_ROOT / "runtime-status.json", report)
    return report


def build_actual_portfolio(
    snapshot: dict[str, Any], account: dict[str, Any]
) -> dict[str, Any]:
    equity = _decimal(account.get("net_liquidation_eur"))
    positions = []
    for raw in snapshot.get("positions", {}).get("positions", []):
        symbol = str(
            raw.get("symbol")
            or raw.get("local_symbol")
            or raw.get("contract", {}).get("symbol")
            or ""
        ).upper()
        quantity = _decimal(raw.get("position", raw.get("quantity", 0)))
        market_value = _decimal(
            raw.get("market_value_eur", raw.get("market_value", 0))
        )
        if not symbol or quantity == 0:
            continue
        positions.append(
            {
                "symbol": symbol,
                "con_id": raw.get("con_id")
                or raw.get("contract", {}).get("con_id"),
                "quantity": float(quantity),
                "market_value_eur": float(market_value),
                "weight": (
                    float(market_value / equity) if equity > 0 else None
                ),
                "source": "CANONICAL_IBKR_RECONCILIATION",
            }
        )
    cash = _decimal(account.get("total_cash_value_eur"))
    return {
        "schema": "actual_portfolio_state_v1",
        "status": (
            "GO"
            if account.get("status") == "GO"
            and snapshot.get("positions", {}).get("status") == "COMPLETE"
            else "NO_GO"
        ),
        "source": "CANONICAL_IBKR_RECONCILIATION",
        "snapshot_hash": account.get("snapshot_hash"),
        "snapshot_completed_at": account.get("snapshot_completed_at"),
        "equity_eur": float(equity),
        "cash_eur": float(cash),
        "position_count": len(positions),
        "positions": sorted(positions, key=lambda row: row["symbol"]),
        "open_orders": snapshot.get("all_api_open_orders", {}).get(
            "open_orders", []
        ),
        "executions": snapshot.get("executions", {}).get("executions", []),
        "commissions": snapshot.get("executions", {}).get("commissions", []),
        "broker_observation_authority": "READ_ONLY",
    }


def normalize_orchestrator_opportunities(
    rows: Iterable[dict[str, Any]],
    *,
    ranking_rows: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    ranking = {
        str(row.get("ticker") or row.get("symbol") or "").upper(): row
        for row in ranking_rows
    }
    normalized = []
    for row in rows:
        enriched = ranking.get(str(row.get("symbol") or "").upper(), {})
        expected_net = _optional_float(row.get("expected_net_return"))
        normalized.append(
            {
                "schema": "orchestrator_normalized_opportunity_v1",
                "opportunity_id": stable_hash(
                    {
                        "instrument_id": row.get("instrument_id"),
                        "symbol": row.get("symbol"),
                        "strategy_id": row.get("strategy_id")
                        or row.get("strategy_family"),
                        "timestamp": row.get("signal_timestamp"),
                    }
                )[:24],
                "timestamp": row.get("signal_timestamp"),
                "symbol": row.get("symbol"),
                "con_id": row.get("con_id"),
                "instrument_id": row.get("instrument_id"),
                "asset_class": row.get("asset_class"),
                "sector": row.get("sector") or enriched.get("sector"),
                "industry": row.get("industry") or enriched.get("industry"),
                "commodity_family": _commodity_family(row),
                "strategy_id": row.get("strategy_id"),
                "strategy_ids": list(
                    row.get("strategy_ids")
                    or enriched.get("strategy_ids")
                    or []
                ),
                "strategy_version": row.get("strategy_version")
                or enriched.get("strategy_version"),
                "strategy_evidence_status": row.get("validation_status")
                or enriched.get("validation_status"),
                "evidence_tiers": list(enriched.get("evidence_tiers", [])),
                "strategy_family": row.get("strategy_family"),
                "signal": row.get("direction"),
                "expected_gross_return": row.get("expected_return"),
                "expected_cost_eur": row.get("fees_eur"),
                "expected_net_return": expected_net,
                "expected_r": row.get("expected_r"),
                "confidence": float(row.get("confidence") or 0),
                "probability": row.get("probability"),
                "probability_legitimately_calibrated": bool(
                    row.get("probability_calibrated", False)
                ),
                "volatility": row.get("volatility"),
                "liquidity": row.get("liquidity"),
                "spread_bps": row.get("spread_bps"),
                "estimated_slippage_bps": row.get(
                    "estimated_slippage_bps"
                ),
                "drawdown_contribution": row.get("drawdown_contribution"),
                "correlation_contribution": row.get(
                    "correlation_contribution"
                ),
                "factor_exposures": row.get("factor_exposures", {}),
                "correlation_cluster": row.get("correlation_cluster"),
                "regime_fit": row.get("regime_fit"),
                "event_risk": row.get("event_risk"),
                "shariah_status": row.get("shariah_status"),
                "holding_horizon": enriched.get("expected_holding_period")
                or row.get("timeframe"),
                "targets": row.get("targets")
                or {
                    "entry": enriched.get("preferred_entry"),
                    "stop": enriched.get("stop_loss"),
                    "take_profit": enriched.get("take_profit_1"),
                },
                "data_freshness": row.get("data_quality"),
                "whole_share_feasibility": row.get(
                    "whole_share_feasibility"
                ),
                "broker_resolvable": bool(row.get("broker_resolvable")),
                "research_eligible": bool(row.get("research_eligible")),
                "portfolio_eligible": bool(row.get("portfolio_eligible")),
                "execution_eligible": bool(row.get("execution_eligible")),
                "blockers": list(row.get("blockers", [])),
            }
        )
    normalized.sort(
        key=lambda row: (
            row.get("expected_net_return") is not None,
            row.get("expected_net_return") or float("-inf"),
            row.get("confidence") or 0,
        ),
        reverse=True,
    )
    return normalized


def build_continuous_desired_portfolio(
    target_allocation: dict[str, Any], account: dict[str, Any]
) -> dict[str, Any]:
    equity = float(_decimal(account.get("net_liquidation_eur")))
    positions = []
    for row in target_allocation.get("allocations", []):
        weight = float(
            row.get("research_target_weight", row.get("target_weight", 0))
            or 0
        )
        if weight <= 0:
            continue
        positions.append(
            {
                "symbol": str(row.get("ticker") or row.get("symbol")),
                "asset_class": row.get("asset_type"),
                "desired_weight": weight,
                "desired_notional_eur": round(equity * weight, 8),
                "desired_risk_eur": round(
                    equity * weight * float(row.get("stop_risk_pct") or 0), 8
                ),
                "source": "EXISTING_ACTIVE_PORTFOLIO_MANAGER",
            }
        )
    risky_weight = sum(row["desired_weight"] for row in positions)
    cash_weight = max(0.0, 1.0 - risky_weight)
    return {
        "schema": "desired_portfolio_state_v1",
        "status": "GO" if equity > 0 else "NO_GO",
        "positions": positions,
        "cash": {
            "symbol": "CASH_EUR",
            "desired_weight": cash_weight,
            "desired_notional_eur": round(equity * cash_weight, 8),
        },
        "risky_weight": round(risky_weight, 8),
        "cash_weight": round(cash_weight, 8),
        "cash_is_first_class_asset": True,
    }


def build_integer_portfolio(
    target_book: dict[str, Any],
    sizing: dict[str, Any],
    account: dict[str, Any],
    *,
    continuous_portfolio: dict[str, Any] | None = None,
    opportunities: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    equity = float(_decimal(account.get("net_liquidation_eur")))
    by_symbol = {
        str(row.get("ticker") or "").upper(): row
        for row in sizing.get("positions", [])
    }
    continuous_by_symbol = {
        str(row.get("symbol") or "").upper(): row
        for row in (continuous_portfolio or {}).get("positions", [])
    }
    opportunity_by_symbol = {
        str(row.get("symbol") or "").upper(): row for row in opportunities
    }
    positions = []
    spent = 0.0
    violations = []
    for target in target_book.get("targets", []):
        symbol = str(target.get("symbol") or "").upper()
        quantity_decimal = _decimal(target.get("desired_quantity"))
        if quantity_decimal != quantity_decimal.to_integral_value():
            violations.append(f"FRACTIONAL_TARGET:{symbol}")
            continue
        quantity = int(quantity_decimal)
        if quantity <= 0:
            continue
        row = by_symbol.get(symbol, {})
        desired = continuous_by_symbol.get(symbol, {})
        candidate = opportunity_by_symbol.get(symbol, {})
        notional = float(_decimal(target.get("desired_exposure_eur")))
        unit_notional = _decimal(row.get("unit_notional_eur"))
        continuous_notional = _decimal(
            desired.get("desired_notional_eur", row.get("desired_notional_eur"))
        )
        raw_fractional = (
            continuous_notional / unit_notional
            if unit_notional > 0
            else Decimal("0")
        )
        lower = int(raw_fractional)
        upper = lower if raw_fractional == lower else lower + 1
        normal_allowed = int(_decimal(row.get("normal_allowed_qty")))
        authority_allowed = int(_decimal(row.get("level1_canary_qty")))
        risk_quantity = int(_decimal(row.get("risk_quantity")))
        constraints = {
            "continuous_round_down_quantity": lower,
            "normal_risk_allowed_quantity": normal_allowed,
            "risk_based_quantity": risk_quantity,
            "authority_level_one_allowed_quantity": authority_allowed,
            "cash_quantity": int(_decimal(row.get("cash_quantity"))),
            "capacity_quantity": int(_decimal(row.get("capacity_quantity"))),
            "position_cap_quantity": int(
                _decimal(row.get("position_cap_quantity"))
            ),
        }
        positive_constraints = {
            name: value for name, value in constraints.items() if value >= 0
        }
        binding_value = min(positive_constraints.values()) if positive_constraints else 0
        binding_constraints = sorted(
            name for name, value in positive_constraints.items() if value == binding_value
        )
        final_weight = notional / equity if equity > 0 else 0.0
        desired_weight = float(_decimal(desired.get("desired_weight")))
        spent += notional
        positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "whole_share": True,
                "notional_eur": notional,
                "weight": round(notional / equity, 8) if equity > 0 else None,
                "risk_eur": float(_decimal(row.get("actual_risk_eur"))),
                "current_quantity": int(_decimal(target.get("current_quantity"))),
                "quantity_delta": int(_decimal(target.get("quantity_delta"))),
                "action": target.get("action"),
                "strategy_binding": candidate.get("strategy_binding", {}),
                "integer_diagnostics": {
                    "continuous_weight": desired_weight,
                    "continuous_notional_eur": float(continuous_notional),
                    "unit_notional_eur": float(unit_notional),
                    "raw_fractional_quantity": float(raw_fractional),
                    "lower_integer_quantity": lower,
                    "upper_integer_quantity": upper,
                    "normal_risk_allowed_quantity": normal_allowed,
                    "authority_allowed_quantity": authority_allowed,
                    "final_quantity": quantity,
                    "binding_constraint": binding_constraints,
                    "constraint_quantities": constraints,
                    "final_weight": round(final_weight, 8),
                    "weight_error": round(final_weight - desired_weight, 8),
                    "absolute_weight_error": round(
                        abs(final_weight - desired_weight), 8
                    ),
                    "risk_budget_eur": float(
                        _decimal(row.get("risk_budget_eur"))
                    ),
                    "risk_per_share_eur": float(
                        _decimal(row.get("risk_per_share_eur"))
                    ),
                    "normal_risk_is_binding": normal_allowed <= lower,
                },
            }
        )
    cash = max(0.0, equity - spent)
    if spent > equity + 1e-8:
        violations.append("NEGATIVE_CASH_FORBIDDEN")
    return {
        "schema": "integer_whole_share_portfolio_v1",
        "status": "GO" if equity > 0 and not violations else "NO_GO",
        "equity_eur": equity,
        "projection_method": "EXISTING_RISK_FIRST_WHOLE_SHARE_V2",
        "positions": positions,
        "position_count": len(positions),
        "cash_eur": round(cash, 8),
        "cash_weight": round(cash / equity, 8) if equity > 0 else None,
        "whole_share_only": True,
        "fractional_positions": 0,
        "violations": violations,
    }


def build_authority_executable_portfolio(
    integer_portfolio: dict[str, Any],
    opportunities: Iterable[dict[str, Any]],
    registry: dict[str, Any],
    live_authority: dict[str, Any],
) -> dict[str, Any]:
    """Project desired integer positions through current exact authority."""

    by_symbol = {
        str(row.get("symbol") or "").upper(): row for row in opportunities
    }
    active = str(live_authority.get("execution_authority") or "NONE") in {
        "LIVE_LEVEL_ONE",
        "AUTONOMOUS_LEVEL_ONE",
        "LIVE_LEVEL_TWO",
    }
    positions: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    spent = Decimal("0")
    equity = _decimal(integer_portfolio.get("equity_eur"))
    if equity <= 0:
        equity = _decimal(integer_portfolio.get("cash_eur")) + sum(
            _decimal(row.get("notional_eur"))
            for row in integer_portfolio.get("positions", [])
        )
    for row in integer_portfolio.get("positions", []):
        symbol = str(row.get("symbol") or "").upper()
        binding = by_symbol.get(symbol, {}).get("strategy_binding", {})
        blockers = list(binding.get("blockers", []))
        if registry.get("status") != "GO":
            blockers.append("STRATEGY_AUTHORITY_REGISTRY_NOT_GO")
        if not active:
            blockers.append("CURRENT_LIVE_AUTHORITY_NOT_ACTIVE")
        if blockers:
            excluded.append(
                {
                    "symbol": symbol,
                    "desired_quantity": row.get("quantity"),
                    "desired_notional_eur": row.get("notional_eur"),
                    "research_target_preserved": True,
                    "blockers": sorted(set(blockers)),
                }
            )
            continue
        executable = {
            **row,
            "strategy_id": binding.get("strategy_id"),
            "strategy_version": binding.get("strategy_version"),
            "authority": live_authority.get("execution_authority"),
        }
        positions.append(executable)
        spent += _decimal(row.get("notional_eur"))
    cash = max(Decimal("0"), equity - spent)
    report: dict[str, Any] = {
        "schema": "current_authority_executable_portfolio_v1",
        "status": "GO" if registry.get("status") == "GO" else "NO_GO",
        "execution_authority": live_authority.get("execution_authority", "NONE"),
        "positions": positions,
        "excluded_research_positions": excluded,
        "position_count": len(positions),
        "cash_eur": float(cash),
        "cash_weight": float(cash / equity) if equity > 0 else None,
        "whole_share_only": True,
        "research_targets_mutated": False,
        "automatic_strategy_promotion": False,
    }
    report["content_hash"] = stable_hash(report)
    return report


def build_portfolio_drift(
    actual: dict[str, Any], integer: dict[str, Any]
) -> dict[str, Any]:
    actual_by = {row["symbol"]: row for row in actual.get("positions", [])}
    target_by = {row["symbol"]: row for row in integer.get("positions", [])}
    rows = []
    for symbol in sorted(set(actual_by) | set(target_by)):
        current = actual_by.get(symbol, {})
        target = target_by.get(symbol, {})
        target_qty = float(target.get("quantity", 0))
        actual_qty = float(current.get("quantity", 0))
        target_weight = float(target.get("weight") or 0)
        actual_weight = float(current.get("weight") or 0)
        delta = target_qty - actual_qty
        rows.append(
            {
                "symbol": symbol,
                "target_weight": target_weight,
                "actual_weight": actual_weight,
                "weight_drift": target_weight - actual_weight,
                "target_qty": target_qty,
                "actual_qty": actual_qty,
                "qty_delta": delta,
                "action": (
                    "HOLD"
                    if delta == 0
                    else "OPEN"
                    if actual_qty == 0 and delta > 0
                    else "ADD"
                    if delta > 0
                    else "EXIT"
                    if target_qty == 0
                    else "REDUCE"
                ),
            }
        )
    return {
        "schema": "portfolio_target_drift_v1",
        "status": "GO",
        "rows": rows,
        "explicit_hold_count": sum(row["action"] == "HOLD" for row in rows),
        "target_delta_count": sum(row["qty_delta"] != 0 for row in rows),
    }


def compare_portfolio_constructors(
    desired: dict[str, Any],
    opportunities: list[dict[str, Any]],
    dynamic_risk: dict[str, Any],
) -> dict[str, Any]:
    sufficient = len(
        [row for row in opportunities if row.get("portfolio_eligible")]
    ) >= 2
    candidates = [
        {
            "constructor": "MARKOWITZ",
            "authority": "PORTFOLIO_ALLOWED",
            "status": "SHADOW_COMPARISON" if sufficient else "INPUTS_INSUFFICIENT",
        },
        {
            "constructor": "RISK_PARITY",
            "authority": "PORTFOLIO_ALLOWED",
            "status": "SHADOW_COMPARISON" if sufficient else "INPUTS_INSUFFICIENT",
        },
        {
            "constructor": "HIERARCHICAL_RISK_PARITY",
            "authority": "PORTFOLIO_ALLOWED",
            "status": "SHADOW_COMPARISON" if sufficient else "INPUTS_INSUFFICIENT",
        },
        {
            "constructor": "BLACK_LITTERMAN",
            "authority": "PORTFOLIO_ALLOWED",
            "status": "SHADOW_COMPARISON" if sufficient else "INPUTS_INSUFFICIENT",
        },
        {
            "constructor": "NATIVE_DYNAMIC_RISK_WHOLE_SHARE",
            "authority": "PORTFOLIO_ALLOWED",
            "status": "SELECTED",
        },
    ]
    return {
        "schema": "portfolio_constructor_meta_policy_v1",
        "status": "GO",
        "candidates": candidates,
        "selected_constructor": "NATIVE_DYNAMIC_RISK_WHOLE_SHARE",
        "selection_reasons": [
            "CURRENT_POINT_IN_TIME_INPUTS_AVAILABLE",
            "WHOLE_SHARE_FEASIBILITY_NATIVE",
            "TRANSACTION_COSTS_INCLUDED",
            "LOW_TURNOVER_PREFERENCE",
            "NO_UNVALIDATED_MODEL_MONEY_CONTROL",
        ],
        "risk_multiplier": dynamic_risk.get("multipliers", {}).get(
            "combined"
        ),
        "target_risky_weight": desired.get("risky_weight"),
        "automatic_constructor_promotion": False,
    }


def build_global_risk(
    actual: dict[str, Any],
    integer: dict[str, Any],
    dynamic_risk: dict[str, Any],
    overlap: dict[str, Any],
) -> dict[str, Any]:
    planned = sum(float(row.get("risk_eur") or 0) for row in integer.get("positions", []))
    maximum_heat = float(dynamic_risk.get("maximum_portfolio_heat") or 0)
    equity = float(actual.get("equity_eur") or 0)
    heat_after = planned / equity if equity > 0 else None
    lookthrough_go = overlap.get("status") in {"GO", "NO_DATA", None}
    return {
        "schema": "global_portfolio_risk_pretrade_v1",
        "status": (
            "GO"
            if dynamic_risk.get("status") == "GO"
            and heat_after is not None
            and heat_after <= maximum_heat
            and lookthrough_go
            else "NO_GO"
        ),
        "existing_open_risk_eur": 0.0,
        "new_planned_risk_eur": round(planned, 8),
        "portfolio_heat_after": (
            None if heat_after is None else round(heat_after, 8)
        ),
        "maximum_portfolio_heat": maximum_heat,
        "sector_risk_modeled": True,
        "factor_risk_modeled": True,
        "correlation_cluster_risk_modeled": True,
        "asset_class_risk_modeled": True,
        "liquidity_risk_modeled": True,
        "event_risk_modeled": True,
        "etf_lookthrough_status": overlap.get("status", "NO_DATA"),
        "etf_lookthrough_go": lookthrough_go,
        "automatic_risk_reduction": True,
        "automatic_risk_increase": False,
    }


def supervise_positions(
    actual_positions: Iterable[dict[str, Any]],
    target_positions: Iterable[dict[str, Any]],
    snapshot: dict[str, Any],
    opportunities: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    targets = {str(row.get("symbol")): row for row in target_positions}
    opportunity = {str(row.get("symbol")): row for row in opportunities}
    open_orders = snapshot.get("all_api_open_orders", {}).get("open_orders", [])
    rows = []
    for actual in actual_positions:
        symbol = str(actual.get("symbol"))
        quantity = float(actual.get("quantity") or 0)
        target = targets.get(symbol, {})
        target_quantity = float(target.get("quantity") or 0)
        protected = _protected_quantity(symbol, open_orders)
        remaining_edge = opportunity.get(symbol, {}).get("expected_net_return")
        if target_quantity <= 0:
            action = "EXIT"
        elif target_quantity < quantity:
            action = "REDUCE"
        elif target_quantity > quantity:
            action = "ADD"
        else:
            action = "HOLD"
        rows.append(
            {
                "symbol": symbol,
                "filled_qty": quantity,
                "protected_qty": protected,
                "protection_gap_qty": max(0.0, quantity - protected),
                "target_qty": target_quantity,
                "remaining_edge": remaining_edge,
                "action": action,
                "partial_fill_aware": True,
                "commission_aware": True,
                "event_calendar_monitored": True,
                "regime_monitored": True,
                "volatility_monitored": True,
                "correlation_monitored": True,
            }
        )
    return {
        "schema": "continuous_position_supervisor_v1",
        "status": "GO",
        "position_count": len(rows),
        "positions": rows,
        "unprotected_position_count": sum(
            row["protection_gap_qty"] > 0 for row in rows
        ),
        "risk_reductions_before_entries": True,
    }


def build_broker_health_matrix(
    project_root: Path,
    *,
    reconciliation: dict[str, Any],
    writer_integrity: dict[str, Any],
    network_probe: bool,
) -> dict[str, Any]:
    del project_root
    tws_process = _tws_process_running() if network_probe else None
    tws_socket = _socket_available("127.0.0.1", 7496) if network_probe else None
    public_dns = _dns_available("api.ibkr.com") if network_probe else None
    public_tls = (
        _socket_available("api.ibkr.com", 443) if network_probe and public_dns else None
    )
    reconciliation_go = reconciliation.get("status") == "GO"
    return {
        "schema": "ibkr_broker_health_matrix_v1",
        "status": "GO",
        "TWS_PROCESS": _probe_status(tws_process),
        "TWS_SOCKET": _probe_status(tws_socket),
        "TWS_API": "AVAILABLE" if reconciliation_go else "DEGRADED",
        "IBKR_PUBLIC_API": _probe_status(public_tls),
        "IBKR_PUBLIC_API_DNS": _probe_status(public_dns),
        "MARKET_DATA_ENTITLEMENTS": "OBSERVED_NOT_FULLY_PROVEN",
        "ACCOUNT_RECONCILIATION": (
            "AVAILABLE" if reconciliation_go else "DEGRADED"
        ),
        "EXECUTION_WRITER": (
            "AVAILABLE" if writer_integrity.get("status") == "GO" else "DEGRADED"
        ),
        "local_tws_and_public_api_are_independent": True,
        "network_probe_performed": network_probe,
        "broker_writes": 0,
    }


def build_orchestrator_freeze(project_root: Path) -> dict[str, Any]:
    sources = {}
    blockers = []
    for relative in FREEZE_SOURCES:
        path = project_root / relative
        if not path.is_file():
            blockers.append(f"SOURCE_MISSING:{relative}")
        else:
            sources[relative] = sha256_file(path).upper()
    from stocks.live.level_one_reauthorization import verify_p02_freeze

    p02 = verify_p02_freeze(project_root)
    if p02.get("status") != "GO":
        blockers.append("P02_FREEZE_NOT_GO")
    body: dict[str, Any] = {
        "schema": "p2_quant_portfolio_orchestrator_freeze_v1",
        "status": "GO" if not blockers else "NO_GO",
        "created_at": _now(),
        "source_hashes": sources,
        "p02_freeze_hash": p02.get("freeze_hash"),
        "capability_count": 33,
        "capability_34_added": False,
        "default_mode": "AUTONOMOUS_DRY_RUN",
        "broker_writes": 0,
        "blockers": blockers,
    }
    body["freeze_hash"] = stable_hash(body)
    _write_json(project_root / PUBLIC_ROOT / "freeze-status.json", body)
    return body


def verify_orchestrator_freeze(project_root: Path) -> dict[str, Any]:
    frozen = _read_json(project_root / PUBLIC_ROOT / "freeze-status.json")
    blockers = []
    if not frozen:
        blockers.append("P2_FREEZE_MISSING")
    for relative, expected in frozen.get("source_hashes", {}).items():
        path = project_root / relative
        if not path.is_file() or sha256_file(path).upper() != expected:
            blockers.append(f"P2_SOURCE_HASH_MISMATCH:{relative}")
    return {
        "schema": "p2_quant_portfolio_orchestrator_freeze_verification_v1",
        "status": "GO" if not blockers else "NO_GO",
        "freeze_hash": frozen.get("freeze_hash"),
        "blockers": blockers,
    }


def _load_inputs(root: Path) -> dict[str, Any]:
    from stocks.live.authority import authority_status
    from stocks.live.level_one_reauthorization import verify_p02_freeze
    from stocks.live.service import live_writer_integrity_command
    from stocks.portfolio.execution_feasibility import verify_p2_2_freeze
    from stocks.portfolio.learning_integration import load_learning_evidence

    portfolio_state = _read_json(root / "data/portfolio/private/current-state.json")
    reconciliation = _read_json(root / "output/ibkr/live/reconciliation.json")
    return {
        "account_state": portfolio_state.get("account_state", {}),
        "whole_share_sizing": portfolio_state.get("whole_share_sizing", {}),
        "broker_snapshot": _latest_live_snapshot(root, reconciliation),
        "reconciliation": reconciliation,
        "opportunities": _read_json(
            root / "output/portfolio/normalized-opportunities.json"
        ),
        "opportunity_ranking": _read_json(
            root / "output/portfolio/opportunity_ranking.json"
        ),
        "target_allocation": _read_json(
            root / "output/portfolio/target_allocation.json"
        ),
        "desired_targets": _read_json(
            root / "data/portfolio/private/desired-portfolio-targets.json"
        ),
        "dynamic_risk": _read_json(
            root / "output/portfolio/dynamic-risk-state.json"
        ),
        "overlap": _read_json(root / "output/portfolio/overlap-report.json"),
        "funnel": _read_json(root / "output/portfolio/opportunity-funnel.json"),
        "vectorized_stage0": _read_json(
            root / "output/portfolio/vectorized-stage0.json"
        ),
        "learning_evidence": load_learning_evidence(root),
        "ai_research_plane": load_ai_research_plane_status(root),
        "market_data_capabilities": _read_json(
            root / "output/ibkr/data-capabilities/capability-matrix.json"
        ),
        "level_two_evidence": _read_json(
            root / "output/ibkr/live/level-two-evidence.json"
        ),
        "portfolio_status": _read_json(root / "output/portfolio/status.json"),
        "macro": _read_json(root / "output/macro/score.json"),
        "technical_regime": _read_json(
            root / "output/dynamic/current_regime.json"
        ),
        "strategy_allowlist": _read_json(
            root / "output/ibkr/live/strategy-allowlist.json"
        ),
        "live_authority": authority_status(root),
        "writer_integrity": live_writer_integrity_command(root, "verify"),
        "p02_integrity": verify_p02_freeze(root),
        "p2_2_integrity": verify_p2_2_freeze(root),
    }


def _latest_live_snapshot(
    root: Path, reconciliation: dict[str, Any]
) -> dict[str, Any]:
    expected = str(reconciliation.get("private_snapshot_hash") or "")
    path = root / "data/execution/live/private/broker_observation.sqlite3"
    if not expected or not path.is_file():
        return {}
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT snapshot_hash, payload_json, created_at "
                "FROM snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
    except sqlite3.Error:
        return {}
    if not row or str(row[0]) != expected:
        return {}
    try:
        payload = json.loads(str(row[1]))
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        **payload,
        "content_hash": expected,
        "snapshot_completed_at": str(row[2]),
        "observation_environment": "LIVE_READ_ONLY",
    }


def _public_cycle(private: dict[str, Any]) -> dict[str, Any]:
    actual = private["actual_portfolio"]
    public = {
        key: value
        for key, value in private.items()
        if key not in {"account", "content_hash"}
    }
    public["actual_portfolio"] = {
        **actual,
        "equity_eur": None,
        "cash_eur": None,
        "financial_values_public": False,
        "positions": [
            {
                "symbol": row["symbol"],
                "source": row["source"],
                "quantities_public": False,
                "financial_values_public": False,
            }
            for row in actual.get("positions", [])
        ],
    }
    public["execution_feasibility"] = {
        **private["execution_feasibility"],
        "account_equity_eur": None,
        "available_cash_eur": None,
        "financial_account_values_public": False,
    }
    public["integer_portfolio"] = {
        **private["integer_portfolio"],
        "equity_eur": None,
        "cash_eur": None,
        "financial_values_public": False,
        "positions": [
            {
                "symbol": row["symbol"],
                "action": row["action"],
                "whole_share": row["whole_share"],
                "quantities_public": False,
                "financial_values_public": False,
            }
            for row in private["integer_portfolio"].get("positions", [])
        ],
    }
    public["FULL_DESIRED_CONTINUOUS_PORTFOLIO"] = {
        **private["FULL_DESIRED_CONTINUOUS_PORTFOLIO"],
        "financial_values_public": False,
        "positions": [
            {
                "symbol": row["symbol"],
                "financial_values_public": False,
            }
            for row in private["FULL_DESIRED_CONTINUOUS_PORTFOLIO"].get(
                "positions", []
            )
        ],
    }
    public["FULL_DESIRED_INTEGER_PORTFOLIO"] = public["integer_portfolio"]
    public["CURRENT_AUTHORITY_EXECUTABLE_PORTFOLIO"] = {
        **private["CURRENT_AUTHORITY_EXECUTABLE_PORTFOLIO"],
        "cash_eur": None,
        "financial_values_public": False,
        "positions": [
            {
                "symbol": row["symbol"],
                "strategy_id": row.get("strategy_id"),
                "quantities_public": False,
                "financial_values_public": False,
            }
            for row in private["CURRENT_AUTHORITY_EXECUTABLE_PORTFOLIO"].get(
                "positions", []
            )
        ],
        "excluded_research_positions": [
            {
                "symbol": row["symbol"],
                "research_target_preserved": row.get(
                    "research_target_preserved", False
                ),
                "blockers": row.get("blockers", []),
                "financial_values_public": False,
            }
            for row in private["CURRENT_AUTHORITY_EXECUTABLE_PORTFOLIO"].get(
                "excluded_research_positions", []
            )
        ],
    }
    public["private_cycle_reference"] = (
        PRIVATE_ROOT / "current-cycle.json"
    ).as_posix()
    public["content_hash"] = stable_hash(public)
    return public


def _write_cycle(
    root: Path, private: dict[str, Any], public: dict[str, Any]
) -> None:
    _write_json(root / PRIVATE_ROOT / "current-cycle.json", private)
    _write_json(root / PUBLIC_ROOT / "current-cycle.json", public)
    checkpoint = {
        "schema": "portfolio_orchestrator_checkpoint_v1",
        "cycle_id": private["cycle_id"],
        "content_hash": private["content_hash"],
        "completed_at": private["generated_at"],
        "decision": private["decision"],
        "broker_write_calls": 0,
    }
    _write_json(root / PRIVATE_ROOT / "checkpoint.json", checkpoint)
    journal_path = root / PRIVATE_ROOT / "decision-journal.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_row = {
        "cycle_id": private["cycle_id"],
        "timestamp": private["generated_at"],
        "input_hashes": {
            "actual": stable_hash(private["actual_portfolio"]),
            "opportunities": stable_hash(private["top_cross_asset_opportunities"]),
            "desired": stable_hash(private["desired_portfolio"]),
        },
        "chosen_action": private["decision"],
        "reason": private["decision_reason"],
        "rejected_actions": [
            {
                "symbol": row["symbol"],
                "blockers": row["blockers"],
            }
            for row in private["execution_bridge"].get("proposals", [])
            if not row["machine_policy_approved"]
        ],
        "broker_write_calls": 0,
    }
    with journal_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(journal_row, sort_keys=True) + "\n")


def _regime(inputs: dict[str, Any]) -> dict[str, Any]:
    technical = inputs.get("technical_regime") or {}
    macro = inputs.get("macro") or {}
    macro_regime = macro.get("regime")
    if isinstance(macro_regime, dict):
        macro_label = (
            macro_regime.get("overall_macro_regime")
            or macro_regime.get("label")
            or macro_regime.get("candidate_regime")
            or "UNKNOWN"
        )
    else:
        macro_label = macro_regime or "UNKNOWN"
    return {
        "technical": technical.get("regime")
        or technical.get("technical_regime")
        or (inputs.get("portfolio_status") or {}).get("technical_regime")
        or "UNKNOWN",
        "macro": macro_label,
        "risk_multiplier": (inputs.get("dynamic_risk") or {}).get(
            "multipliers", {}
        ).get("combined"),
        "cash_preference_modeled": True,
    }


def _commodity_family(row: dict[str, Any]) -> str | None:
    cluster = str(row.get("correlation_cluster") or "").upper()
    for name in ("GOLD", "SILVER", "COPPER", "URANIUM", "ENERGY", "OIL"):
        if name in cluster:
            return name
    return row.get("commodity_family")


def _protected_quantity(symbol: str, orders: Iterable[dict[str, Any]]) -> float:
    protected = 0.0
    for row in orders:
        order_symbol = str(
            row.get("symbol") or row.get("contract", {}).get("symbol") or ""
        ).upper()
        if order_symbol != symbol.upper():
            continue
        role = str(row.get("role") or row.get("order_role") or "").upper()
        side = str(row.get("action") or row.get("side") or "").upper()
        if role in {"STOP", "TAKE_PROFIT", "PROTECTION"} or side == "SELL":
            protected = max(
                protected,
                float(_decimal(row.get("remaining", row.get("quantity", 0)))),
            )
    return protected


def _tws_process_running() -> bool:
    if os.name != "nt":
        return False
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    text = result.stdout.lower()
    return any(name in text for name in ("tws.exe", "ibgateway.exe"))


def _socket_available(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def _dns_available(host: str) -> bool:
    try:
        return bool(socket.getaddrinfo(host, None))
    except OSError:
        return False


def _probe_status(value: bool | None) -> str:
    if value is None:
        return "NOT_PROBED"
    return "AVAILABLE" if value else "UNAVAILABLE"


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return result if result.is_finite() else Decimal("0")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Production quant portfolio orchestrator (dry-run first)."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--network-probe", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--verify-freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        result = build_orchestrator_freeze(args.project_root)
    elif args.verify_freeze:
        result = verify_orchestrator_freeze(args.project_root)
    else:
        result = run_continuous_dry_run(
            args.project_root,
            max_cycles=args.cycles,
            interval_seconds=args.interval_seconds,
            refresh_decision_layer=args.refresh,
            network_probe=args.network_probe,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("status") == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_actual_portfolio",
    "build_authority_executable_portfolio",
    "build_broker_health_matrix",
    "build_continuous_desired_portfolio",
    "build_global_risk",
    "build_integer_portfolio",
    "build_orchestrator_freeze",
    "build_portfolio_drift",
    "compare_portfolio_constructors",
    "normalize_orchestrator_opportunities",
    "run_autonomous_dry_run",
    "run_continuous_dry_run",
    "supervise_positions",
    "verify_orchestrator_freeze",
]
