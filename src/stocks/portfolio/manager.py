from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.analysis.confluence import evaluate_multilayer_confluence
from stocks.capital.canary import evaluate_whole_share_canary
from stocks.costs import estimate_transaction_cost, load_shared_cost_model
from stocks.dynamic.strategy_allocation import infer_strategy_family
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.known_fill import (
    load_latest_phase8_private_snapshot,
)
from stocks.ibkr.reconciliation.account_state import (
    derive_economic_account_state,
)
from stocks.market import load_market_context_map
from stocks.portfolio.dynamic_risk import build_dynamic_risk_state
from stocks.portfolio.coverage import (
    build_coverage_waterfall,
    normalize_asset_class,
)
from stocks.portfolio.attribution import publish_performance_attribution
from stocks.portfolio.intelligence import build_cross_asset_intelligence
from stocks.portfolio.etf_holdings import load_etf_holdings
from stocks.portfolio.monitoring import publish_monitoring_architecture
from stocks.portfolio.opportunities import (
    annotate_opportunity_economics,
    economic_cluster,
    normalize_cross_asset_opportunities,
)
from stocks.portfolio.overlap import build_overlap_report
from stocks.portfolio.targets import build_desired_portfolio_targets
from stocks.portfolio.position_management import evaluate_long_position
from stocks.portfolio.real_assets import (
    active_swing_timeframe_context,
    opportunity_class,
    real_asset_context,
)
from stocks.portfolio.swing_status import publish_active_swing_product_status
from stocks.portfolio.swing import resolve_signal_swing_contract
from stocks.rl.portfolio import build_shadow_portfolio_rotation
from stocks.signals.market_reference import (
    apply_market_reference,
    latest_market_reference,
)
from stocks.signals.freshness import evaluate_signal_freshness
from stocks.signals.price_basis import normalize_research_signal_price_basis
from stocks.signals.timeframe_contracts import (
    declared_research_signal_timeframe_contract,
)
from stocks.screener.config import ScreenerConfig
from stocks.research.stage0 import run_vectorized_stage0
from stocks.universe import (
    broad_asset_metadata,
    commodity_producer_metadata,
)


POLICY_PATH = Path("config/portfolio/active_manager_v1.json")
PUBLIC_ROOT = Path("output/portfolio")
PRIVATE_ROOT = Path("data/portfolio/private")
RESEARCH_ACCOUNT_SNAPSHOT_MAX_AGE = timedelta(hours=36)
RISK_INCREASING_ACTIONS = {"ADD", "REPLACE"}
EXECUTION_BLOCKERS = {
    "CONTRACT_IDENTITY_REQUIRED",
    "CURRENT_DATA_STALE",
    "NEGATIVE_HIGH_IMPACT_NEWS",
    "SHARIAH_ATTESTATION_REQUIRED",
}
POSITIVE_ACTIONS = {"BUY", "STRONG_BUY", "WATCHLIST"}


def active_portfolio_command(
    project_root: Path,
    command: str,
) -> dict[str, Any]:
    report = build_active_portfolio_report(project_root)
    sections = {
        "status": report["engine_status"],
        "plan": report,
        "opportunities": report["opportunities"],
        "actions": report["position_actions"],
        "sizing-audit": report["sizing_audit"],
        "lifecycle-audit": report["lifecycle_audit"],
        "risk": report["risk"],
        "dynamic-risk": report["dynamic_risk"],
        "position-management": report["position_management"],
        "confluence": report["confluence"],
        "exposures": report["exposures"],
        "rebalance-preview": report["rebalance"],
        "decisions": report["capital_decisions"],
        "state": report["portfolio_state"],
        "funnel": report["opportunity_funnel"],
        "coverage": report["coverage_waterfall"],
        "stage0": report["vectorized_stage0"],
        "intelligence": report["cross_asset_intelligence"],
        "attribution": report["performance_attribution"],
        "normalized-opportunities": report["normalized_opportunities"],
        "overlap": report["overlap"],
        "targets": report["desired_targets"],
        "swing-product": report["active_swing_product"],
        "rl-rotation": report["rl_portfolio_rotation"],
    }
    if command not in sections:
        raise ValueError(f"UNKNOWN_ACTIVE_PORTFOLIO_COMMAND:{command}")
    return sections[command]


def build_active_portfolio_report(project_root: Path) -> dict[str, Any]:
    policy = _read_json(project_root / POLICY_PATH)
    broad_metadata = broad_asset_metadata(project_root)
    configured_metadata = policy.get("asset_metadata", {})
    policy["asset_metadata"] = {
        symbol: {
            **broad_metadata.get(symbol, {}),
            **configured_metadata.get(symbol, {}),
        }
        for symbol in set(broad_metadata) | set(configured_metadata)
    }
    signals = _load_signals(project_root)
    contracts = _contract_map(project_root, policy["asset_metadata"])
    fundamentals = _fundamental_map(project_root)
    family_map = _strategy_family_map(project_root)
    macro = _read_json(project_root / "output/macro/score.json")
    technical_regime = _read_json(project_root / "output/dynamic/current_regime.json")
    news = _read_json(project_root / "output/notifications/market-intelligence-digest.json")
    news_events = _read_json(project_root / "output/news/intelligence/portfolio-impact.json")
    news_event_context = _news_event_overlay_map(news_events, policy=policy)
    overrides = _dynamic_overrides(project_root)
    strategy_weights = _dynamic_strategy_weights(project_root)
    market_context = load_market_context_map(project_root)
    asset_context = _asset_context_map(project_root)
    current_positions, private_snapshot_status = _current_positions(project_root)
    private_account = _private_account_state(project_root)
    daily_target = _read_json(project_root / "output/capital/daily_profit_target.json")
    autopilot = _read_json(project_root / "output/research/autopilot/status.json")
    contract_symbols = set(contracts)
    ranked = rank_opportunities(
        signals,
        policy=policy,
        contracts=contracts,
        fundamentals=fundamentals,
        family_map=family_map,
        macro=macro,
        technical_regime=technical_regime,
        news=news,
        dynamic_overrides=overrides,
        strategy_weights=strategy_weights,
        market_context=market_context,
        asset_context=asset_context,
        news_event_context=news_event_context,
    )
    ranked = annotate_opportunity_economics(project_root, ranked)
    opportunity_funnel = build_opportunity_funnel(
        project_root,
        signals=signals,
        ranked=ranked,
        policy=policy,
    )
    confluence_audit = _confluence_audit(ranked)
    news_overlay_audit = _news_overlay_audit(ranked, news_events)
    equity_history, daily_pnl = _private_dynamic_risk_inputs(project_root)
    dynamic_risk = build_dynamic_risk_state(
        equity_eur=(
            _decimal(private_account["net_liquidation_eur"])
            if private_account["status"] == "GO"
            else None
        ),
        equity_history=equity_history,
        daily_pnl=daily_pnl,
        candidates=ranked,
        policy=policy,
        technical_regime=technical_regime,
        macro=macro,
        capital_level=_capital_level(project_root),
    )
    correlation = _correlation_matrix(project_root, ranked, policy)
    whole_share_preflight = _whole_share_candidate_preflight(
        project_root,
        current_positions=current_positions,
        ranked=ranked,
        account=private_account,
        policy=policy,
        dynamic_risk=dynamic_risk,
    )
    whole_share_feasible_tickers = (
        set(whole_share_preflight["feasible_tickers"])
        if whole_share_preflight["selection_filter_applied"]
        else None
    )
    allocation = allocate_research_portfolio(
        ranked,
        policy=policy,
        technical_regime=technical_regime,
        macro=macro,
        correlation=correlation,
        daily_target=daily_target,
        dynamic_risk=dynamic_risk,
        whole_share_feasible_tickers=whole_share_feasible_tickers,
    )
    rl_portfolio_rotation = build_shadow_portfolio_rotation(
        project_root,
        ranked=ranked,
        allocation=allocation,
        whole_share_preflight=whole_share_preflight,
    )
    opportunity_funnel["whole_share_candidate_preflight"] = {
        key: value for key, value in whole_share_preflight.items() if key != "feasible_tickers"
    }
    opportunity_funnel["allocator_selected_count"] = len(allocation.get("allocations", []))
    opportunity_funnel["content_hash"] = stable_hash(
        {key: value for key, value in opportunity_funnel.items() if key != "content_hash"}
    )
    position_management = _build_position_management(
        project_root,
        positions=current_positions,
        ranked=ranked,
        technical_regime=technical_regime,
    )
    actions = position_actions(
        current_positions,
        ranked,
        allocation,
        policy=policy,
        dynamic_overrides=overrides,
        daily_target=daily_target,
        management_states={row["ticker"]: row for row in position_management["private_decisions"]},
    )
    private_sizing = build_private_sizing(
        project_root,
        current_positions=current_positions,
        ranked=ranked,
        allocation=allocation,
        account=private_account,
        policy=policy,
        dynamic_risk=dynamic_risk,
    )
    private_sizing["candidate_preflight"] = whole_share_preflight
    stage0 = run_vectorized_stage0(project_root)
    cross_asset_intelligence = build_cross_asset_intelligence(
        project_root,
        stage0=stage0,
        macro=macro,
    )
    performance_attribution = publish_performance_attribution(project_root)
    normalized_opportunities = normalize_cross_asset_opportunities(
        project_root,
        ranked=ranked,
        stage0_report=stage0,
        sizing_rows=private_sizing.get("positions", []),
    )
    overlap_report = build_overlap_report(
        normalized_opportunities.get("combined_ranking", []),
        correlation,
        threshold=float(policy["portfolio"]["correlation_threshold"]),
        etf_holdings=load_etf_holdings(project_root),
    )
    coverage_waterfall = build_coverage_waterfall(
        project_root,
        ranked=ranked,
        signals=signals,
        portfolio_symbols=(
            row["symbol"]
            for row in opportunity_funnel.get("watchlist_candidates", [])
            if row.get("candidate_stage") in {"PORTFOLIO_CANDIDATE", "EXECUTION_CANDIDATE"}
        ),
        whole_share_symbols=whole_share_preflight.get("feasible_tickers", []),
    )
    desired_targets = build_desired_portfolio_targets(
        project_root,
        current_positions=current_positions,
        sizing_rows=private_sizing.get("positions", []),
        normalized_opportunities=normalized_opportunities,
        policy=policy,
    )
    actions = _apply_portfolio_risk_overlay(
        actions,
        private_sizing=private_sizing,
        policy=policy,
    )
    lifecycle = _record_action_cycle(
        project_root,
        actions=actions,
        private_sizing=private_sizing,
        policy=policy,
    )
    public_actions = [_public_action(row) for row in actions]
    approved_exposure = 0.0
    public_allocations = [
        {
            "ticker": row["ticker"],
            "asset_type": row["asset_type"],
            "sleeve": row["sleeve"],
            "sector": row["sector"],
            "region": row["region"],
            "opportunity_score": row["opportunity_score"],
            "research_target_weight": row["target_weight"],
            "approved_target_weight": 0.0,
            "correlation_penalty": row["correlation_penalty"],
            "maximum_correlation_to_selected": row["maximum_correlation_to_selected"],
            "stop_risk_pct": row["stop_risk_pct"],
            "entry_method": row["entry_method"],
            "stop_method": row["stop_method"],
            "exit_policy": row["exit_policy"],
            "expected_holding_period": row["expected_holding_period"],
            "execution_status": "OBSERVE_ONLY_AUTHORITY_NONE",
        }
        for row in allocation["allocations"]
    ]
    public_opportunities = [
        _public_opportunity(row)
        for row in ranked[: int(policy["ranking"]["maximum_published_opportunities"])]
    ]
    capital_decisions = build_capital_decisions(
        current_positions,
        ranked,
        allocation,
        actions,
        policy=policy,
        private_sizing=private_sizing,
    )
    portfolio_state = {
        "schema": "proactive_portfolio_state_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "base_currency": str(policy.get("base_currency") or "EUR"),
        "current_position_count": len(current_positions),
        "observed_current_gross_exposure": private_sizing["current_gross_exposure_pct"],
        "research_target_exposure": allocation["research_target_exposure"],
        "research_cash_target_weight": allocation["cash_weight"],
        "approved_target_exposure": 0.0,
        "approved_cash_target_weight": 1.0,
        "portfolio_heat": allocation["portfolio_heat"],
        "maximum_portfolio_heat": allocation["maximum_portfolio_heat"],
        "dynamic_risk_multiplier": dynamic_risk["multipliers"]["combined"],
        "technical_regime": technical_regime.get("regime", "UNAVAILABLE"),
        "macro_status": macro.get("status", "UNAVAILABLE"),
        "news_status": news.get("status", "UNAVAILABLE"),
        "news_event_intelligence_status": news_events.get("status", "UNAVAILABLE"),
        "fundamental_symbol_count": len(fundamentals),
        "candidate_count": len(ranked),
        "watchlist_candidate_count": opportunity_funnel["watchlist_candidate_count"],
        "portfolio_candidate_count": opportunity_funnel["portfolio_candidate_count"],
        "execution_candidate_count": opportunity_funnel["execution_candidate_count"],
        "allocatable_candidate_count": sum(not _research_blockers(row) for row in ranked),
        "formal_action_counts": capital_decisions["action_counts"],
        "financial_values_public": False,
        "private_financial_state_reference": ("data/portfolio/private/current-state.json"),
        "automatic_execution_allowed": False,
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }
    risk = {
        "schema": "active_portfolio_risk_v1",
        "status": allocation["status"],
        "research_portfolio_heat": allocation["portfolio_heat"],
        "observed_current_portfolio_heat": private_sizing["current_portfolio_heat"],
        "maximum_portfolio_heat": allocation["maximum_portfolio_heat"],
        "research_target_exposure": allocation["research_target_exposure"],
        "whole_share_shadow_target_exposure": private_sizing["target_gross_exposure_pct"],
        "observed_current_gross_exposure": private_sizing["current_gross_exposure_pct"],
        "approved_target_exposure": approved_exposure,
        "estimated_annualized_volatility": allocation["estimated_annualized_volatility"],
        "maximum_pairwise_correlation": allocation["maximum_pairwise_correlation"],
        "effective_position_count": allocation["effective_position_count"],
        "portfolio_heat_gate": allocation["portfolio_heat_gate"],
        "correlation_gate": allocation["correlation_gate"],
        "daily_profit_target_throttle": _target_throttle(daily_target),
        "dynamic_risk_multiplier": dynamic_risk["multipliers"]["combined"],
        "dynamic_maximum_positions": dynamic_risk["dynamic_research_maximum_positions"],
        "operational_maximum_positions": dynamic_risk["operational_maximum_positions"],
        "authority": "NONE",
    }
    exposures = {
        "schema": "active_portfolio_exposures_v1",
        "status": "GO",
        "research_gross_exposure": allocation["research_target_exposure"],
        "whole_share_shadow_gross_exposure": private_sizing["target_gross_exposure_pct"],
        "observed_current_gross_exposure": private_sizing["current_gross_exposure_pct"],
        "approved_gross_exposure": 0.0,
        "research_cash_weight": allocation["cash_weight"],
        "approved_cash_weight": 1.0,
        "sleeve_weights": allocation["sleeve_weights"],
        "sector_weights": allocation["sector_weights"],
        "region_weights": allocation["region_weights"],
        "correlation_cluster_weights": allocation["correlation_cluster_weights"],
        "asset_class_weights": allocation["asset_class_weights"],
        "margin_enabled": False,
        "leverage_enabled": False,
        "shorting_enabled": False,
        "authority": "NONE",
    }
    rebalance = {
        "schema": "active_portfolio_rebalance_preview_v1",
        "status": "ADVISORY_ONLY",
        "current_position_count": len(current_positions),
        "recommended_action_count": len(public_actions),
        "actions": public_actions,
        "new_candidate_count": len(public_allocations),
        "target_allocations": public_allocations,
        "private_whole_share_plan_status": private_sizing["status"],
        "whole_share_target_position_count": private_sizing["target_position_count"],
        "netted_security_count": private_sizing["netted_security_count"],
        "estimated_turnover_pct": private_sizing["turnover_pct"],
        "turnover_gate": private_sizing["turnover_gate"],
        "automatic_submission": False,
        "orders_generated": 0,
        "broker_calls": 0,
        "execution_authority": "NONE",
    }
    status = {
        "schema": "active_multi_asset_portfolio_status_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "policy_schema": policy["schema"],
        "signal_count": len(signals),
        "opportunity_count": len(ranked),
        "watchlist_candidate_count": opportunity_funnel["watchlist_candidate_count"],
        "portfolio_candidate_count": opportunity_funnel["portfolio_candidate_count"],
        "execution_candidate_count": opportunity_funnel["execution_candidate_count"],
        "allocatable_opportunity_count": sum(not _research_blockers(row) for row in ranked),
        "execution_ready_opportunity_count": sum(not row["deployment_blockers"] for row in ranked),
        "robustness_survivor_opportunity_count": sum(
            "ROBUSTNESS_SURVIVOR" in row["evidence_tiers"] for row in ranked
        ),
        "strategy_family_count": len(
            {family for row in ranked for family in row["strategy_families"]}
        ),
        "observed_timeframes": sorted(
            {timeframe for row in ranked for timeframe in row["timeframes"]}
        ),
        "scanned_signal_timeframes": sorted(
            {str(row.get("timeframe") or "unknown") for row in signals}
        ),
        "positive_signal_timeframes": sorted(
            {
                str(row.get("timeframe") or "unknown")
                for row in signals
                if str(row.get("action", "")).upper() in POSITIVE_ACTIONS
            }
        ),
        "price_invalidated_signal_timeframes": sorted(
            {
                str(row.get("timeframe") or "unknown")
                for row in signals
                if str(row.get("price_validity_status", "")).startswith("CURRENT_")
                and str(row.get("action", "")).upper() == "AVOID"
            }
        ),
        "signal_freshness_policy": ("EXCHANGE_SESSION_AWARE_SIGNAL_EXPIRY_V1"),
        "contract_resolved_symbol_count": len(contract_symbols),
        "current_position_count": len(current_positions),
        "private_position_snapshot_status": private_snapshot_status,
        "private_account_state_status": private_account["status"],
        "private_whole_share_sizing_status": private_sizing["status"],
        "dynamic_risk_status": dynamic_risk["status"],
        "equity_band": dynamic_risk["equity_band"],
        "dynamic_research_maximum_positions": dynamic_risk["dynamic_research_maximum_positions"],
        "operational_maximum_positions": dynamic_risk["operational_maximum_positions"],
        "security_netting_status": private_sizing["security_netting_status"],
        "append_only_action_ledger_status": lifecycle["status"],
        "position_management_status": position_management["public_audit"]["status"],
        "macro_status": macro.get("status", "UNAVAILABLE"),
        "news_status": news.get("status", "UNAVAILABLE"),
        "news_event_intelligence_status": news_events.get("status", "UNAVAILABLE"),
        "news_adjusted_opportunity_count": news_overlay_audit["adjusted_opportunity_count"],
        "news_hard_risk_review_count": news_overlay_audit["hard_risk_review_count"],
        "fundamental_symbol_count": len(fundamentals),
        "multilayer_confluence_status": confluence_audit["status"],
        "multilayer_confluence_counts": confluence_audit["status_counts"],
        "strategy_generator_status": autopilot.get("status", "UNAVAILABLE"),
        "registered_strategy_dna_count": int(autopilot.get("bulk_strategy_catalog_count", 0) or 0),
        "registered_research_trial_count": int(autopilot.get("bulk_trial_count", 0) or 0),
        "continuous_strategy_research": (autopilot.get("status") == "GO"),
        "automatic_live_strategy_promotion": False,
        "p1_coverage_status": coverage_waterfall.get("status", "NO_GO"),
        "p1_stage0_status": stage0.get("status", "NO_GO"),
        "normalized_cross_asset_opportunity_count": (
            normalized_opportunities.get("opportunity_count", 0)
        ),
        "universal_overlap_gate_status": overlap_report["status"],
        "broker_observation_authority": "READ_ONLY",
        "signal_authority": "RESEARCH_AND_SHADOW",
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_order_submission": False,
        "orders_generated": 0,
        "broker_write_calls": 0,
        "live_trading_started": False,
    }
    active_swing_product = publish_active_swing_product_status(
        project_root,
        signals=signals,
        ranked=ranked,
        normalized_opportunities=normalized_opportunities,
        opportunity_funnel=opportunity_funnel,
        current_positions=current_positions,
    )
    report = {
        "schema": "active_multi_asset_portfolio_plan_v1",
        "status": "GO",
        "engine_status": status,
        "opportunities": {
            "schema": "active_opportunity_ranking_v1",
            "status": "GO",
            "count": len(public_opportunities),
            "opportunities": public_opportunities,
            "authority": "NONE",
        },
        "current_allocation": {
            "schema": "current_allocation_v2",
            "status": private_snapshot_status,
            "position_count": len(current_positions),
            "positions": [
                {
                    "ticker": row["symbol"],
                    "position_identity": stable_hash(
                        {
                            "con_id": row["con_id"],
                            "symbol": row["symbol"],
                        }
                    )[:24],
                    "security_type": row["security_type"],
                    "currency": row["currency"],
                }
                for row in current_positions
            ],
            "quantities_public": False,
            "financial_values_public": False,
            "source": "PHASE8_PRIVATE_READ_ONLY_BROKER_OBSERVATION",
        },
        "target_allocation": {
            "schema": "target_allocation_v2",
            "status": allocation["status"],
            "research_target_exposure": allocation["research_target_exposure"],
            "approved_target_exposure": 0.0,
            "research_cash_target_weight": allocation["cash_weight"],
            "cash_target_weight": 1.0,
            "allocations": public_allocations,
            "execution_authority": "NONE",
        },
        "position_actions": {
            "schema": "active_position_actions_v1",
            "status": "GO",
            "actions": public_actions,
            "automatic_actions": 0,
            "execution_authority": "NONE",
        },
        "capital_decisions": capital_decisions,
        "portfolio_state": portfolio_state,
        "opportunity_funnel": opportunity_funnel,
        "coverage_waterfall": coverage_waterfall,
        "vectorized_stage0": stage0,
        "cross_asset_intelligence": cross_asset_intelligence,
        "performance_attribution": performance_attribution,
        "normalized_opportunities": normalized_opportunities,
        "overlap": overlap_report,
        "desired_targets": desired_targets,
        "active_swing_product": active_swing_product,
        "rl_portfolio_rotation": rl_portfolio_rotation,
        "position_management": position_management["public_audit"],
        "confluence": confluence_audit,
        "news_overlay": news_overlay_audit,
        "sizing_audit": {
            "schema": "private_whole_share_sizing_public_audit_v1",
            "status": private_sizing["status"],
            "account_equity_observed": private_account["equity_observed"],
            "available_funds_observed": private_account["available_funds_observed"],
            "current_position_count": len(current_positions),
            "target_position_count": private_sizing["target_position_count"],
            "whole_share_feasible_candidate_count": private_sizing[
                "whole_share_feasible_candidate_count"
            ],
            "whole_share_infeasible_candidate_count": private_sizing[
                "whole_share_infeasible_candidate_count"
            ],
            "sizing_basis": private_sizing.get("sizing_basis"),
            "small_account_whole_share_mode": private_sizing.get(
                "small_account_whole_share_mode", False
            ),
            "netted_security_count": private_sizing["netted_security_count"],
            "whole_share_violation_count": private_sizing["whole_share_violation_count"],
            "negative_cash_violation_count": private_sizing["negative_cash_violation_count"],
            "current_gross_exposure_pct": private_sizing["current_gross_exposure_pct"],
            "target_gross_exposure_pct": private_sizing["target_gross_exposure_pct"],
            "turnover_pct": private_sizing["turnover_pct"],
            "turnover_gate": private_sizing["turnover_gate"],
            "financial_values_public": False,
            "position_quantities_public": False,
            "execution_authority": "NONE",
        },
        "lifecycle_audit": lifecycle,
        "dynamic_risk": dynamic_risk,
        "risk": risk,
        "exposures": exposures,
        "rebalance": rebalance,
        "authority": {
            "broker_observation_authority": "READ_ONLY",
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
            "automatic_submission": False,
            "orders_generated": 0,
            "broker_write_calls": 0,
        },
    }
    report["content_hash"] = stable_hash(report)
    _publish(
        project_root,
        report,
        actions=actions,
        private_sizing=private_sizing,
        private_account=private_account,
        correlation=correlation,
    )
    from stocks.live.portfolio_targets import (
        publish_controlled_purchase_plan,
    )

    publish_controlled_purchase_plan(project_root)
    publish_monitoring_architecture(project_root)
    return report


def rank_opportunities(
    signals: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
    fundamentals: dict[str, dict[str, Any]],
    family_map: dict[str, str],
    macro: dict[str, Any],
    technical_regime: dict[str, Any],
    news: dict[str, Any],
    dynamic_overrides: dict[str, dict[str, Any]],
    strategy_weights: dict[str, float] | None = None,
    market_context: dict[str, dict[str, Any]] | None = None,
    asset_context: dict[str, dict[str, Any]] | None = None,
    news_event_context: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in signals:
        ticker = str(row.get("ticker") or row.get("asset") or "").upper()
        if ticker and str(row.get("action", "")).upper() in POSITIVE_ACTIONS:
            grouped[ticker].append(row)
    weights = policy["opportunity_weights"]
    penalty_policy = policy["penalties"]
    news_map = _news_map(news)
    news_event_context = news_event_context or {}
    rows = []
    for ticker, items in grouped.items():
        latest_by_strategy: dict[str, dict[str, Any]] = {}
        for item in items:
            strategy_id = str(item.get("strategy_id") or "UNKNOWN")
            previous = latest_by_strategy.get(strategy_id)
            if previous is None or str(item.get("data_timestamp", "")) > str(
                previous.get("data_timestamp", "")
            ):
                latest_by_strategy[strategy_id] = item
        selected = list(latest_by_strategy.values())
        research_eligible_source = any(
            bool(
                item.get(
                    "_research_allocation_eligible_source",
                    item.get("_portfolio_eligible_source", True),
                )
            )
            for item in selected
        )
        deployment_eligible_source = any(
            bool(item.get("_deployment_eligible_source", False)) for item in selected
        )
        evidence_tiers = sorted(
            {
                str(
                    item.get(
                        "_evidence_tier",
                        "FROZEN_SHADOW",
                    )
                )
                for item in selected
            }
        )
        families = sorted(
            {
                family_map.get(
                    str(item.get("strategy_id", "")),
                    _infer_family(item),
                )
                for item in selected
            }
        )
        timeframes = sorted({str(item.get("timeframe") or "unknown") for item in selected})
        family_confidences: dict[str, float] = defaultdict(float)
        weighted_family_confidences: dict[str, tuple[float, float]] = {}
        for item in selected:
            strategy_id = str(item.get("strategy_id", ""))
            family = family_map.get(
                strategy_id,
                _infer_family(item),
            )
            confidence = _bounded(item.get("confidence_score"), default=0.0)
            family_confidences[family] = max(family_confidences[family], confidence)
            allocation_weight = max(
                0.0,
                float((strategy_weights or {}).get(strategy_id, 0.0)),
            )
            previous_weighted = weighted_family_confidences.get(family)
            candidate_weighted = confidence * allocation_weight
            if (
                previous_weighted is None
                or candidate_weighted > previous_weighted[0] * previous_weighted[1]
            ):
                weighted_family_confidences[family] = (
                    confidence,
                    allocation_weight,
                )
        confidences = list(family_confidences.values())
        raw_signal_quality = (
            float(np.mean(sorted(confidences, reverse=True)[:5])) if confidences else 0.0
        )
        allocated = [value for value in weighted_family_confidences.values() if value[1] > 0]
        allocated_weight = sum(value[1] for value in allocated)
        if allocated_weight > 0:
            weighted_signal_quality = (
                sum(confidence * allocation_weight for confidence, allocation_weight in allocated)
                / allocated_weight
            )
            allocation_strength = min(1.0, allocated_weight / 0.20)
            signal_quality = weighted_signal_quality * (0.70 + 0.30 * allocation_strength)
            strategy_weight_status = "DYNAMIC_WEIGHTED"
        else:
            weighted_signal_quality = raw_signal_quality
            signal_quality = raw_signal_quality
            allocation_strength = 0.0
            strategy_weight_status = "UNAVAILABLE_FALLBACK_RAW_SIGNAL_QUALITY"
        family_breadth = min(len(families) / 4.0, 1.0)
        timeframe_context = active_swing_timeframe_context(timeframes)
        timeframe_confirmation = float(timeframe_context["score"])
        reasons = sorted({str(reason) for item in selected for reason in item.get("reasons", [])})
        relative_strength = _reason_score(
            reasons,
            ("RELATIVE", "MOMENTUM", "LEADERSHIP", "ROTATION"),
            fallback=signal_quality * 0.65,
        )
        setup_quality = _reason_score(
            reasons,
            (
                "BREAKOUT",
                "PULLBACK",
                "RECLAIM",
                "CONTRACTION",
                "CONFIRMED",
                "TREND",
                "CHANNEL",
            ),
            fallback=signal_quality * 0.7,
        )
        fundamental = fundamentals.get(ticker, {})
        metadata = _metadata(ticker, fundamental, contracts, policy)
        raw_fundamental_score = _optional_number(fundamental.get("fundamental_score"))
        fundamental_required = bool(
            metadata["asset_type"] == "STOCK"
            and policy.get("multi_layer_confluence", {}).get(
                "fundamentals_required_for_stocks", True
            )
        )
        fundamental_quality = (
            0.55
            if metadata["asset_type"] != "STOCK"
            else 0.35
            if raw_fundamental_score is None
            else _bounded(raw_fundamental_score / 100.0, default=0.35)
        )
        liquidity = _bounded(
            _number(fundamental.get("liquidity_score")) / 100.0,
            default=0.5 if ticker in contracts else 0.25,
        )
        generic_regime_fit = _regime_fit(
            metadata,
            macro=macro,
            technical_regime=technical_regime,
        )
        symbol_asset_context = (asset_context or {}).get(ticker, {})
        macro_layer = _macro_confluence_layer(
            symbol_asset_context,
            generic_regime_fit=generic_regime_fit,
        )
        macro_confidence = float(macro_layer["confidence"])
        regime_fit = generic_regime_fit * (1.0 - 0.5 * macro_confidence) + float(
            macro_layer["score"]
        ) * (0.5 * macro_confidence)
        technical_confluence_score = (
            0.35 * signal_quality
            + 0.20 * timeframe_confirmation
            + 0.20 * relative_strength
            + 0.25 * setup_quality
        )
        confluence = evaluate_multilayer_confluence(
            technical_score=technical_confluence_score,
            fundamental_score=(
                fundamental_quality
                if raw_fundamental_score is not None or not fundamental_required
                else None
            ),
            fundamental_required=fundamental_required,
            macro_score=float(macro_layer["score"]),
            macro_confidence=macro_confidence,
            macro_status=str(macro_layer["status"]),
            config=policy.get("multi_layer_confluence"),
        )
        confluence["macro_source"] = macro_layer["source"]
        confluence["transmission_group"] = symbol_asset_context.get("transmission_group")
        confluence["asset_context_hash"] = (
            stable_hash(symbol_asset_context) if symbol_asset_context else None
        )
        event_risk, negative_event = _event_risk(ticker, news_map, news)
        event_overlay = news_event_context.get(
            ticker,
            {
                "event_count": 0,
                "ranking_adjustment": 0.0,
                "hard_risk_flags": [],
                "standalone_entry_allowed": False,
                "execution_authority": "NONE",
            },
        )
        freshness = all(str(item.get("data_freshness", "FRESH")) != "STALE" for item in selected)
        preferred = max(
            selected,
            key=lambda item: _number(item.get("confidence_score")),
        )
        swing_contract = resolve_signal_swing_contract(
            selected,
            symbol=ticker,
        )
        contract_resolved = ticker in contracts
        signal_currency = str(
            metadata.get("currency") or preferred.get("currency") or "UNKNOWN"
        ).upper()
        contract_currency = str(contracts.get(ticker, {}).get("currency") or "UNKNOWN").upper()
        contract_currency_mismatch = (
            contract_resolved
            and signal_currency not in {"", "UNKNOWN"}
            and contract_currency not in {"", "UNKNOWN"}
            and signal_currency != contract_currency
        )
        symbol_market_context = (market_context or {}).get(ticker, {})
        context_components = symbol_market_context.get("ranking_components", {})
        shariah = _resolved_shariah_status(
            fundamental,
            selected,
        )
        data_quality_penalty = 0.0 if freshness else float(penalty_policy["data_quality"])
        contract_penalty = (
            0.0 if contract_resolved else float(penalty_policy["unresolved_contract"])
        )
        components = {
            "signal_quality": signal_quality,
            "strategy_family_breadth": family_breadth,
            "timeframe_confirmation": timeframe_confirmation,
            "fundamental_quality": fundamental_quality,
            "liquidity": liquidity,
            "regime_fit": regime_fit,
            "relative_strength": relative_strength,
            "setup_quality": setup_quality,
            "orderflow_context": _bounded(
                context_components.get("orderflow_context"),
                default=0.5,
            ),
            "gex_context": _bounded(
                context_components.get("gex_context"),
                default=0.5,
            ),
        }
        class_name = opportunity_class(metadata, families)
        real_asset = real_asset_context(metadata, components)
        gross_score = sum(float(weights[key]) * value for key, value in components.items())
        penalties = {
            "event_risk": event_risk * float(penalty_policy["event_risk"]),
            "data_quality": data_quality_penalty,
            "unresolved_contract": contract_penalty,
        }
        base_score_without_confluence = _bounded(gross_score - sum(penalties.values()))
        confluence_only_score = _bounded(
            base_score_without_confluence * float(confluence["ranking_multiplier"])
        )
        news_score_adjustment = float(event_overlay.get("ranking_adjustment") or 0.0)
        score = _bounded(confluence_only_score + news_score_adjustment)
        management_blockers: list[str] = []
        if not contract_resolved:
            management_blockers.append("CONTRACT_IDENTITY_REQUIRED")
        if contract_currency_mismatch:
            management_blockers.append("SIGNAL_CONTRACT_CURRENCY_MISMATCH")
        if not freshness:
            management_blockers.append("CURRENT_DATA_STALE")
        if negative_event:
            management_blockers.append("NEGATIVE_HIGH_IMPACT_NEWS")
        management_blockers.extend(confluence["allocation_blockers"])
        if event_overlay.get("hard_risk_flags"):
            management_blockers.append("MATERIAL_NEWS_RISK_REVIEW_REQUIRED")
        if metadata["asset_type"] in {
            "STOCK",
            "ETF",
            "BOND_ETF",
            "COMMODITY_ETF",
        } and shariah not in {
            "SHARIAH_ELIGIBLE_PIT",
            "SHARIAH_COMPLIANT",
            "NOT_CONFIGURED",
        }:
            management_blockers.append("SHARIAH_ATTESTATION_REQUIRED")
        product_identity = real_asset.get("product_identity", {})
        management_blockers.extend(str(value) for value in product_identity.get("blockers", []))
        research_blockers = list(management_blockers)
        if not research_eligible_source:
            research_blockers.append("RESEARCH_OBSERVER_NOT_PORTFOLIO_ELIGIBLE")
        override = dynamic_overrides.get(ticker, {})
        if str(override.get("action")) == "AVOID":
            raw_override_blockers = [str(item) for item in override.get("risk_blockers", [])]
            effective_override_blockers = _effective_dynamic_override_blockers(
                raw_override_blockers,
                shariah_status=shariah,
            )
            management_blockers.extend(effective_override_blockers)
            if not raw_override_blockers:
                management_blockers.append("DYNAMIC_CONSENSUS_AVOID")
            research_blockers = [
                *research_blockers,
                *management_blockers,
            ]
        deployment_blockers = list(research_blockers)
        if not deployment_eligible_source:
            deployment_blockers.append("STRATEGY_DEPLOYMENT_EVIDENCE_REQUIRED")
        deployment_blockers.append("EXECUTION_AUTHORITY_NONE")
        management = _management_policy(families)
        stop_risks = [_stop_risk(item) for item in selected if _stop_risk(item) > 0]
        stop_risk_pct = float(np.median(stop_risks)) if stop_risks else 0.1
        stop_risk_pct = min(max(stop_risk_pct, 0.02), 0.2)
        rows.append(
            {
                "ticker": ticker,
                **metadata,
                "opportunity_score": round(score, 6),
                "gross_score": round(gross_score, 6),
                "base_score_without_confluence": round(base_score_without_confluence, 6),
                "confluence_adjustment": round(
                    confluence_only_score - base_score_without_confluence,
                    6,
                ),
                "opportunity_score_before_news": round(confluence_only_score, 6),
                "news_score_adjustment": round(news_score_adjustment, 6),
                "news_event_context": event_overlay,
                "multilayer_confluence": confluence,
                "components": {key: round(value, 6) for key, value in components.items()},
                "penalties": {key: round(value, 6) for key, value in penalties.items()},
                "strategy_count": len(selected),
                "strategy_ids": sorted(latest_by_strategy),
                "strategy_families": families,
                "timeframes": timeframes,
                "opportunity_class": class_name,
                "normalized_asset_class": normalize_asset_class(metadata),
                "correlation_cluster": economic_cluster(ticker, metadata),
                "active_swing_timeframe_context": timeframe_context,
                "active_swing_contract": swing_contract,
                "lifecycle_state": swing_contract["lifecycle_state"],
                "setup_id": swing_contract["setup_id"],
                "real_asset_context": real_asset,
                "strategy_allocation": {
                    "status": strategy_weight_status,
                    "raw_signal_quality": round(raw_signal_quality, 6),
                    "weighted_signal_quality": round(weighted_signal_quality, 6),
                    "participating_weight": round(allocated_weight, 6),
                    "allocation_strength": round(allocation_strength, 6),
                    "weighted_strategy_count": len(allocated),
                },
                "reason_count": len(reasons),
                "reasons": reasons[:12],
                "market_context": symbol_market_context
                or {
                    "status": "NO_CONTEXT_NEUTRAL_FALLBACK",
                    "ranking_components": {
                        "orderflow_context": 0.5,
                        "gex_context": 0.5,
                    },
                    "standalone_entry_authority": False,
                    "execution_authority": "NONE",
                },
                "data_timestamp": max(str(item.get("data_timestamp", "")) for item in selected),
                "contract_resolved": contract_resolved,
                "con_id": contracts.get(ticker, {}).get("con_id"),
                "signal_contract_currency_status": (
                    "SIGNAL_CONTRACT_CURRENCY_MISMATCH"
                    if contract_currency_mismatch
                    else "SIGNAL_CONTRACT_CURRENCY_MATCH"
                    if contract_resolved
                    else "CONTRACT_IDENTITY_REQUIRED"
                ),
                "signal_currency": signal_currency,
                "contract_currency": contract_currency,
                "shariah_status": shariah,
                "event_risk": round(event_risk, 6),
                "research_allocation_eligible": not research_blockers,
                "deployment_eligible": not deployment_blockers,
                "evidence_tiers": evidence_tiers,
                "research_allocation_blockers": sorted(set(research_blockers)),
                "position_management_blockers": sorted(set(management_blockers)),
                "execution_blockers": sorted(set(research_blockers)),
                "deployment_blockers": sorted(set(deployment_blockers)),
                "preferred_entry": preferred.get("preferred_entry"),
                "stop_loss": preferred.get("stop_loss"),
                "take_profit_1": preferred.get("take_profit_1"),
                "take_profit_2": preferred.get("take_profit_2"),
                "stop_risk_pct": round(stop_risk_pct, 6),
                "currency": signal_currency,
                "entry_method": str(preferred.get("entry_type") or "NEXT_BAR_LIMIT_OR_STOP_LIMIT"),
                "stop_method": str(preferred.get("stop_method") or management["stop_method"]),
                "exit_policy": str(preferred.get("exit_policy") or management["exit_policy"]),
                "expected_holding_period": str(
                    preferred.get("expected_holding_period")
                    or management["expected_holding_period"]
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["opportunity_score"]),
            len(_research_blockers(row)),
            row["ticker"],
        )
    )
    return rows


def allocate_research_portfolio(
    ranked: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    technical_regime: dict[str, Any],
    macro: dict[str, Any],
    correlation: pd.DataFrame,
    daily_target: dict[str, Any],
    dynamic_risk: dict[str, Any] | None = None,
    whole_share_feasible_tickers: set[str] | None = None,
) -> dict[str, Any]:
    portfolio = policy["portfolio"]
    maximum_portfolio_heat = float(
        (dynamic_risk or {}).get(
            "maximum_portfolio_heat",
            portfolio["maximum_portfolio_heat"],
        )
    )
    minimum_meaningful_target_weight = float(
        (dynamic_risk or {}).get("minimum_meaningful_target_weight", 0.0)
    )
    regime = str(technical_regime.get("regime", "UNKNOWN"))
    market_regime = str(macro.get("regime", {}).get("market_regime", "UNKNOWN"))
    target = (
        float(portfolio["research_maximum_exposure"])
        * float(policy["regime_multipliers"].get(regime, 0.4))
        * float(policy["macro_market_multipliers"].get(market_regime, 0.65))
    )
    position_limit = int(portfolio["maximum_positions"])
    if dynamic_risk is not None:
        position_limit = min(
            position_limit,
            int(dynamic_risk["dynamic_research_maximum_positions"]),
        )
        target *= float(dynamic_risk["multipliers"]["combined"])
        target *= float(dynamic_risk["signal_scarcity_multiplier"])
        if not bool(dynamic_risk.get("new_entries_allowed", True)):
            target = 0.0
    if _target_throttle(daily_target):
        target = 0.0
    candidates = [
        row
        for row in ranked
        if not _research_blockers(row)
        and float(row["opportunity_score"]) >= float(policy["ranking"]["minimum_allocation_score"])
        and (
            whole_share_feasible_tickers is None
            or str(row["ticker"]).upper() in whole_share_feasible_tickers
        )
    ][: position_limit * 3]
    allocations: list[dict[str, Any]] = []
    sleeve_used: dict[str, float] = defaultdict(float)
    sector_used: dict[str, float] = defaultdict(float)
    region_used: dict[str, float] = defaultdict(float)
    cluster_used: dict[str, float] = defaultdict(float)
    asset_class_used: dict[str, float] = defaultdict(float)
    used = 0.0
    heat = 0.0
    for candidate in candidates:
        if len(allocations) >= position_limit:
            break
        ticker = candidate["ticker"]
        max_correlation = _max_selected_correlation(ticker, allocations, correlation)
        correlation_penalty = (
            max(0.0, max_correlation - 0.5) if math.isfinite(max_correlation) else 0.15
        )
        score = max(
            0.0,
            float(
                candidate.get(
                    "portfolio_objective_score",
                    candidate["opportunity_score"],
                )
            )
            - correlation_penalty * float(policy["penalties"]["correlation"]),
        )
        volatility = _volatility_from_matrix(ticker, correlation)
        risk_score = score / max(volatility, 0.12)
        raw_weight = min(target / max(len(candidates), 1), risk_score * 0.03)
        if minimum_meaningful_target_weight > 0:
            raw_weight = max(
                raw_weight,
                min(minimum_meaningful_target_weight, target),
            )
        asset_cap = float(
            portfolio[
                "maximum_etf_weight"
                if candidate["asset_type"] != "STOCK"
                else "maximum_stock_weight"
            ]
        )
        if dynamic_risk is not None:
            dynamic_asset_cap = float(
                dynamic_risk.get(
                    "maximum_etf_position_weight"
                    if candidate["asset_type"] != "STOCK"
                    else "maximum_position_weight",
                    asset_cap,
                )
            )
            asset_cap = max(asset_cap, dynamic_asset_cap)
        sleeve_cap = float(policy["sleeve_caps"].get(candidate["sleeve"], target))
        sector_cap = float(portfolio["maximum_sector_weight"])
        region_cap = float(portfolio["maximum_region_weight"])
        cluster = str(candidate.get("correlation_cluster") or "UNKNOWN")
        cluster_cap = float(portfolio["maximum_correlated_bucket_weight"])
        normalized_asset_class = str(candidate.get("normalized_asset_class") or "EQUITY")
        asset_class_cap = float(
            policy.get("asset_class_caps", {}).get(normalized_asset_class, target)
        )
        weight = min(
            raw_weight,
            asset_cap,
            max(0.0, target - used),
            max(0.0, sleeve_cap - sleeve_used[candidate["sleeve"]]),
            max(0.0, sector_cap - sector_used[candidate["sector"]]),
            max(0.0, region_cap - region_used[candidate["region"]]),
            max(0.0, cluster_cap - cluster_used[cluster]),
            max(
                0.0,
                asset_class_cap - asset_class_used[normalized_asset_class],
            ),
        )
        stop_risk = max(float(candidate["stop_risk_pct"]), 0.01)
        heat_room = max(0.0, maximum_portfolio_heat - heat)
        weight = min(weight, heat_room / stop_risk)
        if weight <= 0.001:
            continue
        allocation = {
            **candidate,
            "target_weight": round(weight, 8),
            "correlation_penalty": round(correlation_penalty, 6),
            "maximum_correlation_to_selected": (
                round(max_correlation, 6) if math.isfinite(max_correlation) else None
            ),
        }
        allocations.append(allocation)
        used += weight
        heat += weight * stop_risk
        sleeve_used[candidate["sleeve"]] += weight
        sector_used[candidate["sector"]] += weight
        region_used[candidate["region"]] += weight
        cluster_used[cluster] += weight
        asset_class_used[normalized_asset_class] += weight
    weights = np.array([float(row["target_weight"]) for row in allocations], dtype=float)
    estimated_volatility = _portfolio_volatility(allocations, correlation)
    max_corr = max(
        (
            float(row["maximum_correlation_to_selected"])
            for row in allocations
            if row["maximum_correlation_to_selected"] is not None
        ),
        default=0.0,
    )
    effective_n = (
        float((weights.sum() ** 2) / np.square(weights).sum())
        if len(weights) and np.square(weights).sum() > 0
        else 0.0
    )
    return {
        "status": "GO" if allocations or target == 0 else "NO_TARGET_POSITIONS",
        "research_target_exposure": round(used, 8),
        "cash_weight": round(max(0.0, 1.0 - used), 8),
        "allocations": allocations,
        "portfolio_heat": round(heat, 8),
        "portfolio_heat_gate": ("GO" if heat <= maximum_portfolio_heat + 1e-12 else "BLOCKED"),
        "maximum_portfolio_heat": round(maximum_portfolio_heat, 8),
        "correlation_gate": (
            "GO" if max_corr <= float(portfolio["correlation_threshold"]) + 1e-12 else "DEGRADED"
        ),
        "estimated_annualized_volatility": estimated_volatility,
        "maximum_pairwise_correlation": round(max_corr, 6),
        "effective_position_count": round(effective_n, 6),
        "sleeve_weights": _rounded_map(sleeve_used),
        "sector_weights": _rounded_map(sector_used),
        "region_weights": _rounded_map(region_used),
        "correlation_cluster_weights": _rounded_map(cluster_used),
        "asset_class_weights": _rounded_map(asset_class_used),
        "regime_multiplier_source": regime,
        "macro_multiplier_source": market_regime,
        "position_limit": position_limit,
        "dynamic_risk_applied": dynamic_risk is not None,
        "dynamic_risk_multiplier": (
            float(dynamic_risk["multipliers"]["combined"]) if dynamic_risk is not None else 1.0
        ),
        "signal_scarcity_multiplier": (
            float(dynamic_risk["signal_scarcity_multiplier"]) if dynamic_risk is not None else 1.0
        ),
        "whole_share_preselection_applied": (whole_share_feasible_tickers is not None),
    }


def _whole_share_candidate_preflight(
    project_root: Path,
    *,
    current_positions: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    account: dict[str, Any],
    policy: dict[str, Any],
    dynamic_risk: dict[str, Any] | None,
) -> dict[str, Any]:
    """Independently test one whole share before portfolio selection."""
    small_policy = policy.get("small_account_whole_share", {})
    equity = _decimal(account.get("net_liquidation_eur"))
    enabled = (
        account.get("status") == "GO"
        and bool(small_policy.get("enabled", False))
        and equity > 0
        and equity <= _decimal(small_policy.get("maximum_equity_eur", 0))
    )
    if not enabled:
        return {
            "schema": "whole_share_candidate_preflight_v1",
            "status": "NOT_APPLICABLE",
            "evaluated_candidate_count": 0,
            "feasible_candidate_count": 0,
            "infeasible_candidate_count": 0,
            "selection_filter_applied": False,
            "feasible_tickers": [],
        }
    limit = int(policy.get("candidate_funnel", {}).get("maximum_portfolio_candidates", 30))
    candidates = [
        row
        for row in ranked
        if not _research_blockers(row)
        and float(row.get("opportunity_score") or 0.0)
        >= float(policy["ranking"]["minimum_allocation_score"])
    ][:limit]
    feasible: list[str] = []
    candidate_results: list[dict[str, Any]] = []
    status_counts: dict[str, int] = defaultdict(int)
    for candidate in candidates:
        sizing = build_private_sizing(
            project_root,
            current_positions=current_positions,
            ranked=ranked,
            allocation={
                "allocations": [
                    {
                        **candidate,
                        "target_weight": small_policy.get(
                            "minimum_meaningful_target_weight",
                            0.10,
                        ),
                    }
                ]
            },
            account=account,
            policy=policy,
            dynamic_risk=dynamic_risk,
        )
        row: dict[str, Any] = next(
            (
                item
                for item in sizing.get("positions", [])
                if str(item.get("ticker", "")).upper() == str(candidate["ticker"]).upper()
            ),
            {},
        )
        status = str(
            row.get("whole_share_feasibility_status") or "WHOLE_SHARE_PREFLIGHT_DATA_BLOCKED"
        )
        status_counts[status] += 1
        candidate_results.append(
            {
                "ticker": str(candidate["ticker"]).upper(),
                "asset_class": _canary_asset_class(candidate.get("asset_type")),
                "opportunity_score": candidate.get("opportunity_score"),
                "status": status,
                "execution_candidate_status": row.get(
                    "execution_candidate_status",
                    "NON_EXECUTABLE_WHOLE_SHARE",
                ),
                "risk_based_quantity": row.get("risk_based_quantity"),
                "cash_based_quantity": row.get("cash_based_quantity"),
                "whole_share_quantity": row.get("whole_share_quantity"),
                "share_price_local": row.get("reference_price"),
                "currency": row.get("currency"),
                "fx_to_eur": row.get("fx_to_eur"),
                "share_price_eur": row.get("unit_notional_eur"),
                "stop_price": row.get("stop_price"),
                "risk_per_share_eur": row.get("risk_per_share_eur"),
                "desired_qty": row.get("desired_qty"),
                "normal_allowed_qty": row.get("normal_allowed_qty"),
                "level1_canary_qty": row.get("level1_canary_qty"),
                "actual_notional_eur": row.get("canary_actual_notional_eur"),
                "actual_portfolio_weight": row.get("canary_actual_portfolio_weight"),
                "planned_risk_eur": row.get("canary_planned_risk_eur"),
                "canary_risk_budget_eur": row.get("canary_risk_budget_eur"),
                "canary_risk_utilization": row.get("canary_risk_utilization"),
                "cash_after_eur": row.get("canary_cash_after_eur"),
                "canary_sizing_reason": row.get("canary_sizing_reason"),
                "canary_blocking_reason": row.get("canary_blocking_reason"),
            }
        )
        if status == "WHOLE_SHARE_FEASIBLE_RISK_FIRST":
            feasible.append(str(candidate["ticker"]).upper())
    return {
        "schema": "whole_share_candidate_preflight_v1",
        "status": "GO" if candidates else "NO_ELIGIBLE_CANDIDATES",
        "evaluated_candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "infeasible_candidate_count": len(candidates) - len(feasible),
        "status_counts": dict(sorted(status_counts.items())),
        "selection_filter_applied": bool(feasible),
        "fallback_when_none_feasible": "RETAIN_TOP_RESEARCH_TARGET",
        "feasible_tickers": feasible,
        "candidate_results": sorted(candidate_results, key=lambda item: item["ticker"]),
        "execution_authority": "NONE",
    }


def position_actions(
    positions: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    allocation: dict[str, Any],
    *,
    policy: dict[str, Any],
    dynamic_overrides: dict[str, dict[str, Any]],
    daily_target: dict[str, Any],
    management_states: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    ranking = {row["ticker"]: row for row in ranked}
    targets = {row["ticker"]: row for row in allocation.get("allocations", [])}
    best_replacement = next(
        (row for row in allocation.get("allocations", []) if not _research_blockers(row)),
        None,
    )
    result = []
    management_states = management_states or {}
    for position in positions:
        ticker = position["symbol"]
        candidate = ranking.get(ticker)
        reasons: list[str] = []
        action = "HOLD"
        score = float(candidate["opportunity_score"]) if candidate else 0.0
        override = dynamic_overrides.get(ticker, {})
        management = management_states.get(ticker, {})
        if str(override.get("action")) == "AVOID":
            action = "EXIT"
            reasons.append("DYNAMIC_CONSENSUS_AVOID")
            reasons.extend(str(item) for item in override.get("risk_blockers", []))
        elif candidate is None:
            action = "REDUCE"
            reasons.append("NO_CURRENT_RANKED_OPPORTUNITY")
        elif _position_blockers(candidate):
            action = "EXIT"
            reasons.extend(_position_blockers(candidate))
        elif management.get("status") == "GO" and management.get("action") in {
            "EXIT",
            "REDUCE_50",
            "TAKE_PARTIAL_25",
            "TAKE_PARTIAL_50",
            "UPDATE_TRAILING_STOP",
        }:
            action = {
                "REDUCE_50": "REDUCE",
                "TAKE_PARTIAL_25": "TAKE_PARTIAL_PROFIT",
                "TAKE_PARTIAL_50": "TAKE_PARTIAL_PROFIT",
            }.get(str(management["action"]), str(management["action"]))
            reasons.extend(str(reason) for reason in management.get("reason_codes", []))
        elif score <= float(policy["ranking"]["exit_score"]):
            action = "EXIT"
            reasons.append("OPPORTUNITY_SCORE_BELOW_EXIT_THRESHOLD")
        elif _price_at_or_below_stop(candidate):
            action = "EXIT"
            reasons.append("STOP_INVALIDATION_OBSERVED")
        elif _price_at_or_above_tp1(candidate, position):
            action = "TAKE_PARTIAL_PROFIT"
            reasons.append("FIRST_PROFIT_OBJECTIVE_REACHED")
        elif (
            best_replacement
            and best_replacement["ticker"] != ticker
            and float(best_replacement["opportunity_score"]) - score
            >= float(policy["ranking"]["replacement_improvement"])
            + float(policy["ranking"]["switching_cost_score"])
        ):
            action = "REPLACE"
            reasons.append("SUPERIOR_RISK_ADJUSTED_REPLACEMENT")
        elif ticker in targets and score >= float(policy["ranking"]["minimum_add_score"]):
            action = "ADD"
            reasons.append("HIGH_SCORE_AND_TARGET_ALLOCATION")
        else:
            reasons.append("THESIS_REMAINS_WITHIN_HOLD_BAND")
        if _target_throttle(daily_target) and action in RISK_INCREASING_ACTIONS:
            action = "BLOCK_NEW_ENTRY"
            reasons.append("DAILY_PROFIT_TARGET_SOFT_THROTTLE")
        executable_action = "NO_ACTION"
        result.append(
            {
                "ticker": ticker,
                "position_identity": stable_hash(
                    {
                        "con_id": position["con_id"],
                        "symbol": ticker,
                    }
                )[:24],
                "advisory_action": action,
                "executable_action": executable_action,
                "reason_codes": sorted(set(reasons)),
                "opportunity_score": round(score, 6),
                "replacement_ticker": (
                    best_replacement["ticker"] if action == "REPLACE" and best_replacement else None
                ),
                "current_quantity": position["quantity"],
                "average_cost": position["average_cost"],
                "currency": position["currency"],
                "risk_increasing": action in RISK_INCREASING_ACTIONS,
                "automatic_execution_allowed": False,
                "execution_authority": "NONE",
                "position_management_status": management.get("status", "UNAVAILABLE"),
            }
        )
    return result


def build_capital_decisions(
    positions: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    allocation: dict[str, Any],
    actions: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    private_sizing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize portfolio advice into one bounded capital action language."""
    current = {str(row.get("symbol") or "").upper() for row in positions}
    ranking = {str(row["ticker"]): row for row in ranked}
    decisions: list[dict[str, Any]] = []
    action_map = {
        "ADD": "ADD",
        "HOLD": "HOLD",
        "REDUCE": "TRIM",
        "TAKE_PARTIAL_PROFIT": "TRIM",
        "REPLACE": "ROTATE",
        "EXIT": "EXIT",
        "UPDATE_TRAILING_STOP": "HOLD",
        "BLOCK_NEW_ENTRY": "HOLD",
    }
    minimum_score = float(policy["ranking"]["minimum_allocation_score"])
    switching_cost = float(policy["ranking"]["switching_cost_score"])
    sizing_by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in (private_sizing or {}).get("positions", [])
    }
    for row in actions:
        source_action = str(row.get("advisory_action") or "HOLD")
        formal_action = action_map.get(source_action, "HOLD")
        replacement = row.get("replacement_ticker")
        replacement_score = float(ranking.get(str(replacement), {}).get("opportunity_score", 0.0))
        current_score = float(row.get("opportunity_score") or 0.0)
        marginal_value = (
            replacement_score - current_score - switching_cost
            if formal_action == "ROTATE"
            else current_score - minimum_score
        )
        decisions.append(
            {
                "ticker": row.get("ticker"),
                "formal_action": formal_action,
                "source_action": source_action,
                "replacement_ticker": replacement,
                "rank_utility_margin": round(marginal_value, 8),
                "reason_codes": list(row.get("reason_codes") or []),
                "implementation_status": "OBSERVE_ONLY_AUTHORITY_NONE",
                "automatic_execution_allowed": False,
                "execution_authority": "NONE",
            }
        )
    for target in allocation.get("allocations", []):
        ticker = str(target.get("ticker") or "").upper()
        if not ticker or ticker in current:
            continue
        candidate = ranking.get(ticker, target)
        blockers = list(candidate.get("deployment_blockers") or [])
        sizing = sizing_by_ticker.get(ticker, {})
        feasibility = str(
            sizing.get("whole_share_feasibility_status")
            or "NOT_EVALUATED_PRIVATE_ACCOUNT_UNAVAILABLE"
        )
        implementation_status = (
            "BLOCKED_WHOLE_SHARE_FEASIBILITY"
            if feasibility.startswith("WHOLE_SHARE_")
            and feasibility != "WHOLE_SHARE_FEASIBLE_RISK_FIRST"
            else "BLOCKED_DEPLOYMENT_GATES"
            if blockers
            else "BLOCKED_EXECUTION_AUTHORITY_NONE"
        )
        decisions.append(
            {
                "ticker": ticker,
                "formal_action": "OPEN",
                "source_action": "TARGET_ALLOCATION_CANDIDATE",
                "replacement_ticker": None,
                "research_target_weight": float(target.get("target_weight") or 0.0),
                "approved_target_weight": 0.0,
                "rank_utility_margin": round(
                    float(candidate.get("opportunity_score") or 0.0) - minimum_score,
                    8,
                ),
                "whole_share_feasibility_status": feasibility,
                "minimum_feasible_weight": sizing.get("minimum_feasible_weight"),
                "sizing_basis": sizing.get("sizing_basis"),
                "reason_codes": (
                    ["RESEARCH_TARGET_SELECTED"]
                    if not blockers
                    else ["RESEARCH_TARGET_SELECTED", *blockers]
                ),
                "implementation_status": implementation_status,
                "automatic_execution_allowed": False,
                "execution_authority": "NONE",
            }
        )
    executable_open_count = sum(
        row["formal_action"] == "OPEN" and row["implementation_status"] == "EXECUTION_READY"
        for row in decisions
    )
    cash_weight = float(allocation.get("cash_weight") or 0.0)
    decisions.append(
        {
            "ticker": "CASH",
            "formal_action": "CASH",
            "source_action": "RESIDUAL_CAPITAL_ALLOCATION",
            "research_target_weight": cash_weight,
            "approved_target_weight": 1.0,
            "rank_utility_margin": 0.0,
            "reason_codes": [
                "CASH_IS_FIRST_CLASS_ASSET",
                (
                    "NO_EXECUTION_READY_ALTERNATIVE"
                    if executable_open_count == 0
                    else "RESIDUAL_AFTER_APPROVED_ALLOCATIONS"
                ),
            ],
            "implementation_status": "CURRENT_FAIL_CLOSED_DESTINATION",
            "automatic_execution_allowed": False,
            "execution_authority": "NONE",
        }
    )
    counts: defaultdict[str, int] = defaultdict(int)
    for row in decisions:
        counts[str(row["formal_action"])] += 1
    report = {
        "schema": "proactive_capital_decisions_v1",
        "status": "GO",
        "actions": decisions,
        "action_counts": dict(sorted(counts.items())),
        "opportunity_cost_is_first_class": True,
        "cash_is_first_class_asset": True,
        "rank_utility_is_not_expected_return": True,
        "automatic_execution_allowed": False,
        "execution_authority": "NONE",
        "orders_generated": 0,
        "broker_write_calls": 0,
    }
    report["content_hash"] = stable_hash(report)
    return report


def build_private_sizing(
    project_root: Path,
    *,
    current_positions: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    allocation: dict[str, Any],
    account: dict[str, Any],
    policy: dict[str, Any],
    dynamic_risk: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if account["status"] != "GO":
        return _blocked_private_sizing(account["status"])
    equity = _decimal(account["net_liquidation_eur"])
    available = _decimal(
        account.get(
            "research_sizing_capacity_eur",
            min(
                _decimal(account["available_funds_eur"]),
                _decimal(account["total_cash_value_eur"]),
            ),
        )
    )
    if equity <= 0 or available < 0:
        return _blocked_private_sizing("INVALID_PRIVATE_ACCOUNT_VALUES")
    ranked_map = {row["ticker"]: row for row in ranked}
    current_map = {row["symbol"]: _decimal(row["quantity"]) for row in current_positions}
    capacity = _capacity_map(project_root)
    fx = _fx_map(project_root)
    shared_cost_model = load_shared_cost_model(project_root)
    sizing_rows: list[dict[str, Any]] = []
    cash_remaining = available
    base_risk = _decimal(
        dynamic_risk["base_risk_per_trade"]
        if dynamic_risk is not None
        else policy["portfolio"]["shadow_base_risk_per_trade"]
    )
    risk_multiplier = _decimal(
        dynamic_risk["multipliers"]["combined"] if dynamic_risk is not None else 1
    )
    small_policy = policy.get("small_account_whole_share", {})
    small_account_mode = bool(small_policy.get("enabled", False)) and (
        equity <= _decimal(small_policy.get("maximum_equity_eur", 0))
    )
    maximum_discrete_exposure = _decimal(
        small_policy.get("maximum_discrete_exposure", 0.5)
        if small_account_mode
        else policy["portfolio"]["research_maximum_exposure"]
    )
    aggregate_notional_room = equity * maximum_discrete_exposure
    maximum_portfolio_heat = _decimal(
        (dynamic_risk or {}).get(
            "maximum_portfolio_heat",
            policy["portfolio"]["maximum_portfolio_heat"],
        )
    )
    aggregate_heat_room = equity * maximum_portfolio_heat
    for target in allocation.get("allocations", []):
        ticker = str(target["ticker"])
        currency = str(target.get("currency") or "EUR").upper()
        fx_rate = fx.get(currency)
        price = _decimal(target.get("preferred_entry"))
        stop = _decimal(target.get("stop_loss"))
        current_quantity = current_map.get(ticker, Decimal("0"))
        blockers: list[str] = []
        if fx_rate is None or fx_rate <= 0:
            blockers.append("FX_RATE_UNAVAILABLE")
        if price <= 0:
            blockers.append("POSITIVE_PRICE_REQUIRED")
        if stop <= 0 or stop >= price:
            blockers.append("VALID_LONG_STOP_REQUIRED")
        capacity_eur = capacity.get(ticker)
        if capacity_eur is None or capacity_eur <= 0:
            blockers.append("LIQUIDITY_CAP_UNAVAILABLE")
        target_quantity = Decimal("0")
        desired_quantity = Decimal("0")
        level1_canary_quantity = Decimal("0")
        canary_sizing_reason = "NOT_EVALUATED_DATA_BLOCKED"
        canary_blocking_reason: str | None = "NOT_EVALUATED_DATA_BLOCKED"
        canary_risk_budget = Decimal("0")
        canary_risk_utilization = Decimal("0")
        canary_actual_notional = Decimal("0")
        canary_actual_weight = Decimal("0")
        canary_planned_risk = Decimal("0")
        canary_cash_after = cash_remaining
        whole_share_feasibility_status = "NOT_EVALUATED_DATA_BLOCKED"
        minimum_feasible_weight: Decimal | None = None
        effective_position_cap = Decimal("0")
        risk_budget = Decimal("0")
        risk_quantity = Decimal("0")
        position_cap_quantity = Decimal("0")
        capacity_quantity = Decimal("0")
        cash_quantity = Decimal("0")
        aggregate_exposure_quantity = Decimal("0")
        aggregate_heat_quantity = Decimal("0")
        desired_notional = equity * _decimal(target.get("target_weight", 0))
        take_profit = _decimal(target.get("take_profit_1"))
        take_profit_source = "SIGNAL_TAKE_PROFIT"
        if take_profit <= price and stop > 0 and stop < price:
            take_profit = price + (price - stop) * _decimal(
                small_policy.get("protective_take_profit_reward_to_risk", 2.0)
            )
            take_profit_source = "PROTECTIVE_REWARD_TO_RISK_FALLBACK"
        estimated_entry_cost = Decimal("0")
        economic_cost_incoherent = False
        per_share_risk = Decimal("0")
        unit_eur = price * fx_rate if fx_rate is not None and price > 0 else Decimal("0")
        if not blockers and unit_eur > 0 and fx_rate is not None and capacity_eur is not None:
            minimum_feasible_weight = unit_eur / equity
            desired_quantity = _whole_units(desired_notional / unit_eur)
            if (
                desired_quantity < 1
                and desired_notional > 0
                and bool(
                    small_policy.get(
                        "round_up_to_one_share_when_risk_feasible",
                        False,
                    )
                )
            ):
                desired_quantity = Decimal("1")
            policy_position_cap = _decimal(
                policy["portfolio"][
                    "maximum_etf_weight"
                    if target["asset_type"] != "STOCK"
                    else "maximum_stock_weight"
                ]
            )
            if small_account_mode:
                small_cap = _decimal(
                    small_policy.get(
                        "maximum_etf_weight"
                        if target["asset_type"] != "STOCK"
                        else "maximum_stock_weight",
                        policy_position_cap,
                    )
                )
                dynamic_cap = _decimal(
                    (dynamic_risk or {}).get("maximum_position_weight", small_cap)
                )
                effective_position_cap = min(small_cap, dynamic_cap)
            else:
                effective_position_cap = policy_position_cap
            risk_budget = (
                equity * base_risk * risk_multiplier * _decimal(target["opportunity_score"])
            )
            one_share_costs = estimate_transaction_cost(
                unit_eur,
                currency=currency,
                model=shared_cost_model,
                round_trip=True,
            )
            adverse_cost_per_share = sum(
                _decimal(one_share_costs[key])
                for key in (
                    "exchange_fees_eur",
                    "spread_eur",
                    "slippage_eur",
                    "market_impact_eur",
                    "fx_conversion_eur",
                )
            )
            per_share_risk = abs(price - stop) * fx_rate + adverse_cost_per_share
            risk_quantity = _whole_units(risk_budget / per_share_risk)
            position_cap_quantity = _whole_units(equity * effective_position_cap / unit_eur)
            capacity_quantity = _whole_units(capacity_eur / unit_eur)
            cash_quantity = _whole_units(cash_remaining / unit_eur)
            aggregate_exposure_quantity = _whole_units(aggregate_notional_room / unit_eur)
            aggregate_heat_quantity = _whole_units(aggregate_heat_room / per_share_risk)
            target_quantity = min(
                desired_quantity,
                risk_quantity,
                position_cap_quantity,
                capacity_quantity,
                cash_quantity + current_quantity,
                aggregate_exposure_quantity + current_quantity,
                aggregate_heat_quantity + current_quantity,
            )
            cost_policy = small_policy.get("transaction_cost_model", {})
            if bool(cost_policy.get("enabled", False)) and target_quantity >= 1:
                shared_entry_cost = _decimal(
                    estimate_transaction_cost(
                        target_quantity * unit_eur,
                        currency=currency,
                        model=shared_cost_model,
                        round_trip=False,
                    )["total_cost_eur"]
                )
                conservative_stress_cost = _decimal(
                    cost_policy.get("estimated_entry_commission_eur", 0)
                ) + (
                    target_quantity
                    * unit_eur
                    * _decimal(cost_policy.get("estimated_slippage_bps", 0))
                    / Decimal("10000")
                )
                estimated_entry_cost = max(shared_entry_cost, conservative_stress_cost)
                actual_initial_risk = target_quantity * per_share_risk
                minimum_notional = max(
                    shared_cost_model.minimum_practical_trade_eur,
                    _decimal(cost_policy.get("minimum_economic_notional_eur", 0)),
                )
                maximum_cost_ratio = _decimal(
                    cost_policy.get("maximum_cost_to_initial_risk_ratio", "Infinity")
                )
                if target_quantity * unit_eur < minimum_notional or (
                    actual_initial_risk <= 0
                    or estimated_entry_cost > actual_initial_risk * maximum_cost_ratio
                ):
                    target_quantity = Decimal("0")
                    economic_cost_incoherent = True
            if economic_cost_incoherent:
                whole_share_feasibility_status = "WHOLE_SHARE_ECONOMIC_COST_INCOHERENT"
            elif target_quantity >= 1:
                whole_share_feasibility_status = "WHOLE_SHARE_FEASIBLE_RISK_FIRST"
            elif risk_quantity < 1:
                whole_share_feasibility_status = "WHOLE_SHARE_RISK_BUDGET_INSUFFICIENT"
            elif position_cap_quantity < 1:
                whole_share_feasibility_status = "WHOLE_SHARE_NOTIONAL_CAP_EXCEEDED"
            elif capacity_quantity < 1:
                whole_share_feasibility_status = "WHOLE_SHARE_LIQUIDITY_CAP_INSUFFICIENT"
            elif cash_quantity < 1:
                whole_share_feasibility_status = "WHOLE_SHARE_CASH_INSUFFICIENT"
            elif aggregate_exposure_quantity < 1:
                whole_share_feasibility_status = "WHOLE_SHARE_PORTFOLIO_EXPOSURE_CAP_EXCEEDED"
            else:
                whole_share_feasibility_status = "WHOLE_SHARE_PORTFOLIO_HEAT_CAP_EXCEEDED"
            asset_class = _canary_asset_class(target.get("asset_type"))
            canary = evaluate_whole_share_canary(
                project_root,
                asset_class=asset_class,
                instrument_currency=currency,
                desired_qty=desired_quantity,
                account_equity_eur=equity,
                available_cash_eur=cash_remaining,
                reserved_cash_eur=Decimal("0"),
                entry_price_local=price,
                protective_stop_local=stop,
                take_profit_local=take_profit,
                fx_rate_to_eur=fx_rate,
                normal_risk_budget_eur=risk_budget,
                normal_maximum_position_weight=effective_position_cap,
                normal_maximum_portfolio_heat_pct=maximum_portfolio_heat,
                liquidity_notional_eur=capacity_eur,
                existing_position_notional_eur=(current_quantity * unit_eur),
                existing_total_exposure_eur=max(
                    Decimal("0"),
                    equity * maximum_discrete_exposure - aggregate_notional_room,
                ),
                existing_portfolio_risk_eur=max(
                    Decimal("0"),
                    equity * maximum_portfolio_heat - aggregate_heat_room,
                ),
            )
            level1_canary_quantity = _decimal(canary.get("canary_qty"))
            canary_sizing_reason = str(canary.get("sizing_reason") or "CANARY_SIZING_BLOCKED")
            canary_blocking_reason = canary.get("blocking_reason")
            if target_quantity < 1:
                level1_canary_quantity = Decimal("0")
                canary_sizing_reason = whole_share_feasibility_status
                canary_blocking_reason = whole_share_feasibility_status
            else:
                level1_canary_quantity = min(level1_canary_quantity, target_quantity)
            canary_risk_budget = _decimal(canary.get("canary_risk_budget_eur"))
            canary_risk_utilization = _decimal(canary.get("canary_risk_utilization"))
            if level1_canary_quantity < 1:
                canary_risk_utilization = Decimal("0")
            canary_actual_notional = level1_canary_quantity * unit_eur
            canary_actual_weight = canary_actual_notional / equity
            canary_planned_risk = level1_canary_quantity * per_share_risk
            canary_cash_after = max(
                Decimal("0"),
                cash_remaining - canary_actual_notional,
            )
            additional = max(Decimal("0"), target_quantity - current_quantity)
            cash_remaining -= additional * unit_eur
            aggregate_notional_room -= additional * unit_eur
            aggregate_heat_room -= additional * per_share_risk
        sizing_rows.append(
            {
                "ticker": ticker,
                "opportunity_score": str(target["opportunity_score"]),
                "research_target_weight": str(target["target_weight"]),
                "currency": currency,
                "fx_to_eur": str(fx_rate) if fx_rate else None,
                "reference_price": str(price),
                "stop_price": str(stop),
                "take_profit_price": str(take_profit),
                "take_profit_source": take_profit_source,
                "current_quantity": str(current_quantity),
                "unconstrained_target_quantity": str(target_quantity),
                "target_quantity": str(target_quantity),
                "planned_quantity_delta": str(target_quantity - current_quantity),
                "unit_notional_eur": str(unit_eur),
                "target_notional_eur": str(target_quantity * unit_eur),
                "desired_notional_eur": str(desired_notional),
                "desired_qty": str(desired_quantity),
                "normal_allowed_qty": str(target_quantity),
                "level1_canary_qty": str(level1_canary_quantity),
                "canary_actual_notional_eur": str(canary_actual_notional),
                "canary_actual_portfolio_weight": str(canary_actual_weight),
                "canary_planned_risk_eur": str(canary_planned_risk),
                "canary_cash_after_eur": str(canary_cash_after),
                "risk_based_quantity": str(risk_quantity),
                "cash_based_quantity": str(cash_quantity),
                "whole_share_quantity": str(target_quantity),
                "actual_notional_eur": str(target_quantity * unit_eur),
                "actual_portfolio_weight": str(target_quantity * unit_eur / equity),
                "actual_risk_eur": str(target_quantity * per_share_risk),
                "remaining_cash_eur": str(cash_remaining),
                "estimated_entry_cost_eur": str(estimated_entry_cost),
                "transaction_cost_model": (
                    "SHARED_TRANSACTION_COST_MODEL_V1_WITH_CONSERVATIVE_STRESS_FLOOR"
                ),
                "planned_notional_eur": str(abs(target_quantity - current_quantity) * unit_eur),
                "risk_per_share_eur": str(per_share_risk),
                "risk_budget_eur": str(risk_budget),
                "canary_risk_budget_eur": str(canary_risk_budget),
                "canary_risk_utilization": str(canary_risk_utilization),
                "canary_sizing_reason": canary_sizing_reason,
                "canary_blocking_reason": canary_blocking_reason,
                "capital_level_1_semantics": ("WHOLE_SHARE_EXECUTION_CANARY"),
                "minimum_feasible_weight": (
                    str(minimum_feasible_weight) if minimum_feasible_weight is not None else None
                ),
                "effective_position_cap": str(effective_position_cap),
                "whole_share_feasibility_status": (whole_share_feasibility_status),
                "execution_candidate_status": (
                    "EXECUTABLE_WHOLE_SHARE"
                    if target_quantity >= 1
                    else "NON_EXECUTABLE_WHOLE_SHARE"
                ),
                "sizing_basis": "RISK_FIRST_WHOLE_SHARE_V2",
                "small_account_whole_share_mode": small_account_mode,
                "risk_quantity": str(risk_quantity),
                "position_cap_quantity": str(position_cap_quantity),
                "capacity_quantity": str(capacity_quantity),
                "cash_quantity": str(cash_quantity),
                "execution_blockers": blockers,
                "whole_share": (target_quantity == target_quantity.to_integral_value()),
            }
        )
    target_symbols = {row["ticker"] for row in sizing_rows}
    for position in current_positions:
        ticker = position["symbol"]
        if ticker in target_symbols:
            continue
        candidate = ranked_map.get(ticker, {})
        currency = str(position.get("currency") or "EUR").upper()
        fx_rate = fx.get(currency)
        price = _decimal(candidate.get("preferred_entry"))
        quantity = _decimal(position["quantity"])
        blockers = []
        if fx_rate is None or fx_rate <= 0:
            blockers.append("FX_RATE_UNAVAILABLE")
        if price <= 0:
            blockers.append("CURRENT_PRICE_UNAVAILABLE")
        unit_eur = price * fx_rate if fx_rate is not None and price > 0 else Decimal("0")
        sizing_rows.append(
            {
                "ticker": ticker,
                "opportunity_score": str(candidate.get("opportunity_score", "0")),
                "research_target_weight": "0",
                "currency": currency,
                "fx_to_eur": str(fx_rate) if fx_rate else None,
                "reference_price": str(price),
                "stop_price": str(_decimal(candidate.get("stop_loss"))),
                "current_quantity": str(quantity),
                "unconstrained_target_quantity": "0",
                "target_quantity": "0",
                "planned_quantity_delta": str(-quantity),
                "unit_notional_eur": str(unit_eur),
                "target_notional_eur": "0",
                "planned_notional_eur": str(quantity * unit_eur),
                "risk_per_share_eur": "0",
                "execution_blockers": blockers,
                "whole_share": (quantity == quantity.to_integral_value()),
            }
        )
    _apply_turnover_gate(sizing_rows, equity, policy)
    cash_remaining = available - sum(
        (
            max(
                Decimal("0"),
                _decimal(row["target_quantity"]) - _decimal(row["current_quantity"]),
            )
            * _decimal(row["unit_notional_eur"])
            for row in sizing_rows
        ),
        Decimal("0"),
    )
    current_notional = sum(
        (
            _decimal(row["current_quantity"]) * _decimal(row["unit_notional_eur"])
            for row in sizing_rows
        ),
        Decimal("0"),
    )
    target_notional = sum(
        (
            _decimal(row["target_quantity"]) * _decimal(row["unit_notional_eur"])
            for row in sizing_rows
        ),
        Decimal("0"),
    )
    turnover = sum(
        (
            abs(_decimal(row["planned_quantity_delta"])) * _decimal(row["unit_notional_eur"])
            for row in sizing_rows
        ),
        Decimal("0"),
    )
    current_heat = sum(
        (
            _decimal(row["current_quantity"])
            * _decimal(row["unit_notional_eur"])
            / equity
            * _decimal(ranked_map.get(row["ticker"], {}).get("stop_risk_pct", 0))
            for row in sizing_rows
        ),
        Decimal("0"),
    )
    target_heat = sum(
        (
            _decimal(row["target_quantity"])
            * _decimal(row["unit_notional_eur"])
            / equity
            * _decimal(ranked_map.get(row["ticker"], {}).get("stop_risk_pct", 0))
            for row in sizing_rows
        ),
        Decimal("0"),
    )
    whole_share_violations = sum(not bool(row["whole_share"]) for row in sizing_rows)
    negative_cash = int(cash_remaining < 0)
    blockers = sorted({blocker for row in sizing_rows for blocker in row["execution_blockers"]})
    hard_blockers = [
        blocker for blocker in blockers if blocker != "RISK_INCREASING_TURNOVER_CAPPED"
    ]
    status = (
        "GO"
        if not hard_blockers and whole_share_violations == 0 and negative_cash == 0
        else "DATA_BLOCKED"
    )
    if status == "GO" and blockers:
        status = "GO_WITH_CONSTRAINTS"
    return {
        "schema": "private_whole_share_portfolio_sizing_v1",
        "status": status,
        "account_snapshot_hash": account["snapshot_hash"],
        "account_equity_eur": str(equity),
        "available_funds_eur": str(available),
        "cash_remaining_eur": str(cash_remaining),
        "current_gross_exposure_pct": _ratio(current_notional, equity),
        "target_gross_exposure_pct": _ratio(target_notional, equity),
        "turnover_pct": _ratio(turnover, equity),
        "current_portfolio_heat": _ratio(current_heat, Decimal("1")),
        "target_portfolio_heat": _ratio(target_heat, Decimal("1")),
        "target_position_count": sum(_decimal(row["target_quantity"]) > 0 for row in sizing_rows),
        "whole_share_feasible_candidate_count": sum(
            row.get("whole_share_feasibility_status") == "WHOLE_SHARE_FEASIBLE_RISK_FIRST"
            for row in sizing_rows
        ),
        "whole_share_infeasible_candidate_count": sum(
            str(row.get("whole_share_feasibility_status", "")).startswith("WHOLE_SHARE_")
            and row.get("whole_share_feasibility_status") != "WHOLE_SHARE_FEASIBLE_RISK_FIRST"
            for row in sizing_rows
        ),
        "netted_security_count": len(sizing_rows),
        "whole_share_violation_count": whole_share_violations,
        "negative_cash_violation_count": negative_cash,
        "security_netting_status": (
            "GO"
            if len({row["ticker"] for row in sizing_rows}) == len(sizing_rows)
            else "DUPLICATE_SECURITY_BLOCKED"
        ),
        "turnover_gate": _turnover_gate_status(sizing_rows, equity, policy),
        "blockers": blockers,
        "positions": sorted(sizing_rows, key=lambda row: row["ticker"]),
        "approved_target_quantities": 0,
        "dynamic_risk_applied": dynamic_risk is not None,
        "dynamic_risk_multiplier": float(risk_multiplier),
        "small_account_whole_share_mode": small_account_mode,
        "sizing_basis": "RISK_FIRST_WHOLE_SHARE_V2",
        "maximum_discrete_exposure": float(maximum_discrete_exposure),
        "maximum_portfolio_heat": float(maximum_portfolio_heat),
        "automatic_submission": False,
        "execution_authority": "NONE",
    }


def _apply_portfolio_risk_overlay(
    actions: list[dict[str, Any]],
    *,
    private_sizing: dict[str, Any],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    if private_sizing["status"] not in {"GO", "GO_WITH_CONSTRAINTS"}:
        return [
            {
                **row,
                "advisory_action": (
                    "BLOCK_NEW_ENTRY" if row["risk_increasing"] else row["advisory_action"]
                ),
                "reason_codes": sorted(
                    {
                        *row["reason_codes"],
                        "PRIVATE_SIZING_DATA_BLOCKED",
                    }
                ),
                "risk_increasing": False,
            }
            for row in actions
        ]
    maximum_heat = float(
        private_sizing.get(
            "maximum_portfolio_heat",
            policy["portfolio"]["maximum_portfolio_heat"],
        )
    )
    maximum_exposure = float(policy["portfolio"]["research_maximum_exposure"])
    heat_breach = float(private_sizing["current_portfolio_heat"]) > maximum_heat
    exposure_breach = float(private_sizing["current_gross_exposure_pct"]) > maximum_exposure
    if not heat_breach and not exposure_breach:
        return actions
    result = []
    for row in actions:
        action = row["advisory_action"]
        reasons = set(row["reason_codes"])
        if heat_breach:
            reasons.add("CURRENT_PORTFOLIO_HEAT_LIMIT_BREACHED")
        if exposure_breach:
            reasons.add("CURRENT_GROSS_EXPOSURE_LIMIT_BREACHED")
        if action in RISK_INCREASING_ACTIONS:
            action = "BLOCK_NEW_ENTRY"
        elif action == "HOLD":
            action = "REDUCE"
        result.append(
            {
                **row,
                "advisory_action": action,
                "reason_codes": sorted(reasons),
                "risk_increasing": action in RISK_INCREASING_ACTIONS,
            }
        )
    return result


def _apply_turnover_gate(
    rows: list[dict[str, Any]],
    equity: Decimal,
    policy: dict[str, Any],
) -> None:
    buy_budget = equity * _decimal(policy["portfolio"]["maximum_turnover_per_rebalance"])
    for row in sorted(
        rows,
        key=lambda item: (
            _decimal(item["planned_quantity_delta"]) <= 0,
            item["ticker"],
        ),
    ):
        delta = _decimal(row["planned_quantity_delta"])
        unit = _decimal(row["unit_notional_eur"])
        if delta <= 0 or unit <= 0:
            continue
        affordable = _whole_units(buy_budget / unit)
        allowed_delta = min(delta, affordable)
        if allowed_delta < delta:
            row["execution_blockers"].append("RISK_INCREASING_TURNOVER_CAPPED")
        current = _decimal(row["current_quantity"])
        target = current + allowed_delta
        row["target_quantity"] = str(target)
        row["planned_quantity_delta"] = str(allowed_delta)
        row["target_notional_eur"] = str(target * unit)
        row["planned_notional_eur"] = str(allowed_delta * unit)
        buy_budget -= allowed_delta * unit


def _turnover_gate_status(
    rows: list[dict[str, Any]],
    equity: Decimal,
    policy: dict[str, Any],
) -> str:
    increasing = sum(
        max(Decimal("0"), _decimal(row["planned_quantity_delta"]))
        * _decimal(row["unit_notional_eur"])
        for row in rows
    )
    cap = equity * _decimal(policy["portfolio"]["maximum_turnover_per_rebalance"])
    reducing = any(_decimal(row["planned_quantity_delta"]) < 0 for row in rows)
    if increasing <= cap:
        return "GO_WITH_RISK_REDUCING_EXIT" if reducing else "GO"
    return "RISK_INCREASING_TURNOVER_BLOCKED"


def _private_account_state(project_root: Path) -> dict[str, Any]:
    snapshot = _latest_portfolio_broker_snapshot(project_root)
    account_authority = (
        "RECONCILED_READ_ONLY"
        if snapshot and snapshot.get("observation_environment") == "LIVE_READ_ONLY"
        else "RESEARCH_ONLY_PHASE8_OBSERVATION"
    )
    if not _snapshot_has_required_eur_account_values(snapshot):
        snapshot = _latest_complete_research_account_snapshot(project_root)
        account_authority = "RESEARCH_ONLY_LAST_OBSERVED"
    if not snapshot:
        return _blocked_private_account("PRIVATE_BROKER_SNAPSHOT_UNAVAILABLE")
    values: dict[str, Decimal] = {}
    for row in snapshot.get("account", {}).get("values", []):
        if str(row.get("currency", "")).upper() != "EUR":
            continue
        tag = str(row.get("tag") or "")
        if tag in {
            "NetLiquidation",
            "AvailableFunds",
            "BuyingPower",
            "TotalCashValue",
            "GrossPositionValue",
        }:
            try:
                observed_value = Decimal(str(row.get("value")))
            except (ArithmeticError, ValueError):
                return _blocked_private_account("INVALID_EUR_ACCOUNT_VALUE")
            if not observed_value.is_finite():
                return _blocked_private_account("INVALID_EUR_ACCOUNT_VALUE")
            values[tag] = observed_value
    required = {
        "NetLiquidation",
        "AvailableFunds",
        "TotalCashValue",
    }
    if not required.issubset(values):
        return _blocked_private_account("REQUIRED_EUR_ACCOUNT_TAGS_UNAVAILABLE")
    if values["NetLiquidation"] <= 0:
        return _blocked_private_account("POSITIVE_NET_LIQUIDATION_REQUIRED")
    if any(
        values.get(tag, Decimal("0")) < 0
        for tag in (
            "AvailableFunds",
            "BuyingPower",
            "TotalCashValue",
            "GrossPositionValue",
        )
    ):
        return _blocked_private_account("NONNEGATIVE_EUR_ACCOUNT_VALUES_REQUIRED")
    economic = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=bool(snapshot.get("content_hash")),
        fx_to_eur=_fx_map(project_root),
    )
    research_capacity = economic.get("research_sizing_capacity_eur")
    if economic.get("research_status") != "RESEARCH_READY" or (research_capacity is None):
        return _blocked_private_account("ECONOMIC_ACCOUNT_RESEARCH_MODEL_NOT_READY")
    return {
        "schema": "private_portfolio_account_state_v1",
        "status": "GO",
        "snapshot_hash": str(snapshot.get("content_hash") or ""),
        "snapshot_completed_at": snapshot.get("snapshot_completed_at"),
        "net_liquidation_eur": str(values["NetLiquidation"]),
        "available_funds_eur": str(values["AvailableFunds"]),
        "buying_power_eur": str(values.get("BuyingPower", "0")),
        "total_cash_value_eur": str(values["TotalCashValue"]),
        "gross_position_value_eur": str(values.get("GrossPositionValue", "0")),
        "reporting_value_eur": economic.get("reporting_value_eur"),
        "reporting_cash_eur": economic.get("reporting_cash_eur"),
        "spendable_eur": economic.get("spendable_eur"),
        "research_sizing_capacity_eur": research_capacity,
        "execution_sizing_capacity_eur": economic.get("execution_sizing_capacity_eur"),
        "eur_available_for_new_longs": economic.get("eur_available_for_new_longs"),
        "economic_account_lifecycle": economic.get("lifecycle_state"),
        "execution_account_status": economic.get("execution_status"),
        "execution_account_blockers": economic.get("execution_blockers", []),
        "buying_power_is_cash": False,
        "net_liquidation_is_deployable_cash": False,
        "implicit_fx_conversion_assumed": False,
        "account_authority": snapshot.get("account_authority", account_authority),
        "snapshot_source": snapshot.get("observation_environment", "PHASE8_READ_ONLY"),
        "snapshot_age_seconds": snapshot.get("snapshot_age_seconds"),
        "fresh_reconciliation": bool(snapshot.get("observation_environment") == "LIVE_READ_ONLY"),
        "execution_eligible": False,
        "equity_observed": True,
        "available_funds_observed": True,
        "raw_account_id_stored": False,
    }


def _record_action_cycle(
    project_root: Path,
    *,
    actions: list[dict[str, Any]],
    private_sizing: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    private_root = project_root / PRIVATE_ROOT
    private_root.mkdir(parents=True, exist_ok=True)
    path = private_root / "portfolio_actions.sqlite3"
    decision_payload = {
        "account_snapshot_hash": private_sizing.get("account_snapshot_hash"),
        "policy_hash": stable_hash(policy),
        "actions": [
            {
                "ticker": row["ticker"],
                "action": row["advisory_action"],
                "reasons": row["reason_codes"],
                "replacement": row["replacement_ticker"],
                "opportunity_score": row.get("opportunity_score"),
            }
            for row in actions
        ],
        "sizing": [
            {
                "ticker": row["ticker"],
                "current_quantity": row["current_quantity"],
                "target_quantity": row["target_quantity"],
                "planned_quantity_delta": row["planned_quantity_delta"],
                "opportunity_score": row.get("opportunity_score"),
                "research_target_weight": row.get("research_target_weight"),
            }
            for row in private_sizing.get("positions", [])
        ],
    }
    decision_id = stable_hash(decision_payload)
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS decision_cycles (
              decision_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              snapshot_hash TEXT,
              policy_hash TEXT NOT NULL,
              payload_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS action_events (
              event_id TEXT PRIMARY KEY,
              decision_id TEXT NOT NULL,
              ticker TEXT NOT NULL,
              advisory_action TEXT NOT NULL,
              reason_json TEXT NOT NULL,
              sizing_json TEXT,
              created_at TEXT NOT NULL
            );
            """
        )
        before = connection.total_changes
        connection.execute(
            """
            INSERT OR IGNORE INTO decision_cycles
            (decision_id, created_at, snapshot_hash, policy_hash, payload_hash)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                now,
                private_sizing.get("account_snapshot_hash"),
                stable_hash(policy),
                stable_hash(decision_payload),
            ),
        )
        cycle_inserted = connection.total_changes > before
        sizing_map = {row["ticker"]: row for row in private_sizing.get("positions", [])}
        inserted_events = 0
        for action in actions:
            private_row = sizing_map.get(action["ticker"])
            event_id = stable_hash(
                {
                    "decision_id": decision_id,
                    "ticker": action["ticker"],
                    "action": action["advisory_action"],
                    "replacement": action["replacement_ticker"],
                }
            )
            before = connection.total_changes
            connection.execute(
                """
                INSERT OR IGNORE INTO action_events
                (event_id, decision_id, ticker, advisory_action,
                 reason_json, sizing_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    decision_id,
                    action["ticker"],
                    action["advisory_action"],
                    json.dumps(action["reason_codes"], sort_keys=True),
                    (json.dumps(private_row, sort_keys=True) if private_row else None),
                    now,
                ),
            )
            inserted_events += int(connection.total_changes > before)
        connection.commit()
        cycle_count = int(connection.execute("SELECT COUNT(*) FROM decision_cycles").fetchone()[0])
        event_count = int(connection.execute("SELECT COUNT(*) FROM action_events").fetchone()[0])
    return {
        "schema": "portfolio_action_lifecycle_audit_v1",
        "status": "GO",
        "decision_id": decision_id,
        "decision_inserted": cycle_inserted,
        "action_events_inserted": inserted_events,
        "decision_cycle_count": cycle_count,
        "action_event_count": event_count,
        "append_only": True,
        "duplicate_decisions_ignored": not cycle_inserted,
        "automatic_state_mutations": 0,
        "automatic_order_submissions": 0,
        "execution_authority": "NONE",
    }


def _capacity_map(project_root: Path) -> dict[str, Decimal]:
    payload = _read_json(project_root / "output/capital/capacity_report.json")
    return {
        str(row.get("symbol", "")).upper(): _decimal(row.get("maximum_order_value_eur"))
        for row in payload.get("instruments", [])
        if row.get("symbol")
    }


def _fx_map(project_root: Path) -> dict[str, Decimal]:
    rates = {"EUR": Decimal("1")}
    path = project_root / "data/fx/fx_daily.parquet"
    if not path.exists():
        return rates
    frame = pd.read_parquet(path)
    required = {"base_currency", "quote_currency", "rate"}
    if not required.issubset(frame.columns):
        return rates
    order = "session_date" if "session_date" in frame else None
    if order:
        frame = frame.sort_values(order)
    for _, row in frame.iterrows():
        if str(row["quote_currency"]).upper() == "EUR":
            rates[str(row["base_currency"]).upper()] = _decimal(row["rate"])
    return rates


def _blocked_private_account(reason: str) -> dict[str, Any]:
    return {
        "schema": "private_portfolio_account_state_v1",
        "status": reason,
        "net_liquidation_eur": "0",
        "available_funds_eur": "0",
        "total_cash_value_eur": "0",
        "snapshot_hash": None,
        "account_authority": "NONE",
        "snapshot_source": None,
        "snapshot_age_seconds": None,
        "fresh_reconciliation": False,
        "execution_eligible": False,
        "equity_observed": False,
        "available_funds_observed": False,
        "raw_account_id_stored": False,
    }


def _blocked_private_sizing(reason: str) -> dict[str, Any]:
    return {
        "schema": "private_whole_share_portfolio_sizing_v1",
        "status": reason,
        "account_snapshot_hash": None,
        "current_gross_exposure_pct": 0.0,
        "target_gross_exposure_pct": 0.0,
        "turnover_pct": 0.0,
        "current_portfolio_heat": 0.0,
        "target_portfolio_heat": 0.0,
        "target_position_count": 0,
        "whole_share_feasible_candidate_count": 0,
        "whole_share_infeasible_candidate_count": 0,
        "netted_security_count": 0,
        "whole_share_violation_count": 0,
        "negative_cash_violation_count": 0,
        "security_netting_status": "DATA_BLOCKED",
        "turnover_gate": "DATA_BLOCKED",
        "blockers": [reason],
        "positions": [],
        "approved_target_quantities": 0,
        "automatic_submission": False,
        "execution_authority": "NONE",
    }


def _load_signals(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical = _read_json(project_root / "output/signals/latest_signals.json")
    rows.extend(
        {
            **row,
            "_portfolio_eligible_source": True,
            "_research_allocation_eligible_source": True,
            "_deployment_eligible_source": False,
            "_evidence_tier": "FROZEN_SHADOW",
        }
        for row in canonical.get("signals", [])
    )
    for name in ("pit_mtf_signals.json", "pit_mtf_research_signals.json"):
        payload = _read_json_or_list(project_root / "output/signals" / name)
        source_rows = payload if isinstance(payload, list) else payload.get("signals", [])
        rows.extend(
            {
                **row,
                "_portfolio_eligible_source": (name == "pit_mtf_signals.json"),
                "_research_allocation_eligible_source": (name == "pit_mtf_signals.json"),
                "_deployment_eligible_source": False,
                "_evidence_tier": (
                    "FROZEN_MTF_CURRENT_PIT"
                    if name == "pit_mtf_signals.json"
                    else "HISTORICALLY_POSITIVE_RESEARCH_OBSERVER"
                ),
            }
            for row in source_rows
        )
    rows.extend(_phase11_13_signals(project_root))
    rows.extend(_phase11_14_signals(project_root))
    rows.extend(_screened_portfolio_signals(project_root))
    unique: dict[str, dict[str, Any]] = {}
    now = datetime.now(UTC)
    market_references: dict[str, dict[str, Any]] = {}
    for row in rows:
        row = normalize_research_signal_price_basis(project_root, row)
        if not isinstance(row.get("strategy_timeframe_contract"), dict):
            row["strategy_timeframe_contract"] = declared_research_signal_timeframe_contract(
                project_root,
                row,
            )
        if not row.get("strategy_family"):
            row["strategy_family"] = str(
                row.get("family") or row.get("entry_strategy") or row.get("formula") or "UNDECLARED"
            )
        symbol = str(row.get("ticker") or row.get("asset") or "").upper()
        if symbol not in market_references:
            market_references[symbol] = latest_market_reference(
                project_root,
                symbol,
                now=now,
            )
        row = apply_market_reference(
            project_root,
            row,
            now=now,
            reference=market_references[symbol],
        )
        freshness = evaluate_signal_freshness(row, now=now)
        if not freshness["is_current"]:
            continue
        row["expiration_timestamp"] = freshness["effective_expiration"].isoformat()
        row["data_freshness"] = "FRESH"
        row["signal_freshness_basis"] = freshness["freshness_basis"]
        row["signal_freshness"] = {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in freshness.items()
            if key != "is_current"
        }
        identity = str(
            row.get("signal_id")
            or stable_hash(
                {
                    "ticker": row.get("ticker"),
                    "strategy_id": row.get("strategy_id"),
                    "data_timestamp": row.get("data_timestamp"),
                }
            )
        )
        unique[identity] = row
    return list(unique.values())


def _screened_portfolio_signals(
    project_root: Path,
) -> list[dict[str, Any]]:
    """Promote current broad-screening rows into research ranking only."""
    records, source = _latest_screener_records(project_root)
    rows: list[dict[str, Any]] = []
    accepted = {"HIGH_POTENTIAL", "WATCHLIST"}
    for item in records:
        classification = str(item.get("classification") or "").upper()
        if classification not in accepted:
            continue
        ticker = str(item.get("symbol") or "").upper()
        price, stop = _latest_price_and_stop(project_root, ticker)
        timestamp = (
            item.get("data_timestamps", {}).get("price_source_timestamp")
            if isinstance(item.get("data_timestamps"), dict)
            else None
        )
        expiration = _runtime_screener_expiration(item)
        if (
            not ticker
            or price <= 0
            or stop <= 0
            or stop >= price
            or not timestamp
            or expiration is None
        ):
            continue
        total_score = _bounded(
            _number(item.get("total_score")) / 100.0,
            default=0.0,
        )
        rows.append(
            {
                "signal_id": "SCREENER-"
                + stable_hash(
                    {
                        "ticker": ticker,
                        "decision_time": item.get("decision_time"),
                        "classification": classification,
                        "source": source,
                    }
                )[:20],
                "ticker": ticker,
                "strategy_id": "BROAD-SCREENER-V1",
                "entry_strategy": "broad_screening_rank",
                "timeframe": "1d",
                "action": "WATCHLIST",
                "confidence_score": round(total_score, 6),
                "data_timestamp": str(timestamp),
                "expiration_timestamp": expiration,
                "preferred_entry": str(price),
                "stop_loss": str(stop),
                "stop_distance_pct": str((price - stop) / price),
                "currency": str(item.get("currency") or "UNKNOWN"),
                "shariah_status": str(item.get("shariah_status") or "SHARIAH_DATA_UNAVAILABLE"),
                "shariah_screened_at": item.get("data_timestamps", {}).get("shariah_screened_at"),
                "shariah_expires_at": item.get("data_timestamps", {}).get("shariah_expires_at"),
                "reasons": [
                    "BROAD_SCREENER_PORTFOLIO_CANDIDATE",
                    f"SCREENER_{classification}",
                    *[str(reason) for reason in item.get("selection_reasons", [])],
                ],
                "_portfolio_eligible_source": True,
                "_research_allocation_eligible_source": True,
                "_deployment_eligible_source": False,
                "_evidence_tier": "BROAD_SCREENING_CURRENT_PIT",
                "_shariah_runtime_verified_source": True,
            }
        )
    return rows


def _runtime_screener_expiration(item: dict[str, Any]) -> str | None:
    try:
        decision = pd.Timestamp(item["decision_time"])
        if decision.tzinfo is None:
            decision = decision.tz_localize("UTC")
        else:
            decision = decision.tz_convert("UTC")
        expiry = decision + pd.Timedelta(days=3)
        shariah_expiry = item.get("data_timestamps", {}).get("shariah_expires_at")
        if shariah_expiry:
            shariah_timestamp = pd.Timestamp(shariah_expiry)
            if shariah_timestamp.tzinfo is None:
                shariah_timestamp = shariah_timestamp.tz_localize("UTC")
            else:
                shariah_timestamp = shariah_timestamp.tz_convert("UTC")
            expiry = min(expiry, shariah_timestamp)
        return expiry.isoformat()
    except (KeyError, TypeError, ValueError):
        return None


def _resolved_shariah_status(
    fundamental: dict[str, Any],
    selected_signals: list[dict[str, Any]],
) -> str:
    verified = {
        str(item.get("shariah_status") or "")
        for item in selected_signals
        if item.get("_shariah_runtime_verified_source") is True
    }
    for eligible in ("SHARIAH_ELIGIBLE_PIT", "SHARIAH_COMPLIANT"):
        if eligible in verified:
            return eligible
    return str(fundamental.get("shariah_status") or "NOT_CONFIGURED")


def _effective_dynamic_override_blockers(
    blockers: list[str],
    *,
    shariah_status: str,
) -> list[str]:
    """Drop only stale blockers contradicted by newer authoritative evidence."""
    shariah_verified = shariah_status in {
        "SHARIAH_ELIGIBLE_PIT",
        "SHARIAH_COMPLIANT",
    }
    return [
        blocker for blocker in blockers if not (shariah_verified and blocker.startswith("SHARIAH_"))
    ]


def _latest_screener_records(
    project_root: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    root = project_root / "output/screener"
    preview_path = root / "latest-preview.json"
    preview = _read_json(preview_path)
    paths = sorted(
        root.glob("????-??-??/screening-results.json"),
        reverse=True,
    )
    canonical_path = paths[0] if paths else None
    canonical = _read_json(canonical_path) if canonical_path else {}
    try:
        current_config_hash = ScreenerConfig.load(project_root).config_hash
    except (FileNotFoundError, KeyError, ValueError):
        current_config_hash = None
    preview_date = str(preview.get("screening_date") or "")
    canonical_date = str(canonical.get("screening_date") or "")
    use_preview = (
        preview.get("status") == "GO"
        and preview.get("canonical_research_evidence") is False
        and preview.get("append_only_history_mutated") is False
        and preview.get("config_hash") == current_config_hash
        and bool(preview.get("records"))
        and preview_date >= canonical_date
    )
    payload = preview if use_preview else canonical
    selected_path = preview_path if use_preview else canonical_path
    if selected_path is None:
        return [], None
    records = [
        row for row in payload.get("records", []) if isinstance(row, dict) and row.get("symbol")
    ]
    return records, str(selected_path.relative_to(project_root))


def build_opportunity_funnel(
    project_root: Path,
    *,
    signals: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Trace broad screening into watch, portfolio and execution sets."""
    settings = policy.get("candidate_funnel", {})
    per_family = int(settings.get("per_family_limit", 20))
    maximum_watch = int(settings.get("maximum_watchlist_candidates", 100))
    maximum_rejections = int(settings.get("maximum_rejection_audit_rows", 50))
    hidden_gem_minimum_market_cap = float(
        settings.get("hidden_gem_minimum_market_cap", 300_000_000)
    )
    hidden_gem_maximum_market_cap = float(
        settings.get("hidden_gem_maximum_market_cap", 10_000_000_000)
    )
    records, screener_source = _latest_screener_records(project_root)
    require_current_classification = bool(
        settings.get("screener_candidates_require_current_classification", True)
    )
    qualified_records = [
        row
        for row in records
        if not require_current_classification
        or str(row.get("classification") or "").upper()
        in {"HIGH_POTENTIAL", "WATCHLIST", "NEUTRAL"}
    ]
    rejected_screen_records = [row for row in records if row not in qualified_records]
    screen_map = {str(row.get("symbol") or "").upper(): row for row in records}
    ranked_map = {str(row["ticker"]): row for row in ranked}
    metadata = policy.get("asset_metadata", {})

    def top_by(key: str, rows: list[dict[str, Any]]) -> list[str]:
        ordered = sorted(
            rows,
            key=lambda row: -_number(row.get(key)),
        )
        return [
            str(row.get("symbol") or "").upper()
            for row in ordered[:per_family]
            if row.get("symbol")
        ]

    def top_per_group(
        key: str,
        rows: list[dict[str, Any]],
        group_key: str,
    ) -> list[str]:
        leaders: list[dict[str, Any]] = []
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group = str(row.get(group_key) or "UNKNOWN").upper()
            groups[group].append(row)
        for group_rows in groups.values():
            leaders.append(max(group_rows, key=lambda row: _number(row.get(key))))
        return top_by(key, leaders)

    def metadata_sector(row: dict[str, Any]) -> str:
        symbol = str(row.get("symbol") or "").upper()
        return str(metadata.get(symbol, {}).get("sector") or row.get("sector") or "").upper()

    def metadata_sleeve(row: dict[str, Any]) -> str:
        symbol = str(row.get("symbol") or "").upper()
        return str(metadata.get(symbol, {}).get("sleeve") or "stock")

    etf_rows = [row for row in qualified_records if str(row.get("asset_type")) != "STOCK"]
    real_asset_rows = [
        row for row in qualified_records if metadata_sleeve(row) == "commodity_security"
    ]
    equity_rows = [row for row in qualified_records if str(row.get("asset_type")) == "STOCK"]
    current_gainers = [row for row in qualified_records if _number(row.get("daily_return")) > 0]
    current_pullbacks = [
        row for row in qualified_records if -0.05 <= _number(row.get("daily_return")) < 0
    ]
    hidden_gem_rows = [
        row
        for row in equity_rows
        if hidden_gem_minimum_market_cap
        <= _number(row.get("market_cap"))
        <= hidden_gem_maximum_market_cap
    ]

    def real_asset_family(*terms: str) -> list[dict[str, Any]]:
        needles = {term.upper() for term in terms}
        return [
            row for row in real_asset_rows if any(term in metadata_sector(row) for term in needles)
        ]

    family_symbols: dict[str, list[str]] = {
        "TREND_LEADERS": top_by("technical_score", qualified_records),
        "QUALITY_FUNDAMENTALS": top_by("fundamental_score", qualified_records),
        "LIQUIDITY_LEADERS": top_by("liquidity_score", qualified_records),
        "RISK_QUALITY": top_by("risk_score", qualified_records),
        "SECTOR_LEADERS": top_per_group("total_score", equity_rows, "sector"),
        "INDUSTRY_LEADERS": top_per_group("total_score", equity_rows, "industry"),
        "HIDDEN_GEMS": top_by("total_score", hidden_gem_rows),
        "CURRENT_GAINERS": top_by("total_score", current_gainers),
        "CURRENT_PULLBACKS": top_by("technical_score", current_pullbacks),
        "ETF_ROTATION": top_by("total_score", etf_rows),
        "REAL_ASSETS": top_by("total_score", real_asset_rows),
        "COPPER_COMPLEX": top_by("total_score", real_asset_family("COPPER")),
        "URANIUM_COMPLEX": top_by("total_score", real_asset_family("URANIUM", "NUCLEAR")),
        "PRECIOUS_METALS": top_by(
            "total_score",
            real_asset_family("GOLD", "SILVER", "PLATINUM", "PALLADIUM"),
        ),
        "ENERGY_COMPLEX": top_by("total_score", real_asset_family("ENERGY", "OIL", "GAS")),
        "DEFENSIVE_ETFS": top_by(
            "total_score",
            [row for row in etf_rows if metadata_sleeve(row) == "bond_defensive"],
        ),
        "REGIONAL_ETFS": top_by(
            "total_score",
            [row for row in etf_rows if metadata_sleeve(row) == "etf_core"],
        ),
    }
    signal_family_symbols: dict[str, list[str]] = defaultdict(list)
    for row in ranked:
        ticker = str(row["ticker"])
        for family in row.get("strategy_families", []):
            family_key = str(family).upper()
            signal_family_symbols[family_key].append(ticker)
    for family, symbols in signal_family_symbols.items():
        family_symbols[f"STRATEGY_{family}"] = symbols[:per_family]

    ordered_symbols: list[str] = []
    for row in ranked:
        ticker = str(row["ticker"])
        if ticker not in ordered_symbols:
            ordered_symbols.append(ticker)
    for symbols in family_symbols.values():
        for ticker in symbols:
            if ticker and ticker not in ordered_symbols:
                ordered_symbols.append(ticker)
    ordered_symbols = ordered_symbols[:maximum_watch]

    minimum_score = float(policy["ranking"]["minimum_allocation_score"])
    candidates: list[dict[str, Any]] = []
    for rank, ticker in enumerate(ordered_symbols, 1):
        opportunity = ranked_map.get(ticker, {})
        screen = screen_map.get(ticker, {})
        score = float(
            opportunity.get("opportunity_score") or (_number(screen.get("total_score")) / 100.0)
        )
        research_blockers = list(opportunity.get("research_allocation_blockers") or [])
        if opportunity.get("deployment_eligible"):
            stage = "EXECUTION_CANDIDATE"
        elif opportunity and not research_blockers and score >= minimum_score:
            stage = "PORTFOLIO_CANDIDATE"
        else:
            stage = "WATCHLIST_CANDIDATE"
        rejection_reasons = sorted(
            {
                *research_blockers,
                *(str(value) for value in screen.get("rejection_reasons", [])),
                *([] if opportunity else ["NOT_IN_CURRENT_STRATEGY_SIGNAL_SET"]),
            }
        )
        candidates.append(
            {
                "rank": rank,
                "symbol": ticker,
                "candidate_stage": stage,
                "asset_type": opportunity.get("asset_type", screen.get("asset_type", "UNKNOWN")),
                "sleeve": opportunity.get(
                    "sleeve",
                    metadata.get(ticker, {}).get("sleeve", "stock"),
                ),
                "sector": opportunity.get("sector", screen.get("sector", "UNKNOWN")),
                "industry": opportunity.get("industry", screen.get("industry", "UNKNOWN")),
                "market_cap": screen.get("market_cap"),
                "fundamental_coverage": screen.get("fundamental_coverage"),
                "opportunity_score": round(score, 6),
                "preliminary_rank_utility": round(score - minimum_score, 6),
                "utility_semantics": ("RANK_MARGIN_NOT_EXPECTED_RETURN"),
                "screener_classification": screen.get("classification"),
                "research_allocation_eligible": bool(
                    opportunity.get("research_allocation_eligible", False)
                ),
                "deployment_eligible": bool(opportunity.get("deployment_eligible", False)),
                "rejection_stage": (
                    None if stage != "WATCHLIST_CANDIDATE" else "PORTFOLIO_QUALIFICATION"
                ),
                "rejection_reasons": rejection_reasons,
                "candidate_sources": sorted(
                    {
                        *(["CURRENT_STRATEGY_RANKING"] if opportunity else []),
                        *(
                            ["CURRENT_SCREENING_CLASSIFICATION"]
                            if screen in qualified_records
                            else []
                        ),
                    }
                ),
                "screening_families": sorted(
                    family for family, symbols in family_symbols.items() if ticker in symbols
                ),
            }
        )
    portfolio_candidates = [
        row
        for row in candidates
        if row["candidate_stage"] in {"PORTFOLIO_CANDIDATE", "EXECUTION_CANDIDATE"}
    ]
    execution_candidates = [
        row for row in candidates if row["candidate_stage"] == "EXECUTION_CANDIDATE"
    ]
    rejection_audit = sorted(
        (row for row in candidates if row["candidate_stage"] == "WATCHLIST_CANDIDATE"),
        key=lambda row: (-float(row["preliminary_rank_utility"]), row["symbol"]),
    )[:maximum_rejections]
    screening_rejection_audit = [
        {
            "symbol": str(row.get("symbol") or "").upper(),
            "candidate_stage": "SCREENING_REJECTED",
            "asset_type": row.get("asset_type", "UNKNOWN"),
            "sector": row.get("sector", "UNKNOWN"),
            "industry": row.get("industry", "UNKNOWN"),
            "market_cap": row.get("market_cap"),
            "fundamental_coverage": row.get("fundamental_coverage"),
            "opportunity_score": round(_number(row.get("total_score")) / 100.0, 6),
            "preliminary_rank_utility": round(
                _number(row.get("total_score")) / 100.0 - minimum_score,
                6,
            ),
            "utility_semantics": "RANK_MARGIN_NOT_EXPECTED_RETURN",
            "rejection_stage": "SCREENING_QUALIFICATION",
            "rejection_reasons": sorted(str(value) for value in row.get("rejection_reasons", [])),
        }
        for row in sorted(
            rejected_screen_records,
            key=lambda row: -_number(row.get("total_score")),
        )[:maximum_rejections]
    ]
    shariah_review_queue = []
    for row in sorted(
        rejected_screen_records,
        key=lambda item: (
            any(
                not str(reason).startswith("SHARIAH_")
                for reason in item.get("rejection_reasons", [])
            ),
            -_number(item.get("total_score")),
        ),
    ):
        rejection_reasons = sorted(str(value) for value in row.get("rejection_reasons", []))
        shariah_reasons = [reason for reason in rejection_reasons if reason.startswith("SHARIAH_")]
        if not shariah_reasons:
            continue
        other_reasons = [
            reason for reason in rejection_reasons if not reason.startswith("SHARIAH_")
        ]
        shariah_review_queue.append(
            {
                "symbol": str(row.get("symbol") or "").upper(),
                "asset_type": row.get("asset_type", "UNKNOWN"),
                "sector": row.get("sector", "UNKNOWN"),
                "industry": row.get("industry", "UNKNOWN"),
                "market_cap": row.get("market_cap"),
                "total_score": _number(row.get("total_score")),
                "shariah_status": row.get("shariah_status"),
                "shariah_reasons": shariah_reasons,
                "other_rejection_reasons": other_reasons,
                "review_status": (
                    "READY_FOR_EXTERNAL_SHARIAH_REVIEW"
                    if not other_reasons
                    else "OTHER_DATA_GATES_MUST_CLEAR_FIRST"
                ),
                "automatic_approval_allowed": False,
            }
        )
        if len(shariah_review_queue) >= per_family:
            break
    universe = _read_json(project_root / "output/analysis/universe-coverage.json")
    report = {
        "schema": "portfolio_opportunity_funnel_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "universe_instrument_count": int(universe.get("universe_instrument_count", 0) or 0),
        "analyzable_instrument_count": int(universe.get("analyzable_instrument_count", 0) or 0),
        "latest_screener_record_count": len(records),
        "qualified_screener_record_count": len(qualified_records),
        "rejected_screener_record_count": len(rejected_screen_records),
        "screener_current_classification_required": (require_current_classification),
        "current_signal_count": len(signals),
        "current_signal_symbol_count": len(
            {
                str(row.get("ticker") or row.get("asset") or "").upper()
                for row in signals
                if row.get("ticker") or row.get("asset")
            }
        ),
        "ranked_opportunity_count": len(ranked),
        "watchlist_candidate_count": len(candidates),
        "portfolio_candidate_count": len(portfolio_candidates),
        "execution_candidate_count": len(execution_candidates),
        "screened_family_counts": {
            family: len(set(symbols)) for family, symbols in sorted(family_symbols.items())
        },
        "screening_family_statuses": {
            "industry_leaders": (
                "UNAVAILABLE_IN_CURRENT_SCREENER_RECORD_SCHEMA"
                if not any(row.get("industry") for row in records)
                else "AVAILABLE"
            ),
            "industry_coverage_count": sum(1 for row in records if row.get("industry")),
            "hidden_gems": (
                "UNAVAILABLE_MARKET_CAP_NOT_IN_CURRENT_SCREENER_RECORD_SCHEMA"
                if not any(row.get("market_cap") for row in records)
                else "AVAILABLE"
            ),
            "market_cap_coverage_count": sum(
                1 for row in records if row.get("market_cap") is not None
            ),
        },
        "watchlist_candidates": candidates,
        "portfolio_candidates": portfolio_candidates,
        "execution_candidates": execution_candidates,
        "top_rejected_before_portfolio": rejection_audit,
        "top_screening_rejections": screening_rejection_audit,
        "shariah_review_queue": shariah_review_queue,
        "shariah_review_queue_count": len(shariah_review_queue),
        "shariah_review_is_advisory_only": True,
        "screener_source": screener_source,
        "ranking_precedes_deployment_qualification": True,
        "cash_is_first_class_asset": True,
        "automatic_execution_allowed": False,
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _phase11_13_signals(
    project_root: Path,
) -> list[dict[str, Any]]:
    return _phase_survivor_signals(
        project_root,
        phase_name="phase11_13",
        signal_prefix="P1113",
        require_forward_candidate=False,
        phase_reason=None,
    )


def _phase11_14_signals(
    project_root: Path,
) -> list[dict[str, Any]]:
    return _phase_survivor_signals(
        project_root,
        phase_name="phase11_14",
        signal_prefix="P1114",
        require_forward_candidate=True,
        phase_reason="PHASE11_14_NESTED_QUALIFIED",
    )


def _phase_survivor_signals(
    project_root: Path,
    *,
    phase_name: str,
    signal_prefix: str,
    require_forward_candidate: bool,
    phase_reason: str | None,
) -> list[dict[str, Any]]:
    root = project_root / f"output/research/{phase_name}"
    status = _read_json(root / "status.json")
    observation = _read_json(root / "latest-forward-observation.json")
    boundary = status.get("qualification_boundary", {})
    if (
        status.get("status") != "GO"
        or boundary.get("status") != "FROZEN"
        or observation.get("status") != "GO"
    ):
        return []
    qualified = {
        str(row["strategy_id"]): row
        for row in status.get("qualification", {}).get(
            "strategies",
            [],
        )
        if row.get("robust_pass")
        and row.get("portfolio_invariants_go")
        and (not require_forward_candidate or row.get("forward_observer_candidate"))
    }
    frozen_ids = {str(value) for value in boundary.get("robust_strategy_ids", [])}
    rows: list[dict[str, Any]] = []
    for item in observation.get("observations", []):
        strategy_id = str(item.get("strategy_id", ""))
        metrics = qualified.get(strategy_id)
        if metrics is None or strategy_id not in frozen_ids:
            continue
        closed = _timestamp(item.get("closed_bar_timestamp"))
        if closed is None:
            continue
        timeframe = str(item.get("timeframe", "1d"))
        declared_expiry = closed + timedelta(days=8 if timeframe == "1w" else 4)
        for ticker, raw_weight in item.get(
            "current_attested_target_weights",
            {},
        ).items():
            weight = _number(raw_weight)
            if weight <= 0:
                continue
            exchange_timezone = _signal_exchange_timezone(
                project_root,
                str(ticker),
                timeframe,
            )
            freshness = evaluate_signal_freshness(
                {
                    "timeframe": timeframe,
                    "data_timestamp": closed,
                    "expiration_timestamp": declared_expiry,
                    "exchange_timezone": exchange_timezone,
                }
            )
            if not freshness["is_current"]:
                continue
            expiry = freshness["effective_expiration"]
            price, stop = _latest_price_and_stop(
                project_root,
                str(ticker),
            )
            if price <= 0 or stop <= 0 or stop >= price:
                continue
            confidence = min(
                0.90,
                0.55
                + max(
                    0.0,
                    _number(metrics.get("combined_period_profit_factor")) - 1.0,
                )
                * 0.35
                + max(
                    0.0,
                    _number(metrics.get("combined_oos_Sharpe")),
                )
                * 0.08,
            )
            rows.append(
                {
                    "signal_id": (
                        f"{signal_prefix}-"
                        + stable_hash(
                            {
                                "strategy_id": strategy_id,
                                "ticker": str(ticker).upper(),
                                "closed_bar_timestamp": closed.isoformat(),
                                "qualification_hash": boundary.get("qualification_hash"),
                            }
                        )[:20]
                    ),
                    "ticker": str(ticker).upper(),
                    "strategy_id": strategy_id,
                    "timeframe": timeframe,
                    "action": "BUY",
                    "confidence_score": round(confidence, 6),
                    "data_timestamp": closed.isoformat(),
                    "data_freshness": ("FRESH" if expiry >= datetime.now(UTC) else "STALE"),
                    "exchange_timezone": exchange_timezone,
                    "signal_freshness_basis": freshness["freshness_basis"],
                    "expiration_timestamp": expiry.isoformat(),
                    "preferred_entry": str(price),
                    "stop_loss": str(stop),
                    "stop_distance_pct": str((price - stop) / price),
                    "reasons": [
                        "ROBUSTNESS_SURVIVOR",
                        "FROZEN_QUALIFICATION_BOUNDARY",
                        "CURRENT_PIT_ATTESTATION_GO",
                        "FORWARD_OBSERVATION_ONLY",
                        *([phase_reason] if phase_reason else []),
                    ],
                    "_portfolio_eligible_source": True,
                    "_research_allocation_eligible_source": True,
                    "_deployment_eligible_source": False,
                    "_evidence_tier": "ROBUSTNESS_SURVIVOR",
                }
            )
    return rows


def _latest_price_and_stop(
    project_root: Path,
    ticker: str,
) -> tuple[float, float]:
    legacy_path = (
        project_root / "data/research/critical_trading/yfinance" / f"{ticker.upper()}.parquet"
    )
    frame = _latest_validated_daily_frame(project_root, ticker)
    if frame.empty and legacy_path.exists():
        frame = pd.read_parquet(legacy_path)
    required = {"close", "high", "low"}
    if frame.empty or not required.issubset(frame.columns):
        return 0.0, 0.0
    timestamp_column = (
        "timestamp_utc"
        if "timestamp_utc" in frame
        else "session_date"
        if "session_date" in frame
        else None
    )
    if timestamp_column is not None:
        frame = frame.sort_values(timestamp_column)
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    previous = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous).abs(),
            (low - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    price = float(close.iloc[-1])
    atr = float(true_range.tail(14).mean())
    stop = max(
        float(low.tail(20).min()),
        price - 2.0 * atr,
    )
    if not math.isfinite(price) or not math.isfinite(stop):
        return 0.0, 0.0
    return price, stop


def _latest_validated_daily_frame(
    project_root: Path,
    ticker: str,
) -> pd.DataFrame:
    root = project_root / "data/research/multitimeframe/private"
    pattern = f"provider=*/symbol={ticker.upper()}/interval=1d/source_interval=1d/bars.parquet"
    candidates: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for path in sorted(root.glob(pattern)):
        try:
            frame = pd.read_parquet(path)
        except (FileNotFoundError, OSError, ValueError):
            continue
        if frame.empty:
            continue
        if "quality_status" in frame:
            frame = frame[frame["quality_status"].astype(str).eq("VALIDATED_OHLC")]
        if "is_partial" in frame:
            frame = frame[~frame["is_partial"].fillna(True).astype(bool)]
        timestamp_column = (
            "timestamp_utc"
            if "timestamp_utc" in frame
            else "session_date"
            if "session_date" in frame
            else None
        )
        required = {"open", "high", "low", "close"}
        if frame.empty or timestamp_column is None or not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        raw_close = pd.to_numeric(frame["close"], errors="coerce")
        adjusted_close = pd.to_numeric(frame.get("adjusted_close", raw_close), errors="coerce")
        adjustment = (
            adjusted_close.div(raw_close).replace([float("inf"), float("-inf")], pd.NA).fillna(1.0)
        )
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce") * adjustment
        frame = frame.dropna(subset=["open", "high", "low", "close"])
        frame = frame.sort_values(timestamp_column)
        if frame.empty:
            continue
        latest = pd.Timestamp(frame[timestamp_column].iloc[-1])
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        else:
            latest = latest.tz_convert("UTC")
        candidates.append((latest, frame))
    if not candidates:
        return pd.DataFrame()
    return max(candidates, key=lambda item: item[0])[1]


def _signal_exchange_timezone(
    project_root: Path,
    ticker: str,
    timeframe: str,
) -> str:
    interval_root = (
        project_root
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / f"symbol={ticker.upper()}"
        / f"interval={timeframe}"
    )
    for path in sorted(interval_root.glob("source_interval=*/bars.parquet")):
        try:
            frame = pd.read_parquet(
                path,
                columns=["exchange_timezone"],
            )
        except (FileNotFoundError, OSError, ValueError):
            continue
        values = frame["exchange_timezone"].dropna().astype(str)
        if not values.empty:
            return values.iloc[-1].strip()
    return ""


def _contract_map(
    project_root: Path,
    asset_metadata: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    path = project_root / "output/ibkr/contracts/stocks.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    contracts = {str(row["symbol"]).upper(): row.to_dict() for _, row in frame.iterrows()}
    for portfolio_symbol, metadata in (asset_metadata or {}).items():
        if not isinstance(metadata, dict):
            continue
        broker_symbol = str(metadata.get("broker_symbol") or "").upper()
        portfolio_symbol = str(portfolio_symbol).upper()
        if not broker_symbol or broker_symbol not in contracts:
            continue
        contracts[portfolio_symbol] = {
            **contracts[broker_symbol],
            "broker_symbol": broker_symbol,
            "portfolio_symbol": portfolio_symbol,
        }
    return contracts


def _fundamental_map(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "output/screener/candidate-history.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if frame.empty or "symbol" not in frame:
        return {}
    order = "decision_time" if "decision_time" in frame else "screening_date"
    latest = frame.sort_values(order).groupby("symbol", sort=False).tail(1)
    return {str(row["symbol"]).upper(): row.to_dict() for _, row in latest.iterrows()}


def _asset_context_map(
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    rows = _read_json(project_root / "output/market_context/asset-context.json").get("contexts", [])
    return {
        str(row["symbol"]).upper(): row
        for row in rows
        if isinstance(row, dict) and row.get("symbol")
    }


def _strategy_family_map(project_root: Path) -> dict[str, str]:
    rows = _read_json(project_root / "output/research/recovered_survivors.json").get(
        "survivors", []
    )
    rows.extend(
        _read_json(project_root / "config/dynamic/strategies_v1.json").get("strategies", [])
    )
    families = {
        str(row["candidate_id"]): str(row.get("family") or row.get("strategy_name") or "unknown")
        for row in rows
        if row.get("candidate_id")
    }
    phase11_14 = project_root / "output/research/phase11_14/strategy-summary.parquet"
    if phase11_14.exists():
        frame = pd.read_parquet(phase11_14)
        if {"strategy_id", "formula"}.issubset(frame.columns):
            families.update(
                {
                    str(row["strategy_id"]): infer_strategy_family(str(row["formula"]))
                    for _, row in frame.iterrows()
                }
            )
    return families


def _dynamic_strategy_weights(project_root: Path) -> dict[str, float]:
    rows = _read_json(project_root / "output/dynamic/strategy_weights.json").get("weights", [])
    return {
        str(row["strategy_id"]): max(0.0, _number(row.get("weight")))
        for row in rows
        if row.get("strategy_id")
    }


def _dynamic_overrides(
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    rows = _read_json(project_root / "output/dynamic/current_signals.json").get("signals", [])
    return {str(row.get("ticker", "")).upper(): row for row in rows if row.get("ticker")}


def _current_positions(
    project_root: Path,
) -> tuple[list[dict[str, Any]], str]:
    snapshot = _latest_portfolio_broker_snapshot(project_root)
    if not snapshot:
        return [], "PRIVATE_BROKER_SNAPSHOT_UNAVAILABLE"
    section = snapshot.get("positions", {})
    if section.get("status") not in {"COMPLETE", "EMPTY_COMPLETE"}:
        return [], "PRIVATE_BROKER_POSITION_SNAPSHOT_INCOMPLETE"
    rows = []
    for item in section.get("positions", []):
        quantity = _number(item.get("position_quantity"))
        if quantity <= 0:
            continue
        rows.append(
            {
                "con_id": int(item["con_id"]),
                "symbol": str(item.get("symbol", "")).upper(),
                "quantity": quantity,
                "average_cost": _number(item.get("average_cost")),
                "currency": str(item.get("currency", "UNKNOWN")),
                "security_type": str(item.get("security_type", "UNKNOWN")),
            }
        )
    return rows, "PRIVATE_BROKER_POSITION_SNAPSHOT_COMPLETE"


def _latest_portfolio_broker_snapshot(
    project_root: Path,
) -> dict[str, Any] | None:
    reconciliation = _read_json(project_root / "output" / "ibkr" / "live" / "reconciliation.json")
    expected_hash = str(reconciliation.get("private_snapshot_hash") or "")
    live_db = (
        project_root / "data" / "execution" / "live" / "private" / "broker_observation.sqlite3"
    )
    if (
        reconciliation.get("status") == "GO"
        and str(reconciliation.get("reconciliation_status", "")).startswith("LIVE_RECONCILED")
        and expected_hash
        and live_db.is_file()
    ):
        try:
            with sqlite3.connect(live_db) as connection:
                row = connection.execute(
                    "SELECT snapshot_hash, payload_json, created_at "
                    "FROM snapshots ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
        except sqlite3.Error:
            row = None
        if row and str(row[0]) == expected_hash:
            try:
                payload = json.loads(str(row[1]))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                return {
                    **payload,
                    "content_hash": expected_hash,
                    "snapshot_completed_at": str(row[2]),
                    "observation_environment": "LIVE_READ_ONLY",
                }
    return load_latest_phase8_private_snapshot(project_root)


def _snapshot_has_required_eur_account_values(
    snapshot: dict[str, Any] | None,
) -> bool:
    if not snapshot:
        return False
    account = snapshot.get("account", {})
    if account.get("status") != "COMPLETE":
        return False
    observed = {
        str(row.get("tag") or "")
        for row in account.get("values", [])
        if str(row.get("currency", "")).upper() == "EUR"
    }
    return {
        "NetLiquidation",
        "AvailableFunds",
        "TotalCashValue",
    }.issubset(observed)


def _latest_complete_research_account_snapshot(
    project_root: Path,
    *,
    now: datetime | None = None,
    max_age: timedelta = RESEARCH_ACCOUNT_SNAPSHOT_MAX_AGE,
) -> dict[str, Any] | None:
    """Load a recent complete account observation for research sizing only.

    The payload is authenticated against its private SQLite hash and must carry
    one consistent account fingerprint. It never becomes position or execution
    authority.
    """
    live_db = (
        project_root / "data" / "execution" / "live" / "private" / "broker_observation.sqlite3"
    )
    if not live_db.is_file():
        return None
    try:
        with sqlite3.connect(live_db) as connection:
            rows = connection.execute(
                "SELECT snapshot_hash, payload_json, created_at "
                "FROM snapshots ORDER BY created_at DESC LIMIT 25"
            ).fetchall()
    except sqlite3.Error:
        return None
    observed_now = (now or datetime.now(UTC)).astimezone(UTC)
    required = {
        "NetLiquidation",
        "AvailableFunds",
        "TotalCashValue",
    }
    allowed = required | {"BuyingPower", "GrossPositionValue"}
    for stored_hash, raw_payload, created_at in rows:
        try:
            payload = json.loads(str(raw_payload))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if str(stored_hash) != stable_hash(payload):
            continue
        completed_at = _timestamp(payload.get("snapshot_completed_at") or created_at)
        if completed_at is None:
            continue
        age = observed_now - completed_at
        if age < -timedelta(minutes=5) or age > max_age:
            continue
        account = payload.get("account", {})
        if account.get("status") != "COMPLETE":
            continue
        values: dict[str, str] = {}
        fingerprints: set[str] = set()
        conflict = False
        for item in account.get("values", []):
            if str(item.get("currency", "")).upper() != "EUR":
                continue
            tag = str(item.get("tag") or "")
            if tag not in allowed:
                continue
            if item.get("source") != "IBKR_ACCOUNT_SUMMARY":
                conflict = True
                break
            fingerprint = str(item.get("account_fingerprint") or "")
            if not fingerprint:
                conflict = True
                break
            fingerprints.add(fingerprint)
            try:
                observed_value = Decimal(str(item.get("value")))
            except (ArithmeticError, ValueError):
                conflict = True
                break
            if not observed_value.is_finite():
                conflict = True
                break
            normalized = str(observed_value)
            if tag in values and values[tag] != normalized:
                conflict = True
                break
            values[tag] = normalized
        if (
            conflict
            or len(fingerprints) != 1
            or not required.issubset(values)
            or _decimal(values["NetLiquidation"]) <= 0
            or any(_decimal(value) < 0 for value in values.values())
        ):
            continue
        return {
            **payload,
            "content_hash": str(stored_hash),
            "snapshot_completed_at": completed_at.isoformat(),
            "observation_environment": "LIVE_RESEARCH_ONLY",
            "account_authority": "RESEARCH_ONLY_LAST_OBSERVED",
            "snapshot_age_seconds": max(0, int(age.total_seconds())),
            "fresh_reconciliation": False,
            "execution_eligible": False,
        }
    return None


def _private_dynamic_risk_inputs(
    project_root: Path,
) -> tuple[
    list[tuple[datetime, Decimal]],
    list[tuple[date, Decimal]],
]:
    equity_history: list[tuple[datetime, Decimal]] = []
    latest = _latest_portfolio_broker_snapshot(project_root)
    live_db = (
        project_root / "data" / "execution" / "live" / "private" / "broker_observation.sqlite3"
    )
    if latest and latest.get("observation_environment") == "LIVE_READ_ONLY" and live_db.is_file():
        try:
            with sqlite3.connect(live_db) as connection:
                rows = connection.execute(
                    "SELECT created_at, payload_json FROM snapshots ORDER BY created_at"
                ).fetchall()
        except sqlite3.Error:
            rows = []
        for created_at, raw_payload in rows:
            timestamp = _timestamp(created_at)
            if timestamp is None:
                continue
            try:
                payload = json.loads(str(raw_payload))
            except json.JSONDecodeError:
                continue
            values = payload.get("account", {}).get("values", [])
            net_liquidation = next(
                (
                    _decimal(item.get("value"))
                    for item in values
                    if item.get("tag") == "NetLiquidation"
                    and str(item.get("currency", "")).upper() == "EUR"
                ),
                Decimal("0"),
            )
            if net_liquidation > 0:
                equity_history.append((timestamp, net_liquidation))

    daily_pnl: list[tuple[date, Decimal]] = []
    history_path = project_root / "data" / "performance" / "private" / "daily-pnl.jsonl"
    if history_path.is_file():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                session_date = date.fromisoformat(str(row["session_date"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            net_pnl = _decimal(row.get("net_pnl_eur"))
            daily_pnl.append((session_date, net_pnl))
    return equity_history, daily_pnl


def _build_position_management(
    project_root: Path,
    *,
    positions: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    technical_regime: dict[str, Any],
) -> dict[str, Any]:
    database = project_root / PRIVATE_ROOT / "position_management.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    ranking = {str(row["ticker"]): row for row in ranked}
    previous = _latest_position_management_states(database)
    private_decisions: list[dict[str, Any]] = []
    public_rows: list[dict[str, Any]] = []
    for position in positions:
        ticker = str(position["symbol"]).upper()
        identity = stable_hash({"con_id": position["con_id"], "symbol": ticker})[:24]
        market = _latest_position_market_state(project_root, ticker)
        candidate = ranking.get(ticker, {})
        prior = previous.get(identity, {})
        entry = _number(position.get("average_cost"))
        current = _number(market.get("current_price"))
        atr = _number(market.get("atr"))
        candidate_stop = _number(candidate.get("stop_loss"))
        initial_stop = _number(prior.get("initial_stop"))
        if initial_stop <= 0 and 0 < candidate_stop < entry:
            initial_stop = candidate_stop
        if initial_stop <= 0 and entry > 0 and atr > 0:
            initial_stop = max(0.01, entry - 2.0 * atr)
        previous_stop = max(
            initial_stop,
            _number(prior.get("proposed_stop")),
        )
        peak = max(
            entry,
            current,
            _number(prior.get("peak_price")),
        )
        trend_strength = _clip_number(
            _number(candidate.get("opportunity_score")),
        )
        volatility_regime = 0.35 if technical_regime.get("status") == "GO" else 0.75
        decision = evaluate_long_position(
            entry_price=entry,
            current_price=current,
            initial_stop=initial_stop,
            previous_stop=previous_stop,
            peak_price=peak,
            atr=atr,
            structural_stop=(candidate_stop if candidate_stop > 0 else None),
            trend_strength=trend_strength,
            volatility_regime=volatility_regime,
            first_target_taken=bool(prior.get("first_target_taken", False)),
            second_target_taken=bool(prior.get("second_target_taken", False)),
        )
        private_row = {
            **decision,
            "ticker": ticker,
            "position_identity": identity,
            "con_id": int(position["con_id"]),
            "quantity": position["quantity"],
            "entry_price": entry,
            "current_price": current,
            "initial_stop": initial_stop,
            "market_data_timestamp": market.get("timestamp"),
            "market_data_source": market.get("source"),
        }
        private_decisions.append(private_row)
        public_rows.append(
            {
                "ticker": ticker,
                "position_identity": identity,
                "status": decision["status"],
                "advisory_action": decision["action"],
                "reason_codes": decision["reason_codes"],
                "current_r": decision.get("current_r"),
                "peak_r": decision.get("peak_r"),
                "profit_giveback": decision.get("profit_giveback"),
                "trailing_active": decision.get("trailing_active", False),
                "market_data_timestamp": market.get("timestamp"),
                "market_data_status": market.get("status"),
                "market_data_reason": market.get("reason"),
                "market_data_age_minutes": market.get("age_minutes"),
                "market_data_source": market.get("source"),
                "atr_timestamp": market.get("atr_timestamp"),
                "atr_source": market.get("atr_source"),
                "price_reference_kind": market.get("price_reference_kind"),
                "financial_values_public": False,
                "automatic_execution_allowed": False,
                "execution_authority": "NONE",
            }
        )
    storage_status = _append_position_management_events(database, private_decisions)
    public = {
        "schema": "position_management_public_audit_v1",
        "status": storage_status,
        "position_count": len(public_rows),
        "positions": public_rows,
        "private_store": str(database.relative_to(project_root)),
        "append_only": True,
        "financial_values_public": False,
        "position_quantities_public": False,
        "automatic_execution_allowed": False,
        "broker_write_calls": 0,
        "execution_authority": "NONE",
    }
    public["content_hash"] = stable_hash(public)
    return {"private_decisions": private_decisions, "public_audit": public}


def _latest_position_management_states(
    database: Path,
) -> dict[str, dict[str, Any]]:
    if not database.is_file():
        return {}
    try:
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT position_identity, payload_json FROM position_events "
                "ORDER BY observed_at, rowid"
            ).fetchall()
    except sqlite3.Error:
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for identity, raw in rows:
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            latest[str(identity)] = payload
    return latest


def _append_position_management_events(
    database: Path,
    decisions: list[dict[str, Any]],
) -> str:
    try:
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS position_events("
                "event_hash TEXT PRIMARY KEY, position_identity TEXT NOT NULL, "
                "observed_at TEXT NOT NULL, payload_json TEXT NOT NULL)"
            )
            observed_at = datetime.now(UTC).isoformat()
            for decision in decisions:
                event_hash = stable_hash(decision)
                connection.execute(
                    "INSERT OR IGNORE INTO position_events VALUES(?,?,?,?)",
                    (
                        event_hash,
                        decision["position_identity"],
                        observed_at,
                        json.dumps(decision, sort_keys=True),
                    ),
                )
    except sqlite3.Error:
        return "PRIVATE_STORE_BLOCKED"
    return "GO"


def _latest_position_market_state(
    project_root: Path,
    ticker: str,
) -> dict[str, Any]:
    reference = latest_market_reference(project_root, ticker)
    if reference.get("status") != "FRESH":
        return {
            "status": "DATA_BLOCKED",
            "reason": (reference.get("reason") or "CURRENT_POSITION_PRICE_REFERENCE_STALE"),
            "age_minutes": reference.get("age_minutes"),
            "timestamp": reference.get("timestamp"),
            "source": reference.get("provider"),
            "price_reference_kind": "INDICATIVE_INTRADAY_BAR_CLOSE",
        }
    atr_state = _latest_daily_atr_state(project_root, ticker)
    if atr_state.get("status") != "GO":
        return {
            "status": "DATA_BLOCKED",
            "reason": atr_state.get("reason", "DAILY_ATR_UNAVAILABLE"),
            "age_minutes": reference.get("age_minutes"),
            "timestamp": reference.get("timestamp"),
            "source": reference.get("provider"),
            "price_reference_kind": "INDICATIVE_INTRADAY_BAR_CLOSE",
            "atr_timestamp": atr_state.get("timestamp"),
            "atr_source": atr_state.get("source"),
        }
    price = _number(reference.get("price"))
    if price <= 0:
        return {
            "status": "DATA_BLOCKED",
            "reason": "CURRENT_POSITION_PRICE_INVALID",
        }
    return {
        "status": "GO",
        "current_price": price,
        "atr": atr_state["atr"],
        "timestamp": reference.get("timestamp"),
        "age_minutes": reference.get("age_minutes"),
        "source": reference.get("provider"),
        "price_reference_kind": "INDICATIVE_INTRADAY_BAR_CLOSE",
        "executable_quote": False,
        "atr_timestamp": atr_state.get("timestamp"),
        "atr_source": atr_state.get("source"),
    }


def _latest_daily_atr_state(
    project_root: Path,
    ticker: str,
) -> dict[str, Any]:
    candidates = list(
        (project_root / "data/research/multitimeframe/private").glob(
            f"provider=*/symbol={ticker}/interval=1d/source_interval=*/bars.parquet"
        )
    )
    selected: tuple[pd.Timestamp, Path, pd.DataFrame] | None = None
    for candidate in candidates:
        try:
            frame = pd.read_parquet(candidate)
        except (OSError, ValueError):
            continue
        if "is_partial" in frame:
            frame = frame.loc[~frame["is_partial"].fillna(False).astype(bool)]
        if frame.empty or not {
            "timestamp_utc",
            "close",
            "high",
            "low",
        }.issubset(frame.columns):
            continue
        timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
        valid = timestamps.notna()
        frame = frame.loc[valid].copy()
        timestamps = timestamps.loc[valid]
        if len(frame) < 15:
            continue
        latest = timestamps.max()
        if selected is None or latest > selected[0]:
            frame["_atr_timestamp"] = timestamps
            selected = (latest, candidate, frame)
    if selected is not None:
        latest, path, frame = selected
        provider = next(
            (part.split("=", 1)[1] for part in path.parts if part.startswith("provider=")),
            "LOCAL_MULTITIMEFRAME_CACHE",
        )
        return _daily_atr_from_frame(
            frame,
            timestamp_column="_atr_timestamp",
            source=provider,
            latest=latest,
        )
    path = project_root / "data/research/critical_trading/yfinance" / f"{ticker}.parquet"
    if not path.is_file():
        return {"status": "DATA_BLOCKED", "reason": "DAILY_ATR_UNAVAILABLE"}
    frame = pd.read_parquet(path)
    required = {"close", "high", "low"}
    if frame.empty or not required.issubset(frame.columns):
        return {"status": "DATA_BLOCKED", "reason": "DAILY_ATR_UNAVAILABLE"}
    sort_column = "session_date" if "session_date" in frame else None
    if sort_column:
        frame = frame.sort_values(sort_column)
        latest = pd.Timestamp(frame.iloc[-1][sort_column])
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
    else:
        return {"status": "DATA_BLOCKED", "reason": "DAILY_ATR_TIMESTAMP_MISSING"}
    return _daily_atr_from_frame(
        frame,
        timestamp_column=sort_column,
        source="LEGACY_LOCAL_YFINANCE_DAILY_ATR",
        latest=latest,
    )


def _daily_atr_from_frame(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    source: str,
    latest: pd.Timestamp,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    atr = float(true_range.tail(14).mean())
    if not math.isfinite(atr) or atr <= 0:
        return {"status": "DATA_BLOCKED", "reason": "DAILY_ATR_INVALID"}
    latest = latest.tz_convert("UTC")
    age_hours = max(
        0.0,
        (datetime.now(UTC) - latest.to_pydatetime()).total_seconds() / 3600.0,
    )
    if age_hours > 240.0:
        return {
            "status": "DATA_BLOCKED",
            "reason": "DAILY_ATR_SOURCE_STALE",
            "timestamp": latest.isoformat(),
            "source": source,
        }
    return {
        "status": "GO",
        "atr": atr,
        "timestamp": latest.isoformat(),
        "source": source,
        "timestamp_column": timestamp_column,
    }


def _clip_number(value: float) -> float:
    return min(1.0, max(0.0, value))


def _correlation_matrix(
    project_root: Path,
    ranked: list[dict[str, Any]],
    policy: dict[str, Any],
) -> pd.DataFrame:
    window = int(policy["portfolio"]["correlation_window_sessions"])
    series: dict[str, pd.Series] = {}
    for row in ranked[:40]:
        ticker = row["ticker"]
        path = project_root / "data/research/critical_trading/yfinance" / f"{ticker}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if not {"session_date", "close"}.issubset(frame.columns):
            continue
        values = (
            frame.sort_values("session_date")
            .set_index("session_date")["close"]
            .astype(float)
            .pct_change()
            .tail(window)
            .rename(ticker)
        )
        if values.notna().sum() >= min(60, window):
            series[ticker] = values
    if not series:
        return pd.DataFrame()
    returns = pd.concat(series.values(), axis=1)
    matrix = returns.corr(min_periods=min(40, window))
    vol = returns.std(ddof=0) * math.sqrt(252)
    matrix.attrs["annualized_volatility"] = vol.to_dict()
    return matrix


def _metadata(
    ticker: str,
    fundamental: dict[str, Any],
    contracts: dict[str, dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    configured = policy["asset_metadata"].get(ticker, {})
    producer = commodity_producer_metadata(ticker)
    contract = contracts.get(ticker, {})
    asset_type = str(configured.get("asset_type") or "STOCK")
    sleeve = str(configured.get("sleeve") or ("stock" if asset_type == "STOCK" else "etf_core"))
    sector = str(
        configured.get("sector")
        or fundamental.get("sector")
        or contract.get("industry")
        or "UNKNOWN"
    )
    currency = str(contract.get("currency") or "").upper()
    exchange = str(contract.get("primary_exchange") or "").upper()
    region = str(
        configured.get("region")
        or ("EUROPE" if currency == "EUR" else "")
        or ("UNITED_STATES" if exchange in {"NASDAQ", "NYSE", "ARCA", "AMEX", "BATS"} else "")
        or "UNKNOWN"
    )
    return {
        "asset_type": asset_type,
        "sleeve": sleeve,
        "sector": sector,
        "region": region,
        "currency": str(configured.get("currency") or "").upper(),
        "product_structure": str(
            configured.get("product_structure")
            or producer.get("product_structure")
            or "UNCLASSIFIED"
        ),
        "commodity_exposure_type": str(
            configured.get("commodity_exposure_type")
            or producer.get("commodity_exposure_type")
            or "NONE"
        ),
        "underlying_commodity": str(
            configured.get("underlying_commodity") or producer.get("underlying_commodity") or "NONE"
        ),
        "product_identity_status": str(
            configured.get("product_identity_status") or "UNVERIFIED_PHYSICAL_STRUCTURE"
        ),
        "physical_structure_verified": bool(configured.get("physical_structure_verified")),
        "product_identity_screened_at": configured.get("product_identity_screened_at"),
        "product_identity_expires_at": configured.get("product_identity_expires_at"),
        "product_identity_source_count": int(configured.get("product_identity_source_count") or 0),
        "shariah_product_status": str(
            configured.get("shariah_product_status") or "ATTESTATION_REQUIRED"
        ),
    }


def _regime_fit(
    metadata: dict[str, str],
    *,
    macro: dict[str, Any],
    technical_regime: dict[str, Any],
) -> float:
    technical = str(technical_regime.get("regime", "UNKNOWN"))
    macro_regime = macro.get("regime", {})
    market = str(macro_regime.get("market_regime", "UNKNOWN"))
    commodity = str(macro_regime.get("commodity_regime", "UNKNOWN"))
    if metadata["sleeve"] == "commodity_security":
        return {
            "STRENGTHENING": 0.9,
            "NEUTRAL": 0.6,
            "WEAKENING": 0.35,
        }.get(commodity, 0.5)
    if metadata["sleeve"] == "bond_defensive":
        return 0.8 if market == "RISK_OFF" else 0.55
    base = {
        "BULL_TREND_LOW_VOL": 0.95,
        "BULL_TREND_HIGH_VOL": 0.75,
        "RECOVERY": 0.8,
        "SIDEWAYS_LOW_VOL": 0.6,
        "SIDEWAYS_HIGH_VOL": 0.4,
        "BEAR_TREND": 0.25,
        "CRISIS": 0.1,
    }.get(technical, 0.45)
    return min(base, 0.55) if market == "RISK_OFF" else base


def _macro_confluence_layer(
    asset_context: dict[str, Any],
    *,
    generic_regime_fit: float,
) -> dict[str, Any]:
    component = asset_context.get("components", {}).get("macro", {})
    raw_score = _optional_number(component.get("score"))
    confidence = _bounded(component.get("confidence"), default=0.0)
    if (
        str(component.get("status", "")).upper() == "AVAILABLE"
        and raw_score is not None
        and confidence > 0
    ):
        return {
            "status": "AVAILABLE",
            "score": _bounded(0.5 + 0.5 * raw_score),
            "confidence": confidence,
            "source": "ASSET_SPECIFIC_MACRO_TRANSMISSION",
        }
    return {
        "status": "GENERIC_REGIME_FALLBACK",
        "score": _bounded(generic_regime_fit),
        "confidence": 0.35,
        "source": "GENERIC_TECHNICAL_AND_MACRO_REGIME_FALLBACK",
    }


def _confluence_audit(
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = defaultdict(int)
    rows = []
    for row in ranked:
        confluence = row.get("multilayer_confluence", {})
        status = str(confluence.get("status", "MISSING"))
        status_counts[status] += 1
        rows.append(
            {
                "ticker": row["ticker"],
                "status": status,
                "confluence_score": confluence.get("confluence_score"),
                "ranking_multiplier": confluence.get("ranking_multiplier"),
                "base_score_without_confluence": row.get("base_score_without_confluence"),
                "final_opportunity_score": row.get("opportunity_score"),
                "score_delta": row.get("confluence_adjustment"),
                "layer_statuses": {
                    name: layer.get("status")
                    for name, layer in confluence.get("layers", {}).items()
                },
                "allocation_allowed": confluence.get("allocation_allowed", False),
                "allocation_blockers": confluence.get("allocation_blockers", []),
                "macro_source": confluence.get("macro_source"),
            }
        )
    return {
        "schema": "technical_fundamental_macro_confluence_audit_v1",
        "status": "GO" if rows else "NO_CURRENT_OPPORTUNITIES",
        "opportunity_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "ablation_method": ("BASE_OPPORTUNITY_SCORE_VERSUS_MULTILAYER_ADJUSTED_SCORE"),
        "rows": rows,
        "technical_signal_required": True,
        "fundamental_or_macro_standalone_entry_allowed": False,
        "automatic_execution": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }


def _news_map(news: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in news.get("important_news", []):
        for symbol in row.get("symbols", []):
            result[str(symbol).upper()].append(row)
    return result


def _news_event_overlay_map(
    payload: dict[str, Any],
    *,
    policy: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    overlay_policy = policy.get("news_event_overlay", {})
    if not bool(overlay_policy.get("enabled", False)):
        return {}
    maximum = float(overlay_policy.get("maximum_absolute_ranking_adjustment", 0.04))
    per_event = float(overlay_policy.get("maximum_individual_event_adjustment", 0.015))
    minimum_materiality = float(overlay_policy.get("minimum_materiality", 0.22))
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "event_count": 0,
            "raw_adjustment": 0.0,
            "hard_risk_flags": set(),
            "event_classes": set(),
            "story_cluster_ids": set(),
        }
    )
    for row in payload.get("rows", []):
        materiality = _bounded(row.get("materiality"), default=0.0)
        if materiality < minimum_materiality:
            continue
        contribution = min(
            per_event,
            max(
                -per_event,
                _number(row.get("raw_impact")) * materiality,
            ),
        )
        for symbol in row.get("symbols", []):
            ticker = str(symbol).upper()
            if not ticker:
                continue
            state = grouped[ticker]
            state["event_count"] += 1
            state["raw_adjustment"] += contribution
            state["hard_risk_flags"].update(str(flag) for flag in row.get("hard_risk_flags", []))
            state["event_classes"].update(str(value) for value in row.get("event_classes", []))
            if row.get("story_cluster_id"):
                state["story_cluster_ids"].add(str(row["story_cluster_id"]))
    return {
        ticker: {
            "schema": "portfolio_news_event_overlay_v1",
            "status": "AVAILABLE_CONTEXT_ONLY",
            "event_count": state["event_count"],
            "ranking_adjustment": round(
                min(max(state["raw_adjustment"], -maximum), maximum),
                8,
            ),
            "hard_risk_flags": sorted(state["hard_risk_flags"]),
            "event_classes": sorted(state["event_classes"]),
            "story_cluster_count": len(state["story_cluster_ids"]),
            "maximum_absolute_adjustment": maximum,
            "financial_validation": overlay_policy.get(
                "historical_financial_validation",
                "PENDING_CAUSAL_CAR_ABLATION",
            ),
            "standalone_entry_allowed": False,
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
        }
        for ticker, state in grouped.items()
    }


def _news_overlay_audit(
    ranked: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for row in ranked:
        context = row.get("news_event_context", {})
        adjustment = float(row.get("news_score_adjustment") or 0.0)
        if not context.get("event_count"):
            continue
        rows.append(
            {
                "ticker": row["ticker"],
                "event_count": int(context.get("event_count", 0)),
                "story_cluster_count": int(context.get("story_cluster_count", 0)),
                "score_without_news": row.get("opportunity_score_before_news"),
                "score_with_news": row.get("opportunity_score"),
                "score_delta": adjustment,
                "hard_risk_flags": context.get("hard_risk_flags", []),
                "standalone_entry_allowed": False,
                "execution_authority": "NONE",
            }
        )
    return {
        "schema": "portfolio_news_overlay_ablation_v1",
        "status": ("GO" if source.get("status") == "GO" else "DATA_UNAVAILABLE"),
        "source_status": source.get("status", "UNAVAILABLE"),
        "source_event_count": int(source.get("portfolio_impact_event_count", 0)),
        "contextualized_opportunity_count": len(rows),
        "adjusted_opportunity_count": sum(abs(float(row["score_delta"])) > 0 for row in rows),
        "news_enhanced_opportunity_count": sum(float(row["score_delta"]) > 0 for row in rows),
        "news_penalized_opportunity_count": sum(float(row["score_delta"]) < 0 for row in rows),
        "hard_risk_review_count": sum(bool(row["hard_risk_flags"]) for row in rows),
        "rows": rows,
        "historical_financial_validation": "PENDING_CAUSAL_CAR_ABLATION",
        "forward_validation": "PENDING_CLOSED_FORWARD_EPISODES",
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_writes": 0,
    }


def _event_risk(
    ticker: str,
    news_map: dict[str, list[dict[str, Any]]],
    news: dict[str, Any],
) -> tuple[float, bool]:
    rows = news_map.get(ticker, [])
    risk = 0.15 if news.get("event_risk_within_24h") else 0.0
    negative_high = False
    for row in rows:
        importance = str(row.get("importance", "LOW"))
        direction = str(row.get("direction", "MIXED_OR_UNCLEAR"))
        if importance == "HIGH":
            risk = max(risk, 0.8)
            negative_high = "NEGATIVE" in direction
        elif importance == "MEDIUM":
            risk = max(risk, 0.4)
    return risk, negative_high


def _infer_family(row: dict[str, Any]) -> str:
    entry_strategy = str(row.get("entry_strategy") or "").strip()
    if entry_strategy:
        return entry_strategy
    text = " ".join(
        [
            str(row.get("strategy_id", "")),
            str(row.get("strategy_version", "")),
            *[str(item) for item in row.get("reasons", [])],
        ]
    ).upper()
    families = (
        ("VWAP", "vwap_reclaim"),
        ("RSI", "rsi_pullback"),
        ("BOLLINGER", "bollinger"),
        ("DONCHIAN", "donchian_breakout"),
        ("CROSS_SECTIONAL_MOMENTUM", "cross_sectional_momentum"),
        ("INVERSE_VOL_TREND", "inverse_volatility_trend"),
        ("BREAKOUT", "breakout"),
        ("PULLBACK", "trend_pullback"),
        ("MEAN", "mean_reversion"),
        ("MOMENTUM", "momentum"),
        ("RELATIVE", "relative_strength"),
        ("QUALITY", "quality_momentum"),
        ("MTF", "multi_timeframe_confirmation"),
        ("MA_", "moving_average"),
    )
    return next(
        (family for token, family in families if token in text),
        "other_technical",
    )


def _management_policy(families: list[str]) -> dict[str, str]:
    text = " ".join(families).lower()
    if any(token in text for token in ("mean_reversion", "reversion", "oscillator")):
        return {
            "stop_method": "STRUCTURAL_OR_VOLATILITY_INVALIDATION",
            "exit_policy": "MEAN_REVERSION_TARGET_OR_SHORT_TIME_STOP",
            "expected_holding_period": "2-10 lower-timeframe bars",
        }
    if any(
        token in text
        for token in (
            "breakout",
            "donchian",
            "bollinger",
            "keltner",
            "contraction",
        )
    ):
        return {
            "stop_method": "BREAKOUT_FAILURE_WITH_ATR_BUFFER",
            "exit_policy": "CHANNEL_FAILURE_OR_ATR_TRAIL_OR_TIME_STOP",
            "expected_holding_period": "2-30 lower-timeframe bars",
        }
    return {
        "stop_method": "STRUCTURAL_ATR_OR_TREND_INVALIDATION",
        "exit_policy": "TREND_INVALIDATION_OR_CHANDELIER_OR_RANK_EXIT",
        "expected_holding_period": "5-60 sessions",
    }


def _timeframe_score(timeframes: list[str]) -> float:
    return float(active_swing_timeframe_context(timeframes)["score"])


def _reason_score(
    reasons: list[str],
    tokens: tuple[str, ...],
    *,
    fallback: float,
) -> float:
    matches = sum(any(token in reason.upper() for token in tokens) for reason in reasons)
    return min(1.0, max(fallback, matches / 3.0))


def _stop_risk(row: dict[str, Any]) -> float:
    direct = _number(row.get("stop_distance_pct"))
    if 0 < direct < 1:
        return direct
    entry = _number(row.get("preferred_entry"))
    stop = _number(row.get("stop_loss"))
    return abs(entry - stop) / entry if entry > 0 and stop > 0 else 0.1


def _price_at_or_below_stop(candidate: dict[str, Any]) -> bool:
    price = _number(candidate.get("preferred_entry"))
    stop = _number(candidate.get("stop_loss"))
    return price > 0 and stop > 0 and price <= stop


def _price_at_or_above_tp1(
    candidate: dict[str, Any],
    position: dict[str, Any],
) -> bool:
    price = _number(candidate.get("preferred_entry"))
    target = _number(candidate.get("take_profit_1"))
    average = _number(position.get("average_cost"))
    return price > 0 and target > 0 and average > 0 and price >= target > average


def _max_selected_correlation(
    ticker: str,
    allocations: list[dict[str, Any]],
    correlation: pd.DataFrame,
) -> float:
    if correlation.empty or ticker not in correlation.index:
        return float("nan")
    values = [
        float(correlation.loc[ticker, row["ticker"]])
        for row in allocations
        if row["ticker"] in correlation.columns
        and math.isfinite(float(correlation.loc[ticker, row["ticker"]]))
    ]
    return max(values, default=0.0)


def _volatility_from_matrix(ticker: str, correlation: pd.DataFrame) -> float:
    return float(correlation.attrs.get("annualized_volatility", {}).get(ticker, 0.25))


def _portfolio_volatility(
    allocations: list[dict[str, Any]],
    correlation: pd.DataFrame,
) -> float | None:
    if not allocations or correlation.empty:
        return None
    tickers = [row["ticker"] for row in allocations if row["ticker"] in correlation.index]
    if not tickers:
        return None
    weights = np.array(
        [
            next(float(row["target_weight"]) for row in allocations if row["ticker"] == ticker)
            for ticker in tickers
        ]
    )
    vols = np.array([_volatility_from_matrix(ticker, correlation) for ticker in tickers])
    corr = correlation.loc[tickers, tickers].fillna(0).to_numpy()
    covariance = np.outer(vols, vols) * corr
    variance = float(weights.T @ covariance @ weights)
    return round(math.sqrt(max(variance, 0.0)), 8)


def _capital_level(project_root: Path) -> int:
    report = _read_json(project_root / "output/capital/current_level.json")
    return int(report.get("CURRENT_CAPITAL_LEVEL", 0) or 0)


def _target_throttle(daily_target: dict[str, Any]) -> bool:
    return bool(daily_target.get("target_reached") and daily_target.get("enforcement_active"))


def _research_blockers(row: dict[str, Any]) -> list[str]:
    return list(
        row.get(
            "research_allocation_blockers",
            row.get("execution_blockers", []),
        )
    )


def _position_blockers(row: dict[str, Any]) -> list[str]:
    return list(
        row.get(
            "position_management_blockers",
            row.get("execution_blockers", []),
        )
    )


def _public_opportunity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "ticker",
            "asset_type",
            "sleeve",
            "sector",
            "region",
            "opportunity_score",
            "base_score_without_confluence",
            "confluence_adjustment",
            "opportunity_score_before_news",
            "news_score_adjustment",
            "news_event_context",
            "multilayer_confluence",
            "components",
            "penalties",
            "strategy_count",
            "strategy_ids",
            "strategy_families",
            "timeframes",
            "opportunity_class",
            "active_swing_timeframe_context",
            "real_asset_context",
            "strategy_allocation",
            "market_context",
            "reason_count",
            "data_timestamp",
            "contract_resolved",
            "signal_contract_currency_status",
            "signal_currency",
            "contract_currency",
            "shariah_status",
            "event_risk",
            "research_allocation_eligible",
            "deployment_eligible",
            "evidence_tiers",
            "research_allocation_blockers",
            "position_management_blockers",
            "execution_blockers",
            "deployment_blockers",
            "stop_risk_pct",
            "currency",
            "entry_method",
            "stop_method",
            "exit_policy",
            "expected_holding_period",
        )
    }


def _public_action(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "ticker",
            "position_identity",
            "advisory_action",
            "executable_action",
            "reason_codes",
            "opportunity_score",
            "replacement_ticker",
            "risk_increasing",
            "automatic_execution_allowed",
            "execution_authority",
        )
    }


def _publish(
    project_root: Path,
    report: dict[str, Any],
    *,
    actions: list[dict[str, Any]],
    private_sizing: dict[str, Any],
    private_account: dict[str, Any],
    correlation: pd.DataFrame,
) -> None:
    output = project_root / PUBLIC_ROOT
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "status.json": report["engine_status"],
        "opportunity_ranking.json": report["opportunities"],
        "current_allocation.json": report["current_allocation"],
        "target_allocation.json": report["target_allocation"],
        "risk_contributions.json": report["risk"],
        "dynamic-risk-state.json": report["dynamic_risk"],
        "position-management.json": report["position_management"],
        "confluence-audit.json": report["confluence"],
        "news-overlay-ablation.json": report["news_overlay"],
        "exposure_report.json": report["exposures"],
        "rebalance_plan.json": report["rebalance"],
        "position_actions.json": report["position_actions"],
        "capital-decisions.json": report["capital_decisions"],
        "portfolio-state.json": report["portfolio_state"],
        "opportunity-funnel.json": report["opportunity_funnel"],
        "coverage-waterfall.json": report["coverage_waterfall"],
        "vectorized-stage0.json": report["vectorized_stage0"],
        "cross-asset-intelligence.json": report["cross_asset_intelligence"],
        "performance-attribution.json": report["performance_attribution"],
        "normalized-opportunities.json": report["normalized_opportunities"],
        "overlap-report.json": report["overlap"],
        "desired-portfolio-targets.json": report["desired_targets"],
        "sizing-audit.json": report["sizing_audit"],
        "lifecycle-audit.json": report["lifecycle_audit"],
        "active_portfolio_plan.json": report,
        "rl-portfolio-rotation.json": report["rl_portfolio_rotation"],
    }
    for name, payload in artifacts.items():
        _write_json(output / name, payload)
    if not correlation.empty:
        correlation.rename_axis("ticker").reset_index().to_parquet(
            output / "correlation_matrix.parquet", index=False
        )
    private = project_root / PRIVATE_ROOT
    private.mkdir(parents=True, exist_ok=True)
    _write_json(
        private / "latest-action-plan.json",
        {
            "schema": "private_active_portfolio_action_plan_v1",
            "generated_at": report["engine_status"]["generated_at"],
            "actions": actions,
            "whole_share_sizing": private_sizing,
            "execution_authority": "NONE",
            "automatic_submission": False,
        },
    )
    _write_json(
        private / "current-state.json",
        {
            "schema": "private_proactive_portfolio_state_v1",
            "generated_at": report["engine_status"]["generated_at"],
            "account_state": private_account,
            "whole_share_sizing": private_sizing,
            "raw_account_identifiers_stored": False,
            "execution_authority": "NONE",
            "automatic_submission": False,
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_json_or_list(path: Path) -> dict[str, Any] | list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, (dict, list)) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _optional_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ArithmeticError, ValueError):
        return Decimal("0")
    return result if result.is_finite() else Decimal("0")


def _whole_units(value: Decimal) -> Decimal:
    if value <= 0:
        return Decimal("0")
    return value.to_integral_value(rounding=ROUND_DOWN)


def _canary_asset_class(value: Any) -> str:
    normalized = str(value or "STOCK").strip().upper()
    if normalized == "STOCK":
        return "STOCK"
    if "COMMODITY" in normalized or "REAL_ASSET" in normalized:
        return "COMMODITY_VEHICLE"
    return "ETF"


def _ratio(numerator: Decimal, denominator: Decimal) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator / denominator), 10)


def _bounded(value: Any, *, default: float = 0.0) -> float:
    number = _number(value)
    if number == 0 and value in (None, ""):
        number = default
    return min(1.0, max(0.0, number))


def _rounded_map(values: dict[str, float]) -> dict[str, float]:
    return {key: round(value, 8) for key, value in sorted(values.items()) if value > 0}
