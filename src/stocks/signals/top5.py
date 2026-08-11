from __future__ import annotations

import html
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.research.autopilot.contracts import stable_hash
from stocks.signals.freshness import (
    SIGNAL_MAX_AGES,
    effective_signal_expiration,
    signal_is_current,
)
from stocks.universe import broad_asset_metadata, broad_commodity_symbols


MINIMUM_OPPORTUNITY_SCORE = 0.60
MAX_SECTOR_COUNT = 2
MAX_REGION_COUNT = 3
MAX_SLEEVE_COUNT = 3
MAX_PAIRWISE_CORRELATION = 0.80
MAX_FAMILY_OVERLAP = 0.80
def publish_top_signals(
    project_root: Path,
    *,
    mode: str = "diversified",
    limit: int = 5,
) -> dict[str, Any]:
    if not 1 <= limit <= 20:
        return _blocked("LIMIT_OUT_OF_RANGE")
    if mode not in {
        "top",
        "raw",
        "diversified",
        "trending",
        "actionable",
        "stocks",
        "etfs",
        "commodities",
        "auto-eligible",
        "dashboard",
    }:
        return _blocked("UNKNOWN_TOP_SIGNAL_MODE")

    now = datetime.now(UTC)
    previous_publication = _read_json(
        project_root
        / "output"
        / "signals"
        / "latest_top_5_publication.json"
    )
    ranking = _read_json(
        project_root / "output" / "portfolio" / "opportunity_ranking.json"
    )
    signal_report = _read_json(
        project_root / "output" / "signals" / "latest_signals.json"
    )
    if not ranking or not signal_report:
        return _blocked("CURRENT_SIGNAL_OR_OPPORTUNITY_ARTIFACT_MISSING")

    opportunities = ranking.get("opportunities", [])
    signals = signal_report.get("signals", [])
    if not isinstance(opportunities, list) or not isinstance(signals, list):
        return _blocked("CURRENT_SIGNAL_OR_OPPORTUNITY_SCHEMA_INVALID")

    generated_at = now.isoformat()
    market_regime = _market_regime(project_root)
    current_signals = [row for row in signals if _signal_is_current(row, now)]
    expired_signal_count = len(signals) - len(current_signals)
    best_signals = _best_signal_by_ticker(current_signals)
    instrument_names = _instrument_names(project_root)
    formatted_candidates = [
        _format_opportunity(
            opportunity,
            best_signals.get(str(opportunity.get("ticker", "")).upper()),
            generated_at=generated_at,
            market_regime=market_regime,
            company_or_fund_name=instrument_names.get(
                str(opportunity.get("ticker", "")).upper()
            ),
        )
        for opportunity in opportunities
        if float(opportunity.get("opportunity_score", 0.0))
        >= MINIMUM_OPPORTUNITY_SCORE
    ]
    candidates: list[dict[str, Any]] = [
        row for row in formatted_candidates if row is not None
    ]
    candidates.sort(
        key=lambda row: (
            -float(row["opportunity_score"]),
            str(row["symbol"]),
        )
    )

    raw_rows = _annotate_rotation(
        _rank(candidates[:limit]),
        previous_publication.get("raw_top_5", []),
        generated_at,
    )
    diversified_rows, exclusions = _diversify(
        candidates,
        project_root=project_root,
        limit=limit,
    )
    diversified_rows = _annotate_rotation(
        _rank(diversified_rows),
        previous_publication.get("diversified_top_5", []),
        generated_at,
    )
    actionable_rows = [
        row for row in diversified_rows if row["manual_signal_eligible"]
    ]
    commodity_symbols = broad_commodity_symbols(project_root)
    stock_rows = _annotate_rotation(_rank(
        [
            row
            for row in candidates
            if str(row.get("instrument_type", "")).upper() == "STOCK"
        ][:limit]
    ), previous_publication.get("top_stocks", []), generated_at)
    etf_rows = _annotate_rotation(_rank(
        [
            row
            for row in candidates
            if "ETF" in str(row.get("instrument_type", "")).upper()
            and str(row.get("symbol", "")).upper()
            not in commodity_symbols
        ][:limit]
    ), previous_publication.get("top_etfs", []), generated_at)
    commodity_rows = _annotate_rotation(_rank(
        [
            row
            for row in candidates
            if str(row.get("symbol", "")).upper() in commodity_symbols
            or str(row.get("sleeve", "")).lower()
            == "commodity_security"
        ][:limit]
    ), previous_publication.get("top_commodity_exposures", []), generated_at)
    if not commodity_rows:
        commodity_rows = _annotate_rotation(_rank(
            _commodity_signal_watchlist(
                project_root,
                best_signals=best_signals,
                generated_at=generated_at,
                market_regime=market_regime,
                limit=limit,
            )
        ), previous_publication.get("top_commodity_exposures", []), generated_at)
    auto_eligible_rows = _rank(
        [
            row
            for row in candidates
            if row["automated_execution_eligible"]
        ][:limit]
    )
    manual_order_plans = [
        _manual_order_plan(row) for row in actionable_rows
    ]

    publication = {
        "schema": "top_signal_publication_v1",
        "status": "GO",
        "generated_at": generated_at,
        "selection_policy": {
            "minimum_opportunity_score": MINIMUM_OPPORTUNITY_SCORE,
            "maximum_signal_count": limit,
            "weak_signals_added_to_fill_limit": False,
            "maximum_sector_count": MAX_SECTOR_COUNT,
            "maximum_region_count": MAX_REGION_COUNT,
            "maximum_sleeve_count": MAX_SLEEVE_COUNT,
            "maximum_pairwise_correlation": MAX_PAIRWISE_CORRELATION,
            "maximum_strategy_family_overlap": MAX_FAMILY_OVERLAP,
            "timeframe_maximum_signal_age": {
                key: int(value.total_seconds())
                for key, value in SIGNAL_MAX_AGES.items()
            },
        },
        "source_signal_generated_at": signal_report.get("generated_at"),
        "source_signal_count": len(signals),
        "current_signal_count": len(current_signals),
        "expired_or_invalid_signal_count": expired_signal_count,
        "rotation_summary": _rotation_summary(
            diversified_rows,
            previous_publication.get("diversified_top_5", []),
            previous_publication.get("generated_at"),
        ),
        "raw_top_5": raw_rows,
        "diversified_top_5": diversified_rows,
        "actionable_signals": actionable_rows,
        "top_stocks": stock_rows,
        "top_etfs": etf_rows,
        "top_commodity_exposures": commodity_rows,
        "auto_eligible_signals": auto_eligible_rows,
        "manual_order_plans": manual_order_plans,
        "diversification_exclusions": exclusions,
        "source_artifacts": [
            "output/portfolio/opportunity_ranking.json",
            "output/portfolio/correlation_matrix.parquet",
            "output/signals/latest_signals.json",
        ],
        "manual_signal_eligible_count": len(actionable_rows),
        "automated_execution_eligible_count": sum(
            bool(row["automated_execution_eligible"])
            for row in diversified_rows
        ),
        "signal_authority": "INFORMATIONAL_MANUAL_ANALYSIS",
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_submission": False,
        "broker_calls": 0,
        "orders_generated": 0,
    }
    publication["content_hash"] = stable_hash(
        {
            "selection_policy": publication["selection_policy"],
            "raw_top_5": _semantic_rows(raw_rows),
            "diversified_top_5": _semantic_rows(diversified_rows),
            "actionable_signals": _semantic_rows(actionable_rows),
            "top_stocks": _semantic_rows(stock_rows),
            "top_etfs": _semantic_rows(etf_rows),
            "top_commodity_exposures": _semantic_rows(commodity_rows),
            "auto_eligible_signals": _semantic_rows(
                auto_eligible_rows
            ),
            "manual_order_plans": manual_order_plans,
            "execution_authority": "NONE",
        }
    )
    _publish(project_root, publication)

    selected = {
        "raw": raw_rows,
        "diversified": diversified_rows,
        "top": diversified_rows,
        "trending": diversified_rows,
        "actionable": actionable_rows,
        "stocks": stock_rows,
        "etfs": etf_rows,
        "commodities": commodity_rows,
        "auto-eligible": auto_eligible_rows,
        "dashboard": diversified_rows,
    }[mode]
    return {
        **{
            key: value
            for key, value in publication.items()
            if key
            not in {
                "raw_top_5",
                "diversified_top_5",
                "actionable_signals",
                "top_stocks",
                "top_etfs",
                "top_commodity_exposures",
                "auto_eligible_signals",
                "manual_order_plans",
            }
        },
        "mode": mode,
        "signal_count": len(selected),
        "signals": selected,
        "manual_order_plans": (
            manual_order_plans if mode == "actionable" else []
        ),
        "collections": (
            {
                "raw_top_5": raw_rows,
                "diversified_top_5": diversified_rows,
                "top_stocks": stock_rows,
                "top_etfs": etf_rows,
                "top_commodity_exposures": commodity_rows,
                "manual_actionable": actionable_rows,
                "auto_eligible": auto_eligible_rows,
            }
            if mode == "dashboard"
            else None
        ),
    }


def _best_signal_by_ticker(
    signals: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    ranked = sorted(
        signals,
        key=lambda row: (
            -float(row.get("confidence_score", 0.0)),
            -float(row.get("reward_risk_1", 0.0)),
            str(row.get("strategy_id", "")),
        ),
    )
    result: dict[str, dict[str, Any]] = {}
    for row in ranked:
        ticker = str(row.get("ticker", "")).upper()
        if ticker and ticker not in result:
            result[ticker] = row
    return result


def _signal_is_current(row: dict[str, Any], now: datetime) -> bool:
    declared_fresh = str(row.get("data_freshness", "")).upper() == "FRESH"
    if declared_fresh:
        return signal_is_current(row, now=now)
    freshness = row.get("signal_freshness", {})
    return bool(
        isinstance(freshness, dict)
        and freshness.get("freshness_basis")
        == "EXCHANGE_SESSION_AWARE_SIGNAL_EXPIRY_V1"
        and signal_is_current(row, now=now)
    )


def _effective_signal_expiration(
    row: dict[str, Any],
    declared_expiration: datetime | None = None,
) -> datetime:
    return effective_signal_expiration(
        row,
        declared_expiration=declared_expiration,
    )


def _annotate_rotation(
    rows: list[dict[str, Any]],
    previous_rows: Any,
    generated_at: str,
) -> list[dict[str, Any]]:
    previous = {
        str(row.get("symbol", "")).upper(): row
        for row in previous_rows
        if isinstance(row, dict) and row.get("symbol")
    } if isinstance(previous_rows, list) else {}
    annotated: list[dict[str, Any]] = []
    for row in rows:
        prior = previous.get(str(row.get("symbol", "")).upper())
        if prior is None:
            appearance = "NEW"
            first_seen = generated_at
            publication_count = 1
        else:
            score_delta = float(row.get("opportunity_score", 0.0)) - float(
                prior.get("opportunity_score", 0.0)
            )
            if row.get("signal_id") != prior.get("signal_id"):
                appearance = "REFRESHED"
            elif (
                abs(score_delta) >= 0.005
                or row.get("signal_status") != prior.get("signal_status")
                or row.get("eligibility_status")
                != prior.get("eligibility_status")
            ):
                appearance = "UPDATED"
            else:
                appearance = "PERSISTENT"
            first_seen = prior.get("first_seen_at") or prior.get(
                "generated_at", generated_at
            )
            publication_count = int(
                prior.get("consecutive_publication_count", 1)
            ) + 1
        prior_score = (
            float(prior.get("opportunity_score", 0.0))
            if prior is not None
            else None
        )
        annotated.append(
            {
                **row,
                "appearance_status": appearance,
                "first_seen_at": first_seen,
                "consecutive_publication_count": publication_count,
                "previous_rank": prior.get("rank") if prior else None,
                "rank_change": (
                    int(prior.get("rank", row["rank"])) - int(row["rank"])
                    if prior
                    else None
                ),
                "score_change": (
                    round(float(row["opportunity_score"]) - prior_score, 6)
                    if prior_score is not None
                    else None
                ),
            }
        )
    return annotated


def _rotation_summary(
    rows: list[dict[str, Any]],
    previous_rows: Any,
    previous_generated_at: Any,
) -> dict[str, Any]:
    previous_symbols = {
        str(row.get("symbol", "")).upper()
        for row in previous_rows
        if isinstance(row, dict) and row.get("symbol")
    } if isinstance(previous_rows, list) else set()
    current_symbols = {str(row.get("symbol", "")).upper() for row in rows}
    statuses = Counter(str(row.get("appearance_status")) for row in rows)
    material = statuses["NEW"] + statuses["REFRESHED"] + statuses["UPDATED"]
    removed = sorted(previous_symbols - current_symbols)
    return {
        "previous_generated_at": previous_generated_at,
        "added_symbols": sorted(current_symbols - previous_symbols),
        "removed_symbols": removed,
        "new_count": statuses["NEW"],
        "refreshed_count": statuses["REFRESHED"],
        "updated_count": statuses["UPDATED"],
        "persistent_count": statuses["PERSISTENT"],
        "material_change_count": material + len(removed),
        "same_symbol_list": current_symbols == previous_symbols,
    }


def _format_opportunity(
    opportunity: dict[str, Any],
    signal: dict[str, Any] | None,
    *,
    generated_at: str,
    market_regime: str,
    company_or_fund_name: str | None,
) -> dict[str, Any] | None:
    ticker = str(opportunity.get("ticker", "")).upper()
    if not ticker or signal is None:
        return None

    manual_blockers = _manual_blockers(opportunity, signal)
    deployment_blockers = sorted(
        {
            *[str(value) for value in opportunity.get("deployment_blockers", [])],
            "AUTOMATED_EXECUTION_AUTHORITY_NOT_GRANTED",
            "STRATEGY_AUTHORITY_NONE",
        }
    )
    manual_eligible = not manual_blockers
    auto_eligible = bool(
        opportunity.get("deployment_eligible")
        and signal.get("automatic_execution_allowed")
        and not deployment_blockers
    )
    current = _number(signal.get("current_market_price"))
    zone_low = _number(signal.get("entry_zone_low"))
    zone_high = _number(signal.get("entry_zone_high"))
    within_entry = (
        current is not None
        and zone_low is not None
        and zone_high is not None
        and zone_low <= current <= zone_high
    )
    signal_status = (
        "ACTIONABLE"
        if manual_eligible and within_entry
        else "NEAR_ENTRY"
        if manual_eligible
        else "WATCH"
    )
    eligibility_status = (
        "AUTO_ELIGIBLE"
        if auto_eligible
        else "MANUAL_ACTIONABLE"
        if manual_eligible and within_entry
        else "AUTO_BLOCKED"
        if manual_eligible
        else "WATCH_ONLY"
    )
    families = sorted(
        {str(value) for value in opportunity.get("strategy_families", [])}
    )
    reasons = [str(value) for value in signal.get("reasons", [])]
    risks = sorted(
        {
            *[str(value) for value in signal.get("risks", [])],
            *manual_blockers,
            *deployment_blockers,
        }
    )
    timeframes = [str(value) for value in opportunity.get("timeframes", [])]
    signal_timeframe = str(signal.get("timeframe") or "1d")
    effective_expiration = _effective_signal_expiration(signal).isoformat()
    components = opportunity.get("components", {})
    return {
        "rank": 0,
        "signal_id": signal.get("signal_id"),
        "generated_at": generated_at,
        "valid_until": effective_expiration,
        "symbol": ticker,
        "company_or_fund_name": company_or_fund_name or ticker,
        "name_status": (
            "UNIVERSE_METADATA"
            if company_or_fund_name
            and company_or_fund_name.upper() != ticker
            else "NAME_UNAVAILABLE_TICKER_FALLBACK"
        ),
        "instrument_type": opportunity.get("asset_type"),
        "exchange": signal.get("exchange"),
        "currency": opportunity.get("currency"),
        "signal_contract_currency_status": opportunity.get(
            "signal_contract_currency_status"
        ),
        "signal_currency": opportunity.get("signal_currency"),
        "contract_currency": opportunity.get("contract_currency"),
        "sector": opportunity.get("sector"),
        "region": opportunity.get("region"),
        "sleeve": opportunity.get("sleeve"),
        "strategy_id": signal.get("strategy_id"),
        "strategy_family": families,
        "direction": "LONG",
        "setup_type": opportunity.get("entry_method"),
        "timeframe": signal_timeframe,
        "signal_timeframe": signal_timeframe,
        "timeframes": timeframes,
        "confirmation_timeframes": timeframes,
        "macro_regime": "CONTEXT_ONLY",
        "market_regime": market_regime,
        "signal_status": signal_status,
        "eligibility_status": eligibility_status,
        "manual_signal_eligible": manual_eligible,
        "automated_execution_eligible": auto_eligible,
        "manual_eligibility_blockers": manual_blockers,
        "automated_execution_blockers": deployment_blockers,
        "confidence": _number(signal.get("confidence_score")),
        "opportunity_score": round(
            float(opportunity.get("opportunity_score", 0.0)), 6
        ),
        "score_components": components,
        "current_price": signal.get("current_market_price"),
        "preferred_entry": signal.get("preferred_entry"),
        "entry_zone_low": signal.get("entry_zone_low"),
        "entry_zone_high": signal.get("entry_zone_high"),
        "do_not_chase_above": signal.get("entry_zone_high"),
        "initial_stop": signal.get("stop_loss"),
        "technical_invalidation": signal.get("invalidation_level"),
        "target_1": signal.get("take_profit_1"),
        "target_2": signal.get("take_profit_2"),
        "trailing_exit_method": opportunity.get("exit_policy"),
        "estimated_holding_period": opportunity.get(
            "expected_holding_period"
        ),
        "risk_reward_to_target_1": signal.get("reward_risk_1"),
        "risk_reward_to_target_2": signal.get("reward_risk_2"),
        "suggested_risk_percent": None,
        "example_position_size": signal.get("suggested_quantity"),
        "model_order_cap_eur": signal.get("maximum_order_value_eur"),
        "estimated_order_value": None,
        "earnings_or_event_date": None,
        "liquidity_status": (
            "GO"
            if float(components.get("liquidity", 0.0)) >= 0.7
            else "REVIEW"
        ),
        "spread_status": "DATA_UNAVAILABLE",
        "fundamental_summary": {
            "quality_score": components.get("fundamental_quality"),
            "shariah_status": opportunity.get("shariah_status"),
        },
        "technical_summary": {
            "signal_quality": components.get("signal_quality"),
            "setup_quality": components.get("setup_quality"),
            "relative_strength": components.get("relative_strength"),
            "timeframe_confirmation": components.get(
                "timeframe_confirmation"
            ),
        },
        "macro_summary": {
            "regime_fit": components.get("regime_fit"),
            "authority": "CONTEXT_ONLY",
        },
        "microstructure_context": opportunity.get(
            "market_context",
            {
                "status": "NO_CONTEXT_NEUTRAL_FALLBACK",
                "standalone_entry_authority": False,
                "execution_authority": "NONE",
            },
        ),
        "why_now": reasons,
        "main_risks": risks,
        "data_quality": signal.get("data_freshness"),
        "contract_resolved": bool(opportunity.get("contract_resolved")),
        "research_evidence_tiers": opportunity.get("evidence_tiers", []),
        "provenance": {
            "opportunity_timestamp": opportunity.get("data_timestamp"),
            "signal_timestamp": signal.get("signal_timestamp"),
            "signal_data_timestamp": signal.get("data_timestamp"),
        },
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _manual_blockers(
    opportunity: dict[str, Any],
    signal: dict[str, Any],
) -> list[str]:
    blockers = {
        str(value)
        for value in opportunity.get("research_allocation_blockers", [])
    }
    blockers.update(
        str(value) for value in opportunity.get("execution_blockers", [])
    )
    if not opportunity.get("research_allocation_eligible"):
        blockers.add("RESEARCH_ALLOCATION_NOT_ELIGIBLE")
    if not opportunity.get("contract_resolved"):
        blockers.add("CONTRACT_NOT_RESOLVED")
    if signal.get("data_freshness") != "FRESH":
        blockers.add("SIGNAL_DATA_NOT_FRESH")
    if _number(signal.get("reward_risk_1")) is None:
        blockers.add("RISK_REWARD_UNAVAILABLE")
    elif float(signal["reward_risk_1"]) < 1.5:
        blockers.add("MINIMUM_RISK_REWARD_NOT_MET")
    required_prices = (
        "current_market_price",
        "entry_zone_low",
        "entry_zone_high",
        "stop_loss",
        "take_profit_1",
    )
    if any(_number(signal.get(field)) is None for field in required_prices):
        blockers.add("ORDER_GEOMETRY_INCOMPLETE")
    return sorted(blockers)


def _commodity_signal_watchlist(
    project_root: Path,
    *,
    best_signals: dict[str, dict[str, Any]],
    generated_at: str,
    market_regime: str,
    limit: int,
) -> list[dict[str, Any]]:
    symbols = broad_commodity_symbols(project_root)
    metadata = broad_asset_metadata(project_root)
    rows: list[dict[str, Any]] = []
    for symbol in sorted(symbols):
        signal = best_signals.get(symbol)
        if signal is None or signal.get("data_freshness") != "FRESH":
            continue
        context = metadata.get(symbol, {})
        confidence = _number(signal.get("confidence_score")) or 0.0
        rows.append(
            {
                "rank": 0,
                "signal_id": signal.get("signal_id"),
                "generated_at": generated_at,
                "valid_until": _effective_signal_expiration(signal).isoformat(),
                "symbol": symbol,
                "company_or_fund_name": symbol,
                "instrument_type": "COMMODITY_ETF",
                "exchange": signal.get("exchange"),
                "currency": signal.get("currency"),
                "sector": context.get("sector"),
                "region": context.get("region"),
                "sleeve": context.get("sleeve"),
                "strategy_id": signal.get("strategy_id"),
                "strategy_family": [],
                "direction": "LONG",
                "setup_type": "TECHNICAL_WATCHLIST",
                "timeframe": str(signal.get("timeframe", "")),
                "signal_timeframe": str(signal.get("timeframe", "")),
                "timeframes": [str(signal.get("timeframe", ""))],
                "confirmation_timeframes": [
                    str(signal.get("timeframe", ""))
                ],
                "macro_regime": "CONTEXT_ONLY",
                "market_regime": market_regime,
                "signal_status": "WATCH",
                "eligibility_status": "WATCH_ONLY",
                "manual_signal_eligible": False,
                "automated_execution_eligible": False,
                "manual_eligibility_blockers": [
                    "PORTFOLIO_OPPORTUNITY_RANKING_UNAVAILABLE"
                ],
                "automated_execution_blockers": [
                    "AUTOMATED_EXECUTION_AUTHORITY_NOT_GRANTED",
                    "PORTFOLIO_OPPORTUNITY_RANKING_UNAVAILABLE",
                    "STRATEGY_AUTHORITY_NONE",
                ],
                "confidence": confidence,
                "opportunity_score": confidence,
                "score_semantics": (
                    "SIGNAL_CONFIDENCE_PROXY_NOT_PORTFOLIO_"
                    "OPPORTUNITY_SCORE"
                ),
                "score_components": {
                    "signal_confidence": confidence,
                },
                "current_price": signal.get("current_market_price"),
                "preferred_entry": signal.get("preferred_entry"),
                "entry_zone_low": signal.get("entry_zone_low"),
                "entry_zone_high": signal.get("entry_zone_high"),
                "do_not_chase_above": signal.get("entry_zone_high"),
                "initial_stop": signal.get("stop_loss"),
                "technical_invalidation": signal.get(
                    "invalidation_level"
                ),
                "target_1": signal.get("take_profit_1"),
                "target_2": signal.get("take_profit_2"),
                "trailing_exit_method": "SIGNAL_SPECIFICATION",
                "estimated_holding_period": signal.get(
                    "expected_holding_period"
                ),
                "risk_reward_to_target_1": signal.get("reward_risk_1"),
                "risk_reward_to_target_2": signal.get("reward_risk_2"),
                "suggested_risk_percent": None,
                "example_position_size": signal.get(
                    "suggested_quantity"
                ),
                "model_order_cap_eur": signal.get(
                    "maximum_order_value_eur"
                ),
                "estimated_order_value": None,
                "earnings_or_event_date": None,
                "liquidity_status": "DATA_UNAVAILABLE",
                "spread_status": "DATA_UNAVAILABLE",
                "fundamental_summary": {
                    "quality_score": None,
                    "shariah_status": "UNSCREENED_PRODUCT_STRUCTURE",
                },
                "technical_summary": {
                    "signal_confidence": confidence,
                },
                "macro_summary": {
                    "authority": "CONTEXT_ONLY",
                },
                "why_now": [
                    str(value) for value in signal.get("reasons", [])
                ],
                "main_risks": sorted(
                    {
                        *[
                            str(value)
                            for value in signal.get("risks", [])
                        ],
                        "COMMODITY_PRODUCT_STRUCTURE_REVIEW_REQUIRED",
                        "PORTFOLIO_OPPORTUNITY_RANKING_UNAVAILABLE",
                    }
                ),
                "data_quality": signal.get("data_freshness"),
                "contract_resolved": bool(
                    signal.get("contract_identity")
                ),
                "research_evidence_tiers": [],
                "provenance": {
                    "signal_timestamp": signal.get("signal_timestamp"),
                    "signal_data_timestamp": signal.get(
                        "data_timestamp"
                    ),
                    "ranking_basis": "SIGNAL_CONFIDENCE_PROXY",
                },
                "execution_authority": "NONE",
                "broker_calls": 0,
                "orders_generated": 0,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["confidence"]),
            str(row["symbol"]),
        )
    )
    return rows[:limit]


def _diversify(
    candidates: list[dict[str, Any]],
    *,
    project_root: Path,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    correlation = _correlations(project_root)
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    sectors: Counter[str] = Counter()
    regions: Counter[str] = Counter()
    sleeves: Counter[str] = Counter()
    for candidate in candidates:
        if len(selected) >= limit:
            break
        reasons: list[str] = []
        sector = str(candidate.get("sector", "UNKNOWN"))
        region = str(candidate.get("region", "UNKNOWN"))
        sleeve = str(candidate.get("sleeve", "UNKNOWN"))
        if sectors[sector] >= MAX_SECTOR_COUNT:
            reasons.append("SECTOR_CAP")
        if regions[region] >= MAX_REGION_COUNT:
            reasons.append("REGION_CAP")
        if sleeves[sleeve] >= MAX_SLEEVE_COUNT:
            reasons.append("SLEEVE_CAP")
        for prior in selected:
            corr = correlation.get(
                (str(candidate["symbol"]), str(prior["symbol"]))
            )
            if corr is not None and corr > MAX_PAIRWISE_CORRELATION:
                reasons.append(
                    f"CORRELATION_CAP:{prior['symbol']}:{corr:.4f}"
                )
            overlap = _family_overlap(candidate, prior)
            if overlap > MAX_FAMILY_OVERLAP:
                reasons.append(
                    f"STRATEGY_FAMILY_OVERLAP:{prior['symbol']}:{overlap:.4f}"
                )
        if reasons:
            exclusions.append(
                {
                    "symbol": candidate["symbol"],
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        selected.append(candidate)
        sectors[sector] += 1
        regions[region] += 1
        sleeves[sleeve] += 1
    return selected, exclusions


def _family_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = set(left.get("strategy_family", []))
    b = set(right.get("strategy_family", []))
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _correlations(project_root: Path) -> dict[tuple[str, str], float]:
    path = project_root / "output" / "portfolio" / "correlation_matrix.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    if "ticker" not in frame.columns:
        return {}
    result: dict[tuple[str, str], float] = {}
    for _, row in frame.iterrows():
        left = str(row["ticker"])
        for right in frame.columns:
            if right == "ticker" or pd.isna(row[right]):
                continue
            result[(left, str(right))] = float(row[right])
    return result


def _manual_order_plan(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "manual_signal_order_plan_v1",
        "signal_id": row["signal_id"],
        "symbol": row["symbol"],
        "direction": row["direction"],
        "order_type": "LIMIT",
        "limit_price": row["preferred_entry"],
        "entry_zone_low": row["entry_zone_low"],
        "entry_zone_high": row["entry_zone_high"],
        "do_not_chase_above": row["do_not_chase_above"],
        "initial_stop": row["initial_stop"],
        "target_1": row["target_1"],
        "target_2": row["target_2"],
        "example_position_size": row["example_position_size"],
        "position_size_is_model_example": True,
        "operator_must_recalculate_account_risk": True,
        "valid_until": row["valid_until"],
        "manual_signal_eligible": True,
        "automated_execution_eligible": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _publish(project_root: Path, publication: dict[str, Any]) -> None:
    root = project_root / "output" / "signals"
    root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "latest_raw_top_5.json": {
            "schema": "raw_top_5_v1",
            "generated_at": publication["generated_at"],
            "signals": publication["raw_top_5"],
        },
        "latest_diversified_top_5.json": {
            "schema": "diversified_top_5_v1",
            "generated_at": publication["generated_at"],
            "signals": publication["diversified_top_5"],
            "diversification_exclusions": publication[
                "diversification_exclusions"
            ],
        },
        "latest_actionable_signals.json": {
            "schema": "manual_actionable_signals_v1",
            "generated_at": publication["generated_at"],
            "signals": publication["actionable_signals"],
        },
        "latest_top_stocks.json": {
            "schema": "top_stocks_v1",
            "generated_at": publication["generated_at"],
            "signals": publication["top_stocks"],
        },
        "latest_top_etfs.json": {
            "schema": "top_etfs_v1",
            "generated_at": publication["generated_at"],
            "signals": publication["top_etfs"],
        },
        "latest_top_commodity_exposures.json": {
            "schema": "top_commodity_exposures_v1",
            "generated_at": publication["generated_at"],
            "signals": publication["top_commodity_exposures"],
        },
        "latest_auto_eligible_signals.json": {
            "schema": "auto_eligible_signals_v1",
            "generated_at": publication["generated_at"],
            "signals": publication["auto_eligible_signals"],
            "execution_authority": "NONE",
        },
        "latest_manual_order_plans.json": {
            "schema": "manual_order_plans_v1",
            "generated_at": publication["generated_at"],
            "plans": publication["manual_order_plans"],
            "execution_authority": "NONE",
            "orders_generated": 0,
        },
        "latest_top_5_publication.json": publication,
    }
    for name, payload in artifacts.items():
        (root / name).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    (root / "latest_top_5.html").write_text(
        _render_html(publication),
        encoding="utf-8",
    )
    history_path = root / "signal_history.jsonl"
    history_record = {
        "generated_at": publication["generated_at"],
        "content_hash": publication["content_hash"],
        "raw_top_5": publication["raw_top_5"],
        "diversified_top_5": publication["diversified_top_5"],
        "actionable_signals": publication["actionable_signals"],
        "execution_authority": "NONE",
        "orders_generated": 0,
    }
    previous_hash = None
    if history_path.exists():
        lines = [
            line
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if lines:
            previous_hash = json.loads(lines[-1]).get("content_hash")
    if previous_hash != publication["content_hash"]:
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(history_record, ensure_ascii=False, default=str)
                + "\n"
            )


def _render_html(publication: dict[str, Any]) -> str:
    rows = publication["diversified_top_5"]
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['rank']}</td>"
            f"<td>{html.escape(str(row['symbol']))}</td>"
            f"<td>{html.escape(str(row['timeframe']))}</td>"
            f"<td>{row['opportunity_score']:.4f}</td>"
            f"<td>{html.escape(str(row['signal_status']))}</td>"
            f"<td>{html.escape(str(row['eligibility_status']))}</td>"
            f"<td>{html.escape(str(row['preferred_entry']))}</td>"
            f"<td>{html.escape(str(row['initial_stop']))}</td>"
            f"<td>{html.escape(str(row['target_1']))}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Diversified Top 5</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#17202a}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccd1d1;padding:8px;text-align:left}"
        "th{background:#f4f6f7}</style></head><body>"
        "<h1>Diversified Top 5</h1>"
        f"<p>Generated {html.escape(publication['generated_at'])}. "
        "Analysis only. Execution authority: NONE.</p>"
        "<table><thead><tr><th>Rank</th><th>Symbol</th><th>Timeframes</th>"
        "<th>Score</th><th>Signal</th><th>Eligibility</th><th>Entry</th>"
        "<th>Stop</th><th>Target 1</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></body></html>"
    )


def _market_regime(project_root: Path) -> str:
    dynamic = _read_json(project_root / "output" / "dynamic" / "status.json")
    return str(dynamic.get("current_regime", "UNAVAILABLE"))


def _instrument_names(project_root: Path) -> dict[str, str]:
    path = project_root / "output" / "universe" / "instruments.parquet"
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path, columns=["symbol", "name"])
    return {
        str(row.symbol).upper(): str(row.name)
        for row in frame.itertuples(index=False)
        if row.symbol and row.name
    }


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "rank": index} for index, row in enumerate(rows, start=1)]


def _semantic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    volatile = {
        "generated_at",
        "appearance_status",
        "first_seen_at",
        "consecutive_publication_count",
        "previous_rank",
        "rank_change",
        "score_change",
    }
    for row in rows:
        semantic = {
            key: value for key, value in row.items() if key not in volatile
        }
        provenance = semantic.get("provenance")
        if isinstance(provenance, dict):
            semantic["provenance"] = {
                key: value
                for key, value in provenance.items()
                if key
                not in {"opportunity_timestamp", "signal_timestamp"}
            }
        result.append(semantic)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "top_signal_publication_v1",
        "status": "BLOCKED",
        "reason": reason,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
