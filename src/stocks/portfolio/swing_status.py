from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.data.multitimeframe import bar_freshness
from stocks.execution.idempotency import stable_hash
from stocks.universe import broad_asset_metadata, broad_universe


CORE_SWING_TIMEFRAMES = ("15m", "1h", "2h", "4h", "1d")
SUPPORTED_SWING_TIMEFRAMES = ("15m", "1h", "2h", "4h", "6h", "12h", "1d", "1w")
TIMEFRAME_ROLES = {
    "15m": ["TACTICAL_ENTRY_ACTIVATION", "TACTICAL_POSITION_MANAGEMENT"],
    "1h": ["PRIMARY_SETUP_DEVELOPMENT", "PRIMARY_OPPORTUNITY_SCAN"],
    "2h": ["SETUP_DEVELOPMENT", "INTERMEDIATE_CONFIRMATION"],
    "4h": ["SWING_STRUCTURE", "SWING_REGIME"],
    "6h": ["OPTIONAL_CONTEXT_WHEN_INCREMENTAL_EVIDENCE_EXISTS"],
    "12h": ["OPTIONAL_CONTEXT_WHEN_INCREMENTAL_EVIDENCE_EXISTS"],
    "1d": ["PRIMARY_REGIME", "STRUCTURAL_RELATIVE_STRENGTH"],
    "1w": ["STRUCTURAL_CONTEXT", "LONG_HORIZON_RISK"],
}
PUBLIC_STATUS_PATH = Path("output/portfolio/active-swing-product-status.json")
PUBLIC_FUNNEL_PATH = Path("output/portfolio/active-swing-funnel.json")
PUBLIC_REQUIRED_REPORT_PATH = Path("output/portfolio/active-swing-required-report.json")
MAX_CONCURRENT_ARTIFACT_SKEW = pd.Timedelta(minutes=2)
CURRENT_OPPORTUNITY_FIELDS = (
    "symbol",
    "asset_class",
    "discovery_reason",
    "strategy",
    "entry_timeframe",
    "setup_timeframe",
    "higher_timeframe_context",
    "signal_age_seconds",
    "expected_net_return",
    "ml_prediction",
    "rl_advisory",
    "whole_share_quantity",
    "risk",
    "costs",
    "portfolio_fit",
    "shariah",
    "authority",
    "quote_status",
    "final_decision",
)


def publish_active_swing_product_status(
    project_root: Path,
    *,
    signals: Iterable[dict[str, Any]],
    ranked: Iterable[dict[str, Any]],
    normalized_opportunities: dict[str, Any],
    opportunity_funnel: dict[str, Any],
    current_positions: Iterable[dict[str, Any]],
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    observed = observed_at or datetime.now(UTC)
    observed_timestamp = pd.Timestamp(observed)
    observed_timestamp = (
        observed_timestamp.tz_localize("UTC")
        if observed_timestamp.tzinfo is None
        else observed_timestamp.tz_convert("UTC")
    )
    signal_rows = list(signals)
    ranked_rows = list(ranked)
    positions = list(current_positions)
    timeframe_health = {
        timeframe: _timeframe_health(
            project_root,
            timeframe=timeframe,
            observed_at=observed,
        )
        for timeframe in SUPPORTED_SWING_TIMEFRAMES
    }
    universe_rows = broad_universe(project_root)
    universe_metadata = broad_asset_metadata(project_root)
    universe_rows = [
        {**row, **universe_metadata.get(str(row.get("symbol")), {})} for row in universe_rows
    ]
    discovery = _discovery_health(universe_rows)
    ai = _read_json(project_root / "output/ai/decision-intelligence/tournament.json")
    active_swing_ml = _read_json(
        project_root
        / "output/ai/decision-intelligence/active-swing-panel-status.json"
    )
    active_swing_inference = _read_json(
        project_root
        / "output/ai/decision-intelligence/current-active-swing-inference.json"
    )
    active_swing_tournament = _read_json(
        project_root
        / "output/ai/decision-intelligence/active-swing-tournament.json"
    )
    rl = _read_json(project_root / "output/rl/status.json")
    rl_rotation = _read_json(
        project_root / "output/portfolio/rl-portfolio-rotation.json"
    )
    candidate_generation = _read_json(
        project_root / "output/research/active_swing/candidate-generation-status.json"
    )
    tactical_candidates = _read_json(project_root / "output/signals/active_swing_15m_signals.json")
    quote = _read_json(project_root / "output/ibkr/live/quote-readiness.json")
    performance = _read_json(project_root / "output/portfolio/performance-attribution.json")
    desired_targets = _read_json(project_root / "output/portfolio/desired-portfolio-targets.json")
    execution = _read_json(project_root / "output/operations/execution-status.json")
    reconciliation = _read_json(project_root / "output/ibkr/phase9/reconciliation-audit.json")
    heartbeat = _read_json(project_root / "runtime/heartbeat.json")
    machine = _read_json(project_root / "output/operations/machine-status.json")
    authority = _read_json(project_root / "output/ibkr/live/authority-status.json")
    normalized_rows = normalized_opportunities.get("combined_ranking", [])

    research_ready = bool(signal_rows and ranked_rows)
    data_ready = all(
        timeframe_health[value]["status"] == "FRESH_CLOSED_BAR" for value in CORE_SWING_TIMEFRAMES
    )
    ml_ready = bool(
        active_swing_tournament.get("performance_gate_go")
        and active_swing_tournament.get("external_data_gate_go")
        and active_swing_tournament.get("forward_evidence_go")
        and active_swing_ml.get("training_ready")
        and active_swing_ml.get("active_swing_candidate_unit_go")
        and active_swing_inference.get("compatible_model")
        and str(active_swing_tournament.get("promotion_status", "")).startswith(
            "APPROVED"
        )
    )
    rl_policy_gate = rl_rotation.get("policy_readiness")
    rl_policy_gate = rl_policy_gate if isinstance(rl_policy_gate, dict) else {}
    rl_ready = bool(
        rl.get("incremental_evidence_go")
        and rl.get("forward_evidence_go")
        and rl.get("promotion_status") in {"ELIGIBLE", "ACTIVE"}
        and rl_policy_gate.get("status") == "GO"
        and rl_policy_gate.get("candidate_identity_deduplicated") is True
        and int(rl_policy_gate.get("architecture_binding_incomplete_count", 0) or 0)
        == 0
    )
    portfolio_ready = bool(normalized_opportunities.get("status") == "GO" and normalized_rows)
    quote_ready = bool(quote.get("quote_valid"))
    current_broker_read_only_usable = _current_broker_read_only_usable(
        reconciliation,
        observed_at=observed,
    )
    reconciliation_ready = bool(
        current_broker_read_only_usable and reconciliation.get("status") == "GO"
    )
    execution_ready = bool(quote_ready and reconciliation_ready)
    authority_ready = bool(
        str(authority.get("execution_authority") or "NONE") != "NONE"
        and authority.get("status") == "GO"
    )
    economically_good = bool(
        ml_ready
        and any(
            _positive(row.get("expected_net_return"))
            and row.get("validation_status") == "VALIDATED_OPPORTUNITY"
            for row in normalized_rows
        )
    )
    heartbeat_status = str(heartbeat.get("status") or heartbeat.get("runtime_status") or "").upper()
    heartbeat_at = pd.to_datetime(
        heartbeat.get("last_heartbeat") or heartbeat.get("updated_at"),
        utc=True,
        errors="coerce",
    )
    heartbeat_fresh = bool(
        not pd.isna(heartbeat_at)
        and -MAX_CONCURRENT_ARTIFACT_SKEW
        <= observed_timestamp - heartbeat_at
        <= pd.Timedelta(minutes=15)
    )
    operational_ready = bool(
        _operational_runtime_usable(heartbeat, machine) and heartbeat_fresh and data_ready
    )
    runtime_control = _runtime_control_state(heartbeat, machine)
    statuses = {
        "SWING_RESEARCH_READY": research_ready,
        "SWING_DATA_READY": data_ready,
        "SWING_15M_CANDIDATE_PIPELINE_READY": bool(
            candidate_generation.get("status") == "GO"
            and int(candidate_generation.get("native_15m_ready_symbol_count", 0)) > 0
            and int(candidate_generation.get("required_context_ready_symbol_count", 0)) > 0
        ),
        "SWING_ML_READY": ml_ready,
        "SWING_RL_READY": rl_ready,
        "SWING_PORTFOLIO_READY": portfolio_ready,
        "SWING_EXECUTION_READY": execution_ready,
        "SWING_AUTHORITY_READY": authority_ready,
        "SWING_ECONOMICALLY_GOOD": economically_good,
        "SWING_OPERATIONALLY_READY": operational_ready,
    }
    statuses["SWING_PRODUCT_READY"] = all(statuses.values())
    blockers = [name for name, ready in statuses.items() if not ready]

    funnel = _active_swing_funnel(
        signal_rows,
        ranked_rows,
        normalized_rows,
        opportunity_funnel,
        timeframe_health,
        candidate_generation,
        quote_ready=quote_ready,
        authority_ready=authority_ready,
    )
    broker_position_count = int(reconciliation.get("broker_position_count", 0) or 0)
    position_state_consistent = bool(
        current_broker_read_only_usable and broker_position_count == len(positions)
    )
    if not current_broker_read_only_usable:
        position_management_status = "BROKER_POSITION_STATE_UNVERIFIED"
    elif not position_state_consistent:
        position_management_status = "BROKER_PORTFOLIO_POSITION_STATE_MISMATCH"
    elif broker_position_count:
        position_management_status = "POSITIONS_REQUIRE_REAL_MANAGEMENT_PATH"
    else:
        position_management_status = "NO_OPEN_POSITION_MANAGEMENT_REQUIRED"
    position_management = {
        "status": position_management_status,
        "position_count": len(positions),
        "current_broker_position_count": broker_position_count,
        "broker_position_state_verified": current_broker_read_only_usable,
        "broker_portfolio_position_state_consistent": (position_state_consistent),
        "fabricated_positions": 0,
    }
    current_opportunities = _current_opportunity_evaluation(
        tactical_candidates,
        observed_at=observed,
        universe_metadata=universe_metadata,
        quote=quote,
    )
    required_report = _required_active_swing_report(
        timeframe_health=timeframe_health,
        signals=signal_rows,
        ranked=ranked_rows,
        candidate_generation=candidate_generation,
        current_opportunities=current_opportunities,
        positions=positions,
        position_management=position_management,
        desired_targets=desired_targets,
        ai=ai,
        active_swing_ml=active_swing_ml,
        active_swing_inference=active_swing_inference,
        active_swing_tournament=active_swing_tournament,
        rl=rl,
        rl_rotation=rl_rotation,
        execution=execution,
        quote=quote,
        performance=performance,
        readiness=statuses,
    )
    report: dict[str, Any] = {
        "schema": "active_swing_product_status_v1",
        "status": "GO" if statuses["SWING_PRODUCT_READY"] else "NO_GO",
        "generated_at": pd.Timestamp(observed).isoformat(),
        "product": "AUTONOMOUS_RETAIL_ACTIVE_SWING_FROM_15M",
        "timeframe_health": timeframe_health,
        "discovery": discovery,
        "funnel": funnel,
        "position_management": position_management,
        "current_opportunities": current_opportunities,
        "required_active_swing_report": required_report,
        "readiness": statuses,
        "blockers": blockers,
        "evidence": {
            "ml_promotion_status": active_swing_tournament.get(
                "promotion_status", "UNAVAILABLE"
            ),
            "legacy_global_ml_promotion_status": ai.get(
                "promotion_status", "UNAVAILABLE"
            ),
            "rl_promotion_status": rl.get("promotion_status", "UNAVAILABLE"),
            "quote_status": quote.get("status", "UNAVAILABLE"),
            "quote_valid": quote_ready,
            "reconciliation_status": reconciliation.get("reconciliation_status", "UNAVAILABLE"),
            "reconciliation_broker_observation_status": reconciliation.get(
                "broker_observation_status", "UNAVAILABLE"
            ),
            "reconciliation_broker_snapshot_status": reconciliation.get(
                "broker_snapshot_status", "UNAVAILABLE"
            ),
            "reconciliation_operational_broker_state": reconciliation.get(
                "operational_broker_state_status", "UNAVAILABLE"
            ),
            "reconciliation_current_read_only_usable": (current_broker_read_only_usable),
            "reconciliation_canonical_execution_evidence": (
                reconciliation.get("canonical_execution_evidence_status", "UNAVAILABLE")
            ),
            "reconciliation_historical_quarantine_status": (
                reconciliation.get("historical_orphan_quarantine_status", "UNAVAILABLE")
            ),
            "reconciliation_historical_execution_blocks_trading": bool(
                reconciliation.get("status") != "GO"
                or reconciliation.get("canonical_execution_evidence_status")
                == "INCOMPLETE_HISTORICAL_EXECUTION_CHAIN"
            ),
            "reconciliation_broker_position_count": broker_position_count,
            "reconciliation_broker_open_order_count": int(
                reconciliation.get("broker_open_order_count", 0) or 0
            ),
            "heartbeat_status": heartbeat_status or "UNAVAILABLE",
            "heartbeat_fresh": heartbeat_fresh,
            "machine_status": machine.get("status", "UNAVAILABLE"),
            "machine_enabled": machine.get("enabled") is True,
            "machine_paused": machine.get("paused") is True,
            "runtime_control_source": runtime_control["source"],
            "runtime_enabled_current": runtime_control["enabled"],
            "runtime_paused_current": runtime_control["paused"],
            "machine_cycle_blockers": machine.get("last_cycle_blockers", []),
            "machine_operationally_usable": _operational_runtime_usable(heartbeat, machine),
            "active_swing_15m_candidate_generation_status": (
                candidate_generation.get("status", "UNAVAILABLE")
            ),
            "active_swing_15m_candidate_count": int(
                candidate_generation.get("candidate_count", 0) or 0
            ),
            "active_swing_15m_candidate_unit": candidate_generation.get(
                "candidate_unit", "UNAVAILABLE"
            ),
        },
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": str(authority.get("execution_authority") or "NONE"),
        "automatic_order_submission": False,
        "orders_generated": 0,
        "broker_write_calls": 0,
    }
    report["content_hash"] = stable_hash(report)
    _write_json(project_root / PUBLIC_STATUS_PATH, report)
    _write_json(project_root / PUBLIC_FUNNEL_PATH, funnel)
    _write_json(project_root / PUBLIC_REQUIRED_REPORT_PATH, required_report)
    return report


def _timeframe_health(
    project_root: Path,
    *,
    timeframe: str,
    observed_at: datetime,
) -> dict[str, Any]:
    root = project_root / "data/research/multitimeframe/private"
    source_policy = {
        "15m": {"source_interval=15m"},
        "1h": {"source_interval=1h"},
        "2h": {"source_interval=1h", "source_interval=2h"},
        "4h": {"source_interval=1h", "source_interval=4h"},
        "6h": {"source_interval=1h", "source_interval=6h"},
        "12h": {"source_interval=1h", "source_interval=12h"},
        "1d": {"source_interval=1d"},
        "1w": {"source_interval=1d", "source_interval=1w"},
        "1mo": {"source_interval=1d", "source_interval=1mo"},
    }
    paths = [
        path
        for path in root.rglob("bars.parquet")
        if f"interval={timeframe}" in path.parts
        and (timeframe not in source_policy or bool(source_policy[timeframe] & set(path.parts)))
    ]
    symbols: set[str] = set()
    providers: set[str] = set()
    row_count = 0
    last_bar: pd.Timestamp | None = None
    last_exchange_timezone = ""
    errors: list[str] = []
    for path in paths:
        for part in path.parts:
            if part.startswith("symbol="):
                symbols.add(part.split("=", 1)[1].upper())
            elif part.startswith("provider="):
                providers.add(part.split("=", 1)[1].upper())
        try:
            frame = pd.read_parquet(path)
        except (KeyError, OSError, ValueError) as exc:
            errors.append(f"{path.name}:{type(exc).__name__}")
            continue
        if "quality_status" in frame:
            frame = frame.loc[frame["quality_status"].astype(str).str.startswith("VALIDATED")]
        for field in ("is_partial", "partial_bucket"):
            if field in frame:
                partial = frame[field].astype("boolean").fillna(False)
                frame = frame.loc[~partial]
        row_count += len(frame)
        if not frame.empty:
            timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
            candidate = timestamps.max()
            if not pd.isna(candidate) and (last_bar is None or candidate > last_bar):
                last_bar = candidate
                latest_index = timestamps.idxmax()
                last_exchange_timezone = str(
                    frame.loc[latest_index].get("exchange_timezone") or ""
                ).strip()
    freshness = bar_freshness(
        last_bar,
        interval=timeframe,
        observed_at=observed_at,
        exchange_timezone=last_exchange_timezone or None,
    )
    return {
        "status": freshness["status"],
        "dataset_count": len(paths),
        "row_count": row_count,
        "symbol_count": len(symbols),
        "providers": sorted(providers),
        "source_policy": (
            "QUALIFIED_SOURCE_INTERVAL_REQUIRED"
            if timeframe in source_policy
            else "CANONICAL_DERIVATION_ALLOWED"
        ),
        "native_source_required": timeframe in {"15m", "1h", "1d"},
        "exchange_timezone": last_exchange_timezone or None,
        "last_complete_bar": last_bar.isoformat() if last_bar is not None else None,
        "expected_last_complete_bar": None,
        "freshness": freshness,
        "read_errors": errors[:10],
    }


def _discovery_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(row.get("asset_type") or "UNKNOWN") for row in rows)
    physical_claims = [
        row for row in rows if str(row.get("commodity_exposure_type")) == "PHYSICAL_COMMODITY"
    ]
    physical = [row for row in physical_claims if row.get("physical_structure_verified") is True]
    shariah_funds = [
        row
        for row in rows
        if "SHARIAH" in str(row.get("sector") or "").upper()
        or str(row.get("asset_type")) == "SUKUK_ETF"
    ]
    return {
        "universe_instrument_count": len(rows),
        "asset_type_counts": dict(sorted(by_type.items())),
        "shariah_fund_count": len(shariah_funds),
        "physical_commodity_claim_count": len(physical_claims),
        "physically_backed_commodity_count": len(physical),
        "physically_backed_symbols": sorted(str(row.get("symbol")) for row in physical),
        "futures_proxy_is_physical": False,
        "physical_claim_requires_current_issuer_attestation": True,
        "physical_structure_does_not_imply_shariah_eligibility": True,
        "broad_discovery_required": True,
    }


def _active_swing_funnel(
    signals: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    normalized: list[dict[str, Any]],
    native_funnel: dict[str, Any],
    timeframe_health: dict[str, dict[str, Any]],
    candidate_generation: dict[str, Any],
    *,
    quote_ready: bool,
    authority_ready: bool,
) -> dict[str, Any]:
    positive = [
        row
        for row in signals
        if str(row.get("action", "")).upper() in {"BUY", "STRONG_BUY", "WATCHLIST"}
    ]
    lifecycle = Counter(str(row.get("lifecycle_state") or "UNDECLARED") for row in ranked)
    rejections = Counter(
        str(blocker) for row in ranked for blocker in row.get("deployment_blockers", [])
    )
    return {
        "schema": "current_active_swing_funnel_v1",
        "universe_scanned": int(candidate_generation.get("scanned_symbol_count", 0) or 0),
        "upstream_opportunity_rows_evaluated": int(
            native_funnel.get("universe_instrument_count", 0) or 0
        ),
        "timeframe_fresh": {
            key: value["status"] == "FRESH_CLOSED_BAR" for key, value in timeframe_health.items()
        },
        "signal_count": len(signals),
        "natural_15m_candidate_count": int(candidate_generation.get("candidate_count", 0) or 0),
        "native_15m_ready_symbol_count": int(
            candidate_generation.get("native_15m_ready_symbol_count", 0) or 0
        ),
        "required_context_ready_symbol_count": int(
            candidate_generation.get("required_context_ready_symbol_count", 0) or 0
        ),
        "candidate_unit": candidate_generation.get("candidate_unit", "UNAVAILABLE"),
        "positive_signal_count": len(positive),
        "strategy_setup_count": len(ranked),
        "near_setup_count": (
            int(candidate_generation.get("near_setup_count", 0) or 0)
            if "near_setup_count" in candidate_generation
            else lifecycle["NEAR_SETUP"]
        ),
        "entry_trigger_count": lifecycle["ENTRY_READY"],
        "lifecycle_undeclared_count": lifecycle["UNDECLARED"],
        "lifecycle_counts": dict(sorted(lifecycle.items())),
        "portfolio_candidate_count": int(native_funnel.get("portfolio_candidate_count", 0) or 0),
        "whole_share_feasible_count": sum(
            "FEASIBLE" in str(row.get("whole_share_feasibility", "")) for row in normalized
        ),
        "economically_positive_count": sum(
            _positive(row.get("expected_net_return"))
            and row.get("validation_status") == "VALIDATED_OPPORTUNITY"
            for row in normalized
        ),
        "unvalidated_positive_estimate_count": sum(
            _positive(row.get("expected_net_return"))
            and row.get("validation_status") != "VALIDATED_OPPORTUNITY"
            for row in normalized
        ),
        "shariah_valid_count": sum(
            row.get("shariah_status")
            in {"SHARIAH_ALLOWED", "SHARIAH_ELIGIBLE_PIT", "SHARIAH_COMPLIANT"}
            for row in normalized
        ),
        "authority_valid_count": len(normalized) if authority_ready else 0,
        "quote_valid_count": len(normalized) if quote_ready else 0,
        "submitted_count": 0,
        "dominant_rejection_reasons": [
            {"reason": reason, "count": count} for reason, count in rejections.most_common(10)
        ],
        "execution_authority": "NONE" if not authority_ready else "EXTERNAL",
        "orders_generated": 0,
    }


def _positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _current_opportunity_evaluation(
    tactical: dict[str, Any],
    *,
    observed_at: datetime,
    universe_metadata: dict[str, dict[str, Any]],
    quote: dict[str, Any],
) -> dict[str, Any]:
    signals = tactical.get("signals", [])
    if not isinstance(signals, list):
        signals = []
    schema_valid = tactical.get("schema") == "active_swing_15m_candidate_generation_v1" and int(
        tactical.get("candidate_count", -1) or 0
    ) == len(signals)
    generated_at = pd.to_datetime(tactical.get("generated_at"), utc=True, errors="coerce")
    observed_timestamp = pd.Timestamp(observed_at)
    observed_timestamp = (
        observed_timestamp.tz_localize("UTC")
        if observed_timestamp.tzinfo is None
        else observed_timestamp.tz_convert("UTC")
    )
    artifact_age = (
        None if pd.isna(generated_at) else (observed_timestamp - generated_at).total_seconds()
    )
    artifact_current = bool(artifact_age is not None and 0 <= artifact_age <= 30 * 60)
    if not schema_valid:
        status = "CANDIDATE_ARTIFACT_INVALID"
    elif not artifact_current:
        status = "CANDIDATE_ARTIFACT_STALE"
    elif not signals:
        status = "NO_NATURAL_CURRENT_OPPORTUNITY"
    else:
        status = "RESEARCH_OBSERVERS_ONLY"

    rows = sorted(
        (row for row in signals if isinstance(row, dict)),
        key=lambda row: float(row.get("confidence_score", 0) or 0),
        reverse=True,
    )[:5]
    evaluations = [
        _current_opportunity_row(
            row,
            observed_at=observed_timestamp,
            universe_metadata=universe_metadata,
            quote=quote,
        )
        for row in rows
    ]
    return {
        "schema": "current_active_swing_evaluation_v1",
        "status": status,
        "candidate_artifact_schema_valid": schema_valid,
        "candidate_artifact_current": artifact_current,
        "candidate_artifact_age_seconds": artifact_age,
        "natural_candidate_count": len(signals),
        "reported_candidate_count": len(evaluations),
        "required_fields": list(CURRENT_OPPORTUNITY_FIELDS),
        "highest_ranked_opportunities": evaluations,
        "older_research_rankings_are_current_opportunities": False,
        "symbol_only_cross_artifact_join_allowed": False,
        "forced_order": False,
        "execution_authority": "NONE",
        "orders_generated": 0,
        "broker_write_calls": 0,
    }


def _current_opportunity_row(
    candidate: dict[str, Any],
    *,
    observed_at: pd.Timestamp,
    universe_metadata: dict[str, dict[str, Any]],
    quote: dict[str, Any],
) -> dict[str, Any]:
    symbol = str(candidate.get("ticker") or candidate.get("asset") or "").upper()
    contract = candidate.get("strategy_timeframe_contract")
    contract = contract if isinstance(contract, dict) else {}
    timestamp = pd.to_datetime(candidate.get("signal_timestamp"), utc=True, errors="coerce")
    signal_age = None if pd.isna(timestamp) else max(0.0, (observed_at - timestamp).total_seconds())
    entry = candidate.get("preferred_entry")
    stop = candidate.get("stop_loss")
    try:
        risk_per_share = round(float(entry) - float(stop), 8)
    except (TypeError, ValueError):
        risk_per_share = None
    metadata = universe_metadata.get(symbol, {})
    quote_contract = quote.get("contract")
    quote_contract = quote_contract if isinstance(quote_contract, dict) else {}
    exact_quote = bool(
        quote.get("quote_valid") and str(quote_contract.get("symbol") or "").upper() == symbol
    )
    blockers = sorted(
        {
            *[str(value) for value in candidate.get("risks", [])],
            "NO_EXACT_CANDIDATE_VALIDATED_NETR",
            "NO_EXACT_CANDIDATE_BOUND_ML_PREDICTION",
            "NO_EXACT_CANDIDATE_BOUND_RL_ADVISORY",
            "SHARIAH_EXACT_CANDIDATE_ATTESTATION_REQUIRED",
            "EXACT_CANDIDATE_QUOTE_REQUIRED",
            "EXECUTION_AUTHORITY_NONE",
        }
    )
    return {
        "symbol": symbol,
        "setup_id": candidate.get("setup_id"),
        "candidate_unit": candidate.get("candidate_unit"),
        "asset_class": metadata.get("asset_type", candidate.get("asset_type", "UNKNOWN")),
        "discovery_reason": candidate.get("reasons", []),
        "strategy": {
            "strategy_id": candidate.get("strategy_id"),
            "strategy_family": candidate.get("strategy_family"),
        },
        "entry_timeframe": contract.get("entry_timeframe"),
        "setup_timeframe": contract.get("setup_timeframe"),
        "higher_timeframe_context": {
            "policy": candidate.get("higher_timeframe_context"),
            "required_timeframes": contract.get("required_timeframes", []),
            "optional_timeframes": contract.get("optional_timeframes", []),
            "evidence": candidate.get("timeframe_evidence", {}),
        },
        "signal_age_seconds": signal_age,
        "expected_net_return": {
            "value": None,
            "status": "UNAVAILABLE_NO_EXACT_CANDIDATE_VALIDATION",
        },
        "ml_prediction": {
            "value": None,
            "status": "UNAVAILABLE_NO_EXACT_CANDIDATE_BINDING",
        },
        "rl_advisory": {
            "action": None,
            "status": "UNAVAILABLE_NO_EXACT_CANDIDATE_BINDING",
        },
        "whole_share_quantity": int(candidate.get("suggested_quantity", 0) or 0),
        "risk": {
            "entry_reference": entry,
            "stop": stop,
            "targets": [
                candidate.get("take_profit_1"),
                candidate.get("take_profit_2"),
            ],
            "risk_per_share": risk_per_share,
            "maximum_planned_loss_eur": candidate.get("maximum_planned_loss_eur", 0),
        },
        "costs": {
            "estimated_transaction_costs_eur": candidate.get("estimated_transaction_costs_eur", 0),
            "status": "UNVALIDATED_OBSERVER_ESTIMATE",
        },
        "portfolio_fit": {
            "eligible": candidate.get("portfolio_eligible") is True,
            "status": candidate.get("qualification_status", "UNQUALIFIED_FORWARD_OBSERVER"),
        },
        "shariah": {
            "status": "EXACT_CANDIDATE_ATTESTATION_REQUIRED",
            "broad_metadata_status": metadata.get("shariah_status"),
        },
        "authority": {
            "strategy": candidate.get("strategy_authority", "NONE"),
            "execution": candidate.get("execution_authority", "NONE"),
        },
        "quote_status": {
            "status": (quote.get("status") if exact_quote else "NOT_OBSERVED_FOR_EXACT_CANDIDATE"),
            "quote_valid": exact_quote,
        },
        "final_decision": {
            "action": "NO_TRADE",
            "status": "UNQUALIFIED_FORWARD_OBSERVER",
            "blockers": blockers,
        },
    }


def _required_active_swing_report(
    *,
    timeframe_health: dict[str, dict[str, Any]],
    signals: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    candidate_generation: dict[str, Any],
    current_opportunities: dict[str, Any],
    positions: list[dict[str, Any]],
    position_management: dict[str, Any],
    desired_targets: dict[str, Any],
    ai: dict[str, Any],
    active_swing_ml: dict[str, Any],
    active_swing_inference: dict[str, Any],
    active_swing_tournament: dict[str, Any],
    rl: dict[str, Any],
    rl_rotation: dict[str, Any],
    execution: dict[str, Any],
    quote: dict[str, Any],
    performance: dict[str, Any],
    readiness: dict[str, bool],
) -> dict[str, Any]:
    (
        architectures,
        undeclared_contracts,
        explicit_strategy_count,
        undeclared_strategy_count,
    ) = _strategy_architectures(
        signals,
        candidate_generation.get("declared_strategy_architectures", []),
    )
    lifecycle = Counter(str(row.get("lifecycle_state") or "UNDECLARED") for row in ranked)
    variants = active_swing_tournament.get("timeframe_ablations")
    if not isinstance(variants, list):
        legacy_ablations = ai.get("timeframe_ablations")
        legacy_ablations = (
            legacy_ablations if isinstance(legacy_ablations, dict) else {}
        )
        variants = legacy_ablations.get("variants")
        variants = variants if isinstance(variants, list) else []
    realized_count = int(performance.get("realized_fact_count", 0) or 0)
    current_rows = current_opportunities.get("highest_ranked_opportunities", [])
    current_rows = current_rows if isinstance(current_rows, list) else []
    current_counts = Counter(
        str(row.get("entry_timeframe") or "UNDECLARED").lower()
        for row in current_rows
        if isinstance(row, dict)
    )
    research_signal_counts = Counter(
        str(row.get("timeframe") or "UNDECLARED").lower()
        for row in signals
        if isinstance(row, dict)
    )
    rl_policy_readiness = rl_rotation.get("policy_readiness")
    rl_policy_readiness = (
        rl_policy_readiness if isinstance(rl_policy_readiness, dict) else {}
    )
    economics_status = (
        "OBSERVED_REALIZED_ECONOMICS_AVAILABLE"
        if realized_count > 0
        else "UNPROVEN_NO_REALIZED_ROUND_TRIP"
    )
    return {
        "schema": "required_active_swing_report_v1",
        "status": "GO" if readiness.get("SWING_PRODUCT_READY") else "NO_GO",
        "A_TIMEFRAME_PIPELINE": {
            "supported_intervals": list(SUPPORTED_SWING_TIMEFRAMES),
            "core_intervals": list(CORE_SWING_TIMEFRAMES),
            "health": timeframe_health,
            "roles": TIMEFRAME_ROLES,
            "closed_bar_only": True,
        },
        "B_STRATEGY_ARCHITECTURES": {
            "count": sum(architectures.values()),
            "counts_by_explicit_entry_setup_context": dict(sorted(architectures.items())),
            "signals_without_explicit_contract": undeclared_contracts,
            "signals_with_explicit_contract": sum(
                isinstance(row.get("strategy_timeframe_contract"), dict) for row in signals
            ),
            "explicit_strategy_count": explicit_strategy_count,
            "unique_strategies_without_explicit_contract": (undeclared_strategy_count),
            "architecture_combination_count": len(architectures),
            "counting_unit": "ONE_UNIQUE_STRATEGY_DNA_NOT_SIGNAL_INSTANCE",
            "bare_timeframes_are_not_architecture": True,
        },
        "C_CURRENT_OPPORTUNITIES": {
            **current_opportunities,
            "natural_candidate_counts_by_entry_timeframe": {
                timeframe: int(current_counts.get(timeframe, 0))
                for timeframe in ("15m", "1h", "2h", "4h", "1d")
            },
            "daily_derived_natural_candidate_count": int(
                current_counts.get("1d", 0)
            ),
            "research_signal_counts_by_timeframe_not_current_opportunities": dict(
                sorted(research_signal_counts.items())
            ),
        },
        "D_NEAR_SETUPS": {
            "count": (
                int(candidate_generation.get("near_setup_count", 0) or 0)
                if "near_setup_count" in candidate_generation
                else lifecycle["NEAR_SETUP"]
            ),
            "current_pre_trigger_observers": candidate_generation.get("near_setups", []),
            "persisted_observer_count": int(candidate_generation.get("near_setup_count", 0) or 0),
            "promoted_in_latest_scan_count": int(
                candidate_generation.get("near_setup_promoted_count", 0) or 0
            ),
            "expired_in_latest_scan_count": int(
                candidate_generation.get("near_setup_expired_count", 0) or 0
            ),
            "entry_ready_count": lifecycle["ENTRY_READY"],
            "lifecycle_counts": dict(sorted(lifecycle.items())),
            "older_research_rows_are_not_current_triggers": True,
            "near_setups_are_not_natural_candidates": True,
            "near_setups_submit_orders": False,
        },
        "E_PORTFOLIO": {
            "current_position_count": len(positions),
            "position_management": position_management,
            "observer_desired_target_count": int(
                desired_targets.get("desired_target_count", 0) or 0
            ),
            "observer_targets_submit_orders": desired_targets.get("submits_orders") is True,
            "rotation_decision_count": len(desired_targets.get("rotation_decisions", [])),
            "cash_is_valid_when_no_edge": True,
            "cash_status": (
                "CASH_NO_AUTHORIZED_EDGE"
                if not positions and not current_rows
                else "PORTFOLIO_STATE_PRESENT"
            ),
            "execution_authority": "NONE",
        },
        "F_ML": {
            "promotion_status": active_swing_tournament.get(
                "promotion_status", "UNAVAILABLE"
            ),
            "performance_gate_go": active_swing_tournament.get(
                "performance_gate_go"
            ) is True,
            "forward_evidence_go": active_swing_tournament.get(
                "forward_evidence_go"
            ) is True,
            "active_swing_candidate_unit_go": active_swing_ml.get(
                "active_swing_candidate_unit_go"
            ) is True,
            "candidate_unit_required": "ONE_NATURAL_STRATEGY_SETUP",
            "candidate_conditioned_panel_status": active_swing_ml.get(
                "status", "UNAVAILABLE"
            ),
            "candidate_conditioned_labeled_row_count": int(
                active_swing_ml.get("row_count", 0) or 0
            ),
            "candidate_conditioned_training_ready": active_swing_ml.get(
                "training_ready"
            ) is True,
            "candidate_conditioned_training_blockers": active_swing_ml.get(
                "training_blockers", []
            ),
            "current_candidate_model_status": active_swing_inference.get(
                "status", "UNAVAILABLE"
            ),
            "current_candidate_model_compatible": active_swing_inference.get(
                "compatible_model"
            ) is True,
            "current_candidate_model_evidence_count": int(
                active_swing_inference.get("evidence_count", 0) or 0
            ),
            "timeframe_ablation_variant_count": len(variants),
            "timeframe_ablation_eligible_count": sum(
                row.get("financial_promotion_eligible") is True
                for row in variants
                if isinstance(row, dict)
            ),
            "timeframe_ablations": variants,
            "cross_timeframe_interaction_trials": active_swing_tournament.get(
                "interaction_trials", []
            ),
            "incremental_performance_by_timeframe_architecture_status": (
                "UNAVAILABLE_NO_ELIGIBLE_FIXED_OOS_ABLATION"
                if not any(
                    row.get("financial_promotion_eligible") is True
                    for row in variants
                    if isinstance(row, dict)
                )
                else "AVAILABLE"
            ),
            "incremental_metrics_required": [
                "OUTER_OOS_NET_R",
                "RANK_IC",
                "DRAWDOWN",
                "TURNOVER",
                "COVERAGE",
            ],
            "candidate_bound_inference_required": True,
        },
        "G_RL": {
            "status": rl.get("status", "UNAVAILABLE"),
            "promotion_status": rl.get("promotion_status", "UNAVAILABLE"),
            "episodes": int(rl.get("episodes", 0) or 0),
            "closed_episodes": int(rl.get("closed_episodes", 0) or 0),
            "action_distribution": rl.get("action_distribution", {}),
            "incremental_evidence_go": rl.get("incremental_evidence_go") is True,
            "forward_evidence_go": rl.get("forward_evidence_go") is True,
            "live_enabled": rl.get("rl_live_enabled") is True,
            "portfolio_rotation_status": rl_rotation.get(
                "status", "UNAVAILABLE"
            ),
            "portfolio_rotation_shadow_action": rl_rotation.get(
                "shadow_action", "CASH"
            ),
            "candidate_unit": rl_policy_readiness.get(
                "candidate_unit_definition", "ONE_NATURAL_STRATEGY_SETUP"
            ),
            "candidate_identity_deduplicated": rl_policy_readiness.get(
                "candidate_identity_deduplicated"
            )
            is True,
            "architecture_binding_complete_count": int(
                rl_policy_readiness.get("architecture_bound_observation_count", 0)
                or 0
            ),
            "architecture_binding_incomplete_count": int(
                rl_policy_readiness.get("architecture_binding_incomplete_count", 0)
                or 0
            ),
            "entry_management_performance_by_timeframe_architecture": {},
            "architecture_performance_status": (
                "UNAVAILABLE_NO_CLOSED_BOUND_CANDIDATE_OUTCOMES"
                if int(rl_policy_readiness.get("closed_candidate_outcome_count", 0) or 0)
                == 0
                else "AVAILABLE"
            ),
        },
        "H_EXECUTION": {
            "mode": execution.get("mode", "UNAVAILABLE"),
            "execution_authority": execution.get("execution_authority", "NONE"),
            "paper_fill_close_canary_go": execution.get("paper_fill_close_canary_go") is True,
            "paper_reconciliation_go": execution.get("paper_reconciliation_go") is True,
            "quote_status": quote.get("status", "UNAVAILABLE"),
            "quote_valid": quote.get("quote_valid") is True,
            "signal_to_fill_degradation_status": (
                "UNPROVEN_NO_REALIZED_ROUND_TRIP" if realized_count == 0 else "OBSERVED"
            ),
            "spread_status": "UNAVAILABLE_NO_EXACT_CURRENT_QUOTE",
            "slippage_status": "UNPROVEN_NO_REALIZED_ROUND_TRIP",
            "cost_status": "UNPROVEN_NO_REALIZED_ROUND_TRIP",
            "orders_generated": 0,
            "broker_write_calls": 0,
        },
        "I_ECONOMICS": {
            "status": economics_status,
            "realized_round_trip_count": realized_count,
            "net_r": None if realized_count == 0 else "PRIVATE_DERIVED_MEASURE",
            "expectancy": None if realized_count == 0 else "PRIVATE_DERIVED_MEASURE",
            "drawdown": None if realized_count == 0 else "PRIVATE_DERIVED_MEASURE",
            "capital_efficiency": (None if realized_count == 0 else "PRIVATE_DERIVED_MEASURE"),
            "financial_values_public": False,
            "historical_estimates_are_not_realized_profit": True,
            "timeframe_architecture_attribution_status": (
                "UNAVAILABLE_NO_REALIZED_ROUND_TRIP"
                if realized_count == 0
                else "AVAILABLE_PRIVATE_DERIVED_MEASURE"
            ),
        },
        "J_PRODUCT_STATUS": dict(readiness),
        "execution_authority": "NONE",
        "orders_generated": 0,
        "broker_write_calls": 0,
    }


def _strategy_architectures(
    signals: list[dict[str, Any]],
    declared_architectures: Iterable[dict[str, Any]] = (),
) -> tuple[Counter[str], int, int, int]:
    counts: Counter[str] = Counter()
    undeclared_signal_rows = 0
    undeclared_strategies: set[str] = set()
    explicit_strategies: set[str] = set()
    definitions = [
        *[row for row in declared_architectures if isinstance(row, dict)],
        *signals,
    ]
    for index, signal in enumerate(definitions):
        contract = signal.get("strategy_timeframe_contract")
        if not isinstance(contract, dict):
            if index >= len(definitions) - len(signals):
                undeclared_signal_rows += 1
                undeclared_strategies.add(
                    str(
                        signal.get("strategy_id")
                        or signal.get("strategy_dna_hash")
                        or "UNIDENTIFIED"
                    )
                )
            continue
        entry = str(contract.get("entry_timeframe") or "").strip()
        setup = str(contract.get("setup_timeframe") or "").strip()
        context = contract.get("context_timeframes")
        if not isinstance(context, list):
            context = contract.get("optional_timeframes", [])
        if not entry or not setup or not isinstance(context, list):
            if index >= len(definitions) - len(signals):
                undeclared_signal_rows += 1
                undeclared_strategies.add(
                    str(
                        signal.get("strategy_id")
                        or signal.get("strategy_dna_hash")
                        or "UNIDENTIFIED"
                    )
                )
            continue
        strategy_identity = str(
            signal.get("strategy_id") or signal.get("strategy_dna_hash") or stable_hash(contract)
        )
        if strategy_identity in explicit_strategies:
            continue
        explicit_strategies.add(strategy_identity)
        context_key = "+".join(sorted(str(value) for value in context)) or "NONE"
        counts[f"{entry}_ENTRY__{setup}_SETUP__{context_key}_CONTEXT"] += 1
    return (
        counts,
        undeclared_signal_rows,
        len(explicit_strategies),
        len(undeclared_strategies),
    )


def _current_broker_read_only_usable(
    payload: dict[str, Any],
    *,
    observed_at: datetime,
) -> bool:
    if payload.get("schema") != "phase9_reconciliation_audit_v1":
        return False
    read_counters = payload.get("read_only_request_counters")
    write_counters = payload.get("broker_write_counters")
    if not isinstance(read_counters, dict) or not isinstance(write_counters, dict):
        return False
    generated_at = pd.to_datetime(payload.get("generated_at"), utc=True, errors="coerce")
    observed_timestamp = pd.Timestamp(observed_at)
    observed_timestamp = (
        observed_timestamp.tz_localize("UTC")
        if observed_timestamp.tzinfo is None
        else observed_timestamp.tz_convert("UTC")
    )
    if pd.isna(generated_at):
        return False
    age = observed_timestamp - generated_at
    required_reads = {
        "read_only_account_summary_requests",
        "read_only_all_api_open_order_requests",
        "read_only_execution_requests",
        "read_only_position_requests",
        "read_only_same_client_open_order_requests",
    }
    return (
        -MAX_CONCURRENT_ARTIFACT_SKEW <= age <= pd.Timedelta(minutes=15)
        and str(payload.get("broker_observation_status", "")).upper() == "GO"
        and str(payload.get("broker_snapshot_status", "")).upper() == "BROKER_SNAPSHOT_OBSERVED"
        and str(payload.get("operational_broker_state_status", "")).upper()
        == "CURRENT_BROKER_FLAT_READ_ONLY"
        and int(payload.get("broker_position_count", 0) or 0) == 0
        and int(payload.get("broker_open_order_count", 0) or 0) == 0
        and all(int(read_counters.get(name, 0) or 0) >= 1 for name in required_reads)
        and all(int(value or 0) == 0 for value in write_counters.values())
        and str(payload.get("execution_authority", "NONE")).upper() == "NONE"
        and int(payload.get("paper_place_order_calls", 0) or 0) == 0
        and int(payload.get("live_place_order_calls", 0) or 0) == 0
        and payload.get("automatic_submission") is False
    )


def _operational_runtime_usable(
    heartbeat: dict[str, Any],
    machine: dict[str, Any],
) -> bool:
    heartbeat_status = str(heartbeat.get("status") or heartbeat.get("runtime_status") or "").upper()
    machine_status = str(machine.get("status") or "").upper()
    blockers = {str(value) for value in machine.get("last_cycle_blockers", [])}
    evidence_only_blockers = {
        "OPERATIONAL_STEP_P4_EVIDENCE_NO_GO",
        "OPERATIONAL_STEP_RL_SHADOW_INSUFFICIENT_EVIDENCE",
    }
    control = _runtime_control_state(heartbeat, machine)
    return bool(
        heartbeat_status in {"GO", "HEALTHY", "RUNNING", "DEGRADED"}
        and machine_status in {"GO", "HEALTHY", "RUNNING", "DEGRADED"}
        and control["enabled"] is True
        and control["paused"] is not True
        and blockers <= evidence_only_blockers
    )


def _runtime_control_state(
    heartbeat: dict[str, Any],
    machine: dict[str, Any],
) -> dict[str, Any]:
    """Prefer an explicitly newer heartbeat over a stale cycle snapshot."""
    heartbeat_at = pd.to_datetime(
        heartbeat.get("last_heartbeat") or heartbeat.get("updated_at"),
        utc=True,
        errors="coerce",
    )
    machine_at = pd.to_datetime(machine.get("last_heartbeat"), utc=True, errors="coerce")
    heartbeat_is_newer = bool(
        not pd.isna(heartbeat_at)
        and (pd.isna(machine_at) or heartbeat_at > machine_at)
        and "enabled" in heartbeat
        and "paused" in heartbeat
    )
    source = heartbeat if heartbeat_is_newer else machine
    return {
        "source": ("NEWER_RUNTIME_HEARTBEAT" if heartbeat_is_newer else "MACHINE_STATUS"),
        "enabled": source.get("enabled") is True,
        "paused": source.get("paused") is True,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["publish_active_swing_product_status"]
