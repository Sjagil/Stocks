from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from filelock import FileLock

from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_9 import _load_current_frames


PRIVATE_ROOT = Path("data/market_context/private")
OUTPUT_ROOT = Path("output/market_context")
TERMINAL_STATES = frozenset(
    {
        "NO_FILL_EXPIRED",
        "STOPPED",
        "TP1_EXIT",
        "TP2_EXIT",
        "TRAIL_EXIT",
        "TIME_EXIT",
        "INVALIDATED_BEFORE_ENTRY",
        "DATA_FAILURE",
    }
)
ENTRY_WINDOW_BARS = {
    "1h": 4,
    "2h": 3,
    "4h": 2,
    "1d": 2,
    "1w": 1,
    "1mo": 1,
}
MAX_HOLDING_BARS = {
    "1h": 40,
    "2h": 24,
    "4h": 15,
    "1d": 10,
    "1w": 8,
    "1mo": 3,
}
ENTRY_EXPIRY = {
    "1h": timedelta(days=2),
    "2h": timedelta(days=3),
    "4h": timedelta(days=4),
    "1d": timedelta(days=5),
    "1w": timedelta(days=14),
    "1mo": timedelta(days=45),
}
ROUND_TRIP_COST_BPS = {
    "STOCK": 10.0,
    "ETF": 8.0,
    "COMMODITY_PROXY": 12.0,
}


def settle_entry_episodes(
    project_root: Path,
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    now = _utc(observed_at or datetime.now(UTC))
    episode_path = project_root / PRIVATE_ROOT / "entry-episodes.jsonl"
    outcome_path = project_root / PRIVATE_ROOT / "entry-episode-outcomes.jsonl"
    revision_path = (
        project_root
        / PRIVATE_ROOT
        / "entry-episode-outcome-revisions.jsonl"
    )
    episodes = _read_jsonl(episode_path)
    current_episodes = [
        row
        for row in episodes
        if row.get("schema") == "active_swing_forward_episode_v1"
        and row.get("feature_snapshot_hash")
    ]
    legacy_episode_ids = {
        str(row.get("episode_id"))
        for row in episodes
        if (
            row.get("schema") != "active_swing_forward_episode_v1"
            or not row.get("feature_snapshot_hash")
        )
        and row.get("episode_id")
    }
    _quarantine_noncanonical_outcomes(
        outcome_path,
        legacy_episode_ids=legacy_episode_ids,
    )
    existing = {
        str(row.get("episode_id")): row
        for row in _read_jsonl(outcome_path)
        if row.get("episode_id")
    }
    pending = [
        row
        for row in current_episodes
        if row.get("episode_id") and str(row["episode_id"]) not in existing
    ]
    revisions = {
        str(row.get("episode_id")): row
        for row in _read_jsonl(revision_path)
        if row.get("episode_id")
        and row.get("research_revision_accepted") is True
    }
    revisable = [
        row
        for row in current_episodes
        if row.get("episode_id")
        and str(row["episode_id"]) not in revisions
        and _research_observation_eligible(row)
        and existing.get(str(row["episode_id"]), {}).get(
            "outcome_classification"
        )
        == "HARD_VETO_BEFORE_ENTRY"
    ]
    frames = (
        _load_episode_frames(
            project_root,
            episodes=[*pending, *revisable],
            observed_at=now,
        )
        if pending or revisable
        else {}
    )
    evaluated: list[dict[str, Any]] = []
    terminal: list[dict[str, Any]] = []
    for episode in pending:
        timeframe = str(episode.get("timeframe") or "")
        symbol = str(episode.get("symbol") or "").upper()
        frame = frames.get(timeframe, {}).get(symbol)
        outcome = _evaluate_episode(
            episode,
            frame=frame,
            evaluated_at=now,
        )
        evaluated.append(outcome)
        if outcome["terminal"]:
            terminal.append(outcome)
    _append_terminal_outcomes(outcome_path, terminal)
    research_revisions: list[dict[str, Any]] = []
    for episode in revisable:
        timeframe = str(episode.get("timeframe") or "")
        symbol = str(episode.get("symbol") or "").upper()
        revised = _evaluate_episode(
            episode,
            frame=frames.get(timeframe, {}).get(symbol),
            evaluated_at=now,
        )
        if not revised["terminal"]:
            continue
        original = existing[str(episode["episode_id"])]
        revision = {
            **revised,
            "research_revision_accepted": True,
            "revision_reason": (
                "EXECUTION_ONLY_CONTRACT_VETO_RECLASSIFIED_FOR_RESEARCH"
            ),
            "supersedes_outcome_hash": original.get("outcome_hash"),
            "original_terminal_status": original.get("terminal_status"),
            "original_outcome_classification": original.get(
                "outcome_classification"
            ),
            "canonical_evidence_replaced": False,
            "execution_authority": "NONE",
        }
        revision["revision_hash"] = stable_hash(revision)
        research_revisions.append(revision)
    _append_terminal_outcomes(revision_path, research_revisions)
    revisions.update(
        {
            str(row["episode_id"]): row
            for row in research_revisions
        }
    )
    all_outcomes = list(existing.values()) + terminal
    terminal_ids = {
        str(row.get("episode_id")) for row in all_outcomes if row.get("episode_id")
    }
    pending_results = [row for row in evaluated if not row["terminal"]]
    expired_count = len(terminal_ids) + sum(
        bool(row.get("eligible_for_terminal_completeness"))
        for row in pending_results
    )
    completed_count = len(terminal_ids)
    completion_ratio = (
        completed_count / expired_count if expired_count else 1.0
    )
    state_counts = _counts(
        str(row.get("terminal_status")) for row in all_outcomes
    )
    status = (
        "GO"
        if completion_ratio >= 0.99
        else "DEGRADED_EPISODE_COMPLETENESS"
    )
    report = {
        "schema": "active_swing_episode_completeness_v1",
        "status": status,
        "generated_at": now.isoformat(),
        "episode_count": len(current_episodes),
        "legacy_episode_count": len(episodes) - len(current_episodes),
        "terminal_episode_count": completed_count,
        "new_terminal_episode_count": len(terminal),
        "research_revision_count": len(revisions),
        "new_research_revision_count": len(research_revisions),
        "research_revision_state_counts": _counts(
            str(row.get("terminal_status")) for row in revisions.values()
        ),
        "pending_episode_count": len(current_episodes) - completed_count,
        "research_observation_eligible_episode_count": sum(
            _research_observation_eligible(row) for row in current_episodes
        ),
        "brokerability_blocked_research_eligible_count": sum(
            _research_observation_eligible(row)
            and not bool(
                _mapping(row.get("decision_contract"))
                .get("gates", {})
                .get("contract_resolved")
            )
            for row in current_episodes
        ),
        "expired_episode_count": expired_count,
        "completion_ratio": round(completion_ratio, 8),
        "terminal_state_counts": state_counts,
        "pending_reason_counts": _counts(
            str(row.get("pending_reason")) for row in pending_results
        ),
        "duplicate_terminal_count": 0,
        "feature_mutation_count": sum(
            row.get("outcome_classification")
            == "FEATURE_SNAPSHOT_MUTATED"
            for row in terminal
        ),
        "intrabar_path_ambiguous_count": sum(
            row.get("outcome_classification")
            == "INTRABAR_PATH_AMBIGUOUS"
            for row in all_outcomes
        ),
        "mfe_mae_scope": "POST_FILL_FROM_NEXT_BAR",
        "missed_entries_counted_as_pnl": False,
        "feature_snapshots_mutated": False,
        "private_episode_store": str(episode_path),
        "private_outcome_store": str(outcome_path),
        "private_research_revision_store": str(revision_path),
        "canonical_outcomes_replaced": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "order_calls": 0,
    }
    report["content_hash"] = stable_hash(report)
    _write_json(project_root / OUTPUT_ROOT / "entry-episode-completeness.json", report)
    return report


def episode_outcome_status(project_root: Path) -> dict[str, Any]:
    path = project_root / OUTPUT_ROOT / "entry-episode-completeness.json"
    if not path.is_file():
        return {
            "schema": "active_swing_episode_completeness_v1",
            "status": "NOT_RUN",
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "order_calls": 0,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {"status": "INVALID"}


def _evaluate_episode(
    episode: Mapping[str, Any],
    *,
    frame: pd.DataFrame | None,
    evaluated_at: datetime,
    counterfactual_long: bool = False,
) -> dict[str, Any]:
    episode_id = str(episode.get("episode_id"))
    decision = _timestamp(
        episode.get("decision_timestamp") or episode.get("observed_at")
    )
    timeframe = str(episode.get("timeframe") or "")
    decision_contract = _mapping(episode.get("decision_contract"))
    context_snapshot = _mapping(episode.get("context_snapshot"))
    setup_snapshot = _mapping(episode.get("setup_snapshot"))
    asset_profile = _mapping(decision_contract.get("asset_profile"))
    asset_metadata = _mapping(setup_snapshot.get("asset_metadata"))
    event_risk = _mapping(context_snapshot.get("event_risk"))
    context_components = _mapping(context_snapshot.get("component_scores"))
    earnings_distance = _number_or_none(
        context_snapshot.get("earnings_distance_days")
        or setup_snapshot.get("earnings_distance_days")
    )
    base = {
        "schema": "active_swing_forward_episode_outcome_v1",
        "episode_id": episode_id,
        "feature_snapshot_hash": episode.get("feature_snapshot_hash"),
        "decision_timestamp": decision.isoformat() if decision else None,
        "signal_timestamp": episode.get("signal_timestamp"),
        "evaluated_at": evaluated_at.isoformat(),
        "timeframe": timeframe,
        "asset_class": (
            episode.get("setup_snapshot", {}).get("asset_class")
        ),
        "symbol_fingerprint": stable_hash(
            {"symbol": str(episode.get("symbol") or "").upper()}
        )[:20],
        "strategy_id": episode.get("strategy_id"),
        "label_source": "COUNTERFACTUAL_BAR_PATH_OBSERVATION",
        "canonical_close": False,
        "canonical_fill_evidence": False,
        "research_observation_eligible": _research_observation_eligible(
            episode
        ),
        "brokerability_status": decision_contract.get(
            "brokerability_status",
            "BROKERABILITY_UNAVAILABLE_AT_DECISION",
        ),
        "outcome_bar_source": (
            frame.attrs.get("provider")
            if frame is not None
            else None
        ),
        "outcome_bar_source_path_hash": (
            frame.attrs.get("source_path_hash")
            if frame is not None
            else None
        ),
        "decision_context_hash": stable_hash(context_snapshot),
        "market_regime": (
            decision_contract.get("market_regime")
            or context_snapshot.get("regime")
            or "UNAVAILABLE_AT_DECISION"
        ),
        "sector": (
            asset_metadata.get("sector")
            or asset_profile.get("sector")
            or "UNAVAILABLE_AT_DECISION"
        ),
        "region": (
            asset_metadata.get("region")
            or asset_profile.get("region")
            or "UNAVAILABLE_AT_DECISION"
        ),
        "event_risk_score": _number_or_none(event_risk.get("risk_score")),
        "event_risk_blocks_new_entry": (
            bool(event_risk.get("blocks_new_entry"))
            if "blocks_new_entry" in event_risk
            else None
        ),
        "earnings_distance_days": earnings_distance,
        "earnings_context_status": (
            "AVAILABLE_AT_DECISION"
            if earnings_distance is not None
            else "UNAVAILABLE_AT_DECISION"
        ),
        "gex_context_score_at_decision": _number_or_none(
            context_components.get("gex")
        ),
        "entry_confirmation_score": _number_or_none(
            decision_contract.get("entry_confirmation_score")
        ),
        "setup_score_within_family": _number_or_none(
            decision_contract.get("setup_score_within_family")
        ),
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "order_calls": 0,
    }
    feature_hash = stable_hash(
        {
            "decision_contract": episode.get("decision_contract"),
            "context_snapshot": episode.get("context_snapshot"),
            "setup_snapshot": episode.get("setup_snapshot"),
            "entry_snapshot": episode.get("entry_snapshot"),
        }
    )
    if feature_hash != episode.get("feature_snapshot_hash"):
        return _terminal(
            base,
            "DATA_FAILURE",
            "FEATURE_SNAPSHOT_MUTATED",
        )
    setup = setup_snapshot
    proposed = episode.get("proposed_order", {})
    action = str(setup.get("action") or "").upper()
    limit_price = _number(proposed.get("limit_price"))
    maximum_valid_entry = _number(
        setup.get("entry_zone_high") or limit_price
    )
    stop = _number(setup.get("stop_loss"))
    tp1 = _number(setup.get("take_profit_1"))
    tp2 = _number(setup.get("take_profit_2"))
    if decision is None or timeframe not in ENTRY_WINDOW_BARS:
        return _terminal(base, "DATA_FAILURE", "INVALID_EPISODE_CLOCK")
    expiry = decision + ENTRY_EXPIRY[timeframe]
    base.update(
        {
            "entry_window_start": decision.isoformat(),
            "entry_window_end": expiry.isoformat(),
            "proposed_limit": limit_price,
            "maximum_valid_entry": maximum_valid_entry,
            "spread_at_decision_bps": _number_or_none(
                episode.get("entry_snapshot", {})
                .get("tape", {})
                .get("spread_bps")
            ),
        }
    )
    if (
        str(episode.get("state")) == "WATCHLIST_HARD_VETO_BLOCKED"
        and not _research_observation_eligible(episode)
    ):
        return _terminal(
            base,
            "INVALIDATED_BEFORE_ENTRY",
            "HARD_VETO_BEFORE_ENTRY",
        )
    if (
        action not in {"BUY", "LONG", "WATCHLIST"}
        and not counterfactual_long
    ):
        return _terminal(
            base,
            "INVALIDATED_BEFORE_ENTRY",
            "LONG_ONLY_SETUP_NOT_ACTIVE",
        )
    if not (
        limit_price > 0
        and maximum_valid_entry >= limit_price
        and 0 < stop < limit_price < tp1
        and (tp2 == 0 or tp2 >= tp1)
    ):
        return _terminal(base, "DATA_FAILURE", "INVALID_PRICE_ENVELOPE")
    prepared = _prepare_frame(frame)
    if prepared.empty:
        if evaluated_at >= expiry:
            return _terminal(base, "DATA_FAILURE", "BAR_DATA_UNAVAILABLE")
        return _pending(base, "AWAITING_BAR_DATA")
    future = prepared.loc[prepared.index > pd.Timestamp(decision)]
    entry_window = future.head(ENTRY_WINDOW_BARS[timeframe])
    if entry_window.empty:
        if evaluated_at >= expiry:
            return _terminal(base, "NO_FILL_EXPIRED", "NO_FUTURE_ENTRY_BAR")
        return _pending(base, "AWAITING_ENTRY_WINDOW")
    fill_timestamp: pd.Timestamp | None = None
    fill_price: float | None = None
    for timestamp, bar in entry_window.iterrows():
        bar_open = float(bar["open"])
        bar_low = float(bar["low"])
        if bar_low <= stop and bar_low <= limit_price:
            return _terminal(
                base,
                "DATA_FAILURE",
                "INTRABAR_PATH_AMBIGUOUS",
            )
        if bar_open > maximum_valid_entry:
            continue
        if bar_low <= limit_price:
            fill_timestamp = timestamp
            fill_price = min(bar_open, limit_price)
            break
    if fill_timestamp is None or fill_price is None:
        if len(entry_window) >= ENTRY_WINDOW_BARS[timeframe] or evaluated_at >= expiry:
            return _terminal(base, "NO_FILL_EXPIRED", "LIMIT_NOT_TOUCHED")
        return _pending(base, "ENTRY_WINDOW_INCOMPLETE")
    path = future.loc[future.index >= fill_timestamp].head(
        MAX_HOLDING_BARS[timeframe]
    )
    fill_bar = path.iloc[0]
    if float(fill_bar["low"]) <= stop or float(fill_bar["high"]) >= tp1:
        return _terminal(
            {
                **base,
                **_fill_fields(episode, fill_timestamp, fill_price),
            },
            "DATA_FAILURE",
            "INTRABAR_PATH_AMBIGUOUS",
        )
    exit_status: str | None = None
    exit_reason: str | None = None
    exit_timestamp: pd.Timestamp | None = None
    exit_price: float | None = None
    measured = path.iloc[1:]
    evaluated_bars: list[tuple[pd.Timestamp, pd.Series]] = []
    for bar_number, (timestamp, bar) in enumerate(
        measured.iterrows(), start=1
    ):
        evaluated_bars.append((timestamp, bar))
        bar_open = float(bar["open"])
        low = float(bar["low"])
        high = float(bar["high"])
        if bar_open <= stop:
            exit_status, exit_reason = "STOPPED", "GAP_THROUGH_STOP"
            exit_timestamp, exit_price = timestamp, bar_open
            break
        if tp2 > 0 and bar_open >= tp2:
            exit_status, exit_reason = "TP2_EXIT", "GAP_THROUGH_TP2"
            exit_timestamp, exit_price = timestamp, bar_open
            break
        if bar_open >= tp1:
            exit_status, exit_reason = "TP1_EXIT", "GAP_THROUGH_TP1"
            exit_timestamp, exit_price = timestamp, bar_open
            break
        stop_hit = low <= stop
        target_hit = high >= tp1
        if stop_hit and target_hit:
            return _terminal(
                {
                    **base,
                    **_fill_fields(episode, fill_timestamp, fill_price),
                },
                "DATA_FAILURE",
                "INTRABAR_PATH_AMBIGUOUS",
            )
        if stop_hit:
            exit_status, exit_reason = "STOPPED", "STOP_HIT"
            exit_timestamp, exit_price = timestamp, stop
            break
        if target_hit:
            exit_status, exit_reason = "TP1_EXIT", "TP1_HIT"
            exit_timestamp, exit_price = timestamp, tp1
            break
    if exit_status is None:
        if len(path) < MAX_HOLDING_BARS[timeframe]:
            return _pending(
                {
                    **base,
                    **_fill_fields(episode, fill_timestamp, fill_price),
                },
                "POSITION_EPISODE_OPEN",
            )
        last_timestamp = path.index[-1]
        exit_status, exit_reason = "TIME_EXIT", "MAX_HOLDING_BARS"
        exit_timestamp, exit_price = last_timestamp, float(path.iloc[-1]["close"])
    if exit_status is None or exit_timestamp is None or exit_price is None:
        raise RuntimeError("terminal episode is missing exit evidence")
    return _closed_trade_outcome(
        base,
        episode=episode,
        fill_timestamp=fill_timestamp,
        fill_price=fill_price,
        exit_status=exit_status,
        exit_reason=str(exit_reason),
        exit_timestamp=exit_timestamp,
        exit_price=float(exit_price),
        measured_bars=evaluated_bars,
    )


def _closed_trade_outcome(
    base: Mapping[str, Any],
    *,
    episode: Mapping[str, Any],
    fill_timestamp: pd.Timestamp,
    fill_price: float,
    exit_status: str,
    exit_reason: str,
    exit_timestamp: pd.Timestamp,
    exit_price: float,
    measured_bars: list[tuple[pd.Timestamp, pd.Series]],
) -> dict[str, Any]:
    setup = episode.get("setup_snapshot", {})
    stop = float(setup["stop_loss"])
    risk = fill_price - stop
    gross_r = (exit_price - fill_price) / risk
    asset_class = str(setup.get("asset_class") or "STOCK")
    cost_bps = ROUND_TRIP_COST_BPS.get(asset_class, 12.0)
    configured_cost = _number_or_none(
        setup.get("estimated_transaction_costs_eur")
    )
    commission = (
        configured_cost
        if configured_cost is not None
        else fill_price * cost_bps / 10_000.0
    )
    spread_bps = _number_or_none(base.get("spread_at_decision_bps")) or 0.0
    estimated_slippage = fill_price * spread_bps / 20_000.0
    net_r = gross_r - (commission + estimated_slippage) / risk
    lows = [float(bar["low"]) for _, bar in measured_bars]
    highs = [float(bar["high"]) for _, bar in measured_bars]
    mfe_index = max(range(len(highs)), key=highs.__getitem__) if highs else None
    mae_index = min(range(len(lows)), key=lows.__getitem__) if lows else None
    mfe_timestamp = (
        measured_bars[mfe_index][0] if mfe_index is not None else None
    )
    mae_timestamp = (
        measured_bars[mae_index][0] if mae_index is not None else None
    )
    maximum_favourable_excursion = (
        (max(highs) - fill_price) / risk if highs else 0.0
    )
    maximum_adverse_excursion = (
        (min(lows) - fill_price) / risk if lows else 0.0
    )
    gross_exit_capture_ratio = (
        gross_r / maximum_favourable_excursion
        if maximum_favourable_excursion > 0
        else None
    )
    holding_duration_seconds = int(
        (
            exit_timestamp.to_pydatetime()
            - fill_timestamp.to_pydatetime()
        ).total_seconds()
    )
    result = {
        **base,
        **_fill_fields(episode, fill_timestamp, fill_price),
        "terminal": True,
        "terminal_status": exit_status,
        "outcome_classification": exit_reason,
        "first_barrier_hit": exit_status,
        "exit_timestamp": exit_timestamp.isoformat(),
        "exit_price": round(exit_price, 8),
        "gross_R": round(gross_r, 8),
        "net_R": round(net_r, 8),
        "estimated_commission": round(commission, 8),
        "estimated_slippage": round(estimated_slippage, 8),
        "cost_model_bps_round_trip": cost_bps,
        "maximum_favourable_excursion": round(
            maximum_favourable_excursion, 8
        ),
        "maximum_adverse_excursion": round(
            maximum_adverse_excursion, 8
        ),
        "mfe_timestamp": (
            mfe_timestamp.isoformat() if mfe_timestamp is not None else None
        ),
        "mae_timestamp": (
            mae_timestamp.isoformat() if mae_timestamp is not None else None
        ),
        "time_to_mfe_seconds": _elapsed_seconds(
            fill_timestamp, mfe_timestamp
        ),
        "time_to_mae_seconds": _elapsed_seconds(
            fill_timestamp, mae_timestamp
        ),
        "time_to_first_barrier_seconds": holding_duration_seconds,
        "time_to_stop_seconds": (
            holding_duration_seconds if exit_status == "STOPPED" else None
        ),
        "gross_exit_capture_ratio": (
            round(gross_exit_capture_ratio, 8)
            if gross_exit_capture_ratio is not None
            else None
        ),
        "hypothetical_spread_at_decision_bps": spread_bps,
        "realized_spread_at_fill_bps": None,
        "spread_observation_status": (
            "DECISION_PROXY_ONLY_FILL_SPREAD_UNAVAILABLE"
            if spread_bps > 0
            else "SPREAD_UNAVAILABLE"
        ),
        "holding_duration_seconds": holding_duration_seconds,
        "eligible_for_terminal_completeness": True,
        "excluded_from_performance_metrics": False,
    }
    result["outcome_hash"] = stable_hash(result)
    return result


def _fill_fields(
    episode: Mapping[str, Any],
    timestamp: pd.Timestamp,
    price: float,
) -> dict[str, Any]:
    decision_price = _number_or_none(
        episode.get("setup_snapshot", {}).get("current_market_price")
    )
    return {
        "would_fill": True,
        "fill_timestamp": timestamp.isoformat(),
        "fill_price": round(price, 8),
        "fill_fraction": 1.0,
        "spread_at_fill_bps": None,
        "slippage_from_decision_bps": (
            round((price / decision_price - 1.0) * 10_000.0, 8)
            if decision_price and decision_price > 0
            else None
        ),
    }


def _terminal(
    base: Mapping[str, Any],
    terminal_status: str,
    classification: str,
) -> dict[str, Any]:
    if terminal_status not in TERMINAL_STATES:
        raise ValueError(f"invalid terminal status: {terminal_status}")
    result = {
        **base,
        "terminal": True,
        "terminal_status": terminal_status,
        "outcome_classification": classification,
        "would_fill": base.get("would_fill", False),
        "fill_timestamp": base.get("fill_timestamp"),
        "fill_price": base.get("fill_price"),
        "fill_fraction": base.get("fill_fraction"),
        "spread_at_fill_bps": base.get("spread_at_fill_bps"),
        "estimated_commission": None,
        "estimated_slippage": None,
        "maximum_favourable_excursion": None,
        "maximum_adverse_excursion": None,
        "mfe_timestamp": None,
        "mae_timestamp": None,
        "time_to_mfe_seconds": None,
        "time_to_mae_seconds": None,
        "time_to_first_barrier_seconds": None,
        "time_to_stop_seconds": None,
        "gross_exit_capture_ratio": None,
        "hypothetical_spread_at_decision_bps": base.get(
            "spread_at_decision_bps"
        ),
        "realized_spread_at_fill_bps": None,
        "spread_observation_status": (
            "DECISION_PROXY_ONLY_FILL_SPREAD_UNAVAILABLE"
            if _number_or_none(base.get("spread_at_decision_bps"))
            else "SPREAD_UNAVAILABLE"
        ),
        "first_barrier_hit": (
            "INTRABAR_PATH_AMBIGUOUS"
            if classification == "INTRABAR_PATH_AMBIGUOUS"
            else None
        ),
        "exit_timestamp": None,
        "exit_price": None,
        "gross_R": None,
        "net_R": None,
        "holding_duration_seconds": None,
        "eligible_for_terminal_completeness": True,
        "excluded_from_performance_metrics": True,
    }
    result["outcome_hash"] = stable_hash(result)
    return result


def _pending(base: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        **base,
        "terminal": False,
        "terminal_status": None,
        "pending_reason": reason,
        "eligible_for_terminal_completeness": False,
    }


def _prepare_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = ("open", "high", "low", "close")
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    prepared = frame.loc[:, required].copy()
    prepared.index = pd.to_datetime(prepared.index, utc=True, errors="coerce")
    prepared = prepared.loc[~prepared.index.isna()]
    for column in required:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna().sort_index()
    valid = (
        prepared["open"].gt(0)
        & prepared["high"].ge(prepared[["open", "close", "low"]].max(axis=1))
        & prepared["low"].le(prepared[["open", "close", "high"]].min(axis=1))
    )
    return prepared.loc[valid]


def _load_episode_frames(
    project_root: Path,
    *,
    episodes: list[Mapping[str, Any]],
    observed_at: datetime,
) -> dict[str, dict[str, pd.DataFrame]]:
    frames = _load_current_frames(project_root, observed_at=observed_at)
    requested = {
        (
            str(row.get("timeframe") or ""),
            str(row.get("symbol") or "").upper(),
        )
        for row in episodes
        if row.get("timeframe") and row.get("symbol")
    }
    for timeframe, symbol in sorted(requested):
        existing = frames.get(timeframe, {}).get(symbol)
        if existing is not None and not existing.empty:
            continue
        provider_frame = _load_single_provider_frame(
            project_root,
            symbol=symbol,
            timeframe=timeframe,
            observed_at=observed_at,
        )
        if provider_frame.empty:
            continue
        frames.setdefault(timeframe, {})[symbol] = provider_frame
    return frames


def _load_single_provider_frame(
    project_root: Path,
    *,
    symbol: str,
    timeframe: str,
    observed_at: datetime,
) -> pd.DataFrame:
    root = project_root / "data" / "research" / "multitimeframe" / "private"
    candidates: list[tuple[pd.Timestamp, int, str, Path, pd.DataFrame]] = []
    provider_priority = {
        "YFINANCE": 4,
        "STOCKS_YFINANCE_LOCAL": 3,
        "STOCKS_PIT_EODHD_LOCAL": 2,
        "DATASCRAPER_EODHD_EXPORT": 1,
    }
    for path in root.glob(
        f"provider=*/symbol={symbol}/interval={timeframe}/"
        "source_interval=*/bars.parquet"
    ):
        try:
            raw = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        timestamp_column = next(
            (
                column
                for column in ("timestamp_utc", "timestamp", "date")
                if column in raw.columns
            ),
            None,
        )
        if timestamp_column is None:
            continue
        raw = raw.copy()
        raw.index = pd.to_datetime(
            raw[timestamp_column], utc=True, errors="coerce"
        )
        prepared = _prepare_frame(raw)
        if prepared.empty:
            continue
        prepared = prepared.loc[
            prepared.index <= pd.Timestamp(observed_at)
        ]
        if prepared.empty:
            continue
        provider = next(
            (
                part.removeprefix("provider=")
                for part in path.parts
                if part.startswith("provider=")
            ),
            "UNKNOWN",
        )
        candidates.append(
            (
                pd.Timestamp(prepared.index.max()),
                provider_priority.get(provider, 0),
                provider,
                path,
                prepared,
            )
        )
    if not candidates:
        return pd.DataFrame()
    _, _, provider, source_path, selected = max(
        candidates,
        key=lambda item: (item[0], item[1], item[2]),
    )
    selected = selected.copy()
    selected.attrs["provider"] = provider
    selected.attrs["provider_selection_policy"] = (
        "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
    )
    selected.attrs["source_path_hash"] = hashlib.sha256(
        str(source_path.resolve()).encode("utf-8")
    ).hexdigest().upper()
    return selected


def _append_terminal_outcomes(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path) + ".lock", timeout=10)
    with lock:
        existing = {
            str(row.get("episode_id"))
            for row in _read_jsonl(path)
            if row.get("episode_id")
        }
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                episode_id = str(row["episode_id"])
                if episode_id in existing:
                    continue
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                existing.add(episode_id)


def _quarantine_noncanonical_outcomes(
    path: Path,
    *,
    legacy_episode_ids: set[str],
) -> Path | None:
    if not path.is_file():
        return None
    rows = _read_jsonl(path)
    if not any(str(row.get("episode_id")) in legacy_episode_ids for row in rows):
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    quarantine = path.parent / "quarantine" / f"{path.stem}-{digest[:24]}.jsonl"
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    if quarantine.exists():
        if hashlib.sha256(quarantine.read_bytes()).hexdigest().upper() != digest:
            raise RuntimeError("outcome quarantine hash collision")
        path.unlink()
    else:
        path.replace(quarantine)
    _write_json(
        quarantine.with_suffix(".manifest.json"),
        {
            "schema": "active_swing_outcome_quarantine_v1",
            "status": "QUARANTINED_NONCANONICAL_LEGACY_EPISODES",
            "content_hash": digest,
            "row_count": len(rows),
            "canonical_evidence": False,
            "execution_authority": "NONE",
            "broker_calls": 0,
        },
    )
    return quarantine


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _timestamp(value: Any) -> datetime | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _number(value: Any) -> float:
    number = _number_or_none(value)
    return number if number is not None else 0.0


def _number_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _research_observation_eligible(episode: Mapping[str, Any]) -> bool:
    decision = _mapping(episode.get("decision_contract"))
    explicit = decision.get("research_observation_eligible")
    if isinstance(explicit, bool):
        return explicit
    hard_vetoes = {
        str(value)
        for value in decision.get("hard_vetoes", [])
        if value
    }
    return bool(hard_vetoes) and hard_vetoes <= {
        "CONTRACT_IDENTITY_REQUIRED"
    }


def _elapsed_seconds(
    start: pd.Timestamp,
    end: pd.Timestamp | None,
) -> int | None:
    if end is None:
        return None
    return int((end.to_pydatetime() - start.to_pydatetime()).total_seconds())


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


__all__ = ["episode_outcome_status", "settle_entry_episodes"]
