from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from stocks.context.candidate_evidence import candidate_evidence_classification
from stocks.execution.idempotency import stable_hash
from stocks.microstructure.orderflow import (
    aggregate_trade_flow,
    classify_trades,
    orderbook_metrics,
)
from stocks.portfolio.swing import StrategyTimeframeContract
from stocks.research.sec_overlay import sec_overlays_for_signals
from stocks.signals.freshness import signal_is_current
from stocks.universe import broad_asset_metadata


PRIVATE_ROOT = Path("data/market_context/private")
OUTPUT_ROOT = Path("output/market_context")
ASSET_PROFILE_WEIGHTS = {
    "STOCK": {
        "trend": 20,
        "relative_strength": 15,
        "volume": 15,
        "orderflow": 15,
        "book": 10,
        "event": 10,
        "execution": 10,
        "macro": 5,
    },
    "ETF": {
        "trend": 15,
        "breadth": 15,
        "underlying": 15,
        "relative_strength": 15,
        "futures_confirmation": 10,
        "orderflow": 10,
        "fair_value": 10,
        "execution": 10,
    },
    "COMMODITY_PROXY": {
        "trend": 15,
        "curve": 15,
        "fundamentals": 15,
        "futures_orderflow": 15,
        "cross_market": 10,
        "relative_strength": 10,
        "proxy_quality": 10,
        "execution": 10,
    },
}


def observe_shortlist(
    project_root: Path,
    *,
    max_symbols: int = 20,
    depth_symbols: int = 5,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    if not 1 <= int(max_symbols) <= 50:
        raise ValueError("max_symbols must be between 1 and 50")
    if not 0 <= int(depth_symbols) <= int(max_symbols):
        raise ValueError("depth_symbols must be between 0 and max_symbols")
    now = _utc(observed_at or datetime.now(UTC))
    canonical_signals = _read_json(
        project_root / "output/signals/latest_signals.json"
    )
    tactical_signals = _read_json(
        project_root / "output/signals/active_swing_15m_signals.json"
    )
    signal_rows = _combined_signal_rows(
        canonical_signals.get("signals", []),
        tactical_signals.get("signals", []),
    )
    market_regime = str(
        _read_json(project_root / "output/dynamic/current_regime.json").get(
            "regime", "UNKNOWN"
        )
    ).upper()
    asset_context = _read_json(
        project_root / OUTPUT_ROOT / "asset-context.json"
    )
    context_map = {
        str(row.get("symbol", "")).upper(): row
        for row in asset_context.get("contexts", [])
        if isinstance(row, dict) and row.get("symbol")
    }
    asset_metadata = broad_asset_metadata(project_root)
    candidate_rows = _candidate_signals(signal_rows)
    selected = _select_signals(candidate_rows, now=now, limit=max_symbols)
    sec_overlays = _sec_ranking_overlays(
        project_root,
        selected,
        now=now,
    )
    selected.sort(
        key=lambda row: (
            _signal_data_valid(row, now=now),
            _number(
                sec_overlays.get(
                    str(row.get("ticker") or row.get("asset") or "").upper(),
                    {},
                ).get("overlay", {}).get("final_rank_score")
            ),
        ),
        reverse=True,
    )
    timeframe_index = _timeframe_index(candidate_rows, now=now)
    _merge_phase11_10_support(
        timeframe_index,
        project_root=project_root,
        now=now,
    )
    trades = _read_parquet(project_root / PRIVATE_ROOT / "equity-trades.parquet")
    books = _read_parquet(
        project_root / PRIVATE_ROOT / "equity-orderbook.parquet"
    )
    observations: list[dict[str, Any]] = []
    for rank, signal in enumerate(selected, start=1):
        symbol = str(signal.get("ticker") or signal.get("asset")).upper()
        tape = _tape_snapshot(trades, symbol=symbol, now=now)
        depth = (
            _depth_snapshot(books, symbol=symbol, now=now)
            if rank <= depth_symbols
            else _depth_not_requested()
        )
        context = context_map.get(symbol, {})
        metadata = asset_metadata.get(symbol, {})
        sec_overlay = sec_overlays.get(symbol, _unavailable_sec_overlay(now))
        decision = _decision_contract(
            signal,
            context=context,
            tape=tape,
            depth=depth,
            asset_metadata=metadata,
            market_regime=market_regime,
            timeframe_support=timeframe_index.get(symbol, {}),
            now=now,
        )
        contract_snapshot = decision["contract_identity"]
        gates = decision["gates"]
        state = _state(decision, tape=tape, depth=depth)
        observation_identity = _signal_candidate_identity(signal)
        evidence_classification = candidate_evidence_classification(signal)
        candidate_identity = (
            observation_identity
            if evidence_classification["natural_strategy_candidate"]
            else None
        )
        timeframe_contract = signal.get("strategy_timeframe_contract")
        timeframe_contract = (
            timeframe_contract if isinstance(timeframe_contract, dict) else {}
        )
        observation = {
            "schema": "active_swing_forward_episode_v1",
            "iteration_rank": rank,
            "symbol": symbol,
            "signal_id": signal.get("signal_id"),
            "strategy_id": signal.get("strategy_id"),
            "strategy_dna_hash": signal.get("strategy_dna_hash"),
            "strategy_family": decision["strategy_family"],
            "model_version": decision["model_version"],
            "entry_timeframe": timeframe_contract.get("entry_timeframe"),
            "setup_timeframe": timeframe_contract.get("setup_timeframe"),
            "context_timeframes": timeframe_contract.get("context_timeframes", []),
            "setup_id": signal.get("setup_id"),
            "candidate_identity": candidate_identity,
            "observation_identity": observation_identity,
            **evidence_classification,
            "timeframe": signal.get("timeframe"),
            "setup_origin_timestamp": signal.get(
                "setup_origin_timestamp"
            ),
            "signal_timestamp": signal.get("signal_timestamp"),
            "data_cutoff_timestamp": signal.get("data_timestamp"),
            "observed_at": now.isoformat(),
            "decision_timestamp": now.isoformat(),
            "state": state,
            "gates": gates,
            "decision_contract": decision,
            "context_snapshot": {
                "asset_bias_score": context.get("asset_bias_score"),
                "asset_bias_confidence": context.get(
                    "asset_bias_confidence"
                ),
                "bias_classification": context.get("bias_classification"),
                "event_risk": context.get("event_risk"),
                "transmission_group": context.get("transmission_group"),
                "component_scores": _context_component_scores(context),
                "content_hash": stable_hash(context) if context else None,
                "sec_ranking_overlay": sec_overlay,
            },
            "setup_snapshot": {
                "contract_identity": contract_snapshot,
                "asset_class": decision["asset_profile"]["asset_class"],
                "asset_metadata": metadata,
                "action": signal.get("action"),
                "current_market_price": signal.get("current_market_price"),
                "preferred_entry": signal.get("preferred_entry"),
                "entry_zone_low": signal.get("entry_zone_low"),
                "entry_zone_high": signal.get("entry_zone_high"),
                "stop_loss": signal.get("stop_loss"),
                "take_profit_1": signal.get("take_profit_1"),
                "take_profit_2": signal.get("take_profit_2"),
                "reward_risk_1": signal.get("reward_risk_1"),
                "estimated_transaction_costs_eur": signal.get(
                    "estimated_transaction_costs_eur"
                ),
                "candidate_unit": signal.get("candidate_unit"),
                "strategy_dna_hash": signal.get("strategy_dna_hash"),
                "negative_sampling_policy": signal.get(
                    "negative_sampling_policy"
                ),
                "strategy_timeframe_contract": signal.get(
                    "strategy_timeframe_contract"
                ),
                "timeframe_evidence": signal.get("timeframe_evidence"),
                "timeframe_evidence_hash": signal.get(
                    "timeframe_evidence_hash"
                ),
                "market_reference_age_minutes": signal.get(
                    "market_reference_age_minutes"
                ),
                "price_validity_status": signal.get("price_validity_status"),
            },
            "entry_snapshot": {
                "tape": tape,
                "depth": depth,
            },
            "forward_outcome": {
                "would_fill": None,
                "fill_timestamp": None,
                "fill_fraction": None,
                "hypothetical_fill_price": None,
                "slippage_bps": None,
                "maximum_favourable_excursion": None,
                "maximum_adverse_excursion": None,
                "tp1_hit": None,
                "tp2_hit": None,
                "stop_hit": None,
                "time_exit": None,
                "net_r": None,
                "holding_time": None,
                "exit_outcome": None,
                "implementation_shortfall_estimate_bps": _shortfall(tape),
                "status": "PENDING_FUTURE_OBSERVATIONS",
            },
            "proposed_order": {
                "mode": "HYPOTHETICAL_OBSERVATION_ONLY",
                "order_type": "LIMIT",
                "limit_price": signal.get("limit_entry_price")
                or signal.get("preferred_entry")
                or signal.get("current_market_price"),
                "quantity": 0,
                "transmit": False,
            },
            "strategy_attribution": signal.get("strategy_id"),
            "orderflow_attribution": tape.get("data_class"),
            "quantity": 0,
            "automatic_execution_allowed": False,
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "order_calls": 0,
        }
        observation["feature_snapshot_hash"] = stable_hash(
            {
                "decision_contract": observation["decision_contract"],
                "context_snapshot": observation["context_snapshot"],
                "setup_snapshot": observation["setup_snapshot"],
                "entry_snapshot": observation["entry_snapshot"],
            }
        )
        observation["episode_id"] = "ENTRY-" + stable_hash(
            {"observation_identity": observation_identity}
        )[:24]
        observation["content_hash"] = stable_hash(observation)
        observations.append(observation)
    appended_rows = _append_episodes(project_root, observations)
    natural_observations = [
        row for row in observations if row["natural_strategy_candidate"]
    ]
    context_observations = [
        row for row in observations if not row["natural_strategy_candidate"]
    ]
    counts = {
        str(key): int(value)
        for key, value in pd.Series(
            [row["state"] for row in observations], dtype="object"
        ).value_counts().items()
    }
    funnel = _signal_funnel(
        input_count=len(signal_rows),
        candidates=candidate_rows,
        observations=observations,
    )
    asset_profile_counts = {
        str(key): int(value)
        for key, value in pd.Series(
            [
                row["decision_contract"]["asset_profile"]["asset_class"]
                for row in observations
            ],
            dtype="object",
        ).value_counts().items()
    }
    contract_identity_status_counts = {
        str(key): int(value)
        for key, value in pd.Series(
            [
                row["decision_contract"]["contract_identity"]["status"]
                for row in observations
            ],
            dtype="object",
        ).value_counts().items()
    }
    payload = {
        "schema": "hierarchical_entry_observer_v1",
        "status": "GO" if observations else "NO_CURRENT_SETUPS",
        "generated_at": now.isoformat(),
        "architecture": "CONTEXT_TO_BIAS_TO_SETUP_TO_ENTRY_TO_RISK_EXIT",
        "shortlist_count": len(observations),
        "depth_subscription_budget": int(depth_symbols),
        "state_counts": counts,
        "signal_funnel": funnel,
        "asset_profile_counts": asset_profile_counts,
        "contract_identity_status_counts": (
            contract_identity_status_counts
        ),
        "market_regime": market_regime,
        "sec_ranking_overlay_status": (
            "GO"
            if any(row.get("status") == "GO" for row in sec_overlays.values())
            else "DEGRADED_OR_UNAVAILABLE"
        ),
        "sec_standalone_entry_allowed": False,
        "natural_strategy_candidate_count": len(natural_observations),
        "canonical_signal_input_count": len(
            canonical_signals.get("signals", [])
        ),
        "tactical_15m_signal_input_count": len(
            tactical_signals.get("signals", [])
        ),
        "combined_deduplicated_signal_input_count": len(signal_rows),
        "context_watchlist_observation_count": len(context_observations),
        "new_episode_count": len(appended_rows),
        "new_natural_candidate_episode_count": sum(
            bool(row["natural_strategy_candidate"]) for row in appended_rows
        ),
        "new_context_observation_episode_count": sum(
            not bool(row["natural_strategy_candidate"]) for row in appended_rows
        ),
        "observed_trade_store_available": not trades.empty,
        "observed_orderbook_store_available": not books.empty,
        "bar_proxy_can_confirm_entry": False,
        "ml_status": "NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS",
        "observations": observations,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "order_calls": 0,
    }
    payload["content_hash"] = stable_hash(payload)
    output = project_root / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "entry-shortlist.json", payload)
    _write_json(output / "entry-observer-status.json", _public_status(payload))
    pd.DataFrame(
        [_flatten_observation(row) for row in observations]
    ).to_parquet(output / "entry-shortlist.parquet", index=False)
    return payload


def _sec_ranking_overlays(
    project_root: Path,
    rows: list[dict[str, Any]],
    *,
    now: datetime,
) -> dict[str, dict[str, Any]]:
    if not (project_root / "sec_ownership_and_event_intelligence_v1.py").is_file():
        return {}
    requests = []
    for row in rows:
        symbol = str(row.get("ticker") or row.get("asset") or "").upper()
        if not symbol:
            continue
        requests.append(
            {
                "symbol": symbol,
                "as_of": now.isoformat(),
                "base_score": _number(row.get("confidence_score")),
                "base_signal_authorized": _signal_data_valid(row, now=now),
            }
        )
    return sec_overlays_for_signals(project_root, requests)


def _unavailable_sec_overlay(now: datetime) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "as_of": now.isoformat(),
        "causal_event_count": 0,
        "sec_intelligence_score": 0.0,
        "overlay": {
            "sec_overlay_points": 0.0,
            "entry_authorized": False,
        },
        "authority": "RANKING_OVERLAY_ONLY",
        "standalone_entry_allowed": False,
        "delayed_context_only": True,
    }


def entry_observer_status(project_root: Path) -> dict[str, Any]:
    path = project_root / OUTPUT_ROOT / "entry-observer-status.json"
    if not path.is_file():
        return {
            "schema": "hierarchical_entry_observer_status_v1",
            "status": "NOT_RUN",
            "execution_authority": "NONE",
            "automatic_orders": 0,
            "broker_calls": 0,
            "order_calls": 0,
        }
    return _read_json(path)


def _candidate_signals(rows: Iterable[Any]) -> list[dict[str, Any]]:
    usable: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timeframe = str(row.get("timeframe", "")).lower()
        if timeframe not in {"15m", "1h", "2h", "4h", "1d"}:
            continue
        action = str(row.get("action", "")).upper()
        original_action = str(row.get("original_action", "")).upper()
        if action not in {
            "BUY",
            "WATCHLIST",
            "ENTRY_READY",
        } and original_action not in {
            "BUY",
            "WATCHLIST",
            "ENTRY_READY",
        }:
            continue
        usable.append(row)
    return usable


def _combined_signal_rows(
    canonical_rows: Iterable[Any], tactical_rows: Iterable[Any]
) -> list[dict[str, Any]]:
    """Combine observer inputs without making tactical rows money signals."""
    latest: dict[str, dict[str, Any]] = {}
    for source, rows in (
        ("CANONICAL_MONEY_SIGNAL_ARTIFACT", canonical_rows),
        ("TACTICAL_15M_OBSERVER_ARTIFACT", tactical_rows),
    ):
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            row = {**raw, "observer_input_source": source}
            identity = str(
                row.get("setup_id")
                or row.get("signal_id")
                or stable_hash(row)
            )
            prior = latest.get(identity)
            timestamp = str(
                row.get("candidate_observed_at")
                or row.get("signal_timestamp")
                or row.get("data_timestamp")
                or ""
            )
            prior_timestamp = str(
                (prior or {}).get("candidate_observed_at")
                or (prior or {}).get("signal_timestamp")
                or (prior or {}).get("data_timestamp")
                or ""
            )
            if prior is None or timestamp >= prior_timestamp:
                latest[identity] = row
    return list(latest.values())


def _select_signals(
    rows: Iterable[dict[str, Any]], *, now: datetime, limit: int
) -> list[dict[str, Any]]:
    usable = list(rows)
    priority = {"15m": 5, "1h": 4, "2h": 3, "4h": 2, "1d": 1}
    usable.sort(
        key=lambda row: (
            _signal_data_valid(row, now=now),
            bool(
                isinstance(row.get("contract_identity"), dict)
                and row["contract_identity"].get("con_id")
            ),
            priority.get(str(row.get("timeframe", "")).lower(), 0),
            _number(row.get("confidence_score")),
            str(row.get("signal_timestamp", "")),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in usable:
        symbol = str(row.get("ticker") or row.get("asset") or "").upper()
        identity = _signal_selection_identity(row)
        if not symbol or identity in seen:
            continue
        selected.append(row)
        seen.add(identity)
        if len(selected) >= limit:
            break
    return selected


def _timeframe_index(
    rows: Iterable[dict[str, Any]], *, now: datetime
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    result: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        symbol = str(row.get("ticker") or row.get("asset") or "").upper()
        timeframe = str(row.get("timeframe", "")).lower()
        if not symbol or timeframe not in {"15m", "1h", "2h", "4h", "1d"}:
            continue
        enriched = {
            **row,
            "_hierarchy_current": _signal_data_valid(row, now=now),
        }
        result.setdefault(symbol, {}).setdefault(timeframe, []).append(enriched)
    return result


def _signal_candidate_identity(signal: dict[str, Any]) -> str:
    setup_id = str(signal.get("setup_id") or "").strip()
    if setup_id:
        return setup_id
    return stable_hash(
        {
            "symbol": str(
                signal.get("ticker") or signal.get("asset") or ""
            ).upper(),
            "strategy_id": str(signal.get("strategy_id") or ""),
            "timeframe": str(signal.get("timeframe") or "").lower(),
            "setup_origin_timestamp": str(
                signal.get("setup_origin_timestamp")
                or signal.get("data_timestamp")
                or signal.get("signal_timestamp")
                or ""
            ),
        }
    )


def _signal_selection_identity(signal: dict[str, Any]) -> str:
    setup_id = str(signal.get("setup_id") or "").strip()
    if setup_id:
        return f"SETUP:{setup_id}"
    symbol = str(
        signal.get("ticker") or signal.get("asset") or ""
    ).upper()
    return f"SYMBOL:{symbol}"


def _merge_phase11_10_support(
    target: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    project_root: Path,
    now: datetime,
) -> None:
    payload = _read_json(
        project_root / "output/research/phase11_10/current-watchlist.json"
    )
    for architecture in payload.get("architecture_watchlists", []):
        if not isinstance(architecture, dict):
            continue
        timeframe = str(architecture.get("lower_timeframe", "")).lower()
        if timeframe not in {"1h", "2h", "4h", "1d"}:
            continue
        for row in architecture.get("all_instruments", []):
            if not isinstance(row, dict) or not row.get("active_signal"):
                continue
            symbol = str(row.get("symbol", "")).upper()
            if not symbol:
                continue
            target.setdefault(symbol, {}).setdefault(timeframe, []).append(
                {
                    **row,
                    "timeframe": timeframe,
                    "strategy_id": architecture.get("strategy_id"),
                    "strategy_family": architecture.get("entry_strategy"),
                    "source": "PHASE11_10_CURRENT_WATCHLIST",
                    "_hierarchy_current": _closed_bar_support_current(
                        row.get("closed_bar_timestamp"),
                        timeframe=timeframe,
                        now=now,
                    ),
                }
            )


def _closed_bar_support_current(
    value: Any, *, timeframe: str, now: datetime
) -> bool:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return False
    if pd.isna(timestamp):
        return False
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    max_age = {
        "1h": timedelta(hours=6),
        "2h": timedelta(hours=12),
        "4h": timedelta(hours=24),
        "1d": timedelta(days=10),
    }[timeframe]
    age = pd.Timestamp(now) - timestamp
    return timedelta(0) <= age <= max_age


def _tape_snapshot(
    frame: pd.DataFrame, *, symbol: str, now: datetime
) -> dict[str, Any]:
    if frame.empty or "symbol" not in frame:
        return _tape_unavailable()
    work = frame.loc[frame["symbol"].astype(str).str.upper().eq(symbol)].copy()
    timestamp_column = "timestamp" if "timestamp" in work else "timestamp_utc"
    if work.empty or timestamp_column not in work:
        return _tape_unavailable()
    work["timestamp"] = pd.to_datetime(
        work[timestamp_column], utc=True, errors="coerce"
    )
    cutoff = pd.Timestamp(now - timedelta(hours=2))
    work = work.dropna(subset=["timestamp"]).loc[
        lambda row: row["timestamp"].between(cutoff, pd.Timestamp(now))
    ]
    if work.empty:
        return _tape_unavailable("OBSERVED_TAPE_STALE_OR_EMPTY")
    classified = classify_trades(work)
    aggregate = aggregate_trade_flow(classified, interval="1h")
    if aggregate.empty:
        return _tape_unavailable("OBSERVED_TAPE_UNCLASSIFIABLE")
    latest = aggregate.iloc[-1]
    first_price = float(classified["price"].iloc[0])
    last_price = float(classified["price"].iloc[-1])
    elapsed = max(
        1.0,
        (
            classified["timestamp"].iloc[-1]
            - classified["timestamp"].iloc[0]
        ).total_seconds(),
    )
    last = classified.iloc[-1]
    bid = _optional(last.get("bid"))
    ask = _optional(last.get("ask"))
    spread_bps = (
        (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
        if bid is not None and ask is not None and ask > bid
        else None
    )
    return {
        "status": "OBSERVED_TRADE_FLOW_AVAILABLE",
        "data_class": "OBSERVED_TRADE_FLOW",
        "latest_timestamp": classified["timestamp"].iloc[-1].isoformat(),
        "trade_count": int(latest["trade_count"]),
        "trades_per_second": round(len(classified) / elapsed, 8),
        "normalized_delta": _optional(latest.get("normalized_delta")),
        "cvd": _optional(latest.get("cvd")),
        "classification_ratio": _optional(
            latest.get("classification_ratio")
        ),
        "price_progression_pct": round(
            last_price / first_price - 1.0, 8
        ),
        "last_price": last_price,
        "bid": bid,
        "ask": ask,
        "spread_bps": round(spread_bps, 8)
        if spread_bps is not None
        else None,
        "confidence": _optional(latest.get("confidence")),
        "standalone_entry_authority": False,
    }


def _depth_snapshot(
    frame: pd.DataFrame, *, symbol: str, now: datetime
) -> dict[str, Any]:
    if frame.empty or "symbol" not in frame:
        return _depth_unavailable()
    work = frame.loc[frame["symbol"].astype(str).str.upper().eq(symbol)].copy()
    timestamp_column = "timestamp" if "timestamp" in work else "timestamp_utc"
    if work.empty or timestamp_column not in work:
        return _depth_unavailable()
    work["timestamp"] = pd.to_datetime(
        work[timestamp_column], utc=True, errors="coerce"
    )
    work = work.dropna(subset=["timestamp"]).loc[
        lambda row: row["timestamp"].between(
            pd.Timestamp(now - timedelta(minutes=15)), pd.Timestamp(now)
        )
    ]
    if work.empty:
        return _depth_unavailable("OBSERVED_DEPTH_STALE_OR_EMPTY")
    latest_timestamp = work["timestamp"].max()
    latest = work.loc[work["timestamp"].eq(latest_timestamp)]
    metrics = orderbook_metrics(latest)
    if metrics["status"] != "ORDERBOOK_CONTEXT_AVAILABLE":
        return _depth_unavailable("OBSERVED_DEPTH_INVALID")
    timestamps = sorted(work["timestamp"].unique())
    replenishment = None
    ask_depletion = None
    if len(timestamps) >= 2:
        previous = work.loc[work["timestamp"].eq(timestamps[-2])]
        bid_before = pd.to_numeric(previous["bid_size"], errors="coerce").sum()
        ask_before = pd.to_numeric(previous["ask_size"], errors="coerce").sum()
        bid_now = pd.to_numeric(latest["bid_size"], errors="coerce").sum()
        ask_now = pd.to_numeric(latest["ask_size"], errors="coerce").sum()
        replenishment = bid_now / bid_before - 1.0 if bid_before > 0 else None
        ask_depletion = 1.0 - ask_now / ask_before if ask_before > 0 else None
    return {
        **metrics,
        "data_class": "OBSERVED_ORDERBOOK",
        "latest_timestamp": pd.Timestamp(latest_timestamp).isoformat(),
        "bid_replenishment": _optional(replenishment),
        "ask_depletion": _optional(ask_depletion),
    }


def _decision_contract(
    signal: dict[str, Any],
    *,
    context: dict[str, Any],
    tape: dict[str, Any],
    depth: dict[str, Any],
    asset_metadata: dict[str, Any],
    market_regime: str,
    timeframe_support: dict[str, list[dict[str, Any]]],
    now: datetime,
) -> dict[str, Any]:
    event = context.get("event_risk", {})
    bias_score = _number(context.get("asset_bias_score"))
    signal_current = signal_is_current(signal, now=now)
    bar_closed = bool(signal.get("bar_closed", True))
    data_fresh = str(signal.get("data_freshness", "FRESH")).upper() != "STALE"
    price_valid = str(signal.get("price_validity_status", "")).endswith("GO")
    contract = _contract_snapshot(signal.get("contract_identity"), now=now)
    contract_resolved = contract["status"] == "RESOLVED"
    reward_risk = _reward_risk(signal)
    economics_viable = reward_risk >= 1.5
    tape_available = tape.get("status") == "OBSERVED_TRADE_FLOW_AVAILABLE"
    tape_confirmation = bool(
        tape_available
        and _number(tape.get("normalized_delta")) >= 0.05
        and _number(tape.get("classification_ratio")) >= 0.50
        and _number(tape.get("price_progression_pct")) > 0.0
        and (
            tape.get("spread_bps") is None
            or _number(tape.get("spread_bps"), 1e9) <= 25.0
        )
    )
    depth_available = depth.get("status") == "ORDERBOOK_CONTEXT_AVAILABLE"
    depth_confirmation = bool(
        depth_available
        and _number(depth.get("obi")) >= -0.10
        and _number(depth.get("microprice_edge")) >= -0.10
    )
    family = _strategy_family(signal)
    route = _regime_route(market_regime, family)
    hierarchy = _hierarchy_status(
        signal,
        support=timeframe_support,
    )
    asset_profile = _asset_profile(
        signal,
        metadata=asset_metadata,
        context=context,
        tape=tape,
        depth=depth,
    )
    hard_vetoes = []
    if not signal_current:
        hard_vetoes.append("SIGNAL_EXPIRED_OR_NOT_CURRENT")
    if not bar_closed:
        hard_vetoes.append("BAR_NOT_CLOSED")
    if not data_fresh:
        hard_vetoes.append("CURRENT_DATA_STALE")
    if not price_valid:
        hard_vetoes.append("CURRENT_PRICE_REFERENCE_INVALID")
    if bool(event.get("blocks_new_entry")):
        hard_vetoes.append("MATERIAL_EVENT_RISK")
    if not contract_resolved:
        hard_vetoes.append("CONTRACT_IDENTITY_REQUIRED")
    if not economics_viable:
        hard_vetoes.append("ENTRY_ECONOMICS_BELOW_MINIMUM")
    research_observation_blockers = sorted(
        veto
        for veto in set(hard_vetoes)
        if veto != "CONTRACT_IDENTITY_REQUIRED"
    )
    research_observation_eligible = not research_observation_blockers
    soft_vetoes = []
    if not context:
        soft_vetoes.append("ASSET_CONTEXT_UNAVAILABLE")
    if bias_score <= -0.25:
        soft_vetoes.append("ASSET_BIAS_ADVERSE")
    if not route["family_allowed"]:
        soft_vetoes.append("STRATEGY_FAMILY_NOT_ROUTED_IN_REGIME")
    if family == "UNCLASSIFIED_EXISTING_FAMILY":
        soft_vetoes.append("STRATEGY_FAMILY_CLASSIFICATION_REQUIRED")
    gates = {
        "context_available": bool(context),
        "bias_not_adverse": bias_score > -0.25,
        "event_risk_clear": not bool(event.get("blocks_new_entry")),
        "signal_current": signal_current,
        "bar_closed": bar_closed,
        "data_fresh": data_fresh,
        "contract_resolved": contract_resolved,
        "economics_viable": economics_viable,
        "setup_current_and_price_valid": signal_current and data_fresh and price_valid,
        "regime_route_allowed": route["family_allowed"],
        "timeframe_hierarchy_ready": hierarchy["ready"],
        "observed_tape_available": tape_available,
        "observed_tape_confirms": tape_confirmation,
        "observed_depth_available": depth_available,
        "observed_depth_confirms": depth_confirmation,
        "bar_proxy_used_as_confirmation": False,
    }
    setup_score = _setup_score(
        signal,
        context=context,
        hierarchy=hierarchy,
        asset_profile=asset_profile,
    )
    entry_score = _entry_confirmation_score(tape=tape, depth=depth)
    decision = {
        "schema": "hierarchical_swing_decision_contract_v2",
        "model_version": "entry-observer-v2.0.0",
        "as_of": now.isoformat(),
        "expires_at": signal.get("expiration_timestamp"),
        "strategy_family": family,
        "strategy_timeframe_contract": signal.get(
            "strategy_timeframe_contract"
        ),
        "candidate_unit": signal.get("candidate_unit"),
        "market_regime": market_regime,
        "contract_identity": contract,
        "hard_vetoes": sorted(set(hard_vetoes)),
        "soft_vetoes": sorted(set(soft_vetoes)),
        "hard_veto_pass": not hard_vetoes,
        "research_observation_eligible": research_observation_eligible,
        "research_observation_blockers": research_observation_blockers,
        "brokerability_status": (
            "BROKERABLE_CONTRACT_RESOLVED"
            if contract_resolved
            else "BROKERABILITY_BLOCKED_CONTRACT_IDENTITY"
        ),
        "standalone_entry_allowed": False,
        "route": route,
        "timeframe_hierarchy": hierarchy,
        "asset_profile": asset_profile,
        "setup_score_within_family": setup_score,
        "entry_confirmation_score": entry_score,
        "gates": gates,
        "authority": "OBSERVATION_ONLY",
        "execution_authority": "NONE",
    }
    decision["layers"] = _layer_results(decision)
    return decision


def _contract_snapshot(raw: Any, *, now: datetime) -> dict[str, Any]:
    identity = raw if isinstance(raw, dict) else {}
    try:
        con_id = int(identity.get("con_id") or 0)
    except (TypeError, ValueError):
        con_id = 0
    resolved_at = pd.to_datetime(
        identity.get("resolved_at"), utc=True, errors="coerce"
    )
    age = (
        pd.Timestamp(now) - resolved_at
        if not pd.isna(resolved_at)
        else None
    )
    cache_fresh = bool(
        con_id > 0
        and age is not None
        and pd.Timedelta(0) <= age < pd.Timedelta(days=7)
    )
    fields = (
        "symbol",
        "local_symbol",
        "security_type",
        "currency",
        "exchange",
        "primary_exchange",
        "trading_class",
        "min_tick",
        "contract_hash",
        "server_version",
        "cache_expires_at",
        "cache_status",
        "contract_source",
    )
    snapshot = {
        field: identity.get(field)
        for field in fields
        if identity.get(field) not in (None, "")
    }
    status = (
        "RESOLVED"
        if cache_fresh
        else (
            "STALE_OR_UNPROVEN_CACHE_BLOCKED"
            if con_id > 0
            else "UNRESOLVED_BLOCKED"
        )
    )
    return {
        "status": status,
        "source": "SIGNAL_IMMUTABLE_SNAPSHOT",
        "con_id": con_id if con_id > 0 else None,
        "resolved_at": (
            resolved_at.isoformat() if not pd.isna(resolved_at) else None
        ),
        "cache_age_seconds": (
            round(age.total_seconds(), 6) if age is not None else None
        ),
        "cache_fresh": cache_fresh,
        **snapshot,
    }


def _state(
    decision: dict[str, Any], *, tape: dict[str, Any], depth: dict[str, Any]
) -> str:
    gates = decision["gates"]
    hard_vetoes = decision["hard_vetoes"]
    if hard_vetoes:
        return "WATCHLIST_HARD_VETO_BLOCKED"
    if decision["soft_vetoes"]:
        return "WATCHLIST_SOFT_VETO_BLOCKED"
    hierarchy = decision["timeframe_hierarchy"]
    timeframe = hierarchy["decision_timeframe"]
    if timeframe == "1d":
        if hierarchy["four_hour_setup_available"]:
            return "DIRECTIONAL_BIAS_WITH_4H_SETUP_1H_CONFIRMATION_PENDING"
        return "DIRECTIONAL_BIAS_ONLY_4H_SETUP_PENDING"
    if timeframe == "4h":
        if not hierarchy["ready"]:
            return "SETUP_4H_DAILY_DIRECTION_PENDING"
        return "SETUP_4H_READY_1H_CONFIRMATION_PENDING"
    if timeframe == "2h":
        if not hierarchy["ready"]:
            return "WATCHLIST_4H_SETUP_REQUIRED"
        return "SETUP_REFINEMENT_READY_1H_CONFIRMATION_PENDING"
    if not hierarchy["ready"]:
        return "WATCHLIST_4H_SETUP_REQUIRED"
    if not gates["observed_tape_available"]:
        return "ENTRY_DATA_PENDING_OBSERVED_TAPE"
    if not gates["observed_tape_confirms"]:
        return "WATCHLIST_TAPE_NOT_CONFIRMED"
    if depth.get("status") == "DEPTH_NOT_REQUESTED_BUDGET":
        return "TAPE_CONFIRMED_DEPTH_NOT_REQUESTED"
    if not gates["observed_depth_available"]:
        return "TAPE_CONFIRMED_DEPTH_PENDING"
    if not gates["observed_depth_confirms"]:
        return "WATCHLIST_DEPTH_NOT_CONFIRMED"
    return "EXECUTION_CANDIDATE_AUTHORITY_NONE"


def _signal_data_valid(signal: dict[str, Any], *, now: datetime) -> bool:
    return bool(
        signal_is_current(signal, now=now)
        and signal.get("bar_closed", True)
        and str(signal.get("data_freshness", "FRESH")).upper() != "STALE"
        and str(signal.get("price_validity_status", "")).endswith("GO")
    )


def _strategy_family(signal: dict[str, Any]) -> str:
    text = " ".join(
        [
            str(signal.get("strategy_id", "")),
            *[str(value) for value in signal.get("reasons", [])],
        ]
    ).upper()
    if any(word in text for word in ("CONTRACTION", "BREAKOUT", "DONCHIAN")):
        return "COMPRESSION_BREAKOUT"
    if any(word in text for word in ("ROTATION", "RELATIVE", "LEADERSHIP")):
        return "RELATIVE_STRENGTH_ROTATION"
    if any(word in text for word in ("PULLBACK", "RECLAIM", "OVERSOLD")):
        return "TREND_PULLBACK"
    if any(
        word in text
        for word in (
            "MOMENTUM",
            "TREND",
            "MOVING_AVERAGE",
            "FAST_MA",
            "SLOW_MA",
        )
    ):
        return "TREND_CONTINUATION"
    return "UNCLASSIFIED_EXISTING_FAMILY"


def _regime_route(regime: str, family: str) -> dict[str, Any]:
    normalized = regime.upper()
    if family == "UNCLASSIFIED_EXISTING_FAMILY":
        return {
            "status": "UNCLASSIFIED_FAMILY_OBSERVATION_ONLY",
            "market_regime": normalized,
            "strategy_family": family,
            "family_allowed": True,
            "allowed_families": [],
            "changes_direction": False,
            "changes_size_only_when_uncertain": True,
        }
    if "BULL" in normalized or "TREND" in normalized:
        allowed = {
            "TREND_PULLBACK",
            "COMPRESSION_BREAKOUT",
            "RELATIVE_STRENGTH_ROTATION",
            "TREND_CONTINUATION",
        }
        route_status = "TREND_REGIME_ROUTE"
    elif any(word in normalized for word in ("RANGE", "SIDEWAYS", "CHOP")):
        allowed = {"COMPRESSION_BREAKOUT", "RELATIVE_STRENGTH_ROTATION"}
        route_status = "RANGE_REGIME_ROUTE"
    elif any(word in normalized for word in ("BEAR", "STRESS", "RISK_OFF")):
        allowed = {"RELATIVE_STRENGTH_ROTATION"}
        route_status = "LONG_ONLY_DEFENSIVE_ROUTE"
    else:
        allowed = {
            "TREND_PULLBACK",
            "COMPRESSION_BREAKOUT",
            "RELATIVE_STRENGTH_ROTATION",
            "TREND_CONTINUATION",
            "UNCLASSIFIED_EXISTING_FAMILY",
        }
        route_status = "UNKNOWN_REGIME_CONSERVATIVE_OBSERVATION"
    return {
        "status": route_status,
        "market_regime": normalized,
        "strategy_family": family,
        "family_allowed": family in allowed,
        "allowed_families": sorted(allowed),
        "changes_direction": False,
        "changes_size_only_when_uncertain": True,
    }


def _hierarchy_status(
    signal: dict[str, Any], *, support: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    explicit = _explicit_hierarchy_status(signal)
    if explicit is not None:
        return explicit
    timeframe = str(signal.get("timeframe", "")).lower()
    current = {
        key: any(bool(row.get("_hierarchy_current")) for row in rows)
        for key, rows in support.items()
    }
    daily = bool(current.get("1d"))
    four_hour = bool(current.get("4h"))
    two_hour = bool(current.get("2h"))
    if timeframe == "1d":
        ready = True
        status = "DAILY_DIRECTIONAL_BIAS_AVAILABLE"
    elif timeframe == "4h":
        ready = daily
        status = (
            "FOUR_HOUR_SETUP_WITH_DAILY_DIRECTION"
            if ready
            else "DAILY_DIRECTION_REQUIRED"
        )
    elif timeframe == "2h":
        ready = daily and four_hour
        status = (
            "TWO_HOUR_REFINEMENT_WITH_FOUR_HOUR_SETUP"
            if ready
            else "FOUR_HOUR_SETUP_AND_DAILY_DIRECTION_REQUIRED"
        )
    else:
        ready = daily and four_hour
        status = (
            "ONE_HOUR_CONFIRMATION_WITH_VALID_FOUR_HOUR_SETUP"
            if ready
            else "FOUR_HOUR_SETUP_AND_DAILY_DIRECTION_REQUIRED"
        )
    return {
        "decision_timeframe": timeframe,
        "status": status,
        "ready": ready,
        "daily_direction_available": daily,
        "four_hour_setup_available": four_hour,
        "two_hour_refinement_available": two_hour,
        "support_sources": sorted(
            {
                str(row.get("source", "LATEST_SIGNALS"))
                for rows in support.values()
                for row in rows
                if row.get("_hierarchy_current")
            }
        ),
        "two_hour_refinement_required": False,
        "fifteen_minute_strategy_can_create_candidate": False,
        "fifteen_minute_execution_can_create_trade": False,
    }


def _explicit_hierarchy_status(
    signal: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw = signal.get("strategy_timeframe_contract")
    if not isinstance(raw, Mapping):
        return None
    try:
        contract = StrategyTimeframeContract(
            entry_timeframe=str(raw["entry_timeframe"]),
            setup_timeframe=str(raw["setup_timeframe"]),
            context_timeframes=tuple(raw.get("context_timeframes", ())),
            structural_timeframe=str(raw["structural_timeframe"]),
            management_timeframe=str(raw["management_timeframe"]),
            exit_timeframe=str(raw["exit_timeframe"]),
            required_timeframes=tuple(raw["required_timeframes"]),
            optional_timeframes=tuple(raw.get("optional_timeframes", ())),
            session=str(raw.get("session", "RTH")),
        )
    except (KeyError, TypeError, ValueError):
        return {
            "decision_timeframe": str(signal.get("timeframe", "")).lower(),
            "status": "EXPLICIT_TIMEFRAME_CONTRACT_INVALID",
            "ready": False,
            "daily_direction_available": False,
            "four_hour_setup_available": False,
            "two_hour_refinement_available": False,
            "required_timeframes": [],
            "observed_timeframes": [],
            "blockers": ["EXPLICIT_TIMEFRAME_CONTRACT_INVALID"],
            "support_sources": [],
            "all_timeframes_need_not_agree": True,
            "two_hour_refinement_required": False,
            "fifteen_minute_strategy_can_create_candidate": True,
            "fifteen_minute_execution_can_create_trade": False,
        }
    raw_evidence = signal.get("timeframe_evidence")
    evidence = raw_evidence if isinstance(raw_evidence, Mapping) else {}
    decision_at = pd.to_datetime(
        signal.get("data_timestamp"), utc=True, errors="coerce"
    )
    observed_at = pd.to_datetime(
        signal.get("signal_timestamp"), utc=True, errors="coerce"
    )
    blockers: list[str] = []
    observed: list[str] = []
    for timeframe in contract.all_timeframes:
        item = evidence.get(timeframe)
        if not isinstance(item, Mapping) or not bool(item.get("available")):
            if timeframe in contract.required_timeframes:
                blockers.append(f"REQUIRED_TIMEFRAME_MISSING:{timeframe}")
            continue
        if item.get("bar_closed") is not True:
            if timeframe in contract.required_timeframes:
                blockers.append(f"REQUIRED_TIMEFRAME_NOT_CLOSED:{timeframe}")
            continue
        available_at = pd.to_datetime(
            item.get("available_at"), utc=True, errors="coerce"
        )
        if (
            pd.isna(available_at)
            or pd.isna(decision_at)
            or available_at > decision_at
        ):
            if timeframe in contract.required_timeframes:
                blockers.append(f"REQUIRED_TIMEFRAME_NONCAUSAL:{timeframe}")
            continue
        knowledge_at = pd.to_datetime(
            item.get("knowledge_available_at"),
            utc=True,
            errors="coerce",
        )
        if (
            not pd.isna(knowledge_at)
            and (pd.isna(observed_at) or knowledge_at > observed_at)
        ):
            if timeframe in contract.required_timeframes:
                blockers.append(
                    f"REQUIRED_TIMEFRAME_NOT_OBSERVED_YET:{timeframe}"
                )
            continue
        observed.append(timeframe)
    ready = not blockers
    return {
        "decision_timeframe": contract.entry_timeframe,
        "status": (
            "EXPLICIT_CAUSAL_TIMEFRAME_CONTRACT_READY"
            if ready
            else "EXPLICIT_TIMEFRAME_CONTRACT_BLOCKED"
        ),
        "ready": ready,
        "daily_direction_available": "1d" in observed,
        "four_hour_setup_available": "4h" in observed,
        "two_hour_refinement_available": "2h" in observed,
        "required_timeframes": list(contract.required_timeframes),
        "observed_timeframes": observed,
        "blockers": blockers,
        "support_sources": ["IMMUTABLE_SIGNAL_TIMEFRAME_EVIDENCE"],
        "all_timeframes_need_not_agree": True,
        "two_hour_refinement_required": False,
        "fifteen_minute_strategy_can_create_candidate": (
            contract.entry_timeframe == "15m"
        ),
        "fifteen_minute_execution_can_create_trade": False,
    }


def _reward_risk(signal: dict[str, Any]) -> float:
    explicit = _optional(signal.get("reward_risk_1"))
    if explicit is not None and explicit > 0:
        return explicit
    entry = _optional(
        signal.get("preferred_entry")
        or signal.get("current_market_price")
        or signal.get("entry_zone_high")
    )
    stop = _optional(signal.get("stop_loss"))
    target = _optional(signal.get("take_profit_1"))
    if entry is None or stop is None or target is None or entry <= stop:
        return 0.0
    return max(0.0, (target - entry) / (entry - stop))


def _asset_profile(
    signal: dict[str, Any],
    *,
    metadata: dict[str, Any],
    context: dict[str, Any],
    tape: dict[str, Any],
    depth: dict[str, Any],
) -> dict[str, Any]:
    asset_class = _specialized_asset_class(signal, metadata)
    weights = ASSET_PROFILE_WEIGHTS[asset_class]
    confidence = _unit(signal.get("confidence_score"))
    reasons = " ".join(str(value) for value in signal.get("reasons", [])).upper()
    relative_strength = (
        confidence
        if any(
            marker in reasons
            for marker in ("RELATIVE", "LEADERSHIP", "MOMENTUM")
        )
        else None
    )
    observed_flow = (
        _signed_unit(tape.get("normalized_delta"))
        if tape.get("status") == "OBSERVED_TRADE_FLOW_AVAILABLE"
        else None
    )
    observed_book = (
        _signed_unit(depth.get("obi"))
        if depth.get("status") == "ORDERBOOK_CONTEXT_AVAILABLE"
        else None
    )
    event = context.get("event_risk")
    event_quality = (
        1.0 - _unit(event.get("risk_score"))
        if isinstance(event, dict)
        else None
    )
    execution_quality = _execution_quality(signal, tape)
    macro_quality = (
        _signed_unit(context.get("asset_bias_score")) if context else None
    )
    fundamental_context = _available_context_average(
        context,
        names=("cot", "macro"),
    )
    common = {
        "trend": confidence,
        "relative_strength": relative_strength,
        "orderflow": observed_flow,
        "execution": execution_quality,
    }
    if asset_class == "STOCK":
        components = {
            **common,
            "volume": None,
            "book": observed_book,
            "event": event_quality,
            "macro": macro_quality,
        }
    elif asset_class == "ETF":
        components = {
            **common,
            "breadth": None,
            "underlying": None,
            "futures_confirmation": None,
            "fair_value": None,
        }
    else:
        components = {
            **common,
            "curve": None,
            "fundamentals": fundamental_context,
            "futures_orderflow": None,
            "cross_market": None,
            "proxy_quality": None,
        }
    available_weight = sum(
        weights[name]
        for name, value in components.items()
        if value is not None
    )
    raw_score = (
        100.0
        * sum(
            weights[name] * float(value)
            for name, value in components.items()
            if value is not None
        )
        / available_weight
        if available_weight
        else 50.0
    )
    coverage = available_weight / sum(weights.values())
    coverage_adjusted = 50.0 + coverage * (raw_score - 50.0)
    status = (
        "SPECIALIZED_DATA_COMPLETE"
        if coverage >= 0.75
        else "SPECIALIZED_DATA_PARTIAL"
        if coverage >= 0.40
        else "SPECIALIZED_DATA_LIMITED"
    )
    return {
        "schema": "active_swing_asset_profile_v1",
        "asset_class": asset_class,
        "configured_asset_type": metadata.get("asset_type"),
        "sleeve": metadata.get("sleeve"),
        "sector": metadata.get("sector"),
        "region": metadata.get("region"),
        "status": status,
        "component_weights": weights,
        "component_scores": {
            name: round(value, 8) if value is not None else None
            for name, value in components.items()
        },
        "missing_components": sorted(
            name for name, value in components.items() if value is None
        ),
        "data_coverage_ratio": round(coverage, 8),
        "available_component_score": round(raw_score, 4),
        "coverage_adjusted_score": round(coverage_adjusted, 4),
        "observed_orderflow_used": observed_flow is not None,
        "bar_flow_proxy_used_as_orderflow": False,
        "score_authority": "RANKING_CONTEXT_ONLY",
        "standalone_entry_authority": False,
    }


def _specialized_asset_class(
    signal: dict[str, Any], metadata: dict[str, Any]
) -> str:
    configured = str(metadata.get("asset_type", "")).upper()
    reported = str(signal.get("asset_class", "")).upper()
    if configured == "COMMODITY_ETF" or "COMMODITY" in reported:
        return "COMMODITY_PROXY"
    if configured.endswith("ETF") or reported == "ETF":
        return "ETF"
    return "STOCK"


def _execution_quality(
    signal: dict[str, Any], tape: dict[str, Any]
) -> float:
    price_valid = str(signal.get("price_validity_status", "")).endswith("GO")
    base = 0.5 if price_valid else 0.0
    spread = _optional(tape.get("spread_bps"))
    if spread is None:
        return base
    spread_quality = (
        1.0 if spread <= 10.0 else 0.7 if spread <= 25.0 else 0.2
    )
    return (base + spread_quality) / 2.0


def _available_context_average(
    context: dict[str, Any], *, names: tuple[str, ...]
) -> float | None:
    components = context.get("components", {})
    values = [
        _signed_unit(component.get("score"))
        for name in names
        if isinstance((component := components.get(name)), dict)
        and _optional(component.get("score")) is not None
    ]
    return sum(values) / len(values) if values else None


def _context_component_scores(context: dict[str, Any]) -> dict[str, Any]:
    components = context.get("components", {})
    return {
        str(name): {
            key: value.get(key)
            for key in ("status", "score", "confidence", "data_class")
            if key in value
        }
        for name, value in components.items()
        if isinstance(value, dict)
    }


def _unit(value: Any) -> float:
    return min(1.0, max(0.0, _number(value)))


def _signed_unit(value: Any) -> float:
    return min(1.0, max(0.0, (_number(value) + 1.0) / 2.0))


def _setup_score(
    signal: dict[str, Any],
    *,
    context: dict[str, Any],
    hierarchy: dict[str, Any],
    asset_profile: dict[str, Any],
) -> float:
    confidence = _number(signal.get("confidence_score"))
    bias = (_number(context.get("asset_bias_score")) + 1.0) / 2.0
    hierarchy_score = (
        1.0
        if hierarchy["ready"]
        else 0.5
        if hierarchy["decision_timeframe"] == "1d"
        else 0.0
    )
    price_valid = float(
        str(signal.get("price_validity_status", "")).endswith("GO")
    )
    specialized = _number(asset_profile.get("coverage_adjusted_score")) / 100.0
    return round(
        100.0
        * (
            0.35 * min(1.0, max(0.0, confidence))
            + 0.15 * min(1.0, max(0.0, bias))
            + 0.20 * hierarchy_score
            + 0.10 * price_valid
            + 0.20 * specialized
        ),
        4,
    )


def _entry_confirmation_score(
    *, tape: dict[str, Any], depth: dict[str, Any]
) -> float | None:
    if tape.get("status") != "OBSERVED_TRADE_FLOW_AVAILABLE":
        return None
    delta = min(1.0, max(0.0, (_number(tape.get("normalized_delta")) + 1) / 2))
    progression = min(
        1.0,
        max(0.0, 0.5 + 25.0 * _number(tape.get("price_progression_pct"))),
    )
    classification = min(1.0, max(0.0, _number(tape.get("classification_ratio"))))
    if depth.get("status") == "ORDERBOOK_CONTEXT_AVAILABLE":
        depth_score = min(1.0, max(0.0, (_number(depth.get("obi")) + 1) / 2))
        weights = (0.35, 0.25, 0.20, 0.20)
    else:
        depth_score = 0.0
        weights = (0.45, 0.30, 0.25, 0.0)
    return round(
        100.0
        * (
            weights[0] * delta
            + weights[1] * progression
            + weights[2] * classification
            + weights[3] * depth_score
        ),
        4,
    )


def _layer_results(decision: dict[str, Any]) -> list[dict[str, Any]]:
    gates = decision["gates"]
    hierarchy = decision["timeframe_hierarchy"]
    hard_vetoes = set(decision["hard_vetoes"])
    data_vetoes = sorted(
        hard_vetoes
        & {
            "SIGNAL_EXPIRED_OR_NOT_CURRENT",
            "BAR_NOT_CLOSED",
            "CURRENT_DATA_STALE",
            "CURRENT_PRICE_REFERENCE_INVALID",
        }
    )
    mandate_vetoes = sorted(hard_vetoes & {"CONTRACT_IDENTITY_REQUIRED"})
    event_vetoes = sorted(hard_vetoes & {"MATERIAL_EVENT_RISK"})
    economics_vetoes = sorted(
        hard_vetoes & {"ENTRY_ECONOMICS_BELOW_MINIMUM"}
    )
    soft_vetoes = set(decision["soft_vetoes"])
    context_warnings = sorted(
        soft_vetoes & {"ASSET_CONTEXT_UNAVAILABLE", "ASSET_BIAS_ADVERSE"}
    )
    route_warnings = sorted(
        soft_vetoes
        & {
            "STRATEGY_FAMILY_NOT_ROUTED_IN_REGIME",
            "STRATEGY_FAMILY_CLASSIFICATION_REQUIRED",
        }
    )
    return [
        {
            "layer": "data_quality",
            "status": "PASS" if not data_vetoes else "VETO",
            "veto_codes": data_vetoes,
        },
        {
            "layer": "mandate_and_instrument",
            "status": "PASS" if not mandate_vetoes else "VETO",
            "veto_codes": mandate_vetoes,
        },
        {
            "layer": "event_risk",
            "status": "PASS" if not event_vetoes else "VETO",
            "veto_codes": event_vetoes,
        },
        {
            "layer": "asset_context",
            "status": (
                "PASS" if not context_warnings else "WARN"
            ),
            "veto_codes": context_warnings,
        },
        {
            "layer": "asset_type_specialization",
            "status": (
                "PASS"
                if decision["asset_profile"]["status"]
                == "SPECIALIZED_DATA_COMPLETE"
                else "WARN"
            ),
            "veto_codes": decision["asset_profile"][
                "missing_components"
            ],
        },
        {
            "layer": "regime_router",
            "status": "PASS" if not route_warnings else "WARN",
            "veto_codes": route_warnings,
        },
        {
            "layer": "timeframe_setup",
            "status": "PASS" if hierarchy["ready"] else "UNKNOWN",
            "veto_codes": [] if hierarchy["ready"] else [hierarchy["status"]],
        },
        {
            "layer": "entry_economics",
            "status": "PASS" if not economics_vetoes else "VETO",
            "veto_codes": economics_vetoes,
        },
        {
            "layer": "entry_confirmation",
            "status": (
                "PASS"
                if gates["observed_tape_confirms"]
                else "UNKNOWN"
                if not gates["observed_tape_available"]
                else "WARN"
            ),
            "veto_codes": [],
        },
    ]


def _signal_funnel(
    *,
    input_count: int,
    candidates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "input_signals": int(input_count),
        "candidate_signals": len(candidates),
        "natural_strategy_candidate_signals": sum(
            candidate_evidence_classification(row)[
                "natural_strategy_candidate"
            ]
            for row in candidates
        ),
        "context_watchlist_signals": sum(
            not candidate_evidence_classification(row)[
                "natural_strategy_candidate"
            ]
            for row in candidates
        ),
        "shortlisted_symbols": len(observations),
        "hard_vetoed": sum(
            bool(row["decision_contract"]["hard_vetoes"])
            for row in observations
        ),
        "regime_routed": sum(
            bool(row["gates"]["regime_route_allowed"])
            for row in observations
        ),
        "timeframe_setup_ready": sum(
            bool(row["gates"]["timeframe_hierarchy_ready"])
            for row in observations
        ),
        "observed_tape_confirmed": sum(
            bool(row["gates"]["observed_tape_confirms"])
            for row in observations
        ),
        "execution_candidates_authority_none": sum(
            row["state"] == "EXECUTION_CANDIDATE_AUTHORITY_NONE"
            for row in observations
        ),
    }


def _append_episodes(
    project_root: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = project_root / PRIVATE_ROOT / "entry-episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing.add(str(value.get("episode_id")))
    appended: list[dict[str, Any]] = []
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if row["episode_id"] in existing:
                continue
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            existing.add(row["episode_id"])
            appended.append(row)
    return appended


def _public_status(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "hierarchical_entry_observer_status_v1",
        "status": payload["status"],
        "generated_at": payload["generated_at"],
        "architecture": payload["architecture"],
        "shortlist_count": payload["shortlist_count"],
        "state_counts": payload["state_counts"],
        "signal_funnel": payload["signal_funnel"],
        "asset_profile_counts": payload["asset_profile_counts"],
        "contract_identity_status_counts": payload[
            "contract_identity_status_counts"
        ],
        "market_regime": payload["market_regime"],
        "natural_strategy_candidate_count": payload[
            "natural_strategy_candidate_count"
        ],
        "context_watchlist_observation_count": payload[
            "context_watchlist_observation_count"
        ],
        "new_episode_count": payload["new_episode_count"],
        "new_natural_candidate_episode_count": payload[
            "new_natural_candidate_episode_count"
        ],
        "new_context_observation_episode_count": payload[
            "new_context_observation_episode_count"
        ],
        "observed_trade_store_available": payload[
            "observed_trade_store_available"
        ],
        "observed_orderbook_store_available": payload[
            "observed_orderbook_store_available"
        ],
        "bar_proxy_can_confirm_entry": False,
        "ml_status": payload["ml_status"],
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "order_calls": 0,
        "content_hash": payload["content_hash"],
    }


def _flatten_observation(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": row["episode_id"],
        "observed_at": row["observed_at"],
        "symbol": row["symbol"],
        "strategy_id": row["strategy_id"],
        "timeframe": row["timeframe"],
        "candidate_identity": row["candidate_identity"],
        "observation_identity": row["observation_identity"],
        "candidate_unit": row["candidate_unit"],
        "natural_strategy_candidate": row["natural_strategy_candidate"],
        "candidate_conditioned_evidence_eligible": row[
            "candidate_conditioned_evidence_eligible"
        ],
        "evidence_scope": row["evidence_scope"],
        "state": row["state"],
        "strategy_family": row["decision_contract"]["strategy_family"],
        "asset_class": row["decision_contract"]["asset_profile"][
            "asset_class"
        ],
        "asset_profile_status": row["decision_contract"][
            "asset_profile"
        ]["status"],
        "asset_profile_coverage": row["decision_contract"][
            "asset_profile"
        ]["data_coverage_ratio"],
        "asset_profile_score": row["decision_contract"][
            "asset_profile"
        ]["coverage_adjusted_score"],
        "feature_snapshot_hash": row["feature_snapshot_hash"],
        "hard_vetoes": "|".join(row["decision_contract"]["hard_vetoes"]),
        "soft_vetoes": "|".join(row["decision_contract"]["soft_vetoes"]),
        "setup_score_within_family": row["decision_contract"][
            "setup_score_within_family"
        ],
        "entry_confirmation_score": row["decision_contract"][
            "entry_confirmation_score"
        ],
        "timeframe_hierarchy_status": row["decision_contract"][
            "timeframe_hierarchy"
        ]["status"],
        "asset_bias_score": row["context_snapshot"]["asset_bias_score"],
        "asset_bias_confidence": row["context_snapshot"][
            "asset_bias_confidence"
        ],
        "tape_status": row["entry_snapshot"]["tape"].get("status"),
        "depth_status": row["entry_snapshot"]["depth"].get("status"),
        "automatic_execution_allowed": False,
        "execution_authority": "NONE",
    }


def _shortfall(tape: dict[str, Any]) -> float | None:
    spread = _optional(tape.get("spread_bps"))
    return round(spread / 2.0, 8) if spread is not None else None


def _tape_unavailable(reason: str = "OBSERVED_TRADE_STORE_UNAVAILABLE") -> dict[str, Any]:
    return {
        "status": reason,
        "data_class": "OBSERVED_TRADE_FLOW_UNAVAILABLE",
        "confidence": 0.0,
        "standalone_entry_authority": False,
    }


def _depth_unavailable(reason: str = "OBSERVED_DEPTH_STORE_UNAVAILABLE") -> dict[str, Any]:
    return {
        "status": reason,
        "data_class": "OBSERVED_ORDERBOOK_UNAVAILABLE",
        "confidence": 0.0,
        "standalone_entry_authority": False,
    }


def _depth_not_requested() -> dict[str, Any]:
    return {
        "status": "DEPTH_NOT_REQUESTED_BUDGET",
        "data_class": "OBSERVED_ORDERBOOK_NOT_REQUESTED",
        "confidence": 0.0,
        "standalone_entry_authority": False,
    }


def _optional(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(value: Any, default: float = 0.0) -> float:
    number = _optional(value)
    return default if number is None else number


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


__all__ = ["entry_observer_status", "observe_shortlist"]
