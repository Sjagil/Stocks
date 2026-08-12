from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from stocks.data.multitimeframe import bar_freshness
from stocks.execution.idempotency import stable_hash
from stocks.portfolio.swing import StrategyTimeframeContract, stable_setup_identity


CONFIG_PATH = Path("config/active_swing/candidate_hypotheses_v1.json")
STATUS_PATH = Path("output/research/active_swing/candidate-generation-status.json")
NEAR_SETUP_STATE_PATH = Path("data/research/active_swing/private/near-setups-state.json")
PRIVATE_DATA_ROOT = Path("data/research/multitimeframe/private/provider=YFINANCE")
BAR_LENGTHS = {
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
}
SOURCE_POLICY = {
    "15m": ("15m",),
    "1h": ("1h",),
    "4h": ("1h", "4h"),
    "1d": ("1d",),
}
REQUIRED_COLUMNS = {
    "timestamp_utc",
    "session_date",
    "open",
    "high",
    "low",
    "close",
}
AUTHORITY = {
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "automatic_execution_allowed": False,
    "automatic_orders": 0,
    "broker_calls": 0,
    "order_calls": 0,
}


def generate_active_swing_candidates(
    project_root: Path,
    symbols: Iterable[str],
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Generate natural 15m setup candidates from native, closed RTH bars.

    These are immutable research hypotheses. They are intentionally not
    financial finalists, portfolio votes, sizing advice, or order intents.
    """
    now = _utc(observed_at or datetime.now(UTC))
    config = _read_json(project_root / CONFIG_PATH)
    blockers = _config_blockers(config)
    if blockers:
        return _publish(
            project_root,
            {
                "schema": "active_swing_15m_candidate_generation_v1",
                "status": "NO_GO",
                "generated_at": now.isoformat(),
                "blockers": blockers,
                "signals": [],
                **AUTHORITY,
            },
        )

    contract = StrategyTimeframeContract(
        entry_timeframe="15m",
        setup_timeframe=str(config["setup_timeframe"]),
        context_timeframes=("1h", "4h", "1d"),
        structural_timeframe="4h",
        management_timeframe="1h",
        exit_timeframe="15m",
        required_timeframes=tuple(config["required_timeframes"]),
        optional_timeframes=tuple(config["optional_timeframes"]),
        session=str(config["session"]),
    )
    declared_strategy_architectures = [
        {
            "strategy_id": str(hypothesis["strategy_id"]),
            "strategy_family": str(hypothesis["family"]),
            "strategy_dna_hash": stable_hash(
                {
                    "hypothesis": hypothesis,
                    "strategy_timeframe_contract": contract.as_dict(),
                    "config_version": config["version"],
                }
            ),
            "strategy_timeframe_contract": contract.as_dict(),
            "declaration_source": str(CONFIG_PATH).replace("\\", "/"),
            "qualification_status": config.get(
                "qualification_status",
                "UNQUALIFIED_FORWARD_OBSERVER",
            ),
            "financial_finalist": False,
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
        }
        for hypothesis in config["hypotheses"]
    ]
    normalized_symbols = sorted(
        {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    )
    previous_state = _read_json(project_root / NEAR_SETUP_STATE_PATH)
    previous_near_setups = {
        _setup_key(row): row
        for row in previous_state.get("near_setups", [])
        if isinstance(row, dict) and _setup_key(row) is not None
    }
    signals: list[dict[str, Any]] = []
    near_setups: list[dict[str, Any]] = []
    promoted_keys: set[tuple[str, str]] = set()
    symbol_rows: list[dict[str, Any]] = []
    for symbol in normalized_symbols:
        frames, frame_status = _load_frames(
            project_root,
            symbol=symbol,
            required=contract.required_timeframes,
            optional=contract.optional_timeframes,
            observed_at=now,
        )
        symbol_candidates: list[dict[str, Any]] = []
        symbol_near_setups: list[dict[str, Any]] = []
        if frame_status["required_timeframes_ready"]:
            for hypothesis in config["hypotheses"]:
                strategy_id = str(hypothesis["strategy_id"])
                key = (symbol, strategy_id)
                candidate = _candidate_for_hypothesis(
                    symbol=symbol,
                    frames=frames,
                    hypothesis=hypothesis,
                    contract=contract,
                    config=config,
                    observed_at=now,
                )
                if candidate is not None:
                    previous = previous_near_setups.get(key)
                    if _can_promote_persisted_near_setup(candidate, previous):
                        candidate = _promote_from_near_setup(candidate, previous)
                        promoted_keys.add(key)
                    symbol_candidates.append(candidate)
                    continue
                near_setup = _near_setup_for_hypothesis(
                    symbol=symbol,
                    frames=frames,
                    hypothesis=hypothesis,
                    contract=contract,
                    config=config,
                    observed_at=now,
                )
                if near_setup is not None:
                    symbol_near_setups.append(
                        _merge_near_setup_observation(
                            near_setup,
                            previous_near_setups.get(key),
                            observed_at=now,
                        )
                    )
        signals.extend(symbol_candidates)
        near_setups.extend(symbol_near_setups)
        symbol_rows.append(
            {
                "symbol_fingerprint": stable_hash({"symbol": symbol})[:20],
                **frame_status,
                "candidate_count": len(symbol_candidates),
                "candidate_setup_ids": [row["setup_id"] for row in symbol_candidates],
                "near_setup_count": len(symbol_near_setups),
                "near_setup_ids": [row["setup_id"] for row in symbol_near_setups],
            }
        )

    signals.sort(
        key=lambda row: (
            -float(row.get("confidence_score", 0.0)),
            str(row.get("ticker", "")),
            str(row.get("strategy_id", "")),
        )
    )
    native_ready = sum(bool(row.get("native_15m_ready")) for row in symbol_rows)
    context_ready = sum(bool(row.get("required_timeframes_ready")) for row in symbol_rows)
    near_setups.sort(
        key=lambda row: (
            float(row.get("distance_to_trigger_atr", math.inf)),
            str(row.get("ticker", "")),
            str(row.get("strategy_id", "")),
        )
    )
    current_near_keys = {key for row in near_setups if (key := _setup_key(row)) is not None}
    scanned_previous_keys = {key for key in previous_near_setups if key[0] in normalized_symbols}
    expired_keys = scanned_previous_keys - current_near_keys - promoted_keys
    retained_unscanned = [
        row for key, row in previous_near_setups.items() if key[0] not in normalized_symbols
    ]
    _publish_near_setup_state(
        project_root,
        observed_at=now,
        near_setups=[*retained_unscanned, *near_setups],
        scanned_symbol_count=len(normalized_symbols),
        expired_count=len(expired_keys),
        promoted_count=len(promoted_keys),
    )
    report = {
        "schema": "active_swing_15m_candidate_generation_v1",
        "status": "GO" if context_ready > 0 else "DATA_BLOCKED",
        "generated_at": now.isoformat(),
        "config_version": config["version"],
        "candidate_unit": config["candidate_unit"],
        "negative_sampling_policy": config["negative_sampling_policy"],
        "higher_timeframe_policy": config["higher_timeframe_policy"],
        "scanned_symbol_count": len(normalized_symbols),
        "native_15m_ready_symbol_count": native_ready,
        "required_context_ready_symbol_count": context_ready,
        "context_blocked_symbol_count": len(normalized_symbols) - context_ready,
        "hypothesis_count": len(config["hypotheses"]),
        "declared_strategy_architecture_count": len(declared_strategy_architectures),
        "declared_strategy_architectures": declared_strategy_architectures,
        "candidate_count": len(signals),
        "current_candidate_count": len(signals),
        "near_setup_count": len(near_setups),
        "near_setups": near_setups,
        "near_setup_expired_count": len(expired_keys),
        "near_setup_promoted_count": len(promoted_keys),
        "near_setups_create_candidates": False,
        "near_setups_change_strategy_thresholds": False,
        "near_setups_submit_orders": False,
        "expired_triggers_are_not_carried_forward": True,
        "native_15m_only": True,
        "derived_or_resampled_5m_as_15m_allowed": False,
        "all_timeframes_need_not_agree": True,
        "next_bar_target_used": False,
        "financial_finalist": False,
        "portfolio_eligible": False,
        "symbol_status": symbol_rows,
        "signals": signals,
        "blockers": [],
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    return _publish(project_root, report)


def active_swing_context_gaps(
    project_root: Path,
    symbols: Iterable[str],
    *,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    """Identify only dynamic symbols lacking fresh 1h/4h setup context."""
    now = _utc(observed_at or datetime.now(UTC))
    normalized_symbols = sorted(
        {str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()}
    )
    gap_rows: list[dict[str, Any]] = []
    for symbol in normalized_symbols:
        _frames, status = _load_frames(
            project_root,
            symbol=symbol,
            required=("1h", "4h"),
            optional=(),
            observed_at=now,
        )
        if status["required_timeframes_ready"]:
            continue
        gap_rows.append(
            {
                "symbol": symbol,
                "timeframe_status": status["timeframe_status"],
                "timeframe_freshness": status["timeframe_freshness"],
                "blockers": status["blockers"],
            }
        )
    return {
        "schema": "active_swing_context_gap_audit_v1",
        "status": "GO" if not gap_rows else "CONTEXT_REFRESH_REQUIRED",
        "generated_at": now.isoformat(),
        "scanned_symbol_count": len(normalized_symbols),
        "gap_symbol_count": len(gap_rows),
        "gap_symbols": [row["symbol"] for row in gap_rows],
        "gaps": gap_rows,
        "required_context_timeframes": ["1h", "4h"],
        "refresh_scope": "GAPS_ONLY",
        **AUTHORITY,
    }


def _candidate_for_hypothesis(
    *,
    symbol: str,
    frames: Mapping[str, pd.DataFrame],
    hypothesis: Mapping[str, Any],
    contract: StrategyTimeframeContract,
    config: Mapping[str, Any],
    observed_at: datetime,
) -> dict[str, Any] | None:
    frame = frames["15m"].copy().reset_index(drop=True)
    if len(frame) < 80:
        return None
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    atr = _atr_series(high, low, close, int(hypothesis["atr_bars"]))
    strategy_id = str(hypothesis["strategy_id"])
    if strategy_id == "ACTIVE-SWING-15M-BREAKOUT-V1":
        channel = int(hypothesis["channel_bars"])
        prior_high = high.shift(1).rolling(channel).max()
        active = close.gt(prior_high)
        event_mask = active & ~active.shift(1, fill_value=False)
        event_index = _latest_recent_event(
            event_mask,
            maximum_age_bars=int(config["maximum_trigger_age_bars"]),
        )
        if event_index is None:
            return None
        trigger_level = float(prior_high.iloc[event_index])
        event_reasons = [
            "NATIVE_15M_CLOSE_ABOVE_PRIOR_20_BAR_HIGH",
            "NATURAL_BREAKOUT_EVENT_NOT_EVERY_BAR",
        ]
    elif strategy_id == "ACTIVE-SWING-15M-PULLBACK-RESUMPTION-V1":
        fast = int(hypothesis["fast_ema"])
        slow = int(hypothesis["slow_ema"])
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        event_mask = (
            close.shift(1).le(ema_fast.shift(1))
            & close.gt(ema_fast)
            & close.gt(close.shift(1))
            & ema_fast.gt(ema_slow)
        )
        event_index = _latest_recent_event(
            event_mask,
            maximum_age_bars=int(config["maximum_trigger_age_bars"]),
        )
        if event_index is None:
            return None
        trigger_level = float(ema_fast.iloc[event_index])
        event_reasons = [
            "NATIVE_15M_CLOSE_RECLAIMS_EMA20",
            "EMA20_ABOVE_EMA50_AT_TRIGGER",
            "NATURAL_PULLBACK_RESUMPTION_EVENT_NOT_EVERY_BAR",
        ]
    else:
        return None

    event_row = frame.iloc[event_index]
    event_available_at = _available_at(event_row, "15m")
    expiration = event_available_at + timedelta(minutes=int(config["maximum_trigger_age_minutes"]))
    if event_available_at > observed_at or observed_at > expiration:
        return None
    event_close = float(close.iloc[event_index])
    event_atr = float(atr.iloc[event_index])
    if not math.isfinite(event_atr) or event_atr <= 0.0:
        return None
    lookback_start = max(0, event_index - 5)
    structural_low = float(low.iloc[lookback_start : event_index + 1].min())
    stop = max(
        structural_low,
        event_close - float(hypothesis["stop_atr"]) * event_atr,
    )
    if not 0.0 < stop < event_close:
        return None
    path_after_trigger = low.iloc[event_index + 1 :]
    if not path_after_trigger.empty and bool(path_after_trigger.le(stop).any()):
        return None

    timeframe_evidence = {
        "15m": _timeframe_state(frame.iloc[: event_index + 1], "15m"),
    }
    for timeframe in ("1h", "4h", "1d"):
        source = frames.get(timeframe)
        if source is None or source.empty:
            timeframe_evidence[timeframe] = _missing_timeframe_state(timeframe)
            continue
        causal = _causal_frame(
            source,
            timeframe=timeframe,
            decision_at=event_available_at,
            decision_session=str(event_row["session_date"]),
        )
        timeframe_evidence[timeframe] = (
            _timeframe_state(causal, timeframe)
            if not causal.empty
            else _missing_timeframe_state(timeframe)
        )
    if not all(
        bool(timeframe_evidence[timeframe]["available"])
        for timeframe in contract.required_timeframes
    ):
        return None

    supportive = sum(
        timeframe_evidence[timeframe]["trend_state"] == "SUPPORTIVE"
        for timeframe in ("1h", "4h", "1d")
        if timeframe_evidence[timeframe]["available"]
    )
    adverse = sum(
        timeframe_evidence[timeframe]["trend_state"] == "ADVERSE"
        for timeframe in ("1h", "4h", "1d")
        if timeframe_evidence[timeframe]["available"]
    )
    context_count = sum(
        timeframe_evidence[timeframe]["available"] for timeframe in ("1h", "4h", "1d")
    )
    context_score = (supportive - adverse) / max(1, context_count)
    trigger_strength = max(0.0, min(1.0, (event_close / trigger_level - 1.0) * 40.0))
    ranking_score = max(
        0.35,
        min(0.80, 0.50 + 0.10 * context_score + 0.15 * trigger_strength),
    )
    lifecycle = "SETUP_VALID" if supportive >= 1 else "WATCHING"
    risk = event_close - stop
    target_1 = event_close + float(hypothesis["target_1_r"]) * risk
    target_2 = event_close + float(hypothesis["target_2_r"]) * risk
    setup_id = stable_setup_identity(
        symbol=symbol,
        strategy_id=strategy_id,
        setup_origin_timestamp=event_available_at.isoformat(),
        setup_timeframe=contract.setup_timeframe,
    )
    evidence_hash = stable_hash(timeframe_evidence)
    return {
        "signal_id": "SIG-AS15-" + setup_id[:24],
        "setup_id": setup_id,
        "candidate_identity": setup_id,
        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "asset": symbol,
        "ticker": symbol,
        "strategy_id": strategy_id,
        "strategy_family": str(hypothesis["family"]),
        "strategy_dna_hash": stable_hash(dict(hypothesis)),
        "strategy_timeframe_contract": contract.as_dict(),
        "timeframe_evidence": timeframe_evidence,
        "timeframe_evidence_hash": evidence_hash,
        "timeframe": "15m",
        "setup_origin_timestamp": event_available_at.isoformat(),
        "trigger_event_timestamp": event_available_at.isoformat(),
        "trigger_bar_timestamp": pd.Timestamp(event_row["timestamp_utc"]).isoformat(),
        "prior_bar_timestamp": (
            pd.Timestamp(frame.iloc[event_index - 1]["timestamp_utc"]).isoformat()
            if event_index > 0
            else None
        ),
        "signal_timestamp": observed_at.isoformat(),
        "data_timestamp": event_available_at.isoformat(),
        "candidate_observed_at": observed_at.isoformat(),
        "expiration_timestamp": expiration.isoformat(),
        "expiration_policy": "STRICT_WALL_CLOCK_NO_SESSION_EXTENSION",
        "strict_expiration": True,
        "data_freshness": "FRESH",
        "bar_closed": True,
        "session": "RTH",
        "source_provider": str(event_row.get("provider") or "YFINANCE"),
        "source_interval": str(event_row.get("source_interval") or "15m"),
        "bar_origin": str(event_row.get("bar_origin") or "NATIVE"),
        "exchange_timezone": str(event_row.get("exchange_timezone") or ""),
        "action": "WATCHLIST",
        "original_action": "BUY",
        "lifecycle_state": lifecycle,
        "lifecycle_status": "WATCHLIST",
        "current_market_price": round(event_close, 4),
        "preferred_entry": round(event_close, 4),
        "entry_zone_low": round(event_close - 0.10 * risk, 4),
        "entry_zone_high": round(event_close + 0.10 * risk, 4),
        "limit_entry_price": round(event_close, 4),
        "invalidation_level": round(stop, 4),
        "stop_loss": round(stop, 4),
        "stop_method": "15M_EVENT_STRUCTURE_WITH_ATR_CAP",
        "take_profit_1": round(target_1, 4),
        "take_profit_2": round(target_2, 4),
        "take_profit_mode": "COUNTERFACTUAL_1_5R_2_5R_THEN_TRAIL",
        "reward_risk_1": float(hypothesis["target_1_r"]),
        "reward_risk_2": float(hypothesis["target_2_r"]),
        "expected_holding_period": "1-20 sessions; maximum outcome window 80 closed 15m bars",
        "confidence_score": round(ranking_score, 6),
        "confidence_semantics": "HEURISTIC_RANK_ONLY_NOT_CALIBRATED_PROBABILITY",
        "multi_timeframe_alignment_score": round(context_score, 6),
        "higher_timeframe_consensus_required": False,
        "higher_timeframe_context": "CAUSAL_1H_4H_WITH_OPTIONAL_PRIOR_1D",
        "reasons": [
            *event_reasons,
            "CAUSAL_HIGHER_TIMEFRAME_CONTEXT_ATTACHED",
            "ALL_TIMEFRAMES_NEED_NOT_AGREE",
        ],
        "risks": [
            "UNQUALIFIED_FORWARD_OBSERVER",
            "NOT_FINANCIAL_FINALIST",
            "NO_CALIBRATED_PROFIT_PROBABILITY",
            "EXECUTION_AUTHORITY_NONE",
        ],
        "qualification_status": "UNQUALIFIED_FORWARD_OBSERVER",
        "financial_finalist": False,
        "portfolio_eligible": False,
        "deployment_eligible": False,
        "execution_eligible": False,
        "negative_sampling_policy": "CANDIDATE_CONDITIONED_ONLY",
        "next_bar_target_used": False,
        "suggested_quantity": 0,
        "maximum_order_value_eur": 0,
        "maximum_planned_loss_eur": 0,
        "estimated_transaction_costs_eur": 0,
        **AUTHORITY,
    }


def _near_setup_for_hypothesis(
    *,
    symbol: str,
    frames: Mapping[str, pd.DataFrame],
    hypothesis: Mapping[str, Any],
    contract: StrategyTimeframeContract,
    config: Mapping[str, Any],
    observed_at: datetime,
) -> dict[str, Any] | None:
    """Observe a pre-trigger condition without creating a trade candidate."""
    frame = frames["15m"].copy().reset_index(drop=True)
    if len(frame) < 80:
        return None
    close = frame["close"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    strategy_id = str(hypothesis["strategy_id"])
    atr = _atr_series(high, low, close, int(hypothesis["atr_bars"]))
    proximity = float(config.get("near_setup_atr_fraction", 0.5))

    if strategy_id == "ACTIVE-SWING-15M-BREAKOUT-V1":
        channel = int(hypothesis["channel_bars"])
        trigger = high.shift(1).rolling(channel).max()
        distance = trigger - close
        near_mask = distance.ge(0.0) & distance.le(proximity * atr)
        awaited_trigger = "CLOSE_ABOVE_PRIOR_20_BAR_HIGH"
        reasons = [
            "CLOSE_WITHIN_CONFIGURED_ATR_DISTANCE_OF_PRIOR_20_BAR_HIGH",
            "BREAKOUT_HAS_NOT_OCCURRED",
        ]
    elif strategy_id == "ACTIVE-SWING-15M-PULLBACK-RESUMPTION-V1":
        fast = int(hypothesis["fast_ema"])
        slow = int(hypothesis["slow_ema"])
        ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
        trigger = ema_fast
        distance = trigger - close
        near_mask = ema_fast.gt(ema_slow) & distance.ge(0.0) & distance.le(proximity * atr)
        awaited_trigger = "CLOSE_RECLAIMS_EMA20_WHILE_EMA20_ABOVE_EMA50"
        reasons = [
            "CLOSE_WITHIN_CONFIGURED_ATR_DISTANCE_BELOW_EMA20",
            "EMA20_ABOVE_EMA50",
            "EMA20_RECLAIM_HAS_NOT_OCCURRED",
        ]
    else:
        return None

    last_index = len(frame) - 1
    if not bool(near_mask.fillna(False).iloc[last_index]):
        return None
    last_atr = float(atr.iloc[last_index])
    last_trigger = float(trigger.iloc[last_index])
    last_distance = float(distance.iloc[last_index])
    if (
        not math.isfinite(last_atr)
        or last_atr <= 0.0
        or not math.isfinite(last_trigger)
        or not math.isfinite(last_distance)
    ):
        return None

    origin_index = last_index
    while origin_index > 0 and bool(near_mask.fillna(False).iloc[origin_index - 1]):
        origin_index -= 1
    maximum_age_bars = int(config.get("maximum_near_setup_age_bars", 64))
    age_bars = last_index - origin_index
    if age_bars >= maximum_age_bars:
        return None

    current_row = frame.iloc[last_index]
    current_available_at = _available_at(current_row, "15m")
    if current_available_at > observed_at:
        return None
    origin_available_at = _available_at(frame.iloc[origin_index], "15m")
    timeframe_evidence = {"15m": _timeframe_state(frame.iloc[: last_index + 1], "15m")}
    for timeframe in ("1h", "4h", "1d"):
        source = frames.get(timeframe)
        if source is None or source.empty:
            timeframe_evidence[timeframe] = _missing_timeframe_state(timeframe)
            continue
        causal = _causal_frame(
            source,
            timeframe=timeframe,
            decision_at=current_available_at,
            decision_session=str(current_row["session_date"]),
        )
        timeframe_evidence[timeframe] = (
            _timeframe_state(causal, timeframe)
            if not causal.empty
            else _missing_timeframe_state(timeframe)
        )
    if not all(
        bool(timeframe_evidence[timeframe]["available"])
        for timeframe in contract.required_timeframes
    ):
        return None

    setup_id = stable_setup_identity(
        symbol=symbol,
        strategy_id=strategy_id,
        setup_origin_timestamp=origin_available_at.isoformat(),
        setup_timeframe=contract.setup_timeframe,
    )
    return {
        "schema": "active_swing_near_setup_v1",
        "setup_id": setup_id,
        "candidate_identity": None,
        "candidate_unit": "PRE_TRIGGER_OBSERVATION_NOT_A_CANDIDATE",
        "asset": symbol,
        "ticker": symbol,
        "strategy_id": strategy_id,
        "strategy_family": str(hypothesis["family"]),
        "strategy_dna_hash": stable_hash(dict(hypothesis)),
        "strategy_timeframe_contract": contract.as_dict(),
        "timeframe_evidence": timeframe_evidence,
        "timeframe_evidence_hash": stable_hash(timeframe_evidence),
        "timeframe": "15m",
        "setup_origin_timestamp": origin_available_at.isoformat(),
        "last_bar_timestamp": pd.Timestamp(current_row["timestamp_utc"]).isoformat(),
        "data_timestamp": current_available_at.isoformat(),
        "last_observed_at": observed_at.isoformat(),
        "near_setup_age_bars": age_bars,
        "maximum_near_setup_age_bars": maximum_age_bars,
        "awaited_trigger": awaited_trigger,
        "trigger_level": round(last_trigger, 8),
        "current_market_price": round(float(close.iloc[last_index]), 8),
        "distance_to_trigger": round(last_distance, 8),
        "distance_to_trigger_atr": round(last_distance / last_atr, 8),
        "near_setup_atr_fraction": proximity,
        "lifecycle_state": "NEAR_SETUP",
        "lifecycle_status": "WATCHLIST",
        "qualification_status": "UNQUALIFIED_NEAR_SETUP_OBSERVER",
        "reasons": [
            *reasons,
            "CAUSAL_CLOSED_BAR_CONTEXT_ATTACHED",
            "EXACT_STRATEGY_TRIGGER_STILL_REQUIRED",
        ],
        "risks": [
            "NOT_A_NATURAL_CANDIDATE",
            "NOT_FINANCIAL_FINALIST",
            "NO_CALIBRATED_PROFIT_PROBABILITY",
            "EXECUTION_AUTHORITY_NONE",
        ],
        "does_not_create_candidate": True,
        "does_not_change_strategy_thresholds": True,
        "financial_finalist": False,
        "portfolio_eligible": False,
        "deployment_eligible": False,
        "execution_eligible": False,
        "suggested_quantity": 0,
        "maximum_order_value_eur": 0,
        "maximum_planned_loss_eur": 0,
        **AUTHORITY,
    }


def _setup_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    symbol = str(row.get("ticker") or row.get("asset") or "").upper()
    strategy_id = str(row.get("strategy_id") or "")
    return (symbol, strategy_id) if symbol and strategy_id else None


def _merge_near_setup_observation(
    current: dict[str, Any],
    previous: Mapping[str, Any] | None,
    *,
    observed_at: datetime,
) -> dict[str, Any]:
    same_setup = bool(
        previous
        and previous.get("setup_id") == current.get("setup_id")
        and previous.get("lifecycle_state") == "NEAR_SETUP"
    )
    current["first_observed_at"] = (
        str(previous.get("first_observed_at"))
        if same_setup and previous and previous.get("first_observed_at")
        else observed_at.isoformat()
    )
    current["observation_count"] = (
        int(previous.get("observation_count", 0) or 0) + 1 if same_setup and previous else 1
    )
    return current


def _can_promote_persisted_near_setup(
    candidate: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> bool:
    return bool(
        previous
        and previous.get("lifecycle_state") == "NEAR_SETUP"
        and previous.get("last_bar_timestamp") == candidate.get("prior_bar_timestamp")
        and _setup_key(previous) == _setup_key(candidate)
    )


def _promote_from_near_setup(
    candidate: dict[str, Any], previous: Mapping[str, Any]
) -> dict[str, Any]:
    setup_id = str(previous["setup_id"])
    promoted = dict(candidate)
    promoted.update(
        {
            "signal_id": "SIG-AS15-" + setup_id[:24],
            "setup_id": setup_id,
            "candidate_identity": setup_id,
            "setup_origin_timestamp": previous["setup_origin_timestamp"],
            "promoted_from_near_setup": True,
            "near_setup_first_observed_at": previous.get("first_observed_at"),
            "near_setup_observation_count": int(previous.get("observation_count", 0) or 0),
        }
    )
    promoted["reasons"] = [
        *promoted.get("reasons", []),
        "PROMOTED_FROM_DIRECTLY_PRECEDING_PERSISTED_NEAR_SETUP",
    ]
    return promoted


def _load_frames(
    project_root: Path,
    *,
    symbol: str,
    required: Iterable[str],
    optional: Iterable[str],
    observed_at: datetime,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    statuses: dict[str, str] = {}
    sources: dict[str, str | None] = {}
    freshness: dict[str, dict[str, Any]] = {}
    for timeframe in (*required, *optional):
        path = _bar_path(project_root, symbol=symbol, timeframe=timeframe)
        if path is None:
            statuses[timeframe] = "MISSING_CANONICAL_PARTITION"
            sources[timeframe] = None
            freshness[timeframe] = {"status": "TIMESTAMP_UNAVAILABLE_BLOCKED"}
            continue
        frame = _closed_frame(path, timeframe=timeframe, observed_at=observed_at)
        if frame.empty:
            statuses[timeframe] = "NO_CAUSALLY_AVAILABLE_CLOSED_BARS"
            sources[timeframe] = path.parent.name.removeprefix("source_interval=")
            freshness[timeframe] = {"status": "TIMESTAMP_UNAVAILABLE_BLOCKED"}
            continue
        latest = frame.iloc[-1]
        timeframe_freshness = bar_freshness(
            latest["timestamp_utc"],
            interval=timeframe,
            observed_at=observed_at,
            exchange_timezone=(str(latest.get("exchange_timezone") or "").strip() or None),
        )
        freshness[timeframe] = timeframe_freshness
        if timeframe_freshness.get("status") != "FRESH_CLOSED_BAR":
            statuses[timeframe] = str(timeframe_freshness.get("status"))
            sources[timeframe] = path.parent.name.removeprefix("source_interval=")
            continue
        frames[timeframe] = frame
        statuses[timeframe] = "READY"
        sources[timeframe] = path.parent.name.removeprefix("source_interval=")
    required_ready = all(statuses.get(value) == "READY" for value in required)
    return frames, {
        "native_15m_ready": (statuses.get("15m") == "READY" and sources.get("15m") == "15m"),
        "required_timeframes_ready": required_ready,
        "timeframe_status": statuses,
        "timeframe_freshness": freshness,
        "source_intervals": sources,
        "blockers": [
            f"REQUIRED_TIMEFRAME_NOT_READY:{timeframe}:{statuses.get(timeframe)}"
            for timeframe in required
            if statuses.get(timeframe) != "READY"
        ],
    }


def _bar_path(project_root: Path, *, symbol: str, timeframe: str) -> Path | None:
    root = project_root / PRIVATE_DATA_ROOT / f"symbol={symbol}" / f"interval={timeframe}"
    for source in SOURCE_POLICY[timeframe]:
        path = root / f"source_interval={source}" / "bars.parquet"
        if path.is_file():
            return path
    return None


def _closed_frame(path: Path, *, timeframe: str, observed_at: datetime) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    if frame.empty or not REQUIRED_COLUMNS.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    work = frame.copy()
    if "quality_status" in work:
        work = work.loc[work["quality_status"].astype(str).str.startswith("VALIDATED")]
    for field in ("is_partial", "partial_bucket"):
        if field in work:
            partial = work[field].astype("boolean").fillna(False)
            work = work.loc[~partial]
    if timeframe in BAR_LENGTHS and "session" in work:
        work = work.loc[work["session"].astype(str).str.upper().eq("RTH")]
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    knowledge_field = next(
        (field for field in ("ingested_at", "received_at", "fetched_at") if field in work),
        None,
    )
    if knowledge_field is not None:
        knowledge_at = pd.to_datetime(work[knowledge_field], utc=True, errors="coerce")
        work = work.loc[knowledge_at.notna() & knowledge_at.le(pd.Timestamp(observed_at))]
    work = work.dropna(subset=["timestamp_utc", "open", "high", "low", "close"])
    available = work.apply(lambda row: _available_at(row, timeframe), axis=1)
    work = work.loc[pd.to_datetime(available, utc=True).le(pd.Timestamp(observed_at))]
    return (
        work.sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
        .reset_index(drop=True)
    )


def _causal_frame(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    decision_at: datetime,
    decision_session: str,
) -> pd.DataFrame:
    if timeframe == "1d":
        return frame.loc[frame["session_date"].astype(str).lt(str(decision_session))].copy()
    available = frame.apply(lambda row: _available_at(row, timeframe), axis=1)
    return frame.loc[pd.to_datetime(available, utc=True).le(pd.Timestamp(decision_at))].copy()


def _available_at(row: Mapping[str, Any], timeframe: str) -> datetime:
    timestamp = pd.Timestamp(row["timestamp_utc"])
    timestamp = (
        timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    )
    if timeframe == "1d":
        return (timestamp + pd.Timedelta(days=1)).to_pydatetime()
    return (timestamp + pd.Timedelta(BAR_LENGTHS[timeframe])).to_pydatetime()


def _latest_recent_event(mask: pd.Series, *, maximum_age_bars: int) -> int | None:
    indices = list(mask.index[mask.fillna(False)])
    if not indices:
        return None
    latest = int(indices[-1])
    return latest if len(mask) - 1 - latest < maximum_age_bars else None


def _timeframe_state(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    if frame.empty or len(frame) < 20:
        return _missing_timeframe_state(timeframe)
    close = frame["close"].astype(float)
    ema_fast = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema_slow = close.ewm(span=50, adjust=False, min_periods=50).mean()
    last = float(close.iloc[-1])
    fast = float(ema_fast.iloc[-1]) if math.isfinite(float(ema_fast.iloc[-1])) else None
    slow = float(ema_slow.iloc[-1]) if math.isfinite(float(ema_slow.iloc[-1])) else None
    if fast is not None and slow is not None and last > fast > slow:
        trend = "SUPPORTIVE"
    elif fast is not None and slow is not None and last < fast < slow:
        trend = "ADVERSE"
    else:
        trend = "MIXED"
    return {
        "timeframe": timeframe,
        "available": True,
        "bar_closed": True,
        "bar_timestamp": pd.Timestamp(frame["timestamp_utc"].iloc[-1]).isoformat(),
        "available_at": _available_at(frame.iloc[-1], timeframe).isoformat(),
        "knowledge_available_at": _knowledge_available_at(frame.iloc[-1]),
        "source_interval": str(frame.iloc[-1].get("source_interval") or timeframe),
        "bar_origin": str(frame.iloc[-1].get("bar_origin") or "UNKNOWN"),
        "trend_state": trend,
        "close": round(last, 8),
        "ema20": round(fast, 8) if fast is not None else None,
        "ema50": round(slow, 8) if slow is not None else None,
        "return_1_bar": (
            round(last / float(close.iloc[-2]) - 1.0, 8)
            if len(close) >= 2 and float(close.iloc[-2]) > 0.0
            else None
        ),
        "return_4_bars": (
            round(last / float(close.iloc[-5]) - 1.0, 8)
            if len(close) >= 5 and float(close.iloc[-5]) > 0.0
            else None
        ),
    }


def _missing_timeframe_state(timeframe: str) -> dict[str, Any]:
    return {
        "timeframe": timeframe,
        "available": False,
        "bar_closed": None,
        "bar_timestamp": None,
        "available_at": None,
        "knowledge_available_at": None,
        "source_interval": None,
        "bar_origin": None,
        "trend_state": "UNAVAILABLE",
        "close": None,
        "ema20": None,
        "ema50": None,
        "return_1_bar": None,
        "return_4_bars": None,
    }


def _atr_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _knowledge_available_at(row: Mapping[str, Any]) -> str | None:
    for field in ("ingested_at", "received_at", "fetched_at"):
        value = row.get(field)
        if value in (None, ""):
            continue
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if not pd.isna(parsed):
            return pd.Timestamp(parsed).isoformat()
    return None


def _config_blockers(config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if config.get("schema") != "active_swing_15m_candidate_hypotheses_v1":
        blockers.append("CONFIG_SCHEMA_INVALID_OR_MISSING")
    if config.get("status") != "ACTIVE_RESEARCH_SHADOW_ONLY":
        blockers.append("CONFIG_NOT_RESEARCH_SHADOW_ONLY")
    if config.get("entry_timeframe") != "15m":
        blockers.append("ENTRY_TIMEFRAME_MUST_BE_15M")
    if config.get("strategy_authority") != "NONE":
        blockers.append("STRATEGY_AUTHORITY_MUST_BE_NONE")
    if config.get("execution_authority") != "NONE":
        blockers.append("EXECUTION_AUTHORITY_MUST_BE_NONE")
    if config.get("automatic_execution_allowed") is not False:
        blockers.append("AUTOMATIC_EXECUTION_MUST_BE_FALSE")
    if not isinstance(config.get("hypotheses"), list) or not config.get("hypotheses"):
        blockers.append("BOUNDED_HYPOTHESES_REQUIRED")
        return blockers
    hypotheses = config["hypotheses"]
    if len(hypotheses) > 5:
        blockers.append("HYPOTHESIS_COUNT_EXCEEDS_BOUNDED_LIMIT")
    required = config.get("required_timeframes")
    if not isinstance(required, list) or not {
        "15m",
        "1h",
        "4h",
    }.issubset(set(required)):
        blockers.append("REQUIRED_15M_1H_4H_CONTEXT_MISSING")
    optional = config.get("optional_timeframes")
    if not isinstance(optional, list):
        blockers.append("OPTIONAL_TIMEFRAMES_MUST_BE_LIST")
    if config.get("session") != "RTH":
        blockers.append("RTH_SESSION_REQUIRED")
    try:
        age_bars = int(config.get("maximum_trigger_age_bars"))
        age_minutes = int(config.get("maximum_trigger_age_minutes"))
    except (TypeError, ValueError):
        blockers.append("TRIGGER_EXPIRY_INVALID")
    else:
        if not 1 <= age_bars <= 32 or not 15 <= age_minutes <= 480:
            blockers.append("TRIGGER_EXPIRY_OUT_OF_BOUNDS")
    try:
        near_fraction = float(config.get("near_setup_atr_fraction", 0.5))
        near_age = int(config.get("maximum_near_setup_age_bars", 64))
    except (TypeError, ValueError):
        blockers.append("NEAR_SETUP_OBSERVER_BOUNDS_INVALID")
    else:
        if not 0.0 < near_fraction <= 1.0 or not 1 <= near_age <= 256:
            blockers.append("NEAR_SETUP_OBSERVER_BOUNDS_OUT_OF_RANGE")
    allowed_ids = {
        "ACTIVE-SWING-15M-BREAKOUT-V1",
        "ACTIVE-SWING-15M-PULLBACK-RESUMPTION-V1",
    }
    identifiers = [
        str(row.get("strategy_id") or "") for row in hypotheses if isinstance(row, Mapping)
    ]
    if len(identifiers) != len(hypotheses):
        blockers.append("HYPOTHESIS_MAPPING_REQUIRED")
    if len(set(identifiers)) != len(identifiers):
        blockers.append("DUPLICATE_HYPOTHESIS_ID")
    if not set(identifiers).issubset(allowed_ids):
        blockers.append("UNREGISTERED_HYPOTHESIS_BLOCKED")
    required_fields = {
        "ACTIVE-SWING-15M-BREAKOUT-V1": {
            "family",
            "channel_bars",
            "atr_bars",
            "stop_atr",
            "target_1_r",
            "target_2_r",
        },
        "ACTIVE-SWING-15M-PULLBACK-RESUMPTION-V1": {
            "family",
            "fast_ema",
            "slow_ema",
            "atr_bars",
            "stop_atr",
            "target_1_r",
            "target_2_r",
        },
    }
    for row in hypotheses:
        if not isinstance(row, Mapping):
            continue
        strategy_id = str(row.get("strategy_id") or "")
        missing = required_fields.get(strategy_id, set()) - set(row)
        if missing:
            blockers.append(f"HYPOTHESIS_FIELDS_MISSING:{strategy_id}:" + ",".join(sorted(missing)))
    return blockers


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _publish(project_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    path = project_root / STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    public = {key: value for key, value in report.items() if key != "signals"}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(public, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return report


def _publish_near_setup_state(
    project_root: Path,
    *,
    observed_at: datetime,
    near_setups: list[dict[str, Any]],
    scanned_symbol_count: int,
    expired_count: int,
    promoted_count: int,
) -> None:
    payload = {
        "schema": "active_swing_near_setup_state_v1",
        "updated_at": observed_at.isoformat(),
        "near_setup_count": len(near_setups),
        "scanned_symbol_count": scanned_symbol_count,
        "expired_in_scan_count": expired_count,
        "promoted_in_scan_count": promoted_count,
        "near_setups": near_setups,
        "candidate_creation_allowed": False,
        "strategy_threshold_changes_allowed": False,
        **AUTHORITY,
    }
    payload["content_hash"] = stable_hash(payload)
    path = project_root / NEAR_SETUP_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


__all__ = ["generate_active_swing_candidates"]
