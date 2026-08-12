from __future__ import annotations

import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import FileLock, Timeout

from stocks.p3.io import atomic_write_json, read_json
from stocks.rl.contracts import (
    ACTION_NAMES,
    RLAction,
    action_name,
    decision_type_for_action,
    stable_hash,
)
from stocks.rl.data import (
    OBSERVATION_FEATURES,
    CausalFeatureScaler,
    DatasetContract,
    load_causal_dataset,
    load_multitimeframe_frames,
)
from stocks.rl.environment import FinanceSwingEnv, SwingEnvironmentConfig
from stocks.rl.experience import (
    ExperienceDecision,
    ExperienceOutcome,
    ExperienceStore,
)
from stocks.rl.reward import RewardInput, calculate_reward
from stocks.rl.registry import PolicyRegistry
from stocks.rl.training import load_maskable_policy, train_default_experiment


STATUS_PATH = Path("output/rl/status.json")
STATE_PATH = Path("runtime/rl-supervisor-state.json")
LOCK_PATH = Path("data/rl/private/supervisor.lock")


def run_supervisor_cycle(
    project_root: Path,
    *,
    allow_challenger_shadow: bool = True,
    allow_training: bool = True,
) -> dict[str, Any]:
    root = project_root.resolve()
    lock = FileLock(str(root / LOCK_PATH), timeout=0)
    try:
        with lock:
            return _run_locked_cycle(
                root,
                allow_challenger_shadow=allow_challenger_shadow,
                allow_training=allow_training,
            )
    except Timeout:
        return _publish_status(
            root,
            {
                "status": "DEGRADED",
                "blockers": ["RL_SUPERVISOR_ALREADY_RUNNING"],
                "training_status": "NOT_STARTED",
            },
        )


def _run_locked_cycle(
    root: Path,
    *,
    allow_challenger_shadow: bool,
    allow_training: bool,
) -> dict[str, Any]:
    store = ExperienceStore(root)
    registry = read_json(root / "models/rl/registry.json")
    policy_config = read_json(root / "config/rl_policy_v1.json")
    data_status = read_json(root / "output/p4/data-catalog-status.json")
    quote = read_json(root / "output/ibkr/live/quote-readiness.json")
    active = registry.get("active")
    challenger = registry.get("challenger")
    selected = active or (challenger if allow_challenger_shadow else None)
    role = "ACTIVE_SHADOW" if active else "CHALLENGER_SHADOW"
    blockers: list[str] = []
    alerts: list[str] = []
    decision_payload: dict[str, Any] | None = None
    last_reward: float | None = None

    scope = policy_config.get("initial_policy_scope", {})
    symbol = str(scope.get("symbol") or "SPUS").upper()
    source = root / f"data/research/critical_trading/yfinance/{symbol}.parquet"
    if not selected:
        blockers.append("RL_POLICY_NOT_TRAINED")
    elif not source.is_file():
        blockers.append(f"RL_PRICE_HISTORY_MISSING:{symbol}")
    else:
        try:
            record = registry.get("policies", {}).get(selected, {})
            verification = PolicyRegistry(root).verify(str(selected))
            if verification.get("status") != "GO":
                raise ValueError(
                    "RL_POLICY_INTEGRITY_BLOCKED:"
                    + ",".join(verification.get("blockers", []))
                )
            policy_root = root / str(record.get("path") or "")
            feature_schema = read_json(policy_root / "feature_schema.json")
            model_config = read_json(policy_root / "config.json")
            expected_environment = str(
                model_config.get("environment", {}).get("version") or ""
            )
            if expected_environment != SwingEnvironmentConfig().version:
                raise ValueError("RL_ENVIRONMENT_MISMATCH")
            if tuple(feature_schema.get("observation_features", [])) != tuple(
                OBSERVATION_FEATURES
            ):
                raise ValueError("RL_FEATURE_SCHEMA_MISMATCH")
            scaler_payload = feature_schema.get("scaler", {})
            scaler = CausalFeatureScaler.from_dict(scaler_payload)
            model = load_maskable_policy(policy_root / "model.zip")
            gates = data_status.get("gates", {})
            quote_valid = quote.get("quote_valid") is True and quote.get(
                "entitlement_state"
            ) == "PROVEN"
            relative_spread = quote.get("relative_spread")
            spread_bps = (
                float(relative_spread) * 10_000.0
                if quote_valid and _finite(relative_spread)
                else None
            )
            contract = DatasetContract(
                symbol=symbol,
                asset_class=str(scope.get("asset_class") or "ETF"),
                strategy_id="PPO_STOCKS_SWING_DECISION_LAYER_V1",
                timeframe=str(scope.get("timeframe") or "1d"),
                evidence_scope="LIVE_FORWARD_SHADOW",
                shariah_point_in_time_verified=bool(gates.get("SHARIAH_PIT_GO")),
                point_in_time_universe_verified=bool(gates.get("PIT_DATA_GO")),
                survivorship_verified=bool(gates.get("SURVIVORSHIP_GO")),
                source_name="LOCAL_SCHEDULED_MARKET_CACHE",
                source_license="RESEARCH_CACHE",
                source_version=str(record.get("training_data_end") or "UNKNOWN"),
            )
            frame = load_causal_dataset(
                source,
                contract,
                spread_bps=spread_bps,
                retain_unresolved_outcome=True,
                multitimeframe_frames=load_multitimeframe_frames(root, symbol),
            )
            latest_timestamp = pd.Timestamp(frame.iloc[-1]["timestamp"])
            fresh = _business_day_age(latest_timestamp) <= 2
            frame.loc[frame.index[-1], "gate_data_fresh"] = fresh
            frame.loc[frame.index[-1], "gate_quote_valid"] = quote_valid
            frame.loc[frame.index[-1], "gate_shariah"] = bool(
                gates.get("SHARIAH_PIT_GO")
            )
            last_window = frame.tail(max(3, min(256, len(frame)))).reset_index(drop=True)
            environment = FinanceSwingEnv(
                last_window,
                scaler=scaler,
                config=SwingEnvironmentConfig(
                    spread_bps=spread_bps or 5.0,
                    require_shariah_gate=True,
                    require_quote_gate=True,
                    require_execution_authority=False,
                ),
                seed=42,
            )
            observation, info = environment.reset(
                seed=42, options={"start_index": len(last_window) - 1}
            )
            mask = environment.action_masks()
            context = environment.decision_context
            _resolve_prior_decisions(store, symbol, context["row"])
            policy_decision = model(
                observation,
                mask,
                context,
                np.random.default_rng(42),
            )
            probabilities = list(policy_decision.probabilities)
            if not all(_finite(value) for value in probabilities):
                raise ValueError("RL_POLICY_OUTPUT_NAN")
            entropy = -sum(
                probability * math.log(max(probability, 1e-15))
                for probability in probabilities
                if probability > 0
            )
            drift_fraction = float(np.mean(np.abs(observation) >= 8.0))
            if drift_fraction > 0.10:
                alerts.append("FEATURE_DRIFT")
            if entropy < 0.01 and sum(mask) > 1:
                alerts.append("POLICY_ENTROPY_COLLAPSE")
            state_hash = stable_hash([round(float(value), 8) for value in observation])
            timestamp = latest_timestamp.isoformat()
            decision = ExperienceDecision(
                timestamp=timestamp,
                policy_version=selected,
                state_hash=state_hash,
                observation=[round(float(value), 8) for value in observation],
                available_actions=[ACTION_NAMES[index] for index in range(len(RLAction))],
                action_mask=mask.astype(int).tolist(),
                chosen_action=action_name(policy_decision.action),
                action_probability=float(probabilities[policy_decision.action]),
                portfolio_state={
                    "cash_pct": 1.0,
                    "exposure": 0.0,
                    "drawdown": 0.0,
                    "setup_score": float(context["row"]["setup_score"]),
                    "close": float(context["row"]["close"]),
                    "quote_valid": quote_valid,
                    "data_fresh": fresh,
                    "policy_role": role,
                    "expected_return": float(context["row"]["expected_return"]),
                    "expected_risk": float(context["row"]["expected_risk"]),
                    "expected_rr": float(context["row"]["expected_rr"]),
                    "signal_confidence": float(
                        context["row"]["signal_confidence"]
                    ),
                    "proposed_turnover": {
                        "OPEN_SMALL": 0.05,
                        "OPEN_NORMAL": 0.10,
                        "OPEN_LARGE": 0.125,
                    }.get(action_name(policy_decision.action), 0.0),
                },
                market_regime=str(context["row"]["market_regime"]),
                signal_id=stable_hash(
                    {"symbol": symbol, "timestamp": timestamp, "scope": "RL_SHADOW"}
                ),
                strategy_id=str(context["row"]["strategy_id"]),
                asset=symbol,
                episode_id=stable_hash(
                    {"policy": selected, "symbol": symbol, "timestamp": timestamp}
                )[:24],
                decision_type=decision_type_for_action(
                    policy_decision.action, position_open=False
                ),
                top_features=_top_features(observation, feature_schema),
                entry=float(context["row"]["close"]),
            )
            inserted = store.append_decision(decision)
            if inserted:
                store.append_episode_event(
                    decision.episode_id,
                    "DECISION_RECORDED",
                    {
                        "episode_id": decision.episode_id,
                        "decision_id": decision.decision_id,
                        "asset": symbol,
                        "action": decision.chosen_action,
                    },
                    timestamp=timestamp,
                )
            decision_payload = {
                "decision_id": decision.decision_id,
                "timestamp": timestamp,
                "policy_version": selected,
                "policy_role": role,
                "chosen_action": decision.chosen_action,
                "decision_type": decision.decision_type,
                "action_probability": decision.action_probability,
                "action_probabilities": {
                    ACTION_NAMES[index]: probability
                    for index, probability in enumerate(probabilities)
                },
                "available_actions": [
                    ACTION_NAMES[index]
                    for index, allowed in enumerate(mask)
                    if allowed
                ],
                "masked_actions": [
                    ACTION_NAMES[index]
                    for index, allowed in enumerate(mask)
                    if not allowed
                ],
                "market_regime": decision.market_regime,
                "setup_score": decision.portfolio_state["setup_score"],
                "quote_valid": quote_valid,
                "data_fresh": fresh,
                "policy_entropy": entropy,
                "feature_drift_fraction": drift_fraction,
                "top_features": decision.top_features,
                "logged": inserted,
                "execution_effect": "NONE",
                "execution_authority": "NONE",
                "broker_writes": 0,
            }
        except (ValueError, KeyError, OSError, RuntimeError) as exc:
            error_text = str(exc)
            blockers.append(
                f"RL_INFERENCE_FAIL_CLOSED:{type(exc).__name__}:{error_text}"
            )
            for marker in (
                "POLICY_HASH_MISMATCH",
                "feature scaler schema",
                "observation",
                "environment",
                "RL_ENVIRONMENT_MISMATCH",
                "RL_FEATURE_SCHEMA_MISMATCH",
                "RL_POLICY_OUTPUT_NAN",
            ):
                if marker.lower() in error_text.lower():
                    alerts.append(marker.upper().replace(" ", "_"))
            alerts.append("NO_RL_ACTION")

    experience = store.publish_status()
    training_status = _maybe_train(
        root,
        store,
        experience,
        allow_training=allow_training,
    )
    if experience.get("mean_reward") is not None:
        last_reward = float(experience["mean_reward"])
    alerts.extend(_policy_health_alerts(experience, decision_payload))
    if training_status.get("status") == "FAILED":
        alerts.append("TRAINING_FAILURE")
    status = {
        "status": (
            "SHADOW_ONLY"
            if selected and not blockers
            else "INSUFFICIENT_EVIDENCE"
        ),
        "blockers": blockers,
        "alerts": sorted(set(alerts)),
        "rl_mode": "SHADOW_ONLY",
        "rl_live_enabled": False,
        "active_policy": active,
        "challenger_policy": challenger,
        "inference_policy": selected,
        "last_inference": decision_payload,
        "last_reward": last_reward,
        "episodes": experience.get("decision_count", 0),
        "closed_episodes": experience.get("closed_episode_count", 0),
        "mean_reward": experience.get("mean_reward"),
        "rolling_reward": experience.get("rolling_50_mean_reward"),
        "net_pnl": experience.get("net_pnl"),
        "maximum_drawdown": experience.get("maximum_drawdown"),
        "policy_entropy": (
            decision_payload.get("policy_entropy") if decision_payload else None
        ),
        "action_distribution": experience.get("action_distribution", {}),
        "trade_frequency": _trade_frequency(experience),
        "skip_frequency": _skip_frequency(experience),
        "reward_by_regime": experience.get("reward_by_regime", {}),
        "pnl_by_regime": experience.get("pnl_by_regime", {}),
        "actions_by_regime": experience.get("actions_by_regime", {}),
        "incremental_evidence_go": False,
        "forward_evidence_go": False,
        "promotion_status": "NOT_ELIGIBLE",
        "training_status": training_status,
        "next_evaluation": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
        "next_training_check": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
        "fallback": "NO_RL_ACTION",
        "active_policy_frozen": True,
        "challenger_cannot_self_promote": True,
        "execution_authority": "NONE",
        "money_control": False,
        "orders_generated": 0,
        "broker_calls": 0,
        "broker_writes": 0,
        "generated_at": _now(),
    }
    return _publish_status(root, status)


def run_forever(
    project_root: Path,
    *,
    interval_seconds: int = 900,
    allow_training: bool = True,
) -> None:
    interval = max(60, int(interval_seconds))
    while True:
        run_supervisor_cycle(project_root, allow_training=allow_training)
        remaining = interval
        while remaining > 0:
            delay = min(60, remaining)
            time.sleep(delay)
            remaining -= delay


def _resolve_prior_decisions(
    store: ExperienceStore,
    symbol: str,
    latest_row: dict[str, Any],
) -> None:
    latest_timestamp = pd.Timestamp(latest_row["timestamp"])
    latest_close = float(latest_row["close"])
    for payload in store.unresolved_decisions(asset=symbol):
        decision_timestamp = pd.Timestamp(payload["timestamp"])
        if decision_timestamp >= latest_timestamp:
            continue
        entry = payload.get("entry")
        if not _finite(entry) or float(entry) <= 0:
            continue
        raw_return = latest_close / float(entry) - 1.0
        action = str(payload.get("chosen_action") or "HOLD")
        weights = {
            "OPEN_SMALL": 0.05,
            "OPEN_NORMAL": 0.10,
            "OPEN_LARGE": 0.125,
        }
        weight = weights.get(action, 0.0)
        cost = weight * 9.0 / 10_000.0 if weight else 0.0
        net_return = weight * raw_return - cost
        breakdown = calculate_reward(
            RewardInput(
                net_return=net_return,
                transaction_cost_return=cost,
                downside_return=min(0.0, net_return),
                turnover=weight,
                skipped_opportunity_return=raw_return if action in {"HOLD", "SKIP"} else 0.0,
            )
        )
        outcome = ExperienceOutcome(
            decision_id=str(payload["decision_id"]),
            episode_id=str(payload["episode_id"]),
            timestamp=latest_timestamp.isoformat(),
            reward=breakdown.total,
            reward_components=breakdown.components,
            realized_pnl=net_return,
            unrealized_pnl=0.0,
            fees=cost,
            slippage=0.0,
            mfe=max(0.0, raw_return),
            mae=min(0.0, raw_return),
            holding_duration=max(1, _business_day_age(decision_timestamp, latest_timestamp)),
            outcome={
                "status": "ONE_BAR_SHADOW_ATTRIBUTION",
                "raw_return": raw_return,
                "net_return": net_return,
                "chosen_action": action,
                "historical_backfill": False,
            },
        )
        if store.append_outcome(outcome):
            store.append_episode_event(
                str(payload["episode_id"]),
                "CLOSED",
                {
                    "episode_id": payload["episode_id"],
                    "decision_id": payload["decision_id"],
                    "outcome_id": outcome.outcome_id,
                },
                timestamp=latest_timestamp.isoformat(),
            )


def _maybe_train(
    root: Path,
    store: ExperienceStore,
    experience: dict[str, Any],
    *,
    allow_training: bool,
) -> dict[str, Any]:
    if not allow_training:
        return {"status": "DISABLED_FOR_THIS_CYCLE"}
    closed = int(experience.get("closed_episode_count", 0) or 0)
    minimum = 100
    if closed < minimum:
        return {
            "status": "NOT_DUE_MINIMUM_CLOSED_EPISODES",
            "closed_episodes": closed,
            "minimum": minimum,
        }
    job_key = f"rl-nightly-{datetime.now(UTC).date().isoformat()}"
    if not store.claim_training_job(job_key, {"closed_episodes": closed}):
        return {"status": "ALREADY_CLAIMED_OR_COMPLETED", "job_key": job_key}
    try:
        result = train_default_experiment(root)
        terminal = "COMPLETED" if result.get("status") else "FAILED"
        store.finish_training_job(job_key, status=terminal, payload=result)
        return {"status": terminal, "job_key": job_key, "result": result.get("status")}
    except Exception as exc:  # supervisor must publish a fail-closed artifact
        store.finish_training_job(
            job_key,
            status="FAILED",
            payload={"error_type": type(exc).__name__, "error": str(exc)},
        )
        return {
            "status": "FAILED",
            "job_key": job_key,
            "error_type": type(exc).__name__,
        }


def _publish_status(root: Path, status: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "schema": "rl_continual_supervisor_status_v1",
        "rl_mode": "SHADOW_ONLY",
        "rl_live_enabled": False,
        "execution_authority": "NONE",
        "money_control": False,
        "orders_generated": 0,
        "broker_calls": 0,
        "broker_writes": 0,
        "generated_at": _now(),
    }
    payload = {**defaults, **status}
    payload["content_hash"] = stable_hash(payload)
    atomic_write_json(root / STATUS_PATH, payload)
    atomic_write_json(
        root / STATE_PATH,
        {
            "schema": "rl_supervisor_state_v1",
            "last_cycle": payload["generated_at"],
            "last_status": payload.get("status"),
            "last_decision_id": (payload.get("last_inference") or {}).get("decision_id"),
            "content_hash": payload["content_hash"],
            "execution_authority": "NONE",
        },
    )
    return payload


def _top_features(observation: np.ndarray, feature_schema: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(feature_schema.get("observation_features", []))
    pairs = sorted(
        enumerate(observation), key=lambda pair: abs(float(pair[1])), reverse=True
    )[:10]
    return [
        {
            "feature": names[index] if index < len(names) else f"feature_{index}",
            "normalized_value": round(float(value), 8),
            "interpretation": "DEBUG_DIAGNOSTIC_NOT_CAUSAL_ATTRIBUTION",
        }
        for index, value in pairs
    ]


def _business_day_age(start: pd.Timestamp, end: pd.Timestamp | None = None) -> int:
    finish = end or pd.Timestamp.now(tz="UTC")
    return int(
        np.busday_count(
            start.tz_convert("UTC").date(),
            finish.tz_convert("UTC").date(),
        )
    )


def _trade_frequency(experience: dict[str, Any]) -> float:
    actions = experience.get("action_distribution", {})
    total = max(1, sum(int(value) for value in actions.values()))
    trades = sum(
        int(value) for name, value in actions.items() if str(name).startswith("OPEN_")
    )
    return trades / total


def _skip_frequency(experience: dict[str, Any]) -> float:
    actions = experience.get("action_distribution", {})
    total = max(1, sum(int(value) for value in actions.values()))
    return sum(int(actions.get(name, 0)) for name in ("HOLD", "SKIP")) / total


def _policy_health_alerts(
    experience: dict[str, Any],
    decision_payload: dict[str, Any] | None,
) -> list[str]:
    alerts: list[str] = []
    decisions = int(experience.get("decision_count", 0) or 0)
    if decisions >= 20:
        trade_frequency = _trade_frequency(experience)
        skip_frequency = _skip_frequency(experience)
        if skip_frequency >= 0.95:
            alerts.append("EXCESSIVE_HOLD")
        if trade_frequency >= 0.50:
            alerts.append("EXCESSIVE_TRADING")
        actions = experience.get("action_distribution", {})
        if (
            actions
            and max(int(value) for value in actions.values()) / decisions >= 0.98
        ):
            alerts.append("ACTION_DISTRIBUTION_COLLAPSE")
    average_turnover = experience.get("average_proposed_turnover")
    if _finite(average_turnover) and float(average_turnover) > 0.25:
        alerts.append("UNUSUAL_TURNOVER")
    mean_reward = experience.get("mean_reward")
    rolling_reward = experience.get("rolling_50_mean_reward")
    if _finite(mean_reward) and _finite(rolling_reward):
        if float(rolling_reward) < min(-0.25, float(mean_reward) - 0.50):
            alerts.append("REWARD_COLLAPSE")
    maximum_drawdown = experience.get("maximum_drawdown")
    if _finite(maximum_drawdown) and float(maximum_drawdown) > 0.10:
        alerts.append("DRAWDOWN_BREACH")
    if decision_payload and _finite(decision_payload.get("policy_entropy")):
        if (
            float(decision_payload["policy_entropy"]) < 0.01
            and len(decision_payload.get("available_actions", [])) > 1
        ):
            alerts.append("POLICY_ENTROPY_COLLAPSE")
    return alerts


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "LOCK_PATH",
    "STATE_PATH",
    "STATUS_PATH",
    "run_forever",
    "run_supervisor_cycle",
]
