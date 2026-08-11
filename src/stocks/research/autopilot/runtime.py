from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from stocks.research.autopilot.contracts import stable_hash
from stocks.research.autopilot.service import (
    autopilot_campaign,
    autopilot_generate,
    autopilot_leaderboard,
    autopilot_status as research_status,
)
from stocks.research.phase11_12 import COSTS_BPS, run_phase11_12
from stocks.research.promotion import recover_survivors


FAMILIES = (
    "quality_momentum",
    "trend_pullback",
    "etf_rotation",
    "volatility_contraction_breakout",
    "commodity_etf_trend",
)


def runtime_command(project_root: Path, command: str) -> dict[str, Any]:
    if command == "run-once":
        return run_once(project_root)
    if command == "leaderboard":
        return autopilot_leaderboard()
    state = _load_state(project_root)
    now = datetime.now(UTC)
    if command == "start":
        state.update(
            enabled=True,
            paused=False,
            scheduler_status="SCHEDULE_ENABLED_EXTERNAL_TRIGGER_REQUIRED",
            next_cycle_at=now.isoformat(),
        )
    elif command == "stop":
        state.update(
            enabled=False,
            paused=False,
            scheduler_status="STOPPED",
            next_cycle_at=None,
        )
    elif command == "pause":
        state.update(paused=True, scheduler_status="PAUSED")
    elif command == "resume":
        state.update(
            enabled=True,
            paused=False,
            scheduler_status="SCHEDULE_ENABLED_EXTERNAL_TRIGGER_REQUIRED",
            next_cycle_at=now.isoformat(),
        )
    elif command == "failures":
        return {
            "status": "GO",
            "failure_count": len(state.get("failures", [])),
            "failures": state.get("failures", []),
            **_authority(),
        }
    elif command != "status":
        return _blocked("UNKNOWN_AUTOPILOT_COMMAND")
    if command != "status":
        _save_state(project_root, state)
    return _status_payload(state)


def run_once(project_root: Path) -> dict[str, Any]:
    state = _load_state(project_root)
    if not state["enabled"]:
        return _blocked("AUTOPILOT_DISABLED")
    if state["paused"]:
        return _blocked("AUTOPILOT_PAUSED")
    lock_path = _private_root(project_root) / "autopilot.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            return _run_locked(project_root, state)
    except Timeout:
        return _blocked("AUTOPILOT_SINGLE_INSTANCE_LOCKED")


def run_if_due(project_root: Path) -> dict[str, Any]:
    state = _load_state(project_root)
    if not state["enabled"] or state["paused"]:
        return _status_payload(state)
    due = state.get("next_cycle_at")
    if due and datetime.fromisoformat(due) > datetime.now(UTC):
        return {**_status_payload(state), "cycle_status": "NOT_DUE"}
    return run_once(project_root)


def _run_locked(project_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    started = datetime.now(UTC)
    limits = _limits()
    day = started.date().isoformat()
    if state.get("budget_date") != day:
        state.update(
            budget_date=day,
            new_strategies_today=0,
            backtests_today=0,
            phase11_12_dna_today=0,
            phase11_12_trials_today=0,
            failures_today=0,
        )
    if int(state["failures_today"]) >= int(
        limits["max_failures_per_cycle"]
    ):
        state["scheduler_status"] = "FAILURE_BUDGET_EXHAUSTED"
        _save_state(project_root, state)
        return _status_payload(state)

    storage_gb = _research_storage_gb(project_root)
    state["research_storage_gb"] = storage_gb
    if storage_gb > float(limits["max_disk_gb"]):
        state["scheduler_status"] = "STORAGE_BUDGET_EXHAUSTED"
        _save_state(project_root, state)
        return _status_payload(state)

    new_budget = min(
        max(
            0,
            int(limits["max_new_strategies_per_day"])
            - int(state["new_strategies_today"]),
        ),
        int(limits["max_parameter_variants_per_strategy"]),
    )
    remaining_backtests = max(
        0,
        int(limits["max_total_backtests_per_day"])
        - int(state["backtests_today"]),
    )
    if remaining_backtests <= 0:
        state["scheduler_status"] = "DAILY_BUDGET_EXHAUSTED"
        _save_state(project_root, state)
        return _status_payload(state)

    cycle_number = int(state.get("cycle_count", 0)) + 1
    family = FAMILIES[(cycle_number - 1) % len(FAMILIES)]
    failures: list[dict[str, Any]] = []
    generated: dict[str, Any] = _skipped(
        "DAILY_GENERATION_BUDGET_EXHAUSTED"
        if new_budget <= 0
        else "LEGACY_CAMPAIGN_NOT_STARTED"
    )
    campaign: dict[str, Any] = _skipped("LEGACY_CAMPAIGN_NOT_STARTED")
    phase11_12_queue: dict[str, Any] = _skipped(
        "PHASE11_12_QUEUE_NOT_STARTED"
    )

    legacy_strategy_budget = min(
        new_budget,
        int(limits["legacy_strategies_per_cycle"]),
        remaining_backtests
        // int(limits["max_legacy_trial_rows_per_strategy"]),
    )
    if legacy_strategy_budget > 0:
        try:
            generated = autopilot_generate(
                budget=new_budget,
                family=family,
                seed=20260726 + cycle_number,
            )
            state["new_strategies_today"] += int(
                generated.get("generated_count", 0)
            )
            campaign = autopilot_campaign(
                family,
                max_trials=legacy_strategy_budget,
            )
            legacy_trial_count = _validated_trial_count(
                campaign,
                remaining_budget=remaining_backtests,
                source="LEGACY_CAMPAIGN",
            )
            state["backtests_today"] += legacy_trial_count
            remaining_backtests -= legacy_trial_count
        except Exception as exc:
            _record_failure(
                state,
                failures,
                cycle_number=cycle_number,
                family=family,
                stage="LEGACY_CAMPAIGN",
                exc=exc,
            )

    elapsed_seconds = (datetime.now(UTC) - started).total_seconds()
    runtime_budget_seconds = (
        int(limits["max_runtime_minutes_per_cycle"]) * 60
    )
    if elapsed_seconds >= runtime_budget_seconds:
        phase11_12_queue = _skipped("CYCLE_RUNTIME_BUDGET_EXHAUSTED")
    else:
        phase11_batch_size = _phase11_12_batch_size(
            remaining_backtests=remaining_backtests,
            limits=limits,
        )
        if phase11_batch_size <= 0:
            phase11_12_queue = _skipped(
                "INSUFFICIENT_BACKTEST_BUDGET_FOR_DNA"
            )
        else:
            try:
                phase11_12_queue = run_phase11_12(
                    project_root,
                    complexity=2,
                    max_strategies=phase11_batch_size,
                    pending_only=True,
                )
                queue_trial_count = _validate_phase11_12_queue(
                    phase11_12_queue,
                    requested_dna=phase11_batch_size,
                    remaining_backtests=remaining_backtests,
                )
                state["backtests_today"] += queue_trial_count
                state["phase11_12_trials_today"] += queue_trial_count
                state["phase11_12_dna_today"] += int(
                    phase11_12_queue.get("catalog_count", 0)
                )
                state["last_phase11_12_queue"] = {
                    "status": phase11_12_queue.get("status"),
                    "catalog_count": phase11_12_queue.get(
                        "catalog_count", 0
                    ),
                    "trial_count": queue_trial_count,
                    "pending_before": phase11_12_queue.get(
                        "pending_before"
                    ),
                    "pending_after": phase11_12_queue.get("pending_after"),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            except Exception as exc:
                _record_failure(
                    state,
                    failures,
                    cycle_number=cycle_number,
                    family=family,
                    stage="PHASE11_12_QUEUE",
                    exc=exc,
                )

    recovery = recover_survivors(project_root)
    completed = datetime.now(UTC)
    queue_completed = phase11_12_queue.get("status") in {
        "GO",
        "PARTIAL",
        "QUEUE_EMPTY",
    }
    if failures:
        cycle_status = "DEGRADED"
    elif campaign.get("status") == "DATA_BLOCKED" and not queue_completed:
        cycle_status = "DATA_BLOCKED"
    else:
        cycle_status = "GO"
    state.update(
        cycle_count=cycle_number,
        last_cycle_started_at=started.isoformat(),
        last_cycle_completed_at=completed.isoformat(),
        last_heartbeat=completed.isoformat(),
        next_cycle_at=(
            completed + timedelta(hours=float(limits["minimum_cycle_interval_hours"]))
        ).isoformat(),
        last_family=family,
        last_cycle_status=cycle_status,
        scheduler_status="SCHEDULE_ENABLED_EXTERNAL_TRIGGER_REQUIRED",
    )
    _save_state(project_root, state)
    payload = {
        "schema": "bounded_research_autopilot_cycle_v1",
        "status": cycle_status,
        "cycle_id": "CYCLE-"
        + stable_hash({"number": cycle_number, "started": started.isoformat()})[:20],
        "family": family,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        "limits": limits,
        "generated": generated,
        "campaign": campaign,
        "phase11_12_queue": phase11_12_queue,
        "survivor_recovery": {
            "status": recovery["status"],
            "survivor_count": recovery["survivor_count"],
            "classification_counts": recovery["classification_counts"],
        },
        "failure_count": len(failures),
        "failures": failures,
        "checkpoint": state,
        "AUTOPILOT_CONTINUOUS_RESEARCH": True,
        "AUTOPILOT_AUTO_LIVE_PROMOTION": False,
        "maximum_automatic_promotion": "FROZEN_SHADOW",
        **_authority(),
    }
    _publish(project_root, "last-cycle.json", payload)
    return payload


def _limits() -> dict[str, int | float]:
    return {
        "max_new_strategies_per_day": int(
            os.environ.get("AUTOPILOT_MAX_NEW_STRATEGIES_PER_DAY", "50")
        ),
        "max_parameter_variants_per_strategy": int(
            os.environ.get("AUTOPILOT_MAX_PARAMETER_VARIANTS_PER_STRATEGY", "25")
        ),
        "legacy_strategies_per_cycle": int(
            os.environ.get("AUTOPILOT_REAL_BACKTESTS_PER_CYCLE", "5")
        ),
        "max_legacy_trial_rows_per_strategy": 9,
        "phase11_12_dna_per_cycle": int(
            os.environ.get("AUTOPILOT_PHASE11_12_DNA_PER_CYCLE", "25")
        ),
        "phase11_12_trial_rows_per_dna": len(COSTS_BPS),
        "max_total_backtests_per_day": int(
            os.environ.get("AUTOPILOT_MAX_TOTAL_BACKTESTS_PER_DAY", "1000")
        ),
        "max_runtime_minutes_per_cycle": int(
            os.environ.get("AUTOPILOT_MAX_RUNTIME_MINUTES_PER_CYCLE", "120")
        ),
        "minimum_cycle_interval_hours": float(
            os.environ.get("AUTOPILOT_MIN_CYCLE_INTERVAL_HOURS", "4")
        ),
        "max_disk_gb": float(os.environ.get("AUTOPILOT_MAX_DISK_GB", "25")),
        "max_failures_per_cycle": int(
            os.environ.get("AUTOPILOT_MAX_FAILURES_PER_CYCLE", "20")
        ),
    }


def _default_state() -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "enabled": True,
        "paused": False,
        "scheduler_status": "SCHEDULE_ENABLED_EXTERNAL_TRIGGER_REQUIRED",
        "cycle_count": 0,
        "last_cycle_started_at": None,
        "last_cycle_completed_at": None,
        "last_heartbeat": None,
        "next_cycle_at": now.isoformat(),
        "budget_date": now.date().isoformat(),
        "new_strategies_today": 0,
        "backtests_today": 0,
        "phase11_12_dna_today": 0,
        "phase11_12_trials_today": 0,
        "failures_today": 0,
        "failures": [],
    }


def _load_state(project_root: Path) -> dict[str, Any]:
    path = _private_root(project_root) / "runtime-state.json"
    if not path.exists():
        return _default_state()
    return {**_default_state(), **json.loads(path.read_text(encoding="utf-8"))}


def _save_state(project_root: Path, state: dict[str, Any]) -> None:
    root = _private_root(project_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "runtime-state.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    _publish(project_root, "runtime-status.json", _status_payload(state))


def _status_payload(state: dict[str, Any]) -> dict[str, Any]:
    research = research_status()
    return {
        "schema": "bounded_research_autopilot_status_v1",
        "status": "GO",
        "enabled": state["enabled"],
        "paused": state["paused"],
        "scheduler_status": state["scheduler_status"],
        "limits": _limits(),
        "last_heartbeat": state.get("last_heartbeat"),
        "last_cycle": state.get("last_cycle_completed_at"),
        "next_cycle": state.get("next_cycle_at"),
        "cycle_count": state.get("cycle_count", 0),
        "new_strategies_today": state.get("new_strategies_today", 0),
        "backtests_today": state.get("backtests_today", 0),
        "phase11_12_dna_today": state.get("phase11_12_dna_today", 0),
        "phase11_12_trials_today": state.get(
            "phase11_12_trials_today", 0
        ),
        "last_phase11_12_queue": state.get("last_phase11_12_queue"),
        "research_storage_gb": state.get("research_storage_gb"),
        "failures_today": state.get("failures_today", 0),
        "research_ledger_counts": research.get("ledger_counts", {}),
        "AUTOPILOT_CONTINUOUS_RESEARCH": True,
        "AUTOPILOT_AUTO_LIVE_PROMOTION": False,
        "external_scheduler_required": True,
        **_authority(),
    }


def _private_root(project_root: Path) -> Path:
    return project_root / "data" / "research" / "autopilot" / "private"


def _publish(project_root: Path, name: str, payload: dict[str, Any]) -> None:
    root = project_root / "output" / "research" / "autopilot"
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def _phase11_12_batch_size(
    *,
    remaining_backtests: int,
    limits: Mapping[str, int | float],
) -> int:
    trials_per_dna = int(limits["phase11_12_trial_rows_per_dna"])
    if trials_per_dna <= 0:
        raise ValueError("PHASE11_12_TRIAL_ROWS_PER_DNA_INVALID")
    return min(
        int(limits["phase11_12_dna_per_cycle"]),
        max(0, remaining_backtests) // trials_per_dna,
    )


def _validated_trial_count(
    payload: Mapping[str, Any],
    *,
    remaining_budget: int,
    source: str,
) -> int:
    trial_count = int(payload.get("trial_count", 0))
    if trial_count < 0 or trial_count > remaining_budget:
        raise RuntimeError(
            f"{source}_TRIAL_BUDGET_VIOLATION:"
            f"{trial_count}>{remaining_budget}"
        )
    return trial_count


def _validate_phase11_12_queue(
    payload: Mapping[str, Any],
    *,
    requested_dna: int,
    remaining_backtests: int,
) -> int:
    status = str(payload.get("status", "")).upper()
    if status not in {"GO", "PARTIAL", "QUEUE_EMPTY"}:
        raise RuntimeError(f"PHASE11_12_QUEUE_STATUS_INVALID:{status}")
    if str(payload.get("EXECUTION_AUTHORITY", "NONE")).upper() != "NONE":
        raise RuntimeError("PHASE11_12_EXECUTION_AUTHORITY_VIOLATION")
    if int(payload.get("BROKER_CALLS", 0)) != 0:
        raise RuntimeError("PHASE11_12_BROKER_CALL_VIOLATION")
    if int(payload.get("ORDER_CALLS", 0)) != 0:
        raise RuntimeError("PHASE11_12_ORDER_CALL_VIOLATION")

    catalog_count = int(payload.get("catalog_count", 0))
    trial_count = _validated_trial_count(
        payload,
        remaining_budget=remaining_backtests,
        source="PHASE11_12_QUEUE",
    )
    expected_trial_count = int(payload.get("expected_trial_count", 0))
    if status == "QUEUE_EMPTY":
        if catalog_count or trial_count or expected_trial_count:
            raise RuntimeError("PHASE11_12_QUEUE_EMPTY_COUNT_VIOLATION")
        return 0
    if catalog_count < 1 or catalog_count > requested_dna:
        raise RuntimeError("PHASE11_12_CATALOG_COUNT_VIOLATION")
    if trial_count != expected_trial_count:
        raise RuntimeError("PHASE11_12_INCOMPLETE_TRIAL_ACCOUNTING")
    if trial_count != catalog_count * len(COSTS_BPS):
        raise RuntimeError("PHASE11_12_COST_STRESS_ACCOUNTING_VIOLATION")
    pending_before = int(payload.get("pending_before", 0))
    pending_after = int(payload.get("pending_after", 0))
    if pending_before - pending_after != catalog_count:
        raise RuntimeError("PHASE11_12_QUEUE_ADVANCE_VIOLATION")
    return trial_count


def _record_failure(
    state: dict[str, Any],
    failures: list[dict[str, Any]],
    *,
    cycle_number: int,
    family: str,
    stage: str,
    exc: Exception,
) -> None:
    failure = {
        "cycle": cycle_number,
        "family": family,
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc)[:500],
        "occurred_at": datetime.now(UTC).isoformat(),
    }
    failures.append(failure)
    state.setdefault("failures", []).append(failure)
    state["failures_today"] += 1


def _research_storage_gb(project_root: Path) -> float:
    roots = (
        project_root / "data" / "research" / "autopilot",
        project_root / "output" / "research" / "autopilot",
        project_root / "output" / "research" / "phase11_12",
    )
    total_bytes = sum(
        path.stat().st_size
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )
    return round(total_bytes / (1024**3), 6)


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "status": "SKIPPED",
        "reason": reason,
        "trial_count": 0,
        **_authority(),
    }


def _authority() -> dict[str, Any]:
    return {
        "strategy_authority": "NONE",
        "signal_authority": "NONE",
        "execution_authority": "NONE",
        "paper_strategy_authority": "NONE",
        "live_strategy_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {"status": "BLOCKED", "reason": reason, **_authority()}
