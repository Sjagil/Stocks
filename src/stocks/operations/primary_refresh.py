from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from filelock import FileLock, Timeout

from stocks.operations.background_jobs import _terminate_process_tree


SCHEMA = "stocks_primary_background_refresh_v1"


@dataclass(frozen=True)
class RefreshStep:
    name: str
    arguments: tuple[str, ...]
    cadence_hours: float
    timeout_seconds: int


def run_primary_refresh(project_root: Path) -> dict[str, Any]:
    """Refresh non-money-loop context and evidence under one bounded lock."""
    project_root = project_root.resolve()
    lock_path = _private_root(project_root) / "primary-refresh.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            return _run_locked(project_root)
    except Timeout:
        return {
            "schema": SCHEMA,
            "status": "SKIPPED_BUSY",
            "money_loop_blocked": False,
            **_authority(),
        }


def _run_locked(project_root: Path) -> dict[str, Any]:
    started = datetime.now(UTC)
    state = _read_json(_state_path(project_root))
    steps: dict[str, dict[str, Any]] = {}
    attempted = 0
    for definition in _steps(project_root):
        prior = state.get("steps", {}).get(definition.name, {})
        if not _due(prior.get("last_attempt_at"), definition.cadence_hours):
            steps[definition.name] = {
                "status": "NOT_DUE",
                "last_attempt_at": prior.get("last_attempt_at"),
                "last_success_at": prior.get("last_success_at"),
                "last_status": prior.get("last_status"),
                "last_process_return_code": prior.get(
                    "last_process_return_code"
                ),
            }
            continue
        attempted += 1
        result = _command_step(
            project_root,
            definition.arguments,
            timeout_seconds=definition.timeout_seconds,
        )
        attempted_at = _now()
        result["last_attempt_at"] = attempted_at
        step_state = {
            "last_attempt_at": attempted_at,
            "last_status": result["status"],
            "last_process_return_code": result.get(
                "process_return_code"
            ),
        }
        if result.get("process_return_code") == 0:
            step_state["last_success_at"] = attempted_at
        elif prior.get("last_success_at"):
            step_state["last_success_at"] = prior["last_success_at"]
        state.setdefault("steps", {})[definition.name] = step_state
        steps[definition.name] = result
        _write_state(project_root, state)

    completed = datetime.now(UTC)
    failures = [
        name
        for name, result in steps.items()
        if result.get("status") in {"ERROR", "TIMEOUT"}
        or (
            result.get("status") != "NOT_DUE"
            and result.get("process_return_code") != 0
        )
        or (
            result.get("status") == "NOT_DUE"
            and result.get("last_process_return_code") not in {None, 0}
        )
    ]
    report = {
        "schema": SCHEMA,
        "status": "DEGRADED" if failures else "GO",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_seconds": (completed - started).total_seconds(),
        "attempted_step_count": attempted,
        "step_count": len(steps),
        "steps": steps,
        "failures": failures,
        "resource_priority": "BELOW_NORMAL",
        "money_loop_blocked": False,
        **_authority(),
    }
    _atomic_json(_public_path(project_root), _redacted(report))
    return report


def _steps(project_root: Path) -> tuple[RefreshStep, ...]:
    symbols = _latest_symbols(project_root)
    return (
        RefreshStep(
            "MARKET_DATA",
            (
                "data",
                "multitimeframe",
                "collect",
                "--symbols",
                ",".join(symbols),
                "--intervals",
                "1h,2h,4h,1d,1w,1mo",
                "--providers",
                "local,datascraper,yfinance",
                "--lookback-days",
                "30",
            ),
            1.0,
            240,
        ),
        RefreshStep("DAILY", ("daily", "--no-autopilot"), 20.0, 600),
        RefreshStep("HMM_REGIME", ("regimes", "current"), 1.0, 120),
        RefreshStep("DYNAMIC", ("dynamic", "daily"), 1.0, 600),
        RefreshStep(
            "MULTITIMEFRAME_WATCHLIST",
            ("research", "phase11-10", "watchlist"),
            20.0,
            120,
        ),
        RefreshStep(
            "FAST_TRACK_OBSERVATION",
            ("research", "phase11-13", "observe"),
            20.0,
            180,
        ),
        RefreshStep(
            "BROAD_SHADOW_OBSERVATION",
            ("research", "phase11-12", "observe"),
            1.0,
            300,
        ),
        RefreshStep(
            "SURVIVOR_SHADOW_OBSERVATION",
            ("research", "phase11-14", "observe"),
            1.0,
            300,
        ),
        RefreshStep(
            "BROAD_SHADOW_NOTIFICATION",
            ("telegram", "send-shadow-digest"),
            1.0,
            60,
        ),
        RefreshStep("MACRO", ("macro", "update"), 1.0, 240),
        RefreshStep(
            "NEWS_DIGEST",
            ("telegram", "market-digest-preview"),
            1.0,
            120,
        ),
        RefreshStep(
            "MARKET_CONTEXT",
            (
                "market",
                "context",
                "build",
                "--symbols",
                ",".join(symbols[:10]),
                "--max-expirations",
                "4",
            ),
            1.0,
            240,
        ),
        RefreshStep(
            "COT_CONTEXT",
            ("market", "context", "cot-update", "--start", "2018-01-01"),
            24.0,
            180,
        ),
        RefreshStep(
            "ASSET_CONTEXT",
            ("market", "context", "transmission"),
            1.0,
            60,
        ),
        RefreshStep(
            "ACTIVE_SWING_SPRINTS",
            ("research", "active-swing", "refresh"),
            1.0,
            120,
        ),
        RefreshStep(
            "ROLE_LEADERBOARDS",
            ("research", "registry", "roles"),
            24.0,
            120,
        ),
        RefreshStep("P3_EVIDENCE", ("p3", "publish"), 1.0, 300),
        RefreshStep("RL_SHADOW", ("rl", "cycle"), 0.25, 300),
        RefreshStep("P4_EVIDENCE", ("p4", "publish"), 1.0, 300),
        RefreshStep(
            "HMM_NOTIFICATION",
            ("telegram", "send-regime-update"),
            1.0,
            60,
        ),
    )


def _latest_symbols(project_root: Path) -> list[str]:
    cycle = _read_json(project_root / "output/operations/last-cycle.json")
    plan = cycle.get("intraday_refresh_plan")
    plan = plan if isinstance(plan, dict) else {}
    symbols = plan.get("symbols")
    if not isinstance(symbols, list):
        symbols = []
    normalized = [
        str(value).strip().upper() for value in symbols if str(value).strip()
    ]
    return normalized[:50] or ["SPY", "QQQ", "AAPL", "ASML", "GLD"]


def _command_step(
    project_root: Path,
    arguments: Sequence[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    python = project_root / ".venv-ibkr" / "Scripts" / "python.exe"
    command = [str(python), str(project_root / "main.py"), *arguments]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminated = _terminate_process_tree(process.pid)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            terminated = False
        return {
            "status": "TIMEOUT",
            "duration_seconds": round(time.monotonic() - started, 3),
            "timeout_seconds": timeout_seconds,
            "process_tree_terminated": terminated,
            "stdout_tail": stdout[-500:],
            "stderr_tail": stderr[-500:],
            "process_return_code": int(process.returncode or 0),
        }
    payload = _json_payload(stdout)
    return {
        "status": str(payload.get("status") or "INVALID_STEP_OUTPUT").upper(),
        "schema": payload.get("schema"),
        "duration_seconds": round(time.monotonic() - started, 3),
        "process_return_code": int(process.returncode or 0),
        "stdout_tail": stdout[-500:],
        "stderr_tail": stderr[-500:],
    }


def _due(value: Any, hours: float) -> bool:
    try:
        previous = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return True
    if previous.tzinfo is None:
        previous = previous.replace(tzinfo=UTC)
    return (datetime.now(UTC) - previous).total_seconds() >= hours * 3600


def _json_payload(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _redacted(payload: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(payload))
    for result in redacted.get("steps", {}).values():
        if isinstance(result, dict):
            result.pop("stdout_tail", None)
            result.pop("stderr_tail", None)
    return redacted


def _authority() -> dict[str, Any]:
    return {
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "orders_generated": 0,
        "automatic_execution": False,
    }


def _private_root(project_root: Path) -> Path:
    return project_root / "data/operations/private/primary-refresh"


def _state_path(project_root: Path) -> Path:
    return _private_root(project_root) / "state.json"


def _public_path(project_root: Path) -> Path:
    return project_root / "output/operations/primary-refresh.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_state(project_root: Path, state: dict[str, Any]) -> None:
    _atomic_json(_state_path(project_root), state)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["RefreshStep", "run_primary_refresh"]
