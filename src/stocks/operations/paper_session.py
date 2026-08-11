from __future__ import annotations

import json
import socket
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo


NEW_YORK = ZoneInfo("America/New_York")
SAFE_RECONCILIATION = {
    "PAPER_RECONCILED",
    "PAPER_RECONCILED_EMPTY",
    "PAPER_RECONCILED_OPEN_LONG",
}
FORBIDDEN_BLOCKER_FRAGMENTS = (
    "STALE",
    "CONTRACT",
    "DUPLICATE",
    "RECONCILIATION",
    "LEDGER",
    "CAP_",
    "LIMIT_EXCEEDED",
)


def audit_paper_session(
    project_root: Path,
    *,
    session_date: str | None = None,
    cycles: Iterable[dict[str, Any]] | None = None,
    paper_tws_reachable: bool | None = None,
) -> dict[str, Any]:
    rows = list(cycles) if cycles is not None else _read_cycles(project_root)
    target = (
        date.fromisoformat(session_date)
        if session_date
        else _latest_session_date(rows)
    )
    selected = [
        row
        for row in rows
        if target is not None and _local_date(row) == target
    ]
    selected.sort(key=lambda row: str(row.get("started_at", "")))
    cycle_ids = [str(row.get("cycle_id", "")) for row in selected]
    duplicate_cycle_count = sum(
        count - 1 for count in Counter(cycle_ids).values() if count > 1
    )
    times = [
        stamp
        for row in selected
        if (stamp := _local_time(row)) is not None
    ]
    preopen_seen = any(time(8, 0) <= stamp < time(9, 30) for stamp in times)
    open_seen = any(time(9, 30) <= stamp <= time(10, 0) for stamp in times)
    close_seen = any(time(15, 45) <= stamp <= time(16, 30) for stamp in times)
    regular = [
        row
        for row in selected
        if (
            (stamp := _local_time(row)) is not None
            and time(9, 30) <= stamp <= time(16, 0)
        )
    ]
    maximum_gap = _maximum_gap_seconds(regular)
    full_coverage = (
        open_seen
        and close_seen
        and maximum_gap is not None
        and maximum_gap <= 900
    )
    paper_mode_count = sum(
        row.get("effective_mode") == "PAPER_AUTOMATIC"
        for row in selected
    )
    nonpaper_count = len(selected) - paper_mode_count
    failed_count = sum(row.get("status") != "GO" for row in selected)
    forbidden_blockers = sorted(
        {
            str(blocker)
            for row in selected
            for blocker in row.get("blockers", [])
            if any(
                fragment in str(blocker)
                for fragment in FORBIDDEN_BLOCKER_FRAGMENTS
            )
        }
    )
    paper_calls = sum(
        int(row.get("paper_place_order_calls", 0) or 0)
        for row in selected
    )
    live_calls = sum(
        int(row.get("live_place_order_calls", 0) or 0)
        for row in selected
    )
    max_calls_per_cycle = max(
        (
            int(row.get("paper_place_order_calls", 0) or 0)
            for row in selected
        ),
        default=0,
    )
    process_ids = {
        int(row["process_id"])
        for row in selected
        if row.get("process_id") is not None
    }
    restart_proven = len(process_ids) >= 2
    final_reconciliation = (
        str(selected[-1].get("reconciliation_status", ""))
        if selected
        else ""
    )
    reconciliation_history = {
        str(row.get("reconciliation_status", "")) for row in selected
    }
    open_position_seen = "PAPER_RECONCILED_OPEN_LONG" in reconciliation_history
    exited_position_seen = (
        open_position_seen
        and final_reconciliation == "PAPER_RECONCILED_EMPTY"
    )
    if paper_tws_reachable is None:
        paper_tws_reachable = _local_port_reachable("127.0.0.1", 7497)
    kill_switch = _read_json(
        project_root
        / "output"
        / "operations"
        / "paper-kill-switch-drill.json"
    )
    kill_switch_go = kill_switch.get("status") == "GO"
    blockers = []
    if target is None or not selected:
        blockers.append("NO_PAPER_SESSION_CYCLES")
    if not preopen_seen:
        blockers.append("PREOPEN_VALIDATION_MISSING")
    if not full_coverage:
        blockers.append("REGULAR_SESSION_COVERAGE_INCOMPLETE")
    if nonpaper_count:
        blockers.append("NON_PAPER_MODE_CYCLE_IN_SESSION")
    if failed_count:
        blockers.append("FAILED_OR_DEGRADED_CYCLE")
    if duplicate_cycle_count:
        blockers.append("DUPLICATE_CYCLE_ID")
    if forbidden_blockers:
        blockers.append("FORBIDDEN_SESSION_INCIDENT")
    if max_calls_per_cycle > 1:
        blockers.append("PAPER_ORDER_PER_CYCLE_CAP_BREACH")
    if live_calls:
        blockers.append("LIVE_CALL_DURING_PAPER_SESSION")
    if final_reconciliation not in SAFE_RECONCILIATION:
        blockers.append("END_OF_SESSION_RECONCILIATION_BLOCKED")
    if not restart_proven:
        blockers.append("RUNTIME_RESTART_NOT_PROVEN")
    if not kill_switch_go:
        blockers.append("PAPER_KILL_SWITCH_DRILL_NOT_PROVEN")
    status = "GO" if not blockers else "NO_GO"
    session_substatus = _paper_session_substatus(
        complete=status == "GO",
        paper_tws_reachable=paper_tws_reachable,
        paper_mode_cycle_count=paper_mode_count,
        paper_place_order_calls=paper_calls,
        open_position_seen=open_position_seen,
        exited_position_seen=exited_position_seen,
        full_regular_session_coverage=full_coverage,
        runtime_restart_proven=restart_proven,
    )
    payload = {
        "schema": "bounded_complete_paper_session_audit_v1",
        "status": status,
        "marker": (
            "ONE_COMPLETE_PAPER_SESSION_GO"
            if status == "GO"
            else "ONE_COMPLETE_PAPER_SESSION_BLOCKED"
        ),
        "generated_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "session_date": None if target is None else target.isoformat(),
        "cycle_count": len(selected),
        "regular_session_cycle_count": len(regular),
        "paper_mode_cycle_count": paper_mode_count,
        "nonpaper_cycle_count": nonpaper_count,
        "preopen_validation_seen": preopen_seen,
        "market_open_seen": open_seen,
        "market_close_seen": close_seen,
        "maximum_regular_cycle_gap_seconds": maximum_gap,
        "full_regular_session_coverage": full_coverage,
        "duplicate_cycle_count": duplicate_cycle_count,
        "failed_cycle_count": failed_count,
        "forbidden_session_incidents": forbidden_blockers,
        "runtime_process_count": len(process_ids),
        "runtime_restart_proven": restart_proven,
        "paper_kill_switch_drill_go": kill_switch_go,
        "final_reconciliation_status": final_reconciliation,
        "paper_place_order_calls": paper_calls,
        "maximum_place_order_calls_per_cycle": max_calls_per_cycle,
        "live_place_order_calls": live_calls,
        "paper_tws_host": "127.0.0.1",
        "paper_tws_port": 7497,
        "paper_tws_reachable": paper_tws_reachable,
        "paper_session_substatus": session_substatus,
        "open_position_seen": open_position_seen,
        "exited_position_seen": exited_position_seen,
        "blockers": blockers,
        "execution_authority": "NONE",
        "live_authority": "NONE",
    }
    _write_json(
        project_root
        / "output"
        / "operations"
        / "paper-session-audit.json",
        payload,
    )
    return payload


def _paper_session_substatus(
    *,
    complete: bool,
    paper_tws_reachable: bool,
    paper_mode_cycle_count: int,
    paper_place_order_calls: int,
    open_position_seen: bool,
    exited_position_seen: bool,
    full_regular_session_coverage: bool,
    runtime_restart_proven: bool,
) -> str:
    if complete:
        return "PAPER_SESSION_COMPLETE"
    if not paper_tws_reachable and paper_mode_cycle_count == 0:
        return "PAPER_SESSION_NOT_STARTED_NO_TWS"
    if paper_mode_cycle_count == 0 or paper_place_order_calls == 0:
        return "PAPER_SESSION_WAITING_FOR_SETUP"
    if open_position_seen and not exited_position_seen:
        return "PAPER_SESSION_POSITION_OPEN"
    if exited_position_seen:
        if full_regular_session_coverage and not runtime_restart_proven:
            return "PAPER_SESSION_WAITING_RESTART_PROOF"
        return "PAPER_SESSION_EXITED"
    return "PAPER_SESSION_ACTIVE"


def _local_port_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


def _read_cycles(project_root: Path) -> list[dict[str, Any]]:
    path = (
        project_root
        / "data"
        / "operations"
        / "private"
        / "cycles.jsonl"
    )
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _latest_session_date(rows: list[dict[str, Any]]) -> date | None:
    dates = [_local_date(row) for row in rows]
    valid = [item for item in dates if item is not None]
    return max(valid) if valid else None


def _local_datetime(row: dict[str, Any]) -> datetime | None:
    value = row.get("started_at")
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(NEW_YORK)


def _local_date(row: dict[str, Any]) -> date | None:
    stamp = _local_datetime(row)
    return None if stamp is None else stamp.date()


def _local_time(row: dict[str, Any]) -> time | None:
    stamp = _local_datetime(row)
    return None if stamp is None else stamp.time().replace(tzinfo=None)


def _maximum_gap_seconds(rows: list[dict[str, Any]]) -> float | None:
    stamps = [
        stamp
        for row in rows
        if (stamp := _local_datetime(row)) is not None
    ]
    if len(stamps) < 2:
        return None
    return max(
        (right - left).total_seconds()
        for left, right in zip(stamps, stamps[1:], strict=False)
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
