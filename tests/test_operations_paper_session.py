from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from stocks.operations.paper_safety import paper_kill_switch_drill
from stocks.operations.paper_session import audit_paper_session


def test_complete_paper_session_requires_coverage_restart_and_cleanup(
    tmp_path: Path,
) -> None:
    paper_kill_switch_drill(tmp_path)
    cycles = _session_cycles()
    result = audit_paper_session(
        tmp_path,
        session_date="2026-07-28",
        cycles=cycles,
        paper_tws_reachable=True,
    )
    assert result["status"] == "GO"
    assert result["preopen_validation_seen"] is True
    assert result["full_regular_session_coverage"] is True
    assert result["runtime_restart_proven"] is True
    assert result["live_place_order_calls"] == 0
    assert result["paper_session_substatus"] == "PAPER_SESSION_COMPLETE"


def test_session_blocks_stale_incident_duplicate_and_missing_restart(
    tmp_path: Path,
) -> None:
    paper_kill_switch_drill(tmp_path)
    cycles = _session_cycles(process_restart=False)
    cycles.append({**cycles[-1]})
    cycles[2]["blockers"] = ["STALE_DATA_BLOCKED"]
    result = audit_paper_session(
        tmp_path,
        session_date="2026-07-28",
        cycles=cycles,
        paper_tws_reachable=True,
    )
    assert result["status"] == "NO_GO"
    assert "RUNTIME_RESTART_NOT_PROVEN" in result["blockers"]
    assert "DUPLICATE_CYCLE_ID" in result["blockers"]
    assert "FORBIDDEN_SESSION_INCIDENT" in result["blockers"]


def test_session_substatus_distinguishes_no_tws_and_waiting_setup(
    tmp_path: Path,
) -> None:
    no_tws = audit_paper_session(
        tmp_path,
        cycles=[],
        paper_tws_reachable=False,
    )
    waiting = audit_paper_session(
        tmp_path,
        session_date="2026-07-28",
        cycles=_session_cycles(),
        paper_tws_reachable=True,
    )

    assert no_tws["paper_session_substatus"] == "PAPER_SESSION_NOT_STARTED_NO_TWS"
    assert waiting["paper_session_substatus"] == "PAPER_SESSION_WAITING_FOR_SETUP"


def test_session_substatus_tracks_position_exit_and_restart_wait(
    tmp_path: Path,
) -> None:
    cycles = _session_cycles(process_restart=False)
    cycles[20]["paper_place_order_calls"] = 1
    cycles[20]["reconciliation_status"] = "PAPER_RECONCILED_OPEN_LONG"
    cycles[-1]["reconciliation_status"] = "PAPER_RECONCILED_EMPTY"

    result = audit_paper_session(
        tmp_path,
        session_date="2026-07-28",
        cycles=cycles,
        paper_tws_reachable=True,
    )

    assert result["open_position_seen"] is True
    assert result["exited_position_seen"] is True
    assert result["paper_session_substatus"] == "PAPER_SESSION_WAITING_RESTART_PROOF"


def test_kill_switch_drill_finishes_clear(tmp_path: Path) -> None:
    result = paper_kill_switch_drill(tmp_path)
    assert result["status"] == "GO"
    assert result["armed_observed"] is True
    assert result["clear_observed"] is True
    assert result["final_kill_switch_active"] is False


def _session_cycles(
    *,
    process_restart: bool = True,
) -> list[dict]:
    start = datetime(2026, 7, 28, 12, 15, tzinfo=UTC)
    rows = []
    stamp = start
    index = 0
    while stamp <= datetime(2026, 7, 28, 20, 15, tzinfo=UTC):
        rows.append(
            {
                "cycle_id": f"CYCLE-{index}",
                "started_at": stamp.isoformat(),
                "completed_at": (stamp + timedelta(seconds=20)).isoformat(),
                "status": "GO",
                "effective_mode": "PAPER_AUTOMATIC",
                "process_id": (
                    101
                    if not process_restart or stamp.hour < 17
                    else 202
                ),
                "blockers": [],
                "paper_place_order_calls": 0,
                "live_place_order_calls": 0,
                "reconciliation_status": "PAPER_RECONCILED_EMPTY",
            }
        )
        stamp += timedelta(minutes=10)
        index += 1
    return rows
