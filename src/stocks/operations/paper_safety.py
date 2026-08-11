from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def paper_kill_switch_status(project_root: Path) -> dict[str, Any]:
    path = _state_path(project_root)
    if not path.exists():
        return {
            "schema": "bounded_paper_kill_switch_v1",
            "status": "GO",
            "active": False,
            "reason": None,
            "updated_at": None,
            "execution_authority": "NONE",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema": "bounded_paper_kill_switch_v1",
            "status": "NO_GO",
            "active": True,
            "reason": "KILL_SWITCH_STATE_UNREADABLE",
            "updated_at": None,
            "execution_authority": "NONE",
        }
    return payload if isinstance(payload, dict) else {
        "schema": "bounded_paper_kill_switch_v1",
        "status": "NO_GO",
        "active": True,
        "reason": "KILL_SWITCH_STATE_INVALID",
        "updated_at": None,
        "execution_authority": "NONE",
    }


def set_paper_kill_switch(
    project_root: Path,
    *,
    active: bool,
    reason: str,
) -> dict[str, Any]:
    payload = {
        "schema": "bounded_paper_kill_switch_v1",
        "status": "GO",
        "active": bool(active),
        "reason": reason,
        "updated_at": datetime.now(UTC).isoformat(),
        "execution_authority": "NONE",
        "paper_place_order_calls": 0,
        "live_place_order_calls": 0,
    }
    _write_json(_state_path(project_root), payload)
    return payload


def paper_kill_switch_drill(project_root: Path) -> dict[str, Any]:
    armed = set_paper_kill_switch(
        project_root,
        active=True,
        reason="BOUNDED_PAPER_KILL_SWITCH_DRILL",
    )
    observed_armed = paper_kill_switch_status(project_root)
    cleared = set_paper_kill_switch(
        project_root,
        active=False,
        reason="BOUNDED_PAPER_KILL_SWITCH_DRILL_COMPLETE",
    )
    observed_clear = paper_kill_switch_status(project_root)
    go = (
        armed.get("active") is True
        and observed_armed.get("active") is True
        and cleared.get("active") is False
        and observed_clear.get("active") is False
    )
    payload = {
        "schema": "bounded_paper_kill_switch_drill_v1",
        "status": "GO" if go else "NO_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "armed_observed": observed_armed.get("active"),
        "clear_observed": not bool(observed_clear.get("active")),
        "final_kill_switch_active": bool(observed_clear.get("active")),
        "execution_authority": "NONE",
        "paper_place_order_calls": 0,
        "live_place_order_calls": 0,
    }
    _write_json(
        project_root
        / "output"
        / "operations"
        / "paper-kill-switch-drill.json",
        payload,
    )
    return payload


def _state_path(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "operations"
        / "private"
        / "paper-kill-switch.json"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
