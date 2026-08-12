from __future__ import annotations

import json
from pathlib import Path

import pytest

import stocks.operations.primary_refresh as refresh


def test_primary_refresh_is_bounded_redacted_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definitions = (
        refresh.RefreshStep("SAFE", ("p3", "publish"), 1.0, 10),
        refresh.RefreshStep("FAILED", ("p4", "publish"), 1.0, 10),
    )
    monkeypatch.setattr(refresh, "_steps", lambda _root: definitions)

    def command_step(_root, arguments, **_kwargs):
        if arguments == ("p3", "publish"):
            return {
                "status": "GO",
                "process_return_code": 0,
                "stdout_tail": "private-success",
                "stderr_tail": "",
            }
        return {
            "status": "NO_GO",
            "process_return_code": 2,
            "stdout_tail": "private-failure",
            "stderr_tail": "secret-detail",
        }

    monkeypatch.setattr(refresh, "_command_step", command_step)
    result = refresh.run_primary_refresh(tmp_path)

    assert result["status"] == "DEGRADED"
    assert result["failures"] == ["FAILED"]
    assert result["attempted_step_count"] == 2
    assert result["money_loop_blocked"] is False
    assert result["resource_priority"] == "BELOW_NORMAL"
    assert result["execution_authority"] == "NONE"
    assert result["broker_calls"] == 0
    assert result["broker_writes"] == 0
    assert result["orders_generated"] == 0

    public = json.loads(
        (
            tmp_path / "output/operations/primary-refresh.json"
        ).read_text(encoding="utf-8")
    )
    assert "stdout_tail" not in public["steps"]["SAFE"]
    assert "stderr_tail" not in public["steps"]["FAILED"]
    state = json.loads(
        (
            tmp_path
            / "data/operations/private/primary-refresh/state.json"
        ).read_text(encoding="utf-8")
    )
    assert state["steps"]["SAFE"]["last_success_at"]
    assert "last_success_at" not in state["steps"]["FAILED"]


def test_primary_refresh_preserves_failed_state_until_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = refresh.RefreshStep(
        "EVIDENCE", ("p4", "publish"), 1.0, 10
    )
    monkeypatch.setattr(refresh, "_steps", lambda _root: (definition,))
    monkeypatch.setattr(
        refresh,
        "_command_step",
        lambda *_args, **_kwargs: {
            "status": "NO_GO",
            "process_return_code": 2,
            "stdout_tail": "",
            "stderr_tail": "",
        },
    )

    first = refresh.run_primary_refresh(tmp_path)
    second = refresh.run_primary_refresh(tmp_path)

    assert first["status"] == "DEGRADED"
    assert second["status"] == "DEGRADED"
    assert second["attempted_step_count"] == 0
    assert second["steps"]["EVIDENCE"]["status"] == "NOT_DUE"
    assert second["steps"]["EVIDENCE"]["last_status"] == "NO_GO"
    assert second["failures"] == ["EVIDENCE"]


def test_primary_refresh_fixed_steps_have_no_execution_command(
    tmp_path: Path,
) -> None:
    definitions = refresh._steps(tmp_path)
    commands = [definition.arguments for definition in definitions]
    flattened = {token.lower() for command in commands for token in command}

    assert commands
    assert all(definition.timeout_seconds > 0 for definition in definitions)
    assert all(definition.cadence_hours > 0 for definition in definitions)
    assert "live" not in flattened
    assert "submit" not in flattened
    assert "approve" not in flattened
    assert "paper-canary" not in flattened
