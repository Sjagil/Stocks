from __future__ import annotations

from pathlib import Path

import stocks.operations.launcher as launcher


def test_launch_live_preserves_activation_blockers(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        launcher,
        "execution_command",
        lambda *_args, **_kwargs: {
            "status": "NO_GO",
            "blockers": ["LIVE_TWS_SOCKET_UNREACHABLE"],
            "execution_authority": "NONE",
        },
    )

    report = launcher.launch_command(
        tmp_path,
        "live",
        approval="operator text",
    )

    assert report["status"] == "NO_GO"
    assert "LIVE_TWS_SOCKET_UNREACHABLE" in report["blockers"]
    assert report["runtime_started"] is False
    assert report["execution_authority"] == "NONE"
    assert report["real_live_order_placed"] is False


def test_launch_live_starts_only_after_level_one_transition(
    tmp_path: Path, monkeypatch
) -> None:
    started = []
    monkeypatch.setattr(
        launcher,
        "execution_command",
        lambda *_args, **_kwargs: {
            "status": "GO",
            "blockers": [],
            "execution_authority": "LIVE_LEVEL_ONE",
            "real_live_order_placed": False,
        },
    )
    monkeypatch.setattr(
        launcher,
        "_start_runtime_task",
        lambda *_args, **kwargs: (
            started.append(kwargs["mode"])
            or {
                "status": "GO",
                "runtime_started": True,
                "task_mode": kwargs["mode"],
            }
        ),
    )

    report = launcher.launch_command(
        tmp_path,
        "live",
        approval="exact operator phrase",
        confirmed=True,
        continuous=True,
    )

    assert report["status"] == "GO"
    assert report["launch_status"] == "LIVE_ACTIVE"
    assert report["runtime_started"] is True
    assert report["execution_authority"] == "LIVE_LEVEL_ONE"
    assert started == ["CONTROLLED_LIVE"]


def test_launch_live_without_explicit_yes_never_starts_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    started = []

    def activation(*_args, **kwargs):
        assert kwargs["confirmed"] is False
        return {
            "status": "NO_GO",
            "blockers": ["EXPLICIT_YES_CONFIRMATION_REQUIRED"],
            "execution_authority": "NONE",
        }

    monkeypatch.setattr(launcher, "execution_command", activation)
    monkeypatch.setattr(
        launcher,
        "_start_runtime_task",
        lambda *_args, **_kwargs: started.append(True),
    )

    report = launcher.launch_command(
        tmp_path,
        "live",
        approval="exact operator phrase",
        continuous=True,
    )

    assert report["status"] == "NO_GO"
    assert report["runtime_started"] is False
    assert report["execution_authority"] == "NONE"
    assert started == []


def test_launch_live_runtime_failure_rolls_back_authority(
    tmp_path: Path, monkeypatch
) -> None:
    commands = []

    def execution(_root, command, **_kwargs):
        commands.append(command)
        if command == "pause-live":
            return {
                "status": "GO",
                "execution_authority": "NONE",
                "transition_status": "LIVE_LEVEL_ONE_PAUSED",
            }
        return {
            "status": "GO",
            "blockers": [],
            "execution_authority": "LIVE_LEVEL_ONE",
        }

    monkeypatch.setattr(launcher, "execution_command", execution)
    monkeypatch.setattr(
        launcher,
        "_start_runtime_task",
        lambda *_args, **_kwargs: {
            "status": "NO_GO",
            "runtime_started": False,
            "blockers": ["RUNTIME_TASK_NOT_RUNNING"],
        },
    )
    monkeypatch.setattr(
        launcher,
        "machine_command",
        lambda *_args, **_kwargs: {"status": "GO"},
    )

    report = launcher.launch_command(
        tmp_path,
        "live",
        approval="exact operator phrase",
        confirmed=True,
        continuous=True,
    )

    assert report["status"] == "NO_GO"
    assert report["execution_authority"] == "NONE"
    assert report["runtime_started"] is False
    assert commands == ["activate-live-canary", "pause-live"]


def test_launch_preflight_preserves_readiness_blockers(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        launcher,
        "system_readiness",
        lambda _root: {"hard_blockers": ["FINANCIAL_GATE_BLOCKED"]},
    )

    report = launcher.launch_command(tmp_path, "preflight")

    assert report["status"] == "NO_GO"
    assert report["blockers"] == ["FINANCIAL_GATE_BLOCKED"]
    assert report["execution_authority"] == "NONE"


def test_launch_status_reports_actual_execution_authority(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        launcher,
        "execution_command",
        lambda *_args, **_kwargs: {
            "status": "GO",
            "execution_authority": "LIVE_LEVEL_ONE",
            "real_live_order_placed": False,
        },
    )
    monkeypatch.setattr(
        launcher,
        "machine_command",
        lambda *_args, **_kwargs: {
            "status": "GO",
            "enabled": True,
        },
    )
    monkeypatch.setattr(
        launcher,
        "system_readiness",
        lambda _root: {"status": "GO"},
    )

    report = launcher.launch_command(tmp_path, "status")

    assert report["schema"] == "stocks_launch_status_v2"
    assert report["execution_authority"] == "LIVE_LEVEL_ONE"
    assert report["execution"]["execution_authority"] == "LIVE_LEVEL_ONE"


def test_launch_stop_pauses_live_authority_before_machine(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    status_reads = iter(["LIVE_LEVEL_ONE", "NONE"])

    def execution(_root, command, **_kwargs):
        calls.append(command)
        if command == "pause-live":
            return {
                "status": "GO",
                "execution_authority": "NONE",
            }
        return {
            "status": "GO",
            "execution_authority": next(status_reads),
        }

    def machine(_root, command, **_kwargs):
        calls.append(f"machine:{command}")
        return {"status": "GO", "enabled": False}

    monkeypatch.setattr(launcher, "execution_command", execution)
    monkeypatch.setattr(launcher, "machine_command", machine)

    report = launcher.launch_command(tmp_path, "stop")

    assert report["status"] == "GO"
    assert report["runtime_status"] == "STOPPED"
    assert report["authority_before_stop"] == "LIVE_LEVEL_ONE"
    assert report["execution_authority"] == "NONE"
    assert calls == ["status", "pause-live", "machine:stop", "status"]


def test_launch_stop_never_claims_go_if_authority_remains(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        launcher,
        "execution_command",
        lambda *_args, **_kwargs: {
            "status": "NO_GO",
            "execution_authority": "LIVE_LEVEL_ONE",
        },
    )
    monkeypatch.setattr(
        launcher,
        "machine_command",
        lambda *_args, **_kwargs: {
            "status": "GO",
            "enabled": False,
        },
    )

    report = launcher.launch_command(tmp_path, "stop")

    assert report["status"] == "NO_GO"
    assert report["runtime_status"] == "STOP_INCOMPLETE"
    assert report["execution_authority"] == "LIVE_LEVEL_ONE"
    assert "EXECUTION_AUTHORITY_NOT_CLEARED" in report["blockers"]


def test_launch_stop_fails_closed_when_restart_task_cannot_be_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        launcher,
        "execution_command",
        lambda *_args, **_kwargs: {
            "status": "GO",
            "execution_authority": "NONE",
        },
    )
    monkeypatch.setattr(
        launcher,
        "machine_command",
        lambda *_args, **_kwargs: {"status": "GO", "enabled": False},
    )
    monkeypatch.setattr(
        launcher,
        "_disable_runtime_task",
        lambda _root: {
            "status": "NO_GO",
            "task_disabled": False,
            "blockers": ["RUNTIME_TASK_DISABLE_FAILED"],
        },
    )

    report = launcher.launch_command(tmp_path, "stop")

    assert report["status"] == "NO_GO"
    assert "RUNTIME_TASK_NOT_DISABLED" in report["blockers"]
