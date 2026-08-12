from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from filelock import FileLock

import stocks.operations.background_jobs as jobs


def test_background_job_executes_and_publishes_redacted_status(
    tmp_path: Path,
) -> None:
    command = (
        sys.executable,
        "-c",
        (
            "import json; print(json.dumps({"
            "'schema':'component_v1','status':'GO'}))"
        ),
    )

    result = jobs._execute_job(
        tmp_path,
        "research",
        ("autopilot", "run-once"),
        timeout_seconds=10,
        command=command,
    )

    assert result["status"] == "COMPLETED"
    assert result["component_status"] == "GO"
    assert result["process_return_code"] == 0
    assert result["money_loop_blocked"] is False
    assert result["resource_priority"] == "BELOW_NORMAL"
    assert result["execution_authority"] == "NONE"
    assert result["broker_writes"] == 0
    public = json.loads(
        (
            tmp_path
            / "output/operations/background-jobs/research.json"
        ).read_text(encoding="utf-8")
    )
    assert public["status"] == "COMPLETED"
    assert "stdout_tail" not in public
    assert "stderr_tail" not in public


def test_background_job_lock_prevents_duplicate_worker(
    tmp_path: Path,
) -> None:
    lock_path = (
        tmp_path
        / "data/operations/private/background-jobs/research.lock"
    )
    lock_path.parent.mkdir(parents=True)
    with FileLock(lock_path, timeout=0):
        result = jobs._execute_job(
            tmp_path,
            "research",
            ("autopilot", "run-once"),
            timeout_seconds=10,
            command=(sys.executable, "-c", "print('{}')"),
        )

    assert result["status"] == "SKIPPED_BUSY"
    assert result["money_loop_blocked"] is False
    assert result["execution_authority"] == "NONE"


def test_background_job_timeout_is_bounded_and_fail_closed(
    tmp_path: Path,
) -> None:
    result = jobs._execute_job(
        tmp_path,
        "research",
        ("autopilot", "run-once"),
        timeout_seconds=1,
        command=(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ),
    )

    assert result["status"] == "TIMED_OUT"
    assert result["blockers"] == ["BACKGROUND_JOB_TIMEOUT"]
    assert result["execution_authority"] == "NONE"
    assert result["broker_writes"] == 0


def test_launcher_is_allowlisted_detached_and_non_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / ".venv-ibkr/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    observed: dict[str, object] = {}

    class Process:
        pid = 43210

    def popen(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return Process()

    monkeypatch.setattr(subprocess, "Popen", popen)

    result = jobs.launch_background_job(
        tmp_path,
        "research",
        ("autopilot", "run-once"),
        timeout_seconds=7200,
    )

    assert result["status"] == "ENQUEUED"
    assert result["worker_pid"] == 43210
    assert result["money_loop_blocked"] is False
    assert result["execution_authority"] == "NONE"
    command = observed["command"]
    assert isinstance(command, list)
    assert "--worker" in command
    assert command[-2:] == ["autopilot", "run-once"]
    kwargs = observed["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    if sys.platform == "win32":
        assert kwargs["creationflags"] & (
            subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )


def test_running_worker_is_not_launched_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs._publish_status(
        tmp_path,
        "research",
        {
            "schema": jobs.SCHEMA,
            "status": "RUNNING",
            "job_name": "research",
            "worker_pid": 12345,
            "started_at": datetime.now(UTC).isoformat(),
            "money_loop_blocked": False,
        },
    )
    monkeypatch.setattr(jobs, "_pid_running", lambda _pid: True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("duplicate worker launch attempted")

    monkeypatch.setattr(subprocess, "Popen", forbidden)

    result = jobs.launch_background_job(
        tmp_path,
        "research",
        ("autopilot", "run-once"),
        timeout_seconds=7200,
    )

    assert result["status"] == "RUNNING"
    assert result["launch_status"] == "SKIPPED_BUSY"
    assert result["worker_pid"] == 12345
    assert result["execution_authority"] == "NONE"


def test_background_job_rejects_arbitrary_commands(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="BACKGROUND_JOB_ARGUMENTS_NOT_ALLOWLISTED",
    ):
        jobs.launch_background_job(
            tmp_path,
            "research",
            ("live", "submit"),
            timeout_seconds=10,
        )


def test_stop_background_jobs_terminates_only_verified_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs._publish_status(
        tmp_path,
        "research",
        {
            "schema": jobs.SCHEMA,
            "status": "RUNNING",
            "job_name": "research",
            "worker_pid": 12345,
        },
    )
    monkeypatch.setattr(
        jobs,
        "_matching_worker_pids",
        lambda _root, name, _current: (
            ([12345], []) if name == "research" else ([], [])
        ),
    )
    monkeypatch.setattr(
        jobs,
        "_worker_identity_matches",
        lambda *_args: True,
    )
    running = {12345}
    monkeypatch.setattr(jobs, "_pid_running", lambda pid: pid in running)

    def terminate(pid: int) -> bool:
        running.discard(pid)
        return True

    monkeypatch.setattr(jobs, "_terminate_process_tree", terminate)

    result = jobs.stop_background_jobs(tmp_path)

    assert result["status"] == "GO"
    assert result["remaining_workers"] == []
    assert result["execution_authority"] == "NONE"
    assert result["broker_writes"] == 0
    stopped = next(
        row for row in result["jobs"] if row["job_name"] == "research"
    )
    assert stopped["status"] == "STOPPED"
    assert stopped["terminated_worker_pids"] == [12345]


def test_stop_background_jobs_fails_closed_on_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        jobs,
        "_matching_worker_pids",
        lambda _root, name, _current: (
            ([54321], []) if name == "research" else ([], [])
        ),
    )
    monkeypatch.setattr(
        jobs,
        "_worker_identity_matches",
        lambda *_args: False,
    )
    monkeypatch.setattr(jobs, "_pid_running", lambda _pid: True)

    def forbidden(_pid: int) -> bool:
        raise AssertionError("unverified process termination attempted")

    monkeypatch.setattr(jobs, "_terminate_process_tree", forbidden)

    result = jobs.stop_background_jobs(tmp_path)

    assert result["status"] == "NO_GO"
    assert result["remaining_workers"] == [54321]
    research = next(
        row for row in result["jobs"] if row["job_name"] == "research"
    )
    assert research["status"] == "STOP_INCOMPLETE"
    assert research["blockers"] == [
        "BACKGROUND_WORKER_IDENTITY_MISMATCH"
    ]


def test_stop_background_jobs_accepts_worker_that_exited_during_tree_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs._publish_status(
        tmp_path,
        "primary_refresh",
        {
            "schema": jobs.SCHEMA,
            "status": "RUNNING",
            "job_name": "primary_refresh",
            "worker_pid": 12345,
        },
    )
    monkeypatch.setattr(
        jobs,
        "_matching_worker_pids",
        lambda _root, name, _current: (
            ([12345, 12346], [])
            if name == "primary_refresh"
            else ([], [])
        ),
    )
    running = {12345, 12346}
    monkeypatch.setattr(jobs, "_pid_running", lambda pid: pid in running)
    monkeypatch.setattr(
        jobs, "_worker_identity_matches", lambda *_args: True
    )

    def terminate(pid: int) -> bool:
        assert pid == 12345
        running.clear()
        return True

    monkeypatch.setattr(jobs, "_terminate_process_tree", terminate)

    result = jobs.stop_background_jobs(tmp_path)

    assert result["status"] == "GO"
    assert result["remaining_workers"] == []
    primary = next(
        row
        for row in result["jobs"]
        if row["job_name"] == "primary_refresh"
    )
    assert primary["status"] == "STOPPED"
    assert primary["terminated_worker_pids"] == [12345]
    assert primary["already_exited_worker_pids"] == [12346]
    assert primary["blockers"] == []


def test_background_jobs_cli_stop_all_reports_machine_readable_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        jobs,
        "stop_background_jobs",
        lambda _root: {
            "schema": "stocks_background_jobs_stop_v1",
            "status": "GO",
        },
    )

    exit_code = jobs.main(
        ["--stop-all", "--project-root", str(tmp_path)]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["status"] == "GO"


def test_stop_background_jobs_preserves_inactive_evidence_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs._publish_status(
        tmp_path,
        "research",
        {
            "schema": jobs.SCHEMA,
            "status": "COMPLETED",
            "job_name": "research",
            "component_status": "GO",
        },
    )
    monkeypatch.setattr(
        jobs,
        "_matching_worker_pids",
        lambda *_args: ([], []),
    )

    result = jobs.stop_background_jobs(tmp_path)

    assert result["status"] == "GO"
    research = next(
        row for row in result["jobs"] if row["job_name"] == "research"
    )
    assert research["status"] == "COMPLETED"
    assert research["stop_status"] == "ALREADY_INACTIVE"
    assert research["component_status"] == "GO"


def test_stop_background_jobs_repairs_stale_incomplete_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs._publish_status(
        tmp_path,
        "primary_refresh",
        {
            "schema": jobs.SCHEMA,
            "status": "STOP_INCOMPLETE",
            "job_name": "primary_refresh",
            "worker_pid": 12345,
            "blockers": ["BACKGROUND_WORKER_STOP_INCOMPLETE"],
        },
    )
    monkeypatch.setattr(
        jobs, "_matching_worker_pids", lambda *_args: ([], [])
    )

    result = jobs.stop_background_jobs(tmp_path)

    assert result["status"] == "GO"
    primary = next(
        row
        for row in result["jobs"]
        if row["job_name"] == "primary_refresh"
    )
    assert primary["status"] == "STOPPED"
    assert primary["stop_status"] == "ALREADY_INACTIVE"
    assert primary["blockers"] == []
