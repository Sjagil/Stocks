from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import psutil
from filelock import FileLock, Timeout


SCHEMA = "stocks_background_job_v1"
ALLOWED_JOBS: dict[str, tuple[str, ...]] = {
    "primary_refresh": ("primary-refresh",),
    "research": ("autopilot", "run-once"),
}
ACTIVE_STATUSES = frozenset({"ENQUEUED", "RUNNING"})
RETRY_BACKOFF = timedelta(minutes=15)


def launch_background_job(
    project_root: Path,
    job_name: str,
    arguments: Sequence[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Launch one allowlisted job without blocking the money loop."""
    normalized = _validated_job(job_name, arguments)
    current = background_job_status(project_root, job_name)
    status = str(current.get("status", "NOT_RUN")).upper()
    worker_pid = int(current.get("worker_pid") or 0)
    if status in ACTIVE_STATUSES and worker_pid and _pid_running(worker_pid):
        return {
            **current,
            "status": "RUNNING",
            "launch_status": "SKIPPED_BUSY",
            **_authority(),
        }
    if status in {"FAILED", "TIMED_OUT", "ABANDONED"} and not _retry_due(
        current
    ):
        return {
            **current,
            "status": "RETRY_BACKOFF",
            "launch_status": "SKIPPED_BACKOFF",
            **_authority(),
        }

    python = project_root / ".venv-ibkr" / "Scripts" / "python.exe"
    worker = Path(__file__).resolve()
    if not python.is_file():
        return _blocked("PYTHON_RUNTIME_NOT_FOUND", job_name)
    command = [
        str(python),
        str(worker),
        "--worker",
        "--project-root",
        str(project_root.resolve()),
        "--job",
        job_name,
        "--timeout-seconds",
        str(max(1, int(timeout_seconds))),
        "--",
        *normalized,
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
            | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        )
    process = subprocess.Popen(
        command,
        cwd=project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=creationflags,
    )
    return {
        "schema": SCHEMA,
        "status": "ENQUEUED",
        "launch_status": "LAUNCHED",
        "job_name": job_name,
        "worker_pid": process.pid,
        "arguments": list(normalized),
        "timeout_seconds": max(1, int(timeout_seconds)),
        "enqueued_at": _now(),
        "money_loop_blocked": False,
        "resource_priority": "BELOW_NORMAL",
        **_authority(),
    }


def background_job_status(project_root: Path, job_name: str) -> dict[str, Any]:
    _validated_name(job_name)
    payload = _read_json(_private_status_path(project_root, job_name))
    if not payload:
        return {
            "schema": SCHEMA,
            "status": "NOT_RUN",
            "job_name": job_name,
            **_authority(),
        }
    status = str(payload.get("status", "UNKNOWN")).upper()
    worker_pid = int(payload.get("worker_pid") or 0)
    if status in ACTIVE_STATUSES and worker_pid and not _pid_running(worker_pid):
        abandoned = {
            **payload,
            "status": "ABANDONED",
            "completed_at": _now(),
            "blockers": ["BACKGROUND_WORKER_PROCESS_MISSING"],
            **_authority(),
        }
        _publish_status(project_root, job_name, abandoned)
        return abandoned
    return {**payload, **_authority()}


def run_background_worker(
    project_root: Path,
    job_name: str,
    arguments: Sequence[str],
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    normalized = _validated_job(job_name, arguments)
    python = project_root / ".venv-ibkr" / "Scripts" / "python.exe"
    main = project_root / "main.py"
    return _execute_job(
        project_root,
        job_name,
        normalized,
        timeout_seconds=max(1, int(timeout_seconds)),
        command=(str(python), str(main), *normalized),
    )


def stop_background_jobs(
    project_root: Path,
    *,
    reason: str = "MACHINE_STOP_REQUESTED",
) -> dict[str, Any]:
    """Stop only verified detached workers owned by this project."""
    project_root = project_root.resolve()
    results: list[dict[str, Any]] = []
    blockers: list[str] = []
    for job_name in ALLOWED_JOBS:
        current = _read_json(_private_status_path(project_root, job_name))
        worker_pids, discovery_blockers = _matching_worker_pids(
            project_root,
            job_name,
            current,
        )
        job_blockers = list(discovery_blockers)
        terminated_pids: list[int] = []
        already_exited_pids: list[int] = []
        remaining_pids: list[int] = []
        for pid in worker_pids:
            if not _pid_running(pid):
                already_exited_pids.append(pid)
                continue
            if not _worker_identity_matches(pid, project_root, job_name):
                if _pid_running(pid):
                    job_blockers.append(
                        "BACKGROUND_WORKER_IDENTITY_MISMATCH"
                    )
                    remaining_pids.append(pid)
                else:
                    already_exited_pids.append(pid)
                continue
            terminated = _terminate_process_tree(pid)
            if not _pid_running(pid):
                if terminated:
                    terminated_pids.append(pid)
                else:
                    already_exited_pids.append(pid)
            else:
                job_blockers.append("BACKGROUND_WORKER_STOP_INCOMPLETE")
                remaining_pids.append(pid)

        previously_active = bool(worker_pids)
        if job_blockers:
            status = "STOP_INCOMPLETE"
            stop_status = "STOP_INCOMPLETE"
        elif previously_active:
            status = "STOPPED"
            stop_status = "STOPPED"
        else:
            prior_status = str(
                current.get("status") or "NOT_RUN"
            ).upper()
            status = (
                "STOPPED"
                if prior_status
                in {"ENQUEUED", "RUNNING", "STOP_INCOMPLETE"}
                else prior_status
            )
            stop_status = "ALREADY_INACTIVE"
        result = {
            **current,
            "schema": SCHEMA,
            "status": status,
            "job_name": job_name,
            "stop_reason": reason,
            "stop_status": stop_status,
            "stopped_at": _now(),
            "worker_pids_found": worker_pids,
            "terminated_worker_pids": terminated_pids,
            "already_exited_worker_pids": already_exited_pids,
            "remaining_worker_pids": remaining_pids,
            "process_tree_terminated": bool(terminated_pids),
            "blockers": sorted(set(job_blockers)),
            **_authority(),
        }
        _publish_status(project_root, job_name, result)
        results.append(result)
        blockers.extend(result["blockers"])

    remaining = sorted(
        {
            int(pid)
            for result in results
            for pid in result["remaining_worker_pids"]
        }
    )
    return {
        "schema": "stocks_background_jobs_stop_v1",
        "status": "GO" if not blockers and not remaining else "NO_GO",
        "stop_reason": reason,
        "jobs": results,
        "remaining_workers": remaining,
        "blockers": sorted(set(blockers)),
        **_authority(),
    }


def _execute_job(
    project_root: Path,
    job_name: str,
    arguments: Sequence[str],
    *,
    timeout_seconds: int,
    command: Sequence[str],
) -> dict[str, Any]:
    started = datetime.now(UTC)
    lock_path = _private_root(project_root) / f"{job_name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(lock_path, timeout=0):
            process = subprocess.Popen(
                list(command),
                cwd=project_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            running = {
                "schema": SCHEMA,
                "status": "RUNNING",
                "job_name": job_name,
                "worker_pid": os.getpid(),
                "child_pid": process.pid,
                "arguments": list(arguments),
                "timeout_seconds": timeout_seconds,
                "started_at": started.isoformat(),
                "heartbeat_at": _now(),
                "money_loop_blocked": False,
                "resource_priority": "BELOW_NORMAL",
                **_authority(),
            }
            _publish_status(project_root, job_name, running)
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
                completed = {
                    **running,
                    "status": "TIMED_OUT",
                    "completed_at": _now(),
                    "duration_seconds": (
                        datetime.now(UTC) - started
                    ).total_seconds(),
                    "process_tree_terminated": terminated,
                    "stdout_tail": stdout[-500:],
                    "stderr_tail": stderr[-500:],
                    "blockers": ["BACKGROUND_JOB_TIMEOUT"],
                }
                _publish_status(project_root, job_name, completed)
                return completed

            component = _json_payload(stdout)
            return_code = int(process.returncode or 0)
            completed_status = "COMPLETED" if return_code == 0 else "FAILED"
            completed = {
                **running,
                "status": completed_status,
                "completed_at": _now(),
                "duration_seconds": (
                    datetime.now(UTC) - started
                ).total_seconds(),
                "process_return_code": return_code,
                "component_status": _component_status(component),
                "component_schema": component.get("schema"),
                "stdout_tail": stdout[-500:],
                "stderr_tail": stderr[-500:],
                "blockers": (
                    []
                    if return_code == 0
                    else ["BACKGROUND_JOB_NON_ZERO_EXIT"]
                ),
            }
            _publish_status(project_root, job_name, completed)
            return completed
    except Timeout:
        return {
            "schema": SCHEMA,
            "status": "SKIPPED_BUSY",
            "job_name": job_name,
            "money_loop_blocked": False,
            **_authority(),
        }


def _publish_status(
    project_root: Path, job_name: str, payload: dict[str, Any]
) -> None:
    private = {**payload, **_authority()}
    public = {
        key: value
        for key, value in private.items()
        if key not in {"stdout_tail", "stderr_tail"}
    }
    _atomic_json(_private_status_path(project_root, job_name), private)
    _atomic_json(_public_status_path(project_root, job_name), public)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _validated_job(job_name: str, arguments: Sequence[str]) -> tuple[str, ...]:
    _validated_name(job_name)
    normalized = tuple(str(item) for item in arguments)
    if normalized != ALLOWED_JOBS[job_name]:
        raise ValueError(f"BACKGROUND_JOB_ARGUMENTS_NOT_ALLOWLISTED:{job_name}")
    return normalized


def _validated_name(job_name: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", str(job_name)):
        raise ValueError("BACKGROUND_JOB_NAME_INVALID")
    if job_name not in ALLOWED_JOBS:
        raise ValueError(f"BACKGROUND_JOB_NOT_ALLOWLISTED:{job_name}")


def _retry_due(payload: dict[str, Any]) -> bool:
    value = payload.get("completed_at") or payload.get("started_at")
    try:
        completed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return True
    if completed.tzinfo is None:
        completed = completed.replace(tzinfo=UTC)
    return datetime.now(UTC) >= completed + RETRY_BACKOFF


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    process_query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information, False, pid
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(
            handle, ctypes.byref(exit_code)
        ):
            return False
        return int(exit_code.value) == still_active
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def _matching_worker_pids(
    project_root: Path,
    job_name: str,
    current: dict[str, Any],
) -> tuple[list[int], list[str]]:
    pids: set[int] = set()
    blockers: list[str] = []
    recorded_pid = int(current.get("worker_pid") or 0)
    if recorded_pid > 0 and _pid_running(recorded_pid):
        pids.add(recorded_pid)
    try:
        for process in psutil.process_iter(["pid"]):
            pid = int(process.info.get("pid") or process.pid)
            if pid > 0 and _worker_identity_matches(
                pid,
                project_root,
                job_name,
            ):
                pids.add(pid)
    except (psutil.Error, OSError):
        blockers.append("BACKGROUND_WORKER_DISCOVERY_FAILED")
    return sorted(pids), blockers


def _worker_identity_matches(
    pid: int,
    project_root: Path,
    job_name: str,
) -> bool:
    try:
        command = psutil.Process(pid).cmdline()
    except (psutil.Error, OSError):
        return False
    if not command:
        return False
    expected_worker = Path(__file__).resolve()
    expected_root = project_root.resolve()
    try:
        worker_index = next(
            index
            for index, value in enumerate(command)
            if Path(value).resolve() == expected_worker
        )
    except (OSError, StopIteration):
        return False
    trailing = command[worker_index + 1 :]
    try:
        root_index = trailing.index("--project-root")
        job_index = trailing.index("--job")
        supplied_root = Path(trailing[root_index + 1]).resolve()
        supplied_job = trailing[job_index + 1]
    except (IndexError, OSError, ValueError):
        return False
    return (
        "--worker" in trailing
        and supplied_root == expected_root
        and supplied_job == job_name
    )


def _terminate_process_tree(pid: int) -> bool:
    if os.name == "nt":
        try:
            completed = subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    try:
        os.kill(pid, 15)
    except OSError:
        return False
    return True


def _json_payload(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _component_status(payload: dict[str, Any]) -> str:
    status = payload.get("status", "INVALID_STEP_OUTPUT")
    if isinstance(status, dict):
        status = status.get("status", "INVALID_STEP_OUTPUT")
    return str(status).upper()


def _private_root(project_root: Path) -> Path:
    return project_root / "data" / "operations" / "private" / "background-jobs"


def _private_status_path(project_root: Path, job_name: str) -> Path:
    return _private_root(project_root) / f"{job_name}.json"


def _public_status_path(project_root: Path, job_name: str) -> Path:
    return (
        project_root
        / "output"
        / "operations"
        / "background-jobs"
        / f"{job_name}.json"
    )


def _authority() -> dict[str, Any]:
    return {
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "orders_generated": 0,
        "automatic_execution": False,
    }


def _blocked(reason: str, job_name: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "job_name": job_name,
        "blockers": [reason],
        **_authority(),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--worker", action="store_true")
    action.add_argument("--stop-all", action="store_true")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--job")
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stop_all:
        result = stop_background_jobs(args.project_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("status") == "GO" else 2
    if not args.job or args.timeout_seconds is None:
        raise SystemExit("--worker requires --job and --timeout-seconds")
    arguments = list(args.arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    result = run_background_worker(
        args.project_root,
        args.job,
        arguments,
        timeout_seconds=args.timeout_seconds,
    )
    return 0 if result.get("status") in {"COMPLETED", "SKIPPED_BUSY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_JOBS",
    "background_job_status",
    "launch_background_job",
    "run_background_worker",
    "stop_background_jobs",
]
