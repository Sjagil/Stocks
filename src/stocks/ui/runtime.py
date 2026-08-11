from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil


RUNTIME_PATH = Path("data/ui/private/runtime.json")
OUTPUT_ROOT = Path("output/ui")
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def ui_command(
    project_root: Path,
    command: str,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> dict[str, Any]:
    if command == "serve":
        return _serve(project_root, host=host, port=port)
    if command == "start":
        return _start(project_root, host=host, port=port)
    if command == "stop":
        return _stop(project_root)
    if command == "status":
        return _status(project_root)
    return _blocked("UNKNOWN_UI_COMMAND")


def _start(project_root: Path, *, host: str, port: int) -> dict[str, Any]:
    invalid = _validate_bind(host, port)
    if invalid:
        return invalid
    current = _status(project_root)
    if current.get("runtime_status") == "RUNNING":
        return current
    output = project_root / OUTPUT_ROOT
    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / "server.stdout.log"
    stderr_path = output / "server.stderr.log"
    command = [
        sys.executable,
        str(project_root / "main.py"),
        "ui",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    with (
        stdout_path.open("ab", buffering=0) as stdout,
        stderr_path.open("ab", buffering=0) as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creationflags,
            close_fds=True,
        )
    _write_runtime(
        project_root,
        {
            "schema": "stocks_ui_runtime_v1",
            "process_id": process.pid,
            "host": host,
            "port": port,
            "url": f"http://{host}:{port}",
            "started_at": datetime.now(UTC).isoformat(),
            "read_only": True,
            "execution_authority": "NONE",
        },
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        health = _http_health(host, port)
        if health:
            return {
                **_status(project_root),
                "status": "GO",
                "startup_status": "STARTED",
            }
        time.sleep(0.25)
    return {
        **_status(project_root),
        "status": "NO_GO",
        "startup_status": "HEALTHCHECK_FAILED",
    }


def _serve(project_root: Path, *, host: str, port: int) -> dict[str, Any]:
    invalid = _validate_bind(host, port)
    if invalid:
        return invalid
    import uvicorn

    from stocks.ui.app import create_app

    uvicorn.run(
        create_app(project_root),
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    return {
        "schema": "stocks_ui_runtime_v1",
        "status": "GO",
        "runtime_status": "STOPPED",
        "read_only": True,
        "execution_authority": "NONE",
    }


def _status(project_root: Path) -> dict[str, Any]:
    runtime = _read_runtime(project_root)
    host = str(runtime.get("host", "127.0.0.1"))
    port = int(runtime.get("port", 8080))
    pid = _owned_server_pid(project_root, runtime)
    running = bool(pid)
    health = _http_health(host, port) if running else None
    if running and health and pid != int(runtime.get("process_id", 0) or 0):
        _write_runtime(project_root, {**runtime, "process_id": pid})
    return {
        "schema": "stocks_ui_runtime_v1",
        "status": "GO",
        "runtime_status": (
            "RUNNING" if running and health else "STOPPED"
        ),
        "process_id": pid if running else None,
        "host": host,
        "port": port,
        "url": f"http://{host}:{port}",
        "health": health or "UNAVAILABLE",
        "read_only": True,
        "external_binding": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _stop(project_root: Path) -> dict[str, Any]:
    runtime = _read_runtime(project_root)
    pid = _owned_server_pid(project_root, runtime)
    if pid:
        process = psutil.Process(pid)
        descendants = process.children(recursive=True)
        for child in descendants:
            child.terminate()
        process.terminate()
        _, alive = psutil.wait_procs([*descendants, process], timeout=10)
        for remaining in alive:
            remaining.kill()
        psutil.wait_procs(alive, timeout=5)
    path = project_root / RUNTIME_PATH
    if path.exists():
        path.unlink()
    return {
        "schema": "stocks_ui_runtime_v1",
        "status": "GO",
        "runtime_status": "STOPPED",
        "read_only": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _validate_bind(host: str, port: int) -> dict[str, Any] | None:
    if host not in LOCAL_HOSTS:
        return _blocked("EXTERNAL_BINDING_NOT_AUTHORIZED")
    if not 1024 <= port <= 65535:
        return _blocked("PORT_OUT_OF_RANGE")
    return None


def _owned_server_pid(
    project_root: Path,
    runtime: dict[str, Any],
) -> int | None:
    expected_main = str((project_root / "main.py").resolve()).casefold()
    port = int(runtime.get("port", 8080))
    candidate_pids: list[int] = []
    try:
        for connection in psutil.net_connections(kind="tcp"):
            local_address = connection.laddr
            if (
                connection.status == psutil.CONN_LISTEN
                and local_address
                and int(local_address.port) == port
                and connection.pid
            ):
                candidate_pids.append(int(connection.pid))
    except (psutil.AccessDenied, psutil.Error, OSError):
        pass
    stored_pid = int(runtime.get("process_id", 0) or 0)
    if stored_pid:
        candidate_pids.append(stored_pid)
    for pid in dict.fromkeys(candidate_pids):
        try:
            command = [part.casefold() for part in psutil.Process(pid).cmdline()]
        except (psutil.Error, OSError):
            continue
        if (
            any(expected_main == str(Path(part).resolve()).casefold() for part in command)
            and "ui" in command
            and "serve" in command
            and "--port" in command
            and str(port) in command
        ):
            return pid
    return None


def _http_health(host: str, port: int) -> dict[str, Any] | None:
    host = "127.0.0.1" if host in {"localhost", "::1"} else host
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/health", timeout=1
        ) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read())
            return {
                "status": payload.get("status", "UNKNOWN"),
                "ui_read_only": payload.get("ui_read_only", False),
            }
    except (
        OSError,
        TimeoutError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        return None


def _write_runtime(project_root: Path, payload: dict[str, Any]) -> None:
    path = project_root / RUNTIME_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_runtime(project_root: Path) -> dict[str, Any]:
    path = project_root / RUNTIME_PATH
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "stocks_ui_runtime_v1",
        "status": "NO_GO",
        "runtime_status": "STOPPED",
        "blockers": [reason],
        "read_only": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
