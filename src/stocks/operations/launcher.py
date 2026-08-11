from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from stocks.live import (
    AUTONOMOUS_LEVEL_ONE,
    AUTONOMOUS_PROFILE,
    LIVE_LEVEL_ONE,
)
from stocks.operations.service import execution_command, machine_command
from stocks.readiness import system_readiness


def launch_command(
    project_root: Path,
    command: str,
    *,
    approval: str | None = None,
    confirmed: bool = False,
    profile: str = AUTONOMOUS_PROFILE,
    continuous: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if command == "stop":
        before = execution_command(project_root, "status")
        pause = None
        if before.get("execution_authority") in {
            LIVE_LEVEL_ONE,
            AUTONOMOUS_LEVEL_ONE,
        }:
            pause = execution_command(
                project_root,
                "pause-live",
                approval="CANONICAL_RUNTIME_STOP",
                env_file=".env.ibkr.live",
            )
        machine = machine_command(project_root, "stop")
        scheduler = _disable_runtime_task(project_root)
        after = execution_command(project_root, "status")
        authority_cleared = after.get("execution_authority") == "NONE"
        machine_stopped = machine.get("enabled") is False
        scheduler_disabled = scheduler.get("status") in {
            "GO",
            "NOT_INSTALLED",
        }
        stopped = authority_cleared and machine_stopped and scheduler_disabled
        return {
            "schema": "stocks_launch_control_v2",
            "status": "GO" if stopped else "NO_GO",
            "runtime_status": (
                "STOPPED" if stopped else "STOP_INCOMPLETE"
            ),
            "authority_before_stop": before.get(
                "execution_authority", "NONE"
            ),
            "authority_transition": pause,
            "machine": machine,
            "scheduler": scheduler,
            "execution_authority": after.get(
                "execution_authority", "NONE"
            ),
            "blockers": (
                []
                if stopped
                else [
                    reason
                    for reason, passed in (
                        ("EXECUTION_AUTHORITY_NOT_CLEARED", authority_cleared),
                        ("MACHINE_NOT_STOPPED", machine_stopped),
                        ("RUNTIME_TASK_NOT_DISABLED", scheduler_disabled),
                    )
                    if not passed
                ]
            ),
            "real_live_order_placed": False,
        }

    if command == "status":
        execution = execution_command(project_root, "status")
        return {
            "schema": "stocks_launch_status_v2",
            "status": "GO",
            "machine": machine_command(project_root, "status"),
            "execution": execution,
            "readiness": system_readiness(project_root),
            "execution_authority": execution.get(
                "execution_authority", "NONE"
            ),
            "real_live_order_placed": bool(
                execution.get("real_live_order_placed", False)
            ),
        }

    if command == "preflight":
        readiness = system_readiness(project_root)
        blockers = list(readiness.get("hard_blockers", []))
        return {
            "schema": "stocks_launch_preflight_v1",
            "status": "GO" if not blockers else "NO_GO",
            "blockers": blockers,
            "readiness": readiness,
            "execution_authority": "NONE",
            "real_live_order_placed": False,
        }

    if command == "live":
        activation = execution_command(
            project_root,
            "activate-live-canary",
            env_file=".env.ibkr.live",
            approval=approval,
            confirmed=confirmed,
            profile=profile,
        )
        activated = (
            activation.get("status") == "GO"
            and activation.get("execution_authority") == LIVE_LEVEL_ONE
        )
        if not activated:
            return {
                "schema": "stocks_launch_live_v2",
                "status": "NO_GO",
                "launch_status": "LIVE_BLOCKED",
                "profile": profile,
                "blockers": activation.get("blockers", []),
                "activation": activation,
                "runtime_requested": continuous,
                "runtime_started": False,
                "resume_requested": resume,
                "execution_authority": "NONE",
                "real_live_order_placed": False,
            }
        runtime = (
            _start_runtime_task(project_root, mode="CONTROLLED_LIVE")
            if continuous
            else {
                "status": "NOT_REQUESTED",
                "runtime_started": False,
                "task_mode": None,
            }
        )
        if continuous and runtime.get("status") != "GO":
            runtime_blockers = runtime.get("blockers")
            if not isinstance(runtime_blockers, list):
                runtime_blockers = ["CONTROLLED_LIVE_RUNTIME_START_FAILED"]
            rollback = execution_command(
                project_root,
                "pause-live",
                approval="RUNTIME_START_FAILURE",
                env_file=".env.ibkr.live",
            )
            machine_command(project_root, "stop")
            return {
                "schema": "stocks_launch_live_v2",
                "status": "NO_GO",
                "launch_status": "LIVE_BLOCKED",
                "profile": profile,
                "blockers": sorted(
                    {str(blocker) for blocker in runtime_blockers}
                ),
                "activation": activation,
                "runtime": runtime,
                "rollback": rollback,
                "runtime_requested": True,
                "runtime_started": False,
                "resume_requested": resume,
                "execution_authority": "NONE",
                "real_live_order_placed": False,
            }
        return {
            "schema": "stocks_launch_live_v2",
            "status": "GO",
            "launch_status": (
                "LIVE_ACTIVE" if continuous else "LIVE_AUTHORITY_ACTIVE"
            ),
            "profile": profile,
            "blockers": [],
            "activation": activation,
            "runtime": runtime,
            "runtime_requested": continuous,
            "runtime_started": bool(runtime.get("runtime_started")),
            "resume_requested": resume,
            "execution_authority": LIVE_LEVEL_ONE,
            "real_live_order_placed": False,
        }

    raise ValueError(f"UNKNOWN_LAUNCH_COMMAND:{command}")


def _disable_runtime_task(project_root: Path) -> dict[str, Any]:
    """Disable restart triggers when the canonical runtime is stopped."""

    if not (project_root / "scripts" / "stop_bot.ps1").is_file():
        return {"status": "NOT_INSTALLED", "task_disabled": True}
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            (
                "$task = Get-ScheduledTask -TaskName "
                "'Stocks Canonical Runtime' -ErrorAction SilentlyContinue; "
                "if ($null -eq $task) { exit 3 }; "
                "Stop-ScheduledTask -TaskName "
                "'Stocks Canonical Runtime' -ErrorAction SilentlyContinue; "
                "Disable-ScheduledTask -TaskName "
                "'Stocks Canonical Runtime' -ErrorAction Stop | Out-Null"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode == 3:
        return {"status": "NOT_INSTALLED", "task_disabled": True}
    return {
        "status": "GO" if result.returncode == 0 else "NO_GO",
        "task_disabled": result.returncode == 0,
        "blockers": (
            [] if result.returncode == 0 else ["RUNTIME_TASK_DISABLE_FAILED"]
        ),
    }


def _start_runtime_task(
    project_root: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    install_script = project_root / "scripts" / "install_windows_service.ps1"
    start_script = project_root / "scripts" / "start_bot.ps1"
    missing = [
        str(path.name)
        for path in (install_script, start_script)
        if not path.exists()
    ]
    if missing:
        return {
            "status": "NO_GO",
            "runtime_started": False,
            "blockers": ["RUNTIME_TASK_SCRIPT_MISSING"],
            "missing_scripts": missing,
        }
    install = _run_powershell_script(
        install_script,
        arguments=("-Mode", mode),
    )
    if install.returncode != 0:
        return {
            "status": "NO_GO",
            "runtime_started": False,
            "blockers": ["RUNTIME_TASK_INSTALL_FAILED"],
            "task_mode": mode,
        }
    started = _run_powershell_script(start_script)
    if started.returncode != 0:
        return {
            "status": "NO_GO",
            "runtime_started": False,
            "blockers": ["RUNTIME_TASK_START_FAILED"],
            "task_mode": mode,
        }
    time.sleep(2)
    status = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            (
                "(Get-ScheduledTask -TaskName "
                "'Stocks Canonical Runtime' -ErrorAction Stop).State"
            ),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    task_state = status.stdout.strip().upper()
    if status.returncode != 0 or task_state != "RUNNING":
        return {
            "status": "NO_GO",
            "runtime_started": False,
            "blockers": ["RUNTIME_TASK_NOT_RUNNING"],
            "task_mode": mode,
            "task_state": task_state or "UNKNOWN",
        }
    return {
        "status": "GO",
        "runtime_started": True,
        "task_name": "Stocks Canonical Runtime",
        "task_mode": mode,
        "task_state": task_state,
    }


def _run_powershell_script(
    script: Path,
    *,
    arguments: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *arguments,
        ],
        cwd=script.parent.parent,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
