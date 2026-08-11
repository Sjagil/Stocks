from __future__ import annotations

import hashlib
import socket
from importlib import metadata
from pathlib import Path
from typing import Any

from stocks.application.config import IbkrSettings
from stocks.ibkr.connection import ReadOnlyIbkrConnectionService

from .context import AppContext


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def ibapi_distribution_version() -> str | None:
    try:
        return metadata.version("ibapi")
    except metadata.PackageNotFoundError:
        return None


def build_doctor_report(context: AppContext, project_root: Path) -> dict[str, Any]:
    api_source = Path(r"C:\TWS API\source\pythonclient")
    return {
        "schema": "stocks_doctor_v1",
        "config": context.ibkr.safe_dict(),
        "phase0_status": "IBKR_PHASE_0_READ_ONLY_CONNECTIVITY_GO",
        "files": {
            "env_ibkr_exists": (project_root / ".env.ibkr").exists(),
            "requirements_exists": (project_root / "requirements.txt").exists(),
            "requirements_lock_exists": (project_root / "requirements.lock.txt").exists(),
            "phase0_probe_exists": (project_root / "ibkr_tws_probe.py").exists(),
            "official_tws_api_source_exists": api_source.exists(),
        },
        "hashes": {
            "ibkr_tws_probe_sha256": sha256_file(project_root / "ibkr_tws_probe.py"),
            "requirements_lock_sha256": sha256_file(project_root / "requirements.lock.txt"),
        },
        "ibapi_distribution_version": ibapi_distribution_version(),
        "canonical_entrypoint": "main.py",
        "read_only_authority": context.ibkr.order_authority,
    }


def build_disconnect_drill_preflight_report(
    context: AppContext,
    *,
    require_socket: bool = True,
) -> dict[str, Any]:
    checks = _disconnect_drill_preflight_checks(context.ibkr)
    if require_socket:
        checks["paper_socket_reachable"] = _socket_reachable(
            context.ibkr.host,
            context.ibkr.port,
            timeout_seconds=min(context.ibkr.connect_timeout_seconds, 3.0),
        )
    else:
        checks["paper_socket_reachable"] = None

    blocking_checks = [name for name, passed in checks.items() if passed is False]
    return {
        "schema": "phase1_disconnect_drill_preflight_v1",
        "status": "GO" if not blocking_checks else "NO_GO",
        "host": context.ibkr.host,
        "port": context.ibkr.port,
        "client_id": context.ibkr.client_id,
        "socket_check_required": require_socket,
        "checks": checks,
        "blocking_checks": blocking_checks,
        "operator_action": (
            "start_disconnect_drill"
            if not blocking_checks
            else "fix_preflight_before_disconnect_drill"
        ),
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def _disconnect_drill_preflight_checks(settings: IbkrSettings) -> dict[str, bool | None]:
    return {
        "tws_paper_host_127_0_0_1": settings.host == "127.0.0.1",
        "tws_paper_port_7497": settings.port == 7497,
        "read_only": settings.read_only is True,
        "order_authority_none": settings.order_authority == "NONE",
        "live_trading_disabled": settings.live_trading_enabled is False,
        "order_transmission_disabled": settings.allow_order_transmission is False,
        "max_order_notional_zero": settings.max_order_notional_eur == 0,
        "max_open_orders_zero": settings.max_open_orders == 0,
        "max_positions_zero": settings.max_positions == 0,
    }


def _socket_reachable(host: str, port: int, *, timeout_seconds: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def build_ibkr_service(context: AppContext) -> ReadOnlyIbkrConnectionService:
    return ReadOnlyIbkrConnectionService(context.ibkr)
