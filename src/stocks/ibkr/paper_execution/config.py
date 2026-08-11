from __future__ import annotations

import os
import json
import sqlite3
from decimal import Decimal
from pathlib import Path

from dotenv import dotenv_values

from stocks.application.config import PAPER_PORTS, load_ibkr_settings
from stocks.ibkr.paper_execution.errors import (
    LIVE_CONFIGURATION_BLOCKED,
    NON_PAPER_ENVIRONMENT_BLOCKED,
    PAPER_ACCOUNT_FINGERPRINT_MISMATCH,
    PAPER_WRITER_DISABLED,
    WRITER_CLIENT_ID_COLLISION_BLOCKED,
    WRITER_CLIENT_ID_ZERO_BLOCKED,
)
from stocks.ibkr.paper_execution.models import PaperWriterConfig
from stocks.ibkr.reconciliation.requests import load_phase8_config


def load_paper_writer_config(project_root: Path, env_file: str | Path = ".env.ibkr") -> tuple[PaperWriterConfig | None, list[str]]:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = project_root / env_path
    errors: list[str] = []
    if env_path.name != ".env.ibkr":
        errors.append("PAPER_WRITER_CONFIG_BLOCKED")
    values = dotenv_values(env_path) if env_path.exists() else {}

    def get(name: str) -> str | None:
        if name in os.environ:
            return os.environ[name]
        value = values.get(name)
        return None if value is None else str(value)

    try:
        base = load_ibkr_settings(env_path)
    except Exception:
        return None, ["PAPER_WRITER_CONFIG_BLOCKED"]
    phase8, phase8_errors = load_phase8_config(project_root, env_path)
    if phase8 is None or phase8_errors:
        errors.append("PHASE8_OBSERVER_CONFIG_BLOCKED")
    observer_client_id = -1 if phase8 is None else phase8.recon_client_id
    writer_client_id = _int(get("IBKR_PAPER_WRITER_CLIENT_ID"), default=-1)
    writer_enabled = _bool(get("IBKR_PAPER_WRITER_ENABLED"), default=False)
    approved_fingerprint = get("IBKR_PAPER_ACCOUNT_FINGERPRINT") or ""
    observed_fingerprint = _approved_baseline_fingerprint(project_root)
    max_notional = _decimal(get("IBKR_PAPER_MAX_ORDER_NOTIONAL_EUR"), default=Decimal("250"))
    max_open_orders = _int(get("IBKR_PAPER_MAX_OPEN_ORDERS"), default=1)
    max_positions = _int(get("IBKR_PAPER_MAX_POSITIONS"), default=1)
    max_new_orders = _int(get("IBKR_PAPER_MAX_NEW_ORDERS_PER_DAY"), default=1)
    max_closing_orders = _int(get("IBKR_PAPER_MAX_CLOSING_ORDERS_PER_DAY"), default=1)
    approval_ttl = _int(get("IBKR_PAPER_APPROVAL_TTL_SECONDS"), default=300)
    callback_timeout = _float(get("IBKR_PAPER_CALLBACK_TIMEOUT_SECONDS"), default=15.0)
    recon_timeout = _float(get("IBKR_PAPER_RECONCILIATION_TIMEOUT_SECONDS"), default=30.0)

    if writer_client_id == 0:
        errors.append(WRITER_CLIENT_ID_ZERO_BLOCKED)
    if writer_client_id in {-1, base.client_id, observer_client_id}:
        errors.append(WRITER_CLIENT_ID_COLLISION_BLOCKED)
    if not writer_enabled:
        errors.append(PAPER_WRITER_DISABLED)
    if base.port not in PAPER_PORTS:
        errors.append(NON_PAPER_ENVIRONMENT_BLOCKED)
    if base.live_trading_enabled or base.allow_order_transmission:
        errors.append(LIVE_CONFIGURATION_BLOCKED)
    if not approved_fingerprint or approved_fingerprint != observed_fingerprint:
        errors.append(PAPER_ACCOUNT_FINGERPRINT_MISMATCH)
    if max_notional > Decimal("250") or max_notional <= 0:
        errors.append("PAPER_NOTIONAL_LIMIT_BLOCKED")
    if max_open_orders > 1 or max_positions > 1:
        errors.append("PAPER_CANARY_LIMIT_BLOCKED")
    if max_new_orders != 1:
        errors.append("OPENING_ORDER_DAILY_LIMIT_BLOCKED")
    if max_closing_orders != 1:
        errors.append("CLOSING_ORDER_DAILY_LIMIT_BLOCKED")
    if any(value <= 0 for value in (approval_ttl, callback_timeout, recon_timeout)):
        errors.append("PAPER_TIMEOUT_CONFIG_BLOCKED")

    config = PaperWriterConfig(
        host=base.host,
        port=base.port,
        phase1_client_id=base.client_id,
        observer_client_id=observer_client_id,
        writer_client_id=writer_client_id,
        writer_enabled=writer_enabled,
        approved_account_fingerprint=approved_fingerprint,
        observed_account_fingerprint=observed_fingerprint,
        max_order_notional_eur=max_notional,
        max_quantity=Decimal("1"),
        max_open_orders=max_open_orders,
        max_positions=max_positions,
        max_new_orders_per_day=max_new_orders,
        max_closing_orders_per_day=max_closing_orders,
        approval_ttl_seconds=approval_ttl,
        callback_timeout_seconds=callback_timeout,
        reconciliation_timeout_seconds=recon_timeout,
        live_trading_enabled=base.live_trading_enabled,
        allow_order_transmission=base.allow_order_transmission,
    )
    return config, sorted(set(errors))


def _approved_baseline_fingerprint(project_root: Path) -> str:
    fingerprints = _private_observed_account_fingerprints(project_root)
    return next(iter(fingerprints)) if len(fingerprints) == 1 else ""


def _private_observed_account_fingerprints(project_root: Path) -> set[str]:
    """Return the newest valid fingerprint set from each observation store."""
    fingerprints: set[str] = set()
    databases = (
        (project_root / "data" / "broker" / "phase8" / "private" / "broker_observation.sqlite3", "snapshots"),
        (project_root / "data" / "broker" / "phase8_1" / "private" / "observation_soak.sqlite3", "snapshots"),
    )
    for db_path, table_name in databases:
        if not db_path.exists():
            continue
        try:
            with sqlite3.connect(db_path) as connection:
                rows = connection.execute(
                    f'SELECT payload_json FROM "{table_name}" ORDER BY rowid DESC LIMIT 50'
                ).fetchall()
        except sqlite3.Error:
            continue
        for (payload_json,) in rows:
            try:
                payload = json.loads(str(payload_json))
            except json.JSONDecodeError:
                continue
            current = {
                value
                for value in _walk_account_fingerprints(payload)
                if _is_hex64(value)
            }
            if current:
                fingerprints.update(current)
                break
    return {value for value in fingerprints if _is_hex64(value)}


def _walk_account_fingerprints(payload: object) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "account_fingerprint" and isinstance(value, str):
                found.add(value.strip())
            found.update(_walk_account_fingerprints(value))
    elif isinstance(payload, list):
        for item in payload:
            found.update(_walk_account_fingerprints(item))
    return found


def _is_hex64(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _int(raw: str | None, *, default: int) -> int:
    return default if raw is None or not raw.isdigit() else int(raw)


def _float(raw: str | None, *, default: float) -> float:
    try:
        return default if raw is None or raw == "" else float(raw)
    except ValueError:
        return default


def _decimal(raw: str | None, *, default: Decimal) -> Decimal:
    try:
        return default if raw is None or raw == "" else Decimal(raw)
    except Exception:
        return default


def _bool(raw: str | None, *, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}
