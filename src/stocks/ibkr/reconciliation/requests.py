from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from stocks.application.config import LIVE_PORTS, PAPER_PORTS, load_ibkr_settings
from stocks.ibkr.reconciliation.errors import BROKER_OBSERVATION_AUTHORITY
from stocks.ibkr.reconciliation.masking import hash_optional_text


ACCOUNT_SUMMARY_TAGS = (
    "AccountType",
    "NetLiquidation",
    "TotalCashValue",
    "SettledCash",
    "AvailableFunds",
    "GrossPositionValue",
    "BuyingPower",
    "InitMarginReq",
    "MaintMarginReq",
    "ExcessLiquidity",
    "$LEDGER:ALL",
)


@dataclass(frozen=True)
class Phase8Config:
    env_file: Path
    host: str
    port: int
    primary_client_id: int
    recon_client_id: int
    account_fingerprint_key: str
    request_timeout_seconds: float
    commission_grace_seconds: float
    snapshot_stability_delay_seconds: float
    read_only: bool
    live_trading_enabled: bool
    allow_order_transmission: bool
    execution_authority: str
    broker_observation_authority: str = BROKER_OBSERVATION_AUTHORITY

    @property
    def masked_recon_client_id(self) -> str:
        masked = hash_optional_text(
            str(self.recon_client_id), self.account_fingerprint_key
        )
        return masked[:16] if masked else "MASKED"

    def safe_dict(self) -> dict[str, Any]:
        return {
            "env_file": str(self.env_file),
            "host": self.host,
            "port": self.port,
            "primary_client_id": self.primary_client_id,
            "recon_client_id_masked": self.masked_recon_client_id,
            "recon_client_id_nonzero": self.recon_client_id != 0,
            "observer_client_id_unique": self.recon_client_id not in {0, self.primary_client_id},
            "fingerprint_key_configured": bool(self.account_fingerprint_key),
            "request_timeout_seconds": self.request_timeout_seconds,
            "commission_grace_seconds": self.commission_grace_seconds,
            "snapshot_stability_delay_seconds": self.snapshot_stability_delay_seconds,
            "read_only": self.read_only,
            "live_trading_enabled": self.live_trading_enabled,
            "allow_order_transmission": self.allow_order_transmission,
            "execution_authority": self.execution_authority,
            "broker_observation_authority": self.broker_observation_authority,
        }


def load_phase8_config(project_root: Path, env_file: str | Path = ".env.ibkr") -> tuple[Phase8Config | None, list[str]]:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = project_root / env_path
    errors: list[str] = []
    if env_path.name != ".env.ibkr":
        errors.append("RECON_CONFIG_BLOCKED")
    values = dotenv_values(env_path) if env_path.exists() else {}
    base = None
    try:
        base = load_ibkr_settings(env_path)
    except Exception:
        errors.append("RECON_CONFIG_BLOCKED")

    def get(name: str) -> str | None:
        if name in os.environ:
            return os.environ[name]
        value = values.get(name)
        return None if value is None else str(value)

    recon_client_id = _int(get("IBKR_RECON_CLIENT_ID"), default=-1)
    fingerprint_key = get("IBKR_ACCOUNT_FINGERPRINT_KEY") or ""
    timeout = _float(get("IBKR_RECON_REQUEST_TIMEOUT_SECONDS"), default=15.0)
    grace = _float(get("IBKR_RECON_COMMISSION_GRACE_SECONDS"), default=2.0)
    stability = _float(get("IBKR_RECON_SNAPSHOT_STABILITY_DELAY_SECONDS"), default=1.0)

    if recon_client_id == 0:
        errors.append("CLIENT_ID_ZERO_BLOCKED")
    if base is not None and recon_client_id == base.client_id:
        errors.append("CLIENT_ID_COLLISION_BLOCKED")
    if recon_client_id < 0:
        errors.append("RECON_CONFIG_BLOCKED")
    if not fingerprint_key:
        errors.append("ACCOUNT_FINGERPRINT_KEY_MISSING")
    if base is not None:
        if base.port in LIVE_PORTS or base.port not in PAPER_PORTS or not base.read_only or base.live_trading_enabled or base.allow_order_transmission:
            errors.append("NON_PAPER_CONFIGURATION_BLOCKED")
    if any(not math.isfinite(item) or item <= 0 for item in (timeout, grace, stability)):
        errors.append("RECON_CONFIG_BLOCKED")

    if base is None:
        return None, sorted(set(errors))
    config = Phase8Config(
        env_file=env_path,
        host=base.host,
        port=base.port,
        primary_client_id=base.client_id,
        recon_client_id=recon_client_id,
        account_fingerprint_key=fingerprint_key,
        request_timeout_seconds=timeout,
        commission_grace_seconds=grace,
        snapshot_stability_delay_seconds=stability,
        read_only=base.read_only,
        live_trading_enabled=base.live_trading_enabled,
        allow_order_transmission=base.allow_order_transmission,
        execution_authority=base.order_authority,
    )
    return config, sorted(set(errors))


def _int(raw: str | None, *, default: int) -> int:
    if raw is None or raw == "":
        return default
    if not raw.isdigit():
        return default
    return int(raw)


def _float(raw: str | None, *, default: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default
