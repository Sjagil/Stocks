from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field, model_validator


PAPER_PORTS = {7497, 4002}
LIVE_PORTS = {7496, 4001}
PHASE1_HOST = "127.0.0.1"
PHASE1_HOST_ALIASES = {"localhost": PHASE1_HOST}
PHASE1_ALLOWED_SECURITY_TYPES = {"STK", "FUT"}
PHASE1_ALLOWED_CURRENCIES = {"EUR", "USD"}
PHASE1_MARKET_DATA_TYPE = 3
PHASE1_OUTPUT_DIR = Path("output/ibkr")
PHASE1_APP_ENVS = {"development", "test", "paper"}
PHASE1_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
BOOL_WORD_TOKENS = {"true", "false", "yes", "no", "on", "off"}
INT_TOKEN_PATTERN = re.compile(r"[0-9]+")
FLOAT_TOKEN_PATTERN = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)")


def _bool_value(raw: str | bool | None, *, default: bool = False) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value {raw!r}; expected true/false")


def _app_env_value(raw: str | None, *, default: str = "development") -> str:
    if raw is None or raw == "":
        return default
    if raw != raw.strip():
        raise ValueError("APP_ENV must not contain surrounding whitespace")
    value = raw.lower()
    if value not in PHASE1_APP_ENVS:
        allowed = ",".join(sorted(PHASE1_APP_ENVS))
        raise ValueError(f"APP_ENV must be one of {allowed} in Phase 1")
    return value


def _host_value(raw: str | None, *, default: str = PHASE1_HOST) -> str:
    if raw is None or raw == "":
        return default
    if raw != raw.strip():
        raise ValueError("IBKR_HOST must not contain surrounding whitespace")
    value = raw.lower()
    value = PHASE1_HOST_ALIASES.get(value, raw)
    if value != PHASE1_HOST:
        raise ValueError("IBKR_HOST=127.0.0.1 is required in Phase 1")
    return PHASE1_HOST


def _int_value(raw: str | None, *, default: int) -> int:
    if raw is None or raw == "":
        return default
    if raw != raw.strip():
        raise ValueError(f"invalid integer value {raw!r}; surrounding whitespace is not allowed")
    if raw.lower() in BOOL_WORD_TOKENS or not INT_TOKEN_PATTERN.fullmatch(raw):
        raise ValueError(f"invalid integer value {raw!r}; expected base-10 digits")
    return int(raw)


def _float_value(raw: str | None, *, default: float) -> float:
    if raw is None or raw == "":
        return default
    if raw != raw.strip():
        raise ValueError(f"invalid float value {raw!r}; surrounding whitespace is not allowed")
    if raw.lower() in BOOL_WORD_TOKENS or not FLOAT_TOKEN_PATTERN.fullmatch(raw):
        raise ValueError(f"invalid float value {raw!r}; expected decimal digits")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"invalid float value {raw!r}; expected finite number")
    return value


def _float_csv_value(raw: str | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if raw is None or raw.strip() == "":
        return default
    return tuple(_float_value(item, default=0.0) for item in raw.split(","))


def _csv_value(raw: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None or raw.strip() == "":
        return default
    return tuple(item.strip().upper() for item in raw.split(",") if item.strip())


def _output_dir_value(raw: str | None, *, default: Path = PHASE1_OUTPUT_DIR) -> Path:
    if raw is None or raw == "":
        return default
    if raw != raw.strip():
        raise ValueError("IBKR_OUTPUT_DIR must not contain surrounding whitespace")
    normalized = raw.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("IBKR_OUTPUT_DIR must be the relative path output/ibkr")
    if tuple(path.parts) != tuple(PHASE1_OUTPUT_DIR.parts):
        raise ValueError("IBKR_OUTPUT_DIR=output/ibkr is required in Phase 1")
    return PHASE1_OUTPUT_DIR


def _account_value(raw: str | None) -> str:
    if raw is None or raw == "":
        return ""
    raise ValueError("IBKR_ACCOUNT must remain empty in Phase 1")


def _order_authority_value(raw: str | None, *, default: str = "NONE") -> str:
    if raw is None or raw == "":
        return default
    if raw != raw.strip():
        raise ValueError("IBKR_ORDER_AUTHORITY must not contain surrounding whitespace")
    value = raw.upper()
    if value != "NONE":
        raise ValueError("IBKR_ORDER_AUTHORITY=NONE is required")
    return "NONE"


def _log_level_value(raw: str | None, *, default: str = "INFO") -> str:
    if raw is None or raw == "":
        return default
    if raw != raw.strip():
        raise ValueError("IBKR_LOG_LEVEL must not contain surrounding whitespace")
    value = raw.upper()
    if value not in PHASE1_LOG_LEVELS:
        allowed = ",".join(sorted(PHASE1_LOG_LEVELS))
        raise ValueError(f"IBKR_LOG_LEVEL must be one of {allowed}")
    return value


class IbkrSettings(BaseModel):
    env_file: Path = Field(default=Path(".env.ibkr"))
    app_env: str = "development"
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 17
    account: str = ""
    read_only: bool = True
    order_authority: str = "NONE"
    live_trading_enabled: bool = False
    allow_order_transmission: bool = False
    market_data_type: int = 3
    allowed_security_types: tuple[str, ...] = ("STK", "FUT")
    allowed_currencies: tuple[str, ...] = ("EUR", "USD")
    max_order_notional_eur: int = 0
    max_open_orders: int = 0
    max_positions: int = 0
    connect_timeout_seconds: float = 12.0
    request_timeout_seconds: float = 15.0
    heartbeat_interval_seconds: float = 25.0
    stale_after_seconds: float = 45.0
    max_reconnect_attempts: int = 5
    reconnect_delays_seconds: tuple[float, ...] = (2.0, 5.0, 15.0, 30.0)
    output_dir: Path = Path("output/ibkr")
    log_level: str = "INFO"

    @model_validator(mode="after")
    def validate_fail_closed(self) -> IbkrSettings:
        self.app_env = _app_env_value(self.app_env)
        self.host = _host_value(self.host)
        if self.port in LIVE_PORTS or self.port not in PAPER_PORTS:
            raise ValueError(f"IBKR Phase 1 only allows paper ports {sorted(PAPER_PORTS)}")
        if not self.read_only:
            raise ValueError("IBKR_READ_ONLY=true is required")
        self.order_authority = _order_authority_value(self.order_authority)
        if self.live_trading_enabled:
            raise ValueError("IBKR_LIVE_TRADING_ENABLED=false is required")
        if self.allow_order_transmission:
            raise ValueError("IBKR_ALLOW_ORDER_TRANSMISSION=false is required")
        if self.market_data_type != PHASE1_MARKET_DATA_TYPE:
            raise ValueError("IBKR_MARKET_DATA_TYPE=3 is required in Phase 1")
        allowed_security_types = set(self.allowed_security_types)
        if (
            not allowed_security_types
            or allowed_security_types - PHASE1_ALLOWED_SECURITY_TYPES
        ):
            raise ValueError("IBKR_ALLOWED_SECURITY_TYPES may only contain STK,FUT")
        allowed_currencies = set(self.allowed_currencies)
        if not allowed_currencies or allowed_currencies - PHASE1_ALLOWED_CURRENCIES:
            raise ValueError("IBKR_ALLOWED_CURRENCIES may only contain EUR,USD")
        if self.max_order_notional_eur != 0:
            raise ValueError("IBKR_MAX_ORDER_NOTIONAL_EUR=0 is required in Phase 1")
        if self.max_open_orders != 0:
            raise ValueError("IBKR_MAX_OPEN_ORDERS=0 is required in Phase 1")
        if self.max_positions != 0:
            raise ValueError("IBKR_MAX_POSITIONS=0 is required in Phase 1")
        if self.client_id <= 0:
            raise ValueError("IBKR_CLIENT_ID must be positive")
        timing_values = (
            self.connect_timeout_seconds,
            self.request_timeout_seconds,
            self.heartbeat_interval_seconds,
            self.stale_after_seconds,
        )
        if any(not math.isfinite(value) for value in timing_values):
            raise ValueError("IBKR timing settings must be finite")
        if self.connect_timeout_seconds <= 0 or self.request_timeout_seconds <= 0:
            raise ValueError("IBKR timeouts must be positive")
        if self.heartbeat_interval_seconds <= 0 or self.stale_after_seconds <= 0:
            raise ValueError("heartbeat and stale thresholds must be positive")
        if self.stale_after_seconds <= self.heartbeat_interval_seconds:
            raise ValueError("IBKR_STALE_AFTER_SECONDS must exceed heartbeat interval")
        if self.max_reconnect_attempts <= 0:
            raise ValueError("IBKR_MAX_RECONNECT_ATTEMPTS must be positive")
        if not self.reconnect_delays_seconds:
            raise ValueError("IBKR_RECONNECT_DELAYS_SECONDS must not be empty")
        if any(
            not math.isfinite(delay) or delay <= 0
            for delay in self.reconnect_delays_seconds
        ):
            raise ValueError("IBKR_RECONNECT_DELAYS_SECONDS values must be positive finite numbers")
        self.log_level = _log_level_value(self.log_level)
        return self

    def safe_dict(self) -> dict[str, Any]:
        return {
            "app_env": self.app_env,
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "account_configured": bool(self.account),
            "read_only": self.read_only,
            "order_authority": self.order_authority,
            "live_trading_enabled": self.live_trading_enabled,
            "allow_order_transmission": self.allow_order_transmission,
            "market_data_type": self.market_data_type,
            "allowed_security_types": self.allowed_security_types,
            "allowed_currencies": self.allowed_currencies,
            "max_order_notional_eur": self.max_order_notional_eur,
            "max_open_orders": self.max_open_orders,
            "max_positions": self.max_positions,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "request_timeout_seconds": self.request_timeout_seconds,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
            "stale_after_seconds": self.stale_after_seconds,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "reconnect_delays_seconds": self.reconnect_delays_seconds,
            "output_dir": str(self.output_dir),
            "log_level": self.log_level,
        }


def load_ibkr_settings(env_file: str | Path = ".env.ibkr") -> IbkrSettings:
    env_path = Path(env_file)
    values = dotenv_values(env_path) if env_path.exists() else {}

    def get(name: str, default: str | None = None) -> str | None:
        if name in os.environ:
            return os.environ[name]
        value = values.get(name)
        if value is None:
            return default
        return str(value)

    return IbkrSettings(
        env_file=env_path,
        app_env=_app_env_value(get("APP_ENV")),
        host=_host_value(get("IBKR_HOST")),
        port=_int_value(get("IBKR_PORT"), default=7497),
        client_id=_int_value(get("IBKR_CLIENT_ID"), default=17),
        account=_account_value(get("IBKR_ACCOUNT")),
        read_only=_bool_value(get("IBKR_READ_ONLY"), default=True),
        order_authority=_order_authority_value(get("IBKR_ORDER_AUTHORITY")),
        live_trading_enabled=_bool_value(
            get("IBKR_LIVE_TRADING_ENABLED"),
            default=False,
        ),
        allow_order_transmission=_bool_value(
            get("IBKR_ALLOW_ORDER_TRANSMISSION"),
            default=False,
        ),
        market_data_type=_int_value(get("IBKR_MARKET_DATA_TYPE"), default=3),
        allowed_security_types=_csv_value(
            get("IBKR_ALLOWED_SECURITY_TYPES"),
            default=("STK", "FUT"),
        ),
        allowed_currencies=_csv_value(
            get("IBKR_ALLOWED_CURRENCIES"),
            default=("EUR", "USD"),
        ),
        max_order_notional_eur=_int_value(
            get("IBKR_MAX_ORDER_NOTIONAL_EUR"),
            default=0,
        ),
        max_open_orders=_int_value(get("IBKR_MAX_OPEN_ORDERS"), default=0),
        max_positions=_int_value(get("IBKR_MAX_POSITIONS"), default=0),
        connect_timeout_seconds=_float_value(
            get("IBKR_CONNECT_TIMEOUT_SECONDS"),
            default=12.0,
        ),
        request_timeout_seconds=_float_value(
            get("IBKR_REQUEST_TIMEOUT_SECONDS"),
            default=15.0,
        ),
        heartbeat_interval_seconds=_float_value(
            get("IBKR_HEARTBEAT_INTERVAL_SECONDS"),
            default=25.0,
        ),
        stale_after_seconds=_float_value(
            get("IBKR_STALE_AFTER_SECONDS"),
            default=45.0,
        ),
        max_reconnect_attempts=_int_value(
            get("IBKR_MAX_RECONNECT_ATTEMPTS"),
            default=5,
        ),
        reconnect_delays_seconds=_float_csv_value(
            get("IBKR_RECONNECT_DELAYS_SECONDS"),
            default=(2.0, 5.0, 15.0, 30.0),
        ),
        output_dir=_output_dir_value(get("IBKR_OUTPUT_DIR"), default=PHASE1_OUTPUT_DIR),
        log_level=_log_level_value(get("IBKR_LOG_LEVEL")),
    )
