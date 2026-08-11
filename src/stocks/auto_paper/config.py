from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


@dataclass(frozen=True)
class AutoPaperConfig:
    enabled: bool
    client_id: int
    strategy_allowlist: tuple[str, ...]
    product_allowlist: tuple[str, ...]
    max_order_notional_eur: Decimal
    max_new_positions_per_day: int
    max_closing_orders_per_day: int
    max_open_positions: int
    max_portfolio_exposure_eur: Decimal
    max_daily_loss_eur: Decimal
    max_sector_exposure_pct: Decimal
    max_event_cluster_exposure_pct: Decimal
    max_signal_age_seconds: int
    max_quote_age_seconds: int
    max_spread_bps: Decimal
    rth_only: bool
    limit_only: bool
    require_shariah_fresh: bool
    heartbeat_timeout_seconds: int
    scheduler_interval_seconds: int = 60

    def safe_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "client_id_nonzero": self.client_id > 0,
            "strategy_allowlist": list(self.strategy_allowlist),
            "product_allowlist": list(self.product_allowlist),
            "max_order_notional_eur": str(self.max_order_notional_eur),
            "max_new_positions_per_day": self.max_new_positions_per_day,
            "max_closing_orders_per_day": self.max_closing_orders_per_day,
            "max_open_positions": self.max_open_positions,
            "max_portfolio_exposure_eur": str(self.max_portfolio_exposure_eur),
            "max_daily_loss_eur": str(self.max_daily_loss_eur),
            "max_sector_exposure_pct": str(self.max_sector_exposure_pct),
            "max_event_cluster_exposure_pct": str(self.max_event_cluster_exposure_pct),
            "max_signal_age_seconds": self.max_signal_age_seconds,
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "max_spread_bps": str(self.max_spread_bps),
            "rth_only": self.rth_only,
            "limit_only": self.limit_only,
            "require_shariah_fresh": self.require_shariah_fresh,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "scheduler_interval_seconds": self.scheduler_interval_seconds,
        }


def load_auto_paper_config(project_root: Path, env_file: str | Path = ".env.ibkr") -> tuple[AutoPaperConfig, list[str]]:
    values = _read_env(project_root / env_file if not Path(env_file).is_absolute() else Path(env_file))
    config = AutoPaperConfig(
        enabled=_bool(values.get("IBKR_AUTO_PAPER_ENABLED"), False),
        client_id=_int(values.get("IBKR_AUTO_PAPER_CLIENT_ID"), 1017),
        strategy_allowlist=_csv(values.get("IBKR_AUTO_PAPER_STRATEGY_ALLOWLIST")),
        product_allowlist=_csv(values.get("IBKR_AUTO_PAPER_PRODUCT_ALLOWLIST")),
        max_order_notional_eur=_decimal(values.get("IBKR_AUTO_PAPER_MAX_ORDER_NOTIONAL_EUR"), "50"),
        max_new_positions_per_day=_int(values.get("IBKR_AUTO_PAPER_MAX_NEW_POSITIONS_PER_DAY"), 1),
        max_closing_orders_per_day=_int(values.get("IBKR_AUTO_PAPER_MAX_CLOSING_ORDERS_PER_DAY"), 4),
        max_open_positions=_int(values.get("IBKR_AUTO_PAPER_MAX_OPEN_POSITIONS"), 2),
        max_portfolio_exposure_eur=_decimal(values.get("IBKR_AUTO_PAPER_MAX_PORTFOLIO_EXPOSURE_EUR"), "100"),
        max_daily_loss_eur=_decimal(values.get("IBKR_AUTO_PAPER_MAX_DAILY_LOSS_EUR"), "20"),
        max_sector_exposure_pct=_decimal(values.get("IBKR_AUTO_PAPER_MAX_SECTOR_EXPOSURE_PCT"), "25"),
        max_event_cluster_exposure_pct=_decimal(values.get("IBKR_AUTO_PAPER_MAX_EVENT_CLUSTER_EXPOSURE_PCT"), "15"),
        max_signal_age_seconds=_int(values.get("IBKR_AUTO_PAPER_MAX_SIGNAL_AGE_SECONDS"), 300),
        max_quote_age_seconds=_int(values.get("IBKR_AUTO_PAPER_MAX_QUOTE_AGE_SECONDS"), 15),
        max_spread_bps=_decimal(values.get("IBKR_AUTO_PAPER_MAX_SPREAD_BPS"), "40"),
        rth_only=_bool(values.get("IBKR_AUTO_PAPER_RTH_ONLY"), True),
        limit_only=_bool(values.get("IBKR_AUTO_PAPER_LIMIT_ONLY"), True),
        require_shariah_fresh=_bool(values.get("IBKR_AUTO_PAPER_REQUIRE_SHARIAH_FRESH"), True),
        heartbeat_timeout_seconds=_int(values.get("IBKR_AUTO_PAPER_HEARTBEAT_TIMEOUT_SECONDS"), 30),
        scheduler_interval_seconds=max(60, _int(values.get("IBKR_AUTO_PAPER_SCHEDULER_INTERVAL_SECONDS"), 60)),
    )
    errors = []
    if config.client_id <= 0 or config.client_id in {0, 17, 817, 917}:
        errors.append("AUTO_PAPER_CLIENT_ID_INVALID")
    if any(value <= 0 for value in (config.max_order_notional_eur, config.max_portfolio_exposure_eur, config.max_daily_loss_eur)):
        errors.append("AUTO_PAPER_LIMIT_CONFIG_INVALID")
    if config.max_new_positions_per_day != 1 or config.max_open_positions != 2:
        errors.append("AUTO_PAPER_FROZEN_LIMIT_CONFIG_MISMATCH")
    if not config.rth_only or not config.limit_only or not config.require_shariah_fresh:
        errors.append("AUTO_PAPER_FAIL_CLOSED_CONFIG_REQUIRED")
    return config, errors


def _read_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _decimal(value: str | None, default: str) -> Decimal:
    try:
        return Decimal(value) if value is not None else Decimal(default)
    except Exception:
        return Decimal(default)


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())
