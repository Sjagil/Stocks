from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import exchange_calendars as xcals

from stocks.market.calendars import IBKR_EXCHANGE_CALENDAR_ALIASES


@dataclass(frozen=True)
class AccountingDataContract:
    point_in_time_fundamentals: bool
    corporate_actions: bool
    delisting_settlement: bool
    eur_fx_accounting: bool
    market_calendar: bool
    stale_data_gate: bool

    def blockers(self) -> list[str]:
        return [
            name
            for name, value in self.__dict__.items()
            if not bool(value)
        ]


def eur_total_return(
    local_total_return: pd.DataFrame | pd.Series,
    fx_return: pd.DataFrame | pd.Series,
) -> pd.DataFrame | pd.Series:
    local, fx = local_total_return.align(fx_return, join="left")
    missing_fx = (
        bool(fx.isna().to_numpy().any())
        if isinstance(fx, pd.DataFrame)
        else bool(fx.isna().any())
    )
    if missing_fx:
        raise ValueError("MISSING_BLOCKING_FX_RETURN")
    return (1.0 + local) * (1.0 + fx) - 1.0


def apply_delisting_settlements(
    returns: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"symbol", "settlement_date", "recovery_return", "available_at"}
    if not required.issubset(events.columns):
        raise ValueError(
            f"DELISTING_FIELDS_MISSING:{','.join(sorted(required - set(events.columns)))}"
        )
    adjusted = returns.copy()
    applied: list[dict[str, Any]] = []
    for row in events.to_dict("records"):
        symbol = str(row["symbol"])
        if symbol not in adjusted:
            continue
        settlement = _utc(row["settlement_date"])
        available = _utc(row["available_at"])
        if available > settlement:
            raise ValueError("DELISTING_SETTLEMENT_NOT_POINT_IN_TIME")
        recovery = float(row["recovery_return"])
        if not -1.0 <= recovery <= 10.0:
            raise ValueError("INVALID_DELISTING_RECOVERY_RETURN")
        if settlement not in adjusted.index:
            raise ValueError("DELISTING_SETTLEMENT_SESSION_MISSING")
        adjusted.loc[settlement, symbol] = recovery
        adjusted.loc[adjusted.index > settlement, symbol] = 0.0
        applied.append(
            {
                "symbol": symbol,
                "settlement_date": settlement.isoformat(),
                "recovery_return": recovery,
            }
        )
    return adjusted, {
        "status": "GO",
        "settlement_count": len(applied),
        "events": applied,
        "unknown_delisting_settlements": 0,
    }


def validate_market_session_index(
    index: pd.DatetimeIndex,
    *,
    exchange: str,
) -> dict[str, Any]:
    exchange_name = str(exchange).strip().upper()
    calendar_code = IBKR_EXCHANGE_CALENDAR_ALIASES.get(exchange_name)
    if calendar_code is None:
        return {
            "status": "UNSUPPORTED_CALENDAR",
            "calendar_code": None,
            "unexpected_sessions": [],
        }
    if index.empty:
        return {
            "status": "NO_DATA",
            "calendar_code": calendar_code,
            "unexpected_sessions": [],
        }
    calendar = xcals.get_calendar(calendar_code)
    normalized = pd.DatetimeIndex(index).tz_convert("UTC").normalize()
    expected = pd.to_datetime(
        calendar.sessions_in_range(
            normalized.min().date().isoformat(),
            normalized.max().date().isoformat(),
        ),
        utc=True,
    ).normalize()
    unexpected = normalized.difference(expected)
    return {
        "status": "GO" if unexpected.empty else "UNEXPECTED_SESSION_BLOCKED",
        "calendar_code": calendar_code,
        "unexpected_sessions": [
            timestamp.date().isoformat() for timestamp in unexpected[:100]
        ],
        "missing_sessions": int(len(expected.difference(normalized))),
    }


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
