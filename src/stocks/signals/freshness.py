from __future__ import annotations

from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any, Mapping

import exchange_calendars as xcals
import pandas as pd

from stocks.data.multitimeframe import (
    EXCHANGE_TIMEZONE_CALENDARS,
    bar_freshness,
)


SIGNAL_MAX_AGES = {
    "1h": timedelta(hours=12),
    "2h": timedelta(hours=24),
    "4h": timedelta(days=3),
    "1d": timedelta(days=10),
    "1w": timedelta(weeks=6),
    "1mo": timedelta(days=62),
}
INTRADAY_BAR_LENGTH = {
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
}


def evaluate_signal_freshness(
    row: Mapping[str, Any],
    *,
    now: datetime | None = None,
    declared_expiration: datetime | None = None,
) -> dict[str, Any]:
    observed = _utc_datetime(now or datetime.now(UTC))
    timeframe = _timeframe(row.get("timeframe"))
    data_timestamp = _utc_datetime(
        row.get("data_timestamp") or row.get("bar_timestamp")
    )
    declared = declared_expiration or _utc_datetime(
        row.get("expiration_timestamp") or row.get("expiration_time")
    )
    if observed is None or data_timestamp is None or declared is None:
        return _blocked("SIGNAL_FRESHNESS_TIMESTAMP_INVALID")

    maximum_age = SIGNAL_MAX_AGES.get(timeframe)
    wall_clock_limit = (
        data_timestamp + maximum_age
        if maximum_age is not None
        else declared
    )
    effective = min(declared, wall_clock_limit)
    exchange_timezone = str(row.get("exchange_timezone") or "").strip()
    session_context: dict[str, Any] = {}

    if timeframe in INTRADAY_BAR_LENGTH and exchange_timezone:
        session_context = bar_freshness(
            data_timestamp,
            interval=timeframe,
            observed_at=observed,
            exchange_timezone=exchange_timezone,
        )
        if session_context.get("status") != "FRESH_CLOSED_BAR":
            return {
                "status": "STALE",
                "is_current": False,
                "effective_expiration": effective,
                "freshness_basis": session_context.get(
                    "freshness_basis", "EXCHANGE_SESSION"
                ),
                "reason": "UNDERLYING_CLOSED_BAR_STALE",
                **_public_session_context(session_context),
            }
        if effective < observed:
            if session_context.get("market_open") is False:
                next_close = _next_relevant_bar_close(
                    observed,
                    timeframe=timeframe,
                    exchange_timezone=exchange_timezone,
                )
                if next_close is not None:
                    effective = max(effective, next_close)
            else:
                effective = max(
                    effective,
                    min(
                        wall_clock_limit,
                        observed + INTRADAY_BAR_LENGTH[timeframe],
                    ),
                )

    current = effective >= observed
    return {
        "status": "FRESH" if current else "STALE",
        "is_current": current,
        "effective_expiration": effective,
        "freshness_basis": (
            "EXCHANGE_SESSION_AWARE_SIGNAL_EXPIRY_V1"
            if session_context
            else "WALL_CLOCK_SIGNAL_EXPIRY"
        ),
        "reason": None if current else "SIGNAL_EXPIRATION_REACHED",
        **_public_session_context(session_context),
    }


def effective_signal_expiration(
    row: Mapping[str, Any],
    *,
    now: datetime | None = None,
    declared_expiration: datetime | None = None,
) -> datetime:
    result = evaluate_signal_freshness(
        row,
        now=now,
        declared_expiration=declared_expiration,
    )
    value = result.get("effective_expiration")
    return (
        value
        if isinstance(value, datetime)
        else datetime.min.replace(tzinfo=UTC)
    )


def signal_is_current(
    row: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    return bool(evaluate_signal_freshness(row, now=now)["is_current"])


def _next_relevant_bar_close(
    observed: datetime,
    *,
    timeframe: str,
    exchange_timezone: str,
) -> datetime | None:
    calendar_code = EXCHANGE_TIMEZONE_CALENDARS.get(exchange_timezone)
    duration = INTRADAY_BAR_LENGTH.get(timeframe)
    if calendar_code is None or duration is None:
        return None
    try:
        calendar = _calendar(calendar_code)
        minute = pd.Timestamp(observed).tz_convert("UTC").floor("min")
        session = calendar.minute_to_session(minute, direction="next")
        session_open = pd.Timestamp(calendar.session_open(session))
        session_close = pd.Timestamp(calendar.session_close(session))
        close = min(session_open + pd.Timedelta(duration), session_close)
    except (KeyError, TypeError, ValueError):
        return None
    return close.to_pydatetime()


@lru_cache(maxsize=None)
def _calendar(calendar_code: str) -> Any:
    return xcals.get_calendar(calendar_code)


def _timeframe(value: Any) -> str:
    timeframe = str(value or "").strip().lower()
    return {
        "60m": "1h",
        "1hour": "1h",
        "2hour": "2h",
        "4hour": "4h",
        "daily": "1d",
        "weekly": "1w",
        "monthly": "1mo",
    }.get(timeframe, timeframe)


def _utc_datetime(value: Any) -> datetime | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def _public_session_context(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: context.get(key)
        for key in (
            "calendar_code",
            "market_open",
            "latest_completed_session",
            "bar_session",
        )
        if key in context
    }


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "status": "STALE",
        "is_current": False,
        "effective_expiration": datetime.min.replace(tzinfo=UTC),
        "freshness_basis": "INVALID_TIMESTAMP_BLOCKED",
        "reason": reason,
    }
