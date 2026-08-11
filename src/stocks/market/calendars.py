from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date
from typing import Any

import exchange_calendars as xcals

from stocks.ibkr.contracts import ResolvedContract


IBKR_EXCHANGE_CALENDAR_ALIASES: dict[str, str] = {
    "NASDAQ": "XNAS",
    "NYSE": "XNYS",
    "ARCA": "ARCX",
    "ARCX": "ARCX",
    "BATS": "BATS",
    "AEB": "XAMS",
    "LSE": "XLON",
    "XLON": "XLON",
    "TSEJ": "XTKS",
    "TSE": "XTKS",
    "SEHK": "XHKG",
    "HKEX": "XHKG",
    "CME": "CME",
    "GLOBEX": "CME",
    "COMEX": "COMEX",
    "NYMEX": "NYMEX",
    "ECBOT": "CBOT",
    "CBOT": "CBOT",
    "EUREX": "XEUR",
}


@dataclass(frozen=True)
class CalendarSessionCheck:
    status: str
    calendar_code: str | None
    calendar_name: str | None
    calendar_timezone: str | None
    session_date: date
    calendar_is_session: bool | None
    ibkr_has_trading_window: bool
    alignment: str
    session_open_utc: str | None = None
    session_close_utc: str | None = None
    break_start_utc: str | None = None
    break_end_utc: str | None = None
    calendar_is_early_close: bool | None = None
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "market_calendar_session_check_v1",
            "status": self.status,
            "calendar_code": self.calendar_code,
            "calendar_name": self.calendar_name,
            "calendar_timezone": self.calendar_timezone,
            "session_date": self.session_date.isoformat(),
            "calendar_is_session": self.calendar_is_session,
            "ibkr_has_trading_window": self.ibkr_has_trading_window,
            "alignment": self.alignment,
            "session_open_utc": self.session_open_utc,
            "session_close_utc": self.session_close_utc,
            "break_start_utc": self.break_start_utc,
            "break_end_utc": self.break_end_utc,
            "calendar_is_early_close": self.calendar_is_early_close,
            "reason": self.reason,
            "financial_calls": _zero_financial_calls(),
        }


def calendar_session_check(
    contract: ResolvedContract,
    *,
    session_date: date,
    ibkr_has_trading_window: bool,
) -> CalendarSessionCheck:
    calendar_code = calendar_code_for_contract(contract)
    if calendar_code is None:
        return CalendarSessionCheck(
            status="UNSUPPORTED_CALENDAR",
            calendar_code=None,
            calendar_name=None,
            calendar_timezone=None,
            session_date=session_date,
            calendar_is_session=None,
            ibkr_has_trading_window=ibkr_has_trading_window,
            alignment="UNKNOWN",
            reason="no supported exchange-calendar mapping for contract",
        )

    calendar = xcals.get_calendar(calendar_code)
    calendar_is_session = bool(calendar.is_session(session_date.isoformat()))
    session_open_utc = None
    session_close_utc = None
    break_start_utc = None
    break_end_utc = None
    calendar_is_early_close = None
    if calendar_is_session:
        session_open = calendar.session_open(session_date.isoformat()).tz_convert(UTC)
        session_close = calendar.session_close(session_date.isoformat()).tz_convert(UTC)
        session_open_utc = session_open.isoformat()
        session_close_utc = session_close.isoformat()
        schedule_row = calendar.schedule.loc[session_date.isoformat()]
        break_start = schedule_row.get("break_start")
        break_end = schedule_row.get("break_end")
        if break_start is not None and not getattr(break_start, "isna", lambda: False)():
            if str(break_start) != "NaT":
                break_start_utc = break_start.tz_convert(UTC).isoformat()
        if break_end is not None and not getattr(break_end, "isna", lambda: False)():
            if str(break_end) != "NaT":
                break_end_utc = break_end.tz_convert(UTC).isoformat()
        calendar_is_early_close = _calendar_has_early_close(calendar, session_date)

    alignment = "MATCH" if calendar_is_session == ibkr_has_trading_window else "MISMATCH"
    return CalendarSessionCheck(
        status="GO",
        calendar_code=calendar_code,
        calendar_name=calendar.name,
        calendar_timezone=str(calendar.tz),
        session_date=session_date,
        calendar_is_session=calendar_is_session,
        ibkr_has_trading_window=ibkr_has_trading_window,
        alignment=alignment,
        session_open_utc=session_open_utc,
        session_close_utc=session_close_utc,
        break_start_utc=break_start_utc,
        break_end_utc=break_end_utc,
        calendar_is_early_close=calendar_is_early_close,
        reason=None if alignment == "MATCH" else "IBKR cached hours disagree with exchange calendar",
    )


def calendar_code_for_contract(contract: ResolvedContract) -> str | None:
    candidates = [
        contract.primary_exchange,
        contract.exchange,
    ]
    if contract.valid_exchanges:
        candidates.extend(contract.valid_exchanges.split(","))

    for candidate in candidates:
        if candidate is None:
            continue
        code = _calendar_alias(candidate)
        if code is not None:
            return code
    return None


def _calendar_alias(value: str) -> str | None:
    text = value.strip().upper()
    if not text or text == "SMART":
        return None
    return IBKR_EXCHANGE_CALENDAR_ALIASES.get(text, text if text in xcals.get_calendar_names() else None)


def _calendar_has_early_close(calendar: Any, session_date: date) -> bool:
    early_closes = getattr(calendar, "early_closes", None)
    if early_closes is None:
        return False
    return any(item.date() == session_date for item in early_closes)


def _zero_financial_calls() -> dict[str, int]:
    return {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }
