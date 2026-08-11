from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time


@dataclass(frozen=True)
class IbkrSessionWindow:
    session_date: date
    start: time | None
    end: time | None
    closed: bool
    start_date: date | None = None
    end_date: date | None = None


def parse_ibkr_hours(value: str) -> tuple[IbkrSessionWindow, ...]:
    text = value.strip()
    if not text:
        raise ValueError("hours string is blank")

    windows: list[IbkrSessionWindow] = []
    for raw_segment in text.split(";"):
        segment = raw_segment.strip()
        if not segment:
            raise ValueError("hours string contains an empty segment")
        if ":" not in segment:
            raise ValueError(f"hours segment lacks date separator: {segment}")
        session_date_text, session_text = segment.split(":", 1)
        session_date = _parse_ibkr_date(session_date_text)
        if session_text == "CLOSED":
            windows.append(
                IbkrSessionWindow(
                    session_date=session_date,
                    start=None,
                    end=None,
                    closed=True,
                )
            )
            continue

        for raw_window in session_text.split(","):
            window = raw_window.strip()
            if "-" not in window:
                raise ValueError(f"hours window lacks range separator: {window}")
            start_text, end_text = window.split("-", 1)
            start_date, start_time = _parse_ibkr_time_endpoint(start_text)
            end_date, end_time = _parse_ibkr_time_endpoint(end_text)
            windows.append(
                IbkrSessionWindow(
                    session_date=session_date,
                    start=start_time,
                    end=end_time,
                    closed=False,
                    start_date=start_date,
                    end_date=end_date,
                )
            )

    if not windows:
        raise ValueError("hours string contains no sessions")
    return tuple(windows)


def validate_ibkr_hours_field(field_name: str, value: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{field_name} is required")
    try:
        parse_ibkr_hours(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be parseable IBKR hours: {exc}") from exc


def _parse_ibkr_date(value: str) -> date:
    text = value.strip()
    try:
        return datetime.strptime(text, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"invalid IBKR date: {text}") from exc


def _parse_ibkr_time_endpoint(value: str) -> tuple[date | None, time]:
    text = value.strip()
    endpoint_date = None
    if ":" in text:
        date_text, time_text = text.split(":", 1)
        endpoint_date = _parse_ibkr_date(date_text)
        text = time_text
    if len(text) != 4 or not text.isdigit():
        raise ValueError(f"invalid IBKR time: {text}")
    hour = int(text[:2])
    minute = int(text[2:])
    try:
        endpoint_time = time(hour=hour, minute=minute)
    except ValueError as exc:
        raise ValueError(f"invalid IBKR time: {text}") from exc
    return endpoint_date, endpoint_time
