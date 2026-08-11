from __future__ import annotations

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def parse_ibkr_timezone_id(value: str) -> ZoneInfo:
    text = value.strip()
    if not text:
        raise ValueError("timeZoneId string is blank")
    try:
        return ZoneInfo(text)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"unknown timeZoneId: {text}") from exc


def validate_ibkr_timezone_id(value: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError("time_zone_id is required")
    try:
        parse_ibkr_timezone_id(value)
    except ValueError as exc:
        raise ValueError(f"time_zone_id must be a known IANA/IBKR timezone: {exc}") from exc
