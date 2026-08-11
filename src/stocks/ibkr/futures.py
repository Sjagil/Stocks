from __future__ import annotations

import re
from datetime import datetime


def validate_future_contract_month(value: str | None, *, field_name: str) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{field_name} is required")
    text = value.strip()
    if not re.fullmatch(r"\d{6}(\d{2})?", text):
        raise ValueError(f"{field_name} must use IBKR YYYYMM or YYYYMMDD format")
    if len(text) == 6:
        _parse_year_month(text, field_name)
    else:
        _parse_yyyymmdd(text, field_name)


def validate_future_expiration_date(value: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError("real_expiration_date is required")
    _parse_yyyymmdd(value.strip(), "real_expiration_date")


def validate_future_last_trade_time(value: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError("last_trade_time is required")
    text = value.strip()
    for time_format in ("%H:%M:%S", "%H:%M"):
        try:
            datetime.strptime(text, time_format).time()
            return
        except ValueError:
            continue
    raise ValueError("last_trade_time must use HH:MM or HH:MM:SS format")


def _parse_year_month(value: str, field_name: str) -> None:
    year = int(value[:4])
    month = int(value[4:6])
    if year < 1900 or month < 1 or month > 12:
        raise ValueError(f"{field_name} contains an invalid year/month")


def _parse_yyyymmdd(value: str, field_name: str) -> None:
    try:
        datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"{field_name} contains an invalid YYYYMMDD date") from exc
