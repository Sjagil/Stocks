from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from stocks.execution.idempotency import stable_hash


def decimal_or_none(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def decimal_value(value: Any, default: str = "0") -> Decimal:
    parsed = decimal_or_none(value)
    return Decimal(default) if parsed is None else parsed


def safe_attr(obj: Any, *names: str, default: Any = "") -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def contract_hash(*, con_id: int, symbol: str, security_type: str, currency: str, exchange: str) -> str:
    return stable_hash(
        {
            "con_id": con_id,
            "symbol": symbol,
            "security_type": security_type,
            "currency": currency,
            "exchange": exchange,
        }
    )
