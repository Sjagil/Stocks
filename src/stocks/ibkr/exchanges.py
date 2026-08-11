from __future__ import annotations

import re


_EXCHANGE_CODE_PATTERN = re.compile(r"^[A-Z0-9._-]+$")


def parse_valid_exchanges(value: str) -> tuple[str, ...]:
    text = value.strip()
    if not text:
        raise ValueError("validExchanges string is blank")

    exchanges: list[str] = []
    for raw_part in text.split(","):
        exchange = raw_part.strip().upper()
        if not exchange:
            raise ValueError("validExchanges contains an empty value")
        if not _EXCHANGE_CODE_PATTERN.fullmatch(exchange):
            raise ValueError(f"validExchanges contains an invalid exchange code: {exchange}")
        exchanges.append(exchange)

    return tuple(exchanges)


def validate_valid_exchanges(value: str | None, *, primary_exchange: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError("valid_exchanges is required")
    try:
        exchanges = parse_valid_exchanges(value)
    except ValueError as exc:
        raise ValueError(f"valid_exchanges must be comma-separated exchange codes: {exc}") from exc

    if primary_exchange is None or not primary_exchange.strip():
        raise ValueError("primary_exchange is required before validating valid_exchanges")
    if primary_exchange.strip().upper() not in exchanges:
        raise ValueError("primary_exchange must be present in valid_exchanges")
