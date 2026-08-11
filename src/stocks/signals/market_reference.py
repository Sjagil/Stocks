from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.data.multitimeframe import bar_freshness


MARKET_REFERENCE_MAX_AGE_MINUTES = 95.0
POSITIVE_SIGNAL_ACTIONS = {"BUY", "STRONG_BUY", "WATCHLIST"}


def latest_market_reference(
    project_root: Path,
    symbol: str,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    candidates: list[dict[str, Any]] = []
    for provider in ("YFINANCE", "EODHD"):
        path = (
            project_root
            / "data/research/multitimeframe/private"
            / f"provider={provider}"
            / f"symbol={symbol.upper()}"
            / "interval=1h"
            / "source_interval=1h"
            / "bars.parquet"
        )
        candidate = _read_latest(path, now=now)
        if candidate:
            candidates.append(candidate)
    if not candidates:
        return {
            "status": "UNAVAILABLE",
            "symbol": symbol.upper(),
            "reason": "QUALIFIED_INTRADAY_REFERENCE_UNAVAILABLE",
        }
    return max(candidates, key=lambda row: row["fetched_timestamp"])


def apply_market_reference(
    project_root: Path,
    signal: dict[str, Any],
    *,
    now: datetime | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    payload = dict(signal)
    symbol = str(payload.get("ticker") or payload.get("asset") or "").upper()
    reference = reference or latest_market_reference(
        project_root,
        symbol,
        now=now,
    )
    payload["market_reference_status"] = reference["status"]
    payload["market_reference_price"] = reference.get("price")
    payload["market_reference_timestamp"] = reference.get("timestamp")
    payload["market_reference_fetched_at"] = reference.get("fetched_at")
    payload["market_reference_provider"] = reference.get("provider")
    payload["market_reference_age_minutes"] = reference.get("age_minutes")
    payload["market_reference_kind"] = "INDICATIVE_INTRADAY_BAR_CLOSE"
    payload["market_reference_is_executable_quote"] = False

    action = str(payload.get("action") or "").upper()
    if action not in POSITIVE_SIGNAL_ACTIONS:
        return payload
    if reference["status"] != "FRESH":
        return _invalidate(
            payload,
            "CURRENT_MARKET_REFERENCE_UNAVAILABLE_OR_STALE",
        )

    price = float(reference["price"])
    entry = _number(
        payload.get("preferred_entry"),
        payload.get("limit_entry_price"),
        payload.get("current_market_price"),
    )
    stop = _number(payload.get("stop_loss"), payload.get("invalidation_level"))
    target = _number(payload.get("take_profit_1"))
    entry_low = _number(payload.get("entry_zone_low"), entry)
    entry_high = _number(payload.get("entry_zone_high"), entry)
    payload["current_market_price"] = _price(price)
    if entry is None or entry <= 0:
        return _invalidate(payload, "ENTRY_REFERENCE_UNAVAILABLE")

    deviation = price / entry - 1.0
    payload["entry_reference_deviation_pct"] = round(deviation, 8)
    stop_distance = abs(entry - stop) / entry if stop is not None else 0.0
    tolerance = max(0.01, min(0.03, stop_distance * 0.5))
    payload["entry_reference_tolerance_pct"] = round(tolerance, 8)
    if stop is not None and price <= stop:
        return _invalidate(payload, "CURRENT_PRICE_BREACHED_SIGNAL_STOP")
    if target is not None and price >= target:
        return _invalidate(payload, "CURRENT_PRICE_REACHED_FIRST_TARGET")
    if entry_low is not None and price < entry_low * (1.0 - tolerance):
        return _invalidate(payload, "CURRENT_PRICE_BELOW_ENTRY_ZONE")
    if entry_high is not None and price > entry_high * (1.0 + tolerance):
        return _invalidate(payload, "CURRENT_PRICE_ABOVE_ENTRY_ZONE")
    payload["price_validity_status"] = "CURRENT_ENTRY_REFERENCE_GO"
    payload["entry_instruction"] = "USE_PUBLISHED_CONDITIONAL_ENTRY_ZONE"
    return payload


def _read_latest(path: Path, *, now: datetime) -> dict[str, Any] | None:
    try:
        frame = pd.read_parquet(path)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if frame.empty or not {"timestamp_utc", "close"}.issubset(frame.columns):
        return None
    frame = frame.sort_values("timestamp_utc")
    row = frame.iloc[-1]
    price = _number(row.get("close"))
    fetched = _timestamp(row.get("fetched_at") or row.get("received_at"))
    timestamp = _timestamp(row.get("timestamp_utc"))
    if price is None or price <= 0 or fetched is None or timestamp is None:
        return None
    fetched_age_minutes = max(
        0.0,
        (now - fetched).total_seconds() / 60.0,
    )
    exchange_timezone = str(
        row.get("exchange_timezone") or ""
    ).strip()
    if exchange_timezone:
        freshness = bar_freshness(
            timestamp,
            interval="1h",
            observed_at=now,
            exchange_timezone=exchange_timezone,
        )
        status = (
            "FRESH"
            if freshness["status"] == "FRESH_CLOSED_BAR"
            else "STALE"
        )
        age_minutes = max(
            0.0,
            (now - timestamp).total_seconds() / 60.0,
        )
    else:
        freshness = {
            "freshness_basis": "FETCH_WALL_CLOCK",
        }
        age_minutes = fetched_age_minutes
        status = (
            "FRESH"
            if age_minutes <= MARKET_REFERENCE_MAX_AGE_MINUTES
            else "STALE"
        )
    return {
        "status": status,
        "symbol": str(row.get("symbol") or "").upper(),
        "provider": str(row.get("provider") or path.parts[-5]),
        "price": _price(price),
        "timestamp": timestamp.isoformat(),
        "fetched_at": fetched.isoformat(),
        "fetched_timestamp": fetched,
        "age_minutes": round(age_minutes, 4),
        "fetched_age_minutes": round(fetched_age_minutes, 4),
        "exchange_timezone": exchange_timezone,
        "freshness_basis": freshness.get("freshness_basis"),
        "quality_status": str(row.get("quality_status") or "UNKNOWN"),
        "is_partial": bool(row.get("is_partial", False)),
    }


def _invalidate(payload: dict[str, Any], reason: str) -> dict[str, Any]:
    risks = list(payload.get("risks", []))
    reasons = list(payload.get("reasons", []))
    if reason not in risks:
        risks.append(reason)
    if "CURRENT_MARKET_REFERENCE_INVALIDATES_ENTRY" not in reasons:
        reasons.append("CURRENT_MARKET_REFERENCE_INVALIDATES_ENTRY")
    payload.update(
        {
            "original_action": payload.get("action"),
            "action": "AVOID",
            "data_freshness": "STALE",
            "lifecycle_status": "INVALIDATED",
            "price_validity_status": reason,
            "entry_instruction": "WAIT_FOR_NEW_CAUSAL_SIGNAL",
            "risks": risks,
            "reasons": reasons,
            "automatic_execution_allowed": False,
        }
    )
    return payload


def _timestamp(value: Any) -> datetime | None:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            result = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(result):
            return result
    return None


def _price(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.0001"))
