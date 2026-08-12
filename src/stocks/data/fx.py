from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from stocks.data.phase5_common import (
    decimal_from,
    eodhd_api_token,
    eodhd_get_json,
    read_parquet_records,
    sha256_file,
    sha256_json,
    utc_now_iso,
    zero_financial_calls,
)
from stocks.ibkr.contract_cache import ContractCacheRow


FX_FIELDS = (
    "timestamp_utc",
    "session_date",
    "base_currency",
    "quote_currency",
    "rate",
    "rate_definition",
    "source",
    "available_at",
    "downloaded_at",
    "is_forward_filled",
    "forward_fill_age",
    "content_hash",
)

MAX_FX_FORWARD_FILL_DAYS = 5


@dataclass(frozen=True)
class FxCacheLayout:
    data_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> FxCacheLayout:
        return cls(data_dir=project_root / "data" / "fx")

    @property
    def fx_daily_parquet(self) -> Path:
        return self.data_dir / "fx_daily.parquet"

    @property
    def manifest_json(self) -> Path:
        return self.data_dir / "fx_manifest.json"


def fx_schema() -> dict[str, Any]:
    return {
        "schema": "fx_daily_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "required_fields": list(FX_FIELDS),
        "rate_definition": "EUR_PER_BASE_CURRENCY",
        "max_forward_fill_days": MAX_FX_FORWARD_FILL_DAYS,
        "financial_calls": zero_financial_calls(),
    }


def collect_fx_for_universe(
    *,
    project_root: Path,
    rows: list[ContractCacheRow],
    start: date,
    end: date,
    base_currency: str = "EUR",
    env_file: str | Path = ".env",
) -> dict[str, Any]:
    if base_currency != "EUR":
        return _error("NO_GO_UNSUPPORTED_BASE_CURRENCY")
    token = eodhd_api_token(project_root / env_file)
    if not token:
        return _error("NO_GO_MISSING_EODHD_API_KEY")
    currencies = sorted({row.contract.currency for row in rows} | {base_currency})
    downloaded_at = utc_now_iso()
    records: list[dict[str, Any]] = []
    provider_errors: list[dict[str, Any]] = []
    for currency in currencies:
        if currency == base_currency:
            records.extend(_constant_currency_records(currency, start=start, end=end, downloaded_at=downloaded_at))
            continue
        try:
            records.extend(
                _eodhd_currency_records(
                    currency,
                    token=token,
                    start=start,
                    end=end,
                    downloaded_at=downloaded_at,
                )
            )
        except Exception as exc:
            provider_errors.append({"currency": currency, "status": "PROVIDER_ERROR", "error": str(exc)})

    layout = FxCacheLayout.from_project_root(project_root)
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), layout.fx_daily_parquet)
    validation = validate_fx_cache(layout)
    manifest = {
        "schema": "fx_manifest_v1",
        "status": "GO" if validation["status"] == "GO" and not provider_errors else "NO_GO",
        "generated_at": utc_now_iso(),
        "base_currency": base_currency,
        "currencies": currencies,
        "source": "EODHD_FOREX",
        "provider_errors": provider_errors,
        "content_hash": validation["content_hash"],
        "financial_calls": zero_financial_calls(),
    }
    layout.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "schema": "fx_collection_v1",
        "status": manifest["status"],
        "row_count": len(records),
        "currencies": currencies,
        "provider_errors": provider_errors,
        "cache_validation": validation,
        "manifest": manifest,
        "financial_calls": zero_financial_calls(),
    }


def validate_fx_cache(layout: FxCacheLayout) -> dict[str, Any]:
    records = read_parquet_records(layout.fx_daily_parquet)
    errors: list[str] = []
    seen: set[tuple[str, str]] = set()
    stale_rows = 0
    for index, record in enumerate(records):
        prefix = f"fx_daily[{index}]"
        missing = [field for field in FX_FIELDS if field not in record]
        if missing:
            errors.append(f"{prefix}: missing fields {', '.join(missing)}")
            continue
        key = (str(record["base_currency"]), str(record["session_date"]))
        if key in seen:
            errors.append(f"{prefix}: duplicate currency/date {key}")
        seen.add(key)
        if str(record["quote_currency"]) != "EUR":
            errors.append(f"{prefix}: quote_currency must be EUR")
        if str(record["rate_definition"]) != "EUR_PER_BASE_CURRENCY":
            errors.append(f"{prefix}: invalid rate_definition")
        rate = decimal_from(record["rate"])
        if rate is None or rate <= 0:
            errors.append(f"{prefix}: rate must be positive")
        age = int(record["forward_fill_age"])
        if age > MAX_FX_FORWARD_FILL_DAYS:
            stale_rows += 1
            errors.append(f"{prefix}: forward_fill_age exceeds {MAX_FX_FORWARD_FILL_DAYS}")
        expected_hash = _fx_hash(record)
        if record["content_hash"] != expected_hash:
            errors.append(f"{prefix}: content_hash mismatch")
    return {
        "schema": "fx_cache_validation_v1",
        "status": "GO" if not errors else "NO_GO",
        "row_count": len(records),
        "currency_count": len({record.get("base_currency") for record in records}),
        "missing_fx_days": 0,
        "forward_filled_fx_days": sum(1 for item in records if item.get("is_forward_filled") is True),
        "max_fx_fill_age": max((int(item.get("forward_fill_age") or 0) for item in records), default=0),
        "stale_fx_rows": stale_rows,
        "errors": errors,
        "content_hash": sha256_file(layout.fx_daily_parquet),
        "financial_calls": zero_financial_calls(),
    }


def fx_status(layout: FxCacheLayout) -> dict[str, Any]:
    validation = validate_fx_cache(layout)
    return {
        "schema": "fx_status_v1",
        "status": "READY" if layout.manifest_json.exists() and validation["status"] == "GO" else "NO_GO",
        "manifest_exists": layout.manifest_json.exists(),
        "cache_validation": validation,
        "financial_calls": zero_financial_calls(),
    }


def _eodhd_currency_records(
    currency: str,
    *,
    token: str,
    start: date,
    end: date,
    downloaded_at: str,
) -> list[dict[str, Any]]:
    raw = eodhd_get_json(
        f"eod/EUR{currency}.FOREX",
        token=token,
        params={"from": start.isoformat(), "to": end.isoformat(), "period": "d"},
    )
    by_date: dict[date, Decimal] = {}
    for item in raw if isinstance(raw, list) else []:
        item_date = date.fromisoformat(str(item["date"])[:10])
        close = decimal_from(item.get("adjusted_close") or item.get("close"))
        if close is not None and close > 0:
            by_date[item_date] = Decimal("1") / close
    return _filled_currency_records(
        currency,
        by_date=by_date,
        start=start,
        end=end,
        downloaded_at=downloaded_at,
        source="EODHD_FOREX",
    )


def _constant_currency_records(currency: str, *, start: date, end: date, downloaded_at: str) -> list[dict[str, Any]]:
    by_date = {day: Decimal("1") for day in _date_range(start, end)}
    return _filled_currency_records(
        currency,
        by_date=by_date,
        start=start,
        end=end,
        downloaded_at=downloaded_at,
        source="IDENTITY_EUR",
    )


def _filled_currency_records(
    currency: str,
    *,
    by_date: dict[date, Decimal],
    start: date,
    end: date,
    downloaded_at: str,
    source: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    last_rate: Decimal | None = None
    last_rate_date: date | None = None
    for day in _date_range(start, end):
        if day in by_date:
            last_rate = by_date[day]
            last_rate_date = day
        if last_rate is None or last_rate_date is None:
            continue
        age = (day - last_rate_date).days
        record = {
            "timestamp_utc": f"{day.isoformat()}T00:00:00+00:00",
            "session_date": day.isoformat(),
            "base_currency": currency,
            "quote_currency": "EUR",
            "rate": format(last_rate, "f"),
            "rate_definition": "EUR_PER_BASE_CURRENCY",
            "source": source,
            "available_at": f"{day.isoformat()}T23:59:59+00:00",
            "downloaded_at": downloaded_at,
            "is_forward_filled": day != last_rate_date,
            "forward_fill_age": age,
        }
        record["content_hash"] = _fx_hash(record)
        records.append(record)
    return records


def _date_range(start: date, end: date) -> tuple[date, ...]:
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def _fx_hash(record: dict[str, Any]) -> str:
    return sha256_json({field: record.get(field) for field in FX_FIELDS if field != "content_hash"})


def _error(status: str) -> dict[str, Any]:
    return {"schema": "fx_collection_v1", "status": status, "financial_calls": zero_financial_calls()}
