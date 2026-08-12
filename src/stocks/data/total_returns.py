from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from stocks.data.bars import BarCacheLayout, BarDataType, BarInterval
from stocks.data.corporate_actions import CorporateActionLayout
from stocks.data.fx import FxCacheLayout
from stocks.data.phase5_common import (
    PHASE5_CALCULATION_VERSION,
    decimal_from,
    read_parquet_records,
    sha256_file,
    sha256_json,
    utc_now_iso,
    zero_financial_calls,
)
from stocks.ibkr.contract_cache import ContractCacheRow
from stocks.research.instrument_manifest import InstrumentManifestLayout, validate_instrument_manifest


TOTAL_RETURN_FIELDS = (
    "timestamp_utc",
    "session_date",
    "con_id",
    "symbol",
    "instrument_currency",
    "base_currency",
    "raw_close",
    "split_adjusted_close",
    "cash_distribution",
    "local_price_return",
    "local_dividend_return",
    "local_total_return",
    "fx_rate",
    "fx_return",
    "eur_total_return",
    "local_total_return_index",
    "eur_total_return_index",
    "price_basis",
    "split_adjusted",
    "dividend_adjusted",
    "total_return",
    "currency_adjusted",
    "corporate_action_source",
    "fx_source",
    "corporate_action_hash",
    "fx_content_hash",
    "calculation_version",
    "content_hash",
)


@dataclass(frozen=True)
class TotalReturnLayout:
    data_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> TotalReturnLayout:
        return cls(data_dir=project_root / "data" / "total_returns")

    @property
    def manifest_json(self) -> Path:
        return self.data_dir / "total_return_manifest.json"

    def total_returns_path(self, *, con_id: int) -> Path:
        return (
            self.data_dir
            / "security_type=STK"
            / f"con_id={con_id}"
            / "interval=1d"
            / "total_returns.parquet"
        )


def total_return_schema() -> dict[str, Any]:
    return {
        "schema": "total_return_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "required_fields": list(TOTAL_RETURN_FIELDS),
        "price_basis": ["RAW", "SPLIT_ADJUSTED", "TOTAL_RETURN_LOCAL", "TOTAL_RETURN_EUR"],
        "calculation_version": PHASE5_CALCULATION_VERSION,
        "financial_calls": zero_financial_calls(),
    }


def build_total_returns_for_universe(
    *,
    project_root: Path,
    rows: list[ContractCacheRow],
    base_currency: str = "EUR",
) -> dict[str, Any]:
    if base_currency != "EUR":
        return _error("NO_GO_UNSUPPORTED_BASE_CURRENCY")
    manifest_validation = validate_instrument_manifest(InstrumentManifestLayout.from_project_root(project_root))
    if manifest_validation["status"] != "GO":
        return _error("NO_GO_INVALID_RESEARCH_MANIFEST")
    instruments = yaml.safe_load(
        InstrumentManifestLayout.from_project_root(project_root).path.read_text(encoding="utf-8")
    )["instruments"]
    by_symbol = {row.contract.symbol.upper(): row for row in rows}
    layout = TotalReturnLayout.from_project_root(project_root)
    bar_layout = BarCacheLayout.from_project_root(project_root)
    ca_layout = CorporateActionLayout.from_project_root(project_root)
    fx_layout = FxCacheLayout.from_project_root(project_root)
    all_events = read_parquet_records(ca_layout.corporate_actions_parquet)
    fx_records = read_parquet_records(fx_layout.fx_daily_parquet)
    fx_by_key = {(item["base_currency"], item["session_date"]): item for item in fx_records}
    ca_content_hash = sha256_file(ca_layout.corporate_actions_parquet)
    fx_file_hash = sha256_file(fx_layout.fx_daily_parquet)

    summaries: list[dict[str, Any]] = []
    for item in instruments:
        row = by_symbol[str(item["symbol"]).upper()]
        raw_path = bar_layout.phase4_bars_path(
            security_type=row.contract.security_type,
            con_id=row.contract.con_id,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
        )
        records = _build_one_instrument(
            row=row,
            raw_bar_records=read_parquet_records(raw_path),
            events=[event for event in all_events if int(event["con_id"]) == row.contract.con_id],
            fx_by_key=fx_by_key,
            base_currency=base_currency,
            ca_content_hash=ca_content_hash,
            fx_file_hash=fx_file_hash,
        )
        path = layout.total_returns_path(con_id=row.contract.con_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), path)
        summaries.append(
            {
                "instrument_id": item["instrument_id"],
                "symbol": row.contract.symbol,
                "con_id": row.contract.con_id,
                "row_count": len(records),
                "first_session": records[0]["session_date"] if records else None,
                "last_session": records[-1]["session_date"] if records else None,
                "content_hash": sha256_file(path),
                "path": str(path),
            }
        )

    validation = validate_total_return_cache(layout)
    manifest = {
        "schema": "total_return_manifest_v1",
        "status": "GO" if validation["status"] == "GO" else "NO_GO",
        "generated_at": utc_now_iso(),
        "base_currency": base_currency,
        "calculation_version": PHASE5_CALCULATION_VERSION,
        "price_basis": "TOTAL_RETURN_EUR",
        "split_adjusted": True,
        "dividend_adjusted": True,
        "total_return": True,
        "currency_adjusted": True,
        "corporate_action_source": "EODHD",
        "fx_source": "EODHD_FOREX",
        "instrument_summaries": summaries,
        "cache_validation": validation,
        "financial_calls": zero_financial_calls(),
    }
    layout.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    layout.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "schema": "total_return_build_v1",
        "status": manifest["status"],
        "instrument_count": len(summaries),
        "row_count": sum(item["row_count"] for item in summaries),
        "cache_validation": validation,
        "manifest": manifest,
        "financial_calls": zero_financial_calls(),
    }


def validate_total_return_cache(layout: TotalReturnLayout) -> dict[str, Any]:
    files = sorted(layout.data_dir.glob("security_type=STK/con_id=*/interval=1d/total_returns.parquet"))
    errors: list[str] = []
    summaries: list[dict[str, Any]] = []
    duplicate_rows = 0
    invalid_rows = 0
    missing_fx_rows = 0
    outlier_count = 0
    for path in files:
        records = read_parquet_records(path)
        seen_dates: set[str] = set()
        previous_date = None
        for index, record in enumerate(records):
            prefix = f"{path}[{index}]"
            missing = [field for field in TOTAL_RETURN_FIELDS if field not in record]
            if missing:
                errors.append(f"{prefix}: missing fields {', '.join(missing)}")
                invalid_rows += 1
                continue
            if record["session_date"] in seen_dates:
                duplicate_rows += 1
                errors.append(f"{prefix}: duplicate session_date")
            seen_dates.add(str(record["session_date"]))
            if previous_date is not None and str(record["session_date"]) < previous_date:
                errors.append(f"{prefix}: session_date values must be sorted")
            previous_date = str(record["session_date"])
            raw_close = decimal_from(record["raw_close"])
            if raw_close is None or raw_close <= 0:
                invalid_rows += 1
                errors.append(f"{prefix}: raw_close must be positive")
            fx_rate = decimal_from(record["fx_rate"])
            if fx_rate is None or fx_rate <= 0:
                missing_fx_rows += 1
                errors.append(f"{prefix}: missing or invalid fx_rate")
            if record["calculation_version"] != PHASE5_CALCULATION_VERSION:
                errors.append(f"{prefix}: invalid calculation_version")
            expected_hash = _tr_hash(record)
            if record["content_hash"] != expected_hash:
                errors.append(f"{prefix}: content_hash mismatch")
            if _is_return_outlier(record):
                outlier_count += 1
        summaries.append(
            {
                "path": str(path),
                "row_count": len(records),
                "first_session": records[0]["session_date"] if records else None,
                "last_session": records[-1]["session_date"] if records else None,
                "content_hash": sha256_file(path),
            }
        )
    return {
        "schema": "total_return_cache_validation_v1",
        "status": "GO" if not errors and files else "NO_GO",
        "file_count": len(files),
        "instrument_count": len(files),
        "row_count": sum(item["row_count"] for item in summaries),
        "duplicate_total_return_rows": duplicate_rows,
        "invalid_total_return_rows": invalid_rows,
        "missing_blocking_fx_rows": missing_fx_rows,
        "return_outlier_count": outlier_count,
        "unexplained_price_jump_count": outlier_count,
        "errors": errors,
        "files": summaries,
        "content_hash": sha256_json({item["path"]: item["content_hash"] for item in summaries}),
        "financial_calls": zero_financial_calls(),
    }


def total_return_status(layout: TotalReturnLayout) -> dict[str, Any]:
    validation = validate_total_return_cache(layout)
    return {
        "schema": "total_return_status_v1",
        "status": "READY" if layout.manifest_json.exists() and validation["status"] == "GO" else "NO_GO",
        "manifest_exists": layout.manifest_json.exists(),
        "cache_validation": validation,
        "financial_calls": zero_financial_calls(),
    }


def _build_one_instrument(
    *,
    row: ContractCacheRow,
    raw_bar_records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    fx_by_key: dict[tuple[str, str], dict[str, Any]],
    base_currency: str,
    ca_content_hash: str | None,
    fx_file_hash: str | None,
) -> list[dict[str, Any]]:
    raw = sorted(raw_bar_records, key=lambda item: item["session_date"])
    split_events = [
        event
        for event in events
        if event.get("event_type") in {"STOCK_SPLIT", "REVERSE_SPLIT"} and event.get("split_factor") is not None
    ]
    split_events_to_apply = _split_events_requiring_adjustment(raw, split_events)
    records: list[dict[str, Any]] = []
    previous_close: Decimal | None = None
    previous_fx: Decimal | None = None
    local_index = Decimal("100")
    eur_index = Decimal("100")
    for bar in raw:
        session_date = str(bar["session_date"])
        raw_close = decimal_from(bar["close"])
        if raw_close is None:
            continue
        split_factor = _cumulative_split_factor(date.fromisoformat(session_date), split_events_to_apply)
        split_adjusted_close = raw_close / split_factor
        fx_record = fx_by_key.get((row.contract.currency, session_date))
        if fx_record is None:
            continue
        fx_rate = decimal_from(fx_record["rate"])
        if fx_rate is None:
            continue
        cash_distribution = _cash_distribution_in_instrument_currency(
            events,
            session_date=session_date,
            instrument_currency=row.contract.currency,
            instrument_fx_rate=fx_rate,
            fx_by_key=fx_by_key,
        )
        if previous_close is None or previous_fx is None:
            local_price_return = Decimal("0")
            local_dividend_return = Decimal("0")
            local_total_return = Decimal("0")
            fx_return = Decimal("0")
            eur_total_return = Decimal("0")
        else:
            local_price_return = split_adjusted_close / previous_close - Decimal("1")
            local_dividend_return = cash_distribution / previous_close
            local_total_return = (split_adjusted_close + cash_distribution) / previous_close - Decimal("1")
            fx_return = fx_rate / previous_fx - Decimal("1")
            eur_total_return = (Decimal("1") + local_total_return) * (Decimal("1") + fx_return) - Decimal("1")
            local_index *= Decimal("1") + local_total_return
            eur_index *= Decimal("1") + eur_total_return
        record = {
            "timestamp_utc": bar["timestamp_utc"],
            "session_date": session_date,
            "con_id": row.contract.con_id,
            "symbol": row.contract.symbol,
            "instrument_currency": row.contract.currency,
            "base_currency": base_currency,
            "raw_close": _d(raw_close),
            "split_adjusted_close": _d(split_adjusted_close),
            "cash_distribution": _d(cash_distribution),
            "local_price_return": _d(local_price_return),
            "local_dividend_return": _d(local_dividend_return),
            "local_total_return": _d(local_total_return),
            "fx_rate": _d(fx_rate),
            "fx_return": _d(fx_return),
            "eur_total_return": _d(eur_total_return),
            "local_total_return_index": _d(local_index),
            "eur_total_return_index": _d(eur_index),
            "price_basis": "TOTAL_RETURN_EUR",
            "split_adjusted": True,
            "dividend_adjusted": True,
            "total_return": True,
            "currency_adjusted": True,
            "corporate_action_source": "EODHD",
            "fx_source": fx_record["source"],
            "corporate_action_hash": ca_content_hash,
            "fx_content_hash": fx_file_hash,
            "calculation_version": PHASE5_CALCULATION_VERSION,
        }
        record["content_hash"] = _tr_hash(record)
        records.append(record)
        previous_close = split_adjusted_close
        previous_fx = fx_rate
    return records


def _cumulative_split_factor(session_date: date, split_events: list[dict[str, Any]]) -> Decimal:
    factor = Decimal("1")
    for event in split_events:
        effective_date = date.fromisoformat(str(event["effective_date"]))
        if effective_date > session_date:
            split_factor = decimal_from(event["split_factor"])
            if split_factor is not None:
                factor *= split_factor
    return factor


def _cash_distribution_in_instrument_currency(
    events: list[dict[str, Any]],
    *,
    session_date: str,
    instrument_currency: str,
    instrument_fx_rate: Decimal,
    fx_by_key: dict[tuple[str, str], dict[str, Any]],
) -> Decimal:
    total = Decimal("0")
    for event in events:
        if event.get("event_type") not in {"CASH_DIVIDEND", "SPECIAL_DIVIDEND"}:
            continue
        if str(event.get("ex_date")) != session_date:
            continue
        amount = decimal_from(event.get("cash_amount"), default=Decimal("0")) or Decimal("0")
        event_currency = str(event.get("currency") or instrument_currency)
        if event_currency != instrument_currency:
            event_fx = fx_by_key.get((event_currency, session_date))
            event_fx_rate = decimal_from(None if event_fx is None else event_fx.get("rate"))
            if event_fx_rate is None:
                continue
            amount = amount * event_fx_rate / instrument_fx_rate
        total += amount
    return total


def _split_events_requiring_adjustment(
    raw_bar_records: list[dict[str, Any]],
    split_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_date = {str(item["session_date"]): item for item in raw_bar_records}
    sorted_dates = sorted(by_date)
    events_to_apply: list[dict[str, Any]] = []
    for event in split_events:
        effective_date = str(event["effective_date"])
        if effective_date not in by_date:
            events_to_apply.append(event)
            continue
        previous_dates = [item for item in sorted_dates if item < effective_date]
        if not previous_dates:
            events_to_apply.append(event)
            continue
        previous = by_date[previous_dates[-1]]
        current = by_date[effective_date]
        previous_close = decimal_from(previous.get("close"))
        current_close = decimal_from(current.get("close"))
        split_factor = decimal_from(event.get("split_factor"))
        if previous_close is None or current_close is None or split_factor is None or split_factor <= 0:
            events_to_apply.append(event)
            continue
        observed_ratio = current_close / previous_close
        unadjusted_ratio = Decimal("1") / split_factor
        if _near_ratio(observed_ratio, unadjusted_ratio):
            events_to_apply.append(event)
    return events_to_apply


def _near_ratio(observed: Decimal, expected: Decimal, *, tolerance: Decimal = Decimal("0.20")) -> bool:
    if expected <= 0:
        return False
    return abs(observed / expected - Decimal("1")) <= tolerance


def _tr_hash(record: dict[str, Any]) -> str:
    return sha256_json({field: record.get(field) for field in TOTAL_RETURN_FIELDS if field != "content_hash"})


def _is_return_outlier(record: dict[str, Any]) -> bool:
    value = float(decimal_from(record["local_price_return"]) or Decimal("0"))
    return abs(math.log1p(value)) > 0.35


def _d(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _error(status: str) -> dict[str, Any]:
    return {"schema": "total_return_build_v1", "status": status, "financial_calls": zero_financial_calls()}
