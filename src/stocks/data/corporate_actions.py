from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from stocks.data.phase5_common import (
    decimal_from,
    eodhd_api_token,
    eodhd_get_json,
    parse_date,
    read_parquet_records,
    sha256_file,
    sha256_json,
    utc_now_iso,
    zero_financial_calls,
)
from stocks.ibkr.contract_cache import ContractCacheRow
from stocks.research.instrument_manifest import InstrumentManifestLayout, validate_instrument_manifest


CORPORATE_ACTION_FIELDS = (
    "event_id",
    "con_id",
    "symbol",
    "event_type",
    "ex_date",
    "record_date",
    "payment_date",
    "effective_date",
    "currency",
    "cash_amount",
    "split_from",
    "split_to",
    "split_factor",
    "distribution_type",
    "gross_amount",
    "withholding_assumption",
    "net_amount",
    "reinvestment_assumption",
    "source",
    "source_event_id",
    "announced_at",
    "available_at",
    "downloaded_at",
    "is_estimated",
    "is_cancelled",
    "provider_count",
    "matching_provider_count",
    "amount_difference",
    "date_difference",
    "classification_difference",
    "resolution_status",
    "event_hash",
)

SUPPORTED_EVENT_TYPES = (
    "CASH_DIVIDEND",
    "SPECIAL_DIVIDEND",
    "STOCK_SPLIT",
    "REVERSE_SPLIT",
    "STOCK_DIVIDEND",
    "SPINOFF",
    "RIGHTS_ISSUE",
    "MERGER",
    "TICKER_CHANGE",
    "DELISTING",
    "UNKNOWN_BLOCKED",
)


@dataclass(frozen=True)
class CorporateActionLayout:
    data_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> CorporateActionLayout:
        return cls(data_dir=project_root / "data" / "corporate_actions")

    @property
    def corporate_actions_parquet(self) -> Path:
        return self.data_dir / "corporate_actions.parquet"

    @property
    def dividends_parquet(self) -> Path:
        return self.data_dir / "dividends.parquet"

    @property
    def splits_parquet(self) -> Path:
        return self.data_dir / "splits.parquet"

    @property
    def manifest_json(self) -> Path:
        return self.data_dir / "event_manifest.json"


def corporate_action_schema() -> dict[str, Any]:
    return {
        "schema": "corporate_action_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "required_fields": list(CORPORATE_ACTION_FIELDS),
        "supported_event_types": list(SUPPORTED_EVENT_TYPES),
        "unknown_event_policy": "UNKNOWN_BLOCKED",
        "financial_calls": zero_financial_calls(),
    }


def collect_corporate_actions_for_universe(
    *,
    project_root: Path,
    rows: list[ContractCacheRow],
    start: date,
    end: date,
    env_file: str | Path = ".env",
) -> dict[str, Any]:
    token = eodhd_api_token(project_root / env_file)
    if not token:
        return _provider_error("NO_GO_MISSING_EODHD_API_KEY")
    manifest_validation = validate_instrument_manifest(InstrumentManifestLayout.from_project_root(project_root))
    if manifest_validation["status"] != "GO":
        return _provider_error("NO_GO_INVALID_RESEARCH_MANIFEST", manifest_validation=manifest_validation)

    manifest_payload = yaml.safe_load(
        InstrumentManifestLayout.from_project_root(project_root).path.read_text(encoding="utf-8")
    )
    instruments = list(manifest_payload["instruments"])
    by_symbol = {row.contract.symbol.upper(): row for row in rows}
    records: list[dict[str, Any]] = []
    provider_errors: list[dict[str, Any]] = []
    downloaded_at = utc_now_iso()
    for item in instruments:
        row = by_symbol.get(str(item["symbol"]).upper())
        if row is None:
            provider_errors.append({"symbol": item["symbol"], "status": "MISSING_CONTRACT_CACHE"})
            continue
        ticker = eodhd_ticker(row.contract.symbol)
        try:
            dividends = eodhd_get_json(
                f"div/{ticker}",
                token=token,
                params={"from": start.isoformat(), "to": end.isoformat()},
            )
            splits = eodhd_get_json(
                f"splits/{ticker}",
                token=token,
                params={"from": start.isoformat(), "to": end.isoformat()},
            )
        except Exception as exc:
            provider_errors.append({"symbol": row.contract.symbol, "status": "PROVIDER_ERROR", "error": str(exc)})
            continue
        records.extend(
            _dividend_record(row, payload, downloaded_at=downloaded_at)
            for payload in _ensure_list(dividends)
        )
        records.extend(
            _split_record(row, payload, downloaded_at=downloaded_at)
            for payload in _ensure_list(splits)
        )

    records = _dedupe_events(records)
    layout = CorporateActionLayout.from_project_root(project_root)
    _write_outputs(layout, records)
    validation = validate_corporate_action_cache(layout)
    manifest = _manifest(layout, records, validation, provider_errors)
    layout.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "schema": "corporate_action_collection_v1",
        "status": "GO" if validation["status"] == "GO" and not provider_errors else "NO_GO",
        "event_count": len(records),
        "provider_errors": provider_errors,
        "cache_validation": validation,
        "manifest": manifest,
        "financial_calls": zero_financial_calls(),
    }


def validate_corporate_action_cache(layout: CorporateActionLayout) -> dict[str, Any]:
    records = read_parquet_records(layout.corporate_actions_parquet)
    errors: list[str] = []
    event_hashes: set[str] = set()
    unresolved = 0
    provider_conflicts = 0
    for index, record in enumerate(records):
        prefix = f"corporate_actions[{index}]"
        missing = [field for field in CORPORATE_ACTION_FIELDS if field not in record]
        if missing:
            errors.append(f"{prefix}: missing fields {', '.join(missing)}")
            continue
        if record["event_type"] not in SUPPORTED_EVENT_TYPES:
            errors.append(f"{prefix}: unsupported event_type {record['event_type']}")
        if record["event_type"] == "UNKNOWN_BLOCKED":
            unresolved += 1
        if record["resolution_status"] not in {
            "MATCHED",
            "MATCHED_WITH_TOLERANCE",
            "DATE_CONFLICT",
            "AMOUNT_CONFLICT",
            "TYPE_CONFLICT",
            "SINGLE_SOURCE",
            "UNRESOLVED_BLOCKED",
        }:
            errors.append(f"{prefix}: invalid resolution_status {record['resolution_status']}")
        if str(record["resolution_status"]).endswith("CONFLICT"):
            provider_conflicts += 1
        expected_hash = _event_hash(record)
        if record["event_hash"] != expected_hash:
            errors.append(f"{prefix}: event_hash mismatch")
        if record["event_hash"] in event_hashes:
            errors.append(f"{prefix}: duplicate event_hash {record['event_hash']}")
        event_hashes.add(record["event_hash"])
    return {
        "schema": "corporate_action_cache_validation_v1",
        "status": "GO" if not errors and unresolved == 0 else "NO_GO",
        "event_count": len(records),
        "split_event_count": sum(1 for item in records if item.get("event_type") in {"STOCK_SPLIT", "REVERSE_SPLIT"}),
        "dividend_event_count": sum(1 for item in records if item.get("event_type") in {"CASH_DIVIDEND", "SPECIAL_DIVIDEND"}),
        "special_event_count": sum(1 for item in records if item.get("event_type") not in {"CASH_DIVIDEND", "STOCK_SPLIT", "REVERSE_SPLIT"}),
        "provider_conflict_count": provider_conflicts,
        "unresolved_event_count": unresolved,
        "errors": errors,
        "content_hash": sha256_file(layout.corporate_actions_parquet),
        "financial_calls": zero_financial_calls(),
    }


def corporate_action_status(layout: CorporateActionLayout) -> dict[str, Any]:
    validation = validate_corporate_action_cache(layout)
    manifest_exists = layout.manifest_json.exists()
    return {
        "schema": "corporate_action_status_v1",
        "status": "READY" if manifest_exists and validation["status"] == "GO" else "NO_GO",
        "manifest_exists": manifest_exists,
        "cache_validation": validation,
        "financial_calls": zero_financial_calls(),
    }


def eodhd_ticker(symbol: str) -> str:
    return f"{symbol.upper()}.US"


def _dividend_record(row: ContractCacheRow, payload: dict[str, Any], *, downloaded_at: str) -> dict[str, Any]:
    ex_date = parse_date(payload.get("date"))
    amount = decimal_from(payload.get("value"), default=Decimal("0"))
    currency = str(payload.get("currency") or row.contract.currency).upper()
    distribution_type = _distribution_type(payload)
    event_type = "SPECIAL_DIVIDEND" if distribution_type == "special distribution" else "CASH_DIVIDEND"
    record = {
        "event_id": "",
        "con_id": row.contract.con_id,
        "symbol": row.contract.symbol,
        "event_type": event_type,
        "ex_date": _date_text(ex_date),
        "record_date": _date_text(parse_date(payload.get("recordDate"))),
        "payment_date": _date_text(parse_date(payload.get("paymentDate"))),
        "effective_date": _date_text(ex_date),
        "currency": currency,
        "cash_amount": _decimal_text(amount),
        "split_from": None,
        "split_to": None,
        "split_factor": None,
        "distribution_type": distribution_type,
        "gross_amount": _decimal_text(amount),
        "withholding_assumption": "GROSS_TOTAL_RETURN_TAX_NOT_PROCESSED",
        "net_amount": _decimal_text(amount),
        "reinvestment_assumption": "CASH_DISTRIBUTION_REINVESTED_IN_TOTAL_RETURN_INDEX",
        "source": "EODHD_DIVIDENDS",
        "source_event_id": f"EODHD_DIVIDENDS:{row.contract.symbol}:{_date_text(ex_date)}:{payload.get('value')}",
        "announced_at": _date_text(parse_date(payload.get("declarationDate"))),
        "available_at": _available_at(ex_date),
        "downloaded_at": downloaded_at,
        "is_estimated": False,
        "is_cancelled": False,
        "provider_count": 1,
        "matching_provider_count": 1,
        "amount_difference": "0",
        "date_difference": 0,
        "classification_difference": False,
        "resolution_status": "SINGLE_SOURCE",
    }
    record["event_id"] = _event_hash({**record, "event_id": "", "event_hash": ""})
    record["event_hash"] = _event_hash(record)
    return record


def _split_record(row: ContractCacheRow, payload: dict[str, Any], *, downloaded_at: str) -> dict[str, Any]:
    effective_date = parse_date(payload.get("date"))
    split_from, split_to, split_factor = _parse_split_ratio(payload)
    event_type = "REVERSE_SPLIT" if split_factor is not None and split_factor < Decimal("1") else "STOCK_SPLIT"
    record = {
        "event_id": "",
        "con_id": row.contract.con_id,
        "symbol": row.contract.symbol,
        "event_type": event_type,
        "ex_date": _date_text(effective_date),
        "record_date": None,
        "payment_date": None,
        "effective_date": _date_text(effective_date),
        "currency": row.contract.currency,
        "cash_amount": None,
        "split_from": _decimal_text(split_from),
        "split_to": _decimal_text(split_to),
        "split_factor": _decimal_text(split_factor),
        "distribution_type": None,
        "gross_amount": None,
        "withholding_assumption": None,
        "net_amount": None,
        "reinvestment_assumption": None,
        "source": "EODHD_SPLITS",
        "source_event_id": f"EODHD_SPLITS:{row.contract.symbol}:{_date_text(effective_date)}:{payload.get('split') or payload}",
        "announced_at": None,
        "available_at": _available_at(effective_date),
        "downloaded_at": downloaded_at,
        "is_estimated": False,
        "is_cancelled": False,
        "provider_count": 1,
        "matching_provider_count": 1,
        "amount_difference": "0",
        "date_difference": 0,
        "classification_difference": False,
        "resolution_status": "SINGLE_SOURCE",
    }
    if split_factor is None:
        record["event_type"] = "UNKNOWN_BLOCKED"
        record["resolution_status"] = "UNRESOLVED_BLOCKED"
    record["event_id"] = _event_hash({**record, "event_id": "", "event_hash": ""})
    record["event_hash"] = _event_hash(record)
    return record


def _parse_split_ratio(payload: dict[str, Any]) -> tuple[Decimal | None, Decimal | None, Decimal | None]:
    raw = payload.get("split") or payload.get("ratio")
    if raw is None:
        return None, None, None
    text = str(raw).strip().replace(":", "/")
    if "/" not in text:
        return None, None, None
    left, right = text.split("/", 1)
    try:
        split_to = Decimal(left)
        split_from = Decimal(right)
    except (InvalidOperation, ValueError):
        return None, None, None
    if split_from <= 0 or split_to <= 0:
        return None, None, None
    return split_from, split_to, split_to / split_from


def _distribution_type(payload: dict[str, Any]) -> str:
    period = str(payload.get("period") or "").strip().lower()
    if "special" in period:
        return "special distribution"
    if "return" in period and "capital" in period:
        return "return of capital"
    if "capital" in period:
        return "capital-gains distribution"
    return "ordinary income"


def _write_outputs(layout: CorporateActionLayout, records: list[dict[str, Any]]) -> None:
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    _write_table(layout.corporate_actions_parquet, records)
    _write_table(layout.dividends_parquet, [item for item in records if item["event_type"] in {"CASH_DIVIDEND", "SPECIAL_DIVIDEND"}])
    _write_table(layout.splits_parquet, [item for item in records if item["event_type"] in {"STOCK_SPLIT", "REVERSE_SPLIT"}])


def _write_table(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        records = [{field: None for field in CORPORATE_ACTION_FIELDS}]
        table = pa.Table.from_pylist(records).slice(0, 0)
    else:
        table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def _manifest(
    layout: CorporateActionLayout,
    records: list[dict[str, Any]],
    validation: dict[str, Any],
    provider_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "corporate_action_event_manifest_v1",
        "status": "GO" if validation["status"] == "GO" and not provider_errors else "NO_GO",
        "generated_at": utc_now_iso(),
        "source": "EODHD",
        "event_count": len(records),
        "provider_errors": provider_errors,
        "files": {
            "corporate_actions": str(layout.corporate_actions_parquet),
            "dividends": str(layout.dividends_parquet),
            "splits": str(layout.splits_parquet),
        },
        "content_hash": validation["content_hash"],
        "financial_calls": zero_financial_calls(),
    }


def _event_hash(record: dict[str, Any]) -> str:
    payload = {key: record.get(key) for key in CORPORATE_ACTION_FIELDS if key not in {"event_hash"}}
    return sha256_json(payload)


def _dedupe_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = {record["event_hash"]: record for record in records}
    return sorted(deduped.values(), key=lambda item: (int(item["con_id"]), str(item["effective_date"]), str(item["event_type"])))


def _ensure_list(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and payload.get("error"):
        raise ValueError(str(payload["error"]))
    return []


def _date_text(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def _available_at(value: date | None) -> str | None:
    if value is None:
        return None
    return (value + timedelta(days=1)).isoformat() + "T00:00:00+00:00"


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value.normalize(), "f")


def _provider_error(status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": "corporate_action_collection_v1",
        "status": status,
        **extra,
        "financial_calls": zero_financial_calls(),
    }
