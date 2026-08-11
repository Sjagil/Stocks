from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from stocks.domain.assets import IbkrSecurityType
from stocks.ibkr.contracts import ContractCandidateEvaluation
from stocks.ibkr.contracts import ContractResolutionRequest
from stocks.ibkr.contracts import ContractResolutionStatus
from stocks.ibkr.contracts import ResolvedContract
from stocks.ibkr.contracts import contract_request_hash


_AUDIT_IBKR_CODE_PATTERN = re.compile(r"^[A-Z0-9._-]+$")


STOCK_CONTRACT_FIELDS = (
    "conId",
    "symbol",
    "localSymbol",
    "secType",
    "currency",
    "exchange",
    "primaryExchange",
    "tradingClass",
    "minTick",
    "validExchanges",
    "timeZoneId",
    "tradingHours",
    "liquidHours",
    "marketRuleIds",
    "longName",
)

FUTURE_CONTRACT_FIELDS = (
    "conId",
    "symbol",
    "localSymbol",
    "secType",
    "currency",
    "exchange",
    "lastTradeDateOrContractMonth",
    "realExpirationDate",
    "lastTradeTime",
    "multiplier",
    "minTick",
    "tradingClass",
    "underConId",
    "timeZoneId",
    "tradingHours",
    "liquidHours",
    "marketRuleIds",
)

FUTURE_REFERENCE_METADATA_FIELDS = (
    "firstNoticeDay",
    "lastTradeDay",
    "deliveryType",
    "settlementType",
    "contractSizeUnit",
    "rollGroup",
)

RESOLUTION_AUDIT_FIELDS = (
    "requested_symbol",
    "requested_security_type",
    "requested_currency",
    "requested_exchange",
    "requested_primary_exchange",
    "returned_match_count",
    "selected_conId",
    "resolution_status",
    "resolved_at",
    "server_version",
    "request_hash",
    "contract_hash",
)

STOCK_CACHE_REQUIRED_COLUMNS = (
    "con_id",
    "symbol",
    "local_symbol",
    "security_type",
    "currency",
    "exchange",
    "primary_exchange",
    "trading_class",
    "min_tick",
    "valid_exchanges",
    "timezone_id",
    "trading_hours",
    "liquid_hours",
    "market_rule_ids",
    "long_name",
    "resolved_at",
    "server_version",
    "contract_hash",
)

FUTURE_CACHE_REQUIRED_COLUMNS = (
    "con_id",
    "symbol",
    "local_symbol",
    "security_type",
    "currency",
    "exchange",
    "last_trade_date_or_contract_month",
    "real_expiration_date",
    "last_trade_time",
    "multiplier",
    "min_tick",
    "trading_class",
    "under_con_id",
    "timezone_id",
    "trading_hours",
    "liquid_hours",
    "market_rule_ids",
    "resolved_at",
    "server_version",
    "contract_hash",
)

CONTRACTS_DUCKDB_DDL = """
CREATE TABLE IF NOT EXISTS ibkr_contracts (
    con_id BIGINT PRIMARY KEY,
    symbol VARCHAR NOT NULL,
    local_symbol VARCHAR NOT NULL,
    security_type VARCHAR NOT NULL,
    currency VARCHAR NOT NULL,
    exchange VARCHAR NOT NULL,
    primary_exchange VARCHAR,
    trading_class VARCHAR,
    expiry VARCHAR,
    multiplier DOUBLE,
    min_tick DOUBLE,
    valid_exchanges VARCHAR,
    market_rule_ids VARCHAR,
    long_name VARCHAR,
    industry VARCHAR,
    category VARCHAR,
    subcategory VARCHAR,
    last_trade_date_or_contract_month VARCHAR,
    real_expiration_date VARCHAR,
    last_trade_time VARCHAR,
    under_con_id BIGINT,
    first_notice_day DATE,
    last_trade_day DATE,
    delivery_type VARCHAR,
    settlement_type VARCHAR,
    contract_size_unit VARCHAR,
    roll_group VARCHAR,
    timezone_id VARCHAR,
    trading_hours VARCHAR,
    liquid_hours VARCHAR,
    resolved_at TIMESTAMP NOT NULL,
    server_version INTEGER,
    contract_hash VARCHAR NOT NULL
);
""".strip()


@dataclass(frozen=True)
class ContractCacheRow:
    contract: ResolvedContract
    resolved_at: datetime
    server_version: int | None = None

    @property
    def contract_hash(self) -> str:
        return contract_hash(self.contract)

    def matches_request(self, request: ContractResolutionRequest) -> bool:
        request.validate_basic()
        return (
            self.contract.symbol.upper() == request.symbol.upper()
            and self.contract.security_type == request.security_type
            and self.contract.currency.upper() == request.currency.upper()
            and self.contract.exchange.upper() == request.exchange.upper()
            and (
                request.primary_exchange is None
                or (self.contract.primary_exchange or "").upper() == request.primary_exchange.upper()
            )
            and (request.expiry is None or self.contract.expiry == request.expiry)
        )

    def is_stale(self, *, now: datetime | None = None) -> bool:
        effective_now = now or datetime.now(UTC)
        return effective_now - self.resolved_at >= contract_cache_ttl(self.contract.security_type)

    def as_resolution_audit(self, request: ContractResolutionRequest, status: str) -> dict[str, Any]:
        request.validate_basic()
        return {
            "requested_symbol": request.symbol.upper(),
            "requested_security_type": request.security_type.value,
            "requested_currency": request.currency.upper(),
            "requested_exchange": request.exchange.upper(),
            "requested_primary_exchange": request.primary_exchange,
            "returned_match_count": 1,
            "selected_conId": self.contract.con_id,
            "resolution_status": status,
            "resolved_at": self.resolved_at.isoformat(),
            "server_version": self.server_version,
            "request_hash": contract_request_hash(request),
            "contract_hash": self.contract_hash,
        }


@dataclass(frozen=True)
class ContractResolutionArtifacts:
    audit_record: dict[str, Any]
    cache_row: ContractCacheRow | None

    @property
    def resolved(self) -> bool:
        return self.cache_row is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "ibkr_contract_resolution_artifacts_v1",
            "resolved": self.resolved,
            "audit_record": self.audit_record,
            "cache_row": {
                "con_id": self.cache_row.contract.con_id,
                "contract_hash": self.cache_row.contract_hash,
                "resolved_at": self.cache_row.resolved_at.isoformat(),
                "server_version": self.cache_row.server_version,
            }
            if self.cache_row
            else None,
            "resolved_contract_identity": contract_identity_document(self.cache_row)
            if self.cache_row
            else None,
            "financial_calls": {
                "place_order": 0,
                "cancel_order": 0,
                "global_cancel": 0,
            },
        }


@dataclass(frozen=True)
class ContractCacheLayout:
    cache_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> ContractCacheLayout:
        return cls(cache_dir=project_root / "output" / "ibkr" / "contracts")

    @property
    def stocks_parquet(self) -> Path:
        return self.cache_dir / "stocks.parquet"

    @property
    def futures_parquet(self) -> Path:
        return self.cache_dir / "futures.parquet"

    @property
    def requests_jsonl(self) -> Path:
        return self.cache_dir / "contract_requests.jsonl"

    @property
    def errors_jsonl(self) -> Path:
        return self.cache_dir / "contract_errors.jsonl"

    @property
    def manifest_json(self) -> Path:
        return self.cache_dir / "contract_manifest.json"

    def as_dict(self) -> dict[str, str]:
        return {
            "cache_dir": str(self.cache_dir),
            "stocks_parquet": str(self.stocks_parquet),
            "futures_parquet": str(self.futures_parquet),
            "contract_requests_jsonl": str(self.requests_jsonl),
            "contract_errors_jsonl": str(self.errors_jsonl),
            "contract_manifest_json": str(self.manifest_json),
        }


def contract_cache_ttl(security_type: IbkrSecurityType) -> timedelta:
    if security_type == IbkrSecurityType.STK:
        return timedelta(days=7)
    if security_type == IbkrSecurityType.FUT:
        return timedelta(hours=24)
    raise ValueError(f"unsupported security type: {security_type.value}")


def canonical_contract_payload(contract: ResolvedContract) -> dict[str, Any]:
    return {
        "conId": contract.con_id,
        "symbol": contract.symbol,
        "localSymbol": contract.local_symbol,
        "secType": contract.security_type.value,
        "exchange": contract.exchange,
        "primaryExchange": contract.primary_exchange,
        "currency": contract.currency,
        "tradingClass": contract.trading_class,
        "multiplier": _decimal_to_string(contract.multiplier),
        "minTick": _decimal_to_string(contract.min_tick),
        "expiry": contract.expiry,
        "validExchanges": contract.valid_exchanges,
        "marketRuleIds": contract.market_rule_ids,
        "longName": contract.long_name,
        "industry": contract.industry,
        "category": contract.category,
        "subcategory": contract.subcategory,
        "lastTradeDateOrContractMonth": contract.last_trade_date_or_contract_month,
        "realExpirationDate": contract.real_expiration_date,
        "lastTradeTime": contract.last_trade_time,
        "underConId": contract.under_con_id,
        "firstNoticeDay": contract.first_notice_day.isoformat() if contract.first_notice_day else None,
        "lastTradeDay": contract.last_trade_day.isoformat() if contract.last_trade_day else None,
        "deliveryType": contract.delivery_type,
        "settlementType": contract.settlement_type,
        "contractSizeUnit": contract.contract_size_unit,
        "rollGroup": contract.roll_group,
        "timeZoneId": contract.time_zone_id,
        "liquidHours": contract.liquid_hours,
        "tradingHours": contract.trading_hours,
    }


def contract_hash(contract: ResolvedContract) -> str:
    payload = canonical_contract_payload(contract)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


def contract_identity_document(row: ContractCacheRow) -> dict[str, Any]:
    row.contract.validate_phase2_required_fields()
    return {
        "schema": "ibkr_contract_identity_v1",
        "contract": canonical_contract_payload(row.contract),
        "resolved_at": row.resolved_at.isoformat(),
        "server_version": row.server_version,
        "contract_hash": row.contract_hash,
        "cache_ttl_seconds": int(contract_cache_ttl(row.contract.security_type).total_seconds()),
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def export_contract_identity(layout: ContractCacheLayout, con_id: int) -> dict[str, Any]:
    if con_id <= 0:
        raise ValueError("con_id must be positive")

    rows = read_contract_cache_rows(layout)
    match = next((row for row in rows if row.contract.con_id == con_id), None)
    return {
        "schema": "ibkr_contract_identity_export_v1",
        "status": "GO" if match else "NOT_FOUND",
        "con_id": con_id,
        "resolved_contract_identity": contract_identity_document(match) if match else None,
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def validate_unique_con_ids(rows: list[ContractCacheRow]) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for row in rows:
        con_id = row.contract.con_id
        if con_id in seen:
            duplicates.add(con_id)
        seen.add(con_id)
    if duplicates:
        duplicate_list = ", ".join(str(item) for item in sorted(duplicates))
        raise ValueError(f"duplicate conId in contract cache: {duplicate_list}")


def find_fresh_contract_cache_hit(
    rows: list[ContractCacheRow],
    request: ContractResolutionRequest,
    *,
    now: datetime | None = None,
) -> ContractCacheRow | None:
    request.validate_basic()
    validate_unique_con_ids(rows)
    matches = [row for row in rows if row.matches_request(request)]
    if len(matches) > 1:
        con_ids = ", ".join(str(row.contract.con_id) for row in matches)
        raise ValueError(f"ambiguous cache hit for {request.symbol}: {con_ids}")
    if not matches:
        return None
    match = matches[0]
    if match.is_stale(now=now):
        return None
    return match


def contract_cache_record(row: ContractCacheRow) -> dict[str, Any]:
    row.contract.validate_phase2_required_fields()
    _validate_timezone_aware_datetime(row.resolved_at, "resolved_at")
    _validate_optional_positive_int(row.server_version, "server_version")
    contract = row.contract
    return {
        "con_id": contract.con_id,
        "symbol": contract.symbol,
        "local_symbol": contract.local_symbol,
        "security_type": contract.security_type.value,
        "currency": contract.currency,
        "exchange": contract.exchange,
        "primary_exchange": contract.primary_exchange,
        "trading_class": contract.trading_class,
        "expiry": contract.expiry,
        "multiplier": _decimal_to_float(contract.multiplier),
        "min_tick": _decimal_to_float(contract.min_tick),
        "valid_exchanges": contract.valid_exchanges,
        "market_rule_ids": contract.market_rule_ids,
        "long_name": contract.long_name,
        "industry": contract.industry,
        "category": contract.category,
        "subcategory": contract.subcategory,
        "last_trade_date_or_contract_month": contract.last_trade_date_or_contract_month,
        "real_expiration_date": contract.real_expiration_date,
        "last_trade_time": contract.last_trade_time,
        "under_con_id": contract.under_con_id,
        "first_notice_day": contract.first_notice_day.isoformat() if contract.first_notice_day else None,
        "last_trade_day": contract.last_trade_day.isoformat() if contract.last_trade_day else None,
        "delivery_type": contract.delivery_type,
        "settlement_type": contract.settlement_type,
        "contract_size_unit": contract.contract_size_unit,
        "roll_group": contract.roll_group,
        "timezone_id": contract.time_zone_id,
        "trading_hours": contract.trading_hours,
        "liquid_hours": contract.liquid_hours,
        "resolved_at": row.resolved_at,
        "server_version": row.server_version,
        "contract_hash": row.contract_hash,
    }


def write_contract_cache_rows(
    layout: ContractCacheLayout,
    rows: list[ContractCacheRow],
) -> dict[str, Any]:
    validate_unique_con_ids(rows)
    grouped: dict[IbkrSecurityType, list[dict[str, Any]]] = {
        IbkrSecurityType.STK: [],
        IbkrSecurityType.FUT: [],
    }
    for row in rows:
        if row.contract.security_type not in grouped:
            raise ValueError(f"unsupported security type: {row.contract.security_type.value}")
        grouped[row.contract.security_type].append(contract_cache_record(row))

    written: dict[str, dict[str, Any]] = {}
    for security_type, records in grouped.items():
        if not records:
            continue
        path = cache_path_for_security_type(layout, security_type)
        _write_contract_cache_file(path, records)
        written[security_type.value] = {
            "path": str(path),
            "row_count": len(records),
        }

    return {
        "schema": "ibkr_contract_cache_write_v1",
        "status": "GO",
        "written": written,
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def upsert_contract_cache_row(layout: ContractCacheLayout, row: ContractCacheRow) -> dict[str, Any]:
    existing_rows = read_contract_cache_rows(layout)
    merged_rows = [existing for existing in existing_rows if existing.contract.con_id != row.contract.con_id]
    merged_rows.append(row)
    return write_contract_cache_rows(layout, merged_rows)


def persist_contract_resolution_artifacts(
    layout: ContractCacheLayout,
    artifacts: ContractResolutionArtifacts,
) -> dict[str, Any]:
    resolution_status = artifacts.audit_record.get("resolution_status")
    if resolution_status == ContractResolutionStatus.RESOLVED.value:
        append_contract_request_audit(layout, artifacts.audit_record)
    else:
        append_contract_error_audit(layout, artifacts.audit_record)

    cache_report = None
    if artifacts.cache_row is not None:
        cache_report = upsert_contract_cache_row(layout, artifacts.cache_row)

    return {
        "schema": "ibkr_contract_resolution_artifact_persist_v1",
        "status": "GO",
        "resolution_status": resolution_status,
        "audit_file": str(layout.requests_jsonl if resolution_status == ContractResolutionStatus.RESOLVED.value else layout.errors_jsonl),
        "cache_written": artifacts.cache_row is not None,
        "cache_report": cache_report,
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def read_contract_cache_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def read_contract_cache_rows(layout: ContractCacheLayout) -> list[ContractCacheRow]:
    validation = validate_contract_cache(layout)
    if validation["status"] != "GO":
        errors = "; ".join(validation["errors"])
        raise ValueError(f"invalid contract cache: {errors}")

    rows: list[ContractCacheRow] = []
    for path in (layout.stocks_parquet, layout.futures_parquet):
        for record in read_contract_cache_records(path):
            rows.append(_contract_cache_row_from_record(record))
    validate_unique_con_ids(rows)
    return rows


def validate_contract_cache(layout: ContractCacheLayout) -> dict[str, Any]:
    stock_summary = _validate_contract_cache_file(
        layout.stocks_parquet,
        security_type=IbkrSecurityType.STK,
        required_columns=STOCK_CACHE_REQUIRED_COLUMNS,
    )
    future_summary = _validate_contract_cache_file(
        layout.futures_parquet,
        security_type=IbkrSecurityType.FUT,
        required_columns=FUTURE_CACHE_REQUIRED_COLUMNS,
    )
    all_errors = list(stock_summary["errors"]) + list(future_summary["errors"])
    duplicate_errors = _duplicate_con_id_errors(
        stock_summary["con_ids"],
        future_summary["con_ids"],
    )
    all_errors.extend(duplicate_errors)
    request_audit_summary = _validate_contract_audit_log(
        layout.requests_jsonl,
        expected_statuses={ContractResolutionStatus.RESOLVED.value},
        require_contract_hash=True,
    )
    error_audit_summary = _validate_contract_audit_log(
        layout.errors_jsonl,
        expected_statuses={
            ContractResolutionStatus.NOT_FOUND.value,
            ContractResolutionStatus.AMBIGUOUS_BLOCKED.value,
            ContractResolutionStatus.VALIDATION_ERROR.value,
            ContractResolutionStatus.INVALID_REQUEST.value,
            ContractResolutionStatus.INVALID_CURRENCY.value,
            ContractResolutionStatus.INVALID_EXCHANGE.value,
            ContractResolutionStatus.MISSING_REQUIRED_FIELDS.value,
            ContractResolutionStatus.STALE_CACHE.value,
            ContractResolutionStatus.CALLBACK_TIMEOUT.value,
            ContractResolutionStatus.PROVIDER_ERROR.value,
            ContractResolutionStatus.CONTRACT_VALIDATION_FAILED.value,
        },
        require_contract_hash=False,
    )
    manifest_summary = _validate_contract_manifest(layout)
    all_errors.extend(request_audit_summary["errors"])
    all_errors.extend(error_audit_summary["errors"])
    all_errors.extend(manifest_summary["errors"])

    return {
        "schema": "ibkr_contract_cache_validation_v1",
        "status": "NO_GO" if all_errors else "GO",
        "validated_at": datetime.now(UTC).isoformat(),
        "files": {
            "stocks_parquet": _public_file_summary(stock_summary),
            "futures_parquet": _public_file_summary(future_summary),
            "contract_requests_jsonl": _public_file_summary(request_audit_summary),
            "contract_errors_jsonl": _public_file_summary(error_audit_summary),
            "contract_manifest_json": _public_file_summary(manifest_summary),
        },
        "row_count": stock_summary["row_count"] + future_summary["row_count"],
        "audit_row_count": request_audit_summary["row_count"] + error_audit_summary["row_count"],
        "errors": all_errors,
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def cache_path_for_security_type(layout: ContractCacheLayout, security_type: IbkrSecurityType) -> Path:
    if security_type == IbkrSecurityType.STK:
        return layout.stocks_parquet
    if security_type == IbkrSecurityType.FUT:
        return layout.futures_parquet
    raise ValueError(f"unsupported security type: {security_type.value}")


def build_resolution_audit_record(
    request: ContractResolutionRequest,
    evaluation: ContractCandidateEvaluation,
    *,
    resolved_at: datetime | None = None,
    server_version: int | None = None,
) -> dict[str, Any]:
    request.validate_basic()
    effective_resolved_at = resolved_at or datetime.now(UTC)
    _validate_timezone_aware_datetime(effective_resolved_at, "resolved_at")
    selected = evaluation.matches[0] if evaluation.status == ContractResolutionStatus.RESOLVED else None
    return {
        "requested_symbol": request.symbol.upper(),
        "requested_security_type": request.security_type.value,
        "requested_currency": request.currency.upper(),
        "requested_exchange": request.exchange.upper(),
        "requested_primary_exchange": request.primary_exchange,
        "returned_match_count": len(evaluation.matches),
        "selected_conId": selected.con_id if selected else None,
        "resolution_status": evaluation.status.value,
        "resolved_at": effective_resolved_at.isoformat(),
        "server_version": server_version,
        "request_hash": contract_request_hash(request),
        "contract_hash": contract_hash(selected) if selected else None,
    }


def build_contract_resolution_artifacts(
    request: ContractResolutionRequest,
    evaluation: ContractCandidateEvaluation,
    *,
    resolved_at: datetime | None = None,
    server_version: int | None = None,
) -> ContractResolutionArtifacts:
    effective_resolved_at = resolved_at or datetime.now(UTC)
    audit_record = build_resolution_audit_record(
        request,
        evaluation,
        resolved_at=effective_resolved_at,
        server_version=server_version,
    )
    if evaluation.status != ContractResolutionStatus.RESOLVED:
        return ContractResolutionArtifacts(audit_record=audit_record, cache_row=None)

    selected = evaluation.matches[0]
    selected.validate_phase2_required_fields()
    return ContractResolutionArtifacts(
        audit_record=audit_record,
        cache_row=ContractCacheRow(
            contract=selected,
            resolved_at=effective_resolved_at,
            server_version=server_version,
        ),
    )


def append_contract_request_audit(
    layout: ContractCacheLayout,
    record: dict[str, Any],
) -> None:
    _append_jsonl(layout.requests_jsonl, record)


def append_contract_error_audit(
    layout: ContractCacheLayout,
    record: dict[str, Any],
) -> None:
    _append_jsonl(layout.errors_jsonl, record)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.open("a", encoding="utf-8").write(
        json.dumps(record, sort_keys=True, ensure_ascii=True, default=str) + "\n"
    )


def empty_contract_manifest(layout: ContractCacheLayout) -> dict[str, Any]:
    return {
        "schema": "ibkr_contract_cache_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "cache_dir": str(layout.cache_dir),
        "files": layout.as_dict(),
        "required_fields": {
            "stocks": list(STOCK_CONTRACT_FIELDS),
            "futures": list(FUTURE_CONTRACT_FIELDS),
            "resolution_audit": list(RESOLUTION_AUDIT_FIELDS),
        },
        "optional_reference_fields": {
            "futures": list(FUTURE_REFERENCE_METADATA_FIELDS),
        },
        "cache_policy": {
            "STK": "7 days",
            "FUT": "24 hours",
        },
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def contract_schema_manifest() -> dict[str, Any]:
    return {
        "schema": "ibkr_contract_schema_v1",
        "supported_security_types": ["STK", "FUT"],
        "duckdb": {
            "contracts_ddl": CONTRACTS_DUCKDB_DDL,
            "primary_key": "con_id",
        },
        "required_fields": {
            "stocks": list(STOCK_CONTRACT_FIELDS),
            "futures": list(FUTURE_CONTRACT_FIELDS),
            "resolution_audit": list(RESOLUTION_AUDIT_FIELDS),
        },
        "optional_reference_fields": {
            "futures": list(FUTURE_REFERENCE_METADATA_FIELDS),
        },
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def initialize_contract_cache(layout: ContractCacheLayout) -> dict[str, Any]:
    layout.cache_dir.mkdir(parents=True, exist_ok=True)
    layout.requests_jsonl.touch(exist_ok=True)
    layout.errors_jsonl.touch(exist_ok=True)

    manifest = empty_contract_manifest(layout)
    manifest["initialized"] = True
    manifest["existing_files"] = {
        "stocks_parquet": layout.stocks_parquet.exists(),
        "futures_parquet": layout.futures_parquet.exists(),
        "contract_requests_jsonl": layout.requests_jsonl.exists(),
        "contract_errors_jsonl": layout.errors_jsonl.exists(),
    }
    layout.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def _decimal_to_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def _decimal_to_float(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _write_contract_cache_file(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)


def _contract_cache_row_from_record(record: dict[str, Any]) -> ContractCacheRow:
    contract = ResolvedContract(
        con_id=int(record["con_id"]),
        symbol=str(record["symbol"]),
        local_symbol=str(record["local_symbol"]),
        security_type=IbkrSecurityType(str(record["security_type"])),
        exchange=str(record["exchange"]),
        currency=str(record["currency"]),
        trading_class=_optional_string(record.get("trading_class")),
        primary_exchange=_optional_string(record.get("primary_exchange")),
        multiplier=_decimal_from_record(record.get("multiplier")),
        min_tick=_decimal_from_record(record.get("min_tick")),
        expiry=_optional_string(record.get("expiry")),
        valid_exchanges=_optional_string(record.get("valid_exchanges")),
        market_rule_ids=_optional_string(record.get("market_rule_ids")),
        long_name=_optional_string(record.get("long_name")),
        industry=_optional_string(record.get("industry")),
        category=_optional_string(record.get("category")),
        subcategory=_optional_string(record.get("subcategory")),
        last_trade_date_or_contract_month=_optional_string(record.get("last_trade_date_or_contract_month")),
        real_expiration_date=_optional_string(record.get("real_expiration_date")),
        last_trade_time=_optional_string(record.get("last_trade_time")),
        under_con_id=_optional_int(record.get("under_con_id")),
        first_notice_day=_optional_date(record.get("first_notice_day")),
        last_trade_day=_optional_date(record.get("last_trade_day")),
        delivery_type=_optional_string(record.get("delivery_type")),
        settlement_type=_optional_string(record.get("settlement_type")),
        contract_size_unit=_optional_string(record.get("contract_size_unit")),
        roll_group=_optional_string(record.get("roll_group")),
        time_zone_id=_optional_string(record.get("timezone_id")),
        trading_hours=_optional_string(record.get("trading_hours")),
        liquid_hours=_optional_string(record.get("liquid_hours")),
    )
    expected_hash = str(record["contract_hash"])
    actual_hash = contract_hash(contract)
    if actual_hash != expected_hash:
        raise ValueError(f"contract_hash mismatch for con_id {contract.con_id}")
    return ContractCacheRow(
        contract=contract,
        resolved_at=_datetime_from_record(record["resolved_at"]),
        server_version=_optional_int(record.get("server_version")),
    )


def _validate_contract_cache_file(
    path: Path,
    *,
    security_type: IbkrSecurityType,
    required_columns: tuple[str, ...],
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "row_count": 0,
            "con_ids": [],
            "errors": [],
        }

    try:
        records = read_contract_cache_records(path)
    except Exception as exc:  # pragma: no cover - exact pyarrow exception is platform-dependent.
        return {
            "path": str(path),
            "exists": True,
            "row_count": 0,
            "con_ids": [],
            "errors": [f"{path.name}: unreadable parquet: {exc}"],
        }

    errors: list[str] = []
    con_ids: list[int] = []
    for index, record in enumerate(records):
        row_label = f"{path.name}[{index}]"
        row_error_count = len(errors)
        con_id = record.get("con_id")
        if isinstance(con_id, int):
            con_ids.append(con_id)
        else:
            errors.append(f"{row_label}: con_id must be an integer")

        if record.get("security_type") != security_type.value:
            errors.append(f"{row_label}: security_type must be {security_type.value}")

        for column in required_columns:
            if column not in record:
                errors.append(f"{row_label}: missing column {column}")
                continue
            value = record[column]
            if value is None:
                errors.append(f"{row_label}: missing value {column}")
            elif isinstance(value, str) and not value.strip():
                errors.append(f"{row_label}: blank value {column}")

        if len(errors) == row_error_count:
            try:
                _contract_cache_row_from_record(record)
            except ValueError as exc:
                errors.append(f"{row_label}: {exc}")

    return {
        "path": str(path),
        "exists": True,
        "row_count": len(records),
        "con_ids": con_ids,
        "errors": errors,
    }


def _validate_contract_audit_log(
    path: Path,
    *,
    expected_statuses: set[str],
    require_contract_hash: bool,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "row_count": 0,
            "errors": [],
        }

    errors: list[str] = []
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row_count += 1
            row_label = f"{path.name}[{index}]"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{row_label}: invalid JSON: {exc.msg}")
                continue

            for field in RESOLUTION_AUDIT_FIELDS:
                if field not in record:
                    errors.append(f"{row_label}: missing audit field {field}")

            status = record.get("resolution_status")
            if status not in expected_statuses:
                expected = ", ".join(sorted(expected_statuses))
                errors.append(f"{row_label}: unexpected resolution_status {status}; expected {expected}")

            try:
                _validate_audit_request_fields(record)
            except ValueError as exc:
                errors.append(f"{row_label}: {exc}")

            try:
                _datetime_from_record(record.get("resolved_at"))
            except (TypeError, ValueError) as exc:
                errors.append(f"{row_label}: resolved_at must be a timezone-aware ISO timestamp: {exc}")

            server_version = record.get("server_version")
            if server_version is not None and (type(server_version) is not int or server_version <= 0):
                errors.append(f"{row_label}: server_version must be null or a positive integer")

            match_count = record.get("returned_match_count")
            if type(match_count) is not int or match_count < 0:
                errors.append(f"{row_label}: returned_match_count must be a non-negative integer")
            elif status == ContractResolutionStatus.RESOLVED.value and match_count != 1:
                errors.append(f"{row_label}: returned_match_count must be 1 for RESOLVED")
            elif status == ContractResolutionStatus.NOT_FOUND.value and match_count != 0:
                errors.append(f"{row_label}: returned_match_count must be 0 for NOT_FOUND")
            elif status == ContractResolutionStatus.AMBIGUOUS_BLOCKED.value and match_count <= 1:
                errors.append(f"{row_label}: returned_match_count must be greater than 1 for AMBIGUOUS_BLOCKED")

            selected_con_id = record.get("selected_conId")
            request_hash_value = record.get("request_hash")
            if not isinstance(request_hash_value, str) or not re.fullmatch(r"[A-F0-9]{64}", request_hash_value):
                errors.append(f"{row_label}: request_hash must be a 64-character uppercase hex digest")
            contract_hash_value = record.get("contract_hash")
            if require_contract_hash:
                if selected_con_id is None:
                    errors.append(f"{row_label}: selected_conId is required for resolved audit records")
                elif type(selected_con_id) is not int or selected_con_id <= 0:
                    errors.append(f"{row_label}: selected_conId must be a positive integer for resolved audit records")
                if not isinstance(contract_hash_value, str) or not contract_hash_value.strip():
                    errors.append(f"{row_label}: contract_hash is required for resolved audit records")
            else:
                if selected_con_id is not None:
                    errors.append(f"{row_label}: selected_conId must be null for error audit records")
                if contract_hash_value is not None:
                    errors.append(f"{row_label}: contract_hash must be null for error audit records")

    return {
        "path": str(path),
        "exists": True,
        "row_count": row_count,
        "errors": errors,
    }


def _validate_audit_request_fields(record: dict[str, Any]) -> None:
    _validate_audit_code_field(record.get("requested_symbol"), "requested_symbol")
    _validate_audit_code_field(record.get("requested_exchange"), "requested_exchange")
    primary_exchange = record.get("requested_primary_exchange")
    if primary_exchange is not None:
        _validate_audit_code_field(primary_exchange, "requested_primary_exchange")
    currency = record.get("requested_currency")
    if not isinstance(currency, str) or currency != currency.strip() or not re.fullmatch(r"[A-Z]{3}", currency):
        raise ValueError("requested_currency must be a 3-letter uppercase code without surrounding whitespace")
    security_type = record.get("requested_security_type")
    if security_type not in {IbkrSecurityType.STK.value, IbkrSecurityType.FUT.value}:
        raise ValueError("requested_security_type must be STK or FUT")


def _validate_audit_code_field(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} is required")
    if value != value.strip():
        raise ValueError(f"{field_name} must not contain leading or trailing whitespace")
    if not _AUDIT_IBKR_CODE_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be an uppercase IBKR code")


def _validate_contract_manifest(layout: ContractCacheLayout) -> dict[str, Any]:
    path = layout.manifest_json
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "row_count": 0,
            "errors": [],
        }

    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "path": str(path),
            "exists": True,
            "row_count": 1,
            "errors": [f"{path.name}: invalid JSON: {exc.msg}"],
        }

    if manifest.get("schema") != "ibkr_contract_cache_manifest_v1":
        errors.append(f"{path.name}: invalid schema {manifest.get('schema')}")
    if manifest.get("cache_dir") != str(layout.cache_dir):
        errors.append(f"{path.name}: cache_dir does not match expected layout")

    expected_files = layout.as_dict()
    actual_files = manifest.get("files")
    if not isinstance(actual_files, dict):
        errors.append(f"{path.name}: files must be an object")
    else:
        for key, expected_value in expected_files.items():
            if actual_files.get(key) != expected_value:
                errors.append(f"{path.name}: files.{key} does not match expected layout")

    expected_required_fields = {
        "stocks": list(STOCK_CONTRACT_FIELDS),
        "futures": list(FUTURE_CONTRACT_FIELDS),
        "resolution_audit": list(RESOLUTION_AUDIT_FIELDS),
    }
    if manifest.get("required_fields") != expected_required_fields:
        errors.append(f"{path.name}: required_fields do not match Phase 2 schema")

    expected_optional_reference_fields = {
        "futures": list(FUTURE_REFERENCE_METADATA_FIELDS),
    }
    if manifest.get("optional_reference_fields") != expected_optional_reference_fields:
        errors.append(f"{path.name}: optional_reference_fields do not match Phase 2 schema")

    if manifest.get("cache_policy") != {"STK": "7 days", "FUT": "24 hours"}:
        errors.append(f"{path.name}: cache_policy does not match Phase 2 policy")

    financial_calls = manifest.get("financial_calls")
    if financial_calls != {"place_order": 0, "cancel_order": 0, "global_cancel": 0}:
        errors.append(f"{path.name}: financial_calls must all be 0")

    return {
        "path": str(path),
        "exists": True,
        "row_count": 1,
        "errors": errors,
    }


def _duplicate_con_id_errors(*con_id_groups: list[int]) -> list[str]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for con_ids in con_id_groups:
        for con_id in con_ids:
            if con_id <= 0:
                duplicates.add(con_id)
            if con_id in seen:
                duplicates.add(con_id)
            seen.add(con_id)
    if not duplicates:
        return []
    duplicate_list = ", ".join(str(item) for item in sorted(duplicates))
    return [f"duplicate or invalid con_id in contract cache: {duplicate_list}"]


def _public_file_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": summary["path"],
        "exists": summary["exists"],
        "row_count": summary["row_count"],
        "error_count": len(summary["errors"]),
    }


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(str(value).strip())


def _validate_optional_positive_int(value: int | None, field_name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{field_name} must be null or a positive integer")


def _validate_timezone_aware_datetime(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _optional_date(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value))


def _datetime_from_record(value: object) -> datetime:
    if value is None:
        raise ValueError("missing timestamp")
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp lacks timezone")
        return value
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp lacks timezone")
    return parsed


def _decimal_from_record(value: object) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value)).normalize()
