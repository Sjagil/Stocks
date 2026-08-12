from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from stocks.domain.assets import IbkrSecurityType


class BarInterval(str, Enum):
    ONE_DAY = "1d"
    ONE_HOUR = "1h"
    # First-class tactical active-swing interval; never execution authority.
    FIFTEEN_MINUTES = "15m"


class BarDataType(str, Enum):
    TRADES = "TRADES"
    MIDPOINT = "MIDPOINT"
    BID = "BID"
    ASK = "ASK"
    ADJUSTED_LAST = "ADJUSTED_LAST"


class BarDataSource(str, Enum):
    IBKR = "IBKR"
    EODHD = "EODHD"


BAR_CACHE_FIELDS = (
    "con_id",
    "sec_type",
    "interval",
    "data_type",
    "source",
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "available_at",
)

TARGET_BAR_COLLECTION_FIELDS = (
    "timestamp_utc",
    "session_date",
    "con_id",
    "symbol",
    "security_type",
    "currency",
    "exchange",
    "primary_exchange",
    "interval",
    "data_type",
    "use_rth",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "wap",
    "bar_count",
    "source",
    "requested_at",
    "received_at",
    "contract_hash",
    "session_hash",
    "request_hash",
    "content_hash",
)


@dataclass(frozen=True)
class BarRequestPolicy:
    max_concurrent_ibkr_requests: int = 3
    max_concurrent_eodhd_requests: int = 2
    request_timeout_seconds: int = 60
    max_retries: int = 3
    retry_backoff_seconds: tuple[int, ...] = (2, 5, 15)

    def validate(self) -> None:
        max_concurrent_ibkr_requests = _int_policy_value(
            self.max_concurrent_ibkr_requests,
            "max_concurrent_ibkr_requests",
        )
        max_concurrent_eodhd_requests = _int_policy_value(
            self.max_concurrent_eodhd_requests,
            "max_concurrent_eodhd_requests",
        )
        _int_policy_value(self.request_timeout_seconds, "request_timeout_seconds")
        max_retries = _int_policy_value(self.max_retries, "max_retries", minimum=0)
        retry_backoff_seconds = _retry_backoff_policy_value(self.retry_backoff_seconds)

        if max_concurrent_ibkr_requests > 5:
            raise ValueError("max_concurrent_ibkr_requests must be between 1 and 5")
        if max_concurrent_eodhd_requests > 5:
            raise ValueError("max_concurrent_eodhd_requests must be between 1 and 5")
        if len(retry_backoff_seconds) < max_retries:
            raise ValueError("retry_backoff_seconds must cover max_retries")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "max_concurrent_ibkr_requests": self.max_concurrent_ibkr_requests,
            "max_concurrent_eodhd_requests": self.max_concurrent_eodhd_requests,
            "request_timeout_seconds": self.request_timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": list(self.retry_backoff_seconds),
            "ibkr_hard_historical_request_cap": 50,
            "authority": "phase4_read_only_historical_ibkr_daily_stk",
        }


@dataclass(frozen=True)
class HistoricalBarRequest:
    con_id: int
    security_type: IbkrSecurityType
    interval: BarInterval
    data_type: BarDataType
    source: BarDataSource
    start: datetime
    end: datetime

    def validate(self, *, data_phase_enabled: bool = False) -> None:
        if not data_phase_enabled:
            raise ValueError("historical bar requests are disabled until the data phase is explicitly enabled")
        validate_bar_request_fields(
            con_id=self.con_id,
            security_type=self.security_type,
            interval=self.interval,
            data_type=self.data_type,
            source=self.source,
            start=self.start,
            end=self.end,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "historical_bar_request_v1",
            "con_id": self.con_id,
            "sec_type": _enum_report_value(self.security_type),
            "interval": _enum_report_value(self.interval),
            "data_type": _enum_report_value(self.data_type),
            "source": _enum_report_value(self.source),
            "start": _datetime_report_value(self.start),
            "end": _datetime_report_value(self.end),
            "financial_calls": _zero_financial_calls(),
        }

    def request_key(self) -> str:
        start = _datetime_value(self.start, "start")
        end = _datetime_value(self.end, "end")
        return "|".join(
            [
                str(self.con_id),
                self.security_type.value,
                self.interval.value,
                self.data_type.value,
                self.source.value,
                start.astimezone(UTC).isoformat(),
                end.astimezone(UTC).isoformat(),
            ]
        )


@dataclass(frozen=True)
class HistoricalBar:
    con_id: int
    security_type: IbkrSecurityType
    interval: BarInterval
    data_type: BarDataType
    source: BarDataSource
    timestamp_utc: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int | None
    available_at: datetime

    def validate(self) -> None:
        _positive_int_value(self.con_id, "con_id")
        timestamp_utc = _datetime_value(self.timestamp_utc, "timestamp_utc")
        available_at = _datetime_value(self.available_at, "available_at")
        if timestamp_utc.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must be expressed in UTC")
        _validate_bar_identity_compatibility(
            security_type=self.security_type,
            interval=self.interval,
            data_type=self.data_type,
            source=self.source,
        )
        availability_floor = _bar_availability_floor(timestamp_utc, self.interval)
        if available_at.astimezone(UTC) < availability_floor:
            raise ValueError("available_at must be on or after the bar availability time")
        for field_name, value in {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
        }.items():
            value = _decimal_value(value, field_name)
            if not value.is_finite():
                raise ValueError(f"{field_name} must be finite")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high must be greater than or equal to open, low and close")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low must be less than or equal to open, high and close")
        _volume_value(self.volume)

    def as_record(self) -> dict[str, Any]:
        self.validate()
        return {
            "con_id": self.con_id,
            "sec_type": self.security_type.value,
            "interval": self.interval.value,
            "data_type": self.data_type.value,
            "source": self.source.value,
            "timestamp_utc": self.timestamp_utc.astimezone(UTC).isoformat(),
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": self.volume,
            "available_at": self.available_at.astimezone(UTC).isoformat(),
        }


@dataclass(frozen=True)
class BarCacheLayout:
    data_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> BarCacheLayout:
        return cls(data_dir=project_root / "data" / "bars")

    def partition_dir(
        self,
        *,
        security_type: IbkrSecurityType,
        con_id: int,
        interval: BarInterval,
        data_type: BarDataType,
        source: BarDataSource,
    ) -> Path:
        con_id = _positive_int_value(con_id, "con_id")
        security_type = _enum_value(IbkrSecurityType, security_type, "security_type")
        interval = _enum_value(BarInterval, interval, "interval")
        data_type = _enum_value(BarDataType, data_type, "data_type")
        source = _enum_value(BarDataSource, source, "source")
        return (
            self.data_dir
            / f"sec_type={security_type.value}"
            / f"con_id={con_id}"
            / f"interval={interval.value}"
            / f"data_type={data_type.value}"
            / f"source={source.value}"
        )

    def bars_path(
        self,
        *,
        security_type: IbkrSecurityType,
        con_id: int,
        interval: BarInterval,
        data_type: BarDataType,
        source: BarDataSource,
    ) -> Path:
        return self.partition_dir(
            security_type=security_type,
            con_id=con_id,
            interval=interval,
            data_type=data_type,
            source=source,
        ) / "bars.parquet"

    def phase4_bars_path(
        self,
        *,
        security_type: IbkrSecurityType,
        con_id: int,
        interval: BarInterval,
        data_type: BarDataType,
    ) -> Path:
        con_id = _positive_int_value(con_id, "con_id")
        security_type = _enum_value(IbkrSecurityType, security_type, "security_type")
        interval = _enum_value(BarInterval, interval, "interval")
        data_type = _enum_value(BarDataType, data_type, "data_type")
        return (
            self.data_dir
            / f"security_type={security_type.value}"
            / f"con_id={con_id}"
            / f"interval={interval.value}"
            / f"data_type={data_type.value}"
            / "bars.parquet"
        )

    @property
    def manifest_json(self) -> Path:
        return self.data_dir / "bar_manifest.json"

    def as_dict(self) -> dict[str, str]:
        return {
            "data_dir": str(self.data_dir),
            "manifest_json": str(self.manifest_json),
            "partitioning": "/".join(
                [
                    "sec_type=<STK|FUT>",
                    "con_id=<positive integer>",
                    "interval=<1d|1h|15m>",
                    "data_type=<TRADES|MIDPOINT|BID|ASK|ADJUSTED_LAST>",
                    "source=<IBKR|EODHD>",
                ]
            ),
            "phase4_partitioning": "/".join(
                [
                    "security_type=<STK>",
                    "con_id=<positive integer>",
                    "interval=<1d>",
                    "data_type=<TRADES>",
                ]
            ),
        }


def validate_bar_request_fields(
    *,
    con_id: int,
    security_type: IbkrSecurityType,
    interval: BarInterval,
    data_type: BarDataType,
    source: BarDataSource,
    start: datetime,
    end: datetime,
) -> None:
    _positive_int_value(con_id, "con_id")
    security_type = _enum_value(IbkrSecurityType, security_type, "security_type")
    interval = _enum_value(BarInterval, interval, "interval")
    data_type = _enum_value(BarDataType, data_type, "data_type")
    source = _enum_value(BarDataSource, source, "source")
    start = _datetime_value(start, "start")
    end = _datetime_value(end, "end")
    if end <= start:
        raise ValueError("end must be after start")
    _validate_bar_identity_compatibility(
        security_type=security_type,
        interval=interval,
        data_type=data_type,
        source=source,
    )


def _validate_bar_identity_compatibility(
    *,
    security_type: IbkrSecurityType,
    interval: BarInterval,
    data_type: BarDataType,
    source: BarDataSource,
) -> None:
    security_type = _enum_value(IbkrSecurityType, security_type, "security_type")
    interval = _enum_value(BarInterval, interval, "interval")
    data_type = _enum_value(BarDataType, data_type, "data_type")
    source = _enum_value(BarDataSource, source, "source")
    if security_type == IbkrSecurityType.FUT and data_type == BarDataType.ADJUSTED_LAST:
        raise ValueError("ADJUSTED_LAST is only valid for STK bars")
    if source == BarDataSource.EODHD and security_type != IbkrSecurityType.STK:
        raise ValueError("EODHD bar planning is only enabled for STK identities in Phase 4")
    if source == BarDataSource.EODHD and interval != BarInterval.ONE_DAY:
        raise ValueError("EODHD is only enabled for daily bar planning in Phase 4")
    if source == BarDataSource.EODHD and data_type not in {BarDataType.TRADES, BarDataType.ADJUSTED_LAST}:
        raise ValueError("EODHD bar planning only supports TRADES or ADJUSTED_LAST")


def plan_bar_request_queue(
    requests: list[HistoricalBarRequest],
    *,
    policy: BarRequestPolicy | None = None,
    contract_rows: list[Any] | None = None,
) -> dict[str, Any]:
    effective_policy = policy or BarRequestPolicy()
    effective_policy.validate()
    queued: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    duplicate_keys: set[str] = set()
    seen: dict[str, int] = {}
    known_identities = None if contract_rows is None else {_contract_row_identity(row) for row in contract_rows}

    for index, request in enumerate(requests):
        try:
            validate_bar_request_fields(
                con_id=request.con_id,
                security_type=request.security_type,
                interval=request.interval,
                data_type=request.data_type,
                source=request.source,
                start=request.start,
                end=request.end,
            )
        except ValueError as exc:
            rejected.append(
                {
                    "index": index,
                    "reason": str(exc),
                    "request": request.as_dict(),
                }
            )
            continue

        if known_identities is not None and (request.con_id, request.security_type.value) not in known_identities:
            rejected.append(
                {
                    "index": index,
                    "reason": (
                        f"request references {request.security_type.value} con_id {request.con_id} "
                        "missing from local contract cache"
                    ),
                    "request": request.as_dict(),
                }
            )
            continue

        key = request.request_key()
        if key in seen:
            duplicate_keys.add(key)
            rejected.append(
                {
                    "index": index,
                    "reason": f"duplicate request of index {seen[key]}",
                    "request": request.as_dict(),
                }
            )
            continue
        seen[key] = index
        queued.append({"index": index, "request_key": key, "request": request.as_dict()})

    source_counts = {source.value: 0 for source in BarDataSource}
    for item in queued:
        source_counts[item["request"]["source"]] += 1

    return {
        "schema": "historical_bar_request_queue_plan_v1",
        "status": "GO" if not rejected else "NO_GO",
        "execution_enabled": False,
        "authority": "phase4_read_only_historical_ibkr_daily_stk",
        "policy": effective_policy.as_dict(),
        "input_count": len(requests),
        "queued_count": len(queued),
        "rejected_count": len(rejected),
        "duplicate_request_count": len(duplicate_keys),
        "contract_identity_validation": {
            "enabled": known_identities is not None,
            "contract_cache_row_count": 0 if contract_rows is None else len(contract_rows),
        },
        "source_counts": source_counts,
        "queued_requests": queued,
        "rejected_requests": rejected,
        "financial_calls": _zero_financial_calls(),
    }


def detect_bar_gaps(bars: list[HistoricalBar], *, interval: BarInterval) -> dict[str, Any]:
    for bar in bars:
        bar.validate()
    ordered = sorted(bars, key=lambda item: item.timestamp_utc)
    duplicates = _duplicate_timestamps(ordered)
    gaps: list[dict[str, Any]] = []
    expected_delta = _expected_delta(interval)
    if expected_delta is not None:
        for previous, current in zip(ordered, ordered[1:]):
            delta = current.timestamp_utc - previous.timestamp_utc
            if delta > expected_delta:
                missing_steps = int(delta / expected_delta) - 1
                gaps.append(
                    {
                        "after": previous.timestamp_utc.isoformat(),
                        "before": current.timestamp_utc.isoformat(),
                        "expected_delta_seconds": int(expected_delta.total_seconds()),
                        "actual_delta_seconds": int(delta.total_seconds()),
                        "missing_steps": missing_steps,
                    }
                )

    return {
        "schema": "historical_bar_gap_report_v1",
        "status": "GO" if not duplicates and not gaps else "NO_GO",
        "interval": interval.value,
        "bar_count": len(ordered),
        "duplicate_timestamps": [timestamp.isoformat() for timestamp in duplicates],
        "gaps": gaps,
        "gap_detection_scope": "intraday_exact_delta" if expected_delta else "requires_market_calendar",
        "financial_calls": _zero_financial_calls(),
    }


def write_bar_cache_file(path: Path, bars: list[HistoricalBar]) -> dict[str, Any]:
    if not bars:
        raise ValueError("cannot write an empty bar cache file")
    for bar in bars:
        bar.validate()
    _validate_single_bar_identity(bars)
    _validate_bars_for_cache_write(path, bars)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([bar.as_record() for bar in bars])
    pq.write_table(table, path)
    return {
        "schema": "historical_bar_cache_write_v1",
        "status": "GO",
        "path": str(path),
        "row_count": len(bars),
        "financial_calls": _zero_financial_calls(),
    }


def _validate_bars_for_cache_write(path: Path, bars: list[HistoricalBar]) -> None:
    ordering_errors = _bar_timestamp_order_errors(path, bars)
    if ordering_errors:
        raise ValueError(ordering_errors[0])

    gap_report = detect_bar_gaps(bars, interval=bars[0].interval)
    if gap_report["status"] == "GO":
        return
    if gap_report["duplicate_timestamps"]:
        raise ValueError(f"{path}: duplicate timestamp {gap_report['duplicate_timestamps'][0]}")
    if gap_report["gaps"]:
        gap = gap_report["gaps"][0]
        raise ValueError(
            f"{path}: gap after {gap['after']} before {gap['before']} "
            f"missing_steps={gap['missing_steps']}"
        )


def read_bar_cache_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.ParquetFile(path).read().to_pylist()


def validate_bar_cache(layout: BarCacheLayout) -> dict[str, Any]:
    legacy_files = sorted(layout.data_dir.glob("sec_type=*/con_id=*/interval=*/data_type=*/source=*/bars.parquet"))
    phase4_files = sorted(layout.data_dir.glob("security_type=*/con_id=*/interval=*/data_type=*/bars.parquet"))
    summaries = [_validate_bar_cache_file(layout, path) for path in legacy_files]
    summaries.extend(_validate_phase4_bar_cache_file(layout, path) for path in phase4_files)
    manifest_summary = _validate_bar_manifest(layout)
    errors = [error for summary in summaries for error in summary["errors"]]
    errors.extend(manifest_summary["errors"])
    quality = _bar_cache_quality_summary(summaries)
    return {
        "schema": "historical_bar_cache_validation_v1",
        "status": "GO" if not errors else "NO_GO",
        "research_readiness_status": _bar_cache_research_readiness_status(errors, quality),
        "data_dir": str(layout.data_dir),
        "manifest": {
            "path": manifest_summary["path"],
            "exists": manifest_summary["exists"],
            "error_count": len(manifest_summary["errors"]),
        },
        **quality,
        "file_count": len(summaries),
        "row_count": sum(int(summary["row_count"]) for summary in summaries),
        "files": summaries,
        "errors": errors,
        "financial_calls": _zero_financial_calls(),
    }


def validate_bar_contract_identity_links(
    layout: BarCacheLayout,
    contract_rows: list[Any],
    *,
    bar_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = bar_validation or validate_bar_cache(layout)
    if validation["status"] != "GO":
        return {
            "schema": "historical_bar_contract_identity_link_validation_v1",
            "status": "NO_GO",
            "reason": "bar_cache_invalid",
            "referenced_count": 0,
            "contract_cache_row_count": len(contract_rows),
            "referenced_identities": [],
            "missing_identities": [],
            "errors": ["bar cache must validate before contract identity links can be checked"],
            "financial_calls": _zero_financial_calls(),
        }

    referenced = _referenced_bar_contract_identities(validation)
    known_identities = {_contract_row_identity(row) for row in contract_rows}
    missing = [identity for identity in referenced if (identity["con_id"], identity["sec_type"]) not in known_identities]
    errors = [
        f"bar cache references {identity['sec_type']} con_id {identity['con_id']} missing from local contract cache"
        for identity in missing
    ]
    return {
        "schema": "historical_bar_contract_identity_link_validation_v1",
        "status": "GO" if not errors else "NO_GO",
        "reason": "validated_against_phase2_contract_cache",
        "referenced_count": len(referenced),
        "contract_cache_row_count": len(contract_rows),
        "referenced_identities": referenced,
        "missing_identities": missing,
        "errors": errors,
        "financial_calls": _zero_financial_calls(),
    }


def bar_cache_manifest(layout: BarCacheLayout) -> dict[str, Any]:
    schema = bar_schema_manifest()
    return {
        "schema": "historical_bar_cache_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "layout": layout.as_dict(),
        "required_fields": list(BAR_CACHE_FIELDS),
        "supported_intervals": schema["supported_intervals"],
        "supported_data_types": schema["supported_data_types"],
        "supported_sources": schema["supported_sources"],
        "target_collection_fields": list(TARGET_BAR_COLLECTION_FIELDS),
        "request_policy": BarRequestPolicy().as_dict(),
        "data_phase_authority": "phase4_read_only_historical_ibkr_daily_stk",
        "phase4_scope": {
            "security_type": "STK",
            "interval": "1d",
            "data_type": "TRADES",
            "source": "IBKR",
            "realtime_data_enabled": False,
            "orders_enabled": False,
        },
        "financial_calls": _zero_financial_calls(),
    }


def initialize_bar_cache(layout: BarCacheLayout) -> dict[str, Any]:
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    manifest = bar_cache_manifest(layout)
    manifest["initialized"] = True
    layout.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest


def bar_schema_manifest() -> dict[str, Any]:
    return {
        "schema": "historical_bar_cache_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "supported_intervals": [item.value for item in BarInterval],
        "supported_data_types": [item.value for item in BarDataType],
        "supported_sources": [item.value for item in BarDataSource],
        "required_fields": list(BAR_CACHE_FIELDS),
        "target_collection_fields": list(TARGET_BAR_COLLECTION_FIELDS),
        "partitioning": [
            "sec_type",
            "con_id",
            "interval",
            "data_type",
            "source",
        ],
        "phase4_partitioning": [
            "security_type",
            "con_id",
            "interval",
            "data_type",
        ],
        "data_phase_authority": "phase4_read_only_historical_ibkr_daily_stk",
        "request_policy": BarRequestPolicy().as_dict(),
        "cache_layout": BarCacheLayout.from_project_root(Path(".")).as_dict(),
        "cache_quality_go_criteria": {
            "file_count_gt_0": True,
            "row_count_gt_0": True,
            "duplicate_rows": 0,
            "invalid_ohlc_rows": 0,
            "timezone_errors": 0,
            "contract_mismatches": 0,
        },
        "financial_calls": _zero_financial_calls(),
    }


def _expected_delta(interval: BarInterval) -> timedelta | None:
    if interval == BarInterval.FIFTEEN_MINUTES:
        return timedelta(minutes=15)
    if interval == BarInterval.ONE_HOUR:
        return timedelta(hours=1)
    return None


def _bar_availability_floor(timestamp_utc: datetime, interval: BarInterval) -> datetime:
    expected_delta = _expected_delta(interval)
    if expected_delta is None:
        return timestamp_utc.astimezone(UTC)
    return timestamp_utc.astimezone(UTC) + expected_delta


def _duplicate_timestamps(bars: list[HistoricalBar]) -> list[datetime]:
    seen: set[datetime] = set()
    duplicates: list[datetime] = []
    for bar in bars:
        timestamp = bar.timestamp_utc
        if timestamp in seen and timestamp not in duplicates:
            duplicates.append(timestamp)
        seen.add(timestamp)
    return duplicates


def _validate_bar_cache_file(layout: BarCacheLayout, path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        partition = _partition_metadata_from_path(layout, path)
    except ValueError as exc:
        return {
            "path": str(path),
            "row_count": 0,
            "errors": [str(exc)],
            "gap_report": None,
            "content_hash": _file_content_hash(path),
        }

    try:
        records = read_bar_cache_records(path)
    except Exception as exc:  # pragma: no cover - pyarrow exception type varies by platform.
        return {
            "path": str(path),
            "row_count": 0,
            "errors": [f"{path}: unreadable parquet: {exc}"],
            "gap_report": None,
            "content_hash": _file_content_hash(path),
        }

    if not records:
        return {
            "path": str(path),
            "row_count": 0,
            "partition": {
                "sec_type": partition["security_type"].value,
                "con_id": partition["con_id"],
                "interval": partition["interval"].value,
                "data_type": partition["data_type"].value,
                "source": partition["source"].value,
            },
            "errors": [f"{path}: bars.parquet must contain at least one bar record"],
            "gap_report": None,
            "content_hash": _file_content_hash(path),
        }

    bars: list[HistoricalBar] = []
    for index, record in enumerate(records):
        row_label = f"{path}[{index}]"
        try:
            _validate_required_bar_record_fields(record)
            bar = _bar_from_record(record)
            _validate_record_matches_partition(bar, partition)
            bars.append(bar)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{row_label}: {exc}")

    gap_report = None
    if not errors:
        ordering_errors = _bar_timestamp_order_errors(path, bars)
        errors.extend(ordering_errors)

    if not errors:
        gap_report = detect_bar_gaps(bars, interval=partition["interval"])
        if gap_report["status"] != "GO":
            for duplicate in gap_report["duplicate_timestamps"]:
                errors.append(f"{path}: duplicate timestamp {duplicate}")
            for gap in gap_report["gaps"]:
                errors.append(
                    f"{path}: gap after {gap['after']} before {gap['before']} "
                    f"missing_steps={gap['missing_steps']}"
                )

    return {
        "path": str(path),
        "row_count": len(records),
        "partition": {
            "sec_type": partition["security_type"].value,
            "con_id": partition["con_id"],
            "interval": partition["interval"].value,
            "data_type": partition["data_type"].value,
            "source": partition["source"].value,
        },
        "errors": errors,
        "gap_report": gap_report,
        "duplicate_rows": 0 if gap_report is None else len(gap_report["duplicate_timestamps"]),
        "invalid_ohlc_rows": _classified_error_count(errors, ("open", "high", "low", "close", "OHLC", "ohlc")),
        "timezone_errors": _classified_error_count(errors, ("timezone", "timestamp_utc", "available_at")),
        "first_timestamp": _first_bar_timestamp(bars),
        "last_timestamp": _last_bar_timestamp(bars),
        "content_hash": _file_content_hash(path),
    }


def _validate_phase4_bar_cache_file(layout: BarCacheLayout, path: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        partition = _phase4_partition_metadata_from_path(layout, path)
    except ValueError as exc:
        return {
            "path": str(path),
            "row_count": 0,
            "errors": [str(exc)],
            "gap_report": None,
            "content_hash": _file_content_hash(path),
        }

    try:
        records = read_bar_cache_records(path)
    except Exception as exc:  # pragma: no cover - pyarrow exception type varies by platform.
        return {
            "path": str(path),
            "row_count": 0,
            "errors": [f"{path}: unreadable parquet: {exc}"],
            "gap_report": None,
            "content_hash": _file_content_hash(path),
        }

    if not records:
        errors.append(f"{path}: bars.parquet must contain at least one bar record")

    timestamps: list[datetime] = []
    for index, record in enumerate(records):
        row_label = f"{path}[{index}]"
        try:
            _validate_required_phase4_bar_record_fields(record)
            _validate_phase4_record_matches_partition(record, partition)
            timestamps.append(_datetime_from_record(record["timestamp_utc"], "timestamp_utc").astimezone(UTC))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{row_label}: {exc}")

    duplicate_rows = len(timestamps) - len(set(timestamps))
    if duplicate_rows:
        errors.append(f"{path}: duplicate timestamp rows={duplicate_rows}")
    for previous, current in zip(timestamps, timestamps[1:]):
        if current < previous:
            errors.append(f"{path}: timestamp_utc values must be sorted ascending")
            break

    return {
        "path": str(path),
        "row_count": len(records),
        "partition": {
            "sec_type": partition["security_type"].value,
            "con_id": partition["con_id"],
            "interval": partition["interval"].value,
            "data_type": partition["data_type"].value,
            "source": BarDataSource.IBKR.value,
        },
        "phase4_schema": True,
        "errors": errors,
        "gap_report": {
            "schema": "historical_bar_gap_report_v1",
            "status": "GO" if duplicate_rows == 0 else "NO_GO",
            "interval": partition["interval"].value,
            "bar_count": len(records),
            "duplicate_timestamps": [],
            "gaps": [],
            "gap_detection_scope": "daily_requires_session_calendar",
            "financial_calls": _zero_financial_calls(),
        },
        "duplicate_rows": duplicate_rows,
        "invalid_ohlc_rows": _classified_error_count(errors, ("open", "high", "low", "close", "OHLC", "ohlc")),
        "timezone_errors": _classified_error_count(errors, ("timezone", "timestamp_utc", "received_at", "requested_at")),
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
        "content_hash": _file_content_hash(path),
    }


def _bar_cache_quality_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    identities: set[tuple[object, object]] = set()
    for summary in summaries:
        partition = summary.get("partition")
        if isinstance(partition, dict):
            identities.add((partition["sec_type"], partition["con_id"]))
    first_timestamps = [summary["first_timestamp"] for summary in summaries if summary.get("first_timestamp")]
    last_timestamps = [summary["last_timestamp"] for summary in summaries if summary.get("last_timestamp")]
    duplicate_rows = sum(int(summary.get("duplicate_rows", 0)) for summary in summaries)
    invalid_ohlc_rows = sum(int(summary.get("invalid_ohlc_rows", 0)) for summary in summaries)
    timezone_errors = sum(int(summary.get("timezone_errors", 0)) for summary in summaries)
    missing_sessions = sum(_gap_missing_steps(summary.get("gap_report")) for summary in summaries)
    content_hashes = [str(summary["content_hash"]) for summary in summaries if summary.get("content_hash")]
    return {
        "instrument_count": len(identities),
        "duplicate_rows": duplicate_rows,
        "invalid_ohlc_rows": invalid_ohlc_rows,
        "missing_sessions": missing_sessions,
        "unexpected_sessions": 0,
        "stale_instruments": 0,
        "timezone_errors": timezone_errors,
        "currency_errors": 0,
        "contract_mismatches": 0,
        "first_timestamp": min(first_timestamps) if first_timestamps else None,
        "last_timestamp": max(last_timestamps) if last_timestamps else None,
        "content_hash": _combined_content_hash(content_hashes),
    }


def _bar_cache_research_readiness_status(errors: list[str], quality: dict[str, Any]) -> str:
    if errors:
        return "NO_GO"
    if int(quality["instrument_count"]) <= 0 or quality["content_hash"] is None:
        return "NO_DATA"
    if (
        quality["duplicate_rows"] != 0
        or quality["invalid_ohlc_rows"] != 0
        or quality["timezone_errors"] != 0
        or quality["contract_mismatches"] != 0
    ):
        return "NO_GO"
    return "GO"


def _gap_missing_steps(gap_report: dict[str, Any] | None) -> int:
    if not gap_report:
        return 0
    return sum(int(gap["missing_steps"]) for gap in gap_report.get("gaps", []))


def _classified_error_count(errors: list[str], needles: tuple[str, ...]) -> int:
    return sum(1 for error in errors if any(needle in error for needle in needles))


def _first_bar_timestamp(bars: list[HistoricalBar]) -> str | None:
    if not bars:
        return None
    return min(bar.timestamp_utc for bar in bars).astimezone(UTC).isoformat()


def _last_bar_timestamp(bars: list[HistoricalBar]) -> str | None:
    if not bars:
        return None
    return max(bar.timestamp_utc for bar in bars).astimezone(UTC).isoformat()


def _file_content_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _combined_content_hash(content_hashes: list[str]) -> str | None:
    if not content_hashes:
        return None
    text = "\n".join(sorted(content_hashes))
    return hashlib.sha256(text.encode("ascii")).hexdigest().upper()


def _is_sha256_hex(value: str) -> bool:
    return bool(re.fullmatch(r"[A-F0-9]{64}", value))


def _bar_timestamp_order_errors(path: Path, bars: list[HistoricalBar]) -> list[str]:
    errors: list[str] = []
    for previous, current in zip(bars, bars[1:]):
        if current.timestamp_utc < previous.timestamp_utc:
            errors.append(
                f"{path}: timestamp_utc values must be sorted ascending; "
                f"{current.timestamp_utc.isoformat()} appears after {previous.timestamp_utc.isoformat()}"
            )
    return errors


def _referenced_bar_contract_identities(validation: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    for summary in validation["files"]:
        partition = summary.get("partition")
        if not partition:
            continue
        key = (int(partition["con_id"]), str(partition["sec_type"]))
        item = grouped.setdefault(
            key,
            {
                "con_id": key[0],
                "sec_type": key[1],
                "intervals": set(),
                "data_types": set(),
                "sources": set(),
                "file_count": 0,
                "row_count": 0,
            },
        )
        item["intervals"].add(str(partition["interval"]))
        item["data_types"].add(str(partition["data_type"]))
        item["sources"].add(str(partition["source"]))
        item["file_count"] += 1
        item["row_count"] += int(summary["row_count"])

    return [
        {
            "con_id": item["con_id"],
            "sec_type": item["sec_type"],
            "intervals": sorted(item["intervals"]),
            "data_types": sorted(item["data_types"]),
            "sources": sorted(item["sources"]),
            "file_count": item["file_count"],
            "row_count": item["row_count"],
        }
        for item in sorted(grouped.values(), key=lambda item: (item["sec_type"], item["con_id"]))
    ]


def _contract_row_identity(row: Any) -> tuple[int, str]:
    security_type = row.contract.security_type
    return int(row.contract.con_id), str(getattr(security_type, "value", security_type))


def _validate_bar_manifest(layout: BarCacheLayout) -> dict[str, Any]:
    path = layout.manifest_json
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "errors": [],
        }

    errors: list[str] = []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "path": str(path),
            "exists": True,
            "errors": [f"{path.name}: invalid JSON: {exc.msg}"],
        }

    if manifest.get("schema") != "historical_bar_cache_manifest_v1":
        errors.append(f"{path.name}: invalid schema {manifest.get('schema')}")
    if manifest.get("layout") != layout.as_dict():
        errors.append(f"{path.name}: layout does not match expected bar cache layout")
    if manifest.get("required_fields") != list(BAR_CACHE_FIELDS):
        errors.append(f"{path.name}: required_fields do not match bar schema")
    if manifest.get("supported_intervals") != [item.value for item in BarInterval]:
        errors.append(f"{path.name}: supported_intervals do not match bar schema")
    if manifest.get("supported_data_types") != [item.value for item in BarDataType]:
        errors.append(f"{path.name}: supported_data_types do not match bar schema")
    if manifest.get("supported_sources") != [item.value for item in BarDataSource]:
        errors.append(f"{path.name}: supported_sources do not match bar schema")
    if manifest.get("request_policy") != BarRequestPolicy().as_dict():
        errors.append(f"{path.name}: request_policy does not match Phase 4 policy")
    if manifest.get("data_phase_authority") != "phase4_read_only_historical_ibkr_daily_stk":
        errors.append(f"{path.name}: data_phase_authority must be phase4_read_only_historical_ibkr_daily_stk")
    if manifest.get("financial_calls") != _zero_financial_calls():
        errors.append(f"{path.name}: financial_calls must all be 0")

    return {
        "path": str(path),
        "exists": True,
        "errors": errors,
    }


def _partition_metadata_from_path(layout: BarCacheLayout, path: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(layout.data_dir)
    except ValueError as exc:
        raise ValueError(f"{path}: bar cache path is outside data_dir") from exc
    parts = relative.parts
    if len(parts) != 6 or parts[-1] != "bars.parquet":
        raise ValueError(f"{path}: invalid bar cache partition path")
    values = dict(_partition_part(part) for part in parts[:-1])
    expected_keys = ["sec_type", "con_id", "interval", "data_type", "source"]
    if list(values) != expected_keys:
        raise ValueError(f"{path}: partition keys must be {', '.join(expected_keys)}")
    con_id = _positive_int_from_record(values["con_id"], "partition con_id")
    return {
        "security_type": _enum_from_record(IbkrSecurityType, values["sec_type"], "partition sec_type"),
        "con_id": con_id,
        "interval": _enum_from_record(BarInterval, values["interval"], "partition interval"),
        "data_type": _enum_from_record(BarDataType, values["data_type"], "partition data_type"),
        "source": _enum_from_record(BarDataSource, values["source"], "partition source"),
    }


def _phase4_partition_metadata_from_path(layout: BarCacheLayout, path: Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(layout.data_dir)
    except ValueError as exc:
        raise ValueError(f"{path}: bar cache path is outside data_dir") from exc
    parts = relative.parts
    if len(parts) != 5 or parts[-1] != "bars.parquet":
        raise ValueError(f"{path}: invalid Phase 4 bar cache partition path")
    values = dict(_partition_part(part) for part in parts[:-1])
    expected_keys = ["security_type", "con_id", "interval", "data_type"]
    if list(values) != expected_keys:
        raise ValueError(f"{path}: partition keys must be {', '.join(expected_keys)}")
    security_type = _enum_from_record(IbkrSecurityType, values["security_type"], "partition security_type")
    interval = _enum_from_record(BarInterval, values["interval"], "partition interval")
    data_type = _enum_from_record(BarDataType, values["data_type"], "partition data_type")
    if security_type != IbkrSecurityType.STK or interval != BarInterval.ONE_DAY or data_type != BarDataType.TRADES:
        raise ValueError(f"{path}: Phase 4 V1 only supports STK 1d TRADES")
    return {
        "security_type": security_type,
        "con_id": _positive_int_from_record(values["con_id"], "partition con_id"),
        "interval": interval,
        "data_type": data_type,
    }


def _partition_part(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"invalid partition segment: {value}")
    key, raw = value.split("=", 1)
    if not key or not raw:
        raise ValueError(f"invalid partition segment: {value}")
    return key, raw


def _validate_required_bar_record_fields(record: dict[str, Any]) -> None:
    extra_fields = sorted(set(record) - set(BAR_CACHE_FIELDS))
    if extra_fields:
        raise ValueError(f"unexpected bar fields: {', '.join(extra_fields)}")
    for field in BAR_CACHE_FIELDS:
        if field not in record:
            raise ValueError(f"missing bar field {field}")
        if record[field] is None and field != "volume":
            raise ValueError(f"missing value {field}")


def _validate_required_phase4_bar_record_fields(record: dict[str, Any]) -> None:
    extra_fields = sorted(set(record) - set(TARGET_BAR_COLLECTION_FIELDS))
    if extra_fields:
        raise ValueError(f"unexpected Phase 4 bar fields: {', '.join(extra_fields)}")
    for field in TARGET_BAR_COLLECTION_FIELDS:
        if field not in record:
            raise ValueError(f"missing Phase 4 bar field {field}")
        if record[field] is None and field not in {"volume", "wap", "bar_count"}:
            raise ValueError(f"missing value {field}")
    _positive_int_from_record(record["con_id"], "con_id")
    _enum_from_record(IbkrSecurityType, record["security_type"], "security_type")
    _enum_from_record(BarInterval, record["interval"], "interval")
    _enum_from_record(BarDataType, record["data_type"], "data_type")
    _enum_from_record(BarDataSource, record["source"], "source")
    if not isinstance(record["use_rth"], bool):
        raise ValueError("use_rth must be boolean")
    for field_name in ("open", "high", "low", "close"):
        value = _decimal_from_record(record[field_name], field_name)
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
    open_ = _decimal_from_record(record["open"], "open")
    high = _decimal_from_record(record["high"], "high")
    low = _decimal_from_record(record["low"], "low")
    close = _decimal_from_record(record["close"], "close")
    if high < max(open_, low, close):
        raise ValueError("high must be greater than or equal to open, low and close")
    if low > min(open_, high, close):
        raise ValueError("low must be less than or equal to open, high and close")
    _volume_from_record(record.get("volume"))
    _datetime_from_record(record["timestamp_utc"], "timestamp_utc")
    _datetime_from_record(record["requested_at"], "requested_at")
    _datetime_from_record(record["received_at"], "received_at")
    for field_name in ("contract_hash", "session_hash", "request_hash", "content_hash"):
        if not _is_sha256_hex(str(record[field_name])):
            raise ValueError(f"{field_name} must be a 64-character uppercase SHA256 hex digest")


def _validate_phase4_record_matches_partition(record: dict[str, Any], partition: dict[str, Any]) -> None:
    if _enum_from_record(IbkrSecurityType, record["security_type"], "security_type") != partition["security_type"]:
        raise ValueError("record security_type does not match partition")
    if _positive_int_from_record(record["con_id"], "con_id") != partition["con_id"]:
        raise ValueError("record con_id does not match partition")
    if _enum_from_record(BarInterval, record["interval"], "interval") != partition["interval"]:
        raise ValueError("record interval does not match partition")
    if _enum_from_record(BarDataType, record["data_type"], "data_type") != partition["data_type"]:
        raise ValueError("record data_type does not match partition")
    if _enum_from_record(BarDataSource, record["source"], "source") != BarDataSource.IBKR:
        raise ValueError("record source must be IBKR for Phase 4")


def _bar_from_record(record: dict[str, Any]) -> HistoricalBar:
    return HistoricalBar(
        con_id=_positive_int_from_record(record["con_id"], "con_id"),
        security_type=_enum_from_record(IbkrSecurityType, record["sec_type"], "sec_type"),
        interval=_enum_from_record(BarInterval, record["interval"], "interval"),
        data_type=_enum_from_record(BarDataType, record["data_type"], "data_type"),
        source=_enum_from_record(BarDataSource, record["source"], "source"),
        timestamp_utc=_datetime_from_record(record["timestamp_utc"], "timestamp_utc"),
        open=_decimal_from_record(record["open"], "open"),
        high=_decimal_from_record(record["high"], "high"),
        low=_decimal_from_record(record["low"], "low"),
        close=_decimal_from_record(record["close"], "close"),
        volume=_volume_from_record(record.get("volume")),
        available_at=_datetime_from_record(record["available_at"], "available_at"),
    )


def _validate_record_matches_partition(bar: HistoricalBar, partition: dict[str, Any]) -> None:
    bar.validate()
    if bar.security_type != partition["security_type"]:
        raise ValueError("record sec_type does not match partition")
    if bar.con_id != partition["con_id"]:
        raise ValueError("record con_id does not match partition")
    if bar.interval != partition["interval"]:
        raise ValueError("record interval does not match partition")
    if bar.data_type != partition["data_type"]:
        raise ValueError("record data_type does not match partition")
    if bar.source != partition["source"]:
        raise ValueError("record source does not match partition")


def _validate_single_bar_identity(bars: list[HistoricalBar]) -> None:
    first = bars[0]
    for bar in bars[1:]:
        if (
            bar.con_id != first.con_id
            or bar.security_type != first.security_type
            or bar.interval != first.interval
            or bar.data_type != first.data_type
            or bar.source != first.source
        ):
            raise ValueError("all bars in one cache file must share con_id, sec_type, interval, data_type and source")


def _datetime_from_record(value: object, field_name: str) -> datetime:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a parseable ISO-8601 datetime")
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value)
        if text != text.strip():
            raise ValueError(f"{field_name} must be a parseable ISO-8601 datetime")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a parseable ISO-8601 datetime") from exc
    _validate_timezone_aware_datetime(parsed, field_name)
    return parsed


def _datetime_value(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    _validate_timezone_aware_datetime(value, field_name)
    return value


def _datetime_report_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return value


def _enum_from_record(enum_type: type[Enum], value: object, field_name: str) -> Any:
    allowed = ", ".join(str(item.value) for item in enum_type)
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be one of: {allowed}")
    text = str(value)
    if text != text.strip():
        raise ValueError(f"{field_name} must be one of: {allowed}")
    try:
        return enum_type(text)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _enum_value(enum_type: type[Enum], value: object, field_name: str) -> Any:
    allowed = ", ".join(str(item.value) for item in enum_type)
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be one of: {allowed}")
    return value


def _enum_report_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    return value


def _decimal_from_record(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a parseable decimal")
    text = str(value)
    if text != text.strip():
        raise ValueError(f"{field_name} must be a parseable decimal")
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a parseable decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return parsed


def _decimal_value(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be a Decimal")
    return value


def _volume_from_record(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("volume must be null or a non-negative integer")
    text = str(value)
    if text != text.strip() or not text.strip().lstrip("-").isdigit():
        raise ValueError("volume must be null or a non-negative integer")
    parsed = int(text)
    if parsed < 0:
        raise ValueError("volume must be null or a non-negative integer")
    return parsed


def _volume_value(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("volume must be null or a non-negative integer")
    return value


def _positive_int_from_record(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    text = str(value)
    if text != text.strip() or not text.strip().isdigit():
        raise ValueError(f"{field_name} must be a positive integer")
    parsed = int(text)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def _positive_int_value(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _int_policy_value(value: object, field_name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        if minimum == 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _retry_backoff_policy_value(value: object) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, tuple):
        raise ValueError("retry_backoff_seconds must be a tuple of positive integers")
    for delay in value:
        _int_policy_value(delay, "retry_backoff_seconds value")
    return value


def _validate_timezone_aware_datetime(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _zero_financial_calls() -> dict[str, int]:
    return {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }
