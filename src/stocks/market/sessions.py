from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from zoneinfo import ZoneInfo

from stocks.ibkr.contract_cache import ContractCacheRow
from stocks.ibkr.timezones import parse_ibkr_timezone_id
from stocks.ibkr.trading_hours import IbkrSessionWindow, parse_ibkr_hours
from stocks.market.calendars import calendar_session_check


SESSION_CACHE_FIELDS = (
    "con_id",
    "symbol",
    "security_type",
    "exchange",
    "primary_exchange",
    "currency",
    "timezone_id",
    "session_date",
    "session_type",
    "session_open_utc",
    "session_close_utc",
    "liquid_open_utc",
    "liquid_close_utc",
    "ibkr_trading_open_utc",
    "ibkr_trading_close_utc",
    "ibkr_liquid_open_utc",
    "ibkr_liquid_close_utc",
    "calendar_open_utc",
    "calendar_close_utc",
    "effective_collection_open_utc",
    "effective_collection_close_utc",
    "effective_source",
    "conflict_minutes",
    "conflict_classification",
    "is_regular_session",
    "is_liquid_session",
    "is_overnight",
    "is_early_close",
    "is_holiday",
    "source",
    "contract_hash",
    "session_hash",
    "session_status",
    "readiness",
)


SESSION_STATUSES = (
    "OPEN_LIQUID",
    "OPEN_EXTENDED",
    "CLOSED",
    "HOLIDAY",
    "EARLY_CLOSE",
    "MAINTENANCE_BREAK",
    "SESSION_NOT_FOUND",
    "TIMEZONE_INVALID",
    "HOURS_PARSE_ERROR",
    "CALENDAR_CONFLICT",
    "CONTRACT_NOT_RESOLVED",
    "STALE_CACHE",
)

_SESSION_CONFLICT_TOLERANCE_MINUTES = 10


@dataclass(frozen=True)
class MarketWindow:
    session_date: date
    start: datetime
    end: datetime
    source: str

    def contains(self, value: datetime) -> bool:
        return self.start <= value < self.end

    def as_dict(self) -> dict[str, str]:
        return {
            "session_date": self.session_date.isoformat(),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "source": self.source,
        }


@dataclass(frozen=True)
class MarketSessionRecord:
    con_id: int
    symbol: str
    security_type: str
    exchange: str
    primary_exchange: str | None
    currency: str
    timezone_id: str
    session_date: date
    session_open_utc: datetime | None
    session_close_utc: datetime | None
    liquid_open_utc: datetime | None
    liquid_close_utc: datetime | None
    ibkr_trading_open_utc: datetime | None
    ibkr_trading_close_utc: datetime | None
    ibkr_liquid_open_utc: datetime | None
    ibkr_liquid_close_utc: datetime | None
    calendar_open_utc: datetime | None
    calendar_close_utc: datetime | None
    effective_collection_open_utc: datetime | None
    effective_collection_close_utc: datetime | None
    effective_source: str
    conflict_minutes: int | None
    conflict_classification: str
    is_regular_session: bool
    is_liquid_session: bool
    is_overnight: bool
    is_early_close: bool
    is_holiday: bool
    source: str
    contract_hash: str
    session_type: str = "PRIMARY"
    session_status: str = "CLOSED"
    readiness: str = "SESSION_READY"

    @property
    def session_hash(self) -> str:
        return session_hash(
            con_id=self.con_id,
            session_date=self.session_date,
            session_open_utc=self.session_open_utc,
            session_close_utc=self.session_close_utc,
            liquid_open_utc=self.liquid_open_utc,
            liquid_close_utc=self.liquid_close_utc,
            timezone_id=self.timezone_id,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "con_id": self.con_id,
            "symbol": self.symbol,
            "security_type": self.security_type,
            "exchange": self.exchange,
            "primary_exchange": self.primary_exchange,
            "currency": self.currency,
            "timezone_id": self.timezone_id,
            "session_date": self.session_date.isoformat(),
            "session_type": self.session_type,
            "session_open_utc": _datetime_utc_iso(self.session_open_utc),
            "session_close_utc": _datetime_utc_iso(self.session_close_utc),
            "liquid_open_utc": _datetime_utc_iso(self.liquid_open_utc),
            "liquid_close_utc": _datetime_utc_iso(self.liquid_close_utc),
            "ibkr_trading_open_utc": _datetime_utc_iso(self.ibkr_trading_open_utc),
            "ibkr_trading_close_utc": _datetime_utc_iso(self.ibkr_trading_close_utc),
            "ibkr_liquid_open_utc": _datetime_utc_iso(self.ibkr_liquid_open_utc),
            "ibkr_liquid_close_utc": _datetime_utc_iso(self.ibkr_liquid_close_utc),
            "calendar_open_utc": _datetime_utc_iso(self.calendar_open_utc),
            "calendar_close_utc": _datetime_utc_iso(self.calendar_close_utc),
            "effective_collection_open_utc": _datetime_utc_iso(self.effective_collection_open_utc),
            "effective_collection_close_utc": _datetime_utc_iso(self.effective_collection_close_utc),
            "effective_source": self.effective_source,
            "conflict_minutes": self.conflict_minutes,
            "conflict_classification": self.conflict_classification,
            "is_regular_session": self.is_regular_session,
            "is_liquid_session": self.is_liquid_session,
            "is_overnight": self.is_overnight,
            "is_early_close": self.is_early_close,
            "is_holiday": self.is_holiday,
            "source": self.source,
            "contract_hash": self.contract_hash,
            "session_hash": self.session_hash,
            "session_status": self.session_status,
            "readiness": self.readiness,
        }

    def parquet_record(self) -> dict[str, Any]:
        return {
            **self.as_dict(),
            "session_date": self.session_date,
            "session_open_utc": self.session_open_utc.astimezone(UTC) if self.session_open_utc else None,
            "session_close_utc": self.session_close_utc.astimezone(UTC) if self.session_close_utc else None,
            "liquid_open_utc": self.liquid_open_utc.astimezone(UTC) if self.liquid_open_utc else None,
            "liquid_close_utc": self.liquid_close_utc.astimezone(UTC) if self.liquid_close_utc else None,
            "ibkr_trading_open_utc": self.ibkr_trading_open_utc.astimezone(UTC)
            if self.ibkr_trading_open_utc
            else None,
            "ibkr_trading_close_utc": self.ibkr_trading_close_utc.astimezone(UTC)
            if self.ibkr_trading_close_utc
            else None,
            "ibkr_liquid_open_utc": self.ibkr_liquid_open_utc.astimezone(UTC)
            if self.ibkr_liquid_open_utc
            else None,
            "ibkr_liquid_close_utc": self.ibkr_liquid_close_utc.astimezone(UTC)
            if self.ibkr_liquid_close_utc
            else None,
            "calendar_open_utc": self.calendar_open_utc.astimezone(UTC) if self.calendar_open_utc else None,
            "calendar_close_utc": self.calendar_close_utc.astimezone(UTC) if self.calendar_close_utc else None,
            "effective_collection_open_utc": self.effective_collection_open_utc.astimezone(UTC)
            if self.effective_collection_open_utc
            else None,
            "effective_collection_close_utc": self.effective_collection_close_utc.astimezone(UTC)
            if self.effective_collection_close_utc
            else None,
        }


@dataclass(frozen=True)
class SessionCacheLayout:
    data_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> SessionCacheLayout:
        return cls(data_dir=project_root / "data" / "sessions")

    @property
    def sessions_parquet(self) -> Path:
        return self.data_dir / "sessions.parquet"

    @property
    def manifest_json(self) -> Path:
        return self.data_dir / "session_manifest.json"

    @property
    def conflicts_jsonl(self) -> Path:
        return self.data_dir / "session_conflicts.jsonl"

    @property
    def errors_jsonl(self) -> Path:
        return self.data_dir / "session_errors.jsonl"

    def as_dict(self) -> dict[str, str]:
        return {
            "data_dir": str(self.data_dir),
            "sessions_parquet": str(self.sessions_parquet),
            "session_manifest_json": str(self.manifest_json),
            "session_conflicts_jsonl": str(self.conflicts_jsonl),
            "session_errors_jsonl": str(self.errors_jsonl),
        }


def session_hash(
    *,
    con_id: int,
    session_date: date,
    session_open_utc: datetime | None,
    session_close_utc: datetime | None,
    liquid_open_utc: datetime | None,
    liquid_close_utc: datetime | None,
    timezone_id: str,
) -> str:
    raw = "|".join(
        (
            str(con_id),
            session_date.isoformat(),
            _datetime_hash_value(session_open_utc),
            _datetime_hash_value(session_close_utc),
            _datetime_hash_value(liquid_open_utc),
            _datetime_hash_value(liquid_close_utc),
            timezone_id,
        )
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


def market_session_schema_manifest() -> dict[str, Any]:
    return {
        "schema": "market_session_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "required_fields": list(SESSION_CACHE_FIELDS),
        "primary_key": ["con_id", "session_date", "session_type"],
        "statuses": list(SESSION_STATUSES),
        "readiness_values": ["SESSION_READY", "SESSION_DEGRADED", "SESSION_BLOCKED"],
        "source_priority": ["IBKR_CONTRACT_CACHE", "EXCHANGE_CALENDAR_VALIDATION"],
        "cache_layout": SessionCacheLayout.from_project_root(Path(".")).as_dict(),
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def market_status_report(row: ContractCacheRow, *, at: datetime | None = None) -> dict[str, Any]:
    row.contract.validate_phase2_required_fields()
    effective_at = _effective_datetime(at)
    timezone = parse_ibkr_timezone_id(row.contract.time_zone_id or "")
    local_at = effective_at.astimezone(timezone)

    trading_windows = market_windows_from_ibkr_hours(
        row.contract.trading_hours or "",
        timezone=timezone,
        source="tradingHours",
    )
    liquid_windows = market_windows_from_ibkr_hours(
        row.contract.liquid_hours or "",
        timezone=timezone,
        source="liquidHours",
    )
    active_trading = _active_window(trading_windows, local_at)
    active_liquid = _active_window(liquid_windows, local_at)
    next_trading = _next_window(trading_windows, local_at)
    trading_coverage = _window_coverage(trading_windows)
    liquid_coverage = _window_coverage(liquid_windows)
    session_status = _current_session_status(
        local_at=local_at,
        active_trading=active_trading,
        active_liquid=active_liquid,
        trading_windows=trading_windows,
        liquid_windows=liquid_windows,
        row=row,
    )

    return {
        "schema": "market_status_v1",
        "status": "GO",
        "session_status": session_status,
        "readiness": "SESSION_READY",
        "source": "local_contract_cache",
        "contract": _contract_payload(row),
        "timeZoneId": row.contract.time_zone_id,
        "evaluated_at_utc": effective_at.astimezone(UTC).isoformat(),
        "evaluated_at_local": local_at.isoformat(),
        "market_state": "OPEN" if active_trading else "CLOSED",
        "is_trading_open": active_trading is not None,
        "is_liquid_open": active_liquid is not None,
        "active_trading_window": active_trading.as_dict() if active_trading else None,
        "active_liquid_window": active_liquid.as_dict() if active_liquid else None,
        "next_trading_open": next_trading.as_dict() if next_trading else None,
        "known_trading_window_count": len(trading_windows),
        "known_liquid_window_count": len(liquid_windows),
        "known_trading_coverage_start": trading_coverage["start"],
        "known_trading_coverage_end": trading_coverage["end"],
        "known_liquid_coverage_start": liquid_coverage["start"],
        "known_liquid_coverage_end": liquid_coverage["end"],
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def market_next_open_report(row: ContractCacheRow, *, at: datetime | None = None) -> dict[str, Any]:
    report = market_status_report(row, at=at)
    return {
        "schema": "market_next_open_v1",
        "status": "GO" if report["next_trading_open"] else "NO_KNOWN_NEXT_OPEN",
        "source": report["source"],
        "contract": report["contract"],
        "timeZoneId": report["timeZoneId"],
        "evaluated_at_utc": report["evaluated_at_utc"],
        "evaluated_at_local": report["evaluated_at_local"],
        "next_trading_open": report["next_trading_open"],
        "reason": None
        if report["next_trading_open"]
        else "no future trading window in local contract cache",
        "known_trading_window_count": report["known_trading_window_count"],
        "known_trading_coverage_start": report["known_trading_coverage_start"],
        "known_trading_coverage_end": report["known_trading_coverage_end"],
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def market_sessions_report(row: ContractCacheRow, *, session_date: date) -> dict[str, Any]:
    try:
        return _market_sessions_report(row, session_date=session_date)
    except ValueError as exc:
        status = _session_error_status(str(exc))
        return {
            "schema": "market_sessions_v1",
            "status": status,
            "readiness": "SESSION_BLOCKED",
            "source": "local_contract_cache",
            "contract": _contract_payload(row, validate=False),
            "timeZoneId": row.contract.time_zone_id,
            "session_date": session_date.isoformat(),
            "reason": str(exc),
            "session": None,
            "calendar_check": None,
            "calendar_conflicts": [],
            "financial_calls": _zero_financial_calls(),
            "market_data_calls": 0,
            "historical_data_calls": 0,
        }


def _market_sessions_report(row: ContractCacheRow, *, session_date: date) -> dict[str, Any]:
    row.contract.validate_phase2_required_fields()
    if row.is_stale():
        return _blocked_session_report(row, session_date=session_date, status="STALE_CACHE")

    timezone = parse_ibkr_timezone_id(row.contract.time_zone_id or "")
    parsed_trading_hours = parse_ibkr_hours(row.contract.trading_hours or "")
    parsed_liquid_hours = parse_ibkr_hours(row.contract.liquid_hours or "")
    trading_windows = tuple(
        window
        for window in market_windows_from_ibkr_hours(
            parsed_trading_hours,
            timezone=timezone,
            source="tradingHours",
        )
        if window.session_date == session_date
    )
    liquid_windows = tuple(
        window
        for window in market_windows_from_ibkr_hours(
            parsed_liquid_hours,
            timezone=timezone,
            source="liquidHours",
        )
        if window.session_date == session_date
    )
    calendar_check = calendar_session_check(
        row.contract,
        session_date=session_date,
        ibkr_has_trading_window=bool(trading_windows),
    )
    conflicts = _calendar_conflicts(calendar_check.as_dict(), liquid_windows)
    record = _session_record_from_windows(
        row,
        session_date=session_date,
        trading_windows=trading_windows,
        liquid_windows=liquid_windows,
        trading_closed=_is_closed_for_date(parsed_trading_hours, session_date),
        liquid_closed=_is_closed_for_date(parsed_liquid_hours, session_date),
        calendar_check=calendar_check.as_dict(),
        conflicts=conflicts,
    )

    return {
        "schema": "market_sessions_v1",
        "status": "GO",
        "session_status": record.session_status,
        "readiness": record.readiness,
        "source": "local_contract_cache",
        "contract": _contract_payload(row),
        "timeZoneId": row.contract.time_zone_id,
        "session_date": session_date.isoformat(),
        "is_trading_day": bool(trading_windows),
        "is_liquid_day": bool(liquid_windows),
        "trading_closed": _is_closed_for_date(parsed_trading_hours, session_date),
        "liquid_closed": _is_closed_for_date(parsed_liquid_hours, session_date),
        "trading_windows": [window.as_dict() for window in trading_windows],
        "liquid_windows": [window.as_dict() for window in liquid_windows],
        "session": record.as_dict(),
        "calendar_check": calendar_check.as_dict(),
        "calendar_conflicts": conflicts,
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def market_sessions_by_date_report(rows: list[ContractCacheRow], *, session_date: date) -> dict[str, Any]:
    contract_reports = [
        market_sessions_report(row, session_date=session_date)
        for row in sorted(rows, key=lambda item: item.contract.con_id)
    ]
    return {
        "schema": "market_sessions_by_date_v1",
        "status": "GO",
        "source": "local_contract_cache",
        "session_date": session_date.isoformat(),
        "contract_count": len(contract_reports),
        "contracts": [
            {
                "contract": report["contract"],
                "timeZoneId": report["timeZoneId"],
                "status": report["status"],
                "session_status": report.get("session_status"),
                "readiness": report.get("readiness"),
                "is_trading_day": report.get("is_trading_day"),
                "is_liquid_day": report.get("is_liquid_day"),
                "trading_closed": report.get("trading_closed"),
                "liquid_closed": report.get("liquid_closed"),
                "trading_windows": report.get("trading_windows", []),
                "liquid_windows": report.get("liquid_windows", []),
                "session": report.get("session"),
                "calendar_check": report.get("calendar_check"),
                "calendar_conflicts": report.get("calendar_conflicts", []),
            }
            for report in contract_reports
        ],
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def market_sessions_range_report(
    row: ContractCacheRow,
    *,
    start: date,
    end: date,
) -> dict[str, Any]:
    if end < start:
        raise ValueError("end date must be on or after start date")
    reports = [
        market_sessions_report(row, session_date=session_date)
        for session_date in _date_range(start, end)
    ]
    return {
        "schema": "market_sessions_range_v1",
        "status": "GO",
        "source": "local_contract_cache",
        "contract": _contract_payload(row, validate=False),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "session_count": len(reports),
        "sessions": reports,
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def build_session_cache_from_contract_rows(
    layout: SessionCacheLayout,
    rows: list[ContractCacheRow],
) -> dict[str, Any]:
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    records: list[MarketSessionRecord] = []
    conflicts: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for row in sorted(rows, key=lambda item: item.contract.con_id):
        for session_date in _known_session_dates(row):
            report = market_sessions_report(row, session_date=session_date)
            session = report.get("session")
            if session is None:
                errors.append(_session_audit_record(row, session_date, report))
                continue
            records.append(_record_from_report_session(session))
            for conflict in report.get("calendar_conflicts", []):
                conflicts.append(
                    {
                        "schema": "market_session_conflict_v1",
                        "con_id": row.contract.con_id,
                        "symbol": row.contract.symbol,
                        "session_date": session_date.isoformat(),
                        "contract_hash": row.contract_hash,
                        "conflict": conflict,
                    }
                )

    _write_session_records(layout.sessions_parquet, records)
    _write_jsonl(layout.conflicts_jsonl, conflicts)
    _write_jsonl(layout.errors_jsonl, errors)
    manifest = session_cache_manifest(layout, record_count=len(records), conflict_count=len(conflicts), error_count=len(errors))
    layout.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )

    validation = validate_session_cache(layout, contract_rows=rows)
    return {
        "schema": "market_session_cache_build_v1",
        "status": validation["status"],
        "cache": manifest,
        "cache_validation": validation,
        "written_records": len(records),
        "conflict_count": len(conflicts),
        "error_count": len(errors),
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def session_cache_manifest(
    layout: SessionCacheLayout,
    *,
    record_count: int = 0,
    conflict_count: int = 0,
    error_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema": "market_session_cache_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "layout": layout.as_dict(),
        "required_fields": list(SESSION_CACHE_FIELDS),
        "primary_key": ["con_id", "session_date", "session_type"],
        "record_count": record_count,
        "conflict_count": conflict_count,
        "error_count": error_count,
        "source": "IBKR_CONTRACT_CACHE",
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def validate_session_cache(
    layout: SessionCacheLayout,
    *,
    contract_rows: list[ContractCacheRow] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    file_count = 0
    records: list[dict[str, Any]] = []
    if not layout.sessions_parquet.exists():
        errors.append(f"{layout.sessions_parquet}: missing sessions parquet")
    else:
        file_count += 1
        try:
            records = pq.read_table(layout.sessions_parquet).to_pylist()
        except Exception as exc:  # pragma: no cover - pyarrow exception varies by platform.
            errors.append(f"{layout.sessions_parquet}: unreadable parquet: {exc}")

    for path in (layout.manifest_json, layout.conflicts_jsonl, layout.errors_jsonl):
        if path.exists():
            file_count += 1
        else:
            errors.append(f"{path}: missing session cache artifact")

    required = set(SESSION_CACHE_FIELDS)
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            errors.append(f"sessions.parquet[{index}]: missing fields {', '.join(missing)}")
        try:
            _validate_cached_session_record(record)
        except ValueError as exc:
            errors.append(f"sessions.parquet[{index}]: {exc}")

    duplicate_rows = _duplicate_primary_key_count(records)
    if duplicate_rows:
        errors.append(f"sessions.parquet: duplicate primary keys={duplicate_rows}")

    contract_mismatches = _contract_mismatch_count(records, contract_rows or [])
    if contract_mismatches:
        errors.append(f"sessions.parquet: contract hash mismatches={contract_mismatches}")

    content_hash = _file_content_hash(layout.sessions_parquet)
    parsed_dates = [
        _date_record_value(record.get("session_date")) for record in records
    ]
    first_dates = [item for item in parsed_dates if item is not None]
    timezone_errors = _classified_error_count(errors, ("timezone", "timezone_id", "TIMEZONE"))
    quality = {
        "instrument_count": len({record.get("con_id") for record in records if record.get("con_id") is not None}),
        "file_count": file_count,
        "row_count": len(records),
        "duplicate_rows": duplicate_rows,
        "invalid_ohlc_rows": 0,
        "missing_sessions": 0,
        "unexpected_sessions": 0,
        "stale_instruments": sum(1 for record in records if record.get("session_status") == "STALE_CACHE"),
        "timezone_errors": timezone_errors,
        "currency_errors": 0,
        "contract_mismatches": contract_mismatches,
        "first_timestamp": min(first_dates).isoformat() if first_dates else None,
        "last_timestamp": max(first_dates).isoformat() if first_dates else None,
        "content_hash": content_hash,
    }
    status = _session_cache_status(errors, quality)
    return {
        "schema": "market_session_cache_validation_v1",
        "status": status,
        "validated_at": datetime.now(UTC).isoformat(),
        "layout": layout.as_dict(),
        **quality,
        "errors": errors,
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def read_session_cache_records(layout: SessionCacheLayout) -> list[dict[str, Any]]:
    if not layout.sessions_parquet.exists():
        return []
    return pq.read_table(layout.sessions_parquet).to_pylist()


def market_windows_from_ibkr_hours(
    value: str | tuple[IbkrSessionWindow, ...],
    *,
    timezone: ZoneInfo,
    source: str,
) -> tuple[MarketWindow, ...]:
    windows: list[MarketWindow] = []
    parsed_windows = parse_ibkr_hours(value) if isinstance(value, str) else value
    for parsed_window in parsed_windows:
        window = _market_window_from_ibkr_window(parsed_window, timezone=timezone, source=source)
        if window is not None:
            windows.append(window)
    return tuple(windows)


def _market_window_from_ibkr_window(
    window: IbkrSessionWindow,
    *,
    timezone: ZoneInfo,
    source: str,
) -> MarketWindow | None:
    if window.closed:
        return None
    if window.start is None or window.end is None:
        raise ValueError("open IBKR session window requires start and end times")

    start_date = window.start_date or window.session_date
    end_date = window.end_date or window.session_date
    start = datetime.combine(start_date, window.start, tzinfo=timezone)
    end = datetime.combine(end_date, window.end, tzinfo=timezone)
    if end <= start:
        end += timedelta(days=1)
    return MarketWindow(
        session_date=window.session_date,
        start=start,
        end=end,
        source=source,
    )


def _session_record_from_windows(
    row: ContractCacheRow,
    *,
    session_date: date,
    trading_windows: tuple[MarketWindow, ...],
    liquid_windows: tuple[MarketWindow, ...],
    trading_closed: bool,
    liquid_closed: bool,
    calendar_check: dict[str, Any],
    conflicts: list[dict[str, Any]],
) -> MarketSessionRecord:
    trading_open = min((window.start for window in trading_windows), default=None)
    trading_close = max((window.end for window in trading_windows), default=None)
    liquid_open = min((window.start for window in liquid_windows), default=None)
    liquid_close = max((window.end for window in liquid_windows), default=None)
    ibkr_trading_open_utc = trading_open.astimezone(UTC) if trading_open else None
    ibkr_trading_close_utc = trading_close.astimezone(UTC) if trading_close else None
    ibkr_liquid_open_utc = liquid_open.astimezone(UTC) if liquid_open else None
    ibkr_liquid_close_utc = liquid_close.astimezone(UTC) if liquid_close else None
    calendar_open_utc = _parse_cached_datetime(calendar_check.get("session_open_utc"))
    calendar_close_utc = _parse_cached_datetime(calendar_check.get("session_close_utc"))
    effective_open_utc = ibkr_liquid_open_utc or ibkr_trading_open_utc
    effective_close_utc = ibkr_liquid_close_utc or ibkr_trading_close_utc
    effective_source = "IBKR_LIQUID_HOURS" if ibkr_liquid_open_utc and ibkr_liquid_close_utc else "IBKR_TRADING_HOURS"
    is_regular = bool(trading_windows)
    is_liquid = bool(liquid_windows)
    conflict_minutes = _max_conflict_minutes(conflicts)
    conflict_classification = _conflict_classification(
        conflicts=conflicts,
        conflict_minutes=conflict_minutes,
        missing_required_hours=is_regular and (effective_open_utc is None or effective_close_utc is None),
    )
    is_holiday = (trading_closed and liquid_closed) or calendar_check.get("calendar_is_session") is False
    is_early_close = bool(calendar_check.get("calendar_is_early_close"))
    status = _resolved_session_status(
        is_regular=is_regular,
        is_holiday=is_holiday,
        is_early_close=is_early_close,
        conflict_classification=conflict_classification,
    )
    readiness = (
        "SESSION_BLOCKED"
        if status in {"SESSION_NOT_FOUND", "TIMEZONE_INVALID", "HOURS_PARSE_ERROR", "CONTRACT_NOT_RESOLVED"}
        or conflict_classification == "OUTSIDE_TOLERANCE"
        else "SESSION_DEGRADED"
        if conflict_classification == "WITHIN_TOLERANCE"
        else "SESSION_READY"
    )
    return MarketSessionRecord(
        con_id=row.contract.con_id,
        symbol=row.contract.symbol,
        security_type=row.contract.security_type.value,
        exchange=row.contract.exchange,
        primary_exchange=row.contract.primary_exchange,
        currency=row.contract.currency,
        timezone_id=row.contract.time_zone_id or "",
        session_date=session_date,
        session_open_utc=ibkr_trading_open_utc,
        session_close_utc=ibkr_trading_close_utc,
        liquid_open_utc=ibkr_liquid_open_utc,
        liquid_close_utc=ibkr_liquid_close_utc,
        ibkr_trading_open_utc=ibkr_trading_open_utc,
        ibkr_trading_close_utc=ibkr_trading_close_utc,
        ibkr_liquid_open_utc=ibkr_liquid_open_utc,
        ibkr_liquid_close_utc=ibkr_liquid_close_utc,
        calendar_open_utc=calendar_open_utc,
        calendar_close_utc=calendar_close_utc,
        effective_collection_open_utc=effective_open_utc,
        effective_collection_close_utc=effective_close_utc,
        effective_source=effective_source,
        conflict_minutes=conflict_minutes,
        conflict_classification=conflict_classification,
        is_regular_session=is_regular,
        is_liquid_session=is_liquid,
        is_overnight=_is_overnight(trading_windows),
        is_early_close=is_early_close,
        is_holiday=is_holiday,
        source="IBKR_CONTRACT_CACHE",
        contract_hash=row.contract_hash,
        session_status=status,
        readiness=readiness,
    )


def _record_from_report_session(session: dict[str, Any]) -> MarketSessionRecord:
    return MarketSessionRecord(
        con_id=int(session["con_id"]),
        symbol=str(session["symbol"]),
        security_type=str(session["security_type"]),
        exchange=str(session["exchange"]),
        primary_exchange=session.get("primary_exchange"),
        currency=str(session["currency"]),
        timezone_id=str(session["timezone_id"]),
        session_date=_parse_cached_date(session["session_date"]),
        session_open_utc=_parse_cached_datetime(session.get("session_open_utc")),
        session_close_utc=_parse_cached_datetime(session.get("session_close_utc")),
        liquid_open_utc=_parse_cached_datetime(session.get("liquid_open_utc")),
        liquid_close_utc=_parse_cached_datetime(session.get("liquid_close_utc")),
        ibkr_trading_open_utc=_parse_cached_datetime(session.get("ibkr_trading_open_utc")),
        ibkr_trading_close_utc=_parse_cached_datetime(session.get("ibkr_trading_close_utc")),
        ibkr_liquid_open_utc=_parse_cached_datetime(session.get("ibkr_liquid_open_utc")),
        ibkr_liquid_close_utc=_parse_cached_datetime(session.get("ibkr_liquid_close_utc")),
        calendar_open_utc=_parse_cached_datetime(session.get("calendar_open_utc")),
        calendar_close_utc=_parse_cached_datetime(session.get("calendar_close_utc")),
        effective_collection_open_utc=_parse_cached_datetime(session.get("effective_collection_open_utc")),
        effective_collection_close_utc=_parse_cached_datetime(session.get("effective_collection_close_utc")),
        effective_source=str(session["effective_source"]),
        conflict_minutes=session.get("conflict_minutes"),
        conflict_classification=str(session["conflict_classification"]),
        is_regular_session=bool(session["is_regular_session"]),
        is_liquid_session=bool(session["is_liquid_session"]),
        is_overnight=bool(session["is_overnight"]),
        is_early_close=bool(session["is_early_close"]),
        is_holiday=bool(session["is_holiday"]),
        source=str(session["source"]),
        contract_hash=str(session["contract_hash"]),
        session_type=str(session.get("session_type", "PRIMARY")),
        session_status=str(session["session_status"]),
        readiness=str(session["readiness"]),
    )


def _blocked_session_report(row: ContractCacheRow, *, session_date: date, status: str) -> dict[str, Any]:
    return {
        "schema": "market_sessions_v1",
        "status": status,
        "session_status": status,
        "readiness": "SESSION_BLOCKED",
        "source": "local_contract_cache",
        "contract": _contract_payload(row, validate=False),
        "timeZoneId": row.contract.time_zone_id,
        "session_date": session_date.isoformat(),
        "is_trading_day": False,
        "is_liquid_day": False,
        "trading_closed": False,
        "liquid_closed": False,
        "trading_windows": [],
        "liquid_windows": [],
        "session": None,
        "calendar_check": None,
        "calendar_conflicts": [],
        "financial_calls": _zero_financial_calls(),
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def _calendar_conflicts(calendar_check: dict[str, Any], liquid_windows: tuple[MarketWindow, ...]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    if calendar_check.get("alignment") == "MISMATCH":
        conflicts.append(
            {
                "type": "SESSION_DAY_MISMATCH",
                "reason": calendar_check.get("reason"),
                "calendar_code": calendar_check.get("calendar_code"),
                "calendar_is_session": calendar_check.get("calendar_is_session"),
                "ibkr_has_trading_window": calendar_check.get("ibkr_has_trading_window"),
            }
        )
    if calendar_check.get("calendar_is_session") is not True or not liquid_windows:
        return conflicts

    calendar_open = _parse_cached_datetime(calendar_check.get("session_open_utc"))
    calendar_close = _parse_cached_datetime(calendar_check.get("session_close_utc"))
    liquid_open = min(window.start for window in liquid_windows).astimezone(UTC)
    liquid_close = max(window.end for window in liquid_windows).astimezone(UTC)
    if calendar_open and abs((liquid_open - calendar_open).total_seconds()) > 60:
        diff_minutes = _rounded_abs_minutes(liquid_open - calendar_open)
        conflicts.append(
            {
                "type": "LIQUID_OPEN_MISMATCH",
                "calendar_open_utc": calendar_open.isoformat(),
                "ibkr_liquid_open_utc": liquid_open.isoformat(),
                "conflict_minutes": diff_minutes,
                "conflict_classification": "WITHIN_TOLERANCE"
                if diff_minutes <= _SESSION_CONFLICT_TOLERANCE_MINUTES
                else "OUTSIDE_TOLERANCE",
            }
        )
    if calendar_close and abs((liquid_close - calendar_close).total_seconds()) > 60:
        diff_minutes = _rounded_abs_minutes(liquid_close - calendar_close)
        conflicts.append(
            {
                "type": "LIQUID_CLOSE_MISMATCH",
                "calendar_close_utc": calendar_close.isoformat(),
                "ibkr_liquid_close_utc": liquid_close.isoformat(),
                "conflict_minutes": diff_minutes,
                "conflict_classification": "WITHIN_TOLERANCE"
                if diff_minutes <= _SESSION_CONFLICT_TOLERANCE_MINUTES
                else "OUTSIDE_TOLERANCE",
            }
        )
    return conflicts


def _max_conflict_minutes(conflicts: list[dict[str, Any]]) -> int | None:
    values = [
        int(conflict["conflict_minutes"])
        for conflict in conflicts
        if conflict.get("conflict_minutes") is not None
    ]
    return max(values) if values else None


def _conflict_classification(
    *,
    conflicts: list[dict[str, Any]],
    conflict_minutes: int | None,
    missing_required_hours: bool,
) -> str:
    if missing_required_hours:
        return "MISSING_OR_UNPARSEABLE_HOURS"
    if not conflicts:
        return "EXACT_MATCH"
    if conflict_minutes is None:
        return "WITHIN_TOLERANCE"
    return (
        "WITHIN_TOLERANCE"
        if conflict_minutes <= _SESSION_CONFLICT_TOLERANCE_MINUTES
        else "OUTSIDE_TOLERANCE"
    )


def _rounded_abs_minutes(delta: timedelta) -> int:
    return int(round(abs(delta.total_seconds()) / 60.0))


def _current_session_status(
    *,
    local_at: datetime,
    active_trading: MarketWindow | None,
    active_liquid: MarketWindow | None,
    trading_windows: tuple[MarketWindow, ...],
    liquid_windows: tuple[MarketWindow, ...],
    row: ContractCacheRow,
) -> str:
    if active_liquid is not None:
        return "OPEN_LIQUID"
    if active_trading is not None:
        if _inside_liquid_session_gap(local_at, liquid_windows):
            return "MAINTENANCE_BREAK"
        return "OPEN_EXTENDED"
    parsed_trading_hours = parse_ibkr_hours(row.contract.trading_hours or "")
    if _is_closed_for_date(parsed_trading_hours, local_at.date()):
        return "HOLIDAY"
    if trading_windows:
        return "CLOSED"
    return "SESSION_NOT_FOUND"


def _resolved_session_status(
    *,
    is_regular: bool,
    is_holiday: bool,
    is_early_close: bool,
    conflict_classification: str,
) -> str:
    if conflict_classification in {"WITHIN_TOLERANCE", "OUTSIDE_TOLERANCE"}:
        return "CALENDAR_CONFLICT"
    if is_early_close and is_regular:
        return "EARLY_CLOSE"
    if is_holiday:
        return "HOLIDAY"
    if not is_regular:
        return "SESSION_NOT_FOUND"
    return "CLOSED"


def _inside_liquid_session_gap(value: datetime, liquid_windows: tuple[MarketWindow, ...]) -> bool:
    session_windows = [window for window in liquid_windows if window.session_date == value.date()]
    if len(session_windows) < 2:
        return False
    return min(window.start for window in session_windows) <= value < max(window.end for window in session_windows)


def _active_window(windows: tuple[MarketWindow, ...], value: datetime) -> MarketWindow | None:
    return next((window for window in windows if window.contains(value)), None)


def _next_window(windows: tuple[MarketWindow, ...], value: datetime) -> MarketWindow | None:
    future_windows = [window for window in windows if window.start > value]
    if not future_windows:
        return None
    return min(future_windows, key=lambda window: window.start)


def _window_coverage(windows: tuple[MarketWindow, ...]) -> dict[str, str | None]:
    if not windows:
        return {"start": None, "end": None}
    return {
        "start": min(window.start for window in windows).isoformat(),
        "end": max(window.end for window in windows).isoformat(),
    }


def _is_closed_for_date(windows: tuple[IbkrSessionWindow, ...], session_date: date) -> bool:
    return any(window.session_date == session_date and window.closed for window in windows)


def _known_session_dates(row: ContractCacheRow) -> tuple[date, ...]:
    dates = {
        window.session_date
        for value in (row.contract.trading_hours, row.contract.liquid_hours)
        if value
        for window in parse_ibkr_hours(value)
    }
    return tuple(sorted(dates))


def _is_overnight(windows: tuple[MarketWindow, ...]) -> bool:
    return any(window.end.date() != window.start.date() for window in windows)


def _contract_payload(row: ContractCacheRow, *, validate: bool = True) -> dict[str, Any]:
    if validate:
        row.contract.validate_phase2_required_fields()
    return {
        "conId": row.contract.con_id,
        "symbol": row.contract.symbol,
        "localSymbol": row.contract.local_symbol,
        "secType": row.contract.security_type.value,
        "exchange": row.contract.exchange,
        "primaryExchange": row.contract.primary_exchange,
        "currency": row.contract.currency,
    }


def _session_error_status(reason: str) -> str:
    lowered = reason.lower()
    if "timezone" in lowered or "time_zone_id" in lowered:
        return "TIMEZONE_INVALID"
    if "hours" in lowered:
        return "HOURS_PARSE_ERROR"
    if "contract" in lowered or "required" in lowered:
        return "CONTRACT_NOT_RESOLVED"
    return "SESSION_NOT_FOUND"


def _effective_datetime(value: datetime | None) -> datetime:
    effective = value or datetime.now(UTC)
    if effective.tzinfo is None or effective.utcoffset() is None:
        raise ValueError("at must be timezone-aware")
    return effective


def _date_range(start: date, end: date) -> tuple[date, ...]:
    days = (end - start).days
    return tuple(start + timedelta(days=offset) for offset in range(days + 1))


def _write_session_records(path: Path, records: list[MarketSessionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([record.parquet_record() for record in records])
    pq.write_table(table, path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True, default=str) + "\n")


def _session_audit_record(row: ContractCacheRow, session_date: date, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "market_session_error_v1",
        "con_id": row.contract.con_id,
        "symbol": row.contract.symbol,
        "session_date": session_date.isoformat(),
        "status": report.get("status"),
        "reason": report.get("reason"),
        "contract_hash": row.contract_hash,
    }


def _validate_cached_session_record(record: dict[str, Any]) -> None:
    con_id = record.get("con_id")
    if isinstance(con_id, bool) or not isinstance(con_id, int) or con_id <= 0:
        raise ValueError("con_id must be positive")
    timezone_id = str(record.get("timezone_id") or "")
    parse_ibkr_timezone_id(timezone_id)
    cached_hash = str(record.get("session_hash") or "")
    expected_hash = session_hash(
        con_id=con_id,
        session_date=_parse_cached_date(record.get("session_date")),
        session_open_utc=_parse_cached_datetime(record.get("session_open_utc")),
        session_close_utc=_parse_cached_datetime(record.get("session_close_utc")),
        liquid_open_utc=_parse_cached_datetime(record.get("liquid_open_utc")),
        liquid_close_utc=_parse_cached_datetime(record.get("liquid_close_utc")),
        timezone_id=timezone_id,
    )
    if cached_hash != expected_hash:
        raise ValueError("session_hash does not match deterministic session payload")


def _duplicate_primary_key_count(records: list[dict[str, Any]]) -> int:
    seen: set[tuple[Any, Any, Any]] = set()
    duplicates = 0
    for record in records:
        key = (record.get("con_id"), str(record.get("session_date")), record.get("session_type"))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _contract_mismatch_count(records: list[dict[str, Any]], contract_rows: list[ContractCacheRow]) -> int:
    if not contract_rows:
        return 0
    expected = {row.contract.con_id: row.contract_hash for row in contract_rows}
    return sum(
        1
        for record in records
        if int(record.get("con_id", 0)) not in expected
        or record.get("contract_hash") != expected[int(record.get("con_id", 0))]
    )


def _session_cache_status(errors: list[str], quality: dict[str, Any]) -> str:
    if errors:
        return "NO_GO"
    if quality["file_count"] <= 0 or quality["row_count"] <= 0 or quality["content_hash"] is None:
        return "NO_DATA"
    if (
        quality["duplicate_rows"] != 0
        or quality["timezone_errors"] != 0
        or quality["contract_mismatches"] != 0
    ):
        return "NO_GO"
    return "GO"


def _classified_error_count(errors: list[str], needles: tuple[str, ...]) -> int:
    return sum(1 for error in errors if any(needle in error for needle in needles))


def _file_content_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _parse_cached_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _date_record_value(value: Any) -> date | None:
    if value is None:
        return None
    return _parse_cached_date(value)


def _parse_cached_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    text = str(value)
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _datetime_utc_iso(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _datetime_hash_value(value: datetime | None) -> str:
    return _datetime_utc_iso(value) or ""


def _zero_financial_calls() -> dict[str, int]:
    return {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }
