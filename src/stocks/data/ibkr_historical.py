from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from stocks.application.config import IbkrSettings
from stocks.data.bars import BarCacheLayout, BarDataSource, BarDataType, BarInterval
from stocks.domain.assets import IbkrSecurityType
from stocks.ibkr.contract_cache import ContractCacheRow
from stocks.ibkr.contracts import IbkrContractSpec, build_native_ibapi_contract
from stocks.market.calendars import calendar_code_for_contract
from stocks.market.sessions import session_hash


class HistoricalCollectorClientIdCollision(RuntimeError):
    pass


@dataclass(frozen=True)
class HistoricalDataRequestPlan:
    request_id: int
    con_id: int
    end_datetime: datetime
    duration: str
    bar_size: str
    what_to_show: str
    use_rth: bool
    format_date: int
    keep_up_to_date: bool
    planned_session_range: dict[str, str]
    request_hash: str
    max_attempts: int = 3
    retry_backoff_seconds: tuple[int, ...] = (2, 5, 15)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "con_id": self.con_id,
            "end_datetime": self.end_datetime.astimezone(UTC).isoformat(),
            "duration": self.duration,
            "bar_size": self.bar_size,
            "what_to_show": self.what_to_show,
            "use_rth": self.use_rth,
            "format_date": self.format_date,
            "keep_up_to_date": self.keep_up_to_date,
            "planned_session_range": self.planned_session_range,
            "request_hash": self.request_hash,
            "max_attempts": self.max_attempts,
            "retry_backoff_seconds": list(self.retry_backoff_seconds),
        }


@dataclass
class HistoricalCallbackState:
    ready_event: threading.Event = field(default_factory=threading.Event)
    done_events: dict[int, threading.Event] = field(default_factory=dict)
    bars: dict[int, list[Any]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    connected: bool = False
    historical_data_calls: int = 0
    financial_calls: dict[str, int] = field(
        default_factory=lambda: {"place_order": 0, "cancel_order": 0, "global_cancel": 0}
    )
    lock: threading.RLock = field(default_factory=threading.RLock)

    def done_event(self, request_id: int) -> threading.Event:
        with self.lock:
            return self.done_events.setdefault(request_id, threading.Event())

    def record_bar(self, request_id: int, bar: Any) -> None:
        with self.lock:
            self.bars.setdefault(request_id, []).append(bar)

    def record_done(self, request_id: int) -> None:
        with self.lock:
            self.done_events.setdefault(request_id, threading.Event()).set()

    def record_error(self, request_id: Any, args: tuple[Any, ...]) -> None:
        with self.lock:
            self.errors.append(
                {
                    "request_id": request_id,
                    "args": [str(item) for item in args],
                }
            )


class IbkrHistoricalDataApp:
    def __init__(self, state: HistoricalCallbackState) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        class _App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self, callback_state: HistoricalCallbackState) -> None:
                self.callback_state = callback_state
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                self.callback_state.ready_event.set()

            def historicalData(self, reqId: int, bar: Any) -> None:  # noqa: N802
                self.callback_state.record_bar(reqId, bar)

            def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # noqa: N802
                self.callback_state.record_done(reqId)

            def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
                self.callback_state.record_error(reqId, args)

        self._app = _App(state)

    def connect(self, host: str, port: int, client_id: int) -> None:
        self._app.connect(host, port, client_id)

    def disconnect(self) -> None:
        self._app.disconnect()

    def isConnected(self) -> bool:  # noqa: N802
        return bool(self._app.isConnected())

    def run(self) -> None:
        self._app.run()

    def reqHistoricalData(  # noqa: N802
        self,
        request_id: int,
        contract: Any,
        end_datetime: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: int,
        format_date: int,
        keep_up_to_date: bool,
        chart_options: list[Any],
    ) -> None:
        self._app.reqHistoricalData(
            request_id,
            contract,
            end_datetime,
            duration,
            bar_size,
            what_to_show,
            use_rth,
            format_date,
            keep_up_to_date,
            chart_options,
        )

    def cancelHistoricalData(self, request_id: int) -> None:  # noqa: N802
        self._app.cancelHistoricalData(request_id)


class IbkrHistoricalDataCollector:
    def __init__(self, settings: IbkrSettings) -> None:
        self.settings = settings
        self.state = HistoricalCallbackState()
        self.app = IbkrHistoricalDataApp(self.state)
        self.thread: threading.Thread | None = None

    def connect(self) -> None:
        self.app.connect(self.settings.host, self.settings.port, self.client_id)
        if not self.app.isConnected():
            raise RuntimeError("IBKR historical socket did not become connected")
        self.thread = threading.Thread(target=self.app.run, name="ibkr-historical-data-client", daemon=True)
        self.thread.start()
        if not self.state.ready_event.wait(self.settings.connect_timeout_seconds):
            raise TimeoutError("IBKR historical API handshake timed out")

    def disconnect(self) -> None:
        self.app.disconnect()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def collect(self, row: ContractCacheRow, plan: HistoricalDataRequestPlan) -> list[Any]:
        last_error: Exception | None = None
        for attempt in range(1, plan.max_attempts + 1):
            try:
                return self._collect_once(row, plan)
            except Exception as exc:
                last_error = exc
                if attempt >= plan.max_attempts:
                    break
                delay = plan.retry_backoff_seconds[min(attempt - 1, len(plan.retry_backoff_seconds) - 1)]
                threading.Event().wait(delay)
        raise RuntimeError(f"IBKR historical data request failed after {plan.max_attempts} attempts: {last_error}")

    def _collect_once(self, row: ContractCacheRow, plan: HistoricalDataRequestPlan) -> list[Any]:
        contract = build_native_ibapi_contract(
            IbkrContractSpec(
                symbol=row.contract.symbol,
                security_type=row.contract.security_type,
                exchange=row.contract.exchange,
                currency=row.contract.currency,
                primary_exchange=row.contract.primary_exchange,
            )
        )
        contract.conId = row.contract.con_id
        done = self.state.done_event(plan.request_id)
        done.clear()
        with self.state.lock:
            self.state.bars[plan.request_id] = []
        self.state.historical_data_calls += 1
        self.app.reqHistoricalData(
            plan.request_id,
            contract,
            _ibkr_end_datetime(plan.end_datetime),
            plan.duration,
            plan.bar_size,
            plan.what_to_show,
            1 if plan.use_rth else 0,
            plan.format_date,
            plan.keep_up_to_date,
            [],
        )
        if not done.wait(self.settings.request_timeout_seconds):
            self.app.cancelHistoricalData(plan.request_id)
            raise TimeoutError("IBKR historical data callback timed out")
        return list(self.state.bars.get(plan.request_id, []))

    @property
    def client_id(self) -> int:
        return self.settings.client_id + 1000


def collect_ibkr_daily_bars(
    *,
    settings: IbkrSettings,
    layout: BarCacheLayout,
    contract_row: ContractCacheRow,
    session_rows: list[dict[str, Any]],
    start: date,
    end: date,
) -> dict[str, Any]:
    requested_at = datetime.now(UTC)
    plan = build_daily_stk_request_plan(contract_row, start=start, end=end)
    _validate_phase4_contract(contract_row)
    path = layout.phase4_bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=contract_row.contract.con_id,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
    )
    output_dir = settings.output_dir / "bars"
    output_dir.mkdir(parents=True, exist_ok=True)
    _append_jsonl(output_dir / "requests.jsonl", {"schema": "ibkr_historical_request_v1", **plan.as_dict()})

    collector = IbkrHistoricalDataCollector(settings)
    try:
        with _HistoricalCollectorSingleFlight(output_dir=output_dir, client_id=collector.client_id):
            collector.connect()
            raw_bars = collector.collect(contract_row, plan)
    except Exception as exc:
        error_class = _historical_collection_error_class(exc)
        error = {
            "schema": "ibkr_historical_collection_error_v1",
            "status": error_class,
            "con_id": contract_row.contract.con_id,
            "request_hash": plan.request_hash,
            "client_id": collector.client_id,
            "error_class": error_class,
            "canonical_run": False,
            "retryable": error_class in {"CLIENT_ID_COLLISION", "API_HANDSHAKE_TIMEOUT", "CALLBACK_TIMEOUT"},
            "data_committed": False,
            "phase4_blocking": False if error_class == "CLIENT_ID_COLLISION" else True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "historical_data_calls": collector.state.historical_data_calls,
            "financial_calls": collector.state.financial_calls,
        }
        _append_jsonl(output_dir / "errors.jsonl", error)
        return error
    finally:
        collector.disconnect()

    received_at = datetime.now(UTC)
    records = [
        record_from_ibkr_bar(
            bar,
            contract_row=contract_row,
            session_rows=session_rows,
            plan=plan,
            requested_at=requested_at,
            received_at=received_at,
        )
        for bar in raw_bars
    ]
    records = _deduplicated_sorted_records(
        record for record in records if start <= date.fromisoformat(record["session_date"]) <= end
    )
    if not records:
        error = {
            "schema": "ibkr_historical_collection_error_v1",
            "status": "PROVIDER_EMPTY_RESPONSE",
            "con_id": contract_row.contract.con_id,
            "request_hash": plan.request_hash,
            "historical_data_calls": collector.state.historical_data_calls,
            "financial_calls": collector.state.financial_calls,
        }
        _append_jsonl(output_dir / "errors.jsonl", error)
        return error

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)
    manifest = _collection_manifest(layout, output_dir, path, records, plan, collector.state.historical_data_calls)
    (output_dir / "collection-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    gap_report = classify_daily_bar_gaps(
        records,
        session_rows=session_rows,
        contract_row=contract_row,
        start=start,
        end=end,
    )
    (output_dir / "gap-report.json").write_text(
        json.dumps(gap_report, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": "ibkr_historical_bar_collection_v1",
        "status": "GO" if gap_report["blocking_gap_count"] == 0 else "NO_GO",
        "path": str(path),
        "row_count": len(records),
        "request": plan.as_dict(),
        "gap_report": gap_report,
        "canonical_run": True,
        "data_committed": True,
        "historical_data_calls": collector.state.historical_data_calls,
        "financial_calls": collector.state.financial_calls,
    }


def build_daily_stk_request_plan(contract_row: ContractCacheRow, *, start: date, end: date) -> HistoricalDataRequestPlan:
    if end < start:
        raise ValueError("end must be on or after start")
    request_id = int(contract_row.contract.con_id % 1_000_000) + 40_000
    end_datetime = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    duration = _duration_for_dates(start, end)
    payload = {
        "request_id": request_id,
        "con_id": contract_row.contract.con_id,
        "end_datetime": end_datetime.isoformat(),
        "duration": duration,
        "bar_size": "1 day",
        "what_to_show": "TRADES",
        "use_rth": True,
        "format_date": 1,
        "keep_up_to_date": False,
        "planned_session_range": {"start": start.isoformat(), "end": end.isoformat()},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return HistoricalDataRequestPlan(
        request_id=request_id,
        con_id=contract_row.contract.con_id,
        end_datetime=end_datetime,
        duration=duration,
        bar_size="1 day",
        what_to_show="TRADES",
        use_rth=True,
        format_date=1,
        keep_up_to_date=False,
        planned_session_range={"start": start.isoformat(), "end": end.isoformat()},
        request_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest().upper(),
        max_attempts=3,
        retry_backoff_seconds=(2, 5, 15),
    )


def record_from_ibkr_bar(
    bar: Any,
    *,
    contract_row: ContractCacheRow,
    session_rows: list[dict[str, Any]],
    plan: HistoricalDataRequestPlan,
    requested_at: datetime,
    received_at: datetime,
) -> dict[str, Any]:
    session_date = _parse_ibkr_bar_date(str(getattr(bar, "date")))
    session = _session_for_date(session_rows, session_date, contract_row=contract_row)
    timestamp_utc = _parse_optional_datetime(session.get("effective_collection_close_utc")) or datetime.combine(
        session_date,
        datetime.min.time(),
        tzinfo=UTC,
    )
    record = {
        "timestamp_utc": timestamp_utc.astimezone(UTC).isoformat(),
        "session_date": session_date.isoformat(),
        "con_id": contract_row.contract.con_id,
        "symbol": contract_row.contract.symbol,
        "security_type": contract_row.contract.security_type.value,
        "currency": contract_row.contract.currency,
        "exchange": contract_row.contract.exchange,
        "primary_exchange": contract_row.contract.primary_exchange,
        "interval": BarInterval.ONE_DAY.value,
        "data_type": BarDataType.TRADES.value,
        "use_rth": plan.use_rth,
        "open": str(Decimal(str(getattr(bar, "open")))),
        "high": str(Decimal(str(getattr(bar, "high")))),
        "low": str(Decimal(str(getattr(bar, "low")))),
        "close": str(Decimal(str(getattr(bar, "close")))),
        "volume": _optional_int(getattr(bar, "volume", None)),
        "wap": _optional_decimal_string(getattr(bar, "wap", None)),
        "bar_count": _optional_int(getattr(bar, "barCount", None)),
        "source": BarDataSource.IBKR.value,
        "requested_at": requested_at.astimezone(UTC).isoformat(),
        "received_at": received_at.astimezone(UTC).isoformat(),
        "contract_hash": contract_row.contract_hash,
        "session_hash": str(session["session_hash"]),
        "request_hash": plan.request_hash,
    }
    record["content_hash"] = _record_content_hash(record)
    return record


def classify_daily_bar_gaps(
    records: list[dict[str, Any]],
    *,
    session_rows: list[dict[str, Any]],
    contract_row: ContractCacheRow,
    start: date,
    end: date,
) -> dict[str, Any]:
    observed_dates = {date.fromisoformat(record["session_date"]) for record in records}
    session_by_date = {date.fromisoformat(str(row["session_date"])): row for row in session_rows}
    first_observed = min(observed_dates) if observed_dates else None
    last_observed = max(observed_dates) if observed_dates else None
    gaps: list[dict[str, Any]] = []
    for day in _date_range(start, end):
        if day in observed_dates:
            continue
        session = session_by_date.get(day)
        calendar_is_session = _calendar_is_session(contract_row, day)
        if day.weekday() >= 5:
            classification = "EXPECTED_WEEKEND"
        elif session and session.get("readiness") == "SESSION_DEGRADED":
            classification = "SESSION_CONFLICT"
        elif session and session.get("is_holiday") is True:
            classification = "EXPECTED_HOLIDAY"
        elif calendar_is_session is False:
            classification = "EXPECTED_HOLIDAY"
        elif first_observed and day < first_observed:
            classification = "EXPECTED_PRE_LISTING"
        elif last_observed and day > last_observed:
            classification = "EXPECTED_POST_DELISTING"
        else:
            classification = "UNEXPECTED_MISSING_SESSION"
        gaps.append(
            {
                "session_date": day.isoformat(),
                "classification": classification,
                "calendar_is_session": calendar_is_session,
                "session_readiness": None if session is None else session.get("readiness"),
                "conflict_classification": None if session is None else session.get("conflict_classification"),
            }
        )
    blocking = [gap for gap in gaps if gap["classification"] in {"UNEXPECTED_MISSING_SESSION", "PROVIDER_EMPTY_RESPONSE"}]
    return {
        "schema": "historical_bar_gap_classification_v1",
        "status": "GO" if not blocking else "NO_GO",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "observed_session_count": len(observed_dates),
        "gap_count": len(gaps),
        "blocking_gap_count": len(blocking),
        "gaps": gaps,
    }


def _calendar_is_session(contract_row: ContractCacheRow, session_date: date) -> bool | None:
    calendar_code = calendar_code_for_contract(contract_row.contract)
    if calendar_code is None:
        return None
    try:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar(calendar_code)
        return bool(calendar.is_session(session_date.isoformat()))
    except Exception:
        return None


def _validate_phase4_contract(row: ContractCacheRow) -> None:
    row.contract.validate_phase2_required_fields()
    if row.contract.security_type != IbkrSecurityType.STK:
        raise ValueError("Phase 4 V1 only supports STK contracts")


class _HistoricalCollectorSingleFlight:
    def __init__(self, *, output_dir: Path, client_id: int) -> None:
        self.path = output_dir / f"collector-client-{client_id}.lock"
        self.client_id = client_id
        self._acquired = False
        self._fd: int | None = None

    def __enter__(self) -> _HistoricalCollectorSingleFlight:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise HistoricalCollectorClientIdCollision(
                f"IBKR historical collector client_id {self.client_id} is already in use"
            ) from exc
        payload = {
            "schema": "ibkr_historical_collector_single_flight_lock_v1",
            "client_id": self.client_id,
            "created_at": datetime.now(UTC).isoformat(),
        }
        os.write(self._fd, json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8"))
        os.close(self._fd)
        self._fd = None
        self._acquired = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._acquired:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self._acquired = False


def _historical_collection_error_class(exc: Exception) -> str:
    if isinstance(exc, HistoricalCollectorClientIdCollision):
        return "CLIENT_ID_COLLISION"
    if isinstance(exc, TimeoutError):
        text = str(exc).lower()
        if "handshake" in text:
            return "API_HANDSHAKE_TIMEOUT"
        if "callback" in text:
            return "CALLBACK_TIMEOUT"
    return "PROVIDER_ERROR"


def _session_for_date(session_rows: list[dict[str, Any]], session_date: date, *, contract_row: ContractCacheRow) -> dict[str, Any]:
    match = next(
        (
            row
            for row in session_rows
            if int(row["con_id"]) == contract_row.contract.con_id
            and date.fromisoformat(str(row["session_date"])) == session_date
        ),
        None,
    )
    if match is not None:
        return match
    return _derived_daily_session(session_date, contract_row=contract_row)


def _derived_daily_session(session_date: date, *, contract_row: ContractCacheRow) -> dict[str, Any]:
    digest = session_hash(
        con_id=contract_row.contract.con_id,
        session_date=session_date,
        session_open_utc=None,
        session_close_utc=None,
        liquid_open_utc=None,
        liquid_close_utc=None,
        timezone_id=contract_row.contract.time_zone_id or "",
    )
    return {
        "con_id": contract_row.contract.con_id,
        "session_date": session_date.isoformat(),
        "session_hash": digest,
        "effective_collection_close_utc": None,
        "readiness": "SESSION_DEGRADED",
        "is_holiday": False,
        "source": "CALENDAR_DERIVED_DAILY_FALLBACK",
    }


def _duration_for_dates(start: date, end: date) -> str:
    days = (end - start).days + 1
    if days <= 365:
        return f"{days} D"
    years = max(1, math.ceil(days / 365.0))
    return f"{years} Y"


def _ibkr_end_datetime(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%d %H:%M:%S UTC")


def _parse_ibkr_bar_date(value: str) -> date:
    text = value.strip()
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    parsed = datetime.fromtimestamp(int(text), tz=UTC) if text.isdigit() else datetime.fromisoformat(text)
    return parsed.date()


def _deduplicated_sorted_records(records: Any) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        deduped[str(record["timestamp_utc"])] = record
    return [deduped[key] for key in sorted(deduped)]


def _record_content_hash(record: dict[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "content_hash"}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()


def _collection_manifest(
    layout: BarCacheLayout,
    output_dir: Path,
    path: Path,
    records: list[dict[str, Any]],
    plan: HistoricalDataRequestPlan,
    historical_data_calls: int,
) -> dict[str, Any]:
    return {
        "schema": "ibkr_historical_bar_collection_manifest_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "data_dir": str(layout.data_dir),
        "bars_path": str(path),
        "request": plan.as_dict(),
        "row_count": len(records),
        "first_timestamp": records[0]["timestamp_utc"],
        "last_timestamp": records[-1]["timestamp_utc"],
        "content_hash": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "artifacts": {
            "requests_jsonl": str(output_dir / "requests.jsonl"),
            "errors_jsonl": str(output_dir / "errors.jsonl"),
            "collection_manifest_json": str(output_dir / "collection-manifest.json"),
            "gap_report_json": str(output_dir / "gap-report.json"),
            "cache_validation_json": str(output_dir / "cache-validation.json"),
        },
        "historical_data_calls": historical_data_calls,
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
    }


def _date_range(start: date, end: date) -> tuple[date, ...]:
    return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True, default=str) + "\n")


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).strip() in {"", "-1"}:
        return None
    return int(float(str(value)))


def _optional_decimal_string(value: Any) -> str | None:
    if value is None or str(value).strip() in {"", "-1"}:
        return None
    return str(Decimal(str(value)))


def _parse_optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
