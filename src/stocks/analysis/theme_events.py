from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_2.providers import (
    SafeJsonClient,
    eodhd_probe,
    load_provider_secrets,
)


CONFIG_PATH = Path("config/themes/frontier_technology_energy_v1.json")
NEWS_PATH = Path("output/analysis/themes/theme-news.json")
MACRO_EVENTS_PATH = Path("output/macro/events.json")
PRIVATE_SEC_ROOT = Path("data/research/themes/private/sec")
PRIVATE_HISTORY = Path(
    "data/research/themes/private/events/event-calendar-snapshots.jsonl"
)
OUTPUT_PATH = Path("output/analysis/themes/event-risk-calendar.json")
MATERIAL_FORMS = frozenset(
    {
        "8-K",
        "8-K/A",
        "10-Q",
        "10-Q/A",
        "10-K",
        "10-K/A",
        "6-K",
        "20-F",
        "40-F",
        "SCHEDULE 13D",
        "SCHEDULE 13D/A",
    }
)

CalendarFetcher = Callable[[str], tuple[dict[str, Any] | None, str | None]]
EodhdFetcher = Callable[
    [list[str], date, date], tuple[dict[str, Any], Any | None]
]


def collect_theme_event_risk(
    project_root: Path,
    *,
    now: datetime | None = None,
    calendar_fetcher: CalendarFetcher | None = None,
    eodhd_fetcher: EodhdFetcher | None = None,
) -> dict[str, Any]:
    """Collect bounded point-in-time event context without creating entries."""
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    config = _read_json(project_root / CONFIG_PATH)
    instruments = _instrument_map(config)
    if not instruments:
        return _publish(project_root, _blocked(observed_at, "THEME_CONFIG_UNAVAILABLE"))

    companies = [
        symbol
        for symbol, metadata in instruments.items()
        if not _is_vehicle(metadata)
    ]
    fetch_calendar = calendar_fetcher or _yfinance_calendar
    yahoo_rows: dict[str, list[date]] = {}
    yahoo_errors: dict[str, str] = {}
    for symbol in companies:
        payload, error = fetch_calendar(symbol)
        dates = _calendar_dates(payload)
        if dates:
            yahoo_rows[symbol] = dates
        if error:
            yahoo_errors[symbol] = error

    start = observed_at.date() - timedelta(days=7)
    end = observed_at.date() + timedelta(days=180)
    fetch_eodhd = eodhd_fetcher or (
        lambda symbols, start_date, end_date: _eodhd_calendar(
            project_root,
            symbols,
            start_date,
            end_date,
        )
    )
    eodhd_probe_result, eodhd_payload = fetch_eodhd(companies, start, end)
    eodhd_rows = _eodhd_dates(eodhd_payload)
    news = _earnings_news(project_root, observed_at)
    macro = _macro_event_context(project_root, observed_at)

    rows = []
    for symbol, metadata in instruments.items():
        filings = _recent_material_filings(
            project_root,
            symbol,
            observed_at,
        )
        row = _event_row(
            symbol=symbol,
            metadata=metadata,
            observed_at=observed_at,
            yahoo_dates=yahoo_rows.get(symbol, []),
            yahoo_error=yahoo_errors.get(symbol),
            eodhd_dates=eodhd_rows.get(symbol, []),
            recent_filings=filings,
            recent_news=news.get(symbol, []),
            macro=macro,
        )
        rows.append(row)
    rows.sort(key=lambda row: (str(row["event_risk_status"]), row["symbol"]))

    counts = Counter(str(row["event_risk_status"]) for row in rows)
    company_rows = [row for row in rows if not row["event_not_applicable"]]
    snapshot_body = {
        "schema": "frontier_theme_event_risk_snapshot_v1",
        "rows": rows,
        "macro_event_context": macro,
        "eodhd_capability": eodhd_probe_result,
    }
    snapshot_hash = stable_hash(snapshot_body)
    snapshot_appended = _append_snapshot(
        project_root / PRIVATE_HISTORY,
        snapshot_body,
        snapshot_hash=snapshot_hash,
        observed_at=observed_at,
    )
    report: dict[str, Any] = {
        "schema": "frontier_theme_event_risk_calendar_v1",
        "status": (
            "GO"
            if company_rows
            and all(
                row["event_risk_status"]
                not in {"EVENT_DATE_UNCERTAIN", "EVENT_DATA_STALE"}
                for row in company_rows
            )
            else "GO_WITH_EXPLICIT_UNCERTAINTY"
        ),
        "generated_at": observed_at.isoformat(),
        "instrument_count": len(rows),
        "company_count": len(company_rows),
        "vehicle_count": len(rows) - len(company_rows),
        "status_counts": dict(sorted(counts.items())),
        "known_future_earnings_count": sum(
            row.get("next_earnings_date") is not None for row in company_rows
        ),
        "uncertain_company_count": sum(
            row["event_risk_status"]
            in {"EVENT_DATE_UNCERTAIN", "EVENT_DATE_CONFLICT", "EVENT_DATA_STALE"}
            for row in company_rows
        ),
        "rows": rows,
        "macro_event_context": macro,
        "providers": {
            "YFINANCE": {
                "status": "GO" if yahoo_rows else "UNAVAILABLE",
                "request_count": len(companies),
                "symbols_with_dates": len(yahoo_rows),
                "error_count": len(yahoo_errors),
                "semantics": "CURRENT_PROVIDER_ESTIMATE; DATE_ONLY; NOT_REVISION_HISTORY",
            },
            "EODHD": eodhd_probe_result,
            "SEC_EDGAR": {
                "status": "GO",
                "semantics": "RECENT_ACCEPTED_FILINGS_ONLY; NOT_A_FUTURE_EARNINGS_CALENDAR",
            },
            "MACRO_ENGINE": {
                "status": macro["status"],
                "source": "EXISTING_OFFICIAL_SCHEDULE_PIPELINE",
            },
        },
        "private_snapshot_hash": snapshot_hash,
        "private_snapshot_appended": snapshot_appended,
        "provider_calls_read_only": len(companies) + 1,
        "broker_calls": 0,
        "orders_generated": 0,
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(report)
    return _publish(project_root, report)


def _event_row(
    *,
    symbol: str,
    metadata: dict[str, Any],
    observed_at: datetime,
    yahoo_dates: list[date],
    yahoo_error: str | None,
    eodhd_dates: list[date],
    recent_filings: list[dict[str, Any]],
    recent_news: list[dict[str, Any]],
    macro: dict[str, Any],
) -> dict[str, Any]:
    if _is_vehicle(metadata):
        return {
            "symbol": symbol,
            "event_risk_status": "EVENT_NOT_APPLICABLE_VEHICLE",
            "event_not_applicable": True,
            "next_earnings_date": None,
            "days_to_event": None,
            "source_confidence": "NOT_APPLICABLE",
            "date_sources": [],
            "recent_material_filings": recent_filings,
            "recent_earnings_news_count": len(recent_news),
            "macro_event_risk_status": macro["status"],
            "hard_block_recommended": False,
            "soft_penalty_recommended": False,
            "authority": "CONTEXT_ONLY",
        }

    source_dates = {
        "YFINANCE": sorted(set(yahoo_dates)),
        "EODHD": sorted(set(eodhd_dates)),
    }
    current_or_future = {
        provider: [value for value in values if value >= observed_at.date()]
        for provider, values in source_dates.items()
        if values
    }
    representatives = {
        provider: min(values)
        for provider, values in current_or_future.items()
        if values
    }
    all_future = sorted({value for values in current_or_future.values() for value in values})
    provider_range_uncertain = any(
        len(values) > 1 and (max(values) - min(values)).days > 1
        for values in current_or_future.values()
    )
    provider_conflict = (
        len(representatives) > 1
        and (max(representatives.values()) - min(representatives.values())).days > 1
    )
    latest_past = max(
        (value for values in source_dates.values() for value in values if value < observed_at.date()),
        default=None,
    )
    recent_filing = bool(recent_filings) and _age_days(
        recent_filings[0].get("accepted_at"),
        observed_at,
    ) <= 7
    recent_news_event = bool(recent_news) and _age_days(
        recent_news[0].get("published_at"),
        observed_at,
    ) <= 7
    post_event = (
        latest_past is not None
        and (observed_at.date() - latest_past).days <= 7
        and (recent_filing or recent_news_event)
    )

    next_date = min(all_future) if all_future else None
    days_to_event = (next_date - observed_at.date()).days if next_date else None
    if provider_conflict or provider_range_uncertain:
        status = "EVENT_DATE_CONFLICT"
        confidence = "CONFLICTING_OR_WIDE_PROVIDER_ESTIMATES"
    elif next_date is not None and days_to_event is not None and days_to_event <= 2:
        status = "EVENT_RISK_IMMINENT"
        confidence = _confidence(representatives)
    elif next_date is not None and days_to_event is not None and days_to_event <= 7:
        status = "EVENT_RISK_NEAR"
        confidence = _confidence(representatives)
    elif next_date is not None:
        status = "EVENT_CLEAR"
        confidence = _confidence(representatives)
    elif post_event:
        status = "EVENT_RISK_POST_EVENT"
        confidence = "PAST_CALENDAR_DATE_CORROBORATED_BY_FILING_OR_NEWS"
    elif yahoo_error and not eodhd_dates:
        status = "EVENT_DATA_STALE"
        confidence = "PROVIDER_ERROR_NO_ALTERNATIVE"
    else:
        status = "EVENT_DATE_UNCERTAIN"
        confidence = "NO_CURRENT_OR_FUTURE_DATE"

    return {
        "symbol": symbol,
        "event_risk_status": status,
        "event_not_applicable": False,
        "next_earnings_date": next_date.isoformat() if next_date else None,
        "days_to_event": days_to_event,
        "latest_past_earnings_date": latest_past.isoformat() if latest_past else None,
        "source_confidence": confidence,
        "date_precision": "DATE_ONLY" if source_dates else "NONE",
        "date_sources": [
            {"provider": provider, "date": value.isoformat()}
            for provider, values in source_dates.items()
            for value in values
        ],
        "provider_error_class": yahoo_error,
        "recent_material_filings": recent_filings,
        "recent_earnings_news_count": len(recent_news),
        "latest_earnings_news_at": (
            recent_news[0].get("published_at") if recent_news else None
        ),
        "macro_event_risk_status": macro["status"],
        "hard_block_recommended": status
        in {
            "EVENT_RISK_IMMINENT",
            "EVENT_DATE_CONFLICT",
            "EVENT_DATE_UNCERTAIN",
            "EVENT_DATA_STALE",
        },
        "soft_penalty_recommended": status
        in {"EVENT_RISK_NEAR", "EVENT_RISK_POST_EVENT"},
        "authority": "RISK_CONTEXT_ONLY",
    }


def _yfinance_calendar(symbol: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        import yfinance as yf  # type: ignore[import-untyped]

        payload = yf.Ticker(symbol).calendar
    except Exception as exc:
        return None, type(exc).__name__
    return (payload if isinstance(payload, dict) else None), None


def _eodhd_calendar(
    project_root: Path,
    symbols: list[str],
    start: date,
    end: date,
) -> tuple[dict[str, Any], Any | None]:
    if not symbols:
        return {"status": "NOT_APPLICABLE", "record_count": 0}, None
    client = SafeJsonClient(
        user_agent="Stocks-Frontier-Theme-Events/1.0 read-only",
        timeout_seconds=20,
        max_attempts=1,
        minimum_interval=0.15,
    )
    key = load_provider_secrets(project_root).get("EODHD")
    provider_symbols = ",".join(_eodhd_symbol(symbol) for symbol in symbols)
    probe, payload = eodhd_probe(
        client,
        key,
        "frontier_theme_earnings_calendar",
        "calendar/earnings",
        {
            "symbols": provider_symbols,
            "from": start.isoformat(),
            "to": end.isoformat(),
        },
    )
    return probe.public_dict(), payload


def _calendar_dates(payload: dict[str, Any] | None) -> list[date]:
    if not isinstance(payload, dict):
        return []
    value = payload.get("Earnings Date") or payload.get("earningsDate")
    values = value if isinstance(value, (list, tuple)) else [value]
    return sorted({parsed for item in values if (parsed := _date_value(item))})


def _eodhd_dates(payload: Any) -> dict[str, list[date]]:
    if isinstance(payload, dict):
        for key in ("earnings", "data", "results", "items"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return {}
    rows: dict[str, list[date]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        raw_symbol = row.get("code") or row.get("symbol") or row.get("ticker")
        parsed = _date_value(
            row.get("report_date")
            or row.get("earnings_date")
            or row.get("date")
        )
        if not raw_symbol or parsed is None:
            continue
        symbol = str(raw_symbol).upper().split(".")[0]
        rows.setdefault(symbol, []).append(parsed)
    return {key: sorted(set(values)) for key, values in rows.items()}


def _recent_material_filings(
    project_root: Path,
    symbol: str,
    observed_at: datetime,
) -> list[dict[str, Any]]:
    root = project_root / PRIVATE_SEC_ROOT / f"symbol={symbol}"
    files = sorted(
        root.glob("submissions-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return []
    outer = _read_json(files[0])
    raw_payload = outer.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else outer
    recent = ((payload.get("filings") or {}).get("recent") or {})
    forms = recent.get("form") or []
    accepted = recent.get("acceptanceDateTime") or []
    filed = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    rows = []
    for index, form_value in enumerate(forms):
        form = str(form_value or "").upper()
        if form not in MATERIAL_FORMS:
            continue
        accepted_at = str(accepted[index]) if index < len(accepted) else ""
        if _age_days(accepted_at, observed_at) > 14:
            continue
        accession = str(accessions[index]) if index < len(accessions) else ""
        rows.append(
            {
                "form": form,
                "accepted_at": accepted_at or None,
                "filing_date": str(filed[index]) if index < len(filed) else None,
                "accession_fingerprint": stable_hash(accession)[:16]
                if accession
                else None,
                "authority": "RECENT_EVENT_CONTEXT_ONLY",
            }
        )
    rows.sort(key=lambda row: str(row.get("accepted_at") or ""), reverse=True)
    return rows[:5]


def _earnings_news(
    project_root: Path,
    observed_at: datetime,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in _read_json(project_root / NEWS_PATH).get("rows", []):
        if row.get("event_type") != "EARNINGS_OR_GUIDANCE":
            continue
        if _age_days(row.get("published_at"), observed_at) > 7:
            continue
        for symbol in row.get("symbols") or []:
            result.setdefault(str(symbol).upper(), []).append(
                {
                    "published_at": row.get("published_at"),
                    "event_hash": row.get("event_hash"),
                    "source": row.get("source"),
                }
            )
    for rows in result.values():
        rows.sort(key=lambda row: str(row.get("published_at") or ""), reverse=True)
    return result


def _macro_event_context(
    project_root: Path,
    observed_at: datetime,
) -> dict[str, Any]:
    payload = _read_json(project_root / MACRO_EVENTS_PATH)
    future = []
    for row in payload.get("scheduled_instances") or []:
        timestamp = _datetime_value(row.get("scheduled_at"))
        if timestamp is None or not observed_at <= timestamp <= observed_at + timedelta(days=7):
            continue
        future.append(
            {
                "event_id": row.get("event_id"),
                "name": row.get("name"),
                "scheduled_at": timestamp.isoformat(),
                "importance": row.get("importance"),
                "hours_to_event": round((timestamp - observed_at).total_seconds() / 3600, 3),
            }
        )
    future.sort(key=lambda row: str(row["scheduled_at"]))
    high = [row for row in future if str(row.get("importance")).upper() == "HIGH"]
    nearest_hours = min((float(row["hours_to_event"]) for row in high), default=None)
    schedule_status = str(payload.get("future_schedule_status") or "UNAVAILABLE")
    schedule_available = (
        schedule_status in {"GO", "PARTIAL"}
        or "AVAILABLE" in schedule_status
    )
    status = (
        "MACRO_EVENT_RISK_IMMINENT"
        if nearest_hours is not None and nearest_hours <= 24
        else "MACRO_EVENT_RISK_NEAR"
        if nearest_hours is not None and nearest_hours <= 72
        else "MACRO_EVENT_CLEAR"
        if schedule_available
        else "MACRO_EVENT_DATE_UNCERTAIN"
    )
    return {
        "status": status,
        "future_schedule_status": schedule_status,
        "high_importance_event_count_7d": len(high),
        "nearest_high_importance_event_hours": nearest_hours,
        "events": future,
        "authority": "CONTEXT_ONLY",
    }


def _instrument_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for theme_id, definition in (config.get("themes") or {}).items():
        for row in definition.get("instruments") or []:
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                result[symbol] = dict(row, theme=theme_id)
    return result


def _is_vehicle(metadata: dict[str, Any]) -> bool:
    maturity = str(metadata.get("business_maturity") or "").upper()
    return "FUND" in maturity or "VEHICLE" in maturity


def _confidence(representatives: dict[str, date]) -> str:
    return (
        "DUAL_PROVIDER_CONSENSUS"
        if len(representatives) > 1
        else "SINGLE_PROVIDER_ESTIMATE"
    )


def _eodhd_symbol(symbol: str) -> str:
    return symbol if "." in symbol else f"{symbol}.US"


def _date_value(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _datetime_value(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _age_days(value: Any, observed_at: datetime) -> float:
    parsed = _datetime_value(value)
    if parsed is None:
        parsed_date = _date_value(value)
        if parsed_date is None:
            return float("inf")
        parsed = datetime.combine(parsed_date, datetime.min.time(), tzinfo=UTC)
    return max(0.0, (observed_at - parsed).total_seconds() / 86400.0)


def _append_snapshot(
    path: Path,
    body: dict[str, Any],
    *,
    snapshot_hash: str,
    observed_at: datetime,
) -> bool:
    previous_hash = None
    if path.is_file():
        for line in reversed(path.read_text(encoding="utf-8").splitlines()):
            try:
                previous = json.loads(line)
            except json.JSONDecodeError:
                continue
            previous_hash = previous.get("snapshot_hash")
            break
    if previous_hash == snapshot_hash:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "observed_at": observed_at.isoformat(),
        "snapshot_hash": snapshot_hash,
        **body,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return True


def _blocked(observed_at: datetime, reason: str) -> dict[str, Any]:
    return {
        "schema": "frontier_theme_event_risk_calendar_v1",
        "status": f"BLOCKED_{reason}",
        "generated_at": observed_at.isoformat(),
        "rows": [],
        "provider_calls_read_only": 0,
        "broker_calls": 0,
        "orders_generated": 0,
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _publish(project_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return report
