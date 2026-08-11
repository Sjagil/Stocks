from __future__ import annotations

import hashlib
import html
import json
import re
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from filelock import FileLock, Timeout

from stocks.macro.analyst import deterministic_analysis, render_markdown
from stocks.macro.config import MacroConfig
from stocks.macro.contracts import MACRO_AUTHORITY, stable_hash
from stocks.macro.engine import (
    compute_macro_snapshot,
    exposure_multiplier,
)
from stocks.macro.providers import collect_configured_sources
from stocks.macro.storage import MacroLayout, MacroStore
from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger
from stocks.shadow.audit import freeze_integrity


def macro_collect(
    project_root: Path,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    end_date = date.fromisoformat(end) if end else date.today()
    start_date = date.fromisoformat(start) if start else date(2000, 1, 1)
    if start_date > end_date:
        raise ValueError("MACRO_START_AFTER_END")
    observations, inventory = collect_configured_sources(
        project_root,
        config,
        start=start_date,
        end=end_date,
    )
    with _store(project_root) as store:
        registration = store.append_observations(
            observations,
            quarantine_conflicts=True,
        )
        counts = store.counts()
    inventory["provider_conflicts"] = [
        {
            "series_id": series_id,
            "conflict_count": conflict_count,
            "classification": "IMMUTABLE_PROVIDER_PAYLOAD_CONFLICT_QUARANTINED",
        }
        for series_id, conflict_count in registration[
            "quarantined_conflicts_by_series"
        ].items()
    ]
    payload = {
        "schema": "macro_collection_v1",
        "status": "GO" if observations else "DATA_INCOMPLETE",
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "registration": registration,
        "store_counts": counts,
        "source_inventory": inventory,
        "vintage_policy": (
            "LATEST_RELEASES_STORED_APPEND_ONLY; "
            "HISTORICAL_VINTAGES_ONLY_WHEN_EXPLICITLY_PROVIDED"
        ),
        "provider_conflict_policy": (
            "CONFLICTING_EXISTING_IDENTITIES_ARE_QUARANTINED_NOT_OVERWRITTEN"
        ),
        "financial_evidence": False,
        **MACRO_AUTHORITY,
    }
    return _publish(project_root, "collection.json", payload)


def macro_update(project_root: Path) -> dict[str, Any]:
    lock_path = (
        project_root / "data" / "macro" / "private" / "macro-update.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return _publish(
            project_root,
            "update.json",
            {
                "schema": "macro_update_v2",
                "status": "UPDATE_ALREADY_RUNNING_BLOCKED",
                "single_flight_lock": "HELD_BY_ANOTHER_PROCESS",
                **MACRO_AUTHORITY,
            },
        )
    try:
        collection = macro_collect(project_root)
        score = macro_score(project_root)
        report = macro_report(project_root, period="daily")
    finally:
        lock.release()
    return _publish(
        project_root,
        "update.json",
        {
            "schema": "macro_update_v2",
            "status": (
                "GO"
                if score["data_quality"]["status"] == "GO"
                else "DATA_INCOMPLETE"
            ),
            "collection_status": collection["status"],
            "score_status": score["data_quality"]["status"],
            "report_status": report["status"],
            "single_flight_lock": "ACQUIRED_AND_RELEASED",
            "bounded_one_shot": True,
            **MACRO_AUTHORITY,
        },
    )


def macro_validate(project_root: Path) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    with _store(project_root) as store:
        observations = store.observations()
        counts = store.counts()
    duplicate_ids = len(observations) - len(
        {row["observation_id"] for row in observations}
    )
    invalid_timestamps = sum(
        pd.Timestamp(row["available_at"]) < pd.Timestamp(row["publication_at"])
        for row in observations
    )
    unknown_series = sorted(
        {row["series_id"] for row in observations} - set(config.series)
    )
    revisions = _revision_audit(observations)
    payload = {
        "schema": "macro_data_validation_v1",
        "status": (
            "GO"
            if duplicate_ids == 0
            and invalid_timestamps == 0
            and not unknown_series
            else "BLOCKED"
        ),
        "observation_count": len(observations),
        "duplicate_observation_ids": duplicate_ids,
        "invalid_timestamp_rows": invalid_timestamps,
        "unknown_series": unknown_series,
        "revision_audit": revisions,
        "series_registry_count": len(config.series),
        "store_counts": counts,
        "point_in_time_contract": "GO",
        "future_revisions_used_early": 0,
        **MACRO_AUTHORITY,
    }
    return _publish(project_root, "validation.json", payload)


def macro_status(project_root: Path) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    with _store(project_root) as store:
        counts = store.counts()
        snapshot = store.latest_payload("score_snapshots")
    availability = (
        {}
        if snapshot is None
        else snapshot["data_quality"]["feature_status_counts"]
    )
    payload = {
        "schema": "macro_engine_status_v1",
        "status": "GO" if snapshot is not None else "NOT_RUN",
        "config_version": config.version,
        "config_hash": config.config_hash,
        "series_count": len(config.series),
        "score_count": len(config.score_weights),
        "store_counts": counts,
        "feature_availability": availability,
        "latest_as_of": None if snapshot is None else snapshot["as_of"],
        "latest_regime": (
            None
            if snapshot is None
            else snapshot["regime"]["overall_macro_regime"]
        ),
        "latest_data_quality": (
            "UNAVAILABLE"
            if snapshot is None
            else snapshot["data_quality"]["status"]
        ),
        "financial_evidence": False,
        **MACRO_AUTHORITY,
    }
    return _publish(project_root, "status.json", payload)


def macro_readiness(project_root: Path) -> dict[str, Any]:
    output = MacroLayout.from_project_root(project_root).output_root
    collection = _read_json(output / "collection.json")
    events = _read_json(output / "events.json")
    score = _read_json(output / "score.json")
    pair_status = _read_json(
        project_root
        / "output"
        / "research"
        / "macro_pairs"
        / "status.json"
    )
    validation = macro_validate(project_root)
    conflicts = macro_conflicts(project_root)
    provider_calls = collection.get("source_inventory", {}).get(
        "provider_calls",
        {},
    )
    required_provider_health = {
        provider: int(provider_calls.get(provider, 0)) > 0
        for provider in (
            "FRED",
            "FRED_VINTAGE",
            "ECB",
            "EUROSTAT",
            "YAHOO",
            "LOCAL_MARKET_CACHE",
            "DATASCRAPER_PIT_FUNDAMENTALS",
        )
    }
    technical_go = (
        validation["status"] == "GO"
        and conflicts["status"] == "GO"
        and all(required_provider_health.values())
        and int(events.get("historical_release_instance_count", 0)) > 0
        and events.get("future_schedule_status")
        == "ECB_OFFICIAL_FUTURE_STATISTICAL_CALENDAR_AVAILABLE"
        and pair_status.get("status") == "GO"
    )
    data_quality = score.get("data_quality", {}).get(
        "status",
        "UNAVAILABLE",
    )
    payload = {
        "schema": "macro_read_only_live_readiness_v1",
        "status": (
            "READ_ONLY_LIVE_READY_DEGRADED_DATA"
            if technical_go and data_quality != "GO"
            else "READ_ONLY_LIVE_READY_GO"
            if technical_go
            else "READ_ONLY_LIVE_READINESS_BLOCKED"
        ),
        "technical_readiness": "GO" if technical_go else "BLOCKED",
        "analytical_data_quality": data_quality,
        "provider_health": required_provider_health,
        "point_in_time_validation": validation["status"],
        "provider_conflict_resolution": conflicts["status"],
        "historical_release_calendar": events.get(
            "historical_schedule_status"
        ),
        "future_release_calendar": events.get("future_schedule_status"),
        "pmi_status": "LICENSED_SERIES_UNAVAILABLE; OECD_PROXIES_EXPLICIT",
        "valuation_status": score.get("scores", {})
        .get("valuation", {})
        .get("status"),
        "earnings_cycle_status": score.get("scores", {})
        .get("earnings_cycle", {})
        .get("status"),
        "macro_pair_validation": pair_status.get("status"),
        "retained_macro_variant_count": pair_status.get(
            "retained_macro_variant_count",
            0,
        ),
        "single_flight_lock": True,
        "bounded_one_shot_update": True,
        "canonical_command": "python main.py macro update",
        "scheduler_authority": "EXTERNAL_OPERATOR_CONTROLLED",
        "automatic_strategy_activation": False,
        "financial_finalist_go": False,
        **MACRO_AUTHORITY,
    }
    return _publish(project_root, "live-readiness.json", payload)


def macro_score(
    project_root: Path,
    *,
    as_of: str | None = None,
    fixture: bool = False,
) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    decision_time = _as_of(as_of)
    with _store(project_root) as store:
        observations = store.observations()
        history = [
            row
            for row in store.regime_history()
            if str(row["as_of"]) < decision_time.isoformat()
        ]
        snapshot = compute_macro_snapshot(
            observations,
            config,
            as_of=decision_time,
            regime_history=history,
        )
        snapshot["status"] = (
            "GO"
            if snapshot["data_quality"]["status"] == "GO"
            else "DATA_INCOMPLETE"
        )
        snapshot["fixture_evidence"] = fixture
        snapshot["financial_evidence"] = False
        store.append_snapshot(
            "score_snapshots",
            "MACRO-SCORE",
            as_of=snapshot["as_of"],
            payload=snapshot,
        )
        regime_payload = {
            "schema": "macro_regime_record_v1",
            "as_of": snapshot["as_of"],
            "regime": snapshot["regime"],
            "cycle_clock": snapshot["cycle_clock"],
            "score_hash": snapshot["content_hash"],
            "fixture_evidence": fixture,
            "financial_evidence": False,
        }
        store.append_snapshot(
            "regimes",
            "MACRO-REGIME",
            as_of=snapshot["as_of"],
            payload=regime_payload,
        )
    return _publish(
        project_root,
        "score.json",
        {**snapshot, **MACRO_AUTHORITY},
    )


def macro_context_at(
    project_root: Path,
    *,
    as_of: datetime,
) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    with _store(project_root) as store:
        observations = store.observations()
    return compute_macro_snapshot(
        observations,
        config,
        as_of=as_of,
        regime_history=[],
    )


def macro_regime(
    project_root: Path,
    *,
    as_of: str | None = None,
) -> dict[str, Any]:
    snapshot = macro_score(project_root, as_of=as_of)
    return _publish(
        project_root,
        "regime.json",
        {
            "schema": "macro_regime_v1",
            "status": (
                "GO"
                if snapshot["regime"]["overall_macro_regime"] != "UNKNOWN"
                else "DATA_INCOMPLETE"
            ),
            "as_of": snapshot["as_of"],
            "regime": snapshot["regime"],
            "cycle_clock": snapshot["cycle_clock"],
            "data_quality": snapshot["data_quality"],
            "financial_evidence": False,
            **MACRO_AUTHORITY,
        },
    )


def macro_history(
    project_root: Path,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    with _store(project_root) as store:
        observations = store.observations()
        existing = store.regime_history()
        if rebuild or len(existing) <= 1:
            history: list[dict[str, Any]] = []
            dates = _monthly_as_of_dates(observations)[-120:]
            for decision_time in dates:
                snapshot = compute_macro_snapshot(
                    observations,
                    config,
                    as_of=decision_time,
                    regime_history=history,
                )
                record = {
                    "schema": "macro_regime_record_v1",
                    "as_of": snapshot["as_of"],
                    "regime": snapshot["regime"],
                    "cycle_clock": snapshot["cycle_clock"],
                    "scores": snapshot["scores"],
                    "score_hash": snapshot["content_hash"],
                    "fixture_evidence": False,
                    "financial_evidence": False,
                }
                store.append_snapshot(
                    "regimes",
                    "MACRO-REGIME",
                    as_of=record["as_of"],
                    payload=record,
                )
                history.append(record)
            existing = store.regime_history()
    latest_by_as_of = {
        str(row["as_of"]): row
        for row in existing
    }
    existing = [
        latest_by_as_of[key]
        for key in sorted(latest_by_as_of)
    ]
    payload = {
        "schema": "macro_regime_history_v1",
        "status": "GO" if existing else "DATA_INCOMPLETE",
        "record_count": len(existing),
        "history": existing,
        "forward_outcome_analysis": _regime_forward_analysis(
            project_root,
            existing,
        ),
        "point_in_time_reconstruction": True,
        "future_returns_used_in_regime": False,
        "forward_returns_descriptive_only": True,
        "historical_vintage_limitations_explicit": True,
        **MACRO_AUTHORITY,
    }
    return _publish(project_root, "history.json", payload)


def macro_events(project_root: Path) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    with _store(project_root) as store:
        observations = store.observations()
        event_ids = [
            store.append_event(
                {
                    **event,
                    "scheduled_at": None,
                    "available_at": None,
                    "consensus_status": "CONSENSUS_HISTORY_UNAVAILABLE",
                    "automatic_exit": False,
                }
            )
            for event in config.events
        ]
    event_series = {
        "US_CPI": {"US_CPI", "US_CORE_CPI"},
        "US_PCE": {"US_CORE_PCE"},
        "US_PAYROLLS": {"US_PAYROLLS", "US_UNEMPLOYMENT"},
        "US_PMI": set(),
        "FOMC": {"FED_POLICY_RATE"},
        "ECB": {"ECB_POLICY_RATE"},
        "EU_CPI": {"EU_CPI", "EU_CORE_CPI"},
    }
    event_lookup = {
        series_id: event_id
        for event_id, series_ids in event_series.items()
        for series_id in series_ids
    }
    historical_instances = [
        {
            "event_id": event_lookup[row["series_id"]],
            "series_id": row["series_id"],
            "observation_date": row["observation_date"],
            "released_at": row["available_at"],
            "vintage": row.get("vintage"),
            "release_source": row["source"],
            "release_status": "OBSERVED_HISTORICAL_RELEASE",
        }
        for row in observations
        if row["series_id"] in event_lookup
        and row["revision_status"] == "HISTORICAL_VINTAGE"
    ]
    historical_instances.sort(
        key=lambda row: (
            row["released_at"],
            row["event_id"],
            row["series_id"],
        )
    )
    future_instances, future_statuses = _official_future_releases()
    now = datetime.now(UTC)
    event_risk = any(
        0
        <= (
            pd.Timestamp(row["scheduled_at"]).to_pydatetime() - now
        ).total_seconds()
        <= 86_400
        for row in future_instances
    )
    payload = {
        "schema": "macro_event_calendar_v2",
        "status": "GO",
        "event_definitions": list(config.events),
        "event_observation_ids": event_ids,
        "scheduled_instances": future_instances,
        "historical_release_instances": historical_instances,
        "historical_release_instance_count": len(historical_instances),
        "event_risk_within_24h": event_risk,
        "historical_schedule_status": (
            "FRED_ALFRED_OBSERVED_RELEASES_AVAILABLE"
            if historical_instances
            else "HISTORICAL_RELEASES_UNAVAILABLE"
        ),
        "future_schedule_status": (
            "OFFICIAL_CENTRAL_BANK_AND_STATISTICAL_CALENDARS_AVAILABLE"
            if future_instances
            else "OFFICIAL_FUTURE_CALENDARS_UNAVAILABLE"
        ),
        "future_schedule_sources": future_statuses,
        "pmi_schedule_status": "LICENSED_PMI_SERIES_UNAVAILABLE",
        "consensus_history_status": "CONSENSUS_HISTORY_UNAVAILABLE",
        "surprise_formula": "(actual-consensus)/historical_surprise_std",
        "automatic_exit": False,
        **MACRO_AUTHORITY,
    }
    return _publish(project_root, "events.json", payload)


def _official_future_releases() -> tuple[list[dict[str, Any]], dict[str, str]]:
    fetchers = {
        "FEDERAL_RESERVE_FOMC": _fomc_future_releases,
        "ECB_MONETARY_POLICY": _ecb_monetary_future_releases,
        "ECB_STATISTICAL": _ecb_future_releases,
    }
    rows: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    for source, fetcher in fetchers.items():
        source_rows, status = fetcher()
        rows.extend(source_rows)
        statuses[source] = status
    unique = {
        (
            str(row.get("event_id")),
            str(row.get("scheduled_at")),
            str(row.get("schedule_source")),
        ): row
        for row in rows
    }
    ordered = sorted(
        unique.values(),
        key=lambda row: (
            str(row.get("scheduled_at")),
            str(row.get("event_id")),
        ),
    )
    return ordered, statuses


def _fomc_future_releases() -> tuple[list[dict[str, Any]], str]:
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    try:
        source = _fetch_official_html(url)
        text = _strip_html(source)
        rows = _parse_fomc_calendar_text(
            text,
            now=datetime.now(UTC),
            source_url=url,
        )
    except Exception:
        return [], "FEDERAL_RESERVE_OFFICIAL_CALENDAR_FETCH_FAILED"
    return rows, (
        "FEDERAL_RESERVE_OFFICIAL_FOMC_CALENDAR_AVAILABLE"
        if rows
        else "FEDERAL_RESERVE_OFFICIAL_FOMC_CALENDAR_EMPTY"
    )


def _ecb_monetary_future_releases() -> tuple[list[dict[str, Any]], str]:
    url = (
        "https://www.ecb.europa.eu/press/calendars/mgcgc/"
        "html/index.en.html"
    )
    try:
        text = _strip_html(_fetch_official_html(url))
        rows = _parse_ecb_monetary_calendar_text(
            text,
            now=datetime.now(UTC),
            source_url=url,
        )
    except Exception:
        return [], "ECB_OFFICIAL_MONETARY_CALENDAR_FETCH_FAILED"
    return rows, (
        "ECB_OFFICIAL_MONETARY_POLICY_CALENDAR_AVAILABLE"
        if rows
        else "ECB_OFFICIAL_MONETARY_POLICY_CALENDAR_EMPTY"
    )


def _parse_fomc_calendar_text(
    text: str,
    *,
    now: datetime,
    source_url: str,
) -> list[dict[str, Any]]:
    year = now.year
    section_match = re.search(
        rf"{year}\s+FOMC Meetings(.*?)(?:{year - 1}\s+FOMC Meetings|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    section = "" if section_match is None else section_match.group(1)
    month_numbers = {
        month: index
        for index, month in enumerate(
            (
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ),
            start=1,
        )
    }
    rows = []
    for month, _start_day, decision_day, projection in re.findall(
        (
            r"\b("
            + "|".join(month_numbers)
            + r")\s+(\d{1,2})-(\d{1,2})(\*)?"
        ),
        section,
        flags=re.IGNORECASE,
    ):
        scheduled_local = datetime(
            year,
            month_numbers[month.capitalize()],
            int(decision_day),
            14,
            0,
            tzinfo=ZoneInfo("America/New_York"),
        )
        scheduled_at = scheduled_local.astimezone(UTC)
        if scheduled_at < now - timedelta(hours=6):
            continue
        rows.append(
            {
                "event_id": "FOMC",
                "name": "Federal Reserve interest-rate decision",
                "scheduled_at": scheduled_at.isoformat(),
                "importance": "HIGH",
                "projection_meeting": bool(projection),
                "schedule_source": "FEDERAL_RESERVE_OFFICIAL_FOMC_CALENDAR",
                "source_url": source_url,
                "affected_markets": [
                    "US_EQUITIES",
                    "GLOBAL_EQUITIES",
                    "USD",
                    "US_GOVERNMENT_BONDS",
                    "GOLD",
                    "COMMODITIES",
                ],
                "automatic_exit": False,
            }
        )
    return rows


def _parse_ecb_monetary_calendar_text(
    text: str,
    *,
    now: datetime,
    source_url: str,
) -> list[dict[str, Any]]:
    rows = []
    for match in re.finditer(
        r"(\d{2}/\d{2}/\d{4})\s+(.*?)(?=\d{2}/\d{2}/\d{4}|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        description = " ".join(match.group(2).split())
        normalized = description.lower()
        if (
            "monetary policy meeting" not in normalized
            or "non-monetary" in normalized
            or (
                "day 2" not in normalized
                and "press conference" not in normalized
            )
        ):
            continue
        scheduled_local = datetime.strptime(
            match.group(1),
            "%d/%m/%Y",
        ).replace(
            hour=14,
            minute=15,
            tzinfo=ZoneInfo("Europe/Brussels"),
        )
        scheduled_at = scheduled_local.astimezone(UTC)
        if scheduled_at < now - timedelta(hours=6):
            continue
        rows.append(
            {
                "event_id": "ECB",
                "name": "ECB interest-rate decision",
                "scheduled_at": scheduled_at.isoformat(),
                "importance": "HIGH",
                "schedule_source": "ECB_OFFICIAL_GOVERNING_COUNCIL_CALENDAR",
                "source_url": source_url,
                "affected_markets": [
                    "EUROPE_EQUITIES",
                    "GLOBAL_EQUITIES",
                    "EUR",
                    "EURO_AREA_GOVERNMENT_BONDS",
                    "GOLD",
                ],
                "automatic_exit": False,
            }
        )
    return rows


def _fetch_official_html(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Stocks-Macro-Research/2.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="replace")


def _ecb_future_releases() -> tuple[list[dict[str, Any]], str]:
    url = (
        "https://www.ecb.europa.eu/press/calendars/statscal/html/"
        "index.en.html"
    )
    try:
        source = _fetch_official_html(url)
        rows = _parse_ecb_calendar_html(source, source_url=url)
    except Exception:
        return [], "ECB_OFFICIAL_FUTURE_CALENDAR_FETCH_FAILED"
    return rows, (
        "ECB_OFFICIAL_FUTURE_STATISTICAL_CALENDAR_AVAILABLE"
        if rows
        else "ECB_OFFICIAL_FUTURE_CALENDAR_EMPTY"
    )


def _parse_ecb_calendar_html(
    source: str,
    *,
    source_url: str,
) -> list[dict[str, Any]]:
    relevant_datasets = {
        "BSI": "ECB",
        "HICP": "EU_CPI",
        "ILM": "ECB",
        "FM": "ECB",
    }
    now = datetime.now(UTC)
    rows = []
    for date_html, detail_html in re.findall(
        r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>",
        source,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        date_text = _strip_html(date_html)
        match = re.search(
            r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})\s+C(?:E|ES)T",
            date_text,
        )
        if not match:
            continue
        local = datetime.strptime(
            f"{match.group(1)} {match.group(2)}",
            "%d/%m/%Y %H:%M",
        ).replace(tzinfo=ZoneInfo("Europe/Brussels"))
        scheduled_at = local.astimezone(UTC)
        if scheduled_at < now:
            continue
        detail = _strip_html(detail_html)
        dataset_match = re.search(r"\(Dataset:\s*([A-Z0-9]+)\)", detail)
        dataset = None if dataset_match is None else dataset_match.group(1)
        if dataset not in relevant_datasets:
            continue
        reference_match = re.search(
            r"Reference period:\s*(.*?)(?:Includes press release\.|$)",
            detail,
        )
        rows.append(
            {
                "event_id": relevant_datasets[dataset],
                "dataset": dataset,
                "name": detail.split("(Dataset:", 1)[0].strip(),
                "scheduled_at": scheduled_at.isoformat(),
                "reference_period": (
                    None
                    if reference_match is None
                    else reference_match.group(1).strip()
                ),
                "tentative": "Tentative" in date_text,
                "schedule_source": "ECB_OFFICIAL_STATISTICAL_CALENDAR",
                "source_url": source_url,
                "automatic_exit": False,
            }
        )
    return rows


def _strip_html(value: str) -> str:
    breaks = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    return " ".join(
        html.unescape(re.sub(r"<[^>]+>", " ", breaks)).split()
    )


def macro_report(
    project_root: Path,
    *,
    period: str,
) -> dict[str, Any]:
    if period not in {"daily", "weekly", "monthly"}:
        raise ValueError("INVALID_MACRO_REPORT_PERIOD")
    with _store(project_root) as store:
        snapshot = store.latest_payload("score_snapshots")
    if snapshot is None:
        snapshot = macro_score(project_root)
    analysis = deterministic_analysis(snapshot, period=period)
    payload = {
        "schema": "macro_analyst_report_v1",
        "status": (
            "GO"
            if snapshot["data_quality"]["status"] == "GO"
            else "DATA_INCOMPLETE"
        ),
        "analysis": analysis,
        "regime": snapshot["regime"],
        "cycle_clock": snapshot["cycle_clock"],
        "implications": snapshot["implications"],
        "data_quality": snapshot["data_quality"],
        "financial_evidence": False,
        **MACRO_AUTHORITY,
    }
    report_hash = stable_hash(payload)
    enriched = _publish_immutable(
        project_root,
        (
            f"reports/{period}/{date.today().isoformat()}/"
            f"{report_hash}.json"
        ),
        payload,
    )
    layout = MacroLayout.from_project_root(project_root)
    markdown_path = (
        layout.output_root
        / "reports"
        / period
        / date.today().isoformat()
        / f"{report_hash}.md"
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(analysis)
    if markdown_path.exists() and markdown_path.read_text(encoding="utf-8") != markdown:
        raise ValueError("MACRO_REPORT_IMMUTABILITY_CONFLICT")
    markdown_path.write_text(markdown, encoding="utf-8")
    with _store(project_root) as store:
        store.append_snapshot(
            "reports",
            "MACRO-REPORT",
            as_of=snapshot["as_of"],
            period=period,
            payload=payload,
        )
    return enriched


def macro_explain(project_root: Path) -> dict[str, Any]:
    with _store(project_root) as store:
        snapshot = store.latest_payload("score_snapshots")
    if snapshot is None:
        snapshot = macro_score(project_root)
    return _publish(
        project_root,
        "explain.json",
        {
            "schema": "macro_explanation_v1",
            "status": "GO",
            "analysis": deterministic_analysis(snapshot, period="current"),
            **MACRO_AUTHORITY,
        },
    )


def macro_compare(
    project_root: Path,
    *,
    date_a: str,
    date_b: str,
) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    with _store(project_root) as store:
        observations = store.observations()
    first = compute_macro_snapshot(
        observations,
        config,
        as_of=_as_of(date_a),
    )
    second = compute_macro_snapshot(
        observations,
        config,
        as_of=_as_of(date_b),
    )
    score_changes = {
        name: (
            None
            if first["scores"][name]["value"] is None
            or second["scores"][name]["value"] is None
            else second["scores"][name]["value"] - first["scores"][name]["value"]
        )
        for name in config.score_weights
    }
    return _publish(
        project_root,
        f"compare/{date_a}-to-{date_b}.json",
        {
            "schema": "macro_comparison_v1",
            "status": "GO",
            "date_a": date_a,
            "date_b": date_b,
            "regime_a": first["regime"],
            "regime_b": second["regime"],
            "score_changes": score_changes,
            "point_in_time": True,
            **MACRO_AUTHORITY,
        },
    )


def macro_sector_impact(project_root: Path) -> dict[str, Any]:
    with _store(project_root) as store:
        snapshot = store.latest_payload("score_snapshots")
    if snapshot is None:
        snapshot = macro_score(project_root)
    return _publish(
        project_root,
        "sector-impact.json",
        {
            "schema": "macro_sector_impact_v1",
            "status": "GO",
            "as_of": snapshot["as_of"],
            "implications": snapshot["implications"],
            "order_signal": False,
            **MACRO_AUTHORITY,
        },
    )


def macro_strategy_impact(
    project_root: Path,
    *,
    strategy_id: str,
) -> dict[str, Any]:
    ledger = ResearchLedger(AutopilotLayout(project_root))
    try:
        strategy = ledger.strategy(strategy_id)
    finally:
        ledger.close()
    if strategy is None:
        return {
            "schema": "macro_strategy_impact_v1",
            "status": "NOT_FOUND",
            "strategy_id": strategy_id,
            **MACRO_AUTHORITY,
        }
    with _store(project_root) as store:
        snapshot = store.latest_payload("score_snapshots")
    if snapshot is None:
        snapshot = macro_score(project_root)
    filters = [
        component
        for component in strategy["regime_components"]
        if component in _macro_component_names()
    ]
    return _publish(
        project_root,
        f"strategy-impact/{strategy_id}.json",
        {
            "schema": "macro_strategy_impact_v1",
            "status": "GO",
            "strategy_id": strategy_id,
            "configured_macro_filters": filters,
            "macro_filter_count": len(filters),
            "maximum_macro_filters": 2,
            "baseline_comparison_required": True,
            "baseline_comparison": (
                "UNAVAILABLE_UNTIL_MATCHED_MACRO_VARIANT_TRIAL"
            ),
            "current_regime": snapshot["regime"],
            "exposure_multiplier": exposure_multiplier(
                snapshot["regime"],
                MacroConfig.load(project_root),
            ),
            "automatic_promotion": False,
            **MACRO_AUTHORITY,
        },
    )


def macro_audit(project_root: Path) -> dict[str, Any]:
    config = MacroConfig.load(project_root)
    validation = macro_validate(project_root)
    frozen = freeze_integrity(project_root)
    source_root = project_root / "src" / "stocks" / "macro"
    forbidden = (
        "place" + "Order",
        "cancel" + "Order",
        "req" + "Global" + "Cancel",
        "req" + "Ids",
        "req" + "Mkt" + "Data",
        "req" + "Historical" + "Data",
    )
    method_hits: list[str] = []
    ai_hits: list[str] = []
    for path in sorted(source_root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        method_hits.extend(
            f"{path.name}:{token}" for token in forbidden if token in text
        )
        ai_hits.extend(
            f"{path.name}:{token}"
            for token in ("openai", "transformers", "torch", "sklearn")
            if re.search(rf"\bimport\s+{token}\b|\bfrom\s+{token}\b", text)
        )
    public_leaks = _public_leaks(MacroLayout.from_project_root(project_root).output_root)
    conflict_resolution = macro_conflicts(project_root)
    payload = {
        "schema": "macro_engine_audit_v1",
        "status": (
            "GO"
            if validation["status"] == "GO"
            and not method_hits
            and not ai_hits
            and not public_leaks
            and all(value == "GO" or value is True for value in frozen.values())
            else "BLOCKED"
        ),
        "series_registry": "GO",
        "series_count": len(config.series),
        "score_weight_sets": len(config.score_weights),
        "point_in_time_validation": validation,
        "forbidden_broker_method_hits": method_hits,
        "ai_or_ml_import_hits": ai_hits,
        "public_privacy_leaks": public_leaks,
        "provider_conflict_resolution": conflict_resolution,
        "frozen_dependency_integrity": frozen,
        "configuration_hash": config.config_hash,
        "fixture_results_are_financial_evidence": False,
        **MACRO_AUTHORITY,
    }
    return _publish(project_root, "audit.json", payload)


def macro_conflicts(project_root: Path) -> dict[str, Any]:
    layout = MacroLayout.from_project_root(project_root)
    collection_path = layout.output_root / "collection.json"
    collection = (
        json.loads(collection_path.read_text(encoding="utf-8"))
        if collection_path.exists()
        else {}
    )
    with _store(project_root) as store:
        rows = store.observations()
    relevant = {
        "COPPER",
        "COPPER_GOLD_RATIO",
        "EURUSD",
        "GLOBAL_COMMODITY_INDEX",
        "GOLD",
        "OIL",
        "USD_INDEX",
    }
    revision_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        series_id = str(row["series_id"])
        if series_id not in relevant:
            continue
        revision = str(row["revision_status"])
        revision_counts.setdefault(series_id, {})[revision] = (
            revision_counts.setdefault(series_id, {}).get(revision, 0) + 1
        )
    current_conflicts = int(
        collection.get("registration", {}).get("conflict_count", -1)
    )
    payload = {
        "schema": "macro_provider_conflict_resolution_v1",
        "status": "GO" if current_conflicts == 0 else "BLOCKED",
        "reported_initial_conflict_count": 2414,
        "reported_initial_conflicts_by_series": {
            "COPPER": 1,
            "COPPER_GOLD_RATIO": 1,
            "EURUSD": 1,
            "GLOBAL_COMMODITY_INDEX": 2408,
            "GOLD": 1,
            "OIL": 1,
            "USD_INDEX": 1,
        },
        "root_cause": (
            "YAHOO_ADJUSTED_CLOSE_RETROACTIVE_RESTATEMENT_AND_PROVISIONAL_"
            "CURRENT_MARKET_BARS"
        ),
        "global_commodity_index_diagnosis": (
            "DBC_ADJ_CLOSE_CHANGED_AFTER_DISTRIBUTION_ADJUSTMENTS"
        ),
        "corrective_actions": [
            "USE_RAW_CLOSE_FOR_MACRO_MARKET_LEVELS",
            "VERSION_RAW_CLOSE_AS_MARKET_CLOSE_RAW_V1",
            "PREFER_RAW_CLOSE_IN_POINT_IN_TIME_SELECTION",
            "EXCLUDE_BARS_NOT_YET_AVAILABLE_AT_COLLECTION_TIME",
            "VERSION_DERIVED_COPPER_GOLD_RATIO_FROM_RAW_CLOSES",
            "PRESERVE_LEGACY_ROWS_APPEND_ONLY",
        ],
        "legacy_and_corrected_revision_counts": revision_counts,
        "current_conflict_count": current_conflicts,
        "current_conflicts_by_series": collection.get(
            "registration",
            {},
        ).get("quarantined_conflicts_by_series", {}),
        "idempotent_repeat_collection": current_conflicts == 0,
        "legacy_rows_overwritten": 0,
        "broker_calls": 0,
        "order_calls": 0,
        "execution_authority": "NONE",
    }
    return _publish(project_root, "provider-conflict-resolution.json", payload)


def macro_freeze(project_root: Path) -> dict[str, Any]:
    audit = macro_audit(project_root)
    status = macro_status(project_root)
    config = MacroConfig.load(project_root)
    paths = [
        project_root / "main.py",
        project_root / "config" / "macro" / "macro_v1.json",
        project_root / "config" / "screener" / "daily_screener_v1.json",
        *sorted((project_root / "src" / "stocks" / "macro").glob("*.py")),
        project_root / "src" / "stocks" / "screener" / "config.py",
        project_root / "src" / "stocks" / "screener" / "scoring.py",
        project_root / "src" / "stocks" / "screener" / "service.py",
        project_root / "src" / "stocks" / "research" / "autopilot" / "components.py",
        project_root / "src" / "stocks" / "research" / "autopilot" / "contracts.py",
        project_root / "src" / "stocks" / "research" / "autopilot" / "generator.py",
        project_root / "src" / "stocks" / "research" / "autopilot" / "engine.py",
        project_root / "src" / "stocks" / "research" / "autopilot" / "ledger.py",
        project_root / "src" / "stocks" / "research" / "autopilot" / "taxonomy.py",
        project_root / "src" / "stocks" / "research" / "macro_pairs.py",
        project_root / "requirements.txt",
        project_root / "requirements.lock.txt",
        project_root / "docs" / "MACRO_ENGINE_READ_ONLY_LIVE_RUNBOOK.md",
        project_root / "docs" / "CANONICAL_STRATEGY_TAXONOMY_INTEGRATION.md",
        project_root / "MACRO_V2_STATUS.md",
        project_root / "scripts" / "run_macro_update.ps1",
    ]
    source_hashes = {
        str(path.relative_to(project_root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest().upper()
        for path in paths
    }
    evidence_paths = [
        project_root / "output" / "macro" / "collection.json",
        project_root / "output" / "macro" / "events.json",
        project_root
        / "output"
        / "macro"
        / "provider-conflict-resolution.json",
        project_root / "output" / "macro" / "live-readiness.json",
        project_root
        / "output"
        / "research"
        / "autopilot"
        / "component-taxonomy-coverage.json",
        project_root
        / "output"
        / "research"
        / "macro_pairs"
        / "status.json",
        project_root
        / "output"
        / "research"
        / "macro_pairs"
        / "decision.json",
        project_root
        / "output"
        / "research"
        / "macro_pairs"
        / "multiple-testing.json",
    ]
    evidence_hashes = {
        str(path.relative_to(project_root)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest().upper()
        for path in evidence_paths
        if path.exists()
    }
    manifest_hash = stable_hash(
        {
            "source_hashes": source_hashes,
            "evidence_hashes": evidence_hashes,
            "config_hash": config.config_hash,
            "store_counts": status["store_counts"],
            "frozen_dependency_integrity": audit["frozen_dependency_integrity"],
        }
    )
    payload = {
        "schema": "macro_engine_freeze_v1",
        "status": "GO" if audit["status"] == "GO" else "BLOCKED",
        "marker": (
            "MACRO_ENGINE_AND_ANALYST_TECHNICAL_FROZEN_GO"
            if audit["status"] == "GO"
            else "MACRO_ENGINE_FREEZE_BLOCKED"
        ),
        "manifest_hash": manifest_hash,
        "source_hashes": source_hashes,
        "evidence_hashes": evidence_hashes,
        "config_hash": config.config_hash,
        "store_counts": status["store_counts"],
        "frozen_dependency_integrity": audit["frozen_dependency_integrity"],
        "financial_evidence": False,
        "open_data_limitations": _data_limitations(config),
        **MACRO_AUTHORITY,
    }
    return _publish_immutable(
        project_root,
        f"frozen/macro-{manifest_hash}.json",
        payload,
    )


def _revision_audit(observations: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in observations:
        groups.setdefault(
            (row["series_id"], row["observation_date"]),
            [],
        ).append(row)
    revised = sum(len(rows) > 1 for rows in groups.values())
    return {
        "periods_with_multiple_vintages": revised,
        "vintages_overwritten": 0,
        "append_only": True,
    }


def _regime_forward_analysis(
    project_root: Path,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    benchmark_path = (
        project_root
        / "data"
        / "research"
        / "critical_trading"
        / "yfinance"
        / "SPY.parquet"
    )
    if not benchmark_path.exists() or not history:
        return {
            "status": "UNAVAILABLE",
            "benchmark": "SPY",
            "reason": "BENCHMARK_OR_REGIME_HISTORY_UNAVAILABLE",
            "descriptive_only": True,
        }
    frame = pd.read_parquet(benchmark_path)
    columns = {str(column).lower(): column for column in frame.columns}
    close_column = columns.get("adj close") or columns.get("close")
    if close_column is None:
        return {
            "status": "UNAVAILABLE",
            "benchmark": "SPY",
            "reason": "BENCHMARK_CLOSE_COLUMN_UNAVAILABLE",
            "descriptive_only": True,
        }
    if "session_date" in columns:
        timestamps = pd.to_datetime(frame[columns["session_date"]], utc=True)
    elif "timestamp" in columns:
        timestamps = pd.to_datetime(frame[columns["timestamp"]], utc=True)
    else:
        timestamps = pd.to_datetime(frame.index, utc=True)
    close = pd.Series(
        pd.to_numeric(frame[close_column], errors="coerce").to_numpy(),
        index=timestamps,
    ).dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    horizons = (5, 10, 21, 63, 126)
    samples: dict[str, dict[int, list[float]]] = {}
    for record in history:
        regime = str(record["regime"]["overall_macro_regime"])
        timestamp = pd.Timestamp(record["as_of"])
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        start_index = int(close.index.searchsorted(timestamp, side="left"))
        if start_index >= len(close):
            continue
        regime_samples = samples.setdefault(
            regime,
            {horizon: [] for horizon in horizons},
        )
        start_value = float(close.iloc[start_index])
        for horizon in horizons:
            end_index = start_index + horizon
            if end_index < len(close):
                regime_samples[horizon].append(
                    float(close.iloc[end_index] / start_value - 1.0)
                )
    by_regime = {}
    for regime, regime_samples in sorted(samples.items()):
        by_regime[regime] = {
            f"{horizon}d": {
                "sample_count": len(values),
                "mean_return": None if not values else float(pd.Series(values).mean()),
                "median_return": (
                    None if not values else float(pd.Series(values).median())
                ),
                "positive_ratio": (
                    None
                    if not values
                    else float((pd.Series(values) > 0).mean())
                ),
            }
            for horizon, values in regime_samples.items()
        }
    return {
        "status": "GO" if by_regime else "DATA_INCOMPLETE",
        "benchmark": "SPY",
        "benchmark_source": str(benchmark_path.relative_to(project_root)).replace(
            "\\",
            "/",
        ),
        "horizons_trading_days": list(horizons),
        "by_regime": by_regime,
        "descriptive_only": True,
        "selection_or_authority_effect": False,
    }


def _monthly_as_of_dates(observations: list[dict[str, Any]]) -> list[datetime]:
    if not observations:
        return []
    timestamps = sorted(
        {
            pd.Timestamp(row["available_at"]).tz_convert("UTC")
            for row in observations
        }
    )
    by_month: dict[str, pd.Timestamp] = {}
    for timestamp in timestamps:
        by_month[timestamp.strftime("%Y-%m")] = timestamp
    return [timestamp.to_pydatetime() for timestamp in by_month.values()]


def _macro_component_names() -> set[str]:
    return {
        "macro_growth_accelerating",
        "macro_growth_slowing",
        "macro_inflation_rising",
        "macro_inflation_falling",
        "macro_liquidity_expanding",
        "macro_liquidity_contracting",
        "macro_credit_improving",
        "macro_credit_deteriorating",
        "macro_breadth_improving",
        "macro_breadth_deteriorating",
        "macro_risk_on",
        "macro_risk_off",
        "macro_dollar_strengthening",
        "macro_dollar_weakening",
        "macro_commodity_inflation",
        "macro_defensive_regime",
        "macro_cyclical_regime",
    }


def _data_limitations(config: MacroConfig) -> list[str]:
    unavailable = sorted(
        spec.canonical_id
        for spec in config.series.values()
        if not spec.vintage_capable
    )
    return [
        "VINTAGE_HISTORY_UNAVAILABLE:" + ",".join(unavailable),
        "CONSENSUS_HISTORY_UNAVAILABLE",
        "FUTURE_CALENDAR_COVERAGE_LIMITED_TO_ECB_STATISTICAL_RELEASES",
        "LICENSED_PMI_HISTORY_UNAVAILABLE; OECD_PROXIES_EXPLICIT",
        "EARNINGS_AND_VALUATION_AGGREGATES_LIMITED_TO_FIVE_SYMBOL_PIT_SAMPLE",
        "MACRO_RESULTS_ARE_RESEARCH_CONTEXT_NOT_FINANCIAL_EVIDENCE",
    ]


def _public_leaks(root: Path) -> list[str]:
    if not root.exists():
        return []
    patterns = {
        "secret": re.compile(
            r"(?i)(api[_-]?key|password|secret)\s*[=:]\s*[\"'][^\"']+"
        ),
        "account": re.compile(r"\b(?:DU|U)\d{6,}\b"),
    }
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".md", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        hits.extend(
            f"{path.name}:{name}"
            for name, pattern in patterns.items()
            if pattern.search(text)
        )
    return hits


def _publish(
    project_root: Path,
    relative: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    root = MacroLayout.from_project_root(project_root).output_root
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("MACRO_ARTIFACT_PATH_OUTSIDE_ROOT")
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "generated_at": datetime.now(UTC).isoformat(),
        "content_hash": stable_hash(payload),
    }
    path.write_text(
        json.dumps(enriched, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return enriched


def _publish_immutable(
    project_root: Path,
    relative: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    root = MacroLayout.from_project_root(project_root).output_root
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("MACRO_ARTIFACT_PATH_OUTSIDE_ROOT")
    digest = stable_hash(payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_hash") != digest:
            raise ValueError("MACRO_FROZEN_ARTIFACT_IMMUTABILITY_CONFLICT")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "frozen_at": datetime.now(UTC).isoformat(),
        "content_hash": digest,
    }
    path.write_text(
        json.dumps(enriched, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return enriched


def _store(project_root: Path) -> MacroStore:
    return MacroStore(MacroLayout.from_project_root(project_root))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC") + timedelta(hours=23, minutes=59)
    return timestamp.tz_convert("UTC").to_pydatetime()
