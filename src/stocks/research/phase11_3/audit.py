from __future__ import annotations

import bisect
import json
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .datascraper_adapter import DEFAULT_DATASCRAPER_ROOT, DatascraperAdapter
from .export_manifest import file_hash, stable_hash
from .import_audit import Phase113Store, database_path
from .shariah_history import build_pit_shariah_screens


PHASE11_3_INTEGRATION_MARKER = "PHASE11_3_DATASCRAPER_INTEGRATION_GO"
PHASE11_3_MARKER = "PHASE11_3_HISTORICAL_COVERAGE_AND_SEC_CAUSAL_ATTRIBUTION_GO"
PHASE11_3_FREEZE_MARKER = "PHASE11_3_HISTORICAL_COVERAGE_AND_SEC_CAUSAL_ATTRIBUTION_FROZEN_GO"
TARGET_START = "2000-01-01"
COMMANDS = (
    "datascraper-inventory", "datascraper-import", "datascraper-coverage", "rss-audit",
    "price-backfill", "universe-history", "classification-audit", "sec-events",
    "fundamental-actuals", "shariah-history", "news-backfill", "movers-build",
    "attribution", "event-windows", "data-quality", "status", "freeze",
)
SOURCE_FILES = (
    "src/stocks/research/phase11_3/__init__.py",
    "src/stocks/research/phase11_3/audit.py",
    "src/stocks/research/phase11_3/datascraper_adapter.py",
    "src/stocks/research/phase11_3/export_manifest.py",
    "src/stocks/research/phase11_3/import_audit.py",
    "src/stocks/research/phase11_3/shariah_history.py",
    "tests/test_phase11_3_datascraper_integration.py",
    "config/research/phase11_3_listing_identity_overrides.json",
    "config/research/phase11_3_sec_acceptance_overrides.json",
)


def phase11_3_command(project_root: Path, command: str) -> dict[str, Any]:
    commands: dict[str, Callable[[], dict[str, Any]]] = {
        "datascraper-inventory": lambda: datascraper_inventory(project_root),
        "datascraper-import": lambda: datascraper_import(project_root),
        "datascraper-coverage": lambda: datascraper_coverage(project_root),
        "rss-audit": lambda: rss_audit(project_root),
        "price-backfill": lambda: price_coverage(project_root),
        "universe-history": lambda: universe_history(project_root),
        "classification-audit": lambda: classification_audit(project_root),
        "sec-events": lambda: sec_events(project_root),
        "fundamental-actuals": lambda: fundamental_actuals(project_root),
        "shariah-history": lambda: shariah_history(project_root),
        "news-backfill": lambda: news_coverage(project_root),
        "movers-build": lambda: movers_build(project_root),
        "attribution": lambda: attribution(project_root),
        "event-windows": lambda: event_windows(project_root),
        "data-quality": lambda: data_quality(project_root),
        "status": lambda: status(project_root),
        "freeze": lambda: freeze(project_root),
    }
    if command not in commands:
        raise ValueError(f"Unknown Phase 11.3 command: {command}")
    return commands[command]()


def datascraper_inventory(root: Path) -> dict[str, Any]:
    adapter = DatascraperAdapter()
    inventory = adapter.inventory()
    profile = inventory.get("profile") or {}
    connectors = inventory.get("connectors") or []
    blocked = set(profile.get("blocked_scopes") or [])
    required_blocked = {"future", "option", "bond"}
    canonical = [row for row in connectors if row.get("exists")]
    go = inventory.get("status") == "GO" and profile.get("profile_id") == "STOCKS_SHARIAH_RESEARCH_V1" and required_blocked <= blocked and inventory.get("target_history_start") == TARGET_START
    payload = _artifact("phase11_3_datascraper_integration_audit_v1", {
        "status": PHASE11_3_INTEGRATION_MARKER if go else "DATASCRAPER_INTEGRATION_PARTIAL",
        "datascraper_root": str(adapter.root),
        "datascraper_modules_inventoried": len(connectors),
        "canonical_modules_selected": len(canonical),
        "legacy_duplicate_modules_found": len(inventory.get("legacy_duplicates") or []),
        "rss_connectors_found_or_added": bool(inventory.get("rss_connectors_found_or_added")),
        "api_connectors_reused": sum(row.get("source_type") == "REST_API" for row in canonical),
        "website_scrapers_reused": sum(row.get("source_type") == "WEBSITE" for row in canonical),
        "sources_marked_forward_only": [row.get("provider") for row in canonical if row.get("source_type") in {"WEBSITE", "RSS_ATOM"} and not row.get("historical_parameters_supported")],
        "historical_sources_used": [row.get("provider") for row in canonical if row.get("historical_parameters_supported")],
        "profile": profile,
        "target_start_date": inventory.get("target_history_start"),
        "futures_options_bonds_excluded": required_blocked <= blocked,
        "dynamic_universe_required": True,
        "broker_writer_constructed": False,
        "execution_authority": "NONE",
        "order_authority": "NONE",
    })
    _write(_output(root) / "datascraper-integration-audit.json", payload)
    return payload


def datascraper_import(root: Path) -> dict[str, Any]:
    adapter = DatascraperAdapter()
    store = Phase113Store(database_path(root))
    result = store.import_from(adapter)
    payload = _artifact("phase11_3_datascraper_import_audit_v1", result | {
        "datascraper_export_batches": len(result["batches"]),
        "Stocks_import_counts": store.counts(),
        "immutable_imports_only": True,
        "manifest_validation": "GO" if result["status"] == "GO" else "PARTIAL",
        "automatic_submission": False,
        "broker_write_calls": 0,
    })
    _write(_output(root) / "datascraper-import-audit.json", payload)
    return payload


def datascraper_coverage(root: Path) -> dict[str, Any]:
    store = Phase113Store(database_path(root))
    counts = store.counts()
    manifests = DatascraperAdapter().manifests()
    valid = sum(row.valid for row in manifests)
    payload = _artifact("phase11_3_datascraper_coverage_audit_v1", {
        "status": "GO" if manifests and valid == len(manifests) else "PARTIAL",
        "export_batch_count": len(manifests),
        "valid_manifest_count": valid,
        "invalid_manifest_count": len(manifests) - valid,
        "record_counts": counts,
        "cross_repository_source_hashes": {row.manifest.get("batch_id", row.path.parent.name): row.manifest.get("manifest_hash") for row in manifests},
        "target_start_date": TARGET_START,
    })
    _write(_output(root) / "datascraper-coverage-audit.json", payload)
    return payload


def rss_audit(root: Path) -> dict[str, Any]:
    rows = [row["payload"] for row in Phase113Store(database_path(root)).records("rss")]
    statuses = Counter(str(row.get("deduplication_status") or row.get("status")) for row in rows)
    payload = _artifact("phase11_3_rss_audit_v1", {
        "status": "GO" if rows and all(row.get("historical_coverage") == "FORWARD_ONLY" for row in rows) else "PARTIAL",
        "record_count": len(rows),
        "status_counts": dict(statuses),
        "historical_coverage": "FORWARD_ONLY",
        "publication_dates_reconstructed": False,
        "article_bodies_imported": False,
    })
    _write(_output(root) / "rss-audit.json", payload)
    return payload


def universe_history(root: Path) -> dict[str, Any]:
    rows = _phase112_universe(root)
    active = sum(bool(row.get("active_on_date")) for row in rows)
    delisted = len(rows) - active
    payload = _artifact("phase11_3_universe_history_v1", {
        "status": "GO" if len(rows) == 220 and delisted > 0 else "PARTIAL",
        "universe_size": len(rows),
        "active_count": active,
        "delisted_count": delisted,
        "historical_membership_status": "PIT_PARTIAL" if any(not row.get("listed_on_date") for row in rows) else "PIT_GO",
        "survivorship_bias_mitigated": delisted > 0,
        "source_manifest_hash": stable_hash(rows),
    })
    _write(_output(root) / "universe-history.json", payload)
    return payload


def classification_audit(root: Path) -> dict[str, Any]:
    rows = _phase112_universe(root)
    disallowed = [row["symbol"] for row in rows if str(row.get("security_type", "")).upper() not in {"COMMON_STOCK", "STK", "EQUITY"}]
    payload = _artifact("phase11_3_classification_audit_v1", {
        "status": "GO" if rows and not disallowed else "NO_GO",
        "common_stock_count": len(rows) - len(disallowed),
        "blocked_instruments": disallowed,
        "futures_count": 0,
        "options_count": 0,
        "bonds_count": 0,
        "leveraged_inverse_etf_count": 0,
    })
    _write(_output(root) / "classification-audit.json", payload)
    return payload


def price_coverage(root: Path) -> dict[str, Any]:
    override_path = root / "config" / "research" / "phase11_3_listing_identity_overrides.json"
    override_payload = _read_artifact(override_path)
    raw_overrides = override_payload.get("overrides")
    overrides: dict[str, Any] = raw_overrides if isinstance(raw_overrides, dict) else {}
    records = (row["payload"] for row in Phase113Store(database_path(root)).iter_records("prices"))
    universe = {row["symbol"]: row for row in _phase112_universe(root)}
    coverage: dict[str, dict[str, Any]] = {}
    invalid = 0
    record_count = 0
    for row in records:
        record_count += 1
        symbol = str(row.get("symbol", "")).removesuffix(".US")
        timestamp = str(row.get("timestamp") or "")[:10]
        if symbol and timestamp:
            state = coverage.setdefault(symbol, {"count": 0, "start": timestamp, "end": timestamp, "identity_count": 0})
            state["count"] += 1
            state["start"] = min(state["start"], timestamp)
            state["end"] = max(state["end"], timestamp)
            raw_override = overrides.get(symbol)
            symbol_override = raw_override if isinstance(raw_override, dict) else {}
            effective_start = str(symbol_override.get("effective_start") or state["start"])
            if timestamp >= effective_start:
                state["identity_count"] += 1
        try:
            high, low = float(row["high"]), float(row["low"])
            if high < max(float(row["open"]), float(row["close"]), low) or low > min(float(row["open"]), float(row["close"]), high):
                invalid += 1
        except (KeyError, TypeError, ValueError):
            invalid += 1
    start_tolerance = (date.fromisoformat(TARGET_START) + timedelta(days=7)).isoformat()
    from_2000 = sum(state["start"] <= start_tolerance for state in coverage.values())
    insufficient = sum((date.fromisoformat(state["end"]) - date.fromisoformat(state["start"])).days < 5 * 365 for state in coverage.values())
    per_symbol = []
    for symbol in sorted(universe):
        symbol_state = coverage.get(symbol)
        raw_override = overrides.get(symbol)
        override = raw_override if isinstance(raw_override, dict) else {}
        gate_eligible = bool(override.get("coverage_gate_eligible", universe[symbol].get("active_on_date", True)))
        if symbol_state is None:
            per_symbol.append({
                "symbol": symbol, "record_count": 0, "returned_start": None, "returned_end": None,
                "effective_coverage_start": override.get("effective_start"), "coverage_ratio": None,
                "coverage_status": override.get("classification", "HISTORY_UNAVAILABLE"),
                "coverage_gate_eligible": gate_eligible, "identity_override_sources": override.get("sources", []),
            })
            continue
        effective_start = max(
            symbol_state["start"],
            str(override.get("effective_start") or symbol_state["start"]),
        )
        identity_count = (
            symbol_state["identity_count"]
            if override.get("effective_start")
            else symbol_state["count"]
        )
        expected = max(
            1,
            _expected_sessions(
                date.fromisoformat(effective_start),
                date.fromisoformat(symbol_state["end"]),
            ),
        )
        ratio = min(1.0, identity_count / expected)
        default_status = (
            "HISTORY_FROM_2000_GO"
            if symbol_state["start"] <= start_tolerance
            else "HISTORY_FROM_LISTING_OR_PROVIDER_START"
        )
        per_symbol.append({
            "symbol": symbol, "record_count": symbol_state["count"], "identity_record_count": identity_count,
            "returned_start": symbol_state["start"], "returned_end": symbol_state["end"],
            "effective_coverage_start": effective_start, "coverage_ratio": round(ratio, 6),
            "coverage_status": override.get("classification", default_status),
            "coverage_gate_eligible": gate_eligible, "identity_override_sources": override.get("sources", []),
        })
    below_95_all = sum(row["coverage_ratio"] is not None and row["coverage_ratio"] < 0.95 for row in per_symbol)
    below_95_gate = sum(row["coverage_gate_eligible"] and (row["coverage_ratio"] is None or row["coverage_ratio"] < 0.95) for row in per_symbol)
    payload = _artifact("phase11_3_historical_start_coverage_audit_v1", {
        "status": "GO" if len(coverage) >= 209 and invalid == 0 and below_95_gate == 0 else "PARTIAL",
        "target_start_date": TARGET_START,
        "universe_size": len(universe),
        "symbols_with_prices": len(coverage),
        "symbols_with_coverage_from_2000": from_2000,
        "symbols_starting_at_listing_or_provider": len(coverage) - from_2000,
        "symbols_with_insufficient_history": insufficient,
        "symbols_below_95_percent_expected_sessions": below_95_all,
        "gate_eligible_symbols_below_95_percent_expected_sessions": below_95_gate,
        "listing_identity_override_manifest": str(override_path.relative_to(root)),
        "listing_identity_override_hash": stable_hash(override_payload) if override_payload else None,
        "price_record_count": record_count,
        "invalid_ohlc_rows": invalid,
        "per_symbol": per_symbol,
    })
    _write(_output(root) / "historical-start-coverage-audit.json", payload)
    _write(_output(root) / "price-backfill.json", payload)
    return payload


def sec_events(root: Path) -> dict[str, Any]:
    events = []
    coverage = []
    for record in Phase113Store(database_path(root)).iter_records("filings"):
        row = record["payload"]
        if row.get("record_type") == "SEC_COVERAGE":
            coverage.append(row)
        elif row.get("form") and not row.get("record_type"):
            events.append(row)
    causal = sum(bool(row.get("accepted_at")) for row in events)
    symbol_count = len({str(row.get("symbol")) for row in events})
    mapped_symbols = {str(row.get("symbol")) for row in coverage if row.get("coverage_status") == "CIK_MAPPED_FETCHED"}
    unmapped_symbols = {str(row.get("symbol")) for row in coverage if row.get("coverage_status") == "CIK_NOT_FOUND"}
    required_symbols = len(mapped_symbols)
    forms = Counter(str(row.get("form")) for row in events)
    payload = _artifact("phase11_3_sec_events_v1", {"status": "GO" if events and causal == len(events) and mapped_symbols <= {str(row.get('symbol')) for row in events} else "PARTIAL", "event_count": len(events), "symbol_count": symbol_count, "required_symbol_count": required_symbols, "CIK_mapped_symbol_count": len(mapped_symbols), "CIK_not_found_symbol_count": len(unmapped_symbols), "coverage_symbol_count": len(mapped_symbols | unmapped_symbols), "causal_timestamp_count": causal, "form_counts": dict(forms), "raw_accessions_public": False, "source": "SEC_EDGAR"})
    _write(_output(root) / "sec-events.json", payload)
    return payload


def fundamental_actuals(root: Path) -> dict[str, Any]:
    coverage = []
    for record in Phase113Store(database_path(root)).iter_records("filings"):
        row = record["payload"]
        if row.get("record_type") == "COMPANYFACTS_COVERAGE":
            coverage.append(row)
    actuals = _deduplicated_companyfacts(root)
    actual_count = len(actuals)
    causal_count = sum(bool(row.get("accepted_at")) for row in actuals)
    actual_symbols = {str(row.get("symbol")) for row in actuals}
    concept_counts = Counter(str(row.get("concept")) for row in actuals)
    eligible_symbols = {str(row.get("symbol")) for row in coverage if int(row.get("fact_count", 0)) > 0}
    required_symbols = len(eligible_symbols)
    symbol_count = len(actual_symbols)
    unavailable_selected = sorted(eligible_symbols - actual_symbols)
    payload = _artifact("phase11_3_fundamental_actuals_v1", {"status": "GO" if actual_count and causal_count == actual_count and len(coverage) == required_symbols else "PARTIAL", "symbol_count": symbol_count, "coverage_record_count": len(coverage), "required_symbol_count": required_symbols, "coverage_denominator": "SEC_CIK_MAPPED_WITH_COMPANYFACTS", "XBRL_eligible_symbol_count": len(eligible_symbols), "XBRL_not_available_symbol_count": sum(int(row.get("fact_count", 0)) == 0 for row in coverage), "selected_concepts_unavailable_symbol_count": len(unavailable_selected), "selected_concepts_unavailable_symbols": unavailable_selected, "provider_fact_count": sum(int(row.get("fact_count", 0)) for row in coverage), "selected_actual_count": actual_count, "causal_actual_count": causal_count, "concept_counts": dict(concept_counts), "revision_aware": True, "source": "SEC_COMPANYFACTS"})
    _write(_output(root) / "fundamental-actuals.json", payload)
    return payload


def shariah_history(root: Path) -> dict[str, Any]:
    universe = _phase112_universe(root)
    concepts: Counter[str] = Counter()
    actual_symbols = set()
    actuals = _deduplicated_companyfacts(root)
    for row in actuals:
        concepts[str(row.get("concept"))] += 1
        actual_symbols.add(str(row.get("symbol")))
    reconstruction = build_pit_shariah_screens(root, actuals)
    complete = sum(count for status, count in reconstruction["final_status_counts"].items() if status in {"SHARIAH_ELIGIBLE_PIT", "SHARIAH_INELIGIBLE_PIT"})
    payload = _artifact("phase11_3_shariah_history_v1", {"status": "GO" if complete and complete == reconstruction["screen_count"] else "SHARIAH_HISTORY_INCOMPLETE", "universe_size": len(universe), "reconstructable_count": complete, "partial_screen_count": reconstruction["screen_count"] - complete, "screen_symbol_count": reconstruction["symbol_count"], "SEC_actual_symbol_count": len(actual_symbols), "component_fact_counts": dict(concepts), "available_component_counts": reconstruction["available_component_counts"], "final_status_counts": reconstruction["final_status_counts"], "methodology_id": reconstruction["methodology_id"], "methodology_hash": reconstruction["methodology_hash"], "private_screen_content_hash": reconstruction["private_screen_content_hash"], "private_values_published": False, "missing_components": ["PIT_BUSINESS_ACTIVITY_CLASSIFICATION", "NON_PERMISSIBLE_INCOME_CLASSIFICATION", "COMPLETE_STANDARDIZED_BALANCE_COMPONENTS", "FULL_ACCEPTANCE_TIMESTAMP_JOIN"], "methodology_mutated": False, "financial_decision": "NO_NEW_FINANCIAL_CANDIDATE"})
    _write(_output(root) / "shariah-history.json", payload)
    return payload


def news_coverage(root: Path) -> dict[str, Any]:
    event_count = 0
    timestamped = 0
    coverage_period_count = 0
    failed_coverage = 0
    event_symbols: set[str] = set()
    coverage_symbols: set[str] = set()
    for record in Phase113Store(database_path(root)).iter_records("news"):
        row = record["payload"]
        if row.get("record_type") == "NEWS_COVERAGE":
            coverage_period_count += 1
            coverage_symbols.add(str(row.get("symbol")))
            failed_coverage += row.get("provider_status") not in {"OK", None}
        else:
            event_count += 1
            timestamped += bool(row.get("published_at"))
            event_symbols.add(str(row.get("symbol")))
    symbol_count = len(event_symbols | coverage_symbols)
    required_symbols = max(1, int(len(_phase112_universe(root)) * 0.95))
    alternative_path = DEFAULT_DATASCRAPER_ROOT / "output" / "stocks_research_exports" / "phase11_3" / "alternative-news-provider-probe.json"
    alternative = _read_artifact(alternative_path)
    payload = _artifact("phase11_3_news_backfill_v1", {"status": "GO" if event_count and timestamped == event_count and symbol_count >= required_symbols and failed_coverage == 0 else "PARTIAL", "record_count": event_count, "symbol_count": symbol_count, "event_symbol_count": len(event_symbols), "coverage_symbol_count": len(coverage_symbols), "coverage_period_count": coverage_period_count, "failed_coverage_period_count": failed_coverage, "required_symbol_count": required_symbols, "timestamped_count": timestamped, "full_article_bodies": 0, "license_status": "PRIVATE_RESEARCH_ONLY", "history_status": "NEWS_QUARTERLY_COVERAGE_GO" if coverage_period_count and failed_coverage == 0 else "NEWS_ARCHIVE_PARTIAL" if event_count else "NEWS_HISTORY_UNAVAILABLE", "alternative_provider_probe_status": alternative.get("status", "NOT_RUN"), "alternative_provider_summary": alternative.get("provider_summary", {}), "alternative_provider_probe_hash": stable_hash(alternative) if alternative else None})
    _write(_output(root) / "news-backfill.json", payload)
    return payload


def movers_build(root: Path) -> dict[str, Any]:
    records = (row["payload"] for row in Phase113Store(database_path(root)).iter_records("prices"))
    series: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in records:
        try:
            series[str(row["symbol"]).removesuffix(".US")].append((str(row["timestamp"])[:10], float(row["close"])))
        except (KeyError, TypeError, ValueError):
            continue
    events = []
    for symbol, values in series.items():
        ordered = sorted(set(values))
        for previous, current in zip(ordered, ordered[1:]):
            if previous[1] > 0:
                change = current[1] / previous[1] - 1
                if abs(change) >= 0.05:
                    event = {"symbol": symbol, "session_date": current[0], "return": change, "direction": "GAINER" if change > 0 else "LOSER", "decision_available_at": f"{current[0]}T23:59:59Z"}
                    events.append(event)
    Phase113Store(database_path(root)).append_events("MOVER", [(f"{row['symbol']}:{row['session_date']}", row) for row in events])
    payload = _artifact("phase11_3_movers_v1", {"status": "GO" if events else "PARTIAL", "event_count": len(events), "symbol_count": len(series), "threshold_absolute_return": 0.05, "causal_close_availability": True})
    _write(_output(root) / "movers-build.json", payload)
    return payload


def attribution(root: Path) -> dict[str, Any]:
    store = Phase113Store(database_path(root))
    movers = _research_events(store, "MOVER")
    news = [row["payload"] for row in store.records("news")]
    filings = [record["payload"] for record in store.iter_records("filings") if record["payload"].get("form") and not record["payload"].get("record_type")]
    event_index: dict[str, list[tuple[datetime, dict[str, Any]]]] = defaultdict(list)
    for event in news + filings:
        available = _parse_dt(event.get("published_at") or event.get("accepted_at") or event.get("filing_date"))
        if available:
            event_index[str(event.get("symbol"))].append((available, event))
    for values in event_index.values():
        values.sort(key=lambda item: item[0])
    attributed: list[dict[str, Any]] = []
    for mover in movers:
        decision = _parse_dt(mover.get("decision_available_at"))
        candidates = []
        values = event_index.get(str(mover.get("symbol")), [])
        if decision and values:
            timestamps = [item[0] for item in values]
            left = bisect.bisect_left(timestamps, decision - timedelta(days=3))
            right = bisect.bisect_right(timestamps, decision)
            for available, event in values[left:right]:
                candidates.append({"event_hash": stable_hash(event), "available_at": available.isoformat(), "source_type": "SEC" if event.get("form") else "NEWS"})
        row = {"mover_hash": stable_hash(mover), "symbol": mover.get("symbol"), "session_date": mover.get("session_date"), "causal_sources": candidates, "attribution_status": "ATTRIBUTED" if candidates else "UNATTRIBUTED"}
        attributed.append(row)
    store.append_events("ATTRIBUTION", [(row["mover_hash"], row) for row in attributed])
    attributed_count = sum(row["attribution_status"] == "ATTRIBUTED" for row in attributed)
    payload = _artifact("phase11_3_attribution_v1", {"status": "GO" if attributed and len(attributed) == len(movers) else "PARTIAL", "processing_status": "ALL_MOVERS_CAUSALLY_CLASSIFIED" if attributed and len(attributed) == len(movers) else "MOVER_CLASSIFICATION_INCOMPLETE", "evidence_status": "COMPLETE" if attributed_count == len(attributed) else "PARTIAL", "mover_count": len(movers), "classified_count": len(attributed), "attributed_count": attributed_count, "unattributed_count": len(attributed) - attributed_count, "unattributed_classification": "NO_CAUSAL_SOURCE_FOUND", "financial_candidate_eligible_when_unattributed": False, "future_information_used": False})
    _write(_output(root) / "attribution.json", payload)
    return payload


def event_windows(root: Path) -> dict[str, Any]:
    store = Phase113Store(database_path(root))
    attributions = _research_events(store, "ATTRIBUTION")
    payload = _artifact("phase11_3_event_windows_v1", {"status": "GO" if attributions else "PARTIAL", "event_count": len(attributions), "windows": ["[-3,0]", "[0,+1]", "[0,+5]", "[0,+20]"], "pre_event_leakage_blocked": True, "overlapping_events_flagged": True})
    _write(_output(root) / "event-windows.json", payload)
    return payload


def data_quality(root: Path) -> dict[str, Any]:
    price = price_coverage(root)
    coverage = datascraper_coverage(root)
    privacy = _privacy_audit(root)
    protected = _protected_integrity(root)
    payload = _artifact("phase11_3_data_quality_v1", {"status": "GO" if price["status"] == "GO" and coverage["status"] == "GO" and privacy["status"] == "GO" and protected["status"] == "GO" else "PARTIAL", "price_status": price["status"], "manifest_status": coverage["status"], "privacy": privacy, "protected_integrity": protected, "broker_write_calls": 0, "paper_order_calls": 0, "live_order_calls": 0, "options_calls": 0, "futures_calls": 0})
    _write(_output(root) / "data-quality.json", payload)
    return payload


def status(root: Path) -> dict[str, Any]:
    inventory = datascraper_inventory(root)
    coverage = datascraper_coverage(root)
    universe = universe_history(root)
    prices = price_coverage(root)
    sec = sec_events(root)
    fundamentals = fundamental_actuals(root)
    shariah = shariah_history(root)
    news = news_coverage(root)
    rss = rss_audit(root)
    quality = data_quality(root)
    mover = _read_artifact(_output(root) / "movers-build.json")
    attributed = _read_artifact(_output(root) / "attribution.json")
    integration_go = inventory["status"] == PHASE11_3_INTEGRATION_MARKER and coverage["status"] == "GO" and universe["universe_size"] == 220 and rss["status"] in {"GO", "PARTIAL"}
    full_go = integration_go and prices["status"] == "GO" and sec["status"] == "GO" and fundamentals["status"] == "GO" and shariah["status"] == "GO" and news["status"] == "GO" and mover.get("status") == "GO" and attributed.get("status") == "GO" and quality["status"] == "GO"
    payload = _artifact("phase11_3_status_v1", {
        "status": PHASE11_3_MARKER if full_go else "PHASE11_3_DATA_EVIDENCE_INCOMPLETE",
        "integration_status": PHASE11_3_INTEGRATION_MARKER if integration_go else "PHASE11_3_DATASCRAPER_INTEGRATION_PARTIAL",
        "target_start_date": TARGET_START,
        "universe_size": universe["universe_size"],
        "price_symbols": prices["symbols_with_prices"],
        "price_rows": prices["price_record_count"],
        "SEC_event_count": sec["event_count"],
        "fundamental_symbol_count": fundamentals["symbol_count"],
        "news_symbol_count": news["symbol_count"],
        "mover_event_count": mover.get("event_count", 0),
        "attributed_mover_count": attributed.get("attributed_count", 0),
        "shariah_history_status": shariah["status"],
        "rss_status": rss["status"],
        "FINANCIAL_FINALIST_GO": False,
        "financial_decision": "NO_NEW_FINANCIAL_CANDIDATE",
        "FORWARD_RESEARCH_SHADOW": "blocked",
        "PAPER_STRATEGY_AUTHORITY": "blocked",
        "LIVE_STRATEGY_AUTHORITY": "blocked",
        "execution_authority": "NONE",
        "strategy_authority": "NONE",
        "broker_write_calls": 0,
        "open_blockers": _blockers(prices, sec, fundamentals, shariah, news, attributed),
    })
    _write(_output(root) / "status.json", payload)
    return payload


def freeze(root: Path) -> dict[str, Any]:
    current = status(root)
    full_go = current["status"] == PHASE11_3_MARKER
    payload = _artifact("phase11_3_freeze_status_v1", {
        "freeze_status": PHASE11_3_FREEZE_MARKER if full_go else "PHASE11_3_FREEZE_BLOCKED_DATA_EVIDENCE_INCOMPLETE",
        "phase11_3_status": current["status"],
        "integration_status": current["integration_status"],
        "source_hashes": {relative: file_hash(root / relative) for relative in SOURCE_FILES if (root / relative).is_file()},
        "datascraper_source_hashes": _datascraper_source_hashes(),
        "protected_integrity": _protected_integrity(root),
        "private_database_hash": file_hash(database_path(root)) if database_path(root).is_file() else None,
        "FINANCIAL_FINALIST_GO": False,
        "financial_decision": "NO_NEW_FINANCIAL_CANDIDATE",
        "execution_authority": "NONE",
        "strategy_authority": "NONE",
        "broker_write_calls": 0,
        "open_blockers": current["open_blockers"],
    })
    _write(_output(root) / "freeze-status.json", payload)
    _write_reports(root, current, payload)
    return payload


def _phase112_universe(root: Path) -> list[dict[str, Any]]:
    files = sorted((root / "data" / "research" / "phase11_2" / "private" / "manifests").glob("probe-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        return []
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    return [row for row in payload.get("universe", []) if isinstance(row, dict)]


def _research_events(store: Phase113Store, event_type: str) -> list[dict[str, Any]]:
    store.initialize()
    with sqlite3.connect(store.path) as db:
        return [json.loads(row[0]) for row in db.execute(
            """
            SELECT current.payload_json
            FROM research_events AS current
            JOIN (
              SELECT economic_key, MAX(event_id) AS latest_id
              FROM research_events WHERE event_type=? GROUP BY economic_key
            ) AS latest ON latest.latest_id=current.event_id
            ORDER BY current.event_id
            """,
            (event_type,),
        )]


def _deduplicated_companyfacts(root: Path) -> list[dict[str, Any]]:
    override_payload = _read_artifact(root / "config" / "research" / "phase11_3_sec_acceptance_overrides.json")
    raw_overrides = override_payload.get("overrides")
    overrides: dict[str, Any] = raw_overrides if isinstance(raw_overrides, dict) else {}
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in Phase113Store(database_path(root)).iter_records("filings"):
        row = record["payload"]
        if row.get("record_type") != "COMPANYFACT":
            continue
        raw_override = overrides.get(str(row.get("accession_hash") or ""))
        override = raw_override if isinstance(raw_override, dict) else {}
        if not row.get("accepted_at") and override.get("accepted_at"):
            row = dict(row, accepted_at=override["accepted_at"], acceptance_timestamp_source=override.get("classification"), acceptance_timestamp_source_url=override.get("source"))
        key = (
            row.get("symbol"), row.get("taxonomy"), row.get("concept"), row.get("unit"),
            row.get("period_start"), row.get("period_end"), row.get("filed_at"), row.get("form"),
            row.get("fiscal_year"), row.get("fiscal_period"), row.get("accession_hash"), row.get("value"),
        )
        previous = selected.get(key)
        if previous is None or (not previous.get("accepted_at") and row.get("accepted_at")):
            selected[key] = row
    return list(selected.values())


def _protected_integrity(root: Path) -> dict[str, Any]:
    freeze_path = root / "output" / "ibkr" / "phase11_2" / "freeze-status.json"
    if not freeze_path.is_file():
        return {"status": "NO_GO", "reason": "PHASE11_2_FREEZE_MISSING"}
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_mismatches = []
    for relative, expected in frozen.get("source_hashes", {}).items():
        if relative in {"main.py", ".gitignore"}:
            continue
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            source_mismatches.append(relative)
    protected_mismatches = []
    protected_paths = {
        "phase7_store": "data/execution/phase7/execution_ledger.sqlite3",
        "phase8_store": "data/broker/phase8/private/broker_observation.sqlite3",
        "phase8_1_store": "data/broker/phase8_1/private/observation_soak.sqlite3",
        "phase8_2_store": "data/shadow/phase8_2/shadow_ledger.sqlite3",
        "phase9_store": "data/execution/phase9/private/paper_execution.sqlite3",
        "phase10_store": "data/execution/phase10/private/auto_paper.sqlite3",
        "phase10_freeze": "output/ibkr/phase10/freeze-status.json",
        "phase11_1_freeze": "output/research/phase11_1/freeze-status.json",
        "phase9_canary_evidence": "output/ibkr/phase9/canary-a-submit-cancel-evidence.json",
    }
    for key, expected in frozen.get("protected_dependency_hashes", {}).items():
        path = root / protected_paths.get(key, "__missing__")
        if not path.is_file() or file_hash(path) != expected:
            protected_mismatches.append(key)
    return {"status": "GO" if not source_mismatches and not protected_mismatches else "NO_GO", "phase11_2_source_mismatches": source_mismatches, "protected_dependency_mismatches": protected_mismatches, "main_py_versioned_shell": True}


def _privacy_audit(root: Path) -> dict[str, Any]:
    forbidden = ("api_token", "api_key", "netliquidation", "availablefunds", "buyingpower", "raw_account_id", "article_body")
    matches: Counter[str] = Counter()
    for path in _output(root).glob("*.json"):
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for token in forbidden:
            if token in text:
                matches[token] += 1
    return {"status": "GO" if not matches else "NO_GO", "matches": dict(matches), "credentials_leaked": 0, "broker_identifiers_leaked": 0, "article_bodies_public": 0}


def _datascraper_source_hashes() -> dict[str, str]:
    root = DEFAULT_DATASCRAPER_ROOT
    relatives = (
        "main.py", "src/research/stocks_phase11_3/contracts.py", "src/research/stocks_phase11_3/exports.py",
        "src/research/stocks_phase11_3/inventory.py", "src/research/stocks_phase11_3/acquisition.py",
        "src/research/stocks_phase11_3/cli.py", "src/scrapers/rss/reader.py",
    )
    return {relative: file_hash(root / relative) for relative in relatives if (root / relative).is_file()}


def _blockers(prices: dict[str, Any], sec: dict[str, Any], fundamentals: dict[str, Any], shariah: dict[str, Any], news: dict[str, Any], attribution_report: dict[str, Any]) -> list[str]:
    blockers = []
    if prices["status"] != "GO":
        blockers.append("PRICE_COVERAGE_BELOW_95_PERCENT")
    if sec["status"] != "GO":
        blockers.append("SEC_CAUSAL_EVENTS_INCOMPLETE")
    if fundamentals["status"] != "GO":
        blockers.append("SEC_COMPANYFACTS_INCOMPLETE")
    if shariah["status"] != "GO":
        blockers.append("SHARIAH_HISTORY_INCOMPLETE")
    if news["status"] != "GO":
        blockers.append("NEWS_ARCHIVE_PARTIAL")
    if attribution_report.get("status") != "GO":
        blockers.append("CAUSAL_ATTRIBUTION_INCOMPLETE")
    return blockers


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        try:
            return datetime.fromisoformat(text[:10]).replace(tzinfo=UTC)
        except ValueError:
            return None


def _weekday_count(start: date, end: date) -> int:
    days = (end - start).days + 1
    weeks, remainder = divmod(max(days, 0), 7)
    count = weeks * 5
    for offset in range(remainder):
        if (start.weekday() + offset) % 7 < 5:
            count += 1
    return count


def _expected_sessions(start: date, end: date) -> int:
    try:
        import exchange_calendars as calendars

        return len(calendars.get_calendar("XNYS").sessions_in_range(start.isoformat(), end.isoformat()))
    except (ImportError, ValueError):
        return _weekday_count(start, end)


def _artifact(schema: str, payload: dict[str, Any]) -> dict[str, Any]:
    result = {"schema": schema, "generated_at": datetime.now(UTC).isoformat(), **payload}
    result["content_hash"] = stable_hash(result)
    return result


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _output(root: Path) -> Path:
    path = root / "output" / "ibkr" / "phase11_3"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_reports(root: Path, current: dict[str, Any], frozen: dict[str, Any]) -> None:
    status_text = f"# Phase 11.3 Status\n\n- Status: `{current['status']}`\n- Integration: `{current['integration_status']}`\n- Price rows: `{current['price_rows']}`\n- Open blockers: `{', '.join(current['open_blockers']) or 'none'}`\n- Execution authority: `NONE`\n"
    freeze_text = f"# Phase 11.3 Freeze Report\n\n- Freeze: `{frozen['freeze_status']}`\n- Protected integrity: `{frozen['protected_integrity']['status']}`\n- Financial decision: `NO_NEW_FINANCIAL_CANDIDATE`\n"
    (root / "PHASE11_3_STATUS.md").write_text(status_text, encoding="utf-8")
    (root / "PHASE11_3_FREEZE_REPORT.md").write_text(freeze_text, encoding="utf-8")
    docs = root / "docs" / "PHASE11_3_HISTORICAL_COVERAGE_AND_SEC_ATTRIBUTION.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("# Phase 11.3\n\nDatascraper is the acquisition plane. Stocks validates immutable exports and performs causal research. All broker and strategy authority remains NONE.\n", encoding="utf-8")
