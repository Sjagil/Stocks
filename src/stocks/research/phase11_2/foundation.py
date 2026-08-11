from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_2.contracts import CapabilityStatus, ProviderId, ProviderSpec
from stocks.research.phase11_2.providers import (
    SEC_BASE,
    SEC_TICKERS_URL,
    SafeJsonClient,
    eodhd_probe,
    load_provider_secrets,
    oxr_probe,
    sec_probe,
    utc_now,
)
from stocks.research.phase11_2.storage import PitFoundationStore, ensure_private_layout


EODHD_DOC = "https://eodhd.com/financial-apis/quick-start-with-our-financial-data-apis"
SEC_DOC = "https://www.sec.gov/search-filings/edgar-application-programming-interfaces"
OXR_DOC = "https://openexchangerates.org/api/historical/:date.json"

DATASET_IDS = (
    "active_symbol_universe",
    "delisted_symbol_universe",
    "daily_ohlcv",
    "adjusted_prices",
    "splits",
    "dividends",
    "fundamentals",
    "income_statements",
    "balance_sheets",
    "cash_flow_statements",
    "shares_outstanding",
    "earnings_actual_eps",
    "earnings_estimate_eps",
    "revenue_actual",
    "revenue_estimate",
    "earnings_announcement_date",
    "earnings_session_classification",
    "earnings_trends",
    "analyst_revisions",
    "news",
    "news_publication_timestamps",
    "news_updates",
    "filing_dates",
    "filing_acceptance_timestamps",
    "xbrl_company_facts",
    "sector",
    "industry",
    "exchange_calendar",
    "fx_rates",
)


def provider_registry(verified_at: str | None = None) -> list[ProviderSpec]:
    verified = verified_at or utc_now()
    return [
        ProviderSpec(
            ProviderId.EODHD,
            "MULTI_DATASET",
            "eodhd.com/api",
            ("STK", "ETF"),
            ("US", "EUROPE", "GLOBAL"),
            True,
            True,
            "provider-dependent",
            ("date", "datetime", "reportDate", "filing_date"),
            "UTC/provider exchange date",
            "NOT_DOCUMENTED_AS_COMPLETE",
            False,
            True,
            False,
            "MITIGATED_ONLY_WITH_DELISTED_ENDPOINT",
            "plan quota; bounded client",
            "LICENSE_REVIEW_REQUIRED",
            "LICENSE_REVIEW_REQUIRED",
            "Paid-plan terms must be reviewed before durable raw redistribution or retention.",
            CapabilityStatus.PARTIAL,
            verified,
            stable_hash(EODHD_DOC),
        ),
        ProviderSpec(
            ProviderId.SEC_EDGAR,
            "SUBMISSIONS_AND_COMPANYFACTS",
            "data.sec.gov/submissions + api/xbrl/companyfacts",
            ("US_REPORTING_ISSUERS",),
            ("SEC_FILERS",),
            True,
            True,
            "full electronic filing history varies",
            ("filingDate", "acceptanceDateTime", "filed", "period"),
            "UTC/US Eastern filing metadata",
            "ACCESSION_VERSIONED",
            True,
            True,
            True,
            "LOW_FOR_REGISTERED_FILERS",
            "bounded polite requests; SEC fair-access rules",
            "PUBLIC_DATA",
            "PUBLIC_DATA",
            "Public SEC data; retain accession identifiers only as hashes in public artifacts.",
            CapabilityStatus.GO,
            verified,
            stable_hash(SEC_DOC),
        ),
        ProviderSpec(
            ProviderId.OPENEXCHANGERATES,
            "HISTORICAL_FX",
            "openexchangerates.org/api/historical",
            ("FX",),
            ("GLOBAL_CURRENCIES",),
            True,
            False,
            "1999-01-01",
            ("timestamp",),
            "UTC",
            "NOT_EXPOSED",
            False,
            True,
            True,
            "NOT_APPLICABLE",
            "plan dependent",
            "LICENSE_REVIEW_REQUIRED",
            "LICENSE_REVIEW_REQUIRED",
            "Historical endpoint timestamp is publication time; base/symbol controls are plan dependent.",
            CapabilityStatus.PARTIAL,
            verified,
            stable_hash(OXR_DOC),
        ),
    ]


def provider_lookup(provider_id: str) -> dict[str, Any]:
    spec = next((item for item in provider_registry() if item.provider_id == provider_id), None)
    if spec is None:
        return {"status": "PROVIDER_CAPABILITY_BLOCKED", "blocker": "UNKNOWN_PROVIDER"}
    return {"status": spec.capability_status, "provider": spec.public_dict()}


def initial_capability_matrix() -> list[dict[str, Any]]:
    eod_go = {
        "active_symbol_universe", "delisted_symbol_universe", "daily_ohlcv", "adjusted_prices", "splits", "dividends",
        "fundamentals", "income_statements", "balance_sheets", "cash_flow_statements", "shares_outstanding",
        "earnings_actual_eps", "earnings_estimate_eps", "revenue_actual", "revenue_estimate", "earnings_announcement_date",
        "earnings_trends", "news", "news_publication_timestamps", "sector", "industry",
    }
    sec_go = {"filing_dates", "filing_acceptance_timestamps", "xbrl_company_facts"}
    rows = []
    for dataset in DATASET_IDS:
        if dataset in sec_go:
            provider = ProviderId.SEC_EDGAR
            pit = True
            revision = True
            timestamp = dataset == "filing_acceptance_timestamps"
            decision = "PIT_USABLE_WITH_ACCEPTED_AT" if timestamp else "PIT_USABLE_AFTER_ACCESSION_JOIN"
            caveat = "Companyfacts facts expose filed dates; join accession to submissions acceptance time."
        elif dataset == "fx_rates":
            provider = ProviderId.OPENEXCHANGERATES
            pit = True
            revision = False
            timestamp = True
            decision = "PIT_ELIGIBLE_WITH_PROVIDER_TIMESTAMP"
            caveat = "Historical values are last published values for the requested UTC day."
        else:
            provider = ProviderId.EODHD
            pit = dataset in {"daily_ohlcv", "adjusted_prices", "splits", "dividends", "news_publication_timestamps"}
            revision = False
            timestamp = dataset == "news_publication_timestamps"
            if dataset == "analyst_revisions":
                decision = "HISTORICAL_CONSENSUS_UNAVAILABLE"
            elif dataset in {"earnings_session_classification", "news_updates"}:
                decision = "TIMESTAMP_GRANULARITY_INSUFFICIENT"
            elif dataset not in eod_go:
                decision = "PROVIDER_CAPABILITY_PARTIAL"
            else:
                decision = "PIT_DATA_PARTIAL" if not pit else "PIT_USABLE"
            caveat = "Endpoint availability does not prove revision-aware point-in-time history."
        rows.append(
            {
                "dataset": dataset,
                "provider": provider,
                "history_available": dataset not in {"news_updates", "analyst_revisions"},
                "PIT_usable": pit,
                "revision_aware": revision,
                "delisted_coverage": dataset in {"delisted_symbol_universe", "daily_ohlcv", "news", "fundamentals"},
                "exact_timestamp_available": timestamp,
                "daily_timestamp_only": not timestamp,
                "provider_first_seen_time_available": False,
                "known_caveats": [caveat],
                "research_decision": decision,
                "probe_status": "NOT_PROBED",
            }
        )
    return rows


def run_bounded_probes(project_root: Path, store: PitFoundationStore) -> dict[str, Any]:
    store.initialize()
    ensure_private_layout(project_root)
    secrets = load_provider_secrets(project_root)
    eod = SafeJsonClient(user_agent="StocksResearch/1.0 read-only-provider-probe", minimum_interval=0.08)
    sec = SafeJsonClient(user_agent="StocksResearch/1.0 research-contact@example.com", minimum_interval=0.12)
    oxr = SafeJsonClient(user_agent="StocksResearch/1.0 read-only-fx-probe", minimum_interval=0.08)
    probes: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}

    eod_requests = [
        ("active_us_symbols", "exchange-symbol-list/US", {}),
        ("delisted_us_symbols", "exchange-symbol-list/US", {"delisted": "1"}),
        ("active_us_stock", "fundamentals/ON.US", {}),
        ("active_european_stock", "fundamentals/ASML.AS", {}),
        ("active_etf", "fundamentals/SPY.US", {}),
        ("historical_prices", "eod/MU.US", {"from": "2024-01-01", "to": "2024-03-31", "order": "a"}),
        ("us_bulk_last_day", "eod-bulk-last-day/US", {"date": "2026-07-21"}),
        ("earnings_calendar", "calendar/earnings", {"symbols": "MU.US", "from": "2025-01-01", "to": "2025-12-31"}),
        ("news", "news", {"s": "MU.US", "from": "2026-07-01", "to": "2026-07-22", "limit": "10"}),
        ("dividends", "div/MU.US", {"from": "2020-01-01"}),
        ("splits", "splits/MU.US", {"from": "2000-01-01"}),
    ]
    for dataset, path, params in eod_requests:
        result, payload = eodhd_probe(eod, secrets["EODHD"], dataset, path, params)
        probes.append(result.public_dict())
        payloads[dataset] = payload

    delisted_symbol = select_delisted_symbol(payloads.get("delisted_us_symbols"))
    if delisted_symbol:
        result, payload = eodhd_probe(eod, secrets["EODHD"], "actual_delisted_stock", f"eod/{delisted_symbol}", {"order": "d"})
        probes.append(result.public_dict())
        payloads["actual_delisted_stock"] = payload

    tickers_result, tickers = sec_probe(sec, "ticker_to_cik", SEC_TICKERS_URL)
    probes.append(tickers_result.public_dict())
    cik = resolve_cik(tickers, "MU")
    submissions = None
    companyfacts = None
    if cik:
        submissions_result, submissions = sec_probe(sec, "submissions", f"{SEC_BASE}/submissions/CIK{cik}.json")
        facts_result, companyfacts = sec_probe(sec, "companyfacts", f"{SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json")
        probes.extend((submissions_result.public_dict(), facts_result.public_dict()))
    else:
        probes.extend((blocked_summary("SEC_EDGAR", "submissions", "CIK_MAPPING_BLOCKED"), blocked_summary("SEC_EDGAR", "companyfacts", "CIK_MAPPING_BLOCKED")))

    fx_result, fx_payload = oxr_probe(oxr, secrets["OPENEXCHANGERATES"], "2025-01-02")
    probes.append(fx_result.public_dict())
    payloads["fx_rates"] = fx_payload

    universe = build_universe(
        payloads.get("active_us_symbols"),
        payloads.get("delisted_us_symbols"),
        target_size=200,
        bulk_prices=payloads.get("us_bulk_last_day"),
    )
    for row in universe:
        store.append_version("symbols", f"{row['symbol']}.{row['exchange']}", row)
        store.append_version("universe_memberships", f"2026-07-22:{row['symbol']}.{row['exchange']}", row)
    for row in _payload_records(payloads.get("historical_prices")):
        date = str(row.get("date") or row.get("datetime") or "UNKNOWN")
        normalized = {
            "symbol": "MU.US",
            "date": date,
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "adjusted_close": row.get("adjusted_close"),
            "volume": row.get("volume"),
            "timestamp_precision": "DATE",
            "decision_availability": "NEXT_SESSION_CONSERVATIVE",
            "source_payload_hash": stable_hash(row),
        }
        store.append_version("prices", f"MU.US:{date}", normalized)
    for dataset, action_type in (("dividends", "DIVIDEND"), ("splits", "SPLIT")):
        for row in _payload_records(payloads.get(dataset)):
            date = str(row.get("date") or row.get("paymentDate") or "UNKNOWN")
            normalized = {
                "symbol": "MU.US",
                "action_type": action_type,
                "date": date,
                "value": row.get("value") or row.get("split") or row.get("dividend"),
                "source_payload_hash": stable_hash(row),
            }
            store.append_version("corporate_actions", f"MU.US:{action_type}:{date}", normalized)
    for row in _payload_records(payloads.get("news")):
        published = str(row.get("date") or row.get("published_at") or "UNKNOWN")
        record_hash = stable_hash(row)
        normalized = {
            "symbol": "MU.US",
            "published_at": published,
            "title": row.get("title"),
            "source": row.get("source"),
            "link_hash": stable_hash(row.get("link")) if row.get("link") else None,
            "sentiment": row.get("sentiment"),
            "source_payload_hash": record_hash,
            "raw_article_body_stored": False,
        }
        store.append_version("news_raw", record_hash, normalized)
        store.append_version("news_versions", record_hash, normalized)
    filings = parse_submissions(submissions)
    for filing in filings:
        store.append_version("filings", filing["accession_hash"], filing)
    facts = parse_companyfacts(companyfacts, filings)
    for fact in facts:
        store.append_version("filing_facts", fact["version_hash"], fact)
    if isinstance(fx_payload, dict) and fx_payload.get("timestamp"):
        fx = {
            "base": fx_payload.get("base"),
            "rates": {key: value for key, value in (fx_payload.get("rates") or {}).items() if key in {"EUR", "USD"}},
            "provider_available_at": datetime.fromtimestamp(int(fx_payload["timestamp"]), UTC).isoformat(),
            "payload_hash": stable_hash(fx_payload),
        }
        store.append_version("fx_rates", "2025-01-02:USD:EUR", fx)

    run_id = stable_hash({"at": utc_now(), "probe_hashes": [row.get("payload_hash") for row in probes]})
    store.append_version("ingestion_runs", run_id, {"run_id": run_id, "probe_count": len(probes), "completed_at": utc_now()})
    private_layout = ensure_private_layout(project_root)
    sec_raw_files = []
    for dataset, raw_payload in (("ticker_to_cik", tickers), ("submissions", submissions), ("companyfacts", companyfacts)):
        if raw_payload is None:
            continue
        payload_hash = stable_hash(raw_payload)
        raw_path = private_layout["raw"] / f"SEC_EDGAR-{dataset}-{payload_hash}.json"
        if not raw_path.exists():
            raw_path.write_text(json.dumps(raw_payload, sort_keys=True, default=str), encoding="utf-8")
        sec_raw_files.append({"dataset": dataset, "payload_hash": payload_hash})
    normalized_payload = {"universe": universe, "filings": filings, "filing_facts": facts, "normalized_at": utc_now()}
    normalized_hash = stable_hash(normalized_payload)
    normalized_path = private_layout["normalized"] / f"normalized-{normalized_hash}.json"
    if not normalized_path.exists():
        normalized_path.write_text(json.dumps(normalized_payload, sort_keys=True, default=str), encoding="utf-8")

    normalized_manifest = {
        "run_id": run_id,
        "probe_results": probes,
        "universe": universe,
        "filing_count": len(filings),
        "fact_count": len(facts),
        "selected_delisted_symbol_hash": stable_hash(delisted_symbol) if delisted_symbol else None,
        "cik_hash": stable_hash(cik) if cik else None,
        "SEC_raw_payloads_stored": sec_raw_files,
        "EODHD_raw_payload_storage_status": "LICENSE_REVIEW_REQUIRED",
        "OPENEXCHANGERATES_raw_payload_storage_status": "LICENSE_REVIEW_REQUIRED",
        "normalized_payload_hash": normalized_hash,
    }
    manifest_path = private_layout["manifests"] / f"probe-{run_id}.json"
    manifest_path.write_text(json.dumps(normalized_manifest, indent=2, sort_keys=True), encoding="utf-8")
    return normalized_manifest


def build_universe(active_payload: Any, delisted_payload: Any, *, target_size: int, bulk_prices: Any = None) -> list[dict[str, Any]]:
    active = normalize_symbol_rows(active_payload, active=True)
    delisted = normalize_symbol_rows(delisted_payload, active=False)
    liquidity = {}
    if isinstance(bulk_prices, list):
        for row in bulk_prices:
            if not isinstance(row, dict):
                continue
            code = str(row.get("code") or row.get("Code") or "")
            close = _float(row.get("adjusted_close") or row.get("close") or row.get("Close"))
            volume = _float(row.get("volume") or row.get("Volume"))
            if code and close is not None and volume is not None:
                liquidity[code] = {"close": close, "volume": volume, "dollar_volume": close * volume}
    for row in active:
        metrics = liquidity.get(row["symbol"], {})
        row.update(
            {
                "reference_close": metrics.get("close"),
                "reference_volume": metrics.get("volume"),
                "reference_dollar_volume": metrics.get("dollar_volume"),
                "liquidity_verified": bool(metrics and metrics.get("close", 0) >= 5 and metrics.get("volume", 0) > 0),
                "penny_stock_blocked": bool(metrics and metrics.get("close", 0) < 5),
            }
        )
    liquid_active = [row for row in active if row["liquidity_verified"] and not row["penny_stock_blocked"]]
    liquid_active.sort(key=lambda row: float(row.get("reference_dollar_volume") or 0), reverse=True)
    targets = {"MU", "DELL", "INTC", "DOCN", "ON", "AMD", "NVDA", "AVGO", "QCOM", "AMAT", "LRCX", "KLAC"}
    active_source = liquid_active if liquidity else []
    selected = [row for row in active_source if row["symbol"] in targets]
    seen = {row["symbol"] for row in selected}
    for row in active_source:
        if len(selected) >= target_size or row["symbol"] in seen:
            continue
        selected.append(row)
        seen.add(row["symbol"])
    for row in delisted[:20]:
        if row["symbol"] not in seen:
            selected.append(row)
            seen.add(row["symbol"])
    return selected[:300]


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _payload_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "results", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def normalize_symbol_rows(payload: Any, *, active: bool) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        security_type = str(item.get("Type") or item.get("type") or "").upper()
        if "COMMON" not in security_type:
            continue
        code = str(item.get("Code") or item.get("code") or "").strip()
        if not code:
            continue
        rows.append(
            {
                "universe_date": "2026-07-22",
                "symbol": code,
                "exchange": str(item.get("Exchange") or item.get("exchange") or "US"),
                "currency": str(item.get("Currency") or item.get("currency") or "USD"),
                "security_type": "COMMON_STOCK",
                "active_on_date": active,
                "listed_on_date": item.get("ListingDate") or item.get("listing_date"),
                "delisted_on_date": item.get("DelistedDate") or item.get("delisted_date"),
                "sector": item.get("Sector") or item.get("sector"),
                "industry": item.get("Industry") or item.get("industry"),
                "inclusion_reason": "LIQUIDITY_REVIEW_REQUIRED" if active else "DELISTED_SURVIVORSHIP_CONTROL",
                "exclusion_reason": None,
                "Shariah_status_at_date": "SHARIAH_HISTORY_UNAVAILABLE",
                "source_record_hash": stable_hash(item),
            }
        )
    return rows


def select_delisted_symbol(payload: Any) -> str | None:
    rows = normalize_symbol_rows(payload, active=False)
    if not rows:
        return None
    return f"{rows[0]['symbol']}.US"


def resolve_cik(payload: Any, ticker: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    for row in payload.values():
        if isinstance(row, dict) and str(row.get("ticker", "")).upper() == ticker.upper():
            return str(row.get("cik_str", "")).zfill(10)
    return None


def parse_submissions(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    recent = ((payload.get("filings") or {}).get("recent") or {})
    forms = recent.get("form") or []
    accepted = recent.get("acceptanceDateTime") or []
    accessions = recent.get("accessionNumber") or []
    filed = recent.get("filingDate") or []
    periods = recent.get("reportDate") or []
    rows = []
    allowed = {"10-K", "10-Q", "8-K", "20-F"}
    for index, form in enumerate(forms):
        if form not in allowed:
            continue
        accession = str(accessions[index]) if index < len(accessions) else ""
        accepted_at = str(accepted[index]) if index < len(accepted) else ""
        if accepted_at and accepted_at.endswith("Z"):
            accepted_at = accepted_at[:-1] + "+00:00"
        rows.append(
            {
                "accession_hash": stable_hash(accession),
                "form": form,
                "filed_at": str(filed[index]) if index < len(filed) else None,
                "accepted_at": accepted_at or None,
                "reporting_period": str(periods[index]) if index < len(periods) else None,
            }
        )
    return rows[:500]


def parse_companyfacts(payload: Any, filings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    accepted_by_accession = {row["accession_hash"]: row.get("accepted_at") for row in filings}
    rows = []
    facts = payload.get("facts") or {}
    for taxonomy, concepts in facts.items():
        if not isinstance(concepts, dict):
            continue
        for tag, concept in concepts.items():
            units = concept.get("units") if isinstance(concept, dict) else None
            if not isinstance(units, dict):
                continue
            for unit, values in units.items():
                if not isinstance(values, list):
                    continue
                for value in values[-20:]:
                    if not isinstance(value, dict) or value.get("form") not in {"10-K", "10-Q", "8-K", "20-F"}:
                        continue
                    accession_hash = stable_hash(str(value.get("accn", "")))
                    fact = {
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "unit": unit,
                        "value": value.get("val"),
                        "period_start": value.get("start"),
                        "period_end": value.get("end"),
                        "filed_at": value.get("filed"),
                        "accepted_at": accepted_by_accession.get(accession_hash),
                        "accession_hash": accession_hash,
                        "form": value.get("form"),
                        "fiscal_year": value.get("fy"),
                        "fiscal_period": value.get("fp"),
                    }
                    fact["version_hash"] = stable_hash(fact)
                    rows.append(fact)
    return rows[:5_000]


def blocked_summary(provider: str, dataset: str, error: str) -> dict[str, Any]:
    return {
        "provider": provider,
        "dataset": dataset,
        "status": error,
        "http_status": None,
        "schema_shape": "NONE",
        "record_count": 0,
        "earliest_timestamp": None,
        "latest_timestamp": None,
        "timestamp_precision": "NONE",
        "null_rate": 0.0,
        "duplicate_rate": 0.0,
        "revision_fields": [],
        "payload_hash": None,
        "attempt_count": 0,
        "error_class": error,
    }


def matrix_with_probes(matrix: list[dict[str, Any]], probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        "active_us_symbols": "active_symbol_universe",
        "delisted_us_symbols": "delisted_symbol_universe",
        "historical_prices": "daily_ohlcv",
        "dividends": "dividends",
        "splits": "splits",
        "active_us_stock": "fundamentals",
        "earnings_calendar": "earnings_announcement_date",
        "news": "news",
        "submissions": "filing_acceptance_timestamps",
        "companyfacts": "xbrl_company_facts",
        "fx_rates": "fx_rates",
    }
    probe_by_dataset = {mapping.get(row["dataset"]): row for row in probes if mapping.get(row["dataset"])}
    result = []
    for row in matrix:
        enriched = dict(row)
        probe = probe_by_dataset.get(row["dataset"])
        if probe:
            enriched["probe_status"] = probe["status"]
            enriched["plan_entitled"] = probe["http_status"] == 200
        result.append(enriched)
    return result


def utc_date_window() -> tuple[str, str]:
    end = datetime.now(UTC).date()
    return (end - timedelta(days=30)).isoformat(), end.isoformat()
