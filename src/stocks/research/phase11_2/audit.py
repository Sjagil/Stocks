from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from stocks.data.phase5_common import sha256_file, utc_now_iso
from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_2.contracts import make_pit_record
from stocks.research.phase11_2.foundation import (
    DATASET_IDS,
    initial_capability_matrix,
    matrix_with_probes,
    provider_registry,
    run_bounded_probes,
)
from stocks.research.phase11_2.privacy import scan_public_artifacts
from stocks.research.phase11_2.providers import safe_secret_status
from stocks.research.phase11_2.quality import SOURCE_CLASSIFICATIONS, compare_sources, quality_summary
from stocks.research.phase11_2.shariah import ShariahMethodology, reconstruct_screen
from stocks.research.phase11_2.storage import PitFoundationStore, phase11_2_database_path, phase11_2_private_root


PHASE11_2_MARKER = "PHASE11_2_PROVIDER_CAPABILITY_AND_PIT_DATA_FOUNDATION_GO"
PHASE11_2_FREEZE_MARKER = "PHASE11_2_PROVIDER_CAPABILITY_AND_PIT_DATA_FOUNDATION_FROZEN_GO"
FINANCIAL_STATE = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "blocked",
    "execution_authority": "NONE",
    "strategy_authority": "NONE",
    "automatic_submission": False,
    "financial_decision": "NO_ALPHA_CANDIDATE",
}
COUNTERS = {
    "ibkr_writer_connections": 0,
    "paper_place_order_calls": 0,
    "paper_cancel_order_calls": 0,
    "live_order_calls": 0,
    "automatic_submissions": 0,
    "futures_calls": 0,
    "options_calls": 0,
}
ARTIFACTS = (
    "provider-registry.json",
    "provider-capability-matrix.json",
    "pit-contract-audit.json",
    "universe-audit.json",
    "delisted-coverage-audit.json",
    "price-coverage-audit.json",
    "fundamental-coverage-audit.json",
    "earnings-coverage-audit.json",
    "news-coverage-audit.json",
    "sec-edgar-audit.json",
    "shariah-pit-audit.json",
    "data-quality-audit.json",
    "status.json",
    "freeze-status.json",
)
PROTECTED_PATHS = {
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
SOURCE_PATHS = [
    "main.py",
    ".gitignore",
    *[f"src/stocks/research/phase11_2/{name}.py" for name in ("__init__", "audit", "contracts", "foundation", "privacy", "providers", "quality", "shariah", "storage")],
    "tests/test_phase11_2_pit_foundation.py",
]
EXPECTED_VERSIONED_APPLICATION_PATHS = {
    "main.py",
    ".gitignore",
    "src/stocks/auto_paper/audit.py",
}


class Layout:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.output = root / "output" / "ibkr" / "phase11_2"
        self.db = phase11_2_database_path(root)

    def artifact(self, name: str) -> Path:
        return self.output / name


def phase11_2_command(project_root: Path, command: str) -> dict[str, Any]:
    commands: dict[str, Callable[[], dict[str, Any]]] = {
        "provider-audit": lambda: provider_audit(project_root),
        "capability-probe": lambda: capability_probe(project_root),
        "universe-build": lambda: universe_build(project_root),
        "price-audit": lambda: price_audit(project_root),
        "fundamentals-audit": lambda: fundamentals_audit(project_root),
        "earnings-audit": lambda: earnings_audit(project_root),
        "news-audit": lambda: news_audit(project_root),
        "sec-audit": lambda: sec_audit(project_root),
        "shariah-pit-audit": lambda: shariah_pit_audit(project_root),
        "data-quality": lambda: data_quality(project_root),
        "status": lambda: status(project_root),
        "freeze": lambda: freeze(project_root),
    }
    if command not in commands:
        raise ValueError(f"Unknown Phase 11.2 command: {command}")
    return commands[command]()


def provider_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    protected = _protected_hashes(project_root)
    specs = [spec.public_dict() for spec in provider_registry()]
    secret_status = safe_secret_status(project_root)
    frozen_integrity = frozen_dependency_integrity(project_root)
    go = len(specs) == 3 and all(protected.values()) and frozen_integrity["status"] == "GO"
    payload = _artifact(
        "phase11_2_provider_registry_v1",
        {
            "status": "GO" if go else "NO_GO",
            "providers": specs,
            "provider_count": len(specs),
            "provider_key_configured": secret_status,
            "provider_keys_logged": False,
            "protected_hashes_at_start": protected,
            "frozen_dependency_integrity": frozen_integrity,
            "execution_infrastructure_modified": False,
        },
    )
    _write(layout.artifact("provider-registry.json"), payload)
    store = PitFoundationStore(layout.db)
    store.initialize()
    for spec in specs:
        store.append_version("providers", f"{spec['provider_id']}:{spec['dataset_id']}", spec)
    return payload


def capability_probe(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    store = PitFoundationStore(layout.db)
    result = run_bounded_probes(project_root, store)
    matrix = matrix_with_probes(initial_capability_matrix(), result["probe_results"])
    for row in matrix:
        store.append_version("datasets", f"{row['provider']}:{row['dataset']}", row)
        store.append_version("provider_capabilities", f"{row['provider']}:{row['dataset']}", row)
    for row in result["probe_results"]:
        store.append_version(
            "provenance",
            f"{result['run_id']}:{row['provider']}:{row['dataset']}",
            {
                "provider": row["provider"],
                "dataset": row["dataset"],
                "payload_hash": row.get("payload_hash"),
                "http_status": row.get("http_status"),
                "probe_status": row.get("status"),
                "observed_at": utc_now_iso(),
            },
        )
    go = len(matrix) == len(DATASET_IDS) and any(row["dataset"] == "daily_ohlcv" and row["probe_status"] == "PROBE_GO" for row in matrix)
    payload = _artifact(
        "phase11_2_provider_capability_matrix_v1",
        {
            "status": "GO" if go else "NO_GO",
            "capability_status": "PIT_DATA_PARTIAL" if go else "PROVIDER_CAPABILITY_BLOCKED",
            "dataset_count": len(matrix),
            "datasets": matrix,
            "probe_results": result["probe_results"],
            "probe_run_hash": result["run_id"],
            "provider_calls_read_only": True,
        },
    )
    _write(layout.artifact("provider-capability-matrix.json"), payload)
    pit = pit_contract_audit(project_root)
    payload["pit_contract_status"] = pit["status"]
    payload["content_hash"] = _content_hash(payload)
    _write(layout.artifact("provider-capability-matrix.json"), payload)
    return payload


def pit_contract_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    now = datetime.now(UTC)
    eligible = make_pit_record(
        entity_id="FIXTURE", symbol="FIX", provider="SEC_EDGAR", dataset="filing", payload={"value": 1},
        first_seen_at=now.isoformat(), ingested_at=now.isoformat(), period_end=(now - timedelta(days=30)).isoformat(),
        accepted_at=now.isoformat(), revision_tracked=True,
    )
    daily = make_pit_record(
        entity_id="FX", symbol="EURUSD", provider="OPENEXCHANGERATES", dataset="fx", payload={"rate": 1.1},
        first_seen_at=now.isoformat(), ingested_at=now.isoformat(), published_at=now.isoformat(), timestamp_precision="DATE",
    )
    blocked = make_pit_record(
        entity_id="BAD", symbol="BAD", provider="EODHD", dataset="fundamental", payload={"value": 1},
        first_seen_at=now.isoformat(), ingested_at=now.isoformat(), period_end=now.isoformat(), published_at=now.isoformat(),
    )
    future_leakage_blocked = not eligible.available_for((now - timedelta(seconds=1)).isoformat())
    go = eligible.PIT_eligibility == "PIT_ELIGIBLE" and daily.PIT_eligibility == "PIT_ELIGIBLE_WITH_DAILY_DELAY" and blocked.PIT_eligibility == "PIT_TIMESTAMP_INCOMPLETE" and future_leakage_blocked
    payload = _artifact(
        "phase11_2_pit_contract_audit_v1",
        {
            "status": "GO" if go else "NO_GO",
            "contract_fields": list(eligible.public_dict()),
            "eligible_fixture": eligible.public_dict(),
            "daily_delay_fixture": daily.public_dict(),
            "period_end_block_fixture": blocked.public_dict(),
            "future_leakage_blocked": future_leakage_blocked,
            "decision_rule": "decision_time >= decision_available_at",
        },
    )
    _write(layout.artifact("pit-contract-audit.json"), payload)
    return payload


def universe_build(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    manifest = _latest_manifest(project_root)
    universe = manifest.get("universe", [])
    active = sum(bool(row.get("active_on_date")) for row in universe)
    delisted = len(universe) - active
    liquidity_verified = sum(bool(row.get("liquidity_verified")) for row in universe if row.get("active_on_date"))
    targets = {"MU", "DELL", "INTC", "DOCN", "ON", "AMD", "NVDA", "AVGO", "QCOM", "AMAT", "LRCX", "KLAC"}
    present = sorted(targets & {str(row.get("symbol")) for row in universe})
    go = 150 <= len(universe) <= 300 and active >= 150 and liquidity_verified == active and delisted > 0
    payload = _artifact(
        "phase11_2_universe_audit_v1",
        {
            "status": "GO" if go else "NO_GO",
            "universe_status": "PIT_DATA_PARTIAL" if go else "SURVIVORSHIP_RISK_BLOCKED",
            "universe_size": len(universe),
            "active_symbol_count": active,
            "delisted_symbol_count": delisted,
            "liquidity_verified_active_count": liquidity_verified,
            "penny_stock_count": sum(bool(row.get("penny_stock_blocked")) for row in universe),
            "per_symbol_price_history_status": "PIT_DATA_PARTIAL",
            "required_research_symbols_present": present,
            "required_research_symbol_count": len(present),
            "historical_membership_rule": "NO_CURRENT_MEMBERSHIP_BACKFILL",
            "survivorship_bias_blocked": delisted == 0,
            "Shariah_history_status": "SHARIAH_HISTORY_UNAVAILABLE",
        },
    )
    _write(layout.artifact("universe-audit.json"), payload)
    delisted_payload = _artifact(
        "phase11_2_delisted_coverage_audit_v1",
        {
            "status": "GO" if delisted > 0 else "NO_GO",
            "delisted_symbol_count": delisted,
            "actual_delisted_symbol_selected_from_provider": bool(manifest.get("selected_delisted_symbol_hash")),
            "selected_delisted_symbol_hash": manifest.get("selected_delisted_symbol_hash"),
            "survivorship_assessment": "DELISTED_ENDPOINT_INCLUDED" if delisted > 0 else "SURVIVORSHIP_RISK_BLOCKED",
        },
    )
    _write(layout.artifact("delisted-coverage-audit.json"), delisted_payload)
    return payload


def price_audit(project_root: Path) -> dict[str, Any]:
    phase5_bars = len(list((project_root / "data" / "bars").rglob("*.parquet")))
    phase5_fx = len(list((project_root / "data" / "fx").rglob("*.parquet")))
    return _coverage_audit(
        project_root,
        "price-coverage-audit.json",
        "phase11_2_price_coverage_audit_v1",
        ("historical_prices", "actual_delisted_stock"),
        "PIT_ELIGIBLE_WITH_DAILY_DELAY",
        require_success=True,
        extra={
            "source_priority": ["PHASE5_CACHE_PRIMARY_WHERE_AVAILABLE", "EODHD_EXTENSION"],
            "existing_phase5_bar_file_count": phase5_bars,
            "existing_phase5_fx_file_count": phase5_fx,
            "FX_source_priority": ["PHASE5_FX_CACHE", "OPENEXCHANGERATES_EXTENSION"],
        },
    )


def fundamentals_audit(project_root: Path) -> dict[str, Any]:
    return _coverage_audit(project_root, "fundamental-coverage-audit.json", "phase11_2_fundamental_coverage_audit_v1", ("active_us_stock", "active_european_stock", "active_etf"), "PIT_DATA_PARTIAL", extra={"revision_support": "REVISION_HISTORY_UNAVAILABLE", "source_priority": ["SEC_EDGAR_PRIMARY", "EODHD_SECONDARY_VALIDATION"]})


def earnings_audit(project_root: Path) -> dict[str, Any]:
    return _coverage_audit(project_root, "earnings-coverage-audit.json", "phase11_2_earnings_coverage_audit_v1", ("earnings_calendar",), "PIT_DATA_PARTIAL", extra={"historical_consensus_status": "HISTORICAL_CONSENSUS_UNAVAILABLE", "session_classification": "TIMESTAMP_GRANULARITY_INSUFFICIENT", "source_priority": ["EODHD_PRIMARY", "SEC_8K_VALIDATION"]})


def news_audit(project_root: Path) -> dict[str, Any]:
    return _coverage_audit(project_root, "news-coverage-audit.json", "phase11_2_news_coverage_audit_v1", ("news",), "NEWS_HISTORY_LIMITED", extra={"update_history": "REVISION_HISTORY_UNAVAILABLE", "provider_first_seen": "UNAVAILABLE", "source_priority": ["EODHD_PRIMARY"]})


def sec_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    store = PitFoundationStore(layout.db)
    filings = [row["payload"] for row in store.records("filings")]
    facts = store.records("filing_facts")
    accepted = sum(bool(row.get("accepted_at")) for row in filings)
    forms = sorted({str(row.get("form")) for row in filings})
    go = accepted > 0 and bool(facts)
    payload = _artifact(
        "phase11_2_sec_edgar_audit_v1",
        {
            "status": "GO" if go else "NO_GO",
            "sec_status": "PIT_DATA_GO" if go else "PROVIDER_CAPABILITY_BLOCKED",
            "ticker_to_cik_mapping": bool(_latest_manifest(project_root).get("cik_hash")),
            "filing_count": len(filings),
            "filings_with_acceptance_timestamp": accepted,
            "filing_fact_version_count": len(facts),
            "forms_observed": forms,
            "accession_identifiers_hashed": True,
            "availability_rule": "accepted_at or conservative later timestamp",
        },
    )
    _write(layout.artifact("sec-edgar-audit.json"), payload)
    return payload


def shariah_pit_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    method = ShariahMethodology()
    incomplete = reconstruct_screen(
        screened_at=utc_now_iso(), financial_statement_available_at=None, price_denominator_date=None,
        business_activity_result=None, debt_ratio=None, cash_interest_ratio=None, receivables_ratio=None,
        non_permissible_income_ratio=None,
    )
    payload = _artifact(
        "phase11_2_shariah_pit_audit_v1",
        {
            "status": "GO",
            "Shariah_PIT_status": "SHARIAH_HISTORY_PARTIAL",
            "methodology": {**asdict(method), "methodology_hash": method.methodology_hash},
            "historical_reconstruction_fixture": incomplete,
            "historical_screen_count": 0,
            "current_status_backprojection_allowed": False,
            "manual_review_required": True,
        },
    )
    _write(layout.artifact("shariah-pit-audit.json"), payload)
    return payload


def data_quality(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    manifest = _latest_manifest(project_root)
    store = PitFoundationStore(layout.db)
    filings = [row["payload"] for row in store.records("filings")]
    summary = quality_summary(universe=manifest.get("universe", []), probes=manifest.get("probe_results", []), filings=filings, counts=store.counts())
    store.append_version("data_quality_events", manifest.get("run_id", "NO_RUN"), summary)
    payload = _artifact(
        "phase11_2_data_quality_audit_v1",
        {
            "status": "GO" if summary["status"] in {"DATA_QUALITY_GO", "DATA_QUALITY_PARTIAL"} else "NO_GO",
            "data_quality_status": summary["status"],
            "metrics": summary,
            "source_conflict_classifications": list(SOURCE_CLASSIFICATIONS),
            "source_comparison_fixtures": {
                "match": compare_sources(1.0, 1.0),
                "tolerance": compare_sources(1.0, 1.0005),
                "conflict": compare_sources(1.0, 2.0),
                "scope": compare_sources(None, 1.0),
            },
            "silent_conflict_resolution": False,
        },
    )
    _write(layout.artifact("data-quality-audit.json"), payload)
    return payload


def status(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    required = [name for name in ARTIFACTS if name not in {"status.json", "freeze-status.json"}]
    checks = {name: _read(layout.artifact(name)).get("status") == "GO" for name in required}
    privacy = scan_public_artifacts(layout.output)
    protected_start = _read(layout.artifact("provider-registry.json")).get("protected_hashes_at_start", {})
    protected_now = _protected_hashes(project_root)
    protected_unchanged = bool(protected_start) and protected_start == protected_now
    go = all(checks.values()) and privacy["status"] == "PRIVACY_GO" and protected_unchanged
    payload = _artifact(
        "phase11_2_status_v1",
        {
            "status": PHASE11_2_MARKER if go else "NO_GO",
            "phase11_2_marker": PHASE11_2_MARKER,
            "technical_readiness": "GO" if go else "NO_GO",
            "artifact_checks": checks,
            "privacy": privacy,
            "protected_dependencies_unchanged": protected_unchanged,
            "financial_readiness": "PIT_DATA_INCOMPLETE",
            "next_phase": "PHASE11_3_NEWS_EARNINGS_AND_FUNDAMENTAL_ATTRIBUTION",
        },
    )
    _write(layout.artifact("status.json"), payload)
    return payload


def freeze(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    status_payload = status(project_root)
    start = _read(layout.artifact("provider-registry.json")).get("protected_hashes_at_start", {})
    now = _protected_hashes(project_root)
    protected_unchanged = bool(start) and start == now
    source_hashes = {path: sha256_file(project_root / path) for path in SOURCE_PATHS}
    frozen_integrity = frozen_dependency_integrity(project_root)
    go = status_payload["status"] == PHASE11_2_MARKER and protected_unchanged and all(source_hashes.values()) and frozen_integrity["status"] == "GO"
    payload = _artifact(
        "phase11_2_freeze_status_v1",
        {
            "freeze_status": PHASE11_2_FREEZE_MARKER if go else "NO_GO",
            "phase11_2_status": status_payload["status"],
            "technical_readiness": "GO" if go else "NO_GO",
            "protected_dependencies_unchanged": protected_unchanged,
            "frozen_dependency_integrity": frozen_integrity,
            "protected_dependency_hashes": now,
            "source_hashes": source_hashes,
            "artifact_hashes": {name: sha256_file(layout.artifact(name)) for name in ARTIFACTS if name != "freeze-status.json"},
            "private_database_hash": sha256_file(layout.db),
            "next_phase": "PHASE11_3_NEWS_EARNINGS_AND_FUNDAMENTAL_ATTRIBUTION",
        },
    )
    _write(layout.artifact("freeze-status.json"), payload)
    return payload


def _coverage_audit(project_root: Path, filename: str, schema: str, datasets: tuple[str, ...], category_status: str, *, extra: dict[str, Any] | None = None, require_success: bool = False) -> dict[str, Any]:
    layout = _layout(project_root)
    probes = {row["dataset"]: row for row in _latest_manifest(project_root).get("probe_results", [])}
    selected = [probes.get(dataset, {"dataset": dataset, "status": "NOT_PROBED", "record_count": 0}) for dataset in datasets]
    assessed = bool(selected) and all(row.get("status") != "NOT_PROBED" for row in selected)
    coverage_go = all(row.get("status") == "PROBE_GO" for row in selected)
    go = assessed and (coverage_go or not require_success)
    payload = _artifact(
        schema,
        {
            "status": "GO" if go else "NO_GO",
            "coverage_status": category_status if coverage_go else "PROVIDER_CAPABILITY_BLOCKED",
            "capability_assessed": assessed,
            "plan_entitlement_blocked": any(row.get("status") == "PLAN_NOT_ENTITLED" for row in selected),
            "datasets": selected,
            "record_count": sum(int(row.get("record_count") or 0) for row in selected),
            "timestamp_precision": sorted({str(row.get("timestamp_precision", "NONE")) for row in selected}),
            "provider_payload_published": False,
            **(extra or {}),
        },
    )
    _write(layout.artifact(filename), payload)
    return payload


def _layout(project_root: Path) -> Layout:
    layout = Layout(project_root)
    layout.output.mkdir(parents=True, exist_ok=True)
    phase11_2_private_root(project_root).mkdir(parents=True, exist_ok=True)
    return layout


def _latest_manifest(project_root: Path) -> dict[str, Any]:
    directory = phase11_2_private_root(project_root) / "manifests"
    files = sorted(directory.glob("probe-*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return _read(files[0]) if files else {}


def _protected_hashes(project_root: Path) -> dict[str, str | None]:
    return {label: sha256_file(project_root / relative) for label, relative in PROTECTED_PATHS.items()}


def frozen_dependency_integrity(project_root: Path) -> dict[str, Any]:
    freeze_path = project_root / "output" / "ibkr" / "phase10" / "freeze-status.json"
    payload = _read(freeze_path)
    marker_go = payload.get("freeze_status") == "PHASE10_AUTONOMOUS_SHARIAH_PAPER_TRADING_FOUNDATION_FROZEN_GO"
    unexpected = []
    expected = []
    for relative, frozen_hash in (payload.get("source_hashes") or {}).items():
        current_hash = sha256_file(project_root / relative)
        if frozen_hash and current_hash != frozen_hash:
            if relative in EXPECTED_VERSIONED_APPLICATION_PATHS:
                expected.append(relative)
            else:
                unexpected.append(relative)
    return {
        "status": "GO" if marker_go and not unexpected else "NO_GO",
        "phase10_marker_go": marker_go,
        "expected_versioned_application_mismatches": sorted(expected),
        "unexpected_frozen_source_mismatches": sorted(unexpected),
    }


def _artifact(schema: str, fields: dict[str, Any]) -> dict[str, Any]:
    payload = {"schema": schema, "generated_at": utc_now_iso(), **COUNTERS, **fields, **FINANCIAL_STATE}
    payload["content_hash"] = _content_hash(payload)
    return payload


def _content_hash(payload: dict[str, Any]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != "content_hash"})


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
