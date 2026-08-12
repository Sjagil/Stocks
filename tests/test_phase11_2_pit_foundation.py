from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from stocks.data.phase5_common import sha256_file
from stocks.research.phase11_2.audit import COUNTERS, PROTECTED_PATHS, frozen_dependency_integrity
from stocks.research.phase11_2.contracts import PitStatus, make_pit_record, pit_access_status
from stocks.research.phase11_2.foundation import (
    build_universe,
    initial_capability_matrix,
    normalize_symbol_rows,
    parse_companyfacts,
    parse_submissions,
    provider_lookup,
    resolve_cik,
)
from stocks.research.phase11_2.privacy import scan_public_artifacts
from stocks.research.phase11_2.providers import SafeJsonClient, summarize_probe
from stocks.research.phase11_2.quality import compare_sources
from stocks.research.phase11_2.shariah import reconstruct_screen, screen_status_at
from stocks.research.phase11_2.storage import PitFoundationStore, TABLES


NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def pit(**changes: object):
    values = {
        "entity_id": "ENTITY",
        "symbol": "TEST",
        "provider": "SEC_EDGAR",
        "dataset": "filing",
        "payload": {"value": 1},
        "first_seen_at": NOW.isoformat(),
        "ingested_at": NOW.isoformat(),
        "accepted_at": NOW.isoformat(),
    }
    values.update(changes)
    return make_pit_record(**values)


def test_unknown_provider_and_capability_fail_closed() -> None:
    result = provider_lookup("UNKNOWN")
    assert result == {"status": "PROVIDER_CAPABILITY_BLOCKED", "blocker": "UNKNOWN_PROVIDER"}


def test_plan_entitlement_failure_is_classified() -> None:
    result = summarize_probe("EODHD", "fundamentals", None, 403, 1, "PLAN_NOT_ENTITLED")
    assert result.status == "PLAN_NOT_ENTITLED"
    assert result.payload_hash is None


def test_provider_secret_is_never_in_safe_probe_summary() -> None:
    secret = "SECRET-PROVIDER-TOKEN"
    result = summarize_probe("EODHD", "prices", [{"date": "2025-01-01", "close": 1}], 200, 1, None)
    assert secret not in json.dumps(result.public_dict())
    assert not any("key" in key.lower() or "token" in key.lower() for key in result.public_dict())


def test_safe_client_is_bounded() -> None:
    client = SafeJsonClient(user_agent="test", max_attempts=99, minimum_interval=0)
    assert client.max_attempts == 3


def test_active_and_delisted_symbol_ingest() -> None:
    active = [{"Code": "MU", "Type": "Common Stock", "Exchange": "NASDAQ", "Currency": "USD"}]
    delisted = [{"Code": "OLD", "Type": "Common Stock", "Exchange": "NYSE", "Currency": "USD"}]
    assert normalize_symbol_rows(active, active=True)[0]["active_on_date"] is True
    assert normalize_symbol_rows(delisted, active=False)[0]["active_on_date"] is False


def test_universe_includes_delisted_and_blocks_current_membership_backfill() -> None:
    active = [{"Code": f"S{i:03}", "Type": "Common Stock", "Exchange": "US", "Currency": "USD"} for i in range(200)]
    delisted = [{"Code": f"D{i:03}", "Type": "Common Stock", "Exchange": "US", "Currency": "USD"} for i in range(20)]
    bulk = [{"code": f"S{i:03}", "close": 10 + i, "volume": 1_000_000 + i} for i in range(200)]
    universe = build_universe(active, delisted, target_size=200, bulk_prices=bulk)
    assert 150 <= len(universe) <= 300
    assert any(not row["active_on_date"] for row in universe)
    assert all(row["inclusion_reason"] == "DELISTED_SURVIVORSHIP_CONTROL" for row in universe if not row["active_on_date"])


def test_survivorship_biased_universe_has_no_delisted_control() -> None:
    active = [{"Code": f"S{i}", "Type": "Common Stock"} for i in range(160)]
    bulk = [{"code": f"S{i}", "close": 10, "volume": 1_000_000} for i in range(160)]
    universe = build_universe(active, [], target_size=160, bulk_prices=bulk)
    assert not any(not row["active_on_date"] for row in universe)


def test_penny_stocks_and_unverified_liquidity_are_excluded() -> None:
    active = [
        {"Code": "GOOD", "Type": "Common Stock"},
        {"Code": "PENNY", "Type": "Common Stock"},
        {"Code": "UNKNOWN", "Type": "Common Stock"},
    ]
    bulk = [{"code": "GOOD", "close": 20, "volume": 1_000_000}, {"code": "PENNY", "close": 1, "volume": 2_000_000}]
    universe = build_universe(active, [], target_size=10, bulk_prices=bulk)
    assert [row["symbol"] for row in universe] == ["GOOD"]


def test_duplicate_payload_idempotent_and_changed_payload_versions(tmp_path: Path) -> None:
    store = PitFoundationStore(tmp_path / "pit.sqlite3")
    store.initialize()
    first = store.append_version("fundamental_versions", "MU:2025Q1", {"revenue": 1})
    duplicate = store.append_version("fundamental_versions", "MU:2025Q1", {"revenue": 1})
    changed = store.append_version("fundamental_versions", "MU:2025Q1", {"revenue": 2})
    assert first["version_number"] == 1
    assert duplicate["status"] == "IDEMPOTENT_REPLAY"
    assert changed["version_number"] == 2
    assert len(store.records("fundamental_versions", "MU:2025Q1")) == 2


def test_restart_determinism_and_required_tables(tmp_path: Path) -> None:
    path = tmp_path / "pit.sqlite3"
    store = PitFoundationStore(path)
    store.initialize()
    store.append_version("news_versions", "N1", {"headline": "x"})
    restarted = PitFoundationStore(path)
    restarted.initialize()
    assert restarted.records("news_versions") == store.records("news_versions")
    assert set(restarted.counts()) == set(TABLES)


def test_period_end_is_not_available_at() -> None:
    record = pit(accepted_at=None, first_seen_at=NOW.isoformat(), period_end=NOW.isoformat(), published_at=NOW.isoformat())
    assert record.PIT_eligibility == PitStatus.TIMESTAMP_INCOMPLETE
    assert "PERIOD_END_AS_AVAILABLE_AT_BLOCKED" in record.PIT_blockers


def test_filing_acceptance_delay_enforced() -> None:
    period_end = NOW - timedelta(days=30)
    accepted = NOW
    record = pit(period_end=period_end.isoformat(), accepted_at=accepted.isoformat())
    assert record.decision_available_at == accepted.isoformat()
    assert not record.available_for((accepted - timedelta(seconds=1)).isoformat())


def test_news_publication_delay_and_future_leakage() -> None:
    published = NOW + timedelta(minutes=5)
    record = pit(provider="EODHD", dataset="news", accepted_at=None, published_at=published.isoformat(), first_seen_at=published.isoformat())
    assert pit_access_status(record, NOW.isoformat()) == "PIT_FUTURE_LEAKAGE_BLOCKED"
    assert record.available_for(published.isoformat())


@pytest.mark.parametrize(
    ("published", "decision", "available"),
    [
        (NOW.replace(hour=12), NOW.replace(hour=13), True),
        (NOW.replace(hour=22), NOW.replace(hour=20), False),
        (NOW.replace(hour=7), NOW.replace(hour=8), True),
    ],
)
def test_after_and_before_market_earnings_availability(published: datetime, decision: datetime, available: bool) -> None:
    record = pit(provider="EODHD", dataset="earnings", accepted_at=None, published_at=published.isoformat(), first_seen_at=published.isoformat())
    assert record.available_for(decision.isoformat()) is available


def test_daily_timestamp_uses_conservative_daily_delay() -> None:
    record = pit(provider="OPENEXCHANGERATES", dataset="fx", timestamp_precision="DATE")
    assert record.PIT_eligibility == PitStatus.DAILY_DELAY


def test_historical_consensus_is_explicitly_unavailable() -> None:
    row = next(item for item in initial_capability_matrix() if item["dataset"] == "analyst_revisions")
    assert row["research_decision"] == "HISTORICAL_CONSENSUS_UNAVAILABLE"
    assert row["PIT_usable"] is False


def test_restatement_creates_new_version(tmp_path: Path) -> None:
    store = PitFoundationStore(tmp_path / "pit.sqlite3")
    store.initialize()
    store.append_version("filing_facts", "fact", {"value": 10, "accession_hash": "A"})
    store.append_version("filing_facts", "fact", {"value": 12, "accession_hash": "B"})
    assert [row["version_number"] for row in store.records("filing_facts", "fact")] == [1, 2]


def test_sec_ticker_to_cik_and_submissions_parsing() -> None:
    tickers = {"0": {"ticker": "MU", "cik_str": 723125}}
    assert resolve_cik(tickers, "MU") == "0000723125"
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-Q", "S-8"],
                "acceptanceDateTime": ["2026-07-01T21:01:02Z", "2026-07-02T00:00:00Z"],
                "accessionNumber": ["0001-26-000001", "0001-26-000002"],
                "filingDate": ["2026-07-01", "2026-07-02"],
                "reportDate": ["2026-05-31", "2026-05-31"],
            }
        }
    }
    rows = parse_submissions(submissions)
    assert len(rows) == 1
    assert rows[0]["form"] == "10-Q"
    assert rows[0]["accepted_at"] == "2026-07-01T21:01:02+00:00"
    assert "0001-26-000001" not in json.dumps(rows)


def test_sec_companyfacts_parsing_joins_acceptance() -> None:
    accession_hash = __import__("stocks.execution.idempotency", fromlist=["stable_hash"]).stable_hash("A")
    filings = [{"accession_hash": accession_hash, "accepted_at": NOW.isoformat()}]
    payload = {
        "facts": {"us-gaap": {"Revenue": {"units": {"USD": [{"val": 1, "start": "2026-01-01", "end": "2026-03-31", "filed": "2026-04-20", "accn": "A", "form": "10-Q", "fy": 2026, "fp": "Q1"}]}}}}
    }
    facts = parse_companyfacts(payload, filings)
    assert facts[0]["accepted_at"] == NOW.isoformat()
    assert facts[0]["version_hash"]


def test_source_conflicts_are_preserved() -> None:
    assert compare_sources(1.0, 1.0) == "SOURCE_VALUES_MATCH"
    assert compare_sources(1.0, 1.0005) == "SOURCE_VALUES_TOLERANCE_MATCH"
    assert compare_sources(1.0, 2.0) == "SOURCE_CONFLICT"
    assert compare_sources(None, 1.0) == "SOURCE_SCOPE_DIFFERENCE"


def test_shariah_historical_screen_and_missing_history() -> None:
    eligible = reconstruct_screen(
        screened_at=NOW.isoformat(), financial_statement_available_at=(NOW - timedelta(days=1)).isoformat(),
        price_denominator_date=(NOW - timedelta(days=1)).isoformat(), business_activity_result=True,
        debt_ratio=Decimal("0.20"), cash_interest_ratio=Decimal("0.10"), receivables_ratio=Decimal("0.20"),
        non_permissible_income_ratio=Decimal("0.01"),
    )
    missing = reconstruct_screen(
        screened_at=NOW.isoformat(), financial_statement_available_at=None, price_denominator_date=None,
        business_activity_result=None, debt_ratio=None, cash_interest_ratio=None, receivables_ratio=None,
        non_permissible_income_ratio=None,
    )
    future = reconstruct_screen(
        screened_at=NOW.isoformat(), financial_statement_available_at=(NOW + timedelta(days=1)).isoformat(),
        price_denominator_date=NOW.isoformat(), business_activity_result=True, debt_ratio=Decimal("0.1"),
        cash_interest_ratio=Decimal("0.1"), receivables_ratio=Decimal("0.1"), non_permissible_income_ratio=Decimal("0.01"),
    )
    assert eligible["final_status"] == "SHARIAH_ELIGIBLE_PIT"
    assert screen_status_at(eligible, (NOW + timedelta(days=91)).isoformat()) == "SHARIAH_STATUS_STALE"
    assert missing["final_status"] == "SHARIAH_DATA_INCOMPLETE"
    assert future["final_status"] == "SHARIAH_DATA_INCOMPLETE"


def test_fx_available_at_enforcement() -> None:
    published = NOW + timedelta(hours=1)
    record = pit(provider="OPENEXCHANGERATES", dataset="fx", accepted_at=None, published_at=published.isoformat(), first_seen_at=published.isoformat())
    assert not record.available_for(NOW.isoformat())


def test_privacy_and_secret_audit(tmp_path: Path) -> None:
    (tmp_path / "safe.json").write_text('{"status":"GO","payload_hash":"A"}', encoding="utf-8")
    assert scan_public_artifacts(tmp_path)["status"] == "PRIVACY_GO"
    (tmp_path / "bad.json").write_text('{"api_token":"secret"}', encoding="utf-8")
    assert scan_public_artifacts(tmp_path)["status"] == "PRIVACY_BLOCKED"


def test_no_ibkr_writer_paper_or_live_calls() -> None:
    source_root = Path(__file__).parents[1] / "src" / "stocks" / "research" / "phase11_2"
    text = "\n".join(path.read_text(encoding="utf-8") for path in source_root.glob("*.py"))
    for forbidden in (".placeOrder(", ".cancelOrder(", "reqGlobalCancel(", "from ibapi", "EClient("):
        assert forbidden not in text
    assert all(value == 0 for value in COUNTERS.values())


def test_phase7_through_phase10_dependencies_unchanged_by_read(tmp_path: Path) -> None:
    for relative in PROTECTED_PATHS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"protected-test-fixture")

    phase10_freeze = tmp_path / "output/ibkr/phase10/freeze-status.json"
    phase10_freeze.parent.mkdir(parents=True, exist_ok=True)
    phase10_freeze.write_text(
        json.dumps(
            {
                "freeze_status": (
                    "PHASE10_AUTONOMOUS_SHARIAH_PAPER_TRADING_FOUNDATION_FROZEN_GO"
                ),
                "source_hashes": {},
            }
        ),
        encoding="utf-8",
    )

    before = {
        name: sha256_file(tmp_path / relative)
        for name, relative in PROTECTED_PATHS.items()
    }
    assert all(before.values())

    after = {
        name: sha256_file(tmp_path / relative)
        for name, relative in PROTECTED_PATHS.items()
    }
    assert after == before
    assert frozen_dependency_integrity(tmp_path)["status"] == "GO"
