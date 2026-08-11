from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stocks.data.corporate_actions import (
    CORPORATE_ACTION_FIELDS,
    CorporateActionLayout,
    _dividend_record,
    _parse_split_ratio,
    validate_corporate_action_cache,
)
from stocks.data.fx import FxCacheLayout, _filled_currency_records, validate_fx_cache
from stocks.data.phase5_common import sha256_json
from stocks.data.total_returns import (
    TotalReturnLayout,
    _build_one_instrument,
    _cumulative_split_factor,
    _tr_hash,
    validate_total_return_cache,
)
from stocks.domain.assets import IbkrSecurityType
from stocks.ibkr.contract_cache import ContractCacheRow
from stocks.ibkr.contracts import ResolvedContract


def test_split_factor_formulas_cover_forward_reverse_and_multiple_splits() -> None:
    two_for_one = {"effective_date": "2026-01-10", "split_factor": "2"}
    four_for_one = {"effective_date": "2026-01-20", "split_factor": "4"}
    reverse = {"effective_date": "2026-01-30", "split_factor": "0.1"}

    assert _cumulative_split_factor(_date("2026-01-09"), [two_for_one]) == Decimal("2")
    assert _cumulative_split_factor(_date("2026-01-19"), [two_for_one, four_for_one]) == Decimal("4")
    assert _cumulative_split_factor(_date("2026-01-01"), [two_for_one, four_for_one]) == Decimal("8")
    assert _cumulative_split_factor(_date("2026-01-29"), [reverse]) == Decimal("0.1")


def test_eodhd_split_ratio_parser_handles_reverse_split() -> None:
    split_from, split_to, split_factor = _parse_split_ratio({"split": "1/10"})

    assert split_from == Decimal("10")
    assert split_to == Decimal("1")
    assert split_factor == Decimal("0.1")


def test_total_return_engine_combines_dividend_fx_and_indexes_without_mutating_raw_bars() -> None:
    row = _contract_row(currency="USD")
    raw_bars = [
        _bar("2026-01-01", "100"),
        _bar("2026-01-02", "101"),
    ]
    raw_hash_before = sha256_json({"raw": raw_bars})
    records = _build_one_instrument(
        row=row,
        raw_bar_records=raw_bars,
        events=[_cash_dividend(row, "2026-01-02", "2")],
        fx_by_key={
            ("USD", "2026-01-01"): _fx("2026-01-01", "0.90"),
            ("USD", "2026-01-02"): _fx("2026-01-02", "0.95"),
        },
        base_currency="EUR",
        ca_content_hash="A" * 64,
        fx_file_hash="B" * 64,
    )

    assert sha256_json({"raw": raw_bars}) == raw_hash_before
    assert Decimal(records[1]["local_price_return"]) == Decimal("0.01")
    assert Decimal(records[1]["local_dividend_return"]) == Decimal("0.02")
    assert Decimal(records[1]["local_total_return"]) == Decimal("0.03")
    assert float(Decimal(records[1]["fx_return"])) == pytest.approx(0.05555555555555556)
    assert float(Decimal(records[1]["eur_total_return"])) == pytest.approx(0.08722222222222223)
    assert records[1]["content_hash"] == _tr_hash(records[1])


def test_total_return_engine_has_no_fx_effect_for_eur_instrument() -> None:
    row = _contract_row(currency="EUR")
    records = _build_one_instrument(
        row=row,
        raw_bar_records=[_bar("2026-01-01", "100"), _bar("2026-01-02", "110")],
        events=[],
        fx_by_key={
            ("EUR", "2026-01-01"): _fx("2026-01-01", "1"),
            ("EUR", "2026-01-02"): _fx("2026-01-02", "1"),
        },
        base_currency="EUR",
        ca_content_hash="A" * 64,
        fx_file_hash="B" * 64,
    )

    assert records[1]["fx_return"] == "0"
    assert records[1]["eur_total_return"] == records[1]["local_total_return"]


def test_multiple_dividends_same_day_are_summed() -> None:
    row = _contract_row(currency="USD")

    records = _build_one_instrument(
        row=row,
        raw_bar_records=[_bar("2026-01-01", "100"), _bar("2026-01-02", "100")],
        events=[_cash_dividend(row, "2026-01-02", "1"), _cash_dividend(row, "2026-01-02", "1.5")],
        fx_by_key={
            ("USD", "2026-01-01"): _fx("2026-01-01", "0.90"),
            ("USD", "2026-01-02"): _fx("2026-01-02", "0.90"),
        },
        base_currency="EUR",
        ca_content_hash="A" * 64,
        fx_file_hash="B" * 64,
    )

    assert Decimal(records[1]["cash_distribution"]) == Decimal("2.5")
    assert Decimal(records[1]["local_dividend_return"]) == Decimal("0.025")


def test_dividend_in_other_currency_is_converted_to_instrument_currency() -> None:
    row = _contract_row(currency="USD")
    event = _cash_dividend(row, "2026-01-02", "1")
    event["currency"] = "GBP"

    records = _build_one_instrument(
        row=row,
        raw_bar_records=[_bar("2026-01-01", "100"), _bar("2026-01-02", "100")],
        events=[event],
        fx_by_key={
            ("USD", "2026-01-01"): _fx("2026-01-01", "0.90"),
            ("USD", "2026-01-02"): _fx("2026-01-02", "0.90"),
            ("GBP", "2026-01-02"): _fx("2026-01-02", "1.20"),
        },
        base_currency="EUR",
        ca_content_hash="A" * 64,
        fx_file_hash="B" * 64,
    )

    assert Decimal(records[1]["cash_distribution"]) == Decimal("1.333333333333333333333333333")


def test_etf_distribution_type_classification_preserves_source_category() -> None:
    row = _contract_row()

    special = _dividend_record(
        row,
        {"date": "2026-01-02", "value": "1", "currency": "USD", "period": "special distribution"},
        downloaded_at="2026-01-03T00:00:00+00:00",
    )
    capital_gains = _dividend_record(
        row,
        {"date": "2026-01-03", "value": "1", "currency": "USD", "period": "capital gains"},
        downloaded_at="2026-01-04T00:00:00+00:00",
    )
    return_of_capital = _dividend_record(
        row,
        {"date": "2026-01-04", "value": "1", "currency": "USD", "period": "return of capital"},
        downloaded_at="2026-01-05T00:00:00+00:00",
    )

    assert special["event_type"] == "SPECIAL_DIVIDEND"
    assert special["distribution_type"] == "special distribution"
    assert capital_gains["distribution_type"] == "capital-gains distribution"
    assert return_of_capital["distribution_type"] == "return of capital"


def test_fx_forward_fill_covers_weekends_and_blocks_stale_rates(tmp_path) -> None:
    layout = FxCacheLayout.from_project_root(tmp_path)
    records = _filled_currency_records(
        "USD",
        by_date={_date("2026-01-02"): Decimal("0.90")},
        start=_date("2026-01-02"),
        end=_date("2026-01-08"),
        downloaded_at="2026-01-09T00:00:00+00:00",
        source="TEST",
    )
    layout.data_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(records), layout.fx_daily_parquet)

    report = validate_fx_cache(layout)

    assert records[1]["is_forward_filled"] is True
    assert records[1]["forward_fill_age"] == 1
    assert report["status"] == "NO_GO"
    assert report["stale_fx_rows"] == 1


def test_corporate_action_cache_detects_deterministic_hashes(tmp_path) -> None:
    layout = CorporateActionLayout.from_project_root(tmp_path)
    record = _cash_dividend(_contract_row(), "2026-01-02", "1.25")
    layout.data_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([record]), layout.corporate_actions_parquet)

    report = validate_corporate_action_cache(layout)

    assert report["status"] == "GO"
    assert report["dividend_event_count"] == 1
    assert report["unresolved_event_count"] == 0


def test_corporate_action_cache_reports_provider_conflicts(tmp_path) -> None:
    layout = CorporateActionLayout.from_project_root(tmp_path)
    record = _cash_dividend(_contract_row(), "2026-01-02", "1.25")
    record["resolution_status"] = "AMOUNT_CONFLICT"
    record["event_hash"] = _corporate_action_hash(record)
    layout.data_dir.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([record]), layout.corporate_actions_parquet)

    report = validate_corporate_action_cache(layout)

    assert report["status"] == "GO"
    assert report["provider_conflict_count"] == 1


def test_total_return_cache_validation_rejects_hash_drift(tmp_path) -> None:
    layout = TotalReturnLayout.from_project_root(tmp_path)
    row = _contract_row()
    records = _build_one_instrument(
        row=row,
        raw_bar_records=[_bar("2026-01-01", "100")],
        events=[],
        fx_by_key={("USD", "2026-01-01"): _fx("2026-01-01", "0.90")},
        base_currency="EUR",
        ca_content_hash="A" * 64,
        fx_file_hash="B" * 64,
    )
    records[0]["content_hash"] = "0" * 64
    path = layout.total_returns_path(con_id=row.contract.con_id)
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(records), path)

    report = validate_total_return_cache(layout)

    assert report["status"] == "NO_GO"
    assert "content_hash mismatch" in report["errors"][0]


def _date(value: str):
    from datetime import date

    return date.fromisoformat(value)


def _contract_row(*, currency: str = "USD") -> ContractCacheRow:
    return ContractCacheRow(
        contract=ResolvedContract(
            con_id=12345,
            symbol="TEST",
            local_symbol="TEST",
            security_type=IbkrSecurityType.STK,
            exchange="SMART",
            primary_exchange="ARCA",
            currency=currency,
            trading_class="TEST",
            min_tick=Decimal("0.01"),
            valid_exchanges="SMART,ARCA",
            market_rule_ids="26",
            long_name="Test Instrument",
            time_zone_id="US/Eastern",
            trading_hours="20260101:0930-1600",
            liquid_hours="20260101:0930-1600",
        ),
        resolved_at=datetime(2026, 1, 1, tzinfo=UTC),
        server_version=225,
    )


def _bar(session_date: str, close: str) -> dict[str, object]:
    return {
        "timestamp_utc": f"{session_date}T20:00:00+00:00",
        "session_date": session_date,
        "close": close,
    }


def _fx(session_date: str, rate: str) -> dict[str, object]:
    return {
        "session_date": session_date,
        "base_currency": "USD",
        "quote_currency": "EUR",
        "rate": rate,
        "source": "TEST",
    }


def _corporate_action_hash(record: dict[str, object]) -> str:
    return sha256_json({field: record.get(field) for field in CORPORATE_ACTION_FIELDS if field != "event_hash"})


def _cash_dividend(row: ContractCacheRow, ex_date: str = "2026-01-02", amount: str = "1") -> dict[str, object]:
    return _dividend_record(
        row,
        {
            "date": ex_date,
            "value": amount,
            "currency": row.contract.currency,
            "period": "quarterly",
            "recordDate": ex_date,
            "paymentDate": ex_date,
            "declarationDate": ex_date,
        },
        downloaded_at="2026-01-03T00:00:00+00:00",
    )
