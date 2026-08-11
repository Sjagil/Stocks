from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import main
from stocks.data.bars import (
    BarCacheLayout,
    BarDataSource,
    BarDataType,
    BarInterval,
    BarRequestPolicy,
    HistoricalBar,
    HistoricalBarRequest,
    TARGET_BAR_COLLECTION_FIELDS,
    bar_cache_manifest,
    bar_schema_manifest,
    detect_bar_gaps,
    initialize_bar_cache,
    plan_bar_request_queue,
    read_bar_cache_records,
    validate_bar_request_fields,
    validate_bar_cache,
    validate_bar_contract_identity_links,
    write_bar_cache_file,
)
from stocks.data.ibkr_historical import (
    HistoricalCollectorClientIdCollision,
    _HistoricalCollectorSingleFlight,
    build_daily_stk_request_plan,
    classify_daily_bar_gaps,
    record_from_ibkr_bar,
)
from stocks.domain.assets import IbkrSecurityType
from stocks.ibkr.contract_cache import (
    ContractCacheLayout,
    ContractCacheRow,
    initialize_contract_cache,
    write_contract_cache_rows,
)
from stocks.ibkr.contracts import ResolvedContract


def test_bar_schema_manifest_is_offline_only() -> None:
    manifest = bar_schema_manifest()

    assert manifest["schema"] == "historical_bar_cache_schema_v1"
    assert manifest["status"] == "OFFLINE_SCHEMA_ONLY"
    assert manifest["supported_intervals"] == ["1d", "1h", "15m"]
    assert "ADJUSTED_LAST" in manifest["supported_data_types"]
    assert manifest["supported_sources"] == ["IBKR", "EODHD"]
    assert manifest["data_phase_authority"] == "phase4_read_only_historical_ibkr_daily_stk"
    assert manifest["request_policy"]["max_concurrent_ibkr_requests"] == 3
    assert manifest["request_policy"]["authority"] == "phase4_read_only_historical_ibkr_daily_stk"
    assert manifest["financial_calls"]["place_order"] == 0


def test_bar_cache_manifest_covers_layout_policy_and_zero_financial_calls(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)

    manifest = bar_cache_manifest(layout)

    assert manifest["schema"] == "historical_bar_cache_manifest_v1"
    assert manifest["layout"] == layout.as_dict()
    assert manifest["required_fields"][0] == "con_id"
    assert manifest["supported_intervals"] == ["1d", "1h", "15m"]
    assert manifest["request_policy"]["max_concurrent_ibkr_requests"] == 3
    assert manifest["data_phase_authority"] == "phase4_read_only_historical_ibkr_daily_stk"
    assert manifest["financial_calls"] == {"place_order": 0, "cancel_order": 0, "global_cancel": 0}


def test_initialize_bar_cache_writes_manifest_and_validates(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)

    manifest = initialize_bar_cache(layout)
    validation = validate_bar_cache(layout)

    assert manifest["initialized"] is True
    assert layout.manifest_json.exists()
    assert validation["status"] == "GO"
    assert validation["manifest"]["exists"] is True
    assert validation["manifest"]["error_count"] == 0


def test_validate_bar_cache_blocks_manifest_financial_call_drift(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    manifest = initialize_bar_cache(layout)
    manifest["financial_calls"]["place_order"] = 1
    layout.manifest_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert report["manifest"]["error_count"] == 1
    assert "financial_calls must all be 0" in report["errors"][0]


def test_historical_bar_request_is_disabled_until_data_phase() -> None:
    request = _bar_request()

    with pytest.raises(ValueError, match="disabled until the data phase"):
        request.validate()

    request.validate(data_phase_enabled=True)
    payload = request.as_dict()
    assert payload["schema"] == "historical_bar_request_v1"
    assert payload["con_id"] == 265598
    assert payload["start"] == "2026-07-20T13:30:00+00:00"
    assert payload["financial_calls"]["place_order"] == 0


def test_bar_request_blocks_invalid_eodhd_intraday_planning() -> None:
    with pytest.raises(ValueError, match="FORBIDDEN_SWING_TIMEFRAME"):
        validate_bar_request_fields(
            con_id=265598,
            security_type=IbkrSecurityType.STK,
            interval=BarInterval.FIFTEEN_MINUTES,
            data_type=BarDataType.TRADES,
            source=BarDataSource.EODHD,
            start=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
            end=datetime(2026, 7, 20, 20, 0, tzinfo=UTC),
        )


def test_bar_request_blocks_eodhd_futures_planning() -> None:
    with pytest.raises(ValueError, match="EODHD bar planning is only enabled for STK"):
        validate_bar_request_fields(
            con_id=999001,
            security_type=IbkrSecurityType.FUT,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
            source=BarDataSource.EODHD,
            start=datetime(2026, 7, 20, tzinfo=UTC),
            end=datetime(2026, 7, 21, tzinfo=UTC),
        )


def test_bar_request_allows_ibkr_futures_trades_planning() -> None:
    validate_bar_request_fields(
        con_id=999001,
        security_type=IbkrSecurityType.FUT,
        interval=BarInterval.ONE_HOUR,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
        start=datetime(2026, 7, 20, 13, 0, tzinfo=UTC),
        end=datetime(2026, 7, 20, 17, 0, tzinfo=UTC),
    )


def test_bar_request_blocks_adjusted_last_for_futures() -> None:
    with pytest.raises(ValueError, match="ADJUSTED_LAST is only valid for STK"):
        validate_bar_request_fields(
            con_id=999001,
            security_type=IbkrSecurityType.FUT,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.ADJUSTED_LAST,
            source=BarDataSource.IBKR,
            start=datetime(2026, 7, 20, tzinfo=UTC),
            end=datetime(2026, 7, 21, tzinfo=UTC),
        )


def test_bar_request_blocks_boolean_con_id() -> None:
    with pytest.raises(ValueError, match="con_id must be a positive integer"):
        validate_bar_request_fields(
            con_id=True,
            security_type=IbkrSecurityType.STK,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
            source=BarDataSource.EODHD,
            start=datetime(2026, 7, 20, tzinfo=UTC),
            end=datetime(2026, 7, 21, tzinfo=UTC),
        )


def test_bar_request_blocks_non_integer_con_id() -> None:
    with pytest.raises(ValueError, match="con_id must be a positive integer"):
        validate_bar_request_fields(
            con_id=265598.0,
            security_type=IbkrSecurityType.STK,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
            source=BarDataSource.EODHD,
            start=datetime(2026, 7, 20, tzinfo=UTC),
            end=datetime(2026, 7, 21, tzinfo=UTC),
        )


def test_bar_request_blocks_string_security_type() -> None:
    with pytest.raises(ValueError, match="security_type must be one of: STK, FUT"):
        validate_bar_request_fields(
            con_id=265598,
            security_type="STK",
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
            source=BarDataSource.EODHD,
            start=datetime(2026, 7, 20, tzinfo=UTC),
            end=datetime(2026, 7, 21, tzinfo=UTC),
        )


def test_bar_request_blocks_string_start_datetime() -> None:
    with pytest.raises(ValueError, match="start must be a timezone-aware datetime"):
        validate_bar_request_fields(
            con_id=265598,
            security_type=IbkrSecurityType.STK,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
            source=BarDataSource.EODHD,
            start="2026-07-20T00:00:00+00:00",
            end=datetime(2026, 7, 21, tzinfo=UTC),
        )


def test_bar_request_policy_blocks_too_many_concurrent_ibkr_requests() -> None:
    policy = BarRequestPolicy(max_concurrent_ibkr_requests=6)

    with pytest.raises(ValueError, match="between 1 and 5"):
        policy.validate()


def test_bar_request_policy_blocks_boolean_concurrency() -> None:
    policy = BarRequestPolicy(max_concurrent_ibkr_requests=True)

    with pytest.raises(ValueError, match="max_concurrent_ibkr_requests must be a positive integer"):
        policy.validate()


def test_bar_request_policy_blocks_float_timeout() -> None:
    policy = BarRequestPolicy(request_timeout_seconds=60.0)

    with pytest.raises(ValueError, match="request_timeout_seconds must be a positive integer"):
        policy.validate()


def test_bar_request_policy_blocks_boolean_retries() -> None:
    policy = BarRequestPolicy(max_retries=False)

    with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
        policy.validate()


def test_bar_request_policy_blocks_non_tuple_backoff() -> None:
    policy = BarRequestPolicy(retry_backoff_seconds=[2, 5, 15])

    with pytest.raises(ValueError, match="retry_backoff_seconds must be a tuple of positive integers"):
        policy.validate()


def test_bar_request_policy_blocks_boolean_backoff_value() -> None:
    policy = BarRequestPolicy(retry_backoff_seconds=(2, True, 15))

    with pytest.raises(ValueError, match="retry_backoff_seconds value must be a positive integer"):
        policy.validate()


def test_plan_bar_request_queue_deduplicates_and_rejects_invalid_requests() -> None:
    valid = _bar_request()
    invalid = HistoricalBarRequest(
        con_id=999001,
        security_type=IbkrSecurityType.FUT,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.ADJUSTED_LAST,
        source=BarDataSource.IBKR,
        start=datetime(2026, 7, 20, tzinfo=UTC),
        end=datetime(2026, 7, 21, tzinfo=UTC),
    )

    plan = plan_bar_request_queue([valid, valid, invalid])

    assert plan["schema"] == "historical_bar_request_queue_plan_v1"
    assert plan["status"] == "NO_GO"
    assert plan["execution_enabled"] is False
    assert plan["queued_count"] == 1
    assert plan["rejected_count"] == 2
    assert plan["duplicate_request_count"] == 1
    assert plan["source_counts"] == {"IBKR": 0, "EODHD": 1}
    assert plan["rejected_requests"][0]["reason"] == "duplicate request of index 0"
    assert plan["rejected_requests"][1]["reason"] == "ADJUSTED_LAST is only valid for STK bars"
    assert plan["financial_calls"]["place_order"] == 0


def test_plan_bar_request_queue_blocks_invalid_policy() -> None:
    with pytest.raises(ValueError, match="request_timeout_seconds must be a positive integer"):
        plan_bar_request_queue([_bar_request()], policy=BarRequestPolicy(request_timeout_seconds=60.0))


def test_plan_bar_request_queue_rejects_legacy_fifteen_minute_request() -> None:
    eodhd = _bar_request()
    ibkr = HistoricalBarRequest(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
        start=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        end=datetime(2026, 7, 20, 20, 0, tzinfo=UTC),
    )

    plan = plan_bar_request_queue([eodhd, ibkr])

    assert plan["status"] == "NO_GO"
    assert plan["queued_count"] == 1
    assert plan["rejected_requests"][0]["reason"] == (
        "FORBIDDEN_SWING_TIMEFRAME:15m"
    )
    assert plan["source_counts"] == {"IBKR": 0, "EODHD": 1}
    assert plan["policy"]["request_timeout_seconds"] == 60
    assert plan["policy"]["max_retries"] == 3
    assert plan["financial_calls"]["place_order"] == 0


def test_plan_bar_request_queue_validates_matching_contract_identity() -> None:
    plan = plan_bar_request_queue([_bar_request()], contract_rows=[_cached_stock_row()])

    assert plan["status"] == "GO"
    assert plan["queued_count"] == 1
    assert plan["rejected_requests"] == []
    assert plan["contract_identity_validation"] == {
        "enabled": True,
        "contract_cache_row_count": 1,
    }
    assert plan["financial_calls"]["place_order"] == 0


def test_plan_bar_request_queue_blocks_missing_contract_identity() -> None:
    plan = plan_bar_request_queue([_bar_request()], contract_rows=[])

    assert plan["status"] == "NO_GO"
    assert plan["queued_count"] == 0
    assert plan["rejected_count"] == 1
    assert "missing from local contract cache" in plan["rejected_requests"][0]["reason"]
    assert plan["contract_identity_validation"]["enabled"] is True
    assert plan["financial_calls"]["place_order"] == 0


def test_plan_bar_request_queue_blocks_security_type_mismatch_for_con_id() -> None:
    plan = plan_bar_request_queue([_bar_request()], contract_rows=[_cached_future_row(con_id=265598)])

    assert plan["status"] == "NO_GO"
    assert plan["queued_count"] == 0
    assert plan["rejected_count"] == 1
    assert plan["rejected_requests"][0]["request"]["sec_type"] == "STK"
    assert "STK con_id 265598 missing from local contract cache" in plan["rejected_requests"][0]["reason"]
    assert plan["financial_calls"]["place_order"] == 0


def test_plan_bar_request_queue_blocks_boolean_con_id() -> None:
    request = HistoricalBarRequest(
        con_id=True,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
        source=BarDataSource.EODHD,
        start=datetime(2026, 7, 20, tzinfo=UTC),
        end=datetime(2026, 7, 21, tzinfo=UTC),
    )

    plan = plan_bar_request_queue([request])

    assert plan["status"] == "NO_GO"
    assert plan["queued_count"] == 0
    assert plan["rejected_count"] == 1
    assert plan["rejected_requests"][0]["reason"] == "con_id must be a positive integer"
    assert plan["financial_calls"]["place_order"] == 0


def test_plan_bar_request_queue_reports_invalid_runtime_enum_without_crashing() -> None:
    request = HistoricalBarRequest(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
        source="EODHD",
        start=datetime(2026, 7, 20, tzinfo=UTC),
        end=datetime(2026, 7, 21, tzinfo=UTC),
    )

    plan = plan_bar_request_queue([request])

    assert plan["status"] == "NO_GO"
    assert plan["queued_count"] == 0
    assert plan["rejected_count"] == 1
    assert plan["rejected_requests"][0]["reason"] == "source must be one of: IBKR, EODHD"
    assert plan["rejected_requests"][0]["request"]["source"] == "EODHD"
    assert plan["financial_calls"]["place_order"] == 0


def test_plan_bar_request_queue_reports_invalid_start_without_crashing() -> None:
    request = HistoricalBarRequest(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
        source=BarDataSource.EODHD,
        start="2026-07-20T00:00:00+00:00",
        end=datetime(2026, 7, 21, tzinfo=UTC),
    )

    plan = plan_bar_request_queue([request])

    assert plan["status"] == "NO_GO"
    assert plan["queued_count"] == 0
    assert plan["rejected_count"] == 1
    assert plan["rejected_requests"][0]["reason"] == "start must be a timezone-aware datetime"
    assert plan["rejected_requests"][0]["request"]["start"] == "2026-07-20T00:00:00+00:00"
    assert plan["financial_calls"]["place_order"] == 0


def test_bar_cache_partition_path_is_canonical(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)

    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
        source=BarDataSource.EODHD,
    )

    assert path == (
        tmp_path
        / "data"
        / "bars"
        / "sec_type=STK"
        / "con_id=265598"
        / "interval=1d"
        / "data_type=TRADES"
        / "source=EODHD"
        / "bars.parquet"
    )


def test_bar_cache_partition_path_blocks_boolean_con_id(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)

    with pytest.raises(ValueError, match="con_id must be a positive integer"):
        layout.bars_path(
            security_type=IbkrSecurityType.STK,
            con_id=True,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
            source=BarDataSource.EODHD,
        )


def test_bar_cache_partition_path_blocks_string_source(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)

    with pytest.raises(ValueError, match="source must be one of: IBKR, EODHD"):
        layout.bars_path(
            security_type=IbkrSecurityType.STK,
            con_id=265598,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
            source="EODHD",
        )


def test_historical_bar_validation_blocks_incoherent_ohlc() -> None:
    bar = _bar(high=Decimal("99"))

    with pytest.raises(ValueError, match="high must be greater"):
        bar.validate()


def test_historical_bar_validation_blocks_boolean_con_id() -> None:
    bar = HistoricalBar(
        con_id=True,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
        timestamp_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1_000_000,
        available_at=datetime(2026, 7, 20, 13, 45, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="con_id must be a positive integer"):
        bar.validate()


def test_historical_bar_validation_blocks_string_interval() -> None:
    bar = HistoricalBar(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval="15m",
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
        timestamp_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1_000_000,
        available_at=datetime(2026, 7, 20, 13, 45, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="interval must be one of: 1d, 1h, 15m"):
        bar.validate()


def test_historical_bar_validation_blocks_string_timestamp_utc() -> None:
    bar = HistoricalBar(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
        timestamp_utc="2026-07-20T13:30:00+00:00",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1_000_000,
        available_at=datetime(2026, 7, 20, 13, 45, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="timestamp_utc must be a timezone-aware datetime"):
        bar.validate()


def test_historical_bar_validation_blocks_string_ohlc() -> None:
    bar = HistoricalBar(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
        timestamp_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        open="100",
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1_000_000,
        available_at=datetime(2026, 7, 20, 13, 45, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="open must be a Decimal"):
        bar.validate()


def test_historical_bar_validation_blocks_boolean_ohlc() -> None:
    bar = HistoricalBar(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
        timestamp_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        open=Decimal("100"),
        high=True,
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1_000_000,
        available_at=datetime(2026, 7, 20, 13, 45, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="high must be a Decimal"):
        bar.validate()


def test_historical_bar_validation_blocks_boolean_volume() -> None:
    bar = HistoricalBar(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
        timestamp_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=True,
        available_at=datetime(2026, 7, 20, 13, 45, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="volume must be null or a non-negative integer"):
        bar.validate()


def test_historical_bar_validation_blocks_available_at_before_timestamp() -> None:
    bar = _bar(
        timestamp=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        available_at=datetime(2026, 7, 20, 13, 44, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="available_at must be on or after the bar availability time"):
        bar.validate()


def test_daily_bar_allows_available_at_on_timestamp_until_calendar_close_is_known() -> None:
    bar = _bar(
        timestamp=datetime(2026, 7, 20, tzinfo=UTC),
        interval=BarInterval.ONE_DAY,
        source=BarDataSource.EODHD,
        available_at=datetime(2026, 7, 20, tzinfo=UTC),
    )

    bar.validate()


def test_detect_bar_gaps_flags_missing_intraday_step_and_duplicates() -> None:
    bars = [
        _bar(timestamp=datetime(2026, 7, 20, 13, 30, tzinfo=UTC)),
        _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)),
        _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)),
        _bar(timestamp=datetime(2026, 7, 20, 14, 15, tzinfo=UTC)),
    ]

    report = detect_bar_gaps(bars, interval=BarInterval.FIFTEEN_MINUTES)

    assert report["schema"] == "historical_bar_gap_report_v1"
    assert report["status"] == "NO_GO"
    assert report["duplicate_timestamps"] == ["2026-07-20T13:45:00+00:00"]
    assert report["gaps"] == [
        {
            "after": "2026-07-20T13:45:00+00:00",
            "before": "2026-07-20T14:15:00+00:00",
            "expected_delta_seconds": 900,
            "actual_delta_seconds": 1800,
            "missing_steps": 1,
        }
    ]
    assert report["financial_calls"]["place_order"] == 0


def test_daily_gap_detection_requires_market_calendar() -> None:
    report = detect_bar_gaps([_bar()], interval=BarInterval.ONE_DAY)

    assert report["status"] == "GO"
    assert report["gap_detection_scope"] == "requires_market_calendar"
    assert report["gaps"] == []


def test_validate_empty_bar_cache_is_go(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)

    report = validate_bar_cache(layout)

    assert report["schema"] == "historical_bar_cache_validation_v1"
    assert report["status"] == "GO"
    assert report["research_readiness_status"] == "NO_DATA"
    assert report["file_count"] == 0
    assert report["row_count"] == 0
    assert report["instrument_count"] == 0
    assert report["content_hash"] is None
    assert report["financial_calls"]["place_order"] == 0


def test_write_and_validate_bar_cache_file_roundtrip(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )

    write_report = write_bar_cache_file(
        path,
        [
            _bar(timestamp=datetime(2026, 7, 20, 13, 30, tzinfo=UTC)),
            _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)),
            _bar(timestamp=datetime(2026, 7, 20, 14, 0, tzinfo=UTC)),
        ],
    )
    validation = validate_bar_cache(layout)

    assert write_report["status"] == "GO"
    assert validation["status"] == "GO"
    assert validation["research_readiness_status"] == "GO"
    assert validation["file_count"] == 1
    assert validation["row_count"] == 3
    assert validation["instrument_count"] == 1
    assert validation["duplicate_rows"] == 0
    assert validation["invalid_ohlc_rows"] == 0
    assert validation["timezone_errors"] == 0
    assert validation["contract_mismatches"] == 0
    assert validation["first_timestamp"] == "2026-07-20T13:30:00+00:00"
    assert validation["last_timestamp"] == "2026-07-20T14:00:00+00:00"
    assert validation["content_hash"] is not None
    assert validation["files"][0]["partition"]["con_id"] == 265598
    assert validation["files"][0]["gap_report"]["status"] == "GO"
    assert validation["financial_calls"]["place_order"] == 0


def test_write_bar_cache_file_blocks_out_of_order_timestamps(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )

    with pytest.raises(ValueError, match="timestamp_utc values must be sorted ascending"):
        write_bar_cache_file(
            path,
            [
                _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)),
                _bar(timestamp=datetime(2026, 7, 20, 13, 30, tzinfo=UTC)),
            ],
        )

    assert not path.exists()


def test_write_bar_cache_file_blocks_duplicate_timestamps(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )

    with pytest.raises(ValueError, match="duplicate timestamp 2026-07-20T13:45:00\\+00:00"):
        write_bar_cache_file(
            path,
            [
                _bar(timestamp=datetime(2026, 7, 20, 13, 30, tzinfo=UTC)),
                _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)),
                _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)),
            ],
        )

    assert not path.exists()


def test_write_bar_cache_file_blocks_intraday_gap(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )

    with pytest.raises(ValueError, match="missing_steps=1"):
        write_bar_cache_file(
            path,
            [
                _bar(timestamp=datetime(2026, 7, 20, 13, 30, tzinfo=UTC)),
                _bar(timestamp=datetime(2026, 7, 20, 14, 0, tzinfo=UTC)),
            ],
        )

    assert not path.exists()


def test_validate_bar_cache_blocks_empty_bar_file(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([_bar().as_record()]).slice(0, 0)
    pq.write_table(table, path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert report["file_count"] == 1
    assert report["row_count"] == 0
    assert "bars.parquet must contain at least one bar record" in report["errors"][0]
    assert report["files"][0]["gap_report"] is None
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_unexpected_record_fields(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["provider_symbol"] = "AAPL.US"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "unexpected bar fields: provider_symbol" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_nonfinite_ohlc(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["open"] = "NaN"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "open must be finite" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_unparseable_ohlc(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["close"] = "not-a-decimal"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "close must be a parseable decimal" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_whitespace_ohlc(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["open"] = " 100.00"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "open must be a parseable decimal" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_boolean_ohlc(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["close"] = True
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "close must be a parseable decimal" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_fractional_volume(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["volume"] = "1000.5"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "volume must be null or a non-negative integer" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_unparseable_volume(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["volume"] = "not-volume"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "volume must be null or a non-negative integer" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_negative_volume(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["volume"] = -1
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "volume must be null or a non-negative integer" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_fractional_record_con_id(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["con_id"] = "265598.0"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "con_id must be a positive integer" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_boolean_record_con_id(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=1,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["con_id"] = True
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "con_id must be a positive integer" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_fractional_partition_con_id(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = (
        layout.data_dir
        / "sec_type=STK"
        / "con_id=265598.0"
        / "interval=15m"
        / "data_type=TRADES"
        / "source=IBKR"
        / "bars.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([_bar().as_record()]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "partition con_id must be a positive integer" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_whitespace_record_sec_type(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["sec_type"] = " STK"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "sec_type must be one of: STK, FUT" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_boolean_record_source(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["source"] = True
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "source must be one of: IBKR, EODHD" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_invalid_partition_interval(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = (
        layout.data_dir
        / "sec_type=STK"
        / "con_id=265598"
        / "interval=5m"
        / "data_type=TRADES"
        / "source=IBKR"
        / "bars.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([_bar().as_record()]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "partition interval must be one of: 1d, 1h, 15m" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_bar_contract_identity_links_allow_empty_bar_cache_without_contracts(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    bar_validation = validate_bar_cache(layout)

    report = validate_bar_contract_identity_links(layout, [], bar_validation=bar_validation)

    assert report["schema"] == "historical_bar_contract_identity_link_validation_v1"
    assert report["status"] == "GO"
    assert report["referenced_count"] == 0
    assert report["contract_cache_row_count"] == 0
    assert report["missing_identities"] == []
    assert report["financial_calls"]["place_order"] == 0


def test_bar_contract_identity_links_block_missing_contract_identity(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    write_bar_cache_file(path, [_bar()])
    bar_validation = validate_bar_cache(layout)

    report = validate_bar_contract_identity_links(layout, [], bar_validation=bar_validation)

    assert report["status"] == "NO_GO"
    assert report["referenced_count"] == 1
    assert report["missing_identities"][0]["con_id"] == 265598
    assert report["missing_identities"][0]["sec_type"] == "STK"
    assert "missing from local contract cache" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_bar_contract_identity_links_accept_matching_contract_identity(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    write_bar_cache_file(path, [_bar()])
    bar_validation = validate_bar_cache(layout)

    report = validate_bar_contract_identity_links(layout, [_cached_stock_row()], bar_validation=bar_validation)

    assert report["status"] == "GO"
    assert report["referenced_identities"] == [
        {
            "con_id": 265598,
            "sec_type": "STK",
            "intervals": ["15m"],
            "data_types": ["TRADES"],
            "sources": ["IBKR"],
            "file_count": 1,
            "row_count": 1,
        }
    ]
    assert report["missing_identities"] == []


def test_validate_bar_cache_blocks_record_partition_mismatch(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    wrong_path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=999999,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    write_bar_cache_file(wrong_path, [_bar()])

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "record con_id does not match partition" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_available_at_before_timestamp(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    write_bar_cache_file(path, [_bar()])
    records = read_bar_cache_records(path)
    records[0]["available_at"] = "2026-07-20T13:44:00+00:00"
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "available_at must be on or after the bar availability time" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_whitespace_timestamp_utc(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["timestamp_utc"] = " 2026-07-20T13:30:00+00:00"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "timestamp_utc must be a parseable ISO-8601 datetime" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_boolean_available_at(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar().as_record()
    record["available_at"] = True
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "available_at must be a parseable ISO-8601 datetime" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_eodhd_futures_file(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.FUT,
        con_id=999001,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
        source=BarDataSource.EODHD,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar(
        timestamp=datetime(2026, 7, 20, tzinfo=UTC),
        interval=BarInterval.ONE_DAY,
        source=BarDataSource.EODHD,
        available_at=datetime(2026, 7, 20, tzinfo=UTC),
    ).as_record()
    record["con_id"] = 999001
    record["sec_type"] = "FUT"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "EODHD bar planning is only enabled for STK identities" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_adjusted_last_futures_file(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.FUT,
        con_id=999001,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.ADJUSTED_LAST,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    record = _bar(
        timestamp=datetime(2026, 7, 20, tzinfo=UTC),
        interval=BarInterval.ONE_DAY,
        available_at=datetime(2026, 7, 20, tzinfo=UTC),
    ).as_record()
    record["con_id"] = 999001
    record["sec_type"] = "FUT"
    record["data_type"] = "ADJUSTED_LAST"
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "ADJUSTED_LAST is only valid for STK bars" in report["errors"][0]
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_out_of_order_timestamps(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)).as_record(),
        _bar(timestamp=datetime(2026, 7, 20, 13, 30, tzinfo=UTC)).as_record(),
    ]
    pq.write_table(pa.Table.from_pylist(records), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert "timestamp_utc values must be sorted ascending" in report["errors"][0]
    assert report["files"][0]["gap_report"] is None
    assert report["financial_calls"]["place_order"] == 0


def test_validate_bar_cache_blocks_intraday_duplicates_and_gaps(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        _bar(timestamp=datetime(2026, 7, 20, 13, 30, tzinfo=UTC)).as_record(),
        _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)).as_record(),
        _bar(timestamp=datetime(2026, 7, 20, 13, 45, tzinfo=UTC)).as_record(),
        _bar(timestamp=datetime(2026, 7, 20, 14, 15, tzinfo=UTC)).as_record(),
    ]
    pq.write_table(pa.Table.from_pylist(records), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "NO_GO"
    assert any("duplicate timestamp 2026-07-20T13:45:00+00:00" in error for error in report["errors"])
    assert any("missing_steps=1" in error for error in report["errors"])
    assert report["financial_calls"]["place_order"] == 0


def test_data_bars_schema_cli_reports_offline_schema(capsys) -> None:
    exit_code = main.main(["data", "bars", "schema"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "historical_bar_cache_schema_v1"
    assert payload["data_phase_authority"] == "phase4_read_only_historical_ibkr_daily_stk"
    assert payload["financial_calls"]["place_order"] == 0


def test_data_bars_validate_cache_cli_reports_empty_cache_go(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "data", "bars", "validate-cache"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "historical_bar_cache_validation_command_v1"
    assert payload["status"] == "GO"
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert payload["phase4"]["enabled"] is True
    assert payload["cache_validation"]["row_count"] == 0
    assert payload["contract_cache_validation"] is None
    assert payload["contract_identity_links"]["referenced_count"] == 0
    assert payload["contract_identity_links"]["status"] == "GO"
    assert payload["financial_calls"]["place_order"] == 0


def test_data_bars_validate_cache_cli_blocks_bar_without_contract_identity(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    layout = BarCacheLayout.from_project_root(tmp_path)
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    write_bar_cache_file(path, [_bar()])

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "data", "bars", "validate-cache"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "NO_GO"
    assert payload["cache_validation"]["status"] == "GO"
    assert payload["contract_cache_validation"]["status"] == "GO"
    assert payload["contract_identity_links"]["missing_identities"][0]["con_id"] == 265598
    assert payload["financial_calls"]["place_order"] == 0


def test_phase4_daily_stk_request_plan_records_required_ibkr_fields() -> None:
    row = _cached_stock_row()

    plan = build_daily_stk_request_plan(row, start=date(2020, 1, 1), end=date(2026, 7, 20))

    assert plan.con_id == 265598
    assert plan.bar_size == "1 day"
    assert plan.what_to_show == "TRADES"
    assert plan.use_rth is True
    assert plan.format_date == 1
    assert plan.keep_up_to_date is False
    assert plan.planned_session_range == {"start": "2020-01-01", "end": "2026-07-20"}
    assert plan.max_attempts == 3
    assert plan.retry_backoff_seconds == (2, 5, 15)
    assert len(plan.request_hash) == 64


def test_phase4_gap_classification_uses_exchange_calendar_for_holidays() -> None:
    row = _cached_stock_row()
    records = [
        {"session_date": "2026-07-02"},
        {"session_date": "2026-07-06"},
    ]

    report = classify_daily_bar_gaps(
        records,
        session_rows=[],
        contract_row=row,
        start=date(2026, 7, 2),
        end=date(2026, 7, 6),
    )

    classifications = {gap["session_date"]: gap["classification"] for gap in report["gaps"]}
    assert report["status"] == "GO"
    assert classifications["2026-07-03"] == "EXPECTED_HOLIDAY"
    assert classifications["2026-07-04"] == "EXPECTED_WEEKEND"
    assert classifications["2026-07-05"] == "EXPECTED_WEEKEND"
    assert report["blocking_gap_count"] == 0


def test_phase4_historical_collector_single_flight_blocks_same_client_id(tmp_path) -> None:
    output_dir = tmp_path / "output" / "ibkr" / "bars"

    with _HistoricalCollectorSingleFlight(output_dir=output_dir, client_id=1017):
        with pytest.raises(HistoricalCollectorClientIdCollision, match="client_id 1017 is already in use"):
            with _HistoricalCollectorSingleFlight(output_dir=output_dir, client_id=1017):
                pass

    with _HistoricalCollectorSingleFlight(output_dir=output_dir, client_id=1017):
        assert (output_dir / "collector-client-1017.lock").exists()

    assert not (output_dir / "collector-client-1017.lock").exists()


def test_phase4_record_from_ibkr_bar_uses_canonical_schema_and_hashes() -> None:
    row = _cached_stock_row()
    plan = build_daily_stk_request_plan(row, start=date(2026, 7, 20), end=date(2026, 7, 20))
    bar = _FakeIbkrBar(
        date="20260720",
        open=100.0,
        high=105.0,
        low=99.0,
        close=104.0,
        volume=1000,
        wap=102.5,
        bar_count=77,
    )
    session_rows = [
        {
            "con_id": 265598,
            "session_date": "2026-07-20",
            "session_hash": "A" * 64,
            "effective_collection_close_utc": "2026-07-20T20:00:00+00:00",
            "readiness": "SESSION_READY",
            "is_holiday": False,
        }
    ]

    record = record_from_ibkr_bar(
        bar,
        contract_row=row,
        session_rows=session_rows,
        plan=plan,
        requested_at=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
        received_at=datetime(2026, 7, 20, 12, 1, tzinfo=UTC),
    )

    assert set(record) == set(TARGET_BAR_COLLECTION_FIELDS)
    assert record["timestamp_utc"] == "2026-07-20T20:00:00+00:00"
    assert record["session_date"] == "2026-07-20"
    assert record["security_type"] == "STK"
    assert record["interval"] == "1d"
    assert record["data_type"] == "TRADES"
    assert record["use_rth"] is True
    assert record["source"] == "IBKR"
    assert record["contract_hash"] == row.contract_hash
    assert record["session_hash"] == "A" * 64
    assert len(record["content_hash"]) == 64


def test_validate_bar_cache_accepts_phase4_partition_and_canonical_schema(tmp_path) -> None:
    layout = BarCacheLayout.from_project_root(tmp_path)
    row = _cached_stock_row()
    plan = build_daily_stk_request_plan(row, start=date(2026, 7, 20), end=date(2026, 7, 20))
    path = layout.phase4_bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
    )
    record = record_from_ibkr_bar(
        _FakeIbkrBar("20260720", 100.0, 105.0, 99.0, 104.0, 1000, 102.5, 77),
        contract_row=row,
        session_rows=[
            {
                "con_id": 265598,
                "session_date": "2026-07-20",
                "session_hash": "B" * 64,
                "effective_collection_close_utc": "2026-07-20T20:00:00+00:00",
                "readiness": "SESSION_READY",
                "is_holiday": False,
            }
        ],
        plan=plan,
        requested_at=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
        received_at=datetime(2026, 7, 20, 12, 1, tzinfo=UTC),
    )
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([record]), path)

    report = validate_bar_cache(layout)

    assert report["status"] == "GO"
    assert report["file_count"] == 1
    assert report["row_count"] == 1
    assert report["duplicate_rows"] == 0
    assert report["invalid_ohlc_rows"] == 0
    assert report["timezone_errors"] == 0
    assert report["files"][0]["phase4_schema"] is True


def test_data_bars_validate_cache_cli_accepts_matching_contract_cache(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    bar_layout = BarCacheLayout.from_project_root(tmp_path)
    contract_layout = ContractCacheLayout.from_project_root(tmp_path)
    initialize_contract_cache(contract_layout)
    write_contract_cache_rows(contract_layout, [_cached_stock_row()])
    path = bar_layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=265598,
        interval=BarInterval.FIFTEEN_MINUTES,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    write_bar_cache_file(path, [_bar()])

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "data", "bars", "validate-cache"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "GO"
    assert payload["contract_cache_validation"]["row_count"] == 1
    assert payload["contract_identity_links"]["status"] == "GO"
    assert payload["contract_identity_links"]["missing_identities"] == []
    assert payload["financial_calls"]["place_order"] == 0


def test_data_bars_init_cache_cli_writes_manifest_without_provider_calls(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "data", "bars", "init-cache"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "historical_bar_cache_init_command_v1"
    assert payload["status"] == "GO"
    assert payload["phase4"]["enabled"] is False
    assert payload["cache"]["schema"] == "historical_bar_cache_manifest_v1"
    assert (tmp_path / "data" / "bars" / "bar_manifest.json").exists()
    assert payload["financial_calls"]["place_order"] == 0


def test_data_bars_request_policy_cli_reports_phase4_read_only_scope(capsys) -> None:
    exit_code = main.main(["data", "bars", "request-policy"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "historical_bar_request_policy_command_v1"
    assert payload["status"] == "GO"
    assert payload["execution_enabled"] is True
    assert payload["phase4"]["authority"] == "read_only_historical_bars_daily_stk_only"
    assert payload["request_policy"]["max_concurrent_ibkr_requests"] == 3
    assert payload["request_policy"]["ibkr_hard_historical_request_cap"] == 50
    assert payload["financial_calls"]["place_order"] == 0


def test_data_bars_status_cli_is_fail_closed_and_does_not_print_key(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        "EODHD_API_KEY=secret-value\nEODHD_ENABLED=true\n",
        encoding="utf-8",
    )

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "data", "bars", "status"])
    output = capsys.readouterr().out
    payload = json.loads(output)

    assert exit_code == 0
    assert "secret-value" not in output
    assert payload["schema"] == "historical_bar_data_status_v1"
    assert payload["status"] == "NO_GO"
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert payload["data_sources"]["ibkr"]["enabled"] is True
    assert payload["data_sources"]["ibkr"]["authority"] == "read_only_historical_bars_daily_stk_only"
    assert payload["data_sources"]["external_providers"]["enabled"] is False
    assert payload["data_sources"]["external_providers"]["authority"] == "disabled_until_explicit_data_phase"
    assert payload["financial_calls"]["place_order"] == 0


def _bar_request() -> HistoricalBarRequest:
    return HistoricalBarRequest(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
        source=BarDataSource.EODHD,
        start=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        end=datetime(2026, 7, 21, 13, 30, tzinfo=UTC),
    )


def _bar(
    *,
    timestamp: datetime = datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
    interval: BarInterval = BarInterval.FIFTEEN_MINUTES,
    source: BarDataSource = BarDataSource.IBKR,
    high: Decimal = Decimal("101"),
    available_at: datetime | None = None,
) -> HistoricalBar:
    availability_lag = timedelta(minutes=15) if interval == BarInterval.FIFTEEN_MINUTES else timedelta(hours=1)
    if interval == BarInterval.ONE_DAY:
        availability_lag = timedelta(0)
    return HistoricalBar(
        con_id=265598,
        security_type=IbkrSecurityType.STK,
        interval=interval,
        data_type=BarDataType.TRADES,
        source=source,
        timestamp_utc=timestamp,
        open=Decimal("100"),
        high=high,
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=1_000_000,
        available_at=available_at or timestamp + availability_lag,
    )


def _cached_stock_row() -> ContractCacheRow:
    return ContractCacheRow(
        contract=ResolvedContract(
            con_id=265598,
            symbol="AAPL",
            local_symbol="AAPL",
            security_type=IbkrSecurityType.STK,
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
            trading_class="NMS",
            min_tick=Decimal("0.01"),
            valid_exchanges="SMART,NASDAQ",
            market_rule_ids="26",
            long_name="Apple Inc.",
            industry="Technology",
            category="Computers",
            subcategory="Computer Hardware",
            time_zone_id="US/Eastern",
            trading_hours="20260720:0930-1600;20260721:CLOSED;20260722:0930-1600",
            liquid_hours="20260720:0930-1600;20260721:CLOSED;20260722:0930-1600",
        ),
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=225,
    )


class _FakeIbkrBar:
    def __init__(
        self,
        date: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        wap: float,
        bar_count: int,
    ) -> None:
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.wap = wap
        self.barCount = bar_count


def _cached_future_row(*, con_id: int = 999001) -> ContractCacheRow:
    return ContractCacheRow(
        contract=ResolvedContract(
            con_id=con_id,
            symbol="GC",
            local_symbol="GCZ6",
            security_type=IbkrSecurityType.FUT,
            exchange="COMEX",
            primary_exchange=None,
            currency="USD",
            trading_class="GC",
            multiplier=Decimal("100"),
            min_tick=Decimal("0.1"),
            valid_exchanges=None,
            market_rule_ids="26",
            expiry="202612",
            last_trade_date_or_contract_month="20261228",
            real_expiration_date="20261228",
            last_trade_time="12:30:00",
            under_con_id=778899,
            time_zone_id="US/Central",
            trading_hours="20260720:1800-20260721:1700;20260721:1800-20260722:1700",
            liquid_hours="20260720:1800-20260721:1700;20260721:1800-20260722:1700",
        ),
        resolved_at=datetime(2026, 7, 20, tzinfo=UTC),
        server_version=225,
    )
