from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal

import main
from stocks.domain.assets import IbkrSecurityType
from stocks.ibkr.contract_cache import ContractCacheLayout, ContractCacheRow, write_contract_cache_rows
from stocks.ibkr.contracts import ResolvedContract
from stocks.market.sessions import (
    SessionCacheLayout,
    build_session_cache_from_contract_rows,
    market_next_open_report,
    market_session_schema_manifest,
    market_sessions_report,
    market_status_report,
    session_hash,
    validate_session_cache,
)


def test_market_status_report_reads_cached_trading_hours() -> None:
    row = _cached_stock_row()

    report = market_status_report(row, at=datetime.fromisoformat("2026-07-20T14:00:00+00:00"))

    assert report["schema"] == "market_status_v1"
    assert report["status"] == "GO"
    assert report["source"] == "local_contract_cache"
    assert report["market_state"] == "OPEN"
    assert report["is_trading_open"] is True
    assert report["is_liquid_open"] is True
    assert report["active_trading_window"]["start"] == "2026-07-20T09:30:00-04:00"
    assert report["known_trading_window_count"] == 2
    assert report["known_trading_coverage_end"] == "2026-07-22T16:00:00-04:00"
    assert report["financial_calls"]["place_order"] == 0


def test_market_cli_status_reads_local_cache_without_tws(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    write_contract_cache_rows(ContractCacheLayout.from_project_root(tmp_path), [_cached_stock_row()])

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "market",
            "status",
            "--con-id",
            "265598",
            "--at",
            "2026-07-20T14:00:00Z",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "market_status_v1"
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert payload["contract"]["conId"] == 265598
    assert payload["market_state"] == "OPEN"
    assert payload["financial_calls"]["place_order"] == 0


def test_market_cli_next_open_uses_next_cached_window(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    write_contract_cache_rows(ContractCacheLayout.from_project_root(tmp_path), [_cached_stock_row()])

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "market",
            "next-open",
            "--con-id",
            "265598",
            "--at",
            "2026-07-20T12:00:00+00:00",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "market_next_open_v1"
    assert payload["next_trading_open"]["start"] == "2026-07-20T09:30:00-04:00"
    assert payload["reason"] is None
    assert payload["financial_calls"]["place_order"] == 0


def test_market_next_open_reports_no_known_window_after_cached_horizon() -> None:
    row = _cached_stock_row()

    report = market_next_open_report(row, at=datetime.fromisoformat("2026-07-23T12:00:00+00:00"))

    assert report["schema"] == "market_next_open_v1"
    assert report["status"] == "NO_KNOWN_NEXT_OPEN"
    assert report["next_trading_open"] is None
    assert report["reason"] == "no future trading window in local contract cache"
    assert report["known_trading_window_count"] == 2
    assert report["known_trading_coverage_end"] == "2026-07-22T16:00:00-04:00"
    assert report["financial_calls"]["place_order"] == 0


def test_market_status_report_handles_overnight_future_session() -> None:
    row = _cached_future_row()

    report = market_status_report(row, at=datetime.fromisoformat("2026-07-21T03:30:00+00:00"))

    assert report["schema"] == "market_status_v1"
    assert report["status"] == "GO"
    assert report["contract"]["secType"] == "FUT"
    assert report["market_state"] == "OPEN"
    assert report["is_trading_open"] is True
    assert report["is_liquid_open"] is True
    assert report["active_trading_window"]["session_date"] == "2026-07-20"
    assert report["active_trading_window"]["start"] == "2026-07-20T18:00:00-05:00"
    assert report["active_trading_window"]["end"] == "2026-07-21T17:00:00-05:00"
    assert report["known_trading_coverage_start"] == "2026-07-20T18:00:00-05:00"
    assert report["known_trading_coverage_end"] == "2026-07-22T17:00:00-05:00"
    assert report["financial_calls"]["place_order"] == 0


def test_market_next_open_skips_active_overnight_future_window() -> None:
    row = _cached_future_row()

    report = market_next_open_report(row, at=datetime.fromisoformat("2026-07-21T15:00:00+00:00"))

    assert report["schema"] == "market_next_open_v1"
    assert report["status"] == "GO"
    assert report["next_trading_open"]["session_date"] == "2026-07-21"
    assert report["next_trading_open"]["start"] == "2026-07-21T18:00:00-05:00"
    assert report["next_trading_open"]["end"] == "2026-07-22T17:00:00-05:00"
    assert report["known_trading_window_count"] == 2
    assert report["financial_calls"]["place_order"] == 0


def test_market_sessions_report_keeps_overnight_window_on_ibkr_session_date() -> None:
    row = _cached_future_row()

    report = market_sessions_report(row, session_date=date(2026, 7, 20))

    assert report["schema"] == "market_sessions_v1"
    assert report["contract"]["secType"] == "FUT"
    assert report["session_date"] == "2026-07-20"
    assert report["is_trading_day"] is True
    assert report["is_liquid_day"] is True
    assert report["trading_closed"] is False
    assert report["trading_windows"] == [
        {
            "session_date": "2026-07-20",
            "start": "2026-07-20T18:00:00-05:00",
            "end": "2026-07-21T17:00:00-05:00",
            "source": "tradingHours",
        }
    ]
    assert report["financial_calls"]["place_order"] == 0


def test_market_cli_sessions_lists_cached_windows_for_date(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    write_contract_cache_rows(ContractCacheLayout.from_project_root(tmp_path), [_cached_stock_row()])

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "market",
            "sessions",
            "--con-id",
            "265598",
            "--date",
            "2026-07-21",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "market_sessions_v1"
    assert payload["session_date"] == "2026-07-21"
    assert payload["is_trading_day"] is False
    assert payload["is_liquid_day"] is False
    assert payload["trading_closed"] is True
    assert payload["liquid_closed"] is True
    assert payload["trading_windows"] == []
    assert payload["liquid_windows"] == []
    assert payload["calendar_check"]["status"] == "GO"
    assert payload["calendar_check"]["calendar_code"] == "XNAS"
    assert payload["calendar_check"]["calendar_is_session"] is True
    assert payload["calendar_check"]["alignment"] == "MISMATCH"
    assert payload["calendar_check"]["financial_calls"]["place_order"] == 0


def test_market_cli_sessions_without_con_id_lists_all_cached_contracts_for_date(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    write_contract_cache_rows(
        ContractCacheLayout.from_project_root(tmp_path),
        [
            _cached_stock_row(),
            _cached_stock_row(con_id=272093, symbol="MSFT", long_name="Microsoft Corporation"),
        ],
    )

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "market",
            "sessions",
            "--date",
            "2026-07-20",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "market_sessions_by_date_v1"
    assert payload["source"] == "local_contract_cache"
    assert payload["session_date"] == "2026-07-20"
    assert payload["contract_count"] == 2
    assert [item["contract"]["symbol"] for item in payload["contracts"]] == ["AAPL", "MSFT"]
    assert all(item["is_trading_day"] is True for item in payload["contracts"])
    assert all(item["trading_closed"] is False for item in payload["contracts"])
    assert all(item["trading_windows"] for item in payload["contracts"])
    assert all(item["calendar_check"]["alignment"] == "MATCH" for item in payload["contracts"])
    assert payload["financial_calls"]["place_order"] == 0


def test_market_sessions_report_matches_supported_exchange_calendar() -> None:
    row = _cached_stock_row()

    report = market_sessions_report(row, session_date=date(2026, 7, 20))

    assert report["calendar_check"]["schema"] == "market_calendar_session_check_v1"
    assert report["calendar_check"]["status"] == "GO"
    assert report["calendar_check"]["calendar_code"] == "XNAS"
    assert report["calendar_check"]["calendar_name"] == "XNYS"
    assert report["calendar_check"]["calendar_timezone"] == "America/New_York"
    assert report["calendar_check"]["calendar_is_session"] is True
    assert report["calendar_check"]["ibkr_has_trading_window"] is True
    assert report["calendar_check"]["alignment"] == "MATCH"
    assert report["calendar_check"]["session_open_utc"] == "2026-07-20T13:30:00+00:00"
    assert report["calendar_check"]["session_close_utc"] == "2026-07-20T20:00:00+00:00"
    assert report["calendar_check"]["financial_calls"]["place_order"] == 0


def test_market_sessions_report_marks_unsupported_exchange_calendar() -> None:
    row = _cached_stock_row(primary_exchange="FOO", valid_exchanges="SMART,FOO")

    report = market_sessions_report(row, session_date=date(2026, 7, 20))

    assert report["calendar_check"]["status"] == "UNSUPPORTED_CALENDAR"
    assert report["calendar_check"]["calendar_code"] is None
    assert report["calendar_check"]["calendar_is_session"] is None
    assert report["calendar_check"]["ibkr_has_trading_window"] is True
    assert report["calendar_check"]["alignment"] == "UNKNOWN"
    assert report["calendar_check"]["reason"] == "no supported exchange-calendar mapping for contract"
    assert report["calendar_check"]["financial_calls"]["place_order"] == 0


def test_market_cli_returns_not_found_for_missing_cached_contract(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "market",
            "status",
            "--con-id",
            "265598",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["schema"] == "market_command_v1"
    assert payload["status"] == "NOT_FOUND"
    assert payload["source"] == "local_contract_cache"
    assert payload["financial_calls"]["place_order"] == 0


def test_market_cli_status_blocks_nonpositive_con_id(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "market",
            "status",
            "--con-id",
            "0",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["schema"] == "market_command_v1"
    assert payload["status"] == "VALIDATION_ERROR"
    assert payload["reason"] == "con_id must be positive"
    assert payload["financial_calls"]["place_order"] == 0


def test_market_cli_sessions_filter_blocks_negative_con_id(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "market",
            "sessions",
            "--con-id",
            "-1",
            "--date",
            "2026-07-20",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["schema"] == "market_command_v1"
    assert payload["status"] == "VALIDATION_ERROR"
    assert payload["reason"] == "con_id must be positive"
    assert payload["financial_calls"]["place_order"] == 0


def test_phase3_market_session_schema_declares_canonical_model() -> None:
    schema = market_session_schema_manifest()

    assert schema["schema"] == "market_session_schema_v1"
    assert schema["primary_key"] == ["con_id", "session_date", "session_type"]
    assert "session_hash" in schema["required_fields"]
    assert "OPEN_LIQUID" in schema["statuses"]
    assert schema["financial_calls"]["place_order"] == 0
    assert schema["market_data_calls"] == 0
    assert schema["historical_data_calls"] == 0


def test_market_cli_nested_sessions_schema_does_not_require_contract_cache(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "market", "sessions", "schema"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "market_session_schema_v1"
    assert payload["market_data_calls"] == 0
    assert payload["historical_data_calls"] == 0


def test_market_cli_nested_sessions_resolve_returns_canonical_session(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    write_contract_cache_rows(ContractCacheLayout.from_project_root(tmp_path), [_cached_stock_row()])

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "market",
            "sessions",
            "resolve",
            "--con-id",
            "265598",
            "--date",
            "2026-07-20",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "market_sessions_v1"
    assert payload["session"]["con_id"] == 265598
    assert payload["session"]["session_open_utc"] == "2026-07-20T13:30:00+00:00"
    assert payload["session"]["liquid_close_utc"] == "2026-07-20T20:00:00+00:00"
    assert payload["session"]["session_hash"]
    assert payload["financial_calls"]["place_order"] == 0


def test_aapl_weekend_and_us_holiday_are_closed() -> None:
    row = _cached_stock_row(
        trading_hours="20260704:CLOSED;20260705:CLOSED",
        liquid_hours="20260704:CLOSED;20260705:CLOSED",
    )

    holiday = market_sessions_report(row, session_date=date(2026, 7, 4))
    weekend = market_sessions_report(row, session_date=date(2026, 7, 5))

    assert holiday["session"]["is_holiday"] is True
    assert holiday["session_status"] == "HOLIDAY"
    assert weekend["session"]["is_holiday"] is True
    assert weekend["session_status"] == "HOLIDAY"


def test_aapl_early_close_is_reported_from_exchange_calendar() -> None:
    row = _cached_stock_row(
        trading_hours="20261224:0400-2000",
        liquid_hours="20261224:0930-1300",
    )

    report = market_sessions_report(row, session_date=date(2026, 12, 24))

    assert report["status"] == "GO"
    assert report["session_status"] == "EARLY_CLOSE"
    assert report["session"]["is_early_close"] is True
    assert report["session"]["liquid_close_utc"] == "2026-12-24T18:00:00+00:00"


def test_asml_european_dst_session_uses_amsterdam_timezone() -> None:
    row = _cached_stock_row(
        con_id=117589399,
        symbol="ASML",
        currency="EUR",
        primary_exchange="AEB",
        valid_exchanges="SMART,AEB",
        time_zone_id="MET",
        trading_hours="20260330:0730-2300",
        liquid_hours="20260330:0900-1740",
    )

    report = market_sessions_report(row, session_date=date(2026, 3, 30))

    assert report["status"] == "GO"
    assert report["session"]["timezone_id"] == "MET"
    assert report["session"]["liquid_open_utc"] == "2026-03-30T07:00:00+00:00"
    assert report["calendar_check"]["calendar_code"] == "XAMS"
    assert report["calendar_conflicts"]


def test_spy_etf_session_is_resolved_as_stk() -> None:
    row = _cached_stock_row(
        con_id=756733,
        symbol="SPY",
        primary_exchange="ARCA",
        valid_exchanges="SMART,ARCA",
    )

    report = market_sessions_report(row, session_date=date(2026, 7, 20))

    assert report["status"] == "GO"
    assert report["contract"]["secType"] == "STK"
    assert report["session"]["security_type"] == "STK"
    assert report["calendar_check"]["calendar_code"] == "ARCX"


def test_invalid_timezone_returns_explicit_phase3_status() -> None:
    row = _cached_stock_row(time_zone_id="BAD/ZONE")

    report = market_sessions_report(row, session_date=date(2026, 7, 20))

    assert report["status"] == "TIMEZONE_INVALID"
    assert report["readiness"] == "SESSION_BLOCKED"
    assert report["session"] is None


def test_malformed_trading_and_liquid_hours_return_explicit_statuses() -> None:
    trading = market_sessions_report(
        _cached_stock_row(trading_hours="20260720"),
        session_date=date(2026, 7, 20),
    )
    liquid = market_sessions_report(
        _cached_stock_row(liquid_hours="20260720:0930"),
        session_date=date(2026, 7, 20),
    )

    assert trading["status"] == "HOURS_PARSE_ERROR"
    assert liquid["status"] == "HOURS_PARSE_ERROR"


def test_explicit_endpoint_dates_support_session_over_midnight() -> None:
    row = _cached_future_row(
        trading_hours="20260721:20260720:1800-20260721:1700",
        liquid_hours="20260721:20260720:1800-20260721:1700",
    )

    report = market_sessions_report(row, session_date=date(2026, 7, 21))

    assert report["status"] == "GO"
    assert report["session"]["is_overnight"] is True
    assert report["trading_windows"][0]["start"] == "2026-07-20T18:00:00-05:00"
    assert report["trading_windows"][0]["end"] == "2026-07-21T17:00:00-05:00"


def test_calendar_conflict_is_explicit_not_silent() -> None:
    row = _cached_stock_row(trading_hours="20260721:CLOSED", liquid_hours="20260721:CLOSED")

    report = market_sessions_report(row, session_date=date(2026, 7, 21))

    assert report["status"] == "GO"
    assert report["session_status"] == "CALENDAR_CONFLICT"
    assert report["readiness"] == "SESSION_DEGRADED"
    assert report["calendar_conflicts"][0]["type"] == "SESSION_DAY_MISMATCH"


def test_stale_contract_cache_blocks_session_resolution() -> None:
    row = _cached_stock_row(resolved_at=datetime(2026, 7, 1, tzinfo=UTC))

    report = market_sessions_report(row, session_date=date(2026, 7, 20))

    assert report["status"] == "STALE_CACHE"
    assert report["readiness"] == "SESSION_BLOCKED"
    assert report["financial_calls"]["place_order"] == 0


def test_session_hash_is_deterministic() -> None:
    first = session_hash(
        con_id=265598,
        session_date=date(2026, 7, 20),
        session_open_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        session_close_utc=datetime(2026, 7, 21, 0, 0, tzinfo=UTC),
        liquid_open_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        liquid_close_utc=datetime(2026, 7, 20, 20, 0, tzinfo=UTC),
        timezone_id="US/Eastern",
    )
    second = session_hash(
        con_id=265598,
        session_date=date(2026, 7, 20),
        session_open_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        session_close_utc=datetime(2026, 7, 21, 0, 0, tzinfo=UTC),
        liquid_open_utc=datetime(2026, 7, 20, 13, 30, tzinfo=UTC),
        liquid_close_utc=datetime(2026, 7, 20, 20, 0, tzinfo=UTC),
        timezone_id="US/Eastern",
    )

    assert first == second
    assert len(first) == 64


def test_session_cache_build_and_validate_records_cache_hits(tmp_path) -> None:
    layout = SessionCacheLayout.from_project_root(tmp_path)
    rows = [
        _cached_stock_row(),
        _cached_stock_row(
            con_id=756733,
            symbol="SPY",
            primary_exchange="ARCA",
            valid_exchanges="SMART,ARCA",
        ),
    ]

    build_report = build_session_cache_from_contract_rows(layout, rows)
    validation = validate_session_cache(layout, contract_rows=rows)

    assert build_report["status"] == "GO"
    assert validation["status"] == "GO"
    assert validation["file_count"] == 4
    assert validation["row_count"] > 0
    assert validation["duplicate_rows"] == 0
    assert validation["timezone_errors"] == 0
    assert validation["contract_mismatches"] == 0
    assert validation["content_hash"]
    assert validation["financial_calls"]["place_order"] == 0


def _cached_stock_row(
    *,
    con_id: int = 265598,
    symbol: str = "AAPL",
    currency: str = "USD",
    long_name: str = "Apple Inc.",
    primary_exchange: str = "NASDAQ",
    valid_exchanges: str = "SMART,NASDAQ",
    time_zone_id: str = "US/Eastern",
    trading_hours: str = "20260720:0930-1600;20260721:CLOSED;20260722:0930-1600",
    liquid_hours: str = "20260720:0930-1600;20260721:CLOSED;20260722:0930-1600",
    resolved_at: datetime = datetime(2099, 1, 1, tzinfo=UTC),
) -> ContractCacheRow:
    return ContractCacheRow(
        contract=ResolvedContract(
            con_id=con_id,
            symbol=symbol,
            local_symbol=symbol,
            security_type=IbkrSecurityType.STK,
            exchange="SMART",
            primary_exchange=primary_exchange,
            currency=currency,
            trading_class="NMS",
            min_tick=Decimal("0.01"),
            valid_exchanges=valid_exchanges,
            market_rule_ids="26",
            long_name=long_name,
            industry="Technology",
            category="Computers",
            subcategory="Computer Hardware",
            time_zone_id=time_zone_id,
            trading_hours=trading_hours,
            liquid_hours=liquid_hours,
        ),
        resolved_at=resolved_at,
        server_version=225,
    )


def _cached_future_row(
    *,
    trading_hours: str = "20260720:1800-20260721:1700;20260721:1800-20260722:1700",
    liquid_hours: str = "20260720:1800-20260721:1700;20260721:1800-20260722:1700",
    resolved_at: datetime = datetime(2099, 1, 1, tzinfo=UTC),
) -> ContractCacheRow:
    return ContractCacheRow(
        contract=ResolvedContract(
            con_id=999001,
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
            first_notice_day=date(2026, 11, 30),
            last_trade_day=date(2026, 12, 28),
            delivery_type="PHYSICAL",
            settlement_type="DELIVERY",
            contract_size_unit="100_TROY_OUNCES",
            roll_group="GC",
            time_zone_id="US/Central",
            trading_hours=trading_hours,
            liquid_hours=liquid_hours,
        ),
        resolved_at=resolved_at,
        server_version=225,
    )
