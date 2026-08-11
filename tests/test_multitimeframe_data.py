from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from stocks.data.multitimeframe import (
    MultiTimeframeLayout,
    _closed_intraday_bars,
    aggregate_bars,
    audit_multitimeframe_sources,
    bar_freshness,
    canonical_interval,
    collect_multitimeframe_data,
    multitimeframe_schema,
    parse_intervals,
    parse_symbols,
    provider_inventory,
    validate_multitimeframe_cache,
)


def _bars(interval: str, periods: int, frequency: str) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-05 14:30:00+00:00", periods=periods, freq=frequency)
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "session_date": ["2026-01-05"] * periods,
            "symbol": ["SPY"] * periods,
            "provider": ["TEST"] * periods,
            "interval": [interval] * periods,
            "source_interval": [interval] * periods,
            "derivation": ["NATIVE_OR_SOURCE_CACHE"] * periods,
            "exchange_timezone": ["America/New_York"] * periods,
            "adjustment_mode": ["RAW"] * periods,
            "open": [100.0 + i for i in range(periods)],
            "high": [101.0 + i for i in range(periods)],
            "low": [99.0 + i for i in range(periods)],
            "close": [100.5 + i for i in range(periods)],
            "adjusted_close": [100.5 + i for i in range(periods)],
            "volume": [1000.0] * periods,
            "dividends": [0.0] * periods,
            "stock_splits": [0.0] * periods,
            "source_bar_count": [1] * periods,
            "is_partial": [False] * periods,
            "received_at": ["2026-01-06T00:00:00+00:00"] * periods,
            "row_hash": [str(i) for i in range(periods)],
        }
    )


def test_interval_alias_is_month_not_minute() -> None:
    assert canonical_interval("1m") == "1mo"
    assert parse_intervals("1h,2h,4h,6h,12h,1d,1w,1m") == [
        "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1mo"
    ]
    for forbidden in ("1min", "5m", "15m", "30m"):
        with pytest.raises(ValueError, match="FORBIDDEN_SWING_TIMEFRAME"):
            canonical_interval(forbidden)


def test_symbol_and_interval_requests_are_bounded() -> None:
    assert parse_symbols("spy,aapl,SPY") == ["AAPL", "SPY"]
    with pytest.raises(ValueError):
        parse_symbols([f"S{i}" for i in range(51)])


def test_session_anchored_four_hour_aggregation_preserves_partial_close_bucket() -> None:
    result = aggregate_bars(_bars("1h", 7, "1h"), target_interval="4h")
    assert len(result) == 2
    assert result.iloc[0]["source_bar_count"] == 4
    assert bool(result.iloc[0]["is_partial"]) is False
    assert result.iloc[1]["source_bar_count"] == 3
    assert bool(result.iloc[1]["is_partial"]) is True
    assert result.iloc[0]["open"] == 100.0
    assert result.iloc[0]["close"] == 103.5


def test_two_hour_aggregation_is_causal_and_complete() -> None:
    result = aggregate_bars(_bars("1h", 4, "1h"), target_interval="2h")
    assert len(result) == 2
    assert result["source_bar_count"].tolist() == [2, 2]
    assert result["timestamp_utc"].tolist() == [
        pd.Timestamp("2026-01-05 14:30:00+00:00"),
        pd.Timestamp("2026-01-05 16:30:00+00:00"),
    ]


def test_daily_to_weekly_and_monthly_uses_last_actual_close() -> None:
    daily = _bars("1d", 31, "1D")
    daily["session_date"] = daily["timestamp_utc"].dt.date.astype(str)
    weekly = aggregate_bars(daily, target_interval="1w")
    monthly = aggregate_bars(daily, target_interval="1mo")
    assert weekly.iloc[0]["open"] == 100.0
    assert weekly.iloc[0]["close"] == 104.5
    assert monthly.iloc[0]["close"] == 126.5


def test_schema_and_inventory_block_datascraper_fixture(tmp_path: Path) -> None:
    schema = multitimeframe_schema(tmp_path)
    inventory = provider_inventory(tmp_path, datascraper_root=tmp_path / "datascraper")
    assert schema["interval_aliases"]["1m"] == "1mo (one month)"
    fixture = next(row for row in inventory["sources"] if row["provider"] == "DATASCRAPER_HISTORICAL_CANDLE_SERVICE_V1")
    assert fixture["status"] == "BLOCKED_DETERMINISTIC_FIXTURE_STUB"
    assert inventory["secret_presence_only"] is True


def test_local_collection_derives_week_and_month_without_network(tmp_path: Path) -> None:
    cache = tmp_path / "data" / "research" / "critical_trading" / "yfinance"
    cache.mkdir(parents=True)
    source = cache / "SPY.parquet"
    daily = _bars("1d", 35, "1D").drop(columns="session_date").rename(columns={"timestamp_utc": "session_date"})
    daily[["session_date", "open", "high", "low", "close", "volume"]].to_parquet(source, index=False)
    (cache / "manifest.json").write_text(
        json.dumps({"instruments": [{"symbol": "SPY", "path": str(source)}]}), encoding="utf-8"
    )
    report = collect_multitimeframe_data(
        tmp_path,
        symbols=["SPY"],
        intervals=["1d", "1w", "1mo"],
        providers=["local"],
    )
    assert report["status"] == "GO"
    assert report["provider_calls_read_only"] == 0
    assert report["broker_calls"] == 0
    assert {row["interval"] for row in report["results"]} == {"1d", "1w", "1mo"}
    assert validate_multitimeframe_cache(tmp_path)["status"] == "GO"


def test_validation_rejects_duplicate_and_invalid_ohlc(tmp_path: Path) -> None:
    layout = MultiTimeframeLayout(tmp_path)
    path = layout.bars_path(provider="TEST", symbol="SPY", interval="1d", source_interval="1d")
    path.parent.mkdir(parents=True)
    frame = _bars("1d", 2, "1D")
    frame.loc[1, "timestamp_utc"] = frame.loc[0, "timestamp_utc"]
    frame.loc[1, "high"] = 1.0
    frame.to_parquet(path, index=False)
    report = validate_multitimeframe_cache(tmp_path)
    assert report["status"] == "NO_GO"
    assert report["duplicate_rows"] == 1
    assert report["invalid_ohlc_rows"] == 1


def test_intraday_freshness_and_closed_bar_gate_are_explicit() -> None:
    observed_at = datetime(2026, 1, 5, 17, 45, tzinfo=UTC)
    frame = _bars("1h", 2, "1h")
    frame["timestamp_utc"] = pd.to_datetime(
        ["2026-01-05T15:30:00Z", "2026-01-05T17:30:00Z"],
        utc=True,
    )

    closed = _closed_intraday_bars(
        frame,
        interval="1h",
        observed_at=observed_at,
    )

    assert closed["timestamp_utc"].tolist() == [
        pd.Timestamp("2026-01-05T15:30:00Z")
    ]
    assert bar_freshness(
        closed["timestamp_utc"].max(),
        interval="1h",
        observed_at=observed_at,
    )["status"] == "FRESH_CLOSED_BAR"
    assert bar_freshness(
        "2026-01-01T15:30:00Z",
        interval="1h",
        observed_at=observed_at,
    )["status"] == "STALE_BAR_BLOCKED"
    assert bar_freshness(
        "2026-01-08T19:30:00Z",
        interval="1h",
        observed_at=datetime(2026, 1, 9, 18, 0, tzinfo=UTC),
    )["status"] == "STALE_BAR_BLOCKED"
    assert bar_freshness(
        "2026-01-09T19:30:00Z",
        interval="1h",
        observed_at=datetime(2026, 1, 12, 13, 0, tzinfo=UTC),
    )["status"] == "FRESH_CLOSED_BAR"


def test_intraday_freshness_uses_latest_completed_exchange_session() -> None:
    weekend = bar_freshness(
        "2026-07-31T19:30:00Z",
        interval="1h",
        observed_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        exchange_timezone="America/New_York",
    )
    missing_friday = bar_freshness(
        "2026-07-30T19:30:00Z",
        interval="1h",
        observed_at=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        exchange_timezone="America/New_York",
    )
    active_session_stale = bar_freshness(
        "2026-07-31T19:30:00Z",
        interval="1h",
        observed_at=datetime(2026, 8, 3, 15, 0, tzinfo=UTC),
        exchange_timezone="America/New_York",
    )
    active_session_current = bar_freshness(
        "2026-08-03T13:30:00Z",
        interval="1h",
        observed_at=datetime(2026, 8, 3, 15, 0, tzinfo=UTC),
        exchange_timezone="America/New_York",
    )

    assert weekend["status"] == "FRESH_CLOSED_BAR"
    assert weekend["freshness_basis"] == "LATEST_COMPLETED_EXCHANGE_SESSION"
    assert weekend["latest_completed_session"] == "2026-07-31"
    assert missing_friday["status"] == "STALE_BAR_BLOCKED"
    assert active_session_stale["status"] == "STALE_BAR_BLOCKED"
    assert active_session_current["status"] == "FRESH_CLOSED_BAR"
    assert (
        active_session_current["freshness_basis"]
        == "ACTIVE_EXCHANGE_SESSION_WALL_CLOCK"
    )


def test_validation_selects_freshest_provider_without_blending(
    tmp_path: Path,
) -> None:
    layout = MultiTimeframeLayout(tmp_path)
    observed_at = datetime(2026, 1, 5, 18, 0, tzinfo=UTC)
    for provider, timestamp in (
        ("EODHD", "2025-12-31T16:30:00Z"),
        ("YFINANCE", "2026-01-05T16:30:00Z"),
    ):
        path = layout.bars_path(
            provider=provider,
            symbol="SPY",
            interval="1h",
            source_interval="1h",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = _bars("1h", 1, "1h")
        frame["provider"] = provider
        frame["timestamp_utc"] = pd.to_datetime([timestamp], utc=True)
        frame["session_date"] = frame["timestamp_utc"].dt.date.astype(str)
        frame.to_parquet(path, index=False)

    report = validate_multitimeframe_cache(
        tmp_path,
        as_of=observed_at,
    )
    coverage = report["coverage"][0]

    assert coverage["current_data_status"] == "FRESH_CLOSED_BAR"
    assert coverage["fresh_provider_count"] == 1
    assert coverage["selected_current_provider"] == "YFINANCE"
    assert coverage["selection_policy"] == (
        "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
    )


def test_cross_provider_audit_preserves_variants_and_reports_divergence(tmp_path: Path) -> None:
    layout = MultiTimeframeLayout(tmp_path)
    for provider, difference in (("LEFT", 0.0), ("RIGHT", 2.0)):
        path = layout.bars_path(provider=provider, symbol="SPY", interval="1d", source_interval="1d")
        path.parent.mkdir(parents=True)
        frame = _bars("1d", 10, "1D")
        frame["provider"] = provider
        frame["close"] += difference
        frame.to_parquet(path, index=False)
    report = audit_multitimeframe_sources(tmp_path)
    assert report["comparison_count"] == 1
    assert report["material_divergence_count"] == 1
    assert report["incompatible_adjustment_pair_count"] == 0
    assert report["selection_policy"]["silent_provider_blending"] is False


def test_privacy_audit_detects_real_credential_value_without_publishing_it(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("EODHD_API_KEY=very-secret-provider-token\n", encoding="utf-8")
    output = tmp_path / "output" / "research" / "multitimeframe"
    output.mkdir(parents=True)
    (output / "leaky.json").write_text('{"value":"very-secret-provider-token"}', encoding="utf-8")
    report = audit_multitimeframe_sources(tmp_path)
    privacy = json.loads((output / "privacy-audit.json").read_text(encoding="utf-8"))
    assert report["privacy_audit_status"] == "NO_GO"
    assert privacy["credential_value_leak_count"] == 1
    assert "very-secret-provider-token" not in json.dumps(privacy)


def test_no_broker_or_execution_methods_in_module() -> None:
    source = (Path(__file__).parents[1] / "src" / "stocks" / "data" / "multitimeframe.py").read_text(encoding="utf-8")
    forbidden = ("placeOrder", "cancelOrder", "reqGlobalCancel", "reqIds", "reqAutoOpenOrders", "exerciseOptions")
    assert all(token not in source for token in forbidden)


def test_privacy_audit_checks_project_env_when_process_value_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    process_secret = (
        "process-provider-token-"
        "must-not-mask-project"
    )
    project_secret = (
        "project-provider-token-"
        "must-be-detected"
    )

    monkeypatch.setenv(
        "EODHD_API_KEY",
        process_secret,
    )

    (tmp_path / ".env").write_text(
        (
            "EODHD_API_KEY="
            f"{project_secret}\n"
        ),
        encoding="utf-8",
    )

    output = (
        tmp_path
        / "output"
        / "research"
        / "multitimeframe"
    )

    output.mkdir(
        parents=True
    )

    (
        output
        / "leaky-project-secret.json"
    ).write_text(
        json.dumps(
            {
                "value": project_secret,
            }
        ),
        encoding="utf-8",
    )

    report = audit_multitimeframe_sources(
        tmp_path
    )

    privacy = json.loads(
        (
            output
            / "privacy-audit.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    serialized = json.dumps(
        privacy
    )

    assert (
        report["privacy_audit_status"]
        == "NO_GO"
    )

    assert project_secret not in serialized
    assert process_secret not in serialized

