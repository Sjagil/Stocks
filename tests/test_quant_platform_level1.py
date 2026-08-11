from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import httpx

from stocks.quant_platform import (
    CANONICAL_MARKET_DATA_COLUMNS,
    AssetClass,
    BitvavoAdapter,
    CoinMarketCapAdapter,
    EodhdAdapter,
    FredAdapter,
    MultiAssetMarketDataExplorer,
    MultiAssetStore,
    OpenExchangeRatesAdapter,
    PerformanceRiskAnalyzer,
    clean_market_data,
    resample_market_data,
)


RECEIVED = "2026-01-05T12:00:00Z"


def _row(
    symbol: str,
    timestamp: str,
    close: float,
    *,
    asset_class: AssetClass = AssetClass.EQUITY,
    source: str = "TEST",
    available_at: str | None = None,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 100.0,
        "asset_class": asset_class,
        "currency": "USD",
        "source": source,
        "available_at": available_at or timestamp,
        "market_cap": None,
    }


def test_eodhd_normalizes_equity_etf_and_commodity_proxy() -> None:
    payload = [
        {
            "date": "2026-01-02",
            "open": 10,
            "high": 12,
            "low": 9,
            "close": 11,
            "adjusted_close": 10.5,
            "volume": 1_000,
        }
    ]
    adapter = EodhdAdapter()
    for asset_class in (AssetClass.EQUITY, AssetClass.ETF, AssetClass.COMMODITY):
        frame = adapter.normalize(
            payload,
            symbol="GLD",
            asset_class=asset_class,
            currency="usd",
            received_at=RECEIVED,
        )
        assert tuple(frame.columns) == CANONICAL_MARKET_DATA_COLUMNS
        assert frame.iloc[0]["asset_class"] == asset_class.value
        assert frame.iloc[0]["close"] == pytest.approx(10.5)


def test_fred_preserves_release_availability_and_negative_macro_values() -> None:
    frame = FredAdapter().normalize(
        {
            "observations": [
                {"date": "2025-12-01", "value": "-0.25", "realtime_start": "2026-01-03"},
                {"date": "2026-01-01", "value": ".", "realtime_start": "2026-01-03"},
            ]
        },
        series_id="T10Y2Y",
        received_at=RECEIVED,
    )
    assert len(frame) == 1
    assert frame.iloc[0]["close"] == pytest.approx(-0.25)
    assert frame.iloc[0]["asset_class"] == "macro"
    assert frame.iloc[0]["available_at"] == pd.Timestamp("2026-01-03", tz="UTC")


def test_fx_bitvavo_and_coinmarketcap_adapters() -> None:
    fx = OpenExchangeRatesAdapter().normalize(
        {"base": "USD", "timestamp": 1_767_268_800, "rates": {"EUR": 0.92}},
        quote_currency="EUR",
        received_at=RECEIVED,
    )
    crypto = BitvavoAdapter().normalize(
        [[1_767_268_800_000, "90000", "92000", "89000", "91000", "2.5"]],
        market="BTC-EUR",
        received_at=RECEIVED,
    )
    caps = CoinMarketCapAdapter().normalize(
        {
            "data": [
                {
                    "symbol": "BTC",
                    "quote": {
                        "USD": {
                            "price": 100_000,
                            "volume_24h": 50_000_000,
                            "market_cap": 2_000_000_000_000,
                            "last_updated": "2026-01-05T10:00:00Z",
                        }
                    },
                }
            ]
        },
        received_at=RECEIVED,
    )
    assert fx.iloc[0]["symbol"] == "USDEUR"
    assert crypto.iloc[0]["symbol"] == "BTCEUR"
    assert caps.iloc[0]["market_cap"] == pytest.approx(2_000_000_000_000)


def test_cleaner_rejects_bad_ohlc_and_future_leakage() -> None:
    bad_ohlc = _row("SPY", "2026-01-02", 10)
    bad_ohlc["high"] = 9
    with pytest.raises(ValueError, match="invalid canonical"):
        clean_market_data([bad_ohlc])
    future_leak = _row(
        "SPY",
        "2026-01-02T12:00:00Z",
        10,
        available_at="2026-01-02T11:59:59Z",
    )
    with pytest.raises(ValueError, match="invalid canonical"):
        clean_market_data([future_leak])


def test_cleaner_retains_distinct_point_in_time_revisions() -> None:
    rows = [
        _row("GDP", "2025-10-01", 1.0, asset_class=AssetClass.MACRO, available_at="2025-11-01"),
        _row("GDP", "2025-10-01", 1.2, asset_class=AssetClass.MACRO, available_at="2025-12-01"),
    ]
    frame = clean_market_data(rows)
    assert frame["close"].tolist() == [1.0, 1.2]


def test_resampling_uses_ohlcv_semantics_and_latest_known_vintage() -> None:
    rows = [
        {**_row("SPY", "2026-01-01", 10), "open": 9, "high": 11, "low": 8},
        {**_row("SPY", "2026-01-02", 12), "open": 10, "high": 13, "low": 9},
    ]
    weekly = resample_market_data(clean_market_data(rows), "W")
    assert len(weekly) == 1
    assert weekly.iloc[0]["open"] == pytest.approx(9)
    assert weekly.iloc[0]["high"] == pytest.approx(13)
    assert weekly.iloc[0]["low"] == pytest.approx(8)
    assert weekly.iloc[0]["close"] == pytest.approx(12)
    assert weekly.iloc[0]["volume"] == pytest.approx(200)


def test_parquet_sqlite_store_supports_as_of_vintages(tmp_path) -> None:
    store = MultiAssetStore(tmp_path / "lake")
    frame = clean_market_data(
        [
            _row("GDP", "2025-10-01", 1.0, asset_class=AssetClass.MACRO, available_at="2025-11-01"),
            _row("GDP", "2025-10-01", 1.2, asset_class=AssetClass.MACRO, available_at="2025-12-01"),
            _row("SPY", "2025-10-01", 100.0),
        ]
    )
    manifest = store.write(frame)
    early = store.read(symbol="GDP", as_of="2025-11-15")
    current = store.read(symbol="GDP")
    assert manifest["broker_writes"] == 0
    assert len(manifest["parquet_files"]) == 2
    assert early["close"].tolist() == [1.0]
    assert current["close"].tolist() == [1.0, 1.2]


def test_returns_rolling_statistics_and_first_period_drawdown() -> None:
    analyzer = PerformanceRiskAnalyzer(periods_per_year=252)
    prices = pd.Series(
        [100.0, 90.0, 99.0],
        index=pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
    )
    assert analyzer.returns(prices).tolist() == pytest.approx([-0.1, 0.1])
    assert analyzer.returns(prices, logarithmic=True).sum() == pytest.approx(math.log(0.99))
    rolling = analyzer.rolling_statistics(prices, window=2)
    assert rolling.iloc[0]["rolling_drawdown"] == pytest.approx(-0.1)


def test_performance_analyzer_produces_complete_risk_distribution_metrics() -> None:
    index = pd.date_range("2025-01-01", periods=300, freq="B", tz="UTC")
    asset_returns = np.tile([0.01, -0.006, 0.004, -0.002, 0.003], 60)
    benchmark_returns = asset_returns * 0.5
    prices = pd.Series(100 * np.cumprod(np.r_[1.0, 1.0 + asset_returns])[:-1], index=index)
    benchmark = pd.Series(100 * np.cumprod(np.r_[1.0, 1.0 + benchmark_returns])[:-1], index=index)
    report = PerformanceRiskAnalyzer().analyze(prices, benchmark_prices=benchmark)
    required = {
        "total_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "maximum_drawdown",
        "historical_var",
        "historical_cvar",
        "best_day",
        "worst_day",
        "best_month",
        "worst_month",
        "skewness",
        "kurtosis",
        "beta",
        "alpha",
        "correlation",
    }
    assert required <= report.keys()
    assert report["beta"] == pytest.approx(2.0, rel=0.02)
    assert report["historical_cvar"] <= report["historical_var"]
    assert report["maximum_drawdown"] < 0


def test_explorer_integrates_ingestion_query_and_analysis_without_execution(tmp_path) -> None:
    explorer = MultiAssetMarketDataExplorer(tmp_path / "quant")
    dates = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    rows = [_row("SPY", date.isoformat(), 100 + index) for index, date in enumerate(dates)]
    rows += [_row("GLD", date.isoformat(), 80 + index * 0.25, asset_class=AssetClass.ETF) for index, date in enumerate(dates)]
    explorer.ingest(clean_market_data(rows))
    report = explorer.analyze("GLD", benchmark_symbol="SPY")
    observations = explorer.observations(asset_class="etf")
    assert len(observations.frame) == 40
    assert report["schema"] == "performance_risk_report_v1"
    assert report["execution_authority"] == "NONE"
    assert report["broker_calls"] == report["broker_writes"] == 0


def test_provider_fetch_routes_are_read_only_and_match_official_endpoints() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "stlouisfed" in request.url.host:
            return httpx.Response(200, json={"observations": []})
        if "eodhd" in request.url.host:
            return httpx.Response(200, json=[])
        if "openexchangerates" in request.url.host:
            return httpx.Response(200, json={"base": "USD", "timestamp": 1, "rates": {"EUR": 0.9}})
        if "bitvavo" in request.url.host:
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"data": {}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        EodhdAdapter().fetch(api_key="secret", ticker="SPY.US", start="2026-01-01", end="2026-01-31", client=client)
        FredAdapter().fetch(api_key="secret", series_id="VIXCLS", client=client)
        OpenExchangeRatesAdapter().fetch_historical(app_id="secret", date="2026-01-01", symbols=["EUR"], client=client)
        BitvavoAdapter().fetch(market="BTC-EUR", interval="1d", client=client)
        CoinMarketCapAdapter().fetch_latest(api_key="secret", symbols=["BTC"], client=client)
    assert [request.method for request in seen] == ["GET"] * 5
    assert seen[0].url.path == "/api/eod/SPY.US"
    assert seen[1].url.path == "/fred/series/observations"
    assert seen[2].url.path == "/api/historical/2026-01-01.json"
    assert seen[3].url.path == "/v2/BTC-EUR/candles"
    assert seen[4].url.path == "/v3/cryptocurrency/quotes/latest"
