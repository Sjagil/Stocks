from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks.quant_platform import (
    CrossSectionalFactorEngine,
    TechnicalStrategyLab,
    build_market_factor_snapshot,
    strategy_signals,
    technical_features,
)


def test_factor_ranking_is_point_in_time_and_weighted() -> None:
    snapshot = pd.DataFrame(
        [
            {"symbol": "AAA", "available_at": "2026-01-01", "momentum": 1, "quality": 1, "value": 1, "volatility": 0.4, "liquidity": 1},
            {"symbol": "AAA", "available_at": "2026-02-01", "momentum": -99, "quality": -99, "value": -99, "volatility": 9, "liquidity": -99},
            {"symbol": "BBB", "available_at": "2026-01-01", "momentum": 3, "quality": 2, "value": 2, "volatility": 0.2, "liquidity": 3},
            {"symbol": "CCC", "available_at": "2026-01-01", "momentum": 2, "quality": 3, "value": 3, "volatility": 0.1, "liquidity": 2},
        ]
    )
    ranked = CrossSectionalFactorEngine().rank(snapshot, as_of="2026-01-15")
    assert ranked.iloc[0]["symbol"] == "CCC"
    assert ranked["symbol"].tolist()[-1] == "AAA"
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_factor_engine_supports_sector_neutralization() -> None:
    snapshot = pd.DataFrame(
        [
            {"symbol": symbol, "sector": sector, "available_at": "2026-01-01", "momentum": value, "quality": value, "value": value, "volatility": 1 / value, "liquidity": value}
            for symbol, sector, value in [("A", "TECH", 4), ("B", "TECH", 2), ("C", "HEALTH", 3), ("D", "HEALTH", 1)]
        ]
    )
    ranked = CrossSectionalFactorEngine().rank(snapshot, as_of="2026-01-02", sector_neutral=True)
    assert set(ranked["symbol"]) == {"A", "B", "C", "D"}
    assert ranked.loc[ranked["symbol"] == "A", "score"].iloc[0] > ranked.loc[ranked["symbol"] == "B", "score"].iloc[0]


def test_market_factor_snapshot_derives_price_and_fundamental_factors() -> None:
    dates = pd.date_range("2024-01-01", periods=300, freq="B", tz="UTC")
    bars = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "timestamp": dates,
                    "available_at": dates,
                    "close": start * np.exp(np.arange(len(dates)) * slope),
                    "volume": volume,
                }
            )
            for symbol, start, slope, volume in [("AAA", 100, 0.002, 1_000_000), ("BBB", 100, 0.0005, 500_000)]
        ],
        ignore_index=True,
    )
    fundamentals = pd.DataFrame(
        [
            {"symbol": "AAA", "available_at": "2025-01-01", "earnings_yield": 0.08, "fcf_yield": 0.07, "ev_to_ebitda": 12, "price_to_book": 4, "roe": 0.30, "roic": 0.22, "gross_profitability": 0.40, "debt_to_equity": 0.2, "earnings_volatility": 0.1, "market_cap": 1e11},
            {"symbol": "BBB", "available_at": "2025-01-01", "earnings_yield": 0.05, "fcf_yield": 0.04, "ev_to_ebitda": 18, "price_to_book": 6, "roe": 0.15, "roic": 0.12, "gross_profitability": 0.25, "debt_to_equity": 0.6, "earnings_volatility": 0.2, "market_cap": 5e10},
        ]
    )
    factors = build_market_factor_snapshot(bars, as_of=dates[-1], fundamentals=fundamentals)
    assert set(factors["symbol"]) == {"AAA", "BBB"}
    aaa = factors.loc[factors["symbol"] == "AAA"].iloc[0]
    bbb = factors.loc[factors["symbol"] == "BBB"].iloc[0]
    assert aaa["momentum"] > bbb["momentum"]
    assert aaa["quality"] > bbb["quality"]
    assert aaa["value"] > bbb["value"]


def _bars(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(0.001 + rng.normal(0, 0.01, rows)))
    high = close * (1.0 + rng.uniform(0.001, 0.02, rows))
    low = close * (1.0 - rng.uniform(0.001, 0.02, rows))
    open_ = np.clip(close * (1.0 + rng.normal(0, 0.003, rows)), low, high)
    volume = rng.integers(100_000, 2_000_000, rows).astype(float)
    volume[-1] = volume[-20:-1].mean() * 3
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=rows, freq="B", tz="UTC"),
    )


def test_technical_features_cover_all_requested_families() -> None:
    features = technical_features(_bars())
    expected = {
        "sma_20", "ema_12", "donchian_high_20", "adx_14", "supertrend", "macd", "ichimoku_span_a",
        "rsi_14", "stochastic_k_14", "roc_12", "cci_20", "williams_r_14",
        "bollinger_upper", "price_zscore_20", "vwap_deviation", "atr_14", "bollinger_squeeze", "keltner_upper",
        "obv", "vpt", "cmf_20", "chaikin_oscillator", "volume_breakout",
    }
    assert expected <= set(features)
    assert features["rsi_14"].dropna().between(0, 100).all()
    assert features["adx_14"].dropna().between(0, 100).all()


def test_strategy_lab_uses_combinations_and_has_no_execution_authority() -> None:
    report = TechnicalStrategyLab().evaluate(_bars())
    signals = report["signals"]
    assert set(signals) == {
        "trend",
        "momentum",
        "mean_reversion",
        "breakout",
        "volatility_expansion",
        "volume_confirmation",
        "combined_trend_pullback",
    }
    assert all(pd.api.types.is_bool_dtype(dtype) for dtype in signals.dtypes)
    assert report["execution_authority"] == "NONE"
    assert report["broker_writes"] == 0


def test_signal_builder_rejects_unqualified_single_indicator_input() -> None:
    with pytest.raises(ValueError, match="missing technical features"):
        strategy_signals(pd.DataFrame({"rsi_14": [20.0]}))
