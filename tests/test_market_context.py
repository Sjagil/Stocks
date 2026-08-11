from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from main import build_parser
from stocks.data.multitimeframe import MultiTimeframeLayout
from stocks.market.context import (
    MarketContextLayout,
    audit_market_context_sources,
    build_market_context,
    load_market_context_map,
    market_context_status,
)
from stocks.microstructure.orderflow import (
    aggregate_trade_flow,
    bar_flow_proxy,
    classify_trades,
    orderbook_metrics,
)
from stocks.options.gex import (
    GexInputError,
    adapt_external_gex_snapshot,
    calculate_gex_snapshot,
)


AS_OF = pd.Timestamp("2026-07-31T18:00:00Z")


def _chain() -> pd.DataFrame:
    expiry = AS_OF + pd.Timedelta(days=30)
    return pd.DataFrame(
        [
            {
                "expiration": expiry,
                "option_type": "call",
                "strike": 95.0,
                "open_interest": 500,
                "implied_volatility": 0.25,
                "gamma": 0.020,
            },
            {
                "expiration": expiry,
                "option_type": "call",
                "strike": 105.0,
                "open_interest": 900,
                "implied_volatility": 0.27,
                "gamma": 0.018,
            },
            {
                "expiration": expiry,
                "option_type": "put",
                "strike": 95.0,
                "open_interest": 800,
                "implied_volatility": 0.29,
                "gamma": 0.019,
            },
            {
                "expiration": expiry,
                "option_type": "put",
                "strike": 90.0,
                "open_interest": 300,
                "implied_volatility": 0.31,
                "gamma": 0.012,
            },
        ]
    )


def test_gex_snapshot_uses_dollar_gamma_formula_and_labels_proxy() -> None:
    summary, profile, scenario = calculate_gex_snapshot(
        _chain(),
        symbol="ABC",
        spot=100.0,
        as_of=AS_OF,
        observed_at=AS_OF,
        source="TEST_CHAIN",
    )
    expected = (
        0.020 * 500 * 100 * 100**2 * 0.01
        + 0.018 * 900 * 100 * 100**2 * 0.01
        - 0.019 * 800 * 100 * 100**2 * 0.01
        - 0.012 * 300 * 100 * 100**2 * 0.01
    )
    assert summary["net_gex_1pct"] == pytest.approx(expected)
    assert summary["dealer_position_observed"] is False
    assert summary["authority"] == "CONTEXT_ONLY"
    assert summary["execution_authority"] == "NONE"
    assert summary["confidence"] <= 0.70
    assert summary["call_wall"] == 105.0
    assert summary["put_wall"] == 95.0
    assert len(profile) == 3
    assert len(scenario) == 161


def test_gex_snapshot_estimates_gamma_when_provider_gamma_absent() -> None:
    chain = _chain().drop(columns="gamma")
    summary, _, _ = calculate_gex_snapshot(
        chain,
        symbol="ABC",
        spot=100.0,
        as_of=AS_OF,
        observed_at=AS_OF,
        source="TEST_CHAIN",
    )
    assert summary["option_rows"] == 4
    assert summary["gamma_provenance_counts"] == {
        "BLACK_SCHOLES_ESTIMATED": 4
    }


def test_gex_snapshot_blocks_empty_or_invalid_inputs() -> None:
    with pytest.raises(GexInputError):
        calculate_gex_snapshot(
            pd.DataFrame(),
            symbol="ABC",
            spot=100.0,
            as_of=AS_OF,
            source="TEST",
        )
    with pytest.raises(GexInputError):
        calculate_gex_snapshot(
            _chain(),
            symbol="ABC",
            spot=0.0,
            as_of=AS_OF,
            source="TEST",
        )


def test_external_legacy_gex_is_not_relabelled_as_dollar_gex() -> None:
    payload = {
        "generated_at": AS_OF.isoformat(),
        "equity_etf_underlyings": {
            "ABC": {
                "GEX": -123.0,
                "open_interest": 5000,
                "source_id": "CURRENT_CHAIN_NOT_PIT",
                "gamma_walls": {
                    "positive_gamma_wall": {"strike": 110},
                    "negative_gamma_wall": {"strike": 90},
                },
            }
        },
    }
    row = adapt_external_gex_snapshot(
        payload,
        observed_at=AS_OF + pd.Timedelta(hours=1),
    )[0]
    assert row["net_gex_legacy"] == -123.0
    assert row["net_gex_1pct"] is None
    assert row["confidence"] == 0.35
    assert "LEGACY_GEX_UNIT_NOT_COMPARABLE_TO_DOLLAR_GEX_1PCT" in row[
        "warnings"
    ]


def test_trade_classification_uses_quote_then_tick_rule() -> None:
    trades = pd.DataFrame(
        [
            {
                "timestamp": "2026-07-31T10:00:00Z",
                "price": 100.1,
                "size": 10,
                "bid": 100.0,
                "ask": 100.1,
            },
            {
                "timestamp": "2026-07-31T10:00:01Z",
                "price": 100.05,
                "size": 5,
                "bid": 100.0,
                "ask": 100.1,
            },
            {
                "timestamp": "2026-07-31T10:00:02Z",
                "price": 99.9,
                "size": 8,
                "bid": 99.9,
                "ask": 100.0,
            },
        ]
    )
    classified = classify_trades(trades)
    assert classified["aggressor_side"].tolist() == [1, -1, -1]
    assert classified["classification_source"].tolist() == [
        "QUOTE_RULE_ASK",
        "TICK_RULE_DOWN",
        "QUOTE_RULE_BID",
    ]


def test_trade_flow_aggregates_delta_cvd_and_confidence() -> None:
    trades = pd.DataFrame(
        [
            {
                "timestamp": "2026-07-31T10:00:00Z",
                "price": 100.1,
                "size": 10,
                "bid": 100.0,
                "ask": 100.1,
            },
            {
                "timestamp": "2026-07-31T10:30:00Z",
                "price": 99.9,
                "size": 4,
                "bid": 99.9,
                "ask": 100.0,
            },
        ]
    )
    flow = aggregate_trade_flow(trades, interval="1h")
    assert len(flow) == 1
    assert flow.iloc[0]["buy_volume"] == 10
    assert flow.iloc[0]["sell_volume"] == 4
    assert flow.iloc[0]["delta"] == 6
    assert flow.iloc[0]["cvd"] == 6
    assert flow.iloc[0]["data_class"] == "OBSERVED_TRADE_FLOW"
    assert flow.iloc[0]["standalone_entry_authority"] == 0


def test_orderbook_metrics_compute_weighted_imbalance_and_microprice() -> None:
    levels = pd.DataFrame(
        [
            {
                "level": 1,
                "bid_price": 99.9,
                "bid_size": 200,
                "ask_price": 100.1,
                "ask_size": 100,
            },
            {
                "level": 2,
                "bid_price": 99.8,
                "bid_size": 100,
                "ask_price": 100.2,
                "ask_size": 100,
            },
        ]
    )
    result = orderbook_metrics(levels)
    assert result["status"] == "ORDERBOOK_CONTEXT_AVAILABLE"
    assert result["obi"] > 0
    assert result["microprice"] > result["midpoint"]
    assert result["standalone_entry_authority"] is False


def test_bar_flow_proxy_is_never_claimed_as_observed_orderflow() -> None:
    index = pd.date_range("2026-07-30T13:30:00Z", periods=8, freq="1h")
    bars = pd.DataFrame(
        {
            "open": [100 + i for i in range(8)],
            "high": [101 + i for i in range(8)],
            "low": [99 + i for i in range(8)],
            "close": [100.5 + i for i in range(8)],
            "volume": [1000 + 100 * i for i in range(8)],
        },
        index=index,
    )
    proxy = bar_flow_proxy(
        bars,
        symbol="ABC",
        interval="1h",
        source="TEST_BARS",
    )
    assert set(proxy["data_class"]) == {
        "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW"
    }
    assert proxy["confidence"].max() <= 0.35
    assert not proxy["observed_aggressor_volume"].any()
    assert not proxy["observed_orderbook"].any()
    assert not proxy["standalone_entry_authority"].any()


def test_source_audit_does_not_map_crypto_flow_to_equities(tmp_path: Path) -> None:
    external = tmp_path / "datascraper"
    micro = external / "data" / "microstructure"
    micro.mkdir(parents=True)
    (micro / "bitvavo_trade_events_v10.jsonl").write_text("{}\n")
    (micro / "bitvavo_book_events_v10.jsonl").write_text("{}\n")
    report = audit_market_context_sources(
        tmp_path,
        datascraper_root=external,
        observed_at=AS_OF.to_pydatetime(),
    )
    assert report["summary"]["crypto_observed_microstructure_available"]
    assert not report["summary"]["equity_observed_trade_flow_available"]
    assert report["hard_truth"]["crypto_flow_can_be_reused_for_equities"] is False


def test_source_audit_uses_latest_file_for_directory_freshness(
    tmp_path: Path,
) -> None:
    private = (
        tmp_path / "data" / "research" / "multitimeframe" / "private"
    )
    private.mkdir(parents=True)
    current = private / "current.parquet"
    current.write_bytes(b"current")
    timestamp = AS_OF.timestamp()
    os.utime(current, (timestamp, timestamp))
    report = audit_market_context_sources(
        tmp_path,
        datascraper_root=tmp_path / "missing-datascraper",
        observed_at=AS_OF.to_pydatetime(),
    )
    source = next(
        row
        for row in report["sources"]
        if row["source_id"] == "STOCKS_MULTITIMEFRAME_BARS"
    )
    assert source["age_hours"] == 0.0
    assert source["file_count"] == 1
    assert source["bytes"] == len(b"current")


def test_build_context_adapts_nested_sector_etf_snapshot(
    tmp_path: Path,
) -> None:
    external = tmp_path / "datascraper"
    report_path = (
        external
        / "output"
        / "worldmonitor_data_plane_launcher"
        / "v4638_etf_greeks_enrichment.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at": AS_OF.isoformat(),
                "computed_greeks_snapshot": {
                    "sector_etf_underlyings": {
                        "XLK": {
                            "GEX": -39658.9,
                            "open_interest": 278989.0,
                            "iv_skew_25d": 0.026,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    status = build_market_context(
        tmp_path,
        symbols=["XLK"],
        fetch_options=False,
        datascraper_root=external,
        observed_at=AS_OF.to_pydatetime(),
    )
    assert status["gex"]["available_symbol_count"] == 1
    context = json.loads(
        MarketContextLayout(tmp_path).gex_json.read_text(encoding="utf-8")
    )["contexts"][0]
    assert context["symbol"] == "XLK"
    assert context["net_gex_legacy"] == pytest.approx(-39658.9)
    assert context["net_gex_1pct"] is None
    assert context["confidence"] == 0.35


def test_build_context_uses_external_gex_and_bar_proxy_offline(
    tmp_path: Path,
) -> None:
    external = tmp_path / "datascraper"
    report_path = (
        external
        / "output"
        / "worldmonitor_data_plane_launcher"
        / "v4638_yfinance_ibkr_options_fetch.json"
    )
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at": AS_OF.isoformat(),
                "equity_etf_underlyings": {
                    "AAPL": {
                        "GEX": 100.0,
                        "open_interest": 5000,
                        "source_id": "YFINANCE_CURRENT_NOT_PIT",
                        "gamma_walls": {
                            "positive_gamma_wall": {"strike": 220},
                            "negative_gamma_wall": {"strike": 190},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    mtf = MultiTimeframeLayout(tmp_path)
    timestamps = {
        "1h": "2026-07-31T16:30:00Z",
        "2h": "2026-07-31T15:30:00Z",
        "4h": "2026-07-31T13:30:00Z",
    }
    for interval, timestamp in timestamps.items():
        frame = _stored_bar_frame(timestamp, interval)
        path = mtf.bars_path(
            provider="YFINANCE",
            symbol="AAPL",
            interval=interval,
            source_interval="1h",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    status = build_market_context(
        tmp_path,
        symbols=["AAPL"],
        fetch_options=False,
        datascraper_root=external,
        observed_at=AS_OF.to_pydatetime(),
    )
    assert status["status"] == "GO"
    assert status["gex"]["available_symbol_count"] == 1
    assert status["orderflow"]["proxy_row_count"] == 18
    assert not status["orderflow"]["observed_trade_flow_available"]
    assert status["execution_authority"] == "NONE"
    layout = MarketContextLayout(tmp_path)
    assert layout.orderflow_parquet.exists()
    current = market_context_status(tmp_path)
    assert current["artifact_integrity"]["gex_context_exists"]


def test_market_context_cli_is_bounded_and_explicit() -> None:
    parser = build_parser()
    parsed = parser.parse_args(
        [
            "market",
            "context",
            "build",
            "--symbols",
            "AAPL,SPY",
            "--max-expirations",
            "3",
            "--no-network",
        ]
    )
    assert parsed.market_context_command == "build"
    assert parsed.max_expirations == 3
    assert parsed.no_network is True


def test_public_context_loader_shrinks_proxy_scores_and_flags_wall(
    tmp_path: Path,
) -> None:
    layout = MarketContextLayout(tmp_path)
    layout.output_root.mkdir(parents=True)
    layout.gex_json.write_text(
        json.dumps(
            {
                "contexts": [
                    {
                        "symbol": "ABC",
                        "status": "AVAILABLE_CONTEXT_ONLY",
                        "source": "TEST_CURRENT_CHAIN",
                        "source_mode": "CURRENT_CHAIN_NOT_PIT",
                        "as_of": AS_OF.isoformat(),
                        "spot": 100.0,
                        "call_wall": 101.0,
                        "put_wall": 95.0,
                        "gamma_flip": 98.0,
                        "net_gex_1pct": 1_000_000.0,
                        "regime_proxy": "POSITIVE_GEX_PROXY",
                        "gex_concentration_top3": 0.8,
                        "near_expiry_gex_concentration": 0.2,
                        "confidence": 0.7,
                        "dealer_position_observed": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rows = []
    for interval in ("1h", "2h", "4h"):
        rows.append(
            {
                "timestamp_utc": AS_OF,
                "symbol": "ABC",
                "interval": interval,
                "source": "TEST_BARS",
                "data_class": "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW",
                "bar_flow_score": 1.0,
                "confidence": 0.3,
                "observed_aggressor_volume": False,
                "observed_orderbook": False,
            }
        )
    pd.DataFrame(rows).to_parquet(layout.orderflow_parquet, index=False)
    context = load_market_context_map(tmp_path)["ABC"]
    assert context["orderflow"]["raw_score"] == 1.0
    assert context["orderflow"]["ranking_score"] == 0.65
    assert context["gex"]["ranking_score"] == 0.325
    assert "CONCENTRATED_CALL_WALL_NEAR_SPOT" in context["advisories"]
    assert context["standalone_entry_authority"] is False
    assert context["execution_authority"] == "NONE"


def _stored_bar_frame(timestamp: str, interval: str) -> pd.DataFrame:
    end = pd.Timestamp(timestamp)
    start = end - pd.Timedelta(hours=5)
    index = pd.date_range(start, end, periods=6)
    return pd.DataFrame(
        {
            "timestamp_utc": index,
            "open": [100.0 + i for i in range(6)],
            "high": [101.0 + i for i in range(6)],
            "low": [99.0 + i for i in range(6)],
            "close": [100.5 + i for i in range(6)],
            "volume": [1000.0 + 10 * i for i in range(6)],
            "interval": interval,
            "provider": "YFINANCE",
            "is_partial": False,
        }
    )
