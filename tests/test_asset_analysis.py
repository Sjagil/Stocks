from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stocks.analysis import analyze_asset, build_analysis_coverage
from stocks.analysis.assets import (
    _effective_bar_timestamp,
    _load_bars,
    _model_order_plan,
)


def _universe(root: Path) -> None:
    path = root / "output" / "universe"
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "name": "Alpha Systems",
                "instrument_type": "STOCK",
                "asset_type": "STOCK",
                "exposure_type": "OPERATING_COMPANY_EQUITY",
                "sector": "Technology",
                "industry": "Software",
                "region": "UNITED_STATES",
                "country": "US",
                "currency": "USD",
                "exchange": "NASDAQ",
                "active_listing": True,
                "signal_eligible": True,
                "live_executable": False,
                "eligibility_status": "SIGNAL_ELIGIBLE",
                "compliance_status": "UNSCREENED",
            },
            {
                "symbol": "NODATA",
                "name": "No Data Asset",
                "instrument_type": "STOCK",
                "asset_type": "STOCK",
                "exposure_type": "OPERATING_COMPANY_EQUITY",
                "sector": "Industrials",
                "industry": "Testing",
                "region": "EUROPE",
                "country": "NL",
                "currency": "EUR",
                "exchange": "SMART",
                "active_listing": True,
                "signal_eligible": False,
                "live_executable": False,
                "eligibility_status": "RESEARCH_ONLY",
                "compliance_status": "UNSCREENED",
            },
        ]
    ).to_parquet(path / "instruments.parquet", index=False)


def _bars(
    root: Path,
    interval: str,
    periods: int = 260,
    *,
    current: bool = False,
) -> None:
    path = (
        root
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
        / "provider=TEST"
        / "symbol=AAA"
        / f"interval={interval}"
        / "source_interval=1h"
    )
    path.mkdir(parents=True)
    frequency = "1h" if interval == "1h" else "2h"
    index = (
        pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor(frequency),
            periods=periods,
            freq=frequency,
        )
        if current
        else pd.date_range(
            "2026-01-01",
            periods=periods,
            freq=frequency,
            tz="UTC",
        )
    )
    close = np.linspace(90.0, 120.0, periods) + np.sin(
        np.arange(periods) / 9.0
    )
    pd.DataFrame(
        {
            "timestamp_utc": index,
            "provider": "TEST",
            "bar_origin": (
                "NATIVE" if interval == "1h" else "DERIVED"
            ),
            "open": close * 0.998,
            "high": close * 1.008,
            "low": close * 0.992,
            "close": close,
            "volume": np.linspace(1000.0, 1500.0, periods),
            "is_partial": False,
        }
    ).to_parquet(path / "bars.parquet", index=False)


def _current_reference(root: Path, price: float = 100.0) -> None:
    path = (
        root
        / "data/research/multitimeframe/private"
        / "provider=YFINANCE"
        / "symbol=AAA"
        / "interval=1h"
        / "source_interval=1h"
    )
    path.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    pd.DataFrame(
        [
            {
                "timestamp_utc": now,
                "fetched_at": now,
                "symbol": "AAA",
                "provider": "YFINANCE",
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
                "is_partial": False,
            }
        ]
    ).to_parquet(path / "bars.parquet", index=False)


def test_asset_analysis_computes_diverse_1h_and_2h_indicators(
    tmp_path: Path,
) -> None:
    _universe(tmp_path)
    _bars(tmp_path, "1h")
    _bars(tmp_path, "2h")

    report = analyze_asset(tmp_path, "AAA")
    rows = {
        row["interval"]: row
        for row in report["multi_timeframe"]["timeframes"]
    }

    assert report["status"] == "GO"
    assert report["analysis_coverage"]["available_timeframes"] == [
        "1h",
        "2h",
    ]
    assert rows["1h"]["status"] == "GO"
    assert rows["2h"]["status"] == "GO"
    assert set(rows["1h"]["components"]) == {
        "trend",
        "momentum",
        "breakout",
        "volume_flow",
        "mean_reversion",
    }
    assert rows["1h"]["rsi_14"] is not None
    assert rows["2h"]["adx_14"] is not None
    assert report["current_market"]["state"] == (
        "CURRENT_REFERENCE_STALE_OR_UNAVAILABLE"
    )
    assert report["decision"]["recommended_action"] == (
        "REFRESH_INTRADAY_REFERENCE_BEFORE_NEW_DECISION"
    )
    assert report["decision"]["execution_authority"] == "NONE"
    assert report["microstructure_context"]["status"] == (
        "NO_CONTEXT_NEUTRAL_FALLBACK"
    )
    assert report["microstructure_context"][
        "standalone_entry_authority"
    ] is False
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0


def test_asset_analysis_prefers_sufficient_history_over_short_provider_file(
    tmp_path: Path,
) -> None:
    root = (
        tmp_path
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
    )
    short = (
        root
        / "provider=EODHD"
        / "symbol=TEST"
        / "interval=1w"
        / "source_interval=1d"
        / "bars.parquet"
    )
    long = (
        root
        / "provider=STOCKS_YFINANCE_LOCAL"
        / "symbol=TEST"
        / "interval=1w"
        / "source_interval=1d"
        / "bars.parquet"
    )
    short.parent.mkdir(parents=True)
    long.parent.mkdir(parents=True)

    def frame(periods: int) -> pd.DataFrame:
        close = np.linspace(90.0, 120.0, periods)
        return pd.DataFrame(
            {
                "timestamp_utc": pd.date_range(
                    "2020-01-01",
                    periods=periods,
                    freq="7D",
                    tz="UTC",
                ),
                "bar_origin": "DERIVED",
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0,
                "is_partial": False,
            }
        )

    frame(5).to_parquet(short, index=False)
    frame(80).to_parquet(long, index=False)

    selected, provenance = _load_bars(tmp_path, "TEST", "1w")

    assert len(selected) == 80
    assert provenance["provider"] == "STOCKS_YFINANCE_LOCAL"


def test_monthly_freshness_uses_closed_month_end() -> None:
    timestamp = pd.Timestamp("2026-06-01T00:00:00Z")

    effective = _effective_bar_timestamp(timestamp, "1mo")

    assert effective.isoformat() == "2026-06-30T00:00:00+00:00"


def test_every_universe_asset_gets_explicit_metadata_or_data_status(
    tmp_path: Path,
) -> None:
    _universe(tmp_path)

    report = analyze_asset(tmp_path, "NODATA")
    unknown = analyze_asset(tmp_path, "UNKNOWN")

    assert report["status"] == "DATA_UNAVAILABLE"
    assert report["metadata"]["name"] == "No Data Asset"
    assert report["multi_timeframe"]["classification"] == "DATA_UNAVAILABLE"
    assert unknown["status"] == "NOT_IN_UNIVERSE"


def test_asset_analysis_uses_news_as_context_and_external_search(
    tmp_path: Path,
) -> None:
    _universe(tmp_path)
    news = tmp_path / "output" / "notifications"
    news.mkdir(parents=True)
    (news / "market-intelligence-digest.json").write_text(
        json.dumps(
            {
                "news_freshness_status": "CURRENT_WITHIN_72H",
                "important_news": [
                    {
                        "title": "Alpha reports results",
                        "source": "TEST",
                        "published_at": "2026-01-02T00:00:00Z",
                        "importance": "HIGH",
                        "direction": "POSITIVE_INFERENCE",
                        "symbols": ["AAA"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = analyze_asset(tmp_path, "AAA")
    item = report["news"]["items"][0]

    assert report["news"]["matched_count"] == 1
    assert item["external_search_url"].startswith(
        "https://news.google.com/search?"
    )
    assert report["news"]["news_is_context_only"] is True
    assert report["decision"]["automatic_submission"] is False


def test_universe_analysis_coverage_counts_1h_and_2h(
    tmp_path: Path,
) -> None:
    _universe(tmp_path)
    _bars(tmp_path, "1h")
    _bars(tmp_path, "2h")

    report = build_analysis_coverage(tmp_path)

    assert report["status"] == "GO"
    assert report["universe_instrument_count"] == 2
    assert report["analyzable_instrument_count"] == 1
    assert report["one_hour_instrument_count"] == 1
    assert report["two_hour_instrument_count"] == 1
    assert report["execution_authority"] == "NONE"
    assert (tmp_path / "output" / "analysis" / "universe-coverage.json").is_file()


def test_model_order_plan_is_concrete_but_never_grants_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "output" / "signals"
    path.mkdir(parents=True)
    (path / "latest_signals.json").write_text(
        json.dumps(
            {
                "signals": [
                    {
                        "ticker": "AAA",
                        "data_freshness": "FRESH",
                        "confidence_score": 0.9,
                        "suggested_quantity": 2,
                        "limit_entry_price": 100.0,
                        "entry_zone_low": 99.0,
                        "entry_zone_high": 101.0,
                        "stop_loss": 95.0,
                        "take_profit_1": 110.0,
                        "take_profit_2": 115.0,
                        "reward_risk_1": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    buy = _model_order_plan(tmp_path, "AAA", "RESEARCH_CANDIDATE")
    sell = _model_order_plan(
        tmp_path,
        "AAA",
        "AVOID_NEW_LONG_RISK_REVIEW_EXISTING_POSITION",
    )

    assert buy["model_action"] == "BUY_SETUP"
    assert buy["quantity"] == 2
    assert buy["protective_stop"] == 95.0
    assert buy["status"] == "DRAFT_NOT_SUBMITTABLE_AUTHORITY_NONE"
    assert sell["model_action"] == "SELL_REVIEW_IF_HELD"
    assert sell["status"] == "BLOCKED_WITHOUT_POSITION_IDENTITY"
    assert buy["execution_authority"] == "NONE"
    assert sell["execution_authority"] == "NONE"


def test_asset_verdict_blocks_price_invalidated_signal(
    tmp_path: Path,
) -> None:
    _universe(tmp_path)
    _bars(tmp_path, "1h")
    _bars(tmp_path, "2h")
    _current_reference(tmp_path, 90.0)
    signals = tmp_path / "output/signals"
    signals.mkdir(parents=True)
    (signals / "latest_signals.json").write_text(
        json.dumps(
            {
                "signals": [
                    {
                        "ticker": "AAA",
                        "strategy_id": "OLD_ENTRY",
                        "original_action": "BUY",
                        "action": "AVOID",
                        "data_freshness": "STALE",
                        "lifecycle_status": "INVALIDATED",
                        "price_validity_status": (
                            "CURRENT_PRICE_BREACHED_SIGNAL_STOP"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_asset(tmp_path, "AAA")

    assert report["current_market"]["price"] == 90.0
    assert report["current_market"]["state"] == (
        "CURRENT_SIGNALS_INVALIDATED"
    )
    assert report["current_market"]["executable_quote"] is False
    assert report["decision"]["recommended_action"] == (
        "AVOID_NEW_LONG_WAIT_FOR_NEW_CAUSAL_SIGNAL"
    )
    assert report["decision"]["model_order_plan"]["model_action"] == (
        "AVOID_NEW_LONG"
    )
    assert report["decision"]["model_order_plan"]["status"] == (
        "CURRENT_PRICE_INVALIDATED_ENTRY"
    )
    assert report["orders_generated"] == 0


def test_asset_verdict_records_valid_signal_without_bypassing_mtf(
    tmp_path: Path,
) -> None:
    _universe(tmp_path)
    _bars(tmp_path, "1h", current=True)
    _bars(tmp_path, "2h", current=True)
    _current_reference(tmp_path, 120.0)
    portfolio = tmp_path / "output/portfolio"
    portfolio.mkdir(parents=True)
    (portfolio / "opportunity_ranking.json").write_text(
        json.dumps(
            {
                "opportunities": [
                    {
                        "ticker": "AAA",
                        "opportunity_score": 0.8,
                        "research_allocation_eligible": True,
                        "deployment_blockers": [
                            "EXECUTION_AUTHORITY_NONE"
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    signals = tmp_path / "output/signals"
    signals.mkdir(parents=True)
    (signals / "latest_signals.json").write_text(
        json.dumps(
            {
                "signals": [
                    {
                        "ticker": "AAA",
                        "strategy_id": "CURRENT_ENTRY",
                        "action": "BUY",
                        "data_freshness": "FRESH",
                        "price_validity_status": (
                            "CURRENT_ENTRY_REFERENCE_GO"
                        ),
                        "confidence_score": 0.9,
                        "suggested_quantity": 2,
                        "limit_entry_price": 120.0,
                        "entry_zone_low": 118.0,
                        "entry_zone_high": 121.0,
                        "stop_loss": 112.0,
                        "take_profit_1": 136.0,
                        "take_profit_2": 144.0,
                        "reward_risk_1": 2.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_asset(tmp_path, "AAA")

    assert report["current_market"]["state"] == (
        "CURRENT_ENTRY_SIGNAL_AVAILABLE"
    )
    assert report["current_market"]["valid_signal_count"] == 1
    assert report["decision"]["recommended_action"] == (
        "NO_NEW_POSITION_WAIT_FOR_ALIGNMENT"
    )
    assert report["decision"]["model_order_plan"]["model_action"] == (
        "NO_ORDER"
    )
    assert report["decision"]["model_order_plan"]["status"] == (
        "NO_CURRENT_ACTIONABLE_SETUP"
    )
    assert report["orders_generated"] == 0


def test_asset_verdict_blocks_signal_contract_currency_mismatch(
    tmp_path: Path,
) -> None:
    _universe(tmp_path)
    _bars(tmp_path, "1h", current=True)
    _current_reference(tmp_path, 120.0)
    portfolio = tmp_path / "output/portfolio"
    portfolio.mkdir(parents=True)
    (portfolio / "opportunity_ranking.json").write_text(
        json.dumps(
            {
                "opportunities": [
                    {
                        "ticker": "AAA",
                        "opportunity_score": 0.8,
                        "research_allocation_eligible": False,
                        "signal_contract_currency_status": (
                            "SIGNAL_CONTRACT_CURRENCY_MISMATCH"
                        ),
                        "signal_currency": "USD",
                        "contract_currency": "EUR",
                        "deployment_blockers": [
                            "SIGNAL_CONTRACT_CURRENCY_MISMATCH",
                            "EXECUTION_AUTHORITY_NONE",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_asset(tmp_path, "AAA")

    assert report["portfolio_context"][
        "signal_contract_currency_status"
    ] == "SIGNAL_CONTRACT_CURRENCY_MISMATCH"
    assert report["decision"]["recommended_action"] == (
        "BLOCKED_SIGNAL_CONTRACT_CURRENCY_REVIEW"
    )
    assert report["decision"]["model_order_plan"]["model_action"] == (
        "NO_ORDER"
    )
    assert "SIGNAL_CONTRACT_CURRENCY_MISMATCH" in report["decision"][
        "reason_codes"
    ]
    assert report["orders_generated"] == 0
