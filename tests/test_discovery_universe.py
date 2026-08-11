from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocks.universe import (
    discovery_universe_command,
    rank_universe_dimension,
    refresh_discovery_universe,
)


def _universe_fixture(root: Path) -> None:
    master = root / "data" / "research" / "phase11_4" / "private"
    master.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "security_id": "SEC-A",
                "ticker": "AAA",
                "name": "Alpha",
                "exchange": "NASDAQ",
                "category": "Domestic",
                "is_delisted": False,
                "sector": "Technology",
                "industry": "Software",
                "currency": "USD",
            },
            {
                "security_id": "SEC-B",
                "ticker": "BBB",
                "name": "Beta",
                "exchange": "NYSE",
                "category": "Domestic",
                "is_delisted": True,
                "sector": "Energy",
                "industry": "Oil",
                "currency": "USD",
            },
        ]
    ).to_parquet(master / "security-master.parquet", index=False)

    universe = root / "config" / "universes"
    universe.mkdir(parents=True)
    (universe / "broad_multi_asset_v1.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "group": "regional_equity",
                        "asset_type": "ETF",
                        "sleeve": "etf_core",
                        "region": "UNITED_STATES",
                        "instruments": {"SPY": "BROAD_MARKET"},
                    },
                    {
                        "group": "commodity",
                        "asset_type": "COMMODITY_ETF",
                        "sleeve": "commodity_security",
                        "region": "GLOBAL",
                        "instruments": {"GLD": "GOLD"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    signals = root / "output" / "signals"
    signals.mkdir(parents=True)
    (signals / "latest_signals.json").write_text(
        json.dumps({"signals": [{"ticker": "AAA"}, {"ticker": "SPY"}]}),
        encoding="utf-8",
    )
    portfolio = root / "output" / "portfolio"
    portfolio.mkdir(parents=True)
    (portfolio / "opportunity_ranking.json").write_text(
        json.dumps(
            {
                "opportunities": [
                    {
                        "ticker": "AAA",
                        "opportunity_score": 0.8,
                        "shariah_status": "SHARIAH_ELIGIBLE_PIT",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    live = root / "output" / "ibkr" / "live"
    live.mkdir(parents=True)
    (live / "strategy-allowlist.json").write_text(
        json.dumps(
            {
                "strategies": [
                    {
                        "status": "PIT_LIVE_ALLOWLISTED",
                        "allowed_symbols": ["AAA"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_discovery_universe_preserves_delisted_and_authority_boundaries(
    tmp_path: Path,
) -> None:
    _universe_fixture(tmp_path)

    report = refresh_discovery_universe(tmp_path)
    frame = pd.read_parquet(
        tmp_path / "output" / "universe" / "instruments.parquet"
    )

    assert report["status"] == "GO"
    assert report["discovery_instrument_count"] == 4
    assert report["delisted_count"] == 1
    assert report["stock_count"] == 2
    assert report["etf_count"] == 1
    assert report["commodity_exposure_count"] == 1
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0
    delisted = frame.loc[frame["symbol"] == "BBB"].iloc[0]
    assert delisted["eligibility_status"] == "DELISTED_BLOCKED"
    live = frame.loc[frame["symbol"] == "AAA"].iloc[0]
    assert bool(live["live_executable"]) is True
    commodity = frame.loc[frame["symbol"] == "GLD"].iloc[0]
    assert commodity["exposure_type"] == "GOLD_PRICE_EXPOSURE"


def test_discovery_universe_commands_and_rankings(
    tmp_path: Path,
) -> None:
    _universe_fixture(tmp_path)
    refresh_discovery_universe(tmp_path)

    stocks = discovery_universe_command(tmp_path, "stocks")
    commodities = discovery_universe_command(tmp_path, "commodities")
    industries = discovery_universe_command(tmp_path, "industries")
    sectors = rank_universe_dimension(tmp_path, "sector")

    assert stocks["count"] == 2
    assert commodities["count"] == 1
    assert industries["status"] == "GO"
    assert industries["dimension"] == "industry"
    assert sectors["status"] == "GO"
    assert sectors["rankings"][0]["sector"] == "Technology"
    assert sectors["rankings"][0]["maximum_opportunity_score"] == 0.8
    assert sectors["execution_authority"] == "NONE"


def test_discovery_universe_tags_known_commodity_producer(
    tmp_path: Path,
) -> None:
    _universe_fixture(tmp_path)
    master_path = (
        tmp_path
        / "data"
        / "research"
        / "phase11_4"
        / "private"
        / "security-master.parquet"
    )
    frame = pd.read_parquet(master_path)
    producer = frame.iloc[[0]].copy()
    producer.loc[:, "security_id"] = "SEC-FCX"
    producer.loc[:, "ticker"] = "FCX"
    producer.loc[:, "name"] = "Freeport McMoRan"
    producer.loc[:, "sector"] = "Basic Materials"
    producer.loc[:, "industry"] = "Copper"
    pd.concat([frame, producer], ignore_index=True).to_parquet(
        master_path, index=False
    )

    refresh_discovery_universe(tmp_path)
    result = pd.read_parquet(
        tmp_path / "output" / "universe" / "instruments.parquet"
    )
    row = result.loc[result["symbol"] == "FCX"].iloc[0]

    assert row["exposure_type"] == "COMMODITY_PRODUCER_EQUITY"
    assert row["commodity_exposure_type"] == "COPPER_PRODUCER"
    assert row["underlying_commodity"] == "COPPER"
    assert row["eligibility_status"] == "RESEARCH_ONLY"
