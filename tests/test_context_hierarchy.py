from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from stocks.context import cot as cot_module
from stocks.context.cot import collect_cot_context
from stocks.context.entry_observer import observe_shortlist
from stocks.context.transmission import build_asset_context
from stocks.research.role_leaderboards import publish_role_leaderboards


NOW = datetime(2026, 8, 3, 15, 0, tzinfo=UTC)


def test_cot_context_applies_conservative_publication_lag(
    tmp_path: Path, monkeypatch
) -> None:
    old = {
        "cftc_contract_market_code": "088691",
        "contract_market_name": "GOLD",
        "commodity_name": "GOLD",
        "report_date_as_yyyy_mm_dd": "2026-07-21T00:00:00",
        "futonly_or_combined": "FutOnly",
        "open_interest_all": "1000",
        "m_money_positions_long_all": "400",
        "m_money_positions_short_all": "100",
        "prod_merc_positions_long": "100",
        "prod_merc_positions_short": "400",
    }
    future = {
        **old,
        "report_date_as_yyyy_mm_dd": "2026-08-04T00:00:00",
        "m_money_positions_long_all": "900",
    }

    def fake_fetch(
        endpoint: str, *, report_type: str, codes: list[str], start: str
    ) -> list[dict[str, str]]:
        del endpoint, codes, start
        return [old, future] if report_type == "DISAGGREGATED" else []

    monkeypatch.setattr(cot_module, "_fetch_rows", fake_fetch)
    report = collect_cot_context(tmp_path, observed_at=NOW)

    gold = next(row for row in report["contexts"] if row["market_id"] == "GOLD")
    assert gold["report_date"].startswith("2026-07-21")
    assert gold["available_at"].startswith("2026-07-24T21:00:00")
    assert report["execution_authority"] == "NONE"
    assert report["broker_calls"] == 0
    assert list(
        (tmp_path / "data/market_context/private/cot/snapshots").glob(
            "snapshot_id=*/history.parquet"
        )
    )


def test_asset_transmission_is_symbol_specific(tmp_path: Path) -> None:
    config = {
        "schema": "asset_context_transmission_v1",
        "version": "TEST",
        "groups": {
            "broad_equity": {
                "sensitivities": {"growth": 1.0},
                "cot_market": "US_EQUITY_INDEX",
            },
            "gold": {
                "sensitivities": {"growth": -1.0, "currency": 1.0},
                "cot_market": "GOLD",
            },
        },
        "symbols": {"SPY": "broad_equity", "GLD": "gold"},
    }
    _json(tmp_path / "config/context/asset_transmission_v1.json", config)
    _json(
        tmp_path / "output/macro/score.json",
        {
            "scores": {
                "growth": {"value": 80, "confidence": 1, "status": "VALID"},
                "currency": {"value": 40, "confidence": 1, "status": "VALID"},
            }
        },
    )
    _json(
        tmp_path / "output/market_context/cot-context.json",
        {
            "contexts": [
                {
                    "market_id": "GOLD",
                    "status": "CONTEXT_AVAILABLE",
                    "positioning_score": 0.5,
                    "confidence": 0.8,
                }
            ]
        },
    )
    _json(
        tmp_path / "output/notifications/market-intelligence-digest.json",
        {"important_news": [], "event_risk_within_24h": False},
    )

    report = build_asset_context(tmp_path, symbols=["SPY", "GLD"], observed_at=NOW)
    by_symbol = {row["symbol"]: row for row in report["contexts"]}

    assert by_symbol["SPY"]["asset_bias_score"] > 0
    assert by_symbol["GLD"]["asset_bias_score"] < by_symbol["SPY"]["asset_bias_score"]
    assert by_symbol["GLD"]["components"]["cot"]["market_id"] == "GOLD"
    assert report["execution_authority"] == "NONE"


def test_asset_transmission_maps_all_major_sectors_from_universe(
    tmp_path: Path,
) -> None:
    source_config = (
        Path(__file__).parents[1]
        / "config/context/asset_transmission_v1.json"
    )
    config = json.loads(source_config.read_text(encoding="utf-8"))
    config["symbols"] = {}
    _json(tmp_path / "config/context/asset_transmission_v1.json", config)
    scores = {
        family: {"value": 20, "confidence": 0.8, "status": "VALID"}
        for group in config["groups"].values()
        for family in group["sensitivities"]
    }
    _json(tmp_path / "output/macro/score.json", {"scores": scores})
    _json(
        tmp_path / "output/notifications/market-intelligence-digest.json",
        {"important_news": [], "event_risk_within_24h": False},
    )
    universe = tmp_path / "output/universe"
    universe.mkdir(parents=True)
    rows = [
        ("TECH", "Technology", "Software - Application"),
        ("FIN", "Financial Services", "Banks - Regional"),
        ("IND", "Industrials", "Specialty Industrial Machinery"),
        ("CYC", "Consumer Cyclical", "Specialty Retail"),
        ("HLTH", "Healthcare", "Medical Devices"),
        ("COMM", "Communication Services", "Telecom Services"),
        ("REIT", "Real Estate", "REIT - Office"),
        ("UTIL", "Utilities", "Utilities - Regulated Electric"),
        ("ENER", "Energy", "Oil & Gas E&P"),
        ("MAT", "Basic Materials", "Specialty Chemicals"),
        ("DEF", "Consumer Defensive", "Packaged Foods"),
    ]
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "sector": sector,
                "industry": industry,
                "region": "UNITED_STATES_LISTED",
                "sleeve": "stock",
                "active_listing": True,
            }
            for symbol, sector, industry in rows
        ]
    ).to_parquet(universe / "instruments.parquet", index=False)

    report = build_asset_context(
        tmp_path,
        symbols=[row[0] for row in rows],
        observed_at=NOW,
    )
    mapped = {
        row["symbol"]: row["transmission_group"]
        for row in report["contexts"]
    }

    assert mapped == {
        "COMM": "communication_equity",
        "CYC": "consumer_cyclical_equity",
        "DEF": "defensive_equity",
        "ENER": "energy",
        "FIN": "financial_equity",
        "HLTH": "healthcare_equity",
        "IND": "industrial_equity",
        "MAT": "materials_equity",
        "REIT": "real_estate_equity",
        "TECH": "technology_equity",
        "UTIL": "utility_equity",
    }
    assert report["mapping_source_counts"] == {
        "UNIVERSE_CLASSIFICATION_MAPPING": 11
    }


def test_entry_observer_requires_observed_tape_and_depth(tmp_path: Path) -> None:
    signal = _signal(NOW, "1h")
    _json(
        tmp_path / "output/signals/latest_signals.json",
        {
            "signals": [
                signal,
                _signal(NOW, "4h"),
                _signal(NOW, "1d"),
            ]
        },
    )
    _json(
        tmp_path / "output/market_context/asset-context.json",
        {
            "contexts": [
                {
                    "symbol": "AAA",
                    "asset_bias_score": 0.4,
                    "asset_bias_confidence": 0.8,
                    "bias_classification": "SUPPORTIVE",
                    "event_risk": {"blocks_new_entry": False},
                }
            ]
        },
    )
    private = tmp_path / "data/market_context/private"
    private.mkdir(parents=True)
    times = [NOW - timedelta(minutes=50 - index * 5) for index in range(10)]
    pd.DataFrame(
        {
            "symbol": "AAA",
            "timestamp": times,
            "price": [100 + index * 0.1 for index in range(10)],
            "size": 100,
            "bid": [99.99 + index * 0.1 for index in range(10)],
            "ask": [100.01 + index * 0.1 for index in range(10)],
        }
    ).to_parquet(private / "equity-trades.parquet", index=False)
    book_rows = []
    for timestamp in (NOW - timedelta(minutes=2), NOW - timedelta(minutes=1)):
        for level in (1, 2, 3):
            book_rows.append(
                {
                    "symbol": "AAA",
                    "timestamp": timestamp,
                    "level": level,
                    "bid_price": 100.8 - level * 0.01,
                    "ask_price": 100.8 + level * 0.01,
                    "bid_size": 200,
                    "ask_size": 100,
                }
            )
    pd.DataFrame(book_rows).to_parquet(
        private / "equity-orderbook.parquet", index=False
    )

    first = observe_shortlist(tmp_path, observed_at=NOW)
    second = observe_shortlist(
        tmp_path, observed_at=NOW + timedelta(minutes=1)
    )

    assert first["observations"][0]["state"] == "EXECUTION_CANDIDATE_AUTHORITY_NONE"
    assert first["observations"][0]["decision_contract"][
        "timeframe_hierarchy"
    ]["ready"]
    assert first["new_episode_count"] == 1
    assert second["new_episode_count"] == 0
    assert first["observations"][0]["schema"] == (
        "active_swing_forward_episode_v1"
    )
    assert first["observations"][0]["feature_snapshot_hash"]
    contract = first["observations"][0]["decision_contract"][
        "contract_identity"
    ]
    assert contract["status"] == "RESOLVED"
    assert contract["source"] == "SIGNAL_IMMUTABLE_SNAPSHOT"
    assert contract["con_id"] == 123
    assert contract["symbol"] == "AAA"
    assert contract["cache_fresh"] is True
    assert first["observations"][0]["setup_snapshot"][
        "contract_identity"
    ] == contract
    assert first["observations"][0]["proposed_order"] == {
        "mode": "HYPOTHETICAL_OBSERVATION_ONLY",
        "order_type": "LIMIT",
        "limit_price": "100.9",
        "quantity": 0,
        "transmit": False,
    }
    assert first["execution_authority"] == "NONE"
    assert first["automatic_orders"] == 0
    assert first["sec_standalone_entry_allowed"] is False
    assert first["observations"][0]["context_snapshot"][
        "sec_ranking_overlay"
    ]["authority"] == "RANKING_OVERLAY_ONLY"


def test_active_swing_profiles_are_asset_type_specific(
    tmp_path: Path,
) -> None:
    def signal(symbol: str, timeframe: str) -> dict[str, object]:
        row = _signal(NOW, timeframe)
        row.update(
            {
                "signal_id": f"SIG-{symbol}-{timeframe}",
                "ticker": symbol,
                "asset": symbol,
                "contract_identity": {
                    "con_id": len(symbol) * 100 + len(timeframe),
                    "symbol": symbol,
                },
            }
        )
        return row

    _json(
        tmp_path / "config/universes/broad_multi_asset_v1.json",
        {
            "groups": [
                {
                    "group": "EQUITY_ETFS",
                    "asset_type": "EQUITY_ETF",
                    "sleeve": "ETF_CORE",
                    "region": "UNITED_STATES",
                    "instruments": {"BBB": ["BROAD_MARKET", "UNITED_STATES"]},
                },
                {
                    "group": "COMMODITY_ETFS",
                    "asset_type": "COMMODITY_ETF",
                    "sleeve": "COMMODITY",
                    "region": "GLOBAL",
                    "instruments": {"CCC": ["GOLD", "GLOBAL"]},
                },
            ]
        },
    )
    _json(
        tmp_path / "output/signals/latest_signals.json",
        {
            "signals": [
                signal(symbol, timeframe)
                for symbol in ("AAA", "BBB", "CCC")
                for timeframe in ("1h", "4h", "1d")
            ]
        },
    )
    _json(
        tmp_path / "output/market_context/asset-context.json",
        {
            "contexts": [
                {
                    "symbol": symbol,
                    "asset_bias_score": 0.4,
                    "event_risk": {
                        "blocks_new_entry": False,
                        "risk_score": 0.0,
                    },
                    "components": {
                        "macro": {"status": "AVAILABLE", "score": 0.2},
                        "cot": {"status": "AVAILABLE", "score": 0.1},
                    },
                }
                for symbol in ("AAA", "BBB", "CCC")
            ]
        },
    )

    report = observe_shortlist(
        tmp_path, max_symbols=3, depth_symbols=3, observed_at=NOW
    )
    profiles = {
        row["symbol"]: row["decision_contract"]["asset_profile"]
        for row in report["observations"]
    }

    assert profiles["AAA"]["asset_class"] == "STOCK"
    assert profiles["BBB"]["asset_class"] == "ETF"
    assert profiles["CCC"]["asset_class"] == "COMMODITY_PROXY"
    assert "fair_value" in profiles["BBB"]["missing_components"]
    assert "curve" in profiles["CCC"]["missing_components"]
    assert all(
        not profile["bar_flow_proxy_used_as_orderflow"]
        for profile in profiles.values()
    )
    assert report["asset_profile_counts"] == {
        "COMMODITY_PROXY": 1,
        "ETF": 1,
        "STOCK": 1,
    }
    assert report["ml_status"] == (
        "NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS"
    )


def test_entry_observer_never_uses_bar_proxy_as_confirmation(
    tmp_path: Path,
) -> None:
    _json(
        tmp_path / "output/signals/latest_signals.json",
        {
            "signals": [
                _signal(NOW, "1h"),
                _signal(NOW, "4h"),
                _signal(NOW, "1d"),
            ]
        },
    )
    _json(
        tmp_path / "output/market_context/asset-context.json",
        {
            "contexts": [
                {
                    "symbol": "AAA",
                    "asset_bias_score": 0.5,
                    "event_risk": {"blocks_new_entry": False},
                }
            ]
        },
    )

    report = observe_shortlist(tmp_path, observed_at=NOW)
    row = report["observations"][0]

    assert row["state"] == "ENTRY_DATA_PENDING_OBSERVED_TAPE"
    assert not row["gates"]["bar_proxy_used_as_confirmation"]
    assert report["execution_authority"] == "NONE"


def test_one_hour_confirmation_requires_current_four_hour_setup(
    tmp_path: Path,
) -> None:
    _json(
        tmp_path / "output/signals/latest_signals.json",
        {"signals": [_signal(NOW, "1h"), _signal(NOW, "1d")]},
    )
    _json(
        tmp_path / "output/market_context/asset-context.json",
        {
            "contexts": [
                {
                    "symbol": "AAA",
                    "asset_bias_score": 0.5,
                    "event_risk": {"blocks_new_entry": False},
                }
            ]
        },
    )

    report = observe_shortlist(tmp_path, observed_at=NOW)
    row = report["observations"][0]

    assert row["state"] == "WATCHLIST_4H_SETUP_REQUIRED"
    assert not row["gates"]["timeframe_hierarchy_ready"]
    assert report["signal_funnel"]["timeframe_setup_ready"] == 0


def test_stale_original_setup_is_reported_as_hard_veto(
    tmp_path: Path,
) -> None:
    stale = _signal(NOW, "4h")
    stale.update(
        {
            "action": "AVOID",
            "original_action": "WATCHLIST",
            "data_freshness": "STALE",
            "price_validity_status": "CURRENT_MARKET_REFERENCE_UNAVAILABLE_OR_STALE",
        }
    )
    _json(
        tmp_path / "output/signals/latest_signals.json",
        {"signals": [stale]},
    )
    _json(
        tmp_path / "output/market_context/asset-context.json",
        {
            "contexts": [
                {
                    "symbol": "AAA",
                    "asset_bias_score": 0.5,
                    "event_risk": {"blocks_new_entry": False},
                }
            ]
        },
    )

    report = observe_shortlist(tmp_path, observed_at=NOW)
    row = report["observations"][0]

    assert row["state"] == "WATCHLIST_HARD_VETO_BLOCKED"
    assert "CURRENT_DATA_STALE" in row["decision_contract"]["hard_vetoes"]
    assert report["signal_funnel"]["hard_vetoed"] == 1


def test_unresolved_contract_is_frozen_and_remains_fail_closed(
    tmp_path: Path,
) -> None:
    signal = _signal(NOW, "4h")
    signal["contract_identity"] = {}
    _json(
        tmp_path / "output/signals/latest_signals.json",
        {"signals": [signal]},
    )
    _json(
        tmp_path / "output/market_context/asset-context.json",
        {
            "contexts": [
                {
                    "symbol": "AAA",
                    "asset_bias_score": 0.5,
                    "event_risk": {"blocks_new_entry": False},
                }
            ]
        },
    )

    report = observe_shortlist(tmp_path, observed_at=NOW)
    row = report["observations"][0]

    expected = {
        "status": "UNRESOLVED_BLOCKED",
        "source": "SIGNAL_IMMUTABLE_SNAPSHOT",
        "con_id": None,
        "resolved_at": None,
        "cache_age_seconds": None,
        "cache_fresh": False,
    }
    assert row["decision_contract"]["contract_identity"] == expected
    assert row["setup_snapshot"]["contract_identity"] == expected
    assert "CONTRACT_IDENTITY_REQUIRED" in row["decision_contract"][
        "hard_vetoes"
    ]
    assert row["decision_contract"]["research_observation_eligible"] is True
    assert row["decision_contract"]["research_observation_blockers"] == []
    assert (
        row["decision_contract"]["brokerability_status"]
        == "BROKERABILITY_BLOCKED_CONTRACT_IDENTITY"
    )
    assert row["decision_contract"]["standalone_entry_allowed"] is False
    assert row["gates"]["contract_resolved"] is False
    assert row["execution_authority"] == "NONE"
    assert row["broker_calls"] == 0
    assert report["contract_identity_status_counts"] == {
        "UNRESOLVED_BLOCKED": 1
    }


def test_role_leaderboards_do_not_compare_monthly_with_active_swing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output/research/phase11_12"
    root.mkdir(parents=True)
    pd.DataFrame(
        [
            _strategy("MONTHLY", "quality_trend", "1mo", 1.5),
            _strategy("FLOW", "flow_consensus", "1h", 1.2),
        ]
    ).to_parquet(root / "strategy-summary.parquet", index=False)
    tactical = tmp_path / "output/research/phase11_15"
    tactical.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "architecture": "daily_1h_pullback",
                "entry_strategy": "pullback",
                "lower_timeframe": "1h",
                "median_incremental_CAGR": -0.01,
                "median_incremental_pf": -0.02,
                "flow_improvement_ratio": 0.4,
                "incremental_flow_evidence_status": "NO_INCREMENTAL_EVIDENCE",
            }
        ]
    ).to_csv(tactical / "architecture-summary.csv", index=False)

    report = publish_role_leaderboards(tmp_path)
    strategic = json.loads(
        (tmp_path / "output/research/strategic_allocation/leaderboard.json").read_text()
    )
    active = json.loads(
        (tmp_path / "output/research/active_swing/leaderboard.json").read_text()
    )

    assert [row["strategy_id"] for row in strategic["leaderboard"]] == ["MONTHLY"]
    assert [row["strategy_id"] for row in active["leaderboard"]] == ["FLOW"]
    assert report["roles"]["tactical_execution"]["status"] == "NO_INCREMENTAL_OBSERVED_FLOW_EVIDENCE"
    assert report["execution_authority"] == "NONE"


def _signal(now: datetime, timeframe: str = "1h") -> dict[str, object]:
    return {
        "signal_id": f"SIG-AAA-{timeframe}",
        "ticker": "AAA",
        "asset": "AAA",
        "strategy_id": "S1",
        "timeframe": timeframe,
        "signal_timestamp": (now - timedelta(minutes=30)).isoformat(),
        "data_timestamp": (now - timedelta(hours=1)).isoformat(),
        "expiration_timestamp": (now + timedelta(hours=3)).isoformat(),
        "action": "WATCHLIST",
        "confidence_score": 0.9,
        "reasons": ["EMA_PULLBACK_ZONE", "TREND_FILTER"],
        "bar_closed": True,
        "data_freshness": "FRESH",
        "price_validity_status": "CURRENT_ENTRY_REFERENCE_GO",
        "contract_identity": {
            "con_id": 123,
            "symbol": "AAA",
            "resolved_at": (now - timedelta(hours=1)).isoformat(),
            "contract_hash": "A" * 64,
            "cache_status": "FRESH",
            "contract_source": "PHASE2_EXACT_STK_CACHE",
        },
        "current_market_price": "100.9",
        "entry_zone_low": "100",
        "entry_zone_high": "101",
        "stop_loss": "98",
        "stop_distance_pct": "0.02",
        "reward_risk_1": "2.0",
        "take_profit_1": "104",
        "take_profit_2": "106",
    }


def _strategy(
    strategy_id: str, formula: str, timeframe: str, sharpe: float
) -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "strategy_hash": strategy_id,
        "formula": formula,
        "timeframe": timeframe,
        "profile": "balanced",
        "asset_class": "STOCK",
        "status": "COMPLETE",
        "CAGR": 0.2,
        "Sharpe": sharpe,
        "period_profit_factor": 1.4,
        "maximum_drawdown": -0.2,
        "fill_count": 100,
        "stress_50bps_profit_factor": 1.2,
        "economic_outcome_fingerprint": strategy_id,
    }


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
