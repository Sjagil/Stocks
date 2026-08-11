from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import psutil
from fastapi.testclient import TestClient

from stocks.ui.app import create_app
from stocks.ui.runtime import _owned_server_pid, ui_command
from stocks.ui.service import (
    ViewModelStore,
    _exchange_clock,
    _period_performance,
)


def _write_json(root: Path, relative: str, payload: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ui_fixture(root: Path) -> None:
    today = pd.Timestamp.now(tz="Europe/Amsterdam").date()
    previous_day = today - pd.Timedelta(days=1)
    today_iso = today.isoformat()
    previous_day_iso = previous_day.isoformat()
    universe = root / "output" / "universe"
    universe.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "instrument_id": "SEC-A",
                "security_id": "SEC-A",
                "symbol": "AAA",
                "name": "Alpha Systems",
                "instrument_type": "STOCK",
                "asset_type": "STOCK",
                "category": "Domestic",
                "exposure_type": "OPERATING_COMPANY_EQUITY",
                "exchange": "NASDAQ",
                "currency": "USD",
                "region": "UNITED_STATES_LISTED",
                "country": "UNKNOWN_ISSUER_DOMICILE",
                "sector": "Technology",
                "industry": "Software",
                "sleeve": "stock",
                "active_listing": True,
                "is_delisted": False,
                "signal_eligible": True,
                "live_executable": False,
                "mega_cap": False,
                "metadata_status": "PIT_SECURITY_MASTER",
                "compliance_status": "UNSCREENED",
                "eligibility_status": "SIGNAL_ELIGIBLE",
                "discovery_source": "TEST",
            },
            {
                "instrument_id": "CURATED:GLD",
                "security_id": None,
                "symbol": "GLD",
                "name": "Gold exposure",
                "instrument_type": "COMMODITY_EXPOSURE",
                "asset_type": "COMMODITY_ETF",
                "category": "commodity",
                "exposure_type": "GOLD_PRICE_EXPOSURE",
                "exchange": "UNRESOLVED",
                "currency": "USD",
                "region": "GLOBAL",
                "country": "MULTI_COUNTRY_OR_GLOBAL",
                "sector": "GOLD",
                "industry": "GOLD",
                "sleeve": "commodity_security",
                "active_listing": True,
                "is_delisted": False,
                "signal_eligible": False,
                "live_executable": False,
                "mega_cap": False,
                "metadata_status": "CURATED_RESEARCH_OVERLAY",
                "compliance_status": "UNSCREENED",
                "eligibility_status": "RESEARCH_ONLY",
                "discovery_source": "TEST",
            },
        ]
    ).to_parquet(universe / "instruments.parquet", index=False)
    ranking = {
        "status": "GO",
        "count": 1,
        "rankings": [
            {
                "rank": 1,
                "sector": "Technology",
                "industry": "Software",
                "region": "UNITED_STATES_LISTED",
                "instrument_count": 1,
                "signal_count": 1,
                "maximum_opportunity_score": 0.8,
            }
        ],
    }
    for dimension in ("sector", "industry", "region"):
        payload = {
            **ranking,
            "dimension": dimension,
            "rankings": [
                {
                    **ranking["rankings"][0],
                    dimension: ranking["rankings"][0][dimension],
                }
            ],
        }
        _write_json(
            root,
            f"output/universe/{dimension}-ranking.json",
            payload,
        )
    signal = {
        "rank": 1,
        "symbol": "AAA",
        "instrument_type": "STOCK",
        "sector": "Technology",
        "region": "UNITED_STATES",
        "timeframe": "4h/1d",
        "timeframes": ["4h", "1d"],
        "opportunity_score": 0.8,
        "signal_status": "ACTIONABLE",
        "entry_zone_low": "99",
        "entry_zone_high": "101",
        "initial_stop": "95",
        "target_1": "108",
    }
    _write_json(
        root,
        "output/signals/latest_top_5_publication.json",
        {
            "status": "GO",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "raw_top_5": [signal],
            "diversified_top_5": [signal],
            "top_stocks": [signal],
            "top_etfs": [],
            "top_commodity_exposures": [],
            "actionable_signals": [signal],
            "auto_eligible_signals": [],
        },
    )
    _write_json(
        root,
        "output/portfolio/opportunity_ranking.json",
        {
            "status": "GO",
            "opportunities": [
                {
                    "ticker": "AAA",
                    "opportunity_score": 0.8,
                    "research_allocation_eligible": True,
                    "contract_resolved": True,
                    "timeframes": ["4h", "1d"],
                    "sector": "Technology",
                    "region": "UNITED_STATES",
                }
            ],
        },
    )
    _write_json(
        root,
        "output/signals/latest_signals.json",
        {
            "status": "GO",
            "signals": [
                {
                    "ticker": "AAA",
                    "strategy_id": "S1",
                    "action": "BUY",
                    "data_freshness": "FRESH",
                    "expiration_timestamp": "2099-01-01T00:00:00+00:00",
                    "confidence_score": 0.82,
                    "suggested_quantity": 3,
                    "limit_entry_price": 100.0,
                    "entry_zone_low": 99.0,
                    "entry_zone_high": 101.0,
                    "stop_loss": 95.0,
                    "take_profit_1": 108.0,
                    "take_profit_2": 114.0,
                    "reward_risk_1": 1.6,
                    "reasons": ["TREND_CONFIRMED"],
                    "risks": ["GAP_RISK"],
                }
            ],
        },
    )
    _write_json(
        root,
        "output/operations/signal-lifecycle.json",
        {
            "status": "GO",
            "rows": [
                {
                    "ticker": "AAA",
                    "strategy_id": "S1",
                    "lifecycle_status": "EXIT",
                    "previous_action": "BUY",
                    "current_action": "WATCHLIST",
                },
                {
                    "ticker": "GLD",
                    "strategy_id": "S2",
                    "lifecycle_status": "AVOID",
                    "previous_action": "WATCHLIST",
                    "current_action": "AVOID",
                },
            ],
        },
    )
    _write_json(
        root,
        "output/portfolio/position-management.json",
        {
            "status": "GO",
            "positions": [
                {
                    "ticker": "BBB",
                    "advisory_action": "REDUCE_50",
                    "reason_codes": ["PROFIT_GIVEBACK_40_PERCENT"],
                    "current_r": 1.8,
                    "peak_r": 3.0,
                    "profit_giveback": 0.4,
                }
            ],
            "execution_authority": "NONE",
        },
    )
    _write_json(
        root,
        "runtime/heartbeat.json",
        {
            "runtime_status": "RUNNING",
            "runtime_state": "NORMAL",
            "IBKR_status": "GO",
            "open_positions": 0,
            "open_orders": 0,
            "execution_authority": "NONE",
            "account_id": "SHOULD_NOT_RENDER",
        },
    )
    _write_json(
        root,
        "output/operations/machine-status.json",
        {"status": "GO", "mode": "SIGNALS_ONLY", "cycle_count": 1},
    )
    _write_json(
        root,
        "output/operations/execution-status.json",
        {
            "status": "GO",
            "execution_authority": "NONE",
            "strategy_authority": "NONE",
        },
    )
    _write_json(
        root,
        "output/ibkr/live/status.json",
        {
            "status": "GO",
            "runtime_state": "SIGNALS_ONLY",
            "broker_connection": "UNREACHABLE",
            "account_reconciliation": "LIVE_RECONCILED_EMPTY",
            "execution_authority": "NONE",
            "kill_switch_state": "CLEAR",
            "open_blockers": ["LIVE_AUTHORITY_NOT_GRANTED"],
            "api_key": "SHOULD_NOT_RENDER",
        },
    )
    snapshot_hash = "VERIFIED-LIVE-SNAPSHOT"
    _write_json(
        root,
        "output/ibkr/live/reconciliation.json",
        {"status": "GO", "private_snapshot_hash": snapshot_hash},
    )
    broker_db = (
        root
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    broker_db.parent.mkdir(parents=True)
    with sqlite3.connect(broker_db) as connection:
        connection.execute(
            "CREATE TABLE snapshots(snapshot_hash TEXT, payload_json TEXT, created_at TEXT)"
        )
        connection.execute(
            "INSERT INTO snapshots VALUES(?,?,?)",
            (
                "PRIOR-LIVE-SNAPSHOT",
                json.dumps(
                    {
                        "account": {
                            "status": "COMPLETE",
                            "values": [
                                {
                                    "tag": "NetLiquidation",
                                    "value": "9800",
                                    "currency": "EUR",
                                },
                                {
                                    "tag": "TotalCashValue",
                                    "value": "1800",
                                    "currency": "EUR",
                                },
                            ],
                        },
                        "snapshot_completed_at": (
                            f"{previous_day_iso}T10:00:00+00:00"
                        ),
                    }
                ),
                f"{previous_day_iso}T10:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO snapshots VALUES(?,?,?)",
            (
                snapshot_hash,
                json.dumps(
                    {
                        "account": {
                            "status": "COMPLETE",
                            "values": [
                                {
                                    "tag": "NetLiquidation",
                                    "value": "10000",
                                    "currency": "EUR",
                                },
                                {
                                    "tag": "TotalCashValue",
                                    "value": "2000",
                                    "currency": "EUR",
                                },
                                {
                                    "tag": "GrossPositionValue",
                                    "value": "8000",
                                    "currency": "EUR",
                                },
                            ]
                        },
                        "snapshot_completed_at": (
                            f"{today_iso}T10:00:00+00:00"
                        ),
                    }
                ),
                f"{today_iso}T10:00:00+00:00",
            ),
        )
    pnl = root / "data/performance/private/daily-pnl.jsonl"
    pnl.parent.mkdir(parents=True)
    pnl.write_text(
        json.dumps(
            {
                "session_date": today_iso,
                "environment": "LIVE",
                "realized_pnl_eur": 75,
                "unrealized_pnl_eur": 25,
                "net_pnl_eur": 100,
                "source": "TEST_BROKER_EVIDENCE",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        root,
        "output/ibkr/phase9/status.json",
        {"status": "NO_GO", "phase9_marker": "PAPER_ONLY"},
    )
    _write_json(
        root,
        "output/portfolio/status.json",
        {
            "status": "GO",
            "signal_count": 1,
            "opportunity_count": 1,
            "registered_strategy_dna_count": 10,
            "current_position_count": 0,
            "private_whole_share_sizing_status": "GO",
            "orders_generated": 0,
        },
    )
    _write_json(
        root,
        "output/portfolio/exposure_report.json",
        {
            "approved_gross_exposure": 0.0,
            "research_gross_exposure": 0.1,
            "sector_weights": {},
            "region_weights": {},
            "sleeve_weights": {},
        },
    )
    for name in ("current_allocation", "rebalance_plan"):
        _write_json(root, f"output/portfolio/{name}.json", {"status": "GO"})
    _write_json(
        root,
        "output/portfolio/risk_contributions.json",
        {
            "status": "GO",
            "research_portfolio_heat": 0.02,
            "observed_current_portfolio_heat": 0.01,
        },
    )
    _write_json(
        root,
        "output/portfolio/target_allocation.json",
        {
            "status": "GO",
            "allocations": [
                {
                    "ticker": "AAA",
                    "research_target_weight": 0.10,
                    "correlation_penalty": 0.0,
                }
            ],
        },
    )
    _write_json(
        root,
        "output/portfolio/dynamic-risk-state.json",
        {
            "status": "GO",
            "equity_band": "EUR_1K_TO_10K",
            "exact_equity_public": False,
            "tier_maximum_positions": 5,
            "dynamic_research_maximum_positions": 2,
            "operational_maximum_positions": 0,
            "regime_confidence": 0.8,
            "signal_quality": 0.8,
            "diversification_score": 0.7,
            "portfolio_drawdown_pct": None,
            "portfolio_drawdown_status": "INSUFFICIENT_HISTORY",
            "drawdown_velocity_per_day": None,
            "drawdown_velocity_status": "INSUFFICIENT_HISTORY",
            "recovery_ratio": None,
            "recovery_status": "INSUFFICIENT_HISTORY",
            "consecutive_loss_sessions": 0,
            "loss_streak_status": "INSUFFICIENT_HISTORY",
            "multipliers": {"combined": 0.85, "data_quality": 0.85},
        },
    )
    _write_json(
        root,
        "output/capital/current_level.json",
        {
            "CURRENT_CAPITAL_LEVEL_NAME": "SIGNALS_AND_SHADOW",
            "CURRENT_TARGET_EXPOSURE": 0.0,
            "level_limits": {},
        },
    )
    _write_json(
        root,
        "output/dynamic/status.json",
        {"current_regime": "BULL_TREND_LOW_VOL"},
    )
    _write_json(
        root,
        "output/notifications/telegram_status.json",
        {
            "schema": "telegram_status_v1",
            "status": "ENABLED",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        },
    )
    _write_json(
        root,
        "output/macro/score.json",
        {
            "status": "DATA_INCOMPLETE",
            "as_of": "2026-07-31T10:00:00+00:00",
            "macro_analysis_authority": "RESEARCH_ONLY",
            "execution_authority": "NONE",
            "data_quality": {
                "status": "DATA_INCOMPLETE",
                "feature_status_counts": {
                    "VALID": 6,
                    "UNAVAILABLE": 2,
                },
            },
            "features": {
                "USD_INDEX": {
                    "status": "VALID",
                    "original_value": 100.25,
                    "transformed_value": 1.2,
                    "observation_date": "2026-07-30",
                    "provider": "TEST",
                },
                "US_YIELD_CURVE_10Y2Y": {
                    "status": "VALID",
                    "original_value": 0.45,
                    "transformed_value": -0.05,
                    "observation_date": "2026-07-30",
                    "provider": "TEST",
                },
                "US_HIGH_YIELD_SPREAD": {
                    "status": "VALID",
                    "original_value": 3.2,
                    "transformed_value": 0.1,
                    "observation_date": "2026-07-30",
                    "provider": "TEST",
                },
                "VIX": {
                    "status": "VALID",
                    "original_value": 17.5,
                    "transformed_value": -2.0,
                    "observation_date": "2026-07-30",
                    "provider": "TEST",
                },
                "EQUITY_BREADTH_GLOBAL": {
                    "status": "VALID",
                    "original_value": 0.62,
                    "transformed_value": 0.12,
                    "observation_date": "2026-07-30",
                    "provider": "TEST",
                },
            },
            "scores": {
                "liquidity": {
                    "status": "VALID",
                    "value": 21.5,
                    "confidence": 0.8,
                    "coverage": 0.9,
                    "missing_inputs": [],
                },
                "commodity": {
                    "status": "VALID",
                    "value": -12.5,
                    "confidence": 1.0,
                    "coverage": 1.0,
                    "missing_inputs": [],
                },
            },
            "regime": {
                "overall_macro_regime": "EXPANSION_DISINFLATION",
                "confidence": 0.72,
                "hysteresis_status": "REGIME_STABLE",
                "reasons": ["GROWTH_POSITIVE", "INFLATION_FALLING"],
            },
        },
    )
    _write_json(
        root,
        "output/macro/history.json",
        {
            "status": "GO",
            "history": [
                {
                    "as_of": "2026-07-30T10:00:00+00:00",
                    "regime": {
                        "overall_macro_regime": "TRANSITION"
                    },
                }
            ],
        },
    )
    _write_json(
        root,
        "output/research/strategies/strategy_registry.json",
        {
            "status": "GO",
            "bulk_strategy_count": 10,
            "standard_strategy_count": 2,
        },
    )
    summary = root / "output" / "research" / "phase11_14"
    summary.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy_id": "S1",
                "formula": "trend",
                "asset_class": "STOCK",
                "timeframe": "4h",
                "combined_oos_CAGR": 0.1,
                "combined_oos_Sharpe": 1.0,
                "combined_period_profit_factor": 1.2,
                "maximum_drawdown": -0.1,
                "positive_fold_ratio": 0.8,
                "cost_50bps_combined_return": 0.05,
                "research_pass": True,
                "robust_pass": True,
                "deployable_pass": False,
                "deployment_blockers": "FORWARD_REQUIRED",
                "financial_finalist": False,
            }
        ]
    ).to_parquet(summary / "strategy-summary.parquet", index=False)
    _write_json(root, "output/research/phase11_14/status.json", {"status": "GO"})
    _write_json(
        root,
        "output/research/phase11_14/latest-forward-observation.json",
        {
            "schema": "phase11_14_forward_observation_v3",
            "status": "GO",
            "observations": [
                {
                    "strategy_id": "P1114-1H-EXPLORE",
                    "formula": "bollinger_breakout",
                    "timeframe": "1h",
                    "asset_class": "COMMODITY_PROXY",
                    "observer_tier": "EXPLORATORY_FORWARD_OBSERVER",
                    "portfolio_eligible": False,
                    "execution_eligible": False,
                    "observation_status": "OBSERVATION_COMPLETE",
                    "closed_bar_timestamp": "2026-07-30T19:30:00+00:00",
                    "data_freshness": "FRESH_CLOSED_BAR",
                    "raw_active_signals": [],
                    "current_attested_target_weights": {},
                    "portfolio_action": "OBSERVE_SIGNAL_ONLY_NO_ALLOCATION",
                }
            ],
        },
    )
    _write_json(
        root,
        "output/research/autopilot/leaderboard.json",
        {"status": "GO", "technical_fixture_rows": []},
    )
    _write_json(
        root,
        "output/notifications/market-intelligence-digest.json",
        {
            "status": "GO",
            "news_freshness_status": "CURRENT_WITHIN_72H",
            "important_news": [],
            "upcoming_macro_events": [],
            "news_source_status": {"record_count": 10},
        },
    )
    _write_json(
        root,
        "output/analysis/groups/coverage.json",
        {
            "status": "GO_WITH_DOCUMENTED_GAPS",
            "sector_count": 1,
            "industry_count": 1,
            "signal_eligible_fundamental_coverage_ratio": 1.0,
            "signal_eligible_fundamental_missing_symbols": [],
        },
    )
    group = {
        "analysis_status": "GO",
        "maximum_opportunity_score": 0.8,
        "fresh_news_event_count": 2,
        "fundamental_coverage_ratio": 1.0,
    }
    _write_json(
        root,
        "output/analysis/groups/sector-analysis.json",
        {"status": "GO", "groups": [{**group, "sector": "Technology"}]},
    )
    _write_json(
        root,
        "output/analysis/groups/industry-analysis.json",
        {"status": "GO", "groups": [{**group, "industry": "Software"}]},
    )
    _write_json(
        root,
        "output/ibkr/news/capabilities.json",
        {
            "status": "TWS_UNAVAILABLE",
            "provider_count": 0,
            "historical_headlines_capability": "UNPROVEN_TWS_UNAVAILABLE",
            "tws_connected": False,
        },
    )
    _write_json(
        root,
        "output/portfolio/confluence-audit.json",
        {
            "status": "GO",
            "rows": [
                {
                    "ticker": "AAA",
                    "status": "THREE_LAYER_CONFIRMED",
                    "confluence_score": 0.75,
                    "ranking_multiplier": 1.05,
                    "base_score_without_confluence": 0.76,
                    "final_opportunity_score": 0.798,
                    "score_delta": 0.038,
                    "layer_statuses": {
                        "technical": "SUPPORTIVE",
                        "fundamental": "SUPPORTIVE",
                        "macro": "SUPPORTIVE",
                    },
                    "allocation_allowed": True,
                    "allocation_blockers": [],
                    "macro_source": "ASSET_SPECIFIC_MACRO_TRANSMISSION",
                }
            ],
            "execution_authority": "NONE",
        },
    )
    bars = (
        root
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
        / "provider=TEST"
        / "symbol=AAA"
        / "interval=4h"
        / "source_interval=1h"
    )
    bars.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-01T12:00:00+00:00",
                "provider": "TEST",
                "bar_origin": "DERIVED",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "volume": 1000.0,
                "is_partial": False,
            }
        ]
    ).to_parquet(bars / "bars.parquet", index=False)


def test_ui_pages_load_and_remain_private(tmp_path: Path) -> None:
    _ui_fixture(tmp_path)
    client = TestClient(create_app(tmp_path))
    pages = (
        "/",
        "/signals",
        "/universe",
        "/sectors",
        "/industries",
        "/regions",
        "/etfs",
        "/commodities",
        "/strategies",
        "/portfolio",
        "/performance",
        "/news",
        "/asset/AAA",
        "/research",
        "/health",
        "/audit",
    )
    for page in pages:
        response = client.get(page)
        assert response.status_code == 200
        assert "SHOULD_NOT_RENDER" not in response.text
        assert "execution authority" in response.text.lower() or (
            "authority none" in response.text.lower()
        )

    strategies = client.get("/strategies")
    assert "Forward Observer Tiers" in strategies.text
    assert "P1114-1H-EXPLORE" in strategies.text
    assert "EXPLORATORY FORWARD OBSERVER" in strategies.text

    health = client.get("/health")
    assert "Telegram" in health.text
    assert "ENABLED" in health.text


def test_ui_filters_chart_and_read_only_http(tmp_path: Path) -> None:
    _ui_fixture(tmp_path)
    client = TestClient(create_app(tmp_path))

    universe = client.get(
        "/api/universe",
        params={"query": "alpha", "page_size": 10},
    ).json()
    signals = client.get(
        "/api/signals",
        params={"collection": "stocks", "minimum_score": 0.7},
    ).json()
    chart = client.get("/api/chart/AAA?interval=4h").json()
    asset = client.get("/api/asset/AAA").json()
    coverage = client.get("/api/analysis/coverage").json()
    news = client.get("/api/news").json()
    performance = client.get("/api/performance?month=2026-01").json()
    dashboard = client.get("/api/dashboard").json()
    portfolio = client.get("/api/portfolio").json()

    assert universe["count"] == 1
    assert universe["instruments"][0]["symbol"] == "AAA"
    assert signals["count"] == 1
    assert signals["trending"][0]["draft_order"]["side"] == "BUY"
    assert signals["trending"][0]["draft_order"]["status"] == (
        "DRAFT_BLOCKED_CAPITAL_LEVEL_ZERO"
    )
    assert signals["trending"][0]["draft_order"]["quantity"] is None
    assert signals["trending"][0]["classification"] == "A"
    assert signals["trending"][0]["expected_rr"] == 1.6
    assert signals["trending"][0]["why_it_exists"] == ["TREND_CONFIRMED"]
    assert "GAP_RISK" in signals["trending"][0]["why_it_can_fail"]
    assert signals["exit_monitor"]["exit_signal_count"] == 2
    assert signals["exit_monitor"]["exit_signals"][0][
        "model_action"
    ] == "SELL_EXIT"
    assert signals["exit_monitor"]["exit_signals"][1][
        "model_action"
    ] == "SELL_REDUCE"
    assert signals["exit_monitor"]["exit_signals"][1][
        "source"
    ] == "POSITION_MANAGEMENT"
    assert signals["exit_monitor"]["avoid_signals"][0][
        "sell_order_status"
    ] == "NOT_AN_ORDER"
    assert chart["status"] == "GO"
    assert chart["bars"][0]["close"] == 101.0
    assert asset["symbol"] == "AAA"
    assert coverage["analyzable_instrument_count"] == 1
    assert news["execution_authority"] == "NONE"
    assert news["sector_industry_coverage"]["sector_count"] == 1
    assert news["top_sectors"][0]["sector"] == "Technology"
    assert news["top_industries"][0]["industry"] == "Software"
    assert news["ibkr_news"]["status"] == "TWS_UNAVAILABLE"
    assert performance["schema"] == "ui_pnl_calendar_v1"
    assert performance["environment"] == "LIVE"
    assert performance["environment_separation"] == "STRICT_NO_MIXING"
    assert performance["execution_authority"] == "NONE"
    assert len(performance["weeks"]) >= 4
    by_period = {
        row["period"]: row for row in performance["period_performance"]
    }
    assert by_period["TODAY"]["net_pnl_eur"] == 100.0
    assert by_period["TODAY"]["return_pct"] == 0.01010101
    assert by_period["TODAY"]["exact_equity_public"] is False
    assert dashboard["risk"]["dynamic_maximum_positions"] == 2
    assert dashboard["risk"]["operational_maximum_positions"] == 0
    assert dashboard["risk"]["open_risk"] == 0.01
    assert dashboard["account"]["portfolio_value_eur"] == 10000.0
    assert dashboard["account"]["cash_eur"] == 2000.0
    assert dashboard["account"]["high_water_mark_eur"] == 10000.0
    assert dashboard["account"]["observation_count"] == 2
    assert dashboard["account"]["persisted_to_public_artifact"] is False
    assert dashboard["performance"]["TODAY"]["return_pct"] == 0.01010101
    assert dashboard["market"]["regime_confidence"] == 0.8
    assert dashboard["macro"]["status"] == "DEGRADED_DATA_INCOMPLETE"
    assert dashboard["macro"]["current_regime"] == (
        "EXPANSION_DISINFLATION"
    )
    assert dashboard["macro"]["previous_regime"] == "TRANSITION"
    assert dashboard["macro"]["regime_change"] == "REGIME_CHANGED"
    indicators = {
        row["series_id"]: row for row in dashboard["macro"]["indicators"]
    }
    assert indicators["USD_INDEX"]["value"] == 100.25
    assert indicators["US_REAL_YIELD_10Y"]["value"] is None
    assert indicators["EM_DEVELOPED_RELATIVE_STRENGTH"]["value"] is None
    assert indicators["liquidity"]["value"] == 21.5
    assert portfolio["dynamic_risk"]["equity_band"] == "EUR_1K_TO_10K"
    assert portfolio["dynamic_risk"]["exact_equity_public"] is False
    assert portfolio["confluence"]["rows"][0]["status"] == (
        "THREE_LAYER_CONFIRMED"
    )
    assert client.post("/api/portfolio", json={}).status_code == 405
    assert client.post("/api/performance", json={}).status_code == 405


def test_dashboard_macro_board_renders_missing_series_fail_closed(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    _write_json(tmp_path, "output/macro/score.json", {})
    _write_json(tmp_path, "output/macro/history.json", {})

    client = TestClient(create_app(tmp_path))
    payload = client.get("/api/dashboard").json()
    page = client.get("/")

    assert payload["macro"]["status"] == "NO_DATA"
    assert payload["macro"]["execution_authority"] == "NONE"
    assert all(
        row["value"] is None for row in payload["macro"]["indicators"]
    )
    assert page.status_code == 200
    assert "Macro Control Board" in page.text
    assert "Real rate (US 10Y)" in page.text
    assert "EM versus developed" in page.text
    assert "Unavailable" in page.text


def test_market_heat_keeps_current_price_for_invalidated_setup(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    _write_json(
        tmp_path,
        "output/portfolio/opportunity_ranking.json",
        {
            "status": "GO",
            "opportunities": [
                {
                    "ticker": "AAPL",
                    "opportunity_score": 0.79,
                    "research_allocation_eligible": True,
                    "contract_resolved": True,
                    "timeframes": ["1d"],
                    "sector": "Technology",
                    "region": "UNITED_STATES",
                    "deployment_blockers": [],
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "output/signals/latest_signals.json",
        {
            "status": "GO",
            "signals": [
                {
                    "ticker": "AAPL",
                    "strategy_id": "OLD_DAILY_SIGNAL",
                    "original_action": "BUY",
                    "action": "AVOID",
                    "data_freshness": "STALE",
                    "lifecycle_status": "INVALIDATED",
                    "entry_zone_low": 329.0,
                    "entry_zone_high": 332.0,
                    "stop_loss": 315.55,
                    "take_profit_1": 356.0,
                    "current_market_price": 302.79,
                    "market_reference_price": 302.79,
                    "market_reference_fetched_at": (
                        "2098-01-01T00:00:00+00:00"
                    ),
                    "market_reference_age_minutes": 12.0,
                    "price_validity_status": (
                        "CURRENT_PRICE_BREACHED_SIGNAL_STOP"
                    ),
                    "entry_instruction": "WAIT_FOR_NEW_CAUSAL_SIGNAL",
                    "risks": ["CURRENT_PRICE_BREACHED_SIGNAL_STOP"],
                }
            ],
        },
    )

    signals = ViewModelStore(tmp_path).signals()
    row = signals["trending"][0]

    assert row["symbol"] == "AAPL"
    assert row["model_action"] == "WATCH"
    assert row["current_price"] == 302.79
    assert row["entry_low"] is None
    assert row["entry_instruction"] == "WAIT_FOR_NEW_CAUSAL_SIGNAL"
    assert "CURRENT_PRICE_BREACHED_SIGNAL_STOP" in row["why_it_can_fail"]
    avoid = next(
        item
        for item in signals["exit_monitor"]["avoid_signals"]
        if item["symbol"] == "AAPL"
    )
    assert avoid["source"] == "MARKET_REFERENCE"
    assert avoid["model_action"] == "AVOID_NEW_LONG"
    assert avoid["current_price"] == 302.79


def test_region_clock_uses_exchange_calendars() -> None:
    rows = _exchange_clock(
        pd.Timestamp("2026-07-30T15:00:00Z").to_pydatetime()
    )
    by_name = {row["name"]: row for row in rows}

    assert by_name["NYSE"]["status"] == "OPEN"
    assert by_name["NYSE"]["next_event_type"] == "CLOSES"
    assert by_name["TSE"]["status"] == "CLOSED"
    assert by_name["TSE"]["next_event_type"] == "OPENS"
    assert all(row["next_event_utc"] for row in rows)
    assert all(row["next_event_local"] for row in rows)
    assert all(row["seconds_to_next_event"] >= 0 for row in rows)


def test_region_api_publishes_live_globe_contract(tmp_path: Path) -> None:
    _ui_fixture(tmp_path)
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/dimensions/region")
    page = client.get("/regions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["exchange_clock_generated_at"]
    assert payload["exchange_open_count"] + payload[
        "exchange_closed_count"
    ] <= len(payload["exchange_clock"])
    assert len(payload["exchange_clock"]) == 12
    assert all(
        {
            "latitude",
            "longitude",
            "local_time",
            "next_event_local",
            "seconds_to_next_event",
        }
        <= set(row)
        for row in payload["exchange_clock"]
    )
    assert page.status_code == 200
    assert 'id="exchange-map"' in page.text
    assert 'data-exchange="NYSE"' in page.text
    javascript = (
        tmp_path / "src" / "stocks" / "ui" / "static" / "app.js"
    )
    if not javascript.exists():
        javascript = Path("src/stocks/ui/static/app.js")
    source = javascript.read_text(encoding="utf-8")
    assert 'type: "scattergeo"' in source
    assert 'fetch("/api/dimensions/region"' in source
    assert "setInterval(refreshExchangeState, 60000)" in source
    assert 'topojsonURL: "/static/"' in source
    assert "cdn.plot.ly" not in source
    topology_path = Path("src/stocks/ui/static/world_110m.json")
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    assert topology["type"] == "Topology"
    assert {"countries", "land", "ocean"} <= set(topology["objects"])


def test_period_performance_never_reports_missing_observation_as_zero() -> None:
    rows = _period_performance(
        {},
        today=pd.Timestamp("2026-07-31").date(),
        current_equity_eur=10000.0,
    )

    assert all(row["status"] == "NO_OBSERVATIONS" for row in rows)
    assert all(row["net_pnl_eur"] is None for row in rows)
    assert all(row["return_pct"] is None for row in rows)


def test_mobile_ui_wraps_long_status_tags_without_page_overflow() -> None:
    stylesheet = Path("src/stocks/ui/static/app.css").read_text(
        encoding="utf-8"
    )

    assert ".panel-header { flex-wrap: wrap; }" in stylesheet
    assert ".panel-header .status-tag" in stylesheet
    assert "white-space: normal;" in stylesheet
    assert "overflow-wrap: anywhere;" in stylesheet


def test_dashboard_blocks_private_values_on_snapshot_hash_mismatch(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    _write_json(
        tmp_path,
        "output/ibkr/live/reconciliation.json",
        {
            "status": "GO",
            "reconciliation_status": "LIVE_RECONCILED_EMPTY",
            "private_snapshot_hash": "MISMATCH",
        },
    )

    account = ViewModelStore(tmp_path).dashboard()["account"]

    assert account["status"] == "BLOCKED"
    assert account["reason"] == "PRIVATE_SNAPSHOT_HASH_MISMATCH"
    assert account["portfolio_value_eur"] is None
    assert account["cash_eur"] is None
    assert account["high_water_mark_eur"] is None
    assert account["persisted_to_public_artifact"] is False


def test_current_reconciliation_overrides_stale_green_live_status(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    _write_json(
        tmp_path,
        "output/ibkr/live/status.json",
        {
            "status": "GO",
            "broker_connection": "LIVE_RECONCILED_EMPTY",
            "account_reconciliation": "LIVE_RECONCILED_EMPTY",
            "position_count": 0,
            "open_order_count": 0,
            "kill_switch_state": "CLEAR",
            "open_blockers": ["EXACT_OPERATOR_APPROVAL_REQUIRED"],
            "execution_authority": "NONE",
        },
    )
    _write_json(
        tmp_path,
        "output/ibkr/live/reconciliation.json",
        {
            "status": "NO_GO",
            "reconciliation_status": "LIVE_TWS_SOCKET_UNREACHABLE",
            "blockers": ["LIVE_TWS_SOCKET_UNREACHABLE"],
            "position_count": None,
            "open_order_count": None,
            "execution_authority": "NONE",
        },
    )

    store = ViewModelStore(tmp_path)
    dashboard = store.dashboard()
    portfolio = store.portfolio()
    page = TestClient(create_app(tmp_path)).get("/")

    assert dashboard["broker"]["ibkr_status"] == "NO_GO"
    assert dashboard["broker"]["live_connection"] == (
        "LIVE_TWS_SOCKET_UNREACHABLE"
    )
    assert dashboard["broker"]["reconciliation"] == (
        "LIVE_TWS_SOCKET_UNREACHABLE"
    )
    assert dashboard["broker"]["reconciliation_go"] is False
    assert dashboard["broker"]["status_source"] == (
        "CURRENT_RECONCILIATION_ARTIFACT"
    )
    assert dashboard["risk"]["kill_switch"] == (
        "TRIGGERED_RECONCILIATION"
    )
    assert "LIVE_TWS_SOCKET_UNREACHABLE" in dashboard["blockers"]
    broker = portfolio["broker_state"]
    assert broker["status"] == "BROKER_OBSERVATION_BLOCKED"
    assert broker["environment"] == "BROKER_OBSERVATION_BLOCKED"
    assert broker["observed_position_count"] is None
    assert broker["observed_open_order_count"] is None
    assert broker["reconciliation"] == "NO_GO"
    assert broker["reconciliation_status"] == (
        "LIVE_TWS_SOCKET_UNREACHABLE"
    )
    assert "LIVE TWS SOCKET UNREACHABLE" in page.text
    portfolio_page = TestClient(create_app(tmp_path)).get("/portfolio")
    assert "Positions observed" in portfolio_page.text
    assert ">Unavailable</strong>" in portfolio_page.text
    assert (
        "Broker positions are unavailable until current reconciliation succeeds."
        in portfolio_page.text
    )


def test_performance_never_mixes_live_and_paper_pnl(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    pnl = tmp_path / "data/performance/private/daily-pnl.jsonl"
    with pnl.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "session_date": pd.Timestamp.now(
                        tz="Europe/Amsterdam"
                    ).date().isoformat(),
                    "environment": "PAPER",
                    "net_pnl_eur": 900,
                    "realized_pnl_eur": 900,
                    "unrealized_pnl_eur": 0,
                    "source": "PAPER_LEDGER_AUDIT",
                }
            )
            + "\n"
        )
    store = ViewModelStore(tmp_path)

    live = store.performance("2026-07", "live")
    paper = store.performance("2026-07", "paper")
    live_period = {
        row["period"]: row for row in live["period_performance"]
    }
    paper_period = {
        row["period"]: row for row in paper["period_performance"]
    }

    assert live["environment"] == "LIVE"
    assert live_period["TODAY"]["net_pnl_eur"] == 100.0
    assert live_period["TODAY"]["return_pct"] == 0.01010101
    assert paper["environment"] == "PAPER"
    assert paper_period["TODAY"]["net_pnl_eur"] == 900.0
    assert paper_period["TODAY"]["return_pct"] is None


def test_ui_frame_reader_fails_soft_during_partial_parquet_write(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "output" / "sample.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"partial-not-parquet")
    store = ViewModelStore(tmp_path)

    monkeypatch.setattr("stocks.ui.service.time.sleep", lambda _value: None)
    frame = store._frame(Path("output/sample.parquet"))

    assert frame.empty


def test_portfolio_prefers_proven_live_counts_over_stale_paper_counts(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    _write_json(
        tmp_path,
        "output/operations/machine-status.json",
        {
            "status": "DEGRADED",
            "last_open_positions": 1,
            "last_open_orders": 1,
        },
    )
    _write_json(
        tmp_path,
        "output/ibkr/live/status.json",
        {
            "status": "GO",
            "account_reconciliation": "LIVE_RECONCILED_EMPTY",
            "position_count": 0,
            "open_order_count": 0,
            "execution_authority": "NONE",
        },
    )

    broker = ViewModelStore(tmp_path).portfolio()["broker_state"]

    assert broker["environment"] == "LIVE_READ_ONLY"
    assert broker["observed_position_count"] == 0
    assert broker["observed_open_order_count"] == 0
    assert broker["paper_observed_position_count"] == 1
    assert broker["paper_observed_open_order_count"] == 1
    assert broker["reconciliation"] == "GO"
    assert broker["execution_authority"] == "NONE"


def test_portfolio_position_overview_merges_only_verified_private_state(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    identity = "POSITION-BBB"
    _write_json(
        tmp_path,
        "output/portfolio/current_allocation.json",
        {
            "status": "PRIVATE_BROKER_POSITION_SNAPSHOT_COMPLETE",
            "position_count": 1,
            "positions": [
                {
                    "ticker": "BBB",
                    "position_identity": identity,
                    "security_type": "STK",
                    "currency": "USD",
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "output/portfolio/position-management.json",
        {
            "status": "GO",
            "positions": [
                {
                    "ticker": "BBB",
                    "position_identity": identity,
                    "status": "GO",
                    "advisory_action": "HOLD",
                    "reason_codes": ["TREND_HEALTHY"],
                    "current_r": 1.5,
                    "peak_r": 2.0,
                    "profit_giveback": 0.25,
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "output/portfolio/opportunity_ranking.json",
        {
            "opportunities": [
                {
                    "ticker": "BBB",
                    "sector": "Industrials",
                    "region": "UNITED_STATES",
                    "strategy_families": ["trend", "quality"],
                }
            ]
        },
    )
    _write_json(
        tmp_path,
        "output/ibkr/live/status.json",
        {
            "status": "GO",
            "account_reconciliation": "LIVE_RECONCILED",
            "position_count": 1,
            "open_order_count": 0,
            "execution_authority": "NONE",
        },
    )
    _write_json(
        tmp_path,
        "output/ibkr/live/reconciliation.json",
        {
            "status": "GO",
            "reconciliation_status": "LIVE_RECONCILED",
            "private_snapshot_hash": "VERIFIED-LIVE-SNAPSHOT",
        },
    )
    position_db = (
        tmp_path
        / "data/portfolio/private/position_management.sqlite3"
    )
    position_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(position_db) as connection:
        connection.execute(
            "CREATE TABLE position_events("
            "event_hash TEXT, position_identity TEXT, observed_at TEXT, "
            "payload_json TEXT)"
        )
        connection.execute(
            "INSERT INTO position_events VALUES(?,?,?,?)",
            (
                "EVENT-1",
                identity,
                "2026-07-31T12:00:00+00:00",
                json.dumps(
                    {
                        "ticker": "BBB",
                        "quantity": 2,
                        "entry_price": 50,
                        "current_price": 55,
                        "initial_stop": 47,
                        "proposed_stop": 52,
                        "current_r": 1.5,
                        "peak_r": 2.0,
                        "profit_giveback": 0.25,
                        "status": "GO",
                        "action": "HOLD",
                        "market_data_source": "YFINANCE",
                    }
                ),
            ),
        )
    action_plan = (
        tmp_path / "data/portfolio/private/latest-action-plan.json"
    )
    action_plan.write_text(
        json.dumps(
            {
                "whole_share_sizing": {
                    "account_snapshot_hash": "VERIFIED-LIVE-SNAPSHOT",
                    "account_equity_eur": 10000,
                    "positions": [
                        {
                            "ticker": "BBB",
                            "current_quantity": 2,
                            "unit_notional_eur": 50,
                            "risk_per_share_eur": 3,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    overview = ViewModelStore(tmp_path).portfolio()["position_overview"]
    row = overview["positions"][0]

    assert overview["status"] == "GO"
    assert overview["position_count"] == 1
    assert overview["strategy_ownership_inferred"] is False
    assert overview["persisted_to_public_artifact"] is False
    assert row["symbol"] == "BBB"
    assert row["strategy"] == "UNATTRIBUTED_BROKER_POSITION"
    assert row["entry_price"] == 50.0
    assert row["current_price"] == 55.0
    assert row["stop"] == 52.0
    assert row["quantity"] == 2.0
    assert row["current_r"] == 1.5
    assert row["peak_r"] == 2.0
    assert row["profit_giveback"] == 0.25
    assert row["position_weight"] == 0.01
    assert row["risk_contribution"] == 0.0006
    assert row["factor_cluster"] == (
        "Industrials / UNITED_STATES"
    )
    assert row["execution_authority"] == "NONE"


def test_portfolio_position_overview_fails_closed_without_private_state(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    _write_json(
        tmp_path,
        "output/portfolio/current_allocation.json",
        {
            "status": "PRIVATE_BROKER_POSITION_SNAPSHOT_COMPLETE",
            "position_count": 1,
            "positions": [
                {
                    "ticker": "BBB",
                    "position_identity": "POSITION-BBB",
                    "security_type": "STK",
                    "currency": "USD",
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "output/ibkr/live/status.json",
        {
            "status": "GO",
            "account_reconciliation": "LIVE_RECONCILED",
            "position_count": 1,
            "open_order_count": 0,
            "execution_authority": "NONE",
        },
    )

    overview = ViewModelStore(tmp_path).portfolio()["position_overview"]
    row = overview["positions"][0]

    assert overview["status"] == "DEGRADED_FAIL_CLOSED"
    assert overview["private_state_missing_count"] == 1
    assert row["entry_price"] is None
    assert row["current_price"] is None
    assert row["stop"] is None
    assert row["position_weight"] is None
    assert row["risk_contribution"] is None
    assert row["data_status"] == "PRIVATE_POSITION_STATE_UNAVAILABLE"
    assert row["execution_authority"] == "NONE"


def test_portfolio_page_renders_unavailable_risk_metrics_as_unknown(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    _write_json(
        tmp_path,
        "output/portfolio/risk_contributions.json",
        {
            "status": "GO",
            "research_target_exposure": None,
            "estimated_annualized_volatility": None,
            "research_portfolio_heat": None,
            "maximum_pairwise_correlation": None,
        },
    )

    response = TestClient(create_app(tmp_path)).get("/portfolio")

    assert response.status_code == 200
    assert response.text.count("<dd>Unavailable</dd>") >= 4


def test_ui_sanitizes_sensitive_keys_and_blocks_external_bind(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    store = ViewModelStore(tmp_path)

    payload = json.dumps(store.health())
    blocked = ui_command(
        tmp_path,
        "start",
        host="0.0.0.0",
        port=8080,
    )

    assert "SHOULD_NOT_RENDER" not in payload
    assert "account_id" not in payload
    assert "api_key" not in payload
    assert blocked["status"] == "NO_GO"
    assert blocked["blockers"] == ["EXTERNAL_BINDING_NOT_AUTHORIZED"]
    assert blocked["execution_authority"] == "NONE"


def test_ui_runtime_recovers_windows_launcher_child_pid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    listener = SimpleNamespace(
        status=psutil.CONN_LISTEN,
        laddr=SimpleNamespace(port=8081),
        pid=456,
    )

    class FakeProcess:
        def cmdline(self) -> list[str]:
            return [
                "python.exe",
                str(tmp_path / "main.py"),
                "ui",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                "8081",
            ]

    monkeypatch.setattr(psutil, "net_connections", lambda **_: [listener])
    monkeypatch.setattr(psutil, "Process", lambda _pid: FakeProcess())

    assert (
        _owned_server_pid(
            tmp_path,
            {"process_id": 123, "host": "127.0.0.1", "port": 8081},
        )
        == 456
    )


def test_strategy_ui_publishes_dynamic_bayesian_allocation(
    tmp_path: Path,
) -> None:
    _write_json(
        tmp_path,
        "output/dynamic/strategy_scores.json",
        {
            "strategies": [
                {
                    "strategy_id": "P1114-TEST",
                    "family": "quality_momentum",
                    "timeframe": "4h",
                    "classification": "FROZEN_SHADOW",
                    "status": "FROZEN_DYNAMIC",
                    "enabled": True,
                    "financial_finalist": False,
                    "evidence": {
                        "evidence_status": "PARTIAL_EVIDENCE",
                        "sample_count": 120,
                        "metric_coverage": 0.7,
                        "missing_metrics": ["sortino", "recency"],
                        "bayesian_positive_probability": {
                            "probability_above_break_even": 0.8
                        },
                        "metrics": {
                            "profit_factor": {"status": "GO", "raw": 1.3},
                            "expectancy": {"status": "GO", "raw": 0.002},
                            "regime_fit": {"status": "GO", "raw": 0.9},
                        },
                    },
                },
                {
                    "strategy_id": "PAUSED-TEST",
                    "family": "mean_reversion",
                    "timeframe": "1d",
                    "classification": "FROZEN_SHADOW",
                    "status": "RESEARCH_ONLY",
                    "enabled": False,
                    "financial_finalist": False,
                    "evidence": {
                        "evidence_status": "INSUFFICIENT_EVIDENCE",
                        "sample_count": 0,
                        "metric_coverage": 0.0,
                        "missing_metrics": ["profit_factor", "expectancy"],
                        "metrics": {},
                    },
                },
            ]
        },
    )
    _write_json(
        tmp_path,
        "output/dynamic/strategy_weights.json",
        {
            "allocated_weight": 0.2,
            "unallocated_weight": 0.8,
            "weights": [
                {
                    "strategy_id": "P1114-TEST",
                    "family": "quality_momentum",
                    "timeframe": "4h",
                    "score": 0.61,
                    "weight": 0.2,
                    "evidence_status": "PARTIAL_EVIDENCE",
                }
            ],
        },
    )
    _write_json(
        tmp_path,
        "output/dynamic/status.json",
        {"status": "GO", "current_regime": "BULL_TREND_LOW_VOL"},
    )
    research = tmp_path / "output/research/phase11_14"
    research.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "strategy_id": "P1114-TEST",
                "formula": "quality_momentum",
                "asset_class": "STOCK",
                "timeframe": "4h",
                "combined_oos_CAGR": 0.12,
                "combined_oos_Sharpe": 1.1,
                "combined_period_profit_factor": 1.3,
                "maximum_drawdown": -0.12,
                "positive_fold_ratio": 0.75,
                "cost_50bps_combined_return": 0.04,
                "research_pass": True,
                "robust_pass": True,
                "deployable_pass": False,
                "deployment_blockers": "FORWARD_REQUIRED",
                "financial_finalist": False,
            }
        ]
    ).to_parquet(research / "strategy-summary.parquet", index=False)
    pd.DataFrame(
        [
            {
                "strategy_id": "P1114-TEST",
                "fold_id": "4h_F001",
                "cost_bps": 10.0,
                "date": timestamp,
                "daily_return": 0.003 if index % 2 == 0 else -0.001,
            }
            for index, timestamp in enumerate(
                pd.date_range(
                    "2026-01-01T00:00:00Z",
                    periods=126,
                    freq="4h",
                )
            )
        ]
    ).to_parquet(research / "oos-returns.parquet", index=False)
    _write_json(
        tmp_path,
        "output/research/phase11_14/forward-performance.json",
        {
            "schema": "phase11_14_forward_performance_v1",
            "status": "FORWARD_OUTCOMES_LOW_CONFIDENCE",
            "evidence_end": "2026-07-31T17:30:00+00:00",
            "counts": {
                "independent_session_count": 4,
                "completed_strategy_count": 1,
                "robust_strategy_count": 1,
                "episode_count": 12,
                "closed_episode_count": 10,
                "open_episode_count": 2,
            },
            "cost_model": {"cost_bps_per_side": 10.0},
            "aggregate": {"sample_status": "LOW_CONFIDENCE"},
            "per_strategy": [
                {
                    "strategy_id": "P1114-TEST",
                    "independent_session_count": 4,
                    "episode_count": 12,
                    "closed_episode_count": 10,
                    "open_episode_count": 2,
                    "net_profit_factor": 1.25,
                    "profit_factor_reason": "DEFINED",
                    "net_expectancy": 0.004,
                    "win_rate": 0.6,
                    "sample_status": "LOW_CONFIDENCE",
                }
            ],
        },
    )

    viewmodel = ViewModelStore(tmp_path).strategies()
    page = TestClient(create_app(tmp_path)).get("/strategies")

    assert viewmodel["dynamic_strategy_count"] == 1
    assert viewmodel["dynamic_strategies"][0]["timeframe"] == "4h"
    assert viewmodel["dynamic_strategies"][0]["bayesian_probability"] == 0.8
    assert viewmodel["dynamic_strategies"][0]["financial_finalist"] is False
    monitor = viewmodel["strategy_monitoring"]
    assert monitor["current_regime"] == "BULL_TREND_LOW_VOL"
    assert monitor["counts"]["SHADOW"] == 1
    assert monitor["counts"]["PAUSED"] == 1
    monitored = {
        row["strategy_id"]: row for row in monitor["rows"]
    }
    assert monitored["P1114-TEST"]["lifecycle"] == "SHADOW"
    assert monitored["P1114-TEST"]["profit_factor"] == 1.3
    assert monitored["P1114-TEST"]["expectancy"] == 0.002
    assert monitored["P1114-TEST"]["maximum_drawdown"] == -0.12
    assert monitored["P1114-TEST"]["regime_fit"] == 0.9
    assert monitored["P1114-TEST"]["rolling"]["status"] == "GO"
    assert monitored["P1114-TEST"]["rolling"]["used_observations"] == 126
    assert monitored["P1114-TEST"]["rolling"]["period_profit_factor"] == 3.0
    assert monitored["P1114-TEST"]["rolling"]["freshness_status"] == (
        "STALE_HISTORICAL_OOS_WINDOW"
    )
    assert monitored["P1114-TEST"]["forward"]["net_profit_factor"] == 1.25
    assert monitored["P1114-TEST"]["forward"]["sample_status"] == (
        "LOW_CONFIDENCE"
    )
    assert monitor["forward_counts"]["independent_session_count"] == 4
    assert monitored["PAUSED-TEST"]["lifecycle"] == "PAUSED"
    assert monitored["PAUSED-TEST"]["rolling"]["return"] is None
    assert page.status_code == 200
    assert "Operational Strategy Monitor" in page.text
    assert "Independent Forward Outcomes" in page.text
    assert "Net episode PF" in page.text
    assert "Rolling period PF" in page.text
    assert "OOS period returns at 10 bps" in page.text


def test_dashboard_surfaces_integrity_capabilities_sec_and_canonical_ml(
    tmp_path: Path,
) -> None:
    _ui_fixture(tmp_path)
    _write_json(
        tmp_path,
        "output/ibkr/live/writer-integrity-verify.json",
        {"status": "NO_GO", "writer_hash_integrity": False},
    )
    _write_json(
        tmp_path,
        "output/ibkr/data-capabilities/capability-matrix.json",
        {
            "status": "GO_DEGRADED",
            "summary": {
                "historical_bars": "AVAILABLE",
                "tick_by_tick_trades": "UNAVAILABLE_ENTITLEMENT",
            },
            "rows": [{"market_data_type": 1}],
            "missing_subscription_classes": ["US_STOCK_TICK_BY_TICK"],
        },
    )
    _write_json(
        tmp_path,
        "output/research/sec_intelligence/status.json",
        {
            "status": "DEGRADED",
            "structured_event_count": 0,
            "metadata_event_count": 100,
            "max_overlay_points": 4.0,
            "authority": "RANKING_OVERLAY_ONLY",
        },
    )
    _write_json(
        tmp_path,
        "output/research/active_swing/selective_ml/status.json",
        {
            "status": "NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS",
            "closed_trainable_episode_count": 0,
            "canonical_label_source": "PHASE9_CANONICAL_BROKER_FILL",
            "model_authority": "NONE",
            "regime_conditioning": (
                "SINGLE_MODEL_EXPLICIT_REGIME_FEATURE"
            ),
            "regime_dataset": {
                "status": "GO",
                "available_regime_count": 3,
            },
            "regime_generalization": {
                "status": "GO",
                "evaluable_regime_count": 3,
                "worst_regime": "VOLATILITY_SHOCK",
                "worst_regime_auc": 0.54,
                "regime_auc_std": 0.03,
                "robust_generalization_score": 0.56,
            },
            "reinforcement_learning_status": (
                "DISABLED_PREMATURE_SAMPLE_SIZE"
            ),
        },
    )
    _write_json(
        tmp_path,
        "output/research/active_swing/rejected_shadow/status.json",
        {
            "status": "GO",
            "rejected_episode_count": 40,
            "counterfactual_performance_count": 12,
            "gate_sample_ready_count": 0,
        },
    )
    _write_json(
        tmp_path,
        (
            "output/research/active_swing/rejected_shadow/"
            "gate-attribution.json"
        ),
        {
            "status": "GO",
            "automatic_gate_relaxation": False,
        },
    )

    dashboard = ViewModelStore(tmp_path).dashboard()

    assert dashboard["authority"]["writer_hash_integrity"] is False
    assert dashboard["broker"]["data_capability_status"] == "GO_DEGRADED"
    assert dashboard["broker"]["market_data_type"] == 1
    assert dashboard["research"]["sec_max_overlay_points"] == 4.0
    assert dashboard["research"]["sec_standalone_entry_allowed"] is False
    assert dashboard["research"]["selective_ml_closed_labels"] == 0
    assert dashboard["research"]["selective_ml_shadow_only"] is True
    assert dashboard["research"]["ml_regime_count"] == 3
    assert dashboard["research"]["ml_loro_status"] == "GO"
    assert dashboard["research"]["ml_worst_regime"] == (
        "VOLATILITY_SHOCK"
    )
    assert dashboard["research"]["ml_worst_regime_auc"] == 0.54
    assert dashboard["research"]["ml_robust_generalization_score"] == 0.56
    assert dashboard["research"]["ml_reinforcement_learning_status"] == (
        "DISABLED_PREMATURE_SAMPLE_SIZE"
    )
    assert dashboard["research"]["rejected_shadow_status"] == "GO"
    assert dashboard["research"]["rejected_episode_count"] == 40
    assert dashboard["research"]["counterfactual_performance_count"] == 12
    assert dashboard["research"]["gate_sample_ready_count"] == 0
    assert dashboard["research"]["automatic_gate_relaxation"] is False
