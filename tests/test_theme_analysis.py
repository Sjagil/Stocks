from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from stocks.analysis.themes import _load_bars, build_frontier_theme_analysis


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bars(root: Path, symbol: str, interval: str, now: datetime) -> None:
    periods = 260
    timestamps = pd.date_range(
        end=now - timedelta(hours=1),
        periods=periods,
        freq="h" if interval == "1h" else "D",
        tz="UTC",
    )
    close = pd.Series(np.linspace(10.0, 20.0, periods))
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 100_000.0,
            "is_partial": False,
        }
    )
    path = (
        root
        / "data/research/multitimeframe/private"
        / "provider=TEST"
        / f"symbol={symbol}"
        / f"interval={interval}"
        / f"source_interval={interval}"
        / "bars.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _provider_bars(
    root: Path,
    *,
    provider: str,
    symbol: str,
    interval: str,
    periods: int,
    end: datetime,
) -> None:
    timestamps = pd.date_range(end=end, periods=periods, freq="D", tz="UTC")
    close = pd.Series(np.linspace(10.0, 20.0, periods))
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 100_000.0,
            "is_partial": False,
        }
    )
    path = (
        root
        / "data/research/multitimeframe/private"
        / f"provider={provider}"
        / f"symbol={symbol}"
        / f"interval={interval}"
        / f"source_interval={interval}"
        / "bars.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _fixture(root: Path, now: datetime) -> None:
    _write_json(
        root / "config/themes/frontier_technology_energy_v1.json",
        {
            "version": "TEST",
            "themes": {
                "quantum_computing": {
                    "label": "Quantum",
                    "macro_sensitivities": ["LIQUIDITY"],
                    "instruments": [
                        {
                            "symbol": "IONQ",
                            "subtheme": "pure_play",
                            "business_maturity": "HIGH_RISK",
                        }
                    ],
                }
            },
            "authority_contract": {
                "order_generation": False,
            },
        },
    )
    for interval in ("1h", "4h", "1d", "1w", "1mo"):
        _bars(root, "IONQ", interval, now)
    _write_json(
        root / "data/news/private/current-news.json",
        {
            "rows": [
                {
                    "published_at": now.isoformat(),
                    "source": "TEST_RSS",
                    "title": "IONQ test event",
                    "symbols": ["IONQ"],
                }
            ]
        },
    )
    _write_json(
        root / "output/macro/status.json",
        {
            "status": "GO",
            "latest_regime": "SLOWDOWN_DISINFLATION",
            "latest_data_quality": "GO",
        },
    )
    _write_json(
        root / "output/macro/score.json",
        {
            "as_of": now.isoformat(),
            "data_quality": {"status": "GO"},
            "scores": {
                "growth": {"value": 20.0, "confidence": 0.8, "status": "VALID"},
                "liquidity": {"value": 40.0, "confidence": 0.9, "status": "VALID"},
                "credit": {"value": 15.0, "confidence": 0.8, "status": "VALID"},
                "risk_appetite": {"value": 30.0, "confidence": 0.9, "status": "VALID"},
                "monetary": {"value": 10.0, "confidence": 0.8, "status": "VALID"},
                "currency": {"value": 5.0, "confidence": 0.8, "status": "VALID"},
                "earnings_cycle": {"value": 20.0, "confidence": 0.7, "status": "VALID"},
                "valuation": {"value": -10.0, "confidence": 0.7, "status": "VALID"},
            },
        },
    )
    _write_json(
        root / "config/context/asset_transmission_v1.json",
        {
            "schema": "asset_context_transmission_v1",
            "groups": {
                "technology_equity": {
                    "sensitivities": {
                        "growth": 0.75,
                        "liquidity": 0.9,
                        "credit": 0.35,
                        "risk_appetite": 0.95,
                        "monetary": 0.55,
                        "currency": 0.3,
                        "earnings_cycle": 0.75,
                    }
                },
                "industrial_equity": {"sensitivities": {"growth": 0.9}},
                "utility_equity": {"sensitivities": {"credit": 0.55}},
                "uranium": {"sensitivities": {"commodity": 0.7}},
            },
            "symbols": {},
        },
    )
    _write_json(
        root / "output/signals/latest_signals.json",
        {
            "signals": [
                {
                    "signal_id": "SIG-IONQ-1",
                    "ticker": "IONQ",
                    "strategy_id": "TEST-STRATEGY",
                    "timeframe": "4h",
                    "action": "AVOID",
                    "lifecycle_status": "INVALIDATED",
                    "execution_eligible": False,
                }
            ]
        },
    )
    _write_json(
        root / "output/analysis/themes/contract-coverage.json",
        {
            "status": "GO",
            "results": [
                {
                    "research_symbol": "IONQ",
                    "status": "RESOLVED",
                    "source": "TEST_CACHE",
                    "cache_hit": True,
                    "returned_match_count": 1,
                    "contract_identity": {
                        "con_id": 123,
                        "symbol": "IONQ",
                        "security_type": "STK",
                        "currency": "USD",
                    },
                }
            ],
        },
    )
    _write_json(
        root / "output/market_context/entry-shortlist.json",
        {
            "observations": [
                {
                    "symbol": "IONQ",
                    "signal_id": "SIG-IONQ-1",
                    "strategy_id": "TEST-STRATEGY",
                    "timeframe": "4h",
                    "state": "DIRECTIONAL_BIAS_ONLY_4H_SETUP_PENDING",
                    "decision_contract": {
                        "hard_veto_pass": True,
                        "hard_vetoes": [],
                        "soft_vetoes": [],
                        "contract_identity": {"status": "RESOLVED"},
                        "gates": {
                            "timeframe_hierarchy_ready": False,
                            "observed_tape_available": False,
                            "observed_depth_available": False,
                        },
                    },
                }
            ]
        },
    )
    _write_json(
        root / "output/analysis/themes/shariah-coverage.json",
        {
            "status": "GO",
            "instruments": [
                {
                    "symbol": "IONQ",
                    "status": "SHARIAH_ELIGIBLE_PIT",
                    "currently_eligible": True,
                    "screened_at": "2026-08-01T00:00:00+00:00",
                    "expires_at": "2026-09-01T00:00:00+00:00",
                    "methodology": "TEST_REVIEW",
                    "source": "TEST",
                }
            ],
        },
    )
    _write_json(
        root / "output/analysis/themes/event-risk-calendar.json",
        {
            "status": "GO",
            "rows": [
                {
                    "symbol": "IONQ",
                    "event_risk_status": "EVENT_CLEAR",
                    "next_earnings_date": "2026-11-04",
                    "days_to_event": 88,
                    "source_confidence": "SINGLE_PROVIDER_ESTIMATE",
                    "macro_event_risk_status": "MACRO_EVENT_CLEAR",
                    "hard_block_recommended": False,
                    "soft_penalty_recommended": False,
                    "authority": "RISK_CONTEXT_ONLY",
                }
            ],
        },
    )
    _write_json(
        root / "output/market_context/entry-episode-completeness.json",
        {
            "status": "GO",
            "episode_count": 10,
            "terminal_episode_count": 9,
            "pending_episode_count": 1,
            "completion_ratio": 0.9,
            "feature_mutation_count": 0,
            "duplicate_terminal_count": 0,
        },
    )
    database = root / "data/research/phase11_3/private/causal_research.sqlite3"
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE records(dataset TEXT, payload_json TEXT)"
        )
        connection.execute(
            "INSERT INTO records(dataset,payload_json) VALUES(?,?)",
            (
                "filings",
                json.dumps(
                    {
                        "symbol": "IONQ",
                        "accepted_at": "2026-08-01T00:00:00Z",
                    }
                ),
            ),
        )


def test_theme_analysis_uses_closed_bars_and_remains_research_only(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    _fixture(tmp_path, now)

    report = build_frontier_theme_analysis(tmp_path, as_of=now)

    theme = report["themes"]["quantum_computing"]
    instrument = theme["instruments"][0]
    assert report["status"] == "GO_WITH_DOCUMENTED_GAPS"
    assert instrument["technical_score"] > 0
    assert instrument["contextual_conviction"]["score"] > 0
    assert instrument["contextual_conviction"]["not_an_entry_signal"] is True
    assert instrument["macro_alignment"]["profile"] == "QUANTUM_EARLY_STAGE_GROWTH"
    assert instrument["macro_alignment"]["classification"] == "SUPPORTIVE"
    assert instrument["macro_alignment"]["standalone_entry_allowed"] is False
    assert instrument["news"]["fresh_event_count"] == 1
    assert instrument["fundamentals"]["status"] == "AVAILABLE"
    assert instrument["contract"]["status"] == "RESOLVED"
    assert instrument["event_risk"]["event_risk_status"] == "EVENT_CLEAR"
    assert theme["contract_coverage_ratio"] == 1.0
    assert instrument["current_forward_observations"][
        "hard_veto_pass_count"
    ] == 1
    assert instrument["shariah_status"] == "SHARIAH_ELIGIBLE_PIT"
    assert theme["current_shariah_coverage_ratio"] == 1.0
    assert theme["forward_observation_coverage"][
        "observed_symbol_count"
    ] == 1
    assert theme["sector_structure"]["status"] == "INSUFFICIENT_DATA"
    assert theme["sector_structure"]["standalone_entry_allowed"] is False
    assert report["forward_evidence"]["feature_mutation_count"] == 0
    assert instrument["current_strategy_setups"]["observed_count"] == 1
    assert instrument["current_strategy_setups"]["valid_setup_count"] == 0
    assert instrument["automatic_execution_eligible"] is False
    assert report["authority"]["execution_authority"] == "NONE"
    assert report["authority"]["orders_generated"] == 0
    assert (
        tmp_path
        / "output/analysis/themes/frontier-technology-energy.json"
    ).is_file()


def test_bar_source_prefers_sufficient_history_before_freshness(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    _provider_bars(
        tmp_path,
        provider="FRESH_SHORT",
        symbol="NVDA",
        interval="1d",
        periods=29,
        end=now - timedelta(hours=1),
    )
    _provider_bars(
        tmp_path,
        provider="COMPLETE_HISTORY",
        symbol="NVDA",
        interval="1d",
        periods=260,
        end=now - timedelta(days=1),
    )

    selected, provenance = _load_bars(tmp_path, "NVDA", "1d")

    assert len(selected) == 260
    assert provenance["provider"] == "COMPLETE_HISTORY"
    assert provenance["provider_candidate_count"] == 2
    assert provenance["selected_profile_rows"] == 260
    assert provenance["selection_reason"] == (
        "SUFFICIENT_HISTORY_THEN_FRESHNESS_THEN_ROWS"
    )


def test_missing_fundamentals_and_unknown_theme_fail_closed(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    _fixture(tmp_path, now)
    database = tmp_path / "data/research/phase11_3/private/causal_research.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM records")

    report = build_frontier_theme_analysis(tmp_path, as_of=now)
    blocked = build_frontier_theme_analysis(
        tmp_path,
        theme="not_configured",
        as_of=now,
    )

    assert report["status"] == "GO_WITH_DOCUMENTED_GAPS"
    assert "FUNDAMENTALS_UNAVAILABLE:IONQ" in report["themes"][
        "quantum_computing"
    ]["documented_gaps"]
    assert blocked["status"] == "BLOCKED"
    assert blocked["execution_authority"] == "NONE"


def test_theme_macro_transmission_is_profile_specific_and_uncertainty_shrunk(
) -> None:
    from stocks.analysis.themes import _macro_alignment

    macro = {
        "data_quality": "DATA_INCOMPLETE",
        "transmission_source": "config/context/asset_transmission_v1.json",
        "transmission_source_hash": "HASH",
        "transmission_profiles": {
            "technology_equity": {
                "growth": 0.75,
                "liquidity": 0.9,
                "risk_appetite": 0.95,
            },
            "industrial_equity": {"growth": 0.9, "credit": 0.55},
            "utility_equity": {"credit": 0.55, "monetary": 0.75},
            "uranium": {
                "growth": 0.35,
                "commodity": 0.7,
                "risk_appetite": 0.45,
                "energy_policy": 0.95,
            },
        },
        "family_scores": {
            "growth": {"value": 0.2, "confidence": 0.8, "status": "VALID"},
            "liquidity": {"value": 0.4, "confidence": 0.9, "status": "VALID"},
            "risk_appetite": {"value": 0.3, "confidence": 0.9, "status": "VALID"},
            "credit": {"value": 0.2, "confidence": 0.8, "status": "VALID"},
            "monetary": {"value": 0.1, "confidence": 0.8, "status": "VALID"},
            "commodity": {"value": -0.5, "confidence": 1.0, "status": "VALID"},
        },
    }

    quantum = _macro_alignment(
        "quantum_computing",
        macro,
        {"subtheme": "pure_play", "business_maturity": "EARLY_STAGE_HIGH_RISK"},
    )
    uranium = _macro_alignment(
        "nuclear_uranium",
        macro,
        {"subtheme": "fuel_cycle", "business_maturity": "OPERATIONAL_PRODUCER"},
    )

    assert quantum["profile"] == "QUANTUM_EARLY_STAGE_GROWTH"
    assert quantum["score"] > 0.5
    assert quantum["standalone_entry_allowed"] is False
    assert uranium["profile"] == "URANIUM_COMMODITY_CHAIN"
    assert uranium["raw_transmission_score"] < 0
    assert uranium["status"] == "DEGRADED_CONTEXT_ONLY"
    assert "energy_policy" in uranium["missing_components"]
    assert abs(uranium["uncertainty_shrunk_score"]) < abs(
        uranium["raw_transmission_score"]
    )


def test_official_nuclear_policy_context_fills_energy_policy_component() -> None:
    from stocks.analysis.themes import _macro_alignment
    from stocks.analysis.themes import _macro_with_theme_policy_context

    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    macro = {
        "data_quality": "DATA_INCOMPLETE",
        "transmission_profiles": {
            "uranium": {"commodity": 0.7, "energy_policy": 0.95},
        },
        "family_scores": {
            "commodity": {
                "value": -0.2,
                "confidence": 1.0,
                "status": "VALID",
            }
        },
    }
    news = [
        {
            "event_hash": "NRC-1",
            "published_at": "2026-08-07T08:00:00+00:00",
            "source": "NRC",
            "source_class": "OFFICIAL_PUBLIC_RSS",
            "event_type": "REGULATORY_OR_LICENSING",
            "direction": "POSITIVE_CONTEXT",
            "importance": "HIGH",
        },
        {
            "event_hash": "DOE-1",
            "published_at": "2026-08-06T08:00:00+00:00",
            "source": "DOE",
            "source_class": "OFFICIAL_PUBLIC_RSS",
            "event_type": "TECHNOLOGY_OR_PROJECT_MILESTONE",
            "direction": "NEUTRAL_OR_UNCERTAIN",
            "importance": "MEDIUM",
        },
    ]

    themed = _macro_with_theme_policy_context(
        macro,
        theme_id="nuclear_uranium",
        theme_news=news,
        as_of=now,
    )
    alignment = _macro_alignment(
        "nuclear_uranium",
        themed,
        {"subtheme": "fuel_cycle"},
    )

    policy = themed["theme_policy_context"]
    assert policy["status"] == "AVAILABLE_OFFICIAL_CURRENT_CONTEXT"
    assert policy["event_count"] == 2
    assert policy["source_count"] == 2
    assert 0 < policy["confidence"] <= 0.75
    assert policy["standalone_entry_allowed"] is False
    assert "energy_policy" not in alignment["missing_components"]
    component = next(
        row for row in alignment["components"]
        if row["family"] == "energy_policy"
    )
    assert component["status"] == "CURRENT_OFFICIAL_CONTEXT"


def test_licensed_or_future_news_cannot_fill_energy_policy() -> None:
    from stocks.analysis.themes import _macro_with_theme_policy_context

    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    macro = {"family_scores": {}}
    themed = _macro_with_theme_policy_context(
        macro,
        theme_id="nuclear_uranium",
        theme_news=[
            {
                "published_at": "2026-08-07T08:00:00+00:00",
                "source": "WIRE",
                "source_class": "LICENSED_NEWS_AGGREGATOR",
                "event_type": "POLICY_OR_PUBLIC_FUNDING",
                "direction": "POSITIVE",
                "importance": "HIGH",
            },
            {
                "published_at": "2026-08-09T08:00:00+00:00",
                "source": "NRC",
                "source_class": "OFFICIAL_PUBLIC_RSS",
                "event_type": "REGULATORY_OR_LICENSING",
                "direction": "POSITIVE_CONTEXT",
                "importance": "HIGH",
            },
        ],
        as_of=now,
    )

    assert themed["theme_policy_context"]["status"] == "UNAVAILABLE"
    assert "energy_policy" not in themed["family_scores"]


def _structure_row(
    symbol: str,
    subtheme: str,
    *,
    return_63: float,
    distance_sma_50: float,
) -> dict:
    return {
        "symbol": symbol,
        "subtheme": subtheme,
        "daily_snapshot": {
            "return_20_bars": return_63 / 2,
            "return_63_bars": return_63,
            "distance_sma_50": distance_sma_50,
        },
    }


def test_nuclear_structure_requires_physical_fund_and_producer_confirmation(
) -> None:
    from stocks.analysis.themes import _theme_sector_structure

    rows = [
        _structure_row(
            "U-UN.TO", "physical_uranium", return_63=0.10,
            distance_sma_50=0.05,
        ),
        _structure_row(
            "URA", "uranium_fund", return_63=0.08,
            distance_sma_50=0.04,
        ),
        _structure_row(
            "URNM", "uranium_fund", return_63=0.06,
            distance_sma_50=0.03,
        ),
        _structure_row(
            "CCJ", "fuel_cycle", return_63=0.12,
            distance_sma_50=0.06,
        ),
        _structure_row(
            "LEU", "fuel_cycle", return_63=0.09,
            distance_sma_50=0.04,
        ),
        _structure_row(
            "NXE", "fuel_cycle", return_63=-0.02,
            distance_sma_50=-0.01,
        ),
    ]

    structure = _theme_sector_structure("nuclear_uranium", rows)

    assert structure["status"] == "PHYSICAL_MINER_CONFIRMATION"
    assert structure["confirmation_component_count"] == 3
    assert structure["standalone_entry_allowed"] is False


def test_quantum_structure_distinguishes_platform_led_breadth() -> None:
    from stocks.analysis.themes import _theme_sector_structure

    rows = [
        _structure_row(
            "IONQ", "pure_play", return_63=-0.10,
            distance_sma_50=-0.04,
        ),
        _structure_row(
            "RGTI", "pure_play", return_63=-0.05,
            distance_sma_50=-0.03,
        ),
        _structure_row(
            "MSFT", "platform_enabler", return_63=0.08,
            distance_sma_50=0.03,
        ),
        _structure_row(
            "IBM", "platform_enabler", return_63=0.05,
            distance_sma_50=0.02,
        ),
    ]

    structure = _theme_sector_structure("quantum_computing", rows)

    assert structure["status"] == "PLATFORM_LED"
    assert structure["pure_play_breadth_positive"] is False
    assert structure["platform_breadth_positive"] is True
    assert structure["standalone_entry_allowed"] is False


def test_news_catalyst_summary_shrinks_single_source_optimism() -> None:
    from stocks.analysis.themes import _news_catalyst_summary

    rows = [
        {
            "source": "ONE_AGGREGATOR",
            "source_class": "LICENSED_NEWS_AGGREGATOR",
            "direction": "POSITIVE_CONTEXT",
            "importance": "HIGH",
            "event_type": "ANALYST_OR_VALUATION_COMMENTARY",
        }
        for _ in range(10)
    ]

    summary = _news_catalyst_summary(rows)

    assert summary["evidence_quality"] == "SOURCE_CONCENTRATED"
    assert summary["directional_balance_before_shrinkage"] == 1.0
    assert summary["directional_balance"] < 0.34
    assert summary["bounded_context_score"] <= 0.55
    assert summary["standalone_entry_allowed"] is False


def test_fundamental_quality_excludes_flagged_extreme_ratios() -> None:
    from stocks.analysis.themes import _fundamental_quality

    quality = _fundamental_quality(
        {
            "status": "AVAILABLE",
            "fundamentals_required": True,
            "annual_revenue_growth": 9.0,
            "annual_net_margin": -20.0,
            "cash_to_assets": 0.6,
            "debt_to_assets": 0.1,
            "data_quality": {
                "status": "REVIEW_REQUIRED",
                "anomalous_metric_fields": [
                    "annual_revenue_growth",
                    "annual_net_margin",
                ],
            },
        }
    )

    assert quality["status"] == "LIMITED_RESEARCH_CONTEXT"
    assert set(quality["excluded_anomalous_metric_fields"]) == {
        "annual_revenue_growth",
        "annual_net_margin",
    }
    assert quality["components"] == {"balance_sheet": 1.0}
    assert quality["uncertainty_shrinkage_factor"] == 0.25
