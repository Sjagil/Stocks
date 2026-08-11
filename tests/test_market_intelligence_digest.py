from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from stocks.macro.service import (
    _parse_ecb_monetary_calendar_text,
    _parse_fomc_calendar_text,
)
from stocks.notifications.market_digest import (
    _datascraper_rss_news,
    _event_study_refresh_reasons,
    build_market_intelligence_digest,
    format_market_intelligence_digest,
)


NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def test_event_study_refreshes_when_material_source_changes() -> None:
    reasons = _event_study_refresh_reasons(
        event_study={
            "status": "GO_WITH_DOCUMENTED_GAPS",
            "generated_at": NOW.isoformat(),
            "source_material_hash": "OLD",
        },
        material_events={"content_hash": "NEW"},
        observed_at=NOW,
    )

    assert reasons == ["MATERIAL_EVENT_SOURCE_HASH_CHANGED"]


def test_event_study_refreshes_after_twelve_hours() -> None:
    reasons = _event_study_refresh_reasons(
        event_study={
            "status": "GO_WITH_DOCUMENTED_GAPS",
            "generated_at": (NOW - timedelta(hours=13)).isoformat(),
            "source_material_hash": "SAME",
        },
        material_events={"content_hash": "SAME"},
        observed_at=NOW,
    )

    assert reasons == ["EVENT_STUDY_OLDER_THAN_12H"]


def test_current_event_study_does_not_refresh() -> None:
    reasons = _event_study_refresh_reasons(
        event_study={
            "status": "GO_WITH_DOCUMENTED_GAPS",
            "generated_at": (NOW - timedelta(hours=1)).isoformat(),
            "source_material_hash": "SAME",
        },
        material_events={"content_hash": "SAME"},
        observed_at=NOW,
    )

    assert reasons == []


def _calendar() -> dict[str, object]:
    return {
        "status": "GO",
        "future_schedule_status": "OFFICIAL_CALENDARS_AVAILABLE",
        "event_definitions": [
            {
                "event_id": "FOMC",
                "name": "Federal Reserve decision",
                "importance": "HIGH",
            }
        ],
        "scheduled_instances": [
            {
                "event_id": "FOMC",
                "name": "Federal Reserve interest-rate decision",
                "scheduled_at": "2026-07-28T18:00:00+00:00",
                "importance": "HIGH",
                "schedule_source": "FEDERAL_RESERVE_OFFICIAL_FOMC_CALENDAR",
                "affected_markets": [
                    "US_EQUITIES",
                    "USD",
                    "US_GOVERNMENT_BONDS",
                ],
            }
        ],
    }


def _news_fetcher(
    _root: Path,
    _symbols: tuple[str, ...],
    _now: datetime,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return (
        [
            {
                "date": "2026-07-28T10:00:00+00:00",
                "title": "AAPL raises earnings guidance after strong demand",
                "symbols": ["AAPL.US"],
                "sentiment": {"polarity": 0.8},
                "_provider": "EODHD",
            }
        ],
        {
            "status": "GO",
            "provider": "EODHD",
            "provider_requests": 1,
        },
    )


def test_digest_maps_news_and_official_event_without_authority(
    tmp_path: Path,
) -> None:
    report = build_market_intelligence_digest(
        tmp_path,
        now=NOW,
        news_fetcher=_news_fetcher,
        calendar_payload=_calendar(),
    )

    assert report["status"] == "GO"
    assert report["important_news_count"] == 1
    assert report["upcoming_macro_event_count"] == 1
    assert report["event_risk_within_24h"] is True
    news = report["important_news"][0]
    assert news["importance"] == "HIGH"
    assert news["direction"] == "POSITIVE_INFERENCE"
    assert "US_EQUITIES" in news["affected_markets"]
    assert report["execution_authority"] == "NONE"
    assert report["broker_calls"] == 0
    assert report["order_calls"] == 0
    artifact = (
        tmp_path
        / "output"
        / "notifications"
        / "market-intelligence-digest.json"
    )
    assert artifact.exists()
    assert "api_token" not in artifact.read_text(encoding="utf-8")


def test_digest_never_labels_stale_empty_archive_as_current(
    tmp_path: Path,
) -> None:
    def empty(
        _root: Path,
        _symbols: tuple[str, ...],
        _now: datetime,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        return [], {"status": "PROVIDER_UNAVAILABLE", "provider_requests": 1}

    report = build_market_intelligence_digest(
        tmp_path,
        now=NOW,
        news_fetcher=empty,
        calendar_payload={"status": "UNAVAILABLE", "scheduled_instances": []},
    )

    assert report["status"] == "DATA_INCOMPLETE"
    assert report["news_freshness_status"] == "CURRENT_NEWS_UNAVAILABLE"
    assert report["important_news"] == []
    message = format_market_intelligence_digest(report)
    assert "stale archive is niet als actueel gebruikt" in message
    assert "Automatische execution: uit." in message


def test_fomc_parser_uses_decision_day_and_eastern_time() -> None:
    text = (
        "2026 FOMC Meetings July 28-29 "
        "September 15-16* 2025 FOMC Meetings"
    )
    rows = _parse_fomc_calendar_text(
        text,
        now=NOW,
        source_url="https://www.federalreserve.gov/test",
    )

    assert [row["scheduled_at"] for row in rows] == [
        "2026-07-29T18:00:00+00:00",
        "2026-09-16T18:00:00+00:00",
    ]
    assert rows[1]["projection_meeting"] is True
    assert rows[0]["automatic_exit"] is False


def test_ecb_parser_keeps_only_monetary_policy_decision_day() -> None:
    text = (
        "09/09/2026 Governing Council of the ECB: monetary policy meeting "
        "(Day 1) "
        "10/09/2026 Governing Council of the ECB: monetary policy meeting "
        "(Day 2), followed by press conference "
        "30/09/2026 Governing Council of the ECB: non-monetary policy meeting"
    )
    rows = _parse_ecb_monetary_calendar_text(
        text,
        now=NOW,
        source_url="https://www.ecb.europa.eu/test",
    )

    assert len(rows) == 1
    assert rows[0]["event_id"] == "ECB"
    assert rows[0]["scheduled_at"] == "2026-09-10T12:15:00+00:00"
    assert "EUR" in rows[0]["affected_markets"]


def test_digest_content_hash_is_deterministic_for_fixed_inputs(
    tmp_path: Path,
) -> None:
    first = build_market_intelligence_digest(
        tmp_path,
        now=NOW,
        news_fetcher=_news_fetcher,
        calendar_payload=_calendar(),
    )
    second = build_market_intelligence_digest(
        tmp_path,
        now=NOW,
        news_fetcher=_news_fetcher,
        calendar_payload=_calendar(),
    )

    assert first["content_hash"] == second["content_hash"]
    public = json.loads(
        (
            tmp_path
            / "output"
            / "notifications"
            / "market-intelligence-digest.json"
        ).read_text(encoding="utf-8")
    )
    assert public["content_hash"] == first["content_hash"]


def test_digest_includes_public_portfolio_actions_without_financial_values(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output/portfolio"
    output.mkdir(parents=True)
    (output / "active_portfolio_plan.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "engine_status": {
                    "generated_at": "2026-07-28T11:00:00+00:00"
                },
                "target_allocation": {
                    "research_target_exposure": 0.2,
                    "approved_target_exposure": 0.0,
                },
                "risk": {
                    "portfolio_heat_gate": "GO",
                    "correlation_gate": "GO",
                },
                "exposures": {"research_cash_weight": 0.8},
                "opportunities": {
                    "opportunities": [
                        {
                            "ticker": "AAPL",
                            "opportunity_score": 0.8,
                            "timeframes": ["1h", "4h", "1d"],
                            "strategy_families": ["rsi", "breakout"],
                            "strategy_ids": [
                                "P1114-SURVIVOR",
                                "BULK-RESEARCH",
                            ],
                            "execution_blockers": [],
                        }
                    ]
                },
                "position_actions": {
                    "actions": [
                        {
                            "ticker": "ON",
                            "advisory_action": "EXIT",
                            "reason_codes": ["RISK_GATE"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    report = build_market_intelligence_digest(
        tmp_path,
        now=NOW,
        news_fetcher=_news_fetcher,
        calendar_payload=_calendar(),
    )
    message = format_market_intelligence_digest(report)

    assert "AAPL score 0.800" in message
    assert "survivor support 1" in message
    assert "ON: EXIT" in message
    assert (
        report["portfolio_decision"]["top_opportunities"][0][
            "survivor_strategy_count"
        ]
        == 1
    )
    assert report["portfolio_decision"]["approved_exposure_pct"] == 0.0
    assert report["portfolio_decision"]["portfolio_heat_gate"] == "GO"
    assert report["portfolio_decision"]["correlation_gate"] == "GO"
    assert report["portfolio_decision"]["research_cash_pct"] == 0.8
    assert report["portfolio_decision"]["financial_values_included"] is False


def test_datascraper_rss_uses_only_current_public_non_crypto_rows(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "news_context_v1.jsonl"
    rows = [
        {
            "authority": "FORWARD_ONLY_PUBLIC_RSS",
            "endpoint_family": "rss_feed",
            "published_at": "2026-07-28T10:30:00+00:00",
            "source": "official_market_rss",
            "payload": {
                "title": "Federal Reserve decision affects global markets",
                "url": "https://example.test/fed",
                "entities": "SPY,GLD",
            },
        },
        {
            "authority": "FORWARD_ONLY_PUBLIC_RSS",
            "endpoint_family": "rss_feed",
            "published_at": "2026-07-28T10:31:00+00:00",
            "source": "kraken_blog_rss",
            "payload": {
                "title": "New crypto token listed",
                "url": "https://example.test/token",
            },
        },
        {
            "authority": "FORWARD_ONLY_PUBLIC_RSS",
            "endpoint_family": "rss_feed",
            "published_at": "2026-07-28T10:31:30+00:00",
            "source": "coindesk_rss",
            "payload": {
                "title": "Digital asset market update",
                "url": "https://example.test/digital-assets",
            },
        },
        {
            "authority": "FORWARD_ONLY_PAID_PROVIDER_API",
            "endpoint_family": "news",
            "published_at": "2026-07-28T10:32:00+00:00",
            "source": "paid_news",
            "payload": {"title": "Not an RSS row"},
        },
    ]
    ledger.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    news, status = _datascraper_rss_news(now=NOW, path=ledger)

    assert [row["title"] for row in news] == [
        "Federal Reserve decision affects global markets"
    ]
    assert news[0]["symbols"] == ["SPY", "GLD"]
    assert status["status"] == "GO"
    assert status["record_count"] == 1
    assert status["rejected_crypto_only_count"] == 2
    assert status["execution_authority"] == "NONE"


def test_digest_publishes_context_without_predictive_or_execution_claim(
    tmp_path: Path,
) -> None:
    macro_root = tmp_path / "output/macro"
    dynamic_root = tmp_path / "output/dynamic"
    macro_root.mkdir(parents=True)
    dynamic_root.mkdir(parents=True)
    (macro_root / "regime.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "regime": {
                    "overall_macro_regime": "EXPANSION_DISINFLATION",
                    "market_regime": "RISK_ON",
                },
                "cycle_clock": {"confidence": 0.7},
                "data_quality": {"status": "GO"},
            }
        ),
        encoding="utf-8",
    )
    (dynamic_root / "current_regime.json").write_text(
        json.dumps(
            {
                "status": "GO",
                "regime": "BULL_TREND_LOW_VOL",
                "inputs": {"drawdown": -0.04},
            }
        ),
        encoding="utf-8",
    )
    (dynamic_root / "status.json").write_text(
        json.dumps(
            {
                "active_timeframes": ["1d", "1w"],
                "observed_timeframes": ["1d", "4h", "1w"],
                "blocked_timeframes": {
                    "1h": "NO_STRATEGY_SURVIVED_50BPS_AND_SAMPLE_GATES"
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_market_intelligence_digest(
        tmp_path,
        now=NOW,
        news_fetcher=_news_fetcher,
        calendar_payload=_calendar(),
    )

    context = report["market_context"]
    assert context["status"] == "GO"
    assert context["technical_regime"] == "BULL_TREND_LOW_VOL"
    assert context["macro_regime"] == "EXPANSION_DISINFLATION"
    assert context["technical_benchmark_drawdown"] == -0.04
    assert context["active_timeframes"] == ["1d", "1w"]
    assert context["observed_timeframes"] == ["1d", "4h", "1w"]
    assert context["blocked_timeframes"]["1h"].startswith("NO_STRATEGY")
    assert context["predictive_claim"] is False
    assert context["execution_authority"] == "NONE"


def test_digest_includes_fresh_frontier_context_without_entry_authority(
    tmp_path: Path,
) -> None:
    theme_root = tmp_path / "output/analysis/themes"
    theme_root.mkdir(parents=True)
    (theme_root / "frontier-technology-energy.json").write_text(
        json.dumps(
            {
                "status": "GO_WITH_DOCUMENTED_GAPS",
                "themes": {
                    "quantum_computing": {
                        "sector_structure": {"status": "PLATFORM_LED"}
                    },
                    "nuclear_uranium": {
                        "sector_structure": {"status": "DIVERGENT_OR_WEAK"}
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (theme_root / "opening-session-watchplan.json").write_text(
        json.dumps(
            {
                "status": "GO_WITH_BLOCKERS",
                "generated_at": NOW.isoformat(),
                "theme_decision_matrix": {
                    "quantum_computing": {
                        "confirmation_status": "PARTIAL_CONTEXT_WAIT",
                        "decision_state": "CURRENT_SETUP_BLOCKED",
                        "leadership_symbols": ["NVDA", "MSFT", "ARQQ"],
                        "instrument_count": 9,
                        "current_shariah_eligible_count": 1,
                        "ready_observation_count": 0,
                        "risk_flags": ["REGIME_CONFLICT_RISK_REDUCING"],
                    },
                    "nuclear_uranium": {
                        "confirmation_status": "UNCONFIRMED_CONTEXT_WAIT",
                        "decision_state": "CURRENT_SETUP_BLOCKED",
                        "leadership_symbols": ["LEU", "NXE", "DNN"],
                        "instrument_count": 15,
                        "current_shariah_eligible_count": 0,
                        "ready_observation_count": 0,
                        "risk_flags": ["THEME_BREADTH_NOT_CONFIRMED"],
                    },
                },
                "rows": [
                    {
                        "symbol": "QUBT",
                        "theme": "quantum_computing",
                        "event_risk_status": "EVENT_RISK_IMMINENT",
                    },
                    {
                        "symbol": "DNN",
                        "theme": "nuclear_uranium",
                        "event_risk_status": "EVENT_RISK_NEAR",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_market_intelligence_digest(
        tmp_path,
        now=NOW,
        news_fetcher=_news_fetcher,
        calendar_payload=_calendar(),
    )
    frontier = report["frontier_theme_context"]
    message = format_market_intelligence_digest(report)

    assert frontier["status"] == "GO_WITH_BLOCKERS"
    assert frontier["themes"][0]["leaders"] == ["NVDA", "MSFT", "ARQQ"]
    assert frontier["themes"][0]["ready_observation_count"] == 0
    assert frontier["themes"][0]["standalone_entry_allowed"] is False
    assert frontier["execution_authority"] == "NONE"
    assert "Quantum: PLATFORM_LED / PARTIAL_CONTEXT_WAIT" in message
    assert "Nuclear/uranium: DIVERGENT_OR_WEAK" in message
    assert "QUBT:EVENT_RISK_IMMINENT" in message
