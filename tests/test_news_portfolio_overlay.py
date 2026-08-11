from __future__ import annotations

import json
from pathlib import Path

import pytest

from stocks.portfolio.manager import (
    _news_event_overlay_map,
    rank_opportunities,
)


def _policy() -> dict[str, object]:
    path = (
        Path(__file__).parents[1]
        / "config"
        / "portfolio"
        / "active_manager_v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _signal() -> dict[str, object]:
    return {
        "ticker": "AAPL",
        "strategy_id": "TECHNICAL-SETUP-1",
        "entry_strategy": "quality_momentum",
        "timeframe": "4h",
        "action": "BUY",
        "confidence_score": "0.80",
        "data_freshness": "FRESH",
        "data_timestamp": "2026-08-08T16:00:00+00:00",
        "preferred_entry": "100",
        "stop_loss": "95",
        "take_profit_1": "110",
        "take_profit_2": "115",
        "stop_distance_pct": "0.05",
        "currency": "USD",
        "reasons": ["POSITIVE_RELATIVE_MOMENTUM"],
        "_portfolio_eligible_source": True,
    }


def _rank(context: dict[str, dict[str, object]]) -> dict[str, object]:
    rows = rank_opportunities(
        [_signal()],
        policy=_policy(),
        contracts={"AAPL": {"con_id": 1, "currency": "USD"}},
        fundamentals={
            "AAPL": {
                "fundamental_score": 80,
                "liquidity_score": 100,
                "sector": "Technology",
                "shariah_status": "SHARIAH_ELIGIBLE_PIT",
            }
        },
        family_map={},
        macro={"regime": {"market_regime": "RISK_ON"}},
        technical_regime={"regime": "BULL_TREND_LOW_VOL"},
        news={},
        dynamic_overrides={},
        news_event_context=context,
    )
    assert len(rows) == 1
    return rows[0]


def test_news_overlay_caps_cumulative_adjustment_and_has_no_authority() -> None:
    payload = {
        "rows": [
            {
                "story_cluster_id": f"STORY-{index}",
                "symbols": ["AAPL"],
                "raw_impact": 1.0,
                "materiality": 1.0,
                "event_classes": ["GUIDANCE_RAISE"],
                "hard_risk_flags": [],
            }
            for index in range(10)
        ]
    }

    context = _news_event_overlay_map(payload, policy=_policy())

    assert context["AAPL"]["ranking_adjustment"] == pytest.approx(0.04)
    assert context["AAPL"]["standalone_entry_allowed"] is False
    assert context["AAPL"]["strategy_authority"] == "NONE"
    assert context["AAPL"]["execution_authority"] == "NONE"


def test_news_overlay_only_adjusts_existing_technical_setup() -> None:
    context = {
        "AAPL": {
            "event_count": 2,
            "ranking_adjustment": 0.04,
            "hard_risk_flags": [],
            "standalone_entry_allowed": False,
            "execution_authority": "NONE",
        }
    }

    candidate = _rank(context)

    assert candidate["news_score_adjustment"] == pytest.approx(0.04)
    assert candidate["opportunity_score"] == pytest.approx(
        candidate["opportunity_score_before_news"] + 0.04
    )
    assert candidate["news_event_context"][
        "standalone_entry_allowed"
    ] is False
    assert candidate["deployment_eligible"] is False
    assert "EXECUTION_AUTHORITY_NONE" in candidate["deployment_blockers"]


def test_material_news_risk_blocks_allocation_without_creating_order() -> None:
    context = {
        "AAPL": {
            "event_count": 1,
            "ranking_adjustment": -0.015,
            "hard_risk_flags": ["FRAUD_OR_ACCOUNTING_RISK"],
            "standalone_entry_allowed": False,
            "execution_authority": "NONE",
        }
    }

    candidate = _rank(context)

    assert candidate["news_score_adjustment"] == pytest.approx(-0.015)
    assert "MATERIAL_NEWS_RISK_REVIEW_REQUIRED" in candidate[
        "research_allocation_blockers"
    ]
    assert candidate["research_allocation_eligible"] is False
    assert candidate["deployment_eligible"] is False

