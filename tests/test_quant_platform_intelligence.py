from __future__ import annotations

import numpy as np
import pytest

from stocks.quant_platform import NewsIntelligenceEngine, SecFilingIntelligenceEngine, event_study


def test_news_engine_decomposes_mixed_earnings_message() -> None:
    engine = NewsIntelligenceEngine({"NVDA": ["NVIDIA"]})
    report = engine.analyze(
        "NVIDIA beats revenue estimates by 12% with strong demand, but guides next-quarter margins lower.",
        history=["NVIDIA announces a routine product update."],
    )
    assert report["entities"] == ["NVDA"]
    assert report["revenue_surprise"] == pytest.approx(0.12)
    assert report["guidance"] == "NEGATIVE"
    assert report["margin_outlook"] == "NEGATIVE"
    assert report["demand_commentary"] == "POSITIVE"
    assert report["simple_positive_negative_only"] is False
    assert report["execution_authority"] == "NONE"


def test_news_novelty_falls_for_near_duplicate() -> None:
    engine = NewsIntelligenceEngine()
    original = "Company raises full-year guidance after strong demand growth"
    unique = engine.analyze(original, history=["Federal Reserve holds interest rates"])["novelty"]
    duplicate = engine.analyze(original, history=[original])["novelty"]
    assert unique > duplicate


@pytest.mark.parametrize("form", ["10-K", "10-Q", "8-K", "4", "5", "13D", "13G", "13F", "S-1", "DEF 14A"])
def test_sec_engine_supports_required_filing_types(form: str) -> None:
    report = SecFilingIntelligenceEngine().analyze(
        form,
        "The company approved a share repurchase. Legal proceedings include litigation.",
        metadata={"transaction_code": "P", "accepted_at": "2026-01-01T12:00:00Z"},
    )
    assert report["features"]["buybacks"] == 1
    assert report["features"]["litigation"] >= 1
    assert report["broker_writes"] == 0


def test_sec_engine_detects_risk_factor_text_change_and_form4_direction() -> None:
    report = SecFilingIntelligenceEngine().analyze(
        "4",
        "Open market purchase acquired by reporting owner",
        metadata={"transaction_code": "P"},
        previous_text="Routine ownership statement",
    )
    assert report["features"]["insider_buying"] >= 2
    assert report["risk_factor_change"] > 0
    assert report["sec_signal"] > 0


def test_event_study_calculates_market_adjusted_event_windows() -> None:
    rng = np.random.default_rng(30)
    market = rng.normal(0, 0.01, 150)
    asset = 0.0001 + 1.2 * market + rng.normal(0, 0.002, 150)
    asset[100] += 0.08
    result = event_study(asset, market, event_position=100)
    assert set(result) == {-20, -5, 0, 1, 5, 20}
    assert result[0]["abnormal_return"] > 0.07
    assert result[5]["cumulative_abnormal_return"] is not None
