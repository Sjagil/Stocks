from __future__ import annotations

import numpy as np
import pandas as pd

from stocks.dynamic.service import (
    FAMILY_MATRIX,
    REGIMES,
    _consensus,
    _family_risk_policy,
    _portfolio,
    capped_weights,
    classify_regime,
    strategy_score,
)
from stocks.dynamic.strategy_allocation import (
    bayesian_positive_probability,
    score_strategy_evidence,
)
from stocks.signals.service import _strategy_signal
from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger


def _survivor(family: str = "ma_crossover", pf: float = 1.25) -> dict[str, object]:
    return {
        "candidate_id": f"TEST-{family}",
        "family": family,
        "profit_factor": pf,
        "parameters": '{"fast": 50, "slow": 200}',
        "classification": "FROZEN_SHADOW",
    }


def test_all_regimes_have_matrix_semantics() -> None:
    assert set(REGIMES) == {
        "BULL_TREND_LOW_VOL", "BULL_TREND_HIGH_VOL", "SIDEWAYS_LOW_VOL",
        "SIDEWAYS_HIGH_VOL", "BEAR_TREND", "CRISIS", "RECOVERY",
        "INFLATIONARY_COMMODITY", "DEFENSIVE", "UNKNOWN",
    }
    assert all(FAMILY_MATRIX[family] for family in FAMILY_MATRIX)


def test_regime_is_causal_and_bounded() -> None:
    index = pd.date_range("2024-01-01", periods=260, freq="B")
    close = pd.Series(np.linspace(100, 160, len(index)), index=index)
    first = classify_regime(close)
    changed_future = pd.concat([close, pd.Series([1.0], index=[index[-1] + pd.offsets.BDay()])])
    second = classify_regime(changed_future.iloc[:-1])
    assert first["regime"] == "BULL_TREND_LOW_VOL"
    assert first["regime"] == second["regime"]
    assert first["future_data_used"] is False


def test_crisis_disables_equity_trend() -> None:
    row = strategy_score(_survivor(), "CRISIS")
    assert row["enabled"] is False
    assert row["components"]["regime_compatibility"] == 0


def test_score_bounds_and_frozen_parameters() -> None:
    row = strategy_score(_survivor(pf=99), "BULL_TREND_LOW_VOL")
    assert 0 <= row["score"] <= 1
    assert row["frozen_parameters"] == {"fast": 50, "slow": 200}


def test_bayesian_positive_probability_uses_conservative_prior() -> None:
    empty = bayesian_positive_probability(0, 0)
    positive = bayesian_positive_probability(8, 2)
    assert empty["status"] == "UNAVAILABLE"
    assert empty["posterior_mean"] == 0.5
    assert positive["posterior_mean"] == 0.65
    assert positive["probability_above_break_even"] > 0.8


def test_missing_metrics_are_explicit_and_not_fabricated() -> None:
    evidence = score_strategy_evidence(
        {
            "profit_factor": 1.3,
            "sample_count": 60,
            "positive_periods": 4,
            "total_periods": 6,
        },
        regime_fit=0.8,
    )
    assert evidence["evidence_status"] == "PARTIAL_EVIDENCE"
    assert evidence["metrics"]["sortino"]["status"] == "UNAVAILABLE"
    assert evidence["metrics"]["recency"]["raw"] is None
    assert evidence["trade_break_even_win_rate"]["status"] == "UNAVAILABLE"


def test_weights_have_strategy_and_family_caps() -> None:
    scores = [
        {
            "strategy_id": f"S{i}",
            "family": "ma_crossover" if i < 2 else f"family{i}",
            "score": 0.9 - i * 0.02,
            "enabled": True,
        }
        for i in range(6)
    ]
    weights = capped_weights(scores)
    assert sum(row["weight"] for row in weights) <= 1.00001
    assert max(row["weight"] for row in weights) <= 0.250001
    trend_family = sum(row["weight"] for row in weights if row["family"] == "ma_crossover")
    assert trend_family <= 0.500001
    assert sum(row["weight"] > 0 for row in weights) > 1


def _plan(strategy_id: str, *, risks: list[str] | None = None) -> dict[str, object]:
    return {
        "ticker": "SPY", "strategy_id": strategy_id, "confidence_score": "0.8",
        "data_freshness": "FRESH", "risks": risks or [], "preferred_entry": "100",
        "stop_loss": "95", "take_profit_1": "107.5", "take_profit_2": "112.5",
        "currency": "USD", "contract_identity": {"con_id": 1},
    }


def test_same_family_votes_are_not_double_counted(tmp_path) -> None:
    weights = [
        {"strategy_id": "A", "family": "trend", "weight": 0.25},
        {"strategy_id": "B", "family": "trend", "weight": 0.25},
    ]
    report = _consensus(
        tmp_path,
        [_plan("A"), _plan("B")],
        weights,
        {"regime": "RECOVERY"},
    )
    assert report["signals"][0]["consensus_score"] == 0.2
    assert report["signals"][0]["independent_family_count"] == 1


def test_earnings_blocker_forces_avoid(tmp_path) -> None:
    weights = [{"strategy_id": "A", "family": "trend", "weight": 0.25}]
    report = _consensus(
        tmp_path,
        [_plan("A", risks=["EARNINGS_EVENT_BLOCKER"])],
        weights,
        {"regime": "RECOVERY"},
    )
    assert report["signals"][0]["action"] == "AVOID"


def test_missing_required_shariah_attestation_forces_avoid(tmp_path) -> None:
    config = tmp_path / "config" / "screener"
    config.mkdir(parents=True)
    (config / "daily_screener_v1.json").write_text(
        '{"shariah_attestations_path":"config/screener/attestations.json"}',
        encoding="utf-8",
    )
    (config / "attestations.json").write_text(
        '{"attestations":[]}',
        encoding="utf-8",
    )
    weights = [{"strategy_id": "A", "family": "trend", "weight": 0.25}]
    report = _consensus(
        tmp_path,
        [_plan("A")],
        weights,
        {"regime": "RECOVERY"},
    )
    assert report["signals"][0]["action"] == "AVOID"
    assert report["signals"][0]["shariah_status"] == "SHARIAH_ATTESTATION_REQUIRED"


def test_portfolio_uses_fx_risk_and_whole_shares(tmp_path) -> None:
    fx = tmp_path / "data" / "fx"
    fx.mkdir(parents=True)
    pd.DataFrame(
        [{"base_currency": "USD", "quote_currency": "EUR", "rate": 0.8}]
    ).to_parquet(fx / "fx_daily.parquet", index=False)
    signals = {
        "signals": [
            {
                **_plan("A"), "action": "BUY", "consensus_score": 0.8,
                "ticker": "SPY",
            }
        ]
    }
    report = _portfolio(tmp_path, signals)
    assert report["positions"][0]["quantity"] == 5
    assert report["gross_exposure_pct"] <= 0.25
    assert report["whole_share_accounting"] is True


def test_frozen_weekly_donchian_uses_prior_channel_only() -> None:
    close = pd.Series([100.0] * 20 + [102.0])
    high = pd.Series([101.0] * 20 + [103.0])
    low = pd.Series([99.0] * 21)
    action, confidence, reasons = _strategy_signal(
        "time_series_momentum",
        close,
        high,
        low,
        {"signal_variant": "donchian_breakout", "channel": 20},
    )
    assert action == "BUY"
    assert 0 < confidence <= 0.9
    assert "CLOSED_WEEKLY_BAR" in reasons


def test_family_specific_exit_policies() -> None:
    trend = _family_risk_policy("ma_crossover")
    breakout = _family_risk_policy("time_series_momentum")
    rotation = _family_risk_policy("etf_rotation")
    assert trend["stop_method"] != breakout["stop_method"]
    assert rotation["exit_policy"].startswith("RELATIVE_RANK")


def test_dynamic_forward_ledger_is_append_only_and_hash_frozen(tmp_path) -> None:
    ledger = ResearchLedger(AutopilotLayout(tmp_path))
    try:
        frozen = {
            "strategy_id": "DYN-1",
            "family": "trend",
            "timeframe": "1d",
            "frozen_parameters": {"fast": 20, "slow": 100},
        }
        first = ledger.register_dynamic_forward(frozen)
        second = ledger.register_dynamic_forward(frozen)
        assert first["inserted"] is True
        assert second["inserted"] is False
        observation = ledger.append_dynamic_forward_observation(
            strategy_id="DYN-1",
            session_date="2026-07-24",
            payload={"signal_assets": ["AAPL"], "used_for_optimization": False},
        )
        duplicate = ledger.append_dynamic_forward_observation(
            strategy_id="DYN-1",
            session_date="2026-07-24",
            payload={"signal_assets": ["AAPL"], "used_for_optimization": False},
        )
        assert observation["inserted"] is True
        assert duplicate["inserted"] is False
        assert ledger.dynamic_forward_status()["observation_count"] == 1
    finally:
        ledger.close()
