from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import stocks.research.phase11_10 as phase11_10_module

from stocks.research.phase11_10 import (
    ARCHITECTURES,
    _architecture_signals,
    _deployment_identity,
    _economic_outcome_audit,
    _load_mtf_checkpoint,
    _phase11_10_worker_limit,
    _rank_viable_strategies,
    _score_summary,
    _shortlist,
    _write_mtf_checkpoint,
    phase11_10_pit_observe,
    phase11_10_schema,
)
from stocks.research.phase11_9 import BASE_STRATEGIES, ENSEMBLES


def test_phase11_10_worker_limit_is_bounded_and_configurable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(phase11_10_module.os, "cpu_count", lambda: 16)
    monkeypatch.setenv("STOCKS_PHASE11_10_MAX_WORKERS", "2")
    assert _phase11_10_worker_limit(20) == 2
    monkeypatch.setenv("STOCKS_PHASE11_10_MAX_WORKERS", "100")
    assert _phase11_10_worker_limit(20) == 12
    monkeypatch.setenv("STOCKS_PHASE11_10_MAX_WORKERS", "invalid")
    assert _phase11_10_worker_limit(20) == 4


def _daily_frame(periods: int = 1_800) -> pd.DataFrame:
    index = pd.bdate_range("2018-01-01", periods=periods)
    trend = np.linspace(50.0, 180.0, periods)
    cycle = np.sin(np.arange(periods) / 11.0) * 5.0
    close = trend + cycle
    return pd.DataFrame(
        {
            "open": close * 0.997,
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=index,
    )


def test_schema_registers_expanded_fail_closed_architectures(
    tmp_path: Path,
) -> None:
    report = phase11_10_schema(tmp_path)

    assert len(report["architectures"]) >= 150
    assert "research_source_hash" in report
    assert report["trial_contract"]["global_hypothesis_count"] == (
        len(report["architectures"]) * 3
    )
    assert report["architectures"]["four_hour_1h_pullback"] == {
        "higher": "4h",
        "lower": "1h",
        "entry": "ema_pullback",
    }
    assert report["architectures"]["daily_4h_1h_breakout"] == {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "volatility_contraction_breakout",
    }
    assert report["architectures"]["four_hour_1h_ma_trend"] == {
        "higher": "4h",
        "lower": "1h",
        "entry": "ma_crossover",
    }
    assert report["architectures"]["weekly_daily_4h_donchian"] == {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "donchian_breakout",
    }
    assert report["architectures"]["four_hour_2h_vpt_breakout"] == {
        "higher": "4h",
        "lower": "2h",
        "entry": "vpt_breakout",
    }
    assert report["architectures"]["daily_4h_2h_trend_consensus"] == {
        "higher": "1d",
        "middle": "4h",
        "lower": "2h",
        "entry": "trend_consensus",
    }
    assert report["architectures"]["weekly_daily_2h_ma_crossover"] == {
        "higher": "1w",
        "middle": "1d",
        "lower": "2h",
        "entry": "ma_crossover",
    }
    assert report["architectures"]["daily_4h_1h_rsi_pullback"] == {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "rsi2_adx_pullback",
    }
    assert report["architectures"]["weekly_daily_4h_bollinger"] == {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "bollinger_breakout",
    }
    assert report["architectures"]["daily_4h_1h_beta_residual"] == {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "beta_residual_pullback",
    }
    assert report["architectures"][
        "weekly_daily_4h_adaptive_breakout"
    ] == {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "adaptive_volatility_breakout",
    }
    registered_entries = {
        row["entry"] for row in report["architectures"].values()
    }
    assert set(BASE_STRATEGIES).issubset(registered_entries)
    assert set(ENSEMBLES).issubset(registered_entries)
    assert {
        "mfi_trend_pullback",
        "cmf_accumulation",
        "obv_breakout",
        "aroon_trend",
        "vortex_trend",
        "choppiness_breakout",
        "connors_rsi_pullback",
        "trend_quality_52w",
        "flow_consensus",
        "structure_consensus",
        "adaptive_momentum_consensus",
    }.issubset(registered_entries)
    assert (
        report["signal_contract"]["gate_availability"]
        == "SHIFTED_ONE_COMPLETE_HIGHER_BAR"
    )
    assert report["EXECUTION_AUTHORITY"] == "NONE"
    assert report["BROKER_CALLS"] == 0


def test_all_daily_based_architectures_emit_bounded_signal_frames() -> None:
    frame = _daily_frame()
    for name, specification in ARCHITECTURES.items():
        if specification["lower"] not in {"1d", "1w"}:
            continue
        lower = (
            frame
            if specification["lower"] == "1d"
            else frame.resample("W-FRI").agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
        )
        result = _architecture_signals(
            {"TEST": lower},
            higher_timeframe=specification["higher"],
            lower_timeframe=specification["lower"],
            entry_strategy=specification["entry"],
            profile="responsive",
        )["TEST"]
        assert result.index.equals(lower.index)
        assert result["signal"].dtype == bool
        assert result["score"].notna().all()


def test_future_lower_bar_change_does_not_revise_prior_signals() -> None:
    frame = _daily_frame()
    changed = frame.copy()
    changed.loc[changed.index[-1], ["high", "close"]] *= 4.0
    original = _architecture_signals(
        {"TEST": frame},
        higher_timeframe="1w",
        lower_timeframe="1d",
        entry_strategy="volatility_contraction_breakout",
        profile="balanced",
    )["TEST"]
    revised = _architecture_signals(
        {"TEST": changed},
        higher_timeframe="1w",
        lower_timeframe="1d",
        entry_strategy="volatility_contraction_breakout",
        profile="balanced",
    )["TEST"]

    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


def test_adaptive_thresholds_and_beta_anchor_are_causal() -> None:
    market = _daily_frame()
    asset = market.copy()
    asset["close"] = asset["close"].mul(
        1 + 0.015 * np.sin(np.arange(len(asset)) / 5.0)
    )
    asset["open"] = asset["close"] * 0.997
    asset["high"] = asset["close"] * 1.015
    asset["low"] = asset["close"] * 0.985
    changed_market = market.copy()
    changed_asset = asset.copy()
    changed_market.loc[changed_market.index[-1], "close"] *= 3.0
    changed_asset.loc[changed_asset.index[-1], "close"] *= 0.2

    kwargs = {
        "higher_timeframe": "1w",
        "lower_timeframe": "1d",
        "entry_strategy": "beta_residual_pullback",
        "profile": "balanced",
    }
    original = _architecture_signals(
        {"AAPL": asset, "SPY": market},
        **kwargs,
    )["AAPL"]
    revised = _architecture_signals(
        {"AAPL": changed_asset, "SPY": changed_market},
        **kwargs,
    )["AAPL"]

    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])
    assert original["signal"].dtype == bool
    assert original["score"].notna().all()


def test_1h_4h_daily_gates_use_only_completed_higher_bars() -> None:
    periods = 7_000
    index = pd.date_range("2025-01-01", periods=periods, freq="1h")
    close = np.linspace(80.0, 140.0, periods) + np.sin(
        np.arange(periods) / 17.0
    )
    frame = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 500_000.0,
        },
        index=index,
    )
    changed = frame.copy()
    changed.loc[changed.index[-1], ["high", "close"]] *= 5.0

    kwargs = {
        "higher_timeframe": "1d",
        "intermediate_timeframes": ("4h",),
        "lower_timeframe": "1h",
        "entry_strategy": "ema_pullback",
        "profile": "balanced",
    }
    original = _architecture_signals({"TEST": frame}, **kwargs)["TEST"]
    revised = _architecture_signals({"TEST": changed}, **kwargs)["TEST"]

    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])
    assert original["signal"].dtype == bool
    assert original["score"].notna().all()


def test_2h_4h_daily_gates_use_only_completed_higher_bars() -> None:
    periods = 3_500
    index = pd.date_range("2025-01-01", periods=periods, freq="2h")
    close = np.linspace(80.0, 140.0, periods) + np.sin(
        np.arange(periods) / 13.0
    )
    frame = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 500_000.0,
        },
        index=index,
    )
    changed = frame.copy()
    changed.loc[changed.index[-1], ["high", "close"]] *= 5.0
    kwargs = {
        "higher_timeframe": "1d",
        "intermediate_timeframes": ("4h",),
        "lower_timeframe": "2h",
        "entry_strategy": "trend_consensus",
        "profile": "balanced",
    }

    original = _architecture_signals({"TEST": frame}, **kwargs)["TEST"]
    revised = _architecture_signals(
        {"TEST": changed},
        **kwargs,
    )["TEST"]

    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])
    assert original["signal"].dtype == bool
    assert original["score"].notna().all()


def test_shortlist_requires_benchmark_incremental_evidence() -> None:
    common = {
        "higher_timeframe": "1w",
        "lower_timeframe": "1d",
        "entry_strategy": "ema_pullback",
        "fold_count": 20,
        "positive_fold_ratio": 0.75,
        "median_oos_portfolio_pf": 1.2,
        "worst_oos_portfolio_pf": 0.8,
        "median_oos_CAGR": 0.1,
        "median_oos_Sharpe": 0.7,
        "worst_oos_drawdown": -0.25,
        "cost_50bps_median_pf": 1.05,
        "plateau_fold_ratio": 0.7,
        "median_fill_count": 30,
        "positive_excess_CAGR_ratio": 0.7,
    }
    summary = pd.DataFrame(
        [
            {
                **common,
                "architecture": "PASS",
                "median_excess_CAGR": 0.03,
            },
            {
                **common,
                "architecture": "FAIL_BENCHMARK",
                "median_excess_CAGR": -0.01,
            },
        ]
    )

    report = _shortlist(summary)

    assert report["candidate_count"] == 1
    assert report["candidates"][0]["architecture"] == "PASS"
    assert report["promising_research_candidate_count"] == 2
    assert report["research_gate_can_grant_authority"] is False
    assert report["financial_finalist"] is False


def test_promising_research_tier_is_realistic_but_grants_no_authority() -> None:
    summary = pd.DataFrame(
        [
            {
                "architecture": "DAILY_4H_1H_PULLBACK",
                "higher_timeframe": "1d",
                "middle_timeframe": "4h",
                "lower_timeframe": "1h",
                "entry_strategy": "ema_pullback",
                "fold_count": 18,
                "positive_fold_ratio": 0.55,
                "median_oos_portfolio_pf": 1.08,
                "worst_oos_portfolio_pf": 0.62,
                "median_oos_CAGR": 0.03,
                "median_oos_Sharpe": 0.61,
                "worst_oos_drawdown": -0.72,
                "cost_50bps_median_pf": 0.98,
                "plateau_fold_ratio": 0.45,
                "median_fill_count": 85,
                "median_excess_CAGR": 0.01,
                "positive_excess_CAGR_ratio": 0.50,
            }
        ]
    )

    report = _shortlist(summary)

    assert report["candidate_count"] == 0
    assert report["promising_research_candidate_count"] == 1
    candidate = report["promising_research_candidates"][0]
    assert candidate["research_classification"] == "PROMISING_RESEARCH"
    assert candidate["stress_status"] == "MARGINAL_AT_50BPS"
    assert report["authority"] == "NONE"


def test_shortlist_uses_exact_modal_deployment_identity() -> None:
    architecture = "daily_4h_ma_trend"
    summary = pd.DataFrame(
        [
            {
                "architecture": architecture,
                "higher_timeframe": "1d",
                "middle_timeframe": None,
                "lower_timeframe": "4h",
                "entry_strategy": "ma_crossover",
                "fold_count": 18,
                "positive_fold_ratio": 0.70,
                "median_oos_portfolio_pf": 1.15,
                "worst_oos_portfolio_pf": 0.80,
                "median_oos_CAGR": 0.12,
                "median_oos_Sharpe": 0.75,
                "worst_oos_drawdown": -0.20,
                "cost_50bps_median_pf": 1.05,
                "plateau_fold_ratio": 0.70,
                "median_fill_count": 60,
                "median_excess_CAGR": 0.04,
                "positive_excess_CAGR_ratio": 0.70,
            }
        ]
    )
    selections = pd.DataFrame(
        {
            "architecture": [architecture, architecture, architecture],
            "selected_profile": [
                "conservative",
                "conservative",
                "balanced",
            ],
        }
    )

    report = _shortlist(summary, selections)
    expected = _deployment_identity(
        architecture=architecture,
        profile="conservative",
        specification=ARCHITECTURES[architecture],
    )

    assert report["candidates"][0]["strategy_id"] == expected["strategy_id"]
    assert (
        report["candidates"][0]["strategy_dna_hash"]
        == expected["strategy_dna_hash"]
    )


def test_weighted_evidence_advances_realistic_candidate_without_authority() -> None:
    summary = pd.DataFrame(
        [
            {
                "architecture": "WEEKLY_DAILY_PULLBACK",
                "higher_timeframe": "1w",
                "middle_timeframe": None,
                "lower_timeframe": "1d",
                "entry_strategy": "ema_pullback",
                "fold_count": 20,
                "positive_fold_ratio": 0.80,
                "median_oos_portfolio_pf": 1.14,
                "worst_oos_portfolio_pf": 0.81,
                "median_oos_CAGR": 0.16,
                "median_oos_Sharpe": 0.76,
                "worst_oos_drawdown": -0.27,
                "cost_50bps_median_pf": 1.06,
                "plateau_fold_ratio": 0.80,
                "median_fill_count": 215,
                "median_excess_CAGR": 0.05,
                "positive_excess_CAGR_ratio": 0.85,
            }
        ]
    )

    report = _shortlist(summary)

    assert report["weighted_evidence_candidate_count"] == 1
    candidate = report["weighted_evidence_candidates"][0]
    assert candidate["weighted_evidence_score"] >= 70
    assert candidate["hard_veto_reasons"] == []
    assert "PIT_BLOCKED" in candidate["evidence_tier"]
    assert report["weighted_evidence_policy"][
        "paper_or_live_authority_granted"
    ] is False


def test_weighted_evidence_never_overrides_hard_financial_veto() -> None:
    summary = pd.DataFrame(
        [
            {
                "architecture": "NEGATIVE_CAGR",
                "higher_timeframe": "4h",
                "middle_timeframe": None,
                "lower_timeframe": "1h",
                "entry_strategy": "ema_pullback",
                "fold_count": 20,
                "positive_fold_ratio": 0.80,
                "median_oos_portfolio_pf": 1.20,
                "worst_oos_portfolio_pf": 0.80,
                "median_oos_CAGR": -0.01,
                "median_oos_Sharpe": 0.90,
                "worst_oos_drawdown": -0.30,
                "cost_50bps_median_pf": 1.10,
                "plateau_fold_ratio": 0.80,
                "median_fill_count": 200,
                "median_excess_CAGR": 0.05,
                "positive_excess_CAGR_ratio": 0.80,
            }
        ]
    )

    report = _shortlist(summary)

    assert report["weighted_evidence_candidate_count"] == 0


def test_top_ranking_excludes_zero_fill_hard_veto() -> None:
    valid = {
        "architecture": "VALID",
        "higher_timeframe": "1d",
        "middle_timeframe": None,
        "lower_timeframe": "4h",
        "entry_strategy": "trend_consensus",
        "fold_count": 18,
        "positive_fold_ratio": 0.60,
        "median_oos_portfolio_pf": 1.10,
        "worst_oos_portfolio_pf": 0.70,
        "median_oos_CAGR": 0.10,
        "median_oos_Sharpe": 0.70,
        "worst_oos_drawdown": -0.20,
        "cost_50bps_median_pf": 1.01,
        "plateau_fold_ratio": 0.60,
        "median_fill_count": 50,
        "median_excess_CAGR": 0.02,
        "positive_excess_CAGR_ratio": 0.60,
    }
    zero_fill = {
        **valid,
        "architecture": "ZERO_FILL",
        "entry_strategy": "cmf_accumulation",
        "median_oos_Sharpe": 3.0,
        "median_oos_portfolio_pf": 4.0,
        "median_fill_count": 0,
    }

    ranked = _rank_viable_strategies(
        _score_summary(pd.DataFrame([zero_fill, valid]))
    )

    assert ranked["architecture"].tolist() == ["VALID"]


def test_pit_observer_requires_new_bar_and_current_attested_target(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "output" / "research" / "phase11_10"
    output.mkdir(parents=True)
    candidate = {
        "architecture": "daily_4h_ma_trend",
        "median_oos_CAGR": 0.15,
        "median_oos_portfolio_pf": 1.10,
    }
    (output / "status.json").write_text(
        json.dumps(
            {
                "schema_hash": "SCHEMA",
                "research_source_hash": "SOURCE",
                "coverage": [
                    {
                        "architecture": "daily_4h_ma_trend",
                        "end": "2026-07-24T17:30:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (output / "shortlist.json").write_text(
        json.dumps({"candidates": [candidate]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "architecture": "daily_4h_ma_trend",
                "selected_profile": "conservative",
            }
        ]
    ).to_csv(output / "parameter-selections.csv", index=False)
    pd.DataFrame([candidate]).to_csv(
        output / "architecture-summary.csv",
        index=False,
    )
    source = tmp_path / "src" / "stocks" / "research" / "phase11_10.py"
    source.parent.mkdir(parents=True)
    source.write_text("FROZEN = True\n", encoding="utf-8")
    index = pd.DatetimeIndex(
        ["2026-07-28T13:30:00", "2026-07-28T17:30:00"]
    )
    frames = {}
    for symbol in ("AAPL", "JPM"):
        frame = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1000.0, 1100.0],
            },
            index=index,
        )
        frame.attrs["provider"] = "EODHD"
        frame.attrs["content_fingerprint"] = f"FP-{symbol}"
        frames[symbol] = frame
    signals = {
        "AAPL": pd.DataFrame(
            {"signal": [False, True], "score": [0.0, 1.0]},
            index=index,
        ),
        "JPM": pd.DataFrame(
            {"signal": [False, True], "score": [0.0, 0.5]},
            index=index,
        ),
    }
    monkeypatch.setattr(
        phase11_10_module,
        "_load_frames",
        lambda _root: {"4h": frames},
    )
    monkeypatch.setattr(
        phase11_10_module,
        "_research_source_hash",
        lambda _root: "SOURCE",
    )
    monkeypatch.setattr(
        phase11_10_module,
        "_architecture_signals",
        lambda *_args, **_kwargs: signals,
    )
    monkeypatch.setattr(
        phase11_10_module,
        "_current_pit_attestations",
        lambda *_args: {"AAPL"},
    )
    monkeypatch.setattr(
        phase11_10_module,
        "bar_freshness",
        lambda *_args, **_kwargs: {"status": "FRESH_CLOSED_BAR"},
    )

    report = phase11_10_pit_observe(tmp_path)

    assert report["status"] == "GO"
    assert report["independent_observation_count"] == 1
    assert report["pit_eligible_observation_count"] == 1
    row = report["observations"][0]
    assert row["current_attested_target_weights"] == {"AAPL": 0.5}
    assert row["provider_continuity_status"] == "SAME_PRIMARY_PROVIDER_GO"
    assert row["automatic_orders"] == 0
    assert row["broker_calls"] == 0
    signals_path = (
        tmp_path / "output" / "signals" / "pit_mtf_signals.json"
    )
    published_signals = json.loads(signals_path.read_text(encoding="utf-8"))
    assert report["signal_count"] == 1
    assert len(published_signals) == 1
    published = published_signals[0]
    assert published["strategy_id"] == row["strategy_id"]
    assert published["ticker"] == "AAPL"
    assert published["timeframe"] == "4h"
    assert published["execution_state"] == "SHADOW"
    assert published["automatic_execution_allowed"] is False
    assert float(published["stop_loss"]) < float(
        published["preferred_entry"]
    )
    assert float(published["take_profit_1"]) > float(
        published["preferred_entry"]
    )
    assert published["broker_calls"] == 0
    assert published["orders_generated"] == 0
    broad_path = (
        tmp_path
        / "output"
        / "signals"
        / "pit_mtf_research_signals.json"
    )
    broad_signals = json.loads(broad_path.read_text(encoding="utf-8"))
    assert report["ten_bps_positive_candidate_count"] == 1
    assert report["ten_bps_positive_evaluated_asset_count"] == 2
    assert report["ten_bps_positive_active_signal_count"] == 2
    assert {row["ticker"] for row in broad_signals} == {"AAPL", "JPM"}
    assert all(
        row["qualification_status"] == "TEN_BPS_POSITIVE_RESEARCH"
        for row in broad_signals
    )
    assert all(row["execution_route"] == "BLOCKED" for row in broad_signals)
    assert all(
        row["automatic_execution_allowed"] is False
        for row in broad_signals
    )
    assert all(row["broker_calls"] == 0 for row in broad_signals)
    jpm = next(row for row in broad_signals if row["ticker"] == "JPM")
    assert jpm["current_pit_attested"] is False
    assert (
        "CURRENT_PIT_ATTESTATION_NOT_AVAILABLE" in jpm["reasons"]
    )
    assert (
        output / "latest-pit-forward-observation.json"
    ).exists()


def test_pit_observer_blocks_changed_research_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "output" / "research" / "phase11_10"
    output.mkdir(parents=True)
    (output / "status.json").write_text(
        json.dumps({"research_source_hash": "OLD"}),
        encoding="utf-8",
    )
    (output / "shortlist.json").write_text(
        json.dumps({"candidates": [{"architecture": "candidate"}]}),
        encoding="utf-8",
    )
    pd.DataFrame([{"architecture": "candidate"}]).to_csv(
        output / "parameter-selections.csv",
        index=False,
    )
    pd.DataFrame([{"architecture": "candidate"}]).to_csv(
        output / "architecture-summary.csv",
        index=False,
    )
    monkeypatch.setattr(
        phase11_10_module,
        "_research_source_hash",
        lambda _root: "NEW",
    )

    report = phase11_10_pit_observe(tmp_path)

    assert report["status"] == "BLOCKED_QUALIFICATION_SOURCE_CHANGED"
    assert report["qualified_source_hash"] == "OLD"
    assert report["current_source_hash"] == "NEW"


def test_deployment_identity_is_deterministic_and_timeframe_specific() -> None:
    one = _deployment_identity(
        architecture="daily_4h_1h_pullback",
        profile="balanced",
        specification=ARCHITECTURES["daily_4h_1h_pullback"],
    )
    replay = _deployment_identity(
        architecture="daily_4h_1h_pullback",
        profile="balanced",
        specification=ARCHITECTURES["daily_4h_1h_pullback"],
    )
    other = _deployment_identity(
        architecture="daily_4h_pullback",
        profile="balanced",
        specification=ARCHITECTURES["daily_4h_pullback"],
    )

    assert one == replay
    assert one["strategy_id"].startswith("MTF-")
    assert one["strategy_id"] != other["strategy_id"]


def test_economic_outcome_audit_groups_identical_architectures(
    tmp_path: Path,
) -> None:
    rows = []
    for architecture, cagr in (
        ("DONCHIAN", 0.1),
        ("MA_CHANNEL", 0.1),
        ("PULLBACK", 0.2),
    ):
        rows.append(
            {
                "architecture": architecture,
                "fold_id": "F1",
                "cost_bps": 10.0,
                "CAGR": cagr,
                "Sharpe": 0.5,
                "maximum_drawdown": -0.1,
                "period_profit_factor": 1.1,
                "terminal_nav": 11_000 if cagr == 0.1 else 12_000,
                "fill_count": 20,
                "turnover_initial_capital": 1.0,
            }
        )
    pd.DataFrame(rows).to_parquet(
        tmp_path / "nested-results.parquet",
        index=False,
    )

    result = _economic_outcome_audit(tmp_path)

    assert result["independent_economic_outcome_count"] == 2
    assert result["duplicate_group_count"] == 1
    assert result["duplicate_groups"][0]["architectures"] == [
        "DONCHIAN",
        "MA_CHANNEL",
    ]


def test_mtf_checkpoint_round_trip_and_signature_isolation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _write_mtf_checkpoint(
        output,
        run_signature="RUN-A",
        completed_architectures={"daily_4h_ma_trend"},
        result_rows=[
            {
                "architecture": "daily_4h_ma_trend",
                "fold_id": "F1",
                "CAGR": 0.12,
            }
        ],
        selection_rows=[
            {
                "architecture": "daily_4h_ma_trend",
                "fold_id": "F1",
                "profile": "conservative",
            }
        ],
        coverage_rows=[
            {
                "architecture": "daily_4h_ma_trend",
                "status": "GO",
            }
        ],
        blocked=[],
    )

    restored = _load_mtf_checkpoint(output, "RUN-A")

    assert restored["completed_architectures"] == [
        "daily_4h_ma_trend"
    ]
    assert restored["result_rows"][0]["fold_id"] == "F1"
    assert restored["selection_rows"][0]["profile"] == "conservative"
    assert restored["coverage_rows"][0]["status"] == "GO"
    assert restored["blocked"] == []

    mismatched = _load_mtf_checkpoint(output, "RUN-B")
    assert mismatched == {
        "result_rows": [],
        "selection_rows": [],
        "coverage_rows": [],
        "blocked": [],
        "completed_architectures": [],
    }
