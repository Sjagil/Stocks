from __future__ import annotations

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd

from stocks.research.phase11_8 import (
    _run_portfolio,
    _signal_frames,
    _select_candidate,
    _strategy_state,
    _synthetic_frames,
    phase11_8_schema,
    portfolio_invariant_audit,
)


def test_schema_requires_realistic_portfolio_and_none_authority(
    tmp_path: Path,
) -> None:
    report = phase11_8_schema(tmp_path)
    contract = report["portfolio_contract"]
    assert contract["whole_shares"] is True
    assert contract["global_max_gross_exposure"] == 1.0
    assert contract["global_security_netting"] is True
    assert contract["portfolio_profit_factor_primary"] is True
    assert report["EXECUTION_AUTHORITY"] == "NONE"
    assert report["BROKER_CALLS"] == 0


def test_portfolio_invariant_audit_is_green(tmp_path: Path) -> None:
    report = portfolio_invariant_audit(tmp_path)
    assert report["status"] == "GO"
    assert all(report["checks"].values())
    assert report["maximum_observed_gross_exposure"] <= 1.0000001


def test_whole_share_global_netted_portfolio() -> None:
    frames = _synthetic_frames()
    signals = _signal_frames(
        frames, "ma_crossover", {"fast": 5, "slow": 20}
    )
    result = _run_portfolio(
        frames,
        signals,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2021-12-31"),
        cost_bps=50,
    )
    assert result["duplicate_position_days"] == 0
    assert result["ledger"]["gross_exposure"].max() <= 1.0000001
    assert result["ledger"]["cash_eur"].min() >= -0.000001
    assert (
        result["fills"]["shares"].astype(float)
        == np.floor(result["fills"]["shares"].astype(float))
    ).all()
    assert (result["fills"]["fx_cost_eur"] > 0).all()
    assert np.allclose(
        result["fills"]["fee_eur"],
        result["fills"]["transaction_cost_eur"]
        + result["fills"]["fx_cost_eur"],
    )
    assert "period_profit_factor" in result["metrics"]


def test_simultaneous_selection_is_deterministic() -> None:
    frames = _synthetic_frames()
    signals = {
        key: pd.DataFrame(
            {"signal": True, "score": 1.0}, index=frame.index
        )
        for key, frame in frames.items()
    }
    result = _run_portfolio(
        frames,
        signals,
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-03-31"),
        cost_bps=10,
    )
    first_buys = result["fills"].loc[
        result["fills"]["side"].eq("BUY")
    ].head(4)
    assert first_buys["security_id"].tolist() == sorted(frames)[:4]
    assert result["selection_rule"] == "SCORE_DESC_SECURITY_ID_ASC_NEXT_BAR"


def test_asynchronous_bars_keep_causal_last_known_mark_value() -> None:
    first_index = pd.date_range(
        "2026-01-02 14:00:00+00:00",
        periods=5,
        freq="2h",
    )
    second_index = first_index + pd.Timedelta(hours=1)

    def frame(index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000_000.0,
            },
            index=index,
        )

    frames = {"A": frame(first_index), "B": frame(second_index)}
    signals = {
        identity: pd.DataFrame(
            {"signal": True, "score": 1.0},
            index=asset.index,
        )
        for identity, asset in frames.items()
    }
    result = _run_portfolio(
        frames,
        signals,
        start=first_index.min(),
        end=second_index.max(),
        cost_bps=0,
    )

    ledger = result["ledger"]
    assert ledger["nav_eur"].min() > 9_900
    assert result["metrics"]["maximum_drawdown"] > -0.02


def test_supported_strategy_states_are_causal_boolean() -> None:
    frame = next(iter(_synthetic_frames().values()))
    cases = {
        "ma_crossover": {"fast": 5, "slow": 20},
        "asymmetric_ma": {
            "entry_fast": 5,
            "entry_slow": 20,
            "exit_fast": 10,
            "exit_slow": 30,
        },
        "ma_channel": {"period": 20},
        "bollinger_breakout": {"period": 20, "sigma": 2},
        "volatility_contraction_breakout": {
            "short_vol": 5,
            "long_vol": 20,
            "ratio": 0.8,
            "breakout": 10,
        },
        "etf_trend": {"momentum": 10, "trend": 20},
        "commodity_etf_trend": {"momentum": 10, "trend": 20},
    }
    for strategy, parameters in cases.items():
        result = _strategy_state(frame, strategy, parameters)
        assert result["signal"].dtype == bool
        assert result.index.equals(frame.index)


def test_candidate_selection_prefers_plateau_and_worst_fold() -> None:
    summary = pd.DataFrame(
        [
            {
                "universe": "etf",
                "strategy": "raw_peak",
                "timeframe": "1mo",
                "fold_count": 20,
                "positive_fold_ratio": 0.8,
                "median_oos_portfolio_pf": 2.8,
                "worst_oos_portfolio_pf": 0.03,
                "median_oos_CAGR": 0.18,
                "worst_oos_drawdown": -0.16,
                "cost_50bps_median_pf": 2.7,
                "plateau_fold_ratio": 0.25,
            },
            {
                "universe": "stocks",
                "strategy": "robust",
                "timeframe": "1w",
                "fold_count": 20,
                "positive_fold_ratio": 0.65,
                "median_oos_portfolio_pf": 1.13,
                "worst_oos_portfolio_pf": 0.76,
                "median_oos_CAGR": 0.04,
                "worst_oos_drawdown": -0.45,
                "cost_50bps_median_pf": 1.12,
                "plateau_fold_ratio": 0.55,
            },
        ]
    )
    candidate = _select_candidate(summary)
    assert candidate["strategy"] == "robust"
    round_tripped = pd.read_csv(StringIO(summary.to_csv(index=False)))
    assert _select_candidate(round_tripped)["candidate_id"] == candidate["candidate_id"]
