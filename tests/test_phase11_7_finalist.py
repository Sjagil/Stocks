from __future__ import annotations

import pandas as pd

from stocks.research.phase11_7 import (
    _apply_execution_regime,
    _breadth_execution_regime,
    _closed_episodes,
    _episode_metrics,
    _is_common_stock,
    _pbo,
    _rotation_candidates,
    phase11_7_schema,
)


def test_common_stock_filter_blocks_non_common_products() -> None:
    assert _is_common_stock({"category": "Domestic Common Stock"})
    assert not _is_common_stock({"category": "ETF"})
    assert not _is_common_stock({"category": "Preferred Stock"})


def test_closed_episode_profit_factor_uses_completed_position_episodes() -> None:
    fills = pd.DataFrame(
        [
            {
                "date": "2020-01-02",
                "security_id": "A",
                "side": "BUY",
                "shares": 1,
                "notional_eur": 100.0,
                "fee_eur": 1.0,
            },
            {
                "date": "2020-01-03",
                "security_id": "A",
                "side": "SELL",
                "shares": 1,
                "notional_eur": 111.0,
                "fee_eur": 1.0,
            },
            {
                "date": "2020-02-02",
                "security_id": "B",
                "side": "BUY",
                "shares": 1,
                "notional_eur": 100.0,
                "fee_eur": 1.0,
            },
            {
                "date": "2020-02-03",
                "security_id": "B",
                "side": "SELL",
                "shares": 1,
                "notional_eur": 96.0,
                "fee_eur": 1.0,
            },
        ]
    )
    episodes = _closed_episodes(fills)
    metrics = _episode_metrics(episodes)
    assert metrics["episode_count"] == 2
    assert metrics["episode_profit_factor"] == 1.5


def test_no_trades_has_null_episode_profit_factor() -> None:
    metrics = _episode_metrics(pd.DataFrame())
    assert metrics["episode_profit_factor"] is None
    assert metrics["sample_status"] == "INSUFFICIENT_SAMPLE"


def test_pbo_is_bounded() -> None:
    index = pd.date_range("2020-01-01", periods=160, freq="B")
    result = _pbo(
        {
            "A": pd.Series([0.001, -0.0005] * 80, index=index),
            "B": pd.Series([0.0005, -0.0007] * 80, index=index),
        }
    )
    assert result["status"] == "GO"
    assert 0.0 <= result["PBO"] <= 1.0


def test_breadth_regime_is_lagged_to_next_session() -> None:
    dates = pd.date_range("2020-01-01", periods=205, freq="B")
    frames = {
        f"S{index}": pd.DataFrame(
            {"date": dates, "close": range(1, 206)}
        )
        for index in range(20)
    }
    regime = _breadth_execution_regime(frames, dates)
    assert not bool(regime.iloc[199])
    assert bool(regime.iloc[200])


def test_regime_splits_candidate_without_same_day_episode() -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    regime = pd.Series([False, True, True, False, True], index=dates)
    candidates = pd.DataFrame(
        [{"entry_date": dates[0], "exit_date": dates[4], "variant_id": "V"}]
    )
    result = _apply_execution_regime(candidates, regime)
    assert len(result) == 1
    assert result.iloc[0]["entry_date"] == dates[1]
    assert result.iloc[0]["exit_date"] == dates[3]


def test_rotation_candidates_use_next_session_and_clip_terminal_exit() -> None:
    calendar = pd.date_range("2020-01-02", periods=320, freq="B")
    dates = calendar[:260]
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": pd.Series(range(100, 360), dtype=float),
            "volume": 100_000.0,
        }
    )
    candidates = _rotation_candidates(
        {"A": frame},
        {
            "A": {
                "ticker": "A",
                "sector": "Technology",
                "currency": "USD",
            }
        },
        calendar,
        lookback=63,
        trend_period=100,
        rebalance="Q",
    )
    assert not candidates.empty
    assert (candidates["entry_date"] < candidates["exit_date"]).all()
    assert candidates["terminal_exit"].any()


def test_schema_keeps_all_authority_disabled(tmp_path) -> None:
    payload = phase11_7_schema(tmp_path)
    assert payload["FINANCIAL_FINALIST_GO"] is False
    assert payload["FORWARD_SHADOW_GO"] is False
    assert payload["STRATEGY_AUTHORITY"] == "NONE"
    assert payload["EXECUTION_AUTHORITY"] == "NONE"
    assert payload["effective_trial_count"] == 48
