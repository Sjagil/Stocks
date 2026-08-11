from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocks.research.phase11_15 import (
    _apply_overlay,
    _entry_gated_state,
    _flow_scores,
    phase11_15_schema,
    phase11_15_status,
)


def _bars(periods: int = 12) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=periods, freq="1h")
    close = pd.Series(
        [100 + index * 0.5 for index in range(periods)],
        index=index,
    )
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


def test_entry_flow_gate_holds_until_base_signal_exits() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="1h")
    base = pd.Series([False, True, True, True, False, True], index=index)
    allowed = pd.Series([False, False, True, False, True, True], index=index)

    result = _entry_gated_state(base, allowed)

    assert result.tolist() == [False, False, True, True, False, True]


def test_flow_overlay_is_context_and_never_changes_authority() -> None:
    bars = _bars()
    signal = pd.DataFrame(
        {"signal": True, "score": 1.0}, index=bars.index
    )
    flow = pd.Series(
        [-1.0] * 4 + [0.5] * 8,
        index=bars.index,
    )

    result = _apply_overlay(
        {"AAA": signal},
        {"AAA": flow},
        overlay="FLOW_CONFIRM",
    )["AAA"]

    assert not result["signal"].iloc[:4].any()
    assert result["signal"].iloc[4:].all()
    assert "flow_proxy_score" in result


def test_one_hour_mtf_flow_uses_prior_closed_higher_bar() -> None:
    bars = _bars(16)
    scores = _flow_scores({"AAA": bars}, lower_timeframe="1h")["AAA"]

    assert scores.index.equals(bars.index)
    assert scores.notna().any()
    assert scores.between(-1, 1).dropna().all()


def test_schema_blocks_historical_gex_claims_and_all_authority(
    tmp_path: Path,
) -> None:
    report = phase11_15_schema(tmp_path)

    assert report["status"] == "GO"
    assert report["orderflow_data_class"] == (
        "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW"
    )
    assert report["gex_backtest_status"] == (
        "BLOCKED_NO_POINT_IN_TIME_OPTION_HISTORY"
    )
    assert report["FINANCIAL_FINALIST_GO"] is False
    assert report["EXECUTION_AUTHORITY"] == "NONE"
    assert report["BROKER_CALLS"] == 0
    assert phase11_15_status(tmp_path)["status"] == "NOT_RUN"
