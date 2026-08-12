from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pandas as pd

from stocks.research.phase11_12_forward import (
    lower_timeframe_forward_status,
    update_lower_timeframe_forward,
)


def _frame(periods: int) -> pd.DataFrame:
    index = pd.date_range("2026-07-28T13:30:00", periods=periods, freq="h")
    return pd.DataFrame(
        {
            "open": [100.0 + index for index in range(periods)],
            "high": [101.0 + index for index in range(periods)],
            "low": [99.0 + index for index in range(periods)],
            "close": [100.5 + index for index in range(periods)],
            "volume": [1000.0] * periods,
        },
        index=index,
    )


def _observation(
    *,
    observed_at: str,
    closed_bar: str,
    active: bool,
) -> dict:
    strategy_id = "BULK-FORWARD-TEST"
    signal = {
        "strategy_id": strategy_id,
        "formula": "flow_consensus",
        "strategy_family": "flow_consensus",
        "strategy_dna_hash": "DNA-FLOW-CONSENSUS-1H",
        "timeframe": "1h",
        "strategy_timeframe_contract": {
            "schema": "active_swing_strategy_timeframe_contract_v1",
            "entry_timeframe": "1h",
            "setup_timeframe": "1h",
            "context_timeframes": [],
            "required_timeframes": ["1h"],
        },
        "model_version": "NO_ML_MODEL_DETERMINISTIC_SIGNAL_V1",
        "profile": "balanced",
        "asset_class": "STOCK",
        "symbol": "AAPL",
        "closed_bar_timestamp": closed_bar,
        "data_freshness": "FRESH_CLOSED_BAR",
    }
    return {
        "schema": "phase11_12_lower_timeframe_shadow_observation_v1",
        "status": "GO",
        "observed_at": observed_at,
        "active_signal_count": 1 if active else 0,
        "observations": [
            {
                "strategy_id": strategy_id,
                "formula": "flow_consensus",
                "timeframe": "1h",
                "profile": "balanced",
                "asset_class": "STOCK",
                "closed_bar_timestamp": closed_bar,
                "observation_status": "OBSERVATION_COMPLETE",
            }
        ],
        "active_signals": [signal] if active else [],
    }


def test_forward_episode_uses_next_bar_entry_and_exit_with_costs(
    tmp_path: Path,
) -> None:
    first = _observation(
        observed_at="2026-07-28T15:31:00Z",
        closed_bar="2026-07-28T15:30:00",
        active=True,
    )
    report = update_lower_timeframe_forward(
        tmp_path,
        first,
        {"1h": {"AAPL": _frame(4)}},
    )
    assert report["pending_entry_count"] == 1
    assert report["open_episode_count"] == 0
    assert report["closed_episode_count"] == 0

    second = _observation(
        observed_at="2026-07-28T16:31:00Z",
        closed_bar="2026-07-28T16:30:00",
        active=True,
    )
    report = update_lower_timeframe_forward(
        tmp_path,
        second,
        {"1h": {"AAPL": _frame(4)}},
    )
    assert report["pending_entry_count"] == 0
    assert report["open_episode_count"] == 1

    third = _observation(
        observed_at="2026-07-28T16:32:00Z",
        closed_bar="2026-07-28T16:30:00",
        active=False,
    )
    report = update_lower_timeframe_forward(
        tmp_path,
        third,
        {"1h": {"AAPL": _frame(4)}},
    )
    assert report["pending_exit_count"] == 1
    assert report["closed_episode_count"] == 0

    fourth = _observation(
        observed_at="2026-07-28T17:31:00Z",
        closed_bar="2026-07-28T17:30:00",
        active=False,
    )
    report = update_lower_timeframe_forward(
        tmp_path,
        fourth,
        {"1h": {"AAPL": _frame(5)}},
    )
    assert report["pending_exit_count"] == 0
    assert report["closed_episode_count"] == 1
    expected = ((104.0 * 0.995) / (103.0 * 1.005)) - 1.0
    assert math.isclose(
        report["aggregate"]["cumulative_net_return"],
        expected,
        rel_tol=1e-12,
    )
    assert report["aggregate"]["sample_status"] == "INSUFFICIENT_SAMPLE"
    assert report["architecture_binding_complete_count"] == 1
    assert report["architecture_binding_incomplete_count"] == 0
    assert report["architectures"][0]["entry_timeframe"] == "1h"
    assert report["architectures"][0]["setup_timeframe"] == "1h"
    assert report["architectures"][0]["context_timeframes"] == []
    assert (
        report["architectures"][0]["model_version"]
        == "NO_ML_MODEL_DETERMINISTIC_SIGNAL_V1"
    )
    assert report["FINANCIAL_FINALIST_GO"] is False
    assert report["EXECUTION_AUTHORITY"] == "NONE"
    assert report["broker_calls"] == 0
    assert report["order_calls"] == 0


def test_forward_observation_and_episode_creation_are_idempotent(
    tmp_path: Path,
) -> None:
    observation = _observation(
        observed_at="2026-07-28T15:31:00Z",
        closed_bar="2026-07-28T15:30:00",
        active=True,
    )
    frames = {"1h": {"AAPL": _frame(4)}}

    first = update_lower_timeframe_forward(tmp_path, observation, frames)
    second = update_lower_timeframe_forward(tmp_path, observation, frames)

    assert first["observation_inserted"] is True
    assert second["observation_inserted"] is False
    assert second["observation_count"] == 1
    assert second["episode_count"] == 1
    assert second["event_count"] == 1


def test_signal_removed_before_next_bar_cancels_pending_entry(
    tmp_path: Path,
) -> None:
    active = _observation(
        observed_at="2026-07-28T15:31:00Z",
        closed_bar="2026-07-28T15:30:00",
        active=True,
    )
    update_lower_timeframe_forward(
        tmp_path,
        active,
        {"1h": {"AAPL": _frame(3)}},
    )
    inactive = _observation(
        observed_at="2026-07-28T15:32:00Z",
        closed_bar="2026-07-28T15:30:00",
        active=False,
    )
    report = update_lower_timeframe_forward(
        tmp_path,
        inactive,
        {"1h": {"AAPL": _frame(3)}},
    )

    assert report["cancelled_before_entry_count"] == 1
    assert report["closed_episode_count"] == 0


def test_forward_status_does_not_expose_episode_prices(tmp_path: Path) -> None:
    observation = _observation(
        observed_at="2026-07-28T15:31:00Z",
        closed_bar="2026-07-28T15:30:00",
        active=True,
    )
    update_lower_timeframe_forward(
        tmp_path,
        observation,
        {"1h": {"AAPL": _frame(4)}},
    )

    report = lower_timeframe_forward_status(tmp_path)
    encoded = str(report)

    assert "entry_price_eur" not in encoded
    assert "exit_price_eur" not in encoded
    assert report["automatic_orders"] == 0
    database = sqlite3.connect(report["private_database"])
    try:
        assert database.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1
    finally:
        database.close()


def test_stale_observation_never_triggers_exit(tmp_path: Path) -> None:
    active = _observation(
        observed_at="2026-07-28T15:31:00Z",
        closed_bar="2026-07-28T15:30:00",
        active=True,
    )
    update_lower_timeframe_forward(
        tmp_path,
        active,
        {"1h": {"AAPL": _frame(4)}},
    )
    next_active = _observation(
        observed_at="2026-07-28T16:31:00Z",
        closed_bar="2026-07-28T16:30:00",
        active=True,
    )
    update_lower_timeframe_forward(
        tmp_path,
        next_active,
        {"1h": {"AAPL": _frame(4)}},
    )
    stale = _observation(
        observed_at="2026-07-28T17:31:00Z",
        closed_bar="2026-07-28T16:30:00",
        active=False,
    )
    stale["observations"][0]["stale_signal_count"] = 1

    report = update_lower_timeframe_forward(
        tmp_path,
        stale,
        {"1h": {"AAPL": _frame(4)}},
    )

    assert report["open_episode_count"] == 1
    assert report["pending_exit_count"] == 0
