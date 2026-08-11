from __future__ import annotations

import pandas as pd
import pytest

from stocks.research.phase11_6 import (
    _episodes,
    canonical_vote_modes,
    confirmation_position,
    confirmation_vote,
    duplicate_classification,
)


def _signals(component_count: int, active: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": [pd.Timestamp("2026-01-05")] * component_count,
            "security_id": ["SEC-1"] * component_count,
            "component": [f"c{index}" for index in range(component_count)],
            "signal": [index < active for index in range(component_count)],
        }
    )


def test_pair_majority_is_not_scheduled_as_duplicate_of_all() -> None:
    assert canonical_vote_modes(2) == ("ALL", "ANY")


def test_triple_majority_requires_strict_majority() -> None:
    assert bool(confirmation_vote(_signals(3, 2), "MAJORITY").iloc[0]["combined_signal"])
    assert not bool(confirmation_vote(_signals(3, 1), "MAJORITY").iloc[0]["combined_signal"])


def test_all_vote_requires_every_component() -> None:
    assert not bool(confirmation_vote(_signals(3, 2), "ALL").iloc[0]["combined_signal"])
    assert bool(confirmation_vote(_signals(3, 3), "ALL").iloc[0]["combined_signal"])


def test_any_vote_requires_one_component() -> None:
    assert bool(confirmation_vote(_signals(3, 1), "ANY").iloc[0]["combined_signal"])
    assert not bool(confirmation_vote(_signals(3, 0), "ANY").iloc[0]["combined_signal"])


def test_unknown_vote_mode_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="UNREGISTERED_VOTE_MODE_BLOCKED"):
        confirmation_vote(_signals(3, 1), "AVERAGE")


def test_near_duplicate_threshold_is_predeclared() -> None:
    assert duplicate_classification(0.90, 0.95, False) == "REJECTED_NEAR_DUPLICATE"
    assert duplicate_classification(0.89, 0.99, True) == "SAME_FAMILY_TREND_ENSEMBLE"
    assert duplicate_classification(0.50, 0.20, False) == "DISTINCT_COMPONENTS"


def test_vote_exit_modes_have_distinct_bounded_state_transitions() -> None:
    dates = pd.date_range("2026-01-01", periods=4)
    signals = pd.DataFrame(
        {
            "decision_time": list(dates) * 2,
            "security_id": ["A"] * 8,
            "component": ["primary"] * 4 + ["secondary"] * 4,
            "signal": [True, True, False, False, True, False, True, False],
        }
    )
    any_exit = confirmation_position(signals, entry_mode="ALL", exit_mode="ANY_EXIT", primary_component="primary")
    all_exit = confirmation_position(signals, entry_mode="ALL", exit_mode="ALL_EXIT", primary_component="primary")
    primary_exit = confirmation_position(signals, entry_mode="ALL", exit_mode="PRIMARY_COMPONENT_EXIT", primary_component="primary")
    assert any_exit["signal"].tolist() == [True, False, False, False]
    assert all_exit["signal"].tolist() == [True, True, True, False]
    assert primary_exit["signal"].tolist() == [True, True, False, False]


def test_closed_signal_executes_on_next_bar_not_same_bar() -> None:
    dates = pd.date_range("2026-01-01", periods=4)
    signals = pd.DataFrame(
        {"decision_time": dates, "security_id": ["A"] * 4, "signal": [True, True, False, False]}
    )
    metadata = pd.DataFrame(
        {"security_id": ["A"], "ticker": ["A"], "sector": ["Tech"], "currency": ["USD"], "median_dollar_volume": [10_000_000.0]}
    )
    episodes = _episodes(signals, "test", metadata)
    assert episodes.iloc[0]["entry_date"] == dates[1]
    assert episodes.iloc[0]["exit_date"] == dates[3]


def test_unknown_exit_mode_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="UNREGISTERED_EXIT_MODE_BLOCKED"):
        confirmation_position(_signals(2, 2), entry_mode="ALL", exit_mode="TRAIL_FOREVER")
