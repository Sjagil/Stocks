from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from stocks.news.event_study import (
    _cluster_receipt_map,
    _measure_intraday,
    _summary_by_event_class,
)


def _hourly(prices: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range(
                "2026-01-02T08:30:00Z",
                periods=len(prices),
                freq="1h",
            ),
            "adjusted_close": prices,
        }
    )


def test_intraday_car_uses_only_a_bar_closed_before_event() -> None:
    market = _hourly([100.0 + index for index in range(80)])
    asset = _hourly([100.0 + 1.5 * index for index in range(80)])
    event_at = datetime(2026, 1, 4, 10, 45, tzinfo=UTC)

    result = _measure_intraday(
        asset,
        market,
        pd.DataFrame(),
        event_at=event_at,
        horizon_bars=1,
        minimum_beta_observations=10,
        maximum_beta=3.0,
    )

    assert result["label_status"] == "COMPLETE"
    baseline = pd.Timestamp(result["baseline_timestamp"])
    target = pd.Timestamp(result["target_timestamp"])
    assert baseline + pd.Timedelta(hours=1) <= pd.Timestamp(event_at)
    assert target >= pd.Timestamp(event_at)


def test_intraday_horizon_remains_pending_without_future_bars() -> None:
    prices = list(np.linspace(100.0, 110.0, 20))
    result = _measure_intraday(
        _hourly(prices),
        _hourly(prices),
        pd.DataFrame(),
        event_at=datetime(2026, 1, 3, 3, 45, tzinfo=UTC),
        horizon_bars=20,
        minimum_beta_observations=5,
        maximum_beta=3.0,
    )

    assert result["label_status"] == "FUTURE_HORIZON_PENDING"


def test_cluster_receipt_time_is_earliest_observed_copy() -> None:
    rows = [
        {
            "story_cluster_id": "STORY-1",
            "received_at": "2026-01-02T12:00:00Z",
        },
        {
            "story_cluster_id": "STORY-1",
            "received_at": "2026-01-02T11:00:00Z",
        },
    ]

    result = _cluster_receipt_map(rows)

    assert result["STORY-1"] == datetime(
        2026, 1, 2, 11, 0, tzinfo=UTC
    )


def test_event_class_summary_keeps_causal_and_descriptive_counts_separate() -> None:
    rows = [
        {
            "event_classes": '["EARNINGS","GUIDANCE_RAISE"]',
            "cumulative_abnormal_return": 0.02,
            "impact_direction_correct": True,
            "event_time_mode": "OPERATIONAL_CAUSAL",
        },
        {
            "event_classes": '["EARNINGS"]',
            "cumulative_abnormal_return": -0.01,
            "impact_direction_correct": False,
            "event_time_mode": "HISTORICAL_DESCRIPTIVE",
        },
    ]

    by_class = {
        row["event_class"]: row for row in _summary_by_event_class(rows)
    }

    assert by_class["EARNINGS"]["label_count"] == 2
    assert by_class["EARNINGS"]["causal_label_count"] == 1
    assert by_class["EARNINGS"]["descriptive_label_count"] == 1
    assert by_class["GUIDANCE_RAISE"]["training_eligible"] is True
