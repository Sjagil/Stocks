from __future__ import annotations

import pandas as pd
import pytest

from stocks.research.phase11_9 import (
    _snap_split_factor,
    _split_adjust_intraday,
)


def test_intraday_split_adjustment_restores_price_and_volume_continuity() -> None:
    intraday_index = pd.to_datetime(
        [
            "2024-06-07 13:30",
            "2024-06-07 17:30",
            "2024-06-10 13:30",
            "2024-06-10 17:30",
        ]
    )
    intraday = pd.DataFrame(
        {
            "open": [1190.0, 1200.0, 120.0, 121.0],
            "high": [1205.0, 1215.0, 122.0, 123.0],
            "low": [1185.0, 1195.0, 119.0, 120.0],
            "close": [1200.0, 1210.0, 121.0, 122.0],
            "volume": [100.0, 100.0, 1000.0, 1000.0],
        },
        index=intraday_index,
    )
    daily = pd.DataFrame(
        {
            "open": [119.0, 120.0],
            "high": [121.5, 123.0],
            "low": [118.5, 119.0],
            "close": [121.0, 122.0],
            "volume": [2000.0, 2000.0],
        },
        index=pd.to_datetime(["2024-06-07", "2024-06-10"]),
    )

    result = _split_adjust_intraday(
        {"NVDA": intraday},
        {"NVDA": daily},
    )["NVDA"]

    assert result.loc[intraday_index[1], "close"] == pytest.approx(121.0)
    assert result.loc[intraday_index[1], "volume"] == pytest.approx(1000.0)
    assert result.loc[intraday_index[3], "close"] == pytest.approx(122.0)
    assert result.attrs["split_adjustment_event_count"] == 1
    assert result.attrs["split_adjustment_events"][0][
        "historical_price_multiplier"
    ] == pytest.approx(0.1)


def test_normal_provider_differences_do_not_create_split_event() -> None:
    index = pd.to_datetime(
        ["2026-01-02 13:30", "2026-01-05 13:30"]
    )
    intraday = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1100.0],
        },
        index=index,
    )
    daily = pd.DataFrame(
        {
            "open": [99.9, 100.9],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.4, 101.4],
            "volume": [1000.0, 1100.0],
        },
        index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
    )

    result = _split_adjust_intraday(
        {"TEST": intraday},
        {"TEST": daily},
    )["TEST"]

    pd.testing.assert_frame_equal(result, intraday)
    assert result.attrs["split_adjustment_event_count"] == 0


@pytest.mark.parametrize(
    ("observed", "expected"),
    [(0.101, 0.1), (0.498, 0.5), (1.49, 1.5), (1.02, None)],
)
def test_split_factor_snapping(
    observed: float, expected: float | None
) -> None:
    assert _snap_split_factor(observed) == expected
