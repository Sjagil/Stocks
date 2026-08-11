from __future__ import annotations

import pandas as pd
import pytest

from stocks.data.multitimeframe import aggregate_bars, canonical_interval
from stocks.research.phase11_6 import causal_higher_timeframe_map


def test_month_aliases_are_unambiguous_and_one_minute_is_reserved() -> None:
    assert canonical_interval("1m") == "1mo"
    assert canonical_interval("1M") == "1mo"
    assert canonical_interval("month") == "1mo"
    with pytest.raises(ValueError, match="FORBIDDEN_SWING_TIMEFRAME"):
        canonical_interval("1min")


def test_higher_timeframe_mapping_uses_only_available_closed_bar() -> None:
    lower = pd.DataFrame(
        {
            "security_id": ["A", "A", "A"],
            "decision_time": pd.to_datetime(["2026-01-06", "2026-01-09", "2026-01-12"]),
        }
    )
    higher = pd.DataFrame(
        {
            "security_id": ["A", "A"],
            "available_at": pd.to_datetime(["2026-01-09", "2026-01-16"]),
            "regime_signal": [True, False],
        }
    )
    mapped = causal_higher_timeframe_map(lower, higher)
    assert pd.isna(mapped.iloc[0]["regime_signal"])
    assert bool(mapped.iloc[1]["regime_signal"])
    assert bool(mapped.iloc[2]["regime_signal"])
    available = mapped["available_at"].dropna()
    assert (available <= mapped.loc[available.index, "decision_time"]).all()


def test_mapping_never_crosses_security_identity() -> None:
    lower = pd.DataFrame({"security_id": ["A", "B"], "decision_time": pd.to_datetime(["2026-01-10", "2026-01-10"])})
    higher = pd.DataFrame({"security_id": ["A", "B"], "available_at": pd.to_datetime(["2026-01-09", "2026-01-09"]), "regime_signal": [True, False]})
    mapped = causal_higher_timeframe_map(lower, higher)
    assert mapped.set_index("security_id")["regime_signal"].to_dict() == {"A": True, "B": False}


def test_derived_bars_publish_phase11_6_provenance_fields() -> None:
    timestamps = pd.date_range("2026-01-05 14:30", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps, "session_date": ["2026-01-05"] * 4,
            "symbol": ["SPY"] * 4, "provider": ["TEST"] * 4,
            "interval": ["1h"] * 4, "source_interval": ["1h"] * 4,
            "derivation": ["NATIVE_OR_SOURCE_CACHE"] * 4,
            "exchange_timezone": ["America/New_York"] * 4,
            "adjustment_mode": ["RAW"] * 4, "open": [1, 2, 3, 4],
            "high": [2, 3, 4, 5], "low": [0.5, 1.5, 2.5, 3.5],
            "close": [1.5, 2.5, 3.5, 4.5], "adjusted_close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10] * 4, "dividends": [0] * 4, "stock_splits": [0] * 4,
            "source_bar_count": [1] * 4, "is_partial": [False] * 4,
            "received_at": ["2026-01-06T00:00:00+00:00"] * 4,
        }
    )
    result = aggregate_bars(frame, target_interval="4h")
    required = {"source_provider", "target_interval", "bar_origin", "aggregation_rule", "partial_bucket", "session", "timezone", "quality_status", "fetched_at", "ingested_at"}
    assert required.issubset(result.columns)
    assert result.iloc[0]["bar_origin"] == "DERIVED"


def test_upsampling_is_blocked() -> None:
    frame = pd.DataFrame({"interval": ["1d"], "timestamp_utc": [pd.Timestamp("2026-01-01", tz="UTC")]})
    with pytest.raises(ValueError, match="cannot aggregate"):
        aggregate_bars(frame, target_interval="1h")


def test_incomplete_week_and_month_are_not_published() -> None:
    timestamps = pd.date_range("2026-01-05", periods=3, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": timestamps, "session_date": timestamps.date.astype(str),
            "symbol": ["SPY"] * 3, "provider": ["TEST"] * 3,
            "interval": ["1d"] * 3, "source_interval": ["1d"] * 3,
            "derivation": ["NATIVE"] * 3, "exchange_timezone": ["UTC"] * 3,
            "adjustment_mode": ["RAW"] * 3, "open": [1.0] * 3, "high": [1.1] * 3,
            "low": [0.9] * 3, "close": [1.0] * 3, "adjusted_close": [1.0] * 3,
            "volume": [1.0] * 3, "dividends": [0.0] * 3, "stock_splits": [0.0] * 3,
            "source_bar_count": [1] * 3, "is_partial": [False] * 3,
            "received_at": ["2026-01-08T00:00:00+00:00"] * 3,
        }
    )
    assert aggregate_bars(frame, target_interval="1w").empty
    assert aggregate_bars(frame, target_interval="1mo").empty
