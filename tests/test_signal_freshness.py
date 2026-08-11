from __future__ import annotations

from datetime import UTC, datetime

from stocks.signals.freshness import evaluate_signal_freshness


def _signal(
    *,
    data_timestamp: str,
    expiration_timestamp: str,
    timeframe: str = "4h",
) -> dict[str, str]:
    return {
        "timeframe": timeframe,
        "data_timestamp": data_timestamp,
        "expiration_timestamp": expiration_timestamp,
        "exchange_timezone": "America/New_York",
    }


def test_friday_intraday_signal_survives_weekend_until_next_bar_close() -> None:
    row = _signal(
        data_timestamp="2026-07-31T17:30:00+00:00",
        expiration_timestamp="2026-08-02T05:30:00+00:00",
    )

    result = evaluate_signal_freshness(
        row,
        now=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )

    assert result["status"] == "FRESH"
    assert result["is_current"] is True
    assert result["freshness_basis"] == (
        "EXCHANGE_SESSION_AWARE_SIGNAL_EXPIRY_V1"
    )
    assert result["latest_completed_session"] == "2026-07-31"
    assert result["effective_expiration"] == datetime(
        2026,
        8,
        3,
        17,
        30,
        tzinfo=UTC,
    )


def test_weekend_bridge_does_not_revive_older_exchange_session() -> None:
    row = _signal(
        data_timestamp="2026-07-30T17:30:00+00:00",
        expiration_timestamp="2026-08-01T05:30:00+00:00",
    )

    result = evaluate_signal_freshness(
        row,
        now=datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
    )

    assert result["status"] == "STALE"
    assert result["is_current"] is False
    assert result["reason"] == "UNDERLYING_CLOSED_BAR_STALE"


def test_signal_expires_after_next_relevant_bar_can_close() -> None:
    row = _signal(
        data_timestamp="2026-07-31T17:30:00+00:00",
        expiration_timestamp="2026-08-02T05:30:00+00:00",
    )

    result = evaluate_signal_freshness(
        row,
        now=datetime(2026, 8, 3, 17, 31, tzinfo=UTC),
    )

    assert result["status"] == "STALE"
    assert result["is_current"] is False


def test_no_exchange_timezone_preserves_wall_clock_clamp() -> None:
    result = evaluate_signal_freshness(
        {
            "timeframe": "1h",
            "data_timestamp": "2026-07-31T12:00:00+00:00",
            "expiration_timestamp": "2026-08-10T12:00:00+00:00",
        },
        now=datetime(2026, 8, 1, 0, 1, tzinfo=UTC),
    )

    assert result["status"] == "STALE"
    assert result["effective_expiration"] == datetime(
        2026,
        8,
        1,
        0,
        0,
        tzinfo=UTC,
    )
