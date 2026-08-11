from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from stocks.signals.market_reference import (
    apply_market_reference,
    latest_market_reference,
)


def _write_reference(
    root: Path,
    *,
    price: float,
    fetched_at: datetime,
) -> None:
    path = (
        root
        / "data/research/multitimeframe/private/provider=YFINANCE"
        / "symbol=AAPL/interval=1h/source_interval=1h"
    )
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": fetched_at.isoformat(),
                "symbol": "AAPL",
                "provider": "YFINANCE",
                "close": price,
                "fetched_at": fetched_at.isoformat(),
                "quality_status": "VALIDATED_OHLC",
                "is_partial": True,
            }
        ]
    ).to_parquet(path / "bars.parquet", index=False)


def _signal() -> dict[str, object]:
    return {
        "signal_id": "AAPL-TEST",
        "ticker": "AAPL",
        "action": "BUY",
        "data_freshness": "FRESH",
        "lifecycle_status": "MANUAL_ACTIONABLE",
        "current_market_price": "331.7350",
        "preferred_entry": "331.7350",
        "entry_zone_low": "330.0000",
        "entry_zone_high": "333.0000",
        "stop_loss": "315.5450",
        "take_profit_1": "356.0200",
        "take_profit_2": "372.2100",
        "risks": [],
        "reasons": [],
    }


def test_large_gap_below_stop_invalidates_old_entry(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 15, 0, tzinfo=UTC)
    _write_reference(tmp_path, price=302.77, fetched_at=now - timedelta(minutes=5))

    result = apply_market_reference(tmp_path, _signal(), now=now)

    assert result["action"] == "AVOID"
    assert result["lifecycle_status"] == "INVALIDATED"
    assert result["data_freshness"] == "STALE"
    assert result["current_market_price"] == Decimal("302.7700")
    assert result["price_validity_status"] == (
        "CURRENT_PRICE_BREACHED_SIGNAL_STOP"
    )
    assert result["entry_instruction"] == "WAIT_FOR_NEW_CAUSAL_SIGNAL"


def test_current_price_inside_entry_zone_remains_valid(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 15, 0, tzinfo=UTC)
    _write_reference(tmp_path, price=332.0, fetched_at=now - timedelta(minutes=5))

    result = apply_market_reference(tmp_path, _signal(), now=now)

    assert result["action"] == "BUY"
    assert result["data_freshness"] == "FRESH"
    assert result["price_validity_status"] == "CURRENT_ENTRY_REFERENCE_GO"


def test_stale_intraday_reference_blocks_signal(tmp_path: Path) -> None:
    now = datetime(2026, 7, 31, 15, 0, tzinfo=UTC)
    _write_reference(tmp_path, price=332.0, fetched_at=now - timedelta(hours=3))

    reference = latest_market_reference(tmp_path, "AAPL", now=now)
    result = apply_market_reference(tmp_path, _signal(), now=now)

    assert reference["status"] == "STALE"
    assert result["action"] == "AVOID"
    assert result["price_validity_status"] == (
        "CURRENT_MARKET_REFERENCE_UNAVAILABLE_OR_STALE"
    )
