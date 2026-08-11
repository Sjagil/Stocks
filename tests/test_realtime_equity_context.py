from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from stocks.context.realtime_equity import (
    RealtimeEquityConfig,
    _append_private,
    _candidate_contracts,
    _load_market_data_endpoint,
)


def test_realtime_equity_bounds_are_fail_closed() -> None:
    RealtimeEquityConfig(duration_seconds=2, max_symbols=10, depth_symbols=5).validate()
    with pytest.raises(ValueError, match="duration_seconds"):
        RealtimeEquityConfig(duration_seconds=61).validate()
    with pytest.raises(ValueError, match="max_symbols"):
        RealtimeEquityConfig(max_symbols=11).validate()
    with pytest.raises(ValueError, match="depth_symbols"):
        RealtimeEquityConfig(max_symbols=3, depth_symbols=4).validate()


def test_realtime_candidates_are_unique_and_bounded() -> None:
    rows = [
        {"ticker": "AAPL", "contract_identity": {"con_id": 1, "currency": "USD"}},
        {"ticker": "AAPL", "contract_identity": {"con_id": 1, "currency": "USD"}},
        {"ticker": "SPY", "contract_identity": {}},
    ]

    selected = _candidate_contracts(rows, limit=2)

    assert [item["symbol"] for item in selected] == ["AAPL", "SPY"]
    assert selected[0]["con_id"] == 1
    assert selected[1]["exchange"] == "SMART"


def test_private_realtime_store_appends_and_deduplicates(tmp_path) -> None:
    path = tmp_path / "equity-trades.parquet"
    timestamp = datetime.now(UTC).isoformat()
    row = {"timestamp": timestamp, "symbol": "AAPL", "price": 100.0, "size": 1.0}

    _append_private(path, [row], days=7)
    _append_private(path, [row], days=7)

    stored = pd.read_parquet(path)
    assert len(stored) == 1
    assert stored.iloc[0]["symbol"] == "AAPL"


def test_live_market_data_endpoint_grants_no_execution_authority(tmp_path) -> None:
    env = tmp_path / ".env.ibkr.live"
    env.write_text(
        "IBKR_HOST=127.0.0.1\n"
        "IBKR_PORT=7496\n"
        "IBKR_CLIENT_ID=91\n"
        "IBKR_READ_ONLY=true\n"
        "IBKR_ORDER_AUTHORITY=NONE\n"
        "IBKR_LIVE_TRADING_ENABLED=false\n"
        "IBKR_ALLOW_ORDER_TRANSMISSION=false\n",
        encoding="ascii",
    )

    endpoint = _load_market_data_endpoint(env)

    assert endpoint.environment == "LIVE_MARKET_DATA_ONLY"
    assert endpoint.market_data_type == 1


def test_market_data_endpoint_rejects_order_authority(tmp_path) -> None:
    env = tmp_path / ".env.ibkr"
    env.write_text("IBKR_ORDER_AUTHORITY=LIVE\n", encoding="ascii")

    with pytest.raises(ValueError, match="IBKR_ORDER_AUTHORITY=NONE"):
        _load_market_data_endpoint(env)
