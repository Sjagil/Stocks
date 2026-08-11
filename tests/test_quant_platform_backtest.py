from __future__ import annotations

import pandas as pd
import pytest

from stocks.quant_platform import (
    BacktestConfig,
    BacktestOrder,
    OrderSide,
    OrderType,
    ProfessionalBacktestEngine,
    orders_from_target_positions,
)


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    closes = [100, 102, 105, 104, 108, 106, 110, 112]
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "timestamp": dates,
            "available_at": dates,
            "open": closes,
            "high": [value + 2 for value in closes],
            "low": [value - 2 for value in closes],
            "close": closes,
            "volume": 100,
            "split_factor": 1.0,
            "dividend": 0.0,
        }
    )


def test_market_orders_fill_next_bar_with_costs_and_round_trip_metrics() -> None:
    bars = _bars()
    orders = [
        BacktestOrder("BUY", "AAA", OrderSide.BUY, 5, bars.iloc[0]["timestamp"]),
        BacktestOrder("SELL", "AAA", OrderSide.SELL, 5, bars.iloc[5]["timestamp"]),
    ]
    result = ProfessionalBacktestEngine().run(bars, orders)
    fills = result["fills"]
    assert fills.iloc[0]["timestamp"] == bars.iloc[1]["timestamp"]
    assert fills.iloc[0]["price"] > bars.iloc[1]["open"]
    assert fills.iloc[1]["price"] < bars.iloc[6]["open"]
    assert result["metrics"]["trade_count"] == 1
    assert result["metrics"]["total_execution_costs"] > 0
    assert result["broker_writes"] == 0


def test_partial_fills_respect_volume_participation() -> None:
    config = BacktestConfig(max_volume_participation=0.10)
    order = BacktestOrder("BIG", "AAA", OrderSide.BUY, 25, _bars().iloc[0]["timestamp"])
    result = ProfessionalBacktestEngine(config).run(_bars(), [order])
    assert result["fills"]["quantity"].tolist() == [10, 10, 5]
    assert result["open_orders"] == []


def test_limit_and_stop_trigger_rules() -> None:
    bars = _bars()
    orders = [
        BacktestOrder("LIMIT", "AAA", OrderSide.BUY, 1, bars.iloc[0]["timestamp"], OrderType.LIMIT, limit_price=101),
        BacktestOrder("STOP", "AAA", OrderSide.BUY, 1, bars.iloc[2]["timestamp"], OrderType.STOP, stop_price=107),
    ]
    fills = ProfessionalBacktestEngine().run(bars, orders)["fills"]
    assert fills.loc[fills["order_id"] == "LIMIT", "price"].iloc[0] <= 101
    assert fills.loc[fills["order_id"] == "STOP", "price"].iloc[0] >= 107


def test_corporate_actions_adjust_positions_and_credit_dividends() -> None:
    bars = _bars()
    bars.loc[3, "split_factor"] = 2.0
    bars.loc[4, "dividend"] = 1.0
    order = BacktestOrder("BUY", "AAA", OrderSide.BUY, 3, bars.iloc[0]["timestamp"])
    result = ProfessionalBacktestEngine().run(bars, [order])
    assert result["final_positions"]["AAA"] == 6
    assert result["equity_curve"].iloc[4]["cash"] > result["equity_curve"].iloc[3]["cash"]


def test_target_positions_create_delayed_whole_quantity_orders() -> None:
    targets = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, tz="UTC"),
            "symbol": ["AAA"] * 3,
            "target_quantity": [2, 5, 0],
        }
    )
    orders = orders_from_target_positions(targets)
    assert [order.side for order in orders] == [OrderSide.BUY, OrderSide.BUY, OrderSide.SELL]
    assert [order.quantity for order in orders] == [2, 3, 5]


def test_backtest_rejects_same_bar_execution_configuration() -> None:
    with pytest.raises(ValueError, match="prevent same-bar"):
        BacktestConfig(execution_delay_bars=0)


def test_backtest_requires_point_in_time_availability_not_before_bar() -> None:
    bars = _bars()
    bars.loc[0, "available_at"] = bars.loc[0, "timestamp"] - pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="causality"):
        ProfessionalBacktestEngine().run(bars, [])


def test_delisting_terminal_bar_forces_position_liquidation() -> None:
    bars = _bars()
    bars["delisted"] = False
    bars.loc[len(bars) - 1, "delisted"] = True
    order = BacktestOrder("BUY", "AAA", OrderSide.BUY, 3, bars.iloc[0]["timestamp"])
    result = ProfessionalBacktestEngine().run(bars, [order])
    assert result["final_positions"]["AAA"] == 0
    assert "DELISTING_LIQUIDATION" in set(result["fills"]["order_type"])
    assert result["assumptions"]["delisted_stocks"] == "TERMINAL_BAR_FORCED_LIQUIDATION"
