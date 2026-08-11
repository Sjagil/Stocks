from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np
import pandas as pd


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass(frozen=True)
class BacktestOrder:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    created_at: Any
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None

    def validate(self) -> None:
        if not self.order_id or not self.symbol:
            raise ValueError("order_id and symbol are required")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("quantity must be a positive whole number")
        if self.order_type == OrderType.LIMIT and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("limit orders require a positive limit_price")
        if self.order_type == OrderType.STOP and (self.stop_price is None or self.stop_price <= 0):
            raise ValueError("stop orders require a positive stop_price")


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 100_000.0
    fixed_commission: float = 0.35
    commission_per_unit: float = 0.005
    spread_bps: float = 2.0
    slippage_bps: float = 3.0
    market_impact_bps: float = 1.0
    execution_delay_bars: int = 1
    latency_bars: int = 0
    max_volume_participation: float = 0.10
    annual_periods: int = 252

    def __post_init__(self) -> None:
        if self.initial_cash <= 0 or self.annual_periods <= 0:
            raise ValueError("initial_cash and annual_periods must be positive")
        if min(self.fixed_commission, self.commission_per_unit, self.spread_bps, self.slippage_bps, self.market_impact_bps) < 0:
            raise ValueError("cost assumptions cannot be negative")
        if self.execution_delay_bars < 1 or self.latency_bars < 0:
            raise ValueError("execution delay must prevent same-bar fills")
        if not 0 < self.max_volume_participation <= 1:
            raise ValueError("max_volume_participation must be in (0, 1]")


@dataclass
class _OpenOrder:
    order: BacktestOrder
    remaining: int
    created_position: int


@dataclass
class _Lot:
    quantity: int
    price: float
    timestamp: pd.Timestamp


class ProfessionalBacktestEngine:
    """Causal multi-asset OHLCV execution simulator with realistic fills."""

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def run(self, bars: pd.DataFrame, orders: Iterable[BacktestOrder]) -> dict[str, Any]:
        data = _validated_bars(bars)
        order_list = list(orders)
        if len({order.order_id for order in order_list}) != len(order_list):
            raise ValueError("order_id values must be unique")
        for order in order_list:
            order.validate()
        order_list.sort(key=lambda item: _timestamp(item.created_at))
        timestamps = pd.DatetimeIndex(sorted(data["timestamp"].unique()))
        timestamp_positions = {timestamp: position for position, timestamp in enumerate(timestamps)}
        pending: list[_OpenOrder] = []
        next_order = 0
        cash = self.config.initial_cash
        positions: dict[str, int] = {}
        lots: dict[str, list[_Lot]] = {}
        last_close: dict[str, float] = {}
        fills: list[dict[str, Any]] = []
        closed_trades: list[dict[str, Any]] = []
        equity_rows: list[dict[str, Any]] = []
        total_notional = 0.0
        total_costs = 0.0

        for position, timestamp in enumerate(timestamps):
            while next_order < len(order_list) and _timestamp(order_list[next_order].created_at) <= timestamp:
                order = order_list[next_order]
                created = timestamp_positions.get(_timestamp(order.created_at), position)
                pending.append(_OpenOrder(order=order, remaining=order.quantity, created_position=created))
                next_order += 1
            session = data.loc[data["timestamp"] == timestamp].set_index("symbol")
            for symbol, bar in session.iterrows():
                split_factor = float(bar.get("split_factor", 1.0) or 1.0)
                dividend = float(bar.get("dividend", 0.0) or 0.0)
                if split_factor <= 0:
                    raise ValueError("split_factor must be positive")
                if split_factor != 1.0 and positions.get(symbol, 0):
                    positions[symbol] = int(round(positions[symbol] * split_factor))
                    for lot in lots.get(symbol, []):
                        lot.quantity = int(round(lot.quantity * split_factor))
                        lot.price /= split_factor
                if dividend and positions.get(symbol, 0):
                    cash += positions[symbol] * dividend

            still_pending: list[_OpenOrder] = []
            for open_order in pending:
                order = open_order.order
                symbol = order.symbol.strip().upper()
                if symbol not in session.index:
                    still_pending.append(open_order)
                    continue
                eligible = open_order.created_position + self.config.execution_delay_bars + self.config.latency_bars
                if position < eligible:
                    still_pending.append(open_order)
                    continue
                bar = session.loc[symbol]
                base_price = _trigger_price(order, bar)
                if base_price is None:
                    still_pending.append(open_order)
                    continue
                volume = float(bar["volume"]) if pd.notna(bar["volume"]) else 0.0
                capacity = max(int(math.floor(volume * self.config.max_volume_participation)), 0)
                requested = min(open_order.remaining, capacity)
                if requested <= 0:
                    still_pending.append(open_order)
                    continue
                price = _execution_price(order, base_price, requested, volume, self.config)
                commission = max(self.config.fixed_commission, requested * self.config.commission_per_unit)
                if order.side == OrderSide.BUY:
                    affordable = int(max(math.floor((cash - commission) / price), 0))
                    quantity = min(requested, affordable)
                    if quantity <= 0:
                        still_pending.append(open_order)
                        continue
                    commission = max(self.config.fixed_commission, quantity * self.config.commission_per_unit)
                    cash -= quantity * price + commission
                    positions[symbol] = positions.get(symbol, 0) + quantity
                    lots.setdefault(symbol, []).append(_Lot(quantity, price, timestamp))
                    realized = 0.0
                else:
                    quantity = min(requested, positions.get(symbol, 0))
                    if quantity <= 0:
                        still_pending.append(open_order)
                        continue
                    commission = max(self.config.fixed_commission, quantity * self.config.commission_per_unit)
                    cash += quantity * price - commission
                    positions[symbol] -= quantity
                    realized = _close_lots(lots.setdefault(symbol, []), quantity, price, timestamp, closed_trades, symbol)
                notional = quantity * price
                total_notional += notional
                total_costs += commission + abs(price - base_price) * quantity
                open_order.remaining -= quantity
                fills.append(
                    {
                        "order_id": order.order_id,
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": order.side.value,
                        "order_type": order.order_type.value,
                        "quantity": quantity,
                        "price": price,
                        "reference_price": base_price,
                        "commission": commission,
                        "realized_pnl_before_exit_commission": realized,
                        "remaining": open_order.remaining,
                    }
                )
                if open_order.remaining > 0:
                    still_pending.append(open_order)
            pending = still_pending
            # A point-in-time universe may contain an explicit terminal
            # delisting bar.  Liquidate any remaining long inventory at that
            # bar's close so disappearing securities cannot create a free
            # survivorship benefit.
            if "delisted" in session:
                for symbol, bar in session.loc[session["delisted"].fillna(False).astype(bool)].iterrows():
                    quantity = positions.get(symbol, 0)
                    if quantity <= 0:
                        continue
                    reference = float(bar["close"])
                    synthetic = BacktestOrder(
                        order_id=f"DELISTING-{symbol}-{timestamp.isoformat()}",
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=quantity,
                        created_at=timestamp,
                    )
                    price = _execution_price(synthetic, reference, quantity, float(bar["volume"] or 0.0), self.config)
                    commission = max(self.config.fixed_commission, quantity * self.config.commission_per_unit)
                    cash += quantity * price - commission
                    positions[symbol] = 0
                    realized = _close_lots(lots.setdefault(symbol, []), quantity, price, timestamp, closed_trades, symbol)
                    total_notional += quantity * price
                    total_costs += commission + abs(price - reference) * quantity
                    fills.append(
                        {
                            "order_id": synthetic.order_id,
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "side": "SELL",
                            "order_type": "DELISTING_LIQUIDATION",
                            "quantity": quantity,
                            "price": price,
                            "reference_price": reference,
                            "commission": commission,
                            "realized_pnl_before_exit_commission": realized,
                            "remaining": 0,
                        }
                    )
            for symbol, bar in session.iterrows():
                last_close[symbol] = float(bar["close"])
            market_value = sum(quantity * last_close.get(symbol, 0.0) for symbol, quantity in positions.items())
            equity = cash + market_value
            gross_exposure = sum(abs(quantity * last_close.get(symbol, 0.0)) for symbol, quantity in positions.items())
            equity_rows.append(
                {
                    "timestamp": timestamp,
                    "cash": cash,
                    "market_value": market_value,
                    "equity": equity,
                    "gross_exposure": gross_exposure / equity if equity > 0 else np.nan,
                }
            )

        equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
        metrics = _backtest_metrics(
            equity_curve,
            closed_trades,
            total_notional=total_notional,
            total_costs=total_costs,
            annual_periods=self.config.annual_periods,
        )
        return {
            "schema": "professional_backtest_result_v1",
            "equity_curve": equity_curve,
            "fills": pd.DataFrame(fills),
            "closed_trades": pd.DataFrame(closed_trades),
            "open_orders": [
                {"order_id": item.order.order_id, "remaining": item.remaining}
                for item in pending
            ],
            "final_positions": positions,
            "metrics": metrics,
            "assumptions": {
                "bid_ask_spread": True,
                "commissions": True,
                "slippage": True,
                "latency": True,
                "partial_fills": True,
                "market_limit_stop_orders": True,
                "corporate_actions": True,
                "missing_candles": "NO_FILL_AND_LAST_PRICE_CARRY",
                "execution_delay": self.config.execution_delay_bars,
                "lookahead_same_bar_fill": False,
                "survivorship_bias": "REQUIRES_POINT_IN_TIME_INPUT_UNIVERSE",
                "delisted_stocks": "TERMINAL_BAR_FORCED_LIQUIDATION",
            },
            "research_only": True,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "broker_writes": 0,
        }


def orders_from_target_positions(targets: pd.DataFrame) -> list[BacktestOrder]:
    required = {"timestamp", "symbol", "target_quantity"}
    if not required <= set(targets):
        raise ValueError("targets require timestamp, symbol and target_quantity")
    orders: list[BacktestOrder] = []
    previous: dict[str, int] = {}
    for index, row in targets.sort_values(["timestamp", "symbol"]).iterrows():
        symbol = str(row["symbol"]).upper()
        target = int(row["target_quantity"])
        delta = target - previous.get(symbol, 0)
        if delta:
            orders.append(
                BacktestOrder(
                    order_id=f"TARGET-{index}-{symbol}",
                    symbol=symbol,
                    side=OrderSide.BUY if delta > 0 else OrderSide.SELL,
                    quantity=abs(delta),
                    created_at=row["timestamp"],
                )
            )
        previous[symbol] = target
    return orders


def _validated_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required - set(bars))
    if missing:
        raise ValueError(f"missing backtest bar columns: {', '.join(missing)}")
    data = bars.copy()
    data["symbol"] = data["symbol"].astype("string").str.upper()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data[["symbol", "timestamp", "open", "high", "low", "close"]].isna().any().any():
        raise ValueError("backtest bars contain missing required values")
    if (data[["open", "high", "low", "close"]] <= 0).any().any() or (data["volume"].dropna() < 0).any():
        raise ValueError("backtest prices must be positive and volume non-negative")
    if (data["high"] < data[["open", "low", "close"]].max(axis=1)).any() or (
        data["low"] > data[["open", "high", "close"]].min(axis=1)
    ).any():
        raise ValueError("backtest bars violate OHLC constraints")
    if data.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("backtest bars must be unique by symbol and timestamp")
    if "available_at" in data:
        available = pd.to_datetime(data["available_at"], utc=True, errors="coerce")
        if available.isna().any() or (available < data["timestamp"]).any():
            raise ValueError("bar availability violates causality")
    return data.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _trigger_price(order: BacktestOrder, bar: pd.Series) -> float | None:
    if order.order_type == OrderType.MARKET:
        return float(bar["open"])
    if order.order_type == OrderType.LIMIT:
        limit = float(order.limit_price)
        if order.side == OrderSide.BUY and float(bar["low"]) <= limit:
            return min(float(bar["open"]), limit)
        if order.side == OrderSide.SELL and float(bar["high"]) >= limit:
            return max(float(bar["open"]), limit)
        return None
    stop = float(order.stop_price)
    if order.side == OrderSide.BUY and float(bar["high"]) >= stop:
        return max(float(bar["open"]), stop)
    if order.side == OrderSide.SELL and float(bar["low"]) <= stop:
        return min(float(bar["open"]), stop)
    return None


def _execution_price(order: BacktestOrder, base: float, quantity: int, volume: float, config: BacktestConfig) -> float:
    direction = 1.0 if order.side == OrderSide.BUY else -1.0
    participation = quantity / volume if volume > 0 else 1.0
    bps = config.spread_bps / 2 + config.slippage_bps + config.market_impact_bps * math.sqrt(participation)
    adjusted = base * (1 + direction * bps / 10_000)
    if order.order_type == OrderType.LIMIT:
        adjusted = min(adjusted, float(order.limit_price)) if order.side == OrderSide.BUY else max(adjusted, float(order.limit_price))
    return adjusted


def _close_lots(
    lots: list[_Lot],
    quantity: int,
    price: float,
    timestamp: pd.Timestamp,
    trades: list[dict[str, Any]],
    symbol: str,
) -> float:
    remaining = quantity
    realized = 0.0
    while remaining and lots:
        lot = lots[0]
        closed = min(remaining, lot.quantity)
        pnl = closed * (price - lot.price)
        realized += pnl
        trades.append(
            {
                "symbol": symbol,
                "entry_at": lot.timestamp,
                "exit_at": timestamp,
                "quantity": closed,
                "entry_price": lot.price,
                "exit_price": price,
                "pnl": pnl,
                "duration_bars": max((timestamp - lot.timestamp).days, 0),
            }
        )
        lot.quantity -= closed
        remaining -= closed
        if lot.quantity == 0:
            lots.pop(0)
    return realized


def _backtest_metrics(
    equity: pd.DataFrame,
    trades: list[dict[str, Any]],
    *,
    total_notional: float,
    total_costs: float,
    annual_periods: int,
) -> dict[str, Any]:
    returns = equity["equity"].pct_change(fill_method=None).dropna()
    total_return = float(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1)
    years = max(len(returns) / annual_periods, 1 / annual_periods)
    cagr = float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else -1.0
    volatility = float(returns.std(ddof=1) * math.sqrt(annual_periods)) if len(returns) > 1 else 0.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(annual_periods)) if len(returns) > 1 and returns.std(ddof=1) > 0 else None
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * math.sqrt(annual_periods)) if len(downside) else 0.0
    sortino = float(returns.mean() * annual_periods / downside_deviation) if downside_deviation > 0 else None
    drawdown = equity["equity"] / equity["equity"].cummax() - 1
    maximum_drawdown = float(drawdown.min())
    ulcer = float(np.sqrt(np.mean(np.square(drawdown * 100))))
    pnl = np.asarray([trade["pnl"] for trade in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else None)
    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "profit_factor": profit_factor,
        "expectancy": float(pnl.mean()) if len(pnl) else 0.0,
        "maximum_drawdown": maximum_drawdown,
        "ulcer_index": ulcer,
        "mar_ratio": cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else None,
        "turnover": total_notional / float(equity["equity"].mean()),
        "average_exposure": float(equity["gross_exposure"].mean()),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "average_trade_duration": float(np.mean([trade["duration_bars"] for trade in trades])) if trades else 0.0,
        "tail_loss_5pct": float(np.quantile(returns, 0.05)) if len(returns) else 0.0,
        "trade_count": len(trades),
        "total_execution_costs": total_costs,
    }


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
