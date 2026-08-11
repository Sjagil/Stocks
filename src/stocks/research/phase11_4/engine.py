from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PitTrade:
    symbol: str
    sector: str
    signal_date: str
    entry_date: str
    exit_signal_date: str | None
    exit_date: str | None
    entry_price: float
    exit_price: float | None
    rsi: float
    historical_dollar_volume: float
    gross_return: float | None
    net_return: float | None
    holding_sessions: int | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def rsi_wilder(values: pd.Series, period: int) -> pd.Series:
    delta = values.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def generate_trades(
    frames: dict[str, pd.DataFrame],
    *,
    period: int = 3,
    threshold: float = 5.0,
    cost_bps_per_side: float = 10.0,
    minimum_price: float = 5.0,
    minimum_median_dollar_volume: float = 1_000_000.0,
    sectors: dict[str, str] | None = None,
) -> tuple[list[PitTrade], list[dict[str, Any]]]:
    trades: list[PitTrade] = []
    signals: list[dict[str, Any]] = []
    sectors = sectors or {}
    for symbol, frame in sorted(frames.items()):
        frame = frame.sort_index()
        rsi = rsi_wilder(frame["close"], period)
        dollar_volume = (frame["close"] * frame["volume"]).rolling(20, min_periods=20).median()
        generated, generated_signals = _generate_frame_trades(
            symbol,
            sectors.get(symbol, "UNKNOWN"),
            frame,
            rsi.to_numpy(dtype=float),
            dollar_volume.to_numpy(dtype=float),
            threshold=threshold,
            cost_bps_per_side=cost_bps_per_side,
            minimum_price=minimum_price,
            minimum_median_dollar_volume=minimum_median_dollar_volume,
        )
        trades.extend(generated)
        signals.extend(generated_signals)
    return trades, signals


def generate_trade_grid_for_frame(
    symbol: str,
    sector: str,
    frame: pd.DataFrame,
    *,
    periods: tuple[int, ...],
    thresholds: tuple[float, ...],
    cost_bps_per_side: float = 10.0,
) -> dict[tuple[int, float], list[PitTrade]]:
    frame = frame.sort_index()
    dollar_volume = (frame["close"] * frame["volume"]).rolling(20, min_periods=20).median().to_numpy(dtype=float)
    result: dict[tuple[int, float], list[PitTrade]] = {}
    for period in periods:
        rsi = rsi_wilder(frame["close"], period).to_numpy(dtype=float)
        for threshold in thresholds:
            trades, _ = _generate_frame_trades(
                symbol,
                sector,
                frame,
                rsi,
                dollar_volume,
                threshold=threshold,
                cost_bps_per_side=cost_bps_per_side,
                minimum_price=5.0,
                minimum_median_dollar_volume=1_000_000.0,
            )
            result[(period, threshold)] = trades
    return result


def _generate_frame_trades(
    symbol: str,
    sector: str,
    frame: pd.DataFrame,
    rsi_values: np.ndarray,
    liquidity_values: np.ndarray,
    *,
    threshold: float,
    cost_bps_per_side: float,
    minimum_price: float,
    minimum_median_dollar_volume: float,
) -> tuple[list[PitTrade], list[dict[str, Any]]]:
    opens = frame["open"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    dates = frame.index
    if len(frame) < 2:
        return [], []
    eligible = (
        np.isfinite(rsi_values)
        & (rsi_values < threshold)
        & (closes >= minimum_price)
        & np.isfinite(liquidity_values)
        & (liquidity_values >= minimum_median_dollar_volume)
    )
    entry_signals = np.flatnonzero(eligible[:-1])
    exit_signals = np.flatnonzero(closes[1:] > closes[:-1]) + 1
    exit_signals = exit_signals[exit_signals < len(frame) - 1]
    trades: list[PitTrade] = []
    signals: list[dict[str, Any]] = []
    cursor = 0
    cost = cost_bps_per_side / 10_000.0
    while True:
        entry_position = int(np.searchsorted(entry_signals, cursor))
        if entry_position >= len(entry_signals):
            break
        signal_index = int(entry_signals[entry_position])
        entry_index = signal_index + 1
        signal_date = str(dates[signal_index].date())
        entry_date = str(dates[entry_index].date())
        signal_rsi = float(rsi_values[signal_index])
        signal_liquidity = float(liquidity_values[signal_index])
        signal = {
            "symbol": symbol,
            "signal_date": signal_date,
            "rsi": signal_rsi,
            "historical_dollar_volume": signal_liquidity,
            "next_open_date": entry_date,
        }
        signals.append(signal)
        exit_position = int(np.searchsorted(exit_signals, entry_index))
        if exit_position >= len(exit_signals):
            trades.append(
                PitTrade(
                    symbol=symbol,
                    sector=sector,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    exit_signal_date=None,
                    exit_date=None,
                    entry_price=float(opens[entry_index]),
                    exit_price=None,
                    rsi=signal_rsi,
                    historical_dollar_volume=signal_liquidity,
                    gross_return=None,
                    net_return=None,
                    holding_sessions=None,
                    status="DELISTING_EXECUTION_UNCERTAIN",
                )
            )
            break
        exit_signal_index = int(exit_signals[exit_position])
        exit_index = exit_signal_index + 1
        entry_price = float(opens[entry_index])
        exit_price = float(opens[exit_index])
        gross = exit_price / entry_price - 1.0
        net = exit_price * (1 - cost) / (entry_price * (1 + cost)) - 1.0
        trades.append(
            PitTrade(
                symbol=symbol,
                sector=sector,
                signal_date=signal_date,
                entry_date=entry_date,
                exit_signal_date=str(dates[exit_signal_index].date()),
                exit_date=str(dates[exit_index].date()),
                entry_price=entry_price,
                exit_price=exit_price,
                rsi=signal_rsi,
                historical_dollar_volume=signal_liquidity,
                gross_return=gross,
                net_return=net,
                holding_sessions=exit_index - entry_index,
                status="EXECUTION_PRICE_GO",
            )
        )
        cursor = exit_index
    return trades, signals


def trade_summary(trades: list[PitTrade]) -> dict[str, Any]:
    closed = [trade for trade in trades if trade.net_return is not None]
    net_returns = [
        value
        for trade in closed
        if (value := trade.net_return) is not None
    ]
    gross_returns = [
        value
        for trade in closed
        if (value := trade.gross_return) is not None
    ]
    values = np.array(net_returns, dtype=float)
    gross_values = np.array(gross_returns, dtype=float)
    gains = float(values[values > 0].sum()) if len(values) else 0.0
    losses = abs(float(values[values < 0].sum())) if len(values) else 0.0
    raw_gains = float(gross_values[gross_values > 0].sum()) if len(gross_values) else 0.0
    raw_losses = abs(float(gross_values[gross_values < 0].sum())) if len(gross_values) else 0.0
    holds = [int(trade.holding_sessions) for trade in closed if trade.holding_sessions is not None]
    return {
        "trade_count": len(closed),
        "uncertain_delisting_exits": sum(trade.status == "DELISTING_EXECUTION_UNCERTAIN" for trade in trades),
        "win_rate": float((values > 0).mean()) if len(values) else None,
        "trade_profit_factor": None if losses == 0 else gains / losses,
        "raw_profit_factor": None if raw_losses == 0 else raw_gains / raw_losses,
        "expectancy": float(values.mean()) if len(values) else None,
        "average_trade": float(values.mean()) if len(values) else None,
        "median_trade": float(np.median(values)) if len(values) else None,
        "average_holding_period": float(np.mean(holds)) if holds else None,
        "maximum_holding_period": max(holds) if holds else None,
    }


def portfolio_simulation(
    trades: list[PitTrade],
    frames: dict[str, pd.DataFrame],
    *,
    max_positions: int = 4,
    initial_cash: float = 2_000.0,
    allocation: float = 0.25,
    cost_bps_per_side: float = 10.0,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [trade for trade in trades if trade.exit_date and trade.net_return is not None]
    by_entry: dict[str, list[PitTrade]] = defaultdict(list)
    for trade in candidates:
        by_entry[trade.entry_date].append(trade)
    dates = sorted({str(day.date()) for frame in frames.values() for day in frame.index})
    cash = initial_cash
    positions: dict[str, dict[str, Any]] = {}
    equity_rows: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    missed = 0
    turnover_notional = 0.0
    costs_paid = 0.0
    cost = cost_bps_per_side / 10_000.0
    price_maps = {symbol: frame["close"].to_dict() for symbol, frame in frames.items()}
    last_prices: dict[str, float] = {}
    for day in dates:
        ts = pd.Timestamp(day)
        for symbol, values in price_maps.items():
            if ts in values and pd.notna(values[ts]):
                last_prices[symbol] = float(values[ts])
        exits = [symbol for symbol, pos in positions.items() if pos["trade"].exit_date == day]
        for symbol in exits:
            pos = positions.pop(symbol)
            trade = pos["trade"]
            proceeds = pos["shares"] * float(trade.exit_price)
            fee = proceeds * cost
            cash += proceeds - fee
            turnover_notional += proceeds
            costs_paid += fee
            fills.append({"date": day, "symbol": symbol, "side": "SELL", "notional": proceeds, "cost": fee})
        equity_before = cash + sum(pos["shares"] * last_prices.get(symbol, pos["trade"].entry_price) for symbol, pos in positions.items())
        ranked = sorted(by_entry.get(day, []), key=lambda item: (item.rsi, -item.historical_dollar_volume, item.symbol))
        for trade in ranked:
            if trade.symbol in positions or len(positions) >= max_positions:
                missed += 1
                continue
            target = min(equity_before * allocation, cash / (1 + cost))
            if target <= 0:
                missed += 1
                continue
            fee = target * cost
            shares = target / trade.entry_price
            cash -= target + fee
            turnover_notional += target
            costs_paid += fee
            positions[trade.symbol] = {"trade": trade, "shares": shares}
            fills.append({"date": day, "symbol": trade.symbol, "side": "BUY", "notional": target, "cost": fee})
        marked = sum(pos["shares"] * last_prices.get(symbol, pos["trade"].entry_price) for symbol, pos in positions.items())
        nav = cash + marked
        equity_rows.append({"date": day, "nav": nav, "cash": cash, "open_positions": len(positions), "exposure": marked / nav if nav > 0 else 0.0})
    equity = pd.DataFrame(equity_rows).set_index(pd.to_datetime([row["date"] for row in equity_rows]))
    returns = equity["nav"].pct_change().fillna(0.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    peak = equity["nav"].cummax()
    downside = returns[returns < 0].std(ddof=0)
    positive = float(returns[returns > 0].sum())
    negative = abs(float(returns[returns < 0].sum()))
    summary: dict[str, Any] = {
        "initial_capital_eur": initial_cash,
        "terminal_nav_eur": float(equity["nav"].iloc[-1]),
        "CAGR": float((equity["nav"].iloc[-1] / initial_cash) ** (1 / years) - 1),
        "Sharpe": float(returns.mean() / returns.std(ddof=0) * math.sqrt(252)) if returns.std(ddof=0) > 0 else None,
        "Sortino": float(returns.mean() / downside * math.sqrt(252)) if downside and downside > 0 else None,
        "maximum_drawdown": float((equity["nav"] / peak - 1).min()),
        "Calmar": None,
        "period_profit_factor": None if negative == 0 else positive / negative,
        "turnover": turnover_notional / initial_cash,
        "transaction_costs": costs_paid,
        "average_exposure": float(equity["exposure"].mean()),
        "maximum_exposure": float(equity["exposure"].max()),
        "cash_drag": float((equity["cash"] / equity["nav"]).mean()),
        "missed_signals_due_to_capacity": missed,
        "maximum_positions": max_positions,
        "maximum_concurrent_sector_concentration": None,
    }
    maximum_drawdown = float(summary["maximum_drawdown"])
    if maximum_drawdown < 0:
        summary["Calmar"] = float(summary["CAGR"]) / abs(maximum_drawdown)
    return summary, equity_rows, fills


def split_summary(trades: list[PitTrade], start: str, end: str) -> dict[str, Any]:
    return trade_summary([trade for trade in trades if start <= trade.entry_date <= end])
