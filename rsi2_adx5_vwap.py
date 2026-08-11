#!/usr/bin/env python3
"""
RSI(2) + ADX(5) + Bollinger + VWAP mean-reversion strategy.

Long-only, no leverage, one position at a time.

Core idea
---------
1. Trade only above the long-term trend filter (SMA 200 by default).
2. Require a strong upward directional regime: ADX is high AND +DI > -DI.
3. Detect an extreme pullback: RSI(2) is low and price touches the lower
   Bollinger Band.
4. Require the setup bar to trade at/below VWAP.
5. Do not buy the falling knife immediately. Enter on a later confirmation
   bar, then execute at the next bar's open.
6. Use ATR-based risk, a hard stop, optional profit target, EMA/RSI recovery
   exits, and a maximum holding period.

The script can:
- load OHLCV from CSV;
- optionally download one ticker with yfinance;
- backtest with fees and slippage;
- save trades, equity curve, enriched candles, summary JSON, and a chart;
- run a parameter sweep for RSI and ADX thresholds.

CSV columns (case-insensitive):
    timestamp, open, high, low, close, volume
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class StrategyConfig:
    initial_capital: float = 10_000.0

    # Signal indicators
    rsi_period: int = 2
    rsi_entry: float = 10.0
    rsi_exit: float = 60.0
    adx_period: int = 5
    adx_entry: float = 25.0
    trend_sma_period: int = 200
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    exit_ema_period: int = 5

    # VWAP
    # auto: session VWAP for intraday, rolling VWAP for daily/weekly bars
    # session: reset every UTC/session date
    # rolling: rolling volume-weighted typical price
    vwap_kind: str = "auto"
    vwap_window: int = 20
    vwap_timezone: str = "UTC"
    # discount: setup below VWAP, confirm by breaking setup high
    # reclaim: setup below VWAP, confirm by closing back above VWAP
    # either: either break setup high or reclaim VWAP
    # off: ignore VWAP
    vwap_mode: str = "either"
    vwap_max_premium: float = 0.0

    # Entry timing
    confirmation_bars: int = 3
    require_bullish_confirmation: bool = True
    require_positive_dmi: bool = False

    # Risk and exits
    risk_per_trade: float = 0.005
    max_position_fraction: float = 0.20
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 1.5
    trailing_atr_mult: float = 0.0
    max_hold_bars: int = 10
    min_hold_bars: int = 1

    # Execution assumptions
    fee_rate: float = 0.001
    slippage_bps: float = 5.0
    minimum_notional: float = 10.0


@dataclass
class PendingEntry:
    signal_time: pd.Timestamp
    signal_index: int
    atr: float
    reason: str


@dataclass
class Position:
    entry_time: pd.Timestamp
    entry_index: int
    entry_price: float
    quantity: float
    entry_fee: float
    stop_price: float
    target_price: float
    highest_price: float
    signal_reason: str


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    bars_held: int
    exit_reason: str
    signal_reason: str


def _validate_config(cfg: StrategyConfig) -> None:
    if cfg.initial_capital <= 0:
        raise ValueError("initial_capital must be positive")
    if not 0 < cfg.risk_per_trade <= 1:
        raise ValueError("risk_per_trade must be in (0, 1]")
    if not 0 < cfg.max_position_fraction <= 1:
        raise ValueError("max_position_fraction must be in (0, 1]")
    if cfg.fee_rate < 0 or cfg.slippage_bps < 0:
        raise ValueError("fee_rate and slippage_bps cannot be negative")
    if cfg.atr_stop_mult <= 0:
        raise ValueError("atr_stop_mult must be positive")
    if cfg.atr_target_mult < 0:
        raise ValueError("atr_target_mult cannot be negative")
    if cfg.confirmation_bars < 1:
        raise ValueError("confirmation_bars must be at least 1")
    if cfg.vwap_kind not in {"auto", "session", "rolling"}:
        raise ValueError("vwap_kind must be auto, session, or rolling")
    if cfg.vwap_mode not in {"discount", "reclaim", "either", "off"}:
        raise ValueError("vwap_mode must be discount, reclaim, either, or off")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No rows found in the input data")

    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    timestamp_candidates = [
        "timestamp",
        "datetime",
        "date",
        "time",
        "open_time",
    ]
    timestamp_col = next((c for c in timestamp_candidates if c in out.columns), None)

    if timestamp_col is not None:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], utc=True, errors="coerce")
        out = out.set_index(timestamp_col)
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError(
            "CSV needs a timestamp/date/datetime column or a DatetimeIndex"
        )
    else:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[required]
    out = out[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
    out = out[out["volume"] >= 0]

    if len(out) < 250:
        raise ValueError(
            f"Only {len(out)} valid rows remain. Use at least 250 bars, preferably much more."
        )

    return out.astype(float)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return _normalize_ohlcv(pd.read_csv(path))


def load_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run: pip install yfinance"
        ) from exc

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may return either (field, ticker) or (ticker, field).
        level0 = {str(x).lower() for x in raw.columns.get_level_values(0)}
        if {"open", "high", "low", "close", "volume"}.issubset(level0):
            raw.columns = raw.columns.get_level_values(0)
        else:
            raw.columns = raw.columns.get_level_values(-1)

    raw = raw.rename(columns={str(c): str(c).lower() for c in raw.columns})
    return _normalize_ohlcv(raw.reset_index())


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder-style recursive moving average."""
    if period < 1:
        raise ValueError("period must be at least 1")
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    out = out.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return out.clip(0.0, 100.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return rma(true_range(df), period)


def adx_components(df: pd.DataFrame, period: int = 5) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )

    atr_n = rma(true_range(df), period)
    plus_di = 100.0 * rma(plus_dm, period) / atr_n.replace(0.0, np.nan)
    minus_di = 100.0 * rma(minus_dm, period) / atr_n.replace(0.0, np.nan)
    denominator = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denominator
    adx_value = rma(dx, period)
    return adx_value, plus_di, minus_di


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return middle, upper, lower


def infer_bar_seconds(index: pd.DatetimeIndex) -> float:
    diffs = index.to_series().diff().dt.total_seconds().dropna()
    if diffs.empty:
        return float("nan")
    return float(diffs.median())


def calculate_vwap(
    df: pd.DataFrame, kind: str, window: int, timezone: str
) -> tuple[pd.Series, str]:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = df["volume"].clip(lower=0.0)
    selected = kind

    if selected == "auto":
        bar_seconds = infer_bar_seconds(df.index)
        selected = "rolling" if (not math.isfinite(bar_seconds) or bar_seconds >= 20 * 3600) else "session"

    if selected == "session":
        # UTC-day reset. For exchange-specific equities, preprocess timestamps into
        # the exchange timezone before loading if that session boundary matters.
        try:
            localized_index = df.index.tz_convert(timezone)
        except Exception as exc:
            raise ValueError(f"Invalid VWAP timezone: {timezone}") from exc
        session_key = pd.Series(localized_index.floor("D"), index=df.index)
        cumulative_pv = (typical * volume).groupby(session_key).cumsum()
        cumulative_volume = volume.groupby(session_key).cumsum()
        vwap = cumulative_pv / cumulative_volume.replace(0.0, np.nan)
    elif selected == "rolling":
        pv_sum = (typical * volume).rolling(window, min_periods=window).sum()
        volume_sum = volume.rolling(window, min_periods=window).sum()
        vwap = pv_sum / volume_sum.replace(0.0, np.nan)
    else:
        raise ValueError(f"Unknown VWAP kind: {selected}")

    return vwap, selected


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> tuple[pd.DataFrame, str]:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["adx"], out["plus_di"], out["minus_di"] = adx_components(out, cfg.adx_period)
    out["trend_sma"] = out["close"].rolling(
        cfg.trend_sma_period, min_periods=cfg.trend_sma_period
    ).mean()
    out["bb_mid"], out["bb_upper"], out["bb_lower"] = bollinger(
        out["close"], cfg.bb_period, cfg.bb_std
    )
    out["atr"] = atr(out, cfg.atr_period)
    out["exit_ema"] = out["close"].ewm(
        span=cfg.exit_ema_period,
        adjust=False,
        min_periods=cfg.exit_ema_period,
    ).mean()
    out["vwap"], selected_vwap_kind = calculate_vwap(
        out, cfg.vwap_kind, cfg.vwap_window, cfg.vwap_timezone
    )
    return out, selected_vwap_kind


def _finite(*values: Any) -> bool:
    return all(pd.notna(v) and np.isfinite(float(v)) for v in values)


def _setup_condition(row: pd.Series, cfg: StrategyConfig) -> bool:
    if not _finite(
        row["close"],
        row["rsi"],
        row["adx"],
        row["plus_di"],
        row["minus_di"],
        row["trend_sma"],
        row["bb_lower"],
        row["atr"],
        row["vwap"],
    ):
        return False

    trend_ok = row["close"] > row["trend_sma"]
    direction_ok = (row["plus_di"] > row["minus_di"]) if cfg.require_positive_dmi else True
    strength_ok = row["adx"] >= cfg.adx_entry
    oversold_ok = row["rsi"] <= cfg.rsi_entry
    band_ok = row["close"] <= row["bb_lower"]

    if cfg.vwap_mode == "off":
        vwap_ok = True
    else:
        vwap_ok = row["close"] <= row["vwap"] * (1.0 + cfg.vwap_max_premium)

    return bool(trend_ok and direction_ok and strength_ok and oversold_ok and band_ok and vwap_ok)


def _confirmation_condition(
    previous_row: pd.Series,
    row: pd.Series,
    setup_high: float,
    cfg: StrategyConfig,
) -> tuple[bool, str]:
    if not _finite(
        row["close"],
        row["open"],
        row["vwap"],
        row["trend_sma"],
        row["plus_di"],
        row["minus_di"],
        row["atr"],
    ):
        return False, ""

    dmi_ok = (row["plus_di"] > row["minus_di"]) if cfg.require_positive_dmi else True
    regime_still_valid = row["close"] > row["trend_sma"] and dmi_ok
    if not regime_still_valid:
        return False, ""

    bullish = row["close"] > row["open"] if cfg.require_bullish_confirmation else True
    breaks_setup_high = row["close"] > setup_high
    reclaims_vwap = row["close"] > row["vwap"]
    if _finite(previous_row.get("close", np.nan), previous_row.get("vwap", np.nan)):
        reclaims_vwap = reclaims_vwap and previous_row["close"] <= previous_row["vwap"]

    if cfg.vwap_mode in {"discount", "off"}:
        confirmed = breaks_setup_high
        reason = "break_setup_high"
    elif cfg.vwap_mode == "reclaim":
        confirmed = reclaims_vwap
        reason = "vwap_reclaim"
    elif cfg.vwap_mode == "either":
        confirmed = breaks_setup_high or reclaims_vwap
        if reclaims_vwap and breaks_setup_high:
            reason = "vwap_reclaim+break_setup_high"
        elif reclaims_vwap:
            reason = "vwap_reclaim"
        else:
            reason = "break_setup_high"
    else:
        raise ValueError(f"Unknown vwap_mode: {cfg.vwap_mode}")

    return bool(bullish and confirmed), reason


def _apply_buy_slippage(price: float, slippage_bps: float) -> float:
    return price * (1.0 + slippage_bps / 10_000.0)


def _apply_sell_slippage(price: float, slippage_bps: float) -> float:
    return price * (1.0 - slippage_bps / 10_000.0)


def _position_size(
    cash: float,
    equity: float,
    entry_price: float,
    stop_price: float,
    cfg: StrategyConfig,
) -> float:
    stop_distance = max(entry_price - stop_price, entry_price * 0.001)
    risk_budget = max(0.0, equity * cfg.risk_per_trade)
    quantity_by_risk = risk_budget / stop_distance
    capital_cap = min(cash, equity * cfg.max_position_fraction)
    quantity_by_capital = capital_cap / (entry_price * (1.0 + cfg.fee_rate))
    quantity = min(quantity_by_risk, quantity_by_capital)
    if not math.isfinite(quantity) or quantity <= 0:
        return 0.0
    if quantity * entry_price < cfg.minimum_notional:
        return 0.0
    return float(quantity)


def infer_annualization(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    span_years = max((index[-1] - index[0]).total_seconds() / (365.25 * 86400), EPS)
    observed = (len(index) - 1) / span_years
    return float(max(observed, 1.0))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max.replace(0.0, np.nan) - 1.0
    return float(drawdown.min())


def _safe_profit_factor(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    gross_profit = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum())
    gross_loss = abs(float(trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()))
    if gross_loss <= EPS:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def calculate_metrics(
    equity_curve: pd.DataFrame,
    trades: pd.DataFrame,
    cfg: StrategyConfig,
) -> dict[str, Any]:
    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ending_equity = float(equity.iloc[-1])
    total_return = ending_equity / cfg.initial_capital - 1.0

    span_years = max(
        (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
        / (365.25 * 86400),
        EPS,
    )
    cagr = (ending_equity / cfg.initial_capital) ** (1.0 / span_years) - 1.0
    annualization = infer_annualization(equity_curve.index)

    mean_ret = float(returns.mean()) if not returns.empty else 0.0
    std_ret = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 1 else 0.0
    sharpe = mean_ret / std_ret * math.sqrt(annualization) if std_ret > EPS else 0.0
    sortino = mean_ret / downside_std * math.sqrt(annualization) if downside_std > EPS else 0.0
    mdd = max_drawdown(equity)

    trade_count = int(len(trades))
    wins = int((trades["net_pnl"] > 0).sum()) if trade_count else 0
    win_rate = wins / trade_count if trade_count else 0.0
    expectancy = float(trades["net_pnl"].mean()) if trade_count else 0.0
    avg_trade_return = float(trades["return_pct"].mean()) if trade_count else 0.0
    avg_bars_held = float(trades["bars_held"].mean()) if trade_count else 0.0
    fees = float(trades["fees"].sum()) if trade_count else 0.0
    exposure = float(equity_curve["in_position"].mean())

    first_close = float(equity_curve["close"].iloc[0])
    last_close = float(equity_curve["close"].iloc[-1])
    buy_hold_return = last_close / first_close - 1.0

    return {
        "starting_equity": cfg.initial_capital,
        "ending_equity": ending_equity,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": _safe_profit_factor(trades),
        "expectancy_currency": expectancy,
        "average_trade_return": avg_trade_return,
        "average_bars_held": avg_bars_held,
        "exposure": exposure,
        "fees_paid": fees,
        "buy_hold_return": buy_hold_return,
        "annualization_factor": annualization,
    }


def backtest(
    raw_df: pd.DataFrame,
    cfg: StrategyConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _validate_config(cfg)
    df, selected_vwap_kind = add_indicators(raw_df, cfg)

    cash = cfg.initial_capital
    position: Optional[Position] = None
    pending: Optional[PendingEntry] = None
    setup: Optional[dict[str, Any]] = None
    trades: list[Trade] = []
    equity_rows: list[dict[str, Any]] = []
    slip = cfg.slippage_bps

    for i in range(len(df)):
        timestamp = df.index[i]
        row = df.iloc[i]
        previous_row = df.iloc[i - 1] if i > 0 else row

        # Execute confirmed entry at this bar's open. The signal was generated
        # only after the preceding bar closed, so there is no close-to-open lookahead.
        if position is None and pending is not None:
            raw_entry = float(row["open"])
            entry_price = _apply_buy_slippage(raw_entry, slip)
            if _finite(pending.atr) and pending.atr > 0:
                stop_price = max(EPS, entry_price - cfg.atr_stop_mult * pending.atr)
                if cfg.atr_target_mult > 0:
                    target_price = entry_price + cfg.atr_target_mult * pending.atr
                else:
                    target_price = float("inf")

                equity_before_entry = cash
                quantity = _position_size(
                    cash,
                    equity_before_entry,
                    entry_price,
                    stop_price,
                    cfg,
                )
                entry_notional = quantity * entry_price
                entry_fee = entry_notional * cfg.fee_rate
                total_cost = entry_notional + entry_fee

                if quantity > 0 and total_cost <= cash + 1e-8:
                    cash -= total_cost
                    position = Position(
                        entry_time=timestamp,
                        entry_index=i,
                        entry_price=entry_price,
                        quantity=quantity,
                        entry_fee=entry_fee,
                        stop_price=stop_price,
                        target_price=target_price,
                        highest_price=float(row["high"]),
                        signal_reason=pending.reason,
                    )
            pending = None

        exit_reason: Optional[str] = None
        exit_price: Optional[float] = None

        if position is not None:
            position.highest_price = max(position.highest_price, float(row["high"]))

            if cfg.trailing_atr_mult > 0 and _finite(row["atr"]):
                trailing_stop = position.highest_price - cfg.trailing_atr_mult * float(row["atr"])
                position.stop_price = max(position.stop_price, trailing_stop)

            # Gaps are executed at the open. Intrabar ambiguity is resolved
            # conservatively: if stop and target are both touched, stop wins.
            if float(row["open"]) <= position.stop_price:
                exit_price = _apply_sell_slippage(float(row["open"]), slip)
                exit_reason = "stop_gap"
            elif float(row["low"]) <= position.stop_price:
                exit_price = _apply_sell_slippage(position.stop_price, slip)
                exit_reason = "stop_loss"
            elif math.isfinite(position.target_price) and float(row["open"]) >= position.target_price:
                exit_price = _apply_sell_slippage(float(row["open"]), slip)
                exit_reason = "target_gap"
            elif math.isfinite(position.target_price) and float(row["high"]) >= position.target_price:
                exit_price = _apply_sell_slippage(position.target_price, slip)
                exit_reason = "take_profit"
            else:
                bars_held = i - position.entry_index
                ema_cross = False
                if i > 0 and _finite(
                    row["close"], row["exit_ema"], previous_row["close"], previous_row["exit_ema"]
                ):
                    ema_cross = (
                        previous_row["close"] < previous_row["exit_ema"]
                        and row["close"] >= row["exit_ema"]
                    )

                if bars_held >= cfg.min_hold_bars and ema_cross:
                    exit_price = _apply_sell_slippage(float(row["close"]), slip)
                    exit_reason = "ema_recovery"
                elif bars_held >= cfg.min_hold_bars and _finite(row["rsi"]) and row["rsi"] >= cfg.rsi_exit:
                    exit_price = _apply_sell_slippage(float(row["close"]), slip)
                    exit_reason = "rsi_recovery"
                elif bars_held >= cfg.max_hold_bars:
                    exit_price = _apply_sell_slippage(float(row["close"]), slip)
                    exit_reason = "time_exit"

            if exit_reason is not None and exit_price is not None:
                exit_notional = position.quantity * exit_price
                exit_fee = exit_notional * cfg.fee_rate
                cash += exit_notional - exit_fee
                gross_pnl = position.quantity * (exit_price - position.entry_price)
                total_fees = position.entry_fee + exit_fee
                net_pnl = gross_pnl - total_fees
                cost_basis = position.quantity * position.entry_price + position.entry_fee
                trade_return = net_pnl / cost_basis if cost_basis > EPS else 0.0

                trades.append(
                    Trade(
                        entry_time=position.entry_time.isoformat(),
                        exit_time=timestamp.isoformat(),
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        quantity=position.quantity,
                        gross_pnl=gross_pnl,
                        fees=total_fees,
                        net_pnl=net_pnl,
                        return_pct=trade_return,
                        bars_held=i - position.entry_index,
                        exit_reason=exit_reason,
                        signal_reason=position.signal_reason,
                    )
                )
                position = None
                setup = None

        # Build or confirm setup only at bar close.
        if position is None and pending is None:
            if setup is not None:
                expired = i > int(setup["expiry_index"])
                if expired:
                    setup = None
                else:
                    confirmed, reason = _confirmation_condition(
                        previous_row,
                        row,
                        float(setup["high"]),
                        cfg,
                    )
                    if confirmed:
                        pending = PendingEntry(
                            signal_time=timestamp,
                            signal_index=i,
                            atr=float(row["atr"]),
                            reason=reason,
                        )
                        setup = None

            # A newer, more extreme setup replaces the old setup. Confirmation
            # is never allowed on the same bar that creates the setup.
            if pending is None and _setup_condition(row, cfg):
                setup = {
                    "time": timestamp,
                    "index": i,
                    "high": float(row["high"]),
                    "expiry_index": i + cfg.confirmation_bars,
                }

        mark_to_market = cash
        in_position = 0
        if position is not None:
            mark_to_market += position.quantity * float(row["close"])
            in_position = 1

        equity_rows.append(
            {
                "timestamp": timestamp,
                "close": float(row["close"]),
                "cash": cash,
                "equity": mark_to_market,
                "in_position": in_position,
            }
        )

    # Force-close at the final close so summary equity and trades reconcile.
    if position is not None:
        timestamp = df.index[-1]
        close_price = _apply_sell_slippage(float(df["close"].iloc[-1]), slip)
        exit_notional = position.quantity * close_price
        exit_fee = exit_notional * cfg.fee_rate
        cash += exit_notional - exit_fee
        gross_pnl = position.quantity * (close_price - position.entry_price)
        total_fees = position.entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees
        cost_basis = position.quantity * position.entry_price + position.entry_fee
        trade_return = net_pnl / cost_basis if cost_basis > EPS else 0.0
        trades.append(
            Trade(
                entry_time=position.entry_time.isoformat(),
                exit_time=timestamp.isoformat(),
                entry_price=position.entry_price,
                exit_price=close_price,
                quantity=position.quantity,
                gross_pnl=gross_pnl,
                fees=total_fees,
                net_pnl=net_pnl,
                return_pct=trade_return,
                bars_held=len(df) - 1 - position.entry_index,
                exit_reason="end_of_data",
                signal_reason=position.signal_reason,
            )
        )
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["in_position"] = 0

    equity_curve = pd.DataFrame(equity_rows).set_index("timestamp")
    trades_df = pd.DataFrame([asdict(t) for t in trades])
    if trades_df.empty:
        trades_df = pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "quantity",
                "gross_pnl",
                "fees",
                "net_pnl",
                "return_pct",
                "bars_held",
                "exit_reason",
                "signal_reason",
            ]
        )

    metrics = calculate_metrics(equity_curve, trades_df, cfg)
    metrics["selected_vwap_kind"] = selected_vwap_kind
    metrics["configuration"] = asdict(cfg)
    return metrics, trades_df, equity_curve, df


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isnan(number):
            return None
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        return number
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def save_chart(equity_curve: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, chart skipped", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_curve.index, equity_curve["equity"], label="Strategy equity")
    normalized_bh = (
        equity_curve["close"] / equity_curve["close"].iloc[0] * equity_curve["equity"].iloc[0]
    )
    ax.plot(equity_curve.index, normalized_bh, label="Buy & hold")
    ax.set_title("RSI(2) + ADX(5) + VWAP strategy")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(metrics: dict[str, Any]) -> None:
    pf = metrics["profit_factor"]
    pf_text = "inf" if pf == float("inf") else f"{pf:.3f}"
    print("\n=== BACKTEST SUMMARY ===")
    print(f"VWAP kind          : {metrics['selected_vwap_kind']}")
    print(f"Starting equity    : {metrics['starting_equity']:.2f}")
    print(f"Ending equity      : {metrics['ending_equity']:.2f}")
    print(f"Total return       : {metrics['total_return']:.2%}")
    print(f"CAGR               : {metrics['cagr']:.2%}")
    print(f"Max drawdown       : {metrics['max_drawdown']:.2%}")
    print(f"Sharpe             : {metrics['sharpe']:.3f}")
    print(f"Sortino            : {metrics['sortino']:.3f}")
    print(f"Trades             : {metrics['trade_count']}")
    print(f"Win rate           : {metrics['win_rate']:.2%}")
    print(f"Profit factor      : {pf_text}")
    print(f"Avg trade return   : {metrics['average_trade_return']:.3%}")
    print(f"Exposure           : {metrics['exposure']:.2%}")
    print(f"Fees paid          : {metrics['fees_paid']:.2f}")
    print(f"Buy & hold return  : {metrics['buy_hold_return']:.2%}")


def parse_number_list(text: str, cast: Any = float) -> list[Any]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(cast(item))
    if not values:
        raise argparse.ArgumentTypeError("List cannot be empty")
    return values


def score_sweep_result(metrics: dict[str, Any], min_trades: int) -> float:
    trades = int(metrics["trade_count"])
    if trades < min_trades:
        return -1e9 + trades
    pf = metrics["profit_factor"]
    pf_capped = 5.0 if pf == float("inf") else min(float(pf), 5.0)
    # Penalize drawdown and low sample size. This score is for ranking research
    # candidates, not proof of out-of-sample profitability.
    return (
        float(metrics["total_return"])
        + 0.10 * pf_capped
        + 0.02 * min(trades, 100) / 100.0
        + 0.50 * float(metrics["max_drawdown"])
    )


def run_sweep(
    df: pd.DataFrame,
    base_cfg: StrategyConfig,
    rsi_values: Iterable[float],
    adx_values: Iterable[float],
    stop_values: Iterable[float],
    target_values: Iterable[float],
    vwap_modes: Iterable[str],
    min_trades: int,
    output_dir: Path,
) -> pd.DataFrame:
    combinations = list(
        itertools.product(rsi_values, adx_values, stop_values, target_values, vwap_modes)
    )
    print(f"Running {len(combinations)} parameter combinations...")
    rows: list[dict[str, Any]] = []

    for n, (rsi_entry, adx_entry, stop_mult, target_mult, vwap_mode) in enumerate(combinations, 1):
        cfg = replace(
            base_cfg,
            rsi_entry=float(rsi_entry),
            adx_entry=float(adx_entry),
            atr_stop_mult=float(stop_mult),
            atr_target_mult=float(target_mult),
            vwap_mode=str(vwap_mode),
        )
        metrics, _, _, _ = backtest(df, cfg)
        row = {
            "rsi_entry": rsi_entry,
            "adx_entry": adx_entry,
            "atr_stop_mult": stop_mult,
            "atr_target_mult": target_mult,
            "vwap_mode": vwap_mode,
            "score": score_sweep_result(metrics, min_trades),
            **{k: v for k, v in metrics.items() if k != "configuration"},
        }
        rows.append(row)
        if n == 1 or n % max(1, len(combinations) // 20) == 0 or n == len(combinations):
            print(f"[{n:>4}/{len(combinations)}] completed")

    results = pd.DataFrame(rows).sort_values(
        ["score", "total_return", "profit_factor"], ascending=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "sweep_results.csv", index=False)

    columns = [
        "rsi_entry",
        "adx_entry",
        "atr_stop_mult",
        "atr_target_mult",
        "vwap_mode",
        "trade_count",
        "win_rate",
        "profit_factor",
        "total_return",
        "max_drawdown",
        "score",
    ]
    print("\n=== TOP SWEEP RESULTS ===")
    print(results[columns].head(20).to_string(index=False))
    print(
        "\nWarning: the sweep ranks in-sample results. Do not treat the top row as a "
        "validated edge. Re-test selected parameters on untouched data."
    )
    return results



def rsi2_adx5_vwap_schema() -> dict[str, Any]:
    """Return the stable offline research contract exposed through main.py."""
    return {
        "schema": "rsi2_adx5_vwap_strategy_contract_v1",
        "status": "GO",
        "strategy_id": "RSI2_ADX5_BOLLINGER_VWAP_MEAN_REVERSION_V1",
        "authority": "offline_backtest_and_parameter_research_only",
        "position_model": {
            "direction": "long_only",
            "leverage": False,
            "short_selling": False,
            "maximum_concurrent_positions": 1,
        },
        "entry_model": {
            "trend_filter": "close_above_long_term_sma",
            "trend_strength": "adx_at_or_above_threshold",
            "pullback": "rsi2_oversold_and_close_at_or_below_lower_bollinger_band",
            "vwap_filter": ["discount", "reclaim", "either", "off"],
            "timing": "setup_on_closed_bar_confirmation_on_later_closed_bar_fill_at_next_open",
            "optional_direction_filter": "plus_di_above_minus_di",
        },
        "risk_model": {
            "position_sizing": "min(account_risk_divided_by_stop_distance, max_position_fraction)",
            "stop": "atr_multiple",
            "target": "optional_atr_multiple",
            "recovery_exits": ["rsi_exit", "ema_exit", "time_exit", "trailing_atr"],
            "costs": ["entry_fee", "exit_fee", "entry_slippage", "exit_slippage"],
        },
        "vwap": {
            "auto": "session_vwap_for_intraday_rolling_vwap_for_daily_or_slower",
            "session": "timezone_session_reset",
            "rolling": "rolling_volume_weighted_typical_price",
        },
        "default_configuration": asdict(StrategyConfig()),
        "required_csv_columns": ["timestamp", "open", "high", "low", "close", "volume"],
        "outputs": [
            "summary.json",
            "trades.csv",
            "equity.csv",
            "indicators.csv",
            "equity.png",
            "sweep_results.csv",
        ],
        "execution": {
            "orders_enabled": False,
            "order_intents_enabled": False,
            "broker_calls_enabled": False,
        },
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def _resolve_job_source(
    *,
    csv_path: Optional[Path],
    ticker: Optional[str],
    period: str,
    interval: str,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    normalized_interval = str(interval).strip().lower()
    aliases = {"1wk": "1w", "1month": "1mo"}
    canonical_interval = aliases.get(normalized_interval, normalized_interval)
    allowed = {"1h", "2h", "4h", "6h", "12h", "1d", "1w", "1mo"}
    if canonical_interval not in allowed:
        raise ValueError(
            f"FORBIDDEN_OR_UNSUPPORTED_SWING_TIMEFRAME:{normalized_interval}"
        )
    if (csv_path is None) == (ticker is None):
        raise ValueError("Provide exactly one data source: csv_path or ticker")
    if csv_path is not None:
        resolved = csv_path.expanduser().resolve()
        frame = load_csv(resolved)
        return frame, resolved.stem, {
            "kind": "csv",
            "path": str(resolved),
        }
    assert ticker is not None
    if canonical_interval not in {"1h", "1d", "1w", "1mo"}:
        raise ValueError(
            "DERIVED_SWING_TIMEFRAME_REQUIRES_CANONICAL_LOCAL_CACHE"
        )
    provider_interval = {"1w": "1wk"}.get(canonical_interval, canonical_interval)
    frame = load_yfinance(ticker, period, provider_interval)
    return frame, str(ticker).replace("/", "_"), {
        "kind": "yfinance",
        "ticker": ticker,
        "period": period,
        "interval": interval,
    }


def run_rsi2_adx5_vwap_backtest_job(
    *,
    csv_path: Optional[Path],
    ticker: Optional[str],
    period: str,
    interval: str,
    output_dir: Path,
    config: StrategyConfig,
) -> dict[str, Any]:
    """Run one deterministic backtest and persist all research artifacts."""
    frame, source_name, source = _resolve_job_source(
        csv_path=csv_path,
        ticker=ticker,
        period=period,
        interval=interval,
    )
    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    metrics, trades, equity_curve, enriched = backtest(frame, config)
    prefix = source_name.replace(" ", "_")
    artifact_paths = {
        "summary": resolved_output / f"{prefix}_summary.json",
        "trades": resolved_output / f"{prefix}_trades.csv",
        "equity": resolved_output / f"{prefix}_equity.csv",
        "indicators": resolved_output / f"{prefix}_indicators.csv",
        "chart": resolved_output / f"{prefix}_equity.png",
    }
    trades.to_csv(artifact_paths["trades"], index=False)
    equity_curve.to_csv(artifact_paths["equity"])
    enriched.to_csv(artifact_paths["indicators"])
    artifact_paths["summary"].write_text(
        json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    save_chart(equity_curve, artifact_paths["chart"])

    return _json_safe(
        {
            "schema": "rsi2_adx5_vwap_backtest_command_v1",
            "status": "GO",
            "strategy_id": "RSI2_ADX5_BOLLINGER_VWAP_MEAN_REVERSION_V1",
            "authority": "offline_backtest_only",
            "source": source,
            "dataset": {
                "rows": len(frame),
                "start": frame.index.min(),
                "end": frame.index.max(),
            },
            "metrics": metrics,
            "artifacts": {key: str(path) for key, path in artifact_paths.items()},
            "execution": {
                "orders_enabled": False,
                "order_intents_enabled": False,
                "broker_calls_enabled": False,
            },
            "financial_calls": {
                "place_order": 0,
                "cancel_order": 0,
                "global_cancel": 0,
            },
        }
    )


def run_rsi2_adx5_vwap_sweep_job(
    *,
    csv_path: Optional[Path],
    ticker: Optional[str],
    period: str,
    interval: str,
    output_dir: Path,
    base_config: StrategyConfig,
    rsi_values: Iterable[float],
    adx_values: Iterable[float],
    stop_values: Iterable[float],
    target_values: Iterable[float],
    vwap_modes: Iterable[str],
    min_trades: int,
) -> dict[str, Any]:
    """Run an in-sample parameter sweep through the canonical main.py entrypoint."""
    frame, _, source = _resolve_job_source(
        csv_path=csv_path,
        ticker=ticker,
        period=period,
        interval=interval,
    )
    resolved_output = output_dir.expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    modes = [str(value).strip() for value in vwap_modes if str(value).strip()]
    invalid_modes = set(modes) - {"discount", "reclaim", "either", "off"}
    if invalid_modes:
        raise ValueError(f"Invalid sweep VWAP modes: {sorted(invalid_modes)}")

    rsi_values = list(rsi_values)
    adx_values = list(adx_values)
    stop_values = list(stop_values)
    target_values = list(target_values)
    combination_count = (
        len(rsi_values)
        * len(adx_values)
        * len(stop_values)
        * len(target_values)
        * len(modes)
    )
    results = run_sweep(
        df=frame,
        base_cfg=base_config,
        rsi_values=rsi_values,
        adx_values=adx_values,
        stop_values=stop_values,
        target_values=target_values,
        vwap_modes=modes,
        min_trades=min_trades,
        output_dir=resolved_output,
    )
    top_columns = [
        "rsi_entry",
        "adx_entry",
        "atr_stop_mult",
        "atr_target_mult",
        "vwap_mode",
        "trade_count",
        "win_rate",
        "profit_factor",
        "total_return",
        "max_drawdown",
        "score",
    ]
    return _json_safe(
        {
            "schema": "rsi2_adx5_vwap_sweep_command_v1",
            "status": "GO",
            "strategy_id": "RSI2_ADX5_BOLLINGER_VWAP_MEAN_REVERSION_V1",
            "authority": "offline_parameter_research_only",
            "source": source,
            "dataset": {
                "rows": len(frame),
                "start": frame.index.min(),
                "end": frame.index.max(),
            },
            "combination_count": combination_count,
            "minimum_trades_for_ranking": min_trades,
            "top_results": results[top_columns].head(20).to_dict(orient="records"),
            "artifacts": {
                "sweep_results": str(resolved_output / "sweep_results.csv"),
            },
            "validation_warning": (
                "These rankings are in-sample research results and are not evidence of "
                "out-of-sample profitability."
            ),
            "execution": {
                "orders_enabled": False,
                "order_intents_enabled": False,
                "broker_calls_enabled": False,
            },
            "financial_calls": {
                "place_order": 0,
                "cancel_order": 0,
                "global_cancel": 0,
            },
        }
    )

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest RSI(2) + ADX(5) + Bollinger + VWAP strategy"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", type=Path, help="OHLCV CSV file")
    source.add_argument("--ticker", help="Ticker for yfinance, e.g. SPY or BTC-USD")

    parser.add_argument("--period", default="5y", help="yfinance period, default: 5y")
    parser.add_argument("--interval", default="1d", help="yfinance interval, default: 1d")
    parser.add_argument("--output-dir", type=Path, default=Path("rsi2_adx5_vwap_output"))

    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--rsi-entry", type=float, default=10.0)
    parser.add_argument("--rsi-exit", type=float, default=60.0)
    parser.add_argument("--adx-entry", type=float, default=25.0)
    parser.add_argument("--trend-sma", type=int, default=200)
    parser.add_argument("--bb-period", type=int, default=20)
    parser.add_argument("--bb-std", type=float, default=2.0)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--exit-ema", type=int, default=5)

    parser.add_argument("--vwap-kind", choices=["auto", "session", "rolling"], default="auto")
    parser.add_argument("--vwap-window", type=int, default=20)
    parser.add_argument(
        "--vwap-timezone",
        default="UTC",
        help="Timezone used for session VWAP resets, e.g. America/New_York",
    )
    parser.add_argument(
        "--vwap-mode",
        choices=["discount", "reclaim", "either", "off"],
        default="either",
    )
    parser.add_argument("--vwap-max-premium", type=float, default=0.0)
    parser.add_argument("--confirmation-bars", type=int, default=3)
    parser.add_argument(
        "--require-positive-dmi",
        action="store_true",
        help="Also require +DI > -DI. This is stricter and can remove many pullback setups.",
    )
    parser.add_argument(
        "--allow-non-bullish-confirmation",
        action="store_true",
        help="Do not require confirmation close > open",
    )

    parser.add_argument("--risk-per-trade", type=float, default=0.005)
    parser.add_argument("--max-position-fraction", type=float, default=0.20)
    parser.add_argument("--atr-stop", type=float, default=2.0)
    parser.add_argument("--atr-target", type=float, default=1.5)
    parser.add_argument("--trailing-atr", type=float, default=0.0)
    parser.add_argument("--max-hold-bars", type=int, default=10)
    parser.add_argument("--min-hold-bars", type=int, default=1)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--minimum-notional", type=float, default=10.0)

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-rsi", default="5,10,15,20")
    parser.add_argument("--sweep-adx", default="20,25,30,35")
    parser.add_argument("--sweep-stops", default="1.5,2.0,2.5")
    parser.add_argument("--sweep-targets", default="0,1.0,1.5,2.0")
    parser.add_argument("--sweep-vwap-modes", default="discount,reclaim,either")
    parser.add_argument("--sweep-min-trades", type=int, default=20)
    return parser


def config_from_args(args: argparse.Namespace) -> StrategyConfig:
    return StrategyConfig(
        initial_capital=args.capital,
        rsi_entry=args.rsi_entry,
        rsi_exit=args.rsi_exit,
        adx_entry=args.adx_entry,
        trend_sma_period=args.trend_sma,
        bb_period=args.bb_period,
        bb_std=args.bb_std,
        atr_period=args.atr_period,
        exit_ema_period=args.exit_ema,
        vwap_kind=args.vwap_kind,
        vwap_window=args.vwap_window,
        vwap_timezone=args.vwap_timezone,
        vwap_mode=args.vwap_mode,
        vwap_max_premium=args.vwap_max_premium,
        confirmation_bars=args.confirmation_bars,
        require_bullish_confirmation=not args.allow_non_bullish_confirmation,
        require_positive_dmi=args.require_positive_dmi,
        risk_per_trade=args.risk_per_trade,
        max_position_fraction=args.max_position_fraction,
        atr_stop_mult=args.atr_stop,
        atr_target_mult=args.atr_target,
        trailing_atr_mult=args.trailing_atr,
        max_hold_bars=args.max_hold_bars,
        min_hold_bars=args.min_hold_bars,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
        minimum_notional=args.minimum_notional,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.csv:
            df = load_csv(args.csv)
            source_name = args.csv.stem
        else:
            df = load_yfinance(args.ticker, args.period, args.interval)
            source_name = str(args.ticker).replace("/", "_")

        cfg = config_from_args(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)

        if args.sweep:
            modes = [x.strip() for x in args.sweep_vwap_modes.split(",") if x.strip()]
            invalid_modes = set(modes) - {"discount", "reclaim", "either", "off"}
            if invalid_modes:
                raise ValueError(f"Invalid sweep VWAP modes: {sorted(invalid_modes)}")
            run_sweep(
                df=df,
                base_cfg=cfg,
                rsi_values=parse_number_list(args.sweep_rsi),
                adx_values=parse_number_list(args.sweep_adx),
                stop_values=parse_number_list(args.sweep_stops),
                target_values=parse_number_list(args.sweep_targets),
                vwap_modes=modes,
                min_trades=args.sweep_min_trades,
                output_dir=args.output_dir,
            )
            return 0

        metrics, trades, equity_curve, enriched = backtest(df, cfg)
        print_summary(metrics)

        prefix = source_name.replace(" ", "_")
        trades.to_csv(args.output_dir / f"{prefix}_trades.csv", index=False)
        equity_curve.to_csv(args.output_dir / f"{prefix}_equity.csv")
        enriched.to_csv(args.output_dir / f"{prefix}_indicators.csv")
        with (args.output_dir / f"{prefix}_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_safe(metrics), handle, indent=2)
        save_chart(equity_curve, args.output_dir / f"{prefix}_equity.png")

        print(f"\nFiles written to: {args.output_dir.resolve()}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
