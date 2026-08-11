from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def technical_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Vectorized trend, momentum, reversion, volatility and volume features."""

    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(missing)}")
    frame = bars.copy().sort_index()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("high violates OHLC constraints")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("low violates OHLC constraints")
    close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
    result = frame.copy()

    result["sma_20"] = close.rolling(20).mean()
    result["sma_50"] = close.rolling(50).mean()
    result["ema_12"] = close.ewm(span=12, adjust=False).mean()
    result["ema_26"] = close.ewm(span=26, adjust=False).mean()
    result["ema_50"] = close.ewm(span=50, adjust=False).mean()
    result["ema_200"] = close.ewm(span=200, adjust=False).mean()
    result["donchian_high_20"] = high.rolling(20).max()
    result["donchian_low_20"] = low.rolling(20).min()
    result["atr_14"] = _true_range(frame).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    plus_di, minus_di, adx = _adx(frame, period=14)
    result["plus_di_14"] = plus_di
    result["minus_di_14"] = minus_di
    result["adx_14"] = adx
    result["supertrend"] = _supertrend(frame, result["atr_14"], multiplier=3.0)
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]
    result["ichimoku_conversion"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    result["ichimoku_base"] = (high.rolling(26).max() + low.rolling(26).min()) / 2
    result["ichimoku_span_a"] = ((result["ichimoku_conversion"] + result["ichimoku_base"]) / 2).shift(26)
    result["ichimoku_span_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)

    result["rsi_14"] = _rsi(close, 14)
    lowest_14, highest_14 = low.rolling(14).min(), high.rolling(14).max()
    spread_14 = (highest_14 - lowest_14).replace(0.0, np.nan)
    result["stochastic_k_14"] = 100 * (close - lowest_14) / spread_14
    result["stochastic_d_3"] = result["stochastic_k_14"].rolling(3).mean()
    result["roc_12"] = close.pct_change(12, fill_method=None) * 100
    typical = (high + low + close) / 3
    typical_mean = typical.rolling(20).mean()
    mean_deviation = typical.rolling(20).apply(lambda values: np.mean(np.abs(values - values.mean())), raw=True)
    result["cci_20"] = (typical - typical_mean) / (0.015 * mean_deviation.replace(0.0, np.nan))
    result["williams_r_14"] = -100 * (highest_14 - close) / spread_14

    standard_deviation = close.rolling(20).std(ddof=0)
    result["bollinger_middle"] = result["sma_20"]
    result["bollinger_upper"] = result["sma_20"] + 2 * standard_deviation
    result["bollinger_lower"] = result["sma_20"] - 2 * standard_deviation
    result["price_zscore_20"] = (close - result["sma_20"]) / standard_deviation.replace(0.0, np.nan)
    cumulative_volume = volume.cumsum().replace(0.0, np.nan)
    result["vwap"] = (typical * volume).cumsum() / cumulative_volume
    result["vwap_deviation"] = close / result["vwap"] - 1.0
    result["bollinger_bandwidth"] = (result["bollinger_upper"] - result["bollinger_lower"]) / result["bollinger_middle"]
    result["keltner_middle"] = close.ewm(span=20, adjust=False).mean()
    result["keltner_upper"] = result["keltner_middle"] + 2 * result["atr_14"]
    result["keltner_lower"] = result["keltner_middle"] - 2 * result["atr_14"]
    result["bollinger_squeeze"] = (result["bollinger_upper"] < result["keltner_upper"]) & (
        result["bollinger_lower"] > result["keltner_lower"]
    )
    result["atr_expansion"] = result["atr_14"] > result["atr_14"].rolling(20).mean()

    direction = np.sign(close.diff()).fillna(0.0)
    result["obv"] = (direction * volume.fillna(0.0)).cumsum()
    result["vpt"] = (close.pct_change(fill_method=None).fillna(0.0) * volume.fillna(0.0)).cumsum()
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0.0, np.nan)
    money_flow_volume = money_flow_multiplier.fillna(0.0) * volume.fillna(0.0)
    result["cmf_20"] = money_flow_volume.rolling(20).sum() / volume.rolling(20).sum().replace(0.0, np.nan)
    accumulation_distribution = money_flow_volume.cumsum()
    result["chaikin_oscillator"] = accumulation_distribution.ewm(span=3, adjust=False).mean() - accumulation_distribution.ewm(span=10, adjust=False).mean()
    result["volume_average_20"] = volume.rolling(20).mean()
    result["volume_breakout"] = volume > result["volume_average_20"] * 1.5
    return result


def strategy_signals(features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "close",
        "ema_12",
        "ema_26",
        "ema_50",
        "ema_200",
        "donchian_high_20",
        "rsi_14",
        "adx_14",
        "volume_breakout",
        "price_zscore_20",
        "atr_expansion",
        "bollinger_squeeze",
        "macd_histogram",
        "cmf_20",
    }
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(f"missing technical features: {', '.join(missing)}")
    result = pd.DataFrame(index=features.index)
    result["trend"] = (
        (features["ema_12"] > features["ema_26"])
        & (features["ema_50"] > features["ema_200"])
        & (features["adx_14"] > 25)
    )
    result["momentum"] = (features["rsi_14"] > 50) & (features["macd_histogram"] > 0)
    result["mean_reversion"] = (features["price_zscore_20"] < -2) & (features["rsi_14"] < 30)
    result["breakout"] = (
        (features["close"] > features["donchian_high_20"].shift(1))
        & features["volume_breakout"]
        & (features["adx_14"] > 20)
    )
    result["volatility_expansion"] = (
        features["bollinger_squeeze"].shift(1, fill_value=False)
        & features["atr_expansion"]
    )
    result["volume_confirmation"] = features["volume_breakout"] & (features["cmf_20"] > 0)
    result["combined_trend_pullback"] = (
        (features["ema_12"] > features["ema_26"])
        & features["rsi_14"].between(35, 60)
        & (features["adx_14"] > 25)
        & result["volume_confirmation"]
    )
    return result.fillna(False).astype(bool)


class TechnicalStrategyLab:
    def evaluate(self, bars: pd.DataFrame) -> dict[str, Any]:
        features = technical_features(bars)
        signals = strategy_signals(features)
        return {
            "features": features,
            "signals": signals,
            "signal_counts": {column: int(signals[column].sum()) for column in signals},
            "research_only": True,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    return rsi.mask((loss == 0) & (gain > 0), 100.0).mask((loss == 0) & (gain == 0), 50.0)


def _adx(frame: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    upward = frame["high"].diff()
    downward = -frame["low"].diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    atr = _true_range(frame).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return plus_di, minus_di, dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _supertrend(frame: pd.DataFrame, atr: pd.Series, *, multiplier: float) -> pd.Series:
    midpoint = (frame["high"] + frame["low"]) / 2
    upper = midpoint + multiplier * atr
    lower = midpoint - multiplier * atr
    values = pd.Series(np.nan, index=frame.index, dtype=float)
    direction = 1
    for position in range(len(frame)):
        if pd.isna(atr.iloc[position]):
            continue
        if position and frame["close"].iloc[position] > upper.iloc[position - 1]:
            direction = 1
        elif position and frame["close"].iloc[position] < lower.iloc[position - 1]:
            direction = -1
        values.iloc[position] = lower.iloc[position] if direction > 0 else upper.iloc[position]
    return values
