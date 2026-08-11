from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import pstdev


def simple_moving_average(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values are required")
    return sum(values) / len(values)


def exponential_moving_average(values: Sequence[float], period: int) -> float:
    if not values:
        raise ValueError("values are required")
    if period <= 0:
        raise ValueError("period must be positive")
    alpha = 2.0 / (period + 1.0)
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1.0 - alpha) * ema
    return ema


def momentum(current_price: float, lagged_price: float) -> float:
    if lagged_price <= 0:
        raise ValueError("lagged_price must be positive")
    return current_price / lagged_price - 1.0


def risk_adjusted_momentum(raw_momentum: float, volatility: float) -> float:
    if volatility <= 0:
        raise ValueError("volatility must be positive")
    return raw_momentum / volatility


def trend_strength(ema_fast: float, ema_slow: float, atr: float) -> float:
    if atr <= 0:
        raise ValueError("atr must be positive")
    return (ema_fast - ema_slow) / atr


def relative_strength(asset_return: float, benchmark_return: float) -> float:
    return asset_return - benchmark_return


def effective_volatility(
    equity_volatility: float,
    fx_volatility: float,
    equity_fx_correlation: float,
) -> float:
    if equity_volatility < 0 or fx_volatility < 0:
        raise ValueError("volatilities cannot be negative")
    if not -1.0 <= equity_fx_correlation <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    variance = (
        equity_volatility**2
        + fx_volatility**2
        + 2.0 * equity_fx_correlation * equity_volatility * fx_volatility
    )
    return math.sqrt(max(variance, 0.0))


def gap_return(open_price: float, previous_close: float) -> float:
    if previous_close <= 0:
        raise ValueError("previous_close must be positive")
    return open_price / previous_close - 1.0


def true_range(high: float, low: float, previous_close: float) -> float:
    if high < low:
        raise ValueError("high cannot be below low")
    if previous_close <= 0:
        raise ValueError("previous_close must be positive")
    return max(
        high - low,
        abs(high - previous_close),
        abs(low - previous_close),
    )


def average_true_range(true_ranges: Sequence[float], period: int) -> float:
    return exponential_moving_average(true_ranges, period=period)


def bollinger_z(price: float, values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values are required")
    sigma = pstdev(values)
    if sigma == 0:
        return 0.0
    return (price - simple_moving_average(values)) / sigma


def bollinger_bands(values: Sequence[float], k: float) -> tuple[float, float, float]:
    if not values:
        raise ValueError("values are required")
    if k < 0:
        raise ValueError("k cannot be negative")
    middle = simple_moving_average(values)
    sigma = pstdev(values)
    return middle, middle + k * sigma, middle - k * sigma


def bollinger_bandwidth(upper_band: float, lower_band: float, middle_band: float) -> float:
    if middle_band == 0:
        raise ValueError("middle_band cannot be zero")
    return (upper_band - lower_band) / middle_band


def percent_b(price: float, lower_band: float, upper_band: float) -> float:
    width = upper_band - lower_band
    if width <= 0:
        raise ValueError("upper_band must exceed lower_band")
    return (price - lower_band) / width


def volume_z(volume: float, historical_volumes: Sequence[float]) -> float:
    if not historical_volumes:
        raise ValueError("historical_volumes are required")
    sigma = pstdev(historical_volumes)
    if sigma == 0:
        return 0.0
    return (volume - simple_moving_average(historical_volumes)) / sigma
