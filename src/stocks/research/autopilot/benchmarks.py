from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stocks.research.autopilot.risk import cost_breakdown


def run_simple_benchmarks(
    close: pd.DataFrame,
    asset_returns: pd.DataFrame,
    eligibility: pd.DataFrame,
    *,
    cost_profile: str,
    periods_per_year: float,
) -> tuple[pd.Series, dict[str, Any]]:
    models: dict[str, pd.DataFrame] = {}
    eligible_float = eligibility.astype(float)
    models["equal_weight"] = eligible_float.div(
        eligible_float.sum(axis=1).replace(0.0, np.nan), axis=0
    ).fillna(0.0)

    volatility = asset_returns.rolling(63, min_periods=30).std()
    inverse = eligible_float.div(volatility.replace(0.0, np.nan))
    models["inverse_volatility"] = inverse.div(
        inverse.sum(axis=1).replace(0.0, np.nan), axis=0
    ).fillna(0.0)

    trend = eligibility & (
        close > close.rolling(200, min_periods=200).mean()
    )
    trend_float = trend.astype(float)
    models["trend_200d"] = trend_float.div(
        trend_float.sum(axis=1).replace(0.0, np.nan), axis=0
    ).fillna(0.0)

    momentum = close.pct_change(252).where(
        eligibility
        & (close > close.rolling(200, min_periods=200).mean())
    )
    month_end = _month_end_mask(close.index)
    rotation = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    current = pd.Series(0.0, index=close.columns)
    for timestamp in close.index:
        if bool(month_end.loc[timestamp]):
            leaders = momentum.loc[timestamp].dropna().nlargest(3).index
            current[:] = 0.0
            if len(leaders):
                current.loc[leaders] = 1.0 / len(leaders)
        rotation.loc[timestamp] = current
    models["momentum_rotation"] = rotation

    world_symbol = next(
        (symbol for symbol in ("ACWI", "SPY") if symbol in close.columns),
        None,
    )
    buy_hold = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if world_symbol is not None:
        buy_hold.loc[
            eligibility[world_symbol].fillna(False), world_symbol
        ] = 1.0
    models["world_buy_and_hold"] = buy_hold

    results: dict[str, dict[str, Any]] = {}
    series_by_name: dict[str, pd.Series] = {}
    for name, target in models.items():
        available = not (
            name == "world_buy_and_hold" and world_symbol is None
        )
        weights = target.shift(1).fillna(0.0).clip(lower=0.0)
        turnover = weights.diff().abs().sum(axis=1).fillna(
            weights.abs().sum(axis=1)
        )
        costs = sum(
            cost_breakdown(turnover, cost_profile).values(),
            start=pd.Series(0.0, index=turnover.index),
        )
        returns = (weights * asset_returns).sum(axis=1) - costs
        series_by_name[name] = returns
        results[name] = {
            **_benchmark_metrics(returns, periods_per_year),
            "available": available,
            "symbol": world_symbol if name == "world_buy_and_hold" else None,
        }
    champion = (
        "world_buy_and_hold"
        if world_symbol is not None
        else "equal_weight"
    )
    return series_by_name[champion], {
        "champion": champion,
        "selection_policy": "FIXED_WORLD_INDEX_ELSE_EQUAL_WEIGHT",
        "results": results,
    }


def _benchmark_metrics(
    returns: pd.Series,
    periods_per_year: float,
) -> dict[str, Any]:
    clean = returns.dropna()
    nav = (1.0 + clean).cumprod()
    standard_deviation = float(clean.std(ddof=1))
    drawdown = nav / nav.cummax() - 1.0
    return {
        "total_return": float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
        "Sharpe": (
            float(
                clean.mean()
                / standard_deviation
                * np.sqrt(periods_per_year)
            )
            if standard_deviation > 0
            else None
        ),
        "maximum_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def _month_end_mask(index: pd.DatetimeIndex) -> pd.Series:
    naive = index.tz_localize(None) if index.tz is not None else index
    period = naive.to_period("M")
    current = pd.Series(period, index=index)
    return current.ne(current.shift(-1))
