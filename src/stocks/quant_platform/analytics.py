from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from stocks.quant_platform.data import validate_market_data


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _drawdown(wealth: pd.Series) -> pd.Series:
    # Initial capital is 1.0, so a loss in the first measured period is a real
    # drawdown rather than an artificial new high.
    return wealth / wealth.cummax().clip(lower=1.0) - 1.0


@dataclass(frozen=True)
class PerformanceRiskAnalyzer:
    periods_per_year: int = 252
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0.5 and 1")

    def returns(self, prices: pd.Series, *, logarithmic: bool = False) -> pd.Series:
        series = self._prices(prices)
        result = np.log(series / series.shift(1)) if logarithmic else series.pct_change(fill_method=None)
        return result.replace([np.inf, -np.inf], np.nan).dropna()

    def rolling_statistics(self, prices: pd.Series, *, window: int = 20) -> pd.DataFrame:
        if window < 2:
            raise ValueError("window must be at least 2")
        returns = self.returns(prices)
        wealth = (1.0 + returns).cumprod()
        result = pd.DataFrame(index=returns.index)
        result["return"] = returns
        result["log_return"] = self.returns(prices, logarithmic=True).reindex(result.index)
        result["rolling_mean"] = returns.rolling(window).mean()
        result["rolling_volatility"] = returns.rolling(window).std(ddof=1) * math.sqrt(self.periods_per_year)
        result["rolling_drawdown"] = _drawdown(wealth)
        return result

    def analyze(
        self,
        prices: pd.Series,
        *,
        benchmark_prices: pd.Series | None = None,
        annual_risk_free_rate: float = 0.0,
    ) -> dict[str, Any]:
        series = self._prices(prices)
        returns = self.returns(series)
        if returns.empty:
            raise ValueError("at least two valid prices are required")
        n = len(returns)
        total_return = float(series.iloc[-1] / series.iloc[0] - 1.0)
        cagr = float((series.iloc[-1] / series.iloc[0]) ** (self.periods_per_year / n) - 1.0)
        volatility = float(returns.std(ddof=1) * math.sqrt(self.periods_per_year)) if n > 1 else 0.0
        period_rf = (1.0 + annual_risk_free_rate) ** (1.0 / self.periods_per_year) - 1.0
        excess = returns - period_rf
        sharpe = excess.mean() / excess.std(ddof=1) * math.sqrt(self.periods_per_year) if excess.std(ddof=1) > 0 else np.nan
        downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(self.periods_per_year))
        sortino = (float(excess.mean()) * self.periods_per_year / downside_deviation) if downside_deviation > 0 else np.nan
        wealth = (1.0 + returns).cumprod()
        drawdown = _drawdown(wealth)
        max_drawdown = float(drawdown.min())
        calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
        var = float(returns.quantile(1.0 - self.confidence_level))
        tail = returns[returns <= var]
        cvar = float(tail.mean()) if not tail.empty else var
        monthly = series.resample("ME").last().pct_change(fill_method=None).dropna() if isinstance(series.index, pd.DatetimeIndex) else pd.Series(dtype=float)
        benchmark = self._benchmark_statistics(returns, benchmark_prices, annual_risk_free_rate)
        return {
            "observations": n,
            "periods_per_year": self.periods_per_year,
            "confidence_level": self.confidence_level,
            "total_return": total_return,
            "cagr": cagr,
            "annualized_volatility": volatility,
            "sharpe_ratio": _finite_or_none(sharpe),
            "sortino_ratio": _finite_or_none(sortino),
            "calmar_ratio": _finite_or_none(calmar),
            "maximum_drawdown": max_drawdown,
            "historical_var": var,
            "historical_cvar": cvar,
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "best_month": _finite_or_none(monthly.max()) if not monthly.empty else None,
            "worst_month": _finite_or_none(monthly.min()) if not monthly.empty else None,
            "skewness": _finite_or_none(returns.skew()),
            "kurtosis": _finite_or_none(returns.kurt()),
            **benchmark,
        }

    def analyze_market_data(
        self,
        frame: pd.DataFrame,
        symbol: str,
        *,
        benchmark_symbol: str | None = None,
        annual_risk_free_rate: float = 0.0,
    ) -> dict[str, Any]:
        canonical = validate_market_data(frame)
        symbol = symbol.strip().upper()
        prices = self._symbol_prices(canonical, symbol)
        benchmark = self._symbol_prices(canonical, benchmark_symbol.strip().upper()) if benchmark_symbol else None
        result = self.analyze(
            prices,
            benchmark_prices=benchmark,
            annual_risk_free_rate=annual_risk_free_rate,
        )
        return {"symbol": symbol, "benchmark_symbol": benchmark_symbol, **result}

    @staticmethod
    def _prices(prices: pd.Series) -> pd.Series:
        series = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
        if not series.index.is_monotonic_increasing:
            series = series.sort_index()
        series = series[~series.index.duplicated(keep="last")]
        if (series <= 0).any():
            raise ValueError("prices must be positive")
        return series

    @staticmethod
    def _symbol_prices(frame: pd.DataFrame, symbol: str) -> pd.Series:
        rows = frame.loc[frame["symbol"] == symbol].sort_values(["timestamp", "available_at"])
        rows = rows.drop_duplicates("timestamp", keep="last")
        if rows.empty:
            raise ValueError(f"no observations for {symbol}")
        return pd.Series(rows["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(rows["timestamp"]), name=symbol)

    def _benchmark_statistics(
        self,
        returns: pd.Series,
        benchmark_prices: pd.Series | None,
        annual_risk_free_rate: float,
    ) -> dict[str, float | None]:
        empty = {"beta": None, "alpha": None, "correlation": None}
        if benchmark_prices is None:
            return empty
        benchmark_returns = self.returns(benchmark_prices)
        aligned = pd.concat([returns.rename("asset"), benchmark_returns.rename("benchmark")], axis=1, join="inner").dropna()
        if len(aligned) < 2:
            return empty
        variance = float(aligned["benchmark"].var(ddof=1))
        if variance <= 0:
            return empty
        beta = float(aligned.cov().loc["asset", "benchmark"] / variance)
        annual_asset = float(aligned["asset"].mean() * self.periods_per_year)
        annual_benchmark = float(aligned["benchmark"].mean() * self.periods_per_year)
        alpha = annual_asset - (annual_risk_free_rate + beta * (annual_benchmark - annual_risk_free_rate))
        return {
            "beta": beta,
            "alpha": alpha,
            "correlation": _finite_or_none(aligned["asset"].corr(aligned["benchmark"])),
        }
