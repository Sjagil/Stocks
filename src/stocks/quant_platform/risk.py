from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm


class VolatilityModelEngine:
    def __init__(self, *, periods_per_year: int = 252):
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        self.periods_per_year = periods_per_year

    def estimate(
        self,
        bars: pd.DataFrame,
        *,
        window: int = 20,
        ewma_lambda: float = 0.94,
    ) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        missing = sorted(required - set(bars))
        if missing:
            raise ValueError(f"missing volatility columns: {', '.join(missing)}")
        if window < 2 or not 0 < ewma_lambda < 1:
            raise ValueError("invalid window or ewma_lambda")
        frame = bars.copy().sort_index()
        for column in required:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if (frame[list(required)] <= 0).any().any():
            raise ValueError("volatility estimators require positive OHLC")
        log_return = np.log(frame["close"] / frame["close"].shift(1))
        result = pd.DataFrame(index=frame.index)
        result["historical"] = log_return.rolling(window).std(ddof=1) * math.sqrt(self.periods_per_year)
        result["ewma"] = np.sqrt(log_return.pow(2).ewm(alpha=1 - ewma_lambda, adjust=False).mean() * self.periods_per_year)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - frame["close"].shift()).abs(),
                (frame["low"] - frame["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        result["atr"] = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / frame["close"] * math.sqrt(self.periods_per_year)
        high_low = np.log(frame["high"] / frame["low"])
        result["parkinson"] = np.sqrt(high_low.pow(2).rolling(window).mean() / (4 * math.log(2)) * self.periods_per_year)
        open_close = np.log(frame["close"] / frame["open"])
        gk_variance = 0.5 * high_low.pow(2) - (2 * math.log(2) - 1) * open_close.pow(2)
        result["garman_klass"] = np.sqrt(gk_variance.clip(lower=0).rolling(window).mean() * self.periods_per_year)
        overnight = np.log(frame["open"] / frame["close"].shift(1))
        intraday = np.log(frame["close"] / frame["open"])
        rs = (
            np.log(frame["high"] / frame["open"]) * np.log(frame["high"] / frame["close"])
            + np.log(frame["low"] / frame["open"]) * np.log(frame["low"] / frame["close"])
        )
        k = 0.34 / (1.34 + (window + 1) / max(window - 1, 1))
        yz_variance = (
            overnight.rolling(window).var(ddof=1)
            + k * intraday.rolling(window).var(ddof=1)
            + (1 - k) * rs.rolling(window).mean()
        )
        result["yang_zhang"] = np.sqrt(yz_variance.clip(lower=0) * self.periods_per_year)
        return result

    def forecast_arch(
        self,
        returns: pd.Series,
        *,
        model: str = "GARCH",
        horizons: Iterable[int] = (1, 5, 20),
    ) -> dict[int, float]:
        from arch import arch_model

        clean = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < 100:
            raise ValueError("at least 100 returns are required for ARCH-family forecasts")
        maximum = max(int(value) for value in horizons)
        if model == "GARCH":
            fitted = arch_model(clean * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        elif model == "EGARCH":
            fitted = arch_model(clean * 100, mean="Constant", vol="EGARCH", p=1, o=0, q=1, dist="t").fit(disp="off")
        elif model == "GJR-GARCH":
            fitted = arch_model(clean * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
        else:
            raise ValueError("model must be GARCH, EGARCH or GJR-GARCH")
        forecast_method = "simulation" if model == "EGARCH" and maximum > 1 else "analytic"
        variance = fitted.forecast(
            horizon=maximum,
            reindex=False,
            method=forecast_method,
            simulations=2_000,
        ).variance.iloc[-1].to_numpy(dtype=float) / 10_000
        return {int(horizon): float(np.sqrt(variance[: int(horizon)].sum())) for horizon in horizons}


@dataclass(frozen=True)
class PortfolioTailRiskEngine:
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if not 0.5 < self.confidence < 1:
            raise ValueError("confidence must be between 0.5 and 1")

    def univariate(self, returns: pd.Series, *, simulations: int = 50_000, seed: int = 42) -> dict[str, float]:
        values = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(values) < 30:
            raise ValueError("at least 30 returns are required")
        alpha = 1 - self.confidence
        historical = float(np.quantile(values, alpha))
        expected_shortfall = float(values[values <= historical].mean())
        mean, standard_deviation = float(values.mean()), float(values.std(ddof=1))
        z = float(norm.ppf(alpha))
        parametric = mean + z * standard_deviation
        skew = float(pd.Series(values).skew())
        excess_kurtosis = float(pd.Series(values).kurt())
        adjusted_z = z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * excess_kurtosis / 24 - (2 * z**3 - 5 * z) * skew**2 / 36
        rng = np.random.default_rng(seed)
        monte_carlo = mean + standard_deviation * rng.standard_t(df=5, size=simulations) / math.sqrt(5 / 3)
        mc_var = float(np.quantile(monte_carlo, alpha))
        ewma_volatility = pd.Series(values).ewm(alpha=0.06, adjust=False).std(bias=False).to_numpy()
        valid = np.isfinite(ewma_volatility) & (ewma_volatility > 0)
        standardized = (values[valid] - mean) / ewma_volatility[valid]
        filtered = mean + ewma_volatility[-1] * rng.choice(standardized, size=simulations, replace=True)
        filtered_var = float(np.quantile(filtered, alpha))
        return {
            "historical_var": historical,
            "historical_expected_shortfall": expected_shortfall,
            "parametric_var": parametric,
            "cornish_fisher_var": mean + adjusted_z * standard_deviation,
            "monte_carlo_var": mc_var,
            "monte_carlo_expected_shortfall": float(monte_carlo[monte_carlo <= mc_var].mean()),
            "filtered_historical_var": filtered_var,
            "filtered_historical_expected_shortfall": float(filtered[filtered <= filtered_var].mean()),
        }

    def portfolio(
        self,
        returns: pd.DataFrame,
        weights: Mapping[str, float],
        *,
        portfolio_value: float = 1.0,
        groups: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        assets = [str(column) for column in returns.columns]
        vector = np.asarray([float(weights.get(asset, 0.0)) for asset in assets])
        if not np.isclose(vector.sum(), 1.0):
            raise ValueError("portfolio weights must sum to one")
        covariance = returns.cov().to_numpy(dtype=float)
        sigma = math.sqrt(max(float(vector @ covariance @ vector), 0.0))
        z = abs(float(norm.ppf(1 - self.confidence)))
        var_fraction = z * sigma
        marginal = covariance @ vector / sigma if sigma > 0 else np.zeros_like(vector)
        component = vector * marginal * z * portfolio_value
        result: dict[str, Any] = {
            "portfolio_value": float(portfolio_value),
            "parametric_var": -var_fraction * portfolio_value,
            "risk_per_asset": dict(zip(assets, (-component).tolist(), strict=True)),
        }
        if groups:
            result["risk_by_group"] = {
                group_name: {
                    label: float(sum(-component[assets.index(asset)] * exposure for asset, exposure in mapping.items() if asset in assets))
                    for label, mapping in definitions.items()
                }
                for group_name, definitions in groups.items()
            }
        return result


class MonteCarloPortfolioSimulator:
    def simulate(
        self,
        returns: pd.Series,
        *,
        simulations: int = 10_000,
        horizon: int = 252,
        method: str = "block_bootstrap",
        block_size: int = 10,
        ruin_drawdown: float = 0.5,
        seed: int = 42,
        regimes: pd.Series | None = None,
    ) -> dict[str, Any]:
        history = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(history) < 30 or simulations <= 0 or horizon <= 0:
            raise ValueError("insufficient history or invalid simulation dimensions")
        rng = np.random.default_rng(seed)
        if method == "bootstrap":
            paths = rng.choice(history, size=(simulations, horizon), replace=True)
        elif method == "block_bootstrap":
            paths = _block_bootstrap(history, simulations, horizon, block_size, rng)
        elif method == "student_t":
            paths = history.mean() + history.std(ddof=1) * rng.standard_t(5, size=(simulations, horizon)) / math.sqrt(5 / 3)
        elif method == "garch":
            paths = _ewma_paths(history, simulations, horizon, rng)
        elif method == "regime_transitions":
            if regimes is None:
                raise ValueError("regimes are required for regime_transitions")
            regime_values = regimes.reindex(returns.dropna().index).dropna()
            regime_history = pd.to_numeric(returns, errors="coerce").reindex(regime_values.index).dropna()
            regime_values = regime_values.reindex(regime_history.index)
            paths = _regime_transition_paths(
                regime_history.to_numpy(dtype=float),
                regime_values.astype(str).to_numpy(),
                simulations,
                horizon,
                rng,
            )
        else:
            raise ValueError("unsupported Monte Carlo method")
        wealth = np.cumprod(1 + paths, axis=1)
        peaks = np.maximum.accumulate(np.column_stack([np.ones(simulations), wealth]), axis=1)[:, 1:]
        drawdowns = wealth / peaks - 1
        maximum_drawdown = drawdowns.min(axis=1)
        terminal = wealth[:, -1]
        cagr = terminal ** (252 / horizon) - 1
        recovery = _recovery_times(drawdowns)
        return {
            "method": method,
            "simulations": simulations,
            "horizon": horizon,
            "probability_10pct_drawdown": float((maximum_drawdown <= -0.10).mean()),
            "probability_20pct_drawdown": float((maximum_drawdown <= -0.20).mean()),
            "probability_of_ruin": float((maximum_drawdown <= -abs(ruin_drawdown)).mean()),
            "expected_cagr": float(cagr.mean()),
            "fifth_percentile_cagr": float(np.quantile(cagr, 0.05)),
            "worst_expected_year": float(np.quantile(terminal - 1, 0.01)),
            "median_recovery_time": float(np.median(recovery)),
            "research_only": True,
            "execution_authority": "NONE",
        }


def _block_bootstrap(history: np.ndarray, simulations: int, horizon: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 0 or block_size > len(history):
        raise ValueError("invalid block_size")
    blocks = math.ceil(horizon / block_size)
    starts = rng.integers(0, len(history) - block_size + 1, size=(simulations, blocks))
    result = np.empty((simulations, blocks * block_size))
    offsets = np.arange(block_size)
    for row in range(simulations):
        result[row] = history[(starts[row, :, None] + offsets).ravel()]
    return result[:, :horizon]


def _ewma_paths(history: np.ndarray, simulations: int, horizon: int, rng: np.random.Generator) -> np.ndarray:
    variance = float(pd.Series(history).ewm(alpha=0.06, adjust=False).var(bias=False).iloc[-1])
    shocks = rng.choice((history - history.mean()) / history.std(ddof=1), size=(simulations, horizon), replace=True)
    paths = np.empty_like(shocks)
    conditional = np.full(simulations, variance)
    for step in range(horizon):
        paths[:, step] = history.mean() + np.sqrt(conditional) * shocks[:, step]
        conditional = 0.94 * conditional + 0.06 * np.square(paths[:, step] - history.mean())
    return paths


def _recovery_times(drawdowns: np.ndarray) -> np.ndarray:
    recovered = drawdowns >= -1e-12
    result = np.full(drawdowns.shape[0], drawdowns.shape[1], dtype=float)
    for row in range(drawdowns.shape[0]):
        trough = int(np.argmin(drawdowns[row]))
        after = np.flatnonzero(recovered[row, trough:])
        if len(after):
            result[row] = float(after[0])
    return result


def _regime_transition_paths(
    returns: np.ndarray,
    regimes: np.ndarray,
    simulations: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = np.unique(regimes)
    if len(labels) < 2:
        raise ValueError("regime transition simulation requires at least two regimes")
    indices = {label: index for index, label in enumerate(labels)}
    transition = np.ones((len(labels), len(labels)), dtype=float)
    for left, right in zip(regimes[:-1], regimes[1:], strict=True):
        transition[indices[left], indices[right]] += 1
    transition /= transition.sum(axis=1, keepdims=True)
    samples = {label: returns[regimes == label] for label in labels}
    states = np.full(simulations, indices[regimes[-1]], dtype=int)
    paths = np.empty((simulations, horizon))
    for step in range(horizon):
        for row in range(simulations):
            states[row] = rng.choice(len(labels), p=transition[states[row]])
            paths[row, step] = rng.choice(samples[labels[states[row]]])
    return paths
