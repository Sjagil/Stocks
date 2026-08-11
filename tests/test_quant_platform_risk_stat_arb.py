from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks.quant_platform import (
    MonteCarloPortfolioSimulator,
    PortfolioTailRiskEngine,
    StatisticalArbitrageEngine,
    VolatilityModelEngine,
)


def _ohlc(rows: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, rows)))
    open_ = close * np.exp(rng.normal(0, 0.003, rows))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, rows))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, rows))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_volatility_engine_covers_range_close_and_ohlc_estimators() -> None:
    result = VolatilityModelEngine().estimate(_ohlc())
    assert {"historical", "ewma", "atr", "parkinson", "garman_klass", "yang_zhang"} == set(result)
    assert (result.dropna() >= 0).all().all()


@pytest.mark.parametrize("model", ["GARCH", "EGARCH", "GJR-GARCH"])
def test_arch_family_forecasts_have_requested_horizons(model: str) -> None:
    returns = np.log(_ohlc(450)["close"]).diff().dropna()
    forecast = VolatilityModelEngine().forecast_arch(returns, model=model, horizons=(1, 5, 20))
    assert set(forecast) == {1, 5, 20}
    assert forecast[1] > 0
    assert forecast[20] >= forecast[5] >= forecast[1]


def test_var_expected_shortfall_methods_preserve_tail_order() -> None:
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.standard_t(5, 2_000) * 0.01)
    risk = PortfolioTailRiskEngine(confidence=0.99).univariate(returns, simulations=10_000)
    assert risk["historical_expected_shortfall"] <= risk["historical_var"] < 0
    assert risk["monte_carlo_expected_shortfall"] <= risk["monte_carlo_var"] < 0
    assert risk["filtered_historical_expected_shortfall"] <= risk["filtered_historical_var"] < 0


def test_portfolio_var_decomposes_to_asset_risk() -> None:
    rng = np.random.default_rng(5)
    returns = pd.DataFrame(rng.normal(0, [0.01, 0.02, 0.007], size=(1_000, 3)), columns=["SPY", "BTC", "GLD"])
    report = PortfolioTailRiskEngine().portfolio(returns, {"SPY": 0.5, "BTC": 0.1, "GLD": 0.4}, portfolio_value=100_000)
    assert report["parametric_var"] < 0
    assert sum(report["risk_per_asset"].values()) == pytest.approx(report["parametric_var"])


@pytest.mark.parametrize("method", ["bootstrap", "block_bootstrap", "student_t", "garch", "regime_transitions"])
def test_monte_carlo_supports_non_normal_and_conditional_paths(method: str) -> None:
    rng = np.random.default_rng(6)
    history = pd.Series(rng.standard_t(6, 1_000) * 0.01 + 0.0002)
    report = MonteCarloPortfolioSimulator().simulate(
        history,
        simulations=1_000,
        horizon=100,
        method=method,
        regimes=pd.Series(np.where(np.arange(len(history)) % 20 < 10, "CALM", "STRESS")),
    )
    assert 0 <= report["probability_10pct_drawdown"] <= 1
    assert 0 <= report["probability_of_ruin"] <= 1
    assert report["execution_authority"] == "NONE"


def test_cointegrated_pair_reports_tests_half_life_and_zscore() -> None:
    rng = np.random.default_rng(8)
    random_walk = 100 + np.cumsum(rng.normal(0, 1, 800))
    stationary = np.zeros(800)
    for index in range(1, len(stationary)):
        stationary[index] = 0.8 * stationary[index - 1] + rng.normal(0, 0.5)
    left = pd.Series(2 * random_walk + stationary + 20)
    right = pd.Series(random_walk)
    report = StatisticalArbitrageEngine().analyze_pair(left, right)
    assert report["hedge_ratio"] == pytest.approx(2.0, rel=0.02)
    assert report["adf_pvalue"] < 0.05
    assert report["half_life"] > 0
    assert report["execution_authority"] == "NONE"


def test_pca_residuals_remove_common_components() -> None:
    rng = np.random.default_rng(10)
    common = rng.normal(size=500)
    returns = pd.DataFrame({f"A{i}": common * (i + 1) + rng.normal(scale=0.1, size=500) for i in range(5)})
    residuals = StatisticalArbitrageEngine().pca_residuals(returns, components=1)
    assert residuals.shape == returns.shape
    assert residuals.var().sum() < ((returns - returns.mean()) / returns.std(ddof=0)).var().sum()


def test_cross_sectional_mean_reversion_is_market_and_sector_neutral() -> None:
    returns = pd.Series({"A": -0.03, "B": 0.02, "C": -0.01, "D": 0.04})
    sectors = pd.Series({"A": "TECH", "B": "TECH", "C": "HEALTH", "D": "HEALTH"})
    result = StatisticalArbitrageEngine().cross_sectional_mean_reversion(returns, sectors)
    assert result["weight"].sum() == pytest.approx(0.0)
    assert result.groupby("sector")["weight"].sum().abs().max() < 1e-12
    assert result["weight"].abs().sum() == pytest.approx(1.0)
