from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks.quant_platform import (
    BlackLittermanAllocator,
    DynamicMultiAssetAllocator,
    HierarchicalRiskParity,
    OptimalExecutionEngine,
    PortfolioExposureRiskEngine,
    TransactionCostModel,
    execution_shortfall,
)


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(40)
    factors = rng.normal(size=(1_000, 3))
    return pd.DataFrame(
        {
            "SPY": 0.0004 + 0.01 * factors[:, 0],
            "GLD": 0.0002 - 0.002 * factors[:, 0] + 0.008 * factors[:, 1],
            "BTC": 0.0008 + 0.005 * factors[:, 0] + 0.03 * factors[:, 2],
            "BONDS": 0.0001 - 0.003 * factors[:, 0] + 0.004 * factors[:, 1],
        }
    )


def test_hrp_is_long_only_fully_invested_and_clustered() -> None:
    result = HierarchicalRiskParity().allocate(_returns())
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert all(weight >= 0 for weight in result["weights"].values())
    assert set(result["cluster_order"]) == set(_returns().columns)


def test_black_litterman_updates_market_equilibrium_with_confident_view() -> None:
    covariance = _returns().cov() * 252
    market = {"SPY": 0.5, "GLD": 0.2, "BTC": 0.1, "BONDS": 0.2}
    baseline = BlackLittermanAllocator().allocate(covariance, market)
    viewed = BlackLittermanAllocator().allocate(
        covariance,
        market,
        views=pd.DataFrame([{"asset": "BTC", "relative_to": "SPY", "expected_return": 0.10, "confidence": 0.8}]),
    )
    assert sum(viewed["weights"].values()) == pytest.approx(1.0)
    assert viewed["posterior_expected_returns"]["BTC"] - viewed["posterior_expected_returns"]["SPY"] > baseline["posterior_expected_returns"]["BTC"] - baseline["posterior_expected_returns"]["SPY"]


def test_dynamic_allocator_respects_risk_budget_and_keeps_cash() -> None:
    assets = ["SPY", "GLD", "BTC", "BONDS"]
    features = pd.DataFrame(
        {
            "expected_return": [0.08, 0.05, 0.15, 0.03],
            "volatility": [0.18, 0.14, 0.60, 0.08],
            "momentum": [0.1, 0.05, 0.2, 0.01],
            "macro": [0.1, 0.2, -0.1, 0.1],
            "liquidity": [1.0, 0.8, 0.6, 0.9],
            "drawdown": [-0.1, -0.05, -0.3, -0.02],
        },
        index=assets,
    )
    result = DynamicMultiAssetAllocator().allocate(features, _returns().cov() * 252, risk_budget=0.08)
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert result["weights"]["CASH"] >= 0
    assert result["expected_portfolio_volatility"] <= 0.08 + 1e-12


def test_portfolio_risk_engine_rejects_excess_technology_exposure() -> None:
    metadata = pd.DataFrame(
        {
            "sector": ["TECH", "METALS", "CRYPTO", "BONDS"],
            "country": ["US", "US", "GLOBAL", "US"],
            "currency": ["USD", "USD", "USD", "USD"],
            "average_daily_volume": [1e8, 1e7, 5e7, 2e7],
        },
        index=["SPY", "GLD", "BTC", "BONDS"],
    )
    factors = pd.DataFrame({"market": [1.0, 0.1, 1.5, -0.2]}, index=metadata.index)
    report = PortfolioExposureRiskEngine().analyze(
        {"SPY": 0.5, "GLD": 0.2, "BTC": 0.1, "BONDS": 0.2},
        _returns(),
        metadata,
        factor_loadings=factors,
        limits={"sector": {"TECH": 0.35}},
    )
    assert not report["risk_approved"]
    assert report["limit_breaches"][0]["label"] == "TECH"


def test_transaction_cost_grows_with_size_volatility_and_illiquidity() -> None:
    model = TransactionCostModel()
    small = model.estimate(price=100, quantity=100, average_daily_volume=1_000_000, volatility=0.01)
    large = model.estimate(price=100, quantity=10_000, average_daily_volume=100_000, volatility=0.04)
    assert large["expected_cost_bps"] > small["expected_cost_bps"]
    assert model.economically_executable(0.02, large, safety_margin=0.001)


@pytest.mark.parametrize("method", ["TWAP", "VWAP", "POV", "IMPLEMENTATION_SHORTFALL", "ALMGREN_CHRISS"])
def test_execution_schedules_exact_integer_quantity(method: str) -> None:
    result = OptimalExecutionEngine().schedule(
        101,
        5,
        method=method,
        forecast_volumes=[100, 200, 400, 200, 100] if method in {"VWAP", "POV"} else None,
    )
    assert result["scheduled_quantity"] == 101
    assert sum(item["quantity"] for item in result["schedule"]) == 101
    assert result["broker_writes"] == 0


def test_execution_shortfall_is_side_aware() -> None:
    fills = pd.DataFrame({"price": [100, 102], "quantity": [10, 30]})
    buy = execution_shortfall(fills, decision_price=100, side="BUY")
    sell = execution_shortfall(fills, decision_price=100, side="SELL")
    assert buy["implementation_shortfall"] > 0
    assert sell["implementation_shortfall"] == pytest.approx(-buy["implementation_shortfall"])
