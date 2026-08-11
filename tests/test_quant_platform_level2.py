from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks.quant_platform import PortfolioOptimizer


def _returns() -> pd.DataFrame:
    random = np.random.default_rng(42)
    factors = random.normal(size=(1_000, 3))
    return pd.DataFrame(
        {
            "SPY": 0.0004 + 0.010 * factors[:, 0],
            "GLD": 0.00025 + 0.004 * factors[:, 0] + 0.007 * factors[:, 1],
            "BTC": 0.0010 + 0.006 * factors[:, 0] + 0.030 * factors[:, 2],
            "BONDS": 0.00015 - 0.002 * factors[:, 0] + 0.003 * factors[:, 1],
        }
    )


def test_optimizer_exposes_covariance_and_correlation() -> None:
    optimizer = PortfolioOptimizer(_returns())
    assert optimizer.covariance.shape == (4, 4)
    assert optimizer.correlation.shape == (4, 4)
    assert np.allclose(optimizer.covariance, optimizer.covariance.T)


@pytest.mark.parametrize(
    "method",
    ["equal_weight", "minimum_variance", "maximum_sharpe", "risk_parity"],
)
def test_portfolio_solutions_are_long_only_fully_invested(method: str) -> None:
    solution = getattr(PortfolioOptimizer(_returns()), method)()
    assert solution.converged
    assert sum(solution.weights.values()) == pytest.approx(1.0)
    assert all(0 <= weight <= 1 for weight in solution.weights.values())
    assert solution.volatility > 0
    assert sum(solution.risk_contributions.values()) == pytest.approx(1.0)


def test_minimum_variance_improves_on_equal_weight() -> None:
    optimizer = PortfolioOptimizer(_returns())
    assert optimizer.minimum_variance().volatility <= optimizer.equal_weight().volatility


def test_risk_parity_equalizes_fractional_risk_contributions() -> None:
    solution = PortfolioOptimizer(_returns()).risk_parity()
    contributions = np.asarray(list(solution.risk_contributions.values()))
    assert contributions.max() - contributions.min() < 1e-4


def test_maximum_sharpe_improves_on_equal_weight() -> None:
    optimizer = PortfolioOptimizer(_returns())
    assert optimizer.maximum_sharpe().sharpe_ratio >= optimizer.equal_weight().sharpe_ratio


def test_efficient_frontier_honors_feasible_targets() -> None:
    optimizer = PortfolioOptimizer(_returns())
    expected = optimizer.expected_returns
    targets = np.linspace(float(expected.min()), float(expected.max()), 5)[1:-1]
    frontier = optimizer.efficient_frontier(targets)
    assert len(frontier) == 3
    for solution, target in zip(frontier, targets, strict=True):
        assert solution.expected_return == pytest.approx(target, abs=1e-7)


def test_comparison_is_research_only_and_contains_all_methods() -> None:
    report = PortfolioOptimizer(_returns()).compare()
    assert {item["method"] for item in report["solutions"]} == {
        "EQUAL_WEIGHT",
        "MINIMUM_VARIANCE",
        "MAXIMUM_SHARPE",
        "RISK_PARITY",
    }
    assert report["execution_authority"] == "NONE"
    assert report["broker_writes"] == 0


def test_optimizer_rejects_incomplete_expected_returns() -> None:
    with pytest.raises(ValueError, match="cover every asset"):
        PortfolioOptimizer(_returns(), expected_returns=pd.Series({"SPY": 0.1}))
