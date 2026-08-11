from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class PortfolioSolution:
    method: str
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float | None
    risk_contributions: dict[str, float]
    converged: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "weights": self.weights,
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_contributions": self.risk_contributions,
            "converged": self.converged,
        }


class PortfolioOptimizer:
    """Long-only Markowitz and equal-risk-contribution optimizer."""

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        periods_per_year: int = 252,
        expected_returns: pd.Series | None = None,
        covariance_shrinkage: float = 1e-6,
    ):
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        frame = returns.apply(pd.to_numeric, errors="coerce").dropna(how="all")
        frame = frame.dropna(axis=1, how="all").dropna()
        if len(frame) < 2 or frame.shape[1] < 2:
            raise ValueError("at least two assets and two complete return rows are required")
        if frame.columns.duplicated().any():
            raise ValueError("asset names must be unique")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("returns must be finite")
        if covariance_shrinkage < 0:
            raise ValueError("covariance_shrinkage must be non-negative")
        self.returns = frame.astype(float)
        self.assets = [str(column) for column in frame.columns]
        self.periods_per_year = periods_per_year
        self.expected_returns = (
            expected_returns.reindex(frame.columns).astype(float)
            if expected_returns is not None
            else frame.mean() * periods_per_year
        )
        if self.expected_returns.isna().any() or not np.isfinite(self.expected_returns).all():
            raise ValueError("expected_returns must cover every asset with finite values")
        raw_covariance = frame.cov().to_numpy(dtype=float) * periods_per_year
        scale = float(np.trace(raw_covariance) / len(self.assets))
        ridge = covariance_shrinkage * max(scale, np.finfo(float).eps)
        self.covariance = pd.DataFrame(
            raw_covariance + np.eye(len(self.assets)) * ridge,
            index=frame.columns,
            columns=frame.columns,
        )
        self.correlation = frame.corr()

    def equal_weight(self, *, annual_risk_free_rate: float = 0.0) -> PortfolioSolution:
        return self._solution("EQUAL_WEIGHT", np.full(len(self.assets), 1.0 / len(self.assets)), True, annual_risk_free_rate)

    def minimum_variance(self, *, annual_risk_free_rate: float = 0.0) -> PortfolioSolution:
        result = self._optimize(lambda weights: self._variance(weights))
        return self._solution("MINIMUM_VARIANCE", result.x, result.success, annual_risk_free_rate)

    def maximum_sharpe(self, *, annual_risk_free_rate: float = 0.0) -> PortfolioSolution:
        def objective(weights: np.ndarray) -> float:
            volatility = self._volatility(weights)
            if volatility <= 0:
                return 1e9
            return -(self._expected_return(weights) - annual_risk_free_rate) / volatility

        result = self._optimize(objective)
        return self._solution("MAXIMUM_SHARPE", result.x, result.success, annual_risk_free_rate)

    def risk_parity(self, *, annual_risk_free_rate: float = 0.0) -> PortfolioSolution:
        covariance = self.covariance.to_numpy(dtype=float)
        budgets = np.full(len(self.assets), 1.0 / len(self.assets))

        # Convex risk-budgeting formulation.  At its optimum
        # x_i(Σx)_i is proportional to the requested risk budget; normalizing x
        # afterwards does not change fractional risk contributions.
        def objective(log_weights: np.ndarray) -> float:
            weights = np.exp(log_weights)
            return float(0.5 * weights @ covariance @ weights - budgets @ log_weights)

        def gradient(log_weights: np.ndarray) -> np.ndarray:
            weights = np.exp(log_weights)
            return weights * (covariance @ weights) - budgets

        initial = -0.5 * np.log(
            np.maximum(np.diag(covariance), np.finfo(float).eps) * len(self.assets)
        )
        result = minimize(
            objective,
            initial,
            jac=gradient,
            method="BFGS",
            options={"gtol": 1e-9, "maxiter": 10_000},
        )
        return self._solution(
            "RISK_PARITY",
            np.exp(result.x),
            result.success,
            annual_risk_free_rate,
        )

    def efficient_frontier(
        self,
        target_returns: Iterable[float],
        *,
        annual_risk_free_rate: float = 0.0,
    ) -> list[PortfolioSolution]:
        solutions: list[PortfolioSolution] = []
        means = self.expected_returns.to_numpy(dtype=float)
        for target in target_returns:
            numeric_target = float(target)
            constraints = [
                {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)},
                {
                    "type": "eq",
                    "fun": lambda weights, expected=numeric_target: float(weights @ means - expected),
                },
            ]
            result = minimize(
                self._variance,
                np.full(len(self.assets), 1.0 / len(self.assets)),
                method="SLSQP",
                bounds=[(0.0, 1.0)] * len(self.assets),
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 2_000},
            )
            if result.success:
                solutions.append(
                    self._solution(
                        f"EFFICIENT_FRONTIER_{numeric_target:.8f}",
                        result.x,
                        True,
                        annual_risk_free_rate,
                    )
                )
        return solutions

    def compare(self, *, annual_risk_free_rate: float = 0.0) -> dict[str, Any]:
        solutions = [
            self.equal_weight(annual_risk_free_rate=annual_risk_free_rate),
            self.minimum_variance(annual_risk_free_rate=annual_risk_free_rate),
            self.maximum_sharpe(annual_risk_free_rate=annual_risk_free_rate),
            self.risk_parity(annual_risk_free_rate=annual_risk_free_rate),
        ]
        return {
            "schema": "portfolio_optimizer_comparison_v1",
            "assets": self.assets,
            "observations": len(self.returns),
            "periods_per_year": self.periods_per_year,
            "expected_returns": dict(zip(self.assets, self.expected_returns.to_numpy(dtype=float), strict=True)),
            "covariance": self.covariance.to_dict(),
            "correlation": self.correlation.to_dict(),
            "solutions": [solution.as_dict() for solution in solutions],
            "research_only": True,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }

    def _optimize(self, objective: Any, *, lower_bound: float = 0.0) -> Any:
        return minimize(
            objective,
            np.full(len(self.assets), 1.0 / len(self.assets)),
            method="SLSQP",
            bounds=[(lower_bound, 1.0)] * len(self.assets),
            constraints=[{"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}],
            options={"ftol": 1e-12, "maxiter": 2_000},
        )

    def _solution(
        self,
        method: str,
        raw_weights: np.ndarray,
        converged: bool,
        annual_risk_free_rate: float,
    ) -> PortfolioSolution:
        weights = np.clip(np.asarray(raw_weights, dtype=float), 0.0, None)
        weights = weights / weights.sum()
        expected_return = self._expected_return(weights)
        volatility = self._volatility(weights)
        sharpe = (expected_return - annual_risk_free_rate) / volatility if volatility > 0 else None
        contributions = self._fractional_risk_contributions(weights)
        return PortfolioSolution(
            method=method,
            weights=dict(zip(self.assets, weights.tolist(), strict=True)),
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            risk_contributions=dict(zip(self.assets, contributions.tolist(), strict=True)),
            converged=bool(converged),
        )

    def _expected_return(self, weights: np.ndarray) -> float:
        return float(weights @ self.expected_returns.to_numpy(dtype=float))

    def _variance(self, weights: np.ndarray) -> float:
        covariance = self.covariance.to_numpy(dtype=float)
        return float(weights @ covariance @ weights)

    def _volatility(self, weights: np.ndarray) -> float:
        return math_sqrt_nonnegative(self._variance(weights))

    def _fractional_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        covariance = self.covariance.to_numpy(dtype=float)
        variance = float(weights @ covariance @ weights)
        if variance <= 0:
            return np.zeros(len(weights))
        component_variance = weights * (covariance @ weights)
        return component_variance / variance


def math_sqrt_nonnegative(value: float) -> float:
    return float(np.sqrt(max(float(value), 0.0)))
