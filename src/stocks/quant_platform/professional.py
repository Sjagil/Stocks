from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd


class FactorRiskModel:
    """Linear multi-factor model R = βF + ε with covariance and attribution."""

    def __init__(self):
        self.betas: pd.DataFrame | None = None
        self.alphas: pd.Series | None = None
        self.factor_covariance: pd.DataFrame | None = None
        self.specific_variance: pd.Series | None = None

    def fit(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> FactorRiskModel:
        aligned = asset_returns.join(factor_returns, how="inner", lsuffix="__asset").dropna()
        assets = list(asset_returns.columns)
        factors = list(factor_returns.columns)
        if len(aligned) < len(factors) * 3 + 10:
            raise ValueError("insufficient observations for factor risk model")
        factor_values = aligned[factors].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(aligned)), factor_values])
        betas: dict[str, np.ndarray] = {}
        intercepts: dict[str, float] = {}
        residual_variance: dict[str, float] = {}
        for asset in assets:
            coefficients = np.linalg.lstsq(design, aligned[asset].to_numpy(dtype=float), rcond=None)[0]
            fitted = design @ coefficients
            residual = aligned[asset].to_numpy(dtype=float) - fitted
            intercepts[str(asset)] = float(coefficients[0])
            betas[str(asset)] = coefficients[1:]
            residual_variance[str(asset)] = float(np.var(residual, ddof=len(factors) + 1))
        self.betas = pd.DataFrame.from_dict(betas, orient="index", columns=factors)
        self.alphas = pd.Series(intercepts)
        self.factor_covariance = aligned[factors].cov()
        self.specific_variance = pd.Series(residual_variance)
        return self

    def covariance(self) -> pd.DataFrame:
        self._require_fitted()
        common = self.betas.to_numpy() @ self.factor_covariance.to_numpy() @ self.betas.to_numpy().T
        total = common + np.diag(self.specific_variance.reindex(self.betas.index).to_numpy())
        return pd.DataFrame(total, index=self.betas.index, columns=self.betas.index)

    def portfolio_exposures(self, weights: Mapping[str, float]) -> dict[str, float]:
        self._require_fitted()
        vector = pd.Series(weights, dtype=float).reindex(self.betas.index).fillna(0.0)
        return self.betas.mul(vector, axis=0).sum().to_dict()

    def attribute(
        self,
        weights: Mapping[str, float],
        factor_realization: Mapping[str, float],
        *,
        portfolio_return: float,
    ) -> dict[str, Any]:
        exposures = self.portfolio_exposures(weights)
        contributions = {
            factor: exposure * float(factor_realization.get(factor, 0.0))
            for factor, exposure in exposures.items()
        }
        alpha = float(portfolio_return - sum(contributions.values()))
        return {
            "portfolio_return": float(portfolio_return),
            "factor_exposures": exposures,
            "factor_contributions": contributions,
            "stock_selection_and_specific": alpha,
            "reconciled": bool(np.isclose(sum(contributions.values()) + alpha, portfolio_return)),
        }

    def report(self) -> dict[str, Any]:
        self._require_fitted()
        return {
            "schema": "factor_risk_model_v1",
            "betas": self.betas.to_dict(orient="index"),
            "specific_variance": self.specific_variance.to_dict(),
            "covariance": self.covariance().to_dict(),
            "execution_authority": "NONE",
            "broker_writes": 0,
        }

    def _require_fitted(self) -> None:
        if self.betas is None or self.alphas is None or self.factor_covariance is None or self.specific_variance is None:
            raise ValueError("factor risk model is not fitted")


class AlphaCombinationEngine:
    def fit(
        self,
        alphas: pd.DataFrame,
        future_returns: pd.Series,
        *,
        regimes: pd.Series | None = None,
        maximum_decay_lag: int = 5,
    ) -> dict[str, Any]:
        signals = alphas.apply(pd.to_numeric, errors="coerce")
        target = pd.to_numeric(future_returns, errors="coerce").reindex(signals.index)
        aligned = signals.assign(__target=target).dropna()
        signals = aligned.drop(columns="__target")
        target = aligned["__target"]
        information_coefficients = {
            column: float(signals[column].corr(target, method="spearman"))
            for column in signals
        }
        alpha_correlation = signals.corr()
        decay = {
            column: {
                lag: float(signals[column].corr(signals[column].shift(lag)))
                for lag in range(1, maximum_decay_lag + 1)
            }
            for column in signals
        }
        turnover = {column: float(signals[column].diff().abs().mean()) for column in signals}
        covariance = signals.cov().to_numpy(dtype=float)
        ic = np.asarray([information_coefficients[column] for column in signals])
        raw = np.linalg.pinv(covariance + np.eye(len(ic)) * 1e-8) @ ic
        if np.isclose(np.abs(raw).sum(), 0):
            raw = np.ones(len(ic))
        weights = raw / np.abs(raw).sum()
        combined = signals.to_numpy(dtype=float) @ weights
        conditional: dict[str, dict[str, float]] = {}
        if regimes is not None:
            labels = regimes.reindex(signals.index)
            for regime in sorted(labels.dropna().astype(str).unique()):
                mask = labels.astype(str) == regime
                conditional[regime] = {
                    column: float(signals.loc[mask, column].corr(target.loc[mask], method="spearman"))
                    for column in signals
                }
        return {
            "weights": dict(zip(signals.columns, weights.tolist(), strict=True)),
            "combined_alpha": pd.Series(combined, index=signals.index, name="combined_alpha"),
            "information_coefficients": information_coefficients,
            "alpha_correlation": alpha_correlation,
            "signal_decay": decay,
            "turnover": turnover,
            "conditional_information_coefficients": conditional,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }


class RegimeConditionalMixtureOfExperts:
    def combine(
        self,
        expert_predictions: pd.DataFrame,
        regime_probabilities: pd.DataFrame,
        expert_for_regime: Mapping[str, str],
    ) -> pd.DataFrame:
        common = expert_predictions.index.intersection(regime_probabilities.index)
        experts = expert_predictions.loc[common]
        probabilities = regime_probabilities.loc[common]
        missing_regimes = sorted(set(probabilities.columns) - set(expert_for_regime))
        if missing_regimes:
            raise ValueError(f"missing regime expert mappings: {', '.join(missing_regimes)}")
        if not np.allclose(probabilities.sum(axis=1), 1.0):
            raise ValueError("regime probabilities must sum to one")
        combined = pd.Series(0.0, index=common)
        contributions = pd.DataFrame(0.0, index=common, columns=experts.columns)
        for regime in probabilities:
            expert = expert_for_regime[regime]
            if expert not in experts:
                raise ValueError(f"unknown expert: {expert}")
            contribution = probabilities[regime] * experts[expert]
            combined += contribution
            contributions[expert] += contribution
        result = contributions.add_prefix("contribution_")
        result["combined_prediction"] = combined
        result["dominant_regime"] = probabilities.idxmax(axis=1)
        return result


@dataclass(frozen=True)
class RewardWeights:
    drawdown: float = 1.0
    turnover: float = 0.1
    transaction_costs: float = 1.0
    tail_risk: float = 0.5


class PortfolioRLEnvironment:
    """Causal portfolio environment; it supplies no claim of learned alpha."""

    def __init__(
        self,
        returns: pd.DataFrame,
        *,
        signals: pd.DataFrame | None = None,
        transaction_cost_bps: float = 5.0,
        reward_weights: RewardWeights | None = None,
    ):
        self.returns = returns.apply(pd.to_numeric, errors="coerce").dropna()
        if len(self.returns) < 2:
            raise ValueError("RL environment requires at least two observations")
        self.signals = signals.reindex(self.returns.index) if signals is not None else None
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.reward_weights = reward_weights or RewardWeights()
        self.assets = [str(column) for column in self.returns.columns]
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.position = 0
        self.weights = pd.Series(0.0, index=self.assets)
        self.cash = 1.0
        self.wealth = 1.0
        self.peak = 1.0
        return self.state()

    def state(self) -> dict[str, Any]:
        history = self.returns.iloc[: self.position + 1]
        return {
            "positions": self.weights.to_dict(),
            "cash_weight": self.cash,
            "signals": {} if self.signals is None else self.signals.iloc[self.position].to_dict(),
            "volatility": history.std(ddof=0).fillna(0.0).to_dict(),
            "correlation": history.corr().fillna(0.0).to_dict(),
            "drawdown": self.wealth / self.peak - 1.0,
            "step": self.position,
        }

    def step(self, target_weights: Mapping[str, float]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        target = pd.Series(target_weights, dtype=float).reindex(self.assets).fillna(0.0)
        if (target < 0).any() or target.sum() > 1 + 1e-12:
            raise ValueError("actions must be long-only weights summing to at most one")
        turnover = float((target - self.weights).abs().sum())
        costs = turnover * self.transaction_cost_bps / 10_000
        next_position = self.position + 1
        gross_return = float(target @ self.returns.iloc[next_position])
        net_return = gross_return - costs
        self.weights = target
        self.cash = 1.0 - float(target.sum())
        self.wealth *= 1 + net_return
        self.peak = max(self.peak, self.wealth)
        drawdown = max(1 - self.wealth / self.peak, 0.0)
        tail_risk = max(-gross_return, 0.0)
        penalties = self.reward_weights
        reward = (
            net_return
            - penalties.drawdown * drawdown
            - penalties.turnover * turnover
            - penalties.transaction_costs * costs
            - penalties.tail_risk * tail_risk
        )
        self.position = next_position
        done = self.position >= len(self.returns) - 1
        info = {
            "gross_return": gross_return,
            "net_return": net_return,
            "turnover": turnover,
            "transaction_costs": costs,
            "drawdown": drawdown,
            "tail_risk": tail_risk,
            "wealth": self.wealth,
        }
        return self.state(), float(reward), done, info


@dataclass
class ConstrainedQPortfolioPolicy:
    """Offline tabular Q-learning policy with cash and long-only actions.

    This policy is deliberately a shadow allocator.  It cannot size whole
    shares, bypass portfolio gates, or call a broker.
    """

    episodes: int = 60
    learning_rate: float = 0.15
    discount: float = 0.95
    epsilon: float = 0.20
    maximum_asset_weight: float = 0.25
    transaction_cost_bps: float = 5.0
    random_state: int = 42

    def __post_init__(self) -> None:
        if (
            self.episodes < 1
            or not 0 < self.learning_rate <= 1
            or not 0 <= self.discount <= 1
            or not 0 <= self.epsilon <= 1
            or not 0 < self.maximum_asset_weight <= 1
        ):
            raise ValueError("invalid constrained Q-policy configuration")
        self.assets: list[str] = []
        self.actions: list[dict[str, float]] = []
        self.q_values: dict[str, np.ndarray] = {}
        self.validation: dict[str, float] = {}

    def fit(
        self,
        returns: pd.DataFrame,
        *,
        signals: pd.DataFrame | None = None,
        train_fraction: float = 0.70,
    ) -> ConstrainedQPortfolioPolicy:
        clean = returns.apply(pd.to_numeric, errors="coerce").dropna()
        if len(clean) < 80 or not 0.5 <= train_fraction < 0.9:
            raise ValueError("RL policy requires at least 80 rows and a temporal holdout")
        self.assets = [str(column) for column in clean.columns]
        aligned_signals = (
            signals.reindex(clean.index).apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if signals is not None
            else clean.rolling(5, min_periods=1).mean()
        )
        self.actions = [{asset: 0.0 for asset in self.assets}]
        for asset in self.assets:
            self.actions.append(
                {
                    name: self.maximum_asset_weight if name == asset else 0.0
                    for name in self.assets
                }
            )
        split = int(len(clean) * train_fraction)
        train_returns = clean.iloc[:split]
        train_signals = aligned_signals.iloc[:split]
        rng = np.random.default_rng(self.random_state)
        self.q_values = {}
        for episode in range(self.episodes):
            environment = PortfolioRLEnvironment(
                train_returns,
                signals=train_signals,
                transaction_cost_bps=self.transaction_cost_bps,
            )
            state = environment.reset()
            done = False
            exploration = self.epsilon * (1 - episode / self.episodes)
            while not done:
                key = self._state_key(state)
                values = self.q_values.setdefault(key, np.zeros(len(self.actions)))
                action_index = (
                    int(rng.integers(len(self.actions)))
                    if rng.random() < exploration
                    else int(np.argmax(values))
                )
                next_state, reward, done, _ = environment.step(
                    self.actions[action_index]
                )
                next_values = self.q_values.setdefault(
                    self._state_key(next_state), np.zeros(len(self.actions))
                )
                target = reward + (0.0 if done else self.discount * float(next_values.max()))
                values[action_index] += self.learning_rate * (
                    target - values[action_index]
                )
                state = next_state
        validation_returns = clean.iloc[split - 1 :]
        validation_signals = aligned_signals.iloc[split - 1 :]
        environment = PortfolioRLEnvironment(
            validation_returns,
            signals=validation_signals,
            transaction_cost_bps=self.transaction_cost_bps,
        )
        state = environment.reset()
        rewards: list[float] = []
        turnovers: list[float] = []
        drawdowns: list[float] = []
        done = False
        while not done:
            _, weights = self.act(state)
            state, reward, done, info = environment.step(weights)
            rewards.append(float(reward))
            turnovers.append(float(info["turnover"]))
            drawdowns.append(float(info["drawdown"]))
        self.validation = {
            "out_of_sample_observations": float(len(validation_returns) - 1),
            "cumulative_net_return": float(environment.wealth - 1.0),
            "mean_reward": float(np.mean(rewards)),
            "mean_turnover": float(np.mean(turnovers)),
            "maximum_drawdown": float(max(drawdowns, default=0.0)),
        }
        return self

    def act(self, state: Mapping[str, Any]) -> tuple[int, dict[str, float]]:
        if not self.actions:
            raise ValueError("RL policy is not fitted")
        values = self.q_values.get(self._state_key(state))
        action_index = int(np.argmax(values)) if values is not None else 0
        weights = dict(self.actions[action_index])
        if any(value < 0 for value in weights.values()) or sum(weights.values()) > 1:
            raise ValueError("learned action violated long-only cash-secured policy")
        return action_index, weights

    def report(self) -> dict[str, Any]:
        return {
            "model_type": "CONSTRAINED_TABULAR_Q_LEARNING",
            "state_count": len(self.q_values),
            "action_count": len(self.actions),
            "actions": self.actions,
            "temporal_holdout_validation": self.validation,
            "long_only": True,
            "cash_action_available": True,
            "leverage_allowed": False,
            "shorting_allowed": False,
            "research_only": True,
            "money_control": False,
            "execution_authority": "NONE",
            "broker_writes": 0,
        }

    def _state_key(self, state: Mapping[str, Any]) -> str:
        signals = {
            str(key): float(value or 0.0)
            for key, value in dict(state.get("signals") or {}).items()
            if str(key) in self.assets
        }
        dominant = max(signals, key=signals.get) if signals else "CASH"
        strength = signals.get(dominant, 0.0)
        signal_bucket = "POS" if strength > 0 else "NON_POS"
        drawdown = float(state.get("drawdown") or 0.0)
        drawdown_bucket = "DD_HIGH" if drawdown < -0.05 else "DD_LOW"
        volatility = dict(state.get("volatility") or {})
        mean_volatility = float(np.mean(list(volatility.values()))) if volatility else 0.0
        volatility_bucket = "VOL_HIGH" if mean_volatility > 0.03 else "VOL_LOW"
        return "|".join((dominant, signal_bucket, drawdown_bucket, volatility_bucket))
