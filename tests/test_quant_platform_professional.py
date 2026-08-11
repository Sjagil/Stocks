from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks.quant_platform import (
    AlphaCombinationEngine,
    ConstrainedQPortfolioPolicy,
    FactorRiskModel,
    PortfolioRLEnvironment,
    RegimeConditionalMixtureOfExperts,
    RewardWeights,
)


def test_factor_risk_model_recovers_exposures_covariance_and_attribution() -> None:
    rng = np.random.default_rng(50)
    factors = pd.DataFrame(rng.normal(0, 0.01, size=(1_000, 4)), columns=["market", "value", "momentum", "quality"])
    true_betas = np.asarray([[1.2, 0.3, 0.5, 0.1], [0.8, -0.2, -0.1, 0.4], [0.1, 0.5, 0.2, 0.3]])
    assets = pd.DataFrame(factors.to_numpy() @ true_betas.T + rng.normal(0, 0.002, size=(1_000, 3)), columns=["A", "B", "C"])
    model = FactorRiskModel().fit(assets, factors)
    assert model.betas.loc["A", "market"] == pytest.approx(1.2, rel=0.05)
    assert model.covariance().shape == (3, 3)
    attribution = model.attribute({"A": 0.5, "B": 0.3, "C": 0.2}, {"market": 0.01, "value": 0.005}, portfolio_return=0.02)
    assert attribution["reconciled"]
    assert model.report()["broker_writes"] == 0


def test_alpha_combination_uses_ic_correlation_decay_turnover_and_regime_performance() -> None:
    rng = np.random.default_rng(51)
    index = pd.date_range("2024-01-01", periods=300, freq="B")
    alphas = pd.DataFrame(rng.normal(size=(300, 3)), columns=["momentum", "value", "nlp"], index=index)
    future = 0.02 * alphas["momentum"] - 0.01 * alphas["value"] + rng.normal(0, 0.01, 300)
    regimes = pd.Series(np.where(np.arange(300) < 150, "TREND", "CHOP"), index=index)
    result = AlphaCombinationEngine().fit(alphas, future, regimes=regimes)
    assert sum(abs(weight) for weight in result["weights"].values()) == pytest.approx(1.0)
    assert result["information_coefficients"]["momentum"] > 0
    assert set(result["conditional_information_coefficients"]) == {"CHOP", "TREND"}
    assert len(result["signal_decay"]["momentum"]) == 5


def test_mixture_of_experts_uses_regime_probabilities_as_gate() -> None:
    index = pd.RangeIndex(2)
    experts = pd.DataFrame({"trend": [0.10, 0.20], "mean_reversion": [-0.05, 0.03]}, index=index)
    probabilities = pd.DataFrame({"TREND": [0.8, 0.2], "CHOP": [0.2, 0.8]}, index=index)
    result = RegimeConditionalMixtureOfExperts().combine(experts, probabilities, {"TREND": "trend", "CHOP": "mean_reversion"})
    assert result.loc[0, "combined_prediction"] == pytest.approx(0.07)
    assert result.loc[1, "dominant_regime"] == "CHOP"


def test_rl_environment_reward_penalizes_turnover_cost_drawdown_and_tail_risk() -> None:
    returns = pd.DataFrame({"SPY": [0.0, -0.10, 0.02], "GLD": [0.0, 0.01, 0.01]})
    environment = PortfolioRLEnvironment(
        returns,
        transaction_cost_bps=10,
        reward_weights=RewardWeights(drawdown=1, turnover=0.1, transaction_costs=1, tail_risk=0.5),
    )
    state, reward, done, info = environment.step({"SPY": 1.0, "GLD": 0.0})
    assert reward < info["net_return"]
    assert info["turnover"] == pytest.approx(1.0)
    assert info["drawdown"] > 0
    assert not done
    assert "correlation" in state


def test_rl_environment_rejects_leveraged_or_short_action() -> None:
    environment = PortfolioRLEnvironment(pd.DataFrame({"A": [0.0, 0.01], "B": [0.0, 0.01]}))
    with pytest.raises(ValueError, match="long-only"):
        environment.step({"A": 1.1, "B": -0.1})


def test_constrained_q_policy_trains_on_temporal_holdout_and_stays_shadow() -> None:
    rng = np.random.default_rng(53)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0.0005, 0.01, 180),
            "B": rng.normal(0.0002, 0.008, 180),
        }
    )
    signals = returns.rolling(10, min_periods=1).mean()
    policy = ConstrainedQPortfolioPolicy(
        episodes=8, maximum_asset_weight=0.25, random_state=4
    ).fit(returns, signals=signals)
    _, weights = policy.act(
        {
            "signals": signals.iloc[-1].to_dict(),
            "volatility": returns.std(ddof=0).to_dict(),
            "drawdown": -0.01,
        }
    )
    assert all(value >= 0 for value in weights.values())
    assert sum(weights.values()) <= 1
    report = policy.report()
    assert report["temporal_holdout_validation"]["out_of_sample_observations"] > 0
    assert report["cash_action_available"]
    assert report["money_control"] is False
    assert report["execution_authority"] == "NONE"
