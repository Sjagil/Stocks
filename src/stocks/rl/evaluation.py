from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Protocol

import numpy as np
import pandas as pd

from stocks.portfolio.position_management import evaluate_long_position
from stocks.rl.contracts import RLAction, stable_hash
from stocks.rl.environment import FinanceSwingEnv, SwingEnvironmentConfig
from stocks.rl.registry import PromotionDecision


@dataclass(frozen=True)
class PolicyDecision:
    action: int
    probabilities: tuple[float, ...]


class DecisionPolicy(Protocol):
    def __call__(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        context: dict[str, Any],
        rng: np.random.Generator,
    ) -> PolicyDecision: ...


@dataclass(frozen=True)
class PromotionGateConfig:
    version: str = "RL_PROMOTION_GATE_V1"
    maximum_drawdown: float = 0.10
    minimum_worst_regime_return: float = -0.02
    minimum_cvar_95: float = -0.03
    maximum_turnover: float = 12.0
    minimum_trade_count: int = 100
    minimum_episode_count: int = 30
    minimum_bootstrap_probability_of_improvement: float = 0.95
    required_cost_stress_multiplier: float = 2.0
    minimum_cost_stress_net_return: float = 0.0

    def validate(self) -> None:
        if not 0 < self.maximum_drawdown < 1:
            raise ValueError("invalid RL maximum drawdown gate")
        if self.minimum_trade_count < 1 or self.minimum_episode_count < 1:
            raise ValueError("RL promotion sample gates must be positive")
        if not 0.5 < self.minimum_bootstrap_probability_of_improvement <= 1:
            raise ValueError("invalid RL bootstrap promotion probability")
        if self.required_cost_stress_multiplier < 1:
            raise ValueError("RL promotion cost stress must not reduce costs")


def cash_policy(
    observation: np.ndarray,
    mask: np.ndarray,
    context: dict[str, Any],
    rng: np.random.Generator,
) -> PolicyDecision:
    return _one_hot_decision(RLAction.HOLD, len(mask))


def highest_ranked_policy(
    observation: np.ndarray,
    mask: np.ndarray,
    context: dict[str, Any],
    rng: np.random.Generator,
) -> PolicyDecision:
    if mask[RLAction.OPEN_LARGE]:
        return _one_hot_decision(RLAction.OPEN_LARGE, len(mask))
    if mask[RLAction.CLOSE] and float(context["row"]["signal_direction"]) <= 0:
        return _one_hot_decision(RLAction.CLOSE, len(mask))
    return _one_hot_decision(RLAction.HOLD, len(mask))


def fixed_sizing_policy(
    observation: np.ndarray,
    mask: np.ndarray,
    context: dict[str, Any],
    rng: np.random.Generator,
) -> PolicyDecision:
    if mask[RLAction.OPEN_NORMAL]:
        return _one_hot_decision(RLAction.OPEN_NORMAL, len(mask))
    position = context["position"]
    row = context["row"]
    if mask[RLAction.CLOSE] and (
        float(row["signal_direction"]) <= 0 or int(position["holding_steps"]) >= 10
    ):
        return _one_hot_decision(RLAction.CLOSE, len(mask))
    return _one_hot_decision(RLAction.HOLD, len(mask))


def deterministic_engine_policy(
    observation: np.ndarray,
    mask: np.ndarray,
    context: dict[str, Any],
    rng: np.random.Generator,
) -> PolicyDecision:
    """Replay existing deterministic position management in the RL environment."""

    if mask[RLAction.OPEN_NORMAL]:
        return _one_hot_decision(RLAction.OPEN_NORMAL, len(mask))
    position = context["position"]
    if not position["weight"]:
        return _one_hot_decision(RLAction.HOLD, len(mask))
    row = context["row"]
    decision = evaluate_long_position(
        entry_price=float(position["entry_price"]),
        current_price=float(row["close"]),
        initial_stop=float(position["stop_price"]),
        previous_stop=float(position["stop_price"]),
        peak_price=float(position["peak_price"]),
        atr=float(row["atr"]),
        structural_stop=None,
        trend_strength=max(0.0, min(1.0, (float(row["trend_strength"]) + 1.0) / 2.0)),
        volatility_regime=max(
            0.0, min(1.0, (float(row["volatility_regime"]) + 1.0) / 2.0)
        ),
    )
    mapping = {
        "EXIT": RLAction.CLOSE,
        "REDUCE_50": RLAction.REDUCE_50,
        "TAKE_PARTIAL_25": RLAction.REDUCE_25,
        "TAKE_PARTIAL_50": RLAction.REDUCE_50,
        "UPDATE_TRAILING_STOP": RLAction.TIGHTEN_STOP,
        "HOLD": RLAction.HOLD,
    }
    action = mapping.get(str(decision.get("action")), RLAction.HOLD)
    if not mask[action]:
        action = RLAction.HOLD
    return _one_hot_decision(action, len(mask))


def random_valid_policy(
    observation: np.ndarray,
    mask: np.ndarray,
    context: dict[str, Any],
    rng: np.random.Generator,
) -> PolicyDecision:
    valid = np.flatnonzero(mask)
    action = int(rng.choice(valid))
    probabilities = np.zeros(len(mask), dtype=float)
    probabilities[valid] = 1.0 / len(valid)
    return PolicyDecision(action, tuple(float(x) for x in probabilities))


BASELINE_POLICIES: dict[str, DecisionPolicy] = {
    "ALWAYS_HIGHEST_RANKED": highest_ranked_policy,
    "FIXED_POSITION_SIZING": fixed_sizing_policy,
    "EXISTING_DETERMINISTIC_ENGINE": deterministic_engine_policy,
    "RANDOM_VALID_ACTIONS": random_valid_policy,
    "DO_NOTHING_CASH": cash_policy,
}


def evaluate_policy(
    frame: pd.DataFrame,
    *,
    scaler: Any,
    environment_config: SwingEnvironmentConfig,
    policy: DecisionPolicy,
    policy_name: str,
    seed: int,
) -> dict[str, Any]:
    env = FinanceSwingEnv(
        frame,
        scaler=scaler,
        config=environment_config,
        seed=seed,
    )
    observation, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    rewards: list[float] = []
    costs: list[float] = []
    turnovers: list[float] = []
    regimes: list[str] = []
    actions: list[str] = []
    holding_durations: list[int] = []
    decision_probabilities: list[float] = []
    strategies: list[str] = []
    asset_classes: list[str] = []
    skipped_opportunity_returns: list[float] = []
    terminated = False
    while not terminated:
        mask = env.action_masks()
        context = env.decision_context
        decision = policy(observation, mask, context, rng)
        if len(decision.probabilities) != len(mask):
            raise ValueError("policy probability vector does not match action space")
        selected_probability = float(decision.probabilities[int(decision.action)])
        observation, reward, terminated, truncated, info = env.step(decision.action)
        if truncated:
            raise RuntimeError("unexpected truncated swing episode")
        returns.append(float(info["portfolio_return"]))
        rewards.append(float(reward))
        costs.append(float(info["transaction_cost_return"]))
        turnovers.append(float(info["turnover"]))
        regimes.append(str(context["row"]["market_regime"]))
        actions.append(str(info["effective_action"]))
        strategies.append(str(context["row"].get("strategy_id") or "UNKNOWN"))
        asset_classes.append(str(context["row"].get("asset_class") or "UNKNOWN"))
        if (
            str(info["effective_action"]) == "HOLD"
            and float(context["row"].get("setup_score", 0.0))
            >= environment_config.minimum_setup_score
        ):
            skipped_opportunity_returns.append(
                float(context["row"].get("outcome_next_return", 0.0))
            )
        holding_durations.append(int(info["holding_duration"]))
        decision_probabilities.append(selected_probability)
    metrics = performance_metrics(
        returns,
        costs=costs,
        turnovers=turnovers,
        actions=actions,
        holding_durations=holding_durations,
    )
    regime_metrics = _regime_metrics(returns, regimes)
    metrics["expected_value_of_skipped_opportunities"] = (
        float(np.mean(skipped_opportunity_returns))
        if skipped_opportunity_returns
        else None
    )
    result: dict[str, Any] = {
        "schema": "rl_policy_evaluation_v1",
        "status": "GO",
        "policy": policy_name,
        "seed": seed,
        "observations": len(returns),
        "episodes": 1,
        "metrics": metrics,
        "action_distribution": dict(Counter(actions)),
        "mean_selected_action_probability": float(np.mean(decision_probabilities)),
        "reward_distribution": _distribution(rewards),
        "regime_performance": regime_metrics,
        "performance_by_strategy": _group_metrics(returns, strategies),
        "performance_by_asset_class": _group_metrics(returns, asset_classes),
        "bootstrap": bootstrap_expectancy(returns, seed=seed),
        "monte_carlo_trade_order": monte_carlo_trade_order(returns, seed=seed),
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "money_control": False,
    }
    result["evaluation_hash"] = stable_hash(result)
    return result


def evaluate_baselines(
    frame: pd.DataFrame,
    *,
    scaler: Any,
    environment_config: SwingEnvironmentConfig,
    seed: int = 42,
) -> dict[str, Any]:
    results = {
        name: evaluate_policy(
            frame,
            scaler=scaler,
            environment_config=environment_config,
            policy=policy,
            policy_name=name,
            seed=seed,
        )
        for name, policy in BASELINE_POLICIES.items()
    }
    payload = {
        "schema": "rl_baseline_comparison_v1",
        "status": "GO",
        "baselines": results,
        "primary_baseline": "EXISTING_DETERMINISTIC_ENGINE",
        "execution_authority": "NONE",
        "broker_writes": 0,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def performance_metrics(
    returns: Iterable[float],
    *,
    costs: Iterable[float] | None = None,
    turnovers: Iterable[float] | None = None,
    actions: Iterable[str] | None = None,
    holding_durations: Iterable[int] | None = None,
    periods_per_year: int = 252,
) -> dict[str, float | int | None]:
    values = np.asarray(list(returns), dtype=float)
    costs_array = np.asarray(list(costs or []), dtype=float)
    turnover_array = np.asarray(list(turnovers or []), dtype=float)
    actions_list = list(actions or [])
    holding = np.asarray(list(holding_durations or []), dtype=float)
    if len(values) == 0:
        raise ValueError("cannot evaluate an empty return series")
    equity = np.cumprod(1.0 + np.clip(values, -0.999999, None))
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    total_return = float(equity[-1] - 1.0)
    years = len(values) / periods_per_year
    cagr = float(equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else -1.0
    volatility = float(np.std(values, ddof=0))
    downside = values[values < 0]
    downside_volatility = float(np.std(downside, ddof=0)) if len(downside) else 0.0
    sharpe = _safe_ratio(float(np.mean(values)), volatility) * math.sqrt(periods_per_year)
    sortino = _safe_ratio(float(np.mean(values)), downside_volatility) * math.sqrt(periods_per_year)
    maximum_drawdown = float(abs(np.min(drawdowns)))
    calmar = _safe_ratio(cagr, maximum_drawdown)
    gross_profit = float(values[values > 0].sum())
    gross_loss = float(abs(values[values < 0].sum()))
    profit_factor = _safe_ratio(gross_profit, gross_loss)
    tail = max(1, int(math.ceil(len(values) * 0.05)))
    worst = np.sort(values)[:tail]
    trades = sum(name.startswith("OPEN_") for name in actions_list)
    participation = trades / max(1, len(values))
    skip_count = sum(name == "HOLD" for name in actions_list)
    return {
        "net_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "maximum_drawdown": maximum_drawdown,
        "profit_factor": profit_factor,
        "expectancy": float(np.mean(values)),
        "hit_rate": float(np.mean(values > 0)),
        "turnover": float(turnover_array.sum()) if len(turnover_array) else 0.0,
        "fees_and_slippage": float(costs_array.sum()) if len(costs_array) else 0.0,
        "number_trades": int(trades),
        "average_holding_period": float(np.mean(holding)) if len(holding) else 0.0,
        "tail_loss_5pct": float(worst[-1]),
        "cvar_95": float(np.mean(worst)),
        "trade_participation_rate": participation,
        "qualified_opportunity_skip_rate": skip_count / max(1, len(values)),
    }


def bootstrap_expectancy(
    returns: Iterable[float], *, seed: int, samples: int = 1_000
) -> dict[str, float | int]:
    values = np.asarray(list(returns), dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(samples)]
    )
    return {
        "samples": samples,
        "mean": float(np.mean(means)),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "probability_positive": float(np.mean(means > 0)),
    }


def _group_metrics(
    returns: list[float], groups: list[str]
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(returns, groups, strict=True):
        grouped[group].append(float(value))
    return {
        group: {
            "observations": len(values),
            "net_return": float(np.prod(1.0 + np.asarray(values)) - 1.0),
            "expectancy": float(np.mean(values)),
            "hit_rate": float(np.mean(np.asarray(values) > 0)),
        }
        for group, values in grouped.items()
    }


def bootstrap_probability_of_improvement(
    challenger_returns: Iterable[float],
    baseline_returns: Iterable[float],
    *,
    seed: int,
    samples: int = 2_000,
) -> float:
    challenger = np.asarray(list(challenger_returns), dtype=float)
    baseline = np.asarray(list(baseline_returns), dtype=float)
    size = min(len(challenger), len(baseline))
    if size == 0:
        return 0.0
    differences = challenger[-size:] - baseline[-size:]
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(differences, size=size, replace=True)))
        for _ in range(samples)
    ]
    return float(np.mean(np.asarray(means) > 0))


def monte_carlo_trade_order(
    returns: Iterable[float], *, seed: int, samples: int = 500
) -> dict[str, float | int]:
    values = np.asarray(list(returns), dtype=float)
    rng = np.random.default_rng(seed)
    drawdowns: list[float] = []
    for _ in range(samples):
        shuffled = rng.permutation(values)
        equity = np.cumprod(1.0 + np.clip(shuffled, -0.999999, None))
        drawdowns.append(float(abs(np.min(equity / np.maximum.accumulate(equity) - 1.0))))
    return {
        "samples": samples,
        "median_max_drawdown": float(np.median(drawdowns)),
        "worst_95_max_drawdown": float(np.quantile(drawdowns, 0.95)),
    }


def cost_stress_evaluation(
    frame: pd.DataFrame,
    *,
    scaler: Any,
    base_config: SwingEnvironmentConfig,
    policy: DecisionPolicy,
    policy_name: str,
    seed: int,
    multipliers: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0),
) -> dict[str, Any]:
    scenarios = {}
    for multiplier in multipliers:
        config = replace(base_config, cost_stress_multiplier=multiplier)
        scenarios[f"{multiplier:.2f}x"] = evaluate_policy(
            frame,
            scaler=scaler,
            environment_config=config,
            policy=policy,
            policy_name=policy_name,
            seed=seed,
        )["metrics"]
    return {
        "schema": "rl_cost_stress_v1",
        "status": "GO",
        "scenarios": scenarios,
        "execution_authority": "NONE",
    }


def evaluate_promotion_gate(
    *,
    challenger_version: str,
    active_version: str | None,
    challenger: dict[str, Any],
    baseline: dict[str, Any],
    cost_stress: dict[str, Any],
    episode_count: int,
    bootstrap_probability: float,
    safety_regression: bool,
    data_blockers: Iterable[str],
    config: PromotionGateConfig | None = None,
) -> PromotionDecision:
    config = config or PromotionGateConfig()
    config.validate()
    challenger_metrics = challenger["metrics"]
    baseline_metrics = baseline["metrics"]
    regimes = challenger.get("regime_performance", {})
    worst_regime = min(
        (float(value["net_return"]) for value in regimes.values()), default=-1.0
    )
    stressed = cost_stress.get("scenarios", {}).get(
        f"{config.required_cost_stress_multiplier:.2f}x", {}
    )
    checks = {
        "NET_EXPECTANCY_BEATS_ACTIVE_OR_DETERMINISTIC": float(
            challenger_metrics["expectancy"]
        )
        > float(baseline_metrics["expectancy"]),
        "MAX_DRAWDOWN_WITHIN_LIMIT": float(challenger_metrics["maximum_drawdown"])
        <= config.maximum_drawdown,
        "WORST_REGIME_ACCEPTABLE": worst_regime
        >= config.minimum_worst_regime_return,
        "CVAR_ACCEPTABLE": float(challenger_metrics["cvar_95"])
        >= config.minimum_cvar_95,
        "TURNOVER_ACCEPTABLE": float(challenger_metrics["turnover"])
        <= config.maximum_turnover,
        "MINIMUM_TRADE_COUNT": int(challenger_metrics["number_trades"])
        >= config.minimum_trade_count,
        "MINIMUM_EPISODE_COUNT": int(episode_count) >= config.minimum_episode_count,
        "BOOTSTRAP_CONFIDENCE": bootstrap_probability
        >= config.minimum_bootstrap_probability_of_improvement,
        "COST_STRESS_SURVIVES": float(stressed.get("net_return", -1.0))
        >= config.minimum_cost_stress_net_return,
        "NO_SAFETY_REGRESSION": not safety_regression,
        "PRODUCTION_DATA_GATES": not list(data_blockers),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    body = {
        "challenger_version": challenger_version,
        "active_version": active_version,
        "checks": checks,
        "config": asdict(config),
        "data_blockers": list(data_blockers),
    }
    return PromotionDecision(
        status="PROMOTION_ELIGIBLE" if not reasons else "REJECT_CHALLENGER",
        passed=not reasons,
        reasons=reasons,
        evidence_hash=stable_hash(body),
        challenger_version=challenger_version,
        active_version=active_version,
        safety_regression=safety_regression,
    )


def _regime_metrics(returns: list[float], regimes: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, regime in zip(returns, regimes, strict=True):
        grouped[regime].append(value)
    return {
        regime: {
            "observations": len(values),
            "net_return": float(np.prod(1.0 + np.asarray(values)) - 1.0),
            "expectancy": float(np.mean(values)),
            "hit_rate": float(np.mean(np.asarray(values) > 0)),
        }
        for regime, values in sorted(grouped.items())
    }


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "std": float(np.std(array, ddof=0)),
    }


def _one_hot_decision(action: int | RLAction, size: int) -> PolicyDecision:
    values = [0.0] * size
    values[int(action)] = 1.0
    return PolicyDecision(int(action), tuple(values))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-15:
        return 0.0
    return numerator / denominator


__all__ = [
    "BASELINE_POLICIES",
    "DecisionPolicy",
    "PolicyDecision",
    "PromotionGateConfig",
    "bootstrap_expectancy",
    "bootstrap_probability_of_improvement",
    "cost_stress_evaluation",
    "evaluate_baselines",
    "evaluate_policy",
    "evaluate_promotion_gate",
    "performance_metrics",
]
