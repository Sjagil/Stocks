from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RLRewardConfig:
    """Central, versionable coefficients for risk-aware swing rewards."""

    version: str = "RL_REWARD_V1"
    return_scale: float = 100.0
    loss_aversion: float = 1.50
    realized_profit_weight: float = 0.15
    risk_adjusted_weight: float = 0.10
    good_management_weight: float = 0.10
    skip_quality_weight: float = 0.05
    transaction_cost_weight: float = 1.00
    drawdown_weight: float = 2.50
    drawdown_convex_power: float = 2.0
    downside_weight: float = 1.25
    excessive_risk_weight: float = 2.00
    concentration_weight: float = 0.75
    turnover_weight: float = 0.35
    bad_execution_weight: float = 1.00
    invalid_action_penalty: float = 2.00
    episode_end_open_position_penalty: float = 0.20
    risk_normalizer: float = 0.01
    component_clip: float = 10.0

    def validate(self) -> None:
        numeric = asdict(self)
        numeric.pop("version")
        if any(not math.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("reward configuration contains non-finite values")
        if any(float(value) < 0 for value in numeric.values()):
            raise ValueError("reward configuration weights must be non-negative")
        if self.loss_aversion < 1.0:
            raise ValueError("loss aversion must be at least one")
        if self.drawdown_convex_power <= 1.0:
            raise ValueError("drawdown penalty must be convex")
        if self.risk_normalizer <= 0 or self.component_clip <= 0:
            raise ValueError("reward normalization values must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class RewardInput:
    net_return: float = 0.0
    realized_return: float = 0.0
    transaction_cost_return: float = 0.0
    drawdown: float = 0.0
    downside_return: float = 0.0
    risk_excess: float = 0.0
    concentration_excess: float = 0.0
    turnover: float = 0.0
    bad_execution: float = 0.0
    invalid_action: bool = False
    good_management: float = 0.0
    skipped_opportunity_return: float = 0.0
    episode_end_open_position: bool = False

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite reward input: {name}")


@dataclass(frozen=True)
class RewardBreakdown:
    components: dict[str, float]
    total: float
    config_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": dict(self.components),
            "total": self.total,
            "config_version": self.config_version,
        }


def calculate_reward(
    inputs: RewardInput,
    config: RLRewardConfig | None = None,
) -> RewardBreakdown:
    config = config or RLRewardConfig()
    config.validate()
    inputs.validate()

    scaled_return = inputs.net_return * config.return_scale
    if scaled_return < 0:
        scaled_return *= config.loss_aversion
    risk_adjusted = (
        inputs.net_return / config.risk_normalizer
    ) * config.risk_adjusted_weight
    skip_quality = -inputs.skipped_opportunity_return * (
        config.return_scale * config.skip_quality_weight
    )

    components = {
        "net_return_reward": _clip(scaled_return, config.component_clip),
        "realized_profit_reward": _clip(
            inputs.realized_return
            * config.return_scale
            * config.realized_profit_weight,
            config.component_clip,
        ),
        "risk_adjusted_reward": _clip(risk_adjusted, config.component_clip),
        "good_trade_management_reward": _clip(
            inputs.good_management * config.good_management_weight,
            config.component_clip,
        ),
        "skip_quality_reward": _clip(skip_quality, config.component_clip),
        "transaction_cost_penalty": -_positive_clip(
            inputs.transaction_cost_return
            * config.return_scale
            * config.transaction_cost_weight,
            config.component_clip,
        ),
        "drawdown_penalty": -_positive_clip(
            config.drawdown_weight
            * ((max(0.0, inputs.drawdown) * config.return_scale) ** config.drawdown_convex_power)
            / config.return_scale,
            config.component_clip,
        ),
        "downside_penalty": -_positive_clip(
            abs(min(0.0, inputs.downside_return))
            * config.return_scale
            * config.downside_weight,
            config.component_clip,
        ),
        "excessive_risk_penalty": -_positive_clip(
            inputs.risk_excess * config.excessive_risk_weight,
            config.component_clip,
        ),
        "concentration_penalty": -_positive_clip(
            inputs.concentration_excess * config.concentration_weight,
            config.component_clip,
        ),
        "turnover_penalty": -_positive_clip(
            inputs.turnover * config.turnover_weight,
            config.component_clip,
        ),
        "bad_execution_penalty": -_positive_clip(
            inputs.bad_execution * config.bad_execution_weight,
            config.component_clip,
        ),
        "invalid_action_penalty": (
            -config.invalid_action_penalty if inputs.invalid_action else 0.0
        ),
        "episode_end_open_position_penalty": (
            -config.episode_end_open_position_penalty
            if inputs.episode_end_open_position
            else 0.0
        ),
    }
    rounded = {name: round(float(value), 12) for name, value in components.items()}
    total = round(sum(rounded.values()), 12)
    return RewardBreakdown(rounded, total, config.version)


def _clip(value: float, limit: float) -> float:
    return min(limit, max(-limit, float(value)))


def _positive_clip(value: float, limit: float) -> float:
    return min(limit, max(0.0, float(value)))


__all__ = [
    "RLRewardConfig",
    "RewardBreakdown",
    "RewardInput",
    "calculate_reward",
]
