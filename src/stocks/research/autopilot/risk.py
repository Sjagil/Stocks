from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    commission_bps: float
    half_spread_bps: float
    slippage_bps: float
    fx_cost_bps: float

    @property
    def total_bps(self) -> float:
        return (
            self.commission_bps
            + self.half_spread_bps
            + self.slippage_bps
            + self.fx_cost_bps
        )

    def as_dict(self) -> dict[str, float]:
        return {**asdict(self), "total_bps": self.total_bps}


COST_MODELS = {
    "NORMAL": CostModel(commission_bps=2.0, half_spread_bps=4.0, slippage_bps=3.0, fx_cost_bps=1.0),
    "DOUBLE": CostModel(commission_bps=4.0, half_spread_bps=8.0, slippage_bps=6.0, fx_cost_bps=2.0),
    "STRESS": CostModel(commission_bps=10.0, half_spread_bps=20.0, slippage_bps=15.0, fx_cost_bps=5.0),
}


@dataclass(frozen=True)
class PortfolioRiskLimits:
    max_position_weight: float = 0.15
    max_sector_weight: float = 0.35
    max_region_weight: float = 0.50
    max_currency_weight: float = 0.60
    max_total_exposure: float = 1.0
    max_one_way_turnover: float = 1.0
    max_position_loss: float = -0.12
    drawdown_circuit_breaker: float = -0.30
    min_median_dollar_volume: float = 1_000_000.0
    min_order_notional_eur: float = 25.0

    def validate(self) -> None:
        bounded = {
            "max_position_weight": self.max_position_weight,
            "max_sector_weight": self.max_sector_weight,
            "max_region_weight": self.max_region_weight,
            "max_currency_weight": self.max_currency_weight,
            "max_total_exposure": self.max_total_exposure,
            "max_one_way_turnover": self.max_one_way_turnover,
        }
        for name, value in bounded.items():
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        if not -1 < self.drawdown_circuit_breaker < 0:
            raise ValueError("drawdown_circuit_breaker must be in (-1,0)")
        if not -1 < self.max_position_loss < 0:
            raise ValueError("max_position_loss must be in (-1,0)")
        if self.min_median_dollar_volume < 0 or self.min_order_notional_eur < 0:
            raise ValueError("liquidity and order limits must be non-negative")


def enforce_portfolio_limits(
    weights: pd.DataFrame,
    *,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
    limits: PortfolioRiskLimits = PortfolioRiskLimits(),
) -> pd.DataFrame:
    limits.validate()
    constrained = weights.fillna(0.0).clip(lower=0.0, upper=limits.max_position_weight)
    if metadata:
        for field, cap in (
            ("sector", limits.max_sector_weight),
            ("region", limits.max_region_weight),
            ("currency", limits.max_currency_weight),
        ):
            groups: dict[str, list[str]] = {}
            for symbol in constrained.columns:
                group = str(metadata.get(symbol, {}).get(field) or "UNKNOWN")
                groups.setdefault(group, []).append(symbol)
            for symbols in groups.values():
                group_total = constrained[symbols].sum(axis=1)
                scale = (cap / group_total).clip(upper=1.0).fillna(1.0)
                constrained.loc[:, symbols] = constrained[symbols].mul(scale, axis=0)
    total = constrained.sum(axis=1)
    total_scale = (limits.max_total_exposure / total.where(total > 0)).clip(upper=1.0)
    constrained = constrained.mul(total_scale.fillna(1.0), axis=0)
    desired_change = constrained.diff().fillna(constrained)
    one_way = desired_change.abs().sum(axis=1) / 2.0
    turnover_scale = (
        limits.max_one_way_turnover / one_way.where(one_way > 0)
    ).clip(upper=1.0).fillna(1.0)
    bounded_change = desired_change.mul(turnover_scale, axis=0)
    output = pd.DataFrame(0.0, index=constrained.index, columns=constrained.columns)
    previous = pd.Series(0.0, index=constrained.columns)
    for timestamp in constrained.index:
        previous = (previous + bounded_change.loc[timestamp]).clip(lower=0.0)
        total_weight = float(previous.sum())
        if total_weight > limits.max_total_exposure:
            previous *= limits.max_total_exposure / total_weight
        output.loc[timestamp] = previous
    return output


def hierarchical_group_weights(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    metadata: Mapping[str, Mapping[str, str]],
    group_field: str,
) -> pd.DataFrame:
    """Allocate equally across groups, then by positive score within each group."""
    if group_field not in {"sector", "region"}:
        raise ValueError("UNSUPPORTED_HIERARCHICAL_GROUP")
    missing = [
        symbol
        for symbol in scores.columns
        if not str(metadata.get(symbol, {}).get(group_field) or "").strip()
    ]
    if missing:
        raise ValueError(
            f"HIERARCHICAL_METADATA_REQUIRED:{group_field}:{','.join(missing)}"
        )
    output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    groups: dict[str, list[str]] = {}
    for symbol in scores.columns:
        group = str(metadata[symbol][group_field])
        groups.setdefault(group, []).append(symbol)
    for timestamp in scores.index:
        active_groups = [
            (group, symbols)
            for group, symbols in groups.items()
            if bool(selected.loc[timestamp, symbols].any())
        ]
        if not active_groups:
            continue
        group_weight = 1.0 / len(active_groups)
        for _, symbols in active_groups:
            active = selected.loc[timestamp, symbols]
            names = list(active[active].index)
            positive = scores.loc[timestamp, names].clip(lower=0.0)
            denominator = float(positive.sum())
            if denominator > 0:
                output.loc[timestamp, names] = (
                    positive / denominator * group_weight
                )
            else:
                output.loc[timestamp, names] = group_weight / len(names)
    return output


def cost_breakdown(
    turnover: pd.Series,
    profile: str,
) -> dict[str, pd.Series]:
    if profile not in COST_MODELS:
        raise ValueError("UNKNOWN_COST_PROFILE")
    model = COST_MODELS[profile]
    divisor = 10_000.0
    return {
        "commission": turnover * model.commission_bps / divisor,
        "spread": turnover * model.half_spread_bps / divisor,
        "slippage": turnover * model.slippage_bps / divisor,
        "fx": turnover * model.fx_cost_bps / divisor,
    }
