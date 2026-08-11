from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ShadowCostModel:
    commission_bps: Decimal = Decimal("1")
    spread_bps: Decimal = Decimal("2")
    market_impact_bps: Decimal = Decimal("1")
    fx_conversion_bps: Decimal = Decimal("0")
    minimum_fee: Decimal = Decimal("0")


def estimate_costs(notional: Decimal, model: ShadowCostModel = ShadowCostModel()) -> dict[str, Decimal]:
    commission = max(notional * model.commission_bps / Decimal("10000"), model.minimum_fee)
    half_spread = notional * (model.spread_bps / Decimal("2")) / Decimal("10000")
    impact = notional * model.market_impact_bps / Decimal("10000")
    fx = notional * model.fx_conversion_bps / Decimal("10000")
    total = commission + half_spread + impact + fx
    return {
        "commission_cost": commission,
        "spread_cost": half_spread,
        "impact_cost": impact,
        "fx_cost": fx,
        "total_cost": total,
    }


def return_after_costs(gross_shadow_return: Decimal, turnover: Decimal, model: ShadowCostModel = ShadowCostModel()) -> dict[str, Decimal]:
    notional = turnover
    costs = estimate_costs(notional, model)
    net = gross_shadow_return - costs["total_cost"]
    return {"gross_shadow_return": gross_shadow_return, **costs, "net_shadow_return": net}
