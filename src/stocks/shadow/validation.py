from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stocks.shadow.errors import (
    CONCENTRATION_LIMIT_BLOCKED,
    INELIGIBLE_INSTRUMENT_BLOCKED,
    INVALID_CASH_WEIGHT,
    NEGATIVE_WEIGHT_BLOCKED,
    TARGET_PORTFOLIO_VALID,
    UNKNOWN_INSTRUMENT_BLOCKED,
    WEIGHTS_NOT_NORMALIZED,
)
from stocks.shadow.models import ShadowTargetPortfolio


@dataclass(frozen=True)
class TargetValidationLimits:
    instrument_cap: Decimal = Decimal("0.60")
    region_cap: Decimal = Decimal("0.80")
    sleeve_cap: Decimal = Decimal("0.80")
    currency_cap: Decimal = Decimal("0.90")
    minimum_cash: Decimal = Decimal("0.00")
    maximum_turnover: Decimal = Decimal("1.00")
    tolerance: Decimal = Decimal("0.000001")


def validate_target_portfolio(
    portfolio: ShadowTargetPortfolio,
    *,
    eligible_con_ids: set[int],
    blocked_con_ids: set[int] | None = None,
    limits: TargetValidationLimits = TargetValidationLimits(),
    turnover: Decimal = Decimal("0"),
) -> dict[str, object]:
    blocked_con_ids = blocked_con_ids or set()
    if portfolio.cash_weight < limits.minimum_cash or portfolio.cash_weight < Decimal("0"):
        return _result(INVALID_CASH_WEIGHT)
    total = portfolio.cash_weight
    by_region: dict[str, Decimal] = {}
    by_sleeve: dict[str, Decimal] = {}
    by_currency: dict[str, Decimal] = {}
    for pos in portfolio.positions:
        weight = Decimal(pos.target_weight)
        if not weight.is_finite():
            return _result(WEIGHTS_NOT_NORMALIZED)
        if weight < 0:
            return _result(NEGATIVE_WEIGHT_BLOCKED)
        if pos.con_id not in eligible_con_ids:
            return _result(UNKNOWN_INSTRUMENT_BLOCKED)
        if pos.con_id in blocked_con_ids:
            return _result(INELIGIBLE_INSTRUMENT_BLOCKED)
        if weight > Decimal("1") or weight > limits.instrument_cap:
            return _result(CONCENTRATION_LIMIT_BLOCKED)
        total += weight
        by_region[pos.region] = by_region.get(pos.region, Decimal("0")) + weight
        by_sleeve[pos.sleeve] = by_sleeve.get(pos.sleeve, Decimal("0")) + weight
        by_currency[pos.currency] = by_currency.get(pos.currency, Decimal("0")) + weight
    if abs(total - Decimal("1")) > limits.tolerance:
        return _result(WEIGHTS_NOT_NORMALIZED)
    if any(value > limits.region_cap for value in by_region.values()):
        return _result(CONCENTRATION_LIMIT_BLOCKED, cap_type="region")
    if any(value > limits.sleeve_cap for value in by_sleeve.values()):
        return _result(CONCENTRATION_LIMIT_BLOCKED, cap_type="sleeve")
    if any(value > limits.currency_cap for value in by_currency.values()):
        return _result(CONCENTRATION_LIMIT_BLOCKED, cap_type="currency")
    if turnover > limits.maximum_turnover:
        return _result(CONCENTRATION_LIMIT_BLOCKED, cap_type="turnover")
    return _result(TARGET_PORTFOLIO_VALID)


def _result(status: str, **extra: object) -> dict[str, object]:
    return {"status": "GO" if status == TARGET_PORTFOLIO_VALID else "NO_GO", "target_status": status, **extra}
