from __future__ import annotations

from decimal import Decimal

from stocks.auto_paper.contracts import Regime


SLEEVES = (
    "CORE_SHARIAH_EQUITIES",
    "EMERGING_GEMS",
    "EVENT_DRIVEN_MOVERS",
    "SHARIAH_EQUITY_ETFS",
    "APPROVED_PHYSICAL_COMMODITY",
    "OPERATIONAL_CASH",
)
BLOCKED_SLEEVES = ("BONDS", "CONVENTIONAL_FIXED_INCOME", "FUTURES", "OPTIONS")


def regime_allocation(regime: Regime) -> dict[str, Decimal]:
    allocations = {
        Regime.RISK_ON: ("0.55", "0.15", "0.10", "0.10", "0.05", "0.05"),
        Regime.INFLATION_SUPPLY_SHOCK: ("0.35", "0.08", "0.07", "0.10", "0.35", "0.05"),
        Regime.GEOPOLITICAL_SUPPLY_SHOCK: ("0.32", "0.08", "0.05", "0.10", "0.40", "0.05"),
        Regime.RECESSION_DEMAND_SHOCK: ("0.45", "0.05", "0.05", "0.20", "0.15", "0.10"),
        Regime.LIQUIDITY_STRESS: ("0.40", "0.03", "0.02", "0.20", "0.25", "0.10"),
    }
    return {sleeve: Decimal(weight) for sleeve, weight in zip(SLEEVES, allocations[regime], strict=True)}


def validate_portfolio_limits(
    *,
    position_weight_pct: Decimal,
    sector_exposure_pct: Decimal,
    event_cluster_exposure_pct: Decimal,
    fallen_angel_combined_pct: Decimal,
    cash_pct: Decimal,
    shariah_eligible: bool,
    high_conviction: bool = False,
    starter_position: bool = False,
    emergency_cash: bool = False,
) -> dict[str, object]:
    max_position = Decimal("10") if high_conviction else Decimal("8")
    blockers = []
    if not shariah_eligible:
        blockers.append("SHARIAH_PORTFOLIO_GATE_BLOCKED")
    if starter_position and not Decimal("2") <= position_weight_pct <= Decimal("3"):
        blockers.append("STARTER_POSITION_RANGE_BLOCKED")
    if position_weight_pct > max_position:
        blockers.append("SINGLE_POSITION_LIMIT_REACHED")
    if sector_exposure_pct > Decimal("25"):
        blockers.append("SECTOR_EXPOSURE_REACHED")
    if event_cluster_exposure_pct > Decimal("15"):
        blockers.append("EVENT_CLUSTER_LIMIT_REACHED")
    if fallen_angel_combined_pct > Decimal("10"):
        blockers.append("FALLEN_ANGEL_LIMIT_REACHED")
    if emergency_cash and cash_pct > Decimal("10"):
        blockers.append("EMERGENCY_CASH_MAXIMUM_REACHED")
    if not emergency_cash and not Decimal("2.5") <= cash_pct <= Decimal("5"):
        blockers.append("OPERATIONAL_CASH_RANGE_BLOCKED")
    return {"status": "PORTFOLIO_RISK_GO" if not blockers else "PORTFOLIO_RISK_BLOCKED", "blockers": blockers}


def risk_off_rotation_order() -> tuple[str, ...]:
    return (
        "APPROVED_PHYSICAL_GOLD",
        "SHARIAH_DEFENSIVE_EQUITIES",
        "SHARIAH_ENERGY_EQUITIES",
        "SHARIAH_MATERIALS_EQUITIES",
        "APPROVED_BROAD_SHARIAH_ETF",
        "LIMITED_OPERATIONAL_CASH",
    )
