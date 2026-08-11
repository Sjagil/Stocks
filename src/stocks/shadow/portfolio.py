from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from stocks.execution.idempotency import stable_hash
from stocks.shadow.costs import ShadowCostModel, estimate_costs
from stocks.shadow.models import ShadowFill, ShadowPosition


@dataclass
class ShadowPortfolioState:
    cash: Decimal = Decimal("100000")
    positions: dict[int, Decimal] = field(default_factory=dict)
    fees: Decimal = Decimal("0")
    nav: Decimal = Decimal("100000")
    booked_fills: set[str] = field(default_factory=set)


def apply_shadow_fill_once(
    state: ShadowPortfolioState,
    fill: ShadowFill,
    *,
    cost_model: ShadowCostModel = ShadowCostModel(),
) -> str:
    if fill.fill_id in state.booked_fills:
        return "DUPLICATE_FILL_BLOCKED"
    if fill.quantity < 0:
        return "NEGATIVE_POSITION_BLOCKED"
    notional = fill.quantity * fill.price
    costs = estimate_costs(notional, cost_model)["total_cost"]
    if state.cash - notional - costs < 0:
        return "NEGATIVE_CASH_BLOCKED"
    state.cash -= notional + costs
    state.positions[fill.con_id] = state.positions.get(fill.con_id, Decimal("0")) + fill.quantity
    state.fees += costs
    state.booked_fills.add(fill.fill_id)
    state.nav = state.cash + sum(qty * fill.price for qty in state.positions.values())
    return "FILL_BOOKED_ONCE"


def reduce_position_once(
    state: ShadowPortfolioState,
    *,
    event_id: str,
    con_id: int,
    quantity: Decimal,
    price: Decimal,
    cost_model: ShadowCostModel = ShadowCostModel(),
) -> str:
    if event_id in state.booked_fills:
        return "DUPLICATE_FILL_BLOCKED"
    current = state.positions.get(con_id, Decimal("0"))
    if quantity <= 0 or quantity > current:
        return "NEGATIVE_POSITION_BLOCKED"
    proceeds = quantity * price
    costs = estimate_costs(proceeds, cost_model)["total_cost"]
    remaining = current - quantity
    if remaining == 0:
        state.positions.pop(con_id, None)
        status = "POSITION_CLOSED"
    else:
        state.positions[con_id] = remaining
        status = "POSITION_REDUCED"
    state.cash += proceeds - costs
    state.fees += costs
    state.booked_fills.add(event_id)
    state.nav = state.cash + sum(qty * price for qty in state.positions.values())
    return status


def book_dividend_once(state: ShadowPortfolioState, *, event_id: str, amount: Decimal) -> str:
    if event_id in state.booked_fills:
        return "DUPLICATE_DIVIDEND_BLOCKED"
    if amount < 0:
        return "NEGATIVE_DIVIDEND_BLOCKED"
    state.cash += amount
    state.nav += amount
    state.booked_fills.add(event_id)
    return "DIVIDEND_BOOKED_ONCE"


def apply_fx_normalization_once(state: ShadowPortfolioState, *, event_id: str, fx_return: Decimal) -> str:
    if event_id in state.booked_fills:
        return "DUPLICATE_FX_NORMALIZATION_BLOCKED"
    if fx_return <= Decimal("-1"):
        return "FX_NORMALIZATION_BLOCKED"
    multiplier = Decimal("1") + fx_return
    state.cash *= multiplier
    state.nav *= multiplier
    state.booked_fills.add(event_id)
    return "FX_NORMALIZATION_BOOKED_ONCE"


def rebalance_cash_residual(state: ShadowPortfolioState, *, target_cash: Decimal) -> str:
    if target_cash < 0 or target_cash > state.nav:
        return "REBALANCE_BLOCKED"
    state.cash = target_cash
    return "REBALANCE_RECORDED"


def portfolio_invariants(state: ShadowPortfolioState) -> dict[str, object]:
    ok = (
        state.cash >= 0
        and all(qty >= 0 for qty in state.positions.values())
        and state.nav >= 0
        and state.fees >= 0
    )
    return {
        "status": "GO" if ok else "NO_GO",
        "cash_non_negative": state.cash >= 0,
        "no_negative_long_position": all(qty >= 0 for qty in state.positions.values()),
        "nav_non_negative": state.nav >= 0,
        "costs_booked": state.fees >= 0,
    }


def state_hash(state: ShadowPortfolioState) -> str:
    return stable_hash(
        {
            "cash": str(state.cash),
            "positions": {str(key): str(value) for key, value in sorted(state.positions.items())},
            "fees": str(state.fees),
            "nav": str(state.nav),
            "booked_fills": sorted(state.booked_fills),
        }
    )


def snapshot_positions(state: ShadowPortfolioState, price: Decimal) -> tuple[ShadowPosition, ...]:
    return tuple(
        ShadowPosition(con_id=con_id, quantity=qty, market_value=qty * price, weight=(qty * price / state.nav if state.nav else Decimal("0")))
        for con_id, qty in sorted(state.positions.items())
    )
