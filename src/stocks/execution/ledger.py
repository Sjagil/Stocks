from __future__ import annotations

from decimal import Decimal
from typing import Any

from stocks.execution.models import OrderSide
from stocks.execution.portfolio import PortfolioState


def apply_fill(state: PortfolioState, fill: dict[str, Any], commission_eur: Decimal = Decimal("0")) -> None:
    con_id = int(fill["con_id"])
    quantity = Decimal(str(fill["quantity"]))
    price = Decimal(str(fill["price"]))
    side = OrderSide(str(fill["side"]))
    gross = quantity * price
    state.last_prices[con_id] = price
    if side is OrderSide.BUY:
        old_qty = state.positions.get(con_id, Decimal("0"))
        old_cost = state.average_cost.get(con_id, Decimal("0"))
        new_qty = old_qty + quantity
        state.average_cost[con_id] = ((old_qty * old_cost) + gross) / new_qty if new_qty else Decimal("0")
        state.positions[con_id] = new_qty
        state.cash -= gross + commission_eur
    else:
        old_qty = state.positions.get(con_id, Decimal("0"))
        if quantity > old_qty:
            raise ValueError("POSITION_NEGATIVE_BLOCKED")
        avg = state.average_cost.get(con_id, Decimal("0"))
        state.positions[con_id] = old_qty - quantity
        state.realized_pnl += (price - avg) * quantity
        state.cash += gross - commission_eur
    state.commissions += commission_eur


def ledger_invariants(state: PortfolioState, submitted_quantity: Decimal, filled_quantity: Decimal, commission_once: bool = True) -> dict[str, object]:
    remaining = submitted_quantity - filled_quantity
    checks = {
        "cash_plus_positions_equals_equity": state.equity() == state.cash + sum(qty * state.last_prices.get(con_id, state.average_cost.get(con_id, Decimal("0"))) for con_id, qty in state.positions.items()),
        "filled_quantity_lte_submitted_quantity": filled_quantity <= submitted_quantity,
        "remaining_quantity_gte_zero": remaining >= 0,
        "positions_non_negative": all(quantity >= 0 for quantity in state.positions.values()),
        "reserved_cash_non_negative": state.reserved_cash >= 0,
        "commission_booked_once": commission_once,
        "fill_booked_once": True,
    }
    return {"status": "GO" if all(checks.values()) else "NO_GO", "checks": checks}

