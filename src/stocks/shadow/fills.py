from __future__ import annotations

from decimal import Decimal

from stocks.execution.idempotency import stable_hash
from stocks.shadow.models import ShadowFill


EXECUTION_PROXIES = ("NEXT_OPEN", "NEXT_CLOSE", "VWAP_PROXY", "FIXED_SLIPPAGE")


def deterministic_fill_id(payload: dict[str, object]) -> str:
    return f"SHADOW-FILL-{stable_hash(payload)[:24]}"


def make_hypothetical_fill(
    *,
    decision_id: str,
    con_id: int,
    quantity: Decimal,
    price: Decimal,
    price_timestamp: str,
    execution_proxy: str = "NEXT_OPEN",
    partial: bool = False,
) -> ShadowFill:
    if execution_proxy not in EXECUTION_PROXIES:
        raise ValueError("invalid execution proxy")
    fill_qty = quantity / Decimal("2") if partial else quantity
    core = {
        "decision_id": decision_id,
        "con_id": con_id,
        "quantity": str(fill_qty),
        "price": str(price),
        "execution_proxy": execution_proxy,
        "price_timestamp": price_timestamp,
    }
    return ShadowFill(
        fill_id=deterministic_fill_id(core),
        decision_id=decision_id,
        con_id=con_id,
        quantity=fill_qty,
        price=price,
        execution_proxy=execution_proxy,
        price_source="FROZEN_FIXTURE_TOTAL_RETURN",
        price_timestamp=price_timestamp,
        slippage_bps=Decimal("1"),
        commission_bps=Decimal("1"),
        spread_bps=Decimal("2"),
        fill_status="HYPOTHETICAL_FILL",
    )
