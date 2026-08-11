from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stocks.execution.idempotency import stable_hash
from stocks.execution.models import FillEvent, OrderSide


@dataclass(frozen=True)
class FakeBrokerAdapter:
    seed: int = 4638

    def simulated_order_id(self, intent_id: str) -> str:
        return "SIM-" + stable_hash({"intent_id": intent_id, "seed": self.seed})[:16]

    def fills_for(self, scenario_id: str, *, intent_id: str, con_id: int, side: OrderSide, quantity: Decimal, price: Decimal) -> list[FillEvent]:
        broker_order_id = self.simulated_order_id(intent_id)
        if scenario_id in {"BROKER_REJECTION", "STALE_SIGNAL", "CLOSED_SESSION", "UNKNOWN_BROKER_ORDER"}:
            return []
        if scenario_id in {"PARTIAL_FILL_THEN_COMPLETE", "RESTART_AFTER_PARTIAL_FILL", "OUT_OF_ORDER_EVENT"}:
            parts = [quantity / Decimal("2"), quantity / Decimal("2")]
        elif scenario_id in {"PARTIAL_FILL_THEN_CANCEL", "DUPLICATE_FILL_EVENT"}:
            parts = [quantity / Decimal("2")]
        else:
            parts = [quantity]
        fills = []
        for index, part in enumerate(parts, start=1):
            fill_price = price + (Decimal("0.01") if scenario_id == "price_slippage" else Decimal("0"))
            fills.append(
                FillEvent(
                    fill_id=f"{broker_order_id}-F{index}",
                    broker_order_id=broker_order_id,
                    intent_id=intent_id,
                    con_id=con_id,
                    side=side,
                    quantity=part,
                    price=fill_price,
                    created_at=f"2026-07-21T10:0{index}:00Z",
                )
            )
        if scenario_id == "DUPLICATE_FILL_EVENT":
            fills.append(fills[0])
        if scenario_id == "OUT_OF_ORDER_EVENT":
            fills = list(reversed(fills))
        return fills

