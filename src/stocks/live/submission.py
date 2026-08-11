from __future__ import annotations

from typing import Protocol

from stocks.live.adapter import order_quantity_is_whole
from stocks.live.store import LiveExecutionStore


class BracketOrderSubmitter(Protocol):
    def placeOrder(
        self, order_id: int, contract: object, order: object
    ) -> None: ...


def submit_bracket_once(
    app: BracketOrderSubmitter,
    *,
    order_ids: tuple[int, int, int],
    contract: object,
    orders: tuple[object, object, object],
    store: LiveExecutionStore,
    intent_id: str,
) -> dict[str, object]:
    quantities = [getattr(order, "totalQuantity", None) for order in orders]
    if (
        not all(order_quantity_is_whole(value) for value in quantities)
        or len({str(value) for value in quantities}) != 1
    ):
        return {
            "status": "NO_GO",
            "submission_status": "FRACTIONAL_QUANTITY_FORBIDDEN",
            "blockers": ["FRACTIONAL_QUANTITY_FORBIDDEN"],
            "live_place_order_calls": 0,
        }
    for order_id in order_ids:
        if store.mark_order_id_used(order_id) != "ORDER_ID_READY":
            return {
                "status": "NO_GO",
                "submission_status": "ORDER_ID_ALREADY_USED",
                "live_place_order_calls": 0,
            }
    for order_id, order in zip(order_ids, orders, strict=True):
        app.placeOrder(order_id, contract, order)
    store.append_event(
        intent_id,
        "LIVE_BRACKET_PLACE_ORDER_CALLED_ONCE",
        {
            "order_count": 3,
            "parent_order_id_hash": "ORDER-ID-USED",
        },
    )
    return {
        "status": "GO",
        "submission_status": "LIVE_BRACKET_SUBMITTED_ONCE",
        "live_place_order_calls": 3,
        "economic_order_count": 1,
    }
