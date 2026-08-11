from __future__ import annotations

from typing import Protocol

from stocks.ibkr.paper_execution.storage import PaperExecutionStore


class OrderIdRequester(Protocol):
    def reqIds(self, number_of_ids: int) -> None: ...


def request_broker_order_id(app: OrderIdRequester) -> None:
    app.reqIds(-1)


def allocate_order_id(store: PaperExecutionStore, *, broker_next_id: int, intent_id: str) -> dict[str, object]:
    status, order_id = store.allocate_order_id(broker_next_id, intent_id)
    return {"status": "GO" if status == "ORDER_ID_READY" else "NO_GO", "order_id_status": status, "order_id_hash": None if order_id is None else "ORDER-ID-ALLOCATED"}
