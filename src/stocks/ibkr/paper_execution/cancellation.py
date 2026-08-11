from __future__ import annotations

from typing import Any

from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def cancel_known_order_once(app: Any, *, order_id: int, writer_client_matches: bool, approved: bool, store: PaperExecutionStore, intent_id: str) -> dict[str, object]:
    if not writer_client_matches:
        return {"status": "NO_GO", "cancel_status": "CANCEL_CLIENT_ID_MISMATCH_BLOCKED", "paper_cancel_order_calls": 0}
    if not approved:
        return {"status": "NO_GO", "cancel_status": "CANCEL_APPROVAL_REQUIRED", "paper_cancel_order_calls": 0}
    try:
        from ibapi.order_cancel import OrderCancel
    except Exception:
        app.cancelOrder(order_id)
    else:
        try:
            app.cancelOrder(order_id, OrderCancel())
        except TypeError:
            app.cancelOrder(order_id)
    store.append_event(intent_id, "CANCEL_ORDER_CALLED_ONCE", {"order_id_hash": "ORDER-ID-CANCELLED"})
    return {"status": "GO", "cancel_status": "CANCEL_REQUESTED", "paper_cancel_order_calls": 1}


def record_broker_cancel_confirmation(
    store: PaperExecutionStore,
    *,
    intent_id: str,
    broker_proof: bool,
) -> dict[str, object]:
    if not broker_proof:
        return {
            "status": "NO_GO",
            "cancel_status": "BROKER_CANCEL_PROOF_REQUIRED",
        }
    events = [
        event
        for event in store.list_events()
        if event["aggregate_id"] == intent_id
    ]
    if not any(
        event["event_type"] == "CANCEL_ORDER_CALLED_ONCE"
        for event in events
    ):
        return {"status": "NO_GO", "cancel_status": "CANCEL_NOT_REQUESTED"}
    if any(
        event["event_type"] == "BROKER_ORDER_CANCELLED" for event in events
    ):
        return {"status": "GO", "cancel_status": "CANCEL_CONFIRMED_IDEMPOTENT"}
    store.append_event(
        intent_id,
        "BROKER_ORDER_CANCELLED",
        {"broker_proof": True},
    )
    has_fill = any(
        str(row["payload"].get("intent_id", "")) == intent_id
        for row in store.list_executions()
    )
    capital_status = store.release_capital_once(
        intent_id,
        reason=(
            "BROKER_CANCEL_CONFIRMED_AFTER_PARTIAL_FILL"
            if has_fill
            else "BROKER_CANCEL_CONFIRMED"
        ),
        allow_deployed=False,
    )
    return {
        "status": "GO",
        "cancel_status": "BROKER_CANCEL_CONFIRMED",
        "capital_status": capital_status,
    }


def block_global_cancel() -> dict[str, object]:
    return {"status": "NO_GO", "cancel_status": "GLOBAL_CANCEL_BLOCKED", "global_cancel_calls": 0}
