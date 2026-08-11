from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def submit_place_order_once(
    app: Any,
    *,
    order_id: int,
    contract: object,
    order: object,
    store: PaperExecutionStore,
    intent_id: str,
    ack_timeout_seconds: float = 0.0,
    ack_settle_seconds: float = 0.0,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    id_status = store.mark_order_id_used(order_id)
    if id_status != "ORDER_ID_READY":
        return {"status": "NO_GO", "submission_status": id_status, "paper_place_order_calls": 0}
    state = getattr(app, "callback_state", None)
    event_offset = 0 if state is None else _state_length(state, "events")
    error_offset = 0 if state is None else _state_length(state, "errors")
    app.placeOrder(order_id, contract, order)
    store.append_event(intent_id, "PLACE_ORDER_CALLED_ONCE", {"order_id_hash": "ORDER-ID-USED"})
    if state is None or ack_timeout_seconds <= 0:
        return {
            "status": "GO",
            "submission_status": "SUBMISSION_ALLOWED_ONCE",
            "paper_place_order_calls": 1,
        }

    deadline = monotonic() + ack_timeout_seconds
    acknowledged_at: float | None = None
    acknowledged_status: str | None = None
    warning_codes: set[str] = set()
    while monotonic() < deadline:
        events = _state_items(state, "events", event_offset)
        statuses = {
            str(event.get("status", "")).strip()
            for event in events
            if event.get("event")
            in {"phase9_open_order", "phase9_order_status"}
        }
        accepted_statuses = statuses & {
            "Filled",
            "PreSubmitted",
            "Submitted",
        }
        errors = _state_items(state, "errors", error_offset)
        hard_errors = [
            error
            for error in errors
            if not (
                str(error.get("code", "")) == "399"
                and accepted_statuses
            )
        ]
        warning_codes.update(
            str(error.get("code", "UNKNOWN"))
            for error in errors
            if str(error.get("code", "")) == "399"
        )
        if hard_errors:
            codes = sorted(
                {
                    str(error.get("code", "UNKNOWN"))
                    for error in hard_errors
                }
            )
            store.append_event(
                intent_id,
                "BROKER_SUBMISSION_REJECTED",
                {"error_codes": codes},
            )
            store.release_capital_once(
                intent_id, reason="BROKER_SUBMISSION_REJECTED"
            )
            return {
                "status": "NO_GO",
                "submission_status": "BROKER_SUBMISSION_REJECTED",
                "broker_acknowledged": False,
                "broker_error_codes": codes,
                "paper_place_order_calls": 1,
            }
        rejected_statuses = statuses & {
            "ApiCancelled",
            "Cancelled",
            "Inactive",
        }
        if rejected_statuses:
            broker_status = sorted(rejected_statuses)[0]
            store.append_event(
                intent_id,
                "BROKER_SUBMISSION_REJECTED",
                {"broker_status": broker_status},
            )
            store.release_capital_once(
                intent_id, reason="BROKER_SUBMISSION_REJECTED"
            )
            return {
                "status": "NO_GO",
                "submission_status": "BROKER_SUBMISSION_REJECTED",
                "broker_acknowledged": False,
                "broker_status": broker_status,
                "paper_place_order_calls": 1,
            }
        if accepted_statuses or any(
            event.get("event") == "phase9_exec_details"
            for event in events
        ):
            acknowledged_status = (
                sorted(accepted_statuses)[0]
                if accepted_statuses
                else "Filled"
            )
            if acknowledged_at is None:
                acknowledged_at = monotonic()
        if (
            acknowledged_at is not None
            and monotonic() - acknowledged_at >= ack_settle_seconds
        ):
            store.append_event(
                intent_id,
                "BROKER_SUBMISSION_ACKNOWLEDGED",
                {
                    "ack_settle_seconds": ack_settle_seconds,
                    "broker_status": acknowledged_status,
                    "order_id_hash": "ORDER-ID-ACKNOWLEDGED",
                },
            )
            return {
                "status": "GO",
                "submission_status": "BROKER_SUBMISSION_ACKNOWLEDGED",
                "broker_acknowledged": True,
                "broker_status": acknowledged_status,
                "broker_warning_codes": sorted(warning_codes),
                "paper_place_order_calls": 1,
            }
        sleeper(min(0.05, max(0.0, deadline - monotonic())))

    timeout_event = (
        "BROKER_SUBMISSION_REJECTED"
        if warning_codes
        else "BROKER_SUBMISSION_ACK_TIMEOUT"
    )
    store.append_event(
        intent_id,
        timeout_event,
        {
            "error_codes": sorted(warning_codes),
            "timeout_seconds": ack_timeout_seconds,
        },
    )
    return {
        "status": "NO_GO",
        "submission_status": timeout_event,
        "broker_acknowledged": False,
        "broker_error_codes": sorted(warning_codes),
        "paper_place_order_calls": 1,
    }


def _state_length(state: object, name: str) -> int:
    with getattr(state, "lock"):
        return len(getattr(state, name, []))


def _state_items(
    state: object,
    name: str,
    offset: int,
) -> list[dict[str, Any]]:
    with getattr(state, "lock"):
        return list(getattr(state, name, [])[offset:])
