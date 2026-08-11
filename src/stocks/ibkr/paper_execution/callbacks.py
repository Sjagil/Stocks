from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.state_mapping import map_order_status


CALLBACK_TYPES = (
    "openOrder",
    "openOrderEnd",
    "orderStatus",
    "execDetails",
    "execDetailsEnd",
    "commissionReport",
    "error",
    "connectionClosed",
)


def empty_callback_counts() -> dict[str, int]:
    return {
        "received": 0,
        "accepted": 0,
        "duplicate": 0,
        "late": 0,
        "out_of_order": 0,
        "orphan": 0,
        "quarantined": 0,
    }


@dataclass
class CallbackAuditState:
    seen: set[str] = field(default_factory=set)
    received: int = 0
    accepted: int = 0
    duplicate: int = 0
    late: int = 0
    out_of_order: int = 0
    orphan: int = 0
    quarantined: int = 0
    by_type: dict[str, dict[str, int]] = field(
        default_factory=lambda: {callback_type: empty_callback_counts() for callback_type in CALLBACK_TYPES},
    )


@dataclass
class OrderStatusProjection:
    original_quantity: Decimal
    status: str = "NONE"
    filled_quantity: Decimal = Decimal("0")
    remaining_quantity: Decimal | None = None
    average_fill_price: Decimal = Decimal("0")
    terminal: bool = False


TERMINAL_IBKR_STATUSES = {"Cancelled", "ApiCancelled", "Filled", "Inactive"}
IBKR_STATUS_PRECEDENCE = {
    "PendingSubmit": 1,
    "ApiPending": 1,
    "PreSubmitted": 2,
    "Submitted": 3,
    "PendingCancel": 4,
    "Cancelled": 5,
    "ApiCancelled": 5,
    "Inactive": 5,
    "Filled": 6,
}


def apply_order_status(
    projection: OrderStatusProjection,
    *,
    status: str,
    filled: Decimal | str,
    remaining: Decimal | str,
    average_fill_price: Decimal | str = Decimal("0"),
) -> dict[str, object]:
    """Apply an IBKR orderStatus without regressing economic state."""
    try:
        filled_value = Decimal(str(filled))
        remaining_value = Decimal(str(remaining))
        average_value = Decimal(str(average_fill_price))
    except (InvalidOperation, ValueError):
        return {"status": "NO_GO", "classification": "INVALID_ORDER_STATUS_VALUE"}
    if any(
        not value.is_finite() or value < 0
        for value in (filled_value, remaining_value, average_value)
    ):
        return {"status": "NO_GO", "classification": "INVALID_ORDER_STATUS_VALUE"}
    if filled_value > projection.original_quantity:
        return {"status": "NO_GO", "classification": "OVERFILL_BLOCKED"}
    if filled_value + remaining_value > projection.original_quantity:
        return {
            "status": "NO_GO",
            "classification": "FILLED_PLUS_REMAINING_EXCEEDS_ORIGINAL",
        }
    same = (
        status == projection.status
        and filled_value == projection.filled_quantity
        and remaining_value == projection.remaining_quantity
        and average_value == projection.average_fill_price
    )
    if same:
        return {"status": "GO", "classification": "ORDER_STATUS_IDEMPOTENT"}
    new_fill = filled_value > projection.filled_quantity
    if filled_value < projection.filled_quantity:
        return {"status": "NO_GO", "classification": "FILLED_QUANTITY_REGRESSION_BLOCKED"}
    prior_rank = IBKR_STATUS_PRECEDENCE.get(projection.status, 0)
    next_rank = IBKR_STATUS_PRECEDENCE.get(status)
    if next_rank is None:
        return {"status": "NO_GO", "classification": "UNKNOWN_ORDER_STATUS_BLOCKED"}
    if projection.terminal and not new_fill:
        return {"status": "GO", "classification": "STALE_TERMINAL_REGRESSION_IGNORED"}
    if next_rank < prior_rank and not new_fill:
        return {"status": "GO", "classification": "STALE_ORDER_STATUS_IGNORED"}
    projection.status = status
    projection.filled_quantity = filled_value
    projection.remaining_quantity = remaining_value
    projection.average_fill_price = average_value
    projection.terminal = status in TERMINAL_IBKR_STATUSES
    return {
        "status": "GO",
        "classification": (
            "TERMINAL_STATE_CORRECTED_BY_NEW_FILL"
            if new_fill and prior_rank >= 5
            else "ORDER_STATUS_APPLIED"
        ),
        "internal_state": map_order_status(status),
    }


def accept_callback(state: CallbackAuditState, callback_type: str, payload: dict[str, object], *, known_order: bool = True, late: bool = False, out_of_order: bool = False) -> dict[str, object]:
    state.received += 1
    counts = state.by_type.setdefault(callback_type, empty_callback_counts())
    counts["received"] += 1
    key = stable_hash({"type": callback_type, "payload": payload})
    if not known_order:
        state.orphan += 1
        state.quarantined += 1
        counts["orphan"] += 1
        counts["quarantined"] += 1
        return {"classification": "ORPHAN_CALLBACK_BLOCKED", "phase9_0_1_classification": "CALLBACK_ORPHAN_QUARANTINED"}
    if late:
        state.late += 1
        state.quarantined += 1
        counts["late"] += 1
        counts["quarantined"] += 1
        return {"classification": "LATE_CALLBACK_QUARANTINED", "phase9_0_1_classification": "CALLBACK_LATE_ACCEPTED"}
    if out_of_order:
        state.out_of_order += 1
        counts["out_of_order"] += 1
        return {"classification": "OUT_OF_ORDER_CALLBACK_BUFFERED", "phase9_0_1_classification": "CALLBACK_OUT_OF_ORDER_BUFFERED"}
    if key in state.seen:
        state.duplicate += 1
        counts["duplicate"] += 1
        return {"classification": "DUPLICATE_CALLBACK_IGNORED", "phase9_0_1_classification": "CALLBACK_DUPLICATE_IGNORED"}
    state.seen.add(key)
    state.accepted += 1
    counts["accepted"] += 1
    return {
        "classification": "CALLBACK_OK",
        "phase9_0_1_classification": "CALLBACK_ACCEPTED",
        "internal_state": map_order_status(str(payload.get("status", ""))),
    }


def callback_audit_payload(state: CallbackAuditState) -> dict[str, Any]:
    return {
        "status": "GO",
        "received": state.received,
        "accepted": state.accepted,
        "duplicate": state.duplicate,
        "late": state.late,
        "out_of_order": state.out_of_order,
        "orphan": state.orphan,
        "quarantined": state.quarantined,
        "phase9_0_1_supported_classifications": [
            "CALLBACK_ACCEPTED",
            "CALLBACK_DUPLICATE_IGNORED",
            "CALLBACK_OUT_OF_ORDER_BUFFERED",
            "CALLBACK_LATE_ACCEPTED",
            "CALLBACK_ORPHAN_QUARANTINED",
            "CALLBACK_CONFLICT_BLOCKED",
        ],
        "callback_types": state.by_type,
    }
