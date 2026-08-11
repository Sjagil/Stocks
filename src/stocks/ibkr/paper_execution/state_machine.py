from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from enum import StrEnum
from typing import Any, Iterable, Mapping

from stocks.execution.idempotency import stable_hash


class OrderLifecycleState(StrEnum):
    NONE = "NONE"
    PREPARED = "PREPARED"
    APPROVED = "APPROVED"
    ORDER_ID_ALLOCATED = "ORDER_ID_ALLOCATED"
    SUBMIT_SENT = "SUBMIT_SENT"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ACK_TIMEOUT_BLOCKED = "ACK_TIMEOUT_BLOCKED"
    ACK_INVALIDATED = "ACK_INVALIDATED"
    CLOSED = "CLOSED"


TERMINAL_STATES = frozenset(
    {
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.REJECTED,
        OrderLifecycleState.ACK_TIMEOUT_BLOCKED,
        OrderLifecycleState.ACK_INVALIDATED,
        OrderLifecycleState.CLOSED,
    }
)


ALLOWED_TRANSITIONS: dict[OrderLifecycleState, frozenset[OrderLifecycleState]] = {
    OrderLifecycleState.NONE: frozenset({OrderLifecycleState.PREPARED}),
    OrderLifecycleState.PREPARED: frozenset({OrderLifecycleState.APPROVED}),
    OrderLifecycleState.APPROVED: frozenset(
        {OrderLifecycleState.ORDER_ID_ALLOCATED}
    ),
    OrderLifecycleState.ORDER_ID_ALLOCATED: frozenset(
        {OrderLifecycleState.SUBMIT_SENT}
    ),
    OrderLifecycleState.SUBMIT_SENT: frozenset(
        {
            OrderLifecycleState.WORKING,
            OrderLifecycleState.PARTIALLY_FILLED,
            OrderLifecycleState.FILLED,
            OrderLifecycleState.CANCEL_REQUESTED,
            OrderLifecycleState.REJECTED,
            OrderLifecycleState.ACK_TIMEOUT_BLOCKED,
            OrderLifecycleState.ACK_INVALIDATED,
        }
    ),
    OrderLifecycleState.WORKING: frozenset(
        {
            OrderLifecycleState.PARTIALLY_FILLED,
            OrderLifecycleState.FILLED,
            OrderLifecycleState.CANCEL_REQUESTED,
            OrderLifecycleState.REJECTED,
            OrderLifecycleState.ACK_INVALIDATED,
        }
    ),
    OrderLifecycleState.PARTIALLY_FILLED: frozenset(
        {
            OrderLifecycleState.PARTIALLY_FILLED,
            OrderLifecycleState.FILLED,
            OrderLifecycleState.CANCEL_REQUESTED,
        }
    ),
    OrderLifecycleState.CANCEL_REQUESTED: frozenset(
        {
            OrderLifecycleState.CANCELLED,
            OrderLifecycleState.PARTIALLY_FILLED,
            OrderLifecycleState.FILLED,
        }
    ),
    OrderLifecycleState.FILLED: frozenset({OrderLifecycleState.CLOSED}),
    OrderLifecycleState.REJECTED: frozenset(
        {OrderLifecycleState.SUBMIT_SENT}
    ),
    OrderLifecycleState.ACK_TIMEOUT_BLOCKED: frozenset(),
    OrderLifecycleState.ACK_INVALIDATED: frozenset(),
    OrderLifecycleState.CANCELLED: frozenset(),
    OrderLifecycleState.CLOSED: frozenset(),
}


EVENT_TARGETS: dict[str, OrderLifecycleState] = {
    "MANUAL_OPERATOR_INTENT": OrderLifecycleState.PREPARED,
    "APPROVAL_RECORDED": OrderLifecycleState.APPROVED,
    "ORDER_ID_ALLOCATED": OrderLifecycleState.ORDER_ID_ALLOCATED,
    "PLACE_ORDER_CALLED_ONCE": OrderLifecycleState.SUBMIT_SENT,
    "BROKER_SUBMISSION_ACKNOWLEDGED": OrderLifecycleState.WORKING,
    "BROKER_SUBMISSION_REJECTED": OrderLifecycleState.REJECTED,
    "BROKER_SUBMISSION_ACK_TIMEOUT": OrderLifecycleState.ACK_TIMEOUT_BLOCKED,
    "BROKER_SUBMISSION_ACK_INVALIDATED": OrderLifecycleState.ACK_INVALIDATED,
    "BROKER_SUBMISSION_WARNING_RECLASSIFIED": OrderLifecycleState.SUBMIT_SENT,
    "CANCEL_ORDER_CALLED_ONCE": OrderLifecycleState.CANCEL_REQUESTED,
    "BROKER_ORDER_CANCELLED": OrderLifecycleState.CANCELLED,
    "CANONICAL_POSITION_EPISODE_CLOSED": OrderLifecycleState.CLOSED,
}


def transition_allowed(
    current: OrderLifecycleState,
    target: OrderLifecycleState,
) -> bool:
    return target == current or target in ALLOWED_TRANSITIONS[current]


def state_machine_schema() -> dict[str, Any]:
    return {
        "schema": "phase9_order_state_machine_v1",
        "status": "GO",
        "states": [state.value for state in OrderLifecycleState],
        "terminal_states": sorted(state.value for state in TERMINAL_STATES),
        "allowed_transitions": {
            state.value: sorted(target.value for target in targets)
            for state, targets in ALLOWED_TRANSITIONS.items()
        },
        "unknown_transition_policy": "UNCLASSIFIED_TRANSITION_BLOCKED",
        "broker_proof_required_for_fill": True,
        "broker_proof_required_for_close": True,
        "negative_position_allowed": False,
        "execution_authority": "NONE",
    }


def audit_order_state_machine(
    intents: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    executions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    intent_map = {
        str(row.get("intent_id")): dict(row)
        for row in intents
        if row.get("intent_id")
    }
    by_intent_events: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_intent_events[str(event.get("aggregate_id"))].append(event)
    filled: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    for row in executions:
        payload = row.get("payload", row)
        intent_id = str(payload.get("intent_id") or row.get("intent_id") or "")
        if intent_id:
            filled[intent_id] += Decimal(str(payload.get("quantity", "0")))

    violations: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    for intent_id in sorted(set(intent_map) | set(by_intent_events)):
        current = OrderLifecycleState.NONE
        transitions: list[dict[str, Any]] = []
        intent = intent_map.get(intent_id, {})
        submitted_quantity = Decimal(str(intent.get("quantity", "0")))
        for event in sorted(
            by_intent_events[intent_id], key=lambda row: int(row.get("event_id", 0))
        ):
            event_type = str(event.get("event_type", ""))
            target = EVENT_TARGETS.get(event_type)
            if event_type == "APPROVAL_RECORDED" and current not in {
                OrderLifecycleState.PREPARED,
                OrderLifecycleState.APPROVED,
            }:
                target = current
            if event_type == "FILL_EXECUTION_ACCEPTED":
                target = (
                    OrderLifecycleState.FILLED
                    if submitted_quantity > 0
                    and filled[intent_id] == submitted_quantity
                    else OrderLifecycleState.PARTIALLY_FILLED
                )
            if target is None:
                continue
            allowed = transition_allowed(current, target)
            transition = {
                "event_id": event.get("event_id"),
                "event_type": event_type,
                "from_state": current.value,
                "to_state": target.value,
                "allowed": allowed,
                "timestamp": event.get("created_at"),
                "cause": event_type,
                "source": _event_source(event_type),
                "correlation_id": _correlation_id(intent_id, event),
            }
            transitions.append(transition)
            if not allowed:
                violations.append(
                    {
                        "intent_id_hash": stable_hash(intent_id),
                        **transition,
                    }
                )
                continue
            current = target
        projections.append(
            {
                "intent_id_hash": stable_hash(intent_id),
                "final_state": current.value,
                "filled_quantity": str(filled[intent_id]),
                "submitted_quantity": str(submitted_quantity),
                "transition_count": len(transitions),
                "terminal": current in TERMINAL_STATES,
            }
        )
    return {
        "schema": "phase9_order_state_machine_audit_v1",
        "status": "GO" if not violations else "NO_GO",
        "intent_count": len(projections),
        "transition_violation_count": len(violations),
        "violations": violations,
        "projections": projections,
        "allowed_transitions": state_machine_schema()["allowed_transitions"],
        "forbidden_transition_policy": "FAIL_CLOSED_AND_RECONCILE",
        "execution_authority": "NONE",
    }


def transition_metadata(
    aggregate_id: str,
    event_type: str,
    payload: Mapping[str, Any],
    created_at: str,
) -> dict[str, str]:
    correlation_id = str(payload.get("correlation_id") or "")
    if not correlation_id:
        correlation_id = stable_hash(
            {
                "aggregate_id": aggregate_id,
                "event_type": event_type,
                "payload": dict(payload),
                "created_at": created_at,
            }
        )[:32]
    return {
        "cause": str(payload.get("cause") or event_type),
        "source": str(payload.get("source") or _event_source(event_type)),
        "correlation_id": correlation_id,
    }


def _event_source(event_type: str) -> str:
    if event_type.startswith("BROKER_") or event_type in {
        "FILL_EXECUTION_ACCEPTED",
        "EXECUTION_RECORDED",
        "COMMISSION_RECORDED",
    }:
        return "IBKR_CALLBACK"
    if event_type in {"PLACE_ORDER_CALLED_ONCE", "CANCEL_ORDER_CALLED_ONCE"}:
        return "PHASE9_BROKER_ADAPTER"
    return "PHASE9_CONTROL_PLANE"


def _correlation_id(
    intent_id: str, event: Mapping[str, Any]
) -> str:
    payload = event.get("payload", {})
    if isinstance(payload, Mapping) and payload.get("transition_meta"):
        meta = payload["transition_meta"]
        if isinstance(meta, Mapping) and meta.get("correlation_id"):
            return str(meta["correlation_id"])
    return stable_hash(
        {
            "intent_id": intent_id,
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
        }
    )[:32]


__all__ = [
    "ALLOWED_TRANSITIONS",
    "OrderLifecycleState",
    "audit_order_state_machine",
    "state_machine_schema",
    "transition_allowed",
    "transition_metadata",
]
