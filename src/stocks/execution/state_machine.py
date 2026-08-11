from __future__ import annotations

from stocks.execution.models import OrderState


ALLOWED_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.CREATED: {OrderState.VALIDATING, OrderState.EXPIRED, OrderState.UNKNOWN_BLOCKED},
    OrderState.VALIDATING: {OrderState.RISK_APPROVED, OrderState.RISK_REJECTED},
    OrderState.RISK_APPROVED: {OrderState.QUEUED},
    OrderState.RISK_REJECTED: {OrderState.INACTIVE},
    OrderState.QUEUED: {OrderState.SUBMITTED_SIMULATED, OrderState.CANCEL_REQUESTED, OrderState.EXPIRED},
    OrderState.SUBMITTED_SIMULATED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.REJECTED, OrderState.EXPIRED, OrderState.CANCEL_REQUESTED},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCEL_REQUESTED, OrderState.REJECTED},
    OrderState.CANCEL_REQUESTED: {OrderState.CANCELLED, OrderState.FILLED},
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
    OrderState.INACTIVE: set(),
    OrderState.UNKNOWN_BLOCKED: set(),
}


def transition_status(current: OrderState, target: OrderState) -> dict[str, object]:
    allowed = target in ALLOWED_TRANSITIONS.get(current, set())
    return {
        "status": "GO" if allowed else "NO_GO",
        "decision_code": "STATE_TRANSITION_ALLOWED" if allowed else "INVALID_STATE_TRANSITION_BLOCKED",
        "from_state": current.value,
        "to_state": target.value,
    }

