from __future__ import annotations


IBKR_STATUS_TO_INTERNAL = {
    "PendingSubmit": "QUEUED",
    "ApiPending": "QUEUED",
    "PreSubmitted": "SUBMITTED_SIMULATED",
    "Submitted": "SUBMITTED_SIMULATED",
    "PendingCancel": "CANCEL_REQUESTED",
    "ApiCancelled": "CANCELLED",
    "Cancelled": "CANCELLED",
    "Filled": "FILLED",
    "Inactive": "INACTIVE",
}


def map_order_status(status: str) -> str:
    return IBKR_STATUS_TO_INTERNAL.get(status, "UNKNOWN_BLOCKED")
