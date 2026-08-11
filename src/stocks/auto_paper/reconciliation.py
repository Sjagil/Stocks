from __future__ import annotations

from decimal import Decimal


def reconcile_shadow_state(
    *,
    local_quantity: Decimal,
    broker_quantity: Decimal,
    unknown_order_count: int = 0,
    unknown_execution_count: int = 0,
    commission_pending_count: int = 0,
    snapshot_complete: bool = True,
) -> dict[str, object]:
    blockers = []
    if not snapshot_complete or local_quantity != broker_quantity:
        blockers.append("BROKER_RECONCILIATION_MISMATCH")
    if unknown_order_count:
        blockers.append("UNKNOWN_BROKER_ORDER")
    if unknown_execution_count:
        blockers.append("UNKNOWN_EXECUTION")
    if commission_pending_count:
        blockers.append("COMMISSION_SCOPE_INCOMPLETE")
    return {
        "status": "SHADOW_RECONCILED" if not blockers else "SHADOW_RECONCILIATION_BLOCKED",
        "blockers": blockers,
        "new_entries_allowed": not blockers,
        "automatic_corrections": 0,
        "manual_review_required": bool(blockers),
    }
