from __future__ import annotations

from decimal import Decimal
from typing import Any

from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def record_commission(store: PaperExecutionStore, *, commission_identity: str, exec_identity: str, amount: Decimal) -> dict[str, object]:
    status = store.append_commission_once(commission_identity, exec_identity, {"amount_hash": "COMMISSION-AMOUNT-PRIVATE", "amount": str(amount)})
    return {"status": "GO", "commission_status": status}


def record_execution_commission(
    store: PaperExecutionStore,
    *,
    execution_id: str,
    commission: Decimal,
    currency: str = "USD",
    correction: bool = False,
    allow_pending: bool = True,
) -> dict[str, object]:
    if not execution_id:
        return {"status": "NO_GO", "commission_status": "COMMISSION_ORPHAN_QUARANTINED"}
    execution_exists = execution_id in {str(row["exec_identity"]) for row in store.list_executions()}
    if not execution_exists and not allow_pending:
        store.append_event(execution_id, "COMMISSION_ORPHAN_QUARANTINED", {"execution_hash": execution_id, "currency": currency})
        return {"status": "NO_GO", "commission_status": "COMMISSION_ORPHAN_QUARANTINED"}
    identity = f"{execution_id}:COMMISSION"
    payload = {
        "execution_id_hash": execution_id,
        "amount": str(commission),
        "currency": currency,
        "correction": correction,
    }
    status = store.append_commission_once(identity, execution_id, payload)
    mapped = {
        "COMMISSION_RECORDED": "COMMISSION_JOINED" if execution_exists else "COMMISSION_PENDING",
        "DUPLICATE_COMMISSION_IGNORED": "COMMISSION_DUPLICATE_IGNORED",
        "COMMISSION_CONFLICT_BLOCKED": "COMMISSION_CONFLICT_BLOCKED",
    }.get(status, status)
    if mapped in {"COMMISSION_JOINED", "COMMISSION_PENDING"}:
        store.append_event(execution_id, mapped, {"execution_hash": execution_id, "currency": currency})
    return {"status": "GO" if mapped != "COMMISSION_CONFLICT_BLOCKED" else "NO_GO", "commission_status": mapped}


def commission_join_audit(store: PaperExecutionStore, *, grace_expired: bool = False) -> dict[str, Any]:
    executions = {str(row["exec_identity"]) for row in store.list_executions()}
    commissions = store.list_commissions()
    joined = 0
    orphan = 0
    for row in commissions:
        if str(row["payload"].get("exec_identity", "")) in executions:
            joined += 1
        else:
            orphan += 1
    pending = len(executions) - joined
    return {
        "status": "GO" if orphan == 0 else "NO_GO",
        "joined_count": joined,
        "pending_count": max(0, pending),
        "orphan_count": orphan,
        "grace_status": "COMMISSION_GRACE_EXPIRED" if grace_expired and pending > 0 else "COMMISSION_PENDING" if pending > 0 else "COMMISSION_JOINED",
        "supported_statuses": [
            "COMMISSION_JOINED",
            "COMMISSION_PENDING",
            "COMMISSION_DUPLICATE_IGNORED",
            "COMMISSION_CONFLICT_BLOCKED",
            "COMMISSION_ORPHAN_QUARANTINED",
            "COMMISSION_GRACE_EXPIRED",
        ],
    }
