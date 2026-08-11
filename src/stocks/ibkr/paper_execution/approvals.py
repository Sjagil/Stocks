from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.errors import (
    APPROVAL_EXPIRED,
    APPROVAL_MISMATCH,
    APPROVED_FOR_SINGLE_SUBMISSION,
    INTENT_CHANGED_AFTER_APPROVAL_BLOCKED,
)
from stocks.ibkr.paper_execution.models import ApprovalRecord, ManualPaperIntent, model_to_jsonable
from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def intent_hash(intent: ManualPaperIntent) -> str:
    return stable_hash(model_to_jsonable(intent))


def approval_challenge(intent: ManualPaperIntent) -> str:
    price = _fmt_decimal(intent.limit_price)
    qty = _fmt_decimal(intent.quantity)
    return f"APPROVE PAPER CANARY {intent.intent_id} {qty} {intent.symbol} {price}"


def cancel_challenge(intent: ManualPaperIntent) -> str:
    return f"CANCEL PAPER CANARY {intent.intent_id} {intent.symbol}"


def make_approval(intent: ManualPaperIntent, *, ttl_seconds: int, approval_type: str = "SUBMIT") -> ApprovalRecord:
    challenge = cancel_challenge(intent) if approval_type == "CANCEL" else approval_challenge(intent)
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
    approval_identity: dict[str, object] = {
        "intent_id": intent.intent_id,
        "intent_hash": intent_hash(intent),
        "approval_type": approval_type,
    }
    if approval_type == "CANCEL":
        approval_identity["expires_at"] = expires_at
    approval_id = stable_hash(approval_identity)[:24]
    return ApprovalRecord(
        approval_id=approval_id,
        intent_id=intent.intent_id,
        intent_hash=intent_hash(intent),
        challenge_hash=stable_hash(challenge),
        expires_at=expires_at,
        used=False,
        approval_type=approval_type,
    )


def approve_intent(store: PaperExecutionStore, intent: ManualPaperIntent, approval_text: str, *, ttl_seconds: int, approval_type: str = "SUBMIT") -> dict[str, object]:
    record = make_approval(intent, ttl_seconds=ttl_seconds, approval_type=approval_type)
    expected = cancel_challenge(intent) if approval_type == "CANCEL" else approval_challenge(intent)
    if approval_text != expected:
        return {"status": "NO_GO", "approval_status": APPROVAL_MISMATCH, "challenge": expected}
    if datetime.fromisoformat(record.expires_at) < datetime.now(timezone.utc):
        return {"status": "NO_GO", "approval_status": APPROVAL_EXPIRED}
    payload = model_to_jsonable(record)
    write_status = store.append_approval(payload)
    return {
        "status": "GO" if write_status in {"APPROVAL_RECORDED", "APPROVAL_IDEMPOTENT"} else "NO_GO",
        "approval_status": APPROVED_FOR_SINGLE_SUBMISSION if write_status in {"APPROVAL_RECORDED", "APPROVAL_IDEMPOTENT"} else write_status,
        "approval_id": record.approval_id,
        "challenge": expected,
        "intent_hash": record.intent_hash,
    }


def consume_approval(store: PaperExecutionStore, intent: ManualPaperIntent, approval_id: str, approved_intent_hash: str) -> dict[str, object]:
    if approved_intent_hash != intent_hash(intent):
        return {"status": "NO_GO", "approval_status": INTENT_CHANGED_AFTER_APPROVAL_BLOCKED}
    status = store.consume_approval(approval_id)
    return {"status": "GO" if status == APPROVED_FOR_SINGLE_SUBMISSION else "NO_GO", "approval_status": status}


def prepare_cancel_approval(store: PaperExecutionStore, intent: ManualPaperIntent, *, ttl_seconds: int) -> dict[str, object]:
    record = make_approval(intent, ttl_seconds=ttl_seconds, approval_type="CANCEL")
    status = store.append_approval(model_to_jsonable(record))
    return {
        "status": "GO" if status in {"APPROVAL_RECORDED", "APPROVAL_IDEMPOTENT"} else "NO_GO",
        "cancel_approval_status": "CANCEL_APPROVAL_REQUIRED",
        "approval_id": record.approval_id,
        "challenge": cancel_challenge(intent),
        "intent_hash": record.intent_hash,
    }


def _fmt_decimal(value: Decimal) -> str:
    return format(value.normalize(), "f")
