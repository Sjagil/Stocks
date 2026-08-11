from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from stocks.execution.idempotency import stable_hash
from stocks.live.models import ManualLiveBracketIntent
from stocks.live.store import LiveExecutionStore


def approval_challenge(intent: ManualLiveBracketIntent) -> str:
    return (
        f"APPROVE LIVE CANARY {intent.intent_id} "
        f"{_fmt(intent.quantity)} {intent.symbol} "
        f"{_fmt(intent.entry_limit_price)} "
        f"{_fmt(intent.stop_price)} {_fmt(intent.take_profit_price)}"
    )


def approve(
    store: LiveExecutionStore,
    intent: ManualLiveBracketIntent,
    approval_text: str,
    *,
    ttl_seconds: int,
) -> dict[str, object]:
    expected = approval_challenge(intent)
    if approval_text != expected:
        return {
            "status": "NO_GO",
            "approval_status": "APPROVAL_MISMATCH",
            "challenge": expected,
        }
    expires_at = (datetime.now(UTC) + timedelta(seconds=ttl_seconds)).isoformat()
    payload = {
        "approval_id": stable_hash(
            {
                "intent_id": intent.intent_id,
                "intent_hash": stable_hash(intent.jsonable()),
                "expires_at": expires_at,
            }
        )[:24],
        "intent_id": intent.intent_id,
        "intent_hash": stable_hash(intent.jsonable()),
        "challenge_hash": stable_hash(expected),
        "expires_at": expires_at,
        "used": False,
        "approval_type": "LIVE_SUBMIT",
    }
    result = store.append_approval(payload)
    return {
        "status": (
            "GO"
            if result in {"APPROVAL_RECORDED", "APPROVAL_IDEMPOTENT"}
            else "NO_GO"
        ),
        "approval_status": result,
        "approval_id": payload["approval_id"],
        "challenge": expected,
    }


def consume(
    store: LiveExecutionStore,
    intent: ManualLiveBracketIntent,
) -> dict[str, object]:
    record = store.find_unconsumed_approval(intent.intent_id, "LIVE_SUBMIT")
    if record is None:
        return {"status": "NO_GO", "approval_status": "APPROVAL_REQUIRED"}
    if datetime.fromisoformat(str(record["expires_at"])) < datetime.now(UTC):
        return {"status": "NO_GO", "approval_status": "APPROVAL_EXPIRED"}
    if str(record["intent_hash"]) != stable_hash(intent.jsonable()):
        return {
            "status": "NO_GO",
            "approval_status": "INTENT_CHANGED_AFTER_APPROVAL_BLOCKED",
        }
    status = store.consume_approval(str(record["approval_id"]))
    return {
        "status": (
            "GO" if status == "APPROVED_FOR_SINGLE_SUBMISSION" else "NO_GO"
        ),
        "approval_status": status,
    }


def _fmt(value: Decimal) -> str:
    return format(value.normalize(), "f")
