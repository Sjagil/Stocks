from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.authority import authority_contract
from stocks.ibkr.paper_execution.storage import COUNTERS, Phase9Layout, artifact, file_hashes, write_json


CANARY_A_MARKER = "PHASE9_CANARY_A_SUBMIT_CANCEL_SAFE_DONE"
CANARY_A_FREEZE_MARKER = "PHASE9_CANARY_A_EVIDENCE_ADOPTION_FROZEN_GO"
EVIDENCE_SCHEMA = "phase9_canary_a_submit_cancel_evidence_v1"
FINANCIAL_STATUS = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "blocked",
    "PAPER_STRATEGY_AUTHORITY": "blocked",
    "LIVE_STRATEGY_AUTHORITY": "blocked",
    "financial_decision": "NO_NEW_FINANCIAL_CANDIDATE",
}
FORBIDDEN_EVENT_TYPES = {
    "LIVE_PLACE_ORDER_CALLED": "live_calls",
    "AUTOMATIC_SUBMISSION": "automatic_submissions",
    "AUTOMATIC_CANCELLATION": "automatic_cancellations",
    "STRATEGY_GENERATED_INTENT": "strategy_generated_intents",
}
FREEZE_SOURCES = [
    "main.py",
    "src/stocks/ibkr/paper_execution/canary_a_evidence.py",
    "src/stocks/ibkr/paper_execution/audit.py",
    "src/stocks/ibkr/paper_execution/storage.py",
    "tests/test_phase9_canary_a_evidence.py",
]


def reconstruct_canary_a_evidence(
    project_root: Path,
    *,
    publish: bool = True,
    verify_existing: bool = True,
) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    evidence_path = layout.artifact("canary-a-submit-cancel-evidence.json")
    reconciliation_path = layout.artifact("reconciliation-audit.json")
    missing = not layout.db_path.exists() or not reconciliation_path.exists()
    ledger_hash = sha256_file(layout.db_path)
    reconciliation_hash = sha256_file(reconciliation_path)
    reconciliation = _read_json(reconciliation_path)
    historical_freeze_valid = _historical_freeze_valid(layout)

    if missing or reconciliation is None:
        payload = _public_payload(
            status="CANARY_A_EVIDENCE_MISSING",
            ledger_hash=ledger_hash,
            reconciliation_hash=reconciliation_hash,
        )
        return _publish(layout, payload, publish=publish)

    ledger = _read_ledger(layout.db_path)
    manual_buy_ids = [
        intent_id
        for intent_id, intent in ledger["intents"].items()
        if intent.get("intent_source") == "MANUAL_OPERATOR"
        and intent.get("side") == "BUY"
        and intent.get("security_type") == "STK"
        and intent.get("order_type") == "LIMIT"
        and intent.get("time_in_force") == "DAY"
        and intent.get("outside_rth") is False
    ]
    events_by_intent = {
        intent_id: [
            event
            for event in ledger["events"]
            if event["aggregate_id"] == intent_id
        ]
        for intent_id in manual_buy_ids
    }
    completed_candidate_ids = [
        intent_id
        for intent_id in manual_buy_ids
        if any(
            event["event_type"] == "PLACE_ORDER_CALLED_ONCE"
            for event in events_by_intent[intent_id]
        )
        and any(
            event["event_type"] == "CANCEL_ORDER_CALLED_ONCE"
            for event in events_by_intent[intent_id]
        )
    ]
    completed_candidate_ids.sort(
        key=lambda intent_id: min(
            _parse_time(event["created_at"])
            for event in events_by_intent[intent_id]
            if event["event_type"] == "PLACE_ORDER_CALLED_ONCE"
        )
    )
    # Canary A is the earliest completed manual submit/cancel lifecycle. Later
    # manual canaries are append-only evidence and must not erase this proof.
    candidate_id = (
        completed_candidate_ids[0] if completed_candidate_ids else None
    )
    accepted_intent_count = sum(
        1
        for event in ledger["events"]
        if event["event_type"] == "MANUAL_OPERATOR_INTENT"
        and event["aggregate_id"] == candidate_id
    )
    candidate_events = events_by_intent.get(candidate_id, [])
    place_events = [event for event in candidate_events if event["event_type"] == "PLACE_ORDER_CALLED_ONCE"]
    cancel_events = [event for event in candidate_events if event["event_type"] == "CANCEL_ORDER_CALLED_ONCE"]
    global_place_count = sum(event["event_type"] == "PLACE_ORDER_CALLED_ONCE" for event in ledger["events"])
    global_cancel_count = sum(event["event_type"] == "CANCEL_ORDER_CALLED_ONCE" for event in ledger["events"])

    submit_approvals = _valid_consumed_approvals(
        ledger["approvals"],
        intent_id=candidate_id,
        approval_type="SUBMIT",
        event_time=place_events[0]["created_at"] if len(place_events) == 1 else None,
    )
    cancel_approvals = _valid_consumed_approvals(
        ledger["approvals"],
        intent_id=candidate_id,
        approval_type="CANCEL",
        event_time=cancel_events[0]["created_at"] if len(cancel_events) == 1 else None,
    )
    effective_cancel_approval_count = 1 if cancel_approvals and len(cancel_events) == 1 else 0

    candidate_executions = [
        execution
        for execution in ledger["executions"]
        if execution["intent_id"] == candidate_id
    ]
    candidate_execution_ids = {
        execution["exec_identity"] for execution in candidate_executions
    }
    all_execution_ids = {
        execution["exec_identity"] for execution in ledger["executions"]
    }
    candidate_commissions = [
        commission
        for commission in ledger["commissions"]
        if commission["exec_identity"] in candidate_execution_ids
        or commission["exec_identity"] not in all_execution_ids
    ]
    candidate_execution_count = len(candidate_executions)
    candidate_commission_count = len(candidate_commissions)
    local_active_intent_ids = {
        intent_id
        for intent_id, intent_events in events_by_intent.items()
        if sum(
            event["event_type"] == "PLACE_ORDER_CALLED_ONCE"
            for event in intent_events
        )
        > sum(
            event["event_type"] == "CANCEL_ORDER_CALLED_ONCE"
            for event in intent_events
        )
        and not any(
            execution["intent_id"] == intent_id
            for execution in ledger["executions"]
        )
    }
    local_active_order_count = int(candidate_id in local_active_intent_ids)
    other_known_active_order_count = len(
        local_active_intent_ids - ({candidate_id} if candidate_id else set())
    )
    broker_open_order_count = int(reconciliation.get("broker_open_order_count", -1))
    broker_position_count = int(reconciliation.get("broker_position_count", -1))
    broker_execution_count = int(reconciliation.get("broker_execution_count", -1))
    broker_commission_count = int(reconciliation.get("broker_commission_count", -1))
    reconciliation_status = str(reconciliation.get("reconciliation_status", "MISSING"))
    reconciliation_content_hash_go = _content_hash_valid(reconciliation)
    subsequent_execution_count = len(ledger["executions"]) - candidate_execution_count
    subsequent_commission_count = len(ledger["commissions"]) - candidate_commission_count
    later_position_reconciled = (
        subsequent_execution_count > 0
        and "local_position_quantity" in reconciliation
        and "broker_position_quantity" in reconciliation
        and reconciliation["local_position_quantity"]
        == reconciliation["broker_position_quantity"]
    )
    reconciliation_scope_valid = (
        reconciliation_status
        in {"PAPER_RECONCILED_EMPTY", "PAPER_RECONCILED_OPEN_ORDER"}
        if subsequent_execution_count == 0
        else reconciliation_status
        in {
            "PAPER_RECONCILED",
            "PAPER_RECONCILED_EMPTY",
            "PAPER_RECONCILED_OPEN_LONG",
            "PAPER_RECONCILED_OPEN_ORDER",
        }
    )

    ordered_lifecycle = (
        len(place_events) == 1
        and len(cancel_events) == 1
        and _parse_time(place_events[0]["created_at"]) < _parse_time(cancel_events[0]["created_at"])
    )
    direct_ack = any(
        event["event_type"] in {"ORDER_ACKNOWLEDGED", "OPEN_ORDER_ACKNOWLEDGED", "ORDER_STATUS_SUBMITTED"}
        for event in candidate_events
    )
    current_reconciliation_proves_absence = (
        local_active_order_count == 0
        and broker_open_order_count == other_known_active_order_count
        and (
            broker_position_count == 0
            if subsequent_execution_count == 0
            else later_position_reconciled
        )
        and broker_execution_count == len(ledger["executions"])
        and broker_commission_count == len(ledger["commissions"])
        and int(reconciliation.get("unknown_broker_open_order_count", 0)) == 0
        and int(reconciliation.get("missing_local_open_order_count", 0)) == 0
        and reconciliation_scope_valid
    )
    frozen_reconciliation_proves_absence = (
        historical_freeze_valid
        and subsequent_execution_count > 0
        and local_active_order_count == 0
        and broker_open_order_count == other_known_active_order_count
        and later_position_reconciled
        and int(reconciliation.get("unknown_broker_open_order_count", 0))
        == 0
        and int(reconciliation.get("missing_local_open_order_count", 0))
        == 0
        and reconciliation_scope_valid
    )
    broker_candidate_absent = (
        current_reconciliation_proves_absence
        or frozen_reconciliation_proves_absence
    )
    acknowledgement_present = direct_ack or (
        ordered_lifecycle and broker_candidate_absent
    )
    acknowledgement_source = (
        "DIRECT_CALLBACK_EVENT"
        if direct_ack
        else "FROZEN_HISTORICAL_RECONCILIATION"
        if frozen_reconciliation_proves_absence
        else "BROKER_RECONCILED_LIFECYCLE"
        if acknowledgement_present
        else "MISSING"
    )
    final_order_state = (
        "API_CANCELLED"
        if ordered_lifecycle and broker_candidate_absent
        else "BLOCKED"
    )
    final_order_state_source = (
        "CANCEL_CALL_PLUS_FROZEN_HISTORICAL_RECONCILIATION"
        if frozen_reconciliation_proves_absence
        else "CANCEL_CALL_PLUS_RECONCILED_EMPTY"
        if final_order_state == "API_CANCELLED"
        else "MISSING"
    )

    forbidden = {name: 0 for name in FORBIDDEN_EVENT_TYPES.values()}
    for event in candidate_events:
        counter = FORBIDDEN_EVENT_TYPES.get(event["event_type"])
        if counter is not None:
            forbidden[counter] += 1
    forbidden["live_calls"] += int(reconciliation.get("live_place_order_calls", 0))
    forbidden["automatic_submissions"] += int(reconciliation.get("automatic_submissions", 0))
    forbidden["automatic_cancellations"] += int(reconciliation.get("automatic_cancellations", 0))
    forbidden["strategy_generated_intents"] += int(reconciliation.get("strategy_generated_intents", 0))
    forbidden["global_cancel_calls"] = int(reconciliation.get("global_cancel_calls", 0))
    forbidden["auto_bind_order_calls"] = int(reconciliation.get("auto_bind_order_calls", 0))
    forbidden["exercise_option_calls"] = int(reconciliation.get("exercise_option_calls", 0))

    status = _status(
        accepted_intent_count=accepted_intent_count,
        candidate_count=1 if candidate_id else 0,
        submit_approval_count=len(submit_approvals),
        effective_cancel_approval_count=effective_cancel_approval_count,
        place_count=len(place_events),
        cancel_count=len(cancel_events),
        acknowledgement_present=acknowledgement_present,
        final_order_state=final_order_state,
        execution_count=candidate_execution_count,
        commission_count=candidate_commission_count,
        broker_empty=broker_candidate_absent,
        reconciliation_content_hash_go=reconciliation_content_hash_go,
        forbidden=forbidden,
    )
    evidence_basis = {
        "accepted_manual_buy_intent_count": accepted_intent_count,
        "valid_one_time_submission_approval_count": len(submit_approvals),
        "effective_cancel_approval_count": effective_cancel_approval_count,
        "place_order_economic_call_count": len(place_events),
        "cancel_order_economic_call_count": len(cancel_events),
        "order_acknowledgement_present": acknowledgement_present,
        "final_order_state": final_order_state,
        "execution_count": candidate_execution_count,
        "commission_count": candidate_commission_count,
        "subsequent_execution_count": subsequent_execution_count,
        "subsequent_commission_count": subsequent_commission_count,
        "local_active_order_count": local_active_order_count,
        "broker_open_order_count": broker_open_order_count,
        "broker_position_count": broker_position_count,
        "reconciliation_status": reconciliation_status,
        "candidate_absence_basis": (
            "FROZEN_HISTORICAL_RECONCILIATION"
            if frozen_reconciliation_proves_absence
            else "CURRENT_RECONCILIATION"
            if current_reconciliation_proves_absence
            else "UNPROVEN"
        ),
        "forbidden_counter_summary": forbidden,
        "source_ledger_hash": ledger_hash,
        "source_reconciliation_hash": reconciliation_hash,
    }
    evidence_hash = stable_hash(evidence_basis)
    if verify_existing and _existing_hash_mismatch(
        evidence_path,
        ledger_hash=ledger_hash,
        reconciliation_hash=reconciliation_hash,
        evidence_hash=evidence_hash,
    ):
        status = "CANARY_A_EVIDENCE_HASH_MISMATCH"

    payload = _public_payload(
        status=status,
        ledger_hash=ledger_hash,
        reconciliation_hash=reconciliation_hash,
        evidence_hash=evidence_hash,
        accepted_manual_buy_intent_count=accepted_intent_count,
        staged_unsubmitted_manual_buy_intent_count=sum(
            not any(
                event["event_type"] == "PLACE_ORDER_CALLED_ONCE"
                for event in events_by_intent[intent_id]
            )
            for intent_id in manual_buy_ids
        ),
        subsequent_manual_buy_lifecycle_count=max(
            0, len(manual_buy_ids) - int(candidate_id is not None)
        ),
        completed_manual_submit_cancel_lifecycle_count=len(
            completed_candidate_ids
        ),
        other_known_active_order_count=other_known_active_order_count,
        valid_one_time_submission_approval_count=len(submit_approvals),
        cancel_approval_attempt_count=sum(
            row["approval_type"] == "CANCEL" and row["intent_id"] == candidate_id
            for row in ledger["approvals"]
        ),
        effective_cancel_approval_count=effective_cancel_approval_count,
        place_order_economic_call_count=len(place_events),
        cancel_order_economic_call_count=len(cancel_events),
        order_acknowledgement_present=acknowledgement_present,
        order_acknowledgement_source=acknowledgement_source,
        final_order_state=final_order_state,
        final_order_state_source=final_order_state_source,
        execution_count=candidate_execution_count,
        commission_count=candidate_commission_count,
        local_active_order_count=local_active_order_count,
        broker_open_order_count=broker_open_order_count,
        broker_position_count=broker_position_count,
        reconciliation_status=reconciliation_status,
        historical_freeze_valid=historical_freeze_valid,
        candidate_absence_basis=evidence_basis[
            "candidate_absence_basis"
        ],
        forbidden_counter_summary=forbidden,
        cumulative_private_ledger_evidence={
            "intent_count": len(ledger["intents"]),
            "approval_attempt_count": len(ledger["approvals"]),
            "place_order_economic_call_count": global_place_count,
            "cancel_order_economic_call_count": global_cancel_count,
            "execution_count": len(ledger["executions"]),
            "commission_count": len(ledger["commissions"]),
        },
    )
    result = _publish(layout, payload, publish=publish)
    if publish and status == "CANARY_A_EVIDENCE_GO":
        _write_freeze(layout, result)
    return result


def _status(
    *,
    accepted_intent_count: int,
    candidate_count: int,
    submit_approval_count: int,
    effective_cancel_approval_count: int,
    place_count: int,
    cancel_count: int,
    acknowledgement_present: bool,
    final_order_state: str,
    execution_count: int,
    commission_count: int,
    broker_empty: bool,
    reconciliation_content_hash_go: bool,
    forbidden: dict[str, int],
) -> str:
    if any(forbidden.values()):
        return "CANARY_A_FORBIDDEN_CALL_DETECTED"
    if candidate_count != 1 or accepted_intent_count != 1:
        return "CANARY_A_EVIDENCE_MISSING"
    if place_count != 1:
        return "CANARY_A_PLACE_COUNT_MISMATCH"
    if cancel_count != 1:
        return "CANARY_A_CANCEL_COUNT_MISMATCH"
    if submit_approval_count != 1 or effective_cancel_approval_count != 1:
        return "CANARY_A_EVIDENCE_MISSING"
    if execution_count != 0:
        return "CANARY_A_EXECUTION_UNEXPECTED"
    if commission_count != 0:
        return "CANARY_A_COMMISSION_UNEXPECTED"
    if not reconciliation_content_hash_go:
        return "CANARY_A_EVIDENCE_HASH_MISMATCH"
    if not broker_empty:
        return "CANARY_A_BROKER_NOT_EMPTY"
    if not acknowledgement_present:
        return "CANARY_A_FINAL_STATE_BLOCKED"
    if final_order_state not in {"CANCELLED", "API_CANCELLED"}:
        return "CANARY_A_FINAL_STATE_BLOCKED"
    return "CANARY_A_EVIDENCE_GO"


def _read_ledger(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        intents = {
            str(row["intent_id"]): json.loads(str(row["payload_json"]))
            for row in conn.execute("SELECT intent_id, payload_json FROM intents")
        }
        approvals = []
        for row in conn.execute("SELECT intent_id, payload_json, created_at, used FROM approvals"):
            payload = json.loads(str(row["payload_json"]))
            approvals.append(
                {
                    "intent_id": str(row["intent_id"]),
                    "approval_type": str(payload.get("approval_type", "")),
                    "created_at": str(row["created_at"]),
                    "expires_at": str(payload.get("expires_at", "")),
                    "used": bool(row["used"]),
                }
            )
        events = [
            {
                "aggregate_id": str(row["aggregate_id"]),
                "event_type": str(row["event_type"]),
                "created_at": str(row["created_at"]),
            }
            for row in conn.execute(
                "SELECT aggregate_id, event_type, created_at FROM events ORDER BY event_id"
            )
        ]
        executions = [
            {
                "exec_identity": str(row["exec_identity"]),
                "intent_id": str(row["intent_id"]),
            }
            for row in conn.execute("SELECT exec_identity, intent_id FROM executions")
        ]
        commissions = [
            {
                "commission_identity": str(row["commission_identity"]),
                "exec_identity": str(row["exec_identity"]),
            }
            for row in conn.execute("SELECT commission_identity, exec_identity FROM commissions")
        ]
    return {
        "intents": intents,
        "approvals": approvals,
        "events": events,
        "executions": executions,
        "commissions": commissions,
    }


def _valid_consumed_approvals(
    approvals: list[dict[str, Any]],
    *,
    intent_id: str | None,
    approval_type: str,
    event_time: str | None,
) -> list[dict[str, Any]]:
    if intent_id is None or event_time is None:
        return []
    event_at = _parse_time(event_time)
    return [
        row
        for row in approvals
        if row["intent_id"] == intent_id
        and row["approval_type"] == approval_type
        and row["used"]
        and _parse_time(row["created_at"]) <= event_at <= _parse_time(row["expires_at"])
    ]


def _public_payload(
    *,
    status: str,
    ledger_hash: str | None,
    reconciliation_hash: str | None,
    evidence_hash: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    return artifact(
        EVIDENCE_SCHEMA,
        {
            "status": status,
            "canary_marker": CANARY_A_MARKER if status == "CANARY_A_EVIDENCE_GO" else "NO_GO",
            **fields,
            "command_local_counters": dict(COUNTERS),
            "source_ledger_hash": ledger_hash,
            "source_reconciliation_hash": reconciliation_hash,
            "evidence_hash": evidence_hash,
            "new_paper_place_order_calls": 0,
            "new_paper_cancel_order_calls": 0,
            **authority_contract(enabled=False),
            **FINANCIAL_STATUS,
        },
    )


def _publish(layout: Phase9Layout, payload: dict[str, Any], *, publish: bool) -> dict[str, Any]:
    if publish:
        write_json(layout.artifact("canary-a-submit-cancel-evidence.json"), payload)
    return payload


def _write_freeze(layout: Phase9Layout, evidence: dict[str, Any]) -> None:
    if _historical_freeze_valid(layout):
        return
    write_json(
        layout.artifact("canary-a-evidence-freeze-status.json"),
        artifact(
            "phase9_canary_a_evidence_freeze_status_v1",
            {
                "freeze_status": CANARY_A_FREEZE_MARKER,
                "canary_marker": CANARY_A_MARKER,
                "evidence_hash": evidence["evidence_hash"],
                "evidence_artifact_hash": sha256_file(
                    layout.artifact("canary-a-submit-cancel-evidence.json")
                ),
                "source_ledger_hash": evidence["source_ledger_hash"],
                "source_reconciliation_hash": evidence["source_reconciliation_hash"],
                "source_hashes": file_hashes(layout.project_root, FREEZE_SOURCES),
                "phase9_full_freeze_status": "BLOCKED_UNTIL_OPERATOR_CANARY_B",
                "new_paper_place_order_calls": 0,
                "new_paper_cancel_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
    )


def _historical_freeze_valid(layout: Phase9Layout) -> bool:
    payload = _read_json(
        layout.artifact("canary-a-evidence-freeze-status.json")
    )
    if payload is None or not _content_hash_valid(payload):
        return False
    return (
        payload.get("schema")
        == "phase9_canary_a_evidence_freeze_status_v1"
        and payload.get("freeze_status") == CANARY_A_FREEZE_MARKER
        and payload.get("canary_marker") == CANARY_A_MARKER
        and isinstance(payload.get("evidence_hash"), str)
        and bool(payload["evidence_hash"])
        and isinstance(payload.get("source_ledger_hash"), str)
        and bool(payload["source_ledger_hash"])
        and isinstance(payload.get("source_reconciliation_hash"), str)
        and bool(payload["source_reconciliation_hash"])
        and payload.get("execution_authority") == "NONE"
        and payload.get("live_authority") == "NONE"
        and int(payload.get("new_paper_place_order_calls", -1)) == 0
        and int(payload.get("new_paper_cancel_order_calls", -1)) == 0
        and int(payload.get("live_place_order_calls", -1)) == 0
    )


def _existing_hash_mismatch(
    path: Path,
    *,
    ledger_hash: str | None,
    reconciliation_hash: str | None,
    evidence_hash: str,
) -> bool:
    existing = _read_json(path)
    if existing is None or existing.get("schema") != EVIDENCE_SCHEMA:
        return False
    if existing.get("source_ledger_hash") != ledger_hash:
        return False
    if existing.get("source_reconciliation_hash") != reconciliation_hash:
        return False
    return not _content_hash_valid(existing) or existing.get("evidence_hash") != evidence_hash


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    observed = payload.get("content_hash")
    return isinstance(observed, str) and observed == stable_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)
