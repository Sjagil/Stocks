from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def build_historical_orphan_quarantine(
    store: PaperExecutionStore,
    *,
    observation: Mapping[str, Any],
    position_projection: Mapping[str, Any],
    operator_completion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate current broker truth from incomplete canonical history.

    This is a read-only classification. It never appends a terminal event,
    execution, commission, or capital release to the Phase 9 ledger.
    """

    current_broker_flat = bool(
        observation.get("status") == "GO"
        and _integer(observation.get("same_client_open_order_count")) == 0
        and _integer(observation.get("all_api_open_order_count")) == 0
        and _integer(observation.get("position_count")) == 0
    )
    records = _unproven_order_records(store)
    records.extend(
        _unmatched_execution_records(
            store,
            observation=observation,
            position_projection=position_projection,
            operator_completion=operator_completion,
        )
    )

    quarantine_active = bool(records)
    return {
        "status": "NO_GO" if quarantine_active else "GO",
        "quarantine_status": (
            "HISTORICAL_ORPHANS_QUARANTINED_FAIL_CLOSED"
            if quarantine_active
            else "NO_HISTORICAL_ORPHANS"
        ),
        "operational_broker_state_status": (
            "CURRENT_BROKER_FLAT_READ_ONLY"
            if current_broker_flat
            else "CURRENT_BROKER_STATE_NOT_FLAT_OR_UNPROVEN"
        ),
        "current_broker_flat": current_broker_flat,
        "current_position_management_required": not current_broker_flat,
        "canonical_execution_evidence_status": (
            "INCOMPLETE_HISTORICAL_EXECUTION_CHAIN"
            if quarantine_active
            else "NO_QUARANTINED_EXECUTION_GAPS"
        ),
        "historical_orphan_count": len(records),
        "historical_orphans": records,
        "operator_attestation_present": operator_completion is not None,
        "operator_attestation_effect": (
            "BROKER_STATE_CONTEXT_ONLY_NO_CANONICAL_LEDGER_EFFECT"
            if operator_completion is not None
            else "NONE"
        ),
        "execution_history_complete": bool(
            observation.get("execution_history_complete", False)
        ),
        "historical_records_retained_in_canonical_ledger": True,
        "phase9_ledger_mutated": False,
        "automatic_terminal_events_appended": 0,
        "automatic_execution_imports": 0,
        "automatic_commission_imports": 0,
        "automatic_capital_releases": 0,
        "broker_write_calls": 0,
        "automatic_financial_actions_allowed": False,
        "execution_authority": "NONE",
        "live_authority": "NONE",
        "required_remediation": _required_remediation(records),
    }


def _unproven_order_records(
    store: PaperExecutionStore,
) -> list[dict[str, Any]]:
    events = store.list_events()
    records: list[dict[str, Any]] = []
    for intent in store.active_local_order_intents():
        intent_id = str(intent["intent_id"])
        intent_events = [
            event for event in events if event["aggregate_id"] == intent_id
        ]
        cancel_requests = [
            event
            for event in intent_events
            if event["event_type"] == "CANCEL_ORDER_CALLED_ONCE"
        ]
        terminal_callbacks = [
            event
            for event in intent_events
            if event["event_type"]
            in {
                "BROKER_ORDER_CANCELLED",
                "BROKER_SUBMISSION_REJECTED",
                "BROKER_SUBMISSION_ACK_INVALIDATED",
            }
        ]
        if terminal_callbacks:
            continue
        order_id = store.latest_order_id_for_intent(intent_id)
        records.append(
            {
                "orphan_type": "UNPROVEN_LOCAL_ORDER_TERMINAL_STATE",
                "classification": (
                    "CANCEL_REQUEST_WITHOUT_BROKER_TERMINAL_CALLBACK"
                    if cancel_requests
                    else "SUBMISSION_WITHOUT_BROKER_TERMINAL_CALLBACK"
                ),
                "intent_id_hash": stable_hash(intent_id),
                "local_order_id_hash": (
                    None
                    if order_id is None
                    else stable_hash({"local_order_id": order_id})
                ),
                "symbol": str(intent.get("symbol", "UNKNOWN")),
                "con_id": int(intent.get("con_id", 0)),
                "side": str(intent.get("side", "")).upper(),
                "quantity": str(intent.get("quantity", "0")),
                "submitted_at": _event_time(
                    intent_events, "PLACE_ORDER_CALLED_ONCE"
                ),
                "cancel_requested_at": (
                    None if not cancel_requests else cancel_requests[-1]["created_at"]
                ),
                "broker_terminal_callback_observed": False,
                "canonical_terminal_state_proven": False,
            }
        )
    return records


def _unmatched_execution_records(
    store: PaperExecutionStore,
    *,
    observation: Mapping[str, Any],
    position_projection: Mapping[str, Any],
    operator_completion: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    executions = store.list_executions()
    commissioned = {
        str(row["payload"].get("exec_identity", ""))
        for row in store.list_commissions()
    }
    sells_by_con_id: dict[int, Decimal] = {}
    for row in executions:
        payload = row["payload"]
        if str(payload.get("side", "")).upper() != "SELL":
            continue
        con_id = int(payload.get("con_id", 0))
        sells_by_con_id[con_id] = sells_by_con_id.get(
            con_id, Decimal("0")
        ) + Decimal(str(payload.get("quantity", "0")))

    local_quantity = Decimal(
        str(
            position_projection.get("position", {}).get(
                "long_quantity", "0"
            )
        )
    )
    broker_quantity = Decimal(str(observation.get("position_count", 0)))
    records: list[dict[str, Any]] = []
    for row in executions:
        payload = row["payload"]
        if str(payload.get("side", "")).upper() != "BUY":
            continue
        execution_id = str(payload.get("exec_id") or row["exec_identity"])
        con_id = int(payload.get("con_id", 0))
        buy_quantity = Decimal(str(payload.get("quantity", "0")))
        reasons: list[str] = []
        if execution_id not in commissioned:
            reasons.append("COMMISSION_MISSING")
        if sells_by_con_id.get(con_id, Decimal("0")) < buy_quantity:
            reasons.append("CANONICAL_CLOSING_EXECUTION_MISSING")
        if local_quantity != broker_quantity:
            reasons.append("LOCAL_BROKER_POSITION_DIVERGENCE")
        if not reasons:
            continue
        records.append(
            {
                "orphan_type": "INCOMPLETE_CANONICAL_EXECUTION_CHAIN",
                "classification": "+".join(sorted(set(reasons))),
                "execution_id_hash": stable_hash(execution_id),
                "intent_id_hash": stable_hash(str(payload.get("intent_id", ""))),
                "symbol": str(payload.get("symbol", "UNKNOWN")),
                "con_id": con_id,
                "side": "BUY",
                "quantity": str(buy_quantity),
                "execution_time": payload.get("execution_time"),
                "commission_observed": execution_id in commissioned,
                "canonical_closing_execution_observed": (
                    sells_by_con_id.get(con_id, Decimal("0")) >= buy_quantity
                ),
                "external_manual_close_attested": bool(
                    operator_completion is not None
                    and operator_completion.get("external_manual_close") is True
                ),
                "external_manual_close_is_canonical_execution": False,
            }
        )
    return records


def _event_time(events: list[dict[str, Any]], event_type: str) -> str | None:
    matching = [
        str(event["created_at"])
        for event in events
        if event["event_type"] == event_type
    ]
    return None if not matching else matching[-1]


def _required_remediation(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return []
    required: set[str] = set()
    for record in records:
        classification = str(record["classification"])
        if "TERMINAL_CALLBACK" in classification:
            required.add("CANONICAL_BROKER_ORDER_TERMINAL_EVIDENCE")
        if "COMMISSION_MISSING" in classification:
            required.add("CANONICAL_EXECUTION_COMMISSION_JOIN")
        if "CLOSING_EXECUTION_MISSING" in classification:
            required.add("CANONICAL_BROKER_CLOSING_EXECUTION")
        if "POSITION_DIVERGENCE" in classification:
            required.add("COMPLETE_BOUNDED_EXECUTION_HISTORY_RECONCILIATION")
    return sorted(required)


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


__all__ = ["build_historical_orphan_quarantine"]
