from __future__ import annotations

from datetime import UTC, datetime

from stocks.ibkr.paper_execution.state_machine import (
    OrderLifecycleState,
    audit_order_state_machine,
    state_machine_schema,
    transition_allowed,
    transition_metadata,
)


def _event(event_id: int, event_type: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "aggregate_id": "I-1",
        "event_type": event_type,
        "created_at": f"2026-08-06T00:00:{event_id:02d}+00:00",
        "payload": {},
    }


def test_schema_publishes_allowed_and_forbidden_policy() -> None:
    schema = state_machine_schema()
    assert schema["status"] == "GO"
    assert schema["unknown_transition_policy"] == (
        "UNCLASSIFIED_TRANSITION_BLOCKED"
    )
    assert "SUBMIT_SENT" in schema["allowed_transitions"]["ORDER_ID_ALLOCATED"]
    assert schema["allowed_transitions"]["CANCELLED"] == []


def test_complete_ack_fill_sequence_is_valid() -> None:
    events = [
        _event(1, "MANUAL_OPERATOR_INTENT"),
        _event(2, "APPROVAL_RECORDED"),
        _event(3, "ORDER_ID_ALLOCATED"),
        _event(4, "PLACE_ORDER_CALLED_ONCE"),
        _event(5, "BROKER_SUBMISSION_ACKNOWLEDGED"),
        _event(6, "FILL_EXECUTION_ACCEPTED"),
    ]
    result = audit_order_state_machine(
        [{"intent_id": "I-1", "quantity": "1"}],
        events,
        [{"payload": {"intent_id": "I-1", "quantity": "1"}}],
    )
    assert result["status"] == "GO"
    assert result["projections"][0]["final_state"] == "FILLED"


def test_fill_without_submit_is_blocked() -> None:
    result = audit_order_state_machine(
        [{"intent_id": "I-1", "quantity": "1"}],
        [
            _event(1, "MANUAL_OPERATOR_INTENT"),
            _event(2, "FILL_EXECUTION_ACCEPTED"),
        ],
        [{"payload": {"intent_id": "I-1", "quantity": "1"}}],
    )
    assert result["status"] == "NO_GO"
    assert result["transition_violation_count"] == 1
    assert result["violations"][0]["to_state"] == "FILLED"


def test_cancelled_order_cannot_return_to_working() -> None:
    assert not transition_allowed(
        OrderLifecycleState.CANCELLED,
        OrderLifecycleState.WORKING,
    )


def test_transition_metadata_has_cause_source_and_correlation() -> None:
    metadata = transition_metadata(
        "I-1",
        "PLACE_ORDER_CALLED_ONCE",
        {},
        datetime.now(UTC).isoformat(),
    )
    assert metadata["cause"] == "PLACE_ORDER_CALLED_ONCE"
    assert metadata["source"] == "PHASE9_BROKER_ADAPTER"
    assert len(metadata["correlation_id"]) == 32

