from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from stocks.ibkr.paper_execution.commissions import record_execution_commission
from stocks.ibkr.paper_execution.executions import (
    FillExecution,
    project_position_from_store,
    record_fill_execution,
)
from stocks.ibkr.paper_execution.historical_quarantine import (
    build_historical_orphan_quarantine,
)
from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def test_historical_gaps_are_quarantined_without_ledger_mutation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    intent_id = "HISTORICAL-CANCEL-INTENT"
    store.register_intent(
        {
            "intent_id": intent_id,
            "economic_order_key": "HISTORICAL-CANCEL-KEY",
            "created_at": "2026-07-21T18:50:22+00:00",
            "symbol": "ON",
            "con_id": 8677881,
            "side": "BUY",
            "quantity": "1",
        }
    )
    assert store.allocate_order_id(1, intent_id) == ("ORDER_ID_READY", 1)
    assert store.mark_order_id_used(1) == "ORDER_ID_READY"
    store.append_event(intent_id, "PLACE_ORDER_CALLED_ONCE", {})
    store.append_event(intent_id, "CANCEL_ORDER_CALLED_ONCE", {})
    assert record_fill_execution(store, _fill("BUY-EXEC", side="BUY"))[
        "status"
    ] == "GO"
    before = store.counts()

    result = build_historical_orphan_quarantine(
        store,
        observation=_flat_observation(execution_history_complete=False),
        position_projection=project_position_from_store(store),
        operator_completion={"external_manual_close": True},
    )

    assert result["status"] == "NO_GO"
    assert result["quarantine_status"] == (
        "HISTORICAL_ORPHANS_QUARANTINED_FAIL_CLOSED"
    )
    assert result["operational_broker_state_status"] == (
        "CURRENT_BROKER_FLAT_READ_ONLY"
    )
    assert result["current_position_management_required"] is False
    assert result["historical_orphan_count"] == 2
    classifications = {
        row["classification"] for row in result["historical_orphans"]
    }
    assert "CANCEL_REQUEST_WITHOUT_BROKER_TERMINAL_CALLBACK" in classifications
    assert any("COMMISSION_MISSING" in value for value in classifications)
    assert any(
        "CANONICAL_CLOSING_EXECUTION_MISSING" in value
        for value in classifications
    )
    assert result["phase9_ledger_mutated"] is False
    assert result["broker_write_calls"] == 0
    assert result["automatic_financial_actions_allowed"] is False
    assert store.counts() == before
    assert not any(
        event["event_type"] == "BROKER_ORDER_CANCELLED"
        for event in store.list_events()
    )


def test_complete_canonical_round_trip_needs_no_quarantine(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    assert record_fill_execution(store, _fill("BUY-EXEC", side="BUY"))[
        "status"
    ] == "GO"
    assert record_execution_commission(
        store, execution_id="BUY-EXEC", commission=Decimal("0.01")
    )["status"] == "GO"
    assert record_fill_execution(store, _fill("SELL-EXEC", side="SELL"))[
        "status"
    ] == "GO"
    assert record_execution_commission(
        store, execution_id="SELL-EXEC", commission=Decimal("0.01")
    )["status"] == "GO"

    result = build_historical_orphan_quarantine(
        store,
        observation=_flat_observation(execution_history_complete=True),
        position_projection=project_position_from_store(store),
    )

    assert result["status"] == "GO"
    assert result["quarantine_status"] == "NO_HISTORICAL_ORPHANS"
    assert result["historical_orphan_count"] == 0
    assert result["canonical_execution_evidence_status"] == (
        "NO_QUARANTINED_EXECUTION_GAPS"
    )


def _store(root: Path) -> PaperExecutionStore:
    store = PaperExecutionStore(root / "paper_execution.sqlite3")
    store.initialize()
    return store


def _fill(exec_id: str, *, side: str) -> FillExecution:
    return FillExecution(
        exec_id=exec_id,
        intent_id=f"{side}-INTENT",
        account_fingerprint="TEST-ACCOUNT",
        perm_id=f"{side}-PERM",
        broker_order_id=f"{side}-ORDER",
        con_id=8677881,
        symbol="ON",
        currency="USD",
        side=side,
        quantity=Decimal("1"),
        price=Decimal("90"),
        execution_time="2026-07-31T09:30:00+00:00",
        submitted_quantity=Decimal("1"),
    )


def _flat_observation(*, execution_history_complete: bool) -> dict[str, object]:
    return {
        "status": "GO",
        "same_client_open_order_count": 0,
        "all_api_open_order_count": 0,
        "position_count": 0,
        "execution_history_complete": execution_history_complete,
    }
