from __future__ import annotations

from pathlib import Path

from stocks.ibkr.paper_execution import audit as phase9_audit
from stocks.ibkr.paper_execution.canary_b_evidence import (
    reconstruct_fill_close_evidence,
)
from stocks.ibkr.paper_execution.storage import (
    PaperExecutionStore,
    Phase9Layout,
    artifact,
    write_json,
)


def test_submitted_buy_without_execution_remains_pending(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _intent(store, "BUY-CANARY", "BUY")
    store.append_event("BUY-CANARY", "PLACE_ORDER_CALLED_ONCE", {})

    result = reconstruct_fill_close_evidence(tmp_path)

    assert result["fill_canary"] == "BUY_FILL_PENDING"
    assert (
        result["closing_sell_canary"]
        == "NOT_RUN_REQUIRES_FILLED_LONG_POSITION"
    )
    assert result["status"] == "NO_GO"


def test_full_buy_requires_separate_closing_sell(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _intent(store, "BUY-CANARY", "BUY")
    store.append_event("BUY-CANARY", "PLACE_ORDER_CALLED_ONCE", {})
    _execution(store, "BUY-EXEC", "BUY-CANARY", "BUY")
    _commission(store, "BUY-EXEC")

    result = reconstruct_fill_close_evidence(tmp_path)

    assert result["fill_canary"] == "GO"
    assert result["closing_sell_canary"] == "CLOSING_SELL_REQUIRED"
    assert result["position_status"] == "PARTIALLY_OPEN"
    assert result["status"] == "NO_GO"


def test_full_buy_and_sell_with_empty_reconciliation_is_go(
    tmp_path: Path,
) -> None:
    store = _complete_round_trip(tmp_path)
    assert store.counts()["execution_count"] == 2

    result = reconstruct_fill_close_evidence(tmp_path)

    assert result["status"] == "GO"
    assert result["fill_canary"] == "GO"
    assert result["closing_sell_canary"] == "GO"
    assert result["position_status"] == "CLOSED"
    assert result["candidate_execution_count"] == 2
    assert result["candidate_commission_count"] == 2


def test_pending_commission_blocks_close_evidence(tmp_path: Path) -> None:
    store = _complete_round_trip(tmp_path)
    with store.connect() as conn:
        conn.execute(
            "DELETE FROM commissions WHERE exec_identity = ?",
            ("SELL-EXEC",),
        )
        conn.commit()

    result = reconstruct_fill_close_evidence(tmp_path)

    assert result["fill_canary"] == "GO"
    assert result["closing_sell_canary"] == "COMMISSION_JOIN_PENDING"
    assert result["status"] == "NO_GO"


def test_nonempty_broker_state_blocks_close_evidence(tmp_path: Path) -> None:
    _complete_round_trip(
        tmp_path,
        reconciliation_overrides={
            "reconciliation_status": "PAPER_RECONCILED_OPEN_ORDER",
            "broker_open_order_count": 1,
        },
    )

    result = reconstruct_fill_close_evidence(tmp_path)

    assert result["closing_sell_canary"] == "PAPER_ACCOUNT_NOT_EMPTY"
    assert result["status"] == "NO_GO"


def test_phase9_audit_preserves_reconstructed_round_trip(
    tmp_path: Path,
) -> None:
    _complete_round_trip(tmp_path)

    phase9_audit.phase9_audit(tmp_path)
    result = phase9_audit.canary_results(tmp_path)

    assert result["status"] == "NO_GO"
    assert result["fill_canary"] == "GO"
    assert result["closing_sell_canary"] == "GO"
    assert (
        Phase9Layout.from_project_root(tmp_path)
        .artifact("canary-results.json")
        .exists()
    )


def _complete_round_trip(
    project_root: Path,
    *,
    reconciliation_overrides: dict[str, object] | None = None,
) -> PaperExecutionStore:
    store = _store(project_root)
    _intent(store, "BUY-CANARY", "BUY", created_at="2026-07-27T12:00:00+00:00")
    store.append_event("BUY-CANARY", "PLACE_ORDER_CALLED_ONCE", {})
    _execution(store, "BUY-EXEC", "BUY-CANARY", "BUY")
    _commission(store, "BUY-EXEC")
    _intent(
        store,
        "SELL-CANARY",
        "SELL",
        created_at="2026-07-27T13:00:00+00:00",
    )
    store.append_event("SELL-CANARY", "PLACE_ORDER_CALLED_ONCE", {})
    _execution(store, "SELL-EXEC", "SELL-CANARY", "SELL")
    _commission(store, "SELL-EXEC")
    reconciliation = {
        "status": "GO",
        "reconciliation_status": "PAPER_RECONCILED_EMPTY",
        "local_active_order_count": 0,
        "broker_open_order_count": 0,
        "broker_position_count": 0,
        "unknown_broker_open_order_count": 0,
        "missing_local_open_order_count": 0,
    }
    reconciliation.update(reconciliation_overrides or {})
    write_json(
        Phase9Layout.from_project_root(project_root).artifact(
            "reconciliation-audit.json"
        ),
        artifact("phase9_reconciliation_audit_v1", reconciliation),
    )
    return store


def _store(project_root: Path) -> PaperExecutionStore:
    store = PaperExecutionStore(
        Phase9Layout.from_project_root(project_root).db_path
    )
    store.initialize()
    return store


def _intent(
    store: PaperExecutionStore,
    intent_id: str,
    side: str,
    *,
    created_at: str = "2026-07-27T12:00:00+00:00",
) -> None:
    assert (
        store.register_intent(
            {
                "intent_id": intent_id,
                "economic_order_key": intent_id + "-KEY",
                "intent_source": "MANUAL_OPERATOR",
                "created_at": created_at,
                "account_fingerprint": "PRIVATE-ACCOUNT",
                "con_id": 8677881,
                "symbol": "PRIVATE-SYMBOL",
                "side": side,
                "quantity": "1",
                "security_type": "STK",
                "order_type": "LIMIT",
                "time_in_force": "DAY",
                "outside_rth": False,
            }
        )
        == "INTENT_REGISTERED"
    )


def _execution(
    store: PaperExecutionStore,
    execution_id: str,
    intent_id: str,
    side: str,
) -> None:
    assert (
        store.append_execution_once(
            execution_id,
            intent_id,
            {
                "exec_id": execution_id,
                "fill_fingerprint": execution_id + "-FILL",
                "intent_id": intent_id,
                "con_id": 8677881,
                "symbol": "PRIVATE-SYMBOL",
                "currency": "USD",
                "side": side,
                "quantity": "1",
                "price": "90",
                "execution_time": "2026-07-27T12:30:00+00:00",
                "submitted_quantity": "1",
                "fx_rate": "0.9",
            },
        )
        == "EXECUTION_RECORDED"
    )


def _commission(store: PaperExecutionStore, execution_id: str) -> None:
    assert (
        store.append_commission_once(
            execution_id + "-COMMISSION",
            execution_id,
            {"amount": "0.35", "currency": "USD"},
        )
        == "COMMISSION_RECORDED"
    )
