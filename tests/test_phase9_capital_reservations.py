from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from stocks.ibkr.callbacks import CallbackState
from stocks.ibkr.paper_execution.cancellation import (
    record_broker_cancel_confirmation,
)
from stocks.ibkr.paper_execution.executions import (
    FillExecution,
    record_fill_execution,
)
from stocks.ibkr.paper_execution.order_ids import allocate_order_id
from stocks.ibkr.paper_execution.storage import PaperExecutionStore
from stocks.ibkr.paper_execution.submission import submit_place_order_once


def test_cancel_request_keeps_order_and_capital_active_until_broker_proof(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _register_buy(store, "BUY-1")
    assert (
        store.reserve_capital_once(
            intent_id="BUY-1", amount_eur=Decimal("90"), con_id=42
        )
        == "CAPITAL_RESERVED"
    )
    store.append_event("BUY-1", "PLACE_ORDER_CALLED_ONCE", {})
    store.append_event("BUY-1", "CANCEL_ORDER_CALLED_ONCE", {})

    assert store.active_local_order_count() == 1
    assert store.capital_summary()["reserved_capital_eur"] == "90"
    assert record_broker_cancel_confirmation(
        store, intent_id="BUY-1", broker_proof=False
    )["cancel_status"] == "BROKER_CANCEL_PROOF_REQUIRED"

    confirmed = record_broker_cancel_confirmation(
        store, intent_id="BUY-1", broker_proof=True
    )
    assert confirmed["cancel_status"] == "BROKER_CANCEL_CONFIRMED"
    assert store.active_local_order_count() == 0
    assert store.capital_reservation_state("BUY-1")["status"] == "RELEASED"


def test_partial_fill_releases_only_unfilled_reservation_after_cancel(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _register_buy(store, "BUY-PARTIAL")
    store.reserve_capital_once(
        intent_id="BUY-PARTIAL", amount_eur=Decimal("90"), con_id=42
    )
    store.append_event("BUY-PARTIAL", "PLACE_ORDER_CALLED_ONCE", {})
    assert record_fill_execution(
        store,
        _fill("EXEC-PARTIAL", "BUY-PARTIAL", "BUY", "0.4", "90"),
    )["execution_status"] == "EXECUTION_ACCEPTED"
    store.append_event("BUY-PARTIAL", "CANCEL_ORDER_CALLED_ONCE", {})

    result = record_broker_cancel_confirmation(
        store, intent_id="BUY-PARTIAL", broker_proof=True
    )
    assert result["status"] == "GO"
    assert store.capital_reservation_state("BUY-PARTIAL")["status"] == "RELEASED"
    assert store.capital_summary()["reserved_capital_eur"] == "0"
    assert store.capital_summary()["deployed_capital_eur"] == "33.120"


def test_full_closing_sell_releases_deployed_capital_and_closes_episode(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    _register_buy(store, "BUY-CLOSE")
    store.reserve_capital_once(
        intent_id="BUY-CLOSE", amount_eur=Decimal("90"), con_id=42
    )
    record_fill_execution(
        store, _fill("EXEC-BUY", "BUY-CLOSE", "BUY", "1", "90")
    )
    store.register_intent(_intent("SELL-CLOSE", side="SELL"))

    result = record_fill_execution(
        store, _fill("EXEC-SELL", "SELL-CLOSE", "SELL", "1", "92")
    )

    assert result["execution_status"] == "EXECUTION_ACCEPTED"
    assert store.capital_reservation_state("BUY-CLOSE")["status"] == "RELEASED"
    assert store.event_type_count("CANONICAL_POSITION_EPISODE_CLOSED") == 1


def test_broker_reject_releases_unfilled_reservation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _register_buy(store, "BUY-REJECT")
    store.reserve_capital_once(
        intent_id="BUY-REJECT", amount_eur=Decimal("90"), con_id=42
    )
    assert allocate_order_id(
        store, broker_next_id=5, intent_id="BUY-REJECT"
    )["order_id_status"] == "ORDER_ID_READY"

    class App:
        def __init__(self) -> None:
            self.callback_state = CallbackState()

        def placeOrder(self, *_args: object) -> None:
            self.callback_state.errors.append({"code": 201})

    result = submit_place_order_once(
        App(),
        order_id=5,
        contract=object(),
        order=object(),
        store=store,
        intent_id="BUY-REJECT",
        ack_timeout_seconds=1,
    )

    assert result["submission_status"] == "BROKER_SUBMISSION_REJECTED"
    assert store.capital_reservation_state("BUY-REJECT")["status"] == "RELEASED"


def test_reservation_is_idempotent_and_conflicts_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _register_buy(store, "BUY-IDEMPOTENT")
    assert store.reserve_capital_once(
        intent_id="BUY-IDEMPOTENT", amount_eur=Decimal("90"), con_id=42
    ) == "CAPITAL_RESERVED"
    assert store.reserve_capital_once(
        intent_id="BUY-IDEMPOTENT", amount_eur=Decimal("90"), con_id=42
    ) == "CAPITAL_RESERVATION_IDEMPOTENT"
    assert store.reserve_capital_once(
        intent_id="BUY-IDEMPOTENT", amount_eur=Decimal("91"), con_id=42
    ) == "CAPITAL_RESERVATION_CONFLICT_BLOCKED"


def _store(tmp_path: Path) -> PaperExecutionStore:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    return store


def _register_buy(store: PaperExecutionStore, intent_id: str) -> None:
    assert store.register_intent(_intent(intent_id)) == "INTENT_REGISTERED"


def _intent(intent_id: str, *, side: str = "BUY") -> dict[str, object]:
    return {
        "intent_id": intent_id,
        "economic_order_key": f"KEY-{intent_id}",
        "created_at": "2026-08-06T10:00:00+00:00",
        "side": side,
        "quantity": "1",
        "con_id": 42,
    }


def _fill(
    exec_id: str,
    intent_id: str,
    side: str,
    quantity: str,
    price: str,
) -> FillExecution:
    return FillExecution(
        exec_id=exec_id,
        intent_id=intent_id,
        account_fingerprint="ACCOUNT-FINGERPRINT",
        perm_id=f"PERM-{exec_id}",
        broker_order_id=f"ORDER-{exec_id}",
        con_id=42,
        symbol="ON",
        currency="USD",
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        execution_time="2026-08-06T10:00:00+00:00",
        submitted_quantity=Decimal("1"),
        fx_rate=Decimal("0.92"),
    )
