from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import main
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.commissions import commission_join_audit, record_execution_commission
from stocks.ibkr.paper_execution.executions import FillExecution, project_position_from_store, record_fill_execution
from stocks.ibkr.paper_execution.known_fill import record_known_fill_from_snapshot
from stocks.ibkr.paper_execution.models import PaperWriterConfig
from stocks.ibkr.paper_execution.reconciliation import reconcile_paper_state, reconcile_position_projection
from stocks.ibkr.paper_execution.risk import evaluate_closing_sell_risk, prepare_intent
from stocks.ibkr.paper_execution.storage import PaperExecutionStore, Phase9Layout
from stocks.ibkr.paper_execution import audit as phase9_audit_module


def test_full_buy_fill_two_partials_and_average_cost(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert record_fill_execution(store, _fill("TEST-EXEC-BUY-1", side="BUY", quantity="0.4", price="89"))["execution_status"] == "EXECUTION_ACCEPTED"
    assert record_fill_execution(store, _fill("TEST-EXEC-BUY-2", side="BUY", quantity="0.6", price="91"))["execution_status"] == "EXECUTION_ACCEPTED"
    assert record_execution_commission(store, execution_id="TEST-EXEC-BUY-1", commission=Decimal("0.004"))["commission_status"] == "COMMISSION_JOINED"
    assert record_execution_commission(store, execution_id="TEST-EXEC-BUY-2", commission=Decimal("0.006"))["commission_status"] == "COMMISSION_JOINED"

    projection = project_position_from_store(store)

    assert projection["status"] == "GO"
    assert projection["position"]["long_quantity"] == "1.0"
    assert projection["position"]["position_status"] == "PARTIALLY_OPEN"
    assert Decimal(projection["position"]["average_cost_local"]) == Decimal("90.210")
    assert projection["order_level_cash_impact_status"] == "ORDER_LEVEL_CASH_IMPACT_RECONCILED"


def test_duplicate_execution_idempotent_and_conflict_blocked(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fill = _fill("TEST-EXEC-DUP", side="BUY", quantity="1", price="90")
    assert record_fill_execution(store, fill)["execution_status"] == "EXECUTION_ACCEPTED"
    assert record_fill_execution(store, fill)["execution_status"] == "IDEMPOTENT_REPLAY"
    assert record_fill_execution(store, _fill("TEST-EXEC-DUP", side="BUY", quantity="1", price="91"))["execution_status"] == "EXECUTION_CONFLICT_BLOCKED"
    assert store.counts()["execution_count"] == 1


def test_cumulative_partial_fills_cannot_exceed_submitted_quantity(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    assert (
        record_fill_execution(
            store,
            _fill(
                "TEST-EXEC-CUMULATIVE-1",
                side="BUY",
                quantity="0.6",
                price="90",
            ),
        )["execution_status"]
        == "EXECUTION_ACCEPTED"
    )
    assert (
        record_fill_execution(
            store,
            _fill(
                "TEST-EXEC-CUMULATIVE-2",
                side="BUY",
                quantity="0.5",
                price="90",
            ),
        )["execution_status"]
        == "FILLED_QUANTITY_EXCEEDS_SUBMITTED"
    )
    assert store.counts()["execution_count"] == 1


def test_known_fill_snapshot_match_is_explicit_idempotent_and_private(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    intent, _ = prepare_intent(
        tmp_path,
        _config(),
        con_id=8677881,
        side="BUY",
        quantity=Decimal("1"),
        limit_price=Decimal("90"),
        reason="known fill observation",
    )
    store.register_intent(intent.__dict__)
    order_id = 12345
    assert store.allocate_order_id(order_id, intent.intent_id)[0] == "ORDER_ID_READY"
    assert store.mark_order_id_used(order_id) == "ORDER_ID_READY"
    store.append_event(
        intent.intent_id,
        "PLACE_ORDER_CALLED_ONCE",
        {"order_id_hash": "PRIVATE"},
    )
    execution_id = "EXECUTION-HASHED-1"
    snapshot = _snapshot_for_intent(
        intent,
        order_id=order_id,
        execution_id=execution_id,
        quantity="1",
    )

    first = record_known_fill_from_snapshot(
        store,
        intent=intent,
        local_order_id=order_id,
        snapshot=snapshot,
    )
    second = record_known_fill_from_snapshot(
        store,
        intent=intent,
        local_order_id=order_id,
        snapshot=snapshot,
    )

    assert first["fill_observation_status"] == "KNOWN_FILL_RECORDED"
    assert first["recorded_execution_count"] == 1
    assert first["joined_commission_count"] == 1
    assert second["fill_observation_status"] == "KNOWN_FILL_IDEMPOTENT"
    assert second["idempotent_execution_count"] == 1
    assert second["unknown_execution_imports"] == 0
    assert second["broker_write_calls"] == 0
    assert store.counts()["execution_count"] == 1
    assert store.counts()["commission_count"] == 1
    assert store.active_local_order_count() == 0


def test_known_fill_snapshot_blocks_identity_mismatch_and_keeps_partial_active(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    intent, _ = prepare_intent(
        tmp_path,
        _config(),
        con_id=8677881,
        side="BUY",
        quantity=Decimal("1"),
        limit_price=Decimal("90"),
        reason="partial fill observation",
    )
    store.register_intent(intent.__dict__)
    order_id = 54321
    store.allocate_order_id(order_id, intent.intent_id)
    store.mark_order_id_used(order_id)
    store.append_event(
        intent.intent_id,
        "PLACE_ORDER_CALLED_ONCE",
        {"order_id_hash": "PRIVATE"},
    )
    partial = _snapshot_for_intent(
        intent,
        order_id=order_id,
        execution_id="EXECUTION-HASHED-PARTIAL",
        quantity="0.4",
        include_commission=False,
    )
    result = record_known_fill_from_snapshot(
        store,
        intent=intent,
        local_order_id=order_id,
        snapshot=partial,
    )
    assert (
        result["fill_observation_status"]
        == "KNOWN_FILL_RECORDED_COMMISSION_PENDING"
    )
    assert result["pending_commission_count"] == 1
    assert store.active_local_order_count() == 1

    mismatch = _snapshot_for_intent(
        intent,
        order_id=order_id,
        execution_id="EXECUTION-HASHED-WRONG",
        quantity="0.1",
    )
    mismatch["executions"]["executions"][0]["con_id"] = 999
    blocked = record_known_fill_from_snapshot(
        store,
        intent=intent,
        local_order_id=order_id,
        snapshot=mismatch,
    )
    assert blocked["status"] == "NO_GO"
    assert (
        blocked["fill_observation_status"]
        == "KNOWN_ORDER_EXECUTION_IDENTITY_MISMATCH"
    )


def test_missing_execution_id_blocked(tmp_path: Path) -> None:
    assert record_fill_execution(_store(tmp_path), _fill("", side="BUY", quantity="1", price="90"))["execution_status"] == "EXECUTION_ID_MISSING_BLOCKED"


def test_commission_before_after_duplicate_orphan_and_grace(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert record_execution_commission(store, execution_id="TEST-EXEC-C1", commission=Decimal("0.01"))["commission_status"] == "COMMISSION_PENDING"
    assert record_fill_execution(store, _fill("TEST-EXEC-C1", side="BUY", quantity="1", price="90"))["execution_status"] == "EXECUTION_ACCEPTED"
    assert commission_join_audit(store)["joined_count"] == 1

    assert record_execution_commission(store, execution_id="TEST-EXEC-C1", commission=Decimal("0.01"))["commission_status"] == "COMMISSION_DUPLICATE_IGNORED"
    assert record_execution_commission(store, execution_id="TEST-EXEC-ORPHAN", commission=Decimal("0.01"), allow_pending=False)["commission_status"] == "COMMISSION_ORPHAN_QUARANTINED"

    pending_store = _store(tmp_path / "pending")
    record_fill_execution(pending_store, _fill("TEST-EXEC-PENDING", side="BUY", quantity="1", price="90"))
    assert commission_join_audit(pending_store, grace_expired=True)["grace_status"] == "COMMISSION_GRACE_EXPIRED"


def test_full_and_partial_closing_sell_realized_pnl_and_no_negative_position(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record_fill_execution(store, _fill("TEST-EXEC-BUY", side="BUY", quantity="1", price="90"))
    record_execution_commission(store, execution_id="TEST-EXEC-BUY", commission=Decimal("0.01"))
    record_fill_execution(store, _fill("TEST-EXEC-SELL", side="SELL", quantity="1", price="92"))
    record_execution_commission(store, execution_id="TEST-EXEC-SELL", commission=Decimal("0.01"))

    closed = project_position_from_store(store)

    assert closed["position"]["long_quantity"] == "0"
    assert closed["position"]["position_status"] == "CLOSED"
    assert Decimal(closed["position"]["realized_pnl_local"]) == Decimal("1.98")

    partial = _store(tmp_path / "partial")
    record_fill_execution(partial, _fill("TEST-EXEC-PBUY", side="BUY", quantity="1", price="90"))
    record_execution_commission(partial, execution_id="TEST-EXEC-PBUY", commission=Decimal("0.01"))
    record_fill_execution(partial, _fill("TEST-EXEC-PSELL", side="SELL", quantity="0.4", price="92"))
    record_execution_commission(partial, execution_id="TEST-EXEC-PSELL", commission=Decimal("0.004"))
    partial_projection = project_position_from_store(partial)
    assert partial_projection["position"]["long_quantity"] == "0.6"
    assert partial_projection["position"]["position_status"] == "PARTIALLY_OPEN"

    no_position = _store(tmp_path / "no_position")
    record_fill_execution(no_position, _fill("TEST-EXEC-NO-POS", side="SELL", quantity="1", price="92"))
    assert project_position_from_store(no_position)["projection_status"] == "NEGATIVE_POSITION_BLOCKED"


def test_closing_sell_risk_gate_statuses(tmp_path: Path) -> None:
    config = _config()
    intent, _ = prepare_intent(tmp_path, config, con_id=8677881, side="SELL", quantity=Decimal("1"), limit_price=Decimal("92"), reason="offline close")

    assert evaluate_closing_sell_risk(intent, config=config, local_long_quantity=Decimal("1"), broker_long_quantity=Decimal("1"), broker_position_snapshot_complete=True, local_position_reconciled=True)["risk_status"] == "CLOSING_SELL_ALLOWED"
    assert evaluate_closing_sell_risk(intent, config=config, local_long_quantity=Decimal("0"), broker_long_quantity=Decimal("0"), broker_position_snapshot_complete=True, local_position_reconciled=True)["risk_status"] == "SELL_WITHOUT_POSITION_BLOCKED"
    assert evaluate_closing_sell_risk(intent, config=config, local_long_quantity=Decimal("0.5"), broker_long_quantity=Decimal("0.5"), broker_position_snapshot_complete=True, local_position_reconciled=True)["risk_status"] == "SELL_EXCEEDS_RECONCILED_POSITION"
    assert evaluate_closing_sell_risk(intent, config=config, local_long_quantity=Decimal("1"), broker_long_quantity=Decimal("1"), broker_position_snapshot_complete=False, local_position_reconciled=True)["risk_status"] == "POSITION_SNAPSHOT_INCOMPLETE"
    assert evaluate_closing_sell_risk(intent, config=config, local_long_quantity=Decimal("1"), broker_long_quantity=Decimal("0.5"), broker_position_snapshot_complete=True, local_position_reconciled=False)["risk_status"] == "LOCAL_BROKER_POSITION_MISMATCH"
    assert evaluate_closing_sell_risk(intent, config=config, local_long_quantity=Decimal("1"), broker_long_quantity=Decimal("1"), broker_position_snapshot_complete=True, local_position_reconciled=True, same_account_fingerprint=False)["risk_status"] == "LOCAL_BROKER_POSITION_MISMATCH"
    assert evaluate_closing_sell_risk(intent, config=config, local_long_quantity=Decimal("1"), broker_long_quantity=Decimal("1"), broker_position_snapshot_complete=True, local_position_reconciled=True, closing_orders_today=1)["risk_status"] == "CLOSING_ORDER_DAILY_LIMIT_BLOCKED"


def test_partial_fill_remainder_requires_existing_order_or_later_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    config = _config()
    record_fill_execution(store, _fill("TEST-EXEC-POLICY-BUY", side="BUY", quantity="1", price="90"))
    record_fill_execution(store, _fill("TEST-EXEC-POLICY-PARTIAL-SELL", side="SELL", quantity="0.4", price="92"))
    projection = project_position_from_store(store)
    assert projection["position"]["long_quantity"] == "0.6"
    intent, _ = prepare_intent(tmp_path, config, con_id=8677881, side="SELL", quantity=Decimal("1"), limit_price=Decimal("92"), reason="offline remainder close")

    same_day = evaluate_closing_sell_risk(
        intent,
        config=config,
        local_long_quantity=Decimal("0.6"),
        broker_long_quantity=Decimal("0.6"),
        broker_position_snapshot_complete=True,
        local_position_reconciled=True,
        closing_orders_today=1,
    )
    later_session = evaluate_closing_sell_risk(
        intent,
        config=config,
        local_long_quantity=Decimal("0.6"),
        broker_long_quantity=Decimal("0.6"),
        broker_position_snapshot_complete=True,
        local_position_reconciled=True,
        closing_orders_today=0,
    )

    assert same_day["risk_status"] == "CLOSING_ORDER_DAILY_LIMIT_BLOCKED"
    assert later_session["risk_status"] == "SELL_EXCEEDS_RECONCILED_POSITION"
    assert project_position_from_store(store)["position"]["long_quantity"] == "0.6"


def test_submit_runtime_blocks_second_closing_order_before_writer_connect(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    config = _config()
    record_fill_execution(store, _fill("TEST-EXEC-RUNTIME-BUY", side="BUY", quantity="1", price="90"))
    intent, _ = prepare_intent(tmp_path, config, con_id=8677881, side="SELL", quantity=Decimal("1"), limit_price=Decimal("92"), reason="offline closing limit")
    store.register_intent(intent.__dict__)
    store.append_event(intent.intent_id, "PLACE_ORDER_CALLED_ONCE", {"order_id_hash": "OFFLINE-FIRST-SELL"})
    monkeypatch.setattr(store, "find_unconsumed_approval", lambda *_args: {"approval_id": "OFFLINE", "intent_hash": "OFFLINE"})
    monkeypatch.setattr(
        phase9_audit_module,
        "_phase8_observation_counts",
        lambda _root: {"status": "GO", "same_client_open_order_count": 0, "all_api_open_order_count": 0, "position_count": 1},
    )
    monkeypatch.setattr(phase9_audit_module, "_phase8_private_position_quantity", lambda _root, _con_id: Decimal("1"))
    monkeypatch.setattr(phase9_audit_module, "connect_phase9_writer", lambda _config: (_ for _ in ()).throw(AssertionError("writer connection must not occur")))

    result = phase9_audit_module._submit_runtime(project_root=tmp_path, config=config, store=store, intent=intent)

    assert result["risk_status"] == "CLOSING_ORDER_DAILY_LIMIT_BLOCKED"
    assert result["paper_place_order_calls"] == 0
    assert store.event_type_count("PLACE_ORDER_CALLED_ONCE") == 1


def test_reconciliation_open_long_empty_mismatch_and_unknown_execution(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record_fill_execution(store, _fill("TEST-EXEC-BUY", side="BUY", quantity="1", price="90"))
    record_execution_commission(store, execution_id="TEST-EXEC-BUY", commission=Decimal("0.01"))
    assert reconcile_paper_state(store, broker_execution_count=1, broker_commission_count=1, broker_position_count=1, local_position_quantity=Decimal("1"), broker_position_quantity=Decimal("1"))["reconciliation_status"] == "PAPER_RECONCILED_OPEN_LONG"
    assert reconcile_position_projection(local_quantity=Decimal("1"), broker_quantity=Decimal("0.5"))["position_reconciliation_status"] == "POSITION_QUANTITY_MISMATCH"
    assert reconcile_position_projection(local_quantity=Decimal("1"), broker_quantity=Decimal("1"), unknown_broker_execution=True)["position_reconciliation_status"] == "POSITION_RECONCILIATION_BLOCKED"


def test_reconciliation_does_not_treat_incomplete_execution_history_as_zero(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    record_fill_execution(
        store,
        _fill(
            "TEST-EXEC-HISTORICAL",
            side="BUY",
            quantity="1",
            price="90",
        ),
    )
    result = reconcile_paper_state(
        store,
        broker_execution_count=0,
        broker_position_count=1,
        execution_history_complete=False,
        local_position_quantity=Decimal("1"),
        broker_position_quantity=Decimal("1"),
    )
    assert result["status"] == "GO"
    assert result["reconciliation_status"] == "PAPER_RECONCILED_OPEN_LONG"
    assert result["broker_execution_count"] == 0
    assert result["execution_count_comparison_applied"] is False


def test_phase9_reconcile_uses_projected_and_private_position_quantities(
    tmp_path: Path, monkeypatch
) -> None:
    store = PaperExecutionStore(
        Phase9Layout.from_project_root(tmp_path).db_path
    )
    store.initialize()
    record_fill_execution(
        store,
        _fill("TEST-EXEC-RECONCILE", side="BUY", quantity="1", price="90"),
    )
    record_execution_commission(
        store,
        execution_id="TEST-EXEC-RECONCILE",
        commission=Decimal("0.01"),
    )
    monkeypatch.setattr(
        phase9_audit_module,
        "_phase8_observation_counts",
        lambda _root: {
            "status": "GO",
            "snapshot_status": "COMPLETE",
            "same_client_open_order_count": 0,
            "all_api_open_order_count": 0,
            "position_count": 1,
            "execution_count": 1,
            "commission_count": 1,
            "execution_scope_current": True,
            "read_only_request_counters": {},
            "broker_write_counters": {},
        },
    )
    monkeypatch.setattr(
        phase9_audit_module,
        "_phase8_open_order_identity_audit",
        lambda _root, _store: {
            "status": "GO",
            "matched_open_order_count": 0,
            "unknown_broker_open_order_count": 0,
            "missing_local_open_order_count": 0,
        },
    )
    monkeypatch.setattr(
        phase9_audit_module,
        "_phase8_private_position_quantity",
        lambda _root, _con_id: Decimal("1"),
    )

    result = phase9_audit_module.phase9_reconcile(tmp_path)

    assert result["status"] == "GO"
    assert result["reconciliation_status"] == "PAPER_RECONCILED_OPEN_LONG"
    assert result["local_position_quantity"] == "1"
    assert result["broker_position_quantity"] == "1"


def test_phase9_prepare_sell_uses_reconciled_position_before_approval(
    tmp_path: Path, monkeypatch
) -> None:
    store = PaperExecutionStore(
        Phase9Layout.from_project_root(tmp_path).db_path
    )
    store.initialize()
    record_fill_execution(
        store,
        _fill("TEST-EXEC-PREPARE-SELL", side="BUY", quantity="1", price="90"),
    )
    monkeypatch.setattr(
        phase9_audit_module,
        "load_paper_writer_config",
        lambda _root, _env: (_config(), []),
    )
    monkeypatch.setattr(
        phase9_audit_module,
        "_phase8_observation_counts",
        lambda _root: {"status": "GO"},
    )
    monkeypatch.setattr(
        phase9_audit_module,
        "_phase8_private_position_quantity",
        lambda _root, _con_id: Decimal("1"),
    )

    result = phase9_audit_module.prepare(
        tmp_path,
        ".env.ibkr",
        con_id=8677881,
        side="SELL",
        quantity=Decimal("1"),
        limit_price=Decimal("84"),
        reason="offline close",
    )

    assert result["status"] == "GO"
    assert result["prepare_status"] == "AWAITING_MANUAL_APPROVAL"
    assert result["risk_status"] == "APPROVAL_REQUIRED"
    assert result["closing_risk_status"] == "CLOSING_SELL_ALLOWED"


def test_restart_replay_is_deterministic(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record_fill_execution(store, _fill("TEST-EXEC-BUY", side="BUY", quantity="1", price="90"))
    record_execution_commission(store, execution_id="TEST-EXEC-BUY", commission=Decimal("0.01"))
    first = project_position_from_store(store)
    second = project_position_from_store(store)
    assert first == second


def test_phase9_0_1_cli_artifacts_and_readiness_preserve_canary_a(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        phase9_audit_module,
        "_canary_a_integrity",
        lambda _layout: {
            "status": "GO",
            "marker": "PHASE9_CANARY_A_SUBMIT_CANCEL_SAFE_DONE",
            "evidence_status": "CANARY_A_EVIDENCE_GO",
            "paper_place_order_calls": 1,
            "paper_cancel_order_calls": 1,
        },
    )
    store = PaperExecutionStore(tmp_path / "data" / "execution" / "phase9" / "private" / "paper_execution.sqlite3")
    store.initialize()
    store.append_event("CANARY_A", "PLACE_ORDER_CALLED_ONCE", {"order_id_hash": "H"})
    store.append_event("CANARY_A", "CANCEL_ORDER_CALLED_ONCE", {"order_id_hash": "H"})
    phase7 = _write_dummy(tmp_path / "data" / "execution" / "phase7" / "private" / "ledger.sqlite3")
    phase8 = _write_dummy(tmp_path / "data" / "broker" / "phase8" / "private" / "broker_observation.sqlite3")
    phase8_2 = _write_dummy(tmp_path / "data" / "shadow" / "phase8_2" / "private" / "shadow.sqlite3")

    assert main.main(["ibkr", "phase9", "fill-close-audit"]) == 0
    assert main.main(["ibkr", "phase9", "canary-b-readiness"]) == 0

    assert (tmp_path / "output" / "ibkr" / "phase9" / "fill-adoption-audit.json").exists()
    assert (tmp_path / "output" / "ibkr" / "phase9" / "canary-b-readiness.json").exists()
    assert (tmp_path / "output" / "ibkr" / "phase9" / "phase9-0-1-freeze-status.json").exists()
    assert phase7.read_text(encoding="utf-8") == "fixture"
    assert phase8.read_text(encoding="utf-8") == "fixture"
    assert phase8_2.read_text(encoding="utf-8") == "fixture"
    assert store.event_type_count("PLACE_ORDER_CALLED_ONCE") == 1
    assert store.event_type_count("CANCEL_ORDER_CALLED_ONCE") == 1


def _store(tmp_path: Path) -> PaperExecutionStore:
    store = PaperExecutionStore(tmp_path / "paper_execution.sqlite3")
    store.initialize()
    return store


def _fill(exec_id: str, *, side: str, quantity: str, price: str) -> FillExecution:
    return FillExecution(
        exec_id=exec_id,
        intent_id=f"{side}_INTENT",
        account_fingerprint="DU_TEST_ACCOUNT",
        perm_id=f"TEST-PERM-{side}",
        broker_order_id=f"TEST-ORDER-{side}",
        con_id=8677881,
        symbol="ON",
        currency="USD",
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        execution_time="2026-07-21T16:00:00+00:00",
        submitted_quantity=Decimal("1"),
        fx_rate=Decimal("0.92"),
    )


def _config() -> PaperWriterConfig:
    return PaperWriterConfig(
        host="127.0.0.1",
        port=7497,
        phase1_client_id=17,
        observer_client_id=817,
        writer_client_id=917,
        writer_enabled=True,
        approved_account_fingerprint="DU_TEST_ACCOUNT",
        observed_account_fingerprint="DU_TEST_ACCOUNT",
        max_order_notional_eur=Decimal("1000"),
        max_quantity=Decimal("1"),
        max_open_orders=1,
        max_positions=1,
        max_new_orders_per_day=1,
        max_closing_orders_per_day=1,
        approval_ttl_seconds=300,
        callback_timeout_seconds=15.0,
        reconciliation_timeout_seconds=30.0,
        live_trading_enabled=False,
        allow_order_transmission=False,
    )


def _write_dummy(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture", encoding="utf-8")
    return path


def _snapshot_for_intent(
    intent,
    *,
    order_id: int,
    execution_id: str,
    quantity: str,
    include_commission: bool = True,
) -> dict:
    execution = {
        "execution_id": execution_id,
        "broker_order_id": stable_hash(
            {"broker_order_id": order_id}
        )[:24],
        "perm_id": "PRIVATE-PERM-ID",
        "account_fingerprint": "MASKED-ACCOUNT",
        "con_id": intent.con_id,
        "symbol": intent.symbol,
        "side": "BOT",
        "quantity": quantity,
        "price": "89.50",
        "execution_time": "2026-07-28T13:30:00+00:00",
    }
    commissions = (
        [
            {
                "execution_id": execution_id,
                "commission": "0.01",
                "currency": intent.currency,
            }
        ]
        if include_commission
        else []
    )
    return {
        "executions": {
            "executions": [execution],
            "commissions": commissions,
        }
    }
