from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import main
from stocks.ibkr.callbacks import CallbackState
from stocks.ibkr.paper_execution.approvals import approval_challenge, approve_intent, cancel_challenge, consume_approval, prepare_cancel_approval
from stocks.ibkr.paper_execution.authority import authority_contract, phase9_authority_status
from stocks.ibkr.paper_execution.callbacks import CallbackAuditState, accept_callback, callback_audit_payload
from stocks.ibkr.paper_execution.cancellation import block_global_cancel, cancel_known_order_once
from stocks.ibkr.paper_execution.commissions import record_commission
from stocks.ibkr.paper_execution.config import (
    _private_observed_account_fingerprints,
    load_paper_writer_config,
)
from stocks.ibkr.paper_execution.executions import record_execution
from stocks.ibkr.paper_execution.models import model_to_jsonable
from stocks.ibkr.paper_execution.order_ids import allocate_order_id
from stocks.ibkr.paper_execution.reconciliation import reconcile_paper_state
from stocks.ibkr.paper_execution.risk import evaluate_risk, prepare_intent
from stocks.ibkr.paper_execution.state_mapping import map_order_status
from stocks.ibkr.paper_execution.storage import PaperExecutionStore
from stocks.ibkr.paper_execution.submission import submit_place_order_once
from stocks.ibkr.paper_execution import audit as phase9_audit_module


def test_phase9_schema_cli(capsys) -> None:
    assert main.main(["ibkr", "phase9", "schema"]) == 0
    out = capsys.readouterr().out
    assert "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO" in out
    assert '"manual_approval_required": true' in out


def test_paper_writer_config_preflight_nonzero_unique_and_fingerprint(tmp_path) -> None:
    _write_env(tmp_path)
    config, errors = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    assert errors == []
    assert config.writer_client_id != 0
    assert config.writer_client_id not in {config.phase1_client_id, config.observer_client_id}
    assert config.approved_account_fingerprint == config.observed_account_fingerprint

    _write_env(tmp_path, writer_id=0)
    _, errors = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert "WRITER_CLIENT_ID_ZERO_BLOCKED" in errors

    _write_env(tmp_path, writer_id=17)
    _, errors = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert "WRITER_CLIENT_ID_COLLISION_BLOCKED" in errors

    _write_env(tmp_path, fingerprint="MISMATCH")
    _, errors = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert "PAPER_ACCOUNT_FINGERPRINT_MISMATCH" in errors

    _write_env(tmp_path, fingerprint="PHASE8_1_BASELINE_APPROVED")
    _, errors = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert "PAPER_ACCOUNT_FINGERPRINT_MISMATCH" in errors


def test_paper_writer_config_uses_latest_observed_fingerprint(tmp_path) -> None:
    old_fingerprint = "b" * 64
    approved_fingerprint = "a" * 64
    _write_env(
        tmp_path,
        fingerprint=approved_fingerprint,
        observed_fingerprint=old_fingerprint,
    )
    _write_private_phase8_fingerprint(tmp_path, approved_fingerprint)

    config, errors = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")

    assert errors == []
    assert config is not None
    assert config.observed_account_fingerprint == approved_fingerprint
    assert _private_observed_account_fingerprints(tmp_path) == {
        approved_fingerprint
    }


def test_paper_writer_config_blocks_conflicting_latest_observation_stores(
    tmp_path,
) -> None:
    approved_fingerprint = "a" * 64
    conflicting_fingerprint = "b" * 64
    _write_env(
        tmp_path,
        fingerprint=approved_fingerprint,
        observed_fingerprint=approved_fingerprint,
    )
    phase8_db = (
        tmp_path
        / "data"
        / "broker"
        / "phase8"
        / "private"
        / "broker_observation.sqlite3"
    )
    _write_fingerprint_snapshot(phase8_db, conflicting_fingerprint)

    _, errors = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")

    assert "PAPER_ACCOUNT_FINGERPRINT_MISMATCH" in errors


def test_authority_target_and_non_canary_blocked() -> None:
    assert authority_contract(enabled=True)["execution_authority"] == "MANUAL_PAPER_CANARY"
    assert phase9_authority_status("MANUAL_PAPER_CANARY")["status"] == "GO"
    assert phase9_authority_status("LIVE")["decision_code"] == "AUTHORITY_NOT_GRANTED"


def test_manual_intent_approval_exact_expiry_one_time_and_change_block(tmp_path) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "data" / "execution" / "phase9" / "private" / "paper_execution.sqlite3")
    store.initialize()
    intent, _ = prepare_intent(tmp_path, config, con_id=756733, side="BUY", quantity=Decimal("1"), limit_price=Decimal("50"), reason="manual paper execution canary")
    assert store.register_intent(intent.__dict__) == "INTENT_REGISTERED"

    assert approval_challenge(intent).startswith("APPROVE PAPER CANARY")
    bad = approve_intent(store, intent, "APPROVE PAPER CANARY WRONG", ttl_seconds=300)
    assert bad["approval_status"] == "APPROVAL_MISMATCH"
    good = approve_intent(store, intent, approval_challenge(intent), ttl_seconds=300)
    assert good["approval_status"] == "APPROVED_FOR_SINGLE_SUBMISSION"
    assert consume_approval(store, intent, str(good["approval_id"]), str(good["intent_hash"]))["approval_status"] == "APPROVED_FOR_SINGLE_SUBMISSION"
    assert consume_approval(store, intent, str(good["approval_id"]), str(good["intent_hash"]))["approval_status"] == "APPROVAL_ALREADY_USED"

    changed = intent.__class__(**{**intent.__dict__, "limit_price": Decimal("101")})
    assert consume_approval(store, changed, "anything", str(good["intent_hash"]))["approval_status"] == "INTENT_CHANGED_AFTER_APPROVAL_BLOCKED"


def test_expired_unsubmitted_intent_rolls_once_in_mutable_command_layer(
    tmp_path,
) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(
        tmp_path
        / "data"
        / "execution"
        / "phase9"
        / "private"
        / "paper_execution.sqlite3"
    )
    store.initialize()
    original, _ = prepare_intent(
        tmp_path,
        config,
        con_id=8677881,
        side="BUY",
        quantity=Decimal("1"),
        limit_price=Decimal("95"),
        reason="expired operator attempt",
    )
    expired = replace(original, expires_at="2020-01-01T00:00:00+00:00")
    assert store.register_intent(expired.__dict__) == "INTENT_REGISTERED"

    proposed, _ = prepare_intent(
        tmp_path,
        config,
        con_id=8677881,
        side="BUY",
        quantity=Decimal("1"),
        limit_price=Decimal("95"),
        reason="fresh operator attempt",
    )
    replacement, risk = (
        phase9_audit_module._rollover_expired_unsubmitted_intent(
            store,
            proposed,
            config=config,
        )
    )
    assert replacement.intent_id != expired.intent_id
    assert replacement.economic_order_key != expired.economic_order_key
    assert risk["risk_status"] == "APPROVAL_REQUIRED"
    assert store.register_intent(replacement.__dict__) == "INTENT_REGISTERED"

    repeated, repeated_risk = (
        phase9_audit_module._rollover_expired_unsubmitted_intent(
            store,
            proposed,
            config=config,
        )
    )
    assert repeated.intent_id == replacement.intent_id
    assert repeated_risk["risk_status"] == "APPROVAL_REQUIRED"
    assert store.register_intent(repeated.__dict__) == "INTENT_IDEMPOTENT"
    assert store.event_type_count("PLACE_ORDER_CALLED_ONCE") == 0


def test_expired_submitted_intent_never_rolls(tmp_path) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    original, _ = prepare_intent(
        tmp_path,
        config,
        con_id=8677881,
        side="BUY",
        quantity=Decimal("1"),
        limit_price=Decimal("95"),
        reason="submitted operator attempt",
    )
    expired = replace(original, expires_at="2020-01-01T00:00:00+00:00")
    store.register_intent(expired.__dict__)
    store.append_event(
        expired.intent_id,
        "PLACE_ORDER_CALLED_ONCE",
        {"order_id_hash": "TEST"},
    )

    proposed, _ = prepare_intent(
        tmp_path,
        config,
        con_id=8677881,
        side="BUY",
        quantity=Decimal("1"),
        limit_price=Decimal("95"),
        reason="must not roll",
    )
    resolved, risk = phase9_audit_module._rollover_expired_unsubmitted_intent(
        store,
        proposed,
        config=config,
    )

    assert resolved.intent_id == expired.intent_id
    assert risk["risk_status"] == "APPROVAL_REQUIRED"
    assert store.counts()["intent_count"] == 1


def test_manual_cancel_approval_and_cancel_blocks_other_client_global_cancel(tmp_path) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    intent, _ = prepare_intent(tmp_path, config, con_id=756733, side="BUY", quantity=Decimal("1"), limit_price=Decimal("50"), reason="manual")
    cancel_record = prepare_cancel_approval(store, intent, ttl_seconds=300)
    assert cancel_record["cancel_approval_status"] == "CANCEL_APPROVAL_REQUIRED"
    assert cancel_challenge(intent).startswith("CANCEL PAPER CANARY")

    class App:
        calls = 0

        def cancelOrder(self, _order_id: int) -> None:
            self.calls += 1

    app = App()
    assert cancel_known_order_once(app, order_id=1, writer_client_matches=False, approved=True, store=store, intent_id=intent.intent_id)["cancel_status"] == "CANCEL_CLIENT_ID_MISMATCH_BLOCKED"
    assert cancel_known_order_once(app, order_id=1, writer_client_matches=True, approved=False, store=store, intent_id=intent.intent_id)["cancel_status"] == "CANCEL_APPROVAL_REQUIRED"
    assert cancel_known_order_once(app, order_id=1, writer_client_matches=True, approved=True, store=store, intent_id=intent.intent_id)["cancel_status"] == "CANCEL_REQUESTED"
    assert app.calls == 1
    assert block_global_cancel()["cancel_status"] == "GLOBAL_CANCEL_BLOCKED"


def test_risk_engine_limits_manual_only_long_limit_day_rth(tmp_path) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    intent, _ = prepare_intent(tmp_path, config, con_id=756733, side="BUY", quantity=Decimal("1"), limit_price=Decimal("50"), reason="manual")
    assert evaluate_risk(intent, config=config, store=store, approval_valid=True)["risk_status"] == "PAPER_RISK_APPROVED_MANUAL_CANARY"
    assert evaluate_risk(intent.__class__(**{**intent.__dict__, "intent_source": "STRATEGY"}), config=config, store=store, approval_valid=True)["risk_status"] == "STRATEGY_INTENT_BLOCKED"
    assert evaluate_risk(intent.__class__(**{**intent.__dict__, "order_type": "MARKET"}), config=config, store=store, approval_valid=True)["risk_status"] == "ORDER_TYPE_BLOCKED"
    assert evaluate_risk(intent.__class__(**{**intent.__dict__, "time_in_force": "GTC"}), config=config, store=store, approval_valid=True)["risk_status"] == "TIME_IN_FORCE_BLOCKED"
    assert evaluate_risk(intent.__class__(**{**intent.__dict__, "outside_rth": True}), config=config, store=store, approval_valid=True)["risk_status"] == "OUTSIDE_RTH_BLOCKED"
    assert evaluate_risk(intent.__class__(**{**intent.__dict__, "quantity": Decimal("2")}), config=config, store=store, approval_valid=True)["risk_status"] == "CANARY_INSTRUMENT_NOT_SUITABLE"
    assert evaluate_risk(intent.__class__(**{**intent.__dict__, "limit_price": Decimal("0")}), config=config, store=store, approval_valid=True)["risk_status"] == "LIMIT_PRICE_INVALID"
    assert evaluate_risk(intent.__class__(**{**intent.__dict__, "estimated_notional_eur": Decimal("251")}), config=config, store=store, approval_valid=True)["risk_status"] == "ORDER_NOTIONAL_EXCEEDED"
    sell = intent.__class__(**{**intent.__dict__, "side": "SELL"})
    assert evaluate_risk(sell, config=config, store=store, approval_valid=True, existing_long_quantity=Decimal("0"))["risk_status"] == "SELL_WITHOUT_POSITION_BLOCKED"
    assert evaluate_risk(intent, config=config, store=store, approval_valid=True, open_orders=1)["risk_status"] == "OPEN_ORDER_LIMIT_EXCEEDED"
    assert evaluate_risk(intent, config=config, store=store, approval_valid=True, open_positions=1)["risk_status"] == "POSITION_LIMIT_EXCEEDED"
    assert evaluate_risk(intent, config=config, store=store, approval_valid=True, new_orders_today=1)["risk_status"] == "OPENING_ORDER_DAILY_LIMIT_BLOCKED"
    assert evaluate_risk(intent, config=config, store=store, approval_valid=True, reconciliation_go=False)["risk_status"] == "RECONCILIATION_BLOCKED"
    assert evaluate_risk(intent, config=config, store=store, approval_valid=True, kill_switch_armed=False)["risk_status"] == "KILL_SWITCH_ACTIVE"


def test_phase9_opening_and_closing_daily_limits_are_separate(tmp_path) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    buy, _ = prepare_intent(tmp_path, config, con_id=8677881, side="BUY", quantity=Decimal("1"), limit_price=Decimal("70"), reason="opening")
    sell, _ = prepare_intent(tmp_path, config, con_id=8677881, side="SELL", quantity=Decimal("1"), limit_price=Decimal("90"), reason="closing")

    assert evaluate_risk(buy, config=config, store=store, approval_valid=True, new_orders_today=0)["risk_status"] == "PAPER_RISK_APPROVED_MANUAL_CANARY"
    assert evaluate_risk(buy, config=config, store=store, approval_valid=True, new_orders_today=1)["risk_status"] == "OPENING_ORDER_DAILY_LIMIT_BLOCKED"
    assert evaluate_risk(sell, config=config, store=store, approval_valid=True, existing_long_quantity=Decimal("1"), closing_orders_today=0, new_orders_today=1)["risk_status"] == "PAPER_RISK_APPROVED_MANUAL_CANARY"
    assert evaluate_risk(sell, config=config, store=store, approval_valid=True, existing_long_quantity=Decimal("1"), closing_orders_today=1)["risk_status"] == "CLOSING_ORDER_DAILY_LIMIT_BLOCKED"
    assert evaluate_risk(sell, config=config, store=store, approval_valid=True, existing_long_quantity=Decimal("0"))["risk_status"] == "SELL_WITHOUT_POSITION_BLOCKED"
    assert evaluate_risk(sell, config=config, store=store, approval_valid=True, existing_long_quantity=Decimal("0.5"))["risk_status"] == "SHORT_POSITION_BLOCKED"


def test_submitted_order_counts_are_separated_by_side_and_session(tmp_path) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    buy, _ = prepare_intent(tmp_path, config, con_id=8677881, side="BUY", quantity=Decimal("1"), limit_price=Decimal("70"), reason="offline opening count")
    sell, _ = prepare_intent(tmp_path, config, con_id=8677881, side="SELL", quantity=Decimal("1"), limit_price=Decimal("92"), reason="offline closing count")
    store.register_intent(model_to_jsonable(buy))
    store.register_intent(model_to_jsonable(sell))
    store.append_event(buy.intent_id, "PLACE_ORDER_CALLED_ONCE", {"order_id_hash": "OFFLINE-BUY"})
    store.append_event(sell.intent_id, "PLACE_ORDER_CALLED_ONCE", {"order_id_hash": "OFFLINE-SELL"})

    assert store.submitted_order_count_for_session(side="BUY", session_date=buy.session_date) == 1
    assert store.submitted_order_count_for_session(side="SELL", session_date=sell.session_date) == 1
    assert store.submitted_order_count_for_session(side="BUY", session_date="1900-01-01") == 0

    store.append_event(
        sell.intent_id,
        "BROKER_SUBMISSION_ACK_TIMEOUT",
        {"timeout_seconds": 1},
    )
    assert (
        store.submitted_order_count_for_session(
            side="SELL",
            session_date=sell.session_date,
        )
        == 0
    )
    assert store.active_local_order_count() == 1

    store.append_event(
        sell.intent_id,
        "BROKER_SUBMISSION_WARNING_RECLASSIFIED",
        {"broker_status": "PreSubmitted"},
    )
    assert (
        store.submitted_order_count_for_session(
            side="SELL",
            session_date=sell.session_date,
        )
        == 1
    )
    assert store.active_local_order_count() == 2


def test_submit_runtime_blocks_second_opening_before_writer_connect(tmp_path, monkeypatch) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    intent, _ = prepare_intent(tmp_path, config, con_id=8677881, side="BUY", quantity=Decimal("1"), limit_price=Decimal("70"), reason="offline opening limit")
    store.register_intent(model_to_jsonable(intent))
    store.append_event(intent.intent_id, "PLACE_ORDER_CALLED_ONCE", {"order_id_hash": "OFFLINE-FIRST-BUY"})
    monkeypatch.setattr(store, "find_unconsumed_approval", lambda *_args: {"approval_id": "OFFLINE", "intent_hash": "OFFLINE"})
    monkeypatch.setattr(
        phase9_audit_module,
        "_phase8_observation_counts",
        lambda _root: {"status": "GO", "same_client_open_order_count": 0, "all_api_open_order_count": 0, "position_count": 0},
    )
    monkeypatch.setattr(phase9_audit_module, "connect_phase9_writer", lambda _config: (_ for _ in ()).throw(AssertionError("writer connection must not occur")))

    result = phase9_audit_module._submit_runtime(project_root=tmp_path, config=config, store=store, intent=intent)

    assert result["risk_status"] == "OPENING_ORDER_DAILY_LIMIT_BLOCKED"
    assert result["paper_place_order_calls"] == 0
    assert store.event_type_count("PLACE_ORDER_CALLED_ONCE") == 1


def test_order_id_management_submission_once_and_no_resubmission(tmp_path) -> None:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert allocate_order_id(store, broker_next_id=5, intent_id="I")["order_id_status"] == "ORDER_ID_READY"
    assert allocate_order_id(store, broker_next_id=5, intent_id="I2")["order_id_status"] == "ORDER_ID_REGRESSION_BLOCKED"

    class App:
        calls = 0

        def placeOrder(self, _order_id: int, _contract: object, _order: object) -> None:
            self.calls += 1

    app = App()
    assert submit_place_order_once(app, order_id=5, contract=object(), order=object(), store=store, intent_id="I")["submission_status"] == "SUBMISSION_ALLOWED_ONCE"
    assert submit_place_order_once(app, order_id=5, contract=object(), order=object(), store=store, intent_id="I")["submission_status"] == "ORDER_ID_ALREADY_USED"
    assert app.calls == 1


def test_submission_waits_for_broker_acknowledgement(tmp_path) -> None:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert (
        allocate_order_id(store, broker_next_id=5, intent_id="I")[
            "order_id_status"
        ]
        == "ORDER_ID_READY"
    )

    class App:
        def __init__(self) -> None:
            self.callback_state = CallbackState()

        def placeOrder(
            self,
            _order_id: int,
            _contract: object,
            _order: object,
        ) -> None:
            self.callback_state.record_event(
                "phase9_order_status",
                status="Submitted",
            )

    result = submit_place_order_once(
        App(),
        order_id=5,
        contract=object(),
        order=object(),
        store=store,
        intent_id="I",
        ack_timeout_seconds=1,
    )

    assert result["status"] == "GO"
    assert result["submission_status"] == "BROKER_SUBMISSION_ACKNOWLEDGED"
    assert store.event_type_count("BROKER_SUBMISSION_ACKNOWLEDGED") == 1


def test_submission_ack_timeout_is_fail_closed(tmp_path) -> None:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert (
        allocate_order_id(store, broker_next_id=5, intent_id="I")[
            "order_id_status"
        ]
        == "ORDER_ID_READY"
    )

    class App:
        def __init__(self) -> None:
            self.callback_state = CallbackState()

        def placeOrder(
            self,
            _order_id: int,
            _contract: object,
            _order: object,
        ) -> None:
            return None

    class Clock:
        value = 0.0

        def __call__(self) -> float:
            self.value += 0.05
            return self.value

    result = submit_place_order_once(
        App(),
        order_id=5,
        contract=object(),
        order=object(),
        store=store,
        intent_id="I",
        ack_timeout_seconds=0.1,
        sleeper=lambda _seconds: None,
        monotonic=Clock(),
    )

    assert result["status"] == "NO_GO"
    assert result["submission_status"] == "BROKER_SUBMISSION_ACK_TIMEOUT"
    assert store.event_type_count("BROKER_SUBMISSION_ACK_TIMEOUT") == 1


def test_submission_error_wins_over_provisional_status(tmp_path) -> None:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert (
        allocate_order_id(store, broker_next_id=5, intent_id="I")[
            "order_id_status"
        ]
        == "ORDER_ID_READY"
    )

    class App:
        def __init__(self) -> None:
            self.callback_state = CallbackState()

        def placeOrder(
            self,
            _order_id: int,
            _contract: object,
            _order: object,
        ) -> None:
            self.callback_state.record_event(
                "phase9_order_status",
                status="PendingSubmit",
            )
            self.callback_state.errors.append({"code": 201})

    result = submit_place_order_once(
        App(),
        order_id=5,
        contract=object(),
        order=object(),
        store=store,
        intent_id="I",
        ack_timeout_seconds=1,
    )

    assert result["status"] == "NO_GO"
    assert result["submission_status"] == "BROKER_SUBMISSION_REJECTED"
    assert result["broker_error_codes"] == ["201"]


def test_submission_late_error_wins_during_ack_settlement(tmp_path) -> None:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert (
        allocate_order_id(store, broker_next_id=5, intent_id="I")[
            "order_id_status"
        ]
        == "ORDER_ID_READY"
    )

    class App:
        def __init__(self) -> None:
            self.callback_state = CallbackState()

        def placeOrder(
            self,
            _order_id: int,
            _contract: object,
            _order: object,
        ) -> None:
            self.callback_state.record_event(
                "phase9_order_status",
                status="Submitted",
            )

    class Clock:
        value = 0.0

        def __call__(self) -> float:
            self.value += 0.01
            return self.value

    app = App()
    sleep_count = 0

    def inject_late_error(_seconds: float) -> None:
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            app.callback_state.errors.append({"code": 201})

    result = submit_place_order_once(
        app,
        order_id=5,
        contract=object(),
        order=object(),
        store=store,
        intent_id="I",
        ack_timeout_seconds=1,
        ack_settle_seconds=0.2,
        sleeper=inject_late_error,
        monotonic=Clock(),
    )

    assert result["status"] == "NO_GO"
    assert result["submission_status"] == "BROKER_SUBMISSION_REJECTED"
    assert result["broker_error_codes"] == ["201"]


def test_submission_399_with_presubmitted_is_acknowledged_warning(
    tmp_path,
) -> None:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert (
        allocate_order_id(store, broker_next_id=5, intent_id="I")[
            "order_id_status"
        ]
        == "ORDER_ID_READY"
    )

    class App:
        def __init__(self) -> None:
            self.callback_state = CallbackState()

        def placeOrder(
            self,
            _order_id: int,
            _contract: object,
            _order: object,
        ) -> None:
            self.callback_state.record_event(
                "phase9_order_status",
                status="PreSubmitted",
            )
            self.callback_state.errors.append({"code": 399})

    result = submit_place_order_once(
        App(),
        order_id=5,
        contract=object(),
        order=object(),
        store=store,
        intent_id="I",
        ack_timeout_seconds=1,
    )

    assert result["status"] == "GO"
    assert result["submission_status"] == "BROKER_SUBMISSION_ACKNOWLEDGED"
    assert result["broker_warning_codes"] == ["399"]


def test_status_mapping_callbacks_fills_and_commissions(tmp_path) -> None:
    assert map_order_status("Submitted") == "SUBMITTED_SIMULATED"
    assert map_order_status("Cancelled") == "CANCELLED"
    callbacks = CallbackAuditState()
    assert accept_callback(callbacks, "openOrder", {"status": "Submitted"})["classification"] == "CALLBACK_OK"
    assert accept_callback(callbacks, "openOrder", {"status": "Submitted"})["classification"] == "DUPLICATE_CALLBACK_IGNORED"
    assert accept_callback(callbacks, "execDetails", {}, out_of_order=True)["classification"] == "OUT_OF_ORDER_CALLBACK_BUFFERED"
    assert accept_callback(callbacks, "commissionReport", {}, late=True)["classification"] == "LATE_CALLBACK_QUARANTINED"
    assert accept_callback(callbacks, "orderStatus", {}, known_order=False)["classification"] == "ORPHAN_CALLBACK_BLOCKED"
    payload = callback_audit_payload(callbacks)
    assert payload["callback_types"]["openOrder"]["received"] == 2
    assert payload["callback_types"]["openOrder"]["duplicate"] == 1
    assert payload["callback_types"]["execDetails"]["out_of_order"] == 1
    assert payload["callback_types"]["commissionReport"]["late"] == 1
    assert payload["callback_types"]["orderStatus"]["orphan"] == 1

    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert record_execution(store, exec_identity="E1", intent_id="I", quantity=Decimal("0.5"), submitted_quantity=Decimal("1"), already_filled=Decimal("0"))["execution_status"] == "EXECUTION_RECORDED"
    assert record_execution(store, exec_identity="E1", intent_id="I", quantity=Decimal("0.5"), submitted_quantity=Decimal("1"), already_filled=Decimal("0"))["execution_status"] == "DUPLICATE_EXECUTION_IGNORED"
    assert record_execution(store, exec_identity="E2", intent_id="I", quantity=Decimal("1.5"), submitted_quantity=Decimal("1"), already_filled=Decimal("0"))["execution_status"] == "FILLED_QUANTITY_EXCEEDS_SUBMITTED"
    assert record_commission(store, commission_identity="C1", exec_identity="E1", amount=Decimal("0.01"))["commission_status"] == "COMMISSION_RECORDED"
    assert record_commission(store, commission_identity="C1", exec_identity="E1", amount=Decimal("0.01"))["commission_status"] == "DUPLICATE_COMMISSION_IGNORED"


def test_reconciliation_empty_go_and_mismatch_blocks(tmp_path) -> None:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert reconcile_paper_state(store)["reconciliation_status"] == "PAPER_RECONCILED_EMPTY"
    assert reconcile_paper_state(store, broker_open_order_count=1)["reconciliation_status"] == "UNKNOWN_BROKER_ORDER"
    store.register_intent(
        {
            "intent_id": "I",
            "economic_order_key": "E",
            "payload_hash": "H",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    assert reconcile_paper_state(store)["reconciliation_status"] == "PAPER_RECONCILED_EMPTY"
    store.append_event("I", "PLACE_ORDER_CALLED_ONCE", {"order_id_hash": "H"})
    assert reconcile_paper_state(store)["reconciliation_status"] == "LOCAL_ORDER_MISSING_AT_BROKER"
    store.append_event("I", "CANCEL_ORDER_CALLED_ONCE", {"order_id_hash": "H"})
    assert reconcile_paper_state(store)["reconciliation_status"] == "LOCAL_ORDER_MISSING_AT_BROKER"
    store.append_event("I", "BROKER_ORDER_CANCELLED", {"broker_proof": True})
    assert reconcile_paper_state(store)["reconciliation_status"] == "PAPER_RECONCILED_EMPTY"
    assert reconcile_paper_state(store, non_atomic_snapshot=True)["reconciliation_status"] == "NON_ATOMIC_SNAPSHOT"
    assert reconcile_paper_state(store, execution_scope_complete=False)["reconciliation_status"] == "EXECUTION_SCOPE_INCOMPLETE"


def test_known_pending_order_requires_private_identity_match(
    tmp_path,
) -> None:
    store = PaperExecutionStore(tmp_path / "phase9.sqlite3")
    store.initialize()
    assert (
        store.register_intent(
            {
                "intent_id": "PENDING-1",
                "economic_order_key": "PENDING-KEY-1",
                "created_at": "2026-07-27T00:00:00+00:00",
            }
        )
        == "INTENT_REGISTERED"
    )
    status, order_id = store.allocate_order_id(101, "PENDING-1")
    assert status == "ORDER_ID_READY"
    assert order_id == 101
    assert store.mark_order_id_used(101) == "ORDER_ID_READY"
    store.append_event(
        "PENDING-1",
        "PLACE_ORDER_CALLED_ONCE",
        {"order_id_hash": "PRIVATE"},
    )

    matched = reconcile_paper_state(
        store,
        broker_open_order_count=1,
        matched_open_order_count=1,
    )
    matched_closing = reconcile_paper_state(
        store,
        broker_open_order_count=1,
        broker_position_count=1,
        local_position_quantity=Decimal("1"),
        broker_position_quantity=Decimal("1"),
        matched_open_order_count=1,
    )
    count_only = reconcile_paper_state(
        store,
        broker_open_order_count=1,
    )
    mismatch = reconcile_paper_state(
        store,
        broker_open_order_count=1,
        matched_open_order_count=0,
        unknown_broker_open_order_count=1,
    )

    assert (
        matched["reconciliation_status"]
        == "PAPER_RECONCILED_OPEN_ORDER"
    )
    assert matched["status"] == "GO"
    assert (
        matched_closing["reconciliation_status"]
        == "PAPER_RECONCILED_OPEN_ORDER_AND_POSITION"
    )
    assert matched_closing["status"] == "GO"
    assert count_only["reconciliation_status"] == "PAPER_RECONCILIATION_BLOCKED"
    assert mismatch["reconciliation_status"] == "UNKNOWN_BROKER_ORDER"


def test_phase9_runtime_submit_consumes_approval_and_calls_place_order_once(tmp_path, monkeypatch) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "data" / "execution" / "phase9" / "private" / "paper_execution.sqlite3")
    store.initialize()
    intent, _ = prepare_intent(tmp_path, config, con_id=756733, side="BUY", quantity=Decimal("1"), limit_price=Decimal("50"), reason="manual")
    assert store.register_intent(intent.__dict__) == "INTENT_REGISTERED"
    approve_intent(store, intent, approval_challenge(intent), ttl_seconds=300)

    class App:
        next_valid_order_id = 101
        calls = 0

        def placeOrder(self, _order_id: int, _contract: object, _order: object) -> None:
            self.calls += 1

    class Service:
        disconnected = False

        def disconnect(self) -> None:
            self.disconnected = True

    app = App()
    service = Service()
    monkeypatch.setattr(phase9_audit_module, "connect_phase9_writer", lambda _config: (service, app, {"connection_status": "HEALTHY", "server_version": 1, "thread_leak": False}))
    monkeypatch.setattr(phase9_audit_module, "build_stock_contract", lambda _intent: object())
    monkeypatch.setattr(phase9_audit_module, "build_limit_day_order", lambda _intent: object())
    monkeypatch.setattr(phase9_audit_module, "_phase8_observation_counts", lambda _project_root: _empty_phase8_observation())

    result = phase9_audit_module.submit(tmp_path, tmp_path / ".env.ibkr", intent_id=intent.intent_id)
    assert result["status"] == "GO"
    assert result["submission_status"] == "SUBMISSION_ALLOWED_ONCE"
    assert result["paper_place_order_calls"] == 1
    assert app.calls == 1
    assert service.disconnected is True

    second = phase9_audit_module.submit(tmp_path, tmp_path / ".env.ibkr", intent_id=intent.intent_id)
    assert second["status"] == "NO_GO"
    assert second["submission_status"] == "APPROVAL_REQUIRED"
    assert app.calls == 1


def test_phase9_runtime_cancel_requires_cancel_approval_and_known_order(tmp_path, monkeypatch) -> None:
    _write_env(tmp_path)
    config, _ = load_paper_writer_config(tmp_path, tmp_path / ".env.ibkr")
    assert config is not None
    store = PaperExecutionStore(tmp_path / "data" / "execution" / "phase9" / "private" / "paper_execution.sqlite3")
    store.initialize()
    intent, _ = prepare_intent(tmp_path, config, con_id=756733, side="BUY", quantity=Decimal("1"), limit_price=Decimal("50"), reason="manual")
    assert store.register_intent(intent.__dict__) == "INTENT_REGISTERED"
    assert store.allocate_order_id(201, intent.intent_id)[0] == "ORDER_ID_READY"
    approve_intent(store, intent, cancel_challenge(intent), ttl_seconds=300, approval_type="CANCEL")

    class App:
        next_valid_order_id = 202
        calls = 0

        def cancelOrder(self, _order_id: int) -> None:
            self.calls += 1

    class Service:
        def disconnect(self) -> None:
            pass

    app = App()
    monkeypatch.setattr(phase9_audit_module, "connect_phase9_writer", lambda _config: (Service(), app, {"connection_status": "HEALTHY", "server_version": 1, "thread_leak": False}))

    result = phase9_audit_module.cancel(tmp_path, tmp_path / ".env.ibkr", intent_id=intent.intent_id)
    assert result["status"] == "GO"
    assert result["cancel_status"] == "CANCEL_REQUESTED"
    assert result["paper_cancel_order_calls"] == 1
    assert app.calls == 1


def test_phase9_cli_preflight_reports_disabled_without_logging_env(capsys) -> None:
    code = main.main(["ibkr", "phase9", "preflight"])
    assert code in {0, 2}
    out = capsys.readouterr().out
    assert "IBKR_PAPER_WRITER_ENABLED" not in out
    assert '"schema": "phase9_preflight_v1"' in out
    assert '"status":' in out


def test_phase9_preflight_go_reports_manual_paper_canary_authority(tmp_path) -> None:
    _write_env(tmp_path)

    result = phase9_audit_module.phase9_preflight(tmp_path, tmp_path / ".env.ibkr")

    assert result["status"] == "GO"
    assert result["execution_authority"] == "MANUAL_PAPER_CANARY"
    assert result["strategy_authority"] == "NONE"
    assert result["live_authority"] == "NONE"
    assert result["config"]["max_new_orders_per_day"] == 1
    assert result["config"]["max_closing_orders_per_day"] == 1
    freeze = json.loads((tmp_path / "output" / "ibkr" / "phase9" / "phase9-limit-semantics-freeze-status.json").read_text(encoding="utf-8"))
    assert freeze["freeze_status"] == "PHASE9_DAILY_ORDER_LIMIT_SEMANTICS_FROZEN_GO"
    assert freeze["phase9_full_freeze_status"] == "BLOCKED_UNTIL_OPERATOR_CANARY_B"
    assert freeze["new_paper_place_order_calls"] == 0
    assert freeze["new_paper_cancel_order_calls"] == 0
    assert "src/stocks/ibkr/paper_execution/storage.py" in freeze["source_hashes"]


def test_phase9_preflight_blocks_non_frozen_daily_limit_semantics(tmp_path) -> None:
    _write_env(tmp_path, max_new_orders=4)
    result = phase9_audit_module.phase9_preflight(tmp_path, tmp_path / ".env.ibkr")
    assert result["status"] == "NO_GO"
    assert "OPENING_ORDER_DAILY_LIMIT_BLOCKED" in result["preflight_errors"]

    _write_env(tmp_path, max_closing_orders=4)
    result = phase9_audit_module.phase9_preflight(tmp_path, tmp_path / ".env.ibkr")
    assert result["status"] == "NO_GO"
    assert "CLOSING_ORDER_DAILY_LIMIT_BLOCKED" in result["preflight_errors"]


def _write_env(
    tmp_path: Path,
    *,
    writer_id: int = 917,
    fingerprint: str = "a" * 64,
    observed_fingerprint: str = "a" * 64,
    max_new_orders: int = 1,
    max_closing_orders: int = 1,
) -> None:
    _write_private_phase8_fingerprint(tmp_path, observed_fingerprint)
    (tmp_path / ".env.ibkr").write_text(
        "\n".join(
            [
                "APP_ENV=development",
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7497",
                "IBKR_CLIENT_ID=17",
                "IBKR_ACCOUNT=",
                "IBKR_READ_ONLY=true",
                "IBKR_ORDER_AUTHORITY=NONE",
                "IBKR_LIVE_TRADING_ENABLED=false",
                "IBKR_ALLOW_ORDER_TRANSMISSION=false",
                "IBKR_MARKET_DATA_TYPE=3",
                "IBKR_ALLOWED_SECURITY_TYPES=STK",
                "IBKR_ALLOWED_CURRENCIES=EUR,USD",
                "IBKR_MAX_ORDER_NOTIONAL_EUR=0",
                "IBKR_MAX_OPEN_ORDERS=0",
                "IBKR_MAX_POSITIONS=0",
                "IBKR_RECON_CLIENT_ID=817",
                "IBKR_ACCOUNT_FINGERPRINT_KEY=fixture-key",
                f"IBKR_PAPER_WRITER_CLIENT_ID={writer_id}",
                "IBKR_PAPER_WRITER_ENABLED=true",
                f"IBKR_PAPER_ACCOUNT_FINGERPRINT={fingerprint}",
                "IBKR_PAPER_MAX_ORDER_NOTIONAL_EUR=100",
                "IBKR_PAPER_MAX_OPEN_ORDERS=1",
                "IBKR_PAPER_MAX_POSITIONS=1",
                f"IBKR_PAPER_MAX_NEW_ORDERS_PER_DAY={max_new_orders}",
                f"IBKR_PAPER_MAX_CLOSING_ORDERS_PER_DAY={max_closing_orders}",
                "IBKR_PAPER_APPROVAL_TTL_SECONDS=300",
                "IBKR_PAPER_CALLBACK_TIMEOUT_SECONDS=15",
                "IBKR_PAPER_RECONCILIATION_TIMEOUT_SECONDS=30",
            ]
        ),
        encoding="utf-8",
    )


def _write_private_phase8_fingerprint(tmp_path: Path, fingerprint: str) -> None:
    db_path = tmp_path / "data" / "broker" / "phase8_1" / "private" / "observation_soak.sqlite3"
    _write_fingerprint_snapshot(db_path, fingerprint)


def _write_fingerprint_snapshot(db_path: Path, fingerprint: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "account": {
            "values": [
                {
                    "account_fingerprint": fingerprint,
                    "tag": "NetLiquidation",
                    "value": "100",
                    "currency": "EUR",
                }
            ]
        }
    }
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS snapshots (row_id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_id TEXT, snapshot_hash TEXT, payload_json TEXT, created_at TEXT)"
        )
        connection.execute(
            "INSERT INTO snapshots(snapshot_id, snapshot_hash, payload_json, created_at) VALUES (?, ?, ?, ?)",
            ("S", "H", json.dumps(payload), "2026-01-01T00:00:00+00:00"),
        )
        connection.commit()


def _empty_phase8_observation() -> dict[str, object]:
    return {
        "status": "GO",
        "snapshot_status": "BROKER_SNAPSHOT_OBSERVED",
        "same_client_open_order_count": 0,
        "all_api_open_order_count": 0,
        "position_count": 0,
        "execution_count": 0,
        "commission_count": 0,
        "execution_scope_current": True,
        "read_only_request_counters": {},
        "broker_write_counters": {},
    }
