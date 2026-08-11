from __future__ import annotations

import inspect
import json
from pathlib import Path

import main
import pytest
from stocks.ibkr.paper_execution import audit as phase9_audit_module
from stocks.ibkr.paper_execution.canary_a_evidence import (
    CANARY_A_MARKER,
    reconstruct_canary_a_evidence,
)
from stocks.ibkr.paper_execution.storage import (
    PaperExecutionStore,
    Phase9Layout,
    artifact,
    write_json,
)


def test_canary_a_reconstructed_from_ledger_and_command_zeros_do_not_erase_it(tmp_path: Path) -> None:
    _fixture(tmp_path)

    result = reconstruct_canary_a_evidence(tmp_path)

    assert result["status"] == "CANARY_A_EVIDENCE_GO"
    assert result["canary_marker"] == CANARY_A_MARKER
    assert result["place_order_economic_call_count"] == 1
    assert result["cancel_order_economic_call_count"] == 1
    assert result["command_local_counters"]["paper_place_order_calls"] == 0
    assert result["command_local_counters"]["paper_cancel_order_calls"] == 0
    assert result["cumulative_private_ledger_evidence"]["place_order_economic_call_count"] == 1
    assert result["cumulative_private_ledger_evidence"]["cancel_order_economic_call_count"] == 1
    assert result["final_order_state"] == "API_CANCELLED"
    assert result["reconciliation_status"] == "PAPER_RECONCILED_EMPTY"


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        ("PLACE_ORDER_CALLED_ONCE", "CANARY_A_PLACE_COUNT_MISMATCH"),
        ("CANCEL_ORDER_CALLED_ONCE", "CANARY_A_CANCEL_COUNT_MISMATCH"),
    ],
)
def test_duplicate_economic_calls_block(tmp_path: Path, event_type: str, expected: str) -> None:
    store = _fixture(tmp_path)
    store.append_event("CANARY-A-INTENT", event_type, {"order_id_hash": "DUPLICATE"})

    assert reconstruct_canary_a_evidence(tmp_path)["status"] == expected


def test_duplicate_callback_does_not_change_economic_call_count(tmp_path: Path) -> None:
    store = _fixture(tmp_path)
    store.append_event("CANARY-A-INTENT", "ORDER_STATUS_CALLBACK", {"status": "ApiCancelled"})
    store.append_event("CANARY-A-INTENT", "ORDER_STATUS_CALLBACK", {"status": "ApiCancelled"})

    result = reconstruct_canary_a_evidence(tmp_path)

    assert result["status"] == "CANARY_A_EVIDENCE_GO"
    assert result["place_order_economic_call_count"] == 1
    assert result["cancel_order_economic_call_count"] == 1


def test_unsubmitted_later_buy_intent_does_not_erase_canary_a(
    tmp_path: Path,
) -> None:
    store = _fixture(tmp_path)
    assert (
        store.register_intent(
            {
                "intent_id": "CANARY-B-STAGED",
                "economic_order_key": "CANARY-B-STAGED-KEY",
                "intent_source": "MANUAL_OPERATOR",
                "created_at": "2026-07-27T12:00:00+00:00",
                "account_fingerprint": "PRIVATE-ACCOUNT",
                "side": "BUY",
                "security_type": "STK",
                "order_type": "LIMIT",
                "time_in_force": "DAY",
                "outside_rth": False,
            }
        )
        == "INTENT_REGISTERED"
    )

    result = reconstruct_canary_a_evidence(tmp_path)

    assert result["status"] == "CANARY_A_EVIDENCE_GO"
    assert result["accepted_manual_buy_intent_count"] == 1
    assert result["staged_unsubmitted_manual_buy_intent_count"] == 1


def test_later_submitted_known_buy_does_not_erase_canary_a(
    tmp_path: Path,
) -> None:
    store = _fixture(
        tmp_path,
        reconciliation_overrides={
            "reconciliation_status": "PAPER_RECONCILED_OPEN_ORDER",
            "local_active_order_count": 1,
            "broker_open_order_count": 1,
            "unknown_broker_open_order_count": 0,
            "missing_local_open_order_count": 0,
        },
    )
    assert (
        store.register_intent(
            {
                "intent_id": "CANARY-B-SUBMITTED",
                "economic_order_key": "CANARY-B-SUBMITTED-KEY",
                "intent_source": "MANUAL_OPERATOR",
                "created_at": "2026-07-27T20:00:00+00:00",
                "account_fingerprint": "PRIVATE-ACCOUNT",
                "side": "BUY",
                "security_type": "STK",
                "order_type": "LIMIT",
                "time_in_force": "DAY",
                "outside_rth": False,
            }
        )
        == "INTENT_REGISTERED"
    )
    store.append_event(
        "CANARY-B-SUBMITTED",
        "PLACE_ORDER_CALLED_ONCE",
        {"order_id_hash": "PRIVATE-ORDER-ID-B"},
    )

    result = reconstruct_canary_a_evidence(tmp_path)

    assert result["status"] == "CANARY_A_EVIDENCE_GO"
    assert result["place_order_economic_call_count"] == 1
    assert result["cancel_order_economic_call_count"] == 1
    assert result["other_known_active_order_count"] == 1
    assert (
        result["cumulative_private_ledger_evidence"][
            "place_order_economic_call_count"
        ]
        == 2
    )


def test_later_known_fill_does_not_erase_canary_a(
    tmp_path: Path,
) -> None:
    store = _fixture(
        tmp_path,
        reconciliation_overrides={
            "reconciliation_status": "PAPER_RECONCILED_OPEN_LONG",
            "broker_position_count": 1,
            "broker_execution_count": 1,
            "broker_commission_count": 1,
            "local_position_quantity": "1",
            "broker_position_quantity": "1",
        },
    )
    assert (
        store.register_intent(
            {
                "intent_id": "CANARY-B-FILLED",
                "economic_order_key": "CANARY-B-FILLED-KEY",
                "intent_source": "MANUAL_OPERATOR",
                "created_at": "2026-07-27T20:00:00+00:00",
                "account_fingerprint": "PRIVATE-ACCOUNT",
                "side": "BUY",
                "security_type": "STK",
                "order_type": "LIMIT",
                "time_in_force": "DAY",
                "outside_rth": False,
            }
        )
        == "INTENT_REGISTERED"
    )
    store.append_event(
        "CANARY-B-FILLED",
        "PLACE_ORDER_CALLED_ONCE",
        {"order_id_hash": "PRIVATE-ORDER-ID-B"},
    )
    store.append_execution_once(
        "PRIVATE-EXEC-B",
        "CANARY-B-FILLED",
        {"side": "BUY", "quantity": "1"},
    )
    store.append_commission_once(
        "PRIVATE-COMMISSION-B",
        "PRIVATE-EXEC-B",
        {"amount": "0.35"},
    )

    result = reconstruct_canary_a_evidence(tmp_path)

    assert result["status"] == "CANARY_A_EVIDENCE_GO"
    assert result["execution_count"] == 0
    assert result["commission_count"] == 0
    assert result["cumulative_private_ledger_evidence"]["execution_count"] == 1


def test_frozen_canary_a_survives_later_known_fill_with_bounded_history(
    tmp_path: Path,
) -> None:
    store = _fixture(tmp_path)
    first = reconstruct_canary_a_evidence(tmp_path)
    layout = Phase9Layout.from_project_root(tmp_path)
    freeze_path = layout.artifact(
        "canary-a-evidence-freeze-status.json"
    )
    frozen_before = freeze_path.read_bytes()
    assert first["status"] == "CANARY_A_EVIDENCE_GO"
    assert (
        store.register_intent(
            {
                "intent_id": "CANARY-B-BOUNDED-HISTORY",
                "economic_order_key": "CANARY-B-BOUNDED-HISTORY-KEY",
                "intent_source": "MANUAL_OPERATOR",
                "created_at": "2026-07-27T20:00:00+00:00",
                "account_fingerprint": "PRIVATE-ACCOUNT",
                "side": "BUY",
                "security_type": "STK",
                "order_type": "LIMIT",
                "time_in_force": "DAY",
                "outside_rth": False,
            }
        )
        == "INTENT_REGISTERED"
    )
    store.append_event(
        "CANARY-B-BOUNDED-HISTORY",
        "PLACE_ORDER_CALLED_ONCE",
        {"order_id_hash": "PRIVATE-ORDER-ID-B"},
    )
    store.append_execution_once(
        "PRIVATE-EXEC-B",
        "CANARY-B-BOUNDED-HISTORY",
        {"side": "BUY", "quantity": "1"},
    )
    reconciliation = {
        "status": "GO",
        "reconciliation_status": "PAPER_RECONCILED_OPEN_LONG",
        "local_active_order_count": 0,
        "broker_open_order_count": 0,
        "broker_position_count": 1,
        "broker_execution_count": 0,
        "broker_commission_count": 0,
        "execution_history_complete": False,
        "local_position_quantity": "1",
        "broker_position_quantity": "1",
        "unknown_broker_open_order_count": 0,
        "missing_local_open_order_count": 0,
        "live_place_order_calls": 0,
        "automatic_submissions": 0,
        "automatic_cancellations": 0,
        "strategy_generated_intents": 0,
        "global_cancel_calls": 0,
        "auto_bind_order_calls": 0,
        "exercise_option_calls": 0,
    }
    write_json(
        layout.artifact("reconciliation-audit.json"),
        artifact("phase9_reconciliation_audit_v1", reconciliation),
    )

    result = reconstruct_canary_a_evidence(tmp_path)

    assert result["status"] == "CANARY_A_EVIDENCE_GO"
    assert result["historical_freeze_valid"] is True
    assert result["candidate_absence_basis"] == (
        "FROZEN_HISTORICAL_RECONCILIATION"
    )
    assert result["order_acknowledgement_source"] == (
        "FROZEN_HISTORICAL_RECONCILIATION"
    )
    assert freeze_path.read_bytes() == frozen_before


def test_tampered_historical_freeze_cannot_rescue_nonempty_broker(
    tmp_path: Path,
) -> None:
    store = _fixture(tmp_path)
    reconstruct_canary_a_evidence(tmp_path)
    layout = Phase9Layout.from_project_root(tmp_path)
    freeze_path = layout.artifact(
        "canary-a-evidence-freeze-status.json"
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["evidence_hash"] = "TAMPERED"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    store.register_intent(
        {
            "intent_id": "CANARY-B-TAMPERED",
            "economic_order_key": "CANARY-B-TAMPERED-KEY",
            "intent_source": "MANUAL_OPERATOR",
            "created_at": "2026-07-27T20:00:00+00:00",
            "account_fingerprint": "PRIVATE-ACCOUNT",
            "side": "BUY",
            "security_type": "STK",
            "order_type": "LIMIT",
            "time_in_force": "DAY",
            "outside_rth": False,
        }
    )
    store.append_event(
        "CANARY-B-TAMPERED",
        "PLACE_ORDER_CALLED_ONCE",
        {},
    )
    store.append_execution_once(
        "PRIVATE-EXEC-B",
        "CANARY-B-TAMPERED",
        {"side": "BUY", "quantity": "1"},
    )
    reconciliation = {
        "status": "GO",
        "reconciliation_status": "PAPER_RECONCILED_OPEN_LONG",
        "local_active_order_count": 0,
        "broker_open_order_count": 0,
        "broker_position_count": 1,
        "broker_execution_count": 0,
        "broker_commission_count": 0,
        "local_position_quantity": "1",
        "broker_position_quantity": "1",
        "unknown_broker_open_order_count": 0,
        "missing_local_open_order_count": 0,
    }
    write_json(
        layout.artifact("reconciliation-audit.json"),
        artifact("phase9_reconciliation_audit_v1", reconciliation),
    )

    result = reconstruct_canary_a_evidence(tmp_path)

    assert result["status"] == "CANARY_A_BROKER_NOT_EMPTY"
    assert result["historical_freeze_valid"] is False


def test_execution_blocks_canary_a(tmp_path: Path) -> None:
    store = _fixture(tmp_path)
    store.append_execution_once("PRIVATE-EXEC", "CANARY-A-INTENT", {})

    assert reconstruct_canary_a_evidence(tmp_path)["status"] == "CANARY_A_EXECUTION_UNEXPECTED"


def test_commission_blocks_canary_a(tmp_path: Path) -> None:
    store = _fixture(tmp_path)
    store.append_commission_once("PRIVATE-COMMISSION", "PRIVATE-EXEC", {})

    assert reconstruct_canary_a_evidence(tmp_path)["status"] == "CANARY_A_COMMISSION_UNEXPECTED"


@pytest.mark.parametrize(
    "overrides",
    [
        {"broker_open_order_count": 1},
        {"broker_position_count": 1},
        {"reconciliation_status": "PAPER_RECONCILED"},
    ],
)
def test_nonempty_or_nonempty_reconciliation_blocks(tmp_path: Path, overrides: dict[str, object]) -> None:
    _fixture(tmp_path, reconciliation_overrides=overrides)

    assert reconstruct_canary_a_evidence(tmp_path)["status"] == "CANARY_A_BROKER_NOT_EMPTY"


@pytest.mark.parametrize(
    "event_type",
    [
        "LIVE_PLACE_ORDER_CALLED",
        "AUTOMATIC_SUBMISSION",
        "AUTOMATIC_CANCELLATION",
        "STRATEGY_GENERATED_INTENT",
    ],
)
def test_forbidden_calls_and_strategy_intent_block(tmp_path: Path, event_type: str) -> None:
    store = _fixture(tmp_path)
    store.append_event("CANARY-A-INTENT", event_type, {})

    assert reconstruct_canary_a_evidence(tmp_path)["status"] == "CANARY_A_FORBIDDEN_CALL_DETECTED"


def test_public_artifact_contains_no_private_identifiers_or_secrets(tmp_path: Path) -> None:
    _fixture(tmp_path)
    reconstruct_canary_a_evidence(tmp_path)
    text = (Phase9Layout.from_project_root(tmp_path).artifact("canary-a-submit-cancel-evidence.json")).read_text(encoding="utf-8")

    for forbidden in (
        "CANARY-A-INTENT",
        "PRIVATE-ACCOUNT",
        "PRIVATE-ORDER-ID",
        "approval_id",
        "challenge_hash",
        "account_fingerprint",
        "exec_id",
        "api_key",
    ):
        assert forbidden not in text


def test_existing_evidence_hash_mismatch_is_fail_closed(tmp_path: Path) -> None:
    _fixture(tmp_path)
    first = reconstruct_canary_a_evidence(tmp_path)
    path = Phase9Layout.from_project_root(tmp_path).artifact("canary-a-submit-cancel-evidence.json")
    first["evidence_hash"] = "TAMPERED"
    path.write_text(json.dumps(first), encoding="utf-8")

    assert reconstruct_canary_a_evidence(tmp_path)["status"] == "CANARY_A_EVIDENCE_HASH_MISMATCH"


def test_phase9_status_recognizes_canary_a_and_fill_close_remain(
    tmp_path: Path, monkeypatch
) -> None:
    _fixture(tmp_path)
    layout = Phase9Layout.from_project_root(tmp_path)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    layout.artifact("schema.json").write_text("{}", encoding="utf-8")
    good_artifacts = {
        "preflight.json": {"status": "GO"},
        "reconciliation-audit.json": json.loads(layout.artifact("reconciliation-audit.json").read_text(encoding="utf-8")),
        "canary-results.json": {"fill_canary": "NOT_RUN_REQUIRES_OPERATOR_APPROVAL"},
    }
    for name in (
        "authority-audit.json",
        "order-id-audit.json",
        "manual-approval-audit.json",
        "risk-audit.json",
        "submission-audit.json",
        "callback-audit.json",
        "execution-audit.json",
        "commission-audit.json",
        "cancellation-audit.json",
            "restart-recovery-audit.json",
            "p0-safety-matrix.json",
            "order-state-machine-audit.json",
        "privacy-audit.json",
    ):
        good_artifacts[name] = {"status": "GO"}
    monkeypatch.setattr(phase9_audit_module, "_read_artifacts", lambda _layout: good_artifacts)
    monkeypatch.setattr(phase9_audit_module, "_freeze_marker", lambda _path: True)

    result = phase9_audit_module.phase9_status(tmp_path)

    assert result["status"] == "NO_GO"
    assert result["checks"]["submit_cancel_canary"] is True
    assert result["checks"]["fill_canary"] is False
    assert result["checks"]["closing_sell_canary"] is False
    assert result["open_blockers"] == [
        "fill_canary",
        "closing_sell_canary",
    ]
    assert result["execution_authority"] == "NONE"
    assert result["strategy_authority"] == "NONE"
    assert result["live_authority"] == "NONE"


def test_cli_is_offline_and_canary_evidence_module_has_no_broker_writes(tmp_path: Path, monkeypatch) -> None:
    _fixture(tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    assert main.main(["ibkr", "phase9", "canary-a-evidence"]) == 0
    source = inspect.getsource(__import__("stocks.ibkr.paper_execution.canary_a_evidence", fromlist=["*"]))
    assert ".placeOrder(" not in source
    assert ".cancelOrder(" not in source


def _fixture(
    project_root: Path,
    *,
    reconciliation_overrides: dict[str, object] | None = None,
) -> PaperExecutionStore:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    intent = {
        "intent_id": "CANARY-A-INTENT",
        "economic_order_key": "CANARY-A-ECONOMIC-KEY",
        "intent_source": "MANUAL_OPERATOR",
        "created_at": "2026-07-21T18:50:00+00:00",
        "account_fingerprint": "PRIVATE-ACCOUNT",
        "side": "BUY",
        "security_type": "STK",
        "order_type": "LIMIT",
        "time_in_force": "DAY",
        "outside_rth": False,
    }
    assert store.register_intent(intent) == "INTENT_REGISTERED"
    _approval(store, "SUBMIT-APPROVAL", "SUBMIT")
    store.append_event("CANARY-A-INTENT", "ORDER_ID_ALLOCATED", {"order_id_hash": "PRIVATE-ORDER-ID"})
    store.append_event("CANARY-A-INTENT", "PLACE_ORDER_CALLED_ONCE", {"order_id_hash": "PRIVATE-ORDER-ID"})
    _approval(store, "CANCEL-APPROVAL", "CANCEL")
    store.append_event("CANARY-A-INTENT", "CANCEL_ORDER_CALLED_ONCE", {"order_id_hash": "PRIVATE-ORDER-ID"})
    reconciliation = {
        "status": "GO",
        "reconciliation_status": "PAPER_RECONCILED_EMPTY",
        "local_active_order_count": 0,
        "broker_open_order_count": 0,
        "broker_position_count": 0,
        "broker_execution_count": 0,
        "broker_commission_count": 0,
        "live_place_order_calls": 0,
        "automatic_submissions": 0,
        "automatic_cancellations": 0,
        "strategy_generated_intents": 0,
        "global_cancel_calls": 0,
        "auto_bind_order_calls": 0,
        "exercise_option_calls": 0,
    }
    reconciliation.update(reconciliation_overrides or {})
    write_json(layout.artifact("reconciliation-audit.json"), artifact("phase9_reconciliation_audit_v1", reconciliation))
    return store


def _approval(store: PaperExecutionStore, approval_id: str, approval_type: str) -> None:
    payload = {
        "approval_id": approval_id,
        "intent_id": "CANARY-A-INTENT",
        "approval_type": approval_type,
        "challenge_hash": "PRIVATE-CHALLENGE",
        "intent_hash": "PRIVATE-INTENT-HASH",
        "expires_at": "2099-01-01T00:00:00+00:00",
    }
    assert store.append_approval(payload) == "APPROVAL_RECORDED"
    assert store.consume_approval(approval_id) == "APPROVED_FOR_SINGLE_SUBMISSION"
