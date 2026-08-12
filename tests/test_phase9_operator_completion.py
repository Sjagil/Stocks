from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

from stocks.ibkr.paper_execution import audit as audit_module
from stocks.ibkr.paper_execution.audit import phase9_status
from stocks.ibkr.paper_execution.executions import FillExecution, record_fill_execution
from stocks.ibkr.paper_execution.operator_completion import (
    EVIDENCE_STATUS,
    accept_operator_attested_manual_completion,
    load_operator_completion_evidence,
)
from stocks.ibkr.paper_execution.storage import (
    PaperExecutionStore,
    Phase9Layout,
    artifact,
    write_json,
)


def test_operator_attested_manual_close_requires_broker_continuity(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)

    result = accept_operator_attested_manual_completion(
        tmp_path,
        symbol="ON",
        con_id=8677881,
        reason="Operator confirms the paper position was manually closed in TWS.",
    )

    assert result["status"] == EVIDENCE_STATUS
    assert result["paper_round_trip_operationally_accepted"] is True
    assert result["api_closing_sell_path_proven"] is False
    assert result["phase9_ledger_mutated"] is False
    assert result["evidence"]["same_account_fingerprint"] is True
    assert result["evidence"]["transition_seconds"] == 73.0
    assert load_operator_completion_evidence(tmp_path) is not None
    assert "DU_TEST_ACCOUNT" not in json.dumps(result)

    store = PaperExecutionStore(Phase9Layout.from_project_root(tmp_path).db_path)
    assert len(store.list_executions()) == 1
    assert all(row["payload"]["side"] == "BUY" for row in store.list_executions())


def test_operator_attestation_blocks_without_empty_post_snapshot(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, post_empty=False)

    result = accept_operator_attested_manual_completion(
        tmp_path,
        symbol="ON",
        con_id=8677881,
        reason="Operator confirms the paper position was manually closed in TWS.",
    )

    assert result["status"] == "NO_GO"
    assert result["classification"] == "BROKER_EMPTY_CONTINUITY_NOT_FOUND"
    assert load_operator_completion_evidence(tmp_path) is None


def test_operator_attestation_preserves_post_callback_timeout_limitation(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, post_timeout=True)

    result = accept_operator_attested_manual_completion(
        tmp_path,
        symbol="ON",
        con_id=8677881,
        reason="Operator confirms the paper position was manually closed in TWS.",
    )

    assert result["status"] == EVIDENCE_STATUS
    assert result["evidence"]["post_broker_state_verified"] is False
    assert result["evidence"]["post_callback_timeout"] is True
    assert result["evidence"]["account_continuity_status"] == (
        "PRE_MATCH_POST_CALLBACK_TIMEOUT_OPERATOR_ATTESTED"
    )
    assert result["api_closing_sell_path_proven"] is False


def test_phase9_status_never_uses_attestation_for_canonical_gates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _fixture(tmp_path)
    accepted = accept_operator_attested_manual_completion(
        tmp_path,
        symbol="ON",
        con_id=8677881,
        reason="Operator confirms the paper position was manually closed in TWS.",
    )
    assert accepted["status"] == EVIDENCE_STATUS

    layout = Phase9Layout.from_project_root(tmp_path)
    write_json(layout.artifact("schema.json"), artifact("test_schema", {"status": "GO"}))
    phase8_freeze = tmp_path / "output" / "shadow" / "phase8_2" / "freeze-status.json"
    write_json(
        phase8_freeze,
        artifact("test_freeze", {"freeze_status": "PHASE8_2_FROZEN_GO"}),
    )
    audit_names = [
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
    ]
    artifacts = {
        "preflight.json": {"status": "GO"},
        "reconciliation-audit.json": {"status": "NO_GO"},
        **{name: {"status": "GO"} for name in audit_names},
    }
    monkeypatch.setattr(audit_module, "_read_artifacts", lambda _layout: artifacts)
    monkeypatch.setattr(
        audit_module,
        "canary_results",
        lambda _root: {
            "status": "NO_GO",
            "fill_canary": "GO",
            "closing_sell_canary": "CLOSING_SELL_REQUIRED",
        },
    )
    monkeypatch.setattr(
        audit_module,
        "reconstruct_canary_a_evidence",
        lambda _root: {"status": "CANARY_A_BROKER_NOT_EMPTY"},
    )
    monkeypatch.setattr(audit_module, "manifest", lambda _root: {"status": "GO"})

    result = phase9_status(tmp_path)

    assert result["status"] == "NO_GO"
    assert result["checks"]["reconciliation"] is False
    assert result["checks"]["submit_cancel_canary"] is False
    assert result["checks"]["fill_canary"] is True
    assert result["checks"]["closing_sell_canary"] is False
    assert result["completion_basis"] == (
        "OPERATOR_ATTESTED_EXTERNAL_CLOSE_NON_CANONICAL"
    )
    assert result["operator_completion_effect"] == (
        "BROKER_STATE_CONTEXT_ONLY_NO_CANONICAL_GATE_SATISFACTION"
    )
    assert result["canonical_execution_evidence_status"] == "NO_GO"
    assert result["api_closing_sell_path_proven"] is False


def _fixture(
    tmp_path: Path,
    *,
    post_empty: bool = True,
    post_timeout: bool = False,
) -> None:
    layout = Phase9Layout.from_project_root(tmp_path)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    fill = FillExecution(
        exec_id="TEST-BUY-EXEC",
        intent_id="TEST-BUY-INTENT",
        account_fingerprint="DU_TEST_ACCOUNT",
        perm_id="TEST-PERM-BUY",
        broker_order_id="TEST-ORDER-BUY",
        con_id=8677881,
        symbol="ON",
        currency="USD",
        side="BUY",
        quantity=Decimal("1"),
        price=Decimal("90"),
        execution_time="2026-07-31T09:30:00+00:00",
        submitted_quantity=Decimal("1"),
        fx_rate=Decimal("0.92"),
    )
    assert record_fill_execution(store, fill)["execution_status"] == "EXECUTION_ACCEPTED"

    write_json(
        layout.artifact("canary-a-evidence-freeze-status.json"),
        artifact(
            "phase9_canary_a_evidence_freeze_status_v1",
            {
                "freeze_status": "PHASE9_CANARY_A_EVIDENCE_ADOPTION_FROZEN_GO",
                "execution_authority": "NONE",
                "live_authority": "NONE",
            },
        ),
    )
    observation_db = (
        tmp_path
        / "data"
        / "broker"
        / "phase8"
        / "private"
        / "broker_observation.sqlite3"
    )
    observation_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(observation_db) as connection:
        connection.execute(
            "CREATE TABLE snapshots ("
            "snapshot_id TEXT PRIMARY KEY, snapshot_hash TEXT NOT NULL, "
            "payload_json TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        pre = _snapshot(position=True, sell_order=True)
        post = _snapshot(
            position=not post_empty,
            sell_order=not post_empty,
            callback_timeout=post_timeout,
        )
        connection.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
            (
                "PRE",
                "PRE-SNAPSHOT-HASH",
                json.dumps(pre),
                "2026-07-31T09:44:39+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
            (
                "POST",
                "POST-SNAPSHOT-HASH",
                json.dumps(post),
                "2026-07-31T09:45:52+00:00",
            ),
        )


def _snapshot(
    *,
    position: bool,
    sell_order: bool,
    callback_timeout: bool = False,
) -> dict[str, object]:
    positions = (
        [
            {
                "account_fingerprint": "DU_TEST_ACCOUNT",
                "symbol": "ON",
                "con_id": 8677881,
                "position_quantity": "1",
            }
        ]
        if position
        else []
    )
    orders = (
        [
            {
                "account_fingerprint": "DU_TEST_ACCOUNT",
                "symbol": "ON",
                "con_id": 8677881,
                "action": "SELL",
                "order_status": "PreSubmitted",
                "total_quantity": "1",
            }
        ]
        if sell_order
        else []
    )
    return {
        "account": {
            "status": "CALLBACK_TIMEOUT" if callback_timeout else "COMPLETE",
            "values": (
                []
                if callback_timeout
                else [
                    {
                        "account_fingerprint": "DU_TEST_ACCOUNT",
                        "tag": "NetLiquidation",
                    }
                ]
            ),
        },
        "positions": {
            "status": "CALLBACK_TIMEOUT" if callback_timeout else "COMPLETE",
            "positions": positions,
        },
        "all_api_open_orders": {
            "status": "CALLBACK_TIMEOUT" if callback_timeout else "COMPLETE",
            "open_orders": orders,
        },
        "same_client_open_orders": {"status": "COMPLETE", "open_orders": []},
        "executions": {
            "status": "EMPTY_COMPLETE",
            "executions": [],
            "commissions": [],
        },
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "server_version": None if callback_timeout else 176,
    }
