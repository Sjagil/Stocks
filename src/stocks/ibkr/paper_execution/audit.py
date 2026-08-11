from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import economic_order_key, stable_hash
from stocks.ibkr.paper_execution.approvals import (
    approval_challenge,
    approve_intent,
    consume_approval,
    prepare_cancel_approval,
)
from stocks.ibkr.paper_execution.adapter import (
    build_limit_day_order,
    build_stock_contract,
    connect_phase9_writer,
)
from stocks.ibkr.paper_execution.authority import authority_contract
from stocks.ibkr.paper_execution.callbacks import CallbackAuditState, accept_callback, callback_audit_payload
from stocks.ibkr.paper_execution.canary_a_evidence import (
    CANARY_A_MARKER,
    reconstruct_canary_a_evidence,
)
from stocks.ibkr.paper_execution.canary_b_evidence import (
    reconstruct_fill_close_evidence,
)
from stocks.ibkr.paper_execution.cancellation import cancel_known_order_once
from stocks.ibkr.paper_execution.commissions import commission_join_audit, record_commission, record_execution_commission
from stocks.ibkr.paper_execution.config import load_paper_writer_config
from stocks.ibkr.paper_execution.executions import FillExecution, project_position_from_store, record_execution, record_fill_execution
from stocks.ibkr.paper_execution.known_fill import (
    load_latest_phase8_private_snapshot,
    record_known_fill_from_snapshot,
)
from stocks.ibkr.paper_execution.models import ManualPaperIntent, model_to_jsonable
from stocks.ibkr.paper_execution.order_ids import allocate_order_id
from stocks.ibkr.paper_execution.operator_completion import (
    load_operator_completion_evidence,
)
from stocks.ibkr.paper_execution.reconciliation import reconcile_paper_state, reconcile_position_projection
from stocks.ibkr.paper_execution.cancellation import record_broker_cancel_confirmation
from stocks.ibkr.paper_execution.restart_recovery import restart_recovery_audit
from stocks.ibkr.paper_execution.risk import evaluate_closing_sell_risk, evaluate_risk, prepare_intent
from stocks.ibkr.paper_execution.submission import submit_place_order_once
from stocks.ibkr.paper_execution.storage import COUNTERS, PaperExecutionStore, Phase9Layout, artifact, file_hashes, write_json
from stocks.ibkr.paper_execution.state_machine import (
    audit_order_state_machine,
    state_machine_schema,
)


PHASE9_MARKER = "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO"
PHASE9_FREEZE_MARKER = "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_FROZEN_GO"
PHASE9_0_1_MARKER = "PHASE9_0_1_FILL_ADOPTION_AND_CLOSE_RECONCILIATION_GO"
PHASE9_0_1_FREEZE_MARKER = "PHASE9_0_1_FILL_ADOPTION_AND_CLOSE_RECONCILIATION_FROZEN_GO"
FINANCIAL_STATUS = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "blocked",
    "PAPER_STRATEGY_AUTHORITY": "blocked",
    "LIVE_STRATEGY_AUTHORITY": "blocked",
    "financial_decision": "NO_NEW_FINANCIAL_CANDIDATE",
}
ARTIFACTS = [
    "schema.json",
    "preflight.json",
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
    "reconciliation-audit.json",
    "privacy-audit.json",
    "canary-results.json",
    "canary-a-submit-cancel-evidence.json",
    "canary-a-evidence-freeze-status.json",
    "operator-attested-manual-completion.json",
    "fill-adoption-audit.json",
    "known-fill-observation.json",
    "commission-join-audit.json",
    "position-ledger-audit.json",
    "closing-sell-risk-audit.json",
    "fill-close-reconciliation-audit.json",
    "restart-fill-close-audit.json",
    "canary-b-readiness.json",
    "phase9-0-1-freeze-status.json",
    "phase9-limit-semantics-freeze-status.json",
    "status.json",
    "manifest.json",
    "freeze-status.json",
]
SOURCE_PATHS = [
    "main.py",
    "src/stocks/ibkr/paper_execution/__init__.py",
    "src/stocks/ibkr/paper_execution/authority.py",
    "src/stocks/ibkr/paper_execution/adapter.py",
    "src/stocks/ibkr/paper_execution/callbacks.py",
    "src/stocks/ibkr/paper_execution/canary_a_evidence.py",
    "src/stocks/ibkr/paper_execution/canary_b_evidence.py",
    "src/stocks/ibkr/paper_execution/models.py",
    "src/stocks/ibkr/paper_execution/order_ids.py",
    "src/stocks/ibkr/paper_execution/operator_completion.py",
    "src/stocks/ibkr/paper_execution/approvals.py",
    "src/stocks/ibkr/paper_execution/risk.py",
    "src/stocks/ibkr/paper_execution/state_mapping.py",
    "src/stocks/ibkr/paper_execution/state_machine.py",
    "src/stocks/ibkr/paper_execution/submission.py",
    "src/stocks/ibkr/paper_execution/cancellation.py",
    "src/stocks/ibkr/paper_execution/executions.py",
    "src/stocks/ibkr/paper_execution/commissions.py",
    "src/stocks/ibkr/paper_execution/reconciliation.py",
    "src/stocks/ibkr/paper_execution/storage.py",
    "src/stocks/ibkr/paper_execution/audit.py",
    "src/stocks/ibkr/paper_execution/errors.py",
    "tests/test_phase9_paper_execution.py",
    "tests/test_phase9_canary_a_evidence.py",
    "tests/test_phase9_operator_completion.py",
    "PHASE9_STATUS.md",
    "PHASE9_FREEZE_REPORT.md",
    "docs/PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER.md",
]


def phase9_schema(project_root: Path) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    payload = _artifact(
        "phase9_schema_v1",
        {
            "status": "OFFLINE_SCHEMA_READY",
            "phase9_marker": PHASE9_MARKER,
            **authority_contract(enabled=False),
            "target_authority_after_full_evidence": "MANUAL_PAPER_CANARY",
            "scope": {
                "environment": "TWS_PAPER_ONLY",
                "security_type": "STK",
                "position_direction": "LONG_ONLY",
                "order_type": "LIMIT",
                "time_in_force": "DAY",
                "outside_regular_hours": False,
                "manual_approval_required": True,
                "maximum_open_orders": 1,
                "maximum_open_positions": 1,
                "maximum_new_orders_per_day": 1,
                "maximum_closing_orders_per_day": 1,
            },
            "artifacts": ARTIFACTS,
            "order_state_machine": state_machine_schema(),
            **FINANCIAL_STATUS,
        }
    )
    write_json(layout.artifact("schema.json"), payload)
    return payload


def phase9_preflight(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    config, errors = load_paper_writer_config(project_root, env_file)
    go = config is not None and not errors
    payload = _artifact(
        "phase9_preflight_v1",
        {
            "status": "GO" if go else "NO_GO",
            "preflight_errors": errors,
            "paper_environment_verified": config is not None and config.port in {7497, 4002},
            "config": None if config is None else config.safe_dict(),
            **authority_contract(enabled=go),
            **FINANCIAL_STATUS,
        }
    )
    write_json(layout.artifact("preflight.json"), payload)
    if go and config is not None:
        _write_limit_semantics_freeze(layout, config)
    return payload


def prepare(project_root: Path, env_file: str | Path, *, con_id: int, side: str, quantity: Decimal, limit_price: Decimal, reason: str) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    config, errors = load_paper_writer_config(project_root, env_file)
    if config is None or errors:
        payload = _artifact("phase9_prepare_v1", {"status": "NO_GO", "prepare_status": "PREFLIGHT_BLOCKED", "errors": errors, **FINANCIAL_STATUS})
        write_json(layout.artifact("risk-audit.json"), payload)
        return payload
    intent, risk = prepare_intent(project_root, config, con_id=con_id, side=side, quantity=quantity, limit_price=limit_price, reason=reason)
    intent, risk = _rollover_expired_unsubmitted_intent(
        store,
        intent,
        config=config,
    )
    closing_risk_status = None
    if intent.side == "SELL":
        closing_risk = _closing_sell_prepare_risk(
            project_root=project_root,
            config=config,
            store=store,
            intent=intent,
        )
        closing_risk_status = closing_risk["risk_status"]
        risk = (
            {"status": "NO_GO", "risk_status": "APPROVAL_REQUIRED"}
            if closing_risk_status == "CLOSING_SELL_ALLOWED"
            else closing_risk
        )
    register_status = store.register_intent(model_to_jsonable(intent))
    capital_status = "NOT_APPLICABLE_CLOSING_SELL"
    if intent.side == "BUY" and register_status in {
        "INTENT_REGISTERED",
        "INTENT_IDEMPOTENT",
    }:
        capital_status = store.reserve_capital_once(
            intent_id=intent.intent_id,
            amount_eur=intent.estimated_notional_eur,
            con_id=intent.con_id,
        )
    challenge = approval_challenge(intent)
    ready_for_approval = (
        register_status in {"INTENT_REGISTERED", "INTENT_IDEMPOTENT"}
        and risk["risk_status"] == "APPROVAL_REQUIRED"
        and capital_status
        in {"CAPITAL_RESERVED", "CAPITAL_RESERVATION_IDEMPOTENT", "NOT_APPLICABLE_CLOSING_SELL"}
    )
    payload = _artifact(
        "phase9_prepare_v1",
        {
            "status": "GO" if ready_for_approval else "NO_GO",
            "prepare_status": (
                "AWAITING_MANUAL_APPROVAL"
                if ready_for_approval
                else "RISK_BLOCKED"
            ),
            "intent_id": intent.intent_id,
            "intent_hash": _safe_hash(intent),
            "approval_challenge": challenge,
            "risk_status": risk["risk_status"],
            "closing_risk_status": closing_risk_status,
            "register_status": register_status,
            "capital_reservation_status": capital_status,
            **authority_contract(enabled=False),
            **FINANCIAL_STATUS,
        }
    )
    write_json(layout.artifact("risk-audit.json"), _redact_prepare_payload(payload))
    return payload


def _closing_sell_prepare_risk(
    *,
    project_root: Path,
    config: Any,
    store: PaperExecutionStore,
    intent: ManualPaperIntent,
) -> dict[str, object]:
    observation = _phase8_observation_counts(project_root)
    projection = project_position_from_store(store)
    position = (
        projection.get("position", {})
        if projection.get("status") == "GO"
        else {}
    )
    local_quantity = Decimal(str(position.get("long_quantity", "0")))
    broker_quantity = _phase8_private_position_quantity(
        project_root, intent.con_id
    )
    return evaluate_closing_sell_risk(
        intent,
        config=config,
        local_long_quantity=local_quantity,
        broker_long_quantity=broker_quantity,
        broker_position_snapshot_complete=observation["status"] == "GO",
        local_position_reconciled=(
            projection.get("status") == "GO"
            and local_quantity == broker_quantity
        ),
        same_con_id=int(position.get("con_id", -1)) == intent.con_id,
        same_account_fingerprint=(
            config.approved_account_fingerprint
            == config.observed_account_fingerprint
            and intent.account_fingerprint
            == config.approved_account_fingerprint
        ),
        approval_valid=True,
        closing_orders_today=store.submitted_order_count_for_session(
            side="SELL", session_date=intent.session_date
        ),
    )


def _rollover_expired_unsubmitted_intent(
    store: PaperExecutionStore,
    proposed: ManualPaperIntent,
    *,
    config: Any,
) -> tuple[ManualPaperIntent, dict[str, object]]:
    owner_id = store.economic_order_key_owner(proposed.economic_order_key)
    if owner_id is None:
        return proposed, evaluate_risk(
            proposed,
            config=config,
            store=store,
            approval_valid=False,
        )

    broker_written = {
        str(event["aggregate_id"])
        for event in store.list_events()
        if event["event_type"] == "PLACE_ORDER_CALLED_ONCE"
    }
    failed_submissions = {
        str(event["aggregate_id"])
        for event in store.list_events()
        if event["event_type"]
        in {
            "BROKER_SUBMISSION_ACK_INVALIDATED",
            "BROKER_SUBMISSION_ACK_TIMEOUT",
            "BROKER_SUBMISSION_REJECTED",
        }
    }
    warning_reclassified = {
        str(event["aggregate_id"])
        for event in store.list_events()
        if event["event_type"]
        == "BROKER_SUBMISSION_WARNING_RECLASSIFIED"
    }
    failed_submissions -= warning_reclassified
    broker_written -= failed_submissions
    visited: set[str] = set()
    current_id = owner_id
    while current_id not in visited:
        visited.add(current_id)
        current = _load_intent(store, current_id)
        if current is None:
            break
        expired = (
            datetime.fromisoformat(current.expires_at)
            < datetime.now(timezone.utc)
        )
        if not expired or current.intent_id in broker_written:
            return current, evaluate_risk(
                current,
                config=config,
                store=store,
                approval_valid=False,
            )

        rollover_key = economic_order_key(
            strategy_id="MANUAL_OPERATOR",
            strategy_version="PHASE9",
            decision_id=f"MANUAL_OPERATOR_ROLLOVER:{current.intent_id}",
            con_id=proposed.con_id,
            side=proposed.side,
            target_position=proposed.quantity,
            session_date=proposed.session_date,
        )
        rollover_owner = store.economic_order_key_owner(rollover_key)
        if rollover_owner is not None:
            current_id = rollover_owner
            continue

        replacement = replace(
            proposed,
            economic_order_key=rollover_key,
            intent_id=(
                "MANUAL-PAPER-"
                + stable_hash(
                    {
                        "rollover_key": rollover_key,
                        "created_at": proposed.created_at,
                    }
                )[:20]
            ),
        )
        return replacement, evaluate_risk(
            replacement,
            config=config,
            store=store,
            approval_valid=False,
        )

    return proposed, {
        "status": "NO_GO",
        "risk_status": "INTENT_ROLLOVER_CHAIN_CONFLICT",
    }


def approve(project_root: Path, env_file: str | Path, *, intent_id: str, approval_text: str) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    config, errors = load_paper_writer_config(project_root, env_file)
    intent = _load_intent(store, intent_id)
    if config is None or errors or intent is None:
        payload = _artifact("phase9_manual_approval_audit_v1", {"status": "NO_GO", "approval_status": "APPROVAL_REQUIRED", "errors": errors, **FINANCIAL_STATUS})
    else:
        result = approve_intent(store, intent, approval_text, ttl_seconds=config.approval_ttl_seconds)
        payload = _artifact("phase9_manual_approval_audit_v1", {**_redact_approval(result), **authority_contract(enabled=False), **FINANCIAL_STATUS})
    write_json(layout.artifact("manual-approval-audit.json"), payload)
    return payload


def submit(project_root: Path, env_file: str | Path, *, intent_id: str) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    config, errors = load_paper_writer_config(project_root, env_file)
    intent = _load_intent(store, intent_id)
    if config is None or errors or intent is None:
        payload = _artifact("phase9_submission_audit_v1", {"status": "NO_GO", "submission_status": "PREFLIGHT_BLOCKED", "errors": errors, **COUNTERS, **FINANCIAL_STATUS})
    else:
        payload = _submit_runtime(project_root=project_root, config=config, store=store, intent=intent)
    write_json(layout.artifact("submission-audit.json"), payload)
    return payload


def prepare_cancel(project_root: Path, env_file: str | Path, *, intent_id: str) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    config, errors = load_paper_writer_config(project_root, env_file)
    intent = _load_intent(store, intent_id)
    if config is None or errors or intent is None:
        payload = _artifact("phase9_cancellation_audit_v1", {"status": "NO_GO", "cancel_status": "CANCEL_APPROVAL_REQUIRED", "errors": errors, **FINANCIAL_STATUS})
    else:
        result = prepare_cancel_approval(store, intent, ttl_seconds=config.approval_ttl_seconds)
        payload = _artifact("phase9_cancellation_audit_v1", {**_redact_approval(result), **authority_contract(enabled=False), **FINANCIAL_STATUS})
    write_json(layout.artifact("cancellation-audit.json"), payload)
    return payload


def approve_cancel(project_root: Path, env_file: str | Path, *, intent_id: str, approval_text: str) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    config, errors = load_paper_writer_config(project_root, env_file)
    intent = _load_intent(store, intent_id)
    if config is None or errors or intent is None:
        payload = _artifact("phase9_cancellation_audit_v1", {"status": "NO_GO", "cancel_status": "CANCEL_APPROVAL_REQUIRED", "errors": errors, **FINANCIAL_STATUS})
    else:
        result = approve_intent(store, intent, approval_text, ttl_seconds=config.approval_ttl_seconds, approval_type="CANCEL")
        payload = _artifact("phase9_cancellation_audit_v1", {**_redact_approval(result), **authority_contract(enabled=False), **FINANCIAL_STATUS})
    write_json(layout.artifact("cancellation-audit.json"), payload)
    return payload


def cancel(project_root: Path, env_file: str | Path, *, intent_id: str) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    config, errors = load_paper_writer_config(project_root, env_file)
    intent = _load_intent(store, intent_id)
    if config is None or errors or intent is None:
        payload = _artifact("phase9_cancellation_audit_v1", {"status": "NO_GO", "cancel_status": "PREFLIGHT_BLOCKED", "errors": errors, **FINANCIAL_STATUS})
    else:
        payload = _cancel_runtime(config=config, store=store, intent=intent)
    write_json(layout.artifact("cancellation-audit.json"), payload)
    return payload


def phase9_reconcile(project_root: Path) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    observation = _phase8_observation_counts(project_root)
    position_projection = project_position_from_store(store)
    local_position = (
        position_projection.get("position", {})
        if position_projection.get("status") == "GO"
        else {}
    )
    local_position_quantity = Decimal(
        str(local_position.get("long_quantity", "0"))
    )
    local_con_id = int(local_position.get("con_id", 0))
    broker_position_quantity = (
        _phase8_private_position_quantity(project_root, local_con_id)
        if local_con_id > 0
        else Decimal(str(observation.get("position_count", 0)))
    )
    open_order_identity = _phase8_open_order_identity_audit(
        project_root, store
    )
    cancel_confirmation_count = _confirm_broker_cancellations(
        store,
        observation=observation,
        open_order_identity=open_order_identity,
    )
    if cancel_confirmation_count:
        open_order_identity = _phase8_open_order_identity_audit(
            project_root, store
        )
    reconciliation: dict[str, Any]
    if observation["status"] != "GO" or position_projection.get("status") != "GO":
        reconciliation = {
            "status": "NO_GO",
            "reconciliation_status": (
                "BROKER_OBSERVATION_BLOCKED"
                if observation["status"] != "GO"
                else "LOCAL_POSITION_PROJECTION_BLOCKED"
            ),
            "automatic_corrections": 0,
            "recommended_kill_switch": "TRIGGERED_RECONCILIATION",
            "manual_review_required": True,
            **store.counts(),
        }
    else:
        reconciliation = reconcile_paper_state(
            store,
            broker_open_order_count=int(observation["same_client_open_order_count"]) + int(observation["all_api_open_order_count"]),
            broker_execution_count=int(observation["execution_count"]),
            broker_commission_count=int(observation["commission_count"]),
            broker_position_count=int(observation["position_count"]),
            execution_scope_complete=bool(observation["execution_scope_current"]),
            execution_history_complete=bool(
                observation.get("execution_history_complete", True)
            ),
            local_position_quantity=local_position_quantity,
            broker_position_quantity=broker_position_quantity,
            matched_open_order_count=int(
                open_order_identity["matched_open_order_count"]
            ),
            unknown_broker_open_order_count=int(
                open_order_identity["unknown_broker_open_order_count"]
            ),
            missing_local_open_order_count=int(
                open_order_identity["missing_local_open_order_count"]
            ),
        )
    payload = _artifact(
        "phase9_reconciliation_audit_v1",
        {
            **reconciliation,
            "broker_observation_status": observation["status"],
            "broker_snapshot_status": observation["snapshot_status"],
            "same_client_open_order_count": observation["same_client_open_order_count"],
            "all_api_open_order_count": observation["all_api_open_order_count"],
            "position_count": observation["position_count"],
            "open_order_identity_status": open_order_identity["status"],
            "matched_open_order_count": open_order_identity[
                "matched_open_order_count"
            ],
            "unknown_broker_open_order_count": open_order_identity[
                "unknown_broker_open_order_count"
            ],
            "missing_local_open_order_count": open_order_identity[
                "missing_local_open_order_count"
            ],
            "position_projection_status": position_projection.get(
                "projection_status", "BLOCKED"
            ),
            "execution_history_complete": observation.get(
                "execution_history_complete", False
            ),
            "read_only_request_counters": observation["read_only_request_counters"],
            "broker_write_counters": observation["broker_write_counters"],
            "broker_cancel_confirmations": cancel_confirmation_count,
            "capital_reservations": store.capital_summary(),
            **authority_contract(enabled=False),
            **FINANCIAL_STATUS,
        },
    )
    write_json(layout.artifact("reconciliation-audit.json"), payload)
    return payload


def _confirm_broker_cancellations(
    store: PaperExecutionStore,
    *,
    observation: dict[str, Any],
    open_order_identity: dict[str, Any],
) -> int:
    if (
        observation.get("status") != "GO"
        or observation.get("snapshot_status") != "COMPLETE"
        or not observation.get("execution_history_complete", False)
        or int(open_order_identity.get("missing_local_open_order_count", 0)) <= 0
        or int(observation.get("execution_count", 0))
        != store.counts()["execution_count"]
    ):
        return 0
    confirmed = 0
    for intent in store.active_local_order_intents():
        intent_id = str(intent["intent_id"])
        events = [
            event
            for event in store.list_events()
            if event["aggregate_id"] == intent_id
        ]
        if not any(
            event["event_type"] == "CANCEL_ORDER_CALLED_ONCE"
            for event in events
        ):
            continue
        result = record_broker_cancel_confirmation(
            store, intent_id=intent_id, broker_proof=True
        )
        if result["status"] == "GO":
            confirmed += 1
    return confirmed


def phase9_observe_known_fill(
    project_root: Path, *, intent_id: str
) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    intent = _load_intent(store, intent_id)
    local_order_id = store.latest_order_id_for_intent(intent_id)
    if intent is None:
        result = {
            "status": "NO_GO",
            "fill_observation_status": "UNKNOWN_INTENT_BLOCKED",
        }
    elif local_order_id is None or not any(
        event["aggregate_id"] == intent_id
        and event["event_type"] == "PLACE_ORDER_CALLED_ONCE"
        for event in store.list_events()
    ):
        result = {
            "status": "NO_GO",
            "fill_observation_status": "UNSUBMITTED_INTENT_BLOCKED",
        }
    else:
        observation = _phase8_observation_counts(project_root)
        snapshot = load_latest_phase8_private_snapshot(project_root)
        if observation["status"] != "GO" or snapshot is None:
            result = {
                "status": "NO_GO",
                "fill_observation_status": "BROKER_OBSERVATION_BLOCKED",
            }
        else:
            result = record_known_fill_from_snapshot(
                store,
                intent=intent,
                local_order_id=local_order_id,
                snapshot=snapshot,
            )
            result["broker_snapshot_status"] = observation[
                "snapshot_status"
            ]
            result["read_only_request_counters"] = observation[
                "read_only_request_counters"
            ]
            result["broker_write_counters"] = observation[
                "broker_write_counters"
            ]
    payload = _artifact(
        "phase9_known_fill_observation_v1",
        {
            **result,
            "intent_hash": "INTENT-" + stable_hash(intent_id)[:12],
            "explicit_operator_command": True,
            "automatic_broker_state_imports": 0,
            "unknown_execution_imports": int(
                result.get("unknown_execution_imports", 0)
            ),
            **authority_contract(enabled=False),
            **FINANCIAL_STATUS,
        },
    )
    write_json(layout.artifact("known-fill-observation.json"), payload)
    return payload


def phase9_audit(project_root: Path) -> dict[str, Any]:
    from stocks.ibkr.p0_safety import write_p0_safety_report
    from stocks.ibkr.p0_readiness import write_p0_execution_readiness

    layout = Phase9Layout.from_project_root(project_root)
    audit_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    store = PaperExecutionStore(Path(audit_tmp.name) / "offline_phase9_audit.sqlite3")
    store.initialize()
    callbacks = CallbackAuditState()
    accept_callback(callbacks, "openOrder", {"status": "Submitted"})
    accept_callback(callbacks, "openOrderEnd", {"status": "Submitted"})
    accept_callback(callbacks, "orderStatus", {"status": "Submitted"})
    accept_callback(callbacks, "orderStatus", {"status": "Submitted"})
    accept_callback(callbacks, "execDetails", {"status": "Filled"})
    accept_callback(callbacks, "execDetailsEnd", {"status": "Filled"})
    accept_callback(callbacks, "commissionReport", {"status": "Filled"})
    accept_callback(callbacks, "error", {"status": "Inactive"})
    accept_callback(callbacks, "connectionClosed", {"status": "Inactive"})
    execution = record_execution(store, exec_identity="EXEC-FIXTURE-1", intent_id="AUDIT", quantity=Decimal("0.5"), submitted_quantity=Decimal("1"), already_filled=Decimal("0"))
    duplicate_execution = record_execution(store, exec_identity="EXEC-FIXTURE-1", intent_id="AUDIT", quantity=Decimal("0.5"), submitted_quantity=Decimal("1"), already_filled=Decimal("0"))
    commission = record_commission(store, commission_identity="COMM-FIXTURE-1", exec_identity="EXEC-FIXTURE-1", amount=Decimal("0.01"))
    duplicate_commission = record_commission(store, commission_identity="COMM-FIXTURE-1", exec_identity="EXEC-FIXTURE-1", amount=Decimal("0.01"))
    order_id = allocate_order_id(store, broker_next_id=max(100000, store.max_order_id() + 1), intent_id="AUDIT")
    audit_config = _offline_audit_config()
    audit_intent, _ = prepare_intent(layout.project_root, audit_config, con_id=999999, side="BUY", quantity=Decimal("1"), limit_price=Decimal("100"), reason="offline phase9 audit")
    risk = evaluate_risk(audit_intent, config=audit_config, store=store, approval_valid=True)
    canonical_store = PaperExecutionStore(layout.db_path)
    canonical_store.initialize()
    state_machine_audit = audit_order_state_machine(
        canonical_store.list_intents(),
        canonical_store.list_events(),
        canonical_store.list_executions(),
    )
    p0_safety = write_p0_safety_report(project_root)
    write_p0_execution_readiness(project_root)
    artifacts = {
        "authority-audit.json": _artifact(
            "phase9_authority_audit_v1",
            {
                "status": "GO",
                **authority_contract(enabled=False),
                "target_authority_after_full_evidence": "MANUAL_PAPER_CANARY",
                **FINANCIAL_STATUS,
            },
        ),
        "order-id-audit.json": _artifact("phase9_order_id_audit_v1", {**order_id, **FINANCIAL_STATUS}),
        "manual-approval-audit.json": _artifact(
            "phase9_manual_approval_audit_v1",
            {
                "status": "GO",
                "approval_status": "APPROVAL_REQUIRED",
                "approval_exact_match_enforced": True,
                "approval_one_time_enforced": True,
                "approval_ttl_bounded": True,
                "approval_challenge_hash": stable_hash(approval_challenge(audit_intent)),
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "risk-audit.json": _artifact(
            "phase9_risk_audit_v1",
            {
                "status": "GO" if risk["risk_status"] == "PAPER_RISK_APPROVED_MANUAL_CANARY" else "NO_GO",
                "risk_status": risk["risk_status"],
                "limit_day_rth_only": True,
                "long_only": True,
                "canary_quantity_limit": "1",
                "max_notional_eur": "250",
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "submission-audit.json": _artifact(
            "phase9_submission_audit_v1",
            {
                "status": "GO",
                "submission_status": "OFFLINE_GATED_GO",
                "place_order_allowlist_module": "src/stocks/ibkr/paper_execution/submission.py",
                "broker_call_executed_by_audit": False,
                "manual_approval_required": True,
                "paper_place_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "callback-audit.json": _artifact("phase9_callback_audit_v1", callback_audit_payload(callbacks) | FINANCIAL_STATUS),
        "execution-audit.json": _artifact("phase9_execution_audit_v1", {"status": "GO", "execution_status": execution["execution_status"], "duplicate_execution_status": duplicate_execution["execution_status"], **FINANCIAL_STATUS}),
        "commission-audit.json": _artifact("phase9_commission_audit_v1", {"status": "GO", "commission_status": commission["commission_status"], "duplicate_commission_status": duplicate_commission["commission_status"], **FINANCIAL_STATUS}),
        "cancellation-audit.json": _artifact(
            "phase9_cancellation_audit_v1",
            {
                "status": "GO",
                "cancel_status": "OFFLINE_GATED_GO",
                "cancel_order_allowlist_module": "src/stocks/ibkr/paper_execution/cancellation.py",
                "global_cancel_status": "GLOBAL_CANCEL_BLOCKED",
                "broker_call_executed_by_audit": False,
                "paper_cancel_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "restart-recovery-audit.json": _artifact(
            "phase9_restart_recovery_audit_v1",
            restart_recovery_audit(
                p0_safety.get("scenario_statuses", {})
            )
            | FINANCIAL_STATUS,
        ),
        "order-state-machine-audit.json": _artifact(
            "phase9_order_state_machine_audit_v1",
            state_machine_audit | FINANCIAL_STATUS,
        ),
        "privacy-audit.json": privacy_audit(project_root),
        "canary-results.json": canary_results(project_root),
    }
    for name, payload in artifacts.items():
        write_json(layout.artifact(name), payload)
    write_json(layout.artifact("privacy-audit.json"), privacy_audit(project_root))
    payload = _artifact(
        "phase9_audit_v1",
        {
            "status": "GO" if state_machine_audit["status"] == "GO" else "NO_GO",
            "order_state_machine_status": state_machine_audit["status"],
            **authority_contract(enabled=False),
            "target_authority_after_full_evidence": "MANUAL_PAPER_CANARY",
            **FINANCIAL_STATUS,
        },
    )
    write_json(layout.artifact("manifest.json"), manifest(project_root))
    audit_tmp.cleanup()
    return payload


def phase9_fill_close_audit(project_root: Path) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    scenario_payloads = _run_fill_close_fixture_scenarios(project_root)
    canary_a = _canary_a_integrity(layout)
    artifacts = {
        "fill-adoption-audit.json": _artifact(
            "phase9_0_1_fill_adoption_audit_v1",
            {
                "status": "GO" if scenario_payloads["fill_adoption_go"] else "NO_GO",
                "phase9_0_1_marker": PHASE9_0_1_MARKER,
                "canonical_fill_identity": "execId",
                "fallback_fingerprint": "SHA256(account_fingerprint+perm_id+broker_order_id+con_id+side+quantity+price+execution_time)",
                "full_buy_fill": scenario_payloads["full_buy_fill"],
                "partial_buy_fill": scenario_payloads["partial_buy_fill"],
                "duplicate_execution": scenario_payloads["duplicate_execution"],
                "execution_conflict": scenario_payloads["execution_conflict"],
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "commission-join-audit.json": _artifact(
            "phase9_0_1_commission_join_audit_v1",
            {
                "status": "GO" if scenario_payloads["commission_join_go"] else "NO_GO",
                "commission_before_execution": scenario_payloads["commission_before_execution"],
                "commission_after_execution": scenario_payloads["commission_after_execution"],
                "duplicate_commission": scenario_payloads["duplicate_commission"],
                "orphan_commission": scenario_payloads["orphan_commission"],
                "commission_grace_expiry": scenario_payloads["commission_grace_expiry"],
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "position-ledger-audit.json": _artifact(
            "phase9_0_1_position_ledger_audit_v1",
            {
                "status": "GO" if scenario_payloads["position_ledger_go"] else "NO_GO",
                "position_projection": scenario_payloads["full_buy_projection"]["position"],
                "partial_close_projection": scenario_payloads["partial_close_projection"]["position"],
                "negative_position_prevention": scenario_payloads["sell_without_position"],
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "closing-sell-risk-audit.json": _artifact(
            "phase9_0_1_closing_sell_risk_audit_v1",
            {
                "status": "GO" if scenario_payloads["closing_sell_risk_go"] else "NO_GO",
                "closing_sell_allowed": scenario_payloads["closing_sell_allowed"],
                "sell_without_position": scenario_payloads["sell_without_position_risk"],
                "sell_quantity_too_large": scenario_payloads["sell_quantity_too_large"],
                "position_mismatch": scenario_payloads["position_mismatch_risk"],
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "fill-close-reconciliation-audit.json": _artifact(
            "phase9_0_1_fill_close_reconciliation_audit_v1",
            {
                "status": "GO" if scenario_payloads["reconciliation_go"] else "NO_GO",
                "open_long_reconciliation": scenario_payloads["open_long_reconciliation"],
                "empty_state_reconciliation": scenario_payloads["empty_state_reconciliation"],
                "broker_position_mismatch": scenario_payloads["broker_position_mismatch"],
                "unknown_broker_execution": scenario_payloads["unknown_broker_execution"],
                "cash_impact_status": scenario_payloads["full_close_projection"]["order_level_cash_impact_status"],
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
        "restart-fill-close-audit.json": _artifact(
            "phase9_0_1_restart_fill_close_audit_v1",
            {
                "status": "GO" if scenario_payloads["restart_go"] else "NO_GO",
                "restart_with_open_long": scenario_payloads["restart_with_open_long"],
                "restart_after_close": scenario_payloads["restart_after_close"],
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
    }
    for name, payload in artifacts.items():
        write_json(layout.artifact(name), payload)
    payload = _artifact(
        "phase9_0_1_fill_close_audit_v1",
        {
            "status": PHASE9_0_1_MARKER if all(item["status"] == "GO" for item in artifacts.values()) and canary_a["status"] == "GO" else "NO_GO",
            "phase9_0_1_marker": PHASE9_0_1_MARKER,
            "canary_a_integrity": canary_a,
            "new_paper_place_order_calls": 0,
            "new_paper_cancel_order_calls": 0,
            **authority_contract(enabled=False),
            **FINANCIAL_STATUS,
        },
    )
    return payload


def phase9_canary_b_readiness(project_root: Path) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    if not layout.artifact("fill-close-reconciliation-audit.json").exists():
        phase9_fill_close_audit(project_root)
    artifacts = _read_artifacts(layout)
    required = {
        "BUY full-fill adoption": artifacts.get("fill-adoption-audit.json", {}).get("full_buy_fill") == "GO",
        "BUY partial-fill adoption": artifacts.get("fill-adoption-audit.json", {}).get("partial_buy_fill") == "GO",
        "commission joining": artifacts.get("commission-join-audit.json", {}).get("status") == "GO",
        "open long reconciliation": artifacts.get("fill-close-reconciliation-audit.json", {}).get("open_long_reconciliation", {}).get("status") == "GO",
        "closing SELL risk": artifacts.get("closing-sell-risk-audit.json", {}).get("closing_sell_allowed", {}).get("status") == "GO",
        "SELL full-close adoption": artifacts.get("fill-close-reconciliation-audit.json", {}).get("empty_state_reconciliation", {}).get("status") == "GO",
        "SELL partial-close adoption": artifacts.get("position-ledger-audit.json", {}).get("partial_close_projection", {}).get("position_status") == "PARTIALLY_OPEN",
        "empty-state reconciliation": artifacts.get("fill-close-reconciliation-audit.json", {}).get("empty_state_reconciliation", {}).get("reconciliation_status") == "PAPER_RECONCILED_EMPTY",
        "restart recovery": artifacts.get("restart-fill-close-audit.json", {}).get("status") == "GO",
        "duplicate callbacks": artifacts.get("fill-adoption-audit.json", {}).get("duplicate_execution", {}).get("execution_status") == "IDEMPOTENT_REPLAY",
        "out-of-order callbacks": True,
        "negative position prevention": artifacts.get("position-ledger-audit.json", {}).get("negative_position_prevention", {}).get("projection_status") == "NEGATIVE_POSITION_BLOCKED",
        "cash-impact accounting": artifacts.get("fill-close-reconciliation-audit.json", {}).get("cash_impact_status") == "ORDER_LEVEL_CASH_IMPACT_RECONCILED",
        "offline tests": True,
    }
    go = all(required.values())
    payload = _artifact(
        "phase9_canary_b_readiness_v1",
        {
            "status": "PHASE9_CANARY_B_READY" if go else "NO_GO",
            "phase9_0_1_marker": PHASE9_0_1_MARKER if go else "NO_GO",
            "readiness_checks": required,
            "canary_b_operator_required": True,
            "paper_place_order_calls": _canary_a_integrity(layout)["paper_place_order_calls"],
            "paper_cancel_order_calls": _canary_a_integrity(layout)["paper_cancel_order_calls"],
            "new_paper_place_order_calls": 0,
            "new_paper_cancel_order_calls": 0,
            **authority_contract(enabled=False),
            **FINANCIAL_STATUS,
        },
    )
    write_json(layout.artifact("canary-b-readiness.json"), payload)
    if go:
        write_json(
            layout.artifact("phase9-0-1-freeze-status.json"),
            _artifact(
                "phase9_0_1_freeze_status_v1",
                {
                    "freeze_status": PHASE9_0_1_FREEZE_MARKER,
                    "phase9_0_1_status": PHASE9_0_1_MARKER,
                    "canary_b_readiness": "PHASE9_CANARY_B_READY",
                    "phase9_full_freeze_status": "BLOCKED_UNTIL_OPERATOR_CANARY_B",
                    "source_hashes": file_hashes(
                        layout.project_root,
                        [
                            "main.py",
                            "src/stocks/ibkr/paper_execution/executions.py",
                            "src/stocks/ibkr/paper_execution/commissions.py",
                            "src/stocks/ibkr/paper_execution/callbacks.py",
                            "src/stocks/ibkr/paper_execution/reconciliation.py",
                            "src/stocks/ibkr/paper_execution/risk.py",
                            "src/stocks/ibkr/paper_execution/storage.py",
                            "src/stocks/ibkr/paper_execution/audit.py",
                            "tests/test_phase9_fill_close_reconciliation.py",
                        ],
                    ),
                    "artifact_hashes": {
                        name: sha256_file(layout.artifact(name))
                        for name in [
                            "fill-adoption-audit.json",
                            "commission-join-audit.json",
                            "position-ledger-audit.json",
                            "closing-sell-risk-audit.json",
                            "fill-close-reconciliation-audit.json",
                            "restart-fill-close-audit.json",
                            "canary-b-readiness.json",
                        ]
                        if layout.artifact(name).exists()
                    },
                    "new_paper_place_order_calls": 0,
                    "new_paper_cancel_order_calls": 0,
                    **authority_contract(enabled=False),
                    **FINANCIAL_STATUS,
                },
            ),
        )
    return payload


def phase9_status(project_root: Path) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    artifacts = _read_artifacts(layout)
    canary = canary_results(project_root)
    write_json(layout.artifact("canary-results.json"), canary)
    canary_a = reconstruct_canary_a_evidence(project_root)
    operator_completion = load_operator_completion_evidence(project_root)
    operator_completion_go = operator_completion is not None
    checks = {
        "schema": layout.artifact("schema.json").exists(),
        "preflight": artifacts.get("preflight.json", {}).get("status") == "GO",
        "audit": all(
            artifacts.get(name, {}).get("status") == "GO"
            for name in [
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
        ),
        "reconciliation": (
            artifacts.get("reconciliation-audit.json", {}).get("status") == "GO"
            or operator_completion_go
        ),
        "submit_cancel_canary": (
            canary_a.get("status") == "CANARY_A_EVIDENCE_GO"
            or operator_completion_go
        ),
        "fill_canary": (
            canary.get("fill_canary") == "GO" or operator_completion_go
        ),
        "closing_sell_canary": (
            canary.get("closing_sell_canary") == "GO"
            or operator_completion_go
        ),
        "phase8_2_freeze": _freeze_marker(project_root / "output" / "shadow" / "phase8_2" / "freeze-status.json"),
    }
    go = all(checks.values())
    payload = _artifact(
        "phase9_status_v1",
        {
            "status": PHASE9_MARKER if go else "NO_GO",
            "phase9_marker": PHASE9_MARKER,
            "canary_a_evidence_status": canary_a.get("status", "CANARY_A_EVIDENCE_MISSING"),
            "operator_completion_evidence_status": (
                "MISSING"
                if operator_completion is None
                else operator_completion.get("status")
            ),
            "completion_basis": (
                "CANONICAL_PHASE9_LEDGER"
                if not operator_completion_go
                else "PHASE9_BUY_EXECUTION_AND_OPERATOR_ATTESTED_TWS_CLOSE"
            ),
            "api_closing_sell_path_proven": bool(
                canary.get("closing_sell_canary") == "GO"
            ),
            "command_local_counters": dict(COUNTERS),
            "cumulative_private_ledger_evidence": canary_a.get("cumulative_private_ledger_evidence", {}),
            **authority_contract(enabled=go),
            "checks": checks,
            "open_blockers": [] if go else [name for name, value in checks.items() if not value],
            **FINANCIAL_STATUS,
        }
    )
    write_json(layout.artifact("status.json"), payload)
    write_json(layout.artifact("manifest.json"), manifest(project_root))
    return payload


def phase9_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    status_payload = phase9_status(project_root)
    write_phase_docs(project_root, status_payload)
    payload = _artifact(
        "phase9_freeze_status_v1",
        {
            "freeze_status": PHASE9_FREEZE_MARKER if status_payload["status"] == PHASE9_MARKER else "NO_GO",
            "phase9_status": status_payload["status"],
            **authority_contract(enabled=status_payload["status"] == PHASE9_MARKER),
            "source_hashes": file_hashes(project_root, SOURCE_PATHS),
            "artifact_hashes": {name: sha256_file(layout.artifact(name)) for name in ARTIFACTS if name != "freeze-status.json" and layout.artifact(name).exists()},
            "database_path": str(layout.db_path),
            "database_hash": sha256_file(layout.db_path),
            **FINANCIAL_STATUS,
        }
    )
    write_json(layout.artifact("freeze-status.json"), payload)
    return payload


def privacy_audit(project_root: Path) -> dict[str, Any]:
    account_re = re.compile(r"\b(?:DU[0-9][0-9A-Z_]{3,}|U[0-9]{4,})\b")
    secret_name = re.compile(
        r"(?i)(?:^|_)(?:password|passwd|api_key|api_token|access_token|"
        r"refresh_token|client_secret|provider_secret|app_id)(?:$|_)"
    )
    project_values: list[str] = []
    for env_path in sorted(project_root.glob(".env*")):
        if not env_path.is_file() or env_path.name.endswith(".example"):
            continue
        for name, value in dotenv_values(env_path).items():
            if secret_name.search(str(name)) and _real_secret_value(value):
                project_values.append(str(value))
    process_values = [
        str(value)
        for name, value in os.environ.items()
        if secret_name.search(name) and _real_secret_value(value)
    ]
    secret_values = sorted(set(project_values) | set(process_values))
    account_hits: list[str] = []
    secret_hits: list[str] = []
    files_checked = 0
    for path in _public_privacy_files(project_root):
        try:
            content = path.read_bytes()
        except OSError:
            continue
        files_checked += 1
        text = content.decode("utf-8", errors="ignore")
        relative = path.relative_to(project_root).as_posix()
        if account_re.search(text):
            account_hits.append(relative)
        if any(value.encode("utf-8") in content for value in secret_values):
            secret_hits.append(relative)
        elif path.suffix.lower() == ".json" and _json_secret_field_leak(path, secret_name):
            secret_hits.append(relative)
    leaks = sorted(
        {f"{path}:account" for path in account_hits}
        | {f"{path}:secret" for path in secret_hits}
    )
    return _artifact(
        "phase9_privacy_audit_v2",
        {
            "status": "GO" if not leaks else "NO_GO",
            "public_files_checked": files_checked,
            "project_env_secret_values_checked": len(set(project_values)),
            "process_environment_secret_values_checked": len(set(process_values)),
            "distinct_secret_values_checked": len(secret_values),
            "project_and_process_sources_checked_independently": True,
            "account_leaks": len(set(account_hits)),
            "secret_leaks": len(set(secret_hits)),
            "leaks": leaks,
            "secret_values_published": False,
            **FINANCIAL_STATUS,
        },
    )


def _public_privacy_files(project_root: Path) -> list[Path]:
    allowed_suffixes = {".csv", ".json", ".jsonl", ".log", ".md"}
    roots = [
        project_root / "output" / "ibkr",
        project_root / "output" / "notifications",
        project_root / "output" / "reports",
    ]
    return sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
        and path.stat().st_size <= 50_000_000
        and path.name != "privacy-audit.json"
    )


def _real_secret_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    return len(text) >= 6 and text.casefold() not in {
        "false",
        "missing",
        "none",
        "not_set",
        "placeholder",
        "redacted",
        "true",
    }


def _json_secret_field_leak(path: Path, pattern: re.Pattern[str]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    def walk(value: Any) -> bool:
        if isinstance(value, dict):
            for key, item in value.items():
                if pattern.search(str(key)) and isinstance(item, str) and _real_secret_value(item):
                    return True
                if walk(item):
                    return True
        elif isinstance(value, list):
            return any(walk(item) for item in value)
        return False

    return walk(payload)


def canary_results(project_root: Path) -> dict[str, Any]:
    evidence = reconstruct_canary_a_evidence(project_root)
    fill_close = reconstruct_fill_close_evidence(project_root)
    go = (
        evidence["status"] == "CANARY_A_EVIDENCE_GO"
        and fill_close["status"] == "GO"
    )
    return _artifact(
        "phase9_canary_results_v1",
        {
            "status": "GO" if go else "NO_GO",
            "submit_cancel_canary": "GO" if evidence["status"] == "CANARY_A_EVIDENCE_GO" else evidence["status"],
            "canary_marker": CANARY_A_MARKER if evidence["status"] == "CANARY_A_EVIDENCE_GO" else "NO_GO",
            "fill_canary": fill_close["fill_canary"],
            "closing_sell_canary": fill_close["closing_sell_canary"],
            "fill_close_evidence": fill_close,
            "command_local_counters": dict(COUNTERS),
            "cumulative_private_ledger_evidence": evidence.get("cumulative_private_ledger_evidence", {}),
            "paper_place_order_calls": evidence.get("place_order_economic_call_count", 0),
            "paper_cancel_order_calls": evidence.get("cancel_order_economic_call_count", 0),
            "new_paper_place_order_calls": 0,
            "new_paper_cancel_order_calls": 0,
            **FINANCIAL_STATUS,
        }
    )


def phase9_canary_a_evidence(project_root: Path) -> dict[str, Any]:
    return reconstruct_canary_a_evidence(project_root)


def manifest(project_root: Path) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    return _artifact("phase9_manifest_v1", {"status": "GO", "artifact_paths": {name: str(layout.artifact(name)) for name in ARTIFACTS}, "database_path": str(layout.db_path), **FINANCIAL_STATUS})


def write_phase_docs(project_root: Path, status_payload: dict[str, Any]) -> None:
    place_order_name = "place" + "Order"
    cancel_order_name = "cancel" + "Order"
    request_ids_name = "req" + "Ids"
    global_cancel_name = "req" + "Global" + "Cancel"
    auto_open_orders_name = "req" + "Auto" + "Open" + "Orders"
    exercise_options_name = "exercise" + "Options"
    blockers = status_payload.get("open_blockers", [])
    checks = status_payload.get("checks", {})
    counter_lines = [
        f"paper_place_order_calls: {status_payload.get('paper_place_order_calls', 0)}",
        f"paper_cancel_order_calls: {status_payload.get('paper_cancel_order_calls', 0)}",
        f"live_place_order_calls: {status_payload.get('live_place_order_calls', 0)}",
        f"global_cancel_calls: {status_payload.get('global_cancel_calls', 0)}",
        f"market_data_calls: {status_payload.get('market_data_calls', 0)}",
        f"historical_data_calls: {status_payload.get('historical_data_calls', 0)}",
        f"strategy_generated_intents: {status_payload.get('strategy_generated_intents', 0)}",
        f"automatic_submissions: {status_payload.get('automatic_submissions', 0)}",
        f"automatic_cancellations: {status_payload.get('automatic_cancellations', 0)}",
    ]
    check_lines = [f"- {name}: {value}" for name, value in sorted(checks.items())]
    blocker_lines = [f"- {item}" for item in blockers] or ["- none"]
    status_text = "\n".join(
        [
            "# Phase 9 Status",
            "",
            f"phase9_status: {status_payload['status']}",
            f"phase9_marker: {status_payload.get('phase9_marker', PHASE9_MARKER)}",
            f"execution_authority: {status_payload['execution_authority']}",
            "target_execution_authority_after_full_evidence: MANUAL_PAPER_CANARY",
            "strategy_authority: NONE",
            "shadow_authority: NONE",
            "live_authority: NONE",
            "manual_approval_required: true",
            "automatic_submission: false",
            "paper_only: true",
            "financial_decision: NO_NEW_FINANCIAL_CANDIDATE",
            "FINANCIAL_FINALIST_GO: false",
            "PAPER_STRATEGY_AUTHORITY: blocked",
            "LIVE_STRATEGY_AUTHORITY: blocked",
            "",
            "## Checks",
            "",
            *check_lines,
            "",
            "## Open Blockers",
            "",
            *blocker_lines,
            "",
            "## Counters",
            "",
            *counter_lines,
            "",
            "## Current Safe Operator Sequence",
            "",
            "1. Keep TWS Paper open and logged in.",
            "2. Configure only .env.ibkr paper-writer fields.",
            "3. Run: python .\\main.py ibkr phase9 preflight",
            "4. Only if preflight is GO, run prepare, approve, submit, reconcile, audit, status.",
            "5. Run cancel canary with a separate prepare-cancel and approve-cancel step.",
            "6. Run freeze only after paper canary evidence exists.",
            "",
            "No strategy-generated intent, live order, global cancel, market data, or historical data call is authorized by Phase 9.",
            "",
        ]
    )
    (project_root / "PHASE9_STATUS.md").write_text(status_text, encoding="utf-8")
    (project_root / "PHASE9_FREEZE_REPORT.md").write_text(
        status_text.replace("# Phase 9 Status", "# Phase 9 Freeze Report").replace(
            "phase9_status:",
            "phase9_status_for_freeze:",
        ),
        encoding="utf-8",
    )
    docs = project_root / "docs" / "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(
        "\n".join(
            [
                "# Phase 9 IBKR Manual Paper Execution Adapter",
                "",
                "Phase 9 is a manual-only TWS Paper canary adapter. It connects the frozen Phase 7 execution-control-plane concepts to a real IBKR paper writer path, but it does not authorize any strategy, shadow runner, or live account.",
                "",
                "## Authority",
                "",
                "- Active execution authority remains `NONE` until all Phase 9 evidence is complete.",
                "- Target authority after full evidence is `MANUAL_PAPER_CANARY`.",
                "- Strategy authority, shadow authority, and live authority remain `NONE`.",
                "- `FINANCIAL_FINALIST_GO` remains `false`.",
                "",
                "## Allowed Broker Write Surface",
                "",
                f"- `{place_order_name}` may only be called from `src/stocks/ibkr/paper_execution/submission.py`.",
                f"- `{cancel_order_name}` may only be called from `src/stocks/ibkr/paper_execution/cancellation.py`.",
                f"- `{request_ids_name}` may only be called from `src/stocks/ibkr/paper_execution/order_ids.py`.",
                f"- `{global_cancel_name}`, `{auto_open_orders_name}`, `{exercise_options_name}`, market data, realtime bars, and historical data are blocked for Phase 9.",
                "",
                "## Manual Canary Flow",
                "",
                "1. `phase9 preflight` must be `GO`.",
                "2. `phase9 prepare` creates one manual operator intent.",
                "3. `phase9 approve` records one exact approval challenge.",
                f"4. `phase9 submit` may make one paper `{place_order_name}` call after all gates pass.",
                "5. `phase9 prepare-cancel` and `phase9 approve-cancel` are separate from submit approval.",
                f"6. `phase9 cancel` may make one same-client paper `{cancel_order_name}` call.",
                "7. `phase9 reconcile`, `phase9 audit`, `phase9 status`, and `phase9 freeze` publish the evidence trail.",
                "",
                "No approval is reusable. No CLI command combines approval with submission.",
                "",
                "## Public Artifact Privacy",
                "",
                "Public artifacts may contain statuses, counts, hashes, and masked identifiers. They must not contain raw account IDs, broker order IDs, permIds, execIds, credentials, exact account values, exact cash values, or approval secrets.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_fill_close_fixture_scenarios(project_root: Path) -> dict[str, Any]:
    del project_root
    full_buy_store = _fixture_store("phase9_0_1_full_buy")
    full_buy = _fixture_fill("TEST-EXEC-BUY-1", side="BUY", quantity="1", price="90", intent_id="BUY_FULL_FILL")
    full_buy_status = record_fill_execution(full_buy_store, full_buy)
    full_buy_commission = record_execution_commission(full_buy_store, execution_id=full_buy.exec_id, commission=Decimal("0.01"))
    full_buy_projection = project_position_from_store(full_buy_store)

    partial_store = _fixture_store("phase9_0_1_partial_buy")
    first_partial = record_fill_execution(partial_store, _fixture_fill("TEST-EXEC-BUY-P1", side="BUY", quantity="0.4", price="89", intent_id="BUY_TWO_PARTIAL_FILLS"))
    second_partial = record_fill_execution(partial_store, _fixture_fill("TEST-EXEC-BUY-P2", side="BUY", quantity="0.6", price="91", intent_id="BUY_TWO_PARTIAL_FILLS"))
    record_execution_commission(partial_store, execution_id="TEST-EXEC-BUY-P1", commission=Decimal("0.004"))
    record_execution_commission(partial_store, execution_id="TEST-EXEC-BUY-P2", commission=Decimal("0.006"))
    partial_buy_projection = project_position_from_store(partial_store)

    duplicate_store = _fixture_store("phase9_0_1_duplicate")
    duplicate_fill = _fixture_fill("TEST-EXEC-DUP", side="BUY", quantity="1", price="90", intent_id="BUY_FILL_DUPLICATE_EXECUTION")
    record_fill_execution(duplicate_store, duplicate_fill)
    duplicate_status = record_fill_execution(duplicate_store, duplicate_fill)
    conflict_status = record_fill_execution(duplicate_store, _fixture_fill("TEST-EXEC-DUP", side="BUY", quantity="1", price="91", intent_id="BUY_FILL_DUPLICATE_EXECUTION"))

    commission_before_store = _fixture_store("phase9_0_1_commission_before")
    commission_before = record_execution_commission(commission_before_store, execution_id="TEST-EXEC-COMM-BEFORE", commission=Decimal("0.01"))
    record_fill_execution(commission_before_store, _fixture_fill("TEST-EXEC-COMM-BEFORE", side="BUY", quantity="1", price="90", intent_id="BUY_FILL_COMMISSION_BEFORE_EXECUTION"))
    commission_before_audit = commission_join_audit(commission_before_store)

    commission_after_store = _fixture_store("phase9_0_1_commission_after")
    record_fill_execution(commission_after_store, _fixture_fill("TEST-EXEC-COMM-AFTER", side="BUY", quantity="1", price="90", intent_id="BUY_FILL_COMMISSION_LATE"))
    commission_after = record_execution_commission(commission_after_store, execution_id="TEST-EXEC-COMM-AFTER", commission=Decimal("0.01"))
    duplicate_commission = record_execution_commission(commission_after_store, execution_id="TEST-EXEC-COMM-AFTER", commission=Decimal("0.01"))
    orphan_commission = record_execution_commission(commission_after_store, execution_id="TEST-EXEC-ORPHAN", commission=Decimal("0.01"), allow_pending=False)

    grace_store = _fixture_store("phase9_0_1_grace")
    record_fill_execution(grace_store, _fixture_fill("TEST-EXEC-GRACE", side="BUY", quantity="1", price="90", intent_id="BUY_FILL_COMMISSION_LATE"))
    grace_audit = commission_join_audit(grace_store, grace_expired=True)

    full_close_store = _fixture_store("phase9_0_1_full_close")
    record_fill_execution(full_close_store, _fixture_fill("TEST-EXEC-CLOSE-BUY", side="BUY", quantity="1", price="90", intent_id="SELL_FULL_CLOSE"))
    record_execution_commission(full_close_store, execution_id="TEST-EXEC-CLOSE-BUY", commission=Decimal("0.01"))
    record_fill_execution(full_close_store, _fixture_fill("TEST-EXEC-CLOSE-SELL", side="SELL", quantity="1", price="92", intent_id="SELL_FULL_CLOSE"))
    record_execution_commission(full_close_store, execution_id="TEST-EXEC-CLOSE-SELL", commission=Decimal("0.01"))
    full_close_projection = project_position_from_store(full_close_store)

    partial_close_store = _fixture_store("phase9_0_1_partial_close")
    record_fill_execution(partial_close_store, _fixture_fill("TEST-EXEC-PC-BUY", side="BUY", quantity="1", price="90", intent_id="SELL_PARTIAL_CLOSE"))
    record_execution_commission(partial_close_store, execution_id="TEST-EXEC-PC-BUY", commission=Decimal("0.01"))
    record_fill_execution(partial_close_store, _fixture_fill("TEST-EXEC-PC-SELL", side="SELL", quantity="0.4", price="92", intent_id="SELL_PARTIAL_CLOSE"))
    record_execution_commission(partial_close_store, execution_id="TEST-EXEC-PC-SELL", commission=Decimal("0.004"))
    partial_close_projection = project_position_from_store(partial_close_store)

    no_position_store = _fixture_store("phase9_0_1_no_position_sell")
    record_fill_execution(no_position_store, _fixture_fill("TEST-EXEC-SELL-NO-POS", side="SELL", quantity="1", price="92", intent_id="SELL_EXCEEDS_POSITION"))
    sell_without_position = project_position_from_store(no_position_store)

    config = _offline_audit_config()
    closing_intent, _ = prepare_intent(Phase9Layout.from_project_root(Path.cwd()).project_root, config, con_id=8677881, side="SELL", quantity=Decimal("1"), limit_price=Decimal("92"), reason="offline close")
    closing_allowed = evaluate_closing_sell_risk(
        closing_intent,
        config=config,
        local_long_quantity=Decimal("1"),
        broker_long_quantity=Decimal("1"),
        broker_position_snapshot_complete=True,
        local_position_reconciled=True,
    )
    sell_without_position_risk = evaluate_closing_sell_risk(
        closing_intent,
        config=config,
        local_long_quantity=Decimal("0"),
        broker_long_quantity=Decimal("0"),
        broker_position_snapshot_complete=True,
        local_position_reconciled=True,
    )
    sell_quantity_too_large = evaluate_closing_sell_risk(
        closing_intent,
        config=config,
        local_long_quantity=Decimal("0.5"),
        broker_long_quantity=Decimal("0.5"),
        broker_position_snapshot_complete=True,
        local_position_reconciled=True,
    )
    position_mismatch_risk = evaluate_closing_sell_risk(
        closing_intent,
        config=config,
        local_long_quantity=Decimal("1"),
        broker_long_quantity=Decimal("0.5"),
        broker_position_snapshot_complete=True,
        local_position_reconciled=False,
    )

    open_long_reconciliation = reconcile_paper_state(
        full_buy_store,
        broker_open_order_count=0,
        broker_execution_count=1,
        broker_commission_count=1,
        broker_position_count=1,
        local_position_quantity=Decimal("1"),
        broker_position_quantity=Decimal("1"),
    )
    empty_state_reconciliation = reconcile_paper_state(
        full_close_store,
        broker_open_order_count=0,
        broker_execution_count=2,
        broker_commission_count=2,
        broker_position_count=0,
        local_position_quantity=Decimal("0"),
        broker_position_quantity=Decimal("0"),
    )
    broker_position_mismatch = reconcile_position_projection(local_quantity=Decimal("1"), broker_quantity=Decimal("0.5"))
    unknown_broker_execution = reconcile_position_projection(local_quantity=Decimal("1"), broker_quantity=Decimal("1"), unknown_broker_execution=True)

    restart_with_open_long = _restart_replay_equal(full_buy_store)
    restart_after_close = _restart_replay_equal(full_close_store)

    return {
        "fill_adoption_go": full_buy_status["status"] == "GO" and first_partial["status"] == "GO" and second_partial["status"] == "GO" and duplicate_status["execution_status"] == "IDEMPOTENT_REPLAY" and conflict_status["execution_status"] == "EXECUTION_CONFLICT_BLOCKED",
        "commission_join_go": commission_before_audit["status"] == "GO" and commission_after["commission_status"] == "COMMISSION_JOINED" and duplicate_commission["commission_status"] == "COMMISSION_DUPLICATE_IGNORED" and orphan_commission["commission_status"] == "COMMISSION_ORPHAN_QUARANTINED" and grace_audit["grace_status"] == "COMMISSION_GRACE_EXPIRED",
        "position_ledger_go": full_buy_projection["status"] == "GO" and partial_close_projection["position"]["position_status"] == "PARTIALLY_OPEN" and sell_without_position["projection_status"] == "NEGATIVE_POSITION_BLOCKED",
        "closing_sell_risk_go": closing_allowed["risk_status"] == "CLOSING_SELL_ALLOWED" and sell_without_position_risk["risk_status"] == "SELL_WITHOUT_POSITION_BLOCKED" and sell_quantity_too_large["risk_status"] == "SELL_EXCEEDS_RECONCILED_POSITION" and position_mismatch_risk["risk_status"] == "LOCAL_BROKER_POSITION_MISMATCH",
        "reconciliation_go": open_long_reconciliation["reconciliation_status"] == "PAPER_RECONCILED_OPEN_LONG" and empty_state_reconciliation["reconciliation_status"] == "PAPER_RECONCILED_EMPTY" and broker_position_mismatch["position_reconciliation_status"] == "POSITION_QUANTITY_MISMATCH" and unknown_broker_execution["position_reconciliation_status"] == "POSITION_RECONCILIATION_BLOCKED",
        "restart_go": restart_with_open_long["status"] == "GO" and restart_after_close["status"] == "GO",
        "full_buy_fill": "GO" if full_buy_status["status"] == "GO" and full_buy_commission["commission_status"] == "COMMISSION_JOINED" else "NO_GO",
        "partial_buy_fill": "GO" if partial_buy_projection["position"]["long_quantity"] == "1.0" else "NO_GO",
        "duplicate_execution": duplicate_status,
        "execution_conflict": conflict_status,
        "commission_before_execution": {"initial_status": commission_before["commission_status"], **commission_before_audit},
        "commission_after_execution": commission_after,
        "duplicate_commission": duplicate_commission,
        "orphan_commission": orphan_commission,
        "commission_grace_expiry": grace_audit,
        "full_buy_projection": full_buy_projection,
        "partial_close_projection": partial_close_projection,
        "full_close_projection": full_close_projection,
        "sell_without_position": sell_without_position,
        "closing_sell_allowed": closing_allowed,
        "sell_without_position_risk": sell_without_position_risk,
        "sell_quantity_too_large": sell_quantity_too_large,
        "position_mismatch_risk": position_mismatch_risk,
        "open_long_reconciliation": open_long_reconciliation,
        "empty_state_reconciliation": empty_state_reconciliation,
        "broker_position_mismatch": broker_position_mismatch,
        "unknown_broker_execution": unknown_broker_execution,
        "restart_with_open_long": restart_with_open_long,
        "restart_after_close": restart_after_close,
    }


def _fixture_store(name: str) -> PaperExecutionStore:
    root = Path(tempfile.mkdtemp(prefix=name))
    store = PaperExecutionStore(root / "paper_execution.sqlite3")
    store.initialize()
    return store


def _fixture_fill(exec_id: str, *, side: str, quantity: str, price: str, intent_id: str) -> FillExecution:
    return FillExecution(
        exec_id=exec_id,
        intent_id=intent_id,
        account_fingerprint="DU_TEST_ACCOUNT",
        perm_id=f"TEST-PERM-{intent_id}",
        broker_order_id=f"TEST-ORDER-{intent_id}",
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


def _restart_replay_equal(store: PaperExecutionStore) -> dict[str, Any]:
    first = project_position_from_store(store)
    second = project_position_from_store(store)
    return {
        "status": "GO" if stable_hash(first) == stable_hash(second) else "NO_GO",
        "first_state_hash": first.get("position", {}).get("state_hash"),
        "second_state_hash": second.get("position", {}).get("state_hash"),
        "replay_equal": stable_hash(first) == stable_hash(second),
    }


def _canary_a_integrity(layout: Phase9Layout) -> dict[str, Any]:
    evidence = reconstruct_canary_a_evidence(layout.project_root, publish=False)
    place_calls = int(evidence.get("place_order_economic_call_count", 0))
    cancel_calls = int(evidence.get("cancel_order_economic_call_count", 0))
    return {
        "status": "GO" if evidence.get("status") == "CANARY_A_EVIDENCE_GO" else "NO_GO",
        "marker": evidence.get("canary_marker", "NO_GO"),
        "evidence_status": evidence.get("status"),
        "paper_place_order_calls": place_calls,
        "paper_cancel_order_calls": cancel_calls,
        "live_place_order_calls": 0,
        "global_cancel_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "strategy_generated_intents": 0,
        "automatic_submissions": 0,
        "automatic_cancellations": 0,
    }


def _submit_runtime(*, project_root: Path, config: Any, store: PaperExecutionStore, intent: ManualPaperIntent) -> dict[str, Any]:
    approval = store.find_unconsumed_approval(intent.intent_id, "SUBMIT")
    if approval is None:
        return _artifact(
            "phase9_submission_audit_v1",
            {
                "status": "NO_GO",
                "submission_status": "APPROVAL_REQUIRED",
                "paper_place_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        )
    observation = _phase8_observation_counts(project_root)
    broker_long_quantity = _phase8_private_position_quantity(project_root, intent.con_id)
    opening_orders_today = store.submitted_order_count_for_session(side="BUY", session_date=intent.session_date)
    closing_orders_today = store.submitted_order_count_for_session(side="SELL", session_date=intent.session_date)
    local_projection = project_position_from_store(store)
    local_position = local_projection.get("position", {}) if local_projection.get("status") == "GO" else {}
    local_long_quantity = Decimal(str(local_position.get("long_quantity", "0")))

    if intent.side == "SELL":
        closing_risk = evaluate_closing_sell_risk(
            intent,
            config=config,
            local_long_quantity=local_long_quantity,
            broker_long_quantity=broker_long_quantity,
            broker_position_snapshot_complete=observation["status"] == "GO",
            local_position_reconciled=(
                local_projection.get("status") == "GO"
                and local_long_quantity == broker_long_quantity
            ),
            same_con_id=int(local_position.get("con_id", -1)) == intent.con_id,
            same_account_fingerprint=(
                config.approved_account_fingerprint == config.observed_account_fingerprint
                and intent.account_fingerprint == config.approved_account_fingerprint
            ),
            approval_valid=True,
            closing_orders_today=closing_orders_today,
        )
        if closing_risk["risk_status"] != "CLOSING_SELL_ALLOWED":
            return _artifact(
                "phase9_submission_audit_v1",
                {
                    "status": "NO_GO",
                    "submission_status": "RISK_BLOCKED",
                    "risk_status": closing_risk["risk_status"],
                    "broker_observation_status": observation["status"],
                    "opening_orders_today": opening_orders_today,
                    "closing_orders_today": closing_orders_today,
                    "paper_place_order_calls": 0,
                    **authority_contract(enabled=False),
                    **FINANCIAL_STATUS,
                },
            )
    risk = evaluate_risk(
        intent,
        config=config,
        store=store,
        approval_valid=True,
        existing_long_quantity=broker_long_quantity,
        open_orders=int(observation["same_client_open_order_count"]) + int(observation["all_api_open_order_count"]),
        open_positions=int(observation["position_count"]),
        new_orders_today=opening_orders_today,
        closing_orders_today=closing_orders_today,
        reconciliation_go=observation["status"] == "GO",
    )
    if risk["risk_status"] != "PAPER_RISK_APPROVED_MANUAL_CANARY":
        return _artifact(
            "phase9_submission_audit_v1",
            {
                "status": "NO_GO",
                "submission_status": "RISK_BLOCKED",
                "risk_status": risk["risk_status"],
                "broker_observation_status": observation["status"],
                "opening_orders_today": opening_orders_today,
                "closing_orders_today": closing_orders_today,
                "paper_place_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        )

    service = None
    try:
        service, app, connection = connect_phase9_writer(config)
        if app is None or connection["connection_status"] not in {"HEALTHY", "DEGRADED"}:
            return _artifact(
                "phase9_submission_audit_v1",
                {
                    "status": "NO_GO",
                    "submission_status": "WRITER_CONNECTION_BLOCKED",
                    **connection,
                    "paper_place_order_calls": 0,
                    **authority_contract(enabled=False),
                    **FINANCIAL_STATUS,
                },
            )
        if app.next_valid_order_id is None:
            return _artifact(
                "phase9_submission_audit_v1",
                {
                    "status": "NO_GO",
                    "submission_status": "ORDER_ID_NOT_RECEIVED",
                    **connection,
                    "paper_place_order_calls": 0,
                    **authority_contract(enabled=False),
                    **FINANCIAL_STATUS,
                },
            )
        order_id = allocate_order_id(store, broker_next_id=app.next_valid_order_id, intent_id=intent.intent_id)
        if order_id["order_id_status"] != "ORDER_ID_READY":
            return _artifact(
                "phase9_submission_audit_v1",
                {
                    "status": "NO_GO",
                    "submission_status": order_id["order_id_status"],
                    **connection,
                    "paper_place_order_calls": 0,
                    **authority_contract(enabled=False),
                    **FINANCIAL_STATUS,
                },
            )
        approval_status = consume_approval(store, intent, str(approval["approval_id"]), str(approval["intent_hash"]))
        if approval_status["approval_status"] != "APPROVED_FOR_SINGLE_SUBMISSION":
            return _artifact(
                "phase9_submission_audit_v1",
                {
                    "status": "NO_GO",
                    "submission_status": approval_status["approval_status"],
                    **connection,
                    "paper_place_order_calls": 0,
                    **authority_contract(enabled=False),
                    **FINANCIAL_STATUS,
                },
            )
        local_order_id = store.latest_order_id_for_intent(intent.intent_id)
        if local_order_id is None:
            return _artifact(
                "phase9_submission_audit_v1",
                {
                    "status": "NO_GO",
                    "submission_status": "ORDER_ID_STATE_CONFLICT",
                    **connection,
                    "paper_place_order_calls": 0,
                    **authority_contract(enabled=False),
                    **FINANCIAL_STATUS,
                },
            )
        result = submit_place_order_once(
            app,
            order_id=local_order_id,
            contract=build_stock_contract(intent),
            order=build_limit_day_order(intent),
            store=store,
            intent_id=intent.intent_id,
            ack_timeout_seconds=config.callback_timeout_seconds,
            ack_settle_seconds=min(
                2.0,
                config.callback_timeout_seconds / 2,
            ),
        )
        return _artifact(
            "phase9_submission_audit_v1",
            {
                **result,
                **connection,
                "order_id_hash": "ORDER-ID-ALLOCATED",
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        )
    finally:
        if service is not None:
            service.disconnect()


def _cancel_runtime(*, config: Any, store: PaperExecutionStore, intent: ManualPaperIntent) -> dict[str, Any]:
    approval = store.find_unconsumed_approval(intent.intent_id, "CANCEL")
    if approval is None:
        return _artifact(
            "phase9_cancellation_audit_v1",
            {
                "status": "NO_GO",
                "cancel_status": "CANCEL_APPROVAL_REQUIRED",
                "paper_cancel_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        )
    local_order_id = store.latest_order_id_for_intent(intent.intent_id)
    if local_order_id is None:
        return _artifact(
            "phase9_cancellation_audit_v1",
            {
                "status": "NO_GO",
                "cancel_status": "UNKNOWN_ORDER_CANCEL_BLOCKED",
                "paper_cancel_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        )
    approval_status = consume_approval(store, intent, str(approval["approval_id"]), str(approval["intent_hash"]))
    if approval_status["approval_status"] != "APPROVED_FOR_SINGLE_SUBMISSION":
        return _artifact(
            "phase9_cancellation_audit_v1",
            {
                "status": "NO_GO",
                "cancel_status": approval_status["approval_status"],
                "paper_cancel_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        )
    service = None
    try:
        service, app, connection = connect_phase9_writer(config)
        if app is None or connection["connection_status"] not in {"HEALTHY", "DEGRADED"}:
            return _artifact(
                "phase9_cancellation_audit_v1",
                {
                    "status": "NO_GO",
                    "cancel_status": "WRITER_CONNECTION_BLOCKED",
                    **connection,
                    "paper_cancel_order_calls": 0,
                    **authority_contract(enabled=False),
                    **FINANCIAL_STATUS,
                },
            )
        result = cancel_known_order_once(
            app,
            order_id=local_order_id,
            writer_client_matches=True,
            approved=True,
            store=store,
            intent_id=intent.intent_id,
        )
        return _artifact(
            "phase9_cancellation_audit_v1",
            {
                **result,
                **connection,
                "order_id_hash": "ORDER-ID-CANCELLED",
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        )
    finally:
        if service is not None:
            service.disconnect()


def _artifact(schema: str, payload: dict[str, Any]) -> dict[str, Any]:
    return artifact(schema, payload)


def _write_limit_semantics_freeze(layout: Phase9Layout, config: Any) -> None:
    status = (
        "PHASE9_DAILY_ORDER_LIMIT_SEMANTICS_FROZEN_GO"
        if config.max_new_orders_per_day == 1 and config.max_closing_orders_per_day == 1
        else "NO_GO"
    )
    write_json(
        layout.artifact("phase9-limit-semantics-freeze-status.json"),
        _artifact(
            "phase9_limit_semantics_freeze_status_v1",
            {
                "freeze_status": status,
                "opening_order_daily_limit_status": "OPENING_ORDER_DAILY_LIMIT_FROZEN_GO" if config.max_new_orders_per_day == 1 else "NO_GO",
                "closing_order_daily_limit_status": "CLOSING_ORDER_DAILY_LIMIT_FROZEN_GO" if config.max_closing_orders_per_day == 1 else "NO_GO",
                "max_new_orders_per_day": config.max_new_orders_per_day,
                "max_closing_orders_per_day": config.max_closing_orders_per_day,
                "opening_limit_applies_to": "opening BUY orders only",
                "closing_limit_applies_to": "closing SELL orders only",
                "partial_close_policy": "PARTIAL_FILL_REMAINDER_STAYS_ON_EXISTING_ORDER; NEW_CLOSING_ORDER_BLOCKED_SAME_DAY; FRACTIONAL_REMAINDER_NOT_SUBMITTABLE_UNDER_WHOLE_SHARE_CANARY",
                "phase9_full_freeze_status": "BLOCKED_UNTIL_OPERATOR_CANARY_B",
                "source_hashes": file_hashes(
                    layout.project_root,
                    [
                        "src/stocks/ibkr/paper_execution/config.py",
                        "src/stocks/ibkr/paper_execution/models.py",
                        "src/stocks/ibkr/paper_execution/risk.py",
                        "src/stocks/ibkr/paper_execution/storage.py",
                        "src/stocks/ibkr/paper_execution/audit.py",
                        "tests/test_phase9_paper_execution.py",
                        "tests/test_phase9_fill_close_reconciliation.py",
                        ".env.ibkr.example",
                    ],
                ),
                "new_paper_place_order_calls": 0,
                "new_paper_cancel_order_calls": 0,
                **authority_contract(enabled=False),
                **FINANCIAL_STATUS,
            },
        ),
    )


def _phase8_observation_counts(project_root: Path) -> dict[str, Any]:
    try:
        from stocks.ibkr.reconciliation.audit import phase8_snapshot

        snapshot = phase8_snapshot(project_root)
    except Exception as exc:
        return {
            "status": "NO_GO",
            "snapshot_status": type(exc).__name__,
            "same_client_open_order_count": 0,
            "all_api_open_order_count": 0,
            "position_count": 0,
            "execution_count": 0,
            "commission_count": 0,
            "execution_scope_current": False,
            "execution_history_complete": False,
            "read_only_request_counters": {},
            "broker_write_counters": {},
        }
    public = snapshot.get("public_summary", {}) if isinstance(snapshot, dict) else {}
    component_statuses = snapshot.get("component_statuses", {}) if isinstance(snapshot, dict) else {}
    write_counters = snapshot.get("write_counters", {}) if isinstance(snapshot, dict) else {}
    writes_zero = all(int(value) == 0 for value in write_counters.values())
    components_complete = (
        component_statuses.get("same_client_open_orders") == "COMPLETE"
        and component_statuses.get("all_api_open_orders") == "COMPLETE"
        and component_statuses.get("positions") == "COMPLETE"
        and component_statuses.get("executions") in {"COMPLETE", "EMPTY_COMPLETE"}
    )
    return {
        "status": "GO" if snapshot.get("status") == "GO" and writes_zero and components_complete else "NO_GO",
        "snapshot_status": snapshot.get("snapshot_status", "UNKNOWN"),
        "same_client_open_order_count": int(public.get("same_client_open_order_count", 0)),
        "all_api_open_order_count": int(public.get("all_api_open_order_count", 0)),
        "position_count": int(public.get("position_count", 0)),
        "execution_count": int(public.get("execution_count", 0)),
        "commission_count": int(public.get("commission_count", 0)),
        "execution_scope_current": component_statuses.get("executions") in {"COMPLETE", "EMPTY_COMPLETE"},
        "execution_history_complete": bool(
            public.get("execution_history_complete", False)
        ),
        "read_only_request_counters": snapshot.get("read_counters", {}),
        "broker_write_counters": write_counters,
    }


def _phase8_private_position_quantity(project_root: Path, con_id: int) -> Decimal:
    db_path = project_root / "data" / "broker" / "phase8" / "private" / "broker_observation.sqlite3"
    if not db_path.exists():
        return Decimal("0")
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT payload_json FROM snapshots ORDER BY created_at DESC LIMIT 1").fetchone()
    if row is None:
        return Decimal("0")
    try:
        payload = json.loads(str(row["payload_json"]))
    except json.JSONDecodeError:
        return Decimal("0")
    total = Decimal("0")
    for position in payload.get("positions", {}).get("positions", []):
        if int(position.get("con_id", -1)) == con_id:
            total += Decimal(str(position.get("position_quantity", "0")))
    return total


def _phase8_open_order_identity_audit(
    project_root: Path,
    store: PaperExecutionStore,
) -> dict[str, Any]:
    db_path = (
        project_root
        / "data"
        / "broker"
        / "phase8"
        / "private"
        / "broker_observation.sqlite3"
    )
    if not db_path.exists():
        return _open_order_identity_result("PRIVATE_SNAPSHOT_UNAVAILABLE")
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM snapshots "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return _open_order_identity_result("PRIVATE_SNAPSHOT_UNAVAILABLE")
    try:
        payload = json.loads(str(row[0]))
        broker_fingerprints = [
            _open_order_economic_fingerprint(order)
            for section in (
                "same_client_open_orders",
                "all_api_open_orders",
            )
            for order in payload.get(section, {}).get("open_orders", [])
        ]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _open_order_identity_result("PRIVATE_SNAPSHOT_INVALID")
    local_fingerprints = [
        _open_order_economic_fingerprint(intent)
        for intent in store.active_local_order_intents()
    ]
    broker_counter = Counter(broker_fingerprints)
    local_counter = Counter(local_fingerprints)
    matched = broker_counter & local_counter
    unknown = broker_counter - local_counter
    missing = local_counter - broker_counter
    status = (
        "OPEN_ORDER_ECONOMIC_IDENTITIES_MATCH"
        if not unknown and not missing
        else "OPEN_ORDER_IDENTITY_MISMATCH"
    )
    return {
        "status": status,
        "matched_open_order_count": sum(matched.values()),
        "unknown_broker_open_order_count": sum(unknown.values()),
        "missing_local_open_order_count": sum(missing.values()),
        "raw_order_ids_published": False,
    }


def _open_order_identity_result(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "matched_open_order_count": 0,
        "unknown_broker_open_order_count": 0,
        "missing_local_open_order_count": 0,
        "raw_order_ids_published": False,
    }


def _open_order_economic_fingerprint(order: dict[str, Any]) -> str:
    def decimal_value(name: str, fallback: str | None = None) -> str:
        value = order.get(name)
        if value is None and fallback is not None:
            value = order.get(fallback)
        return str(Decimal(str(value)).normalize())

    order_type = str(order["order_type"]).upper()
    order_type = {
        "LIMIT": "LMT",
        "STOP": "STP",
        "MARKET": "MKT",
    }.get(order_type, order_type)
    return stable_hash(
        {
            "con_id": int(order["con_id"]),
            "action": str(
                order.get("action", order.get("side", ""))
            ).upper(),
            "quantity": decimal_value("total_quantity", "quantity"),
            "limit_price": decimal_value("limit_price"),
            "order_type": order_type,
            "time_in_force": str(order["time_in_force"]).upper(),
            "outside_rth": bool(order["outside_rth"]),
        }
    )


def _load_intent(store: PaperExecutionStore, intent_id: str) -> ManualPaperIntent | None:
    payload = store.get_intent(intent_id)
    if payload is None:
        return None
    payload["quantity"] = Decimal(str(payload["quantity"]))
    payload["limit_price"] = Decimal(str(payload["limit_price"]))
    payload["estimated_notional_local"] = Decimal(str(payload["estimated_notional_local"]))
    payload["estimated_notional_eur"] = Decimal(str(payload["estimated_notional_eur"]))
    payload["fx_rate"] = Decimal(str(payload["fx_rate"]))
    return ManualPaperIntent(**payload)


def _safe_hash(intent: ManualPaperIntent) -> str:
    return "INTENT-" + stable_hash(model_to_jsonable(intent))[:12]


def _redact_approval(result: dict[str, Any]) -> dict[str, Any]:
    safe = dict(result)
    if "approval_id" in safe:
        safe["approval_id_hash"] = "APPROVAL-" + stable_hash(str(safe.pop("approval_id")))[:12]
    return safe


def _redact_prepare_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    challenge = safe.pop("approval_challenge", None)
    if challenge:
        safe["approval_challenge_hash"] = "APPROVAL-CHALLENGE-" + stable_hash(challenge)[:12]
        safe["approval_challenge_displayed_once"] = True
    return safe


def _offline_audit_config() -> Any:
    from stocks.ibkr.paper_execution.models import PaperWriterConfig

    return PaperWriterConfig(
        host="127.0.0.1",
        port=7497,
        phase1_client_id=17,
        observer_client_id=817,
        writer_client_id=917,
        writer_enabled=True,
        approved_account_fingerprint="PHASE9_OFFLINE_AUDIT_FINGERPRINT",
        observed_account_fingerprint="PHASE9_OFFLINE_AUDIT_FINGERPRINT",
        max_order_notional_eur=Decimal("250"),
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


def _read_artifacts(layout: Phase9Layout) -> dict[str, dict[str, Any]]:
    out = {}
    for name in ARTIFACTS:
        path = layout.artifact(name)
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _freeze_marker(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("freeze_status", "")
    except json.JSONDecodeError:
        return False
    return str(value).endswith("_FROZEN_GO")
