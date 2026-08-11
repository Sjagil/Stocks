from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.callbacks import (
    CallbackAuditState,
    accept_callback,
)
from stocks.ibkr.paper_execution.commissions import (
    commission_join_audit,
    record_execution_commission,
)
from stocks.ibkr.paper_execution.executions import (
    FillExecution,
    project_position_from_store,
    record_fill_execution,
)
from stocks.ibkr.paper_execution.reconciliation import (
    reconcile_paper_state,
    reconcile_position_projection,
)
from stocks.ibkr.paper_execution.storage import PaperExecutionStore
from stocks.ibkr.p0_scenarios import (
    EXTENDED_REQUIRED_SCENARIOS,
    run_extended_scenarios,
)


P0_SCHEMA = "ibkr_execution_p0_safety_matrix_v1"
P0_MARKER = "IBKR_EXECUTION_P0_SAFETY_MATRIX_GO"
P0_ARTIFACT = Path("output/ibkr/phase9/p0-safety-matrix.json")
P0_BLOCKER = "IBKR_P0_SAFETY_MATRIX_REQUIRED"
REQUIRED_SCENARIOS = (
    "NEXT_VALID_ID_MONOTONIC",
    "DUPLICATE_ORDER_STATUS_IDEMPOTENT",
    "OUT_OF_ORDER_STATUS_NO_REGRESSION",
    "PARTIAL_FILL_ACCUMULATION",
    "EXECUTION_DEDUP_BY_EXEC_ID",
    "FILL_BEFORE_COMMISSION_RECONCILES",
    "COMMISSION_BEFORE_EXECUTION_JOIN",
    "MANUAL_TWS_ORDER_FAILS_RECONCILIATION",
    "RESTART_REPLAY_IDEMPOTENT",
    "EXTERNAL_POSITION_CHANGE_BLOCKS",
    "CONNECTION_LOSS_FAIL_CLOSED",
) + EXTENDED_REQUIRED_SCENARIOS
CRITICAL_SOURCE_PATHS = (
    "src/stocks/ibkr/p0_safety.py",
    "src/stocks/ibkr/p0_scenarios.py",
    "src/stocks/ibkr/p0_readiness.py",
    "src/stocks/ibkr/reconciliation/account_state.py",
    "src/stocks/ibkr/reconciliation/callbacks.py",
    "src/stocks/ibkr/reconciliation/recovery.py",
    "src/stocks/ibkr/reconciliation/requests.py",
    "src/stocks/ibkr/paper_execution/callbacks.py",
    "src/stocks/ibkr/paper_execution/cancellation.py",
    "src/stocks/ibkr/paper_execution/commissions.py",
    "src/stocks/ibkr/paper_execution/executions.py",
    "src/stocks/ibkr/paper_execution/reconciliation.py",
    "src/stocks/ibkr/paper_execution/restart_recovery.py",
    "src/stocks/ibkr/paper_execution/state_machine.py",
    "src/stocks/ibkr/paper_execution/storage.py",
    "src/stocks/live/authority.py",
    "src/stocks/live/adapter.py",
    "src/stocks/live/config.py",
    "src/stocks/live/evidence.py",
    "src/stocks/live/models.py",
    "src/stocks/live/portfolio_targets.py",
    "src/stocks/live/service.py",
    "src/stocks/live/submission.py",
    "src/stocks/capital/service.py",
    "src/stocks/portfolio/manager.py",
    "src/stocks/portfolio/targets.py",
    "config/capital_scaling/levels_v1.json",
    "config/portfolio/active_manager_v1.json",
)
RECONCILIATION_INVARIANTS = (
    "FILLED_QUANTITY_NONNEGATIVE",
    "REMAINING_QUANTITY_NONNEGATIVE",
    "FILLED_QUANTITY_NOT_ABOVE_ORIGINAL_WITHOUT_BROKER_CORRECTION",
    "EXEC_ID_CHANGES_POSITION_EXACTLY_ONCE",
    "COMMISSION_NEVER_CHANGES_QUANTITY",
    "DUPLICATE_ORDER_STATUS_HAS_NO_ECONOMIC_EFFECT",
    "TERMINAL_ORDER_STATE_NEVER_REGRESSES_FROM_STALE_CALLBACK",
    "UNKNOWN_BROKER_POSITION_HAS_EXPLICIT_OWNERSHIP",
    "CAPITAL_RESERVED_ONCE_PER_ECONOMIC_ORDER",
    "CANCELLED_UNFILLED_QUANTITY_RELEASES_RESERVATION",
    "PARTIAL_FILL_RETAINS_ONLY_UNFILLED_RESERVATION",
    "RESTART_REPLAY_PRESERVES_ECONOMIC_STATE_HASH",
)


def build_p0_safety_report() -> dict[str, Any]:
    """Run deterministic, broker-free regression scenarios."""
    with tempfile.TemporaryDirectory(
        prefix="ibkr-p0-", ignore_cleanup_errors=True
    ) as directory:
        root = Path(directory)
        scenarios = _run_scenarios(root)
    statuses = {
        scenario_id: str(scenarios[scenario_id]["status"])
        for scenario_id in REQUIRED_SCENARIOS
    }
    all_go = all(value == "GO" for value in statuses.values())
    source_hashes = _current_source_hashes()
    sources_complete = all(source_hashes.values())
    body: dict[str, Any] = {
        "schema": P0_SCHEMA,
        "marker": P0_MARKER if all_go and sources_complete else "NO_GO",
        "status": "GO" if all_go and sources_complete else "NO_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "scenario_statuses": statuses,
        "scenarios": scenarios,
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "source_hashes": source_hashes,
        "source_hash_root": "INSTALLED_PACKAGE_PROJECT_ROOT",
        "reconciliation_invariants": list(RECONCILIATION_INVARIANTS),
        "canonical_sources_of_truth": {
            "ibkr_orders": "CANONICAL_BROKER_OBSERVATION_PLUS_PAPER_ORDER_EVENTS",
            "executions": "PAPER_EXECUTION_STORE_EXEC_ID_UNIQUE_RECORDS",
            "commissions": "PAPER_EXECUTION_STORE_JOINED_BY_EXEC_ID",
            "positions": "DERIVED_FROM_CANONICAL_EXECUTIONS_AND_BROKER_RECONCILIATION",
            "cash_account_state": "DERIVED_FROM_CANONICAL_BROKER_ACCOUNT_VALUES_AND_RESERVATIONS",
        },
        "parallel_financial_ledger_created": False,
        "broker_write_calls": 0,
        "network_calls": 0,
        "automatic_corrections": 0,
        "execution_authority": "NONE",
        "open_blockers": [
            scenario_id
            for scenario_id, status in statuses.items()
            if status != "GO"
        ]
        + ([] if sources_complete else ["P0_CRITICAL_SOURCE_MISSING"]),
    }
    body["content_hash"] = _content_hash(body)
    return body


def write_p0_safety_report(project_root: Path) -> dict[str, Any]:
    report = build_p0_safety_report()
    path = project_root / P0_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return report


def inspect_p0_regression_gate(project_root: Path) -> dict[str, Any]:
    path = project_root / P0_ARTIFACT
    blockers: list[str] = []
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        report = {}
        blockers.append("P0_SAFETY_MATRIX_MISSING_OR_INVALID")
    if report:
        if report.get("schema") != P0_SCHEMA:
            blockers.append("P0_SAFETY_MATRIX_SCHEMA_MISMATCH")
        if report.get("content_hash") != _content_hash(report):
            blockers.append("P0_SAFETY_MATRIX_CONTENT_HASH_MISMATCH")
        if report.get("status") != "GO" or report.get("marker") != P0_MARKER:
            blockers.append("P0_SAFETY_MATRIX_NOT_GO")
        statuses = report.get("scenario_statuses", {})
        if not isinstance(statuses, dict) or any(
            statuses.get(scenario_id) != "GO"
            for scenario_id in REQUIRED_SCENARIOS
        ):
            blockers.append("P0_REQUIRED_SCENARIO_NOT_GO")
        if report.get("broker_write_calls") != 0:
            blockers.append("P0_BROKER_WRITE_DETECTED")
        if report.get("execution_authority") != "NONE":
            blockers.append("P0_AUDIT_AUTHORITY_VIOLATION")
        recorded_hashes = report.get("source_hashes", {})
        if recorded_hashes != _current_source_hashes():
            blockers.append("P0_CRITICAL_SOURCE_HASH_CHANGED")
    status = "GO" if not blockers else "NO_GO"
    return {
        "schema": "ibkr_execution_p0_gate_status_v1",
        "status": status,
        "marker": report.get("marker"),
        "attestation_hash": report.get("content_hash"),
        "artifact_path": P0_ARTIFACT.as_posix(),
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
    }


def inspect_p0_safety_gate(project_root: Path) -> dict[str, Any]:
    """Backward-compatible name for the deterministic regression gate."""
    return inspect_p0_regression_gate(project_root)


def _run_scenarios(root: Path) -> dict[str, dict[str, Any]]:
    store = PaperExecutionStore(root / "primary.sqlite3")
    store.initialize()

    first_id = store.allocate_order_id(100, "P0-ORDER")
    regression = store.allocate_order_id(100, "P0-ORDER-REPLAY")
    next_id = store.allocate_order_id(101, "P0-ORDER-NEXT")
    next_valid_go = (
        first_id == ("ORDER_ID_READY", 100)
        and regression == ("ORDER_ID_REGRESSION_BLOCKED", None)
        and next_id == ("ORDER_ID_READY", 101)
    )

    callbacks = CallbackAuditState()
    callback_payload = {"orderId": 100, "status": "Submitted"}
    accepted = accept_callback(callbacks, "orderStatus", callback_payload)
    duplicate = accept_callback(callbacks, "orderStatus", callback_payload)
    out_of_order = accept_callback(
        callbacks,
        "orderStatus",
        {"orderId": 100, "status": "PreSubmitted"},
        out_of_order=True,
    )

    partial_one = record_fill_execution(
        store,
        _fill("P0-EXEC-1", quantity="0.4", submitted="1"),
    )
    partial_two = record_fill_execution(
        store,
        _fill("P0-EXEC-2", quantity="0.6", submitted="1"),
    )
    projection = project_position_from_store(store)
    duplicate_fill = record_fill_execution(
        store,
        _fill("P0-EXEC-1", quantity="0.4", submitted="1"),
    )
    partial_go = (
        partial_one.get("execution_status") == "EXECUTION_ACCEPTED"
        and partial_two.get("execution_status") == "EXECUTION_ACCEPTED"
        and projection.get("position", {}).get("long_quantity") == "1.0"
    )
    dedup_go = (
        duplicate_fill.get("execution_status") == "IDEMPOTENT_REPLAY"
        and len(store.list_executions()) == 2
    )

    fill_first_store = PaperExecutionStore(root / "fill-first.sqlite3")
    fill_first_store.initialize()
    fill_first = record_fill_execution(
        fill_first_store,
        _fill("P0-FILL-FIRST", quantity="1", submitted="1"),
    )
    pending_projection = project_position_from_store(fill_first_store)
    joined_after_fill = record_execution_commission(
        fill_first_store,
        execution_id="P0-FILL-FIRST",
        commission=Decimal("0.25"),
    )
    joined_projection = project_position_from_store(fill_first_store)
    fill_before_commission_go = (
        fill_first.get("execution_status") == "EXECUTION_ACCEPTED"
        and pending_projection.get("projection_status")
        == "RECONCILIATION_PENDING_COMMISSION"
        and joined_after_fill.get("commission_status") == "COMMISSION_JOINED"
        and joined_projection.get("projection_status") == "POSITION_PROJECTED"
    )

    commission_first_store = PaperExecutionStore(
        root / "commission-first.sqlite3"
    )
    commission_first_store.initialize()
    pending_commission = record_execution_commission(
        commission_first_store,
        execution_id="P0-COMMISSION-FIRST",
        commission=Decimal("0.15"),
    )
    execution_after_commission = record_fill_execution(
        commission_first_store,
        _fill("P0-COMMISSION-FIRST", quantity="1", submitted="1"),
    )
    commission_audit = commission_join_audit(commission_first_store)
    commission_before_execution_go = (
        pending_commission.get("commission_status") == "COMMISSION_PENDING"
        and execution_after_commission.get("execution_status")
        == "EXECUTION_ACCEPTED"
        and commission_audit.get("status") == "GO"
        and commission_audit.get("joined_count") == 1
        and commission_audit.get("pending_count") == 0
    )

    empty_store = PaperExecutionStore(root / "reconciliation.sqlite3")
    empty_store.initialize()
    manual_order = reconcile_paper_state(
        empty_store,
        broker_open_order_count=1,
        unknown_broker_open_order_count=1,
    )
    connection_loss = reconcile_paper_state(
        empty_store,
        execution_scope_complete=False,
    )
    external_position = reconcile_position_projection(
        local_quantity=Decimal("0"),
        broker_quantity=Decimal("1"),
        unknown_broker_position=True,
    )

    before_restart = project_position_from_store(fill_first_store)
    restarted_store = PaperExecutionStore(fill_first_store.path)
    restarted_store.initialize()
    after_restart = project_position_from_store(restarted_store)
    restart_go = stable_hash(before_restart) == stable_hash(after_restart)

    scenarios = {
        "NEXT_VALID_ID_MONOTONIC": _scenario(
            next_valid_go,
            first=first_id[0],
            regression=regression[0],
            next=next_id[0],
        ),
        "DUPLICATE_ORDER_STATUS_IDEMPOTENT": _scenario(
            accepted.get("classification") == "CALLBACK_OK"
            and duplicate.get("classification") == "DUPLICATE_CALLBACK_IGNORED"
            and callbacks.accepted == 1,
            accepted=accepted.get("classification"),
            duplicate=duplicate.get("classification"),
            accepted_count=callbacks.accepted,
        ),
        "OUT_OF_ORDER_STATUS_NO_REGRESSION": _scenario(
            out_of_order.get("classification")
            == "OUT_OF_ORDER_CALLBACK_BUFFERED"
            and callbacks.accepted == 1,
            classification=out_of_order.get("classification"),
            accepted_count=callbacks.accepted,
        ),
        "PARTIAL_FILL_ACCUMULATION": _scenario(
            partial_go,
            projected_quantity=projection.get("position", {}).get(
                "long_quantity"
            ),
            execution_count=len(store.list_executions()),
        ),
        "EXECUTION_DEDUP_BY_EXEC_ID": _scenario(
            dedup_go,
            replay_status=duplicate_fill.get("execution_status"),
            execution_count=len(store.list_executions()),
        ),
        "FILL_BEFORE_COMMISSION_RECONCILES": _scenario(
            fill_before_commission_go,
            before=pending_projection.get("projection_status"),
            commission=joined_after_fill.get("commission_status"),
            after=joined_projection.get("projection_status"),
        ),
        "COMMISSION_BEFORE_EXECUTION_JOIN": _scenario(
            commission_before_execution_go,
            initial=pending_commission.get("commission_status"),
            joined_count=commission_audit.get("joined_count"),
            pending_count=commission_audit.get("pending_count"),
        ),
        "MANUAL_TWS_ORDER_FAILS_RECONCILIATION": _scenario(
            manual_order.get("status") == "NO_GO"
            and manual_order.get("reconciliation_status")
            == "UNKNOWN_BROKER_ORDER",
            reconciliation_status=manual_order.get("reconciliation_status"),
            automatic_corrections=manual_order.get("automatic_corrections"),
        ),
        "RESTART_REPLAY_IDEMPOTENT": _scenario(
            restart_go,
            state_hash=after_restart.get("position", {}).get("state_hash"),
            execution_count=len(restarted_store.list_executions()),
        ),
        "EXTERNAL_POSITION_CHANGE_BLOCKS": _scenario(
            external_position.get("status") == "NO_GO"
            and external_position.get("position_reconciliation_status")
            == "UNKNOWN_BROKER_POSITION",
            reconciliation_status=external_position.get(
                "position_reconciliation_status"
            ),
            automatic_position_imports=external_position.get(
                "automatic_position_imports"
            ),
        ),
        "CONNECTION_LOSS_FAIL_CLOSED": _scenario(
            connection_loss.get("status") == "NO_GO"
            and connection_loss.get("reconciliation_status")
            == "EXECUTION_SCOPE_INCOMPLETE",
            reconciliation_status=connection_loss.get(
                "reconciliation_status"
            ),
            kill_switch=connection_loss.get("recommended_kill_switch"),
        ),
    }
    scenarios.update(run_extended_scenarios(root))
    return scenarios


def _fill(exec_id: str, *, quantity: str, submitted: str) -> FillExecution:
    return FillExecution(
        exec_id=exec_id,
        intent_id="P0-INTENT",
        account_fingerprint="P0-OFFLINE-ACCOUNT",
        perm_id="P0-PERM-ID",
        broker_order_id="P0-BROKER-ORDER-ID",
        con_id=265598,
        symbol="AAPL",
        currency="USD",
        side="BUY",
        quantity=Decimal(quantity),
        price=Decimal("100"),
        execution_time=f"2026-08-09T10:00:0{exec_id[-1]}+00:00",
        submitted_quantity=Decimal(submitted),
        fx_rate=Decimal("0.90"),
    )


def _scenario(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {
        "status": "GO" if passed else "NO_GO",
        "result": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


def _package_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _current_source_hashes() -> dict[str, str | None]:
    root = _package_project_root()
    return {path: sha256_file(root / path) for path in CRITICAL_SOURCE_PATHS}


def _content_hash(payload: dict[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the offline IBKR P0 gate")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = write_p0_safety_report(args.project_root.resolve())
    print(
        json.dumps(
            {
                "status": report["status"],
                "marker": report["marker"],
                "scenario_statuses": report["scenario_statuses"],
                "broker_write_calls": report["broker_write_calls"],
                "execution_authority": report["execution_authority"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
