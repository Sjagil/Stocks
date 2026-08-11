from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.p0_safety import inspect_p0_regression_gate
from stocks.ibkr.reconciliation.account_state import (
    derive_economic_account_state,
)


P0_READINESS_ARTIFACT = Path(
    "output/ibkr/live/p0-execution-readiness.json"
)
P0_READY_MARKER = "P0_EXECUTION_INFRASTRUCTURE_READY"
P0_READINESS_BLOCKER = "P0_EXECUTION_INFRASTRUCTURE_NOT_READY"
REQUIRED_SUB_GATES = (
    "IBKR_CONNECTION_READY",
    "ACCOUNT_STATE_READY",
    "ACCOUNT_STATE_FRESH",
    "ORDER_ID_STATE_READY",
    "ORDER_RECONCILIATION_READY",
    "POSITION_RECONCILIATION_READY",
    "CASH_RECONCILIATION_READY",
    "REGRESSION_MATRIX_PASS",
    "NO_UNEXPLAINED_CRITICAL_DIVERGENCE",
)


def build_p0_execution_readiness(project_root: Path) -> dict[str, Any]:
    regression = inspect_p0_regression_gate(project_root)
    reconciliation = _read_json(
        project_root / "output/ibkr/live/reconciliation.json"
    )
    private = _latest_private_snapshot(
        project_root
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    snapshot = private.get("payload", {}) if private else {}
    snapshot_hash_verified = bool(
        private
        and private.get("snapshot_hash")
        == reconciliation.get("private_snapshot_hash")
        and private.get("snapshot_hash") == stable_hash(snapshot)
    )
    account = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=snapshot_hash_verified,
    )
    reconciliation_hash_valid = _content_hash_valid(reconciliation)
    reconciliation_go = bool(
        reconciliation_hash_valid
        and reconciliation.get("status") == "GO"
        and str(reconciliation.get("reconciliation_status", "")).startswith(
            "LIVE_RECONCILED"
        )
    )
    unknown_orders_clear = reconciliation.get("unknown_orders") == 0
    unknown_positions_clear = reconciliation.get("unknown_positions") == 0
    sub_gates = {
        "IBKR_CONNECTION_READY": bool(
            reconciliation_go and snapshot.get("server_version")
        ),
        "ACCOUNT_STATE_READY": (
            account.get("execution_status") == "EXECUTION_ACCOUNT_READY"
        ),
        "ACCOUNT_STATE_FRESH": bool(
            account.get("account_state_fresh")
            and account.get("position_state_fresh")
            and account.get("order_state_fresh")
        ),
        "ORDER_ID_STATE_READY": bool(
            regression.get("status") == "GO"
            and reconciliation_go
            and snapshot.get("server_version")
        ),
        "ORDER_RECONCILIATION_READY": bool(
            reconciliation_go
            and unknown_orders_clear
            and account.get("order_reconciliation_completed")
        ),
        "POSITION_RECONCILIATION_READY": bool(
            reconciliation_go
            and unknown_positions_clear
            and account.get("position_reconciliation_completed")
        ),
        "CASH_RECONCILIATION_READY": bool(
            reconciliation_go
            and account.get("execution_sizing_capacity_eur") is not None
            and account.get("open_order_reserved_capital_complete") is True
        ),
        "REGRESSION_MATRIX_PASS": regression.get("status") == "GO",
        "NO_UNEXPLAINED_CRITICAL_DIVERGENCE": bool(
            reconciliation_go
            and unknown_orders_clear
            and unknown_positions_clear
            and not reconciliation.get("blockers")
        ),
    }
    ready = all(sub_gates.values())
    body: dict[str, Any] = {
        "schema": "ibkr_p0_execution_readiness_v1",
        "status": "GO" if ready else "NO_GO",
        "marker": P0_READY_MARKER if ready else "NO_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "sub_gates": sub_gates,
        "required_sub_gates": list(REQUIRED_SUB_GATES),
        "open_blockers": [
            name for name, passed in sub_gates.items() if not passed
        ],
        "account_lifecycle_state": account.get("lifecycle_state"),
        "account_execution_blockers": account.get("execution_blockers", []),
        "account_semantics": {
            "base_currency": account.get("account_base_currency"),
            "spendable_eur_proven": account.get("spendable_eur") is not None,
            "reporting_value_eur_proven": account.get("reporting_value_eur")
            is not None,
            "buying_power_is_cash": False,
            "implicit_fx_conversion_assumed": False,
            "non_eur_cash_excluded": True,
        },
        "regression_attestation_hash": regression.get("attestation_hash"),
        "reconciliation_hash": reconciliation.get("content_hash"),
        "private_snapshot_hash": private.get("snapshot_hash") if private else None,
        "input_binding_hash": _input_binding_hash(
            regression, reconciliation, private
        ),
        "orders_submitted": 0,
        "orders_cancelled": 0,
        "orders_modified": 0,
        "fx_trades": 0,
        "live_permissions_changed": False,
        "execution_authority": "NONE",
    }
    body["content_hash"] = _content_hash(body)
    return body


def write_p0_execution_readiness(project_root: Path) -> dict[str, Any]:
    report = build_p0_execution_readiness(project_root)
    path = project_root / P0_READINESS_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return report


def inspect_p0_execution_readiness_gate(
    project_root: Path,
) -> dict[str, Any]:
    path = project_root / P0_READINESS_ARTIFACT
    blockers: list[str] = []
    report = _read_json(path)
    if not report:
        blockers.append("P0_EXECUTION_READINESS_MISSING_OR_INVALID")
    else:
        if report.get("content_hash") != _content_hash(report):
            blockers.append("P0_EXECUTION_READINESS_HASH_MISMATCH")
        if report.get("status") != "GO" or report.get("marker") != P0_READY_MARKER:
            blockers.append(P0_READINESS_BLOCKER)
        gates = report.get("sub_gates", {})
        if any(gates.get(name) is not True for name in REQUIRED_SUB_GATES):
            blockers.append("P0_EXECUTION_SUB_GATE_NOT_READY")
        regression = inspect_p0_regression_gate(project_root)
        reconciliation = _read_json(
            project_root / "output/ibkr/live/reconciliation.json"
        )
        private = _latest_private_snapshot(
            project_root
            / "data/execution/live/private/broker_observation.sqlite3"
        )
        if report.get("input_binding_hash") != _input_binding_hash(
            regression, reconciliation, private
        ):
            blockers.append("P0_EXECUTION_INPUT_BINDING_CHANGED")
    return {
        "schema": "ibkr_p0_execution_readiness_gate_v1",
        "status": "GO" if not blockers else "NO_GO",
        "marker": report.get("marker"),
        "attestation_hash": report.get("content_hash"),
        "artifact_path": P0_READINESS_ARTIFACT.as_posix(),
        "sub_gates": report.get("sub_gates", {}),
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
    }


def _latest_private_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with sqlite3.connect(path) as connection:
            row = connection.execute(
                "SELECT snapshot_hash, payload_json, created_at "
                "FROM snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
    except (sqlite3.Error, OSError):
        return None
    if row is None:
        return None
    try:
        payload = json.loads(str(row[1]))
    except json.JSONDecodeError:
        return None
    return {
        "snapshot_hash": str(row[0]),
        "payload": payload if isinstance(payload, dict) else {},
        "created_at": str(row[2]),
    }


def _input_binding_hash(
    regression: dict[str, Any],
    reconciliation: dict[str, Any],
    private: dict[str, Any] | None,
) -> str:
    return stable_hash(
        {
            "regression": regression.get("attestation_hash"),
            "reconciliation": reconciliation.get("content_hash"),
            "private_snapshot": private.get("snapshot_hash") if private else None,
        }
    )


def _content_hash(payload: dict[str, Any]) -> str:
    return stable_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    return bool(payload and payload.get("content_hash") == _content_hash(payload))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = [
    "P0_READINESS_BLOCKER",
    "P0_READY_MARKER",
    "build_p0_execution_readiness",
    "inspect_p0_execution_readiness_gate",
    "write_p0_execution_readiness",
]
