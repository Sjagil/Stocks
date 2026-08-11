from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from stocks.ibkr.reconciliation.models import BrokerObservationSnapshot


def compare_phase7_to_broker(project_root: Path, broker: BrokerObservationSnapshot) -> dict[str, Any]:
    phase7_db = project_root / "data" / "execution" / "phase7" / "execution_ledger.sqlite3"
    before_hash = _file_hash(phase7_db)
    local_orders, local_fills = _phase7_ids(phase7_db)
    after_hash = _file_hash(phase7_db)
    broker_order_count = len(broker.same_client_open_orders.open_orders) + len(broker.all_api_open_orders.open_orders)
    broker_execution_count = len(broker.executions.executions)
    broker_position_count = len(broker.positions.positions)
    broker_account_value_count = len(broker.account.values)
    statuses: list[str] = []
    if broker_order_count or broker_execution_count or broker_position_count or broker_account_value_count:
        statuses.append("OBSERVED_EXTERNAL_BROKER_STATE")
    if any(str(item).startswith("SIM-") for item in local_orders):
        statuses.append("SIM_ORDER_IDS_IGNORED")
    if broker_position_count:
        statuses.append("POSITION_MISMATCH")
    if broker_account_value_count:
        statuses.append("CASH_OBSERVATION_MISMATCH")
    if broker_order_count:
        statuses.append("UNKNOWN_BROKER_ORDER")
    if broker_execution_count:
        statuses.append("UNKNOWN_BROKER_EXECUTION")
    if broker.executions.execution_history_complete is False:
        statuses.append("EXECUTION_SCOPE_INCOMPLETE")
    if broker.content_hash and broker.snapshot_atomic is False:
        statuses.append("NON_ATOMIC_BROKER_SNAPSHOT")
    status = "RECONCILED_EMPTY_STATE" if not statuses else "RECONCILIATION_BLOCKED"
    if statuses == ["EXECUTION_SCOPE_INCOMPLETE", "NON_ATOMIC_BROKER_SNAPSHOT"]:
        status = "RECONCILED_EMPTY_STATE"
    return {
        "status": status,
        "classifications": sorted(set(statuses)),
        "phase7_ledger_path": str(phase7_db),
        "phase7_ledger_hash_before": before_hash,
        "phase7_ledger_hash_after": after_hash,
        "phase7_ledger_unchanged": before_hash == after_hash,
        "local_order_count": len(local_orders),
        "local_fill_count": len(local_fills),
        "broker_open_order_count": broker_order_count,
        "broker_execution_count": broker_execution_count,
        "broker_position_count": broker_position_count,
        "broker_account_fingerprint_count": len({item.account_fingerprint for item in broker.account.values}),
        "automatic_corrections": 0,
        "recommended_manual_review": bool(statuses),
        "recommended_kill_switch_state": "TRIGGERED_RECONCILIATION" if statuses else "ARMED",
        "reconciliation_gate": "BLOCKED" if statuses else "GO",
    }


def _phase7_ids(path: Path) -> tuple[list[str], list[str]]:
    if not path.exists():
        return [], []
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        orders = [row["payload_json"] for row in conn.execute("SELECT payload_json FROM intents").fetchall()]
        fills = [row["fill_id"] for row in conn.execute("SELECT fill_id FROM fills").fetchall()]
    local_orders = []
    for raw in orders:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        local_orders.append(str(payload.get("broker_order_id") or payload.get("intent_id") or ""))
    return local_orders, [str(item) for item in fills]


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()
