from __future__ import annotations

import json
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.commissions import record_execution_commission
from stocks.ibkr.paper_execution.executions import FillExecution, record_fill_execution
from stocks.ibkr.paper_execution.models import ManualPaperIntent
from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def load_latest_phase8_private_snapshot(project_root: Path) -> dict[str, Any] | None:
    db_path = (
        project_root
        / "data"
        / "broker"
        / "phase8"
        / "private"
        / "broker_observation.sqlite3"
    )
    if not db_path.exists():
        return None
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT payload_json FROM snapshots ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def record_known_fill_from_snapshot(
    store: PaperExecutionStore,
    *,
    intent: ManualPaperIntent,
    local_order_id: int,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    expected_order_hash = stable_hash(
        {"broker_order_id": local_order_id}
    )[:24]
    broker_executions = snapshot.get("executions", {}).get(
        "executions", []
    )
    order_executions = [
        item
        for item in broker_executions
        if str(item.get("broker_order_id", "")) == expected_order_hash
    ]
    if not order_executions:
        return {
            "status": "GO",
            "fill_observation_status": "KNOWN_FILL_NOT_OBSERVED",
            "matched_execution_count": 0,
            "recorded_execution_count": 0,
            "idempotent_execution_count": 0,
            "joined_commission_count": 0,
            "pending_commission_count": 0,
            "unknown_execution_imports": 0,
            "broker_write_calls": 0,
        }

    invalid = [
        item
        for item in order_executions
        if int(item.get("con_id", -1)) != intent.con_id
        or _normalize_side(str(item.get("side", ""))) != intent.side
        or Decimal(str(item.get("quantity", "0"))) <= 0
    ]
    if invalid:
        return _blocked("KNOWN_ORDER_EXECUTION_IDENTITY_MISMATCH")

    total_quantity = sum(
        (
            Decimal(str(item.get("quantity", "0")))
            for item in order_executions
        ),
        Decimal("0"),
    )
    if total_quantity > intent.quantity:
        return _blocked("KNOWN_ORDER_EXECUTION_QUANTITY_EXCEEDED")

    commissions = {
        str(item.get("execution_id", "")): item
        for item in snapshot.get("executions", {}).get("commissions", [])
    }
    recorded = 0
    idempotent = 0
    joined_commissions = 0
    pending_commissions = 0
    for item in sorted(
        order_executions,
        key=lambda row: str(row.get("execution_id", "")),
    ):
        execution_id = str(item.get("execution_id", ""))
        if not execution_id:
            return _blocked("KNOWN_ORDER_EXECUTION_ID_MISSING")
        fill = FillExecution(
            exec_id=execution_id,
            intent_id=intent.intent_id,
            account_fingerprint=str(
                item.get("account_fingerprint", "")
            ),
            perm_id=str(item.get("perm_id", "")),
            broker_order_id=str(item.get("broker_order_id", "")),
            con_id=int(item["con_id"]),
            symbol=str(item.get("symbol", intent.symbol)),
            currency=intent.currency,
            side=_normalize_side(str(item.get("side", ""))),
            quantity=Decimal(str(item["quantity"])),
            price=Decimal(str(item.get("price", "0"))),
            execution_time=str(item.get("execution_time", "")),
            submitted_quantity=intent.quantity,
            fx_rate=intent.fx_rate,
        )
        fill_result = record_fill_execution(store, fill)
        fill_status = str(fill_result["execution_status"])
        if fill_result["status"] != "GO":
            return _blocked(fill_status)
        if fill_status == "EXECUTION_ACCEPTED":
            recorded += 1
        elif fill_status == "IDEMPOTENT_REPLAY":
            idempotent += 1

        commission = commissions.get(execution_id)
        if commission is None:
            pending_commissions += 1
            continue
        commission_result = record_execution_commission(
            store,
            execution_id=execution_id,
            commission=Decimal(str(commission.get("commission", "0"))),
            currency=str(commission.get("currency", intent.currency)),
            allow_pending=False,
        )
        if commission_result["status"] != "GO":
            return _blocked(str(commission_result["commission_status"]))
        if commission_result["commission_status"] in {
            "COMMISSION_JOINED",
            "COMMISSION_DUPLICATE_IGNORED",
        }:
            joined_commissions += 1

    status = (
        "KNOWN_FILL_RECORDED"
        if recorded
        else "KNOWN_FILL_IDEMPOTENT"
    )
    if pending_commissions:
        status += "_COMMISSION_PENDING"
    return {
        "status": "GO",
        "fill_observation_status": status,
        "matched_execution_count": len(order_executions),
        "recorded_execution_count": recorded,
        "idempotent_execution_count": idempotent,
        "joined_commission_count": joined_commissions,
        "pending_commission_count": pending_commissions,
        "unknown_execution_imports": 0,
        "broker_write_calls": 0,
    }


def _normalize_side(side: str) -> str:
    return {
        "BOT": "BUY",
        "BUY": "BUY",
        "SLD": "SELL",
        "SELL": "SELL",
    }.get(side.upper(), side.upper())


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "status": "NO_GO",
        "fill_observation_status": reason,
        "matched_execution_count": 0,
        "recorded_execution_count": 0,
        "idempotent_execution_count": 0,
        "joined_commission_count": 0,
        "pending_commission_count": 0,
        "unknown_execution_imports": 0,
        "broker_write_calls": 0,
    }
