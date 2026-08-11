from __future__ import annotations

from decimal import Decimal
from typing import Any

from stocks.execution.models import KillSwitchState, KillSwitchStatus, ReconciliationResult


def reconcile(local: dict[str, Any], broker: dict[str, Any]) -> tuple[ReconciliationResult, KillSwitchState]:
    mismatches: list[str] = []
    local_orders = set(local.get("open_orders", []))
    broker_orders = set(broker.get("open_orders", []))
    local_fills = set(local.get("fills", []))
    broker_fills = set(broker.get("fills", []))
    if local_orders - broker_orders:
        mismatches.append("LOCAL_ORDER_MISSING_AT_BROKER")
    if broker_orders - local_orders:
        mismatches.append("UNKNOWN_BROKER_ORDER")
    if local_fills - broker_fills:
        mismatches.append("LOCAL_FILL_MISSING_AT_BROKER")
    if broker_fills - local_fills:
        mismatches.append("UNKNOWN_BROKER_FILL")
    if len(local_fills) != len(local.get("fills", [])):
        mismatches.append("DUPLICATE_FILL")
    if len(local_orders) != len(local.get("open_orders", [])):
        mismatches.append("DUPLICATE_ORDER")
    if local.get("positions", {}) != broker.get("positions", {}):
        mismatches.append("POSITION_MISMATCH")
    if Decimal(str(local.get("cash", "0"))) != Decimal(str(broker.get("cash", "0"))):
        mismatches.append("CASH_MISMATCH")
    status = "RECONCILED" if not mismatches else "RECONCILIATION_BLOCKED"
    kill = KillSwitchState(
        status=KillSwitchStatus.ARMED if not mismatches else KillSwitchStatus.TRIGGERED_RECONCILIATION,
        reason=None if not mismatches else ",".join(mismatches),
        updated_at="2026-07-21T12:00:00Z",
    )
    return ReconciliationResult(status=status, mismatches=tuple(mismatches), kill_switch_triggered=bool(mismatches)), kill

