from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Mapping

from stocks.ibkr.paper_execution.storage import PaperExecutionStore


def reconcile_paper_state(
    store: PaperExecutionStore,
    *,
    broker_open_order_count: int = 0,
    broker_execution_count: int | None = None,
    broker_commission_count: int | None = None,
    broker_position_count: int = 0,
    broker_cash_observation_count: int | None = None,
    non_atomic_snapshot: bool = False,
    execution_scope_complete: bool = True,
    execution_history_complete: bool = True,
    local_position_quantity: Decimal | None = None,
    broker_position_quantity: Decimal | None = None,
    matched_open_order_count: int | None = None,
    unknown_broker_open_order_count: int = 0,
    missing_local_open_order_count: int = 0,
) -> dict[str, object]:
    counts = store.counts()
    local_active_order_count = store.active_local_order_count()
    observed_exec_count = (
        counts["execution_count"]
        if broker_execution_count is None
        else broker_execution_count
    )
    observed_commission_count = (
        counts["commission_count"]
        if broker_commission_count is None
        else broker_commission_count
    )
    exec_count = (
        observed_exec_count
        if execution_history_complete
        else counts["execution_count"]
    )
    commission_count = (
        observed_commission_count
        if execution_history_complete
        else counts["commission_count"]
    )
    local_qty = Decimal("0") if local_position_quantity is None else local_position_quantity
    broker_qty = Decimal(str(broker_position_count)) if broker_position_quantity is None else broker_position_quantity
    if non_atomic_snapshot:
        status = "NON_ATOMIC_SNAPSHOT"
    elif not execution_scope_complete:
        status = "EXECUTION_SCOPE_INCOMPLETE"
    elif unknown_broker_open_order_count > 0:
        status = "UNKNOWN_BROKER_ORDER"
    elif missing_local_open_order_count > 0:
        status = "LOCAL_ORDER_MISSING_AT_BROKER"
    elif broker_open_order_count > 0 and local_active_order_count == 0:
        status = "UNKNOWN_BROKER_ORDER"
    elif local_active_order_count > 0 and broker_open_order_count == 0 and counts["execution_count"] == 0:
        status = "LOCAL_ORDER_MISSING_AT_BROKER"
    elif (
        execution_history_complete
        and exec_count > counts["execution_count"]
    ):
        status = "UNKNOWN_BROKER_EXECUTION"
    elif (
        execution_history_complete
        and exec_count < counts["execution_count"]
    ):
        status = "LOCAL_EXECUTION_MISSING_AT_BROKER"
    elif execution_history_complete and (
        commission_count > counts["commission_count"]
        or commission_count < counts["commission_count"]
    ):
        status = "COMMISSION_MISMATCH"
    elif local_qty != broker_qty and (local_position_quantity is not None or broker_position_quantity is not None):
        status = "POSITION_QUANTITY_MISMATCH"
    elif broker_position_count > counts["execution_count"] and local_position_quantity is None:
        status = "POSITION_MISMATCH"
    elif broker_cash_observation_count is not None and counts["execution_count"] > 0 and broker_cash_observation_count == 0:
        status = "CASH_OBSERVATION_MISMATCH"
    elif (
        matched_open_order_count is not None
        and matched_open_order_count > 0
        and matched_open_order_count == local_active_order_count
        and matched_open_order_count == broker_open_order_count
        and local_position_quantity is not None
        and broker_position_quantity is not None
        and local_qty > 0
        and local_qty == broker_qty
        and exec_count == counts["execution_count"]
        and commission_count == counts["commission_count"]
    ):
        status = "PAPER_RECONCILED_OPEN_ORDER_AND_POSITION"
    elif (
        matched_open_order_count is not None
        and matched_open_order_count > 0
        and matched_open_order_count == local_active_order_count
        and matched_open_order_count == broker_open_order_count
        and broker_position_count == 0
        and exec_count == counts["execution_count"]
        and commission_count == counts["commission_count"]
    ):
        status = "PAPER_RECONCILED_OPEN_ORDER"
    elif (
        local_position_quantity is not None
        and broker_position_quantity is not None
        and local_active_order_count == 0
        and broker_open_order_count == 0
        and local_qty == 0
        and broker_qty == 0
        and exec_count == counts["execution_count"]
        and commission_count == counts["commission_count"]
    ):
        status = "PAPER_RECONCILED_EMPTY"
    elif local_active_order_count == 0 and broker_open_order_count == 0 and exec_count == 0 and broker_position_count == 0:
        status = "PAPER_RECONCILED_EMPTY"
    elif local_active_order_count == 0 and broker_open_order_count == 0 and local_qty > 0 and local_qty == broker_qty:
        status = "PAPER_RECONCILED_OPEN_LONG"
    elif local_active_order_count == 0 and broker_open_order_count == 0 and exec_count == counts["execution_count"] and commission_count == counts["commission_count"]:
        status = "PAPER_RECONCILED"
    else:
        status = "PAPER_RECONCILIATION_BLOCKED"
    go_statuses = {
        "PAPER_RECONCILED",
        "PAPER_RECONCILED_EMPTY",
        "PAPER_RECONCILED_OPEN_LONG",
        "PAPER_RECONCILED_OPEN_ORDER",
        "PAPER_RECONCILED_OPEN_ORDER_AND_POSITION",
    }
    return {
        "status": "GO" if status in go_statuses else "NO_GO",
        "reconciliation_status": status,
        "automatic_corrections": 0,
        "recommended_kill_switch": "ARMED" if status in go_statuses else "TRIGGERED_RECONCILIATION",
        "manual_review_required": status not in go_statuses,
        "local_active_order_count": local_active_order_count,
        "broker_open_order_count": broker_open_order_count,
        "broker_position_count": broker_position_count,
        "broker_execution_count": observed_exec_count,
        "broker_commission_count": observed_commission_count,
        "execution_history_complete": execution_history_complete,
        "execution_count_comparison_applied": execution_history_complete,
        "local_position_quantity": str(local_qty),
        "broker_position_quantity": str(broker_qty),
        "matched_open_order_count": matched_open_order_count,
        "unknown_broker_open_order_count": unknown_broker_open_order_count,
        "missing_local_open_order_count": missing_local_open_order_count,
        **counts,
    }


def reconcile_position_projection(
    *,
    local_quantity: Decimal,
    broker_quantity: Decimal,
    snapshot_complete: bool = True,
    unknown_broker_position: bool = False,
    unknown_broker_execution: bool = False,
) -> dict[str, object]:
    if not snapshot_complete:
        status = "NON_ATOMIC_POSITION_SNAPSHOT"
    elif unknown_broker_position:
        status = "UNKNOWN_BROKER_POSITION"
    elif unknown_broker_execution:
        status = "POSITION_RECONCILIATION_BLOCKED"
    elif local_quantity > 0 and broker_quantity == 0:
        status = "LOCAL_POSITION_MISSING_AT_BROKER"
    elif local_quantity == 0 and broker_quantity > 0:
        status = "BROKER_POSITION_MISSING_LOCALLY"
    elif local_quantity != broker_quantity:
        status = "POSITION_QUANTITY_MISMATCH"
    else:
        status = "PAPER_POSITION_RECONCILED"
    return {
        "status": "GO" if status == "PAPER_POSITION_RECONCILED" else "NO_GO",
        "position_reconciliation_status": status,
        "local_long_quantity": str(local_quantity),
        "broker_position_quantity": str(broker_quantity),
        "automatic_position_imports": 0,
        "automatic_position_deletions": 0,
    }


def classify_broker_order_ownership(
    broker_order: Mapping[str, Any],
    *,
    local_orders: Iterable[Mapping[str, Any]],
) -> str:
    """Classify ownership without assigning external orders to a strategy."""
    perm_id = str(broker_order.get("perm_id") or "")
    broker_order_id = str(broker_order.get("broker_order_id") or "")
    for local in local_orders:
        if perm_id and perm_id == str(local.get("perm_id") or ""):
            return "OWNED_STRATEGY"
        if broker_order_id and broker_order_id == str(
            local.get("broker_order_id") or ""
        ):
            return "OWNED_STRATEGY"
    client_id = broker_order.get("client_id")
    if client_id == 0 or bool(broker_order.get("manual_order", False)):
        return "EXTERNAL_MANUAL"
    return "UNKNOWN_EXTERNAL"


def reconcile_open_orders(
    local_orders: Iterable[Mapping[str, Any]],
    broker_orders: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    local = list(local_orders)
    broker = list(broker_orders)
    local_by_perm = {
        str(row.get("perm_id")): row for row in local if row.get("perm_id")
    }
    broker_by_perm = {
        str(row.get("perm_id")): row for row in broker if row.get("perm_id")
    }
    missing_at_broker = sorted(set(local_by_perm) - set(broker_by_perm))
    missing_locally = sorted(set(broker_by_perm) - set(local_by_perm))
    remaining_mismatches: list[str] = []
    status_mismatches: list[str] = []
    for identity in sorted(set(local_by_perm) & set(broker_by_perm)):
        local_row = local_by_perm[identity]
        broker_row = broker_by_perm[identity]
        if Decimal(str(local_row.get("remaining_quantity", "0"))) != Decimal(
            str(broker_row.get("remaining_quantity", "0"))
        ):
            remaining_mismatches.append(identity)
        if str(local_row.get("order_status")) != str(
            broker_row.get("order_status")
        ):
            status_mismatches.append(identity)
    ownership = {
        str(row.get("perm_id") or row.get("broker_order_id") or index):
        classify_broker_order_ownership(row, local_orders=local)
        for index, row in enumerate(broker)
    }
    blockers = []
    if missing_at_broker:
        blockers.append("LOCAL_ORDER_MISSING_AT_BROKER")
    if missing_locally:
        blockers.append("BROKER_ORDER_MISSING_LOCALLY")
    if remaining_mismatches:
        blockers.append("OPEN_ORDER_REMAINING_QUANTITY_MISMATCH")
    if status_mismatches:
        blockers.append("OPEN_ORDER_STATUS_MISMATCH")
    if any(value != "OWNED_STRATEGY" for value in ownership.values()):
        blockers.append("EXTERNAL_ORDER_OWNERSHIP_REVIEW_REQUIRED")
    return {
        "status": "GO" if not blockers else "NO_GO",
        "blockers": blockers,
        "ownership": ownership,
        "missing_at_broker_count": len(missing_at_broker),
        "missing_locally_count": len(missing_locally),
        "remaining_quantity_mismatch_count": len(remaining_mismatches),
        "status_mismatch_count": len(status_mismatches),
        "automatic_strategy_adoptions": 0,
    }


def reconcile_position_details(
    *,
    local_quantity: Decimal,
    broker_quantity: Decimal,
    local_average_cost: Decimal | None,
    broker_average_cost: Decimal | None,
    broker_ownership: str = "OWNED_STRATEGY",
) -> dict[str, Any]:
    blockers: list[str] = []
    if local_quantity != broker_quantity:
        blockers.append("POSITION_QUANTITY_MISMATCH")
    if (
        local_average_cost is not None
        and broker_average_cost is not None
        and local_quantity != 0
        and local_average_cost != broker_average_cost
    ):
        blockers.append("POSITION_AVERAGE_COST_MISMATCH")
    if broker_quantity != 0 and broker_ownership != "OWNED_STRATEGY":
        blockers.append("EXTERNAL_POSITION_OWNERSHIP_REVIEW_REQUIRED")
    return {
        "status": "GO" if not blockers else "NO_GO",
        "blockers": blockers,
        "broker_ownership": broker_ownership,
        "automatic_position_imports": 0,
        "automatic_position_deletions": 0,
        "ordinary_strategy_sell_allowed": not blockers,
    }


def reconcile_cash_state(
    *,
    local_cash_eur: Decimal,
    broker_spendable_eur: Decimal,
    reserved_capital_eur: Decimal,
    unexplained_tolerance_eur: Decimal = Decimal("0.01"),
) -> dict[str, Any]:
    expected_available = local_cash_eur - reserved_capital_eur
    divergence = broker_spendable_eur - expected_available
    explained = abs(divergence) <= unexplained_tolerance_eur
    return {
        "status": "GO" if explained else "NO_GO",
        "cash_reconciliation_status": (
            "CASH_RECONCILED" if explained else "UNEXPLAINED_CASH_DIVERGENCE"
        ),
        "expected_available_eur": str(expected_available),
        "broker_spendable_eur": str(broker_spendable_eur),
        "reserved_capital_eur": str(reserved_capital_eur),
        "divergence_eur": str(divergence),
        "broker_truth_wins_after_review": True,
        "automatic_cash_overwrite": False,
    }
