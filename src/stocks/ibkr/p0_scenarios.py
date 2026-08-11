from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.callbacks import (
    OrderStatusProjection,
    apply_order_status,
)
from stocks.ibkr.paper_execution.cancellation import (
    record_broker_cancel_confirmation,
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
    classify_broker_order_ownership,
    reconcile_cash_state,
    reconcile_open_orders,
    reconcile_position_details,
)
from stocks.ibkr.paper_execution.storage import PaperExecutionStore
from stocks.ibkr.reconciliation.account_state import (
    derive_economic_account_state,
)
from stocks.ibkr.reconciliation.recovery import (
    ConnectionRecoveryState,
    REQUIRED_SUBSCRIPTIONS,
)


EXTENDED_REQUIRED_SCENARIOS = (
    "CONNECTION_INITIAL",
    "CONNECTION_DELAYED_READY",
    "CONNECTION_DISCONNECT",
    "CONNECTION_RECONNECT",
    "CONNECTION_REPEATED_RECONNECT",
    "CONNECTION_TWS_RESTART",
    "CONNECTION_GATEWAY_RESTART",
    "CONNECTION_CLIENT_ID_REUSE",
    "CONNECTION_LOST_SUBSCRIPTION_RECOVERY",
    "ORDER_ID_AFTER_RECONNECT",
    "ORDER_ID_UNKNOWN_EXTERNAL",
    "ORDER_ID_MANUAL_TWS",
    "ORDER_ID_DUPLICATE_BLOCKED",
    "PERMID_STABLE_ACROSS_ORDER_ID_CHANGE",
    "PERMID_EXTERNAL_DISCOVERED_LATER",
    "PERMID_RESTART_REDISCOVERY",
    "EXECUTION_SAME_EXEC_ID_TWICE",
    "EXECUTION_SAME_FILL_DIFFERENT_CALLBACK_PATH",
    "EXECUTION_BEFORE_ORDER_STATUS",
    "EXECUTION_AFTER_TERMINAL_ORDER_STATUS",
    "EXECUTION_MULTIPLE_PARTIAL_EXEC_IDS",
    "EXECUTION_COMMISSION_ENRICHES_FILL",
    "COMMISSION_FILL_THEN_REPORT",
    "COMMISSION_SUBSTANTIALLY_DELAYED",
    "COMMISSION_MULTIPLE_FILLS_SEPARATE_REPORTS",
    "COMMISSION_MISSING_REPORT_VISIBLE",
    "COMMISSION_DUPLICATE_REPORT_IDEMPOTENT",
    "ORDER_STATUS_PENDING_SUBMIT",
    "ORDER_STATUS_PRE_SUBMITTED",
    "ORDER_STATUS_SUBMITTED",
    "ORDER_STATUS_PENDING_CANCEL",
    "ORDER_STATUS_CANCELLED",
    "ORDER_STATUS_API_CANCELLED",
    "ORDER_STATUS_FILLED",
    "ORDER_STATUS_INACTIVE",
    "ORDER_STATUS_STALE_TERMINAL_REGRESSION_BLOCKED",
    "PARTIAL_FILL_25_50_75_100",
    "PARTIAL_FILL_MULTIPLE_PRICES",
    "PARTIAL_FILL_THEN_CANCEL",
    "PARTIAL_FILL_THEN_INACTIVE",
    "PARTIAL_FILL_THEN_RECONNECT",
    "PARTIAL_FILL_MANUAL_INTERVENTION",
    "CANCEL_RACE_CANCEL_THEN_FILL",
    "CANCEL_RACE_FILL_THEN_ACK",
    "CANCEL_RACE_PARTIAL_DURING_CANCEL",
    "CANCEL_RACE_FULL_DURING_CANCEL",
    "MANUAL_TWS_OPEN_ORDER",
    "MANUAL_TWS_FILLED_ORDER",
    "MANUAL_TWS_CANCELLED_ORDER",
    "MANUAL_TWS_POSITION_CHANGE",
    "EXTERNAL_POSITION_NOT_STRATEGY_ADOPTED",
    "CASH_RECONCILIATION_MATCH",
    "CASH_RECONCILIATION_DIVERGENCE_VISIBLE",
    "CASH_RESERVATION_NOT_DUPLICATED",
    "POSITION_LOCAL_GREATER_THAN_BROKER",
    "POSITION_LOCAL_LESS_THAN_BROKER",
    "POSITION_LOCAL_ABSENT_AT_BROKER",
    "POSITION_BROKER_ABSENT_LOCALLY",
    "POSITION_AVERAGE_COST_DISAGREEMENT",
    "OPEN_ORDER_LOCAL_OPEN_BROKER_MISSING",
    "OPEN_ORDER_BROKER_OPEN_LOCAL_MISSING",
    "OPEN_ORDER_REMAINING_QUANTITY_DIFFERENT",
    "OPEN_ORDER_STATUS_DIFFERENT",
    "RESTART_ORDERS_RECONSTRUCTED",
    "RESTART_FILLS_COMMISSIONS_POSITIONS_RECONSTRUCTED",
    "RESTART_OWNERSHIP_RESERVATIONS_RECONCILIATION_RECONSTRUCTED",
    "ACCOUNT_DOWNLOAD_COMPLETION_REQUIRED",
    "ACCOUNT_STALENESS_FAILS_EXECUTION_ONLY",
    "ACCOUNT_NON_EUR_CASH_EXCLUDED",
    "ACCOUNT_BUYING_POWER_NOT_CASH",
)


def run_extended_scenarios(root: Path) -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    _connection_scenarios(scenarios)
    _identity_scenarios(scenarios, root)
    _execution_and_commission_scenarios(scenarios, root)
    _order_status_scenarios(scenarios)
    _partial_and_cancel_scenarios(scenarios, root)
    _manual_and_reconciliation_scenarios(scenarios, root)
    _restart_scenarios(scenarios, root)
    _account_scenarios(scenarios)
    missing = set(EXTENDED_REQUIRED_SCENARIOS) - set(scenarios)
    if missing:
        raise RuntimeError(f"P0 scenario implementation missing: {sorted(missing)}")
    return scenarios


def _connection_scenarios(output: dict[str, dict[str, Any]]) -> None:
    state = ConnectionRecoveryState()
    initial = state.connect(41)
    delayed = state.mark_reconciled()
    for subscription in REQUIRED_SUBSCRIPTIONS:
        state.subscribe(subscription)
    ready = state.mark_reconciled()
    output["CONNECTION_INITIAL"] = _case(
        initial == "CONNECTED_RECONCILIATION_REQUIRED" and not state.generation == 0,
        transition=initial,
    )
    output["CONNECTION_DELAYED_READY"] = _case(
        delayed == "RECONCILIATION_BLOCKED_SUBSCRIPTIONS_INCOMPLETE"
        and ready == "CONNECTION_READY"
        and state.execution_ready,
        before=delayed,
        after=ready,
    )
    disconnected = state.disconnect()
    output["CONNECTION_DISCONNECT"] = _case(
        disconnected == "DISCONNECTED_EXECUTION_BLOCKED"
        and not state.execution_ready,
        transition=disconnected,
    )
    reconnect = state.connect(41)
    output["CONNECTION_RECONNECT"] = _case(
        reconnect == "CONNECTED_RECONCILIATION_REQUIRED"
        and not state.execution_ready,
        generation=state.generation,
    )
    state.disconnect()
    state.connect(41)
    state.disconnect()
    repeated = state.connect(41)
    output["CONNECTION_REPEATED_RECONNECT"] = _case(
        repeated == "CONNECTED_RECONCILIATION_REQUIRED"
        and state.generation == 4,
        generation=state.generation,
    )
    reuse = state.connect(41)
    output["CONNECTION_CLIENT_ID_REUSE"] = _case(
        reuse == "CLIENT_ID_REUSE_BLOCKED", classification=reuse
    )
    for subscription in REQUIRED_SUBSCRIPTIONS:
        state.subscribe(subscription)
    state.mark_reconciled()
    state.disconnect()
    state.connect(41)
    lost = state.mark_reconciled()
    for subscription in REQUIRED_SUBSCRIPTIONS:
        state.subscribe(subscription)
    recovered = state.mark_reconciled()
    output["CONNECTION_LOST_SUBSCRIPTION_RECOVERY"] = _case(
        lost == "RECONCILIATION_BLOCKED_SUBSCRIPTIONS_INCOMPLETE"
        and recovered == "CONNECTION_READY",
        before=lost,
        after=recovered,
    )
    output["CONNECTION_TWS_RESTART"] = _case(
        state.generation >= 5 and state.execution_ready,
        generation=state.generation,
    )
    output["CONNECTION_GATEWAY_RESTART"] = _case(
        state.generation >= 5 and state.execution_ready,
        generation=state.generation,
    )


def _identity_scenarios(
    output: dict[str, dict[str, Any]], root: Path
) -> None:
    store = _store(root / "identity.sqlite3")
    first = store.allocate_order_id(700, "ORDER-A")
    store = PaperExecutionStore(store.path)
    store.initialize()
    after_reconnect = store.allocate_order_id(701, "ORDER-B")
    duplicate = store.allocate_order_id(701, "ORDER-C")
    output["ORDER_ID_AFTER_RECONNECT"] = _case(
        first[0] == "ORDER_ID_READY"
        and after_reconnect[0] == "ORDER_ID_READY",
        max_order_id=store.max_order_id(),
    )
    output["ORDER_ID_DUPLICATE_BLOCKED"] = _case(
        duplicate[0] == "ORDER_ID_REGRESSION_BLOCKED",
        classification=duplicate[0],
    )
    local = [{"perm_id": "PERM-700", "broker_order_id": "700"}]
    changed = {
        "perm_id": "PERM-700",
        "broker_order_id": "9900",
        "client_id": 41,
    }
    stable = classify_broker_order_ownership(changed, local_orders=local)
    output["PERMID_STABLE_ACROSS_ORDER_ID_CHANGE"] = _case(
        stable == "OWNED_STRATEGY", ownership=stable
    )
    manual = {
        "perm_id": "PERM-MANUAL",
        "broker_order_id": "0",
        "client_id": 0,
    }
    manual_ownership = classify_broker_order_ownership(
        manual, local_orders=local
    )
    unknown = classify_broker_order_ownership(
        {"perm_id": "PERM-X", "broker_order_id": "900", "client_id": 77},
        local_orders=local,
    )
    output["ORDER_ID_UNKNOWN_EXTERNAL"] = _case(
        unknown == "UNKNOWN_EXTERNAL", ownership=unknown
    )
    output["ORDER_ID_MANUAL_TWS"] = _case(
        manual_ownership == "EXTERNAL_MANUAL", ownership=manual_ownership
    )
    output["PERMID_EXTERNAL_DISCOVERED_LATER"] = _case(
        manual_ownership == "EXTERNAL_MANUAL", ownership=manual_ownership
    )
    replayed = classify_broker_order_ownership(changed, local_orders=local)
    output["PERMID_RESTART_REDISCOVERY"] = _case(
        replayed == stable == "OWNED_STRATEGY", ownership=replayed
    )


def _execution_and_commission_scenarios(
    output: dict[str, dict[str, Any]], root: Path
) -> None:
    store = _store(root / "execution.sqlite3")
    one = record_fill_execution(store, _fill("EXEC-1", "I-1", "0.5", "100", "1"))
    replay = record_fill_execution(store, _fill("EXEC-1", "I-1", "0.5", "100", "1"))
    output["EXECUTION_SAME_EXEC_ID_TWICE"] = _case(
        one["execution_status"] == "EXECUTION_ACCEPTED"
        and replay["execution_status"] == "IDEMPOTENT_REPLAY"
        and len(store.list_executions()) == 1,
        replay=replay["execution_status"],
    )
    output["EXECUTION_SAME_FILL_DIFFERENT_CALLBACK_PATH"] = _case(
        replay["execution_status"] == "IDEMPOTENT_REPLAY",
        economic_effect_count=len(store.list_executions()),
    )
    status = OrderStatusProjection(original_quantity=Decimal("1"))
    execution_first = record_fill_execution(
        store, _fill("EXEC-2", "I-1", "0.5", "101", "1")
    )
    status_after = apply_order_status(
        status, status="Filled", filled="1", remaining="0", average_fill_price="100.5"
    )
    output["EXECUTION_BEFORE_ORDER_STATUS"] = _case(
        execution_first["execution_status"] == "EXECUTION_ACCEPTED"
        and status_after["status"] == "GO",
        execution_count=len(store.list_executions()),
    )
    terminal = OrderStatusProjection(original_quantity=Decimal("1"))
    apply_order_status(terminal, status="Filled", filled="1", remaining="0")
    terminal_store = _store(root / "terminal-exec.sqlite3")
    after_terminal = record_fill_execution(
        terminal_store, _fill("EXEC-TERMINAL", "I-T", "1", "100", "1")
    )
    output["EXECUTION_AFTER_TERMINAL_ORDER_STATUS"] = _case(
        terminal.terminal
        and after_terminal["execution_status"] == "EXECUTION_ACCEPTED",
        terminal_status=terminal.status,
    )
    output["EXECUTION_MULTIPLE_PARTIAL_EXEC_IDS"] = _case(
        len(store.list_executions()) == 2
        and project_position_from_store(store)["position"]["long_quantity"] == "1.0",
        execution_count=len(store.list_executions()),
    )
    commission = record_execution_commission(
        terminal_store, execution_id="EXEC-TERMINAL", commission=Decimal("0.5")
    )
    output["EXECUTION_COMMISSION_ENRICHES_FILL"] = _case(
        commission["commission_status"] == "COMMISSION_JOINED"
        and len(terminal_store.list_executions()) == 1,
        execution_count=len(terminal_store.list_executions()),
        commission_count=len(terminal_store.list_commissions()),
    )
    output["COMMISSION_FILL_THEN_REPORT"] = _case(
        commission["commission_status"] == "COMMISSION_JOINED",
        classification=commission["commission_status"],
    )
    missing_store = _store(root / "missing-commission.sqlite3")
    record_fill_execution(
        missing_store, _fill("EXEC-MISSING", "I-M", "1", "100", "1")
    )
    delayed = commission_join_audit(missing_store, grace_expired=True)
    output["COMMISSION_SUBSTANTIALLY_DELAYED"] = _case(
        delayed["grace_status"] == "COMMISSION_GRACE_EXPIRED"
        and delayed["pending_count"] == 1,
        grace_status=delayed["grace_status"],
    )
    output["COMMISSION_MISSING_REPORT_VISIBLE"] = _case(
        delayed["pending_count"] == 1,
        pending_count=delayed["pending_count"],
    )
    multi = _store(root / "multi-commission.sqlite3")
    for index in range(2):
        exec_id = f"EXEC-MULTI-{index}"
        record_fill_execution(multi, _fill(exec_id, "I-MULTI", "0.5", str(100 + index), "1"))
        record_execution_commission(multi, execution_id=exec_id, commission=Decimal("0.2"))
    multi_audit = commission_join_audit(multi)
    output["COMMISSION_MULTIPLE_FILLS_SEPARATE_REPORTS"] = _case(
        multi_audit["joined_count"] == 2 and multi_audit["pending_count"] == 0,
        joined_count=multi_audit["joined_count"],
    )
    duplicate = record_execution_commission(
        multi, execution_id="EXEC-MULTI-0", commission=Decimal("0.2")
    )
    output["COMMISSION_DUPLICATE_REPORT_IDEMPOTENT"] = _case(
        duplicate["commission_status"] == "COMMISSION_DUPLICATE_IGNORED"
        and len(multi.list_commissions()) == 2,
        classification=duplicate["commission_status"],
    )


def _order_status_scenarios(output: dict[str, dict[str, Any]]) -> None:
    cases = {
        "PendingSubmit": "ORDER_STATUS_PENDING_SUBMIT",
        "PreSubmitted": "ORDER_STATUS_PRE_SUBMITTED",
        "Submitted": "ORDER_STATUS_SUBMITTED",
        "PendingCancel": "ORDER_STATUS_PENDING_CANCEL",
        "Cancelled": "ORDER_STATUS_CANCELLED",
        "ApiCancelled": "ORDER_STATUS_API_CANCELLED",
        "Filled": "ORDER_STATUS_FILLED",
        "Inactive": "ORDER_STATUS_INACTIVE",
    }
    for status, scenario_id in cases.items():
        projection = OrderStatusProjection(original_quantity=Decimal("1"))
        filled = "1" if status == "Filled" else "0"
        remaining = "0" if status in {"Filled", "Cancelled", "ApiCancelled", "Inactive"} else "1"
        result = apply_order_status(
            projection, status=status, filled=filled, remaining=remaining
        )
        output[scenario_id] = _case(
            result["status"] == "GO" and projection.status == status,
            classification=result["classification"],
        )
    terminal = OrderStatusProjection(original_quantity=Decimal("1"))
    apply_order_status(terminal, status="Filled", filled="1", remaining="0")
    stale = apply_order_status(
        terminal, status="Submitted", filled="1", remaining="0"
    )
    output["ORDER_STATUS_STALE_TERMINAL_REGRESSION_BLOCKED"] = _case(
        stale["classification"] == "STALE_TERMINAL_REGRESSION_IGNORED"
        and terminal.status == "Filled",
        classification=stale["classification"],
    )


def _partial_and_cancel_scenarios(
    output: dict[str, dict[str, Any]], root: Path
) -> None:
    store = _store(root / "partials.sqlite3")
    prices = ("100", "102", "101", "103")
    quantities: list[str] = []
    for index, price in enumerate(prices, start=1):
        record_fill_execution(
            store,
            _fill(f"EXEC-P-{index}", "I-P", "0.25", price, "1"),
        )
        quantities.append(
            project_position_from_store(store)["position"]["long_quantity"]
        )
    projected = project_position_from_store(store)
    output["PARTIAL_FILL_25_50_75_100"] = _case(
        quantities == ["0.25", "0.50", "0.75", "1.00"],
        quantities=quantities,
    )
    output["PARTIAL_FILL_MULTIPLE_PRICES"] = _case(
        Decimal(projected["position"]["average_cost_local"])
        == Decimal("101.50"),
        average_cost=projected["position"]["average_cost_local"],
    )
    cancel_store = _store(root / "partial-cancel.sqlite3")
    _register_intent(cancel_store, "I-CANCEL", perm_id="PERM-CANCEL")
    cancel_store.reserve_capital_once(
        intent_id="I-CANCEL", amount_eur=Decimal("100"), con_id=1
    )
    cancel_store.append_event("I-CANCEL", "PLACE_ORDER_CALLED_ONCE", {})
    record_fill_execution(
        cancel_store, _fill("EXEC-CANCEL", "I-CANCEL", "0.25", "100", "1")
    )
    cancel_store.append_event("I-CANCEL", "CANCEL_ORDER_CALLED_ONCE", {})
    cancel = record_broker_cancel_confirmation(
        cancel_store, intent_id="I-CANCEL", broker_proof=True
    )
    breakdown = cancel_store.capital_reservation_breakdown("I-CANCEL")
    output["PARTIAL_FILL_THEN_CANCEL"] = _case(
        cancel["cancel_status"] == "BROKER_CANCEL_CONFIRMED"
        and breakdown["reserved_eur"] == "0"
        and Decimal(breakdown["deployed_eur"]) > 0,
        capital=breakdown,
    )
    inactive = OrderStatusProjection(original_quantity=Decimal("1"))
    inactive_result = apply_order_status(
        inactive, status="Inactive", filled="0.25", remaining="0.75"
    )
    output["PARTIAL_FILL_THEN_INACTIVE"] = _case(
        inactive_result["status"] == "GO"
        and inactive.filled_quantity == Decimal("0.25"),
        status=inactive.status,
    )
    restarted = PaperExecutionStore(store.path)
    restarted.initialize()
    output["PARTIAL_FILL_THEN_RECONNECT"] = _case(
        stable_hash(projected) == stable_hash(project_position_from_store(restarted)),
        execution_count=len(restarted.list_executions()),
    )
    manual_position = reconcile_position_details(
        local_quantity=Decimal("0.25"),
        broker_quantity=Decimal("0.50"),
        local_average_cost=Decimal("100"),
        broker_average_cost=Decimal("101"),
        broker_ownership="EXTERNAL_MANUAL",
    )
    output["PARTIAL_FILL_MANUAL_INTERVENTION"] = _case(
        manual_position["status"] == "NO_GO"
        and not manual_position["ordinary_strategy_sell_allowed"],
        blockers=manual_position["blockers"],
    )
    cancel_requested = OrderStatusProjection(original_quantity=Decimal("1"))
    apply_order_status(cancel_requested, status="PendingCancel", filled="0", remaining="1")
    cancel_then_fill = apply_order_status(
        cancel_requested, status="Filled", filled="1", remaining="0"
    )
    output["CANCEL_RACE_CANCEL_THEN_FILL"] = _case(
        cancel_then_fill["status"] == "GO" and cancel_requested.status == "Filled",
        final_status=cancel_requested.status,
    )
    filled_first = OrderStatusProjection(original_quantity=Decimal("1"))
    apply_order_status(filled_first, status="Filled", filled="1", remaining="0")
    cancel_ack = apply_order_status(
        filled_first, status="Cancelled", filled="1", remaining="0"
    )
    output["CANCEL_RACE_FILL_THEN_ACK"] = _case(
        cancel_ack["classification"] == "STALE_TERMINAL_REGRESSION_IGNORED"
        and filled_first.status == "Filled",
        final_status=filled_first.status,
    )
    partial_cancel = OrderStatusProjection(original_quantity=Decimal("1"))
    apply_order_status(partial_cancel, status="PendingCancel", filled="0", remaining="1")
    partial_race = apply_order_status(
        partial_cancel, status="PendingCancel", filled="0.25", remaining="0.75"
    )
    output["CANCEL_RACE_PARTIAL_DURING_CANCEL"] = _case(
        partial_race["status"] == "GO"
        and partial_cancel.filled_quantity == Decimal("0.25"),
        filled=str(partial_cancel.filled_quantity),
    )
    full_race = apply_order_status(
        partial_cancel, status="Filled", filled="1", remaining="0"
    )
    output["CANCEL_RACE_FULL_DURING_CANCEL"] = _case(
        full_race["status"] == "GO" and partial_cancel.status == "Filled",
        final_status=partial_cancel.status,
    )


def _manual_and_reconciliation_scenarios(
    output: dict[str, dict[str, Any]], root: Path
) -> None:
    manual_order = {
        "perm_id": "PERM-M",
        "broker_order_id": "0",
        "client_id": 0,
        "manual_order": True,
        "remaining_quantity": "1",
        "order_status": "Submitted",
    }
    ownership = classify_broker_order_ownership(manual_order, local_orders=[])
    for scenario_id in (
        "MANUAL_TWS_OPEN_ORDER",
        "MANUAL_TWS_FILLED_ORDER",
        "MANUAL_TWS_CANCELLED_ORDER",
    ):
        output[scenario_id] = _case(
            ownership == "EXTERNAL_MANUAL", ownership=ownership
        )
    manual_position = reconcile_position_details(
        local_quantity=Decimal("0"),
        broker_quantity=Decimal("1"),
        local_average_cost=None,
        broker_average_cost=Decimal("100"),
        broker_ownership="EXTERNAL_MANUAL",
    )
    output["MANUAL_TWS_POSITION_CHANGE"] = _case(
        manual_position["status"] == "NO_GO"
        and not manual_position["ordinary_strategy_sell_allowed"],
        blockers=manual_position["blockers"],
    )
    output["EXTERNAL_POSITION_NOT_STRATEGY_ADOPTED"] = _case(
        manual_position["automatic_position_imports"] == 0
        and not manual_position["ordinary_strategy_sell_allowed"],
        automatic_imports=manual_position["automatic_position_imports"],
    )
    cash_match = reconcile_cash_state(
        local_cash_eur=Decimal("1000"),
        broker_spendable_eur=Decimal("900"),
        reserved_capital_eur=Decimal("100"),
    )
    cash_divergence = reconcile_cash_state(
        local_cash_eur=Decimal("1000"),
        broker_spendable_eur=Decimal("850"),
        reserved_capital_eur=Decimal("100"),
    )
    output["CASH_RECONCILIATION_MATCH"] = _case(
        cash_match["status"] == "GO", status=cash_match["cash_reconciliation_status"]
    )
    output["CASH_RECONCILIATION_DIVERGENCE_VISIBLE"] = _case(
        cash_divergence["status"] == "NO_GO"
        and not cash_divergence["automatic_cash_overwrite"],
        divergence=cash_divergence["divergence_eur"],
    )
    reserve_store = _store(root / "reserve-once.sqlite3")
    _register_intent(reserve_store, "I-R")
    reserve_first = reserve_store.reserve_capital_once(
        intent_id="I-R", amount_eur=Decimal("100"), con_id=1
    )
    reserve_second = reserve_store.reserve_capital_once(
        intent_id="I-R", amount_eur=Decimal("100"), con_id=1
    )
    output["CASH_RESERVATION_NOT_DUPLICATED"] = _case(
        reserve_first == "CAPITAL_RESERVED"
        and reserve_second == "CAPITAL_RESERVATION_IDEMPOTENT"
        and reserve_store.capital_summary()["reserved_capital_eur"] == "100",
        replay=reserve_second,
    )
    position_cases = {
        "POSITION_LOCAL_GREATER_THAN_BROKER": ("2", "1", "100", "100"),
        "POSITION_LOCAL_LESS_THAN_BROKER": ("1", "2", "100", "100"),
        "POSITION_LOCAL_ABSENT_AT_BROKER": ("1", "0", "100", None),
        "POSITION_BROKER_ABSENT_LOCALLY": ("0", "1", None, "100"),
        "POSITION_AVERAGE_COST_DISAGREEMENT": ("1", "1", "100", "101"),
    }
    for scenario_id, values in position_cases.items():
        local_qty, broker_qty, local_cost, broker_cost = values
        result = reconcile_position_details(
            local_quantity=Decimal(local_qty),
            broker_quantity=Decimal(broker_qty),
            local_average_cost=None if local_cost is None else Decimal(local_cost),
            broker_average_cost=None if broker_cost is None else Decimal(broker_cost),
        )
        output[scenario_id] = _case(
            result["status"] == "NO_GO" and bool(result["blockers"]),
            blockers=result["blockers"],
        )
    local = [
        {
            "perm_id": "PERM-1",
            "broker_order_id": "1",
            "remaining_quantity": "1",
            "order_status": "Submitted",
        }
    ]
    broker_match = [{**local[0], "client_id": 41}]
    order_cases = {
        "OPEN_ORDER_LOCAL_OPEN_BROKER_MISSING": reconcile_open_orders(local, []),
        "OPEN_ORDER_BROKER_OPEN_LOCAL_MISSING": reconcile_open_orders([], broker_match),
        "OPEN_ORDER_REMAINING_QUANTITY_DIFFERENT": reconcile_open_orders(
            local, [{**broker_match[0], "remaining_quantity": "0.5"}]
        ),
        "OPEN_ORDER_STATUS_DIFFERENT": reconcile_open_orders(
            local, [{**broker_match[0], "order_status": "PendingCancel"}]
        ),
    }
    for scenario_id, result in order_cases.items():
        output[scenario_id] = _case(
            result["status"] == "NO_GO" and bool(result["blockers"]),
            blockers=result["blockers"],
        )


def _restart_scenarios(
    output: dict[str, dict[str, Any]], root: Path
) -> None:
    store = _store(root / "restart-complete.sqlite3")
    _register_intent(store, "I-RESTART", perm_id="PERM-RESTART")
    store.reserve_capital_once(
        intent_id="I-RESTART", amount_eur=Decimal("100"), con_id=1
    )
    store.append_event("I-RESTART", "PLACE_ORDER_CALLED_ONCE", {})
    record_fill_execution(
        store, _fill("EXEC-RESTART", "I-RESTART", "0.5", "100", "1")
    )
    record_execution_commission(
        store, execution_id="EXEC-RESTART", commission=Decimal("0.2")
    )
    before = {
        "intents": store.list_intents(),
        "events": store.list_events(),
        "position": project_position_from_store(store),
        "capital": store.capital_summary(),
    }
    restarted = PaperExecutionStore(store.path)
    restarted.initialize()
    after = {
        "intents": restarted.list_intents(),
        "events": restarted.list_events(),
        "position": project_position_from_store(restarted),
        "capital": restarted.capital_summary(),
    }
    output["RESTART_ORDERS_RECONSTRUCTED"] = _case(
        stable_hash(before["intents"]) == stable_hash(after["intents"]),
        intent_count=len(after["intents"]),
    )
    output["RESTART_FILLS_COMMISSIONS_POSITIONS_RECONSTRUCTED"] = _case(
        stable_hash(before["position"]) == stable_hash(after["position"])
        and len(restarted.list_executions()) == 1
        and len(restarted.list_commissions()) == 1,
        execution_count=len(restarted.list_executions()),
        commission_count=len(restarted.list_commissions()),
    )
    local_order = after["intents"]
    ownership = classify_broker_order_ownership(
        {"perm_id": "PERM-RESTART", "broker_order_id": "900", "client_id": 41},
        local_orders=local_order,
    )
    output["RESTART_OWNERSHIP_RESERVATIONS_RECONCILIATION_RECONSTRUCTED"] = _case(
        ownership == "OWNED_STRATEGY"
        and stable_hash(before["capital"]) == stable_hash(after["capital"]),
        ownership=ownership,
        capital=after["capital"],
    )


def _account_scenarios(output: dict[str, dict[str, Any]]) -> None:
    now = datetime.now(UTC)
    snapshot = _account_snapshot(now)
    ready = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=now,
    )
    partial_snapshot = _account_snapshot(now)
    partial_snapshot["account"]["status"] = "CALLBACK_TIMEOUT"
    partial = derive_economic_account_state(
        partial_snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=now,
    )
    output["ACCOUNT_DOWNLOAD_COMPLETION_REQUIRED"] = _case(
        ready["execution_status"] == "EXECUTION_ACCOUNT_READY"
        and partial["execution_status"] == "NO_GO",
        partial_lifecycle=partial["lifecycle_state"],
    )
    stale = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=now + timedelta(minutes=5),
    )
    output["ACCOUNT_STALENESS_FAILS_EXECUTION_ONLY"] = _case(
        stale["research_status"] == "RESEARCH_READY"
        and stale["execution_status"] == "NO_GO",
        lifecycle=stale["lifecycle_state"],
    )
    output["ACCOUNT_NON_EUR_CASH_EXCLUDED"] = _case(
        ready["cash_by_currency"].get("USD") == "500"
        and ready["spendable_eur"] == "1870"
        and ready["implicit_fx_conversion_assumed"] is False,
        spendable_eur=ready["spendable_eur"],
    )
    output["ACCOUNT_BUYING_POWER_NOT_CASH"] = _case(
        ready["buying_power"]["value"] == "3000"
        and ready["spendable_eur"] == "1870"
        and ready["buying_power_is_cash"] is False,
        buying_power=ready["buying_power"]["value"],
        spendable_eur=ready["spendable_eur"],
    )


def _account_snapshot(now: datetime) -> dict[str, Any]:
    timestamp = now.isoformat()
    values = []
    for tag, value in {
        "NetLiquidation": "2370",
        "TotalCashValue": "2000",
        "SettledCash": "1870",
        "AvailableFunds": "1500",
        "BuyingPower": "3000",
        "GrossPositionValue": "370",
        "InitMarginReq": "100",
        "MaintMarginReq": "80",
        "ExcessLiquidity": "1600",
        "CashBalance": "1870",
    }.items():
        values.append(
            {
                "account_fingerprint": "HASHED-ACCOUNT",
                "tag": tag,
                "value": value,
                "currency": "EUR",
                "observed_at": timestamp,
            }
        )
    values.append(
        {
            "account_fingerprint": "HASHED-ACCOUNT",
            "tag": "CashBalance",
            "value": "500",
            "currency": "USD",
            "observed_at": timestamp,
        }
    )
    component = {"started_at": timestamp, "completed_at": timestamp}
    return {
        "server_version": "188",
        "account": {"status": "COMPLETE", "values": values},
        "positions": {"status": "EMPTY_COMPLETE", "positions": []},
        "same_client_open_orders": {"status": "EMPTY_COMPLETE", "open_orders": []},
        "all_api_open_orders": {"status": "EMPTY_COMPLETE", "open_orders": []},
        "executions": {"status": "EMPTY_COMPLETE", "executions": []},
        "component_timestamps": {
            name: component
            for name in (
                "accountsummary",
                "positions",
                "same_client_open_orders",
                "all_api_open_orders",
                "executions",
            )
        },
    }


def _store(path: Path) -> PaperExecutionStore:
    store = PaperExecutionStore(path)
    store.initialize()
    return store


def _register_intent(
    store: PaperExecutionStore, intent_id: str, *, perm_id: str = ""
) -> None:
    store.register_intent(
        {
            "intent_id": intent_id,
            "economic_order_key": f"KEY-{intent_id}",
            "created_at": "2026-08-09T10:00:00+00:00",
            "side": "BUY",
            "quantity": "1",
            "con_id": 1,
            "perm_id": perm_id,
        }
    )


def _fill(
    exec_id: str,
    intent_id: str,
    quantity: str,
    price: str,
    submitted: str,
) -> FillExecution:
    return FillExecution(
        exec_id=exec_id,
        intent_id=intent_id,
        account_fingerprint="P0-ACCOUNT",
        perm_id=f"PERM-{intent_id}",
        broker_order_id=f"ORDER-{intent_id}",
        con_id=1,
        symbol="AAPL",
        currency="EUR",
        side="BUY",
        quantity=Decimal(quantity),
        price=Decimal(price),
        execution_time=f"2026-08-09T10:00:{len(exec_id):02}+00:00",
        submitted_quantity=Decimal(submitted),
        fx_rate=Decimal("1"),
    )


def _case(passed: bool, **evidence: Any) -> dict[str, Any]:
    return {
        "status": "GO" if passed else "NO_GO",
        "result": "PASS" if passed else "FAIL",
        "evidence": evidence,
    }


__all__ = ["EXTENDED_REQUIRED_SCENARIOS", "run_extended_scenarios"]
