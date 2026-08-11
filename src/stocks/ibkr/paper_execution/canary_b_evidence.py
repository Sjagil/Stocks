from __future__ import annotations

import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.executions import project_position
from stocks.ibkr.paper_execution.storage import PaperExecutionStore, Phase9Layout


def reconstruct_fill_close_evidence(project_root: Path) -> dict[str, Any]:
    """Reconstruct the first complete manual paper round trip from private evidence."""
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    events = store.list_events()
    executions = store.list_executions()
    commissions = store.list_commissions()
    reconciliation = _load_json(layout.artifact("reconciliation-audit.json"))

    event_counts = Counter(
        (str(event["aggregate_id"]), str(event["event_type"]))
        for event in events
    )
    intent_ids = list(
        dict.fromkeys(
            str(event["aggregate_id"])
            for event in events
            if event["event_type"] == "MANUAL_OPERATOR_INTENT"
        )
    )
    intents = [
        intent
        for intent_id in intent_ids
        if (intent := store.get_intent(intent_id)) is not None
        and intent.get("intent_source") == "MANUAL_OPERATOR"
    ]
    executions_by_intent = _group_executions(executions)

    submitted_buys = [
        intent
        for intent in intents
        if _side(intent) == "BUY"
        and event_counts[(str(intent["intent_id"]), "PLACE_ORDER_CALLED_ONCE")]
        > 0
    ]
    complete_buys = [
        intent
        for intent in submitted_buys
        if _intent_fill_status(intent, executions_by_intent) == "FULLY_FILLED"
    ]
    buy = complete_buys[0] if complete_buys else None
    sell = (
        _matching_complete_sell(
            buy,
            intents,
            executions_by_intent,
            event_counts,
        )
        if buy is not None
        else None
    )

    candidate_intents = [] if buy is None else [buy]
    if sell is not None:
        candidate_intents.append(sell)
    candidate_ids = {str(intent["intent_id"]) for intent in candidate_intents}
    candidate_executions = [
        row
        for row in executions
        if str(row["payload"].get("intent_id", "")) in candidate_ids
    ]
    candidate_execution_ids = {
        str(row["exec_identity"]) for row in candidate_executions
    }
    candidate_commissions = [
        row
        for row in commissions
        if str(row["payload"].get("exec_identity", ""))
        in candidate_execution_ids
    ]
    projection = project_position(
        candidate_executions,
        candidate_commissions,
    )

    fill_status = _fill_canary_status(
        submitted_buys=submitted_buys,
        complete_buys=complete_buys,
        buy=buy,
        event_counts=event_counts,
        executions_by_intent=executions_by_intent,
    )
    close_status = _close_canary_status(
        buy=buy,
        sell=sell,
        event_counts=event_counts,
        executions_by_intent=executions_by_intent,
        projection=projection,
        candidate_execution_ids=candidate_execution_ids,
        candidate_commissions=candidate_commissions,
        all_executions=executions,
        all_commissions=commissions,
        reconciliation=reconciliation,
    )
    go = fill_status == "GO" and close_status == "GO"
    evidence = {
        "status": "GO" if go else "NO_GO",
        "fill_canary": fill_status,
        "closing_sell_canary": close_status,
        "submitted_manual_buy_count": len(submitted_buys),
        "fully_filled_manual_buy_count": len(complete_buys),
        "candidate_buy_hash": (
            None
            if buy is None
            else "INTENT-" + stable_hash(str(buy["intent_id"]))[:12]
        ),
        "candidate_sell_hash": (
            None
            if sell is None
            else "INTENT-" + stable_hash(str(sell["intent_id"]))[:12]
        ),
        "candidate_execution_count": len(candidate_executions),
        "candidate_commission_count": len(candidate_commissions),
        "position_status": (
            projection.get("position", {}).get("position_status", "EMPTY")
        ),
        "projection_status": projection.get(
            "projection_status", "NO_EXECUTIONS"
        ),
        "pending_commission_count": int(
            projection.get("pending_commission_count", 0)
        ),
        "orphan_commission_count": _orphan_commission_count(
            executions,
            commissions,
        ),
        "duplicate_economic_fill_count": int(
            projection.get("duplicate_economic_fills", 0)
        ),
        "negative_position_prevented": int(
            projection.get("negative_position_prevented", 0)
        ),
        "reconciliation_status": reconciliation.get(
            "reconciliation_status", "MISSING"
        ),
        "broker_open_order_count": int(
            reconciliation.get("broker_open_order_count", 0)
        ),
        "broker_position_count": int(
            reconciliation.get("broker_position_count", 0)
        ),
        "unknown_broker_open_order_count": int(
            reconciliation.get("unknown_broker_open_order_count", 0)
        ),
        "missing_local_open_order_count": int(
            reconciliation.get("missing_local_open_order_count", 0)
        ),
        "automatic_state_adoptions": 0,
        "strategy_generated_intents": 0,
    }
    evidence["evidence_hash"] = stable_hash(evidence)
    return evidence


def _fill_canary_status(
    *,
    submitted_buys: list[dict[str, Any]],
    complete_buys: list[dict[str, Any]],
    buy: dict[str, Any] | None,
    event_counts: Counter[tuple[str, str]],
    executions_by_intent: dict[str, list[dict[str, Any]]],
) -> str:
    if buy is None:
        if submitted_buys:
            if any(
                executions_by_intent.get(str(intent["intent_id"]))
                for intent in submitted_buys
            ):
                return "BUY_PARTIAL_FILL_PENDING"
            return "BUY_FILL_PENDING"
        return "NOT_RUN_REQUIRES_OPERATOR_APPROVAL"
    intent_id = str(buy["intent_id"])
    if len(complete_buys) > 1:
        return "MULTIPLE_FILLED_BUYS_REVIEW_REQUIRED"
    if event_counts[(intent_id, "PLACE_ORDER_CALLED_ONCE")] != 1:
        return "BUY_PLACE_COUNT_MISMATCH"
    if _intent_fill_status(buy, executions_by_intent) != "FULLY_FILLED":
        return "BUY_FILL_INCOMPLETE"
    return "GO"


def _close_canary_status(
    *,
    buy: dict[str, Any] | None,
    sell: dict[str, Any] | None,
    event_counts: Counter[tuple[str, str]],
    executions_by_intent: dict[str, list[dict[str, Any]]],
    projection: dict[str, Any],
    candidate_execution_ids: set[str],
    candidate_commissions: list[dict[str, Any]],
    all_executions: list[dict[str, Any]],
    all_commissions: list[dict[str, Any]],
    reconciliation: dict[str, Any],
) -> str:
    if buy is None:
        return "NOT_RUN_REQUIRES_FILLED_LONG_POSITION"
    if sell is None:
        return "CLOSING_SELL_REQUIRED"
    sell_id = str(sell["intent_id"])
    if event_counts[(sell_id, "PLACE_ORDER_CALLED_ONCE")] != 1:
        return "SELL_PLACE_COUNT_MISMATCH"
    if _intent_fill_status(sell, executions_by_intent) != "FULLY_FILLED":
        return "SELL_FILL_INCOMPLETE"
    if projection.get("status") != "GO":
        return str(projection.get("projection_status", "POSITION_LEDGER_BLOCKED"))
    if projection.get("position", {}).get("position_status") != "CLOSED":
        return "POSITION_NOT_CLOSED"
    if projection.get("duplicate_economic_fills", 0) != 0:
        return "DUPLICATE_ECONOMIC_FILL_BLOCKED"
    if _orphan_commission_count(all_executions, all_commissions) != 0:
        return "ORPHAN_COMMISSION_BLOCKED"
    commissioned_execution_ids = {
        str(row["payload"].get("exec_identity", ""))
        for row in candidate_commissions
    }
    if candidate_execution_ids != commissioned_execution_ids:
        return "COMMISSION_JOIN_PENDING"
    if reconciliation.get("status") != "GO":
        return "RECONCILIATION_BLOCKED"
    if reconciliation.get("reconciliation_status") != "PAPER_RECONCILED_EMPTY":
        return "PAPER_ACCOUNT_NOT_EMPTY"
    if any(
        int(reconciliation.get(field, 0)) != 0
        for field in (
            "local_active_order_count",
            "broker_open_order_count",
            "broker_position_count",
            "unknown_broker_open_order_count",
            "missing_local_open_order_count",
        )
    ):
        return "PAPER_ACCOUNT_NOT_EMPTY"
    return "GO"


def _matching_complete_sell(
    buy: dict[str, Any],
    intents: list[dict[str, Any]],
    executions_by_intent: dict[str, list[dict[str, Any]]],
    event_counts: Counter[tuple[str, str]],
) -> dict[str, Any] | None:
    buy_created = str(buy.get("created_at", ""))
    for intent in intents:
        intent_id = str(intent["intent_id"])
        if _side(intent) != "SELL":
            continue
        if int(intent.get("con_id", -1)) != int(buy.get("con_id", -2)):
            continue
        if str(intent.get("account_fingerprint", "")) != str(
            buy.get("account_fingerprint", "")
        ):
            continue
        if str(intent.get("created_at", "")) < buy_created:
            continue
        if event_counts[(intent_id, "PLACE_ORDER_CALLED_ONCE")] == 0:
            continue
        if _decimal(intent.get("quantity")) != _decimal(buy.get("quantity")):
            continue
        if _intent_fill_status(intent, executions_by_intent) == "FULLY_FILLED":
            return intent
    return None


def _group_executions(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        intent_id = str(row["payload"].get("intent_id", ""))
        grouped.setdefault(intent_id, []).append(row)
    return grouped


def _intent_fill_status(
    intent: dict[str, Any],
    executions_by_intent: dict[str, list[dict[str, Any]]],
) -> str:
    intent_id = str(intent["intent_id"])
    rows = executions_by_intent.get(intent_id, [])
    quantity = sum(
        (_decimal(row["payload"].get("quantity")) for row in rows),
        Decimal("0"),
    )
    submitted = _decimal(intent.get("quantity"))
    if quantity == 0:
        return "NOT_FILLED"
    if quantity < submitted:
        return "PARTIALLY_FILLED"
    if quantity == submitted:
        return "FULLY_FILLED"
    return "OVERFILLED_BLOCKED"


def _orphan_commission_count(
    executions: list[dict[str, Any]],
    commissions: list[dict[str, Any]],
) -> int:
    execution_ids = {str(row["exec_identity"]) for row in executions}
    return sum(
        str(row["payload"].get("exec_identity", "")) not in execution_ids
        for row in commissions
    )


def _side(intent: dict[str, Any]) -> str:
    return str(intent.get("side", "")).upper()


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("NaN")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
