from __future__ import annotations

import json
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from stocks.research.autopilot.contracts import stable_hash


DEFAULT_MINIMUM_ROUND_TRIPS = 5


def live_level_two_evidence(
    project_root: Path,
    *,
    minimum_round_trips: int = DEFAULT_MINIMUM_ROUND_TRIPS,
) -> dict[str, Any]:
    """Derive Level-2 evidence from immutable private broker observations.

    Broker execution IDs are joined to locally allocated order IDs.  A round
    trip counts only when a known Level-1 long position returns exactly to
    zero.  This function never creates authority or contacts the broker.
    """
    blockers: list[str] = []
    reconciliation = _read_json(
        project_root / "output/ibkr/live/reconciliation.json"
    )
    if not _content_hash_valid(reconciliation):
        blockers.append("LIVE_RECONCILIATION_INTEGRITY_BLOCKED")
    if (
        reconciliation.get("status") != "GO"
        or reconciliation.get("reconciliation_status")
        != "LIVE_RECONCILED_EMPTY"
        or reconciliation.get("unknown_orders") != 0
        or reconciliation.get("unknown_positions") != 0
    ):
        blockers.append("LIVE_RECONCILIATION_EMPTY_REQUIRED")

    observation_path = (
        project_root
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    authority_state = _read_json(
        project_root
        / "data/execution/live/private/authority-state.json"
    )
    level_one_activated_at = str(
        authority_state.get("activated_at") or ""
    )
    if not level_one_activated_at:
        blockers.append("LIVE_LEVEL_ONE_ACTIVATION_EVIDENCE_REQUIRED")
        snapshots: list[dict[str, Any]] = []
        snapshot_blockers: list[str] = []
    else:
        snapshots, snapshot_blockers = _observation_snapshots(
            observation_path,
            since=level_one_activated_at,
        )
    blockers.extend(snapshot_blockers)
    latest_snapshot_hash = (
        snapshots[-1]["snapshot_hash"] if snapshots else None
    )
    if (
        level_one_activated_at
        and latest_snapshot_hash != reconciliation.get("private_snapshot_hash")
    ):
        blockers.append("LIVE_OBSERVATION_BINDING_MISMATCH")

    ledger_path = (
        project_root
        / "data/execution/live/private/live_execution.sqlite3"
    )
    ledger, ledger_blockers = _live_ledger(ledger_path)
    blockers.extend(ledger_blockers)
    intents = ledger.get("intents", {})
    order_owners = ledger.get("order_owners", {})

    executions: dict[str, dict[str, Any]] = {}
    commissions: dict[str, Decimal] = {}
    for snapshot in snapshots:
        payload = snapshot["payload"]
        execution_component = payload.get("executions", {})
        if execution_component.get("execution_history_complete") is not True:
            blockers.append("LIVE_EXECUTION_HISTORY_INCOMPLETE")
        for commission in execution_component.get("commissions", []):
            execution_id = str(commission.get("execution_id") or "")
            if execution_id:
                commission_value = _nonnegative_decimal(
                    commission.get("commission")
                )
                if commission_value is None:
                    blockers.append("LIVE_EXECUTION_COMMISSION_INVALID")
                else:
                    commissions[execution_id] = commission_value
        for execution in execution_component.get("executions", []):
            execution_id = str(execution.get("execution_id") or "")
            if not execution_id:
                blockers.append("LIVE_EXECUTION_ID_REQUIRED")
                continue
            prior = executions.get(execution_id)
            if prior is not None and stable_hash(prior) != stable_hash(execution):
                blockers.append("LIVE_EXECUTION_REVISION_CONFLICT")
                continue
            executions[execution_id] = execution

    positions: dict[int, Decimal] = {}
    round_trips = 0
    known_execution_count = 0
    account_fingerprints: set[str] = set()
    execution_ids_by_intent: dict[str, list[str]] = {}
    entry_value_by_intent: dict[str, Decimal] = {}
    exit_value_by_intent: dict[str, Decimal] = {}
    verified_round_trips: list[dict[str, Any]] = []
    completed_intents: set[str] = set()
    ordered = sorted(
        executions.values(),
        key=lambda row: (
            str(row.get("execution_time") or row.get("observed_at") or ""),
            str(row.get("execution_id") or ""),
        ),
    )
    for execution in ordered:
        broker_order_hash = str(execution.get("broker_order_id") or "")
        intent_id = order_owners.get(broker_order_hash)
        intent = intents.get(intent_id or "")
        if intent is None:
            blockers.append("UNKNOWN_LIVE_EXECUTION_BLOCKED")
            continue
        if not _level_one_intent_valid(intent, ledger):
            blockers.append("NON_LEVEL_ONE_EXECUTION_IN_EVIDENCE")
            continue
        con_id = _positive_int(execution.get("con_id"))
        if con_id is None or con_id != _positive_int(intent.get("con_id")):
            blockers.append("LIVE_EXECUTION_CONTRACT_MISMATCH")
            continue
        quantity = _whole_positive_decimal(execution.get("quantity"))
        side = _side(execution.get("side"))
        if quantity is None or side is None:
            blockers.append("LIVE_EXECUTION_ECONOMICS_INVALID")
            continue
        execution_id = str(execution.get("execution_id"))
        if execution_id not in commissions:
            blockers.append("LIVE_EXECUTION_COMMISSION_REQUIRED")
        account_fingerprint = str(
            execution.get("account_fingerprint") or ""
        )
        if not account_fingerprint:
            blockers.append("LIVE_EXECUTION_ACCOUNT_REQUIRED")
        else:
            account_fingerprints.add(account_fingerprint)
        previous = positions.get(con_id, Decimal("0"))
        current = previous + quantity if side == "BUY" else previous - quantity
        if current < 0:
            blockers.append("LIVE_SHORT_OR_OVERSELL_BLOCKED")
            continue
        positions[con_id] = current
        execution_ids_by_intent.setdefault(str(intent_id), []).append(
            execution_id
        )
        execution_price = _positive_decimal(execution.get("price"))
        intent_fx = _positive_decimal(intent.get("fx_rate_to_eur"))
        if execution_price is None or intent_fx is None:
            blockers.append("LIVE_EXECUTION_ECONOMICS_INVALID")
            continue
        value_eur = quantity * execution_price * intent_fx
        if side == "BUY":
            entry_value_by_intent[str(intent_id)] = (
                entry_value_by_intent.get(str(intent_id), Decimal("0"))
                + value_eur
            )
        else:
            exit_value_by_intent[str(intent_id)] = (
                exit_value_by_intent.get(str(intent_id), Decimal("0"))
                + value_eur
            )
        if (
            previous > 0
            and current == 0
            and str(intent_id) not in completed_intents
        ):
            round_trips += 1
            completed_intents.add(str(intent_id))
            intent_execution_ids = execution_ids_by_intent.get(
                str(intent_id), []
            )
            commission_total = sum(
                (commissions.get(item, Decimal("0")) for item in intent_execution_ids),
                Decimal("0"),
            )
            realized_pnl = (
                exit_value_by_intent.get(str(intent_id), Decimal("0"))
                - entry_value_by_intent.get(str(intent_id), Decimal("0"))
                - commission_total
            )
            verified_round_trips.append(
                {
                    "strategy_id": intent.get("strategy_id"),
                    "target_id": intent.get("target_id"),
                    "intent_id": intent_id,
                    "planned_entry": intent.get("entry_limit_price"),
                    "planned_stop": intent.get("stop_price"),
                    "planned_quantity": intent.get("canary_qty"),
                    "actual_whole_share_quantity": str(quantity),
                    "allocated_bracket_order_count": ledger.get(
                        "order_count_by_intent", {}
                    ).get(str(intent_id), 0),
                    "exec_ids": intent_execution_ids,
                    "commission_count": sum(
                        item in commissions for item in intent_execution_ids
                    ),
                    "protective_bracket_submission_proven": (
                        "LIVE_BRACKET_PLACE_ORDER_CALLED_ONCE"
                        in ledger.get("events_by_intent", {}).get(
                            str(intent_id), set()
                        )
                    ),
                    "exit_fill_proven": True,
                    "realized_pnl_eur": str(realized_pnl),
                    "reconciliation_status": "LIVE_RECONCILED_EMPTY",
                }
            )
        known_execution_count += 1

    if any(quantity != 0 for quantity in positions.values()):
        blockers.append("LIVE_ROUND_TRIP_POSITION_STILL_OPEN")
    if len(account_fingerprints) > 1:
        blockers.append("LIVE_EVIDENCE_ACCOUNT_CHANGED")
    if round_trips < minimum_round_trips:
        blockers.append("MINIMUM_LIVE_LEVEL_ONE_ROUND_TRIPS_NOT_REACHED")

    report = {
        "schema": "live_level_two_evidence_v1",
        "status": "GO" if not blockers else "NO_GO",
        "minimum_round_trips": minimum_round_trips,
        "verified_round_trip_count": round_trips,
        "verified_execution_count": known_execution_count,
        "verified_round_trips": verified_round_trips,
        "round_trip_semantics": "VERIFIED_LEVEL_ONE_WHOLE_SHARE_FULL_LIFECYCLE",
        "observation_snapshot_count": len(snapshots),
        "level_one_activation_bound": bool(level_one_activated_at),
        "account_fingerprint_count": len(account_fingerprints),
        "latest_private_snapshot_hash": latest_snapshot_hash,
        "live_ledger_binding_hash": ledger.get("binding_hash"),
        "blockers": sorted(set(blockers)),
        "execution_authority": "NONE",
        "changes_authority": False,
        "broker_calls": 0,
        "broker_writes": 0,
    }
    report["content_hash"] = stable_hash(report)
    _write_json(
        project_root / "output/ibkr/live/level-two-evidence.json",
        report,
    )
    return report


def _observation_snapshots(
    path: Path,
    *,
    since: str | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.is_file():
        return [], ["PRIVATE_LIVE_OBSERVATIONS_REQUIRED"]
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            query = (
                "SELECT snapshot_hash, payload_json, created_at "
                "FROM snapshots"
            )
            parameters: tuple[str, ...] = ()
            if since:
                query += " WHERE created_at >= ?"
                parameters = (since,)
            rows = conn.execute(
                query + " ORDER BY created_at, snapshot_id", parameters
            ).fetchall()
    except (OSError, sqlite3.Error):
        return [], ["PRIVATE_LIVE_OBSERVATIONS_UNREADABLE"]
    snapshots: list[dict[str, Any]] = []
    blockers: list[str] = []
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            blockers.append("PRIVATE_LIVE_OBSERVATION_INVALID")
            continue
        if not isinstance(payload, dict):
            blockers.append("PRIVATE_LIVE_OBSERVATION_INVALID")
            continue
        snapshot_hash = str(row["snapshot_hash"])
        if snapshot_hash != stable_hash(payload):
            blockers.append("PRIVATE_LIVE_OBSERVATION_INTEGRITY_BLOCKED")
            continue
        snapshots.append(
            {
                "snapshot_hash": snapshot_hash,
                "payload": payload,
                "created_at": str(row["created_at"]),
            }
        )
    if not snapshots:
        blockers.append("PRIVATE_LIVE_OBSERVATIONS_REQUIRED")
    return snapshots, blockers


def _live_ledger(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, ["PRIVATE_LIVE_EXECUTION_LEDGER_REQUIRED"]
    try:
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            intent_rows = conn.execute(
                "SELECT intent_id, payload_hash, payload_json FROM intents"
            ).fetchall()
            order_rows = conn.execute(
                "SELECT order_id, intent_id FROM order_ids WHERE used = 1"
            ).fetchall()
            event_rows = conn.execute(
                "SELECT aggregate_id, event_type FROM events"
            ).fetchall()
    except (OSError, sqlite3.Error):
        return {}, ["PRIVATE_LIVE_EXECUTION_LEDGER_UNREADABLE"]
    blockers: list[str] = []
    intents: dict[str, dict[str, Any]] = {}
    for row in intent_rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            blockers.append("LIVE_INTENT_LEDGER_INTEGRITY_BLOCKED")
            continue
        if not isinstance(payload, dict) or str(row["payload_hash"]) != stable_hash(
            payload
        ):
            blockers.append("LIVE_INTENT_LEDGER_INTEGRITY_BLOCKED")
            continue
        intents[str(row["intent_id"])] = payload
    owners = {
        stable_hash({"broker_order_id": int(row["order_id"])})[:24]: str(
            row["intent_id"]
        )
        for row in order_rows
    }
    order_count_by_intent: dict[str, int] = {}
    for row in order_rows:
        intent_id = str(row["intent_id"])
        order_count_by_intent[intent_id] = (
            order_count_by_intent.get(intent_id, 0) + 1
        )
    events_by_intent: dict[str, set[str]] = {}
    for row in event_rows:
        events_by_intent.setdefault(str(row["aggregate_id"]), set()).add(
            str(row["event_type"])
        )
    binding = {
        "intent_hashes": sorted(stable_hash(value) for value in intents.values()),
        "allocated_order_hashes": sorted(owners),
    }
    return {
        "intents": intents,
        "order_owners": owners,
        "order_count_by_intent": order_count_by_intent,
        "events_by_intent": events_by_intent,
        "binding_hash": stable_hash(binding),
    }, blockers


def _level_one_intent_valid(
    intent: dict[str, Any], ledger: dict[str, Any]
) -> bool:
    quantity = _positive_decimal(intent.get("quantity"))
    desired = _positive_decimal(intent.get("desired_qty"))
    normal = _positive_decimal(intent.get("normal_allowed_qty"))
    canary = _positive_decimal(intent.get("canary_qty"))
    notional = _positive_decimal(intent.get("estimated_notional_eur"))
    hard_cap = _positive_decimal(
        intent.get("canary_notional_hard_cap_eur")
    )
    intent_id = str(intent.get("intent_id") or "")
    return bool(
        quantity is not None
        and quantity == quantity.to_integral_value()
        and desired is not None
        and desired == desired.to_integral_value()
        and normal is not None
        and normal == normal.to_integral_value()
        and canary is not None
        and canary == canary.to_integral_value()
        and desired >= normal >= canary == quantity
        and notional is not None
        and hard_cap is not None
        and notional <= hard_cap
        and intent.get("fractional_allowed") is False
        and int(intent.get("capital_level", 0) or 0) == 1
        and str(intent.get("asset_class"))
        in {"STOCK", "ETF", "COMMODITY_VEHICLE"}
        and bool(str(intent.get("strategy_id") or "").strip())
        and bool(str(intent.get("target_id") or "").strip())
        and _positive_decimal(intent.get("risk_per_share_eur")) is not None
        and _positive_decimal(intent.get("planned_total_risk_eur")) is not None
        and ledger.get("order_count_by_intent", {}).get(intent_id) == 3
        and "LIVE_BRACKET_PLACE_ORDER_CALLED_ONCE"
        in ledger.get("events_by_intent", {}).get(intent_id, set())
        and str(intent.get("security_type")) == "STK"
        and _positive_int(intent.get("con_id")) is not None
    )


def _side(value: Any) -> str | None:
    normalized = str(value or "").upper()
    if normalized in {"BUY", "BOT"}:
        return "BUY"
    if normalized in {"SELL", "SLD"}:
        return "SELL"
    return None


def _positive_decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed > 0 else None


def _whole_positive_decimal(value: Any) -> Decimal | None:
    parsed = _positive_decimal(value)
    if parsed is None or parsed != parsed.to_integral_value():
        return None
    return parsed


def _nonnegative_decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed >= 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    observed = payload.get("content_hash")
    return isinstance(observed, str) and observed == stable_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["live_level_two_evidence"]
