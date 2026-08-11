from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable

from stocks.application.phase_gates import phase1_freeze_status
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution import (
    PHASE9_FREEZE_MARKER,
    PHASE9_MARKER,
    phase9_reconcile,
    phase9_status,
)
from stocks.ibkr.paper_execution.adapter import (
    build_limit_day_order,
    build_stock_contract,
    connect_phase9_writer,
)
from stocks.ibkr.paper_execution.config import load_paper_writer_config
from stocks.ibkr.paper_execution.models import ManualPaperIntent
from stocks.ibkr.paper_execution.order_ids import allocate_order_id
from stocks.ibkr.paper_execution.storage import (
    PaperExecutionStore,
    Phase9Layout,
)
from stocks.ibkr.paper_execution.submission import submit_place_order_once
from stocks.operations.paper_runtime import (
    PaperRuntimeStore,
    database_path,
    phase9_position_quantities,
)
from stocks.operations.paper_safety import paper_kill_switch_status


AUTO_PAPER_AUTHORITY = "AUTOMATIC_BOUNDED_PAPER"
AUTO_PAPER_MARKER = "BOUNDED_AUTOMATIC_PAPER_WRITER_GO"
AUTO_PAPER_LIMITS: dict[str, Any] = {
    "max_orders_per_cycle": 1,
    "max_quantity_per_order": 1,
    "max_order_notional_eur": "100",
    "max_open_positions": 1,
    "max_open_orders": 1,
    "security_types": ["STK"],
    "order_types": ["LIMIT"],
    "time_in_force": ["DAY"],
    "outside_rth": False,
    "shorting": False,
    "margin": False,
    "leverage": False,
}
NativeSubmitter = Callable[
    [Path, str | Path, dict[str, Any]], dict[str, Any]
]


def build_paper_strategy_allowlist(project_root: Path) -> dict[str, Any]:
    top20 = _read_json(
        project_root
        / "output"
        / "research"
        / "phase11_10"
        / "top20-strategies.json"
    )
    verification = _read_json(
        project_root
        / "output"
        / "research"
        / "autopilot"
        / "verification-status.json"
    )
    shariah_status = (
        verification.get("historical_shariah_status", {}).get("status")
    )
    qualified = [
        {
            "strategy_id": str(row.get("strategy_id", "")),
            "architecture": str(row.get("architecture", "")),
            "strategy_dna_hash": str(
                row.get("strategy_dna_hash", "")
            ),
            "evidence_tier": str(row.get("evidence_tier", "")),
            "evidence_hash": stable_hash(row),
        }
        for row in top20.get("strategies", [])
        if row.get("strategy_id")
        and row.get("strategy_dna_hash")
        and row.get("evidence_tier") == "PAPER_CANDIDATE"
        and not row.get("hard_veto_reasons")
    ]
    blockers = []
    if shariah_status != "PIT_ELIGIBILITY_GO":
        blockers.append("HISTORICAL_SHARIAH_PIT_UNAVAILABLE")
    if not qualified:
        blockers.append("NO_PIT_QUALIFIED_PAPER_STRATEGY")
    status = "GO" if not blockers else "NO_GO"
    payload = {
        "schema": "automatic_paper_strategy_allowlist_v1",
        "status": status,
        "generated_at": _now(),
        "strategy_count": len(qualified),
        "strategies": qualified,
        "historical_shariah_status": shariah_status,
        "blockers": blockers,
        "strategy_authority": (
            "FROZEN_PIT_ALLOWLIST" if status == "GO" else "NONE"
        ),
        "execution_authority": "NONE",
        "paper_place_order_calls": 0,
        "live_place_order_calls": 0,
    }
    _publish(project_root, "paper-strategy-allowlist.json", payload)
    return payload


def automatic_paper_preflight(
    project_root: Path,
    env_file: str | Path = ".env.ibkr",
) -> dict[str, Any]:
    phase1 = phase1_freeze_status(project_root)
    phase9 = phase9_status(project_root)
    reconciliation = phase9_reconcile(project_root)
    allowlist = build_paper_strategy_allowlist(project_root)
    config, config_errors = load_paper_writer_config(
        project_root, env_file
    )
    freeze = _read_json(
        project_root / "output" / "ibkr" / "phase9" / "freeze-status.json"
    )
    blockers = []
    kill_switch = paper_kill_switch_status(project_root)
    if kill_switch.get("active") or kill_switch.get("status") != "GO":
        blockers.append("PAPER_KILL_SWITCH_ACTIVE")
    if not phase1.frozen:
        blockers.append("PHASE1_FREEZE_INTEGRITY_BLOCKED")
    if phase9.get("status") != PHASE9_MARKER:
        blockers.append("PHASE9_FILL_CLOSE_CANARY_REQUIRED")
    if freeze.get("freeze_status") != PHASE9_FREEZE_MARKER:
        blockers.append("PHASE9_FULL_FREEZE_REQUIRED")
    if reconciliation.get("reconciliation_status") not in {
        "PAPER_RECONCILED_EMPTY",
        "PAPER_RECONCILED",
        "PAPER_RECONCILED_OPEN_LONG",
    }:
        blockers.append("PAPER_RECONCILIATION_NOT_GO")
    if allowlist.get("status") != "GO":
        blockers.extend(allowlist.get("blockers", []))
    blockers.extend(config_errors)
    if config is None:
        blockers.append("PAPER_WRITER_CONFIG_BLOCKED")
    status = "GO" if not blockers else "NO_GO"
    payload = {
        "schema": "bounded_automatic_paper_preflight_v1",
        "status": status,
        "generated_at": _now(),
        "blockers": sorted(set(blockers)),
        "phase1_freeze_status": phase1.status,
        "phase9_status": phase9.get("status"),
        "phase9_freeze_status": freeze.get("freeze_status"),
        "reconciliation_status": reconciliation.get(
            "reconciliation_status"
        ),
        "broker_open_order_count": reconciliation.get(
            "broker_open_order_count"
        ),
        "broker_position_count": reconciliation.get(
            "broker_position_count"
        ),
        "paper_kill_switch_active": bool(kill_switch.get("active")),
        "allowlist_status": allowlist.get("status"),
        "allowlisted_strategy_count": allowlist.get("strategy_count", 0),
        "paper_config": None if config is None else config.safe_dict(),
        "execution_authority": "NONE",
        "automatic_orders_allowed": status == "GO",
        "paper_place_order_calls": 0,
        "live_place_order_calls": 0,
    }
    _publish(project_root, "paper-writer-preflight.json", payload)
    return payload


def execute_automatic_paper_cycle(
    project_root: Path,
    *,
    execution_authority: str,
    env_file: str | Path = ".env.ibkr",
    preflight: dict[str, Any] | None = None,
    submitter: NativeSubmitter | None = None,
) -> dict[str, Any]:
    gate = (
        automatic_paper_preflight(project_root, env_file)
        if preflight is None
        else preflight
    )
    if execution_authority != AUTO_PAPER_AUTHORITY:
        return _blocked_cycle(
            project_root,
            "AUTHORITY_NOT_GRANTED",
            gate,
        )
    if gate.get("status") != "GO":
        return _blocked_cycle(
            project_root,
            "AUTOMATIC_PAPER_PREFLIGHT_BLOCKED",
            gate,
        )
    allowlist = _read_json(
        project_root
        / "output"
        / "operations"
        / "paper-strategy-allowlist.json"
    )
    allowed = {
        str(row.get("strategy_id", ""))
        for row in allowlist.get("strategies", [])
    }
    store = PaperRuntimeStore(database_path(project_root))
    store.initialize()
    plans = [
        *reversed(store.plans("exit_plans")),
        *reversed(store.plans("entry_plans")),
    ]
    positions = phase9_position_quantities(project_root)
    phase9_store = PaperExecutionStore(
        Phase9Layout.from_project_root(project_root).db_path
    )
    phase9_store.initialize()
    effective_notional_limit = _effective_notional_limit(gate)
    daily_target_gate = _daily_profit_target_gate(project_root)
    accepted = []
    blocked = []
    economic_keys: set[str] = set()
    for plan in plans:
        validation = validate_automatic_paper_plan(
            plan,
            allowed_strategies=allowed,
            position_quantities=positions,
            max_order_notional_eur=effective_notional_limit,
            new_entries_allowed=bool(
                daily_target_gate["new_entries_allowed"]
            ),
        )
        if validation["status"] != "GO":
            blocked.append(
                {
                    "plan_hash": stable_hash(plan),
                    "reason": validation["reason"],
                }
            )
            continue
        submission_state = _prior_submission_state(phase9_store, plan)
        if submission_state != "NEW":
            blocked.append(
                {
                    "plan_hash": stable_hash(plan),
                    "reason": submission_state,
                }
            )
            continue
        key = str(plan["economic_key"])
        if key in economic_keys:
            blocked.append(
                {
                    "plan_hash": stable_hash(plan),
                    "reason": "DUPLICATE_PLAN_IN_CYCLE",
                }
            )
            continue
        economic_keys.add(key)
        accepted.append(plan)
    accepted = accepted[: AUTO_PAPER_LIMITS["max_orders_per_cycle"]]
    native_submitter = submitter or _submit_native
    submissions = [
        native_submitter(project_root, env_file, plan)
        for plan in accepted
    ]
    place_calls = sum(
        int(item.get("paper_place_order_calls", 0))
        for item in submissions
    )
    status = (
        "GO"
        if all(item.get("status") == "GO" for item in submissions)
        else "NO_GO"
    )
    payload = {
        "schema": "bounded_automatic_paper_cycle_v1",
        "status": status,
        "marker": AUTO_PAPER_MARKER if status == "GO" else "NO_GO",
        "generated_at": _now(),
        "eligible_plan_count": len(accepted),
        "blocked_plan_count": len(blocked),
        "blocked_plans": blocked,
        "submissions": submissions,
        "execution_authority": execution_authority,
        "strategy_authority": "FROZEN_PIT_ALLOWLIST",
        "paper_place_order_calls": place_calls,
        "paper_cancel_order_calls": 0,
        "live_place_order_calls": 0,
        "automatic_submissions": place_calls,
        "effective_max_order_notional_eur": str(
            effective_notional_limit
        ),
        "daily_profit_target_gate": daily_target_gate,
    }
    _publish(project_root, "paper-writer-cycle.json", payload)
    return payload


def validate_automatic_paper_plan(
    plan: dict[str, Any],
    *,
    allowed_strategies: Iterable[str],
    position_quantities: dict[int, int],
    max_order_notional_eur: Decimal | str = Decimal("100"),
    new_entries_allowed: bool = True,
    now: datetime | None = None,
) -> dict[str, str]:
    allowed = set(allowed_strategies)
    if str(plan.get("strategy_id", "")) not in allowed:
        return _blocked("STRATEGY_NOT_ALLOWLISTED")
    if plan.get("broker_submission_status") != "READY_AFTER_AUTHORITY":
        return _blocked("PLAN_NOT_READY")
    if plan.get("execution_authority") != AUTO_PAPER_AUTHORITY:
        return _blocked("PLAN_AUTHORITY_NOT_GRANTED")
    if str(plan.get("security_type", "STK")) != "STK":
        return _blocked("SECURITY_TYPE_BLOCKED")
    if str(plan.get("order_type")) != "LIMIT":
        return _blocked("ORDER_TYPE_BLOCKED")
    if str(plan.get("time_in_force")) != "DAY":
        return _blocked("TIME_IN_FORCE_BLOCKED")
    if bool(plan.get("outside_rth")):
        return _blocked("OUTSIDE_RTH_BLOCKED")
    if str(plan.get("data_freshness", "")) != "FRESH":
        return _blocked("STALE_DATA_BLOCKED")
    expiration = _timestamp(plan.get("expiration_timestamp"))
    reference_time = now or datetime.now(UTC)
    if expiration is None:
        return _blocked("PLAN_EXPIRATION_REQUIRED")
    if expiration <= reference_time:
        return _blocked("PLAN_EXPIRED")
    side = str(plan.get("side", "")).upper()
    if side not in {"BUY", "SELL"}:
        return _blocked("SIDE_BLOCKED")
    quantity = _integer(plan.get("target_quantity"))
    if quantity != 1:
        return _blocked("WHOLE_SHARE_QUANTITY_BLOCKED")
    con_id = _integer(plan.get("con_id"))
    if con_id <= 0:
        return _blocked("CONTRACT_IDENTITY_REQUIRED")
    price = _decimal(plan.get("limit_price"))
    if price <= 0:
        return _blocked("LIMIT_PRICE_INVALID")
    if side == "BUY":
        if not new_entries_allowed:
            return _blocked("DAILY_PROFIT_TARGET_REACHED")
        if sum(position_quantities.values()) > 0:
            return _blocked("MAX_OPEN_POSITIONS_REACHED")
        if plan.get("proposal_status") != "VALID_SIGNAL_EXECUTABLE":
            return _blocked("ENTRY_RISK_NOT_APPROVED")
        required_cash = _decimal(plan.get("required_cash_eur"))
        if required_cash <= 0 or required_cash > _decimal(
            max_order_notional_eur
        ):
            return _blocked("ORDER_NOTIONAL_EXCEEDED")
    if side == "SELL":
        if plan.get("proposal_status") != "VALID_RISK_REDUCING_EXIT":
            return _blocked("EXIT_NOT_RISK_REDUCING")
        if position_quantities.get(con_id, 0) < quantity:
            return _blocked("SELL_WITHOUT_RECONCILED_POSITION")
    return {"status": "GO", "reason": "PLAN_APPROVED"}


def _daily_profit_target_gate(project_root: Path) -> dict[str, Any]:
    report = _read_json(
        project_root / "output" / "capital" / "daily_profit_target.json"
    )
    reached = bool(
        report.get("status") == "GO"
        and report.get("target_reached")
        and report.get("enforcement_active")
    )
    return {
        "status": (
            "DAILY_PROFIT_TARGET_REACHED"
            if reached
            else "DAILY_PROFIT_TARGET_NOT_REACHED"
        ),
        "target_reached": reached,
        "new_entries_allowed": not reached,
        "risk_reducing_exits_allowed": True,
        "force_liquidation": False,
        "risk_chasing_allowed": False,
    }


def _submit_native(
    project_root: Path,
    env_file: str | Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    config, errors = load_paper_writer_config(project_root, env_file)
    if config is None or errors:
        return {
            "status": "NO_GO",
            "submission_status": "PAPER_WRITER_CONFIG_BLOCKED",
            "errors": errors,
            "paper_place_order_calls": 0,
        }
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    intent = _automatic_intent(plan, config.approved_account_fingerprint)
    owner = store.economic_order_key_owner(intent.economic_order_key)
    events = store.list_events()
    if owner is not None:
        already_called = any(
            event["aggregate_id"] == owner
            and event["event_type"] == "PLACE_ORDER_CALLED_ONCE"
            for event in events
        )
        return {
            "status": "GO" if already_called else "NO_GO",
            "submission_status": (
                "IDEMPOTENT_REPLAY"
                if already_called
                else "INCOMPLETE_PRIOR_SUBMISSION_REVIEW"
            ),
            "intent_hash": stable_hash(owner),
            "paper_place_order_calls": 0,
        }
    registration = store.register_intent(asdict(intent))
    if registration != "INTENT_REGISTERED":
        return {
            "status": "NO_GO",
            "submission_status": registration,
            "paper_place_order_calls": 0,
        }
    service = None
    try:
        service, app, connection = connect_phase9_writer(config)
        if (
            app is None
            or connection["connection_status"]
            not in {"HEALTHY", "DEGRADED"}
            or app.next_valid_order_id is None
        ):
            return {
                "status": "NO_GO",
                "submission_status": "WRITER_CONNECTION_BLOCKED",
                **connection,
                "paper_place_order_calls": 0,
            }
        allocated = allocate_order_id(
            store,
            broker_next_id=app.next_valid_order_id,
            intent_id=intent.intent_id,
        )
        if allocated["order_id_status"] != "ORDER_ID_READY":
            return {
                "status": "NO_GO",
                "submission_status": allocated["order_id_status"],
                "paper_place_order_calls": 0,
            }
        local_order_id = store.latest_order_id_for_intent(intent.intent_id)
        if local_order_id is None:
            return {
                "status": "NO_GO",
                "submission_status": "ORDER_ID_STATE_CONFLICT",
                "paper_place_order_calls": 0,
            }
        result = submit_place_order_once(
            app,
            order_id=local_order_id,
            contract=build_stock_contract(intent),
            order=build_limit_day_order(intent),
            store=store,
            intent_id=intent.intent_id,
        )
        if result["status"] == "GO":
            store.append_event(
                intent.intent_id,
                "AUTOMATIC_PAPER_SUBMISSION",
                {
                    "strategy_hash": stable_hash(
                        str(plan["strategy_id"])
                    ),
                    "plan_hash": stable_hash(plan),
                },
            )
        return {
            **result,
            **connection,
            "intent_hash": stable_hash(intent.intent_id),
        }
    finally:
        if service is not None:
            service.disconnect()


def _automatic_intent(
    plan: dict[str, Any], account_fingerprint: str
) -> ManualPaperIntent:
    now = datetime.now(UTC)
    quantity = Decimal(str(plan["target_quantity"]))
    price = Decimal(str(plan["limit_price"]))
    required_cash = _decimal(plan.get("required_cash_eur"))
    if required_cash <= 0:
        required_cash = quantity * price
    economic_key = _automatic_economic_key(plan, now)
    estimated_local = quantity * price
    fx_rate = (
        required_cash / estimated_local
        if estimated_local > 0
        else Decimal("1")
    )
    return ManualPaperIntent(
        intent_id="AUTO-PAPER-INTENT-" + economic_key[:20],
        economic_order_key=economic_key,
        intent_source="AUTOMATIC_PAPER_ALLOWLIST",
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=10)).isoformat(),
        account_fingerprint=account_fingerprint,
        con_id=int(plan["con_id"]),
        symbol=str(plan["ticker"]),
        security_type=str(plan.get("security_type", "STK")),
        currency=str(plan.get("currency", "EUR")),
        exchange=str(plan.get("exchange", "SMART")),
        side=str(plan["side"]),
        quantity=quantity,
        order_type="LIMIT",
        limit_price=price,
        estimated_notional_local=estimated_local,
        estimated_notional_eur=required_cash,
        fx_rate=fx_rate,
        fx_rate_timestamp=now.isoformat(),
        session_date=now.date().isoformat(),
        outside_rth=False,
        time_in_force="DAY",
        contract_hash=stable_hash(
            {
                "con_id": int(plan["con_id"]),
                "symbol": str(plan["ticker"]),
                "security_type": str(plan.get("security_type", "STK")),
                "currency": str(plan.get("currency", "EUR")),
                "exchange": str(plan.get("exchange", "SMART")),
            }
        ),
        operator_reason="frozen PIT allowlisted automatic paper plan",
    )


def _blocked_cycle(
    project_root: Path,
    reason: str,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "bounded_automatic_paper_cycle_v1",
        "status": "NO_GO",
        "generated_at": _now(),
        "blocker": reason,
        "preflight_status": preflight.get("status"),
        "preflight_blockers": preflight.get("blockers", []),
        "execution_authority": "NONE",
        "paper_place_order_calls": 0,
        "paper_cancel_order_calls": 0,
        "live_place_order_calls": 0,
        "automatic_submissions": 0,
    }
    _publish(project_root, "paper-writer-cycle.json", payload)
    return payload


def _blocked(reason: str) -> dict[str, str]:
    return {"status": "NO_GO", "reason": reason}


def _integer(value: Any) -> int:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return -1
    if decimal != decimal.to_integral_value():
        return -1
    return int(decimal)


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("-1")


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _effective_notional_limit(gate: dict[str, Any]) -> Decimal:
    configured = _decimal(
        gate.get("paper_config", {}).get("max_order_notional_eur")
    )
    hard_limit = Decimal(AUTO_PAPER_LIMITS["max_order_notional_eur"])
    if configured <= 0:
        return hard_limit
    return min(configured, hard_limit)


def _automatic_economic_key(
    plan: dict[str, Any], now: datetime
) -> str:
    return stable_hash(
        {
            "automatic_plan_key": plan["economic_key"],
            "strategy_id": plan["strategy_id"],
            "session_date": now.date().isoformat(),
        }
    )


def _prior_submission_state(
    store: PaperExecutionStore, plan: dict[str, Any]
) -> str:
    economic_key = _automatic_economic_key(plan, datetime.now(UTC))
    owner = store.economic_order_key_owner(economic_key)
    if owner is None:
        return "NEW"
    called = any(
        event["aggregate_id"] == owner
        and event["event_type"] == "PLACE_ORDER_CALLED_ONCE"
        for event in store.list_events()
    )
    return (
        "ALREADY_SUBMITTED_IDEMPOTENT"
        if called
        else "INCOMPLETE_PRIOR_SUBMISSION_REVIEW"
    )


def _publish(
    project_root: Path, name: str, payload: dict[str, Any]
) -> None:
    path = project_root / "output" / "operations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _now() -> str:
    return datetime.now(UTC).isoformat()
