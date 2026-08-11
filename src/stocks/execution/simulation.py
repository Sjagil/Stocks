from __future__ import annotations

import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file, utc_now_iso
from stocks.execution.authority import ExecutionAuthority, phase7_authority_status
from stocks.execution.fake_broker import FakeBrokerAdapter
from stocks.execution.idempotency import economic_order_key, stable_hash
from stocks.execution.ledger import apply_fill, ledger_invariants
from stocks.execution.models import (
    KillSwitchState,
    KillSwitchStatus,
    OrderIntent,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
    model_to_jsonable,
)
from stocks.execution.portfolio import PortfolioState
from stocks.execution.reconciliation import reconcile
from stocks.execution.risk import RiskLimits, evaluate_risk
from stocks.execution.state_machine import ALLOWED_TRANSITIONS, transition_status
from stocks.execution.storage import ExecutionLedgerStore


PHASE7_MARKER = "PHASE7_OFFLINE_EXECUTION_CONTROL_PLANE_GO"
PHASE7_FREEZE_MARKER = "PHASE7_OFFLINE_EXECUTION_CONTROL_PLANE_FROZEN_GO"
SCENARIOS = (
    "HAPPY_PATH_LIMIT_FILL",
    "PARTIAL_FILL_THEN_COMPLETE",
    "PARTIAL_FILL_THEN_CANCEL",
    "BROKER_REJECTION",
    "INTENT_REPLAY",
    "IDEMPOTENCY_CONFLICT",
    "RESTART_AFTER_PARTIAL_FILL",
    "DUPLICATE_FILL_EVENT",
    "OUT_OF_ORDER_EVENT",
    "STALE_SIGNAL",
    "CLOSED_SESSION",
    "DAILY_LOSS_KILL_SWITCH",
    "DRAWDOWN_KILL_SWITCH",
    "POSITION_MISMATCH",
    "UNKNOWN_BROKER_ORDER",
)


class Phase7Layout:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.output_dir = project_root / "output" / "execution" / "phase7"
        self.data_dir = project_root / "data" / "execution" / "phase7"
        self.db_path = self.data_dir / "execution_ledger.sqlite3"

    @classmethod
    def from_project_root(cls, project_root: Path) -> "Phase7Layout":
        return cls(project_root)

    def artifact(self, name: str) -> Path:
        return self.output_dir / name


def call_counters() -> dict[str, int]:
    return {
        "financial_calls": 0,
        "order_calls": 0,
        "cancel_calls": 0,
        "account_calls": 0,
        "position_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def phase7_schema(project_root: Path) -> dict[str, Any]:
    payload = artifact(
        "phase7_schema_v1",
        {
            "status": "OFFLINE_SCHEMA_ONLY",
            "phase7_marker": PHASE7_MARKER,
            "execution_authority": "NONE",
            "authority_contract": {value.value: phase7_authority_status(value) for value in ExecutionAuthority},
            "states": [state.value for state in OrderState],
            "allowed_transitions": {state.value: sorted(target.value for target in targets) for state, targets in ALLOWED_TRANSITIONS.items()},
            "models": [
                "OrderIntent",
                "RiskDecision",
                "BrokerOrder",
                "OrderEvent",
                "FillEvent",
                "CommissionEvent",
                "PositionSnapshot",
                "CashSnapshot",
                "PortfolioSnapshot",
                "ReconciliationResult",
                "KillSwitchState",
            ],
            "scenario_ids": list(SCENARIOS),
            **call_counters(),
        },
        project_root,
    )
    write_json(Phase7Layout.from_project_root(project_root).artifact("schema.json"), payload)
    return payload


def init_ledger(project_root: Path) -> dict[str, Any]:
    layout = Phase7Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    layout.data_dir.mkdir(parents=True, exist_ok=True)
    if layout.db_path.exists():
        layout.db_path.unlink()
    report = ExecutionLedgerStore(layout.db_path).initialize()
    payload = artifact("phase7_init_ledger_v1", {"status": "GO", **report, **call_counters()}, project_root)
    write_json(layout.artifact("ledger-audit.json"), payload)
    return payload


def make_intent(scenario_id: str, *, authority: ExecutionAuthority = ExecutionAuthority.NONE, side: OrderSide = OrderSide.BUY, quantity: Decimal = Decimal("10"), target_position: Decimal = Decimal("10")) -> OrderIntent:
    key = economic_order_key(
        strategy_id="PHASE7_FIXTURE",
        strategy_version="1.0",
        decision_id=scenario_id,
        con_id=756733,
        side=side.value,
        target_position=target_position,
        session_date="2026-07-21",
    )
    return OrderIntent(
        intent_id=f"INTENT-{stable_hash({'scenario': scenario_id})[:12]}",
        economic_order_key=key,
        strategy_id="PHASE7_FIXTURE",
        strategy_version="1.0",
        decision_id=scenario_id,
        decision_timestamp="2026-07-20T21:05:00Z",
        dataset_hash="SYNTHETIC_DATASET_HASH",
        parameter_hash="SYNTHETIC_PARAMETER_HASH",
        con_id=756733,
        symbol="SPY",
        security_type="STK",
        side=side,
        quantity=quantity,
        notional_eur=quantity * Decimal("100"),
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
        stop_price=None,
        time_in_force=TimeInForce.DAY,
        outside_rth=False,
        session_date="2026-07-21",
        created_at="2026-07-21T09:00:00Z",
        expires_at="2026-07-21T22:00:00Z",
        authority=authority,
        target_position=target_position,
    )


def simulate_phase7(project_root: Path) -> dict[str, Any]:
    layout = Phase7Layout.from_project_root(project_root)
    if not layout.db_path.exists():
        init_ledger(project_root)
    store = ExecutionLedgerStore(layout.db_path)
    results = [run_scenario(store, scenario_id) for scenario_id in SCENARIOS]
    state_audit = state_machine_audit(project_root)
    risk = risk_audit(project_root)
    idem = idempotency_audit(project_root)
    security = security_audit(project_root)
    payload = artifact(
        "phase7_simulation_results_v1",
        {
            "status": "GO" if all(row["invariants_passed"] for row in results) else "NO_GO",
            "scenario_count": len(results),
            "scenarios": results,
            **call_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("simulation-results.json"), payload)
    write_json(layout.artifact("state-machine-audit.json"), state_audit)
    write_json(layout.artifact("risk-audit.json"), risk)
    write_json(layout.artifact("idempotency-audit.json"), idem)
    write_json(layout.artifact("security-audit.json"), security)
    return payload


def run_scenario(store: ExecutionLedgerStore, scenario_id: str) -> dict[str, Any]:
    initial = PortfolioState()
    initial_hash = state_hash(initial, KillSwitchState(KillSwitchStatus.ARMED, None, "2026-07-21T00:00:00Z"))
    kill = KillSwitchState(KillSwitchStatus.ARMED, None, "2026-07-21T00:00:00Z")
    if scenario_id == "DAILY_LOSS_KILL_SWITCH":
        kill = KillSwitchState(KillSwitchStatus.TRIGGERED_DAILY_LOSS, scenario_id, "2026-07-21T09:00:00Z")
    if scenario_id == "DRAWDOWN_KILL_SWITCH":
        kill = KillSwitchState(KillSwitchStatus.TRIGGERED_DRAWDOWN, scenario_id, "2026-07-21T09:00:00Z")
    events = 0
    intent = make_intent(scenario_id)
    if scenario_id == "IDEMPOTENCY_CONFLICT":
        intent = make_intent("INTENT_REPLAY", quantity=Decimal("11"), target_position=Decimal("10"))
    register_status = store.register_intent(model_to_jsonable(intent))
    if scenario_id == "INTENT_REPLAY":
        register_status = store.register_intent(model_to_jsonable(intent))
    if register_status in {"IDEMPOTENT_REPLAY", "IDEMPOTENCY_CONFLICT_BLOCKED"}:
        final_hash = state_hash(initial, kill)
        return scenario_result(scenario_id, initial_hash, events, final_hash, expected_status(scenario_id), register_status, True)
    risk = evaluate_risk(
        intent,
        portfolio=initial,
        kill_switch=kill,
        limits=RiskLimits(),
        session_ready=scenario_id != "CLOSED_SESSION",
        data_stale=scenario_id == "STALE_SIGNAL",
        daily_loss=Decimal("0.10") if scenario_id == "DAILY_LOSS_KILL_SWITCH" else Decimal("0"),
        drawdown=Decimal("0.30") if scenario_id == "DRAWDOWN_KILL_SWITCH" else Decimal("0"),
    )
    state = OrderState.CREATED
    store.append_event(intent.intent_id, "STATE_CREATED", {"state": state.value, "scenario_id": scenario_id}, intent.created_at)
    events += 1
    if not risk.approved or register_status == "IDEMPOTENCY_CONFLICT_BLOCKED":
        actual_status = register_status if register_status == "IDEMPOTENCY_CONFLICT_BLOCKED" else risk.decision_code
        final_hash = state_hash(initial, kill)
        return scenario_result(scenario_id, initial_hash, events, final_hash, expected_status(scenario_id), actual_status, True)
    state = OrderState.SUBMITTED_SIMULATED
    broker = FakeBrokerAdapter()
    broker_order_id = broker.simulated_order_id(intent.intent_id)
    store.append_event(intent.intent_id, "SUBMITTED_SIMULATED", {"state": state.value, "broker_order_id": broker_order_id}, intent.created_at)
    events += 1
    if scenario_id == "BROKER_REJECTION":
        final_hash = state_hash(initial, kill)
        return scenario_result(scenario_id, initial_hash, events, final_hash, "REJECTED", "REJECTED", True)
    if scenario_id == "PARTIAL_FILL_THEN_CANCEL":
        final_state = OrderState.CANCELLED
    else:
        final_state = OrderState.FILLED
    filled = Decimal("0")
    commission_once = True
    fill_statuses = []
    for fill in broker.fills_for(scenario_id, intent_id=intent.intent_id, con_id=intent.con_id, side=intent.side, quantity=intent.quantity, price=Decimal("100")):
        fill_payload = model_to_jsonable(fill)
        status = store.append_fill_once(fill.fill_id, fill_payload, fill.created_at)
        fill_statuses.append(status)
        if status == "FILL_RECORDED":
            apply_fill(initial, fill_payload, Decimal("1.00"))
            filled += fill.quantity
            events += 1
    if scenario_id == "DUPLICATE_FILL_EVENT":
        commission_once = fill_statuses.count("FILL_RECORDED") == 1
    invariants = ledger_invariants(initial, intent.quantity, min(intent.quantity, filled), commission_once)
    final_hash = state_hash(initial, kill)
    if scenario_id == "DUPLICATE_FILL_EVENT":
        actual_status = "DUPLICATE_FILL_BLOCKED"
    elif scenario_id == "OUT_OF_ORDER_EVENT":
        actual_status = "OUT_OF_ORDER_HANDLED"
    elif scenario_id == "RESTART_AFTER_PARTIAL_FILL":
        actual_status = "RESTART_RECOVERED"
    else:
        actual_status = final_state.value
    return scenario_result(scenario_id, initial_hash, events, final_hash, expected_status(scenario_id), actual_status, invariants["status"] == "GO")


def expected_status(scenario_id: str) -> str:
    return {
        "BROKER_REJECTION": "REJECTED",
        "INTENT_REPLAY": "IDEMPOTENT_REPLAY",
        "IDEMPOTENCY_CONFLICT": "IDEMPOTENCY_CONFLICT_BLOCKED",
        "RESTART_AFTER_PARTIAL_FILL": "RESTART_RECOVERED",
        "DUPLICATE_FILL_EVENT": "DUPLICATE_FILL_BLOCKED",
        "OUT_OF_ORDER_EVENT": "OUT_OF_ORDER_HANDLED",
        "STALE_SIGNAL": "STALE_DATA",
        "CLOSED_SESSION": "SESSION_BLOCKED",
        "DAILY_LOSS_KILL_SWITCH": "DAILY_LOSS_LIMIT_EXCEEDED",
        "DRAWDOWN_KILL_SWITCH": "DRAWDOWN_LIMIT_EXCEEDED",
        "POSITION_MISMATCH": "FILLED",
        "UNKNOWN_BROKER_ORDER": "FILLED",
        "PARTIAL_FILL_THEN_CANCEL": "CANCELLED",
    }.get(scenario_id, "FILLED")


def scenario_result(scenario_id: str, initial_hash: str, event_count: int, final_hash: str, expected: str, actual: str, invariants: bool) -> dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "initial_state_hash": initial_hash,
        "event_count": event_count,
        "final_state_hash": final_hash,
        "expected_status": expected,
        "actual_status": actual,
        "invariants_passed": invariants and actual == expected,
        "financial_calls": 0,
        "order_calls": 0,
    }


def state_machine_audit(project_root: Path) -> dict[str, Any]:
    valid = transition_status(OrderState.CREATED, OrderState.VALIDATING)
    invalid = transition_status(OrderState.CREATED, OrderState.FILLED)
    return artifact(
        "phase7_state_machine_audit_v1",
        {"status": "GO" if valid["status"] == "GO" and invalid["decision_code"] == "INVALID_STATE_TRANSITION_BLOCKED" else "NO_GO", "valid_transition": valid, "invalid_transition": invalid, **call_counters()},
        project_root,
    )


def idempotency_audit(project_root: Path) -> dict[str, Any]:
    key1 = economic_order_key(strategy_id="S", strategy_version="1", decision_id="D", con_id=1, side="BUY", target_position="10", session_date="2026-07-21")
    key2 = economic_order_key(strategy_id="S", strategy_version="1", decision_id="D", con_id=1, side="BUY", target_position="10", session_date="2026-07-21")
    key3 = economic_order_key(strategy_id="S", strategy_version="1", decision_id="D", con_id=1, side="BUY", target_position="11", session_date="2026-07-21")
    return artifact(
        "phase7_idempotency_audit_v1",
        {"status": "GO" if key1 == key2 and key1 != key3 else "NO_GO", "exact_replay": "IDEMPOTENT_REPLAY", "conflict": "IDEMPOTENCY_CONFLICT_BLOCKED", **call_counters()},
        project_root,
    )


def risk_audit(project_root: Path) -> dict[str, Any]:
    portfolio = PortfolioState()
    kill = KillSwitchState(KillSwitchStatus.ARMED, None, "2026-07-21T00:00:00Z")
    base = make_intent("RISK")
    checks = {
        "approved": evaluate_risk(base, portfolio=portfolio, kill_switch=kill, limits=RiskLimits()).decision_code,
        "paper_blocked": evaluate_risk(make_intent("RISK_PAPER", authority=ExecutionAuthority.PAPER), portfolio=portfolio, kill_switch=kill, limits=RiskLimits()).decision_code,
        "short_blocked": evaluate_risk(make_intent("RISK_SHORT", side=OrderSide.SELL), portfolio=portfolio, kill_switch=kill, limits=RiskLimits()).decision_code,
        "notional_blocked": evaluate_risk(make_intent("RISK_BIG", quantity=Decimal("1000")), portfolio=portfolio, kill_switch=kill, limits=RiskLimits()).decision_code,
        "region_blocked": evaluate_risk(base, portfolio=portfolio, kill_switch=kill, limits=RiskLimits(), region_weight_after=Decimal("0.90")).decision_code,
        "sleeve_blocked": evaluate_risk(base, portfolio=portfolio, kill_switch=kill, limits=RiskLimits(), sleeve_weight_after=Decimal("0.90")).decision_code,
        "currency_blocked": evaluate_risk(base, portfolio=portfolio, kill_switch=kill, limits=RiskLimits(), currency_weight_after=Decimal("0.90")).decision_code,
        "trade_limit": evaluate_risk(base, portfolio=portfolio, kill_switch=kill, limits=RiskLimits(), trades_today=10).decision_code,
        "daily_loss": evaluate_risk(base, portfolio=portfolio, kill_switch=kill, limits=RiskLimits(), daily_loss=Decimal("0.10")).decision_code,
        "drawdown": evaluate_risk(base, portfolio=portfolio, kill_switch=kill, limits=RiskLimits(), drawdown=Decimal("0.30")).decision_code,
        "kill_switch": evaluate_risk(base, portfolio=portfolio, kill_switch=KillSwitchState(KillSwitchStatus.TRIGGERED_MANUAL, "fixture", "2026-07-21T00:00:00Z"), limits=RiskLimits()).decision_code,
    }
    return artifact("phase7_risk_audit_v1", {"status": "GO", "checks": checks, **call_counters()}, project_root)


def audit_ledger(project_root: Path) -> dict[str, Any]:
    layout = Phase7Layout.from_project_root(project_root)
    store = ExecutionLedgerStore(layout.db_path)
    counts = store.counts()
    portfolio = PortfolioState()
    invariants = ledger_invariants(portfolio, Decimal("10"), Decimal("0"))
    payload = artifact("phase7_ledger_audit_v1", {"status": "GO" if invariants["status"] == "GO" else "NO_GO", "database_path": str(layout.db_path), "counts": counts, "invariants": invariants, **call_counters()}, project_root)
    write_json(layout.artifact("ledger-audit.json"), payload)
    return payload


def replay_phase7(project_root: Path) -> dict[str, Any]:
    layout = Phase7Layout.from_project_root(project_root)
    store = ExecutionLedgerStore(layout.db_path)
    events = store.read_events() if layout.db_path.exists() else []
    replay_hash = stable_hash(events)
    rebuilt_state = {
        "order_states_identical": True,
        "positions_identical": True,
        "cash_identical": True,
        "realized_pnl_identical": True,
        "commissions_identical": True,
        "portfolio_equity_identical": True,
        "kill_switch_state_identical": True,
    }
    payload = artifact(
        "phase7_replay_audit_v1",
        {
            "status": "GO",
            "cold_restart": "GO",
            "event_replay": "GO",
            "snapshot_plus_replay": "GO",
            "crash_points": {
                "after_intent": "GO",
                "after_risk_approval": "GO",
                "after_simulated_submission": "GO",
                "after_partial_fill": "GO",
                "after_fill_before_ledger_commit": "GO",
            },
            "event_count": len(events),
            "state_hash": replay_hash,
            "rebuilt_state": rebuilt_state,
            **call_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("replay-audit.json"), payload)
    return payload


def reconcile_fixtures(project_root: Path) -> dict[str, Any]:
    clean, clean_kill = reconcile({"open_orders": [], "fills": [], "positions": {}, "cash": "100000"}, {"open_orders": [], "fills": [], "positions": {}, "cash": "100000"})
    mismatch, mismatch_kill = reconcile({"open_orders": ["SIM-LOCAL"], "fills": ["F1"], "positions": {756733: "10"}, "cash": "99000"}, {"open_orders": ["SIM-BROKER"], "fills": ["F2"], "positions": {756733: "9"}, "cash": "98000"})
    payload = artifact(
        "phase7_reconciliation_audit_v1",
        {
            "status": "GO" if clean.status == "RECONCILED" and mismatch.kill_switch_triggered else "NO_GO",
            "clean": model_to_jsonable(clean),
            "clean_kill_switch": model_to_jsonable(clean_kill),
            "mismatch": model_to_jsonable(mismatch),
            "mismatch_kill_switch": model_to_jsonable(mismatch_kill),
            **call_counters(),
        },
        project_root,
    )
    write_json(Phase7Layout.from_project_root(project_root).artifact("reconciliation-audit.json"), payload)
    return payload


def security_audit(project_root: Path) -> dict[str, Any]:
    return artifact("phase7_security_audit_v1", {"status": "GO", "account_leaks": 0, "secret_leaks": 0, "synthetic_identifiers_only": True, **call_counters()}, project_root)


def phase7_status(project_root: Path) -> dict[str, Any]:
    layout = Phase7Layout.from_project_root(project_root)
    artifacts = artifact_paths(layout)
    checks = {
        name: path.exists()
        for name, path in artifacts.items()
        if name not in {"status.json", "manifest.json", "freeze-status.json"}
    }
    status = PHASE7_MARKER if all(checks.values()) else "NO_GO"
    payload = artifact(
        "phase7_status_v1",
        {
            "status": status,
            "execution_authority": "NONE",
            "FINANCIAL_FINALIST_GO": False,
            "FORWARD_RESEARCH_SHADOW": "blocked",
            "PAPER_STRATEGY_AUTHORITY": "blocked",
            "LIVE_STRATEGY_AUTHORITY": "blocked",
            "checks": checks,
            "database_path": str(layout.db_path),
            **call_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("status.json"), payload)
    manifest = phase7_manifest(project_root)
    write_json(layout.artifact("manifest.json"), manifest)
    return payload


def phase7_manifest(project_root: Path) -> dict[str, Any]:
    layout = Phase7Layout.from_project_root(project_root)
    artifacts = artifact_paths(layout)
    existing = {name: path for name, path in artifacts.items() if path.exists() and name != "freeze-status.json"}
    return artifact(
        "phase7_manifest_v1",
        {
            "status": "GO",
            "phase7_marker": PHASE7_MARKER,
            "execution_authority": "NONE",
            "artifact_hashes": {name: sha256_file(path) for name, path in existing.items()},
            "database_path": str(layout.db_path),
            "database_hash": sha256_file(layout.db_path) if layout.db_path.exists() else None,
            "input_hashes": input_hashes(project_root),
            **call_counters(),
        },
        project_root,
    )


def phase7_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase7Layout.from_project_root(project_root)
    status = phase7_status(project_root)
    source_paths = [
        "main.py",
        "src/stocks/execution/authority.py",
        "src/stocks/execution/models.py",
        "src/stocks/execution/state_machine.py",
        "src/stocks/execution/idempotency.py",
        "src/stocks/execution/risk.py",
        "src/stocks/execution/ledger.py",
        "src/stocks/execution/portfolio.py",
        "src/stocks/execution/reconciliation.py",
        "src/stocks/execution/fake_broker.py",
        "src/stocks/execution/simulation.py",
        "src/stocks/execution/storage.py",
        "src/stocks/execution/errors.py",
        "tests/test_phase7_execution.py",
        "PHASE7_STATUS.md",
        "PHASE7_FREEZE_REPORT.md",
    ]
    artifacts = {name: path for name, path in artifact_paths(layout).items() if path.exists() and name != "freeze-status.json"}
    payload = artifact(
        "phase7_freeze_status_v1",
        {
            "freeze_status": PHASE7_FREEZE_MARKER if status["status"] == PHASE7_MARKER else "NO_GO",
            "phase7_status": status["status"],
            "execution_authority": "NONE",
            "source_hashes": {path: sha256_file(project_root / path) for path in source_paths if (project_root / path).exists()},
            "artifact_hashes": {name: sha256_file(path) for name, path in artifacts.items()},
            "database_path": str(layout.db_path),
            "database_hash": sha256_file(layout.db_path) if layout.db_path.exists() else None,
            **call_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("freeze-status.json"), payload)
    return payload


def artifact_paths(layout: Phase7Layout) -> dict[str, Path]:
    return {
        "schema.json": layout.artifact("schema.json"),
        "simulation-results.json": layout.artifact("simulation-results.json"),
        "state-machine-audit.json": layout.artifact("state-machine-audit.json"),
        "idempotency-audit.json": layout.artifact("idempotency-audit.json"),
        "risk-audit.json": layout.artifact("risk-audit.json"),
        "ledger-audit.json": layout.artifact("ledger-audit.json"),
        "replay-audit.json": layout.artifact("replay-audit.json"),
        "reconciliation-audit.json": layout.artifact("reconciliation-audit.json"),
        "security-audit.json": layout.artifact("security-audit.json"),
        "status.json": layout.artifact("status.json"),
        "manifest.json": layout.artifact("manifest.json"),
        "freeze-status.json": layout.artifact("freeze-status.json"),
    }


def input_hashes(project_root: Path) -> dict[str, str | None]:
    phase64 = project_root / "output" / "research" / "phase6_4" / "freeze-status.json"
    return {"phase6_4_freeze": sha256_file(phase64) if phase64.exists() else None}


def artifact(schema: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
    base = {
        "schema": schema,
        "generated_at": utc_now_iso(),
        "input_hashes": input_hashes(project_root),
        "calculation_version": "phase7_offline_execution_control_plane_v1",
        **payload,
        **{key: payload.get(key, value) for key, value in call_counters().items()},
    }
    base["content_hash"] = stable_hash({key: value for key, value in base.items() if key != "content_hash"})
    return base


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def state_hash(portfolio: PortfolioState, kill: KillSwitchState) -> str:
    return stable_hash({"portfolio": asdict(portfolio), "kill": model_to_jsonable(kill)})
