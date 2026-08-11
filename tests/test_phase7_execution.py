from __future__ import annotations

from decimal import Decimal

import main
from stocks.execution.authority import ExecutionAuthority, phase7_authority_status
from stocks.execution.fake_broker import FakeBrokerAdapter
from stocks.execution.idempotency import economic_order_key
from stocks.execution.ledger import apply_fill, ledger_invariants
from stocks.execution.models import KillSwitchState, KillSwitchStatus, OrderSide, OrderState, model_to_jsonable
from stocks.execution.portfolio import PortfolioState
from stocks.execution.reconciliation import reconcile
from stocks.execution.risk import RiskLimits, evaluate_risk
from stocks.execution.simulation import call_counters, make_intent, phase7_schema, run_scenario
from stocks.execution.state_machine import transition_status
from stocks.execution.storage import ExecutionLedgerStore


def test_authority_none_only_and_phase7_schema_cli(capsys) -> None:
    assert phase7_authority_status(ExecutionAuthority.NONE)["status"] == "GO"
    assert phase7_authority_status(ExecutionAuthority.PAPER)["decision_code"] == "AUTHORITY_NOT_GRANTED"
    assert phase7_authority_status(ExecutionAuthority.LIVE)["decision_code"] == "AUTHORITY_NOT_GRANTED"

    exit_code = main.main(["execution", "phase7", "schema"])
    assert exit_code == 0
    assert "PHASE7_OFFLINE_EXECUTION_CONTROL_PLANE_GO" in capsys.readouterr().out


def test_economic_order_key_is_deterministic() -> None:
    first = economic_order_key(strategy_id="S", strategy_version="1", decision_id="D", con_id=1, side="BUY", target_position="10", session_date="2026-07-21")
    second = economic_order_key(strategy_id="S", strategy_version="1", decision_id="D", con_id=1, side="BUY", target_position="10", session_date="2026-07-21")
    changed = economic_order_key(strategy_id="S", strategy_version="1", decision_id="D", con_id=1, side="BUY", target_position="11", session_date="2026-07-21")

    assert first == second
    assert first != changed


def test_idempotency_replay_and_conflict_use_unique_constraint(tmp_path) -> None:
    store = ExecutionLedgerStore(tmp_path / "ledger.sqlite3")
    store.initialize()
    intent = model_to_jsonable(make_intent("IDEMPOTENCY"))
    changed = {**intent, "quantity": "11"}

    assert store.register_intent(intent) == "INTENT_REGISTERED"
    assert store.register_intent(intent) == "IDEMPOTENT_REPLAY"
    assert store.register_intent(changed) == "IDEMPOTENCY_CONFLICT_BLOCKED"


def test_state_transitions_allow_and_block() -> None:
    assert transition_status(OrderState.CREATED, OrderState.VALIDATING)["status"] == "GO"
    assert transition_status(OrderState.CREATED, OrderState.FILLED)["decision_code"] == "INVALID_STATE_TRANSITION_BLOCKED"


def test_risk_rejection_codes_cover_limits_and_kill_switch() -> None:
    portfolio = PortfolioState()
    armed = KillSwitchState(KillSwitchStatus.ARMED, None, "2026-07-21T00:00:00Z")
    base = make_intent("RISK")

    assert evaluate_risk(base, portfolio=portfolio, kill_switch=armed, limits=RiskLimits()).decision_code == "RISK_APPROVED_SIMULATION_ONLY"
    assert evaluate_risk(make_intent("PAPER", authority=ExecutionAuthority.PAPER), portfolio=portfolio, kill_switch=armed, limits=RiskLimits()).decision_code == "AUTHORITY_NOT_GRANTED"
    assert evaluate_risk(make_intent("SELL", side=OrderSide.SELL), portfolio=portfolio, kill_switch=armed, limits=RiskLimits()).decision_code == "SHORT_POSITION_BLOCKED"
    assert evaluate_risk(make_intent("BIG", quantity=Decimal("1000")), portfolio=portfolio, kill_switch=armed, limits=RiskLimits()).decision_code == "ORDER_NOTIONAL_EXCEEDED"
    assert evaluate_risk(base, portfolio=portfolio, kill_switch=armed, limits=RiskLimits(), region_weight_after=Decimal("0.9")).decision_code == "REGION_LIMIT_EXCEEDED"
    assert evaluate_risk(base, portfolio=portfolio, kill_switch=armed, limits=RiskLimits(), sleeve_weight_after=Decimal("0.9")).decision_code == "SLEEVE_LIMIT_EXCEEDED"
    assert evaluate_risk(base, portfolio=portfolio, kill_switch=armed, limits=RiskLimits(), currency_weight_after=Decimal("0.9")).decision_code == "CURRENCY_LIMIT_EXCEEDED"
    assert evaluate_risk(base, portfolio=portfolio, kill_switch=armed, limits=RiskLimits(), trades_today=10).decision_code == "DAILY_TRADE_LIMIT_EXCEEDED"
    assert evaluate_risk(base, portfolio=portfolio, kill_switch=armed, limits=RiskLimits(), daily_loss=Decimal("0.1")).decision_code == "DAILY_LOSS_LIMIT_EXCEEDED"
    assert evaluate_risk(base, portfolio=portfolio, kill_switch=armed, limits=RiskLimits(), drawdown=Decimal("0.3")).decision_code == "DRAWDOWN_LIMIT_EXCEEDED"
    assert evaluate_risk(base, portfolio=portfolio, kill_switch=KillSwitchState(KillSwitchStatus.TRIGGERED_MANUAL, "fixture", "now"), limits=RiskLimits()).decision_code == "KILL_SWITCH_ACTIVE"


def test_fake_broker_uses_synthetic_ids_and_partial_duplicate_out_of_order() -> None:
    broker = FakeBrokerAdapter()
    intent = make_intent("BROKER")

    assert broker.simulated_order_id(intent.intent_id).startswith("SIM-")
    assert len(broker.fills_for("PARTIAL_FILL_THEN_COMPLETE", intent_id=intent.intent_id, con_id=1, side=OrderSide.BUY, quantity=Decimal("10"), price=Decimal("100"))) == 2
    assert len(broker.fills_for("DUPLICATE_FILL_EVENT", intent_id=intent.intent_id, con_id=1, side=OrderSide.BUY, quantity=Decimal("10"), price=Decimal("100"))) == 2
    out_of_order = broker.fills_for("OUT_OF_ORDER_EVENT", intent_id=intent.intent_id, con_id=1, side=OrderSide.BUY, quantity=Decimal("10"), price=Decimal("100"))
    assert out_of_order[0].fill_id.endswith("F2")


def test_ledger_commission_fill_and_cash_position_invariants(tmp_path) -> None:
    state = PortfolioState()
    fill = {"con_id": 1, "quantity": "10", "price": "100", "side": "BUY"}
    apply_fill(state, fill, Decimal("1"))

    inv = ledger_invariants(state, Decimal("10"), Decimal("10"), commission_once=True)
    assert state.positions[1] == Decimal("10")
    assert state.commissions == Decimal("1")
    assert inv["status"] == "GO"

    store = ExecutionLedgerStore(tmp_path / "ledger.sqlite3")
    store.initialize()
    assert store.append_fill_once("F1", fill, "now") == "FILL_RECORDED"
    assert store.append_fill_once("F1", fill, "now") == "DUPLICATE_FILL_BLOCKED"


def test_scenarios_restart_replay_duplicate_and_blocks(tmp_path) -> None:
    store = ExecutionLedgerStore(tmp_path / "ledger.sqlite3")
    store.initialize()

    for scenario, expected in {
        "HAPPY_PATH_LIMIT_FILL": "FILLED",
        "PARTIAL_FILL_THEN_CANCEL": "CANCELLED",
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
    }.items():
        result = run_scenario(store, scenario)
        assert result["actual_status"] == expected
        assert result["invariants_passed"] is True


def test_reconciliation_mismatches_trigger_kill_switch() -> None:
    clean, clean_kill = reconcile({"open_orders": [], "fills": [], "positions": {}, "cash": "1"}, {"open_orders": [], "fills": [], "positions": {}, "cash": "1"})
    mismatch, kill = reconcile({"open_orders": ["SIM-A"], "fills": ["F1"], "positions": {1: "1"}, "cash": "1"}, {"open_orders": ["SIM-B"], "fills": ["F2"], "positions": {1: "2"}, "cash": "2"})

    assert clean.status == "RECONCILED"
    assert clean_kill.status == KillSwitchStatus.ARMED
    assert mismatch.status == "RECONCILIATION_BLOCKED"
    assert "UNKNOWN_BROKER_ORDER" in mismatch.mismatches
    assert "POSITION_MISMATCH" in mismatch.mismatches
    assert "CASH_MISMATCH" in mismatch.mismatches
    assert kill.status == KillSwitchStatus.TRIGGERED_RECONCILIATION


def test_phase7_schema_counters_zero(tmp_path) -> None:
    schema = phase7_schema(tmp_path)

    for key, value in call_counters().items():
        assert schema[key] == value
    assert schema["execution_authority"] == "NONE"
