from __future__ import annotations

from decimal import Decimal

import main
from stocks.shadow.authority import authority_contract, authority_status
from stocks.shadow.audit import (
    PHASE8_2_MARKER,
    activation_audit,
    cost_model_audit,
    decision_clock_audit,
    init_ledger,
    register_fixtures,
    replay,
    security_audit,
    shadow_ledger_audit,
    simulate,
    status,
    target_weight_audit,
)
from stocks.shadow.clock import validate_decision_clock
from stocks.shadow.costs import ShadowCostModel, estimate_costs, return_after_costs
from stocks.shadow.decisions import (
    build_decision,
    build_fixture_signals,
    build_fixture_target,
    fixture_decision_request,
    frozen_fixture_universe,
)
from stocks.shadow.evaluation import evaluate_decision
from stocks.shadow.fills import make_hypothetical_fill
from stocks.shadow.models import ShadowTargetPortfolio, model_to_jsonable
from stocks.shadow.portfolio import (
    ShadowPortfolioState,
    apply_fx_normalization_once,
    apply_shadow_fill_once,
    book_dividend_once,
    portfolio_invariants,
    rebalance_cash_residual,
    reduce_position_once,
)
from stocks.shadow.provenance import validate_signal
from stocks.shadow.registry import activation_gate, default_contracts, registry_audit, registry_hash
from stocks.shadow.storage import COUNTERS, Phase82Layout, ShadowLedgerStore
from stocks.shadow.validation import TargetValidationLimits, validate_target_portfolio


def test_phase8_2_schema_cli_keeps_authorities_none(capsys) -> None:
    assert main.main(["shadow", "phase8-2", "schema"]) == 0
    out = capsys.readouterr().out
    assert PHASE8_2_MARKER in out
    assert '"strategy_authority": "NONE"' in out
    assert '"shadow_authority": "NONE"' in out
    assert '"execution_authority": "NONE"' in out


def test_authority_contract_blocks_every_non_none_authority() -> None:
    assert authority_contract() == {
        "strategy_authority": "NONE",
        "shadow_authority": "NONE",
        "execution_authority": "NONE",
    }
    assert authority_status("NONE")["status"] == "GO"
    for value in ["RESEARCH_SHADOW", "PAPER", "LIMITED_LIVE", "LIVE"]:
        assert authority_status(value)["decision_code"] == "AUTHORITY_NOT_GRANTED"


def test_strategy_registry_hash_disabled_rejected_and_synthetic_allowed() -> None:
    contracts = default_contracts("2026-07-21T21:00:00+00:00")
    assert registry_hash(contracts) == registry_hash(default_contracts("2026-07-21T21:00:00+00:00"))
    audit = registry_audit(contracts)
    assert audit["status"] == "GO"
    assert audit["rejected_strategies_blocked"] == "GO"
    assert audit["synthetic_fixture_status"] == "SYNTHETIC_FIXTURE_ALLOWED"
    synthetic = next(contract for contract in contracts if contract.strategy_id == "SYNTHETIC_SHADOW_FIXTURE_V1")
    rejected = next(contract for contract in contracts if contract.strategy_id.startswith("PHASE6"))
    assert activation_gate(synthetic)["decision_code"] == "SYNTHETIC_FIXTURE_ALLOWED"
    assert activation_gate(rejected)["decision_code"] == "STRATEGY_ACTIVATION_BLOCKED_NO_FINANCIAL_ELIGIBILITY"


def test_decision_clock_causality_same_close_future_data_and_hashes(tmp_path) -> None:
    contract = default_contracts("2026-07-21T21:00:00+00:00")[0]
    req = fixture_decision_request(tmp_path, contract)
    assert validate_decision_clock(
        frequency="FIXTURE_MANUAL",
        information_cutoff_timestamp=req.information_cutoff_timestamp,
        decision_timestamp=req.decision_timestamp,
        first_executable_timestamp=req.first_executable_timestamp,
        dataset_content_hashes=req.dataset_content_hashes,
    )["status"] == "GO"
    assert validate_decision_clock(
        frequency="FIXTURE_MANUAL",
        information_cutoff_timestamp=req.information_cutoff_timestamp,
        decision_timestamp=req.first_executable_timestamp,
        first_executable_timestamp=req.decision_timestamp,
        dataset_content_hashes=req.dataset_content_hashes,
    )["decision_code"] == "SAME_CLOSE_EXECUTION_BLOCKED"
    assert validate_decision_clock(
        frequency="FIXTURE_MANUAL",
        information_cutoff_timestamp=req.information_cutoff_timestamp,
        decision_timestamp=req.decision_timestamp,
        first_executable_timestamp=req.first_executable_timestamp,
        dataset_content_hashes={},
    )["decision_code"] == "MISSING_DATASET_HASH"
    assert validate_decision_clock(
        frequency="FIXTURE_MANUAL",
        information_cutoff_timestamp=req.information_cutoff_timestamp,
        decision_timestamp=req.decision_timestamp,
        first_executable_timestamp=req.first_executable_timestamp,
        dataset_content_hashes=req.dataset_content_hashes,
        feature_available_at="2026-07-22T00:00:00+00:00",
    )["decision_code"] == "FUTURE_DATA_BLOCKED"


def test_immutable_universe_signal_provenance_and_non_causal_block(tmp_path) -> None:
    contract = default_contracts("2026-07-21T21:00:00+00:00")[0]
    universe = frozen_fixture_universe(tmp_path)
    req = fixture_decision_request(tmp_path, contract)
    signals = build_fixture_signals(req, universe)
    assert universe.instrument_count == 3
    assert universe.universe_hash == frozen_fixture_universe(tmp_path).universe_hash
    assert all(validate_signal(signal, req.information_cutoff_timestamp)["status"] == "GO" for signal in signals)
    late_signal = signals[0].__class__(**{**model_to_jsonable(signals[0]), "available_at": "2026-07-22T00:00:00+00:00"})
    assert validate_signal(late_signal, req.information_cutoff_timestamp)["signal_status"] == "NON_CAUSAL_BLOCKED"


def test_target_weight_validation_caps_cash_turnover_and_unknowns(tmp_path) -> None:
    contract = default_contracts("2026-07-21T21:00:00+00:00")[0]
    req = fixture_decision_request(tmp_path, contract)
    target = build_fixture_target(req)
    eligible = set(frozen_fixture_universe(tmp_path).con_ids)
    assert validate_target_portfolio(target, eligible_con_ids=eligible)["target_status"] == "TARGET_PORTFOLIO_VALID"
    first = target.positions[0]
    negative = ShadowTargetPortfolio(target.decision_id, (first.__class__(first.con_id, first.symbol, first.region, first.sleeve, first.currency, Decimal("-0.1")), *target.positions[1:]), target.cash_weight, target.target_portfolio_hash, target.status)
    assert validate_target_portfolio(negative, eligible_con_ids=eligible)["target_status"] == "NEGATIVE_WEIGHT_BLOCKED"
    unknown = ShadowTargetPortfolio(target.decision_id, (first.__class__(999, first.symbol, first.region, first.sleeve, first.currency, first.target_weight), *target.positions[1:]), target.cash_weight, target.target_portfolio_hash, target.status)
    assert validate_target_portfolio(unknown, eligible_con_ids=eligible)["target_status"] == "UNKNOWN_INSTRUMENT_BLOCKED"
    assert validate_target_portfolio(target, eligible_con_ids=eligible, limits=TargetValidationLimits(region_cap=Decimal("0.1")))["target_status"] == "CONCENTRATION_LIMIT_BLOCKED"
    assert validate_target_portfolio(target, eligible_con_ids=eligible, limits=TargetValidationLimits(sleeve_cap=Decimal("0.1")))["target_status"] == "CONCENTRATION_LIMIT_BLOCKED"
    assert validate_target_portfolio(target, eligible_con_ids=eligible, limits=TargetValidationLimits(currency_cap=Decimal("0.1")))["target_status"] == "CONCENTRATION_LIMIT_BLOCKED"
    assert validate_target_portfolio(target, eligible_con_ids=eligible, limits=TargetValidationLimits(maximum_turnover=Decimal("0.1")), turnover=Decimal("0.2"))["target_status"] == "CONCENTRATION_LIMIT_BLOCKED"


def test_hypothetical_fills_cost_once_cash_position_and_nav_invariants() -> None:
    fill = make_hypothetical_fill(decision_id="D", con_id=1, quantity=Decimal("10"), price=Decimal("100"), price_timestamp="2026-07-22T13:30:00+00:00")
    partial = make_hypothetical_fill(decision_id="D", con_id=2, quantity=Decimal("4"), price=Decimal("100"), price_timestamp="2026-07-22T13:30:00+00:00", partial=True)
    assert fill.fill_id.startswith("SHADOW-FILL-")
    assert fill.fill_status == "HYPOTHETICAL_FILL"
    assert partial.quantity == Decimal("2")
    state = ShadowPortfolioState()
    assert apply_shadow_fill_once(state, fill) == "FILL_BOOKED_ONCE"
    fees_after_first = state.fees
    assert apply_shadow_fill_once(state, fill) == "DUPLICATE_FILL_BLOCKED"
    assert state.fees == fees_after_first
    assert portfolio_invariants(state)["status"] == "GO"
    costs = estimate_costs(Decimal("1000"), ShadowCostModel())
    assert costs["total_cost"] == costs["commission_cost"] + costs["spread_cost"] + costs["impact_cost"] + costs["fx_cost"]
    assert return_after_costs(Decimal("0.01"), Decimal("1"))["net_shadow_return"] < Decimal("0.01")


def test_position_accounting_reduce_close_dividend_fx_and_rebalance() -> None:
    state = ShadowPortfolioState()
    first = make_hypothetical_fill(decision_id="D", con_id=1, quantity=Decimal("10"), price=Decimal("100"), price_timestamp="2026-07-22T13:30:00+00:00")
    second = make_hypothetical_fill(decision_id="D", con_id=1, quantity=Decimal("4"), price=Decimal("100"), price_timestamp="2026-07-22T13:31:00+00:00")
    assert apply_shadow_fill_once(state, first) == "FILL_BOOKED_ONCE"
    assert apply_shadow_fill_once(state, second) == "FILL_BOOKED_ONCE"
    assert state.positions[1] == Decimal("14")
    assert reduce_position_once(state, event_id="REDUCE-1", con_id=1, quantity=Decimal("4"), price=Decimal("100")) == "POSITION_REDUCED"
    assert state.positions[1] == Decimal("10")
    assert reduce_position_once(state, event_id="CLOSE-1", con_id=1, quantity=Decimal("10"), price=Decimal("100")) == "POSITION_CLOSED"
    assert 1 not in state.positions
    assert book_dividend_once(state, event_id="DIV-1", amount=Decimal("5")) == "DIVIDEND_BOOKED_ONCE"
    assert apply_fx_normalization_once(state, event_id="FX-1", fx_return=Decimal("0.01")) == "FX_NORMALIZATION_BOOKED_ONCE"
    assert rebalance_cash_residual(state, target_cash=Decimal("1000")) == "REBALANCE_RECORDED"
    assert portfolio_invariants(state)["status"] == "GO"


def test_shadow_ledger_idempotency_conflict_replay_and_distinct_databases(tmp_path) -> None:
    layout = Phase82Layout.from_project_root(tmp_path)
    store = ShadowLedgerStore(layout.db_path)
    assert store.initialize()["status"] == "GO"
    payload = {"decision_id": "D1", "value": "A"}
    assert store.append_decision(payload) == "DECISION_CREATED"
    assert store.append_decision(payload) == "DUPLICATE_DECISION"
    assert store.append_decision({"decision_id": "D1", "value": "B"}) == "DECISION_ID_CONFLICT"
    replay_one = replay(tmp_path)
    replay_two = replay(tmp_path)
    assert replay_one["deterministic_state_hash"] == replay_two["deterministic_state_hash"]
    audit = shadow_ledger_audit(tmp_path)
    assert audit["phase7_ledger_distinct"] is True
    assert audit["phase8_database_distinct"] is True
    assert audit["phase8_1_database_distinct"] is True


def test_evaluation_lifecycle_benchmarks_and_audits_go(tmp_path) -> None:
    early = evaluate_decision(decision_id="D", decision_timestamp="2026-07-21T00:00:00+00:00", evaluation_start="2026-07-22T00:00:00+00:00", evaluation_end="2026-07-23T00:00:00+00:00", now="2026-07-22T12:00:00+00:00", costs=Decimal("0.01"))
    done = evaluate_decision(decision_id="D", decision_timestamp="2026-07-21T00:00:00+00:00", evaluation_start="2026-07-22T00:00:00+00:00", evaluation_end="2026-07-23T00:00:00+00:00", now="2026-07-24T00:00:00+00:00", costs=Decimal("0.01"))
    assert early.evaluation_status == "AWAITING_EVALUATION_HORIZON"
    assert done.evaluation_status == "EVALUATED"
    assert decision_clock_audit(tmp_path)["status"] == "GO"
    assert target_weight_audit(tmp_path)["status"] == "GO"
    assert cost_model_audit(tmp_path)["status"] == "GO"
    assert activation_audit(tmp_path)["status"] == "GO"


def test_simulate_replay_status_security_and_zero_counters(tmp_path) -> None:
    assert init_ledger(tmp_path)["status"] == "GO"
    assert register_fixtures(tmp_path)["status"] == "GO"
    sim = simulate(tmp_path)
    assert sim["status"] == "GO"
    assert sim["decision_status"] == "SHADOW_FIXTURE_VALIDATED"
    assert len(sim["scenarios"]) == 16
    assert {row["broker_calls"] for row in sim["scenarios"]} == {0}
    assert replay(tmp_path)["status"] == "GO"
    assert security_audit(tmp_path)["status"] == "GO"
    for key, value in COUNTERS.items():
        assert sim[key] == value == 0


def test_phase8_2_status_after_required_artifacts(tmp_path) -> None:
    init_ledger(tmp_path)
    register_fixtures(tmp_path)
    simulate(tmp_path)
    report = status(tmp_path)
    assert report["strategy_authority"] == "NONE"
    assert report["shadow_authority"] == "NONE"
    assert report["execution_authority"] == "NONE"
    assert report["FINANCIAL_FINALIST_GO"] is False


def test_build_decision_authority_none_and_status_validated(tmp_path) -> None:
    contract = default_contracts("2026-07-21T21:00:00+00:00")[0]
    universe = frozen_fixture_universe(tmp_path)
    req = fixture_decision_request(tmp_path, contract)
    target = build_fixture_target(req)
    decision = build_decision(contract, req, target, universe)
    assert decision.authority == "NONE"
    assert decision.status == "SHADOW_FIXTURE_VALIDATED"
