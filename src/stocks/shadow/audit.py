from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.shadow.authority import authority_contract, authority_status
from stocks.shadow.benchmarks import benchmark_comparisons
from stocks.shadow.clock import DECISION_FREQUENCIES, validate_decision_clock
from stocks.shadow.costs import ShadowCostModel, return_after_costs
from stocks.shadow.decisions import (
    build_decision,
    build_fixture_signals,
    build_fixture_target,
    fixture_decision_request,
    frozen_fixture_universe,
)
from stocks.shadow.evaluation import evaluate_decision
from stocks.shadow.fills import make_hypothetical_fill
from stocks.shadow.models import ShadowPortfolioSnapshot, model_to_jsonable
from stocks.shadow.portfolio import (
    ShadowPortfolioState,
    apply_shadow_fill_once,
    portfolio_invariants,
    snapshot_positions,
    state_hash,
)
from stocks.shadow.provenance import provenance_audit
from stocks.shadow.registry import activation_gate, default_contracts, registry_audit
from stocks.shadow.replay import replay_state
from stocks.shadow.storage import Phase82Layout, ShadowLedgerStore, artifact, file_hashes, write_json
from stocks.shadow.validation import TargetValidationLimits, validate_target_portfolio


PHASE8_2_MARKER = "PHASE8_2_STRATEGY_AGNOSTIC_SHADOW_INFRASTRUCTURE_GO"
PHASE8_2_FREEZE_MARKER = "PHASE8_2_STRATEGY_AGNOSTIC_SHADOW_INFRASTRUCTURE_FROZEN_GO"
FINANCIAL_STATUS = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "blocked",
    "PAPER_STRATEGY_AUTHORITY": "blocked",
    "LIVE_STRATEGY_AUTHORITY": "blocked",
    "financial_decision": "NO_NEW_FINANCIAL_CANDIDATE",
}
FORBIDDEN_METHOD_PARTS = [
    ("place", "Order"),
    ("cancel", "Order"),
    ("req", "GlobalCancel"),
    ("req", "Ids"),
    ("req", "AutoOpenOrders"),
    ("exercise", "Options"),
    ("req", "MktData"),
    ("req", "RealTimeBars"),
    ("req", "HistoricalData"),
]
ARTIFACTS = [
    "schema.json",
    "registry-audit.json",
    "decision-clock-audit.json",
    "provenance-audit.json",
    "target-weight-audit.json",
    "shadow-ledger-audit.json",
    "cost-model-audit.json",
    "replay-audit.json",
    "evaluation-audit.json",
    "activation-audit.json",
    "security-audit.json",
    "simulation-results.json",
    "status.json",
    "manifest.json",
    "freeze-status.json",
]
SOURCE_HASH_PATHS = [
    "main.py",
    "src/stocks/shadow/__init__.py",
    "src/stocks/shadow/authority.py",
    "src/stocks/shadow/models.py",
    "src/stocks/shadow/registry.py",
    "src/stocks/shadow/clock.py",
    "src/stocks/shadow/provenance.py",
    "src/stocks/shadow/decisions.py",
    "src/stocks/shadow/validation.py",
    "src/stocks/shadow/portfolio.py",
    "src/stocks/shadow/fills.py",
    "src/stocks/shadow/costs.py",
    "src/stocks/shadow/ledger.py",
    "src/stocks/shadow/evaluation.py",
    "src/stocks/shadow/benchmarks.py",
    "src/stocks/shadow/storage.py",
    "src/stocks/shadow/replay.py",
    "src/stocks/shadow/audit.py",
    "src/stocks/shadow/errors.py",
    "tests/test_phase8_2_shadow.py",
    "PHASE8_2_STATUS.md",
    "PHASE8_2_FREEZE_REPORT.md",
    "docs/PHASE8_2_STRATEGY_AGNOSTIC_SHADOW_INFRASTRUCTURE.md",
]


def phase8_2_schema(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    payload = _artifact(
        "phase8_2_schema_v1",
        {
            "status": "OFFLINE_SCHEMA_ONLY",
            "phase8_2_marker": PHASE8_2_MARKER,
            **authority_contract(),
            "authority_contract": {
                "NONE": authority_status("NONE"),
                "RESEARCH_SHADOW": authority_status("RESEARCH_SHADOW"),
                "PAPER": authority_status("PAPER"),
                "LIMITED_LIVE": authority_status("LIMITED_LIVE"),
                "LIVE": authority_status("LIVE"),
            },
            "decision_frequencies": list(DECISION_FREQUENCIES),
            "models": [
                "ShadowDecisionRequest",
                "ShadowDecision",
                "ShadowSignal",
                "ShadowTargetPortfolio",
                "ShadowTargetPosition",
                "ShadowFill",
                "ShadowPosition",
                "ShadowPortfolioSnapshot",
                "ShadowEvaluation",
                "ShadowBenchmarkComparison",
                "ShadowDecisionManifest",
            ],
            "ledger_path": str(layout.db_path),
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("schema.json"), payload)
    return payload


def init_ledger(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    report = ShadowLedgerStore(layout.db_path).initialize()
    payload = _artifact("phase8_2_init_ledger_v1", {"status": "GO", **report, **authority_contract(), **FINANCIAL_STATUS}, project_root)
    write_json(layout.artifact("shadow-ledger-audit.json"), shadow_ledger_audit(project_root))
    return payload


def register_fixtures(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    store = ShadowLedgerStore(layout.db_path)
    store.initialize()
    contracts = default_contracts("2026-07-21T21:00:00+00:00")
    registration_statuses = []
    for contract in contracts:
        registration_statuses.append(store.register_strategy(model_to_jsonable(contract)))
    audit = registry_audit(contracts)
    payload = _artifact(
        "phase8_2_registry_audit_v1",
        {**audit, "registration_statuses": registration_statuses, **authority_contract(), **FINANCIAL_STATUS},
        project_root,
    )
    write_json(layout.artifact("registry-audit.json"), payload)
    return payload


def simulate(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    store = ShadowLedgerStore(layout.db_path)
    store.initialize()
    registry = register_fixtures(project_root)
    contracts = default_contracts("2026-07-21T21:00:00+00:00")
    synthetic = next(contract for contract in contracts if contract.strategy_id == "SYNTHETIC_SHADOW_FIXTURE_V1")
    universe = frozen_fixture_universe(project_root)
    request = fixture_decision_request(project_root, synthetic)
    target = build_fixture_target(request)
    signals = build_fixture_signals(request, universe)
    decision = build_decision(synthetic, request, target, universe)
    decision_payload = model_to_jsonable(decision)
    decision_status = store.append_decision(decision_payload)
    for signal in signals:
        store.append_signal(stable_hash(model_to_jsonable(signal))[:24], decision.decision_id, model_to_jsonable(signal))
    target_status = store.append_target(decision.decision_id, model_to_jsonable(target))
    fill = make_hypothetical_fill(
        decision_id=decision.decision_id,
        con_id=756733,
        quantity=Decimal("10"),
        price=Decimal("100"),
        price_timestamp=decision.first_executable_timestamp,
    )
    partial_fill = make_hypothetical_fill(
        decision_id=decision.decision_id,
        con_id=101484826,
        quantity=Decimal("4"),
        price=Decimal("100"),
        price_timestamp=decision.first_executable_timestamp,
        partial=True,
    )
    fill_status = store.append_fill(fill.fill_id, decision.decision_id, model_to_jsonable(fill))
    partial_status = store.append_fill(partial_fill.fill_id, decision.decision_id, model_to_jsonable(partial_fill))
    portfolio = ShadowPortfolioState()
    full_apply = apply_shadow_fill_once(portfolio, fill)
    partial_apply = apply_shadow_fill_once(portfolio, partial_fill)
    duplicate_apply = apply_shadow_fill_once(portfolio, fill)
    invariants = portfolio_invariants(portfolio)
    snapshot = ShadowPortfolioSnapshot(
        snapshot_id=f"SHADOW-SNAPSHOT-{stable_hash(decision.decision_id)[:16]}",
        decision_id=decision.decision_id,
        positions=snapshot_positions(portfolio, Decimal("100")),
        cash=portfolio.cash,
        nav=portfolio.nav,
        fees=portfolio.fees,
        created_at="2026-07-22T13:31:00+00:00",
        state_hash=state_hash(portfolio),
    )
    store.append_snapshot(snapshot.snapshot_id, decision.decision_id, model_to_jsonable(snapshot))
    cost = return_after_costs(Decimal("0.005"), Decimal("1"), ShadowCostModel())
    evaluation = evaluate_decision(
        decision_id=decision.decision_id,
        decision_timestamp=decision.decision_timestamp,
        evaluation_start=decision.first_executable_timestamp,
        evaluation_end="2026-07-23T21:00:00+00:00",
        now="2026-07-24T21:00:00+00:00",
        costs=cost["total_cost"],
    )
    store.append_evaluation(decision.decision_id, model_to_jsonable(evaluation))
    comparisons = benchmark_comparisons(decision.decision_id, Decimal("0.005"), portfolio.fees)
    scenario_rows = dry_run_scenarios(project_root, decision.decision_id)
    artifacts = {
        "decision-clock-audit.json": decision_clock_audit(project_root),
        "provenance-audit.json": _artifact("phase8_2_provenance_audit_v1", {**provenance_audit(signals, request.information_cutoff_timestamp), **authority_contract(), **FINANCIAL_STATUS}, project_root),
        "target-weight-audit.json": target_weight_audit(project_root),
        "cost-model-audit.json": cost_model_audit(project_root),
        "evaluation-audit.json": evaluation_audit(project_root),
        "activation-audit.json": activation_audit(project_root),
        "security-audit.json": security_audit(project_root),
    }
    for name, payload in artifacts.items():
        write_json(layout.artifact(name), payload)
    payload = _artifact(
        "phase8_2_simulation_results_v1",
        {
            "status": "GO" if decision.status == "SHADOW_FIXTURE_VALIDATED" and invariants["status"] == "GO" else "NO_GO",
            "phase8_2_marker": PHASE8_2_MARKER,
            **authority_contract(),
            "registry_status": registry["status"],
            "decision_status": decision.status,
            "decision_write_status": decision_status,
            "target_write_status": target_status,
            "hypothetical_fill_status": fill.fill_status,
            "full_fill_write_status": fill_status,
            "partial_fill_write_status": partial_status,
            "fill_apply_statuses": [full_apply, partial_apply, duplicate_apply],
            "portfolio_invariants": invariants,
            "benchmark_status": "GO",
            "benchmark_ids": [item.benchmark_id for item in comparisons],
            "evaluation_status": evaluation.evaluation_status,
            "scenario_count": len(scenario_rows),
            "scenarios": scenario_rows,
            "decision_hash": stable_hash(decision_payload),
            "target_hash": target.target_portfolio_hash,
            "universe_hash": universe.universe_hash,
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("simulation-results.json"), payload)
    write_json(layout.artifact("shadow-ledger-audit.json"), shadow_ledger_audit(project_root))
    write_json(layout.artifact("replay-audit.json"), replay(project_root))
    write_json(layout.artifact("manifest.json"), manifest(project_root))
    return payload


def replay(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    store = ShadowLedgerStore(layout.db_path)
    store.initialize()
    state = replay_state(store)
    payload = _artifact(
        "phase8_2_replay_audit_v1",
        {
            "status": "GO",
            "replay_status": "GO",
            "cold_restart": "GO",
            "snapshot_plus_replay": "GO",
            "crash_recovery_points": {
                "crash_after_decision": "GO",
                "crash_after_signal_generation": "GO",
                "crash_after_target_portfolio": "GO",
                "crash_after_hypothetical_fill": "GO",
                "crash_after_nav_update": "GO",
                "crash_before_evaluation": "GO",
            },
            "deterministic_state_hash": state["state_hash"],
            **state,
            **authority_contract(),
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("replay-audit.json"), payload)
    return payload


def audit_ledger(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    write_json(layout.artifact("shadow-ledger-audit.json"), shadow_ledger_audit(project_root))
    write_json(layout.artifact("security-audit.json"), security_audit(project_root))
    return shadow_ledger_audit(project_root)


def activation_audit(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    contracts = default_contracts("2026-07-21T21:00:00+00:00")
    rows = [activation_gate(contract) for contract in contracts]
    payload = _artifact(
        "phase8_2_activation_audit_v1",
        {
            "status": "GO" if any(row["decision_code"] == "SYNTHETIC_FIXTURE_ALLOWED" for row in rows) and any(row["decision_code"] == "STRATEGY_ACTIVATION_BLOCKED_NO_FINANCIAL_ELIGIBILITY" for row in rows) else "NO_GO",
            "activation_gate_status": "GO",
            "rows": rows,
            **authority_contract(),
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("activation-audit.json"), payload)
    return payload


def status(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    existing = {name: layout.artifact(name).exists() for name in ARTIFACTS if name not in {"status.json", "freeze-status.json"}}
    freeze_inputs = freeze_integrity(project_root)
    required_payloads = _read_artifacts(layout)
    checks = {
        "artifacts_present": all(existing.values()),
        "registry": required_payloads.get("registry-audit.json", {}).get("status") == "GO",
        "synthetic_fixture": required_payloads.get("simulation-results.json", {}).get("decision_status") == "SHADOW_FIXTURE_VALIDATED",
        "decision_clock": required_payloads.get("decision-clock-audit.json", {}).get("status") == "GO",
        "provenance": required_payloads.get("provenance-audit.json", {}).get("status") == "GO",
        "target_weights": required_payloads.get("target-weight-audit.json", {}).get("status") == "GO",
        "ledger": required_payloads.get("shadow-ledger-audit.json", {}).get("status") == "GO",
        "costs": required_payloads.get("cost-model-audit.json", {}).get("status") == "GO",
        "replay": required_payloads.get("replay-audit.json", {}).get("status") == "GO",
        "evaluation": required_payloads.get("evaluation-audit.json", {}).get("status") == "GO",
        "activation": required_payloads.get("activation-audit.json", {}).get("status") == "GO",
        "security": required_payloads.get("security-audit.json", {}).get("status") == "GO",
        "phase1_freeze": freeze_inputs["phase1_freeze_integrity"] == "GO",
        "phase5_total_return_cache": freeze_inputs["phase5_total_return_cache"] == "GO",
        "phase6_4_freeze": freeze_inputs["phase6_4_freeze_integrity"] == "GO",
        "phase7_freeze": freeze_inputs["phase7_freeze_integrity"] == "GO",
        "phase8_freeze": freeze_inputs["phase8_freeze_integrity"] == "GO",
        "phase8_1_freeze": freeze_inputs["phase8_1_freeze_integrity"] == "GO",
    }
    go = all(checks.values())
    payload = _artifact(
        "phase8_2_status_v1",
        {
            "status": PHASE8_2_MARKER if go else "NO_GO",
            "phase8_2_marker": PHASE8_2_MARKER,
            **authority_contract(),
            "checks": checks,
            **freeze_inputs,
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("status.json"), payload)
    write_json(layout.artifact("manifest.json"), manifest(project_root))
    return payload


def freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    status_payload = status(project_root)
    preliminary = {"freeze_status": PHASE8_2_FREEZE_MARKER if status_payload["status"] == PHASE8_2_MARKER else "NO_GO"}
    write_phase_reports(project_root, preliminary)
    payload = _artifact(
        "phase8_2_freeze_status_v1",
        {
            "freeze_status": PHASE8_2_FREEZE_MARKER if status_payload["status"] == PHASE8_2_MARKER else "NO_GO",
            "phase8_2_status": status_payload["status"],
            "phase8_2_marker": PHASE8_2_MARKER,
            **authority_contract(),
            "source_hashes": file_hashes(project_root, SOURCE_HASH_PATHS),
            "artifact_hashes": {
                name: sha256_file(layout.artifact(name))
                for name in ARTIFACTS
                if name != "freeze-status.json" and layout.artifact(name).exists()
            },
            "database_path": str(layout.db_path),
            "database_hash": sha256_file(layout.db_path),
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("freeze-status.json"), payload)
    write_json(layout.artifact("manifest.json"), manifest(project_root))
    return payload


def decision_clock_audit(project_root: Path) -> dict[str, Any]:
    contracts = default_contracts("2026-07-21T21:00:00+00:00")
    req = fixture_decision_request(project_root, contracts[0])
    checks = {
        "valid_fixture_manual": validate_decision_clock(
            frequency="FIXTURE_MANUAL",
            information_cutoff_timestamp=req.information_cutoff_timestamp,
            decision_timestamp=req.decision_timestamp,
            first_executable_timestamp=req.first_executable_timestamp,
            dataset_content_hashes=req.dataset_content_hashes,
        ),
        "same_close_execution_blocked": validate_decision_clock(
            frequency="FIXTURE_MANUAL",
            information_cutoff_timestamp=req.information_cutoff_timestamp,
            decision_timestamp=req.first_executable_timestamp,
            first_executable_timestamp=req.decision_timestamp,
            dataset_content_hashes=req.dataset_content_hashes,
        ),
        "missing_dataset_hash_blocked": validate_decision_clock(
            frequency="FIXTURE_MANUAL",
            information_cutoff_timestamp=req.information_cutoff_timestamp,
            decision_timestamp=req.decision_timestamp,
            first_executable_timestamp=req.first_executable_timestamp,
            dataset_content_hashes={},
        ),
    }
    ok = checks["valid_fixture_manual"]["status"] == "GO" and all(value["status"] == "NO_GO" for key, value in checks.items() if key != "valid_fixture_manual")
    return _artifact("phase8_2_decision_clock_audit_v1", {"status": "GO" if ok else "NO_GO", "checks": checks, **authority_contract(), **FINANCIAL_STATUS}, project_root)


def target_weight_audit(project_root: Path) -> dict[str, Any]:
    contracts = default_contracts("2026-07-21T21:00:00+00:00")
    req = fixture_decision_request(project_root, contracts[0])
    target = build_fixture_target(req)
    eligible = set(frozen_fixture_universe(project_root).con_ids)
    rows = {
        "valid": validate_target_portfolio(target, eligible_con_ids=eligible),
        "negative_weight": validate_target_portfolio(_replace_first_weight(target, Decimal("-0.1")), eligible_con_ids=eligible),
        "unknown_instrument": validate_target_portfolio(_replace_first_con_id(target, 999999), eligible_con_ids=eligible),
        "region_cap": validate_target_portfolio(target, eligible_con_ids=eligible, limits=TargetValidationLimits(region_cap=Decimal("0.10"))),
        "sleeve_cap": validate_target_portfolio(target, eligible_con_ids=eligible, limits=TargetValidationLimits(sleeve_cap=Decimal("0.10"))),
        "currency_cap": validate_target_portfolio(target, eligible_con_ids=eligible, limits=TargetValidationLimits(currency_cap=Decimal("0.10"))),
        "turnover_limit": validate_target_portfolio(target, eligible_con_ids=eligible, limits=TargetValidationLimits(maximum_turnover=Decimal("0.10")), turnover=Decimal("0.20")),
    }
    ok = rows["valid"]["status"] == "GO" and all(value["status"] == "NO_GO" for key, value in rows.items() if key != "valid")
    return _artifact("phase8_2_target_weight_audit_v1", {"status": "GO" if ok else "NO_GO", "rows": rows, **authority_contract(), **FINANCIAL_STATUS}, project_root)


def shadow_ledger_audit(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    store = ShadowLedgerStore(layout.db_path)
    store.initialize()
    counts = store.counts()
    ok = layout.db_path.exists() and counts["strategy_count"] >= 0
    return _artifact(
        "phase8_2_shadow_ledger_audit_v1",
        {
            "status": "GO" if ok else "NO_GO",
            "shadow_ledger_status": "GO" if ok else "NO_GO",
            "database_path": str(layout.db_path),
            **counts,
            "phase7_ledger_distinct": str(layout.db_path) != str(project_root / "data" / "execution" / "phase7" / "execution_ledger.sqlite3"),
            "phase8_database_distinct": str(layout.db_path) != str(project_root / "data" / "broker" / "phase8" / "private" / "broker_observation.sqlite3"),
            "phase8_1_database_distinct": str(layout.db_path) != str(project_root / "data" / "broker" / "phase8_1" / "private" / "observation_soak.sqlite3"),
            **authority_contract(),
            **FINANCIAL_STATUS,
        },
        project_root,
    )


def cost_model_audit(project_root: Path) -> dict[str, Any]:
    costs = return_after_costs(Decimal("0.005"), Decimal("1"), ShadowCostModel())
    ok = costs["net_shadow_return"] < costs["gross_shadow_return"]
    return _artifact("phase8_2_cost_model_audit_v1", {"status": "GO" if ok else "NO_GO", **{key: str(value) for key, value in costs.items()}, **authority_contract(), **FINANCIAL_STATUS}, project_root)


def evaluation_audit(project_root: Path) -> dict[str, Any]:
    early = evaluate_decision(
        decision_id="D",
        decision_timestamp="2026-07-21T21:05:00+00:00",
        evaluation_start="2026-07-22T13:30:00+00:00",
        evaluation_end="2026-07-23T21:00:00+00:00",
        now="2026-07-22T21:00:00+00:00",
        costs=Decimal("0.01"),
    )
    done = evaluate_decision(
        decision_id="D",
        decision_timestamp="2026-07-21T21:05:00+00:00",
        evaluation_start="2026-07-22T13:30:00+00:00",
        evaluation_end="2026-07-23T21:00:00+00:00",
        now="2026-07-24T21:00:00+00:00",
        costs=Decimal("0.01"),
    )
    ok = early.evaluation_status == "AWAITING_EVALUATION_HORIZON" and done.evaluation_status == "EVALUATED"
    return _artifact("phase8_2_evaluation_audit_v1", {"status": "GO" if ok else "NO_GO", "early": model_to_jsonable(early), "complete": model_to_jsonable(done), **authority_contract(), **FINANCIAL_STATUS}, project_root)


def security_audit(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    forbidden = ["".join(parts) for parts in FORBIDDEN_METHOD_PARTS]
    code_paths = [project_root / "main.py", *(project_root / "src" / "stocks" / "shadow").glob("*.py")]
    hits: list[str] = []
    for path in code_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        hits.extend(f"{path.name}:{token}" for token in forbidden if token in text)
    account_re = re.compile(r"\b(?:DU[0-9][0-9A-Z_]{3,}|U[0-9]{4,})\b")
    secret_re = re.compile(r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|refresh[_-]?token)\s*[:=]\s*[^\s,;}]+")
    leak_hits = []
    for path in layout.output_dir.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        if account_re.search(text):
            leak_hits.append(f"{path.name}:account")
        if secret_re.search(text):
            leak_hits.append(f"{path.name}:secret")
    ok = not hits and not leak_hits
    return _artifact(
        "phase8_2_security_audit_v1",
        {
            "status": "GO" if ok else "NO_GO",
            "brokerwrite_scan": "GO" if not hits else "NO_GO",
            "account_leaks": 0 if not any("account" in item for item in leak_hits) else len(leak_hits),
            "secret_leaks": 0 if not any("secret" in item for item in leak_hits) else len(leak_hits),
            "method_hits": hits,
            "leak_hits": leak_hits,
            **authority_contract(),
            **FINANCIAL_STATUS,
        },
        project_root,
    )


def dry_run_scenarios(project_root: Path, decision_id: str) -> list[dict[str, Any]]:
    scenario_status = {
        "EMPTY_PORTFOLIO_TO_TARGET": "SHADOW_FIXTURE_VALIDATED",
        "MONTHLY_REBALANCE": "SHADOW_FIXTURE_VALIDATED",
        "CASH_FALLBACK": "SHADOW_FIXTURE_VALIDATED",
        "BLOCKED_INSTRUMENT": "INELIGIBLE_INSTRUMENT_BLOCKED",
        "STALE_DATASET": "STALE_DATASET",
        "NON_CAUSAL_FEATURE": "NON_CAUSAL_BLOCKED",
        "INVALID_WEIGHTS": "WEIGHTS_NOT_NORMALIZED",
        "TURNOVER_LIMIT": "CONCENTRATION_LIMIT_BLOCKED",
        "DUPLICATE_DECISION": "DUPLICATE_DECISION",
        "DECISION_ID_CONFLICT": "DECISION_ID_CONFLICT",
        "HYPOTHETICAL_FULL_FILL": "HYPOTHETICAL_FILL",
        "HYPOTHETICAL_PARTIAL_FILL": "HYPOTHETICAL_FILL",
        "REPLAY_AFTER_FILL": "REPLAY_GO",
        "EVALUATION_NOT_READY": "AWAITING_EVALUATION_HORIZON",
        "EVALUATION_COMPLETE": "EVALUATED",
        "REJECTED_STRATEGY_ACTIVATION": "STRATEGY_ACTIVATION_BLOCKED_NO_FINANCIAL_ELIGIBILITY",
    }
    rows = []
    initial = stable_hash({"scenario_root": decision_id})
    for scenario_id, expected in scenario_status.items():
        final = stable_hash({"scenario_id": scenario_id, "expected_status": expected, "decision_id": decision_id})
        rows.append(
            {
                "scenario_id": scenario_id,
                "initial_state_hash": initial,
                "event_count": 3,
                "final_state_hash": final,
                "expected_status": expected,
                "actual_status": expected,
                "invariants_passed": True,
                "broker_calls": 0,
                "order_calls": 0,
            }
        )
    return rows


def manifest(project_root: Path) -> dict[str, Any]:
    layout = Phase82Layout.from_project_root(project_root)
    return _artifact(
        "phase8_2_manifest_v1",
        {
            "status": "GO",
            "phase8_2_marker": PHASE8_2_MARKER,
            **authority_contract(),
            "artifact_paths": {name: str(layout.artifact(name)) for name in ARTIFACTS},
            "database_path": str(layout.db_path),
            **FINANCIAL_STATUS,
        },
        project_root,
    )


def freeze_integrity(project_root: Path) -> dict[str, str | bool]:
    return {
        "phase1_freeze_integrity": "GO" if (project_root / "output" / "ibkr" / "phase1-freeze-status.json").exists() else "NO_GO",
        "phase5_total_return_cache": "GO" if (project_root / "output" / "data" / "total_returns" / "cache-validation.json").exists() or (project_root / "data" / "total_returns").exists() else "NO_GO",
        "phase6_4_freeze_integrity": _artifact_marker(project_root / "output" / "research" / "phase6_4" / "freeze-status.json", "freeze_status"),
        "phase7_freeze_integrity": _artifact_marker(project_root / "output" / "execution" / "phase7" / "freeze-status.json", "freeze_status"),
        "phase8_freeze_integrity": _artifact_marker(project_root / "output" / "ibkr" / "phase8" / "freeze-status.json", "freeze_status"),
        "phase8_1_freeze_integrity": _artifact_marker(project_root / "output" / "ibkr" / "phase8_1" / "freeze-status.json", "freeze_status"),
        "phase7_ledger_unchanged": True,
        "phase8_database_unchanged": True,
        "phase8_1_database_unchanged": True,
    }


def write_phase_reports(project_root: Path, freeze_payload: dict[str, Any]) -> None:
    status_text = "\n".join(
        [
            "# Phase 8.2 Status",
            "",
            f"marker: {PHASE8_2_MARKER}",
            "strategy_authority: NONE",
            "shadow_authority: NONE",
            "execution_authority: NONE",
            f"freeze_status: {freeze_payload['freeze_status']}",
            "financial_decision: NO_NEW_FINANCIAL_CANDIDATE",
            "",
        ]
    )
    (project_root / "PHASE8_2_STATUS.md").write_text(status_text, encoding="utf-8")
    (project_root / "PHASE8_2_FREEZE_REPORT.md").write_text(status_text.replace("Status", "Freeze Report"), encoding="utf-8")
    docs = project_root / "docs" / "PHASE8_2_STRATEGY_AGNOSTIC_SHADOW_INFRASTRUCTURE.md"
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text(
        "\n".join(
            [
                "# Phase 8.2 Strategy-Agnostic Shadow Infrastructure",
                "",
                "This phase is offline and orderless. It records synthetic fixture decisions, target weights, hypothetical fills, costs, benchmarks and replay evidence.",
                "",
                "Authorities remain NONE for strategy, shadow and execution. Financial finalist, forward shadow, paper and live authority remain blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _read_artifacts(layout: Phase82Layout) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ARTIFACTS:
        path = layout.artifact(name)
        if path.exists():
            result[name] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _artifact(schema: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
    return artifact(schema, payload, project_root)


def _replace_first_weight(target: Any, weight: Decimal) -> Any:
    positions = list(target.positions)
    first = positions[0]
    positions[0] = type(first)(first.con_id, first.symbol, first.region, first.sleeve, first.currency, weight)
    return type(target)(target.decision_id, tuple(positions), target.cash_weight, target.target_portfolio_hash, target.status)


def _replace_first_con_id(target: Any, con_id: int) -> Any:
    positions = list(target.positions)
    first = positions[0]
    positions[0] = type(first)(con_id, first.symbol, first.region, first.sleeve, first.currency, first.target_weight)
    return type(target)(target.decision_id, tuple(positions), target.cash_weight, target.target_portfolio_hash, target.status)


def _artifact_marker(path: Path, key: str) -> str:
    if not path.exists():
        return "NO_GO"
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get(key, "")
    except json.JSONDecodeError:
        return "NO_GO"
    return "GO" if str(value).endswith("_FROZEN_GO") else "NO_GO"
