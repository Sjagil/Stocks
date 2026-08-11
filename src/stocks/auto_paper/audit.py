from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from stocks.auto_paper.authority import AuthorityDependencies, entry_authority, foundation_authority
from stocks.auto_paper.config import load_auto_paper_config
from stocks.auto_paper.contracts import (
    AssetGroup,
    AutoSignal,
    MarketQuote,
    PortfolioState,
    Regime,
    ShariahSnapshot,
    model_to_jsonable,
)
from stocks.auto_paper.entries import prepare_shadow_entry
from stocks.auto_paper.evaluation import financial_evaluation_fixture
from stocks.auto_paper.exits import EXIT_REASONS, EXIT_STATUSES, evaluate_risk_reducing_exit
from stocks.auto_paper.kill_switch import KILL_SWITCHES, evaluate_kill_switches
from stocks.auto_paper.movers_adapter import MOVER_TYPES, classify_candidate, phase11_1_adoption
from stocks.auto_paper.portfolio import BLOCKED_SLEEVES, SLEEVES, regime_allocation, risk_off_rotation_order, validate_portfolio_limits
from stocks.auto_paper.privacy import scan_public_artifacts
from stocks.auto_paper.reconciliation import reconcile_shadow_state
from stocks.auto_paper.replay import replay_fixture
from stocks.auto_paper.risk import EntryRiskContext, evaluate_entry_risk
from stocks.auto_paper.scheduler import run_bounded_scheduler
from stocks.auto_paper.shariah_gate import evaluate_shariah
from stocks.auto_paper.storage import AutoPaperStore, phase10_database_path
from stocks.auto_paper.strategies import REQUIREMENTS, STRATEGY_IDS, evaluate_strategy
from stocks.data.phase5_common import sha256_file, utc_now_iso
from stocks.execution.idempotency import stable_hash


PHASE10_MARKER = "PHASE10_AUTONOMOUS_SHARIAH_PAPER_TRADING_FOUNDATION_GO"
PHASE10_FREEZE_MARKER = "PHASE10_AUTONOMOUS_SHARIAH_PAPER_TRADING_FOUNDATION_FROZEN_GO"
FINANCIAL_STATUS = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "blocked",
    "PAPER_STRATEGY_AUTHORITY": "blocked",
    "LIVE_STRATEGY_AUTHORITY": "blocked",
    "financial_decision": "NO_ALPHA_CANDIDATE",
}
COUNTERS = {
    "paper_place_order_calls": 0,
    "paper_cancel_order_calls": 0,
    "live_place_order_calls": 0,
    "global_cancel_calls": 0,
    "request_order_id_calls": 0,
    "auto_bind_order_calls": 0,
    "exercise_option_calls": 0,
    "market_data_calls": 0,
    "historical_data_calls": 0,
    "automatic_submissions": 0,
    "automatic_cancellations": 0,
    "broker_writer_connections": 0,
}
ARTIFACTS = (
    "preflight.json",
    "canary-a-evidence-adoption.json",
    "shariah-gate-audit.json",
    "movers-integration-audit.json",
    "strategy-contract-audit.json",
    "portfolio-risk-audit.json",
    "automatic-entry-audit.json",
    "automatic-exit-audit.json",
    "scheduler-audit.json",
    "kill-switch-audit.json",
    "replay-audit.json",
    "financial-evaluation.json",
    "status.json",
    "freeze-status.json",
)
SOURCE_PATHS = [
    "main.py",
    ".env.ibkr.example",
    ".gitignore",
    *[f"src/stocks/auto_paper/{name}.py" for name in (
        "__init__",
        "audit",
        "authority",
        "config",
        "contracts",
        "entries",
        "evaluation",
        "exits",
        "kill_switch",
        "movers_adapter",
        "portfolio",
        "privacy",
        "reconciliation",
        "replay",
        "risk",
        "scheduler",
        "shariah_gate",
        "signal_registry",
        "storage",
        "strategies",
    )],
    "tests/test_phase10_auto_paper.py",
]
FROZEN_DEPENDENCIES = {
    "phase1": ("output/ibkr/phase1-freeze-status.json", "FROZEN_GO"),
    "phase5": ("output/ibkr/phase5-freeze-status.json", "IBKR_PHASE5_TOTAL_RETURN_AND_FX_FROZEN_GO"),
    "phase7": ("output/execution/phase7/freeze-status.json", "PHASE7_OFFLINE_EXECUTION_CONTROL_PLANE_FROZEN_GO"),
    "phase8": ("output/ibkr/phase8/freeze-status.json", "PHASE8_IBKR_READ_ONLY_RECONCILIATION_ADAPTER_FROZEN_GO"),
    "phase8_1": ("output/ibkr/phase8_1/freeze-status.json", "PHASE8_1_READ_ONLY_OBSERVATION_SOAK_AND_RECOVERY_FROZEN_GO"),
    "phase8_2": ("output/shadow/phase8_2/freeze-status.json", "PHASE8_2_STRATEGY_AGNOSTIC_SHADOW_INFRASTRUCTURE_FROZEN_GO"),
    "phase9_0_1": ("output/ibkr/phase9/phase9-0-1-freeze-status.json", "PHASE9_0_1_FILL_ADOPTION_AND_CLOSE_RECONCILIATION_FROZEN_GO"),
    "phase9_limits": ("output/ibkr/phase9/phase9-limit-semantics-freeze-status.json", "PHASE9_DAILY_ORDER_LIMIT_SEMANTICS_FROZEN_GO"),
    "phase9_canary_a": ("output/ibkr/phase9/canary-a-evidence-freeze-status.json", "PHASE9_CANARY_A_EVIDENCE_ADOPTION_FROZEN_GO"),
    "phase11_1": ("output/research/phase11_1/freeze-status.json", "PHASE11_1_ORTHOGONAL_PIT_DATA_AND_ALPHA_STRATEGY_FOUNDATION_FROZEN_GO"),
}
PRIVATE_DEPENDENCIES = {
    "phase7": "data/execution/phase7/execution_ledger.sqlite3",
    "phase8": "data/broker/phase8/private/broker_observation.sqlite3",
    "phase8_1": "data/broker/phase8_1/private/observation_soak.sqlite3",
    "phase8_2": "data/shadow/phase8_2/shadow_ledger.sqlite3",
    "phase9": "data/execution/phase9/private/paper_execution.sqlite3",
}
IMMUTABLE_PHASE9_ARTIFACTS = {
    "phase9_canary_a_evidence": "output/ibkr/phase9/canary-a-submit-cancel-evidence.json",
    "phase9_canary_a_freeze": "output/ibkr/phase9/canary-a-evidence-freeze-status.json",
    "phase9_status": "output/ibkr/phase9/status.json",
}
IMMUTABLE_PRIVATE_DEPENDENCY_NAMES = {"phase7"}
IMMUTABLE_PHASE9_ARTIFACT_NAMES = {"phase9_canary_a_freeze"}
EXPECTED_MUTABLE_PATHS = {
    ".env.ibkr.example",
    "main.py",
    "src/stocks/ibkr/paper_execution/audit.py",
    "src/stocks/ibkr/paper_execution/canary_a_evidence.py",
    "src/stocks/ibkr/paper_execution/executions.py",
    "src/stocks/ibkr/paper_execution/reconciliation.py",
    "src/stocks/ibkr/paper_execution/storage.py",
    "tests/test_phase9_fill_close_reconciliation.py",
    "tests/test_phase9_canary_a_evidence.py",
}


class Phase10Layout:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.output_dir = project_root / "output" / "ibkr" / "phase10"
        self.db_path = phase10_database_path(project_root)

    def artifact(self, name: str) -> Path:
        return self.output_dir / name


def phase10_command(project_root: Path, command: str, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    commands: dict[str, Callable[[], dict[str, Any]]] = {
        "preflight": lambda: preflight(project_root, env_file),
        "shariah-audit": lambda: shariah_audit(project_root, env_file),
        "movers-audit": lambda: movers_audit(project_root),
        "strategy-audit": lambda: strategy_audit(project_root),
        "portfolio-audit": lambda: portfolio_audit(project_root),
        "entry-audit": lambda: entry_audit(project_root, env_file),
        "exit-audit": lambda: exit_audit(project_root),
        "scheduler-audit": lambda: scheduler_audit(project_root, env_file),
        "kill-switch-audit": lambda: kill_switch_audit(project_root),
        "replay-audit": lambda: replay_audit(project_root),
        "financial-evaluation": lambda: financial_evaluation(project_root),
        "status": lambda: status(project_root),
        "freeze": lambda: freeze(project_root),
    }
    if command not in commands:
        raise ValueError(f"Unknown Phase 10 command: {command}")
    return commands[command]()


def preflight(project_root: Path, env_file: str | Path) -> dict[str, Any]:
    layout = _layout(project_root)
    config, errors = load_auto_paper_config(project_root, env_file)
    frozen = frozen_dependency_audit(project_root)
    canary = canary_a_adoption(project_root)
    private_hashes = {name: sha256_file(project_root / relative) for name, relative in PRIVATE_DEPENDENCIES.items()}
    phase9_artifact_hashes = {
        name: sha256_file(project_root / relative) for name, relative in IMMUTABLE_PHASE9_ARTIFACTS.items()
    }
    immutable_sources_present = all(private_hashes.values()) and all(phase9_artifact_hashes.values())
    go = not errors and frozen["status"] == "GO" and canary["status"] == "GO" and not config.enabled and immutable_sources_present
    payload = _artifact(
        "phase10_preflight_v1",
        {
            "status": "GO" if go else "NO_GO",
            "technical_preflight": "GO" if go else "NO_GO",
            "config": config.safe_dict(),
            "config_errors": errors,
            "auto_paper_default_disabled": not config.enabled,
            "frozen_dependency_integrity": frozen,
            "private_dependency_hashes_at_preflight": private_hashes,
            "immutable_phase9_artifact_hashes_at_preflight": phase9_artifact_hashes,
            "immutable_sources_present": immutable_sources_present,
            "canary_a_evidence_adoption_status": canary["status"],
            **foundation_authority(),
        },
    )
    _write(layout.artifact("preflight.json"), payload)
    return payload


def canary_a_adoption(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    source = project_root / "output" / "ibkr" / "phase9" / "canary-a-submit-cancel-evidence.json"
    data = _read(source)
    phase9_status_path = project_root / "output" / "ibkr" / "phase9" / "status.json"
    phase9_status = _read(phase9_status_path)
    phase9_state_go = (
        _phase9_progression_valid(phase9_status)
        and _content_hash_valid(phase9_status)
    )
    go = data.get("status") == "CANARY_A_EVIDENCE_GO" and _content_hash_valid(data) and phase9_state_go
    payload = _artifact(
        "phase10_canary_a_evidence_adoption_v1",
        {
            "status": "GO" if go else "NO_GO",
            "adoption_status": "PHASE9_CANARY_A_EVIDENCE_ADOPTED_READ_ONLY" if go else "PHASE9_CANARY_A_EVIDENCE_BLOCKED",
            "canary_marker": data.get("canary_marker"),
            "source_evidence_hash": data.get("evidence_hash"),
            "source_artifact_hash": sha256_file(source),
            "phase9_status_hash": sha256_file(phase9_status_path),
            "phase9_state_valid": phase9_state_go,
            "source_ledger_mutated": False,
            **foundation_authority(),
        },
    )
    _write(layout.artifact("canary-a-evidence-adoption.json"), payload)
    return payload


def _phase9_progression_valid(payload: dict[str, Any]) -> bool:
    checks = payload.get("checks", {})
    if (
        checks.get("submit_cancel_canary") is not True
        or payload.get("execution_authority") != "NONE"
        or payload.get("strategy_authority") != "NONE"
        or payload.get("shadow_authority") != "NONE"
        or payload.get("live_authority") != "NONE"
    ):
        return False
    fill_go = checks.get("fill_canary") is True
    close_go = checks.get("closing_sell_canary") is True
    blockers = payload.get("open_blockers")
    if payload.get("status") == "NO_GO":
        expected = (
            ["closing_sell_canary"]
            if fill_go and not close_go
            else ["fill_canary", "closing_sell_canary"]
            if not fill_go and not close_go
            else None
        )
        return expected is not None and blockers == expected
    return (
        payload.get("status")
        == "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO"
        and fill_go
        and close_go
        and blockers == []
    )


def shariah_audit(project_root: Path, env_file: str | Path) -> dict[str, Any]:
    layout = _layout(project_root)
    config, _ = load_auto_paper_config(project_root, env_file)
    now = datetime.now(UTC)
    stock = _shariah_snapshot(now)
    eligible_stock = evaluate_shariah(security_type="STK", asset_group=AssetGroup.SHARIAH_STOCK, product_id="STOCK", snapshot=stock, product_allowlist=config.product_allowlist, decision_time=now.isoformat())
    stale_stock = evaluate_shariah(security_type="STK", asset_group=AssetGroup.SHARIAH_STOCK, product_id="STOCK", snapshot=replace(stock, expires_at=(now - timedelta(seconds=1)).isoformat()), product_allowlist=(), decision_time=now.isoformat())
    etf_snapshot = replace(stock, product_structure="PHYSICAL_EQUITY_ETF", underlying_assets=("STOCK-BASKET",), shariah_certificate=True)
    etf = evaluate_shariah(security_type="STK", asset_group=AssetGroup.APPROVED_SHARIAH_EQUITY_ETF, product_id="ETF-FIXTURE", snapshot=etf_snapshot, product_allowlist=(), decision_time=now.isoformat())
    eligible_etf = evaluate_shariah(security_type="STK", asset_group=AssetGroup.APPROVED_SHARIAH_EQUITY_ETF, product_id="ETF-FIXTURE", snapshot=etf_snapshot, product_allowlist=("ETF-FIXTURE",), decision_time=now.isoformat())
    commodity = evaluate_shariah(security_type="STK", asset_group=AssetGroup.APPROVED_PHYSICAL_COMMODITY_PRODUCT, product_id="GOLD-FIXTURE", snapshot=replace(stock, product_structure="PHYSICAL_COMMODITY", underlying_assets=("PHYSICAL_GOLD",), physical_backing=True, shariah_certificate=True), product_allowlist=("GOLD-FIXTURE",), decision_time=now.isoformat())
    blocked_types = {security_type: evaluate_shariah(security_type=security_type, asset_group=AssetGroup.SHARIAH_STOCK, product_id="BLOCKED", snapshot=stock, product_allowlist=(), decision_time=now.isoformat())["status"] for security_type in ("FUT", "OPT", "BOND", "CFD", "SWAP")}
    go = eligible_stock["status"] == "SHARIAH_ELIGIBLE" and stale_stock["status"] == "SHARIAH_STATUS_STALE" and etf["status"] == "SHARIAH_MANUAL_REVIEW_REQUIRED" and eligible_etf["status"] == "SHARIAH_ELIGIBLE" and commodity["status"] == "SHARIAH_ELIGIBLE" and all(value == "SHARIAH_PRODUCT_STRUCTURE_BLOCKED" for value in blocked_types.values())
    payload = _artifact("phase10_shariah_gate_audit_v1", {"status": "GO" if go else "NO_GO", "stock_gate": eligible_stock, "stale_gate": stale_stock, "etf_allowlist_gate": etf, "eligible_etf_gate": eligible_etf, "physical_commodity_gate": commodity, "blocked_security_types": blocked_types, **foundation_authority()})
    _write(layout.artifact("shariah-gate-audit.json"), payload)
    return payload


def movers_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    adoption = phase11_1_adoption(project_root)
    accepted = [classify_candidate({"mover_type": mover, "shariah_eligible": True, "liquid": True, "news_attributed": True, "fundamentals_available": True, "technical_acceptance": True, "event_cluster_known": True, "unexplained_pump": False, "value_trap": False, "permanent_impairment": False}) for mover in MOVER_TYPES]
    rejected_pump = classify_candidate({"mover_type": "TOP_GAINERS", "shariah_eligible": True, "liquid": True, "news_attributed": False, "fundamentals_available": True, "technical_acceptance": True, "event_cluster_known": True, "unexplained_pump": True})
    go = adoption["status"] == "GO" and all(row["status"] == "MOVER_CANDIDATE_ACCEPTED" for row in accepted) and rejected_pump["status"] == "MOVER_CANDIDATE_REJECTED"
    payload = _artifact("phase10_movers_integration_audit_v1", {"status": "GO" if go else "NO_GO", "phase11_1_adoption": adoption, "mover_types": list(MOVER_TYPES), "accepted_fixture_count": len(accepted), "pump_rejection": rejected_pump, "source_modified": False, **foundation_authority()})
    _write(layout.artifact("movers-integration-audit.json"), payload)
    return payload


def strategy_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    decisions = [evaluate_strategy(strategy_id, {name: True for name in REQUIREMENTS[strategy_id]}) for strategy_id in STRATEGY_IDS]
    go = all(row["status"] == "SHADOW_CANDIDATE" and row["automatic_paper_eligibility"] is False for row in decisions)
    payload = _artifact("phase10_strategy_contract_audit_v1", {"status": "GO" if go else "NO_GO", "strategy_ids": list(STRATEGY_IDS), "decisions": decisions, "optimizer_used": False, "machine_learning_used": False, **foundation_authority()})
    _write(layout.artifact("strategy-contract-audit.json"), payload)
    return payload


def portfolio_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    allocations = {regime.value: {key: str(value) for key, value in regime_allocation(regime).items()} for regime in Regime}
    sums_go = all(sum(Decimal(value) for value in row.values()) == Decimal("1") for row in allocations.values())
    valid = validate_portfolio_limits(position_weight_pct=Decimal("8"), sector_exposure_pct=Decimal("25"), event_cluster_exposure_pct=Decimal("15"), fallen_angel_combined_pct=Decimal("10"), cash_pct=Decimal("5"), shariah_eligible=True)
    starter = validate_portfolio_limits(position_weight_pct=Decimal("2.5"), sector_exposure_pct=Decimal("5"), event_cluster_exposure_pct=Decimal("5"), fallen_angel_combined_pct=Decimal("5"), cash_pct=Decimal("3"), shariah_eligible=True, starter_position=True)
    blocked = validate_portfolio_limits(position_weight_pct=Decimal("11"), sector_exposure_pct=Decimal("26"), event_cluster_exposure_pct=Decimal("16"), fallen_angel_combined_pct=Decimal("11"), cash_pct=Decimal("11"), shariah_eligible=True, high_conviction=True)
    shariah_blocked = validate_portfolio_limits(position_weight_pct=Decimal("3"), sector_exposure_pct=Decimal("5"), event_cluster_exposure_pct=Decimal("5"), fallen_angel_combined_pct=Decimal("5"), cash_pct=Decimal("3"), shariah_eligible=False)
    shariah_blockers = shariah_blocked.get("blockers")
    go = sums_go and valid["status"] == "PORTFOLIO_RISK_GO" and starter["status"] == "PORTFOLIO_RISK_GO" and blocked["status"] == "PORTFOLIO_RISK_BLOCKED" and isinstance(shariah_blockers, list) and "SHARIAH_PORTFOLIO_GATE_BLOCKED" in shariah_blockers
    payload = _artifact("phase10_portfolio_risk_audit_v1", {"status": "GO" if go else "NO_GO", "sleeves": list(SLEEVES), "blocked_sleeves": list(BLOCKED_SLEEVES), "regime_allocations": allocations, "risk_off_rotation_order": list(risk_off_rotation_order()), "limit_fixture": valid, "starter_fixture": starter, "blocked_fixture": blocked, "shariah_blocked_fixture": shariah_blocked, "bonds_used": False, **foundation_authority()})
    _write(layout.artifact("portfolio-risk-audit.json"), payload)
    return payload


def entry_audit(project_root: Path, env_file: str | Path) -> dict[str, Any]:
    layout = _layout(project_root)
    config, _ = load_auto_paper_config(project_root, env_file)
    config = replace(config, strategy_allowlist=("QUALITY_GAINER_CONTINUATION_V1",))
    store = AutoPaperStore(layout.db_path)
    store.initialize()
    now = datetime.now(UTC)
    run_id = stable_hash(now.isoformat())[:16]
    signal = _signal(now, signal_id=f"SYNTHETIC-{run_id}")
    portfolio = _empty_portfolio()
    context = _entry_context(now)
    risk = evaluate_entry_risk(signal, portfolio, config, context)
    first = prepare_shadow_entry(store, signal, account_fingerprint="SYNTHETIC-ACCOUNT-FIXTURE", risk=risk)
    duplicate = prepare_shadow_entry(store, signal, account_fingerprint="SYNTHETIC-ACCOUNT-FIXTURE", risk=risk)
    authority = entry_authority(config, AuthorityDependencies(False, False, False, True, True, True, True, True, True, True), signal.strategy_id)
    go = risk["status"] == "ENTRY_RISK_GO" and first.decision_code == "SIGNAL_SHADOW_ONLY" and duplicate.decision_code == "SIGNAL_DUPLICATE" and authority["entry_authority"] == "NONE"
    payload = _artifact("phase10_automatic_entry_audit_v1", {"status": "GO" if go else "NO_GO", "valid_synthetic_buy": model_to_jsonable(first), "duplicate_buy": model_to_jsonable(duplicate), "runtime_authority_check": authority, "store_counts": store.counts(), "broker_order_constructed": False, **foundation_authority()})
    _write(layout.artifact("automatic-entry-audit.json"), payload)
    return payload


def exit_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    allowed = evaluate_risk_reducing_exit(con_id=1, local_con_id=1, broker_con_id=1, sell_quantity=Decimal("1"), local_quantity=Decimal("1"), broker_quantity=Decimal("1"), account_match=True, snapshot_complete=True, reason="HARD_STOP_LOSS", limit_price=Decimal("49"), entries_today=999)
    exceeds = evaluate_risk_reducing_exit(con_id=1, local_con_id=1, broker_con_id=1, sell_quantity=Decimal("2"), local_quantity=Decimal("1"), broker_quantity=Decimal("1"), account_match=True, snapshot_complete=True, reason="ATR_STOP", limit_price=Decimal("49"))
    mismatch = evaluate_risk_reducing_exit(con_id=1, local_con_id=1, broker_con_id=2, sell_quantity=Decimal("1"), local_quantity=Decimal("1"), broker_quantity=Decimal("1"), account_match=True, snapshot_complete=True, reason="THESIS_INVALIDATED", limit_price=Decimal("49"))
    incomplete = evaluate_risk_reducing_exit(con_id=1, local_con_id=1, broker_con_id=1, sell_quantity=Decimal("1"), local_quantity=Decimal("1"), broker_quantity=Decimal("1"), account_match=True, snapshot_complete=False, reason="DATA_INTEGRITY_EMERGENCY", limit_price=Decimal("49"))
    go = allowed["status"] == "EXIT_RISK_REDUCING_ALLOWED" and exceeds["status"] == "SELL_EXCEEDS_RECONCILED_POSITION" and mismatch["status"] == "EXIT_BLOCKED_POSITION_MISMATCH" and incomplete["status"] == "EXIT_BLOCKED_SNAPSHOT_INCOMPLETE"
    payload = _artifact("phase10_automatic_exit_audit_v1", {"status": "GO" if go else "NO_GO", "exit_reasons": list(EXIT_REASONS), "exit_status_contract": list(EXIT_STATUSES), "risk_reducing_sell": allowed, "oversell_block": exceeds, "position_mismatch_block": mismatch, "snapshot_incomplete_block": incomplete, "entry_limit_independent": allowed.get("entry_count_ignored") == 999, **foundation_authority()})
    _write(layout.artifact("automatic-exit-audit.json"), payload)
    return payload


def scheduler_audit(project_root: Path, env_file: str | Path) -> dict[str, Any]:
    layout = _layout(project_root)
    config, _ = load_auto_paper_config(project_root, env_file)
    result = run_bounded_scheduler(start_time=datetime.now(UTC).isoformat(), interval_seconds=config.scheduler_interval_seconds, max_iterations=6)
    go = result["status"] == "SCHEDULER_BOUNDED_GO" and result["bounded"] and not result["busy_loop"]
    payload = _artifact(
        "phase10_scheduler_audit_v1",
        {**result, "scheduler_status": result["status"], "status": "GO" if go else "NO_GO", **foundation_authority()},
    )
    _write(layout.artifact("scheduler-audit.json"), payload)
    return payload


def kill_switch_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    clear = evaluate_kill_switches({})
    individual = {name: evaluate_kill_switches({name: True})["status"] for name in KILL_SWITCHES}
    unknown_order = reconcile_shadow_state(local_quantity=Decimal("0"), broker_quantity=Decimal("0"), unknown_order_count=1)
    unknown_execution = reconcile_shadow_state(local_quantity=Decimal("0"), broker_quantity=Decimal("0"), unknown_execution_count=1)
    commission_pending = reconcile_shadow_state(local_quantity=Decimal("0"), broker_quantity=Decimal("0"), commission_pending_count=1)
    go = clear["status"] == "KILL_SWITCH_CLEAR" and all(value == "KILL_SWITCH_ACTIVE" for value in individual.values()) and not unknown_order["new_entries_allowed"] and not unknown_execution["new_entries_allowed"] and not commission_pending["new_entries_allowed"]
    payload = _artifact("phase10_kill_switch_audit_v1", {"status": "GO" if go else "NO_GO", "kill_switches": list(KILL_SWITCHES), "individual_fixture_statuses": individual, "unknown_order": unknown_order, "unknown_execution": unknown_execution, "commission_pending": commission_pending, "clear_fixture": clear, **foundation_authority()})
    _write(layout.artifact("kill-switch-audit.json"), payload)
    return payload


def replay_audit(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    store = AutoPaperStore(layout.db_path)
    store.initialize()
    run_id = stable_hash(datetime.now(UTC).isoformat())[:16]
    result = replay_fixture(store, run_id=run_id)
    payload = _artifact(
        "phase10_replay_audit_v1",
        {
            **result,
            "replay_status": result["status"],
            "status": "GO" if result["status"] == "REPLAY_GO" else "NO_GO",
            "store_counts": store.counts(),
            **foundation_authority(),
        },
    )
    _write(layout.artifact("replay-audit.json"), payload)
    return payload


def financial_evaluation(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    result = financial_evaluation_fixture()
    payload = _artifact(
        "phase10_financial_evaluation_v1",
        {
            **result,
            "evaluation_status": result["status"],
            "status": "GO" if result["status"] == "FINANCIAL_EVALUATION_CONTRACT_GO" else "NO_GO",
            **foundation_authority(),
        },
    )
    _write(layout.artifact("financial-evaluation.json"), payload)
    return payload


def status(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    required = [name for name in ARTIFACTS if name not in {"status.json", "freeze-status.json"}]
    statuses = {name: _read(layout.artifact(name)).get("status") == "GO" for name in required}
    privacy = scan_public_artifacts(layout.output_dir)
    technical_go = all(statuses.values()) and privacy["status"] == "PRIVACY_GO"
    payload = _artifact("phase10_status_v1", {"status": PHASE10_MARKER if technical_go else "NO_GO", "phase10_marker": PHASE10_MARKER, "artifact_checks": statuses, "privacy": privacy, "technical_readiness": "GO" if technical_go else "NO_GO", "open_runtime_blockers": foundation_authority()["runtime_blockers"], **foundation_authority()})
    _write(layout.artifact("status.json"), payload)
    return payload


def freeze(project_root: Path) -> dict[str, Any]:
    layout = _layout(project_root)
    status_payload = status(project_root)
    preflight_payload = _read(layout.artifact("preflight.json"))
    frozen = frozen_dependency_audit(project_root)
    before = preflight_payload.get("private_dependency_hashes_at_preflight", {})
    after = {name: sha256_file(project_root / relative) for name, relative in PRIVATE_DEPENDENCIES.items()}
    private_unchanged = _selected_hashes_unchanged(
        before,
        after,
        IMMUTABLE_PRIVATE_DEPENDENCY_NAMES,
    )
    append_only_private_present = all(after.values())
    phase9_before = preflight_payload.get("immutable_phase9_artifact_hashes_at_preflight", {})
    phase9_after = {name: sha256_file(project_root / relative) for name, relative in IMMUTABLE_PHASE9_ARTIFACTS.items()}
    phase9_unchanged = _selected_hashes_unchanged(
        phase9_before,
        phase9_after,
        IMMUTABLE_PHASE9_ARTIFACT_NAMES,
    )
    versioned_phase9_present = all(phase9_after.values())
    current_canary = canary_a_adoption(project_root)
    go = (
        status_payload["status"] == PHASE10_MARKER
        and frozen["status"] == "GO"
        and private_unchanged
        and append_only_private_present
        and phase9_unchanged
        and versioned_phase9_present
        and current_canary["status"] == "GO"
    )
    payload = _artifact(
        "phase10_freeze_status_v1",
        {
            "freeze_status": PHASE10_FREEZE_MARKER if go else "NO_GO",
            "phase10_status": status_payload["status"],
            "technical_readiness": "GO" if go else "NO_GO",
            "frozen_dependency_integrity": frozen,
            "private_dependency_hashes_unchanged": private_unchanged,
            "immutable_private_dependency_names": sorted(
                IMMUTABLE_PRIVATE_DEPENDENCY_NAMES
            ),
            "append_only_private_dependencies_present": (
                append_only_private_present
            ),
            "private_dependency_hashes": after,
            "immutable_phase9_artifact_hashes_unchanged": phase9_unchanged,
            "immutable_phase9_artifact_names": sorted(
                IMMUTABLE_PHASE9_ARTIFACT_NAMES
            ),
            "versioned_phase9_artifacts_present": (
                versioned_phase9_present
            ),
            "current_canary_a_adoption_status": current_canary["status"],
            "immutable_phase9_artifact_hashes": phase9_after,
            "source_hashes": {path: sha256_file(project_root / path) for path in SOURCE_PATHS},
            "artifact_hashes": {name: sha256_file(layout.artifact(name)) for name in ARTIFACTS if name != "freeze-status.json" and layout.artifact(name).exists()},
            "phase10_private_database_hash": sha256_file(layout.db_path),
            **foundation_authority(),
        },
    )
    _write(layout.artifact("freeze-status.json"), payload)
    return payload


def _selected_hashes_unchanged(
    before: dict[str, Any],
    after: dict[str, Any],
    names: set[str],
) -> bool:
    return bool(names) and all(
        before.get(name) is not None
        and before.get(name) == after.get(name)
        for name in names
    )


def frozen_dependency_audit(project_root: Path) -> dict[str, Any]:
    markers = {}
    unexpected = []
    expected_mutable = []
    artifact_hashes = {}
    for name, (relative, expected_marker) in FROZEN_DEPENDENCIES.items():
        path = project_root / relative
        data = _read(path)
        marker = next((data.get(key) for key in ("freeze_status", "status") if data.get(key) is not None), None)
        markers[name] = {"expected": expected_marker, "observed": marker, "go": marker == expected_marker}
        artifact_hashes[name] = sha256_file(path)
        hashes = data.get("source_hashes") or data.get("immutable_phase1_service_hashes") or {}
        for source, expected_hash in hashes.items():
            source_path = project_root / source
            if expected_hash and source_path.exists() and sha256_file(source_path) != expected_hash:
                item = f"{name}:{source}"
                if source in EXPECTED_MUTABLE_PATHS:
                    expected_mutable.append(item)
                else:
                    unexpected.append(item)
    go = all(item["go"] for item in markers.values()) and not unexpected
    return {"status": "GO" if go else "NO_GO", "markers": markers, "unexpected_immutable_mismatches": unexpected, "expected_versioned_mismatches": sorted(expected_mutable), "freeze_artifact_hashes": artifact_hashes}


def _layout(project_root: Path) -> Phase10Layout:
    layout = Phase10Layout(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    return layout


def _artifact(schema: str, fields: dict[str, Any]) -> dict[str, Any]:
    payload = {"schema": schema, "generated_at": utc_now_iso(), **COUNTERS, **fields, **FINANCIAL_STATUS}
    payload["content_hash"] = stable_hash({key: value for key, value in payload.items() if key != "content_hash"})
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    return payload.get("content_hash") == stable_hash({key: value for key, value in payload.items() if key != "content_hash"})


def _shariah_snapshot(now: datetime) -> ShariahSnapshot:
    return ShariahSnapshot("SHARIAH_ELIGIBLE", "AAOIFI_FIXTURE", "1.0", now.isoformat(), (now - timedelta(days=1)).isoformat(), (now + timedelta(days=1)).isoformat(), True, True, True, "OPERATING_COMPANY")


def _signal(now: datetime, *, signal_id: str) -> AutoSignal:
    provenance = {"fixture": stable_hash("fixture")}
    return AutoSignal(signal_id, "QUALITY_GAINER_CONTINUATION_V1", "1.0", now.isoformat(), now.isoformat(), (now + timedelta(minutes=5)).isoformat(), now.date().isoformat(), 1, "SYNTH", "SMART", "EUR", "STK", AssetGroup.SHARIAH_STOCK, "BUY", Decimal("1"), Decimal("40"), Decimal("41"), "synthetic offline fixture", None, Decimal("0.8"), "5d", provenance, stable_hash(provenance), stable_hash("features"), stable_hash("portfolio"), stable_hash("shariah"))


def _empty_portfolio() -> PortfolioState:
    return PortfolioState(Decimal("100"), Decimal("0"), Decimal("0"), (), {}, {}, "PAPER_RECONCILED_EMPTY", True)


def _entry_context(now: datetime) -> EntryRiskContext:
    return EntryRiskContext(now.isoformat(), True, False, "SHARIAH_ELIGIBLE", True, True, True, MarketQuote(Decimal("39.99"), Decimal("40.01"), now.isoformat()), False, False, 0, "TECHNOLOGY", "EARNINGS", True)
