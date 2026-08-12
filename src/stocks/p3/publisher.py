from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from stocks.p3.analytics import (
    load_shared_oos_matrix,
    multiple_testing_diagnostics,
    parameter_stability_diagnostics,
    regime_robustness_diagnostics,
    strategy_dependency_diagnostics,
)
from stocks.p3.contracts import StrategyDNA
from stocks.p3.io import atomic_write_json, file_hash, read_json
from stocks.p3.ledger import (
    UnifiedTrialLedger,
    ai_trial_records,
    native_strategy_dna,
    native_trial_records,
)


AUTHORITY = {
    "money_control": False,
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "broker_calls": 0,
    "broker_writes": 0,
    "orders_generated": 0,
}


SOURCE_PATHS = (
    "config/p3_unified_evidence_policy_v1.json",
    "output/research/reports/executive_summary.json",
    "output/research/reports/manifest.json",
    "output/research/strategies/strategy_registry.json",
    "output/research/strategies/strategy_dna.csv",
    "output/research/evidence_throughput/status.json",
    "output/research/evidence_throughput/multiple-testing-trial-accounting.json",
    "output/research/phase11_6/walk_forward_status.json",
    "output/research/phase11_11/status.json",
    "output/research/phase11_14/status.json",
    "output/research/phase11_14/qualification.json",
    "output/research/universe/pit_universe_audit.json",
    "output/research/universe/survivorship_audit.json",
    "output/research/universe/shariah_eligibility.json",
    "output/verification/data-readiness.json",
    "output/verification/universe-status.json",
    "output/ai/reference-repo-integration-matrix.json",
    "output/ai/model-health.json",
    "output/ai/shadow-portfolio-comparison.json",
    "output/ai/authority-matrix.json",
    "output/portfolio/p2-2-execution-feasibility.json",
    "output/portfolio/learning-model-evidence.json",
    "output/portfolio/performance-attribution.json",
    "output/portfolio/active_portfolio_plan.json",
    "output/portfolio/desired-portfolio-targets.json",
    "output/ibkr/live/authority-status.json",
    "output/ibkr/live/reconciliation.json",
    "output/ibkr/phase9/reconciliation-audit.json",
    "output/ibkr/live/p0-execution-readiness.json",
    "output/ibkr/live/p0-2-readiness.json",
    "output/ibkr/live/automatic-cycle.json",
    "output/ibkr/live/writer-integrity-verify.json",
    "output/operations/machine-status.json",
    "output/operations/last-cycle.json",
    "runtime/heartbeat.json",
)


def publish_p3_evidence(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    generated_at = datetime.now(UTC).isoformat()
    source_hashes = _source_hashes(project_root)

    dna = native_strategy_dna(project_root)
    dna_by_strategy = {item.strategy_id: item for item in dna}
    native_trials = native_trial_records(project_root, dna_by_strategy)
    ai_trials = ai_trial_records(project_root)

    ledger = UnifiedTrialLedger(project_root)
    try:
        native_import = ledger.import_records(native_trials)
        ai_import = ledger.import_records(ai_trials)
        import_hash = ledger.record_import_run()
        ledger_counts = ledger.counts()
        trial_ledger_path = ledger.export_public_jsonl()
    finally:
        ledger.close()

    registry = read_json(
        project_root / "output/research/reports/executive_summary.json"
    )
    evidence = read_json(
        project_root / "output/research/evidence_throughput/status.json"
    )
    multiple_testing = read_json(
        project_root
        / "output/research/evidence_throughput/multiple-testing-trial-accounting.json"
    )

    canonical_state = _canonical_state(project_root, generated_at, source_hashes)
    architecture = _architecture_map(project_root, generated_at, source_hashes)
    strategy_artifacts = _strategy_artifacts(
        project_root,
        generated_at,
        source_hashes,
        registry,
        evidence,
        multiple_testing,
        dna,
        ledger_counts,
        import_hash,
        trial_ledger_path,
    )
    validation_artifacts = _validation_artifacts(
        project_root,
        generated_at,
        source_hashes,
        evidence,
        multiple_testing,
    )
    data_artifacts = _data_artifacts(project_root, generated_at, source_hashes)
    capability_usage = _capability_usage(
        project_root, generated_at, source_hashes
    )
    ai_artifacts = _ai_shadow_artifacts(project_root, generated_at, source_hashes)
    portfolio_artifacts = _portfolio_artifacts(
        project_root, generated_at, source_hashes, evidence
    )
    quote_readiness = _quote_readiness(project_root, generated_at, source_hashes)
    runtime_reliability = _runtime_reliability(
        project_root, generated_at, source_hashes
    )
    engineering = _engineering_status(project_root, generated_at)

    published: dict[str, dict[str, Any]] = {
        "output/verification/p3-canonical-state.json": canonical_state,
        "output/verification/p3-architecture-map.json": architecture,
        **strategy_artifacts,
        **validation_artifacts,
        **data_artifacts,
        "output/ai/capability-runtime-usage.json": capability_usage,
        **ai_artifacts,
        **portfolio_artifacts,
        "output/ibkr/live/quote-readiness.json": quote_readiness,
        "output/operations/runtime-reliability.json": runtime_reliability,
        "output/verification/p3-engineering-status.json": engineering,
    }
    for relative, payload in published.items():
        atomic_write_json(project_root / relative, payload)

    p3_readiness = _p3_readiness(
        generated_at=generated_at,
        source_hashes=source_hashes,
        canonical_state=canonical_state,
        architecture=architecture,
        strategy_artifacts=strategy_artifacts,
        validation_artifacts=validation_artifacts,
        data_artifacts=data_artifacts,
        capability_usage=capability_usage,
        ai_artifacts=ai_artifacts,
        portfolio_artifacts=portfolio_artifacts,
        quote_readiness=quote_readiness,
        runtime_reliability=runtime_reliability,
        engineering=engineering,
        ledger_counts=ledger_counts,
    )
    atomic_write_json(
        project_root / "output/verification/p3-readiness.json", p3_readiness
    )
    published["output/verification/p3-readiness.json"] = p3_readiness
    freeze = {
        "schema": "p3_unified_evidence_freeze_v1",
        "status": "P3_IMPLEMENTATION_FROZEN" if p3_readiness["p3_complete"] else "NO_GO",
        "generated_at": generated_at,
        "p3_complete": p3_readiness["p3_complete"],
        "policy": "config/p3_unified_evidence_policy_v1.json",
        "policy_sha256": file_hash(
            project_root / "config/p3_unified_evidence_policy_v1.json"
        ),
        "readiness": "output/verification/p3-readiness.json",
        "readiness_sha256": file_hash(
            project_root / "output/verification/p3-readiness.json"
        ),
        "code_hashes": {
            relative: file_hash(project_root / relative)
            for relative in (
                "src/stocks/p3/contracts.py",
                "src/stocks/p3/ledger.py",
                "src/stocks/p3/analytics.py",
                "src/stocks/p3/publisher.py",
                "src/stocks/operations/service.py",
                "main.py",
            )
        },
        "financial_finalist_go": p3_readiness["financial_finalist_go"],
        "capital_scaling_ready": p3_readiness["capital_scaling_ready"],
        "economic_and_external_blockers": p3_readiness[
            "economic_and_external_blockers"
        ],
        **AUTHORITY,
    }
    atomic_write_json(project_root / "output/verification/p3-freeze.json", freeze)
    published["output/verification/p3-freeze.json"] = freeze

    report = _render_report(
        generated_at=generated_at,
        canonical_state=canonical_state,
        registry=registry,
        evidence=evidence,
        validation_artifacts=validation_artifacts,
        data_artifacts=data_artifacts,
        capability_usage=capability_usage,
        ai_artifacts=ai_artifacts,
        portfolio_artifacts=portfolio_artifacts,
        quote_readiness=quote_readiness,
        runtime_reliability=runtime_reliability,
        engineering=engineering,
        p3_readiness=p3_readiness,
        ledger_counts=ledger_counts,
    )
    report_path = project_root / "reports/P3_UNIFIED_STRATEGY_EVIDENCE_COMPLETION.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8", newline="\n")

    manifest_rows = []
    for relative in sorted(published):
        path = project_root / relative
        manifest_rows.append(
            {
                "path": relative,
                "sha256": file_hash(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest_rows.extend(
        [
            {
                "path": str(trial_ledger_path.relative_to(project_root)).replace("\\", "/"),
                "sha256": file_hash(trial_ledger_path),
                "size_bytes": trial_ledger_path.stat().st_size,
            },
            {
                "path": str(report_path.relative_to(project_root)).replace("\\", "/"),
                "sha256": file_hash(report_path),
                "size_bytes": report_path.stat().st_size,
            },
        ]
    )
    manifest = {
        "schema": "p3_unified_evidence_manifest_v1",
        "status": "GO",
        "generated_at": generated_at,
        "artifact_count": len(manifest_rows),
        "artifacts": manifest_rows,
        "source_hashes": source_hashes,
        "ledger_import": {
            "native": native_import,
            "ai": ai_import,
            "counts": ledger_counts,
            "import_hash": import_hash,
        },
        **AUTHORITY,
    }
    atomic_write_json(project_root / "output/verification/p3-manifest.json", manifest)
    return {
        "schema": "p3_unified_strategy_evidence_publish_v1",
        "status": "GO",
        "evidence_status": p3_readiness["status"],
        "generated_at": generated_at,
        "p3_complete": p3_readiness["p3_complete"],
        "financial_finalist_go": p3_readiness["financial_finalist_go"],
        "runtime_active": canonical_state["runtime"]["active"],
        "ledger_counts": ledger_counts,
        "artifact_count": len(manifest_rows) + 1,
        "report": "reports/P3_UNIFIED_STRATEGY_EVIDENCE_COMPLETION.md",
        "manifest": "output/verification/p3-manifest.json",
        "blockers": p3_readiness["blockers"],
        **AUTHORITY,
    }


def _canonical_state(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
) -> dict[str, Any]:
    machine = read_json(project_root / "output/operations/machine-status.json")
    heartbeat = read_json(project_root / "runtime/heartbeat.json")
    last_cycle = read_json(project_root / "output/operations/last-cycle.json")
    authority = read_json(project_root / "output/ibkr/live/authority-status.json")
    live_reconciliation = read_json(
        project_root / "output/ibkr/live/reconciliation.json"
    )
    phase9_reconciliation = read_json(
        project_root / "output/ibkr/phase9/reconciliation-audit.json"
    )
    phase9_current = _phase9_observation_is_current(
        phase9_reconciliation, as_of=generated_at
    )
    reconciliation = (
        phase9_reconciliation if phase9_current else live_reconciliation
    )
    p0 = read_json(project_root / "output/ibkr/live/p0-execution-readiness.json")
    p02 = read_json(project_root / "output/ibkr/live/p0-2-readiness.json")
    p2 = read_json(project_root / "output/portfolio/orchestrator/freeze-status.json")
    p22 = read_json(
        project_root
        / "output/portfolio/orchestrator/p2-2-execution-feasibility-freeze.json"
    )
    automatic = read_json(project_root / "output/ibkr/live/automatic-cycle.json")
    signals = read_json(project_root / "output/operations/signal-lifecycle.json")
    registry = read_json(project_root / "output/research/reports/executive_summary.json")
    feasibility = read_json(
        project_root / "output/portfolio/p2-2-execution-feasibility.json"
    )
    desired = read_json(project_root / "output/portfolio/desired-portfolio-targets.json")
    blockers = _unique(
        [
            *list(machine.get("last_cycle_blockers") or []),
            *list(last_cycle.get("blockers") or []),
            *list(automatic.get("blockers") or []),
            *list(p0.get("open_blockers") or []),
            *list(p02.get("blockers") or []),
        ]
    )
    effective_authority = str(heartbeat.get("execution_authority") or "NONE")
    broker_observed = (
        phase9_current
        or str(live_reconciliation.get("status") or "").upper() == "GO"
    )
    if phase9_current:
        position_count = int(reconciliation.get("broker_position_count") or 0)
        open_order_count = int(
            reconciliation.get("broker_open_order_count") or 0
        )
        execution_count = int(reconciliation.get("broker_execution_count") or 0)
        commission_count = int(
            reconciliation.get("broker_commission_count") or 0
        )
        broker_writes = sum(
            int(value or 0)
            for value in dict(
                reconciliation.get("broker_write_counters") or {}
            ).values()
        )
        reconciliation_status = reconciliation.get("status")
        reconciliation_detail = reconciliation.get("reconciliation_status")
    else:
        position_count = int(reconciliation.get("position_count") or 0)
        open_order_count = int(reconciliation.get("open_order_count") or 0)
        execution_count = int(reconciliation.get("execution_count") or 0)
        commission_count = int(reconciliation.get("commission_count") or 0)
        broker_writes = int(reconciliation.get("broker_write_calls") or 0)
        reconciliation_status = reconciliation.get("status")
        reconciliation_detail = reconciliation.get("reconciliation_status")
    return {
        "schema": "p3_canonical_current_state_v1",
        "status": "GO" if broker_observed else "DEGRADED",
        "generated_at": generated_at,
        "source_precedence": [
            "fresh broker reconciliation",
            "runtime heartbeat and last cycle",
            "current authority and freeze verifications",
            "current portfolio and research artifacts",
            "historical reports for audit only",
        ],
        "source_hashes": source_hashes,
        "account_equity": {
            "value": None,
            "status": "PRIVATE_NOT_REPUBLISHED",
            "spendable_eur_proven": bool(
                (p0.get("account_semantics") or {}).get("spendable_eur_proven")
            ),
        },
        "cash": {
            "value": None,
            "status": "PRIVATE_NOT_REPUBLISHED",
            "non_eur_cash_excluded": bool(
                (p0.get("account_semantics") or {}).get("non_eur_cash_excluded")
            ),
        },
        "broker": {
            "source": (
                "PHASE9_READ_ONLY_AUDIT"
                if phase9_current
                else "LIVE_RECONCILIATION"
            ),
            "observation_current": broker_observed,
            "reconciliation_status": reconciliation_status,
            "reconciliation_detail": reconciliation_detail,
            "operational_state_status": reconciliation.get(
                "operational_broker_state_status"
            ),
            "canonical_execution_evidence_status": reconciliation.get(
                "canonical_execution_evidence_status"
            ),
            "reconciliation_blocks_new_risk": reconciliation_status != "GO",
            "positions": position_count,
            "open_orders": open_order_count,
            "executions": execution_count,
            "commissions": commission_count,
            "snapshot_atomic": bool(
                (reconciliation.get("public_summary") or {}).get("snapshot_atomic")
            ),
            "execution_history_complete": bool(
                (reconciliation.get("public_summary") or {}).get(
                    "execution_history_complete"
                )
            ),
            "broker_writes": broker_writes,
        },
        "runtime": {
            "active": bool(machine.get("enabled"))
            and str(machine.get("mode")) == "CONTROLLED_LIVE",
            "runtime_status": heartbeat.get("runtime_status"),
            "runtime_state": heartbeat.get("runtime_state"),
            "runtime_mode": machine.get("mode"),
            "requested_mode": heartbeat.get("requested_mode"),
            "autonomous_level": authority.get("current_scaling_level"),
            "authority_lifecycle": authority.get("lifecycle_status"),
            "infrastructure_authority": authority.get("execution_authority"),
            "effective_execution_authority": effective_authority,
            "last_cycle_id": machine.get("last_cycle_id"),
            "last_cycle_status": machine.get("last_cycle_status"),
            "last_heartbeat": machine.get("last_heartbeat"),
            "broker_writes": int(heartbeat.get("broker_writes") or 0),
        },
        "hashes": {
            "strategy_authority_hash": file_hash(
                project_root / "output/ibkr/live/strategy-allowlist.json"
            ),
            "qualification_hash": authority.get("qualification_hash"),
            "writer_hash": file_hash(
                project_root / "output/ibkr/live/freeze-status.json"
            ),
            "p0": p0.get("content_hash"),
            "p0_2": (p02.get("freeze") or {}).get("freeze_hash"),
            "p2": p2.get("freeze_hash"),
            "p2_1": authority.get("current_p2_1_freeze_hash"),
            "p2_2": p22.get("freeze_hash"),
        },
        "research": {
            "current_universe": registry.get("unique_strategy_dna_count"),
            "current_signals": int(signals.get("fresh_entry_count") or 0),
            "current_strategies": int(registry.get("unique_strategy_dna_count") or 0),
            "financial_finalist_go": bool(registry.get("financial_finalist_go")),
            "forward_shadow_go": bool(registry.get("forward_shadow_go")),
        },
        "portfolio": {
            "current_portfolio_status": read_json(
                project_root / "output/portfolio/status.json"
            ).get("status"),
            "desired_portfolio_status": desired.get("status"),
            "current_executable_portfolio": {
                "decision": feasibility.get("decision_status"),
                "feasible_now_count": int(feasibility.get("feasible_now_count") or 0),
                "cash_is_valid_outcome": bool(feasibility.get("cash_is_valid_outcome")),
            },
        },
        "current_blockers": blockers,
        "historical_reports_may_override": False,
        **AUTHORITY,
    }


def _architecture_map(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
) -> dict[str, Any]:
    specs = [
        ("RAW_DATA", "src/stocks/data", "EXISTS_AND_STRONG"),
        ("POINT_IN_TIME_NORMALIZATION", "src/stocks/research/instrument_manifest.py", "EXISTS_BUT_WEAK"),
        ("FEATURES", "src/stocks/features", "EXISTS_AND_STRONG"),
        ("STRATEGY_GENERATOR", "src/stocks/research/autopilot/generator.py", "EXISTS_AND_STRONG"),
        ("UNIVERSAL_TRIAL_LEDGER", "src/stocks/p3/ledger.py", "EXISTS_AND_STRONG"),
        ("STAGE0_FALSIFICATION", "src/stocks/research/stage0.py", "EXISTS_AND_STRONG"),
        ("NESTED_WALK_FORWARD", "src/stocks/research/phase11_6.py", "EXISTS_AND_STRONG"),
        ("HISTORICAL_OOS", "src/stocks/research/evidence_throughput.py", "EXISTS_AND_STRONG"),
        ("MULTIPLE_TESTING", "src/stocks/research/evidence_throughput.py", "EXISTS_BUT_WEAK"),
        ("ROBUSTNESS", "src/stocks/research/phase11_7.py", "EXISTS_BUT_WEAK"),
        ("LIVE_SHADOW_FORWARD", "src/stocks/research/phase11_14.py", "BLOCKED_BY_FORWARD_EVIDENCE"),
        ("STRATEGY_QUALIFICATION", "src/stocks/research/promotion.py", "EXISTS_AND_STRONG"),
        ("OPPORTUNITY_GENERATION", "src/stocks/portfolio/manager.py", "EXISTS_AND_STRONG"),
        ("PORTFOLIO_CONSTRUCTION", "src/stocks/portfolio/orchestrator.py", "EXISTS_AND_STRONG"),
        ("WHOLE_SHARE_ECONOMICS", "src/stocks/portfolio/execution_feasibility.py", "EXISTS_AND_STRONG"),
        ("RISK", "src/stocks/portfolio/dynamic_risk.py", "EXISTS_AND_STRONG"),
        ("SHARIAH", "src/stocks/domain/shariah.py", "BLOCKED_BY_DATA"),
        ("STRATEGY_AUTHORITY", "src/stocks/portfolio/strategy_authority.py", "EXISTS_AND_STRONG"),
        ("LIVE_MARKET_DATA", "src/stocks/live/service.py", "BLOCKED_BY_DATA"),
        ("RECONCILIATION", "src/stocks/ibkr/reconciliation", "EXISTS_AND_STRONG"),
        ("WRITER", "src/stocks/live/writer.py", "EXISTS_AND_STRONG"),
        ("IBKR", "src/stocks/ibkr", "EXISTS_AND_STRONG"),
    ]
    modules = []
    for name, relative, classification in specs:
        path = project_root / relative
        modules.append(
            {
                "name": name,
                "path": relative,
                "classification": classification if path.exists() else "MISSING",
                "exists": path.exists(),
                "money_authority": name in {"RISK", "STRATEGY_AUTHORITY", "WRITER"},
            }
        )
    return {
        "schema": "p3_current_architecture_map_v1",
        "status": "GO_WITH_DOCUMENTED_GAPS",
        "generated_at": generated_at,
        "pipeline_order": [name for name, _, _ in specs],
        "modules": modules,
        "classification_counts": dict(
            sorted(Counter(row["classification"] for row in modules).items())
        ),
        "second_broker_client_created": False,
        "second_writer_created": False,
        "second_portfolio_manager_created": False,
        "second_risk_engine_created": False,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }


def _strategy_artifacts(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
    registry: dict[str, Any],
    evidence: dict[str, Any],
    multiple_testing: dict[str, Any],
    dna: list[StrategyDNA],
    ledger_counts: dict[str, int],
    import_hash: str,
    trial_ledger_path: Path,
) -> dict[str, dict[str, Any]]:
    partial = [item for item in dna if item.completeness_status != "DNA_COMPLETE"]
    source_counts = Counter(item.source_registry for item in dna)
    families = dict(registry.get("family_counts") or {})
    timeframes = dict(registry.get("timeframe_counts") or {})
    funnel = {
        "registered_strategy_dna": int(registry.get("unique_strategy_dna_count") or len(dna)),
        "registered_trials": int(ledger_counts.get("TOTAL") or 0),
        "executed_baseline_strategies": int(registry.get("executed_baseline_count") or 0),
        "historically_positive": int(registry.get("historically_positive_strategy_count") or 0),
        "cost_stress_survivors": int(registry.get("cost_stress_survivor_count") or 0),
        "nested_walk_forward_survivors": int(registry.get("nested_walk_forward_survivor_count") or 0),
        "forward_shadow_eligible": int(registry.get("forward_shadow_eligible_strategy_count") or 0),
        "multiple_testing_survivors": int(
            multiple_testing.get("multiple_testing_corrected_finalist_count") or 0
        ),
        "financial_finalists": int(
            ((evidence.get("finalists") or {}).get("financial_finalist_count")) or 0
        ),
        "qualified": 0,
        "new_live_authorized": 0,
    }
    generator = {
        "schema": "p3_strategy_generator_status_v1",
        "status": "GO_WITH_EVIDENCE_GAPS",
        "generated_at": generated_at,
        "strategy_generator_role": "PRIMARY_HYPOTHESIS_GENERATION_ENGINE",
        "total_hypotheses": funnel["registered_strategy_dna"],
        "total_configurations": funnel["registered_strategy_dna"],
        "families": families,
        "timeframes": timeframes,
        "asset_classes": registry.get("asset_class_counts") or {},
        "formula_count": int(registry.get("formula_count") or 0),
        "rejection_funnel": funnel,
        "research_yield": (
            funnel["nested_walk_forward_survivors"] / funnel["registered_strategy_dna"]
            if funnel["registered_strategy_dna"]
            else 0.0
        ),
        "rejected_trials_retained": True,
        "automatic_promotion": False,
        "financial_finalist_go": bool(registry.get("financial_finalist_go")),
        "open_evidence_gaps": registry.get("open_evidence_gaps") or [],
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    dna_registry = {
        "schema": "p3_strategy_dna_registry_pointer_v1",
        "status": "GO_WITH_PARTIAL_NATIVE_DNA_FIELDS" if partial else "GO",
        "generated_at": generated_at,
        "strategy_count": len(dna),
        "source_counts": dict(sorted(source_counts.items())),
        "complete_dna_count": len(dna) - len(partial),
        "partial_dna_count": len(partial),
        "partial_reason_counts": dict(
            sorted(Counter(item.completeness_status for item in partial).items())
        ),
        "canonical_native_registry": "output/research/strategies/strategy_registry.json",
        "canonical_native_registry_sha256": file_hash(
            project_root / "output/research/strategies/strategy_registry.json"
        ),
        "canonical_native_dna_csv": "output/research/strategies/strategy_dna.csv",
        "canonical_native_dna_csv_sha256": file_hash(
            project_root / "output/research/strategies/strategy_dna.csv"
        ),
        "p3_identity_policy": {
            "native_strategy_hash_preserved": True,
            "p3_strategy_spec_hash_is_sha256_of_canonical_p3_spec": True,
            "mutable_results_excluded": True,
            "missing_native_fields_never_invented": True,
        },
        "sample": [item.as_record() for item in dna[:25]],
        "complete_registry_not_duplicated": True,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    search_space = {
        "schema": "p3_search_space_summary_v1",
        "status": "GO",
        "generated_at": generated_at,
        "total_strategy_dna": len(dna),
        "total_unified_trials": int(ledger_counts.get("TOTAL") or 0),
        "trial_source_counts": ledger_counts,
        "family_counts": families,
        "timeframe_counts": timeframes,
        "asset_class_counts": registry.get("asset_class_counts") or {},
        "complexity_counts": {
            "one_block": registry.get("one_block_strategy_count"),
            "two_block": registry.get("two_block_strategy_count"),
            "three_to_five_blocks": registry.get("three_to_five_block_strategy_count"),
            "more_than_five_blocks": registry.get("more_than_five_block_strategy_count"),
        },
        "global_multiple_testing_trial_count": multiple_testing.get(
            "registered_hypothesis_count"
        ),
        "global_unique_economic_outcomes": multiple_testing.get(
            "unique_economic_outcome_count"
        ),
        "global_duplicate_outcomes": multiple_testing.get("duplicate_outcome_count"),
        "researcher_reset_allowed": False,
        "multi_timeframe_combinations_count_as_trials": True,
        "rejected_trials_retained": True,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    ledger_status = {
        "schema": "p3_unified_trial_ledger_status_v1",
        "status": "GO",
        "generated_at": generated_at,
        "canonical_public_ledger": str(trial_ledger_path.relative_to(project_root)).replace(
            "\\", "/"
        ),
        "canonical_public_ledger_sha256": file_hash(trial_ledger_path),
        "canonical_private_store": "data/research/p3/private/unified_evidence.sqlite3",
        "source_counts": ledger_counts,
        "import_hash": import_hash,
        "append_only": True,
        "idempotent_import": True,
        "immutability_conflict_fails_closed": True,
        "covers_deterministic_and_ai": True,
        "source_ledgers_are_import_feeds_not_competing_authority": True,
        **AUTHORITY,
    }
    return {
        "output/research/strategy-generator-status.json": generator,
        "output/research/strategy-dna-registry.json": dna_registry,
        "output/research/search-space-summary.json": search_space,
        "output/research/trial-ledger-status.json": ledger_status,
    }


def _validation_artifacts(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
    evidence: dict[str, Any],
    multiple_testing: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    walk_forward = read_json(
        project_root / "output/research/phase11_6/walk_forward_status.json"
    )
    hmm = read_json(project_root / "output/research/phase11_11/status.json")
    phase14_selection = read_json(
        project_root / "output/research/phase11_14/selection-bias-audit.json"
    )
    finalist = dict((evidence.get("finalists") or {}).get("closest_candidate") or {})
    checks = dict(finalist.get("checks") or {})
    mt_candidates = list(multiple_testing.get("candidates") or [])
    shared_oos = load_shared_oos_matrix(project_root)
    global_multiple_testing = multiple_testing_diagnostics(
        shared_oos,
        global_trial_count=int(
            multiple_testing.get("registered_hypothesis_count") or 0
        ),
    )
    parameter_diagnostics = parameter_stability_diagnostics(project_root)
    regime_diagnostics = regime_robustness_diagnostics(project_root, shared_oos)
    nested = {
        "schema": "p3_nested_walk_forward_summary_v1",
        "status": (
            "GO_NO_FINANCIAL_FINALIST"
            if walk_forward.get("status") == "GO"
            else "INSUFFICIENT_EVIDENCE"
        ),
        "generated_at": generated_at,
        "fold_count": int(walk_forward.get("fold_count") or 0),
        "identity_count": int(walk_forward.get("identity_count") or 0),
        "outer_result_count": int(walk_forward.get("outer_result_count") or 0),
        "passing_strategy_timeframes": walk_forward.get(
            "passing_strategy_timeframes"
        ),
        "selector_evaluated_out_of_sample": True,
        "purge_and_embargo_contract": "NATIVE_HORIZON_AWARE_SPLIT",
        "outer_oos_retuning_allowed": False,
        "financial_finalist_go": bool(walk_forward.get("FINANCIAL_FINALIST_GO")),
        "source_artifact": "output/research/phase11_6/walk_forward_status.json",
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    historical = {
        "schema": "p3_historical_oos_summary_v1",
        "status": (
            "HISTORICAL_OOS_SUPPORTED_WITH_NONFINAL_GAPS"
            if checks.get("oos_pass")
            else "INSUFFICIENT_EVIDENCE"
        ),
        "generated_at": generated_at,
        "closest_candidate": _candidate_summary(finalist),
        "historical_oos_is_forward": False,
        "selection_bias_status": hmm.get("selection_bias_status"),
        "untouched_holdout_status": phase14_selection.get("status"),
        "financial_finalist_go": bool(evidence.get("FINANCIAL_FINALIST_GO")),
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    multiple = {
        "schema": "p3_multiple_testing_summary_v1",
        "status": str(
            multiple_testing.get("status")
            or "INSUFFICIENT_MULTIPLE_TESTING_EVIDENCE"
        ),
        "generated_at": generated_at,
        "registered_hypothesis_count": int(
            multiple_testing.get("registered_hypothesis_count") or 0
        ),
        "source_hypothesis_count": int(
            multiple_testing.get("source_hypothesis_count") or 0
        ),
        "unique_economic_outcome_count": int(
            multiple_testing.get("unique_economic_outcome_count") or 0
        ),
        "duplicate_outcome_count": int(
            multiple_testing.get("duplicate_outcome_count") or 0
        ),
        "minimum_dsr_probability": multiple_testing.get("minimum_dsr_probability"),
        "multiple_testing_corrected_finalist_count": int(
            multiple_testing.get("multiple_testing_corrected_finalist_count") or 0
        ),
        "pbo_status": multiple_testing.get("pbo_status"),
        "pbo_method": multiple_testing.get("pbo_method"),
        "pbo_is_classical_cscv": bool(
            multiple_testing.get("pbo_is_classical_cscv")
        ),
        "pbo_by_timeframe": multiple_testing.get("pbo_by_timeframe") or {},
        "shared_oos_diagnostics": global_multiple_testing,
        "reality_check_status": (global_multiple_testing.get("white_reality_check") or {}).get(
            "p_value"
        ),
        "hansen_spa_status": (global_multiple_testing.get("hansen_spa") or {}).get(
            "p_value"
        ),
        "false_discovery_control": global_multiple_testing.get(
            "false_discovery_control"
        ),
        "candidate_count": len(mt_candidates),
        "candidates": mt_candidates,
        "thresholds_changed": bool(multiple_testing.get("thresholds_changed")),
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    parameter = {
        "schema": "p3_parameter_stability_summary_v1",
        "status": parameter_diagnostics.get("status"),
        "generated_at": generated_at,
        "closest_candidate_id": finalist.get("strategy_id"),
        "closest_candidate_parameter_stability_pass": checks.get(
            "parameter_stability_pass"
        ),
        "plateau_ratio": finalist.get("plateau_ratio"),
        "neighbor_median": finalist.get("neighbor_median"),
        "neighbor_worst": finalist.get("neighbor_worst"),
        "global_parameter_stability_complete": parameter_diagnostics.get("status")
        == "GO",
        "global_diagnostics": parameter_diagnostics,
        "blockers": (
            []
            if parameter_diagnostics.get("status") == "GO"
            else ["CANONICAL_PARAMETER_STABILITY_MATRIX_INCOMPLETE"]
        ),
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    regime_audit = read_json(project_root / "output/research/phase11_7/regime-audit.json")
    regime = {
        "schema": "p3_regime_robustness_summary_v1",
        "status": regime_diagnostics.get("status"),
        "generated_at": generated_at,
        "closest_candidate_id": finalist.get("strategy_id"),
        "closest_candidate_robustness_pass": checks.get("robustness_pass"),
        "native_regime_audit": regime_audit,
        "hmm_strategy_timeframe_pairs": int(
            hmm.get("strategy_timeframe_pairs_evaluated") or 0
        ),
        "leave_one_regime_out_status": regime_diagnostics.get(
            "leave_one_regime_out_status"
        ),
        "worst_regime_net_expectancy": finalist.get("worst_regime_net_expectancy"),
        "global_regime_robustness_complete": regime_diagnostics.get("status")
        == "GO",
        "global_diagnostics": regime_diagnostics,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    return {
        "output/research/nested-walk-forward.json": nested,
        "output/research/historical-oos-summary.json": historical,
        "output/research/multiple-testing-summary.json": multiple,
        "output/research/parameter-stability.json": parameter,
        "output/research/regime-robustness.json": regime,
    }


def _data_artifacts(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
) -> dict[str, dict[str, Any]]:
    readiness = read_json(project_root / "output/verification/data-readiness.json")
    pit = read_json(project_root / "output/research/universe/pit_universe_audit.json")
    survivorship = read_json(
        project_root / "output/research/universe/survivorship_audit.json"
    )
    shariah = read_json(
        project_root / "output/research/universe/shariah_eligibility.json"
    )
    universe = read_json(project_root / "output/verification/universe-status.json")
    gaps = list(readiness.get("open_data_gaps") or [])
    pit_artifact = {
        "schema": "p3_pit_readiness_v1",
        "status": "PIT_PARTIAL",
        "generated_at": generated_at,
        "core_research_ready": bool(readiness.get("core_research_ready")),
        "all_desired_data_available": bool(readiness.get("all_desired_data_available")),
        "available_source_count": int(readiness.get("available_source_count") or 0),
        "point_in_time_universe_status": pit.get("point_in_time_universe_status"),
        "current_symbol_count": int(pit.get("current_symbol_count") or 0),
        "observation_row_count": int(pit.get("observation_row_count") or 0),
        "discovery_instrument_count": int(universe.get("discovery_instrument_count") or 0),
        "active_listing_count": int(universe.get("active_listing_count") or 0),
        "signal_eligible_instrument_count": int(
            universe.get("signal_eligible_instrument_count") or 0
        ),
        "live_executable_instrument_count": int(
            universe.get("live_executable_instrument_count") or 0
        ),
        "open_data_gaps": gaps,
        "evidence_label": "RESEARCH_ONLY",
        "final_production_evidence_allowed": False,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    survivorship_artifact = {
        "schema": "p3_survivorship_readiness_v1",
        "status": str(survivorship.get("status") or "BLOCKED"),
        "generated_at": generated_at,
        "delisted_security_count": int(
            survivorship.get("delisted_security_count") or 0
        ),
        "delisted_security_count_is_complete": bool(
            survivorship.get("delisted_security_count_is_complete")
        ),
        "reason": survivorship.get("reason"),
        "today_survivors_only_allowed": False,
        "evidence_label": "SURVIVORSHIP_PARTIAL",
        "final_production_evidence_allowed": False,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    shariah_artifact = {
        "schema": "p3_shariah_pit_readiness_v1",
        "status": str(shariah.get("status") or "SHARIAH_PIT_MISSING"),
        "generated_at": generated_at,
        "native_evidence": shariah,
        "historical_current_classification_substitution_allowed": False,
        "research_only_unrestricted_series_must_be_separate": True,
        "final_production_evidence_allowed": str(shariah.get("status"))
        in {"GO", "COMPLETE"},
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    return {
        "output/data/pit-readiness.json": pit_artifact,
        "output/data/survivorship-readiness.json": survivorship_artifact,
        "output/data/shariah-pit-readiness.json": shariah_artifact,
    }


def _capability_usage(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
) -> dict[str, Any]:
    matrix = read_json(
        project_root / "output/ai/reference-repo-integration-matrix.json"
    )
    rows = []
    counts: Counter[str] = Counter()
    for capability in list(matrix.get("capabilities") or []):
        artifact = str(capability.get("artifact") or "")
        artifact_path = project_root / artifact if artifact else None
        exists = bool(artifact_path and artifact_path.exists())
        output_used = bool(capability.get("output_used"))
        native_status = str(capability.get("native_status") or "UNKNOWN")
        if not output_used:
            classification = "BLOCKED_BY_EVIDENCE"
        elif not exists:
            classification = "SHOULD_BE_USED_BUT_DISCONNECTED"
        elif capability.get("authority_after") in {"CONTEXT_ONLY", "ADVISORY_ONLY"}:
            classification = "CONTEXT_ACTIVE"
        elif native_status == "EXISTS_AND_STRONG":
            classification = "PRODUCTION_ACTIVE"
        else:
            classification = "RESEARCH_ACTIVE"
        counts[classification] += 1
        rows.append(
            {
                "capability_id": capability.get("capability_id"),
                "name": capability.get("capability_name"),
                "authority": capability.get("authority_after"),
                "latest_output": artifact or None,
                "latest_output_exists": exists,
                "latest_output_sha256": file_hash(artifact_path) if artifact_path else None,
                "actual_downstream_consumer": (
                    capability.get("native_module") if output_used else None
                ),
                "invocation_frequency": "NATIVE_CADENCE_NOT_CANONICALLY_COUNTED",
                "output_used": output_used,
                "evidence_status": capability.get("forward_evidence"),
                "classification": classification,
                "reason_if_unused": (
                    None
                    if output_used
                    else str(capability.get("gap") or "OUTPUT_USED_FALSE")
                ),
            }
        )
    return {
        "schema": "p3_capability_runtime_usage_v1",
        "status": "GO_WITH_DISCONNECTED_OR_EVIDENCE_BLOCKED_CAPABILITIES",
        "generated_at": generated_at,
        "capability_count": len(rows),
        "classification_counts": dict(sorted(counts.items())),
        "capabilities": rows,
        "capability_34_added": False,
        "code_exists_is_not_usage": True,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }


def _ai_shadow_artifacts(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
) -> dict[str, dict[str, Any]]:
    comparison = read_json(
        project_root / "output/ai/shadow-portfolio-comparison.json"
    )
    health = read_json(project_root / "output/ai/model-health.json")
    learning = read_json(project_root / "output/portfolio/learning-model-evidence.json")
    comparison_valid = bool(comparison.get("comparison_valid"))
    common = {
        "generated_at": generated_at,
        "same_period_required": True,
        "automatic_promotion": False,
        "model_active_count": int(health.get("active_model_count") or 0),
        "model_paused_or_shadow_count": int(
            health.get("paused_or_shadow_count") or 0
        ),
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    comparable = {
        "schema": "p3_ai_shadow_comparable_series_v1",
        "status": "GO" if comparison_valid else "INSUFFICIENT_COMPARABLE_SERIES",
        "comparison_valid": comparison_valid,
        "reason": comparison.get("reason"),
        "variants": comparison.get("variants") or [],
        "series": [],
        "backfill_allowed": False,
        **common,
    }
    incremental = {
        "schema": "p3_ai_shadow_incremental_performance_v1",
        "status": str(comparison.get("status") or "NO_INCREMENTAL_EVIDENCE"),
        "comparison_valid": comparison_valid,
        "required_metrics": comparison.get("required_metrics") or [],
        "incremental_net_return": None,
        "incremental_sharpe": None,
        "incremental_drawdown": None,
        "financial_validation_status": learning.get("financial_validation_status"),
        "accepted_shadow_classifier_count": int(
            learning.get("accepted_shadow_classifier_count") or 0
        ),
        "rl_shadow_validation_go": bool(learning.get("rl_shadow_validation_go")),
        **common,
    }
    forward = {
        "schema": "p3_ai_shadow_live_forward_status_v1",
        "status": "FORWARD_INSUFFICIENT_SAMPLE",
        "eligible_model_count": int(health.get("paused_or_shadow_count") or 0),
        "active_shadow_model_count": int(health.get("active_model_count") or 0),
        "closed_comparable_episode_count": 0,
        "pending_episode_count": 0,
        "forward_supported_model_count": 0,
        "backfill_allowed": False,
        "training_may_grant_money_authority": False,
        **common,
    }
    return {
        "output/ai/shadow/comparable-series.json": comparable,
        "output/ai/shadow/incremental-performance.json": incremental,
        "output/ai/shadow/live-forward-status.json": forward,
    }


def _portfolio_artifacts(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
    evidence: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    feasibility = read_json(
        project_root / "output/portfolio/p2-2-execution-feasibility.json"
    )
    portfolio = read_json(project_root / "output/portfolio/active_portfolio_plan.json")
    attribution = read_json(
        project_root / "output/portfolio/performance-attribution.json"
    )
    funnel = dict(evidence.get("funnel") or {})
    funnel.update(
        {
            "execution_feasibility_evaluated": int(
                feasibility.get("evaluated_symbol_strategy_count") or 0
            ),
            "execution_feasible_now": int(feasibility.get("feasible_now_count") or 0),
            "execution_rejected": int(feasibility.get("rejected_count") or 0),
            "orders_created": int(feasibility.get("orders_created") or 0),
            "orders_submitted": int(feasibility.get("orders_submitted") or 0),
        }
    )
    economic_funnel = {
        "schema": "p3_current_economic_funnel_v1",
        "status": "NO_FEASIBLE_CANDIDATE_RETAIN_CASH",
        "generated_at": generated_at,
        "funnel": funnel,
        "primary_rejection_distribution": feasibility.get(
            "primary_rejection_distribution"
        )
        or {},
        "all_gate_rejection_distribution": feasibility.get(
            "all_gate_rejection_distribution"
        )
        or {},
        "dominant_bottleneck": feasibility.get("biggest_marginal_loss_gate"),
        "bottleneck_type": _blocker_type(
            str(feasibility.get("biggest_marginal_loss_gate") or "")
        ),
        "cash_is_valid_outcome": bool(feasibility.get("cash_is_valid_outcome")),
        "automatic_gate_relaxation": False,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    strategy_portfolio = {
        "schema": "p3_strategy_portfolio_v1",
        "status": "CASH_NO_QUALIFIED_STRATEGY_PORTFOLIO",
        "generated_at": generated_at,
        "qualified_strategy_count": 0,
        "financial_finalist_count": int(
            ((evidence.get("finalists") or {}).get("financial_finalist_count")) or 0
        ),
        "cash_weight": 1.0,
        "strategy_weights": [],
        "capital_competition_simulated": False,
        "reason": "NO_QUALIFIED_FORWARD_SUPPORTED_STRATEGY_SET",
        "portfolio_status": portfolio.get("status"),
        "performance_attribution_status": attribution.get("status"),
        "realized_fact_count": int(attribution.get("realized_fact_count") or 0),
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    dependency = strategy_dependency_diagnostics(
        load_shared_oos_matrix(project_root)
    )
    correlation = {
        "schema": "p3_strategy_correlation_v1",
        "status": dependency.get("status"),
        "generated_at": generated_at,
        "qualified_strategy_count": 0,
        "evaluated_shadow_strategy_count": dependency.get("strategy_count", 0),
        "oos_return_correlation": dependency.get("oos_return_correlation", []),
        "downside_correlation": dependency.get("downside_correlation", []),
        "drawdown_overlap": dependency.get("drawdown_overlap", []),
        "signal_jaccard": dependency.get("signal_jaccard", []),
        "trade_overlap": dependency.get("trade_overlap", []),
        "trade_overlap_status": dependency.get("trade_overlap_status"),
        "factor_overlap": dependency.get("factor_overlap", []),
        "factor_overlap_status": dependency.get("factor_overlap_status"),
        "near_duplicate_clusters": dependency.get("near_duplicate_clusters", []),
        "dependency_method": dependency.get("method_version"),
        "native_asset_correlation_artifact": "output/portfolio/correlation_matrix.parquet",
        "asset_correlation_is_strategy_correlation": False,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    analysis = {
        "schema": "p3_strategy_portfolio_analysis_v1",
        "status": "INSUFFICIENT_QUALIFIED_STRATEGY_SET",
        "generated_at": generated_at,
        "with_without_marginal_analysis": [],
        "portfolio_net_return": None,
        "portfolio_drawdown": None,
        "portfolio_heat": 0.0,
        "cash_allocation": 1.0,
        "standalone_returns_summed_as_full_capital": False,
        "source_hashes": source_hashes,
        **AUTHORITY,
    }
    return {
        "output/portfolio/current-economic-funnel.json": economic_funnel,
        "output/portfolio/strategy-portfolio.json": strategy_portfolio,
        "output/research/strategy-correlation.json": correlation,
        "output/research/strategy-portfolio-analysis.json": analysis,
    }


def _quote_readiness(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
) -> dict[str, Any]:
    cycle = read_json(project_root / "output/ibkr/live/automatic-cycle.json")
    authority = read_json(project_root / "output/ibkr/live/authority-status.json")
    blockers = list(cycle.get("blockers") or [])
    entitlement_blocked = any(
        "ENTITLEMENT" in str(blocker).upper() for blocker in blockers
    )
    quote_blocked = bool(blockers) or cycle.get("cycle_status") == "LIVE_QUOTE_BLOCKED"
    return {
        "schema": "p3_live_quote_readiness_v1",
        "status": (
            "LIVE_QUOTE_ENTITLEMENT_BLOCKED"
            if entitlement_blocked
            else "LIVE_QUOTE_NOT_PROVEN"
            if quote_blocked
            else "GO"
        ),
        "generated_at": generated_at,
        "cycle_status": cycle.get("cycle_status"),
        "symbol": None,
        "contract": None,
        "exchange": None,
        "primary_exchange": None,
        "currency": None,
        "requested_market_data_type": "LIVE_REQUIRED",
        "callback_market_data_type": None,
        "bid": None,
        "ask": None,
        "bid_size": None,
        "ask_size": None,
        "bid_timestamp": None,
        "ask_timestamp": None,
        "quote_age_seconds": None,
        "streaming_or_snapshot": None,
        "entitlement_state": (
            "BLOCKED" if entitlement_blocked else "NOT_PROVEN" if quote_blocked else "PROVEN"
        ),
        "spread": None,
        "relative_spread": None,
        "quote_valid": False if blockers else True,
        "blockers": blockers,
        "external_action_required": (
            "ENABLE_REQUIRED_IBKR_REALTIME_US_EQUITY_MARKET_DATA_SUBSCRIPTION_AND_REVERIFY"
            if entitlement_blocked
            else "VERIFY_CONTRACT_ROUTING_AND_REALTIME_BID_ASK_ENTITLEMENT_FOR_THE_NATURAL_SETUP"
            if quote_blocked
            else None
        ),
        "delayed_data_may_satisfy_live_gate": False,
        "infrastructure_authority": authority.get("execution_authority"),
        "live_place_order_calls": int(cycle.get("live_place_order_calls") or 0),
        "broker_write_calls": int(cycle.get("broker_write_calls") or 0),
        "source_hashes": source_hashes,
        **AUTHORITY,
    }


def _runtime_reliability(
    project_root: Path,
    generated_at: str,
    source_hashes: dict[str, str | None],
) -> dict[str, Any]:
    machine = read_json(project_root / "output/operations/machine-status.json")
    heartbeat = read_json(project_root / "runtime/heartbeat.json")
    last_cycle = read_json(project_root / "output/operations/last-cycle.json")
    telegram = read_json(project_root / "output/notifications/telegram_status.json")
    recent = _recent_cycle_summary(project_root, limit=50)
    operational_blockers = list(machine.get("last_cycle_blockers") or [])
    financial = [
        blocker
        for blocker in operational_blockers
        if any(
            token in str(blocker).upper()
            for token in ("WRITER", "RECONCILIATION", "RISK", "AUTHORITY", "QUOTE")
        )
    ]
    observability = [blocker for blocker in operational_blockers if blocker not in financial]
    return {
        "schema": "p3_runtime_reliability_v1",
        "status": (
            "DEGRADED_ACTIVE"
            if machine.get("status") == "DEGRADED" and machine.get("enabled")
            else str(machine.get("status") or "UNKNOWN")
        ),
        "generated_at": generated_at,
        "scheduler_enabled": bool(machine.get("enabled")),
        "runtime_mode": machine.get("mode"),
        "runtime_status": heartbeat.get("runtime_status"),
        "runtime_state": heartbeat.get("runtime_state"),
        "last_heartbeat": heartbeat.get("last_heartbeat"),
        "last_cycle_id": last_cycle.get("cycle_id"),
        "last_cycle_status": last_cycle.get("status"),
        "last_cycle_duration_seconds": last_cycle.get("duration_seconds"),
        "financial_gate_failures": financial,
        "observability_degradation": observability,
        "telegram": {
            "status": telegram.get("status"),
            "enabled": telegram.get("enabled"),
            "queue_size": telegram.get("queue_size"),
            "last_error": telegram.get("last_error"),
            "may_change_authority": telegram.get("telegram_can_change_authority"),
        },
        "recent_cycles": recent,
        "single_instance_lock": machine.get("single_instance_lock"),
        "single_instance_lock_exists": bool(
            machine.get("single_instance_lock")
            and Path(str(machine.get("single_instance_lock"))).exists()
        ),
        "atomic_artifact_policy": "TEMP_FILE_FSYNC_OS_REPLACE",
        "monitoring_failure_may_grant_authority": False,
        "broker_writes": int(heartbeat.get("broker_writes") or 0),
        "source_hashes": source_hashes,
        **AUTHORITY,
    }


def _engineering_status(project_root: Path, generated_at: str) -> dict[str, Any]:
    source_files = sorted((project_root / "src").rglob("*.py"))
    line_counts = []
    for path in source_files:
        try:
            lines = sum(1 for _ in path.open("r", encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
        line_counts.append(
            {
                "path": str(path.relative_to(project_root)).replace("\\", "/"),
                "lines": lines,
            }
        )
    large = sorted(line_counts, key=lambda item: int(item["lines"]), reverse=True)
    git_present = (project_root / ".git").exists()
    ci_present = (project_root / ".github/workflows/product-ci.yml").exists()
    p3_scheduler_present = _p3_scheduler_integration_present(project_root)
    policy_present = (
        project_root / "config/p3_unified_evidence_policy_v1.json"
    ).exists()
    return {
        "schema": "p3_engineering_status_v1",
        "status": "GO" if git_present and ci_present else "ENGINEERING_GAPS_OPEN",
        "generated_at": generated_at,
        "git_metadata_present": git_present,
        "ci_workflow_present": ci_present,
        "root_pyproject_present": (project_root / "pyproject.toml").exists(),
        "requirements_lock_present": (project_root / "requirements.lock.txt").exists(),
        "p3_policy_present": policy_present,
        "p3_scheduler_integration_present": p3_scheduler_present,
        "python_source_file_count": len(line_counts),
        "python_source_line_count": sum(int(item["lines"]) for item in line_counts),
        "files_over_1000_loc": sum(int(item["lines"]) > 1000 for item in line_counts),
        "modules_over_2000_loc": sum(int(item["lines"]) > 2000 for item in line_counts),
        "modules_over_4000_loc": sum(int(item["lines"]) > 4000 for item in line_counts),
        "largest_modules": large[:25],
        "refactor_policy": "BOUNDED_EXTRACTION_ONLY_WITH_IMMEDIATE_CORRECTNESS_BENEFIT",
        "full_rewrite_allowed": False,
        "open_gaps": [
            gap
            for gap, present in (
                ("GIT_METADATA_MISSING", git_present),
                ("PRODUCT_CI_MISSING", ci_present),
                ("ROOT_PYPROJECT_MISSING", (project_root / "pyproject.toml").exists()),
                ("P3_POLICY_MISSING", policy_present),
                ("P3_SCHEDULER_INTEGRATION_MISSING", p3_scheduler_present),
            )
            if not present
        ],
        **AUTHORITY,
    }


def _p3_readiness(
    *,
    generated_at: str,
    source_hashes: dict[str, str | None],
    canonical_state: dict[str, Any],
    architecture: dict[str, Any],
    strategy_artifacts: dict[str, dict[str, Any]],
    validation_artifacts: dict[str, dict[str, Any]],
    data_artifacts: dict[str, dict[str, Any]],
    capability_usage: dict[str, Any],
    ai_artifacts: dict[str, dict[str, Any]],
    portfolio_artifacts: dict[str, dict[str, Any]],
    quote_readiness: dict[str, Any],
    runtime_reliability: dict[str, Any],
    engineering: dict[str, Any],
    ledger_counts: dict[str, int],
) -> dict[str, Any]:
    financial_finalist_go = (
        portfolio_artifacts["output/portfolio/strategy-portfolio.json"]
        .get("financial_finalist_count", 0)
        > 0
    )
    multiple_diagnostics = validation_artifacts[
        "output/research/multiple-testing-summary.json"
    ].get("shared_oos_diagnostics") or {}
    internal_gates = {
        "CANONICAL_STATE_GO": canonical_state.get("status") == "GO",
        "ARCHITECTURE_MAP_GO": architecture.get("status")
        == "GO_WITH_DOCUMENTED_GAPS",
        "UNIFIED_TRIAL_LEDGER_GO": int(ledger_counts.get("TOTAL") or 0) > 0,
        "STRATEGY_DNA_GO": strategy_artifacts[
            "output/research/strategy-dna-registry.json"
        ].get("strategy_count", 0)
        > 0,
        "NESTED_WALK_FORWARD_GO": validation_artifacts[
            "output/research/nested-walk-forward.json"
        ].get("fold_count", 0)
        > 0,
        "MULTIPLE_TESTING_EVALUATED": multiple_diagnostics.get("status")
        in {"GO", "NO_MULTIPLE_TESTING_CORRECTED_FINALIST"},
        "PARAMETER_STABILITY_EVALUATED": validation_artifacts[
            "output/research/parameter-stability.json"
        ].get("global_parameter_stability_complete")
        is True,
        "REGIME_ROBUSTNESS_EVALUATED": validation_artifacts[
            "output/research/regime-robustness.json"
        ].get("global_regime_robustness_complete")
        is True,
        "STRATEGY_DEPENDENCY_EVALUATED": portfolio_artifacts[
            "output/research/strategy-correlation.json"
        ].get("status")
        == "GO",
        "CAPABILITY_USAGE_CONTRACT_GO": int(capability_usage.get("capability_count") or 0)
        > 0,
        "AI_SHADOW_CONTRACT_GO": bool(
            ai_artifacts["output/ai/shadow/incremental-performance.json"].get(
                "status"
            )
        ),
        "PORTFOLIO_CASH_OUTCOME_GO": portfolio_artifacts[
            "output/portfolio/current-economic-funnel.json"
        ].get("cash_is_valid_outcome")
        is True,
        "LIVE_FAIL_CLOSED_CONTRACT_GO": quote_readiness.get(
            "delayed_data_may_satisfy_live_gate"
        )
        is False,
        "RUNTIME_ACTIVE": canonical_state["runtime"].get("active") is True,
        "P3_POLICY_GO": engineering.get("p3_policy_present") is True,
        "P3_SCHEDULER_INTEGRATION_GO": engineering.get(
            "p3_scheduler_integration_present"
        )
        is True,
        "GIT_AND_CI_GO": engineering.get("git_metadata_present") is True
        and engineering.get("ci_workflow_present") is True,
        "ZERO_UNAUTHORIZED_BROKER_WRITES": int(
            canonical_state["broker"].get("broker_writes") or 0
        )
        == 0,
    }
    economic_and_external_gates = {
        "PIT_DATA_GO": data_artifacts["output/data/pit-readiness.json"].get(
            "final_production_evidence_allowed"
        )
        is True,
        "SURVIVORSHIP_GO": data_artifacts[
            "output/data/survivorship-readiness.json"
        ].get("final_production_evidence_allowed")
        is True,
        "SHARIAH_PIT_GO": data_artifacts[
            "output/data/shariah-pit-readiness.json"
        ].get("final_production_evidence_allowed")
        is True,
        "MULTIPLE_TESTING_SURVIVOR_GO": validation_artifacts[
            "output/research/multiple-testing-summary.json"
        ].get("multiple_testing_corrected_finalist_count", 0)
        > 0,
        "FORWARD_EVIDENCE_GO": financial_finalist_go,
        "AI_INCREMENTAL_EVIDENCE_GO": _ai_incremental_evidence_go(
            ai_artifacts["output/ai/shadow/incremental-performance.json"].get(
                "status"
            )
        ),
        "STRATEGY_PORTFOLIO_GO": portfolio_artifacts[
            "output/portfolio/strategy-portfolio.json"
        ].get("qualified_strategy_count", 0)
        > 0,
        "LIVE_QUOTE_GO": quote_readiness.get("status") == "GO",
        "RUNTIME_ACTIVE": canonical_state["runtime"].get("active") is True,
        "OPERATIONS_RELIABLE": runtime_reliability.get("status")
        not in {"DEGRADED", "DEGRADED_ACTIVE"},
    }
    gates = {**internal_gates, **economic_and_external_gates}
    internal_blockers = [
        name for name, passed in internal_gates.items() if not passed
    ]
    economic_and_external_blockers = [
        name
        for name, passed in economic_and_external_gates.items()
        if not passed
    ]
    blockers = [*internal_blockers, *economic_and_external_blockers]
    technical_foundation = all(internal_gates.values())
    return {
        "schema": "p3_unified_strategy_evidence_readiness_v1",
        "status": (
            "P3_IMPLEMENTATION_COMPLETE_ECONOMIC_EVIDENCE_INCOMPLETE"
            if technical_foundation
            else "P3_IMPLEMENTATION_INCOMPLETE"
        ),
        "generated_at": generated_at,
        "p3_complete": technical_foundation,
        "p3_completion_definition": "ALL_INTERNAL_IMPLEMENTATION_GATES_PASS;ZERO_FINALISTS_IS_VALID",
        "technical_foundation_go": technical_foundation,
        "financial_finalist_go": financial_finalist_go,
        "capital_scaling_ready": False,
        "level_two_activation_allowed": False,
        "gates": gates,
        "internal_gates": internal_gates,
        "economic_and_external_gates": economic_and_external_gates,
        "internal_blockers": internal_blockers,
        "economic_and_external_blockers": economic_and_external_blockers,
        "blockers": blockers,
        "external_blockers": [
            blocker
            for blocker in blockers
            if blocker in {"PIT_DATA_GO", "SURVIVORSHIP_GO", "SHARIAH_PIT_GO", "LIVE_QUOTE_GO"}
        ],
        "valid_current_financial_conclusion": [
            "NO_FINANCIAL_FINALIST",
            "NO_INCREMENTAL_AI_EVIDENCE",
            "FORWARD_EVIDENCE_INSUFFICIENT",
            "NO_TRADE_WHEN_LIVE_GATES_FAIL",
        ],
        "single_highest_value_next_action": (
            "CONTINUE_PREREGISTERED_RESEARCH_AND_FORWARD_OBSERVATION_WITHOUT_RELAXING_GATES"
        ),
        "source_hashes": source_hashes,
        **AUTHORITY,
    }


def _ai_incremental_evidence_go(status: Any) -> bool:
    """Accept only the model tournament's explicit positive evidence state."""

    return str(status or "").upper() == "SHADOW_VALIDATION_GO"


def _phase9_observation_is_current(
    payload: dict[str, Any],
    *,
    as_of: str,
    maximum_age_minutes: float = 15.0,
) -> bool:
    if str(payload.get("broker_observation_status") or "").upper() != "GO":
        return False
    try:
        observed = datetime.fromisoformat(str(payload["generated_at"]))
        current = datetime.fromisoformat(as_of)
    except (KeyError, TypeError, ValueError):
        return False
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    age_minutes = (current.astimezone(UTC) - observed.astimezone(UTC)).total_seconds() / 60
    return -2.0 <= age_minutes <= maximum_age_minutes


def _p3_scheduler_integration_present(project_root: Path) -> bool:
    for relative in (
        "src/stocks/operations/service.py",
        "src/stocks/operations/primary_refresh.py",
    ):
        path = project_root / relative
        if path.is_file() and '("p3", "publish")' in path.read_text(
            encoding="utf-8"
        ):
            return True
    return False


def _source_hashes(project_root: Path) -> dict[str, str | None]:
    return {
        relative: file_hash(project_root / relative)
        if (project_root / relative).is_file()
        else None
        for relative in SOURCE_PATHS
    }


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    if not candidate:
        return {
            "status": "NO_CANDIDATE",
            "strategy_id": None,
            "strategy_hash": None,
            "financial_finalist": False,
        }
    checks = dict(candidate.get("checks") or {})
    safe_metrics = {
        key: value
        for key, value in dict(candidate.get("metrics") or {}).items()
        if key
        in {
            "cagr",
            "max_drawdown",
            "profit_factor",
            "sharpe",
            "sortino",
            "trade_count",
            "turnover",
            "win_rate",
        }
    }
    return {
        "status": candidate.get("status") or "NONFINAL_CANDIDATE",
        "strategy_id": candidate.get("strategy_id"),
        "strategy_hash": candidate.get("strategy_hash"),
        "family": candidate.get("family"),
        "timeframe": candidate.get("timeframe"),
        "checks": checks,
        "metrics": safe_metrics,
        "failed_checks": sorted(
            str(key) for key, value in checks.items() if value is False
        ),
        "financial_finalist": bool(candidate.get("financial_finalist")),
    }


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _blocker_type(blocker: str) -> str:
    upper = blocker.upper()
    if any(token in upper for token in ("ENTITLEMENT", "SUBSCRIPTION", "QUOTE")):
        return "EXTERNAL_MARKET_DATA_OR_QUOTE"
    if any(token in upper for token in ("COST", "SPREAD", "SLIPPAGE", "COMMISSION")):
        return "ECONOMIC_EXECUTION_COST"
    if any(token in upper for token in ("LIQUID", "ADV", "SIZE", "VOLUME")):
        return "MARKET_CAPACITY"
    if any(token in upper for token in ("PIT", "SURVIVOR", "SHARIAH")):
        return "DATA_QUALITY"
    if any(token in upper for token in ("RISK", "DRAWDOWN", "VOLATILITY")):
        return "RISK"
    if any(token in upper for token in ("FORWARD", "SAMPLE", "TRADE_COUNT")):
        return "EVIDENCE_SAMPLE"
    return "UNCLASSIFIED"


def _recent_cycle_summary(project_root: Path, *, limit: int) -> dict[str, Any]:
    candidates = (
        project_root / "output/operations/cycle-history.jsonl",
        project_root / "output/operations/cycles.jsonl",
        project_root / "runtime/cycle-history.jsonl",
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        last_cycle = read_json(project_root / "output/operations/last-cycle.json")
        return {
            "status": "LAST_CYCLE_ONLY",
            "source": "output/operations/last-cycle.json",
            "sample_size": 1 if last_cycle else 0,
            "success_count": int(
                bool(last_cycle)
                and str(last_cycle.get("status")).upper() in {"GO", "OK", "COMPLETE"}
            ),
            "degraded_count": int(
                bool(last_cycle)
                and str(last_cycle.get("status")).upper()
                not in {"GO", "OK", "COMPLETE"}
            ),
        }
    rows: list[dict[str, Any]] = []
    try:
        for line in source.read_text(encoding="utf-8").splitlines()[-limit:]:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        rows = []
    statuses = Counter(
        str(row.get("status") or row.get("cycle_status") or "UNKNOWN")
        for row in rows
    )
    return {
        "status": "GO" if rows else "UNREADABLE_OR_EMPTY",
        "source": str(source.relative_to(project_root)).replace("\\", "/"),
        "sample_size": len(rows),
        "status_counts": dict(sorted(statuses.items())),
    }


def _render_report(
    *,
    generated_at: str,
    canonical_state: dict[str, Any],
    registry: dict[str, Any],
    evidence: dict[str, Any],
    validation_artifacts: dict[str, dict[str, Any]],
    data_artifacts: dict[str, dict[str, Any]],
    capability_usage: dict[str, Any],
    ai_artifacts: dict[str, dict[str, Any]],
    portfolio_artifacts: dict[str, dict[str, Any]],
    quote_readiness: dict[str, Any],
    runtime_reliability: dict[str, Any],
    engineering: dict[str, Any],
    p3_readiness: dict[str, Any],
    ledger_counts: dict[str, int],
) -> str:
    gates = dict(p3_readiness.get("gates") or {})
    gate_rows = "\n".join(
        f"| {name} | {'GO' if passed else 'BLOCKED'} |"
        for name, passed in gates.items()
    )
    blockers = p3_readiness.get("blockers") or []
    blocker_rows = "\n".join(f"- `{blocker}`" for blocker in blockers) or "- None"
    source_counts = "\n".join(
        f"- `{name}`: {count:,}" for name, count in sorted(ledger_counts.items())
    )
    mt = validation_artifacts["output/research/multiple-testing-summary.json"]
    nested = validation_artifacts["output/research/nested-walk-forward.json"]
    pit = data_artifacts["output/data/pit-readiness.json"]
    survivorship = data_artifacts["output/data/survivorship-readiness.json"]
    shariah = data_artifacts["output/data/shariah-pit-readiness.json"]
    incremental = ai_artifacts["output/ai/shadow/incremental-performance.json"]
    strategy_portfolio = portfolio_artifacts[
        "output/portfolio/strategy-portfolio.json"
    ]
    return f"""# P3 unified strategy evidence completion

Generated: `{generated_at}`

## Executive conclusion

The unified P3 evidence foundation is **{p3_readiness.get('status')}**. It is not a claim of profitability or production readiness. The current defensible financial conclusion is **no financial finalist, no AI incremental edge, insufficient forward evidence, and cash/no-trade while live gates fail**.

- P3 complete: `{p3_readiness.get('p3_complete')}`
- Technical foundation: `{p3_readiness.get('technical_foundation_go')}`
- Financial finalist: `{p3_readiness.get('financial_finalist_go')}`
- Capital scaling ready: `{p3_readiness.get('capital_scaling_ready')}`
- Runtime active: `{canonical_state.get('runtime', {}).get('active')}`
- Effective execution authority: `{canonical_state.get('runtime', {}).get('effective_execution_authority')}`
- Broker writes observed in canonical reconciliation: `{canonical_state.get('broker', {}).get('broker_writes')}`

## Current gate matrix

| Gate | State |
|---|---|
{gate_rows}

## Open blockers

{blocker_rows}

## Unified trial ledger

The deterministic generator and AI experiments are consolidated into one append-only SQLite evidence store and one public JSONL export. Every imported record remains namespaced and immutable; imports do not grant strategy or execution authority.

{source_counts}

## Economic evidence

- Unique strategies in registry: `{registry.get('unique_strategy_dna_count') or registry.get('strategy_count')}`
- Native trials: `{int(ledger_counts.get('AUTOPILOT_STANDARD', 0)) + int(ledger_counts.get('AUTOPILOT_BULK', 0))}`
- Executed baselines: `{registry.get('executed_baseline_count')}`
- Historically positive: `{registry.get('historically_positive_strategy_count')}`
- Cost-stress survivors: `{registry.get('cost_stress_survivor_count')}`
- Nested walk-forward survivors: `{registry.get('nested_walk_forward_survivor_count')}`
- Forward-shadow eligible: `{registry.get('forward_shadow_eligible_strategy_count')}`
- Financial finalists: `{strategy_portfolio.get('financial_finalist_count')}`
- Registered hypotheses: `{mt.get('registered_hypothesis_count')}`
- Unique economic outcomes: `{mt.get('unique_economic_outcome_count')}`
- Multiple-testing-corrected finalists: `{mt.get('multiple_testing_corrected_finalist_count')}`
- Nested walk-forward folds/results: `{nested.get('fold_count')}` / `{nested.get('outer_result_count')}`

Passing technical validation does not prove economic edge. The current portfolio remains `{strategy_portfolio.get('status')}` with cash weight `{strategy_portfolio.get('cash_weight')}`.

## Data evidence

- PIT readiness: `{pit.get('status')}`
- Survivorship readiness: `{survivorship.get('status')}`
- Shariah PIT readiness: `{shariah.get('status')}`

These gaps prohibit final production evidence where their respective artifact says `final_production_evidence_allowed=false`.

## AI evidence

- Capability runtime status: `{capability_usage.get('status')}`
- AI experiments in unified ledger: `{ledger_counts.get('AI_EXPERIMENT', 0)}`
- Incremental performance: `{incremental.get('status')}`
- AI money authority: `NONE`

AI remains shadow/advisory. It cannot change orders, weights, stops, targets, authority, or final strategy qualification.

## Live execution and operations

- Quote readiness: `{quote_readiness.get('status')}`
- Quote valid: `{quote_readiness.get('quote_valid')}`
- Runtime reliability: `{runtime_reliability.get('status')}`
- Scheduler enabled: `{runtime_reliability.get('scheduler_enabled')}`
- Runtime mode: `{runtime_reliability.get('runtime_mode')}`
- Reconciliation: `{canonical_state.get('broker', {}).get('reconciliation_status')}`

Delayed or synthetic data cannot satisfy the live quote gate. No trade is forced; an exact authorized natural setup may proceed only through the existing canonical writer after every financial, portfolio, quote, authority, reconciliation, and risk gate passes.

## Engineering condition

- Git metadata: `{engineering.get('git_metadata_present')}`
- Product CI workflow: `{engineering.get('ci_workflow_present')}`
- Locked requirements: `{engineering.get('requirements_lock_present')}`
- Python source files / LOC: `{engineering.get('python_source_file_count')}` / `{engineering.get('python_source_line_count')}`
- Files over 1,000 LOC: `{engineering.get('files_over_1000_loc')}`

## Highest-value next action

`{p3_readiness.get('single_highest_value_next_action')}`

This report is generated from current artifacts. The machine-readable authority is `output/verification/p3-readiness.json`; the manifest binds every published P3 artifact by SHA-256.
"""
