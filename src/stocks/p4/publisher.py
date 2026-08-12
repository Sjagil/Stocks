from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.p3.io import atomic_write_json, file_hash, read_json
from stocks.p4.completion import (
    AUDIT_PATH,
    REPORT_PATH as AUDIT_REPORT_PATH,
    publish_requirement_audit,
)
from stocks.p4.data import PITDataCatalog
from stocks.p4.forward import (
    freeze_forward_evaluation_protocol,
    preregister_phase11_14_candidates,
)
from stocks.rl.contracts import stable_hash


READINESS_PATH = Path("output/verification/p4-readiness.json")
MANIFEST_PATH = Path("output/verification/p4-manifest.json")
REPORT_PATH = Path("reports/P4_DATA_FORWARD_RL_STATUS.md")


def publish_p4_readiness(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    generated_at = _now()
    data_status = PITDataCatalog(root).audit()
    registration = preregister_phase11_14_candidates(root)
    forward_protocol = freeze_forward_evaluation_protocol(root)
    forward = read_json(root / "output/research/phase11_14/forward-performance.json")
    forward_audit = read_json(root / "output/research/phase11_14/forward-audit.json")
    quote = read_json(root / "output/ibkr/live/quote-readiness.json")
    p3 = read_json(root / "output/verification/p3-readiness.json")
    operations = read_json(root / "output/operations/last-cycle.json")
    registry = read_json(root / "models/rl/registry.json")
    rl_status = read_json(root / "output/rl/status.json")
    rl_evidence = _rl_evidence(root, registry)

    internal_gates = {
        "P3_FOUNDATION_GO": p3.get("p3_complete") is True,
        "P4_DATA_CONTRACT_GO": _files_exist(
            root,
            (
                "config/p4_data_policy_v1.json",
                "src/stocks/p4/data.py",
            ),
        ),
        "FORWARD_COHORT_FROZEN_GO": registration.get("status") == "FROZEN"
        and int(registration.get("candidate_count", 0)) == 9,
        "FORWARD_PROTOCOL_FROZEN_GO": forward_protocol.get("status")
        == "FROZEN"
        and len(forward_protocol.get("candidate_protocols", [])) == 9,
        "EXISTING_FORWARD_OBSERVER_REUSED_GO": bool(
            forward.get("schema") == "phase11_14_forward_performance_v1"
        ),
        "RL_CONTRACT_GO": _files_exist(
            root,
            (
                "config/rl_policy_v1.json",
                "config/rl_reward_v1.json",
                "config/rl_promotion_v1.json",
                "config/rl_training_v1.json",
            ),
        ),
        "RL_IMPLEMENTATION_GO": _files_exist(
            root,
            (
                "src/stocks/rl/environment.py",
                "src/stocks/rl/reward.py",
                "src/stocks/rl/experience.py",
                "src/stocks/rl/registry.py",
                "src/stocks/rl/training.py",
                "src/stocks/rl/evaluation.py",
                "src/stocks/rl/supervisor.py",
            ),
        ),
        "REFERENCE_INTEGRATION_AUDITED_GO": (
            root / "docs/P4_RL_REFERENCE_INTEGRATION.md"
        ).is_file(),
        "RUNTIME_ACTIVE": str(operations.get("status")) in {"GO", "DEGRADED"},
        "ZERO_RL_BROKER_WRITES": int(rl_status.get("broker_writes", 0) or 0) == 0,
    }
    registry_policies = registry.get("policies", {}) if isinstance(registry, dict) else {}
    rl_baseline_trained = bool(registry_policies)
    external_gates = {
        **data_status.get("gates", {}),
        "LIVE_QUOTE_GO": quote.get("quote_valid") is True
        and quote.get("entitlement_state") == "PROVEN",
        "FORWARD_EVIDENCE_GO": forward.get("status") == "GO"
        and int(forward.get("observation_count", 0)) > 0,
        "RL_BASELINE_TRAINED": rl_baseline_trained,
        "RL_INCREMENTAL_EVIDENCE_GO": rl_status.get("incremental_evidence_go") is True,
        "RL_FORWARD_EVIDENCE_GO": rl_status.get("forward_evidence_go") is True,
        "RL_POLICY_PROMOTION_GO": rl_status.get("promotion_status")
        == "PROMOTION_ELIGIBLE",
        "CORRECTED_RETEST_GO": _artifact_go(
            root / "output/p4/corrected-data-retest.json"
        ),
        "STRATEGY_PORTFOLIO_GO": _artifact_go(
            root / "output/p4/strategy-portfolio-evaluation.json"
        ),
        "PAPER_CANARY_GO": _artifact_go(
            root / "output/p4/paper-canary-readiness.json"
        ),
    }
    internal_blockers = [name for name, passed in internal_gates.items() if not passed]
    economic_external_blockers = [
        name for name, passed in external_gates.items() if not passed
    ]
    p4_complete = not internal_blockers and not economic_external_blockers
    readiness: dict[str, Any] = {
        "schema": "p4_data_forward_rl_readiness_v1",
        "status": "P4_COMPLETE" if p4_complete else "P4_ACTIVE_EVIDENCE_INCOMPLETE",
        "generated_at": generated_at,
        "p4_complete": p4_complete,
        "internal_gates": internal_gates,
        "economic_and_external_gates": external_gates,
        "internal_blockers": internal_blockers,
        "economic_and_external_blockers": economic_external_blockers,
        "data_status": data_status.get("status"),
        "forward_registration_status": registration.get("status"),
        "forward_protocol_status": forward_protocol.get("status"),
        "forward_protocol_hash": forward_protocol.get("protocol_hash"),
        "forward_candidate_count": registration.get("candidate_count", 0),
        "forward_observation_count": forward.get("observation_count", 0),
        "forward_performance_status": forward.get("status", "NOT_AVAILABLE"),
        "independent_forward_audit_status": forward_audit.get(
            "status", "NOT_AVAILABLE"
        ),
        "rl_policy_count": len(registry_policies),
        "rl_active_policy": registry.get("active") if registry else None,
        "rl_challenger_policy": registry.get("challenger") if registry else None,
        "rl_initial_experiment": rl_evidence,
        "rl_mode": "SHADOW_ONLY",
        "rl_live_enabled": False,
        "capital_scaling_ready": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "money_control": False,
        "orders_generated": 0,
        "broker_calls": 0,
        "broker_writes": 0,
        "valid_current_conclusion": [
            "P4_IMPLEMENTATION_IN_PROGRESS" if internal_blockers else "P4_INTERNAL_FOUNDATION_GO",
            "PIT_DATA_EXTERNAL_EVIDENCE_INCOMPLETE"
            if not all(data_status.get("gates", {}).values())
            else "PIT_DATA_GO",
            str(forward.get("status") or "FORWARD_EVIDENCE_NOT_AVAILABLE"),
            str(rl_status.get("status") or "RL_BASELINE_NOT_TRAINED"),
            "CASH_NO_TRADE_WHILE_FINANCIAL_GATES_FAIL",
        ],
        "single_highest_value_next_action": (
            "INGEST_ATTESTED_PIT_SECURITY_MASTER_MEMBERSHIP_DELISTING_SHARIAH_AND_FUNDAMENTALS"
            if not all(data_status.get("gates", {}).values())
            else "CONTINUE_PREREGISTERED_FORWARD_AND_RL_SHADOW_EVIDENCE"
        ),
        "source_hashes": _source_hashes(root),
    }
    readiness["content_hash"] = stable_hash(readiness)
    atomic_write_json(root / READINESS_PATH, readiness)
    _write_report(root, readiness, data_status, forward, rl_status, rl_evidence)
    requirement_audit = publish_requirement_audit(root, readiness, rl_status)
    manifest = _write_manifest(root, readiness)
    return {
        "status": "GO" if not internal_blockers else "NO_GO",
        "readiness_status": readiness["status"],
        "p4_complete": p4_complete,
        "readiness": readiness,
        "manifest": manifest,
        "requirement_audit": requirement_audit,
        "report": REPORT_PATH.as_posix(),
        "execution_authority": "NONE",
        "broker_writes": 0,
    }


def _write_report(
    root: Path,
    readiness: dict[str, Any],
    data_status: dict[str, Any],
    forward: dict[str, Any],
    rl_status: dict[str, Any],
    rl_evidence: dict[str, Any],
) -> None:
    gate_rows = [
        f"| {name} | {'GO' if passed else 'BLOCKED'} |"
        for name, passed in {
            **readiness["internal_gates"],
            **readiness["economic_and_external_gates"],
        }.items()
    ]
    challenger_metrics = rl_evidence.get("challenger_metrics", {})
    baseline_metrics = rl_evidence.get("deterministic_baseline_metrics", {})
    promotion = rl_evidence.get("promotion_decision", {})
    report = "\n".join(
        [
            "# P4 data, forward evidence and RL status",
            "",
            f"Generated: `{readiness['generated_at']}`",
            "",
            "## Executive conclusion",
            "",
            f"Current status: **{readiness['status']}**. PPO remains `SHADOW_ONLY`, "
            "has no broker or money authority, and cannot relax deterministic gates.",
            "",
            f"- P4 complete: `{readiness['p4_complete']}`",
            f"- Frozen forward candidates: `{readiness['forward_candidate_count']}`",
            f"- Forward observations: `{readiness['forward_observation_count']}`",
            f"- Forward status: `{readiness['forward_performance_status']}`",
            f"- Registered RL policies: `{readiness['rl_policy_count']}`",
            f"- RL status: `{rl_status.get('status', 'NOT_TRAINED')}`",
            f"- PIT data status: `{data_status.get('status')}`",
            "- Execution authority: `NONE`",
            "- Broker writes: `0`",
            "",
            "## Gate matrix",
            "",
            "| Gate | State |",
            "|---|---|",
            *gate_rows,
            "",
            "## Open evidence blockers",
            "",
            *[
                f"- `{name}`"
                for name in readiness["economic_and_external_blockers"]
            ],
            "",
            "## Data boundary",
            "",
            "Current-universe membership and current Shariah classifications may not "
            "substitute for history. Only provider-versioned, licensed, hashed and "
            "point-in-time attested snapshots can turn the data gates green.",
            "",
            "## Forward and RL boundary",
            "",
            f"The existing Phase 11.14 frozen observer is reused and currently reports "
            f"`{forward.get('status', 'NOT_AVAILABLE')}`. Historical OOS results are not "
            "forward evidence. A PPO challenger is compared against the existing "
            "deterministic engine and can be rejected without changing any execution path.",
            "",
            "## Initial PPO experiment",
            "",
            f"- Policy: `{rl_evidence.get('policy_version', 'NOT_AVAILABLE')}`",
            f"- Training period: `{rl_evidence.get('training_start')}` through "
            f"`{rl_evidence.get('training_end')}`",
            f"- Test observations: `{rl_evidence.get('test_observations', 0)}`",
            f"- Test episodes: `{rl_evidence.get('test_episodes', 0)}`",
            f"- PPO trades: `{challenger_metrics.get('number_trades', 0)}`",
            f"- PPO net return: `{challenger_metrics.get('net_return')}`",
            f"- PPO Sharpe: `{challenger_metrics.get('sharpe')}`",
            f"- PPO max drawdown: `{challenger_metrics.get('maximum_drawdown')}`",
            f"- PPO CVaR 95: `{challenger_metrics.get('cvar_95')}`",
            f"- PPO turnover: `{challenger_metrics.get('turnover')}`",
            f"- PPO costs: `{challenger_metrics.get('fees_and_slippage')}`",
            f"- Deterministic net return: `{baseline_metrics.get('net_return')}`",
            f"- Deterministic Sharpe: `{baseline_metrics.get('sharpe')}`",
            f"- Bootstrap probability of improvement: "
            f"`{rl_evidence.get('bootstrap_probability_of_improvement')}`",
            f"- Promotion: `{promotion.get('status', 'NOT_AVAILABLE')}`",
            f"- Promotion blockers: `{', '.join(promotion.get('reasons', []))}`",
            "- Performance by strategy and asset class is the same bounded SPUS/ETF "
            "test result; no cross-asset inference is permitted.",
            "",
            "## Highest-value next action",
            "",
            f"`{readiness['single_highest_value_next_action']}`",
            "",
        ]
    )
    path = root / REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(path)


def _write_manifest(root: Path, readiness: dict[str, Any]) -> dict[str, Any]:
    paths = [
        Path("config/p4_data_policy_v1.json"),
        Path("config/p4_forward_evaluation_v1.json"),
        Path("config/rl_policy_v1.json"),
        Path("config/rl_reward_v1.json"),
        Path("config/rl_promotion_v1.json"),
        Path("config/rl_training_v1.json"),
        Path("docs/P4_RL_REFERENCE_INTEGRATION.md"),
        Path("docs/P4_EXTERNAL_EVIDENCE_RUNBOOK.md"),
        Path("output/p4/data-catalog-status.json"),
        Path("output/p4/preregistered-forward-cohort.json"),
        Path("output/p4/frozen-forward-evaluation-protocol.json"),
        Path("output/rl/status.json"),
        Path("models/rl/registry.json"),
        READINESS_PATH,
        REPORT_PATH,
        AUDIT_PATH,
        AUDIT_REPORT_PATH,
    ]
    artifacts = [
        {
            "path": path.as_posix(),
            "sha256": file_hash(root / path),
            "size_bytes": (root / path).stat().st_size,
        }
        for path in paths
        if (root / path).is_file()
    ]
    payload = {
        "schema": "p4_data_forward_rl_manifest_v1",
        "status": "GO",
        "generated_at": readiness["generated_at"],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "execution_authority": "NONE",
        "broker_writes": 0,
        "money_control": False,
    }
    payload["content_hash"] = stable_hash(payload)
    atomic_write_json(root / MANIFEST_PATH, payload)
    return payload


def _source_hashes(root: Path) -> dict[str, str | None]:
    paths = (
        "config/p4_data_policy_v1.json",
        "config/p4_forward_evaluation_v1.json",
        "config/rl_policy_v1.json",
        "config/rl_reward_v1.json",
        "config/rl_promotion_v1.json",
        "config/rl_training_v1.json",
        "output/research/phase11_14/qualification-boundary.json",
        "output/research/phase11_14/forward-performance.json",
        "output/ibkr/live/quote-readiness.json",
        "output/verification/p3-readiness.json",
    )
    return {path: file_hash(root / path) for path in paths}


def _rl_evidence(root: Path, registry: dict[str, Any]) -> dict[str, Any]:
    version = registry.get("challenger") or registry.get("active")
    record = registry.get("policies", {}).get(version, {}) if version else {}
    policy_root = root / str(record.get("path") or "")
    training = read_json(policy_root / "training_metadata.json")
    evaluation = read_json(policy_root / "evaluation.json")
    promotion = read_json(policy_root / "promotion-decision.json")
    experiment_id = training.get("experiment_id")
    experiment = read_json(
        root / "output" / "rl" / "experiments" / str(experiment_id) / "report.json"
    ) if experiment_id else {}
    dataset_contract = training.get("dataset_contract", {})
    metrics = evaluation.get("metrics", {})
    return {
        "schema": "p4_rl_initial_experiment_summary_v1",
        "status": experiment.get("status", "NOT_AVAILABLE"),
        "policy_version": version,
        "training_start": training.get("training_start"),
        "training_end": training.get("training_end"),
        "test_observations": evaluation.get("observations", 0),
        "test_episodes": evaluation.get("episodes", 0),
        "selected_seed": training.get("selected_seed"),
        "selection_rule": training.get("selection_rule"),
        "challenger_metrics": metrics,
        "deterministic_baseline_metrics": experiment.get(
            "deterministic_baseline_evaluation", {}
        ).get("metrics", {}),
        "cost_stress": experiment.get("cost_stress", {}),
        "regime_performance": evaluation.get("regime_performance", {}),
        "performance_by_strategy": evaluation.get(
            "performance_by_strategy",
            {str(dataset_contract.get("strategy_id") or "UNKNOWN"): metrics},
        ),
        "performance_by_asset_class": evaluation.get(
            "performance_by_asset_class",
            {str(dataset_contract.get("asset_class") or "UNKNOWN"): metrics},
        ),
        "bootstrap_probability_of_improvement": experiment.get(
            "bootstrap_probability_of_improvement"
        ),
        "promotion_decision": promotion,
        "historical_oos_is_forward": False,
        "forward_evidence_available": False,
        "execution_authority": "NONE",
        "broker_writes": 0,
    }


def _files_exist(root: Path, paths: tuple[str, ...]) -> bool:
    return all((root / path).is_file() for path in paths)


def _artifact_go(path: Path) -> bool:
    payload = read_json(path)
    expected = payload.get("content_hash")
    body = {key: value for key, value in payload.items() if key != "content_hash"}
    return (
        payload.get("status") == "GO"
        and isinstance(expected, str)
        and expected == stable_hash(body)
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "MANIFEST_PATH",
    "READINESS_PATH",
    "REPORT_PATH",
    "publish_p4_readiness",
]
