from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.p3.io import atomic_write_json
from stocks.rl.contracts import stable_hash


AUDIT_PATH = Path("output/verification/p4-rl-requirement-audit.json")
REPORT_PATH = Path("reports/P4_RL_REQUIREMENT_AUDIT.md")


def publish_requirement_audit(
    project_root: Path,
    readiness: dict[str, Any],
    rl_status: dict[str, Any],
) -> dict[str, Any]:
    """Publish an explicit implementation/evidence split for all 43 requirements."""
    root = project_root.resolve()
    requirements = _requirements(
        external=readiness.get("economic_and_external_gates", {}),
        rl_status=rl_status,
    )
    evidence_paths = sorted(
        {path for item in requirements for path in item["evidence"]}
    )
    missing_evidence_paths = [
        path for path in evidence_paths if not (root / path).exists()
    ]
    audit: dict[str, Any] = {
        "schema": "p4_rl_requirement_audit_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "requirement_count": len(requirements),
        "all_requirements_mapped": len(requirements) == 43,
        "evidence_paths_verified": not missing_evidence_paths,
        "missing_evidence_paths": missing_evidence_paths,
        "implementation_counts": dict(
            sorted(Counter(item["implementation_status"] for item in requirements).items())
        ),
        "evidence_counts": dict(
            sorted(Counter(item["evidence_status"] for item in requirements).items())
        ),
        "production_ready": False,
        "profitability_proven": False,
        "rl_live_enabled": False,
        "execution_authority": "NONE",
        "broker_writes": 0,
        "valid_conclusion": (
            "INTERNAL_P4_RL_IMPLEMENTED_EXTERNAL_DATA_AND_NATURAL_FORWARD_EVIDENCE_INCOMPLETE"
        ),
        "requirements": requirements,
    }
    audit["content_hash"] = stable_hash(audit)
    atomic_write_json(root / AUDIT_PATH, audit)
    _write_report(root, audit)
    return audit


def _entry(
    number: int,
    title: str,
    implementation_status: str,
    evidence_status: str,
    evidence: list[str],
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": title,
        "implementation_status": implementation_status,
        "evidence_status": evidence_status,
        "evidence": evidence,
        "blockers": blockers or [],
    }


def _requirements(
    *, external: dict[str, Any], rl_status: dict[str, Any]
) -> list[dict[str, Any]]:
    pit_blockers = [
        gate
        for gate in ("PIT_DATA_GO", "SURVIVORSHIP_GO", "SHARIAH_PIT_GO")
        if not external.get(gate, False)
    ]
    forward_blockers = [
        gate
        for gate in (
            "FORWARD_EVIDENCE_GO",
            "RL_INCREMENTAL_EVIDENCE_GO",
            "RL_FORWARD_EVIDENCE_GO",
            "RL_POLICY_PROMOTION_GO",
        )
        if not external.get(gate, False)
    ]
    closed_episodes = int(rl_status.get("closed_episodes", 0))
    forward_sample = "FORWARD_SAMPLE_AVAILABLE" if closed_episodes else "FORWARD_SAMPLE_EMPTY"
    no_closed = [] if closed_episodes else ["NO_CLOSED_FORWARD_EPISODES"]
    return [
        _entry(1, "Constrained RL decision layer", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/environment.py", "tests/test_rl_decision_layer.py"]),
        _entry(2, "Canonical MaskablePPO baseline", "IMPLEMENTED_AND_TESTED", "INITIAL_EXPERIMENT_COMPLETE", ["src/stocks/rl/training.py", "config/rl_training_v1.json"]),
        _entry(3, "Causal swing environment", "IMPLEMENTED_AND_TESTED", "PARTIAL_DATA_COVERAGE", ["src/stocks/rl/environment.py", "src/stocks/rl/data.py"], ["15M_SOURCE_DATA_MISSING"]),
        _entry(4, "Normalized observations and missingness", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/data.py", "models/rl/registry.json"]),
        _entry(5, "Small masked action space", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/contracts.py", "src/stocks/rl/environment.py"]),
        _entry(6, "Decomposed risk-aware reward", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/reward.py", "config/rl_reward_v1.json"]),
        _entry(7, "Positive and negative reward semantics", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["tests/test_rl_decision_layer.py"]),
        _entry(8, "Asymmetric loss and convex drawdown", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/reward.py", "tests/test_rl_decision_layer.py"]),
        _entry(9, "Central reward configuration", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["config/rl_reward_v1.json", "src/stocks/rl/reward.py"]),
        _entry(10, "Reward-hacking and terminal MTM controls", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/environment.py", "tests/test_rl_decision_layer.py"]),
        _entry(11, "Top-N opportunity selection", "IMPLEMENTED_AND_TESTED", "NOT_YET_ECONOMICALLY_EVALUABLE", ["src/stocks/rl/environment.py"], ["QUALIFIED_PIT_OPPORTUNITY_HISTORY_REQUIRED"]),
        _entry(12, "Extensible single-agent architecture", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/contracts.py", "src/stocks/rl/environment.py"]),
        _entry(13, "Chronological purged walk-forward", "IMPLEMENTED_AND_TESTED", "HISTORICAL_OOS_ONLY", ["src/stocks/rl/training.py"], ["HISTORICAL_OOS_IS_NOT_FORWARD"]),
        _entry(14, "Regime diversity and reporting", "IMPLEMENTED_AND_TESTED", "INSUFFICIENT_EVIDENCE", ["src/stocks/rl/training.py"], ["MACRO_REGIME_HISTORY_MISSING"]),
        _entry(15, "Five mandatory baselines", "IMPLEMENTED_AND_TESTED", "INITIAL_EXPERIMENT_COMPLETE", ["src/stocks/rl/evaluation.py"]),
        _entry(16, "Costs and 1x-2x stress", "IMPLEMENTED_AND_TESTED", "INITIAL_EXPERIMENT_COMPLETE", ["src/stocks/rl/environment.py", "src/stocks/rl/evaluation.py"]),
        _entry(17, "Continual frozen-policy operation", "IMPLEMENTED_AND_TESTED", "SHADOW_ONLY", ["src/stocks/rl/supervisor.py", "src/stocks/operations/service.py"]),
        _entry(18, "Persistent experience store", "IMPLEMENTED_AND_TESTED", forward_sample, ["src/stocks/rl/experience.py"], no_closed),
        _entry(19, "Active/challenger separation", "IMPLEMENTED_AND_TESTED", "CHALLENGER_ONLY", ["src/stocks/rl/registry.py", "src/stocks/rl/supervisor.py"]),
        _entry(20, "Fail-closed promotion gate", "IMPLEMENTED_AND_TESTED", "CHALLENGER_REJECTED", ["src/stocks/rl/evaluation.py", "src/stocks/rl/registry.py", "config/rl_promotion_v1.json"]),
        _entry(21, "Shadow-first deployment", "IMPLEMENTED_AND_TESTED", "SHADOW_ONLY", ["config/rl_policy_v1.json", "output/rl/status.json"]),
        _entry(22, "Paper-stage boundary", "IMPLEMENTED", "NOT_YET_ELIGIBLE", ["config/rl_policy_v1.json"], ["SHADOW_EVIDENCE_INSUFFICIENT"]),
        _entry(23, "Live RL disabled by default", "IMPLEMENTED_AND_TESTED", "SAFETY_GATE_GO", ["config/rl_policy_v1.json", "tests/test_rl_decision_layer.py"]),
        _entry(24, "Restart-safe supervisor", "IMPLEMENTED_AND_TESTED", "RUNTIME_ACTIVE", ["src/stocks/rl/supervisor.py", "src/stocks/operations/service.py"]),
        _entry(25, "Step and terminal reward attribution", "IMPLEMENTED_AND_TESTED", forward_sample, ["src/stocks/rl/environment.py", "src/stocks/rl/experience.py"], no_closed),
        _entry(26, "Trade episode and portfolio context", "IMPLEMENTED_AND_TESTED", "NOT_YET_FORWARD_EVALUABLE", ["src/stocks/rl/environment.py"], no_closed),
        _entry(27, "Reward explainability", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/reward.py", "src/stocks/rl/experience.py"]),
        _entry(28, "Policy diagnostics", "IMPLEMENTED_AND_TESTED", "SHADOW_ONLY", ["src/stocks/rl/supervisor.py", "output/rl/status.json"]),
        _entry(29, "Anti-overfitting controls", "IMPLEMENTED_AND_TESTED", "INSUFFICIENT_SAMPLE", ["src/stocks/rl/training.py", "src/stocks/rl/evaluation.py"], ["MINIMUM_TRADE_COUNT", "MINIMUM_EPISODE_COUNT", "BOOTSTRAP_CONFIDENCE"]),
        _entry(30, "Catastrophic-forgetting controls", "IMPLEMENTED_AND_TESTED", "PARTIAL_REGIME_COVERAGE", ["config/rl_training_v1.json", "src/stocks/rl/training.py"], ["MACRO_REGIME_HISTORY_MISSING"]),
        _entry(31, "Immutable policy registry", "IMPLEMENTED_AND_TESTED", "CHALLENGER_REGISTERED", ["src/stocks/rl/registry.py", "models/rl/registry.json"]),
        _entry(32, "Dashboard and regime status", "IMPLEMENTED_AND_TESTED", "SHADOW_ONLY", ["src/stocks/ui/service.py", "src/stocks/ui/templates/research.html"]),
        _entry(33, "Health alerts and fail-closed fallback", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["src/stocks/rl/supervisor.py"]),
        _entry(34, "Required automated tests", "IMPLEMENTED_AND_TESTED", "INTERNAL_EVIDENCE_GO", ["tests/test_rl_decision_layer.py", "tests/test_p4_data_contract.py"]),
        _entry(35, "Initial multi-seed experiment", "IMPLEMENTED_AND_TESTED", "CHALLENGER_REJECTED", ["models/rl/registry.json", "output/verification/p4-readiness.json"]),
        _entry(36, "Asset-class extensible; ETF first", "IMPLEMENTED_AND_TESTED", "ETF_RESEARCH_ONLY", ["src/stocks/rl/contracts.py", "output/verification/p4-readiness.json"]),
        _entry(37, "Active-swing multi-timeframe focus", "IMPLEMENTED_AND_TESTED", "PARTIAL_DATA_COVERAGE", ["src/stocks/rl/data.py"], ["15M_SOURCE_DATA_MISSING"]),
        _entry(38, "Cash/skip and collapse monitoring", "IMPLEMENTED_AND_TESTED", "SHADOW_ONLY", ["src/stocks/rl/environment.py", "src/stocks/rl/supervisor.py"]),
        _entry(39, "Incremental value by decision function", "INTERFACES_IMPLEMENTED", "NOT_YET_EVALUABLE", ["src/stocks/rl/environment.py", "src/stocks/rl/evaluation.py"], forward_blockers or ["FORWARD_INCREMENTAL_ABLATIONS_REQUIRED"]),
        _entry(40, "No premature RL complexity", "IMPLEMENTED_AND_TESTED", "SCOPE_CONTROL_GO", ["config/rl_training_v1.json"]),
        _entry(41, "End-to-end evidence architecture", "IMPLEMENTED_AND_TESTED", "SHADOW_ONLY", ["src/stocks/rl/supervisor.py", "src/stocks/operations/service.py"]),
        _entry(42, "Hard controls remain outside RL", "IMPLEMENTED_AND_TESTED", "SAFETY_GATE_GO", ["src/stocks/rl/environment.py", "src/stocks/rl/supervisor.py"]),
        _entry(43, "Executed implementation and truthful report", "IMPLEMENTED_AND_TESTED", "P4_ACTIVE_EVIDENCE_INCOMPLETE", ["output/verification/p4-readiness.json", "reports/P4_DATA_FORWARD_RL_STATUS.md"], pit_blockers + forward_blockers),
    ]


def _write_report(root: Path, audit: dict[str, Any]) -> None:
    rows = [
        "# P4 + RL Requirement Audit",
        "",
        f"Generated: `{audit['generated_at']}`",
        "",
        "Implementation and economic evidence are separate. Passing tests or training is not proof of profitability or production readiness.",
        "",
        f"- Requirements mapped: `{audit['requirement_count']}/43`",
        f"- Evidence paths verified: `{audit['evidence_paths_verified']}`",
        f"- Production ready: `{audit['production_ready']}`",
        f"- Profitability proven: `{audit['profitability_proven']}`",
        f"- RL live enabled: `{audit['rl_live_enabled']}`",
        f"- Execution authority: `{audit['execution_authority']}`",
        f"- Broker writes: `{audit['broker_writes']}`",
        "",
        "| # | Requirement | Implementation | Evidence | Blockers |",
        "|---:|---|---|---|---|",
    ]
    for item in audit["requirements"]:
        blockers = ", ".join(item["blockers"]) if item["blockers"] else "-"
        rows.append(
            f"| {item['number']} | {item['title']} | {item['implementation_status']} | "
            f"{item['evidence_status']} | {blockers} |"
        )
    rows += [
        "",
        "## Current valid conclusion",
        "",
        "The internal P4/RL system is implemented and runs in shadow mode. Production data attestation, real-time quote evidence, statistically sufficient natural forward outcomes, incremental RL ablations, portfolio qualification and a paper canary remain required. Cash and zero RL execution authority are therefore the correct current outcome.",
        "",
    ]
    (root / REPORT_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT_PATH).write_text("\n".join(rows), encoding="utf-8")


__all__ = ["AUDIT_PATH", "REPORT_PATH", "publish_requirement_audit"]
