from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.shadow.errors import STRATEGY_ACTIVATION_BLOCKED_NO_FINANCIAL_ELIGIBILITY, SYNTHETIC_FIXTURE_ALLOWED
from stocks.shadow.models import model_to_jsonable


ALLOWED_PHASE8_2_STATUSES = {
    "REGISTERED_DISABLED",
    "FIXTURE_ONLY",
    "REJECTED_FINANCIAL",
    "FROZEN_RESEARCH_REFERENCE",
}
DISALLOWED_ACTIVE_STATUSES = {"ACTIVE", "FORWARD_ENABLED", "PAPER_ENABLED", "LIVE_ENABLED"}


@dataclass(frozen=True)
class StrategyContract:
    strategy_id: str
    strategy_version: str
    strategy_hash: str
    status: str
    economic_hypothesis: str
    decision_frequency: str
    required_datasets: tuple[str, ...]
    required_features: tuple[str, ...]
    minimum_history: str
    universe_rule: str
    signal_schema: str
    target_weight_schema: str
    cash_rule: str
    risk_rule_reference: str
    cost_model_reference: str
    evaluation_horizon: str
    authority_required: str
    registered_at: str


def contract_hash(payload: dict[str, Any]) -> str:
    return stable_hash(payload)


def default_contracts(registered_at: str) -> list[StrategyContract]:
    synthetic_core = {
        "strategy_id": "SYNTHETIC_SHADOW_FIXTURE_V1",
        "strategy_version": "1.0",
        "status": "FIXTURE_ONLY",
        "economic_hypothesis": "Synthetic invariant fixture; no financial hypothesis.",
        "decision_frequency": "FIXTURE_MANUAL",
        "required_datasets": ("FROZEN_TOTAL_RETURN_FIXTURE",),
        "required_features": ("fixture_momentum", "fixture_quality"),
        "minimum_history": "2 rows",
        "universe_rule": "deterministic_fixture_universe",
        "signal_schema": "phase8_2_shadow_signal_v1",
        "target_weight_schema": "phase8_2_target_weights_v1",
        "cash_rule": "long_only_cash_residual",
        "risk_rule_reference": "phase8_2_fixture_caps",
        "cost_model_reference": "phase8_2_fixed_fixture_costs",
        "evaluation_horizon": "1 synthetic period",
        "authority_required": "NONE",
        "registered_at": registered_at,
    }
    rejected_core = {
        "strategy_id": "PHASE6_RESEARCH_REFERENCE",
        "strategy_version": "6.4",
        "status": "REJECTED_FINANCIAL",
        "economic_hypothesis": "Frozen Phase 6.4 research reference; no new financial candidate.",
        "decision_frequency": "MONTHLY_AFTER_CLOSE",
        "required_datasets": ("FROZEN_TOTAL_RETURN_CACHE",),
        "required_features": ("momentum", "trend", "risk"),
        "minimum_history": "frozen research sample",
        "universe_rule": "phase6_4_research_registry",
        "signal_schema": "research_reference_only",
        "target_weight_schema": "research_reference_only",
        "cash_rule": "not_active",
        "risk_rule_reference": "not_active",
        "cost_model_reference": "not_active",
        "evaluation_horizon": "not_active",
        "authority_required": "FORWARD_RESEARCH_SHADOW_ELIGIBLE",
        "registered_at": registered_at,
    }
    return [_contract_from_core(synthetic_core), _contract_from_core(rejected_core)]


def _contract_from_core(core: dict[str, Any]) -> StrategyContract:
    hashed = {k: v for k, v in core.items() if k != "registered_at"}
    return StrategyContract(strategy_hash=contract_hash(hashed), **core)


def registry_hash(contracts: list[StrategyContract]) -> str:
    return stable_hash([model_to_jsonable(contract) for contract in sorted(contracts, key=lambda item: item.strategy_id)])


def registry_audit(contracts: list[StrategyContract]) -> dict[str, Any]:
    rows = [model_to_jsonable(contract) for contract in contracts]
    statuses = {row["status"] for row in rows}
    active_hits = sorted(statuses & DISALLOWED_ACTIVE_STATUSES)
    rejected_blocked = all(
        row["status"] in {"REJECTED_FINANCIAL", "FROZEN_RESEARCH_REFERENCE"}
        for row in rows
        if row["strategy_id"].startswith("PHASE6")
    )
    synthetic = next((row for row in rows if row["strategy_id"] == "SYNTHETIC_SHADOW_FIXTURE_V1"), None)
    ok = not active_hits and statuses <= ALLOWED_PHASE8_2_STATUSES and rejected_blocked and synthetic is not None
    return {
        "status": "GO" if ok else "NO_GO",
        "registry_status": "GO" if ok else "NO_GO",
        "strategy_count": len(rows),
        "registered_disabled_or_fixture_count": sum(row["status"] in ALLOWED_PHASE8_2_STATUSES for row in rows),
        "rejected_strategies_blocked": "GO" if rejected_blocked else "NO_GO",
        "synthetic_fixture_status": "SYNTHETIC_FIXTURE_ALLOWED" if synthetic else "MISSING",
        "disallowed_active_status_hits": active_hits,
        "registry_hash": registry_hash(contracts),
        "strategies": rows,
    }


def activation_gate(contract: StrategyContract, eligibility_artifact: dict[str, Any] | None = None) -> dict[str, str]:
    if contract.strategy_id == "SYNTHETIC_SHADOW_FIXTURE_V1" and contract.status == "FIXTURE_ONLY":
        return {"status": "GO", "strategy_id": contract.strategy_id, "decision_code": SYNTHETIC_FIXTURE_ALLOWED}
    if not eligibility_artifact or eligibility_artifact.get("status") != "FORWARD_RESEARCH_SHADOW_ELIGIBLE":
        return {
            "status": "NO_GO",
            "strategy_id": contract.strategy_id,
            "decision_code": STRATEGY_ACTIVATION_BLOCKED_NO_FINANCIAL_ELIGIBILITY,
        }
    return {"status": "NO_GO", "strategy_id": contract.strategy_id, "decision_code": "PHASE8_2_DOES_NOT_ACTIVATE_STRATEGIES"}
