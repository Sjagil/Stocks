from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.p3.io import atomic_write_json, file_hash, read_json
from stocks.rl.contracts import stable_hash


REGISTRATION_PATH = Path("output/p4/preregistered-forward-cohort.json")
PROTOCOL_PATH = Path("output/p4/frozen-forward-evaluation-protocol.json")


def preregister_phase11_14_candidates(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    boundary_path = root / "output/research/phase11_14/qualification-boundary.json"
    qualification_path = root / "output/research/phase11_14/qualification.json"
    if not boundary_path.is_file() or not qualification_path.is_file():
        return _blocked("PHASE11_14_FROZEN_COHORT_MISSING")
    boundary = read_json(boundary_path)
    qualification = read_json(qualification_path)
    if boundary.get("status") != "FROZEN":
        return _blocked("PHASE11_14_BOUNDARY_NOT_FROZEN")
    robust_ids = sorted(str(value) for value in boundary.get("robust_strategy_ids", []))
    strategies = {
        str(row.get("strategy_id")): row
        for row in qualification.get("strategies", [])
        if isinstance(row, dict) and row.get("strategy_id") in robust_ids
    }
    cohort = []
    for strategy_id in robust_ids:
        row = strategies.get(strategy_id, {})
        cohort.append(
            {
                "strategy_id": strategy_id,
                "source_strategy_id": row.get("source_strategy_id"),
                "formula": row.get("formula"),
                "timeframe": row.get("timeframe"),
                "asset_class": row.get("asset_class"),
                "frozen_profile": row.get("frozen_profile"),
                "data_end": boundary.get("data_end_by_strategy", {}).get(strategy_id),
                "parameters_mutable": False,
                "forced_trade_allowed": False,
                "automatic_promotion": False,
            }
        )
    semantics = {
        "source_boundary_hash": file_hash(boundary_path),
        "source_qualification_hash": file_hash(qualification_path),
        "original_frozen_at": boundary.get("frozen_at"),
        "cohort": cohort,
        "forward_only_after_original_data_end": True,
        "historical_backfill_counts_as_forward": False,
        "every_change_is_new_hypothesis": True,
    }
    registration_hash = stable_hash(semantics)
    payload: dict[str, Any] = {
        "schema": "p4_preregistered_forward_cohort_v1",
        "status": "FROZEN",
        "registration_hash": registration_hash,
        "registered_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(cohort),
        **semantics,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "money_control": False,
    }
    destination = root / REGISTRATION_PATH
    if destination.is_file():
        existing = read_json(destination)
        if existing.get("registration_hash") != registration_hash:
            return _blocked("P4_FORWARD_REGISTRATION_IMMUTABILITY_MISMATCH")
        return existing
    atomic_write_json(destination, payload)
    return payload


def freeze_forward_evaluation_protocol(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    registration = preregister_phase11_14_candidates(root)
    policy_path = root / "config/p4_forward_evaluation_v1.json"
    policy = read_json(policy_path)
    if registration.get("status") != "FROZEN":
        return _blocked("P4_FORWARD_COHORT_NOT_FROZEN")
    if policy.get("status") != "FROZEN":
        return _blocked("P4_FORWARD_EVALUATION_POLICY_NOT_FROZEN")
    candidate_protocols = [
        {
            "strategy_id": row["strategy_id"],
            "timeframe": row.get("timeframe"),
            "asset_class": row.get("asset_class"),
            "baselines": list(policy.get("candidate_baselines", [])),
            "comparison_clock": policy.get("comparison_clock"),
        }
        for row in registration.get("cohort", [])
    ]
    semantics = {
        "registration_hash": registration.get("registration_hash"),
        "policy_hash": file_hash(policy_path),
        "candidate_protocols": candidate_protocols,
        "policy": policy,
    }
    protocol_hash = stable_hash(semantics)
    payload = {
        "schema": "p4_frozen_forward_evaluation_protocol_v1",
        "status": "FROZEN",
        "protocol_hash": protocol_hash,
        "frozen_at": datetime.now(UTC).isoformat(),
        **semantics,
        "execution_authority": "NONE",
        "broker_writes": 0,
        "money_control": False,
    }
    destination = root / PROTOCOL_PATH
    if destination.is_file():
        existing = read_json(destination)
        if existing.get("protocol_hash") != protocol_hash:
            return _blocked("P4_FORWARD_PROTOCOL_IMMUTABILITY_MISMATCH")
        return existing
    atomic_write_json(destination, payload)
    return payload


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "p4_preregistered_forward_cohort_v1",
        "status": "BLOCKED",
        "blockers": [reason],
        "execution_authority": "NONE",
        "broker_writes": 0,
    }


__all__ = [
    "PROTOCOL_PATH",
    "REGISTRATION_PATH",
    "freeze_forward_evaluation_protocol",
    "preregister_phase11_14_candidates",
]
