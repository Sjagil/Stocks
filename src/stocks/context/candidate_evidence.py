from __future__ import annotations

from collections.abc import Mapping
from typing import Any


NATURAL_CANDIDATE_UNIT = "ONE_NATURAL_STRATEGY_SETUP"
NATURAL_CANDIDATE_SCOPE = "NATURAL_STRATEGY_CANDIDATE"
CONTEXT_OBSERVATION_SCOPE = "CONTEXT_WATCHLIST_OBSERVATION"


def candidate_evidence_classification(value: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one signal/episode without trusting a self-declared unit alone."""

    setup = _mapping(value.get("setup_snapshot"))
    contract = _mapping(
        value.get("strategy_timeframe_contract")
        or setup.get("strategy_timeframe_contract")
    )
    unit = str(value.get("candidate_unit") or setup.get("candidate_unit") or "")
    setup_id = str(value.get("setup_id") or "").strip()
    candidate_identity = str(value.get("candidate_identity") or "").strip()
    timeframe = str(value.get("timeframe") or "").lower()
    origin = str(value.get("setup_origin_timestamp") or "").strip()
    strategy_id = str(value.get("strategy_id") or "").strip()
    strategy_dna_hash = str(
        value.get("strategy_dna_hash") or setup.get("strategy_dna_hash") or ""
    ).strip()
    evidence_hash = str(
        value.get("timeframe_evidence_hash")
        or setup.get("timeframe_evidence_hash")
        or ""
    ).strip()
    negative_sampling_policy = str(
        value.get("negative_sampling_policy")
        or setup.get("negative_sampling_policy")
        or ""
    ).upper()
    entry_timeframe = str(contract.get("entry_timeframe") or "").lower()
    setup_timeframe = str(contract.get("setup_timeframe") or "").lower()
    required_timeframes = {
        str(item).lower()
        for item in contract.get("required_timeframes", [])
        if str(item).strip()
    }

    blockers: list[str] = []
    if unit != NATURAL_CANDIDATE_UNIT:
        blockers.append("NATURAL_CANDIDATE_UNIT_MISSING")
    if not setup_id:
        blockers.append("STABLE_SETUP_ID_MISSING")
    if not candidate_identity or candidate_identity != setup_id:
        blockers.append("CANDIDATE_IDENTITY_NOT_SETUP_ID")
    if not origin:
        blockers.append("SETUP_ORIGIN_TIMESTAMP_MISSING")
    if not strategy_id or not strategy_dna_hash:
        blockers.append("FROZEN_STRATEGY_HYPOTHESIS_IDENTITY_MISSING")
    if not timeframe or entry_timeframe != timeframe:
        blockers.append("ENTRY_TIMEFRAME_CONTRACT_MISMATCH")
    if (
        contract.get("schema")
        != "active_swing_strategy_timeframe_contract_v1"
        or not setup_timeframe
        or entry_timeframe not in required_timeframes
        or setup_timeframe not in required_timeframes
    ):
        blockers.append("CAUSAL_TIMEFRAME_CONTRACT_INVALID")
    if not evidence_hash:
        blockers.append("CAUSAL_TIMEFRAME_EVIDENCE_HASH_MISSING")
    if negative_sampling_policy != "CANDIDATE_CONDITIONED_ONLY":
        blockers.append("CANDIDATE_CONDITIONED_POLICY_MISSING")

    eligible = not blockers
    return {
        "natural_strategy_candidate": eligible,
        "candidate_conditioned_evidence_eligible": eligible,
        "candidate_unit": unit or None,
        "evidence_scope": (
            NATURAL_CANDIDATE_SCOPE if eligible else CONTEXT_OBSERVATION_SCOPE
        ),
        "candidate_evidence_blockers": blockers,
    }


def is_natural_strategy_candidate(value: Mapping[str, Any]) -> bool:
    return bool(
        candidate_evidence_classification(value)["natural_strategy_candidate"]
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "CONTEXT_OBSERVATION_SCOPE",
    "NATURAL_CANDIDATE_SCOPE",
    "NATURAL_CANDIDATE_UNIT",
    "candidate_evidence_classification",
    "is_natural_strategy_candidate",
]
