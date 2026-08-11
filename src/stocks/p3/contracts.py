from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping


class EvidenceState(StrEnum):
    REJECTED = "REJECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    HISTORICAL_OOS_SUPPORTED = "HISTORICAL_OOS_SUPPORTED"
    MULTIPLE_TESTING_FAILED = "MULTIPLE_TESTING_FAILED"
    COST_FRAGILE = "COST_FRAGILE"
    PARAMETER_FRAGILE = "PARAMETER_FRAGILE"
    REGIME_FRAGILE = "REGIME_FRAGILE"
    FORWARD_REQUIRED = "FORWARD_REQUIRED"
    FORWARD_INSUFFICIENT_SAMPLE = "FORWARD_INSUFFICIENT_SAMPLE"
    FORWARD_SUPPORTED = "FORWARD_SUPPORTED"
    QUALIFICATION_CANDIDATE = "QUALIFICATION_CANDIDATE"
    QUALIFIED = "QUALIFIED"
    PROMOTION_RECOMMENDED = "PROMOTION_RECOMMENDED"
    LIVE_AUTHORIZED = "LIVE_AUTHORIZED"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"


FINAL_EVIDENCE_STATES = frozenset(state.value for state in EvidenceState)


def stable_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


def normalized_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(sorted((value or {}).items(), key=lambda item: str(item[0])))


@dataclass(frozen=True)
class StrategyDNA:
    strategy_id: str
    native_strategy_hash: str
    strategy_family: str
    economic_hypothesis: str
    direction: str
    universe_scope: tuple[str, ...]
    entry_rule: str
    exit_rule: str
    stop_rule: str
    target_rule: str
    time_exit: str
    entry_timeframe: str
    setup_timeframe: str | None
    context_timeframes: tuple[str, ...]
    feature_set: tuple[str, ...]
    parameters: dict[str, Any]
    regime_filter: str
    position_management: str
    cost_model_version: str
    fill_model_version: str
    source_registry: str
    completeness_status: str

    def __post_init__(self) -> None:
        required = {
            "strategy_id": self.strategy_id,
            "native_strategy_hash": self.native_strategy_hash,
            "strategy_family": self.strategy_family,
            "economic_hypothesis": self.economic_hypothesis,
            "direction": self.direction,
            "entry_rule": self.entry_rule,
            "exit_rule": self.exit_rule,
            "stop_rule": self.stop_rule,
            "target_rule": self.target_rule,
            "time_exit": self.time_exit,
            "entry_timeframe": self.entry_timeframe,
            "regime_filter": self.regime_filter,
            "position_management": self.position_management,
            "cost_model_version": self.cost_model_version,
            "fill_model_version": self.fill_model_version,
            "source_registry": self.source_registry,
            "completeness_status": self.completeness_status,
        }
        missing = sorted(name for name, value in required.items() if not str(value).strip())
        if missing:
            raise ValueError(f"STRATEGY_DNA_REQUIRED_FIELDS_MISSING:{'|'.join(missing)}")
        if self.direction not in {"LONG_ONLY", "LONG_SHORT", "SHORT_ONLY"}:
            raise ValueError("STRATEGY_DNA_DIRECTION_INVALID")

    def canonical_spec(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("native_strategy_hash")
        payload.pop("source_registry")
        payload.pop("completeness_status")
        payload["universe_scope"] = sorted(self.universe_scope)
        payload["context_timeframes"] = sorted(self.context_timeframes)
        payload["feature_set"] = sorted(self.feature_set)
        payload["parameters"] = normalized_mapping(self.parameters)
        return payload

    @property
    def strategy_spec_hash(self) -> str:
        return stable_hash(self.canonical_spec())

    def as_record(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "strategy_spec_hash": self.strategy_spec_hash,
            "canonical_spec": self.canonical_spec(),
        }


@dataclass(frozen=True)
class UnifiedTrialRecord:
    source_namespace: str
    source_record_id: str
    research_family: str
    hypothesis_id: str
    strategy_id: str | None
    strategy_spec_hash: str | None
    model_id: str | None
    parameters: dict[str, Any]
    timeframes: tuple[str, ...]
    universe: tuple[str, ...]
    features: tuple[str, ...]
    regime_filter: str
    entry: str
    exit: str
    stop: str
    target: str
    holding_rule: str
    cost_version: str
    fill_model_version: str
    data_hash: str
    cutoff: str
    code_hash: str
    seed: int | None
    created_at: str
    status: str
    rejection_reason: tuple[str, ...]
    metrics: dict[str, Any]
    provenance: dict[str, Any]
    money_control: bool = False
    execution_authority: str = "NONE"

    def __post_init__(self) -> None:
        if not self.source_namespace or not self.source_record_id:
            raise ValueError("TRIAL_SOURCE_ID_REQUIRED")
        if self.money_control or self.execution_authority != "NONE":
            raise ValueError("RESEARCH_TRIAL_MONEY_AUTHORITY_FORBIDDEN")
        if not self.created_at:
            raise ValueError("TRIAL_CREATED_AT_REQUIRED")

    def immutable_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameters"] = normalized_mapping(self.parameters)
        payload["timeframes"] = sorted(self.timeframes)
        payload["universe"] = sorted(self.universe)
        payload["features"] = sorted(self.features)
        payload["rejection_reason"] = sorted(self.rejection_reason)
        payload["metrics"] = normalized_mapping(self.metrics)
        payload["provenance"] = normalized_mapping(self.provenance)
        return payload

    @property
    def trial_hash(self) -> str:
        return stable_hash(self.immutable_payload())

    @property
    def trial_id(self) -> str:
        return f"P3TRIAL-{self.trial_hash[:24]}"

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": "p3_unified_research_trial_v1",
            "trial_id": self.trial_id,
            "trial_hash": self.trial_hash,
            **self.immutable_payload(),
        }
