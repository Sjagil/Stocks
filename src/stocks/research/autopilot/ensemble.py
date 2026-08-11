from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Mapping

import pandas as pd

from stocks.research.autopilot.contracts import stable_hash


class VoteMode(StrEnum):
    CONFIRMATION = "confirmation"
    MAJORITY = "majority"
    WEIGHTED = "weighted"
    UNANIMOUS = "unanimous"
    HIERARCHICAL = "hierarchical"
    SLEEVES = "sleeves"


@dataclass(frozen=True)
class EnsembleSpec:
    ensemble_id: str
    ensemble_hash: str
    version: str
    component_strategy_ids: tuple[str, ...]
    component_families: tuple[str, ...]
    weights: tuple[float, ...]
    vote_mode: str
    vote_threshold: float
    max_strategy_exposure: float
    max_family_exposure: float
    regime_required: bool
    frozen_weights: bool = True
    strategy_authority: str = "NONE"
    execution_authority: str = "NONE"

    def validate(self) -> None:
        count = len(self.component_strategy_ids)
        if count < 2:
            raise ValueError("ENSEMBLE_REQUIRES_AT_LEAST_TWO_STRATEGIES")
        if len(set(self.component_strategy_ids)) != count:
            raise ValueError("DUPLICATE_ENSEMBLE_COMPONENT")
        if len(self.component_families) != count or len(self.weights) != count:
            raise ValueError("ENSEMBLE_COMPONENT_LENGTH_MISMATCH")
        VoteMode(self.vote_mode)
        if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
            raise ValueError("INVALID_ENSEMBLE_WEIGHTS")
        if not 0 < self.vote_threshold <= 1:
            raise ValueError("INVALID_VOTE_THRESHOLD")
        if not 0 < self.max_strategy_exposure <= 1:
            raise ValueError("INVALID_STRATEGY_EXPOSURE")
        if not 0 < self.max_family_exposure <= 1:
            raise ValueError("INVALID_FAMILY_EXPOSURE")
        if not self.frozen_weights:
            raise ValueError("ADAPTIVE_ENSEMBLE_WEIGHTS_FORBIDDEN")
        if self.strategy_authority != "NONE" or self.execution_authority != "NONE":
            raise ValueError("ENSEMBLE_AUTHORITY_NOT_GRANTED")
        payload = asdict(self)
        payload.pop("ensemble_id")
        payload.pop("ensemble_hash")
        if stable_hash(payload) != self.ensemble_hash:
            raise ValueError("ENSEMBLE_HASH_MISMATCH")


def build_ensemble(
    strategy_ids: list[str],
    families: list[str],
    *,
    vote_mode: str = VoteMode.MAJORITY,
    weights: list[float] | None = None,
    vote_threshold: float = 0.5,
    max_strategy_exposure: float = 0.35,
    max_family_exposure: float = 0.60,
    regime_required: bool = True,
) -> EnsembleSpec:
    if weights is None:
        weights = [1.0 / len(strategy_ids)] * len(strategy_ids)
    total = sum(weights)
    normalized = tuple(float(weight / total) for weight in weights)
    core: dict[str, Any] = {
        "version": "1.0.0",
        "component_strategy_ids": tuple(strategy_ids),
        "component_families": tuple(families),
        "weights": normalized,
        "vote_mode": VoteMode(vote_mode).value,
        "vote_threshold": float(vote_threshold),
        "max_strategy_exposure": float(max_strategy_exposure),
        "max_family_exposure": float(max_family_exposure),
        "regime_required": bool(regime_required),
        "frozen_weights": True,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }
    digest = stable_hash(core)
    spec = EnsembleSpec(
        ensemble_id=f"ENSEMBLE-{digest[:20]}",
        ensemble_hash=digest,
        **core,
    )
    spec.validate()
    return spec


def combine_signals(
    signals: Mapping[str, pd.DataFrame],
    spec: EnsembleSpec,
    *,
    regime_gate: pd.DataFrame | None = None,
) -> pd.DataFrame:
    spec.validate()
    missing = set(spec.component_strategy_ids) - set(signals)
    if missing:
        raise ValueError(f"MISSING_ENSEMBLE_SIGNALS:{','.join(sorted(missing))}")
    aligned = [
        signals[strategy_id].clip(lower=-1.0, upper=1.0)
        for strategy_id in spec.component_strategy_ids
    ]
    index = aligned[0].index
    columns = aligned[0].columns
    aligned = [
        frame.reindex(index=index, columns=columns).fillna(0.0)
        for frame in aligned
    ]
    positive = [frame > 0 for frame in aligned]
    negative = [frame < 0 for frame in aligned]
    mode = VoteMode(spec.vote_mode)
    if mode == VoteMode.UNANIMOUS:
        result = sum(positive).eq(len(positive)).astype(float)
    elif mode in {VoteMode.MAJORITY, VoteMode.CONFIRMATION}:
        vote_ratio = sum(positive) / len(positive)
        result = (
            vote_ratio.gt(spec.vote_threshold).astype(float)
            if mode == VoteMode.MAJORITY
            else vote_ratio.ge(spec.vote_threshold).astype(float)
        )
    elif mode == VoteMode.HIERARCHICAL:
        primary = positive[0]
        confirmations = sum(positive[1:]) / max(1, len(positive) - 1)
        result = (primary & confirmations.ge(spec.vote_threshold)).astype(float)
    else:
        component_weights = _capped_component_weights(spec)
        result = sum(
            frame.clip(lower=0.0) * weight
            for frame, weight in zip(aligned, component_weights, strict=True)
        )
        if mode == VoteMode.WEIGHTED:
            result = result.ge(spec.vote_threshold).astype(float)
    # A simultaneous explicit exit/conflict from at least half the components
    # blocks a long target. It never creates a short target.
    conflict_ratio = sum(negative) / len(negative)
    result = result.where(conflict_ratio < 0.5, 0.0).clip(lower=0.0)
    if spec.regime_required:
        if regime_gate is None:
            raise ValueError("ENSEMBLE_REGIME_GATE_REQUIRED")
        result = result.where(
            regime_gate.reindex(index=index, columns=columns).fillna(False),
            0.0,
        )
    row_total = result.sum(axis=1)
    normalized = result.div(row_total.where(row_total > 0), axis=0).fillna(0.0)
    return normalized


def _capped_component_weights(spec: EnsembleSpec) -> tuple[float, ...]:
    weights = [
        min(float(weight), spec.max_strategy_exposure)
        for weight in spec.weights
    ]
    by_family: dict[str, list[int]] = {}
    for index, family in enumerate(spec.component_families):
        by_family.setdefault(family, []).append(index)
    for indices in by_family.values():
        total = sum(weights[index] for index in indices)
        if total > spec.max_family_exposure:
            scale = spec.max_family_exposure / total
            for index in indices:
                weights[index] *= scale
    total = sum(weights)
    return tuple(weight / total for weight in weights)
