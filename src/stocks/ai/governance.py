from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.ai.contracts import (
    AIAuthority,
    ExperimentRecord,
    ExperimentStatus,
    FORBIDDEN_AI_POWERS,
    ModelLifecycle,
    ModelRecord,
    NLPEvent,
    ResearchHypothesis,
)


_RESEARCH_TRANSITIONS: dict[str, set[str]] = {
    ExperimentStatus.PROPOSED: {
        ExperimentStatus.REJECTED_STAGE0,
        ExperimentStatus.VALIDATING,
        ExperimentStatus.FAILED,
    },
    ExperimentStatus.VALIDATING: {
        ExperimentStatus.FAILED,
        ExperimentStatus.RESEARCH_VALIDATED,
        ExperimentStatus.FORWARD_REQUIRED,
    },
    ExperimentStatus.RESEARCH_VALIDATED: {
        ExperimentStatus.FORWARD_REQUIRED,
    },
    ExperimentStatus.FORWARD_REQUIRED: {
        ExperimentStatus.FAILED,
        ExperimentStatus.FORWARD_VALIDATED,
    },
}


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def causal_time_splits(
    index: Sequence[Any] | pd.Index,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    forward_fraction: float = 0.10,
    purge_observations: int = 5,
) -> dict[str, list[int]]:
    """Return ordered, purged positional splits; random splitting is impossible."""

    if purge_observations < 0:
        raise ValueError("purge_observations must be non-negative")
    fractions = (validation_fraction, test_fraction, forward_fraction)
    if any(value <= 0 or value >= 1 for value in fractions):
        raise ValueError("temporal split fractions must be between zero and one")
    if sum(fractions) >= 0.8:
        raise ValueError("at least twenty percent of observations must train")
    timestamps = pd.to_datetime(pd.Index(index), utc=True, errors="raise")
    if len(timestamps) < 25:
        raise ValueError("insufficient observations for five temporal splits")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("temporal index must be ordered and unique")
    count = len(timestamps)
    validation = max(1, int(count * validation_fraction))
    test = max(1, int(count * test_fraction))
    forward = max(1, int(count * forward_fraction))
    live_shadow = max(1, int(count * 0.05))
    train_end = count - validation - test - forward - live_shadow
    if train_end <= purge_observations:
        raise ValueError("insufficient train window after purge")
    starts = {
        "TRAIN": (0, train_end),
        "VALIDATION": (train_end + purge_observations, train_end + validation),
        "TEST": (
            train_end + validation + purge_observations,
            train_end + validation + test,
        ),
        "FORWARD": (
            train_end + validation + test + purge_observations,
            train_end + validation + test + forward,
        ),
        "LIVE_SHADOW": (
            train_end + validation + test + forward + purge_observations,
            count,
        ),
    }
    result = {
        name: list(range(min(start, end), end))
        for name, (start, end) in starts.items()
    }
    if any(not values for values in result.values()):
        raise ValueError("purge gap consumed a required temporal split")
    return result


def validate_point_in_time_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    decision_time: datetime,
) -> dict[str, Any]:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    cutoff = decision_time.astimezone(UTC)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        event_time = _timestamp(row.get("event_time"))
        available_at = _timestamp(row.get("available_at"))
        reasons = []
        if event_time is None or available_at is None:
            reasons.append("CAUSAL_TIMESTAMP_MISSING")
        elif available_at < event_time:
            reasons.append("AVAILABLE_BEFORE_EVENT")
        elif available_at > cutoff:
            reasons.append("NOT_AVAILABLE_AT_DECISION_TIME")
        target = rejected if reasons else accepted
        target.append({**dict(row), "causality_blockers": reasons})
    return {
        "schema": "ai_point_in_time_validation_v1",
        "status": "GO" if not rejected else "PARTIAL",
        "decision_time": cutoff.isoformat(),
        "accepted": accepted,
        "rejected": rejected,
        "lookahead_rows": sum(
            "NOT_AVAILABLE_AT_DECISION_TIME" in row["causality_blockers"]
            or "AVAILABLE_BEFORE_EVENT" in row["causality_blockers"]
            for row in rejected
        ),
    }


def normalize_nlp_events(
    events: Iterable[NLPEvent],
    *,
    decision_time: datetime,
) -> dict[str, Any]:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    cutoff = decision_time.astimezone(UTC)
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: item.available_at):
        reasons = []
        if event.event_id in seen_ids or event.raw_hash in seen_hashes:
            reasons.append("DUPLICATE_EVENT")
        if event.available_at > cutoff:
            reasons.append("LATE_EVENT_NOT_AVAILABLE")
        payload = event.model_dump(mode="json")
        payload["causality_blockers"] = reasons
        (rejected if reasons else accepted).append(payload)
        seen_ids.add(event.event_id)
        seen_hashes.add(event.raw_hash)
    return {
        "schema": "ai_nlp_event_normalization_v1",
        "status": "GO" if not rejected else "PARTIAL",
        "accepted": accepted,
        "rejected": rejected,
        "duplicate_count": sum(
            "DUPLICATE_EVENT" in row["causality_blockers"]
            for row in rejected
        ),
        "late_count": sum(
            "LATE_EVENT_NOT_AVAILABLE" in row["causality_blockers"]
            for row in rejected
        ),
        "standalone_entry_allowed": False,
        "execution_authority": "NONE",
    }


def false_discovery_control(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    ordered = sorted((name, float(value)) for name, value in p_values.items())
    if any(value < 0 or value > 1 for _, value in ordered):
        raise ValueError("p-values must be between zero and one")
    ordered.sort(key=lambda item: item[1])
    count = len(ordered)
    threshold_rank = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= alpha * rank / max(1, count):
            threshold_rank = rank
    discoveries = [name for name, _ in ordered[:threshold_rank]]
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank in range(count, 0, -1):
        name, value = ordered[rank - 1]
        running = min(running, value * count / rank)
        adjusted[name] = min(1.0, running)
    return {
        "schema": "ai_false_discovery_control_v1",
        "method": "BENJAMINI_HOCHBERG",
        "alpha": alpha,
        "hypothesis_count": count,
        "discoveries": discoveries,
        "adjusted_p_values": dict(sorted(adjusted.items())),
    }


def multiple_testing_penalty(
    raw_sharpe: float,
    *,
    hypothesis_count: int,
) -> dict[str, Any]:
    if hypothesis_count < 1:
        raise ValueError("hypothesis_count must be positive")
    penalty = float(np.sqrt(2.0 * np.log(max(1, hypothesis_count))))
    return {
        "schema": "ai_multiple_testing_penalty_v1",
        "raw_sharpe": float(raw_sharpe),
        "hypothesis_count": hypothesis_count,
        "selection_penalty": penalty,
        "penalized_sharpe": float(raw_sharpe - penalty),
        "single_prespecified_experiment": hypothesis_count == 1,
    }


def assess_model_health(
    model: ModelRecord,
    *,
    now: datetime,
    feature_drift: float | None,
    prediction_drift: float | None,
    calibration_drift: float | None,
    performance_drift: float | None,
    regime_drift: float | None,
    schema_matches: bool,
) -> dict[str, Any]:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    checks = {
        "FEATURE_DRIFT": _drift_status(
            feature_drift, model.drift_limits.get("feature", 0.20)
        ),
        "PREDICTION_DRIFT": _drift_status(
            prediction_drift, model.drift_limits.get("prediction", 0.20)
        ),
        "CALIBRATION_DRIFT": _drift_status(
            calibration_drift, model.drift_limits.get("calibration", 0.05)
        ),
        "PERFORMANCE_DRIFT": _drift_status(
            performance_drift, model.drift_limits.get("performance", 0.20)
        ),
        "REGIME_DRIFT": _drift_status(
            regime_drift, model.drift_limits.get("regime", 0.25)
        ),
        "DATA_SCHEMA_DRIFT": "GO" if schema_matches else "BREACH",
    }
    expired = now.astimezone(UTC) >= model.expires_at
    breached = any(value == "BREACH" for value in checks.values())
    missing = any(value == "INSUFFICIENT_BASELINE" for value in checks.values())
    if model.lifecycle in {ModelLifecycle.PAUSED, ModelLifecycle.RETIRED}:
        lifecycle = model.lifecycle
        status = "PAUSED_EXISTING_EVIDENCE_GATE"
    elif expired or breached:
        lifecycle = ModelLifecycle.PAUSED
        status = "PAUSED_FAIL_CLOSED"
    elif missing:
        lifecycle = ModelLifecycle.SHADOW
        status = "SHADOW_INSUFFICIENT_HEALTH_BASELINE"
    else:
        lifecycle = model.lifecycle
        status = "GO"
    return {
        "schema": "ai_model_health_record_v1",
        "model_id": model.model_id,
        "status": status,
        "expired": expired,
        "checks": checks,
        "recommended_lifecycle": str(lifecycle),
        "risk_limit_increase_allowed": False,
        "automatic_model_promotion": False,
        "execution_authority": "NONE",
    }


def validate_ai_authority(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    allowed = {str(value) for value in AIAuthority}
    count = 0
    for row in rows:
        count += 1
        component = str(row.get("component_id") or row.get("model_id") or count)
        if str(row.get("authority")) not in allowed:
            violations.append(f"INVALID_AUTHORITY:{component}")
        if bool(row.get("money_control")):
            violations.append(f"MONEY_CONTROL_FORBIDDEN:{component}")
        if str(row.get("execution_authority", "NONE")) != "NONE":
            violations.append(f"EXECUTION_AUTHORITY_FORBIDDEN:{component}")
        for power in FORBIDDEN_AI_POWERS:
            if bool(row.get(power.lower())) or power in row.get(
                "granted_powers", []
            ):
                violations.append(f"FORBIDDEN_POWER:{component}:{power}")
    return {
        "schema": "ai_authority_validation_v1",
        "status": "GO" if not violations else "NO_GO",
        "component_count": count,
        "violations": sorted(set(violations)),
        "money_control": False,
        "direct_broker_access": False,
        "execution_authority": "NONE",
    }


def transition_hypothesis(
    hypothesis: ResearchHypothesis,
    status: ExperimentStatus,
) -> ResearchHypothesis:
    current = str(hypothesis.status)
    target = str(status)
    if target not in _RESEARCH_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid research transition {current} -> {target}")
    return hypothesis.model_copy(update={"status": target})


def write_immutable_experiment(
    root: Path,
    record: ExperimentRecord,
) -> Path:
    path = root / "output/ai/experiments" / f"{record.experiment_id}.json"
    payload = record.model_dump(mode="json")
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("immutable experiment id already contains other data")
        return path
    _atomic_json(path, payload)
    return path


def audit_ai_import_boundary(ai_root: Path) -> dict[str, Any]:
    violations: list[str] = []
    files = sorted(ai_root.glob("*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    if imported.name.startswith(("stocks.ibkr", "stocks.live")):
                        violations.append(f"{path.name}:import:{imported.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith(("stocks.ibkr", "stocks.live")):
                    violations.append(f"{path.name}:import:{module}")
            elif isinstance(node, ast.Call):
                called = node.func
                name = (
                    called.attr
                    if isinstance(called, ast.Attribute)
                    else called.id
                    if isinstance(called, ast.Name)
                    else ""
                )
                forbidden_calls = {
                    "place" + "Order",
                    "submit_order",
                    "broker_writer",
                }
                if name in forbidden_calls:
                    violations.append(f"{path.name}:call:{name}")
    return {
        "schema": "ai_import_boundary_audit_v1",
        "status": "GO" if not violations else "NO_GO",
        "files_scanned": len(files),
        "violations": violations,
        "broker_imports": 0 if not violations else None,
        "writer_calls": 0 if not violations else None,
    }


def _drift_status(value: float | None, limit: float) -> str:
    if value is None:
        return "INSUFFICIENT_BASELINE"
    return "BREACH" if abs(float(value)) > abs(float(limit)) else "GO"


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        return None
    return parsed.to_pydatetime().astimezone(UTC)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


__all__ = [
    "assess_model_health",
    "audit_ai_import_boundary",
    "canonical_hash",
    "causal_time_splits",
    "false_discovery_control",
    "multiple_testing_penalty",
    "normalize_nlp_events",
    "transition_hypothesis",
    "validate_ai_authority",
    "validate_point_in_time_rows",
    "write_immutable_experiment",
]
