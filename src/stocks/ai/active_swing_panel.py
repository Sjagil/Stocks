from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.context.candidate_evidence import (
    NATURAL_CANDIDATE_UNIT,
    candidate_evidence_classification,
)
from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.p3.io import atomic_write_json, read_json


EPISODES_PATH = Path("data/market_context/private/entry-episodes.jsonl")
OUTCOMES_PATH = Path("data/market_context/private/entry-episode-outcomes.jsonl")
PANEL_PATH = Path("data/ai/private/active-swing-candidate-panel.parquet")
STATUS_PATH = Path("output/ai/decision-intelligence/active-swing-panel-status.json")
INFERENCE_PATH = Path(
    "output/ai/decision-intelligence/current-active-swing-inference.json"
)
CURRENT_CANDIDATES_PATH = Path("output/signals/active_swing_15m_signals.json")
TIMEFRAMES = ("15m", "1h", "2h", "4h", "6h", "12h", "1d", "1w")

IDENTITY_COLUMNS = (
    "candidate_identity",
    "candidate_unit",
    "episode_id",
    "feature_snapshot_hash",
    "timeframe_evidence_hash",
    "symbol",
    "security_id",
    "strategy_id",
    "strategy_family",
    "strategy_dna_hash",
    "fold_id",
    "decision_timestamp",
    "feature_timestamp",
    "label_available_at",
)
CONTRACT_COLUMNS = (
    "entry_timeframe",
    "setup_timeframe",
    "structural_timeframe",
    "management_timeframe",
    "exit_timeframe",
    "context_architecture",
    "asset_class",
    "market_regime",
)
SCALAR_FEATURE_COLUMNS = (
    "setup_score_within_family",
    "entry_confirmation_score",
    "asset_bias_score",
    "asset_bias_confidence",
    "event_risk_score",
    "asset_profile_score",
    "asset_profile_coverage",
    "reward_risk_1",
    "estimated_transaction_costs_eur",
)
TARGET_COLUMNS = (
    "native_exit_net_return",
    "positive_net_trade",
    "gross_R",
    "maximum_favourable_excursion_R",
    "maximum_adverse_excursion_R",
    "holding_duration_seconds",
    "first_barrier_hit",
)
TIMEFRAME_FEATURE_COLUMNS = tuple(
    name
    for timeframe in TIMEFRAMES
    for name in (
        f"has_{timeframe}",
        f"return_1bar_{timeframe}",
        f"momentum_4bars_{timeframe}",
        f"distance_ema20_{timeframe}",
        f"distance_ema50_{timeframe}",
        f"trend_state_{timeframe}",
    )
)
INTERACTION_FEATURE_COLUMNS = (
    "interaction_momentum_15m_x_momentum_1h",
    "interaction_pullback_1h_x_trend_4h",
    "interaction_momentum_4h_x_momentum_1d",
)
PANEL_COLUMNS = (
    *IDENTITY_COLUMNS,
    *CONTRACT_COLUMNS,
    *SCALAR_FEATURE_COLUMNS,
    *TIMEFRAME_FEATURE_COLUMNS,
    *INTERACTION_FEATURE_COLUMNS,
    *TARGET_COLUMNS,
)


def build_active_swing_candidate_panel(
    project_root: Path,
    *,
    publish: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one causal row per naturally emitted active-swing setup."""

    root = project_root.resolve()
    episodes, malformed_episodes = _read_jsonl(root / EPISODES_PATH)
    outcomes, malformed_outcomes = _read_jsonl(root / OUTCOMES_PATH)
    rejection_counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for episode in episodes:
        reason = _episode_rejection_reason(episode)
        if reason:
            rejection_counts[reason] += 1
        else:
            eligible.append(episode)

    eligible.sort(key=lambda row: _timestamp(row.get("decision_timestamp")))
    deduplicated: dict[str, dict[str, Any]] = {}
    for episode in eligible:
        identity = str(episode["candidate_identity"])
        if identity in deduplicated:
            rejection_counts["DUPLICATE_CANDIDATE_OBSERVATION"] += 1
            continue
        deduplicated[identity] = episode

    outcomes_by_episode: dict[str, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        outcomes_by_episode.setdefault(str(outcome.get("episode_id") or ""), []).append(
            outcome
        )

    rows: list[dict[str, Any]] = []
    for episode in deduplicated.values():
        matches = outcomes_by_episode.get(str(episode["episode_id"]), [])
        valid_matches = [
            outcome
            for outcome in matches
            if _outcome_rejection_reason(episode, outcome) is None
        ]
        for outcome in matches:
            reason = _outcome_rejection_reason(episode, outcome)
            if reason:
                rejection_counts[reason] += 1
        if not valid_matches:
            rejection_counts["NO_VALID_TERMINAL_PERFORMANCE_OUTCOME"] += 1
            continue
        if len(valid_matches) > 1:
            rejection_counts["DUPLICATE_VALID_OUTCOME"] += 1
            continue
        rows.append(_panel_row(episode, valid_matches[0]))

    panel = pd.DataFrame(rows, columns=PANEL_COLUMNS)
    if not panel.empty:
        panel = panel.sort_values(
            ["decision_timestamp", "candidate_identity"]
        ).reset_index(drop=True)
        if panel["candidate_identity"].duplicated().any():
            raise ValueError("active-swing panel contains duplicate candidate identities")
        if not panel["candidate_unit"].eq(NATURAL_CANDIDATE_UNIT).all():
            raise ValueError("active-swing panel contains a non-natural candidate unit")
        if not (
            pd.to_datetime(panel["feature_timestamp"], utc=True)
            <= pd.to_datetime(panel["decision_timestamp"], utc=True)
        ).all():
            raise ValueError("active-swing panel contains feature lookahead")
        if not (
            pd.to_datetime(panel["label_available_at"], utc=True)
            >= pd.to_datetime(panel["decision_timestamp"], utc=True)
        ).all():
            raise ValueError("active-swing panel contains premature labels")

    positives = int(panel["positive_net_trade"].sum()) if not panel.empty else 0
    negatives = len(panel) - positives
    dependence = _sample_dependence_audit(panel)
    strategy_clustering = _strategy_clustering_audit(panel)
    redundancy = _timeframe_redundancy_audit(panel)
    training_blockers: list[str] = []
    if len(panel) < 500:
        training_blockers.append("FEWER_THAN_500_NATURAL_LABELED_CANDIDATES")
    if positives < 100:
        training_blockers.append("FEWER_THAN_100_POSITIVE_CANDIDATES")
    if negatives < 100:
        training_blockers.append("FEWER_THAN_100_NONPOSITIVE_CANDIDATES")
    if (int(panel["decision_timestamp"].nunique()) if not panel.empty else 0) < 50:
        training_blockers.append("FEWER_THAN_50_DISTINCT_DECISION_TIMESTAMPS")
    if dependence["independent_market_time_bucket_count"] < 100:
        training_blockers.append("FEWER_THAN_100_INDEPENDENT_MARKET_TIME_BUCKETS")
    if dependence["effective_sample_size"] < 100:
        training_blockers.append("EFFECTIVE_SAMPLE_SIZE_BELOW_100")

    status: dict[str, Any] = {
        "schema": "active_swing_candidate_ml_panel_status_v1",
        "status": "TRAINING_PANEL_GO" if not training_blockers else "INSUFFICIENT_FORWARD_EVIDENCE",
        "generated_at": datetime.now(UTC).isoformat(),
        "panel_path": PANEL_PATH.as_posix(),
        "episode_source": EPISODES_PATH.as_posix(),
        "outcome_source": OUTCOMES_PATH.as_posix(),
        "candidate_source": "NATURAL_ACTIVE_SWING_FORWARD_OBSERVER",
        "candidate_unit": NATURAL_CANDIDATE_UNIT,
        "prediction_unit": "ONE_CANDIDATE_PER_PREDICTION",
        "negative_sampling_policy": "CANDIDATE_CONDITIONED_ONLY",
        "row_count": len(panel),
        "positive_count": positives,
        "nonpositive_count": negatives,
        "class_balance_positive": (
            round(positives / len(panel), 8) if len(panel) else None
        ),
        "distinct_candidate_count": (
            int(panel["candidate_identity"].nunique()) if not panel.empty else 0
        ),
        "distinct_decision_timestamp_count": (
            int(panel["decision_timestamp"].nunique()) if not panel.empty else 0
        ),
        "strategy_count": int(panel["strategy_id"].nunique()) if not panel.empty else 0,
        "architecture_count": (
            int(panel["context_architecture"].nunique()) if not panel.empty else 0
        ),
        "eligible_episode_count_before_deduplication": len(eligible),
        "deduplicated_candidate_count": len(deduplicated),
        "malformed_episode_line_count": malformed_episodes,
        "malformed_outcome_line_count": malformed_outcomes,
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "sample_dependence": dependence,
        "strategy_clustering": strategy_clustering,
        "timeframe_redundancy": redundancy,
        "active_swing_candidate_unit_go": bool(len(panel)),
        "candidate_identity_deduplicated": True,
        "causal_snapshot_validation": "STRICT_GO" if len(panel) else "NO_ELIGIBLE_ROWS_YET",
        "training_ready": not training_blockers,
        "training_blockers": training_blockers,
        "numeric_features": [*SCALAR_FEATURE_COLUMNS, *[name for name in TIMEFRAME_FEATURE_COLUMNS if not name.startswith("trend_state_")], *INTERACTION_FEATURE_COLUMNS],
        "categorical_features": [*CONTRACT_COLUMNS, *[name for name in TIMEFRAME_FEATURE_COLUMNS if name.startswith("trend_state_")], "symbol", "strategy_id", "strategy_family", "strategy_dna_hash"],
        "targets": list(TARGET_COLUMNS),
        "target_rule": "VALID_TERMINAL_POST_DECISION_COUNTERFACTUAL_BAR_PATH_ONLY",
        "point_in_time_universe_complete": False,
        "shariah_history_complete": False,
        "production_training_allowed": False,
        "automatic_promotion": False,
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "source_hashes": _source_hashes(root),
    }
    if publish:
        destination = root / PANEL_PATH
        _atomic_parquet(destination, panel)
        status["panel_sha256"] = sha256_file(destination).upper()
        status["content_hash"] = stable_hash(status)
        atomic_write_json(root / STATUS_PATH, status)
    return panel, status


def infer_current_active_swing_candidates(
    project_root: Path,
    bundle: Mapping[str, Any],
    tournament: Mapping[str, Any],
    *,
    publish: bool = True,
) -> dict[str, Any]:
    """Fail closed unless a model was trained on the exact natural candidate unit."""

    root = project_root.resolve()
    payload = read_json(root / CURRENT_CANDIDATES_PATH)
    rows = payload.get("candidates") or payload.get("signals") or []
    candidates: dict[str, Mapping[str, Any]] = {}
    rejected = 0
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            rejected += 1
            continue
        classification = candidate_evidence_classification(row)
        identity = str(row.get("candidate_identity") or "")
        if not classification["natural_strategy_candidate"] or not identity:
            rejected += 1
            continue
        candidates.setdefault(identity, row)
    compatible = bundle.get("candidate_unit") == NATURAL_CANDIDATE_UNIT
    evidence: list[dict[str, Any]] = []
    missing_observations: list[str] = []
    if candidates and compatible:
        from stocks.ai.active_swing_modeling import predict_active_swing_bundle

        episodes, _ = _read_jsonl(root / EPISODES_PATH)
        by_identity: dict[str, Mapping[str, Any]] = {}
        for episode in episodes:
            identity = str(episode.get("candidate_identity") or "")
            if identity in candidates and identity not in by_identity:
                if _episode_rejection_reason(episode) is None:
                    by_identity[identity] = episode
        feature_rows: list[dict[str, Any]] = []
        feature_episodes: list[Mapping[str, Any]] = []
        for identity in candidates:
            episode = by_identity.get(identity)
            if episode is None:
                missing_observations.append(identity)
                continue
            feature_rows.append(active_swing_feature_row(episode))
            feature_episodes.append(episode)
        if feature_rows:
            features = pd.DataFrame(feature_rows)
            predictions = predict_active_swing_bundle(bundle, features)
            training_identities = set(bundle.get("training_candidate_identities", ()))
            for index, episode in enumerate(feature_episodes):
                feature = features.iloc[index]
                prediction = predictions.iloc[index]
                identity = str(episode["candidate_identity"])
                evidence.append(
                    {
                        "schema": "active_swing_candidate_model_evidence_v1",
                        "evidence_id": stable_hash(
                            {
                                "model": bundle.get("model_version"),
                                "candidate_identity": identity,
                                "feature_snapshot_hash": episode.get(
                                    "feature_snapshot_hash"
                                ),
                            }
                        )[:32],
                        "model_version": bundle.get("model_version"),
                        "candidate_identity": identity,
                        "candidate_unit": NATURAL_CANDIDATE_UNIT,
                        "symbol": feature["symbol"],
                        "strategy_id": feature["strategy_id"],
                        "strategy_dna_hash": feature["strategy_dna_hash"],
                        "decision_timestamp": feature["decision_timestamp"],
                        "feature_timestamp": feature["feature_timestamp"],
                        "feature_snapshot_hash": episode.get(
                            "feature_snapshot_hash"
                        ),
                        "probability_positive_net": float(
                            prediction["probability_positive_net"]
                        ),
                        "predicted_net_R": float(prediction["predicted_net_R"]),
                        "conservative_expected_R": float(
                            prediction["conservative_expected_R"]
                        ),
                        "uncertainty": float(prediction["uncertainty"]),
                        "abstained": bool(prediction["abstained"]),
                        "candidate_seen_in_training": identity in training_identities,
                        "validation_status": tournament.get("promotion_status"),
                        "financial_fields_mutated": False,
                        "authority": "SHADOW_ONLY",
                        "execution_authority": "NONE",
                        "broker_writes": 0,
                    }
                )
    status = (
        "NO_CURRENT_NATURAL_CANDIDATES"
        if not candidates
        else "NO_COMPATIBLE_ACTIVE_SWING_MODEL"
        if not compatible
        else "SHADOW_EVIDENCE_AVAILABLE_NOT_PROMOTED"
        if evidence
        else "AWAITING_EXACT_CANDIDATE_OBSERVATION"
    )
    result = {
        "schema": "current_active_swing_candidate_inference_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": status,
        "candidate_unit": NATURAL_CANDIDATE_UNIT,
        "current_candidate_count": len(candidates),
        "rejected_or_context_row_count": rejected,
        "candidate_identities": sorted(candidates),
        "model_version": bundle.get("model_version"),
        "model_candidate_unit": bundle.get("candidate_unit") or "LEGACY_OR_UNDECLARED",
        "compatible_model": compatible,
        "exact_candidate_observation_missing": sorted(missing_observations),
        "evidence_count": len(evidence),
        "model_evidence": evidence,
        "promotion_status": tournament.get("promotion_status"),
        "financial_fields_mutated": False,
        "automatic_promotion": False,
        "authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }
    result["content_hash"] = stable_hash(result)
    if publish:
        atomic_write_json(root / INFERENCE_PATH, result)
    return result


def _episode_rejection_reason(episode: Mapping[str, Any]) -> str | None:
    if episode.get("schema") != "active_swing_forward_episode_v1":
        return "NONCANONICAL_EPISODE_SCHEMA"
    classification = candidate_evidence_classification(episode)
    if not classification["natural_strategy_candidate"]:
        return "CONTEXT_OR_INVALID_CANDIDATE"
    required = (
        "episode_id",
        "candidate_identity",
        "feature_snapshot_hash",
        "decision_timestamp",
        "strategy_dna_hash",
    )
    if any(not str(episode.get(name) or "").strip() for name in required):
        return "REQUIRED_EPISODE_IDENTITY_MISSING"
    expected_feature_hash = stable_hash(
        {
            "decision_contract": episode.get("decision_contract"),
            "context_snapshot": episode.get("context_snapshot"),
            "setup_snapshot": episode.get("setup_snapshot"),
            "entry_snapshot": episode.get("entry_snapshot"),
        }
    )
    if expected_feature_hash != episode.get("feature_snapshot_hash"):
        return "FEATURE_SNAPSHOT_MUTATED"
    decision = _optional_timestamp(episode.get("decision_timestamp"))
    if decision is None:
        return "INVALID_DECISION_TIMESTAMP"
    setup = _mapping(episode.get("setup_snapshot"))
    evidence = _mapping(setup.get("timeframe_evidence"))
    if stable_hash(evidence) != setup.get("timeframe_evidence_hash"):
        return "TIMEFRAME_EVIDENCE_HASH_MISMATCH"
    contract = _mapping(setup.get("strategy_timeframe_contract"))
    required_timeframes = {
        str(value) for value in contract.get("required_timeframes", [])
    }
    if not required_timeframes:
        return "REQUIRED_TIMEFRAMES_MISSING"
    for timeframe in required_timeframes:
        state = _mapping(evidence.get(timeframe))
        if state.get("available") is not True or state.get("bar_closed") is not True:
            return "REQUIRED_TIMEFRAME_NOT_CLOSED_AVAILABLE"
    for state_value in evidence.values():
        state = _mapping(state_value)
        if state.get("available") is not True:
            continue
        available_at = _optional_timestamp(state.get("available_at"))
        if available_at is None or available_at > decision:
            return "FEATURE_AVAILABLE_AFTER_DECISION"
        knowledge = _optional_timestamp(state.get("knowledge_available_at"))
        if state.get("knowledge_available_at") and (knowledge is None or knowledge > decision):
            return "KNOWLEDGE_AVAILABLE_AFTER_DECISION"
    return None


def _outcome_rejection_reason(
    episode: Mapping[str, Any], outcome: Mapping[str, Any]
) -> str | None:
    if outcome.get("schema") != "active_swing_forward_episode_outcome_v1":
        return "NONCANONICAL_OUTCOME_SCHEMA"
    if outcome.get("terminal") is not True:
        return "OUTCOME_NOT_TERMINAL"
    if outcome.get("research_observation_eligible") is not True:
        return "OUTCOME_NOT_RESEARCH_ELIGIBLE"
    if outcome.get("excluded_from_performance_metrics") is not False:
        return "OUTCOME_EXCLUDED_FROM_PERFORMANCE"
    if outcome.get("natural_strategy_candidate") is not True:
        return "OUTCOME_NOT_NATURAL_CANDIDATE"
    if outcome.get("feature_snapshot_hash") != episode.get("feature_snapshot_hash"):
        return "OUTCOME_FEATURE_HASH_MISMATCH"
    if outcome.get("candidate_identity") != episode.get("candidate_identity"):
        return "OUTCOME_CANDIDATE_IDENTITY_MISMATCH"
    net_r = _finite_or_none(outcome.get("net_R"))
    if net_r is None:
        return "OUTCOME_NET_R_MISSING"
    decision = _optional_timestamp(episode.get("decision_timestamp"))
    fill = _optional_timestamp(outcome.get("fill_timestamp"))
    exit_at = _optional_timestamp(outcome.get("exit_timestamp"))
    if decision is None or fill is None or exit_at is None or fill < decision or exit_at < fill:
        return "OUTCOME_CLOCK_INVALID"
    return None


def _panel_row(episode: Mapping[str, Any], outcome: Mapping[str, Any]) -> dict[str, Any]:
    row = active_swing_feature_row(episode, outcome=outcome)
    row.update(
        {
            "native_exit_net_return": float(outcome["net_R"]),
            "positive_net_trade": int(float(outcome["net_R"]) > 0.0),
            "gross_R": _finite_or_none(outcome.get("gross_R")),
            "maximum_favourable_excursion_R": _finite_or_none(
                outcome.get("maximum_favourable_excursion")
            ),
            "maximum_adverse_excursion_R": _finite_or_none(
                outcome.get("maximum_adverse_excursion")
            ),
            "holding_duration_seconds": _finite_or_none(
                outcome.get("holding_duration_seconds")
            ),
            "first_barrier_hit": str(
                outcome.get("first_barrier_hit") or "UNKNOWN"
            ),
        }
    )
    return row


def active_swing_feature_row(
    episode: Mapping[str, Any],
    *,
    outcome: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract the immutable, decision-time feature row for one episode."""

    setup = _mapping(episode.get("setup_snapshot"))
    contract = _mapping(setup.get("strategy_timeframe_contract"))
    evidence = _mapping(setup.get("timeframe_evidence"))
    decision_contract = _mapping(episode.get("decision_contract"))
    context = _mapping(episode.get("context_snapshot"))
    asset_profile = _mapping(decision_contract.get("asset_profile"))
    event_risk = _mapping(context.get("event_risk"))
    decision = _timestamp(episode["decision_timestamp"])
    feature_times = [
        _timestamp(_mapping(value)["available_at"])
        for value in evidence.values()
        if _mapping(value).get("available") is True
    ]
    context_timeframes = [str(value) for value in contract.get("context_timeframes", [])]
    row: dict[str, Any] = {
        "candidate_identity": str(episode["candidate_identity"]),
        "candidate_unit": NATURAL_CANDIDATE_UNIT,
        "episode_id": str(episode["episode_id"]),
        "feature_snapshot_hash": str(episode["feature_snapshot_hash"]),
        "timeframe_evidence_hash": str(setup["timeframe_evidence_hash"]),
        "symbol": str(episode.get("symbol") or "").upper(),
        "security_id": str(
            _mapping(outcome).get("symbol_fingerprint")
            or f"SYMBOL:{episode.get('symbol')}"
        ),
        "strategy_id": str(episode.get("strategy_id") or ""),
        "strategy_family": str(episode.get("strategy_family") or "UNKNOWN"),
        "strategy_dna_hash": str(episode.get("strategy_dna_hash") or ""),
        "fold_id": "NATURAL_FORWARD",
        "decision_timestamp": decision.isoformat(),
        "feature_timestamp": max(feature_times).isoformat(),
        "label_available_at": (
            _timestamp(_mapping(outcome)["exit_timestamp"]).isoformat()
            if _mapping(outcome).get("exit_timestamp")
            else None
        ),
        "entry_timeframe": contract.get("entry_timeframe"),
        "setup_timeframe": contract.get("setup_timeframe"),
        "structural_timeframe": contract.get("structural_timeframe"),
        "management_timeframe": contract.get("management_timeframe"),
        "exit_timeframe": contract.get("exit_timeframe"),
        "context_architecture": ">".join(context_timeframes),
        "asset_class": setup.get("asset_class") or "UNKNOWN",
        "market_regime": decision_contract.get("market_regime") or "UNKNOWN",
        "setup_score_within_family": _finite_or_none(decision_contract.get("setup_score_within_family")),
        "entry_confirmation_score": _finite_or_none(decision_contract.get("entry_confirmation_score")),
        "asset_bias_score": _finite_or_none(context.get("asset_bias_score")),
        "asset_bias_confidence": _finite_or_none(context.get("asset_bias_confidence")),
        "event_risk_score": _finite_or_none(event_risk.get("risk_score")),
        "asset_profile_score": _finite_or_none(asset_profile.get("coverage_adjusted_score")),
        "asset_profile_coverage": _finite_or_none(asset_profile.get("data_coverage_ratio")),
        "reward_risk_1": _finite_or_none(setup.get("reward_risk_1")),
        "estimated_transaction_costs_eur": _finite_or_none(setup.get("estimated_transaction_costs_eur")),
    }
    for timeframe in TIMEFRAMES:
        state = _mapping(evidence.get(timeframe))
        available = state.get("available") is True
        close = _finite_or_none(state.get("close"))
        ema20 = _finite_or_none(state.get("ema20"))
        ema50 = _finite_or_none(state.get("ema50"))
        row[f"has_{timeframe}"] = int(available)
        row[f"return_1bar_{timeframe}"] = _finite_or_none(state.get("return_1_bar"))
        row[f"momentum_4bars_{timeframe}"] = _finite_or_none(state.get("return_4_bars"))
        row[f"distance_ema20_{timeframe}"] = _relative_distance(close, ema20)
        row[f"distance_ema50_{timeframe}"] = _relative_distance(close, ema50)
        row[f"trend_state_{timeframe}"] = str(state.get("trend_state") or "UNAVAILABLE")
    row["interaction_momentum_15m_x_momentum_1h"] = _product(
        row["momentum_4bars_15m"], row["momentum_4bars_1h"]
    )
    pullback_1h = row["distance_ema20_1h"]
    row["interaction_pullback_1h_x_trend_4h"] = _product(
        -float(pullback_1h) if pullback_1h is not None else None,
        row["momentum_4bars_4h"],
    )
    row["interaction_momentum_4h_x_momentum_1d"] = _product(
        row["momentum_4bars_4h"], row["momentum_4bars_1d"]
    )
    return row


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    if not path.is_file():
        return [], 0
    rows: list[dict[str, Any]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            rows.append(value)
        else:
            malformed += 1
    return rows, malformed


def _source_hashes(root: Path) -> dict[str, str | None]:
    return {
        path.as_posix(): (
            sha256_file(root / path).upper() if (root / path).is_file() else None
        )
        for path in (EPISODES_PATH, OUTCOMES_PATH)
    }


def _sample_dependence_audit(panel: pd.DataFrame) -> dict[str, Any]:
    if panel.empty:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "raw_candidate_count": 0,
            "independent_market_time_bucket_count": 0,
            "symbol_time_cluster_count": 0,
            "strategy_time_cluster_count": 0,
            "architecture_time_cluster_count": 0,
            "trade_interval_overlap_cluster_count": 0,
            "effective_sample_size": 0.0,
            "raw_candidate_count_is_not_effective_sample_size": True,
        }
    work = panel.copy()
    work["decision_timestamp"] = pd.to_datetime(
        work["decision_timestamp"], utc=True
    )
    work["label_available_at"] = pd.to_datetime(
        work["label_available_at"], utc=True
    )
    work["market_time_bucket"] = work["decision_timestamp"].dt.floor("1D")
    architecture = (
        work["entry_timeframe"].astype(str)
        + ">"
        + work["setup_timeframe"].astype(str)
        + ">"
        + work["context_architecture"].astype(str)
    )
    market_groups = work.groupby("market_time_bucket", dropna=False).size()
    symbol_groups = work.groupby(
        ["symbol", "market_time_bucket"], dropna=False
    ).size()
    strategy_groups = work.groupby(
        ["strategy_id", "market_time_bucket"], dropna=False
    ).size()
    architecture_groups = pd.DataFrame(
        {
            "architecture": architecture,
            "market_time_bucket": work["market_time_bucket"],
        }
    ).groupby(["architecture", "market_time_bucket"], dropna=False).size()
    maximum_mean_cluster_size = max(
        float(groups.mean())
        for groups in (
            market_groups,
            symbol_groups,
            strategy_groups,
            architecture_groups,
        )
        if len(groups)
    )
    overlap_clusters = _interval_cluster_count(
        list(
            zip(
                work["decision_timestamp"],
                work["label_available_at"],
                strict=True,
            )
        )
    )
    returns = work.sort_values("decision_timestamp")[
        "native_exit_net_return"
    ].astype(float)
    autocorrelation = float(returns.autocorr(lag=1)) if len(returns) > 2 else 0.0
    autocorrelation = autocorrelation if math.isfinite(autocorrelation) else 0.0
    serial_penalty = (
        (1.0 + autocorrelation) / max(1e-12, 1.0 - autocorrelation)
        if autocorrelation > 0.0
        else 1.0
    )
    effective = min(
        float(len(work)) / maximum_mean_cluster_size / serial_penalty,
        float(len(market_groups)),
    )
    return {
        "status": "EVALUABLE" if effective >= 100 else "LOW_EFFECTIVE_SAMPLE",
        "raw_candidate_count": len(work),
        "independent_market_time_bucket_count": len(market_groups),
        "symbol_time_cluster_count": len(symbol_groups),
        "strategy_time_cluster_count": len(strategy_groups),
        "architecture_time_cluster_count": len(architecture_groups),
        "trade_interval_overlap_cluster_count": overlap_clusters,
        "mean_candidates_per_market_time_bucket": round(
            float(market_groups.mean()), 8
        ),
        "maximum_mean_cluster_size_penalty": round(
            maximum_mean_cluster_size, 8
        ),
        "lag1_net_R_autocorrelation": round(autocorrelation, 8),
        "serial_correlation_penalty": round(serial_penalty, 8),
        "effective_sample_size": round(max(0.0, effective), 8),
        "dimensions": [
            "TRADE_INTERVAL_OVERLAP",
            "SYMBOL_BY_DECISION_DAY",
            "STRATEGY_BY_DECISION_DAY",
            "ARCHITECTURE_BY_DECISION_DAY",
            "LAG1_NET_R_AUTOCORRELATION",
        ],
        "raw_candidate_count_is_not_effective_sample_size": True,
    }


def _interval_cluster_count(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> int:
    ordered = sorted(intervals, key=lambda value: (value[0], value[1]))
    count = 0
    current_end: pd.Timestamp | None = None
    for start, end in ordered:
        if current_end is None or start > current_end:
            count += 1
            current_end = end
        else:
            current_end = max(current_end, end)
    return count


def _strategy_clustering_audit(panel: pd.DataFrame) -> dict[str, Any]:
    if panel.empty:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "strategy_count": 0,
            "frozen_hypothesis_cluster_count": 0,
            "empirical_pair_count": 0,
            "empirical_correlated_pair_count": 0,
            "empirical_return_clusters": [],
        }
    work = panel.copy()
    work["decision_day"] = pd.to_datetime(
        work["decision_timestamp"], utc=True
    ).dt.floor("1D")
    hypothesis_key = (
        work["strategy_dna_hash"].astype(str)
        + "|"
        + work["entry_timeframe"].astype(str)
        + ">"
        + work["setup_timeframe"].astype(str)
        + ">"
        + work["context_architecture"].astype(str)
    )
    strategy_ids = sorted(work["strategy_id"].astype(str).unique())
    returns = work.pivot_table(
        index="decision_day",
        columns="strategy_id",
        values="native_exit_net_return",
        aggfunc="mean",
    )
    parent = {strategy: strategy for strategy in strategy_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    empirical_pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(strategy_ids):
        for right in strategy_ids[left_index + 1 :]:
            common = returns[[left, right]].dropna()
            if len(common) < 20:
                continue
            correlation = float(common[left].corr(common[right]))
            if not math.isfinite(correlation):
                continue
            overlap_ratio = len(common) / max(
                1, int(returns[left].notna().sum()), int(returns[right].notna().sum())
            )
            correlated = correlation >= 0.75 and overlap_ratio >= 0.25
            empirical_pairs.append(
                {
                    "left_strategy_id": left,
                    "right_strategy_id": right,
                    "common_decision_day_count": len(common),
                    "return_correlation": round(correlation, 8),
                    "decision_day_overlap_ratio": round(overlap_ratio, 8),
                    "same_empirical_cluster": correlated,
                }
            )
            if correlated:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parent[right_root] = left_root
    clusters: dict[str, list[str]] = {}
    for strategy in strategy_ids:
        clusters.setdefault(find(strategy), []).append(strategy)
    return {
        "status": "EVALUABLE" if empirical_pairs else "INSUFFICIENT_PAIRWISE_OVERLAP",
        "strategy_count": len(strategy_ids),
        "frozen_hypothesis_cluster_count": int(hypothesis_key.nunique()),
        "empirical_pair_count": len(empirical_pairs),
        "empirical_correlated_pair_count": sum(
            row["same_empirical_cluster"] for row in empirical_pairs
        ),
        "empirical_return_clusters": [
            sorted(values) for values in clusters.values()
        ],
        "pairwise_evidence": empirical_pairs,
        "clustering_dimensions": [
            "STRATEGY_DNA",
            "TIMEFRAME_ARCHITECTURE",
            "DECISION_DAY_OVERLAP",
            "OOS_NET_R_CORRELATION",
        ],
        "correlation_threshold": 0.75,
        "minimum_common_decision_days": 20,
    }


def _timeframe_redundancy_audit(panel: pd.DataFrame) -> dict[str, Any]:
    feature_names = [
        name
        for name in TIMEFRAME_FEATURE_COLUMNS
        if name.startswith("momentum_4bars_")
        or name.startswith("distance_ema20_")
        or name.startswith("distance_ema50_")
    ]
    if panel.empty:
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "feature_count": len(feature_names),
            "evaluable_pair_count": 0,
            "high_redundancy_pair_count": 0,
            "high_redundancy_pairs": [],
        }
    high: list[dict[str, Any]] = []
    evaluable = 0
    for left_index, left in enumerate(feature_names):
        for right in feature_names[left_index + 1 :]:
            common = panel[[left, right]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            if len(common) < 100:
                continue
            correlation = float(common[left].corr(common[right]))
            if not math.isfinite(correlation):
                continue
            evaluable += 1
            if abs(correlation) >= 0.90:
                high.append(
                    {
                        "left_feature": left,
                        "right_feature": right,
                        "complete_case_count": len(common),
                        "correlation": round(correlation, 8),
                    }
                )
    return {
        "status": "EVALUABLE" if evaluable else "INSUFFICIENT_PAIRWISE_COVERAGE",
        "feature_count": len(feature_names),
        "evaluable_pair_count": evaluable,
        "high_redundancy_pair_count": len(high),
        "high_redundancy_pairs": high,
        "minimum_complete_cases_per_pair": 100,
        "absolute_correlation_threshold": 0.90,
        "automatic_feature_pruning": False,
        "reason": "PRUNING_REQUIRES_OUTER_HOLDOUT_INCREMENTAL_EVIDENCE",
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _relative_distance(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None or reference == 0.0:
        return None
    return value / reference - 1.0


def _product(left: Any, right: Any) -> float | None:
    left_value = _finite_or_none(left)
    right_value = _finite_or_none(right)
    if left_value is None or right_value is None:
        return None
    return left_value * right_value


def _optional_timestamp(value: Any) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp)


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = _optional_timestamp(value)
    if timestamp is None:
        raise ValueError(f"invalid timestamp: {value!r}")
    return timestamp


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = [
    "INFERENCE_PATH",
    "PANEL_PATH",
    "STATUS_PATH",
    "CONTRACT_COLUMNS",
    "INTERACTION_FEATURE_COLUMNS",
    "SCALAR_FEATURE_COLUMNS",
    "TIMEFRAME_FEATURE_COLUMNS",
    "active_swing_feature_row",
    "build_active_swing_candidate_panel",
    "infer_current_active_swing_candidates",
]
