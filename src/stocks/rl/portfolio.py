from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from stocks.context.candidate_evidence import is_natural_strategy_candidate
from stocks.execution.idempotency import stable_hash


CONFIG_PATH = Path("config/rl_portfolio_rotation_v1.json")
OUTPUT_PATH = Path("output/rl/portfolio-rotation-status.json")
PORTFOLIO_FEATURES = (
    "portfolio_objective_score",
    "opportunity_score",
    "expected_net_return",
    "expected_r",
    "stop_risk_pct",
    "liquidity",
    "regime_fit",
    "target_weight",
)
FEATURE_SCHEMA_HASH = stable_hash(
    {
        "schema": "rl_portfolio_rotation_features_v1",
        "features": list(PORTFOLIO_FEATURES),
    }
)
DEFAULT_FAIL_CLOSED_CONFIG: dict[str, Any] = {
    "schema": "rl_portfolio_rotation_contract_v1",
    "mode": "SHADOW_ONLY",
    "canonical_environment": "OpportunitySelectionEnv",
    "top_n": 10,
    "accepted_entry_lifecycle_states": ["ENTRY_READY", "ROTATION_CANDIDATE"],
    "accepted_shariah_statuses": ["SHARIAH_ELIGIBLE_PIT", "SHARIAH_COMPLIANT"],
    "accepted_product_shariah_statuses": [
        "SHARIAH_PRODUCT_ELIGIBLE_PIT",
        "SHARIAH_PRODUCT_COMPLIANT",
    ],
    "require_contract_resolved": True,
    "require_positive_expected_net_return": True,
    "require_positive_expected_r": True,
    "require_whole_share_preflight": True,
    "require_causal_candidate_history": True,
    "require_promoted_dedicated_policy": True,
    "require_oos_promotion_gate": True,
    "candidate_history_path": (
        "data/market_context/private/entry-episodes.jsonl"
    ),
    "candidate_outcome_path": (
        "data/market_context/private/entry-episode-outcomes.jsonl"
    ),
    "minimum_natural_candidate_episodes": 500,
    "minimum_closed_candidate_outcomes": 250,
    "minimum_independent_candidate_clusters": 150,
    "minimum_decision_periods": 100,
    "minimum_supported_timeframes": 3,
    "minimum_market_regimes": 3,
    "promotion_evidence_path": "output/rl/portfolio-rotation/promotion.json",
    "policy_may_create_signals": False,
    "policy_may_change_eligibility": False,
    "policy_may_increase_weight": False,
    "policy_may_override_caps": False,
    "financial_effect_applied": False,
    "automatic_order_submission": False,
    "execution_authority": "NONE",
}


def build_shadow_portfolio_rotation(
    project_root: Path,
    *,
    ranked: Sequence[Mapping[str, Any]],
    allocation: Mapping[str, Any],
    whole_share_preflight: Mapping[str, Any],
    proposed_ticker: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed RL rotation report without changing allocation.

    A dedicated policy may only narrow the deterministic allocator's output.
    It cannot create candidates, change eligibility, increase weights or gain
    execution authority. Until dedicated causal/OOS evidence is promoted, the
    sole effective shadow action is CASH.
    """

    loaded_config = _read_json(project_root / CONFIG_PATH)
    config = loaded_config or dict(DEFAULT_FAIL_CLOSED_CONFIG)
    _validate_config(config)
    readiness = _policy_readiness(project_root, config)
    readiness["config_loaded"] = bool(loaded_config)
    report = evaluate_shadow_portfolio_rotation(
        ranked=ranked,
        allocation=allocation,
        whole_share_preflight=whole_share_preflight,
        config=config,
        policy_readiness=readiness,
        proposed_ticker=proposed_ticker,
    )
    return report


def evaluate_shadow_portfolio_rotation(
    *,
    ranked: Sequence[Mapping[str, Any]],
    allocation: Mapping[str, Any],
    whole_share_preflight: Mapping[str, Any],
    config: Mapping[str, Any],
    policy_readiness: Mapping[str, Any],
    proposed_ticker: str | None = None,
) -> dict[str, Any]:
    _validate_config(config)
    allocation_rows = list(allocation.get("allocations") or [])
    allocation_by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in allocation_rows
        if str(row.get("ticker") or "").strip()
    }
    ranked_by_ticker = {
        str(row.get("ticker") or "").upper(): row
        for row in ranked
        if str(row.get("ticker") or "").strip()
    }
    top_n = int(config["top_n"])
    ordered_tickers = list(allocation_by_ticker)[:top_n]
    feasible = {
        str(value).upper()
        for value in whole_share_preflight.get("feasible_tickers", [])
    }
    preflight_applied = bool(
        whole_share_preflight.get("selection_filter_applied", False)
    )

    candidate_rows: list[dict[str, Any]] = []
    candidate_mask = [True]
    for rank, ticker in enumerate(ordered_tickers, start=1):
        candidate = ranked_by_ticker.get(ticker, {})
        baseline = allocation_by_ticker[ticker]
        blockers = _candidate_blockers(
            candidate,
            ticker=ticker,
            config=config,
            preflight_applied=preflight_applied,
            feasible_tickers=feasible,
        )
        admissible = not blockers
        candidate_mask.append(admissible)
        candidate_rows.append(
            {
                "action_index": rank,
                "ticker": ticker,
                "admissible": admissible,
                "blockers": blockers,
                "baseline_target_weight": _finite_or_none(
                    baseline.get("target_weight")
                ),
                "maximum_shadow_target_weight": _finite_or_none(
                    baseline.get("target_weight")
                ),
                "lifecycle_state": str(
                    candidate.get("lifecycle_state") or "UNKNOWN"
                ),
                "feature_values": {
                    name: _feature_value(name, candidate, baseline)
                    for name in PORTFOLIO_FEATURES
                },
            }
        )

    readiness_blockers = _readiness_blockers(policy_readiness, config)
    policy_ready = not readiness_blockers
    policy_action_mask = [True] + [
        bool(value and policy_ready) for value in candidate_mask[1:]
    ]
    requested = str(proposed_ticker or "CASH").strip().upper()
    by_ticker = {row["ticker"]: row for row in candidate_rows}
    proposal_blockers: list[str] = []
    if requested != "CASH":
        if requested not in by_ticker:
            proposal_blockers.append("PROPOSAL_OUTSIDE_DETERMINISTIC_ALLOCATOR")
        elif not by_ticker[requested]["admissible"]:
            proposal_blockers.append("PROPOSAL_CANDIDATE_MASKED")
        proposal_blockers.extend(readiness_blockers)
    shadow_action = (
        requested
        if requested != "CASH" and not proposal_blockers
        else "CASH"
    )
    chosen = by_ticker.get(shadow_action)
    status = (
        "SHADOW_PROPOSAL_AVAILABLE"
        if shadow_action != "CASH"
        else "INSUFFICIENT_EVIDENCE"
        if readiness_blockers
        else "CASH_SELECTED"
    )
    generated_at = datetime.now(UTC).isoformat()
    report: dict[str, Any] = {
        "schema": "rl_portfolio_rotation_status_v1",
        "status": status,
        "generated_at": generated_at,
        "mode": "SHADOW_ONLY",
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "action_space": ["CASH", *ordered_tickers],
        "candidate_gate_mask": [int(value) for value in candidate_mask],
        "policy_action_mask": [int(value) for value in policy_action_mask],
        "cash_action_available": True,
        "candidate_count": len(candidate_rows),
        "admissible_candidate_count": sum(candidate_mask[1:]),
        "candidates": candidate_rows,
        "policy_readiness": {
            "status": "GO" if policy_ready else "NO_GO",
            "policy_id": policy_readiness.get("policy_id"),
            "blockers": readiness_blockers,
            "causal_candidate_history": bool(
                policy_readiness.get("causal_candidate_history", False)
            ),
            "dedicated_policy_promoted": bool(
                policy_readiness.get("dedicated_policy_promoted", False)
            ),
            "oos_promotion_gate": bool(
                policy_readiness.get("oos_promotion_gate", False)
            ),
            "feature_schema_match": bool(
                policy_readiness.get("feature_schema_match", False)
            ),
            "natural_candidate_episode_count": int(
                policy_readiness.get("natural_candidate_episode_count", 0)
            ),
            "canonical_observation_episode_count": int(
                policy_readiness.get("canonical_observation_episode_count", 0)
            ),
            "context_observation_episode_count": int(
                policy_readiness.get("context_observation_episode_count", 0)
            ),
            "candidate_identity_deduplicated": bool(
                policy_readiness.get("candidate_identity_deduplicated", False)
            ),
            "duplicate_candidate_episode_count": int(
                policy_readiness.get("duplicate_candidate_episode_count", 0)
            ),
            "candidate_unit_definition": policy_readiness.get(
                "candidate_unit_definition",
                "ONE_NATURAL_STRATEGY_SETUP",
            ),
            "architecture_bound_observation_count": int(
                policy_readiness.get("architecture_bound_observation_count", 0)
            ),
            "architecture_binding_incomplete_count": int(
                policy_readiness.get("architecture_binding_incomplete_count", 0)
            ),
            "architecture_counts": policy_readiness.get(
                "architecture_counts", {}
            ),
            "forward_records_require_explicit_architecture_binding": bool(
                policy_readiness.get(
                    "forward_records_require_explicit_architecture_binding", False
                )
            ),
            "closed_candidate_outcome_count": int(
                policy_readiness.get("closed_candidate_outcome_count", 0)
            ),
            "independent_candidate_cluster_count": int(
                policy_readiness.get("independent_candidate_cluster_count", 0)
            ),
            "decision_period_count": int(
                policy_readiness.get("decision_period_count", 0)
            ),
            "supported_timeframe_count": int(
                policy_readiness.get("supported_timeframe_count", 0)
            ),
            "market_regime_count": int(
                policy_readiness.get("market_regime_count", 0)
            ),
            "evidence_minima": policy_readiness.get("evidence_minima", {}),
        },
        "requested_shadow_action": requested,
        "proposal_blockers": sorted(set(proposal_blockers)),
        "shadow_action": shadow_action,
        "shadow_target_weight": (
            chosen["maximum_shadow_target_weight"] if chosen else 0.0
        ),
        "deterministic_allocator_unchanged": True,
        "baseline_allocation_count": len(allocation_rows),
        "baseline_research_target_exposure": _finite_or_none(
            allocation.get("research_target_exposure")
        ),
        "policy_may_create_signals": False,
        "policy_may_change_eligibility": False,
        "policy_may_increase_weight": False,
        "policy_may_override_caps": False,
        "financial_effect_applied": False,
        "orders_generated": 0,
        "broker_calls": 0,
        "broker_writes": 0,
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(report)
    return report


def _candidate_blockers(
    candidate: Mapping[str, Any],
    *,
    ticker: str,
    config: Mapping[str, Any],
    preflight_applied: bool,
    feasible_tickers: set[str],
) -> list[str]:
    blockers = {
        str(value)
        for value in candidate.get("research_allocation_blockers", [])
    }
    if not candidate:
        blockers.add("RANKED_CANDIDATE_MISSING")
    if bool(config.get("require_contract_resolved", True)) and not bool(
        candidate.get("contract_resolved", False)
    ):
        blockers.add("CONTRACT_IDENTITY_REQUIRED")
    lifecycle = str(candidate.get("lifecycle_state") or "UNKNOWN").upper()
    accepted_lifecycle = {
        str(value).upper()
        for value in config.get("accepted_entry_lifecycle_states", [])
    }
    if lifecycle not in accepted_lifecycle:
        blockers.add(f"LIFECYCLE_NOT_ENTRY_READY:{lifecycle}")
    shariah = str(candidate.get("shariah_status") or "UNKNOWN").upper()
    accepted_shariah = {
        str(value).upper()
        for value in config.get("accepted_shariah_statuses", [])
    }
    if shariah not in accepted_shariah:
        blockers.add(f"SHARIAH_NOT_CURRENTLY_ELIGIBLE:{shariah}")
    expected_net = _finite_or_none(candidate.get("expected_net_return"))
    if bool(config.get("require_positive_expected_net_return", True)) and (
        expected_net is None or expected_net <= 0
    ):
        blockers.add("POSITIVE_EXPECTED_NET_RETURN_REQUIRED")
    expected_r = _finite_or_none(candidate.get("expected_r"))
    if bool(config.get("require_positive_expected_r", True)) and (
        expected_r is None or expected_r <= 0
    ):
        blockers.add("POSITIVE_EXPECTED_R_REQUIRED")
    if bool(config.get("require_whole_share_preflight", True)):
        if not preflight_applied:
            blockers.add("WHOLE_SHARE_PREFLIGHT_REQUIRED")
        elif ticker not in feasible_tickers:
            blockers.add("WHOLE_SHARE_INFEASIBLE")
    product = candidate.get("real_asset_context", {}).get(
        "product_identity", {}
    )
    product_structure = str(
        product.get("product_structure")
        or candidate.get("product_structure")
        or ""
    ).upper()
    claims_physical = "PHYSICAL" in product_structure or bool(
        product.get("physical_structure_claimed", False)
    )
    if claims_physical:
        if not bool(product.get("physical_structure_verified", False)):
            blockers.add("PHYSICAL_STRUCTURE_NOT_CURRENTLY_VERIFIED")
        product_shariah = str(
            product.get("shariah_product_status")
            or candidate.get("shariah_product_status")
            or "ATTESTATION_REQUIRED"
        ).upper()
        accepted_product = {
            str(value).upper()
            for value in config.get(
                "accepted_product_shariah_statuses", []
            )
        }
        if product_shariah not in accepted_product:
            blockers.add(
                f"SHARIAH_PRODUCT_NOT_ELIGIBLE:{product_shariah}"
            )
    return sorted(blockers)


def _policy_readiness(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    history_path = project_root / str(config["candidate_history_path"])
    outcome_path = project_root / str(config["candidate_outcome_path"])
    history = candidate_history_readiness(
        history_path,
        outcome_path,
        config=config,
    )
    evidence = _read_json(
        project_root / str(config["promotion_evidence_path"])
    )
    return {
        "policy_id": evidence.get("policy_id"),
        **history,
        "dedicated_policy_promoted": (
            str(evidence.get("status") or "").upper() == "PROMOTED"
            and str(evidence.get("mode") or "").upper() == "SHADOW_ONLY"
            and str(evidence.get("environment") or "")
            == str(config["canonical_environment"])
        ),
        "oos_promotion_gate": bool(evidence.get("oos_promotion_gate")),
        "feature_schema_match": (
            str(evidence.get("feature_schema_hash") or "")
            == FEATURE_SCHEMA_HASH
        ),
    }


def candidate_history_readiness(
    history_path: Path,
    outcome_path: Path,
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_observations = [
        row
        for row in _read_jsonl(history_path)
        if row.get("schema") == "active_swing_forward_episode_v1"
        and row.get("episode_id")
        and row.get("feature_snapshot_hash")
    ]
    natural_candidate_observations = [
        row
        for row in canonical_observations
        if is_natural_strategy_candidate(row)
    ]
    architecture_bound_observations = [
        row
        for row in natural_candidate_observations
        if not _forward_architecture_binding_blockers(row)
    ]
    candidate_by_identity: dict[str, dict[str, Any]] = {}
    for row in architecture_bound_observations:
        identity = str(row["candidate_identity"])
        previous = candidate_by_identity.get(identity)
        if previous is None or str(
            row.get("decision_timestamp") or row.get("observed_at") or ""
        ) < str(
            previous.get("decision_timestamp")
            or previous.get("observed_at")
            or ""
        ):
            candidate_by_identity[identity] = row
    episodes = list(candidate_by_identity.values())
    by_id = {str(row["episode_id"]): row for row in episodes}
    outcomes = []
    for row in _read_jsonl(outcome_path):
        episode = by_id.get(str(row.get("episode_id") or ""))
        if episode is None:
            continue
        if (
            row.get("schema") != "active_swing_forward_episode_outcome_v1"
            or row.get("terminal") is not True
            or row.get("research_observation_eligible") is not True
            or row.get("excluded_from_performance_metrics") is not False
            or row.get("feature_snapshot_hash")
            != episode.get("feature_snapshot_hash")
            or _finite_or_none(row.get("net_R")) is None
            or not row.get("fill_timestamp")
            or not row.get("exit_timestamp")
        ):
            continue
        outcomes.append(row)
    decision_periods = {
        str(row.get("decision_timestamp") or row.get("observed_at"))
        for row in episodes
        if row.get("decision_timestamp") or row.get("observed_at")
    }
    timeframes = {
        str(_forward_architecture_metadata(row)["entry_timeframe"]).lower()
        for row in episodes
        if str(_forward_architecture_metadata(row)["entry_timeframe"]).lower()
        in {"15m", "1h", "2h", "4h", "1d", "1w"}
    }
    regimes = {
        str(row.get("market_regime") or "")
        for row in outcomes
        if str(row.get("market_regime") or "")
        not in {"", "UNKNOWN", "UNAVAILABLE_AT_DECISION"}
    }
    clusters = {
        (
            str(row.get("symbol_fingerprint") or ""),
            str(row.get("strategy_id") or ""),
            str(_forward_architecture_metadata(episode)["architecture_key"]),
            str(row.get("fill_timestamp") or "")[:10],
        )
        for row in outcomes
        for episode in [by_id[str(row["episode_id"])]]
    }
    architectures = Counter(
        str(_forward_architecture_metadata(row)["architecture_key"])
        for row in episodes
    )
    minima = {
        "natural_candidate_episodes": int(
            config.get("minimum_natural_candidate_episodes", 500)
        ),
        "closed_candidate_outcomes": int(
            config.get("minimum_closed_candidate_outcomes", 250)
        ),
        "independent_candidate_clusters": int(
            config.get("minimum_independent_candidate_clusters", 150)
        ),
        "decision_periods": int(config.get("minimum_decision_periods", 100)),
        "supported_timeframes": int(
            config.get("minimum_supported_timeframes", 3)
        ),
        "market_regimes": int(config.get("minimum_market_regimes", 3)),
    }
    counts = {
        "natural_candidate_episode_count": len(episodes),
        "closed_candidate_outcome_count": len(outcomes),
        "independent_candidate_cluster_count": len(clusters),
        "decision_period_count": len(decision_periods),
        "supported_timeframe_count": len(timeframes),
        "market_regime_count": len(regimes),
    }
    ready = (
        counts["natural_candidate_episode_count"]
        >= minima["natural_candidate_episodes"]
        and counts["closed_candidate_outcome_count"]
        >= minima["closed_candidate_outcomes"]
        and counts["independent_candidate_cluster_count"]
        >= minima["independent_candidate_clusters"]
        and counts["decision_period_count"] >= minima["decision_periods"]
        and counts["supported_timeframe_count"]
        >= minima["supported_timeframes"]
        and counts["market_regime_count"] >= minima["market_regimes"]
    )
    return {
        "causal_candidate_history": ready,
        **counts,
        "evidence_minima": minima,
        "raw_episode_count_is_not_promotion_sample": True,
        "canonical_observation_episode_count": len(canonical_observations),
        "context_observation_episode_count": (
            len(canonical_observations) - len(natural_candidate_observations)
        ),
        "architecture_bound_observation_count": len(
            architecture_bound_observations
        ),
        "architecture_binding_incomplete_count": (
            len(natural_candidate_observations)
            - len(architecture_bound_observations)
        ),
        "duplicate_candidate_episode_count": (
            len(architecture_bound_observations) - len(episodes)
        ),
        "candidate_unit_definition": "ONE_NATURAL_STRATEGY_SETUP",
        "candidate_identity_deduplicated": True,
        "forward_records_require_explicit_architecture_binding": True,
        "architecture_counts": dict(sorted(architectures.items())),
        "candidate_history_path": str(history_path),
        "candidate_outcome_path": str(outcome_path),
    }


def _forward_architecture_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    setup = row.get("setup_snapshot")
    setup = setup if isinstance(setup, Mapping) else {}
    decision = row.get("decision_contract")
    decision = decision if isinstance(decision, Mapping) else {}
    contract = row.get("strategy_timeframe_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    if not contract:
        nested = setup.get("strategy_timeframe_contract")
        contract = nested if isinstance(nested, Mapping) else {}
    context = sorted(
        {
            str(value).lower()
            for value in contract.get("context_timeframes", [])
            if str(value).strip()
        }
    )
    entry = str(
        row.get("entry_timeframe") or contract.get("entry_timeframe") or ""
    ).lower()
    setup_timeframe = str(
        row.get("setup_timeframe") or contract.get("setup_timeframe") or ""
    ).lower()
    family = str(
        row.get("strategy_family") or decision.get("strategy_family") or ""
    ).strip()
    model_version = str(
        row.get("model_version") or decision.get("model_version") or ""
    ).strip()
    strategy_dna_hash = str(
        row.get("strategy_dna_hash") or setup.get("strategy_dna_hash") or ""
    ).strip()
    architecture_key = (
        f"{family}__{entry}_ENTRY__{setup_timeframe}_SETUP__"
        f"{'+'.join(context) if context else 'NO_CONTEXT'}__MODEL_{model_version}"
    )
    return {
        "strategy_family": family,
        "strategy_dna_hash": strategy_dna_hash,
        "entry_timeframe": entry,
        "setup_timeframe": setup_timeframe,
        "context_timeframes": context,
        "model_version": model_version,
        "architecture_key": architecture_key,
    }


def _forward_architecture_binding_blockers(row: Mapping[str, Any]) -> list[str]:
    metadata = _forward_architecture_metadata(row)
    blockers: list[str] = []
    if not metadata["strategy_family"]:
        blockers.append("STRATEGY_FAMILY_MISSING")
    if not metadata["strategy_dna_hash"]:
        blockers.append("STRATEGY_DNA_HASH_MISSING")
    if not metadata["entry_timeframe"] or not metadata["setup_timeframe"]:
        blockers.append("ENTRY_SETUP_TIMEFRAME_BINDING_MISSING")
    if not metadata["model_version"]:
        blockers.append("MODEL_VERSION_MISSING")
    return blockers


def _readiness_blockers(
    readiness: Mapping[str, Any], config: Mapping[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if readiness.get("config_loaded") is False:
        blockers.append("RL_PORTFOLIO_ROTATION_CONFIG_MISSING")
    if bool(config.get("require_causal_candidate_history", True)) and not bool(
        readiness.get("causal_candidate_history", False)
    ):
        blockers.append("CAUSAL_CANDIDATE_UNIT_HISTORY_REQUIRED")
    if bool(config.get("require_promoted_dedicated_policy", True)) and not bool(
        readiness.get("dedicated_policy_promoted", False)
    ):
        blockers.append("DEDICATED_PORTFOLIO_RL_POLICY_NOT_PROMOTED")
    if bool(config.get("require_oos_promotion_gate", True)) and not bool(
        readiness.get("oos_promotion_gate", False)
    ):
        blockers.append("PORTFOLIO_RL_OOS_PROMOTION_GATE_REQUIRED")
    if not bool(readiness.get("feature_schema_match", False)):
        blockers.append("PORTFOLIO_RL_FEATURE_SCHEMA_MISMATCH")
    return blockers


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != "rl_portfolio_rotation_contract_v1":
        raise ValueError("RL_PORTFOLIO_ROTATION_CONFIG_INVALID")
    if str(config.get("mode") or "").upper() != "SHADOW_ONLY":
        raise ValueError("RL_PORTFOLIO_ROTATION_MUST_BE_SHADOW_ONLY")
    if int(config.get("top_n") or 0) < 1:
        raise ValueError("RL_PORTFOLIO_ROTATION_TOP_N_INVALID")
    forbidden_true = (
        "policy_may_create_signals",
        "policy_may_change_eligibility",
        "policy_may_increase_weight",
        "policy_may_override_caps",
        "financial_effect_applied",
        "automatic_order_submission",
    )
    if any(bool(config.get(name)) for name in forbidden_true):
        raise ValueError("RL_PORTFOLIO_ROTATION_AUTHORITY_INVALID")
    if str(config.get("execution_authority") or "") != "NONE":
        raise ValueError("RL_PORTFOLIO_ROTATION_AUTHORITY_INVALID")


def _feature_value(
    name: str,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> float | None:
    if name == "target_weight":
        return _finite_or_none(baseline.get("target_weight"))
    if name == "liquidity":
        return _finite_or_none(
            candidate.get("components", {}).get("liquidity")
        )
    if name == "regime_fit":
        return _finite_or_none(
            candidate.get("components", {}).get("regime_fit")
        )
    return _finite_or_none(candidate.get(name))


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


__all__ = [
    "FEATURE_SCHEMA_HASH",
    "PORTFOLIO_FEATURES",
    "build_shadow_portfolio_rotation",
    "candidate_history_readiness",
    "evaluate_shadow_portfolio_rotation",
]
