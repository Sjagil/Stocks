from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.research.autopilot.statistics import (
    probability_of_backtest_overfitting,
)
from stocks.research.phase6_3 import deflated_sharpe_probability


OUTPUT_ROOT = Path("output/research/evidence_throughput")
CONFIG_PATH = Path("config/research/evidence_throughput_v1.json")
AUTHORITY = {
    "FINANCIAL_FINALIST_GO": False,
    "strategy_authority": "NONE",
    "execution_authority": "NONE",
    "automatic_orders": 0,
    "broker_calls": 0,
    "order_calls": 0,
}


def publish_evidence_throughput(project_root: Path) -> dict[str, Any]:
    """Publish the current evidence funnel without changing policy or state."""
    config = _read_json(project_root / CONFIG_PATH)
    signals = _read_json(project_root / "output/signals/latest_signals.json")
    ranking = _read_json(
        project_root / "output/portfolio/opportunity_ranking.json"
    )
    top = _read_json(project_root / "output/reports/top_opportunities.json")
    qualification = _read_json(
        project_root / "output/research/phase11_14/qualification.json"
    )
    selection_bias = _read_json(
        project_root
        / "output/research/phase11_14/selection-bias-audit.json"
    )
    forward = _read_json(
        project_root / "output/research/phase11_14/forward-performance.json"
    )
    gate_attribution = _read_json(
        project_root
        / "output/research/active_swing/rejected_shadow"
        / "gate-attribution.json"
    )
    paper_session = _read_json(
        project_root / "output/operations/paper-session-audit.json"
    )
    phase9 = _read_json(project_root / "output/ibkr/phase9/status.json")
    shariah = _read_json(
        project_root / "output/research/universe/shariah_eligibility.json"
    )
    shariah_history = _read_json(
        project_root / "output/ibkr/phase11_3/shariah-history.json"
    )
    live_allowlist = _read_json(
        project_root / "output/ibkr/live/strategy-allowlist.json"
    )
    forward_observation = _read_json(
        project_root
        / "output/research/phase11_14/latest-forward-observation.json"
    )
    macro_score = _read_json(project_root / "output/macro/score.json")
    fractional_capability = _read_json(
        project_root / "output/ibkr/capabilities/fractional-shares.json"
    )
    summary = _read_parquet(
        project_root / "output/research/phase11_12/strategy-summary.parquet"
    )
    nested_folds = _read_parquet(
        project_root / "output/research/phase11_14/fold-results.parquet"
    )
    oos_returns = _read_parquet(
        project_root / "output/research/phase11_14/oos-returns.parquet"
    )
    macro_events = _read_json(project_root / "output/macro/events.json")
    trial_accounting = _multiple_testing_trial_accounting(
        qualification,
        selection_bias=selection_bias,
        summary=summary,
        nested_folds=nested_folds,
        config=config,
    )
    monte_carlo = _fold_monte_carlo(
        nested_folds,
        qualification=qualification,
        config=config,
    )
    event_driven = _exact_event_driven(
        oos_returns,
        macro_events=macro_events,
        qualification=qualification,
        config=config,
    )
    pit_boundary = _pit_shariah_boundary(
        shariah,
        historical=shariah_history,
    )
    forward_independence = _forward_independence_audit(
        forward,
        config=config,
    )
    forward = _forward_with_independence(
        forward,
        forward_independence,
    )
    opportunities = _dict_rows(ranking.get("opportunities"))
    raw_signals = _dict_rows(signals.get("signals"))
    stages, rejection_rows = _build_funnel(
        raw_signals,
        opportunities,
        top=top,
        gate_attribution=gate_attribution,
    )
    validation = _validation_throughput(
        project_root,
        config=config,
        qualification=qualification,
        selection_bias=selection_bias,
        forward=forward,
        monte_carlo=monte_carlo,
        event_driven=event_driven,
    )
    finalists = _finalist_funnel(
        qualification,
        selection_bias=selection_bias,
        forward=forward,
        config=config,
        trial_accounting=trial_accounting,
        pit_boundary=pit_boundary,
        paper_session=paper_session,
        phase9=phase9,
    )
    sample_plan = _forward_sample_plan(
        finalists,
        forward=forward,
        config=config,
    )
    authority_recommendations = _strategy_authority_recommendations(
        finalists,
        config=config,
    )
    canary_readiness = _provisional_canary_readiness(
        authority_recommendations,
        live_allowlist=live_allowlist,
        forward_observation=forward_observation,
        opportunities=opportunities,
        macro_score=macro_score,
        fractional_capability=fractional_capability,
    )
    now = datetime.now(UTC).isoformat()
    funnel = {
        "schema": "evidence_throughput_funnel_attribution_v1",
        "status": "GO" if stages else "DATA_UNAVAILABLE",
        "generated_at": now,
        "stages": stages,
        "rejection_attribution": rejection_rows,
        "stage_units_are_explicit": True,
        "raw_to_composite_is_aggregation_not_one_to_one": True,
        "no_generic_no_opportunities_reason": True,
        "policy_changed": False,
        **AUTHORITY,
    }
    funnel["content_hash"] = stable_hash(funnel)
    report = {
        "schema": "evidence_throughput_status_v1",
        "status": "GO",
        "generated_at": now,
        "funnel": {
            "raw_signals": _stage_pass(stages, "RAW_SIGNALS"),
            "composite_opportunities": _stage_pass(
                stages, "COMPOSITE_OPPORTUNITIES"
            ),
            "research_allocatable": _stage_pass(
                stages, "RESEARCH_ALLOCATABLE"
            ),
            "strategy_qualified": _stage_pass(
                stages, "STRATEGY_QUALIFIED"
            ),
            "economically_qualified": _stage_pass(
                stages, "ECONOMICALLY_QUALIFIED"
            ),
            "portfolio_qualified": _stage_pass(
                stages, "PORTFOLIO_QUALIFIED"
            ),
            "execution_ready": _stage_pass(stages, "EXECUTION_READY"),
        },
        "validation": validation,
        "monte_carlo": {
            "status": monte_carlo["status"],
            "evaluable_candidate_count": monte_carlo[
                "evaluable_candidate_count"
            ],
            "bootstrap_runs": monte_carlo["bootstrap_runs"],
        },
        "event_driven": {
            "status": event_driven["status"],
            "event_scope": event_driven["event_scope"],
            "causal_release_count": event_driven["causal_release_count"],
            "evaluable_candidate_count": event_driven[
                "evaluable_candidate_count"
            ],
        },
        "forward_independence": {
            "status": forward_independence["status"],
            "raw_closed_episode_count": forward_independence["counts"][
                "raw_closed_episode_count"
            ],
            "canonical_closed_episode_count": forward_independence[
                "counts"
            ]["canonical_closed_episode_count"],
            "independent_closed_episode_count": forward_independence[
                "counts"
            ]["independent_closed_episode_count"],
            "effective_sample_size": forward_independence["counts"][
                "effective_sample_size"
            ],
        },
        "finalists": {
            "near_finalist_count": finalists["near_finalist_count"],
            "financial_finalist_count": finalists[
                "financial_finalist_count"
            ],
            "closest_candidate": finalists.get("closest_candidate"),
        },
        "graduated_strategy_authority": {
            "status": authority_recommendations["status"],
            "tier_counts": authority_recommendations["tier_counts"],
            "canary_eligible_count": authority_recommendations[
                "canary_eligible_count"
            ],
        },
        "provisional_canary_readiness": {
            "status": canary_readiness["status"],
            "candidate_count": canary_readiness["candidate_count"],
            "affordable_whole_share_count": canary_readiness[
                "affordable_whole_share_count"
            ],
            "execution_ready_count": canary_readiness[
                "execution_ready_count"
            ],
            "blockers": canary_readiness["blockers"],
        },
        "automatic_gate_relaxation": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    root = project_root / OUTPUT_ROOT
    _write_json(root / "funnel-attribution.json", funnel)
    _write_json(root / "validation-throughput.json", validation)
    _write_json(
        root / "successive-halving.json",
        {
            "schema": "research_successive_halving_v1",
            "status": (
                "EVIDENCE_GAPS"
                if validation["missing_halving_stage_evidence"]
                else "GO"
            ),
            "generated_at": now,
            "stages": validation["successive_halving"],
            "missing_stage_evidence": validation[
                "missing_halving_stage_evidence"
            ],
            "automatic_promotion": False,
            **AUTHORITY,
        },
    )
    _write_json(root / "finalist-funnel.json", finalists)
    _write_json(root / "multiple-testing-trial-accounting.json", trial_accounting)
    _write_json(root / "monte-carlo.json", monte_carlo)
    _write_json(root / "event-driven.json", event_driven)
    _write_json(root / "pit-shariah-boundary.json", pit_boundary)
    _write_json(root / "forward-independence-audit.json", forward_independence)
    _write_json(root / "forward-sample-plan.json", sample_plan)
    _write_json(
        root / "strategy-authority-recommendations.json",
        authority_recommendations,
    )
    _write_json(
        root / "provisional-canary-readiness.json",
        canary_readiness,
    )
    _write_json(
        root / "evidence-distance.json",
        {
            "schema": "candidate_evidence_distance_v1",
            "status": "GO",
            "generated_at": now,
            "weights": config.get("evidence_distance_weights", {}),
            "candidates": [
                {
                    key: row.get(key)
                    for key in (
                        "strategy_id",
                        "formula",
                        "timeframe",
                        "asset_class",
                        "status",
                        "normalized_evidence_distance",
                        "evidence_completeness",
                    )
                }
                for row in finalists["candidates"]
            ],
            **AUTHORITY,
        },
    )
    _write_json(root / "status.json", report)
    pd.DataFrame(stages).to_parquet(
        root / "funnel-stages.parquet", index=False
    )
    pd.DataFrame(rejection_rows).to_parquet(
        root / "funnel-rejections.parquet", index=False
    )
    pd.DataFrame(finalists["candidates"]).to_parquet(
        root / "near-finalists.parquet", index=False
    )
    pd.DataFrame(sample_plan["candidates"]).to_parquet(
        root / "forward-sample-plan.parquet", index=False
    )
    pd.DataFrame(forward_independence["per_strategy"]).to_parquet(
        root / "forward-independence.parquet", index=False
    )
    return report


def evidence_throughput_status(project_root: Path) -> dict[str, Any]:
    path = project_root / OUTPUT_ROOT / "status.json"
    if not path.is_file():
        return {"schema": "evidence_throughput_status_v1", "status": "NOT_RUN"}
    return _read_json(path)


def _build_funnel(
    raw_signals: list[dict[str, Any]],
    opportunities: list[dict[str, Any]],
    *,
    top: Mapping[str, Any],
    gate_attribution: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gates = {
        str(row.get("gate")): row
        for row in _dict_rows(gate_attribution.get("gates"))
    }
    stages: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    raw_count = len(raw_signals)
    composite_count = len(opportunities)
    stages.append(
        _stage(
            "RAW_SIGNALS",
            raw_count,
            raw_count,
            before_scores=_scores(raw_signals, "confidence_score"),
            after_scores=_scores(raw_signals, "confidence_score"),
            input_unit="SIGNAL",
            output_unit="SIGNAL",
        )
    )
    residual = max(0, raw_count - composite_count)
    stages.append(
        _stage(
            "COMPOSITE_OPPORTUNITIES",
            raw_count,
            composite_count,
            before_scores=_scores(raw_signals, "confidence_score"),
            after_scores=_scores(opportunities, "opportunity_score"),
            input_unit="SIGNAL",
            output_unit="UNIQUE_ASSET_OPPORTUNITY",
            reason="AGGREGATED_DUPLICATE_OVERLAPPING_OR_NONCOMPOSITE_SIGNAL",
        )
    )
    if residual:
        rejections.append(
            _rejection_row(
                "RAW_TO_COMPOSITE",
                "AGGREGATED_DUPLICATE_OVERLAPPING_OR_NONCOMPOSITE_SIGNAL",
                residual,
                gates,
                attribution="AGGREGATE_RESIDUAL_NOT_ITEM_MAPPED",
            )
        )
    research = [
        row for row in opportunities if row.get("research_allocation_eligible")
    ]
    _append_transition(
        stages,
        rejections,
        stage="RESEARCH_ALLOCATABLE",
        transition="COMPOSITE_TO_RESEARCH",
        source=opportunities,
        passed=research,
        blocker_field="research_allocation_blockers",
        gates=gates,
    )
    strategy = [row for row in research if _strategy_qualified(row)]
    _append_transition(
        stages,
        rejections,
        stage="STRATEGY_QUALIFIED",
        transition="RESEARCH_TO_STRATEGY",
        source=research,
        passed=strategy,
        blocker_field="research_allocation_blockers",
        gates=gates,
        fallback_reason="STRATEGY_QUALIFICATION_REQUIRED",
    )
    economic = [row for row in strategy if _economically_qualified(row)]
    _append_transition(
        stages,
        rejections,
        stage="ECONOMICALLY_QUALIFIED",
        transition="STRATEGY_TO_ECONOMIC",
        source=strategy,
        passed=economic,
        blocker_field="execution_blockers",
        gates=gates,
        fallback_reason="ECONOMIC_QUALIFICATION_REQUIRED",
    )
    portfolio = [row for row in economic if _portfolio_qualified(row)]
    _append_transition(
        stages,
        rejections,
        stage="PORTFOLIO_QUALIFIED",
        transition="ECONOMIC_TO_PORTFOLIO",
        source=economic,
        passed=portfolio,
        blocker_field="position_management_blockers",
        gates=gates,
        fallback_reason="PORTFOLIO_QUALIFICATION_REQUIRED",
    )
    execution = [row for row in portfolio if row.get("deployment_eligible")]
    _append_transition(
        stages,
        rejections,
        stage="EXECUTION_READY",
        transition="PORTFOLIO_TO_EXECUTION",
        source=portfolio,
        passed=execution,
        blocker_field="deployment_blockers",
        gates=gates,
        fallback_reason="EXECUTION_READINESS_REQUIRED",
    )
    published_auto = int(top.get("automated_execution_eligible_count") or 0)
    stages[-1]["published_automated_execution_eligible_count"] = published_auto
    stages[-1]["count_consistent_with_publication"] = (
        published_auto == len(execution)
    )
    return stages, rejections


def _append_transition(
    stages: list[dict[str, Any]],
    rejections: list[dict[str, Any]],
    *,
    stage: str,
    transition: str,
    source: list[dict[str, Any]],
    passed: list[dict[str, Any]],
    blocker_field: str,
    gates: Mapping[str, Mapping[str, Any]],
    fallback_reason: str | None = None,
) -> None:
    passed_ids = {id(row) for row in passed}
    failed = [row for row in source if id(row) not in passed_ids]
    reasons = Counter(
        reason
        for row in failed
        for reason in _blockers(row, blocker_field, fallback_reason)
    )
    stages.append(
        _stage(
            stage,
            len(source),
            len(passed),
            before_scores=_scores(source, "opportunity_score"),
            after_scores=_scores(passed, "opportunity_score"),
            input_unit="UNIQUE_ASSET_OPPORTUNITY",
            output_unit="UNIQUE_ASSET_OPPORTUNITY",
            reason=reasons.most_common(1)[0][0] if reasons else None,
        )
    )
    for reason, count in reasons.most_common():
        rejections.append(
            _rejection_row(transition, reason, count, gates)
        )


def _stage(
    name: str,
    input_count: int,
    pass_count: int,
    *,
    before_scores: list[float],
    after_scores: list[float],
    input_unit: str,
    output_unit: str,
    reason: str | None = None,
) -> dict[str, Any]:
    rejected = max(0, input_count - pass_count)
    return {
        "stage": name,
        "input_count": input_count,
        "pass_count": pass_count,
        "reject_count": rejected,
        "pass_rate": _ratio(pass_count, input_count),
        "top_rejection_reason": reason,
        "median_score_before": _median(before_scores),
        "median_score_after": _median(after_scores),
        "input_unit": input_unit,
        "output_unit": output_unit,
    }


def _rejection_row(
    transition: str,
    reason: str,
    count: int,
    gates: Mapping[str, Mapping[str, Any]],
    *,
    attribution: str = "ITEM_MAPPED_BLOCKER",
) -> dict[str, Any]:
    evidence = gates.get(reason, {})
    return {
        "transition": transition,
        "reason": reason,
        "reject_count": count,
        "attribution": attribution,
        "counterfactual_EV_R": evidence.get("mean_counterfactual_net_R"),
        "counterfactual_sample_count": evidence.get(
            "performance_sample_count", 0
        ),
        "counterfactual_evidence_status": evidence.get(
            "evidence_status", "INSUFFICIENT_EVIDENCE"
        ),
    }


def _validation_throughput(
    project_root: Path,
    *,
    config: Mapping[str, Any],
    qualification: Mapping[str, Any],
    selection_bias: Mapping[str, Any],
    forward: Mapping[str, Any],
    monte_carlo: Mapping[str, Any],
    event_driven: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = (
        project_root / "output/research/phase11_12/strategy-summary.parquet"
    )
    summary = (
        pd.read_parquet(summary_path)
        if summary_path.is_file()
        else pd.DataFrame()
    )
    generated = int(len(summary))
    exact = int(summary.get("status", pd.Series(dtype=str)).eq("COMPLETE").sum())
    robust = int(qualification.get("robust_pass_count") or 0)
    selected = int(qualification.get("selected_candidate_count") or 0)
    source_hypotheses = int(selection_bias.get("source_hypothesis_count") or 0)
    forward_counts = forward.get("counts", {})
    closed = int(
        forward_counts.get("independent_closed_episode_count")
        or 0
    )
    raw_closed = int(forward_counts.get("closed_episode_count") or 0)
    validation_ratio = _ratio(robust, source_hypotheses or generated)
    exact_ratio = _ratio(exact, generated)
    threshold = float(config.get("validation_ratio_low_threshold") or 0.1)
    backlog = validation_ratio is None or validation_ratio < threshold
    allocations = config.get("compute_allocations", {})
    profile = "validation_backlog" if backlog else "balanced"
    allocation = dict(allocations.get(profile, {}))
    allocation_sum = round(sum(float(value) for value in allocation.values()), 8)
    summary_numeric = summary.copy()
    for column in (
        "CAGR",
        "period_profit_factor",
        "stress_50bps_profit_factor",
    ):
        if column in summary_numeric:
            summary_numeric[column] = pd.to_numeric(
                summary_numeric[column], errors="coerce"
            )
    qualified_rows = _dict_rows(qualification.get("strategies"))
    successive_halving = [
        _halving_stage("SANITY", exact, "MEASURED"),
        _halving_stage("CAUSALITY", exact, "CANONICAL_ENGINE_CONTRACT"),
        _halving_stage("FAST_BACKTEST", exact, "MEASURED"),
        _halving_stage(
            "NET_COST_TEST",
            int(
                summary_numeric.get(
                    "stress_50bps_profit_factor", pd.Series(dtype=float)
                ).gt(1.0).sum()
            ),
            "MEASURED",
        ),
        _halving_stage(
            "PARAMETER_PLATEAU",
            sum(
                float(row.get("parameter_plateau_ratio") or 0) >= 0.5
                for row in qualified_rows
            ),
            "MEASURED_SELECTED_COHORT",
        ),
        _halving_stage("UNIVERSE_STABILITY", robust, "MEASURED"),
        _halving_stage(
            "EXACT_EVENT_DRIVEN",
            (
                int(event_driven.get("evaluable_candidate_count") or 0)
                if event_driven.get("status") == "GO"
                else None
            ),
            (
                "MEASURED_CAUSAL_MACRO_RELEASE_WINDOWS"
                if event_driven.get("status") == "GO"
                else "UNAVAILABLE_CAUSAL_EVENT_WINDOW_EVIDENCE"
            ),
        ),
        _halving_stage("REGIME_SPLITS", robust, "NESTED_ROBUSTNESS_PROXY"),
        _halving_stage(
            "WALK_FORWARD",
            sum(int(row.get("fold_count") or 0) >= 5 for row in qualified_rows),
            "MEASURED",
        ),
        _halving_stage(
            "MONTE_CARLO",
            int(monte_carlo.get("evaluable_candidate_count") or 0),
            "MEASURED_OOS_FOLD_CLUSTER_BOOTSTRAP",
        ),
        _halving_stage(
            "DSR_PBO",
            int(
                selection_bias.get(
                    "multiple_testing_corrected_finalist_count"
                )
                or 0
            ),
            "MEASURED_GLOBAL_CORRECTION",
        ),
        _halving_stage(
            "SHADOW",
            int(qualification.get("forward_observer_candidate_count") or 0),
            "MEASURED",
        ),
        _halving_stage(
            "PAPER",
            0,
            "NO_STRATEGY_PAPER_EVIDENCE",
        ),
    ]
    return {
        "schema": "research_validation_throughput_v1",
        "status": "VALIDATION_BACKLOG" if backlog else "BALANCED",
        "source_hypothesis_count": source_hypotheses,
        "generated_strategy_variant_count": generated,
        "exact_backtest_complete_count": exact,
        "selected_nested_candidate_count": selected,
        "strategies_fully_validated_count": robust,
        "closed_forward_episode_count": closed,
        "raw_closed_forward_episode_count": raw_closed,
        "forward_episode_count_semantics": (
            "INDEPENDENT_CAUSAL_CLUSTERS_FOR_PROMOTION"
        ),
        "monte_carlo_evaluable_candidate_count": int(
            monte_carlo.get("evaluable_candidate_count") or 0
        ),
        "event_driven_evaluable_candidate_count": int(
            event_driven.get("evaluable_candidate_count") or 0
        ),
        "validation_ratio": validation_ratio,
        "exact_ratio": exact_ratio,
        "validation_ratio_low_threshold": threshold,
        "new_hypothesis_generation_throttled": backlog,
        "compute_profile": profile,
        "recommended_compute_allocation": allocation,
        "successive_halving": successive_halving,
        "missing_halving_stage_evidence": [
            row["stage"]
            for row in successive_halving
            if row["count"] is None
        ],
        "compute_allocation_sum": allocation_sum,
        "compute_allocation_valid": abs(allocation_sum - 1.0) < 1e-8,
        "configuration_source": str(CONFIG_PATH),
        "automatic_strategy_generation": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }


def _finalist_funnel(
    qualification: Mapping[str, Any],
    *,
    selection_bias: Mapping[str, Any],
    forward: Mapping[str, Any],
    config: Mapping[str, Any],
    trial_accounting: Mapping[str, Any],
    pit_boundary: Mapping[str, Any],
    paper_session: Mapping[str, Any],
    phase9: Mapping[str, Any],
) -> dict[str, Any]:
    forward_by_strategy = {
        str(row.get("strategy_id")): row
        for row in _dict_rows(forward.get("per_strategy"))
    }
    minimum_forward = int(
        config.get("minimum_forward_closed_episodes_per_candidate") or 30
    )
    maximum_blockers = int(config.get("near_finalist_maximum_blockers") or 4)
    trials_by_strategy = {
        str(row.get("strategy_id")): row
        for row in _dict_rows(trial_accounting.get("candidates"))
    }
    distance_weights = _normalized_weights(
        config.get("evidence_distance_weights")
    )
    paper_complete = (
        paper_session.get("paper_session_substatus")
        == "PAPER_SESSION_COMPLETE"
    )
    phase9_status = str(
        phase9.get("phase9_marker") or phase9.get("status") or "NOT_RUN"
    )
    phase9_adapter_go = (
        phase9_status == "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO"
    )
    candidates: list[dict[str, Any]] = []
    for row in _dict_rows(qualification.get("strategies")):
        strategy_id = str(row.get("strategy_id") or "")
        forward_row = forward_by_strategy.get(strategy_id, {})
        trial_row = trials_by_strategy.get(strategy_id, {})
        checks = {
            "positive_gross": _positive(row.get("combined_oos_return")),
            "positive_net": _positive(row.get("combined_period_profit_factor"), 1.0),
            "cost_stress_pass": _positive(row.get("cost_50bps_combined_return")),
            "oos_pass": bool(row.get("research_pass")),
            "walk_forward_pass": int(row.get("fold_count") or 0) >= 5
            and float(row.get("positive_fold_ratio") or 0) >= 0.6,
            "parameter_stability_pass": float(
                row.get("parameter_plateau_ratio") or 0
            )
            >= 0.5,
            "minimum_sample_pass": int(row.get("normal_cost_fill_count") or 0)
            >= 100,
            "portfolio_invariants_pass": bool(
                row.get("portfolio_invariants_go")
            ),
            "robustness_pass": bool(row.get("robust_pass")),
            "multiple_testing_pass": int(
                selection_bias.get("multiple_testing_corrected_finalist_count") or 0
            ) > 0 and bool(trial_row.get("candidate_multiple_testing_pass")),
            "historical_shariah_pit_pass": str(
                row.get("shariah_product_structure_status") or ""
            )
            == "HISTORICAL_ELIGIBILITY_VERIFIED",
            "independent_forward_sample_pass": int(
                forward_row.get("independent_closed_episode_count") or 0
            )
            >= minimum_forward,
        }
        blockers = [name.upper() for name, passed in checks.items() if not passed]
        raw_closed_forward = int(
            forward_row.get("closed_episode_count") or 0
        )
        closed_forward = int(
            forward_row.get("independent_closed_episode_count") or 0
        )
        evidence_components = _evidence_components(
            trial_row=trial_row,
            pit_boundary=pit_boundary,
            closed_forward=closed_forward,
            minimum_forward=minimum_forward,
            paper_complete=paper_complete,
            paper_session=paper_session,
            phase9_status=phase9_status,
            phase9_adapter_go=phase9_adapter_go,
            config=config,
        )
        evidence_completeness = {
            name: float(component["completeness"])
            for name, component in evidence_components.items()
        }
        normalized_distance = round(
            sum(
                distance_weights[name] * (1.0 - evidence_completeness[name])
                for name in distance_weights
            ),
            8,
        )
        historical_core_pass = all(
            checks[name]
            for name in (
                "positive_gross",
                "positive_net",
                "cost_stress_pass",
                "oos_pass",
                "walk_forward_pass",
                "parameter_stability_pass",
                "minimum_sample_pass",
                "portfolio_invariants_pass",
                "robustness_pass",
            )
        )
        financial_finalist = not blockers
        graduated = _graduated_strategy_authority(
            config=config,
            trial_row=trial_row,
            closed_forward=closed_forward,
            historical_core_pass=historical_core_pass,
            robust_pass=bool(row.get("robust_pass")),
            financial_finalist=financial_finalist,
            paper_complete=paper_complete,
        )
        status = (
            "FINANCIAL_FINALIST"
            if financial_finalist
            else "NEAR_FINALIST"
            if historical_core_pass and len(blockers) <= maximum_blockers
            else "ROBUST_RESEARCH_CANDIDATE"
            if bool(row.get("robust_pass"))
            else "RESEARCH_CANDIDATE"
        )
        priority = _research_priority(row, blocker_count=len(blockers))
        priority = round(priority * (1.0 - 0.5 * normalized_distance), 8)
        candidates.append(
            {
                "strategy_id": strategy_id,
                "formula": row.get("formula"),
                "timeframe": row.get("timeframe"),
                "asset_class": row.get("asset_class"),
                "status": status,
                "distance_to_finalist": len(blockers),
                "normalized_evidence_distance": normalized_distance,
                "evidence_completeness": evidence_completeness,
                "evidence_components": evidence_components,
                "dominant_evidence_gap": max(
                    evidence_components,
                    key=lambda name: distance_weights[name]
                    * float(evidence_components[name]["distance"]),
                ),
                "blockers": "|".join(blockers),
                "research_priority": priority,
                "closed_forward_episodes": closed_forward,
                "raw_closed_forward_episodes": raw_closed_forward,
                "independent_closed_forward_episodes": closed_forward,
                "forward_effective_sample_size": forward_row.get(
                    "effective_sample_size"
                ),
                "forward_independence_status": forward_row.get(
                    "independence_status"
                ),
                "minimum_forward_closed_episodes": minimum_forward,
                "combined_oos_CAGR": row.get("combined_oos_CAGR"),
                "combined_oos_Sharpe": row.get("combined_oos_Sharpe"),
                "combined_period_profit_factor": row.get(
                    "combined_period_profit_factor"
                ),
                "cost_50bps_combined_return": row.get(
                    "cost_50bps_combined_return"
                ),
                "maximum_drawdown": row.get("maximum_drawdown"),
                "checks": checks,
                "multiple_testing": trial_row,
                "paper_session_substatus": paper_session.get(
                    "paper_session_substatus", "PAPER_SESSION_STATUS_UNAVAILABLE"
                ),
                "phase9_adapter_status": phase9_status,
                "phase9_adapter_go": phase9_adapter_go,
                "phase9_adapter_is_not_natural_session_evidence": True,
                "evidence_tier": graduated["evidence_tier"],
                "recommended_strategy_authority": graduated[
                    "recommended_strategy_authority"
                ],
                "strategy_canary_eligible": graduated[
                    "strategy_canary_eligible"
                ],
                "soft_evidence_risk_multiplier": graduated[
                    "soft_evidence_risk_multiplier"
                ],
                "provisional_risk_fraction": graduated[
                    "provisional_risk_fraction"
                ],
                "risk_multiplier_components": graduated[
                    "risk_multiplier_components"
                ],
                "runtime_hard_gates_still_required": graduated[
                    "runtime_hard_gates_still_required"
                ],
                "strategy_authority_applied": False,
                "execution_authority": "NONE",
                "automatic_promotion": False,
            }
        )
    candidates.sort(
        key=lambda row: (
            {
                "FINANCIAL_FINALIST": 0,
                "NEAR_FINALIST": 1,
                "ROBUST_RESEARCH_CANDIDATE": 2,
                "RESEARCH_CANDIDATE": 3,
            }.get(str(row["status"]), 4),
            float(row["normalized_evidence_distance"]),
            int(row["distance_to_finalist"]),
            -float(row["research_priority"]),
            str(row["strategy_id"]),
        )
    )
    near = sum(row["status"] == "NEAR_FINALIST" for row in candidates)
    finalists = sum(
        row["status"] == "FINANCIAL_FINALIST" for row in candidates
    )
    stage_counts = {
        "strategies_considered": len(candidates),
        "positive_gross": _check_count(candidates, "positive_gross"),
        "positive_net": _check_count(candidates, "positive_net"),
        "cost_stress_pass": _check_count(candidates, "cost_stress_pass"),
        "oos_pass": _check_count(candidates, "oos_pass"),
        "walk_forward_pass": _check_count(candidates, "walk_forward_pass"),
        "parameter_stability_pass": _check_count(
            candidates, "parameter_stability_pass"
        ),
        "minimum_sample_pass": _check_count(
            candidates, "minimum_sample_pass"
        ),
        "multiple_testing_pass": _check_count(
            candidates, "multiple_testing_pass"
        ),
        "independent_forward_sample_pass": _check_count(
            candidates, "independent_forward_sample_pass"
        ),
        "financial_finalist": finalists,
    }
    report = {
        "schema": "financial_finalist_funnel_v1",
        "status": "FINANCIAL_FINALIST_GO" if finalists else "NO_FINANCIAL_FINALIST",
        "generated_at": datetime.now(UTC).isoformat(),
        "stages": stage_counts,
        "near_finalist_count": near,
        "financial_finalist_count": finalists,
        "closest_candidate": candidates[0] if candidates else None,
        "candidates": candidates,
        "thresholds_not_lowered": True,
        "selection_conditioned_history_not_independent": True,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _graduated_strategy_authority(
    *,
    config: Mapping[str, Any],
    trial_row: Mapping[str, Any],
    closed_forward: int,
    historical_core_pass: bool,
    robust_pass: bool,
    financial_finalist: bool,
    paper_complete: bool,
) -> dict[str, Any]:
    policy_raw = config.get("graduated_strategy_authority", {})
    policy = policy_raw if isinstance(policy_raw, Mapping) else {}
    enabled = bool(policy.get("enabled", True))
    pbo = _finite_or_none(trial_row.get("pbo"))
    pbo_multiplier = _threshold_multiplier(
        pbo,
        policy.get("pbo_multipliers"),
        unavailable=float(policy.get("unavailable_pbo_multiplier", 0.25)),
    )
    forward_multiplier = _threshold_multiplier(
        float(max(0, closed_forward)),
        policy.get("forward_episode_multipliers"),
        unavailable=0.0,
    )
    dsr = _bounded_float(trial_row.get("dsr_probability"))
    minimum_dsr = _bounded_float(
        policy.get("minimum_dsr_multiplier", 0.25)
    )
    dsr_multiplier = max(minimum_dsr, dsr)
    raw_multiplier = pbo_multiplier * forward_multiplier * dsr_multiplier
    provisional_cap = _bounded_float(
        policy.get("maximum_provisional_risk_multiplier", 0.2)
    )
    paper_required = bool(
        policy.get("canary_requires_complete_paper_session", True)
    )

    if not enabled:
        evidence_tier = "RESEARCH"
        recommended = "NONE"
        risk_multiplier = 0.0
    elif financial_finalist:
        evidence_tier = "FINANCIAL_FINALIST"
        recommended = "LIVE_NORMAL"
        risk_multiplier = min(1.0, raw_multiplier)
    elif historical_core_pass:
        evidence_tier = "PROVISIONAL_TRADABLE"
        canary_ready = paper_complete or not paper_required
        recommended = "CANARY" if canary_ready else "PROVISIONAL"
        risk_multiplier = min(provisional_cap, raw_multiplier)
    elif robust_pass:
        evidence_tier = "ROBUST_OBSERVER"
        recommended = "SHADOW"
        risk_multiplier = 0.0
    else:
        evidence_tier = "RESEARCH"
        recommended = "NONE"
        risk_multiplier = 0.0

    strategy_canary_eligible = bool(
        enabled
        and historical_core_pass
        and (paper_complete or not paper_required)
    )
    base_risk_fraction = max(
        0.0,
        min(0.02, float(policy.get("base_risk_fraction", 0.006))),
    )
    return {
        "evidence_tier": evidence_tier,
        "recommended_strategy_authority": recommended,
        "strategy_canary_eligible": strategy_canary_eligible,
        "soft_evidence_risk_multiplier": round(risk_multiplier, 8),
        "provisional_risk_fraction": round(
            base_risk_fraction * risk_multiplier,
            10,
        ),
        "risk_multiplier_components": {
            "pbo": round(pbo_multiplier, 8),
            "forward_sample": round(forward_multiplier, 8),
            "dsr": round(dsr_multiplier, 8),
        },
        "runtime_hard_gates_still_required": [
            "FRESH_VALID_PRICE",
            "CURRENT_PIT_SHARIAH_PASS",
            "POSITIVE_NET_EXPECTED_VALUE",
            "COMPLETE_STOP_AND_EXIT_STRUCTURE",
            "ACCEPTABLE_SPREAD_AND_LIQUIDITY",
            "PORTFOLIO_RISK_WITHIN_CAPS",
            "BROKER_RECONCILIATION_GO",
            "IDEMPOTENCY_GO",
            "EXPLICIT_EXECUTION_AUTHORITY",
        ],
        "strategy_authority_applied": False,
        "execution_authority": "NONE",
    }


def _strategy_authority_recommendations(
    finalists: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = []
    tier_counts: Counter[str] = Counter()
    for row in _dict_rows(finalists.get("candidates")):
        item = {
            key: row.get(key)
            for key in (
                "strategy_id",
                "formula",
                "timeframe",
                "asset_class",
                "status",
                "evidence_components",
                "evidence_tier",
                "recommended_strategy_authority",
                "strategy_canary_eligible",
                "soft_evidence_risk_multiplier",
                "provisional_risk_fraction",
                "risk_multiplier_components",
                "runtime_hard_gates_still_required",
                "phase9_adapter_status",
                "phase9_adapter_go",
                "phase9_adapter_is_not_natural_session_evidence",
                "paper_session_substatus",
            )
        }
        tier_counts[str(item["evidence_tier"])] += 1
        candidates.append(item)
    payload = {
        "schema": "graduated_strategy_authority_recommendations_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "policy": config.get("graduated_strategy_authority", {}),
        "tier_counts": dict(sorted(tier_counts.items())),
        "canary_eligible_count": sum(
            bool(row.get("strategy_canary_eligible")) for row in candidates
        ),
        "candidates": candidates,
        "statistical_uncertainty_reduces_risk": True,
        "runtime_hard_gates_remain_fail_closed": True,
        "strategy_authority_applied": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _provisional_canary_readiness(
    authority: Mapping[str, Any],
    *,
    live_allowlist: Mapping[str, Any],
    forward_observation: Mapping[str, Any],
    opportunities: list[dict[str, Any]],
    macro_score: Mapping[str, Any],
    fractional_capability: Mapping[str, Any],
) -> dict[str, Any]:
    authority_rows = {
        str(row.get("strategy_id")): row
        for row in _dict_rows(authority.get("candidates"))
        if row.get("recommended_strategy_authority")
        in {"PROVISIONAL", "CANARY", "LIVE_SMALL", "LIVE_NORMAL"}
    }
    allowlist_rows = {
        str(row.get("strategy_id")): row
        for row in _dict_rows(live_allowlist.get("strategies"))
        if row.get("status") == "PIT_LIVE_ALLOWLISTED"
    }
    observation_rows = {
        str(row.get("strategy_id")): row
        for row in _dict_rows(forward_observation.get("observations"))
    }
    opportunity_by_symbol = {
        str(row.get("ticker") or row.get("symbol") or "").upper(): row
        for row in opportunities
        if row.get("ticker") or row.get("symbol")
    }
    eurusd, fx_status = _eurusd_rate(macro_score)
    fractional_reference = fractional_capability.get(
        "contract_reference", {}
    )
    if not isinstance(fractional_reference, Mapping):
        fractional_reference = {}
    fractional_symbol = str(
        fractional_reference.get("symbol") or ""
    ).upper()
    fractional_increment_observed = (
        fractional_capability.get("classification")
        == "CONTRACT_FRACTIONAL_INCREMENT_OBSERVED"
    )
    account_fractional_permission_proven = bool(
        fractional_capability.get("account_fractional_permission_proven")
    )
    bracket_fractional_support_proven = bool(
        fractional_capability.get("fractional_bracket_support_proven")
    )
    rows: list[dict[str, Any]] = []
    for strategy_id in sorted(set(authority_rows) & set(allowlist_rows)):
        authority_row = authority_rows[strategy_id]
        allowlist_row = allowlist_rows[strategy_id]
        observed = observation_rows.get(strategy_id, {})
        allowed_symbols = {
            str(symbol).upper()
            for symbol in allowlist_row.get("allowed_symbols", [])
        }
        maximum_order_eur = _finite_or_none(
            allowlist_row.get("maximum_order_value_eur")
        )
        for signal in _dict_rows(observed.get("raw_active_signals")):
            symbol = str(signal.get("symbol") or "").upper()
            if symbol not in allowed_symbols:
                continue
            opportunity = opportunity_by_symbol.get(symbol, {})
            currency = str(opportunity.get("currency") or "").upper()
            entry_reference = _finite_or_none(signal.get("entry_reference"))
            fx_to_eur: float | None = None
            if currency == "EUR":
                fx_to_eur = 1.0
            elif currency == "USD" and eurusd is not None and eurusd != 0.0:
                fx_to_eur = 1.0 / eurusd
            notional_eur = (
                entry_reference * fx_to_eur
                if entry_reference is not None and fx_to_eur is not None
                else None
            )
            affordable = bool(
                maximum_order_eur is not None
                and notional_eur is not None
                and 0 < notional_eur <= maximum_order_eur
            )
            symbol_fractional_increment_observed = bool(
                fractional_increment_observed and symbol == fractional_symbol
            )
            fractional_execution_eligible = bool(
                symbol_fractional_increment_observed
                and account_fractional_permission_proven
                and bracket_fractional_support_proven
            )
            blockers = []
            if not authority_row.get("strategy_canary_eligible"):
                blockers.append("STRATEGY_CANARY_AUTHORITY_NOT_RECOMMENDED")
            if not bool(signal.get("currently_attested")):
                blockers.append("CURRENT_PIT_SHARIAH_ATTESTATION_REQUIRED")
            if str(signal.get("action") or "").upper() != "BUY":
                blockers.append("CURRENT_BUY_SETUP_REQUIRED")
            if str(signal.get("data_freshness") or "").upper() != "FRESH":
                blockers.append("FRESH_SIGNAL_REQUIRED")
            if signal.get("execution_envelope_status") != "GO":
                blockers.append("EXECUTION_ENVELOPE_REQUIRED")
            if maximum_order_eur is None or maximum_order_eur <= 0:
                blockers.append("VALID_LEVEL_ONE_ORDER_CAP_REQUIRED")
            if fx_to_eur is None:
                blockers.append("FRESH_FX_TO_EUR_REQUIRED")
            if entry_reference is None or entry_reference <= 0:
                blockers.append("VALID_ENTRY_REFERENCE_REQUIRED")
            elif not affordable:
                blockers.append("WHOLE_SHARE_NOTIONAL_EXCEEDS_LEVEL_ONE_CAP")
                if symbol_fractional_increment_observed:
                    blockers.append(
                        "FRACTIONAL_CONTRACT_INCREMENT_OBSERVED_BUT_ACCOUNT_OR_BRACKET_SUPPORT_UNPROVEN"
                    )
            blockers.extend(
                [
                    "REALTIME_TOP_OF_BOOK_REQUIRED_AT_EXECUTION",
                    "EXPLICIT_EXECUTION_AUTHORITY_REQUIRED",
                ]
            )
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "formula": authority_row.get("formula"),
                    "timeframe": authority_row.get("timeframe"),
                    "symbol": symbol,
                    "currency": currency or "UNAVAILABLE",
                    "current_pit_attested": bool(
                        signal.get("currently_attested")
                    ),
                    "signal_freshness": signal.get("data_freshness"),
                    "closed_bar_timestamp": observed.get(
                        "closed_bar_timestamp"
                    ),
                    "entry_reference": entry_reference,
                    "entry_reference_source": (
                        "FROZEN_FORWARD_OBSERVATION_NOT_LIVE_QUOTE"
                    ),
                    "fx_to_eur": fx_to_eur,
                    "fx_status": fx_status,
                    "estimated_one_share_notional_eur": (
                        round(notional_eur, 6)
                        if notional_eur is not None
                        else None
                    ),
                    "maximum_order_value_eur": maximum_order_eur,
                    "whole_share_affordable": affordable,
                    "fractional_contract_increment_observed": (
                        symbol_fractional_increment_observed
                    ),
                    "account_fractional_permission_proven": (
                        account_fractional_permission_proven
                    ),
                    "fractional_bracket_support_proven": (
                        bracket_fractional_support_proven
                    ),
                    "fractional_execution_eligible": (
                        fractional_execution_eligible
                    ),
                    "recommended_strategy_authority": authority_row.get(
                        "recommended_strategy_authority"
                    ),
                    "strategy_canary_eligible": bool(
                        authority_row.get("strategy_canary_eligible")
                    ),
                    "execution_ready": not blockers,
                    "blockers": sorted(set(blockers)),
                }
            )
    blocker_set: set[str] = set()
    for row in rows:
        row_blockers = row.get("blockers")
        if isinstance(row_blockers, list):
            blocker_set.update(str(blocker) for blocker in row_blockers)
    blockers = sorted(blocker_set)
    if not rows:
        blockers.append("NO_OVERLAPPING_PROVISIONAL_ALLOWLISTED_SETUP")
    payload = {
        "schema": "provisional_live_canary_readiness_v1",
        "status": (
            "CANARY_EXECUTION_READY"
            if any(row["execution_ready"] for row in rows)
            else "NO_EXECUTABLE_PROVISIONAL_CANDIDATE"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(rows),
        "affordable_whole_share_count": sum(
            bool(row["whole_share_affordable"]) for row in rows
        ),
        "execution_ready_count": sum(
            bool(row["execution_ready"]) for row in rows
        ),
        "candidates": rows,
        "blockers": sorted(set(blockers)),
        "fractional_shares_assumed": False,
        "fractional_contract_capability_status": fractional_capability.get(
            "status", "UNPROVEN"
        ),
        "fractional_writer_activation_allowed": False,
        "live_quote_used": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _eurusd_rate(macro_score: Mapping[str, Any]) -> tuple[float | None, str]:
    features = macro_score.get("features", {})
    if not isinstance(features, Mapping):
        return None, "EURUSD_UNAVAILABLE"
    row = features.get("EURUSD", {})
    if not isinstance(row, Mapping):
        return None, "EURUSD_UNAVAILABLE"
    value = _finite_or_none(row.get("original_value"))
    if row.get("status") != "VALID" or bool(row.get("stale")):
        return None, "EURUSD_STALE_OR_INVALID"
    if value is None or value <= 0:
        return None, "EURUSD_INVALID_VALUE"
    return value, "EURUSD_VALID_MARKET_CLOSE_REFERENCE"


def _strategy_qualified(row: Mapping[str, Any]) -> bool:
    allocation = row.get("strategy_allocation", {})
    return (
        isinstance(allocation, Mapping)
        and float(allocation.get("participating_weight") or 0) > 0
        and str(allocation.get("status") or "") != "UNAVAILABLE_FALLBACK_RAW_SIGNAL_QUALITY"
    )


def _multiple_testing_trial_accounting(
    qualification: Mapping[str, Any],
    *,
    selection_bias: Mapping[str, Any],
    summary: pd.DataFrame,
    nested_folds: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    registered = len(summary)
    unique = (
        int(summary["economic_outcome_fingerprint"].nunique(dropna=True))
        if "economic_outcome_fingerprint" in summary
        else registered
    )
    dsr_threshold = float(config.get("dsr_sharpe_threshold") or 0.0)
    dsr_gate = float(config.get("minimum_dsr_probability") or 0.95)
    maximum_pbo = float(config.get("maximum_pbo") or 0.40)
    pbo_by_timeframe = _walk_forward_rank_pbo(nested_folds)
    candidates = []
    for row in _dict_rows(qualification.get("strategies")):
        family = summary
        for column in ("formula", "timeframe", "asset_class"):
            if column in family and row.get(column) is not None:
                family = family.loc[family[column] == row.get(column)]
        family_registered = len(family)
        family_unique = (
            int(family["economic_outcome_fingerprint"].nunique(dropna=True))
            if "economic_outcome_fingerprint" in family
            else family_registered
        )
        sharpe = _finite_or_none(row.get("combined_oos_Sharpe"))
        dsr = deflated_sharpe_probability(
            sharpe,
            dsr_threshold,
            max(1, family_unique),
        )
        pbo = pbo_by_timeframe.get(str(row.get("timeframe")), {})
        pbo_value = _finite_or_none(pbo.get("PBO"))
        pbo_pass = pbo_value is not None and pbo_value <= maximum_pbo
        dsr_pass = dsr >= dsr_gate
        candidates.append(
            {
                "strategy_id": row.get("strategy_id"),
                "formula": row.get("formula"),
                "timeframe": row.get("timeframe"),
                "asset_class": row.get("asset_class"),
                "family_registered_trial_count": family_registered,
                "family_unique_economic_outcome_count": family_unique,
                "family_duplicate_outcome_count": max(
                    0, family_registered - family_unique
                ),
                "candidate_oos_sharpe": sharpe,
                "dsr_probability": round(dsr, 10),
                "dsr_gate": dsr_gate,
                "dsr_pass": dsr_pass,
                "pbo": pbo_value,
                "pbo_status": pbo.get(
                    "status", "UNAVAILABLE_INSUFFICIENT_FOLD_MATRIX"
                ),
                "pbo_method": "EXPANDING_WALK_FORWARD_RANK_BY_TIMEFRAME",
                "pbo_maximum": maximum_pbo,
                "pbo_pass": pbo_pass,
                "pbo_completeness": (
                    round(max(0.0, 1.0 - pbo_value), 10)
                    if pbo_value is not None
                    else 0.0
                ),
                "candidate_multiple_testing_pass": dsr_pass and pbo_pass,
            }
        )
    corrected_count = sum(
        bool(row["candidate_multiple_testing_pass"]) for row in candidates
    )
    payload = {
        "schema": "multiple_testing_trial_accounting_v1",
        "status": (
            "GO_MULTIPLE_TESTING_CORRECTED_CANDIDATES_AVAILABLE"
            if corrected_count
            else "BLOCKED_NO_MULTIPLE_TESTING_CORRECTED_CANDIDATE"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "registered_hypothesis_count": registered,
        "unique_economic_outcome_count": unique,
        "duplicate_outcome_count": max(0, registered - unique),
        "source_hypothesis_count": int(
            selection_bias.get("source_hypothesis_count") or 0
        ),
        "dsr_method": "EXISTING_SELECTION_COUNT_PENALTY",
        "dsr_sharpe_threshold": dsr_threshold,
        "minimum_dsr_probability": dsr_gate,
        "pbo_status": (
            "EXPANDING_WALK_FORWARD_RANK_AVAILABLE"
            if pbo_by_timeframe
            else "UNAVAILABLE_INSUFFICIENT_FOLD_MATRIX"
        ),
        "pbo_method": "EXPANDING_WALK_FORWARD_RANK_BY_TIMEFRAME",
        "pbo_is_classical_cscv": False,
        "pbo_by_timeframe": pbo_by_timeframe,
        "multiple_testing_corrected_finalist_count": corrected_count,
        "candidates": candidates,
        "thresholds_changed": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _fold_monte_carlo(
    folds: pd.DataFrame,
    *,
    qualification: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "strategy_id",
        "fold_id",
        "cost_bps",
        "period_profit_factor",
        "CAGR",
        "maximum_drawdown",
    }
    requested_runs = int(config.get("oos_fold_bootstrap_runs") or 5_000)
    runs = min(10_000, max(100, requested_runs))
    candidates = []
    if required.issubset(folds.columns):
        normal = folds.loc[
            pd.to_numeric(folds["cost_bps"], errors="coerce").eq(10.0)
        ].copy()
    else:
        normal = pd.DataFrame()
    for candidate in _dict_rows(qualification.get("strategies")):
        strategy_id = str(candidate.get("strategy_id") or "")
        if normal.empty:
            group = pd.DataFrame()
        else:
            group = normal.loc[normal["strategy_id"].astype(str) == strategy_id]
        clean = (
            group[
                [
                    "fold_id",
                    "period_profit_factor",
                    "CAGR",
                    "maximum_drawdown",
                ]
            ].copy()
            if not group.empty
            else pd.DataFrame()
        )
        if not clean.empty:
            for column in (
                "period_profit_factor",
                "CAGR",
                "maximum_drawdown",
            ):
                clean[column] = pd.to_numeric(clean[column], errors="coerce")
            clean = (
                clean.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .drop_duplicates(subset=["fold_id"], keep="first")
                .sort_values("fold_id")
            )
        fold_count = len(clean)
        if fold_count < 5:
            candidates.append(
                {
                    "strategy_id": strategy_id,
                    "formula": candidate.get("formula"),
                    "timeframe": candidate.get("timeframe"),
                    "status": "INSUFFICIENT_OOS_FOLDS",
                    "fold_count": fold_count,
                    "bootstrap_runs": 0,
                }
            )
            continue
        values = clean[
            ["period_profit_factor", "CAGR", "maximum_drawdown"]
        ].to_numpy(dtype=float)
        seed = int(
            stable_hash(
                {
                    "namespace": "EVIDENCE_OOS_FOLD_BOOTSTRAP_V1",
                    "strategy_id": strategy_id,
                }
            )[:8],
            16,
        )
        random = np.random.default_rng(seed)
        indices = random.integers(0, fold_count, size=(runs, fold_count))
        samples = values[indices]
        median_pf = np.median(samples[:, :, 0], axis=1)
        median_cagr = np.median(samples[:, :, 1], axis=1)
        worst_drawdown = np.min(samples[:, :, 2], axis=1)
        candidates.append(
            {
                "strategy_id": strategy_id,
                "formula": candidate.get("formula"),
                "timeframe": candidate.get("timeframe"),
                "status": "EVALUABLE_OOS_FOLD_BOOTSTRAP",
                "fold_count": fold_count,
                "bootstrap_runs": runs,
                "random_seed": seed,
                "probability_median_pf_above_one": round(
                    float(np.mean(median_pf > 1.0)), 8
                ),
                "probability_median_cagr_positive": round(
                    float(np.mean(median_cagr > 0.0)), 8
                ),
                "median_pf_p05": round(float(np.quantile(median_pf, 0.05)), 8),
                "median_pf_p50": round(float(np.quantile(median_pf, 0.50)), 8),
                "median_pf_p95": round(float(np.quantile(median_pf, 0.95)), 8),
                "median_cagr_p05": round(
                    float(np.quantile(median_cagr, 0.05)), 8
                ),
                "median_cagr_p50": round(
                    float(np.quantile(median_cagr, 0.50)), 8
                ),
                "worst_drawdown_p05": round(
                    float(np.quantile(worst_drawdown, 0.05)), 8
                ),
            }
        )
    evaluable = sum(
        row["status"] == "EVALUABLE_OOS_FOLD_BOOTSTRAP"
        for row in candidates
    )
    payload = {
        "schema": "evidence_oos_fold_monte_carlo_v1",
        "status": "GO" if evaluable else "INSUFFICIENT_OOS_FOLDS",
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "RESAMPLE_COMPLETE_NESTED_OOS_FOLDS_WITH_REPLACEMENT",
        "unit_of_resampling": "COMPLETE_OOS_FOLD",
        "daily_block_bootstrap_claimed": False,
        "bootstrap_runs": runs,
        "candidate_count": len(candidates),
        "evaluable_candidate_count": evaluable,
        "candidates": candidates,
        "thresholds_changed": False,
        "authority_gate_changed": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _exact_event_driven(
    oos_returns: pd.DataFrame,
    *,
    macro_events: Mapping[str, Any],
    qualification: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Segment existing OOS returns by causal macro-release windows."""
    releases = _contemporary_macro_releases(macro_events)
    required = {"strategy_id", "cost_bps", "date", "daily_return"}
    minimum_event = int(config.get("minimum_event_window_observations") or 20)
    minimum_non_event = int(
        config.get("minimum_non_event_window_observations") or 100
    )
    minimum_aligned_releases = int(
        config.get("minimum_aligned_macro_releases") or 10
    )
    candidates: list[dict[str, Any]] = []
    normal = pd.DataFrame()
    if required.issubset(oos_returns.columns):
        normal = oos_returns.loc[
            pd.to_numeric(oos_returns["cost_bps"], errors="coerce").eq(10.0)
        ].copy()
    for candidate in _dict_rows(qualification.get("strategies")):
        strategy_id = str(candidate.get("strategy_id") or "")
        timeframe = str(candidate.get("timeframe") or "").lower()
        base = {
            "strategy_id": strategy_id,
            "formula": candidate.get("formula"),
            "timeframe": timeframe,
            "asset_class": candidate.get("asset_class"),
        }
        if timeframe in {"1w", "1mo", "1m"}:
            candidates.append(
                {
                    **base,
                    "status": "UNAVAILABLE_TIMEFRAME_TOO_COARSE_FOR_EXACT_WINDOW",
                    "reason": "WEEKLY_OR_MONTHLY_RETURN_CANNOT_ISOLATE_RELEASE_WINDOW",
                    "evaluable": False,
                }
            )
            continue
        group = (
            normal.loc[normal["strategy_id"].astype(str) == strategy_id].copy()
            if not normal.empty
            else pd.DataFrame()
        )
        if group.empty or releases.empty:
            candidates.append(
                {
                    **base,
                    "status": "UNAVAILABLE_OOS_RETURNS_OR_CAUSAL_RELEASES",
                    "evaluable": False,
                }
            )
            continue
        group["date"] = pd.to_datetime(group["date"], errors="coerce", utc=True)
        group["daily_return"] = pd.to_numeric(
            group["daily_return"], errors="coerce"
        )
        group = group.dropna(subset=["date", "daily_return"])
        conflicts = (
            group.groupby("date", sort=False)["daily_return"].nunique(dropna=False)
            > 1
        )
        conflicting_timestamp_count = int(conflicts.sum())
        duplicate_timestamp_count = int(group.duplicated("date", keep=False).sum())
        if conflicting_timestamp_count:
            candidates.append(
                {
                    **base,
                    "status": "BLOCKED_CONFLICTING_OOS_TIMESTAMPS",
                    "evaluable": False,
                    "duplicate_timestamp_count": duplicate_timestamp_count,
                    "conflicting_timestamp_count": conflicting_timestamp_count,
                }
            )
            continue
        group = (
            group.drop_duplicates("date", keep="first")
            .sort_values("date")
            .reset_index(drop=True)
        )
        masks, aligned = _event_window_masks(
            group["date"], releases=releases, timeframe=timeframe
        )
        event_mask = masks.pop("ALL")
        event_metrics = _event_return_metrics(
            group.loc[event_mask, "daily_return"]
        )
        non_event_metrics = _event_return_metrics(
            group.loc[~event_mask, "daily_return"]
        )
        family_metrics = {
            family: {
                **_event_return_metrics(group.loc[mask, "daily_return"]),
                "aligned_release_count": aligned.get(family, 0),
            }
            for family, mask in sorted(masks.items())
        }
        evaluable = bool(
            event_metrics["observation_count"] >= minimum_event
            and non_event_metrics["observation_count"] >= minimum_non_event
            and aligned.get("ALL", 0) >= minimum_aligned_releases
        )
        candidates.append(
            {
                **base,
                "status": (
                    "EVALUABLE_CAUSAL_MACRO_RELEASE_WINDOWS"
                    if evaluable
                    else "INSUFFICIENT_EVENT_WINDOW_SAMPLE"
                ),
                "evaluable": evaluable,
                "oos_observation_count": len(group),
                "duplicate_timestamp_count": duplicate_timestamp_count,
                "conflicting_timestamp_count": 0,
                "aligned_release_count": aligned.get("ALL", 0),
                "event_window": "RELEASE_ANCHOR_THROUGH_NEXT_OBSERVED_SESSION",
                "event_returns": event_metrics,
                "non_event_returns": non_event_metrics,
                "mean_return_delta": _rounded_difference(
                    event_metrics.get("mean_return"),
                    non_event_metrics.get("mean_return"),
                ),
                "event_families": family_metrics,
            }
        )
    evaluable_count = sum(bool(row.get("evaluable")) for row in candidates)
    payload = {
        "schema": "evidence_exact_event_driven_v1",
        "status": "GO" if evaluable_count else "INSUFFICIENT_CAUSAL_EVENT_EVIDENCE",
        "generated_at": datetime.now(UTC).isoformat(),
        "method": "POINT_IN_TIME_MACRO_RELEASE_WINDOW_SEGMENTATION_OF_NESTED_OOS_RETURNS",
        "event_scope": "MACRO_RELEASE_WINDOWS_ONLY",
        "event_families": ["US_CPI", "US_PAYROLLS", "US_PCE"],
        "release_source": "FRED_ALFRED",
        "exact_causal_release_join": True,
        "causal_release_count": len(releases),
        "minimum_event_window_observations": minimum_event,
        "minimum_non_event_window_observations": minimum_non_event,
        "minimum_aligned_macro_releases": minimum_aligned_releases,
        "candidate_count": len(candidates),
        "evaluable_candidate_count": evaluable_count,
        "candidates": candidates,
        "earnings_event_coverage_status": "UNAVAILABLE_PLAN_NOT_ENTITLED",
        "sec_event_security_join_status": "NOT_INCLUDED_SECURITY_ID_MAPPING_UNPROVEN",
        "weekly_monthly_exact_window_status": "UNAVAILABLE_TIMEFRAME_TOO_COARSE",
        "daily_block_bootstrap_claimed": False,
        "thresholds_changed": False,
        "authority_gate_changed": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _contemporary_macro_releases(
    macro_events: Mapping[str, Any],
) -> pd.DataFrame:
    rows = _dict_rows(macro_events.get("historical_release_instances"))
    frame = pd.DataFrame(rows)
    required = {"event_id", "observation_date", "released_at", "release_status"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=["event_id", "released_at"])
    frame = frame.loc[
        frame["event_id"].isin({"US_CPI", "US_PAYROLLS", "US_PCE"})
        & frame["release_status"].eq("OBSERVED_HISTORICAL_RELEASE")
    ].copy()
    frame["released_at"] = pd.to_datetime(
        frame["released_at"], errors="coerce", utc=True
    )
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], errors="coerce", utc=True
    )
    lag = frame["released_at"].dt.normalize() - frame["observation_date"]
    frame = frame.loc[
        frame["released_at"].notna()
        & frame["observation_date"].notna()
        & lag.ge(pd.Timedelta(0))
        & lag.le(pd.Timedelta(days=120))
    ]
    return (
        frame[["event_id", "released_at"]]
        .drop_duplicates(["event_id", "released_at"])
        .sort_values(["released_at", "event_id"])
        .reset_index(drop=True)
    )


def _event_window_masks(
    timestamps: pd.Series,
    *,
    releases: pd.DataFrame,
    timeframe: str,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    ordered = pd.DatetimeIndex(timestamps)
    sessions = pd.DatetimeIndex(ordered.normalize().unique()).sort_values()
    families = sorted(str(value) for value in releases["event_id"].unique())
    masks: dict[str, np.ndarray] = {
        family: np.zeros(len(ordered), dtype=bool) for family in families
    }
    aligned = {family: 0 for family in families}
    intraday = timeframe.endswith("h") or timeframe.endswith("min")
    for row in releases.itertuples(index=False):
        released_at = pd.Timestamp(row.released_at)
        if intraday:
            anchor_position = int(ordered.searchsorted(released_at, side="left"))
        else:
            anchor_position = int(
                ordered.normalize().searchsorted(
                    released_at.normalize(), side="left"
                )
            )
        if anchor_position >= len(ordered):
            continue
        anchor = ordered[anchor_position]
        if anchor.normalize() - released_at.normalize() > pd.Timedelta(days=4):
            continue
        session_position = int(sessions.searchsorted(anchor.normalize()))
        selected_sessions = sessions[session_position : session_position + 2]
        mask = np.asarray(ordered.normalize().isin(selected_sessions))
        if intraday:
            mask &= np.asarray(ordered >= anchor)
        family = str(row.event_id)
        masks[family] |= mask
        aligned[family] += 1
    masks["ALL"] = (
        np.logical_or.reduce(list(masks.values()))
        if masks
        else np.zeros(len(ordered), dtype=bool)
    )
    aligned["ALL"] = sum(aligned.values())
    return masks, aligned


def _event_return_metrics(values: pd.Series) -> dict[str, Any]:
    clean = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    positives = clean.loc[clean > 0]
    negatives = clean.loc[clean < 0]
    gross_positive = float(positives.sum())
    gross_negative = abs(float(negatives.sum()))
    if clean.empty:
        profit_factor = None
        profit_factor_status = "NO_OBSERVATIONS"
    elif gross_negative == 0:
        profit_factor = None
        profit_factor_status = (
            "NO_LOSS_PERIODS" if gross_positive > 0 else "ALL_ZERO_RETURNS"
        )
    else:
        profit_factor = round(gross_positive / gross_negative, 8)
        profit_factor_status = "FINITE"
    return {
        "observation_count": len(clean),
        "positive_period_count": len(positives),
        "negative_period_count": len(negatives),
        "zero_period_count": int(clean.eq(0).sum()),
        "mean_return": round(float(clean.mean()), 10) if not clean.empty else None,
        "median_return": round(float(clean.median()), 10) if not clean.empty else None,
        "compounded_return": (
            round(float((1.0 + clean).prod() - 1.0), 10)
            if not clean.empty
            else None
        ),
        "period_profit_factor": profit_factor,
        "period_profit_factor_status": profit_factor_status,
    }


def _rounded_difference(left: Any, right: Any) -> float | None:
    left_value = _finite_or_none(left)
    right_value = _finite_or_none(right)
    if left_value is None or right_value is None:
        return None
    return round(left_value - right_value, 10)


def _walk_forward_rank_pbo(folds: pd.DataFrame) -> dict[str, dict[str, Any]]:
    required = {"strategy_id", "fold_id", "timeframe", "cost_bps", "Sharpe"}
    if folds.empty or not required.issubset(folds.columns):
        return {}
    normal = folds.loc[pd.to_numeric(folds["cost_bps"], errors="coerce").eq(10.0)]
    result: dict[str, dict[str, Any]] = {}
    for timeframe, group in normal.groupby("timeframe", sort=True):
        matrix = group.pivot_table(
            index="fold_id",
            columns="strategy_id",
            values="Sharpe",
            aggfunc="first",
        ).sort_index()
        in_sample = matrix.expanding(min_periods=1).mean().shift(1)
        report = probability_of_backtest_overfitting(in_sample, matrix)
        result[str(timeframe)] = {
            **report,
            "status": (
                "GO_EXPANDING_WALK_FORWARD_RANK"
                if report.get("status") == "GO"
                else report.get("status")
            ),
            "method": "PRIOR_FOLD_MEAN_RANK_VERSUS_NEXT_OOS_FOLD",
            "classical_cscv": False,
        }
    return result


def _pit_shariah_boundary(
    shariah: Mapping[str, Any],
    *,
    historical: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counts = shariah.get("current_status_counts", {})
    if not isinstance(counts, Mapping):
        counts = {}
    history = historical if isinstance(historical, Mapping) else {}
    total = sum(int(value or 0) for value in counts.values())
    current_eligible = int(counts.get("SHARIAH_ELIGIBLE_PIT") or 0)
    complete_screens = int(history.get("reconstructable_count") or 0)
    partial_screens = int(history.get("partial_screen_count") or 0)
    screen_count = complete_screens + partial_screens
    screen_symbols = int(history.get("screen_symbol_count") or 0)
    component_counts = {
        str(key): int(value or 0)
        for key, value in dict(
            history.get("available_component_counts") or {}
        ).items()
    }
    required_financial_components = (
        "market_cap",
        "debt_ratio",
        "cash_interest_ratio",
        "receivables_ratio",
    )
    component_observation_count = sum(
        component_counts.get(component, 0)
        for component in required_financial_components
    )
    component_observation_target = (
        screen_count * len(required_financial_components)
    )
    historical_coverage = _ratio(complete_screens, screen_count) or 0.0
    component_coverage = (
        _ratio(component_observation_count, component_observation_target)
        or 0.0
    )
    missing_components = sorted(
        {
            str(value)
            for value in history.get("missing_components", [])
            if value
        }
    )
    historical_gate_pass = bool(
        screen_count > 0 and complete_screens == screen_count
    )
    payload = {
        "schema": "historical_pit_shariah_boundary_v1",
        "status": (
            "GO_HISTORICAL_POINT_IN_TIME_COVERAGE_COMPLETE"
            if historical_gate_pass
            else "BLOCKED_HISTORICAL_POINT_IN_TIME_COVERAGE_INCOMPLETE"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "source_status": shariah.get("status", "UNAVAILABLE"),
        "historical_source_status": history.get(
            "status",
            shariah.get("historical_status", "SHARIAH_HISTORY_UNAVAILABLE"),
        ),
        "current_instrument_count": total,
        "current_pit_eligible_count": current_eligible,
        "current_pit_coverage_ratio": _ratio(current_eligible, total) or 0.0,
        "historical_screen_count": screen_count,
        "historical_screen_symbol_count": screen_symbols,
        "historical_complete_screen_count": complete_screens,
        "historical_partial_screen_count": partial_screens,
        "historical_point_in_time_coverage_ratio": historical_coverage,
        "historical_financial_component_coverage_ratio": component_coverage,
        "historical_financial_component_counts": component_counts,
        "historical_missing_components": missing_components,
        "historical_backtest_status": (
            "HISTORICAL_ELIGIBILITY_VERIFIED"
            if historical_gate_pass
            else "HISTORICAL_ELIGIBILITY_UNVERIFIED"
        ),
        "historical_evidence_scope": (
            "CAUSAL_SEC_FINANCIAL_COMPONENTS_PARTIAL"
            if screen_count > 0 and not historical_gate_pass
            else "CAUSAL_COMPLETE_SCREENS"
            if historical_gate_pass
            else "UNAVAILABLE"
        ),
        "complete_screen_requires_all_components": True,
        "current_attestation_allowed_for_live_screening": True,
        "current_status_backprojection_allowed": False,
        "financial_finalist_gate_pass": historical_gate_pass,
        "fallback_forbidden": True,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _forward_independence_audit(
    forward: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    policy_raw = config.get("forward_independence", {})
    policy = policy_raw if isinstance(policy_raw, Mapping) else {}
    cluster_gap_bars = max(0, int(policy.get("cluster_gap_bars") or 3))
    timeframe_seconds = {
        "1h": 3_600,
        "2h": 7_200,
        "4h": 14_400,
        "1d": 86_400,
        "1w": 604_800,
        "1mo": 2_592_000,
    }
    configured_seconds = policy.get("timeframe_seconds", {})
    if isinstance(configured_seconds, Mapping):
        for timeframe, seconds in configured_seconds.items():
            value = _finite_or_none(seconds)
            if value is not None and value > 0:
                timeframe_seconds[str(timeframe)] = int(value)

    source_rows = _dict_rows(forward.get("episodes"))
    raw_closed = [
        row
        for row in source_rows
        if str(row.get("outcome_status") or "").startswith("CLOSED_")
    ]
    canonical: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for row in raw_closed:
        reason = _forward_episode_exclusion_reason(row)
        if reason:
            excluded[reason] += 1
            continue
        canonical.append(row)

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in canonical:
        key = (
            str(row.get("strategy_id")),
            str(row.get("symbol")).upper(),
            str(row.get("market_regime") or "UNAVAILABLE_AT_DECISION"),
            _forward_event_class(row),
        )
        grouped.setdefault(key, []).append(row)

    clusters: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        strategy_id, symbol, regime, event_class = key
        ordered = sorted(
            rows,
            key=lambda row: (
                _forward_episode_start(row),
                str(row.get("episode_id") or row.get("episode_hash") or ""),
            ),
        )
        current: list[dict[str, Any]] = []
        current_end: pd.Timestamp | None = None
        for row in ordered:
            start = _forward_episode_start(row)
            end = _forward_episode_end(row)
            timeframe = str(row.get("timeframe") or "1d")
            cooldown = pd.Timedelta(
                seconds=timeframe_seconds.get(timeframe, 86_400)
                * cluster_gap_bars
            )
            if current and current_end is not None and start > current_end + cooldown:
                clusters.append(
                    _forward_cluster(
                        current,
                        strategy_id=strategy_id,
                        symbol=symbol,
                        regime=regime,
                        event_class=event_class,
                    )
                )
                current = []
                current_end = None
            current.append(row)
            current_end = end if current_end is None else max(current_end, end)
        if current:
            clusters.append(
                _forward_cluster(
                    current,
                    strategy_id=strategy_id,
                    symbol=symbol,
                    regime=regime,
                    event_class=event_class,
                )
            )

    source_strategy_ids = {
        str(row.get("strategy_id") or "")
        for row in _dict_rows(forward.get("per_strategy"))
        if row.get("strategy_id")
    }
    source_strategy_ids.update(
        str(row.get("strategy_id") or "")
        for row in source_rows
        if row.get("strategy_id")
    )
    per_strategy = [
        _forward_independence_summary(
            [row for row in raw_closed if str(row.get("strategy_id")) == strategy_id],
            [row for row in canonical if str(row.get("strategy_id")) == strategy_id],
            [row for row in clusters if row["strategy_id"] == strategy_id],
            strategy_id=strategy_id,
        )
        for strategy_id in sorted(source_strategy_ids)
    ]
    aggregate = _forward_independence_summary(
        raw_closed,
        canonical,
        clusters,
        strategy_id="ALL",
    )
    status = (
        "GO"
        if not excluded
        else "GO_WITH_DOCUMENTED_NONCANONICAL_EXCLUSIONS"
        if canonical
        else "INSUFFICIENT_CANONICAL_CLOSED_EPISODES"
    )
    report = {
        "schema": "forward_episode_independence_audit_v1",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "counts": aggregate,
        "per_strategy": per_strategy,
        "clusters": clusters,
        "exclusion_reason_counts": dict(sorted(excluded.items())),
        "cluster_policy": {
            "dimensions": [
                "strategy_id",
                "symbol",
                "market_regime",
                "event_class",
                "overlapping_or_nearby_time_window",
            ],
            "cluster_gap_bars": cluster_gap_bars,
            "timeframe_seconds": timeframe_seconds,
            "cluster_return": "MEAN_NET_RETURN_WITHIN_CLUSTER",
            "effective_sample": "LAG1_AUTOCORRELATION_OF_CLUSTER_RETURNS",
        },
        "raw_episode_count_is_not_promotion_sample": True,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    return report


def _forward_episode_exclusion_reason(row: Mapping[str, Any]) -> str | None:
    if not row.get("strategy_id"):
        return "MISSING_STRATEGY_ID"
    if not row.get("symbol"):
        return "MISSING_SYMBOL"
    if _finite_or_none(row.get("net_return")) is None:
        return "MISSING_FINITE_NET_RETURN"
    if _forward_timestamp(row.get("signal_timestamp")) is None:
        return "MISSING_SIGNAL_TIMESTAMP"
    if _forward_timestamp(
        row.get("exit_timestamp") or row.get("mark_timestamp")
    ) is None:
        return "MISSING_TERMINAL_TIMESTAMP"
    return None


def _forward_episode_start(row: Mapping[str, Any]) -> pd.Timestamp:
    return _forward_timestamp(
        row.get("entry_timestamp") or row.get("signal_timestamp")
    ) or pd.Timestamp.min.tz_localize("UTC")


def _forward_episode_end(row: Mapping[str, Any]) -> pd.Timestamp:
    return _forward_timestamp(
        row.get("exit_timestamp") or row.get("mark_timestamp")
    ) or _forward_episode_start(row)


def _forward_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _forward_event_class(row: Mapping[str, Any]) -> str:
    distance = _finite_or_none(row.get("earnings_distance_days"))
    if distance is None:
        return "EVENT_CONTEXT_UNAVAILABLE"
    if abs(distance) <= 5:
        return "EARNINGS_WINDOW_5D"
    if abs(distance) <= 20:
        return "EARNINGS_CONTEXT_20D"
    return "NO_NEAR_EARNINGS_EVENT"


def _forward_cluster(
    episodes: list[dict[str, Any]],
    *,
    strategy_id: str,
    symbol: str,
    regime: str,
    event_class: str,
) -> dict[str, Any]:
    starts = [_forward_episode_start(row) for row in episodes]
    ends = [_forward_episode_end(row) for row in episodes]
    returns = [float(row["net_return"]) for row in episodes]
    identity = {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "market_regime": regime,
        "event_class": event_class,
        "cluster_start": min(starts).isoformat(),
        "cluster_end": max(ends).isoformat(),
        "episode_ids": sorted(
            str(row.get("episode_id") or row.get("episode_hash") or "")
            for row in episodes
        ),
    }
    return {
        "cluster_id": stable_hash(identity),
        **identity,
        "episode_count": len(episodes),
        "cluster_net_return": round(float(np.mean(returns)), 10),
    }


def _forward_independence_summary(
    raw_closed: list[dict[str, Any]],
    canonical: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    *,
    strategy_id: str,
) -> dict[str, Any]:
    ordered = sorted(clusters, key=lambda row: (row["cluster_start"], row["cluster_id"]))
    returns = [float(row["cluster_net_return"]) for row in ordered]
    effective, autocorrelation = _forward_effective_sample_size(returns)
    return {
        "strategy_id": strategy_id,
        "raw_closed_episode_count": len(raw_closed),
        "canonical_closed_episode_count": len(canonical),
        "independent_closed_episode_count": len(clusters),
        "effective_sample_size": effective,
        "lag1_autocorrelation": autocorrelation,
        "distinct_asset_count": len({row["symbol"] for row in clusters}),
        "distinct_regime_count": len(
            {row["market_regime"] for row in clusters}
        ),
        "distinct_event_class_count": len(
            {row["event_class"] for row in clusters}
        ),
        "independence_status": (
            "EVALUABLE"
            if effective >= 30
            else "LOW_CONFIDENCE"
            if effective >= 10
            else "INSUFFICIENT_SAMPLE"
        ),
    }


def _forward_effective_sample_size(
    returns: list[float],
) -> tuple[float, float | None]:
    count = len(returns)
    if count < 3:
        return float(count), None
    left = np.asarray(returns[:-1], dtype=float)
    right = np.asarray(returns[1:], dtype=float)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float(count), None
    autocorrelation = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(autocorrelation):
        return float(count), None
    autocorrelation = min(0.95, max(-0.95, autocorrelation))
    effective = count * (1.0 - autocorrelation) / (1.0 + autocorrelation)
    return round(min(float(count), max(1.0, effective)), 6), round(
        autocorrelation, 6
    )


def _forward_with_independence(
    forward: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    independence_by_strategy = {
        str(row.get("strategy_id")): row
        for row in _dict_rows(audit.get("per_strategy"))
    }
    per_strategy = []
    for source in _dict_rows(forward.get("per_strategy")):
        strategy_id = str(source.get("strategy_id") or "")
        independence = independence_by_strategy.get(strategy_id, {})
        per_strategy.append(
            {
                **source,
                "raw_closed_episode_count": int(
                    independence.get("raw_closed_episode_count") or 0
                ),
                "independent_closed_episode_count": int(
                    independence.get("independent_closed_episode_count") or 0
                ),
                "effective_sample_size": independence.get(
                    "effective_sample_size", 0.0
                ),
                "independence_status": independence.get(
                    "independence_status", "INSUFFICIENT_SAMPLE"
                ),
            }
        )
    counts = dict(forward.get("counts", {}))
    audit_counts = audit.get("counts", {})
    counts["independent_closed_episode_count"] = int(
        audit_counts.get("independent_closed_episode_count") or 0
    )
    counts["forward_effective_sample_size"] = audit_counts.get(
        "effective_sample_size", 0.0
    )
    return {**forward, "counts": counts, "per_strategy": per_strategy}


def _forward_sample_plan(
    finalists: Mapping[str, Any],
    *,
    forward: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    acceleration = config.get("forward_sample_acceleration", {})
    if not isinstance(acceleration, Mapping):
        acceleration = {}
    minimum_assets = int(
        acceleration.get("minimum_distinct_assets_per_candidate") or 5
    )
    minimum_stock_sectors = int(
        acceleration.get("minimum_distinct_sectors_per_stock_candidate") or 3
    )
    maximum_candidates = int(
        acceleration.get("maximum_priority_candidates") or 8
    )
    episodes_by_strategy: dict[str, list[dict[str, Any]]] = {}
    for episode in _dict_rows(forward.get("episodes")):
        episodes_by_strategy.setdefault(
            str(episode.get("strategy_id") or ""), []
        ).append(episode)
    rows = []
    prioritized = [
        row
        for row in _dict_rows(finalists.get("candidates"))
        if row.get("status") == "NEAR_FINALIST"
    ]
    for index, row in enumerate(prioritized[:maximum_candidates], start=1):
        closed = int(row.get("closed_forward_episodes") or 0)
        raw_closed = int(row.get("raw_closed_forward_episodes") or 0)
        minimum = int(row.get("minimum_forward_closed_episodes") or 0)
        strategy_id = str(row.get("strategy_id") or "")
        episodes = episodes_by_strategy.get(strategy_id, [])
        observed_symbols = sorted(
            {
                str(episode.get("symbol"))
                for episode in episodes
                if episode.get("symbol")
            }
        )
        observed_sectors = sorted(
            {
                str(episode.get("sector"))
                for episode in episodes
                if episode.get("sector")
                and episode.get("sector") != "UNAVAILABLE_AT_DECISION"
            }
        )
        outcome_counts = Counter(
            str(episode.get("outcome_status") or "UNAVAILABLE")
            for episode in episodes
        )
        open_count = int(outcome_counts.get("OPEN", 0))
        awaiting_count = int(outcome_counts.get("AWAITING_NEXT_BAR", 0))
        blocked_count = sum(
            count
            for status, count in outcome_counts.items()
            if status.startswith("BLOCKED_")
        )
        asset_deficit = max(0, minimum_assets - len(observed_symbols))
        is_stock = str(row.get("asset_class")) == "STOCK"
        sector_deficit = (
            max(0, minimum_stock_sectors - len(observed_sectors))
            if is_stock
            else 0
        )
        next_actions = []
        if open_count:
            next_actions.append("CLOSE_EXISTING_OPEN_EPISODES_CAUSALLY")
        if awaiting_count:
            next_actions.append("WAIT_FOR_NEXT_CLOSED_BAR")
        if blocked_count:
            next_actions.append("AUDIT_EXECUTION_ENVELOPE_COVERAGE")
        if asset_deficit:
            next_actions.append("OBSERVE_MORE_FROZEN_UNIVERSE_ASSETS")
        if sector_deficit:
            next_actions.append("CAPTURE_SECTOR_AT_DECISION_POINT_IN_TIME")
        if max(0, minimum - closed):
            next_actions.append("COLLECT_NATURAL_CLOSED_FORWARD_EPISODES")
        rows.append(
            {
                "priority_rank": index,
                "strategy_id": strategy_id,
                "formula": row.get("formula"),
                "timeframe": row.get("timeframe"),
                "asset_class": row.get("asset_class"),
                "closed_forward_episodes": closed,
                "raw_closed_forward_episodes": raw_closed,
                "closed_episode_count_semantics": (
                    "INDEPENDENT_CAUSAL_CLUSTERS"
                ),
                "forward_effective_sample_size": row.get(
                    "forward_effective_sample_size"
                ),
                "closed_episode_deficit": max(0, minimum - closed),
                "observed_episode_count": len(episodes),
                "open_episode_count": open_count,
                "awaiting_next_bar_count": awaiting_count,
                "blocked_episode_count": blocked_count,
                "observed_distinct_asset_count": len(observed_symbols),
                "minimum_distinct_asset_target": minimum_assets,
                "distinct_asset_deficit": asset_deficit,
                "observed_distinct_sector_count": len(observed_sectors),
                "minimum_distinct_sector_target": (
                    minimum_stock_sectors if is_stock else 0
                ),
                "distinct_sector_deficit": sector_deficit,
                "sector_metadata_status": (
                    "AVAILABLE_AT_DECISION"
                    if observed_sectors
                    else "UNAVAILABLE_AT_DECISION"
                    if is_stock
                    else "NOT_APPLICABLE"
                ),
                "next_evidence_actions": next_actions,
                "normalized_evidence_distance": row.get(
                    "normalized_evidence_distance"
                ),
                "observation_scope": "FROZEN_QUALIFICATION_UNIVERSE_ONLY",
                "force_signal": False,
            }
        )
    return {
        "schema": "forward_sample_acceleration_plan_v1",
        "status": "FORWARD_COLLECTION_ACTIVE",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_count": len(rows),
        "candidates": rows,
        "prioritizes_existing_open_episodes": bool(
            acceleration.get("prioritize_existing_open_episodes", True)
        ),
        "sector_metadata_is_not_inferred": True,
        "strategy_contract_changed": False,
        "universe_contract_changed": False,
        "qualification_thresholds_changed": False,
        "forced_trades": 0,
        "forced_signals": 0,
        **AUTHORITY,
    }


def _evidence_components(
    *,
    trial_row: Mapping[str, Any],
    pit_boundary: Mapping[str, Any],
    closed_forward: int,
    minimum_forward: int,
    paper_complete: bool,
    paper_session: Mapping[str, Any],
    phase9_status: str,
    phase9_adapter_go: bool,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    dsr = _finite_or_none(trial_row.get("dsr_probability"))
    dsr_gate = float(config.get("minimum_dsr_probability") or 0.95)
    dsr_completeness = (
        min(1.0, max(0.0, dsr / dsr_gate))
        if dsr is not None and dsr_gate > 0
        else 0.0
    )
    pbo = _finite_or_none(trial_row.get("pbo"))
    maximum_pbo = float(config.get("maximum_pbo") or 0.40)
    if pbo is None:
        pbo_completeness = 0.0
    elif pbo <= maximum_pbo:
        pbo_completeness = 1.0
    elif maximum_pbo < 1.0:
        pbo_completeness = max(
            0.0, min(1.0, (1.0 - pbo) / (1.0 - maximum_pbo))
        )
    else:
        pbo_completeness = 0.0
    shariah_coverage = _bounded_float(
        pit_boundary.get("historical_point_in_time_coverage_ratio")
    )
    forward_completeness = min(
        1.0, closed_forward / max(1, minimum_forward)
    )
    values: dict[str, dict[str, Any]] = {
        "dsr": {
            "observed": dsr,
            "target": dsr_gate,
            "comparison": "GREATER_THAN_OR_EQUAL",
            "completeness": dsr_completeness,
            "gate_pass": dsr is not None and dsr >= dsr_gate,
            "next_action": "INCREASE_INDEPENDENT_ROBUSTNESS_EVIDENCE",
        },
        "pbo": {
            "observed": pbo,
            "target": maximum_pbo,
            "comparison": "LESS_THAN_OR_EQUAL",
            "completeness": pbo_completeness,
            "gate_pass": pbo is not None and pbo <= maximum_pbo,
            "next_action": "INCREASE_STABLE_OUT_OF_SAMPLE_RANK_EVIDENCE",
        },
        "historical_shariah": {
            "observed": shariah_coverage,
            "target": 1.0,
            "comparison": "GREATER_THAN_OR_EQUAL",
            "completeness": shariah_coverage,
            "gate_pass": shariah_coverage >= 1.0,
            "next_action": "ACQUIRE_OR_EXPLICITLY_BOUND_HISTORICAL_PIT_SHARIAH",
        },
        "forward": {
            "observed": closed_forward,
            "target": minimum_forward,
            "comparison": "GREATER_THAN_OR_EQUAL",
            "completeness": forward_completeness,
            "gate_pass": closed_forward >= minimum_forward,
            "remaining_closed_episodes": max(
                0, minimum_forward - closed_forward
            ),
            "next_action": "COLLECT_NATURAL_CLOSED_FORWARD_EPISODES",
        },
        "paper_session": {
            "observed": paper_session.get(
                "paper_session_substatus", "PAPER_SESSION_STATUS_UNAVAILABLE"
            ),
            "target": "PAPER_SESSION_COMPLETE",
            "comparison": "EQUAL",
            "completeness": 1.0 if paper_complete else 0.0,
            "gate_pass": paper_complete,
            "phase9_adapter_status": phase9_status,
            "phase9_adapter_gate_pass": phase9_adapter_go,
            "phase9_adapter_is_not_natural_session_evidence": True,
            "natural_strategy_session_gate_pass": paper_complete,
            "next_action": "COMPLETE_ONE_NATURAL_PAPER_SESSION",
        },
    }
    for component in values.values():
        completeness = round(float(component["completeness"]), 8)
        component["completeness"] = completeness
        component["distance"] = round(1.0 - completeness, 8)
        if component["gate_pass"]:
            component["next_action"] = "NO_ADDITIONAL_EVIDENCE_REQUIRED"
    return values


def _economically_qualified(row: Mapping[str, Any]) -> bool:
    markers = (
        "ECONOMIC",
        "EXPECTED_VALUE",
        "ECR",
        "SPREAD",
        "LIQUIDITY",
        "FUNDAMENTAL",
        "EVENT_RISK",
    )
    return not any(
        any(marker in reason for marker in markers)
        for reason in _blockers(row, "execution_blockers", None)
    )


def _portfolio_qualified(row: Mapping[str, Any]) -> bool:
    return not _blockers(row, "position_management_blockers", None)


def _research_priority(row: Mapping[str, Any], *, blocker_count: int) -> float:
    robustness = (
        min(1.0, max(0.0, float(row.get("positive_fold_ratio") or 0)))
        + min(1.0, max(0.0, float(row.get("parameter_plateau_ratio") or 0)))
        + min(1.0, max(0.0, float(row.get("combined_oos_Sharpe") or 0) / 1.5))
    ) / 3.0
    economic = (
        min(1.0, max(0.0, float(row.get("combined_oos_CAGR") or 0) / 0.25))
        + min(1.0, max(0.0, float(row.get("cost_50bps_combined_return") or 0)))
        + min(1.0, max(0.0, 1.0 - abs(float(row.get("maximum_drawdown") or 0))))
    ) / 3.0
    return round(robustness * economic / (1.0 + blocker_count), 8)


def _blockers(
    row: Mapping[str, Any], field: str, fallback: str | None
) -> list[str]:
    values = row.get(field, [])
    if not isinstance(values, list):
        values = []
    normalized = [str(value) for value in values if value]
    return normalized or ([fallback] if fallback else [])


def _scores(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    result = []
    for row in rows:
        candidate = row.get(key)
        if candidate is None:
            continue
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def _stage_pass(stages: list[dict[str, Any]], name: str) -> int:
    return next(
        (int(row["pass_count"]) for row in stages if row["stage"] == name),
        0,
    )


def _check_count(rows: list[dict[str, Any]], check: str) -> int:
    return sum(bool(row["checks"].get(check)) for row in rows)


def _positive(value: Any, threshold: float = 0.0) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > threshold
    except (TypeError, ValueError):
        return False


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bounded_float(value: Any) -> float:
    candidate = _finite_or_none(value)
    return min(1.0, max(0.0, candidate)) if candidate is not None else 0.0


def _threshold_multiplier(
    value: float | None,
    rows: Any,
    *,
    unavailable: float,
) -> float:
    if value is None or not math.isfinite(value):
        return _bounded_float(unavailable)
    if not isinstance(rows, list):
        return _bounded_float(unavailable)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        maximum = _finite_or_none(row.get("maximum_exclusive"))
        multiplier = _finite_or_none(row.get("multiplier"))
        if maximum is not None and multiplier is not None and value < maximum:
            return min(1.0, max(0.0, multiplier))
    return _bounded_float(unavailable)


def _normalized_weights(value: Any) -> dict[str, float]:
    names = ("dsr", "pbo", "historical_shariah", "forward", "paper_session")
    supplied = value if isinstance(value, Mapping) else {}
    weights = {
        name: max(0.0, float(supplied.get(name, 0.2))) for name in names
    }
    total = sum(weights.values())
    if total <= 0:
        return {name: 1.0 / len(names) for name in names}
    return {name: weight / total for name, weight in weights.items()}


def _halving_stage(
    stage: str,
    count: int | None,
    evidence_status: str,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "count": count,
        "evidence_status": evidence_status,
    }


def _median(values: list[float]) -> float | None:
    return round(float(pd.Series(values).median()), 8) if values else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 8) if denominator else None


def _dict_rows(value: Any) -> list[dict[str, Any]]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
