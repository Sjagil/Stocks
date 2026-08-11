from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash


THEME_ANALYSIS_PATH = Path(
    "output/analysis/themes/frontier-technology-energy.json"
)
OUTPUT_PATH = Path("output/analysis/themes/opening-session-watchplan.json")
MAX_RESEARCH_LEADERS_PER_THEME = 5


def build_theme_opening_session_plan(
    project_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Turn completed theme research into a fail-closed observation plan."""
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    analysis = _read_json(project_root / THEME_ANALYSIS_PATH)
    themes = analysis.get("themes") or {}
    if not themes:
        return _publish(
            project_root,
            {
                "schema": "theme_opening_session_watchplan_v1",
                "status": "BLOCKED_THEME_ANALYSIS_UNAVAILABLE",
                "generated_at": observed_at.isoformat(),
                "rows": [],
                **_authority_contract(),
            },
        )

    rows: list[dict[str, Any]] = []
    theme_confirmation_gates: dict[str, dict[str, Any]] = {}
    for theme_id, theme in sorted(themes.items()):
        instruments = theme.get("instruments") or []
        sector_structure = theme.get("sector_structure") or {}
        theme_confirmation_gates[theme_id] = _theme_confirmation_gate(
            theme_id,
            sector_structure,
        )
        leaders = {
            str(row.get("symbol"))
            for row in (theme.get("leadership") or [])[
                :MAX_RESEARCH_LEADERS_PER_THEME
            ]
        }
        structure_status = str(sector_structure.get("status") or "UNKNOWN")
        for instrument in instruments:
            forward = instrument.get("current_forward_observations") or {}
            observations = forward.get("items") or []
            symbol = str(instrument.get("symbol") or "")
            if not observations and symbol not in leaders:
                continue
            rows.append(
                _plan_row(
                    theme_id=theme_id,
                    structure_status=structure_status,
                    instrument=instrument,
                    observations=observations,
                    is_research_leader=symbol in leaders,
                )
            )

    rows.sort(
        key=lambda row: (
            0 if row["has_forward_observation"] else 1,
            -float(row.get("technical_score") or -2.0),
            row["symbol"],
        )
    )
    state_counts: dict[str, int] = {}
    for row in rows:
        state = str(row["session_readiness"])
        state_counts[state] = state_counts.get(state, 0) + 1
    regime_context = _regime_context(project_root, observed_at)
    theme_decision_matrix = _theme_decision_matrix(
        themes=themes,
        confirmation_gates=theme_confirmation_gates,
        plan_rows=rows,
        regime_context=regime_context,
    )
    report = {
        "schema": "theme_opening_session_watchplan_v1",
        "status": "GO_WITH_BLOCKERS" if rows else "NO_CURRENT_CANDIDATES",
        "generated_at": observed_at.isoformat(),
        "analysis_content_hash": analysis.get("content_hash"),
        "candidate_count": len(rows),
        "forward_observation_count": sum(
            row["has_forward_observation"] for row in rows
        ),
        "ready_observation_count": sum(
            row["session_readiness"] == "OBSERVATION_READY"
            for row in rows
        ),
        "theme_confirmation_gates": theme_confirmation_gates,
        "theme_decision_matrix": theme_decision_matrix,
        "regime_context": regime_context,
        "state_counts": dict(sorted(state_counts.items())),
        "rows": rows,
        "session_open_requirements": [
            "REFRESH_LATEST_COMPLETED_BARS",
            "RECOMPUTE_CONDITIONAL_LEVELS_FROM_FROZEN_STRATEGY",
            "REPLACE_INDICATIVE_REFERENCE_WITH_FRESH_EXECUTABLE_QUOTE",
            "VERIFY_CURRENT_SHARIAH_ATTESTATION",
            "VERIFY_CONTRACT_IDENTITY",
            "VERIFY_DATA_CAPABILITIES_WITHOUT_FALLBACK",
            "VERIFY_EVENT_CALENDAR_AND_MATERIAL_FILING_WINDOW",
            "OBSERVE_FRESH_TOP_OF_BOOK_AND_SPREAD_WHEN_ENTITLED",
            "REQUIRE_EXISTING_STRATEGY_SETUP",
            "KEEP_NEWS_MACRO_AND_THEME_CONTEXT_NON_TRIGGERING",
        ],
        **_authority_contract(),
    }
    report["content_hash"] = stable_hash(report)
    return _publish(project_root, report)


def _theme_confirmation_gate(
    theme_id: str,
    structure: dict[str, Any],
) -> dict[str, Any]:
    """Expose measured theme context without turning it into an entry gate."""
    cohorts = structure.get("cohorts") or {}
    if theme_id == "nuclear_uranium":
        components = [
            _confirmation_component(
                component="physical_proxy",
                cohort=cohorts.get("physical_uranium") or {},
                threshold=1.0,
            ),
            _confirmation_component(
                component="uranium_fund_breadth",
                cohort=cohorts.get("uranium_fund") or {},
                threshold=2 / 3,
            ),
            _confirmation_component(
                component="fuel_cycle_breadth",
                cohort=cohorts.get("fuel_cycle") or {},
                threshold=0.6,
            ),
        ]
        confirmation_status = _confirmation_status(components)
        return {
            "schema": "nuclear_uranium_confirmation_gate_v1",
            "theme": theme_id,
            "structure_status": structure.get("status") or "UNKNOWN",
            "confirmation_status": confirmation_status,
            "components": components,
            "required_component_count": 3,
            "confirmed_component_count": sum(
                bool(row["threshold_met"]) for row in components
            ),
            "session_open_revalidation_required": True,
            "recommended_action": (
                "REVALIDATE_PHYSICAL_FUNDS_AND_FUEL_CYCLE_AT_SESSION_OPEN"
            ),
            "context_only": True,
            "standalone_entry_allowed": False,
        }

    if theme_id == "quantum_computing":
        components = [
            _confirmation_component(
                component="platform_breadth",
                cohort=cohorts.get("platform_enabler") or {},
                threshold=0.5,
            ),
            _confirmation_component(
                component="pure_play_breadth",
                cohort=cohorts.get("pure_play") or {},
                threshold=0.5,
            ),
        ]
        confirmation_status = _confirmation_status(components)
        return {
            "schema": "quantum_ecosystem_confirmation_gate_v1",
            "theme": theme_id,
            "structure_status": structure.get("status") or "UNKNOWN",
            "confirmation_status": confirmation_status,
            "components": components,
            "required_component_count": 2,
            "confirmed_component_count": sum(
                bool(row["threshold_met"]) for row in components
            ),
            "session_open_revalidation_required": True,
            "recommended_action": (
                "REVALIDATE_PLATFORM_AND_PURE_PLAY_BREADTH_AT_SESSION_OPEN"
            ),
            "context_only": True,
            "standalone_entry_allowed": False,
        }

    return {
        "schema": "generic_theme_confirmation_gate_v1",
        "theme": theme_id,
        "structure_status": structure.get("status") or "UNKNOWN",
        "confirmation_status": "DESCRIPTIVE_ONLY",
        "components": [],
        "session_open_revalidation_required": True,
        "context_only": True,
        "standalone_entry_allowed": False,
    }


def _confirmation_component(
    *,
    component: str,
    cohort: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    evaluable_count = int(cohort.get("structure_evaluable_count") or 0)
    ratio_value = cohort.get("positive_structure_ratio")
    ratio = float(ratio_value) if ratio_value is not None else None
    available = evaluable_count > 0 and ratio is not None
    threshold_met = bool(
        evaluable_count > 0 and ratio is not None and ratio >= threshold
    )
    return {
        "component": component,
        "available": available,
        "structure_evaluable_count": evaluable_count,
        "positive_structure_count": int(
            cohort.get("positive_structure_count") or 0
        ),
        "positive_structure_ratio": ratio,
        "required_positive_structure_ratio": round(threshold, 6),
        "threshold_met": threshold_met,
        "positive_symbols": sorted(
            str(symbol) for symbol in (cohort.get("positive_symbols") or [])
        ),
    }


def _confirmation_status(components: list[dict[str, Any]]) -> str:
    if not components or not all(row["available"] for row in components):
        return "INSUFFICIENT_MEASUREMENTS"
    confirmed_count = sum(bool(row["threshold_met"]) for row in components)
    if confirmed_count == len(components):
        return "CONFIRMED_CONTEXT"
    if confirmed_count:
        return "PARTIAL_CONTEXT_WAIT"
    return "UNCONFIRMED_CONTEXT_WAIT"


def _theme_decision_matrix(
    *,
    themes: dict[str, Any],
    confirmation_gates: dict[str, dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    regime_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Summarize independent evidence dimensions without creating a signal."""
    matrix: dict[str, dict[str, Any]] = {}
    for theme_id, theme in sorted(themes.items()):
        instruments = [
            row
            for row in (theme.get("instruments") or [])
            if isinstance(row, dict)
        ]
        rows = [row for row in plan_rows if row.get("theme") == theme_id]
        gate = confirmation_gates.get(theme_id) or {}
        confirmation_status = str(
            gate.get("confirmation_status") or "INSUFFICIENT_MEASUREMENTS"
        )
        readiness_counts: dict[str, int] = {}
        hard_blocker_counts: dict[str, int] = {}
        for row in rows:
            readiness = str(row.get("session_readiness") or "UNKNOWN")
            readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
            for blocker in row.get("hard_blockers") or []:
                blocker_name = str(blocker)
                hard_blocker_counts[blocker_name] = (
                    hard_blocker_counts.get(blocker_name, 0) + 1
                )

        instrument_count = len(instruments)
        shariah_eligible_count = sum(
            bool((row.get("shariah") or {}).get("currently_eligible"))
            for row in instruments
        )
        fundamental_required = [
            row
            for row in instruments
            if (row.get("fundamentals") or {}).get(
                "fundamentals_required", True
            )
        ]
        fundamental_usable_count = sum(
            bool(
                ((row.get("fundamentals") or {}).get("data_quality") or {}).get(
                    "decision_usable"
                )
            )
            or str(
                ((row.get("fundamentals") or {}).get("data_quality") or {}).get(
                    "status"
                )
            )
            == "GO"
            for row in fundamental_required
        )
        macro_scores = [
            float(score)
            for row in instruments
            if (score := (row.get("macro_alignment") or {}).get("score"))
            is not None
        ]
        macro_scores.sort()
        macro_median = (
            macro_scores[len(macro_scores) // 2]
            if len(macro_scores) % 2 == 1
            else (
                macro_scores[len(macro_scores) // 2 - 1]
                + macro_scores[len(macro_scores) // 2]
            )
            / 2
            if macro_scores
            else None
        )
        observation_count = sum(
            bool(row.get("has_forward_observation")) for row in rows
        )
        ready_count = sum(
            str(row.get("session_readiness") or "").startswith(
                "OBSERVATION_READY"
            )
            for row in rows
        )
        risk_flags = []
        if confirmation_status != "CONFIRMED_CONTEXT":
            risk_flags.append("THEME_BREADTH_NOT_CONFIRMED")
        if instrument_count and shariah_eligible_count == 0:
            risk_flags.append("CURRENT_SHARIAH_COVERAGE_ZERO")
        elif shariah_eligible_count < instrument_count:
            risk_flags.append("CURRENT_SHARIAH_COVERAGE_PARTIAL")
        if (
            fundamental_required
            and fundamental_usable_count < len(fundamental_required)
        ):
            risk_flags.append("FUNDAMENTAL_DECISION_COVERAGE_PARTIAL")
        if observation_count == 0:
            risk_flags.append("NO_CURRENT_FORWARD_SETUP")
        if ready_count == 0:
            risk_flags.append("NO_OBSERVATION_READY")
        if any(
            blocker.startswith("MATERIAL_EVENT_")
            or blocker in {"EVENT_DATE_CONFLICT", "EVENT_CALENDAR_UNCERTAIN"}
            for blocker in hard_blocker_counts
        ):
            risk_flags.append("EVENT_RISK_BLOCKS_PRESENT")
        if regime_context.get("status") == "REGIME_CONFLICT_RISK_REDUCING":
            risk_flags.append("REGIME_CONFLICT_RISK_REDUCING")
        if regime_context.get("hmm_freshness_status") == "STALE":
            risk_flags.append("HMM_REGIME_STALE")
        if regime_context.get("macro_data_quality") not in {None, "GO"}:
            risk_flags.append("MACRO_DATA_QUALITY_INCOMPLETE")

        if ready_count:
            decision_state = "SESSION_REVALIDATION_CANDIDATE"
            recommended_action = "REVALIDATE_READY_OBSERVATIONS_AT_SESSION_OPEN"
        elif observation_count:
            decision_state = "CURRENT_SETUP_BLOCKED"
            recommended_action = "RESOLVE_VISIBLE_SETUP_BLOCKERS_WITHOUT_RELAXING_GATES"
        elif confirmation_status == "CONFIRMED_CONTEXT":
            decision_state = "CONFIRMED_CONTEXT_NO_SETUP"
            recommended_action = "WAIT_FOR_FROZEN_STRATEGY_SETUP"
        elif confirmation_status == "PARTIAL_CONTEXT_WAIT":
            decision_state = "PARTIAL_CONTEXT_WATCH"
            recommended_action = "REVALIDATE_THEME_BREADTH_AT_SESSION_OPEN"
        else:
            decision_state = "UNCONFIRMED_RESEARCH_WATCH"
            recommended_action = "REQUIRE_BREADTH_AND_STRATEGY_CONFIRMATION"

        matrix[theme_id] = {
            "schema": "theme_decision_matrix_v1",
            "theme": theme_id,
            "decision_state": decision_state,
            "confirmation_status": confirmation_status,
            "structure_status": gate.get("structure_status") or "UNKNOWN",
            "confirmed_component_count": int(
                gate.get("confirmed_component_count") or 0
            ),
            "required_component_count": int(
                gate.get("required_component_count") or 0
            ),
            "instrument_count": instrument_count,
            "candidate_count": len(rows),
            "forward_observation_count": observation_count,
            "ready_observation_count": ready_count,
            "readiness_counts": dict(sorted(readiness_counts.items())),
            "hard_blocker_counts": dict(sorted(hard_blocker_counts.items())),
            "current_shariah_eligible_count": shariah_eligible_count,
            "current_shariah_coverage_ratio": round(
                shariah_eligible_count / instrument_count, 6
            )
            if instrument_count
            else 0.0,
            "fundamental_required_count": len(fundamental_required),
            "fundamental_decision_usable_count": fundamental_usable_count,
            "fundamental_decision_usable_ratio": round(
                fundamental_usable_count / len(fundamental_required), 6
            )
            if fundamental_required
            else 1.0,
            "macro_alignment_median_score": (
                round(macro_median, 6) if macro_median is not None else None
            ),
            "leadership_symbols": [
                str(row.get("symbol"))
                for row in (theme.get("leadership") or [])[
                    :MAX_RESEARCH_LEADERS_PER_THEME
                ]
                if row.get("symbol")
            ],
            "risk_flags": sorted(risk_flags),
            "recommended_action": recommended_action,
            "context_only": True,
            "standalone_entry_allowed": False,
            "execution_authority": "NONE",
        }
    return matrix


def _regime_context(
    project_root: Path,
    observed_at: datetime,
) -> dict[str, Any]:
    """Compare independent regime models and expose disagreement explicitly."""
    dynamic = _read_json(project_root / "output/dynamic/current_regime.json")
    hmm = _read_json(
        project_root / "output/research/phase11_11/current.json"
    )
    macro = _read_json(project_root / "output/macro/score.json")
    dynamic_regime = str(dynamic.get("regime") or "UNAVAILABLE")
    dynamic_as_of = _as_utc(dynamic.get("as_of"))
    hmm_state = hmm.get("state") or {}
    hmm_as_of = _as_utc(hmm_state.get("as_of"))
    probabilities = {
        str(key): float(value)
        for key, value in (hmm_state.get("probabilities") or {}).items()
        if isinstance(value, (int, float))
    }
    dominant_hmm = (
        max(probabilities, key=lambda key: probabilities[key])
        if probabilities
        else "UNAVAILABLE"
    )
    dominant_probability = probabilities.get(dominant_hmm)
    risk_on_probability = float(probabilities.get("RISK_ON_TREND", 0.0))
    stress_probability = float(probabilities.get("STRESS_HIGH_VOL", 0.0))
    dynamic_risk_on = any(
        token in dynamic_regime for token in ("BULL", "RISK_ON")
    )
    dynamic_risk_off = any(
        token in dynamic_regime for token in ("BEAR", "RISK_OFF", "STRESS")
    )
    conflict = bool(
        (dynamic_risk_on and stress_probability >= 0.5)
        or (dynamic_risk_off and risk_on_probability >= 0.5)
    )
    hmm_age_days = _age_days(observed_at, hmm_as_of)
    dynamic_age_days = _age_days(observed_at, dynamic_as_of)
    hmm_freshness = (
        "FRESH"
        if hmm_age_days is not None and hmm_age_days <= 4.0
        else "STALE"
        if hmm_age_days is not None
        else "UNAVAILABLE"
    )
    macro_quality = str(
        (macro.get("data_quality") or {}).get("status") or "UNAVAILABLE"
    )
    macro_regime_payload = macro.get("regime")
    if isinstance(macro_regime_payload, dict):
        macro_regime = str(
            macro_regime_payload.get("overall_macro_regime")
            or macro_regime_payload.get("candidate_regime")
            or "UNAVAILABLE"
        )
    else:
        macro_regime = str(macro_regime_payload or "UNAVAILABLE")
    if conflict:
        status = "REGIME_CONFLICT_RISK_REDUCING"
        recommended_action = "USE_MOST_CONSERVATIVE_AVAILABLE_RISK_OVERLAY"
    elif dynamic_regime == "UNAVAILABLE" or dominant_hmm == "UNAVAILABLE":
        status = "DEGRADED_REGIME_INPUT_UNAVAILABLE"
        recommended_action = "WAIT_FOR_COMPLETE_REGIME_INPUTS"
    elif hmm_freshness != "FRESH":
        status = "DEGRADED_HMM_STALE"
        recommended_action = "REFRESH_CLOSED_CROSS_ASSET_HMM_INPUTS"
    elif macro_quality != "GO":
        status = "GO_WITH_MACRO_DATA_GAPS"
        recommended_action = "RETAIN_REDUCED_CONFIDENCE_FOR_MACRO_CONTEXT"
    else:
        status = "GO"
        recommended_action = "REVALIDATE_AT_NEXT_COMPLETED_SESSION"
    return {
        "schema": "theme_regime_context_v1",
        "status": status,
        "generated_at": observed_at.isoformat(),
        "dynamic_regime": dynamic_regime,
        "dynamic_as_of": dynamic_as_of.isoformat() if dynamic_as_of else None,
        "dynamic_age_days": (
            round(dynamic_age_days, 6)
            if dynamic_age_days is not None
            else None
        ),
        "hmm_dominant_state": dominant_hmm,
        "hmm_dominant_probability": (
            round(float(dominant_probability), 6)
            if dominant_probability is not None
            else None
        ),
        "hmm_probabilities": {
            key: round(value, 6) for key, value in sorted(probabilities.items())
        },
        "hmm_regime_multiplier": hmm_state.get("regime_multiplier"),
        "hmm_as_of": hmm_as_of.isoformat() if hmm_as_of else None,
        "hmm_age_days": round(hmm_age_days, 6) if hmm_age_days is not None else None,
        "hmm_freshness_status": hmm_freshness,
        "macro_regime": macro_regime,
        "macro_data_quality": macro_quality,
        "regime_conflict": conflict,
        "recommended_action": recommended_action,
        "risk_overlay_semantics": "CAN_ONLY_REDUCE_OR_BLOCK_RISK",
        "standalone_entry_allowed": False,
        "execution_authority": "NONE",
    }


def _as_utc(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _age_days(observed_at: datetime, value: datetime | None) -> float | None:
    if value is None:
        return None
    return max(0.0, (observed_at - value).total_seconds() / 86_400.0)


def _plan_row(
    *,
    theme_id: str,
    structure_status: str,
    instrument: dict[str, Any],
    observations: list[dict[str, Any]],
    is_research_leader: bool,
) -> dict[str, Any]:
    symbol = str(instrument.get("symbol") or "")
    contract_ok = (instrument.get("contract") or {}).get("status") == "RESOLVED"
    shariah_ok = bool((instrument.get("shariah") or {}).get("currently_eligible"))
    data_fresh = (
        (instrument.get("daily_snapshot") or {}).get("freshness_status") == "FRESH"
    )
    hard_veto_pass = bool(observations) and all(
        bool(row.get("hard_veto_pass")) for row in observations
    )
    hierarchy_ready = bool(observations) and all(
        bool(row.get("timeframe_hierarchy_ready")) for row in observations
    )
    conditional_plans = _conditional_trade_plans(instrument, observations)
    complete_conditional_plan = any(
        bool(row.get("levels_complete")) for row in conditional_plans
    )
    hard_blockers = []
    soft_evidence_penalties = []
    if not shariah_ok:
        hard_blockers.append("CURRENT_SHARIAH_ATTESTATION_REQUIRED")
    if not contract_ok:
        hard_blockers.append("CONTRACT_IDENTITY_NOT_RESOLVED")
    if not data_fresh:
        hard_blockers.append("LATEST_DAILY_BAR_NOT_FRESH")
    if observations and not hard_veto_pass:
        hard_blockers.append("FORWARD_HARD_VETO_FAILED")
    if observations and not hierarchy_ready:
        hard_blockers.append("TIMEFRAME_HIERARCHY_PENDING")
    if observations and not complete_conditional_plan:
        hard_blockers.append("CONDITIONAL_RISK_LEVELS_INCOMPLETE")
    if not observations:
        hard_blockers.append("NO_CURRENT_FORWARD_STRATEGY_SETUP")
    if structure_status in {"BROAD_WEAKNESS", "DIVERGENT_OR_WEAK"}:
        soft_evidence_penalties.append(f"THEME_STRUCTURE_{structure_status}")
    fundamentals = instrument.get("fundamentals") or {}
    fundamental_data_quality = fundamentals.get("data_quality") or {}
    fundamental_quality_status = str(
        fundamental_data_quality.get("status") or "UNKNOWN"
    )
    if (
        fundamentals.get("fundamentals_required", True)
        and fundamental_quality_status != "GO"
    ):
        soft_evidence_penalties.append(
            f"FUNDAMENTAL_DECISION_QUALITY_{fundamental_quality_status}"
        )
    event_risk = instrument.get("event_risk") or {}
    event_risk_status = str(
        event_risk.get("event_risk_status") or "EVENT_DATE_UNCERTAIN"
    )
    if event_risk_status == "EVENT_RISK_IMMINENT":
        hard_blockers.append("MATERIAL_EVENT_RISK_IMMINENT")
    elif event_risk_status == "EVENT_DATE_CONFLICT":
        hard_blockers.append("EVENT_DATE_CONFLICT")
    elif event_risk_status in {"EVENT_DATE_UNCERTAIN", "EVENT_DATA_STALE"}:
        hard_blockers.append("EVENT_CALENDAR_UNCERTAIN")
    elif event_risk_status in {"EVENT_RISK_NEAR", "EVENT_RISK_POST_EVENT"}:
        soft_evidence_penalties.append(event_risk_status)
    macro_event_risk_status = str(
        event_risk.get("macro_event_risk_status") or "MACRO_EVENT_DATE_UNCERTAIN"
    )
    if macro_event_risk_status in {
        "MACRO_EVENT_RISK_IMMINENT",
        "MACRO_EVENT_RISK_NEAR",
    }:
        soft_evidence_penalties.append(macro_event_risk_status)
    readiness = (
        "OBSERVATION_READY"
        if observations and not hard_blockers and not soft_evidence_penalties
        else "OBSERVATION_READY_WITH_SOFT_PENALTIES"
        if observations and not hard_blockers
        else "BLOCKED_CURRENT_SETUP"
        if observations
        else "RESEARCH_MONITOR_ONLY"
    )
    catalyst = (instrument.get("news") or {}).get("catalyst_summary") or {}
    macro_alignment = instrument.get("macro_alignment") or {}
    return {
        "theme": theme_id,
        "symbol": symbol,
        "selection_reasons": [
            reason
            for reason, selected in (
                ("CURRENT_FORWARD_OBSERVATION", bool(observations)),
                ("TOP_THEME_TECHNICAL_LEADER", is_research_leader),
            )
            if selected
        ],
        "has_forward_observation": bool(observations),
        "strategy_ids": sorted(
            {str(row.get("strategy_id")) for row in observations if row.get("strategy_id")}
        ),
        "forward_states": sorted(
            {str(row.get("state") or "UNKNOWN") for row in observations}
        ),
        "technical_score": instrument.get("technical_score"),
        "technical_classification": instrument.get(
            "technical_classification"
        ),
        "macro_profile": macro_alignment.get("profile"),
        "macro_classification": macro_alignment.get("classification"),
        "macro_context_score": macro_alignment.get("score"),
        "macro_context_confidence": macro_alignment.get("confidence"),
        "macro_missing_components": list(
            macro_alignment.get("missing_components") or []
        ),
        "theme_structure_status": structure_status,
        "catalyst_classification": catalyst.get("classification"),
        "catalyst_evidence_quality": catalyst.get("evidence_quality"),
        "fundamental_decision_quality": fundamental_quality_status,
        "event_risk_status": event_risk_status,
        "next_earnings_date": event_risk.get("next_earnings_date"),
        "days_to_event": event_risk.get("days_to_event"),
        "event_source_confidence": event_risk.get("source_confidence"),
        "macro_event_risk_status": macro_event_risk_status,
        "contract_status": (instrument.get("contract") or {}).get("status"),
        "shariah_status": instrument.get("shariah_status"),
        "daily_freshness_status": (
            instrument.get("daily_snapshot") or {}
        ).get("freshness_status"),
        "conditional_trade_plan_count": len(conditional_plans),
        "conditional_trade_plans": conditional_plans,
        "session_open_revalidation_required": True,
        "levels_are_orders": False,
        "session_readiness": readiness,
        "hard_blockers": hard_blockers,
        "soft_evidence_penalties": soft_evidence_penalties,
        "blockers": hard_blockers,
        "recommended_action": (
            "REVALIDATE_CONDITIONAL_PLAN_AT_SESSION_OPEN"
            if readiness.startswith("OBSERVATION_READY")
            else "NO_ACTION_RESEARCH_WATCH_ONLY"
        ),
        "entry_or_order_created": False,
    }


def _conditional_trade_plans(
    instrument: dict[str, Any],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    setups = (instrument.get("current_strategy_setups") or {}).get("items") or []
    signal_ids = {
        str(row.get("signal_id")) for row in observations if row.get("signal_id")
    }
    strategy_timeframes = {
        (str(row.get("strategy_id")), str(row.get("timeframe")))
        for row in observations
        if row.get("strategy_id")
    }
    selected = [
        row
        for row in setups
        if str(row.get("signal_id")) in signal_ids
        or (
            str(row.get("strategy_id")),
            str(row.get("timeframe")),
        )
        in strategy_timeframes
    ]
    plans = []
    for row in selected:
        required = (
            row.get("entry_zone_low"),
            row.get("entry_zone_high"),
            row.get("stop_loss"),
            row.get("take_profit_1"),
        )
        plans.append(
            {
                "signal_id": row.get("signal_id"),
                "strategy_id": row.get("strategy_id"),
                "timeframe": row.get("timeframe"),
                "signal_timestamp": row.get("signal_timestamp"),
                "data_timestamp": row.get("data_timestamp"),
                "expiration_timestamp": row.get("expiration_timestamp"),
                "preferred_entry": row.get("preferred_entry"),
                "entry_zone_low": row.get("entry_zone_low"),
                "entry_zone_high": row.get("entry_zone_high"),
                "invalidation_level": row.get("invalidation_level"),
                "stop_loss": row.get("stop_loss"),
                "stop_method": row.get("stop_method"),
                "take_profit_1": row.get("take_profit_1"),
                "take_profit_2": row.get("take_profit_2"),
                "take_profit_mode": row.get("take_profit_mode"),
                "reward_risk_1": row.get("reward_risk_1"),
                "reward_risk_2": row.get("reward_risk_2"),
                "market_reference_status": row.get("market_reference_status"),
                "market_reference_price": row.get("market_reference_price"),
                "market_reference_timestamp": row.get(
                    "market_reference_timestamp"
                ),
                "market_reference_provider": row.get(
                    "market_reference_provider"
                ),
                "market_reference_kind": row.get("market_reference_kind"),
                "market_reference_is_executable_quote": bool(
                    row.get("market_reference_is_executable_quote", False)
                ),
                "price_validity_status": row.get("price_validity_status"),
                "source_provider": row.get("source_provider"),
                "source_interval": row.get("source_interval"),
                "bar_closed": bool(row.get("bar_closed")),
                "levels_complete": all(value is not None for value in required),
                "quantity": None,
                "sizing_status": "RECOMPUTE_AFTER_SESSION_OPEN_REVALIDATION",
                "order_created": False,
                "execution_authority": "NONE",
            }
        )
    return plans


def _authority_contract() -> dict[str, Any]:
    return {
        "standalone_entry_allowed": False,
        "automatic_execution": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _publish(project_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return report
