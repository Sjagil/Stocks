from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash


SESSION_PLAN_PATH = Path(
    "output/analysis/themes/opening-session-watchplan.json"
)
STRATEGY_SCORES_PATH = Path("output/dynamic/strategy_scores.json")
EPISODE_LEDGER_PATH = Path(
    "data/market_context/private/entry-episodes.jsonl"
)
OUTCOME_LEDGER_PATH = Path(
    "data/market_context/private/entry-episode-outcomes.jsonl"
)
OUTPUT_PATH = Path(
    "output/analysis/themes/provisional-candidate-assessment.json"
)
PERFORMANCE_TERMINALS = frozenset(
    {"STOPPED", "TP1_EXIT", "TP2_EXIT", "TRAIL_EXIT", "TIME_EXIT"}
)


def build_theme_provisional_assessment(
    project_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Assess tiny-risk research eligibility without granting authority."""
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    plan = _read_json(project_root / SESSION_PLAN_PATH)
    candidates = [
        row
        for row in plan.get("rows", [])
        if row.get("has_forward_observation")
    ]
    if not plan or not candidates:
        return _publish(
            project_root,
            {
                "schema": "theme_provisional_candidate_assessment_v1",
                "status": "NO_CURRENT_FORWARD_THEME_SETUP",
                "generated_at": observed_at.isoformat(),
                "rows": [],
                **_authority_contract(),
            },
        )

    scores = {
        str(row.get("strategy_id")): row
        for row in _read_json(project_root / STRATEGY_SCORES_PATH).get(
            "strategies", []
        )
        if isinstance(row, dict)
    }
    episodes = {
        str(row.get("episode_id")): row
        for row in _read_jsonl(project_root / EPISODE_LEDGER_PATH)
    }
    outcomes = {
        str(row.get("episode_id")): row
        for row in _read_jsonl(project_root / OUTCOME_LEDGER_PATH)
    }
    rows = []
    for candidate in candidates:
        strategy_ids = sorted(
            {
                str(value)
                for value in candidate.get("strategy_ids", [])
                if value
            }
            or {
                str(value)
                for value in _strategy_ids_from_plan(
                    plan,
                    symbol=str(candidate.get("symbol") or ""),
                )
            }
        )
        if not strategy_ids:
            strategy_ids = ["UNKNOWN"]
        for strategy_id in strategy_ids:
            rows.append(
                _assessment_row(
                    candidate=candidate,
                    strategy_id=strategy_id,
                    score=scores.get(strategy_id),
                    episodes=episodes,
                    outcomes=outcomes,
                )
            )

    tier_counts = Counter(str(row["recommended_tier"]) for row in rows)
    report = {
        "schema": "theme_provisional_candidate_assessment_v1",
        "status": "GO_WITH_HARD_BLOCKERS"
        if any(row["hard_gate_status"] == "BLOCKED" for row in rows)
        else "GO_OBSERVATION_ONLY",
        "generated_at": observed_at.isoformat(),
        "session_plan_content_hash": plan.get("content_hash"),
        "assessment_count": len(rows),
        "hard_gate_eligible_count": sum(
            row["hard_gate_status"] == "PASS" for row in rows
        ),
        "canonical_performance_episode_count": sum(
            row["forward_evidence"]["canonical_performance_episode_count"]
            for row in rows
        ),
        "tier_counts": dict(sorted(tier_counts.items())),
        "rows": rows,
        "policy": {
            "hard_failures_block": True,
            "soft_uncertainty_reduces_risk": True,
            "financial_finalist_required_for_normal_allocation": True,
            "financial_finalist_required_for_observation": False,
            "automatic_strategy_promotion": False,
            "automatic_execution_promotion": False,
        },
        **_authority_contract(),
    }
    report["content_hash"] = stable_hash(report)
    return _publish(project_root, report)


def _assessment_row(
    *,
    candidate: dict[str, Any],
    strategy_id: str,
    score: dict[str, Any] | None,
    episodes: dict[str, dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    symbol = str(candidate.get("symbol") or "")
    matched = []
    for episode_id, outcome in outcomes.items():
        episode = episodes.get(episode_id, {})
        if outcome.get("strategy_id") != strategy_id:
            continue
        if str(episode.get("symbol") or "") != symbol:
            continue
        matched.append(outcome)
    performance = [
        row
        for row in matched
        if row.get("terminal_status") in PERFORMANCE_TERMINALS
        and not row.get("excluded_from_performance_metrics")
        and row.get("net_R") is not None
    ]
    forward_multiplier = _forward_multiplier(len(performance))
    evidence = (score or {}).get("evidence") or {}
    historical_score = _bounded(float((score or {}).get("score") or 0.0))
    limitations = sorted(str(item) for item in evidence.get("limitations", []))
    multiple_testing_multiplier = (
        0.35
        if {"PBO_PASS_FAIL", "MULTIPLE_TESTING_PASS_FAIL"} & set(limitations)
        else 0.6
        if "DSR_PASS_FAIL" in limitations
        else 1.0
    )
    structure_multiplier = (
        0.5
        if any(
            str(item).startswith("THEME_STRUCTURE_")
            for item in candidate.get("soft_evidence_penalties", [])
        )
        else 1.0
    )
    fundamental_multiplier = (
        0.75
        if any(
            str(item).startswith("FUNDAMENTAL_DECISION_QUALITY_")
            for item in candidate.get("soft_evidence_penalties", [])
        )
        else 1.0
    )
    catalyst_multiplier = (
        0.9
        if candidate.get("catalyst_evidence_quality")
        in {"SOURCE_CONCENTRATED", "SPARSE"}
        else 1.0
    )
    soft_penalties = {
        str(item) for item in candidate.get("soft_evidence_penalties", [])
    }
    event_context_multiplier = (
        0.65
        if {"EVENT_RISK_NEAR", "EVENT_RISK_POST_EVENT"} & soft_penalties
        else 1.0
    )
    macro_event_multiplier = (
        0.85
        if {
            "MACRO_EVENT_RISK_IMMINENT",
            "MACRO_EVENT_RISK_NEAR",
        }
        & soft_penalties
        else 1.0
    )
    soft_multiplier = (
        historical_score
        * forward_multiplier
        * multiple_testing_multiplier
        * structure_multiplier
        * fundamental_multiplier
        * catalyst_multiplier
        * event_context_multiplier
        * macro_event_multiplier
    )
    hard_blockers = list(candidate.get("hard_blockers") or [])
    if score is None:
        hard_blockers.append("STRATEGY_EVIDENCE_RECORD_UNAVAILABLE")
    hard_blockers = sorted(set(hard_blockers))
    recommended_tier = (
        "RESEARCH_ONLY"
        if score is None
        else "ROBUST_OBSERVER"
        if hard_blockers or not performance
        else "PROVISIONAL_REVIEW_REQUIRED"
    )
    return {
        "theme": candidate.get("theme"),
        "symbol": symbol,
        "strategy_id": strategy_id,
        "strategy_family": (score or {}).get("family"),
        "timeframe": (score or {}).get("timeframe"),
        "hard_gate_status": "BLOCKED" if hard_blockers else "PASS",
        "hard_blockers": hard_blockers,
        "soft_evidence_penalties": candidate.get(
            "soft_evidence_penalties", []
        ),
        "historical_evidence": {
            "status": evidence.get("evidence_status") or "UNAVAILABLE",
            "score": round(historical_score, 8),
            "sample_count": evidence.get("sample_count", 0),
            "profit_factor": (
                (evidence.get("metrics") or {})
                .get("profit_factor", {})
                .get("raw")
            ),
            "expectancy": (
                (evidence.get("metrics") or {})
                .get("expectancy", {})
                .get("raw")
            ),
            "limitations": limitations,
        },
        "forward_evidence": {
            "terminal_outcome_count": len(matched),
            "terminal_status_counts": dict(
                sorted(Counter(str(row.get("terminal_status")) for row in matched).items())
            ),
            "canonical_performance_episode_count": len(performance),
            "forward_sample_multiplier": forward_multiplier,
        },
        "soft_risk_multiplier": round(soft_multiplier, 8),
        "soft_risk_components": {
            "historical_evidence": round(historical_score, 8),
            "forward_sample": forward_multiplier,
            "multiple_testing": multiple_testing_multiplier,
            "theme_structure": structure_multiplier,
            "fundamental_quality": fundamental_multiplier,
            "catalyst_quality": catalyst_multiplier,
            "event_context": event_context_multiplier,
            "macro_event_context": macro_event_multiplier,
        },
        "maximum_experimental_fraction_of_normal_risk": round(
            0.2 * soft_multiplier, 8
        ),
        "executable_risk_fraction": 0.0,
        "recommended_tier": recommended_tier,
        "strategy_authority_applied": False,
        "automatic_promotion": False,
        "entry_or_order_created": False,
    }


def _strategy_ids_from_plan(
    plan: dict[str, Any],
    *,
    symbol: str,
) -> list[str]:
    for row in plan.get("rows", []):
        if str(row.get("symbol") or "") == symbol:
            return list(row.get("strategy_ids") or [])
    return []


def _forward_multiplier(count: int) -> float:
    if count < 5:
        return 0.25
    if count < 15:
        return 0.4
    if count < 30:
        return 0.6
    if count < 60:
        return 0.8
    return 1.0


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))


def _authority_contract() -> dict[str, Any]:
    return {
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "order_calls": 0,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


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
