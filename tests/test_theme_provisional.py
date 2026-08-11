from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from stocks.analysis.theme_provisional import build_theme_provisional_assessment


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(root: Path, *, hard_blockers: list[str]) -> None:
    _json(
        root / "output/analysis/themes/opening-session-watchplan.json",
        {
            "content_hash": "PLAN",
            "rows": [
                {
                    "theme": "quantum_computing",
                    "symbol": "AAA",
                    "has_forward_observation": True,
                    "strategy_ids": ["STRAT-1"],
                    "hard_blockers": hard_blockers,
                    "soft_evidence_penalties": [
                        "THEME_STRUCTURE_BROAD_WEAKNESS"
                    ],
                    "catalyst_evidence_quality": "SOURCE_CONCENTRATED",
                }
            ],
        },
    )
    _json(
        root / "output/dynamic/strategy_scores.json",
        {
            "strategies": [
                {
                    "strategy_id": "STRAT-1",
                    "family": "trend",
                    "timeframe": "1d",
                    "score": 0.6,
                    "evidence": {
                        "evidence_status": "PARTIAL_EVIDENCE",
                        "sample_count": 50,
                        "limitations": ["PBO_PASS_FAIL"],
                        "metrics": {
                            "profit_factor": {"raw": 1.2},
                            "expectancy": {"raw": 0.01},
                        },
                    },
                }
            ]
        },
    )
    _jsonl(
        root / "data/market_context/private/entry-episodes.jsonl",
        [
            {
                "episode_id": "EP-1",
                "symbol": "AAA",
                "strategy_id": "STRAT-1",
            }
        ],
    )
    _jsonl(
        root / "data/market_context/private/entry-episode-outcomes.jsonl",
        [
            {
                "episode_id": "EP-1",
                "strategy_id": "STRAT-1",
                "terminal_status": "TP1_EXIT",
                "excluded_from_performance_metrics": False,
                "net_R": 0.5,
            }
        ],
    )


def test_provisional_assessment_penalizes_uncertainty_without_authority(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, hard_blockers=[])

    report = build_theme_provisional_assessment(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
    )
    row = report["rows"][0]

    assert row["hard_gate_status"] == "PASS"
    assert row["recommended_tier"] == "PROVISIONAL_REVIEW_REQUIRED"
    assert row["forward_evidence"]["canonical_performance_episode_count"] == 1
    assert 0 < row["maximum_experimental_fraction_of_normal_risk"] < 0.02
    assert row["soft_risk_components"]["fundamental_quality"] == 1.0
    assert row["soft_risk_components"]["event_context"] == 1.0
    assert row["executable_risk_fraction"] == 0.0
    assert row["strategy_authority_applied"] is False
    assert report["execution_authority"] == "NONE"
    assert report["order_calls"] == 0


def test_current_shariah_failure_remains_a_hard_blocker(tmp_path: Path) -> None:
    _fixture(tmp_path, hard_blockers=["CURRENT_SHARIAH_ATTESTATION_REQUIRED"])

    report = build_theme_provisional_assessment(tmp_path)
    row = report["rows"][0]

    assert report["status"] == "GO_WITH_HARD_BLOCKERS"
    assert row["hard_gate_status"] == "BLOCKED"
    assert row["recommended_tier"] == "ROBUST_OBSERVER"
    assert row["executable_risk_fraction"] == 0.0
    assert row["entry_or_order_created"] is False


def test_limited_fundamentals_reduce_soft_risk_not_authority(tmp_path: Path) -> None:
    _fixture(tmp_path, hard_blockers=[])
    plan_path = (
        tmp_path / "output/analysis/themes/opening-session-watchplan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["rows"][0]["soft_evidence_penalties"].append(
        "FUNDAMENTAL_DECISION_QUALITY_LIMITED_CORE_METRICS"
    )
    _json(plan_path, plan)

    report = build_theme_provisional_assessment(tmp_path)
    row = report["rows"][0]

    assert row["hard_gate_status"] == "PASS"
    assert row["soft_risk_components"]["fundamental_quality"] == 0.75
    assert row["executable_risk_fraction"] == 0.0


def test_near_event_reduces_hypothetical_risk_without_authority(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path, hard_blockers=[])
    plan_path = (
        tmp_path / "output/analysis/themes/opening-session-watchplan.json"
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["rows"][0]["soft_evidence_penalties"].append("EVENT_RISK_NEAR")
    _json(plan_path, plan)

    report = build_theme_provisional_assessment(tmp_path)
    row = report["rows"][0]

    assert row["soft_risk_components"]["event_context"] == 0.65
    assert row["executable_risk_fraction"] == 0.0
    assert report["execution_authority"] == "NONE"


def test_missing_strategy_evidence_fails_closed(tmp_path: Path) -> None:
    _fixture(tmp_path, hard_blockers=[])
    _json(tmp_path / "output/dynamic/strategy_scores.json", {"strategies": []})

    report = build_theme_provisional_assessment(tmp_path)
    row = report["rows"][0]

    assert row["recommended_tier"] == "RESEARCH_ONLY"
    assert "STRATEGY_EVIDENCE_RECORD_UNAVAILABLE" in row["hard_blockers"]
    assert row["soft_risk_multiplier"] == 0.0
