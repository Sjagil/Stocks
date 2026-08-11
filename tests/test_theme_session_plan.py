from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from stocks.analysis.theme_session_plan import build_theme_opening_session_plan


def _write_analysis(root: Path, *, shariah_eligible: bool) -> None:
    path = root / "output/analysis/themes/frontier-technology-energy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "content_hash": "THEME-HASH",
                "themes": {
                    "quantum_computing": {
                        "sector_structure": {
                            "status": "BROAD_CONFIRMATION",
                            "cohorts": {
                                "platform_enabler": {
                                    "structure_evaluable_count": 4,
                                    "positive_structure_count": 3,
                                    "positive_structure_ratio": 0.75,
                                    "positive_symbols": ["AAA", "BBB", "CCC"],
                                },
                                "pure_play": {
                                    "structure_evaluable_count": 4,
                                    "positive_structure_count": 2,
                                    "positive_structure_ratio": 0.5,
                                    "positive_symbols": ["DDD", "EEE"],
                                },
                            },
                        },
                        "leadership": [{"symbol": "AAA"}],
                        "instruments": [
                            {
                                "symbol": "AAA",
                                "technical_score": 0.8,
                                "technical_classification": "STRONG_POSITIVE",
                                "daily_snapshot": {"freshness_status": "FRESH"},
                                "contract": {"status": "RESOLVED"},
                                "shariah": {
                                    "currently_eligible": shariah_eligible
                                },
                                "shariah_status": (
                                    "SHARIAH_ELIGIBLE_PIT"
                                    if shariah_eligible
                                    else "SHARIAH_ATTESTATION_REQUIRED"
                                ),
                                "fundamentals": {
                                    "fundamentals_required": True,
                                    "data_quality": {"status": "GO"},
                                },
                                "event_risk": {
                                    "event_risk_status": "EVENT_CLEAR",
                                    "next_earnings_date": "2026-10-28",
                                    "days_to_event": 81,
                                    "source_confidence": (
                                        "SINGLE_PROVIDER_ESTIMATE"
                                    ),
                                    "macro_event_risk_status": (
                                        "MACRO_EVENT_CLEAR"
                                    ),
                                },
                                "news": {
                                    "catalyst_summary": {
                                        "classification": (
                                            "POSITIVE_CATALYST_BALANCE"
                                        ),
                                        "evidence_quality": "DIVERSE_DIRECTIONAL",
                                    }
                                },
                                "current_forward_observations": {
                                    "items": [
                                        {
                                            "signal_id": "SIG-AAA",
                                            "strategy_id": "STRAT-AAA",
                                            "timeframe": "4h",
                                            "state": "ENTRY_CONFIRMATION_PENDING",
                                            "hard_veto_pass": True,
                                            "timeframe_hierarchy_ready": True,
                                        }
                                    ]
                                },
                                "current_strategy_setups": {
                                    "items": [
                                        {
                                            "signal_id": "SIG-AAA",
                                            "strategy_id": "STRAT-AAA",
                                            "timeframe": "4h",
                                            "signal_timestamp": "2026-08-08T11:00:00Z",
                                            "data_timestamp": "2026-08-08T10:00:00Z",
                                            "expiration_timestamp": "2026-08-10T20:00:00Z",
                                            "preferred_entry": "101.00",
                                            "entry_zone_low": "100.00",
                                            "entry_zone_high": "102.00",
                                            "invalidation_level": "96.00",
                                            "stop_loss": "96.00",
                                            "stop_method": "STRUCTURAL",
                                            "take_profit_1": "108.50",
                                            "take_profit_2": "113.50",
                                            "take_profit_mode": "PARTIAL_TARGETS",
                                            "reward_risk_1": "1.50",
                                            "reward_risk_2": "2.50",
                                            "market_reference_status": "FRESH",
                                            "market_reference_price": "101.10",
                                            "market_reference_timestamp": "2026-08-08T10:00:00Z",
                                            "market_reference_provider": "TEST",
                                            "market_reference_kind": "INDICATIVE_BAR_CLOSE",
                                            "market_reference_is_executable_quote": False,
                                            "price_validity_status": "CURRENT_ENTRY_REFERENCE_GO",
                                            "source_provider": "TEST",
                                            "source_interval": "4h",
                                            "bar_closed": True,
                                        }
                                    ]
                                },
                                "macro_alignment": {
                                    "profile": "QUANTUM_EARLY_STAGE_GROWTH",
                                    "classification": "MIXED_OR_NEUTRAL",
                                    "score": 0.49,
                                    "confidence": 0.65,
                                    "missing_components": [],
                                },
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_session_plan_is_observation_only_when_all_gates_pass(tmp_path: Path) -> None:
    _write_analysis(tmp_path, shariah_eligible=True)

    report = build_theme_opening_session_plan(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
    )

    assert report["status"] == "GO_WITH_BLOCKERS"
    assert report["ready_observation_count"] == 1
    assert report["rows"][0]["session_readiness"] == "OBSERVATION_READY"
    assert report["rows"][0]["conditional_trade_plan_count"] == 1
    assert report["rows"][0]["conditional_trade_plans"][0][
        "levels_complete"
    ] is True
    assert report["rows"][0]["conditional_trade_plans"][0][
        "market_reference_is_executable_quote"
    ] is False
    assert report["rows"][0]["macro_profile"] == (
        "QUANTUM_EARLY_STAGE_GROWTH"
    )
    assert report["rows"][0]["levels_are_orders"] is False
    assert report["rows"][0]["entry_or_order_created"] is False
    assert report["execution_authority"] == "NONE"
    assert report["orders_generated"] == 0
    gate = report["theme_confirmation_gates"]["quantum_computing"]
    assert gate["confirmation_status"] == "CONFIRMED_CONTEXT"
    assert gate["confirmed_component_count"] == 2
    assert gate["standalone_entry_allowed"] is False
    assert gate["context_only"] is True
    decision = report["theme_decision_matrix"]["quantum_computing"]
    assert decision["decision_state"] == "SESSION_REVALIDATION_CANDIDATE"
    assert decision["forward_observation_count"] == 1
    assert decision["ready_observation_count"] == 1
    assert decision["current_shariah_coverage_ratio"] == 1.0
    assert decision["fundamental_decision_usable_ratio"] == 1.0
    assert decision["standalone_entry_allowed"] is False
    assert decision["execution_authority"] == "NONE"


def test_session_plan_fails_closed_without_current_shariah(tmp_path: Path) -> None:
    _write_analysis(tmp_path, shariah_eligible=False)

    report = build_theme_opening_session_plan(tmp_path)
    row = report["rows"][0]

    assert row["session_readiness"] == "BLOCKED_CURRENT_SETUP"
    assert "CURRENT_SHARIAH_ATTESTATION_REQUIRED" in row["blockers"]
    assert row["recommended_action"] == "NO_ACTION_RESEARCH_WATCH_ONLY"
    assert report["automatic_execution"] is False
    decision = report["theme_decision_matrix"]["quantum_computing"]
    assert decision["decision_state"] == "CURRENT_SETUP_BLOCKED"
    assert "CURRENT_SHARIAH_COVERAGE_ZERO" in decision["risk_flags"]


def test_session_plan_blocks_when_analysis_is_missing(tmp_path: Path) -> None:
    report = build_theme_opening_session_plan(tmp_path)

    assert report["status"] == "BLOCKED_THEME_ANALYSIS_UNAVAILABLE"
    assert report["execution_authority"] == "NONE"


def test_session_plan_blocks_imminent_or_unknown_event_date(
    tmp_path: Path,
) -> None:
    _write_analysis(tmp_path, shariah_eligible=True)
    path = tmp_path / "output/analysis/themes/frontier-technology-energy.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    instrument = payload["themes"]["quantum_computing"]["instruments"][0]
    instrument["event_risk"] = {
        "event_risk_status": "EVENT_RISK_IMMINENT",
        "next_earnings_date": "2026-08-10",
        "days_to_event": 2,
        "source_confidence": "SINGLE_PROVIDER_ESTIMATE",
        "macro_event_risk_status": "MACRO_EVENT_CLEAR",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_theme_opening_session_plan(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
    )

    assert report["rows"][0]["session_readiness"] == "BLOCKED_CURRENT_SETUP"
    assert "MATERIAL_EVENT_RISK_IMMINENT" in report["rows"][0]["hard_blockers"]
    assert report["orders_generated"] == 0


def test_session_plan_blocks_incomplete_conditional_risk_levels(
    tmp_path: Path,
) -> None:
    _write_analysis(tmp_path, shariah_eligible=True)
    path = tmp_path / "output/analysis/themes/frontier-technology-energy.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    instrument = payload["themes"]["quantum_computing"]["instruments"][0]
    instrument["current_strategy_setups"]["items"][0]["stop_loss"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_theme_opening_session_plan(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
    )
    row = report["rows"][0]

    assert row["session_readiness"] == "BLOCKED_CURRENT_SETUP"
    assert "CONDITIONAL_RISK_LEVELS_INCOMPLETE" in row["hard_blockers"]
    assert row["conditional_trade_plans"][0]["levels_complete"] is False
    assert row["conditional_trade_plans"][0]["quantity"] is None
    assert row["entry_or_order_created"] is False


def test_session_plan_publishes_nuclear_confirmation_thresholds(
    tmp_path: Path,
) -> None:
    _write_analysis(tmp_path, shariah_eligible=True)
    path = tmp_path / "output/analysis/themes/frontier-technology-energy.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["themes"] = {
        "nuclear_uranium": {
            "sector_structure": {
                "status": "PARTIAL_CONFIRMATION",
                "cohorts": {
                    "physical_uranium": {
                        "structure_evaluable_count": 1,
                        "positive_structure_count": 1,
                        "positive_structure_ratio": 1.0,
                        "positive_symbols": ["U.UN"],
                    },
                    "uranium_fund": {
                        "structure_evaluable_count": 3,
                        "positive_structure_count": 2,
                        "positive_structure_ratio": 2 / 3,
                        "positive_symbols": ["URA", "URNM"],
                    },
                    "fuel_cycle": {
                        "structure_evaluable_count": 5,
                        "positive_structure_count": 2,
                        "positive_structure_ratio": 0.4,
                        "positive_symbols": ["CCJ", "LEU"],
                    },
                },
            },
            "leadership": [],
            "instruments": [],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_theme_opening_session_plan(tmp_path)
    gate = report["theme_confirmation_gates"]["nuclear_uranium"]

    assert gate["confirmation_status"] == "PARTIAL_CONTEXT_WAIT"
    assert gate["confirmed_component_count"] == 2
    assert gate["required_component_count"] == 3
    assert [
        row["required_positive_structure_ratio"]
        for row in gate["components"]
    ] == [1.0, 0.666667, 0.6]
    assert gate["standalone_entry_allowed"] is False
    assert report["execution_authority"] == "NONE"
    assert report["orders_generated"] == 0


def test_session_plan_marks_missing_theme_measurements_insufficient(
    tmp_path: Path,
) -> None:
    _write_analysis(tmp_path, shariah_eligible=True)
    path = tmp_path / "output/analysis/themes/frontier-technology-energy.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["themes"]["quantum_computing"]["sector_structure"] = {
        "status": "INSUFFICIENT_DATA",
        "cohorts": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    report = build_theme_opening_session_plan(tmp_path)
    gate = report["theme_confirmation_gates"]["quantum_computing"]

    assert gate["confirmation_status"] == "INSUFFICIENT_MEASUREMENTS"
    assert all(row["available"] is False for row in gate["components"])
    assert gate["standalone_entry_allowed"] is False


def test_session_plan_exposes_fresh_risk_reducing_regime_conflict(
    tmp_path: Path,
) -> None:
    _write_analysis(tmp_path, shariah_eligible=True)
    dynamic = tmp_path / "output/dynamic/current_regime.json"
    dynamic.parent.mkdir(parents=True)
    dynamic.write_text(
        json.dumps(
            {
                "as_of": "2026-08-08T10:00:00+00:00",
                "regime": "BULL_TREND_LOW_VOL",
            }
        ),
        encoding="utf-8",
    )
    hmm = tmp_path / "output/research/phase11_11/current.json"
    hmm.parent.mkdir(parents=True)
    hmm.write_text(
        json.dumps(
            {
                "state": {
                    "as_of": "2026-08-07T00:00:00",
                    "probabilities": {
                        "RISK_ON_TREND": 0.39,
                        "NEUTRAL_CHOPPY": 0.01,
                        "STRESS_HIGH_VOL": 0.60,
                    },
                    "regime_multiplier": 0.48,
                }
            }
        ),
        encoding="utf-8",
    )
    macro = tmp_path / "output/macro/score.json"
    macro.parent.mkdir(parents=True)
    macro.write_text(
        json.dumps(
            {
                "regime": {
                    "overall_macro_regime": "SLOWDOWN_DISINFLATION"
                },
                "data_quality": {"status": "DATA_INCOMPLETE"},
            }
        ),
        encoding="utf-8",
    )

    report = build_theme_opening_session_plan(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
    )

    context = report["regime_context"]
    assert context["status"] == "REGIME_CONFLICT_RISK_REDUCING"
    assert context["regime_conflict"] is True
    assert context["hmm_freshness_status"] == "FRESH"
    assert context["macro_regime"] == "SLOWDOWN_DISINFLATION"
    assert context["standalone_entry_allowed"] is False
    assert context["execution_authority"] == "NONE"
    matrix = report["theme_decision_matrix"]["quantum_computing"]
    assert "REGIME_CONFLICT_RISK_REDUCING" in matrix["risk_flags"]
    assert "MACRO_DATA_QUALITY_INCOMPLETE" in matrix["risk_flags"]
