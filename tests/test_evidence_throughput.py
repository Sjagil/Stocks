from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.research.evidence_throughput import (
    _exact_event_driven,
    _fold_monte_carlo,
    _forward_independence_audit,
    _walk_forward_rank_pbo,
    publish_evidence_throughput,
)


def _write_json(root: Path, relative: str, payload: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _opportunity(
    ticker: str,
    *,
    research: bool,
    blockers: list[str],
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "currency": "USD",
        "opportunity_score": 0.8 if research else 0.6,
        "research_allocation_eligible": research,
        "research_allocation_blockers": blockers,
        "execution_blockers": blockers,
        "position_management_blockers": blockers,
        "deployment_eligible": False,
        "deployment_blockers": (
            ["EXECUTION_AUTHORITY_NONE", "STRATEGY_DEPLOYMENT_EVIDENCE_REQUIRED"]
            if research
            else blockers
        ),
        "strategy_allocation": {
            "status": "DYNAMIC_WEIGHTED" if research else "UNAVAILABLE_FALLBACK_RAW_SIGNAL_QUALITY",
            "participating_weight": 0.5 if research else 0,
        },
    }


def _qualification_row() -> dict[str, object]:
    return {
        "strategy_id": "Q1",
        "formula": "nr7_breakout",
        "timeframe": "4h",
        "asset_class": "STOCK",
        "combined_oos_CAGR": 0.2,
        "combined_oos_Sharpe": 1.1,
        "combined_oos_return": 0.5,
        "combined_period_profit_factor": 1.3,
        "cost_50bps_combined_return": 0.2,
        "maximum_drawdown": -0.15,
        "fold_count": 6,
        "positive_fold_ratio": 0.8,
        "parameter_plateau_ratio": 0.7,
        "normal_cost_fill_count": 200,
        "portfolio_invariants_go": True,
        "research_pass": True,
        "robust_pass": True,
        "shariah_product_structure_status": "HISTORICAL_ELIGIBILITY_UNVERIFIED",
    }


def _seed(root: Path) -> None:
    _write_json(
        root,
        "config/research/evidence_throughput_v1.json",
        {
            "validation_ratio_low_threshold": 0.1,
            "minimum_forward_closed_episodes_per_candidate": 30,
            "near_finalist_maximum_blockers": 4,
            "graduated_strategy_authority": {
                "enabled": True,
                "base_risk_fraction": 0.006,
                "maximum_provisional_risk_multiplier": 0.2,
                "unavailable_pbo_multiplier": 0.25,
                "canary_requires_complete_paper_session": True,
                "pbo_multipliers": [
                    {"maximum_exclusive": 1.01, "multiplier": 0.5}
                ],
                "forward_episode_multipliers": [
                    {"maximum_exclusive": 5, "multiplier": 0.25},
                    {"maximum_exclusive": 1000, "multiplier": 1.0},
                ],
                "minimum_dsr_multiplier": 0.25,
            },
            "compute_allocations": {
                "validation_backlog": {
                    "new_hypotheses": 0.05,
                    "fast_validation": 0.2,
                    "exact_backtest": 0.2,
                    "cost_stress": 0.1,
                    "walk_forward": 0.15,
                    "robustness_dsr_pbo": 0.2,
                    "forward_analysis": 0.1,
                },
                "balanced": {"new_hypotheses": 1.0},
            },
        },
    )
    _write_json(
        root,
        "output/signals/latest_signals.json",
        {
            "signals": [
                {"confidence_score": 0.9},
                {"confidence_score": 0.8},
                {"confidence_score": 0.7},
                {"confidence_score": 0.6},
            ]
        },
    )
    _write_json(
        root,
        "output/portfolio/opportunity_ranking.json",
        {
            "opportunities": [
                _opportunity("AAA", research=True, blockers=[]),
                _opportunity(
                    "BBB",
                    research=False,
                    blockers=["SHARIAH_ATTESTATION_REQUIRED"],
                ),
            ]
        },
    )
    _write_json(
        root,
        "output/reports/top_opportunities.json",
        {"automated_execution_eligible_count": 0},
    )
    _write_json(
        root,
        "output/research/phase11_14/qualification.json",
        {
            "selected_candidate_count": 1,
            "robust_pass_count": 1,
            "strategies": [_qualification_row()],
        },
    )
    _write_json(
        root,
        "output/research/phase11_14/selection-bias-audit.json",
        {
            "source_hypothesis_count": 100,
            "multiple_testing_corrected_finalist_count": 0,
        },
    )
    _write_json(
        root,
        "output/research/phase11_14/forward-performance.json",
        {
            "counts": {"closed_episode_count": 2},
            "per_strategy": [
                {"strategy_id": "Q1", "closed_episode_count": 2}
            ],
            "episodes": [
                {
                    "strategy_id": "Q1",
                    "symbol": "AAA",
                    "sector": "Technology",
                    "outcome_status": "OPEN",
                },
                {
                    "strategy_id": "Q1",
                    "symbol": "BBB",
                    "sector": "UNAVAILABLE_AT_DECISION",
                    "outcome_status": "AWAITING_NEXT_BAR",
                },
                {
                    "strategy_id": "Q1",
                    "symbol": "CCC",
                    "sector": "Industrials",
                    "outcome_status": "CLOSED_STOP",
                    "timeframe": "4h",
                    "market_regime": "RISK_ON",
                    "signal_timestamp": "2026-07-20T12:00:00Z",
                    "entry_timestamp": "2026-07-20T16:00:00Z",
                    "exit_timestamp": "2026-07-21T12:00:00Z",
                    "net_return": -0.02,
                    "episode_id": "E1",
                },
                {
                    "strategy_id": "Q1",
                    "symbol": "CCC",
                    "sector": "Industrials",
                    "outcome_status": "CLOSED_TARGET",
                    "timeframe": "4h",
                    "market_regime": "RISK_ON",
                    "signal_timestamp": "2026-07-21T08:00:00Z",
                    "entry_timestamp": "2026-07-21T12:00:00Z",
                    "exit_timestamp": "2026-07-22T08:00:00Z",
                    "net_return": 0.03,
                    "episode_id": "E2",
                },
            ],
        },
    )
    _write_json(
        root,
        "output/ibkr/live/strategy-allowlist.json",
        {
            "status": "GO",
            "strategies": [
                {
                    "strategy_id": "Q1",
                    "status": "PIT_LIVE_ALLOWLISTED",
                    "allowed_symbols": ["AAA"],
                    "canary_notional_hard_cap_eur": "250",
                    "maximum_order_value_eur": "250",
                    "primary_sizing_authority": "RISK_PER_WHOLE_SHARE",
                }
            ],
        },
    )
    _write_json(
        root,
        "output/ibkr/phase9/status.json",
        {
            "status": "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO",
            "phase9_marker": "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO",
            "execution_authority": "MANUAL_PAPER_CANARY",
            "strategy_authority": "NONE",
        },
    )
    _write_json(
        root,
        "output/research/phase11_14/latest-forward-observation.json",
        {
            "status": "GO",
            "observations": [
                {
                    "strategy_id": "Q1",
                    "closed_bar_timestamp": "2026-07-21T00:00:00+00:00",
                    "raw_active_signals": [
                        {
                            "symbol": "AAA",
                            "action": "BUY",
                            "currently_attested": True,
                            "data_freshness": "FRESH",
                            "execution_envelope_status": "GO",
                            "entry_reference": 5.0,
                        }
                    ],
                }
            ],
        },
    )
    _write_json(
        root,
        "output/macro/score.json",
        {
            "features": {
                "EURUSD": {
                    "status": "VALID",
                    "stale": False,
                    "original_value": 2.0,
                }
            }
        },
    )
    summary = pd.DataFrame(
        {"status": ["COMPLETE"] * 10, "strategy_id": range(10)}
    )
    path = root / "output/research/phase11_12/strategy-summary.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(path, index=False)


def test_evidence_funnel_is_monotonic_and_fail_closed(tmp_path: Path) -> None:
    _seed(tmp_path)

    report = publish_evidence_throughput(tmp_path)

    assert report["funnel"] == {
        "raw_signals": 4,
        "composite_opportunities": 2,
        "research_allocatable": 1,
        "strategy_qualified": 1,
        "economically_qualified": 1,
        "portfolio_qualified": 1,
        "execution_ready": 0,
    }
    assert report["execution_authority"] == "NONE"
    assert report["automatic_orders"] == 0
    assert report["FINANCIAL_FINALIST_GO"] is False


def test_validation_backlog_prioritizes_evidence_not_generation(
    tmp_path: Path,
) -> None:
    _seed(tmp_path)

    report = publish_evidence_throughput(tmp_path)
    validation = report["validation"]

    assert validation["status"] == "VALIDATION_BACKLOG"
    assert validation["validation_ratio"] == 0.01
    assert validation["new_hypothesis_generation_throttled"] is True
    assert validation["recommended_compute_allocation"]["new_hypotheses"] == 0.05
    assert validation["compute_allocation_valid"] is True


def test_near_finalist_keeps_real_blockers_visible(tmp_path: Path) -> None:
    _seed(tmp_path)

    publish_evidence_throughput(tmp_path)
    finalist = json.loads(
        (
            tmp_path
            / "output/research/evidence_throughput/finalist-funnel.json"
        ).read_text(encoding="utf-8")
    )
    candidate = finalist["candidates"][0]

    assert candidate["status"] == "NEAR_FINALIST"
    assert candidate["distance_to_finalist"] == 3
    assert 0.0 <= candidate["normalized_evidence_distance"] <= 1.0
    assert set(candidate["evidence_completeness"]) == {
        "dsr",
        "pbo",
        "historical_shariah",
        "forward",
        "paper_session",
    }
    assert candidate["evidence_components"]["pbo"]["gate_pass"] is False
    assert candidate["evidence_components"]["forward"][
        "remaining_closed_episodes"
    ] == 29
    assert candidate["raw_closed_forward_episodes"] == 2
    assert candidate["independent_closed_forward_episodes"] == 1
    assert candidate["forward_effective_sample_size"] == 1.0
    paper = candidate["evidence_components"]["paper_session"]
    assert paper["phase9_adapter_gate_pass"] is True
    assert paper["natural_strategy_session_gate_pass"] is False
    assert paper["phase9_adapter_is_not_natural_session_evidence"] is True
    assert candidate["phase9_adapter_go"] is True
    assert candidate["dominant_evidence_gap"] in candidate[
        "evidence_components"
    ]
    assert "MULTIPLE_TESTING_PASS" in candidate["blockers"]
    assert "HISTORICAL_SHARIAH_PIT_PASS" in candidate["blockers"]
    assert "INDEPENDENT_FORWARD_SAMPLE_PASS" in candidate["blockers"]
    assert finalist["automatic_promotion"] is False


def test_forward_independence_clusters_overlap_and_excludes_bad_rows() -> None:
    episodes = [
        {
            "strategy_id": "S1",
            "symbol": "AAA",
            "timeframe": "1h",
            "market_regime": "RISK_ON",
            "outcome_status": "CLOSED_TARGET",
            "signal_timestamp": "2026-01-01T10:00:00Z",
            "entry_timestamp": "2026-01-01T11:00:00Z",
            "exit_timestamp": "2026-01-01T14:00:00Z",
            "net_return": 0.02,
            "episode_id": "E1",
        },
        {
            "strategy_id": "S1",
            "symbol": "AAA",
            "timeframe": "1h",
            "market_regime": "RISK_ON",
            "outcome_status": "CLOSED_STOP",
            "signal_timestamp": "2026-01-01T13:00:00Z",
            "entry_timestamp": "2026-01-01T14:00:00Z",
            "exit_timestamp": "2026-01-01T16:00:00Z",
            "net_return": -0.01,
            "episode_id": "E2",
        },
        {
            "strategy_id": "S1",
            "symbol": "AAA",
            "timeframe": "1h",
            "market_regime": "RISK_ON",
            "outcome_status": "CLOSED_TARGET",
            "signal_timestamp": "2026-01-02T10:00:00Z",
            "entry_timestamp": "2026-01-02T11:00:00Z",
            "exit_timestamp": "2026-01-02T13:00:00Z",
            "net_return": 0.03,
            "episode_id": "E3",
        },
        {
            "strategy_id": "S1",
            "symbol": "BBB",
            "timeframe": "1h",
            "market_regime": "STRESS",
            "outcome_status": "CLOSED_STOP",
            "signal_timestamp": "2026-01-02T10:00:00Z",
            "entry_timestamp": "2026-01-02T11:00:00Z",
            "exit_timestamp": "2026-01-02T12:00:00Z",
            "net_return": -0.02,
            "episode_id": "E4",
        },
        {
            "strategy_id": "S1",
            "symbol": "CCC",
            "timeframe": "1h",
            "outcome_status": "CLOSED_STOP",
            "net_return": -0.02,
            "episode_id": "E5",
        },
    ]

    report = _forward_independence_audit(
        {
            "episodes": episodes,
            "per_strategy": [{"strategy_id": "S1"}],
        },
        config={"forward_independence": {"cluster_gap_bars": 3}},
    )

    assert report["status"] == "GO_WITH_DOCUMENTED_NONCANONICAL_EXCLUSIONS"
    assert report["counts"]["raw_closed_episode_count"] == 5
    assert report["counts"]["canonical_closed_episode_count"] == 4
    assert report["counts"]["independent_closed_episode_count"] == 3
    assert report["counts"]["distinct_asset_count"] == 2
    assert report["exclusion_reason_counts"] == {
        "MISSING_SIGNAL_TIMESTAMP": 1
    }
    assert report["clusters"][0]["episode_count"] == 2
    assert report["execution_authority"] == "NONE"


def test_graduated_authority_separates_strategy_from_execution(
    tmp_path: Path,
) -> None:
    _seed(tmp_path)

    report = publish_evidence_throughput(tmp_path)
    authority = json.loads(
        (
            tmp_path
            / "output/research/evidence_throughput/strategy-authority-recommendations.json"
        ).read_text(encoding="utf-8")
    )
    candidate = authority["candidates"][0]

    assert report["graduated_strategy_authority"]["status"] == "GO"
    assert candidate["evidence_tier"] == "PROVISIONAL_TRADABLE"
    assert candidate["recommended_strategy_authority"] == "PROVISIONAL"
    assert candidate["strategy_canary_eligible"] is False
    assert 0 < candidate["soft_evidence_risk_multiplier"] <= 0.2
    assert candidate["provisional_risk_fraction"] <= 0.0012
    assert authority["strategy_authority_applied"] is False
    assert authority["execution_authority"] == "NONE"
    assert authority["automatic_promotion"] is False
    assert authority["content_hash"] == stable_hash(
        {
            key: value
            for key, value in authority.items()
            if key != "content_hash"
        }
    )


def test_complete_paper_session_only_changes_canary_recommendation(
    tmp_path: Path,
) -> None:
    _seed(tmp_path)
    _write_json(
        tmp_path,
        "output/operations/paper-session-audit.json",
        {
            "status": "GO",
            "marker": "ONE_COMPLETE_PAPER_SESSION_GO",
            "paper_session_substatus": "PAPER_SESSION_COMPLETE",
        },
    )

    publish_evidence_throughput(tmp_path)
    authority = json.loads(
        (
            tmp_path
            / "output/research/evidence_throughput/strategy-authority-recommendations.json"
        ).read_text(encoding="utf-8")
    )
    candidate = authority["candidates"][0]

    assert candidate["recommended_strategy_authority"] == "CANARY"
    assert candidate["strategy_canary_eligible"] is True
    assert candidate["phase9_adapter_go"] is True
    assert candidate["evidence_components"]["paper_session"][
        "natural_strategy_session_gate_pass"
    ] is True
    assert authority["execution_authority"] == "NONE"
    assert authority["strategy_authority_applied"] is False

    readiness = json.loads(
        (
            tmp_path
            / "output/research/evidence_throughput/provisional-canary-readiness.json"
        ).read_text(encoding="utf-8")
    )
    assert readiness["candidate_count"] == 1
    assert readiness["affordable_whole_share_count"] == 1
    assert readiness["execution_ready_count"] == 0
    assert readiness["candidates"][0]["whole_share_affordable"] is True
    assert readiness["candidates"][0]["strategy_canary_eligible"] is True
    assert "REALTIME_TOP_OF_BOOK_REQUIRED_AT_EXECUTION" in readiness["blockers"]
    assert readiness["execution_authority"] == "NONE"


def test_provisional_canary_readiness_blocks_unaffordable_whole_share(
    tmp_path: Path,
) -> None:
    _seed(tmp_path)
    observation_path = (
        tmp_path
        / "output/research/phase11_14/latest-forward-observation.json"
    )
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["observations"][0]["raw_active_signals"][0][
        "entry_reference"
        ] = 600.0
    observation_path.write_text(json.dumps(observation), encoding="utf-8")
    _write_json(
        tmp_path,
        "output/ibkr/capabilities/fractional-shares.json",
        {
            "status": "GO_CONTRACT_METADATA_ONLY",
            "classification": "CONTRACT_FRACTIONAL_INCREMENT_OBSERVED",
            "contract_reference": {"symbol": "AAA"},
            "account_fractional_permission_proven": False,
            "fractional_bracket_support_proven": False,
        },
    )

    publish_evidence_throughput(tmp_path)
    readiness = json.loads(
        (
            tmp_path
            / "output/research/evidence_throughput/provisional-canary-readiness.json"
        ).read_text(encoding="utf-8")
    )

    candidate = readiness["candidates"][0]
    assert candidate["estimated_one_share_notional_eur"] == 300.0
    assert candidate["whole_share_affordable"] is False
    assert "WHOLE_SHARE_NOTIONAL_EXCEEDS_LEVEL_ONE_CAP" in candidate["blockers"]
    assert candidate["fractional_contract_increment_observed"] is True
    assert candidate["fractional_execution_eligible"] is False
    assert (
        "FRACTIONAL_CONTRACT_INCREMENT_OBSERVED_BUT_ACCOUNT_OR_BRACKET_SUPPORT_UNPROVEN"
        in candidate["blockers"]
    )
    assert readiness["fractional_writer_activation_allowed"] is False
    assert readiness["execution_ready_count"] == 0
    assert readiness["automatic_orders"] == 0


def test_trial_accounting_and_pit_boundary_remain_fail_closed(
    tmp_path: Path,
) -> None:
    _seed(tmp_path)
    _write_json(
        tmp_path,
        "output/ibkr/phase11_3/shariah-history.json",
        {
            "status": "SHARIAH_HISTORY_INCOMPLETE",
            "reconstructable_count": 0,
            "partial_screen_count": 10,
            "screen_symbol_count": 3,
            "available_component_counts": {
                "market_cap": 10,
                "debt_ratio": 8,
                "cash_interest_ratio": 7,
                "receivables_ratio": 9,
            },
            "missing_components": [
                "PIT_BUSINESS_ACTIVITY_CLASSIFICATION",
                "NON_PERMISSIBLE_INCOME_CLASSIFICATION",
            ],
        },
    )

    publish_evidence_throughput(tmp_path)
    root = tmp_path / "output/research/evidence_throughput"
    trials = json.loads(
        (root / "multiple-testing-trial-accounting.json").read_text()
    )
    boundary = json.loads((root / "pit-shariah-boundary.json").read_text())
    plan = json.loads((root / "forward-sample-plan.json").read_text())

    assert trials["registered_hypothesis_count"] == 10
    assert trials["pbo_status"] == "UNAVAILABLE_INSUFFICIENT_FOLD_MATRIX"
    assert trials["multiple_testing_corrected_finalist_count"] == 0
    assert boundary["historical_point_in_time_coverage_ratio"] == 0.0
    assert boundary["historical_screen_count"] == 10
    assert boundary["historical_complete_screen_count"] == 0
    assert boundary["historical_partial_screen_count"] == 10
    assert boundary["historical_financial_component_coverage_ratio"] == 0.85
    assert boundary["historical_evidence_scope"] == (
        "CAUSAL_SEC_FINANCIAL_COMPONENTS_PARTIAL"
    )
    assert boundary["financial_finalist_gate_pass"] is False
    assert boundary["current_status_backprojection_allowed"] is False
    assert plan["universe_contract_changed"] is False
    assert plan["forced_signals"] == 0
    candidate = plan["candidates"][0]
    assert candidate["observed_distinct_asset_count"] == 3
    assert candidate["open_episode_count"] == 1
    assert candidate["awaiting_next_bar_count"] == 1
    assert candidate["sector_metadata_status"] == "AVAILABLE_AT_DECISION"
    assert "CLOSE_EXISTING_OPEN_EPISODES_CAUSALLY" in candidate[
        "next_evidence_actions"
    ]
    assert plan["sector_metadata_is_not_inferred"] is True


def test_walk_forward_rank_pbo_uses_prior_folds_only() -> None:
    rows = []
    for fold in range(6):
        for strategy in range(4):
            rows.append(
                {
                    "strategy_id": f"S{strategy}",
                    "fold_id": f"1h_F{fold:03d}",
                    "timeframe": "1h",
                    "cost_bps": 10.0,
                    "Sharpe": float(strategy + fold / 10),
                }
            )

    report = _walk_forward_rank_pbo(pd.DataFrame(rows))["1h"]

    assert report["status"] == "GO_EXPANDING_WALK_FORWARD_RANK"
    assert report["fold_count"] == 5
    assert report["configuration_count"] == 4
    assert report["classical_cscv"] is False


def test_oos_fold_monte_carlo_is_deterministic_and_not_an_authority_gate() -> None:
    rows = []
    for fold in range(6):
        rows.append(
            {
                "strategy_id": "Q1",
                "fold_id": f"F{fold}",
                "cost_bps": 10.0,
                "period_profit_factor": 1.05 + fold * 0.03,
                "CAGR": 0.02 + fold * 0.01,
                "maximum_drawdown": -0.20 + fold * 0.01,
            }
        )
    folds = pd.DataFrame(rows)
    qualification = {"strategies": [_qualification_row()]}
    config = {"oos_fold_bootstrap_runs": 500}

    first = _fold_monte_carlo(
        folds, qualification=qualification, config=config
    )
    second = _fold_monte_carlo(
        folds, qualification=qualification, config=config
    )

    first_without_time = {**first, "generated_at": None, "content_hash": None}
    second_without_time = {**second, "generated_at": None, "content_hash": None}
    assert first_without_time == second_without_time
    assert first["status"] == "GO"
    assert first["evaluable_candidate_count"] == 1
    assert first["candidates"][0]["bootstrap_runs"] == 500
    assert first["candidates"][0]["probability_median_pf_above_one"] == 1.0
    assert first["daily_block_bootstrap_claimed"] is False
    assert first["authority_gate_changed"] is False
    assert first["execution_authority"] == "NONE"


def test_exact_event_driven_uses_release_time_and_preserves_authority_none() -> None:
    dates = pd.to_datetime(
        [
            "2025-01-02 14:00:00",
            "2025-01-02 15:00:00",
            "2025-01-02 16:00:00",
            "2025-01-03 14:00:00",
            "2025-01-03 15:00:00",
            "2025-01-06 14:00:00",
            "2025-01-06 15:00:00",
        ]
    )
    returns = pd.DataFrame(
        {
            "strategy_id": ["Q1"] * len(dates),
            "cost_bps": [10.0] * len(dates),
            "date": dates,
            "daily_return": [-0.01, 0.01, 0.01, 0.01, 0.01, -0.01, -0.01],
        }
    )
    events = {
        "historical_release_instances": [
            {
                "event_id": "US_CPI",
                "observation_date": "2024-12-01",
                "released_at": "2025-01-02T15:00:00+00:00",
                "release_status": "OBSERVED_HISTORICAL_RELEASE",
            }
        ]
    }

    report = _exact_event_driven(
        returns,
        macro_events=events,
        qualification={"strategies": [_qualification_row()]},
        config={
            "minimum_event_window_observations": 4,
            "minimum_non_event_window_observations": 3,
            "minimum_aligned_macro_releases": 1,
        },
    )

    candidate = report["candidates"][0]
    assert report["status"] == "GO"
    assert report["exact_causal_release_join"] is True
    assert candidate["status"] == "EVALUABLE_CAUSAL_MACRO_RELEASE_WINDOWS"
    assert candidate["event_returns"]["observation_count"] == 4
    assert candidate["non_event_returns"]["observation_count"] == 3
    assert candidate["event_returns"]["mean_return"] == 0.01
    assert candidate["non_event_returns"]["mean_return"] == -0.01
    assert report["authority_gate_changed"] is False
    assert report["execution_authority"] == "NONE"
    assert report["automatic_orders"] == 0


def test_exact_event_driven_blocks_conflicting_duplicate_oos_timestamp() -> None:
    returns = pd.DataFrame(
        {
            "strategy_id": ["Q1", "Q1"],
            "cost_bps": [10.0, 10.0],
            "date": pd.to_datetime(["2025-01-02 15:00", "2025-01-02 15:00"]),
            "daily_return": [0.01, -0.01],
        }
    )
    events = {
        "historical_release_instances": [
            {
                "event_id": "US_CPI",
                "observation_date": "2024-12-01",
                "released_at": "2025-01-02T15:00:00+00:00",
                "release_status": "OBSERVED_HISTORICAL_RELEASE",
            }
        ]
    }

    report = _exact_event_driven(
        returns,
        macro_events=events,
        qualification={"strategies": [_qualification_row()]},
        config={},
    )

    candidate = report["candidates"][0]
    assert candidate["status"] == "BLOCKED_CONFLICTING_OOS_TIMESTAMPS"
    assert candidate["conflicting_timestamp_count"] == 1
    assert candidate["evaluable"] is False
    assert report["execution_authority"] == "NONE"
