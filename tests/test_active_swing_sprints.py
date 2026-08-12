from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from stocks.research.active_swing_sprints import (
    publish_active_swing_leaderboards,
    publish_shortlist_coverage,
    run_entry_filter_experiment,
    train_selective_ml,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _episode(
    index: int,
    *,
    tape: bool = False,
    depth: bool = False,
    regime: str = "BULL_TREND_LOW_VOL",
) -> dict[str, object]:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    candidate_identity = f"CANDIDATE-{index:04d}"
    return {
        "schema": "active_swing_forward_episode_v1",
        "episode_id": f"ENTRY-{index:04d}",
        "feature_snapshot_hash": f"HASH-{index:04d}",
        "setup_id": candidate_identity,
        "candidate_identity": candidate_identity,
        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "setup_origin_timestamp": timestamp.isoformat(),
        "strategy_id": "ACTIVE-SWING-TEST",
        "strategy_dna_hash": "DNA-TEST",
        "timeframe_evidence_hash": f"EVIDENCE-{index:04d}",
        "negative_sampling_policy": "CANDIDATE_CONDITIONED_ONLY",
        "strategy_timeframe_contract": {
            "schema": "active_swing_strategy_timeframe_contract_v1",
            "entry_timeframe": "1h",
            "setup_timeframe": "4h",
            "required_timeframes": ["1h", "4h"],
        },
        "decision_timestamp": timestamp.isoformat(),
        "timeframe": "1h",
        "decision_contract": {
            "hard_veto_pass": True,
            "market_regime": regime,
            "setup_score_within_family": 60 + index % 20,
            "gates": {
                "observed_tape_available": tape,
                "observed_tape_confirms": tape,
                "observed_depth_available": depth,
                "observed_depth_confirms": depth,
            },
            "asset_profile": {
                "asset_class": "STOCK",
                "data_coverage_ratio": 0.8,
                "coverage_adjusted_score": 70.0,
            },
        },
        "context_snapshot": {
            "asset_bias_score": 0.2,
            "asset_bias_confidence": 0.7,
        },
        "setup_snapshot": {
            "asset_class": "STOCK",
            "reward_risk_1": 2.0,
        },
    }


def _outcome(index: int, *, winner: bool) -> dict[str, object]:
    return {
        "schema": "active_swing_forward_episode_outcome_v1",
        "episode_id": f"ENTRY-{index:04d}",
        "feature_snapshot_hash": f"HASH-{index:04d}",
        "terminal": True,
        "terminal_status": "TP1_EXIT" if winner else "STOPPED",
        "would_fill": True,
        "net_R": 1.5 if winner else -1.0,
        "maximum_adverse_excursion": -0.2 if winner else -1.0,
        "maximum_favourable_excursion": 1.5 if winner else 0.2,
        "slippage_from_decision_bps": 2.0,
        "spread_at_decision_bps": 4.0,
        "asset_class": "STOCK",
        "label_source": "PHASE9_CANONICAL_BROKER_FILL",
        "canonical_close": True,
        "canonical_fill_evidence": True,
    }


def test_shortlist_funnel_is_bounded_and_bar_proxy_is_not_tape(tmp_path: Path) -> None:
    observations = []
    for index in range(30):
        observations.append(
            {
                "episode_id": f"E-{index}",
                "symbol": f"S{index}",
                "timeframe": "1h",
                "entry_snapshot": {
                    "tape": {
                        "status": "BAR_FLOW_PROXY_ONLY",
                        "data_class": "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW",
                    },
                    "depth": {
                        "status": "OBSERVED_DEPTH_STORE_UNAVAILABLE",
                        "data_class": "OBSERVED_ORDERBOOK_UNAVAILABLE",
                    },
                },
                "decision_contract": {
                    "asset_profile": {
                        "asset_class": "STOCK",
                        "data_coverage_ratio": 0.6,
                        "component_scores": {"relative_strength": 0.5},
                    }
                },
            }
        )
    _write_json(
        tmp_path / "output/market_context/entry-shortlist.json",
        {
            "signal_funnel": {"input_signals": 254},
            "observations": observations,
        },
    )

    report = publish_shortlist_coverage(tmp_path)

    assert report["funnel"]["structural_candidates"] == 20
    assert report["funnel"]["level1_tape_requested"] == 10
    assert report["funnel"]["depth_requested"] == 5
    assert report["observed_tape_count"] == 0
    assert report["bar_proxy_can_claim_tape_or_depth"] is False
    assert all(row["tape_status"] == "UNAVAILABLE" for row in report["rows"][:10])


def test_entry_filter_experiment_uses_same_outcomes_and_sample_gate(tmp_path: Path) -> None:
    episodes = [
        _episode(index, tape=index % 2 == 0, depth=index % 4 == 0)
        for index in range(60)
    ]
    outcomes = [
        _outcome(index, winner=(index % 2 == 0)) for index in range(60)
    ]
    private = tmp_path / "data/market_context/private"
    _write_jsonl(private / "entry-episodes.jsonl", episodes)
    _write_jsonl(private / "entry-episode-outcomes.jsonl", outcomes)

    report = run_entry_filter_experiment(tmp_path)
    treatments = {row["treatment"]: row for row in report["treatments"]}

    assert report["status"] == "GO"
    assert treatments["BASE"]["closed_trade_count"] == 60
    assert treatments["LEVEL1_TAPE"]["closed_trade_count"] == 30
    assert treatments["LEVEL1_TAPE"]["net_expectancy_R"] > treatments["BASE"]["net_expectancy_R"]
    assert report["same_frozen_setups_stops_targets"] is True
    assert report["execution_authority"] == "NONE"


def test_role_leaderboards_cap_champion_and_challengers(tmp_path: Path) -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "strategy_id": f"S-{index}",
                "strategy_hash": f"H-{index}",
                "formula": f"formula_{index}",
                "timeframe": "1h" if index < 5 else "1w",
                "profile": "balanced",
                "asset_class": "COMMODITY_PROXY" if index == 0 else "STOCK",
                "status": "COMPLETE",
                "CAGR": 0.1 + index / 100,
                "Sharpe": 0.8 + index / 100,
                "period_profit_factor": 1.2,
                "maximum_drawdown": -0.2,
                "fill_count": 100 + index,
                "stress_50bps_profit_factor": 1.1,
                "economic_outcome_fingerprint": f"F-{index}",
            }
        )
    source = tmp_path / "output/research/phase11_12/strategy-summary.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(source, index=False)
    _write_json(
        tmp_path / "output/research/active_swing/entry_filter_experiment/results.json",
        {"treatments": []},
    )

    report = publish_active_swing_leaderboards(tmp_path)

    active = report["roles"]["ACTIVE_SWING"]
    assert active["champion_count"] == 1
    assert active["challenger_count"] == 2
    assert active["published_count"] == 3
    assert report["roles"]["TACTICAL_ENTRY"]["status"] == "DATA_UNAVAILABLE"
    assert report["roles"]["TACTICAL_ENTRY"]["published_count"] == 0
    assert report["cross_role_ranking_allowed"] is False
    assert report["automatic_promotion"] is False


def test_selective_ml_stays_blocked_below_100_closed_labels(tmp_path: Path) -> None:
    private = tmp_path / "data/market_context/private"
    _write_jsonl(private / "entry-episodes.jsonl", [_episode(0)])
    _write_jsonl(private / "entry-episode-outcomes.jsonl", [_outcome(0, winner=True)])

    report = train_selective_ml(tmp_path)

    assert report["trained"] is False
    assert report["status"] == "NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS"
    assert report["model_authority"] == "NONE"
    assert report["automatic_retraining"] is False


def test_selective_ml_trains_only_after_threshold_and_remains_observational(tmp_path: Path) -> None:
    episodes = [_episode(index) for index in range(160)]
    outcomes = [_outcome(index, winner=index % 2 == 0) for index in range(160)]
    private = tmp_path / "data/market_context/private"
    _write_jsonl(private / "entry-episodes.jsonl", episodes)
    _write_jsonl(private / "entry-episode-outcomes.jsonl", outcomes)

    report = train_selective_ml(tmp_path)

    assert report["trained"] is True
    assert report["status"] == "EXPERIMENTAL_SMOKE_ELIGIBLE"
    assert report["chronological_split"] is True
    assert report["purged_time_series_split"] is True
    assert report["model_authority"] == "NONE"
    assert report["predictions_are_observation_only"] is True
    assert report["regime_conditioning"] == (
        "SINGLE_MODEL_EXPLICIT_REGIME_FEATURE"
    )
    assert report["feature_availability_indicators"] is True
    assert report["reinforcement_learning_authority"] == "NONE"


def test_selective_ml_leave_one_regime_out_is_causal_and_shadow_only(
    tmp_path: Path,
) -> None:
    regimes = (
        "BULL_TREND_LOW_VOL",
        "BEAR_TREND",
        "VOLATILITY_SHOCK",
    )
    episodes = [
        _episode(index, regime=regimes[index % len(regimes)])
        for index in range(360)
    ]
    outcomes = [
        _outcome(index, winner=index % 2 == 0) for index in range(360)
    ]
    private = tmp_path / "data/market_context/private"
    _write_jsonl(private / "entry-episodes.jsonl", episodes)
    _write_jsonl(private / "entry-episode-outcomes.jsonl", outcomes)

    report = train_selective_ml(tmp_path)
    generalization = report["regime_generalization"]

    assert report["trained"] is True
    assert generalization["status"] == "GO"
    assert generalization["evaluable_regime_count"] == 3
    assert generalization["all_folds_causally_ordered"] is True
    assert generalization["worst_regime"] in regimes
    assert 0.0 <= generalization["worst_regime_auc"] <= 1.0
    assert generalization["selection_authority"] == "RESEARCH_SHADOW_ONLY"
    assert generalization["model_authority"] == "NONE"
    assert generalization["mixture_of_experts_trained"] is False
    assert generalization["reinforcement_learning_trained"] is False
    assert (
        tmp_path
        / "output/research/active_swing/selective_ml/regime-generalization.json"
    ).is_file()


def test_selective_ml_reports_unavailable_regime_without_zero_filling(
    tmp_path: Path,
) -> None:
    episodes = [_episode(index, regime="") for index in range(120)]
    outcomes = [
        _outcome(index, winner=index % 2 == 0) for index in range(120)
    ]
    private = tmp_path / "data/market_context/private"
    _write_jsonl(private / "entry-episodes.jsonl", episodes)
    _write_jsonl(private / "entry-episode-outcomes.jsonl", outcomes)

    report = train_selective_ml(tmp_path)

    assert report["regime_dataset"]["unavailable_regime_row_count"] == 120
    assert report["regime_dataset"]["missing_values_are_not_zero_filled"] is True
    assert report["regime_generalization"]["status"] == "NOT_EVALUABLE"
    assert report["model_authority"] == "NONE"


def test_noncanonical_bar_outcomes_are_never_ml_training_rows(
    tmp_path: Path,
) -> None:
    episodes = [_episode(index) for index in range(120)]
    outcomes = [_outcome(index, winner=index % 2 == 0) for index in range(120)]
    for outcome in outcomes:
        outcome["label_source"] = "BAR_SIMULATION"
        outcome["canonical_fill_evidence"] = False
    private = tmp_path / "data/market_context/private"
    _write_jsonl(private / "entry-episodes.jsonl", episodes)
    _write_jsonl(private / "entry-episode-outcomes.jsonl", outcomes)

    report = train_selective_ml(tmp_path)

    assert report["closed_trainable_episode_count"] == 0
    assert report["trained"] is False
    assert report["bar_simulation_labels_trainable"] is False
    assert report["model_can_raise_risk_caps"] is False


def test_selective_ml_requires_matching_immutable_feature_snapshot(
    tmp_path: Path,
) -> None:
    episodes = [_episode(index) for index in range(120)]
    outcomes = [_outcome(index, winner=index % 2 == 0) for index in range(120)]
    outcomes[0]["feature_snapshot_hash"] = "MUTATED-SNAPSHOT"
    private = tmp_path / "data/market_context/private"
    _write_jsonl(private / "entry-episodes.jsonl", episodes)
    _write_jsonl(private / "entry-episode-outcomes.jsonl", outcomes)

    report = train_selective_ml(tmp_path)

    assert report["closed_trainable_episode_count"] == 119
    provenance = report["label_provenance"]
    assert provenance["canonical_exclusion_reasons"] == {
        "FEATURE_SNAPSHOT_HASH_MISMATCH": 1
    }
    assert report["feature_snapshot_hash_match_required"] is True


def test_label_provenance_keeps_research_paper_and_live_cohorts_separate(
    tmp_path: Path,
) -> None:
    private = tmp_path / "data/market_context/private"
    episodes = [_episode(0)]
    outcome = _outcome(0, winner=True)
    outcome["label_source"] = "COUNTERFACTUAL_BAR_PATH_OBSERVATION"
    outcome["canonical_fill_evidence"] = False
    _write_jsonl(private / "entry-episodes.jsonl", episodes)
    _write_jsonl(private / "entry-episode-outcomes.jsonl", [outcome])
    historical = tmp_path / "output/research/phase11_7/selected-closed-episodes.csv"
    historical.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"security_id": "A", "entry_date": "2025-01-01", "net_pnl": 1.0}]
    ).to_csv(historical, index=False)

    report = train_selective_ml(tmp_path)
    provenance = report["label_provenance"]

    assert provenance["canonical_trainable_label_count"] == 0
    assert provenance["cohort_isolation"] is True
    assert provenance["cohorts"]["HISTORICAL_BACKTEST_RESEARCH"][
        "observed_label_count"
    ] == 1
    assert provenance["cohorts"]["FORWARD_COUNTERFACTUAL_OBSERVATION"][
        "terminal_outcome_count"
    ] == 1
    assert provenance["cohorts"]["PAPER_BROKER_EXECUTION"]["execution_count"] == 0
    assert provenance["cohorts"]["LIVE_BROKER_EXECUTION"]["execution_count"] == 0
    assert provenance["historical_backtest_labels_are_execution_evidence"] is False
    assert provenance["counterfactual_forward_labels_are_broker_fill_evidence"] is False
    assert (
        tmp_path
        / "output/research/active_swing/selective_ml/label-provenance.json"
    ).is_file()
