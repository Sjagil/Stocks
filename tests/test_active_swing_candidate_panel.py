from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from stocks.ai.active_swing_panel import (
    INTERACTION_FEATURE_COLUMNS,
    build_active_swing_candidate_panel,
    infer_current_active_swing_candidates,
)
from stocks.ai.active_swing_modeling import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    predict_active_swing_bundle,
    run_active_swing_model_tournament,
)
from stocks.ai.active_swing_runtime import infer_active_swing_fast_path
from stocks.execution.idempotency import stable_hash


def _episode(*, identity: str = "SETUP-1") -> dict:
    decision = "2026-08-10T15:00:00+00:00"
    contract = {
        "schema": "active_swing_strategy_timeframe_contract_v1",
        "entry_timeframe": "15m",
        "setup_timeframe": "1h",
        "context_timeframes": ["1h", "4h", "1d"],
        "structural_timeframe": "4h",
        "management_timeframe": "1h",
        "exit_timeframe": "15m",
        "required_timeframes": ["15m", "1h", "4h", "1d"],
    }
    evidence = {
        timeframe: {
            "timeframe": timeframe,
            "available": True,
            "bar_closed": True,
            "available_at": available_at,
            "knowledge_available_at": available_at,
            "close": close,
            "ema20": close - 1.0,
            "ema50": close - 2.0,
            "return_1_bar": 0.002,
            "return_4_bars": 0.01,
            "trend_state": "SUPPORTIVE",
        }
        for timeframe, available_at, close in (
            ("15m", "2026-08-10T14:45:00+00:00", 101.0),
            ("1h", "2026-08-10T14:00:00+00:00", 102.0),
            ("4h", "2026-08-10T12:00:00+00:00", 103.0),
            ("1d", "2026-08-10T00:00:00+00:00", 104.0),
        )
    }
    setup = {
        "asset_class": "STOCK",
        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "strategy_dna_hash": "DNA-1",
        "negative_sampling_policy": "CANDIDATE_CONDITIONED_ONLY",
        "strategy_timeframe_contract": contract,
        "timeframe_evidence": evidence,
        "timeframe_evidence_hash": stable_hash(evidence),
        "reward_risk_1": 1.5,
        "estimated_transaction_costs_eur": 0.5,
    }
    episode = {
        "schema": "active_swing_forward_episode_v1",
        "episode_id": f"ENTRY-{identity}",
        "candidate_identity": identity,
        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "setup_id": identity,
        "natural_strategy_candidate": True,
        "candidate_conditioned_evidence_eligible": True,
        "symbol": "TEST",
        "strategy_id": "AS15-1",
        "strategy_family": "PULLBACK_RESUMPTION",
        "strategy_dna_hash": "DNA-1",
        "timeframe": "15m",
        "setup_origin_timestamp": "2026-08-10T14:45:00+00:00",
        "decision_timestamp": decision,
        "decision_contract": {
            "market_regime": "TREND",
            "setup_score_within_family": 0.7,
            "entry_confirmation_score": 0.6,
            "asset_profile": {
                "coverage_adjusted_score": 61.0,
                "data_coverage_ratio": 0.8,
            },
        },
        "context_snapshot": {
            "asset_bias_score": 0.2,
            "asset_bias_confidence": 0.7,
            "event_risk": {"risk_score": 0.1},
        },
        "setup_snapshot": setup,
        "entry_snapshot": {"tape": {}, "depth": {}},
    }
    episode["feature_snapshot_hash"] = stable_hash(
        {
            "decision_contract": episode["decision_contract"],
            "context_snapshot": episode["context_snapshot"],
            "setup_snapshot": episode["setup_snapshot"],
            "entry_snapshot": episode["entry_snapshot"],
        }
    )
    return episode


def _outcome(episode: dict) -> dict:
    return {
        "schema": "active_swing_forward_episode_outcome_v1",
        "episode_id": episode["episode_id"],
        "candidate_identity": episode["candidate_identity"],
        "natural_strategy_candidate": True,
        "candidate_conditioned_evidence_eligible": True,
        "feature_snapshot_hash": episode["feature_snapshot_hash"],
        "terminal": True,
        "research_observation_eligible": True,
        "excluded_from_performance_metrics": False,
        "symbol_fingerprint": "SECURITY-TEST",
        "fill_timestamp": "2026-08-10T15:15:00+00:00",
        "exit_timestamp": "2026-08-11T15:00:00+00:00",
        "net_R": 0.75,
        "gross_R": 0.8,
        "maximum_favourable_excursion": 1.1,
        "maximum_adverse_excursion": -0.2,
        "holding_duration_seconds": 85_500,
        "first_barrier_hit": "TIME_EXIT",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_candidate_panel_emits_exactly_one_row_per_natural_setup(tmp_path: Path) -> None:
    episode = _episode()
    _write_jsonl(
        tmp_path / "data/market_context/private/entry-episodes.jsonl",
        [episode, copy.deepcopy(episode)],
    )
    _write_jsonl(
        tmp_path / "data/market_context/private/entry-episode-outcomes.jsonl",
        [_outcome(episode)],
    )

    panel, status = build_active_swing_candidate_panel(tmp_path)

    assert len(panel) == 1
    assert panel.loc[0, "candidate_identity"] == "SETUP-1"
    assert panel.loc[0, "candidate_unit"] == "ONE_NATURAL_STRATEGY_SETUP"
    assert panel.loc[0, "native_exit_net_return"] == 0.75
    assert panel.loc[0, "positive_net_trade"] == 1
    assert panel.loc[0, "momentum_4bars_15m"] == 0.01
    assert panel.loc[0, "return_1bar_1h"] == 0.002
    assert status["rejection_counts"]["DUPLICATE_CANDIDATE_OBSERVATION"] == 1
    assert status["training_ready"] is False
    assert status["sample_dependence"]["raw_candidate_count"] == 1
    assert status["sample_dependence"]["effective_sample_size"] == 1.0
    assert status["strategy_clustering"]["frozen_hypothesis_cluster_count"] == 1
    persisted = pd.read_parquet(
        tmp_path / "data/ai/private/active-swing-candidate-panel.parquet"
    )
    assert len(persisted) == 1


def test_candidate_panel_rejects_context_future_and_mismatched_outcomes(
    tmp_path: Path,
) -> None:
    context_only = _episode(identity="CONTEXT")
    context_only["candidate_unit"] = "CONTEXT_WATCHLIST_OBSERVATION"
    future = _episode(identity="FUTURE")
    future["setup_snapshot"]["timeframe_evidence"]["15m"]["available_at"] = (
        "2026-08-10T15:15:00+00:00"
    )
    future["setup_snapshot"]["timeframe_evidence_hash"] = stable_hash(
        future["setup_snapshot"]["timeframe_evidence"]
    )
    future["feature_snapshot_hash"] = stable_hash(
        {
            "decision_contract": future["decision_contract"],
            "context_snapshot": future["context_snapshot"],
            "setup_snapshot": future["setup_snapshot"],
            "entry_snapshot": future["entry_snapshot"],
        }
    )
    mismatch = _episode(identity="MISMATCH")
    mismatched_outcome = _outcome(mismatch)
    mismatched_outcome["feature_snapshot_hash"] = "WRONG"
    _write_jsonl(
        tmp_path / "data/market_context/private/entry-episodes.jsonl",
        [context_only, future, mismatch],
    )
    _write_jsonl(
        tmp_path / "data/market_context/private/entry-episode-outcomes.jsonl",
        [mismatched_outcome],
    )

    panel, status = build_active_swing_candidate_panel(tmp_path)

    assert panel.empty
    assert status["rejection_counts"]["CONTEXT_OR_INVALID_CANDIDATE"] == 1
    assert status["rejection_counts"]["FEATURE_AVAILABLE_AFTER_DECISION"] == 1
    assert status["rejection_counts"]["OUTCOME_FEATURE_HASH_MISMATCH"] == 1
    assert status["active_swing_candidate_unit_go"] is False


def test_current_candidate_inference_rejects_legacy_model_bundle(tmp_path: Path) -> None:
    episode = _episode()
    signal = {
        "candidate_identity": episode["candidate_identity"],
        "candidate_unit": episode["candidate_unit"],
        "setup_id": episode["setup_id"],
        "setup_origin_timestamp": episode["setup_origin_timestamp"],
        "strategy_id": episode["strategy_id"],
        "strategy_dna_hash": episode["strategy_dna_hash"],
        "timeframe": episode["timeframe"],
        "strategy_timeframe_contract": episode["setup_snapshot"][
            "strategy_timeframe_contract"
        ],
        "timeframe_evidence_hash": episode["setup_snapshot"][
            "timeframe_evidence_hash"
        ],
        "negative_sampling_policy": "CANDIDATE_CONDITIONED_ONLY",
    }
    path = tmp_path / "output/signals/active_swing_15m_signals.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidates": [signal]}), encoding="utf-8")

    result = infer_current_active_swing_candidates(
        tmp_path,
        {"model_version": "LEGACY-1"},
        {"promotion_status": "REJECTED"},
    )

    assert result["status"] == "NO_COMPATIBLE_ACTIVE_SWING_MODEL"
    assert result["current_candidate_count"] == 1
    assert result["evidence_count"] == 0
    assert result["execution_authority"] == "NONE"


def test_fast_path_inference_is_no_retrain_and_fail_closed_without_model(
    tmp_path: Path,
) -> None:
    result = infer_active_swing_fast_path(tmp_path)

    assert result["status"] == "NO_CURRENT_NATURAL_CANDIDATES"
    assert result["model_load_status"] == "NO_CURRENT_MODEL_ARTIFACT"
    assert result["model_artifact_hash_verified"] is False
    assert result["fast_path_retraining_performed"] is False
    assert result["execution_authority"] == "NONE"
    assert result["broker_writes"] == 0


def test_fast_path_inference_skips_when_machine_is_disabled(tmp_path: Path) -> None:
    machine = tmp_path / "output/operations/machine-status.json"
    machine.parent.mkdir(parents=True, exist_ok=True)
    machine.write_text(json.dumps({"enabled": False}), encoding="utf-8")

    result = infer_active_swing_fast_path(tmp_path)

    assert result["status"] == "SKIPPED_MACHINE_DISABLED"
    assert result["model_load_status"] == "NOT_ATTEMPTED_MACHINE_DISABLED"
    assert result["fast_path_retraining_performed"] is False
    assert result["broker_writes"] == 0


def test_candidate_model_trains_only_after_purged_fixed_evidence_minima(
    tmp_path: Path,
) -> None:
    timestamps = pd.date_range("2024-01-01", periods=520, freq="D", tz="UTC")
    rows = []
    for index, timestamp in enumerate(timestamps):
        positive = index % 2
        row = {
            "candidate_identity": f"SETUP-{index:04d}",
            "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
            "decision_timestamp": timestamp,
            "label_available_at": timestamp + pd.Timedelta(hours=12),
            "native_exit_net_return": 0.5 if positive else -0.4,
            "positive_net_trade": positive,
        }
        row.update({name: float(positive) for name in NUMERIC_FEATURES})
        row.update(
            {
                name: 1.0
                for name in NUMERIC_FEATURES
                if name.startswith("has_")
            }
        )
        row.update({name: float(positive) for name in INTERACTION_FEATURE_COLUMNS})
        row.update({name: "CATEGORY" for name in CATEGORICAL_FEATURES})
        rows.append(row)
    panel = pd.DataFrame(rows)
    report, oos, bundle = run_active_swing_model_tournament(
        panel,
        {
            "training_ready": True,
            "training_blockers": [],
            "panel_sha256": "PANEL-HASH",
        },
    )

    assert report["status"] == "SHADOW_MODEL_TRAINED_NOT_PROMOTED"
    assert report["promotion_status"] == (
        "NOT_ELIGIBLE_FORWARD_AND_EXTERNAL_GATES_REQUIRED"
    )
    assert report["label_availability_purged"] is True
    assert len(report["challengers"]) == 2
    assert len(report["timeframe_ablations"]) == 5
    assert all(
        row["trial_counted"] is True for row in report["timeframe_ablations"]
    )
    assert all(
        row["model_fitted"] is True for row in report["timeframe_ablations"]
    )
    assert len(report["interaction_trials"]) == 3
    assert all(row["trial_counted"] is True for row in report["interaction_trials"])
    assert all(row["model_fitted"] is True for row in report["interaction_trials"])
    assert not oos.empty
    assert bundle is not None
    assert bundle["candidate_unit"] == "ONE_NATURAL_STRATEGY_SETUP"
    prediction = predict_active_swing_bundle(bundle, panel.iloc[[0]])
    assert 0.0 <= prediction.iloc[0]["probability_positive_net"] <= 1.0
    assert bundle["execution_authority"] == "NONE"

    episode = _episode()
    _write_jsonl(
        tmp_path / "data/market_context/private/entry-episodes.jsonl",
        [episode],
    )
    signal = {
        "candidate_identity": episode["candidate_identity"],
        "candidate_unit": episode["candidate_unit"],
        "setup_id": episode["setup_id"],
        "setup_origin_timestamp": episode["setup_origin_timestamp"],
        "strategy_id": episode["strategy_id"],
        "strategy_dna_hash": episode["strategy_dna_hash"],
        "timeframe": episode["timeframe"],
        "strategy_timeframe_contract": episode["setup_snapshot"][
            "strategy_timeframe_contract"
        ],
        "timeframe_evidence_hash": episode["setup_snapshot"][
            "timeframe_evidence_hash"
        ],
        "negative_sampling_policy": "CANDIDATE_CONDITIONED_ONLY",
    }
    candidate_path = tmp_path / "output/signals/active_swing_15m_signals.json"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_text(json.dumps({"candidates": [signal]}), encoding="utf-8")
    current = infer_current_active_swing_candidates(
        tmp_path,
        bundle,
        report,
    )
    assert current["status"] == "SHADOW_EVIDENCE_AVAILABLE_NOT_PROMOTED"
    assert current["evidence_count"] == 1
    assert current["model_evidence"][0]["candidate_identity"] == "SETUP-1"
    assert current["model_evidence"][0]["financial_fields_mutated"] is False
