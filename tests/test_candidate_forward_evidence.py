from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from stocks.context.entry_observer import (
    _candidate_signals,
    _select_signals,
    _signal_candidate_identity,
)
from stocks.context.candidate_evidence import candidate_evidence_classification
from stocks.rl.portfolio import candidate_history_readiness


def test_entry_observer_keeps_distinct_natural_candidates_and_accepts_15m() -> None:
    rows = [
        {
            "ticker": "SPUS",
            "strategy_id": "BREAKOUT",
            "setup_id": "SETUP-A",
            "timeframe": "15m",
            "action": "WATCHLIST",
            "confidence_score": 0.8,
        },
        {
            "ticker": "SPUS",
            "strategy_id": "PULLBACK",
            "setup_id": "SETUP-B",
            "timeframe": "1h",
            "action": "WATCHLIST",
            "confidence_score": 0.7,
        },
        {
            "ticker": "SPUS",
            "strategy_id": "BREAKOUT",
            "setup_id": "SETUP-A",
            "timeframe": "15m",
            "action": "WATCHLIST",
            "confidence_score": 0.6,
        },
    ]

    candidates = _candidate_signals(rows)
    selected = _select_signals(
        candidates,
        now=datetime.now(UTC),
        limit=20,
    )

    assert len(candidates) == 3
    assert [row["setup_id"] for row in selected] == ["SETUP-A", "SETUP-B"]
    assert selected[0]["timeframe"] == "15m"


def test_publication_timestamp_cannot_inflate_generic_observation_identity() -> None:
    first = {
        "ticker": "SPUS",
        "strategy_id": "GENERIC-WATCHLIST",
        "timeframe": "1d",
        "data_timestamp": "2026-08-11T20:00:00Z",
        "signal_timestamp": "2026-08-12T02:00:00Z",
    }
    republished = {
        **first,
        "signal_timestamp": "2026-08-12T03:00:00Z",
    }

    assert _signal_candidate_identity(first) == _signal_candidate_identity(
        republished
    )
    classification = candidate_evidence_classification(first)
    assert classification["natural_strategy_candidate"] is False
    assert classification["evidence_scope"] == "CONTEXT_WATCHLIST_OBSERVATION"


def test_rl_history_requires_canonical_closed_independent_candidate_evidence(
    tmp_path: Path,
) -> None:
    history = tmp_path / "episodes.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    episode_rows = [
        {
            "schema": "active_swing_forward_episode_v1",
            "episode_id": "E1",
            "feature_snapshot_hash": "H1",
            "candidate_identity": "C1",
            **_natural_candidate_fields(
                candidate_identity="C1",
                timeframe="15m",
                origin="2026-01-01T10:00:00Z",
            ),
            "decision_timestamp": "2026-01-01T10:00:00Z",
            "timeframe": "15m",
        },
        {
            "schema": "active_swing_forward_episode_v1",
            "episode_id": "E2",
            "feature_snapshot_hash": "H2",
            "candidate_identity": "C2",
            **_natural_candidate_fields(
                candidate_identity="C2",
                timeframe="1h",
                origin="2026-01-02T10:00:00Z",
            ),
            "decision_timestamp": "2026-01-02T10:00:00Z",
            "timeframe": "1h",
        },
    ]
    outcome_rows = [
        {
            "schema": "active_swing_forward_episode_outcome_v1",
            "episode_id": "E1",
            "feature_snapshot_hash": "H1",
            "terminal": True,
            "research_observation_eligible": True,
            "excluded_from_performance_metrics": False,
            "net_R": 1.0,
            "fill_timestamp": "2026-01-01T10:15:00Z",
            "exit_timestamp": "2026-01-01T11:00:00Z",
            "symbol_fingerprint": "S1",
            "strategy_id": "A",
            "timeframe": "15m",
            "market_regime": "RISK_ON",
        },
        {
            "schema": "active_swing_forward_episode_outcome_v1",
            "episode_id": "E2",
            "feature_snapshot_hash": "H2",
            "terminal": True,
            "research_observation_eligible": True,
            "excluded_from_performance_metrics": False,
            "net_R": -1.0,
            "fill_timestamp": "2026-01-02T10:00:00Z",
            "exit_timestamp": "2026-01-02T14:00:00Z",
            "symbol_fingerprint": "S2",
            "strategy_id": "B",
            "timeframe": "1h",
            "market_regime": "RISK_OFF",
        },
    ]
    history.write_text(
        "".join(json.dumps(row) + "\n" for row in episode_rows),
        encoding="utf-8",
    )
    outcomes.write_text(
        "".join(json.dumps(row) + "\n" for row in outcome_rows),
        encoding="utf-8",
    )
    config = {
        "minimum_natural_candidate_episodes": 2,
        "minimum_closed_candidate_outcomes": 2,
        "minimum_independent_candidate_clusters": 2,
        "minimum_decision_periods": 2,
        "minimum_supported_timeframes": 2,
        "minimum_market_regimes": 2,
    }

    report = candidate_history_readiness(
        history,
        outcomes,
        config=config,
    )

    assert report["causal_candidate_history"] is True
    assert report["natural_candidate_episode_count"] == 2
    assert report["closed_candidate_outcome_count"] == 2
    assert report["independent_candidate_cluster_count"] == 2
    assert report["raw_episode_count_is_not_promotion_sample"] is True
    assert report["canonical_observation_episode_count"] == 2
    assert report["context_observation_episode_count"] == 0
    assert report["candidate_identity_deduplicated"] is True


def test_mutated_or_noncanonical_outcome_never_counts(tmp_path: Path) -> None:
    history = tmp_path / "episodes.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    history.write_text(
        json.dumps(
            {
                "schema": "active_swing_forward_episode_v1",
                "episode_id": "E1",
                "feature_snapshot_hash": "ORIGINAL",
                "candidate_identity": "C1",
                **_natural_candidate_fields(
                    candidate_identity="C1",
                    timeframe="15m",
                    origin="2026-01-01T10:00:00Z",
                ),
                "decision_timestamp": "2026-01-01T10:00:00Z",
                "timeframe": "15m",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outcomes.write_text(
        json.dumps(
            {
                "schema": "active_swing_forward_episode_outcome_v1",
                "episode_id": "E1",
                "feature_snapshot_hash": "MUTATED",
                "terminal": True,
                "research_observation_eligible": True,
                "excluded_from_performance_metrics": False,
                "net_R": 99.0,
                "fill_timestamp": "2026-01-01T10:15:00Z",
                "exit_timestamp": "2026-01-01T10:30:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = candidate_history_readiness(
        history,
        outcomes,
        config={
            "minimum_natural_candidate_episodes": 1,
            "minimum_closed_candidate_outcomes": 1,
            "minimum_independent_candidate_clusters": 1,
            "minimum_decision_periods": 1,
            "minimum_supported_timeframes": 1,
            "minimum_market_regimes": 1,
        },
    )

    assert report["natural_candidate_episode_count"] == 1
    assert report["closed_candidate_outcome_count"] == 0
    assert report["causal_candidate_history"] is False


def test_generic_context_observations_never_count_as_rl_candidate_evidence(
    tmp_path: Path,
) -> None:
    history = tmp_path / "episodes.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    history.write_text(
        json.dumps(
            {
                "schema": "active_swing_forward_episode_v1",
                "episode_id": "GENERIC",
                "feature_snapshot_hash": "H1",
                "candidate_identity": "PUBLICATION-HASH",
                "decision_timestamp": "2026-01-01T10:00:00Z",
                "timeframe": "1d",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outcomes.write_text("", encoding="utf-8")

    report = candidate_history_readiness(history, outcomes, config={})

    assert report["canonical_observation_episode_count"] == 1
    assert report["context_observation_episode_count"] == 1
    assert report["natural_candidate_episode_count"] == 0
    assert report["causal_candidate_history"] is False


def test_republished_natural_candidate_is_deduplicated_by_setup_identity(
    tmp_path: Path,
) -> None:
    history = tmp_path / "episodes.jsonl"
    outcomes = tmp_path / "outcomes.jsonl"
    common = {
        "schema": "active_swing_forward_episode_v1",
        "feature_snapshot_hash": "H1",
        "candidate_identity": "C1",
        **_natural_candidate_fields(
            candidate_identity="C1",
            timeframe="15m",
            origin="2026-01-01T10:00:00Z",
        ),
        "timeframe": "15m",
    }
    history.write_text(
        json.dumps(
            {
                **common,
                "episode_id": "E1",
                "decision_timestamp": "2026-01-01T10:00:00Z",
            }
        )
        + "\n"
        + json.dumps(
            {
                **common,
                "episode_id": "E2",
                "decision_timestamp": "2026-01-01T10:05:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    outcomes.write_text("", encoding="utf-8")

    report = candidate_history_readiness(history, outcomes, config={})

    assert report["canonical_observation_episode_count"] == 2
    assert report["natural_candidate_episode_count"] == 1
    assert report["duplicate_candidate_episode_count"] == 1
    assert report["context_observation_episode_count"] == 0


def _natural_candidate_fields(
    *, candidate_identity: str, timeframe: str, origin: str
) -> dict[str, object]:
    setup_timeframe = "1h" if timeframe == "15m" else "4h"
    return {
        "setup_id": candidate_identity,
        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
        "setup_origin_timestamp": origin,
        "strategy_id": "ACTIVE-SWING-TEST",
        "strategy_dna_hash": "DNA-TEST",
        "strategy_family": "TEST_FAMILY",
        "model_version": "TEST-MODEL-V1",
        "timeframe_evidence_hash": "EVIDENCE-TEST",
        "negative_sampling_policy": "CANDIDATE_CONDITIONED_ONLY",
        "strategy_timeframe_contract": {
            "schema": "active_swing_strategy_timeframe_contract_v1",
            "entry_timeframe": timeframe,
            "setup_timeframe": setup_timeframe,
            "required_timeframes": [timeframe, setup_timeframe],
        },
    }
