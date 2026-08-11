from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from stocks.ai import (
    AIAuthority,
    AIPortfolioProposal,
    ExperimentRecord,
    ExperimentStatus,
    ModelLifecycle,
    ModelPrediction,
    ModelRecord,
    NLPEvent,
    ResearchHypothesis,
    assess_model_health,
    audit_ai_import_boundary,
    causal_time_splits,
    false_discovery_control,
    normalize_nlp_events,
    publish_ai_research_plane,
    transition_hypothesis,
    validate_ai_authority,
    validate_point_in_time_rows,
)
from stocks.ai.governance import write_immutable_experiment


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 11, 12, tzinfo=UTC)


def _model(**changes: object) -> ModelRecord:
    values = {
        "model_id": "MODEL-1",
        "family": "LOGISTIC",
        "version": "1",
        "feature_set": ("momentum",),
        "target": "NET_RETURN_POSITIVE",
        "training_interval": "2024..2025",
        "validation_interval": "2025H1",
        "test_interval": "2025H2",
        "forward_interval": "NOT_AVAILABLE",
        "universe": ("AAPL",),
        "horizon": "5D",
        "regime_scope": "ALL_OBSERVED",
        "data_hash": "DATA",
        "code_hash": "CODE",
        "hyperparameters": {},
        "metrics": {},
        "calibration": {},
        "drift_limits": {
            "feature": 0.2,
            "prediction": 0.2,
            "calibration": 0.05,
            "performance": 0.2,
            "regime": 0.25,
        },
        "authority": AIAuthority.SHADOW_ONLY,
        "lifecycle": ModelLifecycle.SHADOW,
        "created_at": NOW,
        "expires_at": NOW + timedelta(days=30),
        "incremental_evidence": "NO_INCREMENTAL_EVIDENCE",
    }
    values.update(changes)
    return ModelRecord(**values)


def _event(event_id: str, raw_hash: str, available_at: datetime) -> NLPEvent:
    return NLPEvent(
        event_id=event_id,
        source="TEST",
        published_at=NOW,
        available_at=available_at,
        entities=("Apple",),
        tickers=("AAPL",),
        event_type="EARNINGS",
        sentiment=0.2,
        uncertainty=0.3,
        novelty=0.8,
        relevance=0.9,
        expected_horizon="5D",
        source_quality=0.8,
        raw_hash=raw_hash,
    )


def test_model_prediction_enforces_point_in_time_availability() -> None:
    with pytest.raises(ValidationError, match="cannot precede"):
        ModelPrediction(
            prediction_id="P1",
            model_id="M1",
            model_version="1",
            feature_set_id="F1",
            symbol="AAPL",
            timestamp=NOW,
            available_at=NOW - timedelta(seconds=1),
            horizon="5D",
            prediction=0.01,
            probability=0.6,
            data_hash="D",
            training_manifest="T",
            validation_manifest="V",
        )


def test_temporal_splits_are_ordered_purged_and_non_overlapping() -> None:
    index = pd.date_range("2024-01-01", periods=100, tz="UTC")
    splits = causal_time_splits(index, purge_observations=2)
    assert list(splits) == [
        "TRAIN",
        "VALIDATION",
        "TEST",
        "FORWARD",
        "LIVE_SHADOW",
    ]
    flattened = [value for rows in splits.values() for value in rows]
    assert len(flattened) == len(set(flattened))
    assert max(splits["TRAIN"]) + 2 < min(splits["VALIDATION"])
    assert max(splits["VALIDATION"]) + 2 < min(splits["TEST"])


def test_point_in_time_validation_rejects_future_and_revision_leakage() -> None:
    report = validate_point_in_time_rows(
        [
            {
                "feature": "closed_bar",
                "event_time": NOW - timedelta(days=1),
                "available_at": NOW - timedelta(hours=20),
            },
            {
                "feature": "late_revision",
                "event_time": NOW - timedelta(days=30),
                "available_at": NOW + timedelta(seconds=1),
            },
        ],
        decision_time=NOW,
    )
    assert len(report["accepted"]) == 1
    assert report["lookahead_rows"] == 1
    assert report["status"] == "PARTIAL"


def test_nlp_normalization_rejects_duplicate_and_late_news() -> None:
    report = normalize_nlp_events(
        [
            _event("E1", "HASH", NOW),
            _event("E2", "HASH", NOW),
            _event("E3", "OTHER", NOW + timedelta(minutes=1)),
        ],
        decision_time=NOW,
    )
    assert report["duplicate_count"] == 1
    assert report["late_count"] == 1
    assert not report["standalone_entry_allowed"]
    assert report["execution_authority"] == "NONE"


def test_research_loop_has_no_live_authorized_transition() -> None:
    hypothesis = ResearchHypothesis(
        hypothesis_id="H1",
        source="TEST",
        description="test",
        economic_rationale="test net returns",
        feature_dependencies=("momentum",),
        target="NET_RETURN",
        horizon="5D",
        created_at=NOW,
        status=ExperimentStatus.PROPOSED,
    )
    validating = transition_hypothesis(hypothesis, ExperimentStatus.VALIDATING)
    assert validating.status == ExperimentStatus.VALIDATING
    with pytest.raises(ValueError, match="invalid research transition"):
        transition_hypothesis(validating, ExperimentStatus.FORWARD_VALIDATED)
    assert "LIVE" not in {status.value for status in ExperimentStatus}


def test_model_health_pauses_on_drift_and_never_expands_risk() -> None:
    report = assess_model_health(
        _model(),
        now=NOW + timedelta(days=1),
        feature_drift=0.5,
        prediction_drift=0.0,
        calibration_drift=0.0,
        performance_drift=0.0,
        regime_drift=0.0,
        schema_matches=True,
    )
    assert report["status"] == "PAUSED_FAIL_CLOSED"
    assert report["recommended_lifecycle"] == ModelLifecycle.PAUSED
    assert not report["risk_limit_increase_allowed"]
    assert not report["automatic_model_promotion"]


def test_false_discovery_control_counts_all_hypotheses() -> None:
    report = false_discovery_control(
        {"h1": 0.001, "h2": 0.02, "h3": 0.9}, alpha=0.05
    )
    assert report["hypothesis_count"] == 3
    assert report["discoveries"] == ["h1", "h2"]
    assert report["adjusted_p_values"]["h3"] == pytest.approx(0.9)


def test_ai_authority_rejects_money_and_order_power() -> None:
    report = validate_ai_authority(
        [
            {
                "component_id": "BAD",
                "authority": "SHADOW_ONLY",
                "money_control": True,
                "execution_authority": "LIVE",
                "granted_powers": ["ORDER_AUTHORITY"],
            }
        ]
    )
    assert report["status"] == "NO_GO"
    assert any("MONEY_CONTROL_FORBIDDEN" in row for row in report["violations"])
    assert any("ORDER_AUTHORITY" in row for row in report["violations"])


def test_ai_portfolio_proposal_is_weight_only_and_long_only() -> None:
    proposal = AIPortfolioProposal(
        proposal_id="P1",
        model_id="M1",
        as_of=NOW,
        target_weights={"AAPL": 0.25, "CASH": 0.75},
    )
    assert not proposal.publishes_broker_quantity
    assert proposal.native_translation_required
    assert proposal.execution_authority == "NONE"
    with pytest.raises(ValidationError):
        AIPortfolioProposal(
            proposal_id="P2",
            model_id="M1",
            as_of=NOW,
            target_weights={"AAPL": 1.1},
        )


def test_experiment_records_are_reproducible_and_immutable(
    tmp_path: Path,
) -> None:
    record = ExperimentRecord(
        experiment_id="EXP-1",
        hypothesis_id="H1",
        code_hash="CODE",
        dataset_hash="DATA",
        cutoff=NOW,
        parameters={"alpha": 1},
        seed=42,
        transaction_cost_model_version="COST-V1",
        result_artifact="output/result.json",
        decision=ExperimentStatus.FAILED,
        hypothesis_count_at_selection=5,
        multiple_testing={"method": "BH"},
    )
    path = write_immutable_experiment(tmp_path, record)
    assert write_immutable_experiment(tmp_path, record) == path
    changed = record.model_copy(update={"dataset_hash": "OTHER"})
    with pytest.raises(ValueError, match="immutable experiment"):
        write_immutable_experiment(tmp_path, changed)


def test_ai_package_has_no_broker_import_or_writer_call() -> None:
    report = audit_ai_import_boundary(ROOT / "src/stocks/ai")
    assert report["status"] == "GO"
    assert report["broker_imports"] == 0
    assert report["writer_calls"] == 0


def test_publish_plane_inspects_all_repos_and_keeps_33_capabilities() -> None:
    status = publish_ai_research_plane(ROOT)
    assert status["status"] == "GO"
    assert status["reference_repo_count"] == 14
    assert status["capability_count"] == 33
    assert status["financial_validation_status"] == "NO_INCREMENTAL_EVIDENCE"
    assert not status["ai_money_control"]
    assert status["broker_calls"] == status["writer_calls"] == 0
    matrix = json.loads(
        (ROOT / "output/ai/reference-repo-integration-matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(matrix["repositories"]) == 14
    assert all(row["status"] == "PRESENT_INSPECTED" for row in matrix["repositories"])
    assert len(matrix["capabilities"]) == 33
    assert not matrix["capability_34_added"]
