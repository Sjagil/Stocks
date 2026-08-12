from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from stocks.ai.contracts import ModelEvidence
from stocks.ai.intelligence import enqueue_refresh_if_due
from stocks.ai.modeling import (
    PlattCalibrator,
    _oos_uncertainty_evidence,
    _classifier_templates,
    _regressor_templates,
    active_swing_candidate_unit_readiness,
    performance_gate_components,
    purged_walk_forward_splits,
    ranker_query_layout,
    timeframe_ablation_readiness,
)
from stocks.ai.panel import build_causal_bar_features
from stocks.portfolio.learning_integration import integrate_learning_evidence


def test_model_evidence_is_typed_shadow_only_and_causal() -> None:
    now = datetime.now(UTC)
    evidence = ModelEvidence(
        evidence_id="E-1",
        model_version="M-1",
        symbol="AAPL",
        as_of=now,
        feature_timestamp=now - timedelta(days=1),
        probability_positive_net=0.61,
        predicted_net_return=0.012,
        expected_win=0.08,
        expected_loss=-0.04,
        conservative_expected_value=0.0332,
        uncertainty=0.7,
        cross_sectional_rank=0.8,
        meta_take=True,
        abstained=False,
        out_of_distribution=False,
        validation_status="REJECTED_NO_INCREMENTAL_OOS_VALUE",
        tournament_hash="T-1",
        feature_hash="F-1",
    )
    assert evidence.authority == "SHADOW_ONLY"
    assert evidence.money_control is False
    assert evidence.mutates_financial_fields is False
    assert evidence.execution_authority == "NONE"

    with pytest.raises(ValidationError):
        ModelEvidence(
            **{
                **evidence.model_dump(),
                "feature_timestamp": now + timedelta(seconds=1),
            }
        )


def test_purged_walk_forward_never_uses_unavailable_labels() -> None:
    dates = pd.date_range("2022-01-01", periods=420, tz="UTC")
    panel = pd.DataFrame(
        {
            "decision_timestamp": np.repeat(dates, 2),
            "label_available_at": np.repeat(dates + pd.Timedelta(days=12), 2),
        }
    )
    folds = list(purged_walk_forward_splits(panel, folds=3))
    assert len(folds) >= 2
    decision = pd.to_datetime(panel["decision_timestamp"], utc=True)
    available = pd.to_datetime(panel["label_available_at"], utc=True)
    for fold in folds:
        train = fold.train_indices
        validation = fold.validation_indices
        test = fold.test_indices
        assert available.iloc[train].max() < decision.iloc[validation].min()
        assert decision.iloc[validation].max() < decision.iloc[test].min()
        assert set(train).isdisjoint(validation)
        assert set(train).isdisjoint(test)
        assert set(validation).isdisjoint(test)


def test_lightgbm_is_a_deterministic_shadow_challenger() -> None:
    classifier = _classifier_templates(42)["LIGHTGBM"].named_steps["model"]
    regressor = _regressor_templates(42)["LIGHTGBM"].named_steps["model"]
    assert classifier.get_params()["deterministic"] is True
    assert regressor.get_params()["deterministic"] is True
    assert classifier.get_params()["n_jobs"] == 1
    assert regressor.get_params()["objective"] == "regression_l1"


def test_timeframe_ablations_count_blocked_trials_without_fitting() -> None:
    rows = 600
    panel = pd.DataFrame(
        {
            "decision_timestamp": pd.date_range(
                "2020-01-01", periods=rows, freq="D", tz="UTC"
            ),
            "return_15m": [np.nan] * rows,
            "return_1h": np.linspace(-0.01, 0.01, rows),
            "return_2h": np.linspace(-0.02, 0.02, rows),
            "return_4h": np.linspace(-0.03, 0.03, rows),
            "return_1d": np.linspace(-0.04, 0.04, rows),
        }
    )
    report = timeframe_ablation_readiness(panel)
    assert report["variant_count"] == 5
    assert report["eligible_variant_count"] == 0
    assert report["blocked_variant_count"] == 5
    assert all(row["trial_counted"] is True for row in report["variants"])
    assert all(row["model_fitted"] is False for row in report["variants"])
    one_day_variants = [
        row for row in report["variants"] if "PLUS_1D" in row["variant"] or row["variant"].endswith("_1D")
    ]
    assert one_day_variants
    assert all("return_1d" in row["features"] for row in one_day_variants)
    assert all("return_1" not in row["features"] for row in report["variants"])
    assert all(
        "INSUFFICIENT_CAUSAL_COVERAGE" in row["blockers"]
        for row in report["variants"]
    )


def test_active_swing_candidate_unit_requires_unique_natural_setups() -> None:
    valid = pd.DataFrame(
        {
            "candidate_unit": [
                "ONE_NATURAL_STRATEGY_SETUP",
                "ONE_NATURAL_STRATEGY_SETUP",
            ],
            "candidate_identity": ["SETUP-1", "SETUP-2"],
        }
    )
    assert active_swing_candidate_unit_readiness(valid) is True
    duplicate = valid.copy()
    duplicate.loc[1, "candidate_identity"] = "SETUP-1"
    assert active_swing_candidate_unit_readiness(duplicate) is False
    legacy = valid.assign(candidate_unit="ONE_EXECUTED_OOS_TRADE_EPISODE")
    assert active_swing_candidate_unit_readiness(legacy) is False


def test_oos_uncertainty_undercoverage_is_explicit() -> None:
    rows = 100
    report = _oos_uncertainty_evidence(
        pd.DataFrame(
            {
                "classifier_probability_std": np.full(rows, 0.03),
                "regressor_prediction_std": np.full(rows, 0.01),
                "return_interval_lower_90": np.full(rows, -0.05),
                "return_interval_upper_90": np.full(rows, 0.05),
                "return_interval_contains_realized": [True] * 85
                + [False] * 15,
            }
        )
    )
    assert report["empirical_return_interval_coverage"] == 0.85
    assert report["status"] == "UNDER_COVERAGE_RESEARCH_ONLY"
    assert report["future_test_labels_used_for_interval_fit"] is False


def test_ranker_queries_use_exact_decision_timestamp_and_contiguous_groups() -> None:
    frame = pd.DataFrame(
        {
            "decision_timestamp": [
                "2026-08-10T14:00:00Z",
                "2026-08-10T14:15:00Z",
                "2026-08-10T14:00:00Z",
                "2026-08-10T14:15:00Z",
                "2026-08-10T14:15:00Z",
            ],
            "strategy_id": ["B", "A", "A", "C", "B"],
            "security_id": ["2", "1", "1", "3", "2"],
            "native_exit_net_return": [0.03, -0.01, 0.01, 0.04, 0.02],
        }
    )
    layout = ranker_query_layout(frame)
    assert layout["group_sizes"] == [2, 3]
    assert layout["multi_candidate_query_count"] == 2
    assert layout["query_semantics"] == "EXACT_DECISION_TIMESTAMP"
    assert layout["relevance"].tolist() == [2, 4, 1, 3, 4]
    assert sum(layout["group_sizes"]) == len(frame)


def test_positive_rank_ic_cannot_override_negative_ranking_economics() -> None:
    components = performance_gate_components(
        {
            "meta_label": {
                "delta_net_expectancy": 0.01,
                "bootstrap_probability_of_improvement": 0.99,
            },
            "ranking": {
                "mean_rank_ic": 0.10,
                "delta_net_expectancy": -0.001,
                "bootstrap_probability_of_improvement": 0.10,
            },
        }
    )
    assert components["RANK_IC_POSITIVE"] is True
    assert components["RANKING_DELTA_NET_EXPECTANCY_POSITIVE"] is False
    assert components["RANKING_BOOTSTRAP_PROBABILITY_GE_095"] is False
    assert all(components.values()) is False


def test_platt_calibrator_handles_single_class_validation_fail_closed() -> None:
    raw = np.array([0.2, 0.4, 0.6, 0.8])
    calibrator = PlattCalibrator().fit(raw, np.ones(4, dtype=int))
    np.testing.assert_allclose(calibrator.transform(raw), raw)


def test_causal_features_do_not_depend_on_future_bar() -> None:
    dates = pd.date_range("2024-01-01", periods=90, tz="UTC")
    close = np.linspace(100.0, 125.0, len(dates))
    bars = pd.DataFrame(
        {
            "security_id": "ID-1",
            "ticker": "AAA",
            "date": dates,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1_000, 2_000, len(dates)),
            "sector": "TECH",
            "currency": "USD",
        }
    )
    before = build_causal_bar_features(bars).iloc[-2]
    changed = bars.copy()
    changed.loc[changed.index[-1], ["high", "close", "volume"]] = [500, 400, 9_000]
    after = build_causal_bar_features(changed).iloc[-2]
    for name in ("return_20d", "volatility_20d", "rsi_14d", "market_breadth_20d"):
        np.testing.assert_allclose(before[name], after[name], equal_nan=True)


def test_global_evidence_is_overlay_only_and_never_mutates_financial_fields() -> None:
    opportunity = {
        "symbol": "AAPL",
        "expected_net_return": 0.10,
        "expected_loss": 0.04,
        "execution_authority": "NONE",
    }
    evidence = {
        "global_model_evidence": [
            {
                "symbol": "AAPL",
                "validation_status": "REJECTED_NO_INCREMENTAL_OOS_VALUE",
                "meta_take": False,
                "abstained": True,
                "out_of_distribution": False,
            }
        ]
    }
    authority = {
        "capabilities": [
            {"id": 17, "authority": "SHADOW_ONLY"},
            {"id": 10, "authority": "CONTEXT_ONLY"},
            {"id": 32, "authority": "SHADOW_ONLY"},
        ]
    }
    overlaid, report = integrate_learning_evidence(
        [opportunity], evidence, authority
    )
    assert overlaid[0]["expected_net_return"] == opportunity["expected_net_return"]
    assert overlaid[0]["expected_loss"] == opportunity["expected_loss"]
    assert overlaid[0]["learning_overlay"]["financial_fields_mutated"] is False
    assert overlaid[0]["learning_overlay"]["execution_influence"] == "NONE"
    assert report["predictions_change_financial_fields"] is False
    assert report["broker_writes"] == 0


def test_fresh_async_refresh_dispatch_does_not_spawn(tmp_path) -> None:
    manifest = tmp_path / "output/ai/decision-intelligence/model-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"generated_at":"'
        + datetime.now(UTC).isoformat()
        + '","model_version":"M-1","promotion_status":"SHADOW_ONLY"}',
        encoding="utf-8",
    )
    status = enqueue_refresh_if_due(tmp_path)
    assert status["status"] == "SKIPPED_FRESH"
    assert "process_id" not in status
    assert status["execution_authority"] == "NONE"
    assert status["broker_writes"] == 0
