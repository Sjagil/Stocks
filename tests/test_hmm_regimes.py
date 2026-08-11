from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stocks.regimes.service as regime_service
from stocks.dynamic.service import _portfolio
from stocks.regimes.audit import (
    HMMStatePersistence,
    audit_transition_stability,
)
from stocks.regimes.canonicalize import canonicalize_states
from stocks.regimes.features import (
    engineer_daily_cross_asset_features,
    load_point_in_time_macro,
    standardize_train_oos,
)
from stocks.regimes.filter import IntradayHMMFilter, hamilton_filter
from stocks.regimes.model import FrozenHMM
from stocks.regimes.risk_overlay import (
    allowed_trade_risk,
    regime_multiplier,
)
from stocks.regimes.service import (
    _frozen_shadow_registry,
    _promotion_registry,
    regimes_current,
)
from stocks.research.phase11_8 import _run_portfolio


def _model() -> FrozenHMM:
    return FrozenHMM(
        n_regimes=3,
        feature_names=("x",),
        transition=(
            (0.90, 0.08, 0.02),
            (0.08, 0.84, 0.08),
            (0.02, 0.08, 0.90),
        ),
        intercepts=(0.01, 0.0, -0.02),
        coefficients=((0.1,), (0.0,), (-0.1,)),
        variances=(0.01, 0.04, 0.16),
        initial_probabilities=(0.5, 0.3, 0.2),
        raw_to_label={
            0: "RISK_ON_TREND",
            1: "NEUTRAL_CHOPPY",
            2: "STRESS_HIGH_VOL",
        },
        converged=True,
        log_likelihood=-10.0,
        expected_durations=(10.0, 6.25, 10.0),
        training_state_occupancy=(0.40, 0.35, 0.25),
        training_observations=200,
        feature_means={"world_index_ret": 0.0, "x": 0.0},
        feature_scales={"world_index_ret": 1.0, "x": 1.0},
    )


def test_daily_features_are_causal_and_use_requested_annualization() -> None:
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    raw = pd.DataFrame(
        {
            "world_index": np.linspace(100, 130, len(index)),
            "bond_index": np.linspace(100, 105, len(index)),
            "commodity_index": np.linspace(90, 120, len(index)),
        },
        index=index,
    )
    original = engineer_daily_cross_asset_features(
        raw,
        periods_per_year=52,
    )
    changed = raw.copy()
    changed.loc[index[-1], "world_index"] = 1_000
    mutated = engineer_daily_cross_asset_features(
        changed,
        periods_per_year=52,
    )
    pd.testing.assert_frame_equal(
        original.loc[original.index < index[-1]],
        mutated.loc[mutated.index < index[-1]],
    )


def test_macro_available_at_blocks_pre_release_visibility(
    tmp_path: Path,
) -> None:
    database = tmp_path / "data" / "macro" / "private" / "macro.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE observations (
                series_id TEXT,
                available_at TEXT,
                observation_date TEXT,
                created_at TEXT,
                payload_json TEXT
            )
            """
        )
        connection.execute(
            "INSERT INTO observations VALUES (?, ?, ?, ?, ?)",
            (
                "VIX",
                "2024-01-03T00:00:00",
                "2024-01-02",
                "2024-01-03T00:00:01",
                json.dumps({"original_value": 20.0}),
            ),
        )
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    result = load_point_in_time_macro(tmp_path, index, ("VIX",))
    assert pd.isna(result.loc["2024-01-02", "VIX"])
    assert result.loc["2024-01-03", "VIX"] == 20.0
    assert result.loc["2024-01-04", "VIX"] == 20.0


def test_standardization_is_fit_on_train_only() -> None:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    oos = pd.DataFrame({"a": [1_000_000.0]})
    scaled_train, _, scaler = standardize_train_oos(train, oos)
    assert scaler["a"]["mean"] == 2.0
    assert scaled_train["a"].mean() == pytest.approx(0.0)


def test_state_canonicalization_is_deterministic() -> None:
    mapping = canonicalize_states(
        np.array([0.01, 0.20, 0.04]),
        np.array([0.03, -0.04, 0.01]),
    )
    assert mapping.raw_to_label == {
        0: "RISK_ON_TREND",
        1: "STRESS_HIGH_VOL",
        2: "NEUTRAL_CHOPPY",
    }


def test_hamilton_filter_is_recursive_and_probabilities_sum_to_one() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="D")
    features = pd.DataFrame(
        {
            "world_index_ret": [0.01, 0.02, -0.01, 0.0, -0.05, 0.01],
            "x": [0.1, 0.2, -0.1, 0.0, -0.5, 0.1],
        },
        index=index,
    )
    original = hamilton_filter(_model(), features)
    changed = features.copy()
    changed.iloc[-1] = [0.9, 0.9]
    mutated = hamilton_filter(_model(), changed)
    assert np.allclose(original.sum(axis=1), 1.0)
    pd.testing.assert_frame_equal(original.iloc[:-1], mutated.iloc[:-1])


def test_transition_audit_uses_training_occupancy() -> None:
    probabilities = pd.DataFrame(
        {
            "RISK_ON_TREND": [0.99] * 10,
            "NEUTRAL_CHOPPY": [0.005] * 10,
            "STRESS_HIGH_VOL": [0.005] * 10,
        }
    )
    report = audit_transition_stability(
        _model(),
        probabilities,
        minimum_state_fraction=0.05,
        minimum_duration=5.0,
        maximum_chatter_ratio=0.15,
    )
    assert report["checks"]["minimum_state_fraction"] is True


def test_hysteresis_requires_confirmations() -> None:
    filter_ = IntradayHMMFilter(
        activation_threshold=0.75,
        deactivation_threshold=0.30,
        minimum_confirmations=2,
    )
    probabilities = {
        "RISK_ON_TREND": 0.05,
        "NEUTRAL_CHOPPY": 0.10,
        "STRESS_HIGH_VOL": 0.85,
    }
    assert filter_.update(probabilities)["active_state"] == "NEUTRAL_CHOPPY"
    assert filter_.update(probabilities)["active_state"] == "STRESS_HIGH_VOL"


def test_overlay_is_bounded_and_only_reduces_risk() -> None:
    probabilities = pd.DataFrame(
        {
            "RISK_ON_TREND": [1.0, 0.0],
            "NEUTRAL_CHOPPY": [0.0, 0.0],
            "STRESS_HIGH_VOL": [0.0, 1.0],
        }
    )
    multiplier = regime_multiplier(
        probabilities,
        {
            "RISK_ON_TREND": 1.0,
            "NEUTRAL_CHOPPY": 0.6,
            "STRESS_HIGH_VOL": 0.15,
        },
    )
    assert multiplier.tolist() == [1.0, 0.15]
    assert allowed_trade_risk(100.0, 0.15, 0.5) == 7.5
    with pytest.raises(ValueError, match="INVALID_HMM_RISK_INPUT"):
        allowed_trade_risk(100.0, 1.01, 1.0)


def test_state_store_is_append_only(tmp_path: Path) -> None:
    store = HMMStatePersistence(tmp_path)
    store.append_state({"as_of": "2024-01-01", "state": "A"})
    store.append_state({"as_of": "2024-01-02", "state": "B"})
    lines = (tmp_path / "filtered-states.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["state"] == "A"


def test_current_regime_uses_validated_current_provider_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private = tmp_path / "data" / "regimes" / "private"
    private.mkdir(parents=True)
    model_path = private / "model.json"
    model_path.write_text("{}", encoding="utf-8")
    (private / "current-model.json").write_text(
        json.dumps(
            {
                "model_path": str(model_path),
                "model_hash": "MODEL-HASH",
            }
        ),
        encoding="utf-8",
    )
    (private / "current-state.json").write_text(
        json.dumps(
            {
                "as_of": "2024-01-01T00:00:00",
                "probabilities": {
                    "RISK_ON_TREND": 0.5,
                    "NEUTRAL_CHOPPY": 0.3,
                    "STRESS_HIGH_VOL": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    features = pd.DataFrame(
        {
            "world_index_ret": [0.01, 0.02],
            "x": [0.1, 0.2],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    current_frames = {"SPY": pd.DataFrame()}
    calls = {"current": 0}

    def load_current(_: Path) -> dict[str, dict[str, pd.DataFrame]]:
        calls["current"] += 1
        return {"1d": current_frames}

    monkeypatch.setattr(regime_service, "_load_current_frames", load_current)
    monkeypatch.setattr(
        regime_service,
        "_load_frames",
        lambda _: pytest.fail("historical loader used for current regime"),
    )
    monkeypatch.setattr(
        regime_service,
        "frozen_hmm_from_payload",
        lambda _: _model(),
    )
    monkeypatch.setattr(
        regime_service,
        "_feature_matrix",
        lambda _root, frames, timeframe: (
            features,
            {
                "feature_set": "TEST",
                "received_current_frames": frames is current_frames,
                "timeframe": timeframe,
            },
        ),
    )
    monkeypatch.setattr(
        regime_service,
        "_scale_with_model",
        lambda frame, _model: frame,
    )
    monkeypatch.setattr(
        regime_service,
        "hamilton_filter",
        lambda _model, frame, **_: pd.DataFrame(
            {
                "RISK_ON_TREND": [0.6],
                "NEUTRAL_CHOPPY": [0.3],
                "STRESS_HIGH_VOL": [0.1],
            },
            index=frame.index,
        ),
    )
    monkeypatch.setattr(
        regime_service,
        "_probability_multiplier",
        lambda *_: 0.75,
    )
    monkeypatch.setattr(regime_service, "_config", lambda _: {})

    result = regimes_current(tmp_path)

    assert calls["current"] == 1
    assert result["new_filtered_rows"] == 1
    assert result["state"]["as_of"] == "2024-01-02T00:00:00"
    assert result["feature_audit"]["received_current_frames"] is True
    assert result["EXECUTION_AUTHORITY"] == "NONE"


def test_portfolio_overlay_uses_previous_bar_and_reduces_position() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "open": [10.0] * 5,
            "high": [11.0] * 5,
            "low": [9.0] * 5,
            "close": [10.0] * 5,
            "volume": [1_000.0] * 5,
        },
        index=dates,
    )
    signals = {
        "A": pd.DataFrame(
            {"signal": [True] * 5, "score": [1.0] * 5},
            index=dates,
        )
    }
    multiplier = pd.Series(
        [1.0, 1.0, 0.0, 0.0, 0.0],
        index=dates,
    )
    result = _run_portfolio(
        {"A": frame},
        signals,
        start=dates[1],
        end=dates[-1],
        cost_bps=10.0,
        exposure_multiplier=multiplier,
        entry_block_below_multiplier=0.2,
    )
    fills = result["fills"]
    assert fills.iloc[0]["side"] == "BUY"
    assert fills.iloc[0]["date"] == dates[1]
    assert "SELL" in fills.loc[
        fills["date"].eq(dates[3]), "side"
    ].iloc[0]


def test_promotion_never_grants_execution_authority() -> None:
    summary = pd.DataFrame(
        [
            {
                "strategy": "trend",
                "timeframe": "1d",
                "fold_count": 12,
                "baseline_median_pf": 1.1,
                "hmm_median_pf": 1.2,
                "baseline_median_CAGR": 0.1,
                "hmm_median_CAGR": 0.11,
                "hmm_positive_fold_ratio": 0.8,
                "baseline_cost_50bps_median_pf": 1.05,
                "hmm_cost_50bps_median_pf": 1.1,
                "sharpe_ablation_ratio": 1.2,
                "baseline_worst_drawdown": -0.3,
                "hmm_worst_drawdown": -0.2,
                "drawdown_reduction_fraction": 1 / 3,
                "hmm_incremental_pf_fold_ratio": 0.7,
                "stable_model_fold_ratio": 0.8,
            }
        ]
    )
    config = {
        "promotion": {
            "minimum_fold_count": 10,
            "minimum_baseline_median_pf": 1.0,
            "minimum_hmm_median_pf": 1.0,
            "minimum_hmm_positive_fold_ratio": 0.6,
            "minimum_cost_50bps_pf": 1.0,
            "minimum_sharpe_ablation_ratio": 1.15,
            "minimum_drawdown_reduction_fraction": 0.2,
            "minimum_incremental_fold_ratio": 0.6,
        }
    }
    registry = _promotion_registry(summary, config)
    assert registry["hmm_overlay_promoted_count"] == 1
    assert registry["promotions"][0]["destination"] == "FROZEN_SHADOW"
    assert registry["EXECUTION_AUTHORITY"] == "NONE"
    assert registry["automatic_execution_promotion"] is False


def test_runtime_hmm_overlay_can_only_reduce_exposure(
    tmp_path: Path,
) -> None:
    without_hmm = _portfolio(tmp_path, {"signals": []})
    path = (
        tmp_path
        / "output"
        / "research"
        / "phase11_11"
        / "current.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "GO",
                "state": {
                    "as_of": "2026-07-24T00:00:00",
                    "regime_multiplier": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )
    with_hmm = _portfolio(tmp_path, {"signals": []})
    assert (
        with_hmm["max_total_exposure_pct"]
        <= without_hmm["max_total_exposure_pct"]
    )
    assert with_hmm["hmm_risk_multiplier"] == 0.5
    assert with_hmm["execution_authority"] == "NONE"


def test_frozen_shadow_registry_contains_only_promotions() -> None:
    promotions = {
        "promotions": [
            {
                "strategy": "ma_crossover",
                "timeframe": "1w",
                "destination": "FROZEN_SHADOW",
                "selected_variant": "BASELINE",
            },
            {
                "strategy": "ma_channel",
                "timeframe": "1d",
                "destination": "RESEARCH_CANDIDATE",
                "selected_variant": "NONE",
            },
        ]
    }
    selections = pd.DataFrame(
        [
            {
                "strategy": "ma_crossover",
                "timeframe": "1w",
                "fold_id": "1w_F001",
                "selected_profile": "balanced",
            }
        ]
    )
    registry = _frozen_shadow_registry(promotions, selections)
    assert registry["candidate_count"] == 1
    assert registry["candidates"][0]["classification"] == "FROZEN_SHADOW"
    assert registry["candidates"][0]["execution_authority"] == "NONE"
