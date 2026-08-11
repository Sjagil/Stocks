from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks.quant_platform.ml import (
    MetaLabelingEngine,
    ProbabilityCalibratedSignalEngine,
    ReturnPredictionModel,
    TemporalConvolutionalReturnModel,
    build_economic_return_target,
)


def _dataset(rows: int = 240) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(21)
    x = pd.DataFrame(
        {
            "momentum": rng.normal(size=rows),
            "volatility": rng.lognormal(-2, 0.3, rows),
            "breadth": rng.uniform(0, 1, rows),
            "regime": rng.normal(size=rows),
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="B"),
    )
    future = 0.01 * x["momentum"] - 0.02 * x["volatility"] + 0.004 * x["breadth"] + rng.normal(0, 0.006, rows)
    return x, future


def test_economic_target_requires_return_above_cost_threshold() -> None:
    prices = pd.Series([100, 101, 102, 101, 105, 104, 108], dtype=float)
    result = build_economic_return_target(prices, horizon=2, transaction_cost_threshold=0.025)
    assert result.loc[0, "target"] == 0
    assert result.loc[3, "target"] == 1
    assert pd.isna(result.iloc[-1]["target"])


@pytest.mark.parametrize("model_type", ["logistic", "ridge", "lasso", "elastic_net", "random_forest"])
def test_classical_models_use_temporal_validation_and_calibrated_probabilities(model_type: str) -> None:
    features, future = _dataset()
    model = ReturnPredictionModel(model_type=model_type, splits=3).fit(features, (future > 0).astype(int))
    probability = model.predict_proba(features.tail(10))
    assert probability.between(0, 1).all()
    assert 0 <= model.validation["brier_score"] <= 1
    assert model.report()["probability_calibrated"]


def test_meta_labeling_filters_primary_strategy_instead_of_replacing_it() -> None:
    features, future = _dataset()
    side = pd.Series(np.where(features["momentum"] > 0, 1, -1), index=features.index)
    engine = MetaLabelingEngine(random_state=2).fit(features, side, future, 0.001)
    decisions = engine.decide(features.tail(20))
    assert set(decisions["meta_label"]) <= {"TAKE_TRADE", "SKIP_TRADE"}
    assert decisions["take_probability"].between(0, 1).all()


def test_probability_signal_reports_return_downside_uncertainty_and_vol_scaled_size() -> None:
    features, future = _dataset()
    engine = ProbabilityCalibratedSignalEngine(random_state=3).fit(features, future, cost_threshold=0.001)
    forecast_volatility = pd.Series(0.02, index=features.tail(10).index)
    result = engine.predict(features.tail(10), forecast_volatility, maximum_position=0.25)
    assert {
        "expected_return",
        "probability_positive_after_costs",
        "expected_downside",
        "uncertainty",
        "forecast_volatility",
        "position_size",
    } == set(result)
    assert result["position_size"].between(0, 0.25).all()
    assert result["uncertainty"].between(0, 1).all()


def test_temporal_model_rejects_noncausal_row_order() -> None:
    features, future = _dataset()
    with pytest.raises(ValueError, match="ordered causally"):
        ReturnPredictionModel().fit(features.iloc[::-1], (future > 0).astype(int))


def test_tcn_trains_causal_dilated_filters_with_temporal_holdout() -> None:
    features, future = _dataset(220)
    length = 20
    values = features.to_numpy(dtype=float)
    sequences = np.asarray(
        [values[index - length + 1 : index + 1] for index in range(length - 1, len(values))]
    )
    labels = (future.iloc[length - 1 :] > 0).astype(int)
    model = TemporalConvolutionalReturnModel(
        dilations=(1, 2, 4),
        channels=3,
        kernel_size=3,
        epochs=35,
        splits=3,
        random_state=9,
    ).fit(sequences, labels)
    probability = model.predict_proba(sequences[-8:])
    assert np.logical_and(probability >= 0, probability <= 1).all()
    report = model.report()
    assert report["backend"] == "NUMPY_TRAINED_CAUSAL_DILATED_CONVOLUTION"
    assert report["receptive_field"] == 9
    assert report["temporal_validation"]["out_of_sample_observations"] > 0
    assert report["money_control"] is False
    assert report["execution_authority"] == "NONE"


def test_tcn_rejects_sequence_shorter_than_receptive_field() -> None:
    sequences = np.ones((100, 5, 2))
    labels = np.tile([0, 1], 50)
    with pytest.raises(ValueError, match="receptive field"):
        TemporalConvolutionalReturnModel(
            dilations=(1, 4), kernel_size=3, epochs=2
        ).fit(sequences, labels)
