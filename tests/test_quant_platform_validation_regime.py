from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stocks.quant_platform import (
    HiddenMarkovRegimeDetector,
    RuleBasedRegimeDetector,
    StatisticalRegimeDetector,
    StrategyAllocationEngine,
    WalkForwardValidator,
    cost_stress,
    parameter_sensitivity,
    stability_report,
)
from stocks.regimes.model import FrozenHMM


def test_walk_forward_has_causal_purged_nonoverlapping_folds() -> None:
    validator = WalkForwardValidator(train_size=100, test_size=20, step_size=20, purge_size=5)
    folds = validator.split(200)
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_indices.max() + 5 < fold.test_indices.min()
        assert len(fold.train_indices) == 100
        assert len(fold.test_indices) == 20


def test_walk_forward_selects_on_train_and_scores_out_of_sample() -> None:
    data = pd.DataFrame({"x": np.arange(180.0)})
    validator = WalkForwardValidator(train_size=80, test_size=20, step_size=20)
    result = validator.evaluate(
        data,
        parameter_grid=[{"scale": 1}, {"scale": 2}],
        train_score=lambda frame, params: float(frame["x"].mean() * params["scale"]),
        test_score=lambda frame, params: float(frame["x"].mean() / params["scale"]),
    )
    assert all(item == {"scale": 2} for item in result["parameters"])
    report = stability_report(result)
    assert report["folds"] == len(result)
    assert report["execution_authority"] == "NONE"


def test_cost_stress_and_parameter_plateau() -> None:
    stressed = cost_stress(pd.Series([0.01, -0.002]), pd.Series([1.0, 0.5]), [0, 10, 50])
    assert stressed["total_net_return"].is_monotonic_decreasing
    sensitivity = parameter_sensitivity(
        pd.DataFrame({"fast": [5, 10, 15, 20, 25], "score": [1.0, 1.8, 2.0, 1.9, 1.1]}),
        parameter_columns=["fast"],
    )
    assert sensitivity["best_parameters"] == {"fast": 15.0}
    assert sensitivity["plateau_fraction"] > 0


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        ({"vix": 45, "credit_spread": 6, "liquidity_growth": -1}, "LIQUIDITY_CRISIS"),
        ({"vix": 20, "pmi": 43, "yield_curve_10y_2y": -0.5}, "RECESSION"),
        ({"vix": 18, "pmi": 55, "spx_momentum": 0.1, "inflation_trend": 1}, "EXPANSION_INFLATION"),
        ({"vix": 18, "pmi": 55, "spx_momentum": 0.1, "inflation_trend": -1}, "EXPANSION_DISINFLATION"),
        ({"vix": 28, "pmi": 51, "spx_momentum": 0.01, "market_breadth": 0.3}, "RISK_OFF"),
    ],
)
def test_rule_based_regime_classification(features: dict[str, float], expected: str) -> None:
    assert RuleBasedRegimeDetector().classify(features)["regime"] == expected


@pytest.mark.parametrize("method", ["kmeans", "gmm"])
def test_statistical_regimes_produce_calibrated_rows(method: str) -> None:
    rng = np.random.default_rng(9)
    first = rng.normal(loc=[-0.08, 35, -1], scale=[0.02, 3, 0.2], size=(80, 3))
    second = rng.normal(loc=[0.08, 15, 1], scale=[0.02, 2, 0.2], size=(80, 3))
    frame = pd.DataFrame(np.vstack([first, second]), columns=["spx_momentum", "vix", "liquidity_growth"])
    detector = StatisticalRegimeDetector(method=method, n_regimes=2).fit(frame)
    probabilities = detector.predict_proba(frame.tail(5))
    probability_columns = [column for column in probabilities if column not in {"regime", "confidence"}]
    assert np.allclose(probabilities[probability_columns].sum(axis=1), 1.0)
    assert probabilities["confidence"].between(0.5, 1.0).all()


def test_strategy_allocation_mixes_regimes_and_sums_to_one() -> None:
    result = StrategyAllocationEngine().allocate({"RISK_ON": 0.7, "RISK_OFF": 0.3})
    assert sum(result["strategy_weights"].values()) == pytest.approx(1.0)
    assert result["strategy_weights"]["trend"] == pytest.approx(0.275)
    assert result["broker_writes"] == 0


def test_hidden_markov_adapter_uses_recursive_hamilton_probabilities() -> None:
    model = FrozenHMM(
        n_regimes=3,
        feature_names=("x",),
        transition=((0.9, 0.08, 0.02), (0.08, 0.84, 0.08), (0.02, 0.08, 0.9)),
        intercepts=(0.01, 0.0, -0.02),
        coefficients=((0.1,), (0.0,), (-0.1,)),
        variances=(0.01, 0.04, 0.16),
        initial_probabilities=(0.5, 0.3, 0.2),
        raw_to_label={0: "RISK_ON_TREND", 1: "NEUTRAL_CHOPPY", 2: "STRESS_HIGH_VOL"},
        converged=True,
        log_likelihood=-10.0,
        expected_durations=(10.0, 6.25, 10.0),
        training_state_occupancy=(0.4, 0.35, 0.25),
        training_observations=200,
        feature_means={"world_index_ret": 0.0, "x": 0.0},
        feature_scales={"world_index_ret": 1.0, "x": 1.0},
    )
    features = pd.DataFrame({"world_index_ret": [0.01, -0.02, 0.03], "x": [0.1, -0.2, 0.3]})
    detector = HiddenMarkovRegimeDetector(model)
    probabilities = detector.predict_proba(features)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert detector.report()["causal_filter"] == "HAMILTON_FILTER"
