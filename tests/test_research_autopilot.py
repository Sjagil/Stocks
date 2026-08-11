from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

import stocks.research.autopilot.service as autopilot_service
from stocks.research.autopilot.accounting import (
    apply_delisting_settlements,
    eur_total_return,
    validate_market_session_index,
)
from stocks.research.autopilot.components import component_registry
from stocks.research.autopilot.contracts import (
    ALLOWED_SWING_TIMEFRAMES,
    ResearchLevel,
    StrategyStatus,
    canonical_swing_timeframe,
    causal_higher_timeframe_map,
    validate_closed_candles,
)
from stocks.research.autopilot.engine import deterministic_fixture, run_backtest
from stocks.research.autopilot.benchmarks import run_simple_benchmarks
from stocks.research.autopilot.ensemble import (
    build_ensemble,
    combine_signals,
)
from stocks.research.autopilot.generator import (
    generate_strategies,
    validate_strategy,
)
from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger
from stocks.research.autopilot.risk import (
    COST_MODELS,
    PortfolioRiskLimits,
    enforce_portfolio_limits,
    hierarchical_group_weights,
)
from stocks.research.autopilot.service import (
    _decide,
    _publish,
    _publish_immutable,
    _walk_forward_results,
)
from stocks.research.autopilot.taxonomy import taxonomy_coverage_report
from stocks.research.autopilot.statistics import (
    cohort_stability,
    parameter_neighbor_stability,
    probability_of_backtest_overfitting,
    robustness_statistics,
)


@pytest.mark.parametrize("timeframe", ALLOWED_SWING_TIMEFRAMES)
def test_allowed_swing_timeframes(timeframe: str) -> None:
    assert canonical_swing_timeframe(timeframe) == timeframe


@pytest.mark.parametrize("timeframe", ["1m", "5m", "15m", "30m", "tick"])
def test_forbidden_timeframes_fail_closed(timeframe: str) -> None:
    with pytest.raises(ValueError, match="FORBIDDEN_SWING_TIMEFRAME"):
        canonical_swing_timeframe(timeframe)


def test_open_candle_is_excluded() -> None:
    frame = pd.DataFrame(
        {
            "bar_close_utc": [
                "2026-01-02T10:00:00Z",
                "2026-01-02T11:00:00Z",
            ],
            "close": [10.0, 11.0],
        }
    )
    closed = validate_closed_candles(
        frame, decision_time=pd.Timestamp("2026-01-02T10:30:00Z")
    )
    assert list(closed["close"]) == [10.0]


def test_higher_timeframe_mapping_is_delayed_until_close() -> None:
    low = pd.DataFrame(
        {
            "bar_close_utc": [
                "2026-01-02T09:00:00Z",
                "2026-01-02T10:00:00Z",
                "2026-01-02T11:00:00Z",
            ]
        }
    )
    high = pd.DataFrame(
        {
            "bar_close_utc": [
                "2026-01-02T08:30:00Z",
                "2026-01-02T10:30:00Z",
            ],
            "feature": [10.0, 20.0],
        }
    )
    mapped = causal_higher_timeframe_map(low, high)
    assert list(mapped["htf_feature"]) == [10.0, 10.0, 20.0]


def test_component_registry_is_complete_and_valid() -> None:
    registry = component_registry()
    assert len(registry) >= 80
    assert {
        "trend",
        "structure",
        "momentum",
        "mean_reversion",
        "volatility",
        "volume",
        "liquidity",
        "fundamental",
        "valuation",
        "regime",
        "sizing",
        "exit",
    }.issubset({component.category for component in registry.values()})
    for component in registry.values():
        component.validate()


def test_canonical_taxonomy_metadata_and_scope_are_enforced() -> None:
    report = taxonomy_coverage_report()
    assert report["status"] == "GO"
    assert report["canonical_section_count"] == 24
    assert report["component_metadata_failures"] == {}
    assert report["component_count"] >= 100
    assert report["scope"]["minimum_timeframe"] == "1h"
    assert {"5m", "15m"} <= set(report["scope"]["forbidden_timeframes"])
    assert report["equal_weight_benchmark_required"] is True
    assert report["strategy_authority"] == "NONE"


def test_generator_is_deterministic_bounded_and_covers_required_families() -> None:
    first = generate_strategies(budget=100, seed=42)
    second = generate_strategies(budget=100, seed=42)
    assert [item.strategy_hash for item in first] == [
        item.strategy_hash for item in second
    ]
    assert len(first) <= 100
    assert {item.family for item in first} == {
        "quality_momentum",
        "trend_pullback",
        "etf_rotation",
        "volatility_contraction_breakout",
        "commodity_etf_trend",
    }
    assert all(item.long_only for item in first)
    assert all(not item.leverage_allowed for item in first)
    assert all(not item.shorting_allowed for item in first)
    assert all("FUT" not in item.asset_scope for item in first)


def test_generator_rejects_excess_budget() -> None:
    with pytest.raises(ValueError, match="budget must be"):
        generate_strategies(budget=101)


def test_strategy_hash_mutation_is_blocked() -> None:
    strategy = generate_strategies(budget=1)[0]
    with pytest.raises(ValueError, match="STRATEGY_HASH_MISMATCH"):
        validate_strategy(replace(strategy, strategy_hash="BAD"))


def test_unregistered_and_out_of_bounds_parameters_are_blocked() -> None:
    strategy = generate_strategies(
        budget=1, family="etf_rotation"
    )[0]
    with pytest.raises(ValueError, match="PARAMETER_OUT_OF_BOUNDS"):
        validate_strategy(
            replace(strategy, parameters={**strategy.parameters, "top_n": 999})
        )
    with pytest.raises(ValueError, match="UNREGISTERED_PARAMETERS"):
        validate_strategy(
            replace(strategy, parameters={**strategy.parameters, "mystery": 1})
        )


def test_backtest_uses_long_only_capped_exposure_and_next_bar_execution() -> None:
    strategy = generate_strategies(
        budget=1, family="etf_rotation"
    )[0]
    bars, eligible = deterministic_fixture(periods=700)
    result = run_backtest(strategy, bars, eligible=eligible, fixture=True)
    assert result.status == "COMPLETE"
    assert result.metrics["maximum_exposure"] <= 1.0
    assert (result.weights >= 0).all().all()
    assert result.provenance["next_bar_execution"] is True
    assert result.provenance["closed_candles_only"] is True
    assert result.metrics["commission_cost"] >= 0
    assert result.metrics["spread_cost"] >= 0
    assert result.metrics["slippage_cost"] >= 0
    assert "closed_episodes" in result.metrics


def test_monthly_rotation_only_opens_after_scheduled_month_end_signal() -> None:
    strategy = generate_strategies(budget=1, family="etf_rotation")[0]
    bars, eligible = deterministic_fixture(periods=800)
    result = run_backtest(strategy, bars, eligible=eligible, fixture=True)
    opened = (result.weights > 0) & (
        result.weights.shift(1).fillna(0.0) == 0.0
    )
    entry_dates = result.weights.index[opened.any(axis=1)]
    assert len(entry_dates) > 0
    for timestamp in entry_dates:
        location = result.weights.index.get_loc(timestamp)
        assert location > 0
        previous = result.weights.index[location - 1]
        assert (timestamp.year, timestamp.month) != (
            previous.year,
            previous.month,
        )


def test_backtest_blocks_missing_pit_eligibility() -> None:
    strategy = generate_strategies(budget=1)[0]
    bars, eligible = deterministic_fixture(periods=700)
    eligible.loc[:, :] = False
    result = run_backtest(strategy, bars, eligible=eligible, fixture=False)
    assert result.status == "BLOCKED:PIT_ELIGIBILITY_UNAVAILABLE"


def test_actual_backtest_requires_complete_accounting_contract() -> None:
    strategy = generate_strategies(budget=1)[0]
    bars, eligible = deterministic_fixture(periods=700)
    result = run_backtest(
        strategy,
        bars,
        eligible=eligible,
        fixture=False,
        accounting_returns=pd.DataFrame(),
        data_contract={"point_in_time_fundamentals": True},
    )
    assert result.status.startswith("BLOCKED:DATA_CONTRACT_INCOMPLETE")


def test_eur_return_uses_exact_multiplicative_identity() -> None:
    local = pd.Series([0.10])
    fx = pd.Series([0.05])
    assert eur_total_return(local, fx).iloc[0] == pytest.approx(0.155)
    with pytest.raises(ValueError, match="MISSING_BLOCKING_FX_RETURN"):
        eur_total_return(local, pd.Series([float("nan")]))


def test_delisting_settlement_is_explicit_and_causal() -> None:
    index = pd.date_range("2026-01-01", periods=3, tz="UTC")
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=index)
    events = pd.DataFrame(
        {
            "symbol": ["A"],
            "settlement_date": [index[1]],
            "recovery_return": [-0.8],
            "available_at": [index[1]],
        }
    )
    adjusted, audit = apply_delisting_settlements(returns, events)
    assert adjusted.loc[index[1], "A"] == -0.8
    assert adjusted.loc[index[2], "A"] == 0.0
    assert audit["unknown_delisting_settlements"] == 0
    events.loc[0, "available_at"] = index[2]
    with pytest.raises(ValueError, match="NOT_POINT_IN_TIME"):
        apply_delisting_settlements(returns, events)


def test_market_calendar_blocks_weekend_bar() -> None:
    valid = validate_market_session_index(
        pd.DatetimeIndex(["2026-01-02T00:00:00Z"]), exchange="NASDAQ"
    )
    invalid = validate_market_session_index(
        pd.DatetimeIndex(["2026-01-03T00:00:00Z"]), exchange="NASDAQ"
    )
    assert valid["status"] == "GO"
    assert invalid["status"] == "UNEXPECTED_SESSION_BLOCKED"


def test_screener_eligibility_is_exact_date_point_in_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "screener.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE screener_observations (
          screening_date TEXT, symbol TEXT, classification TEXT, public_json TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO screener_observations VALUES (?,?,?,?)",
        (
            "2026-01-02",
            "A",
            "WATCHLIST",
            json.dumps(
                {
                    "asset_type": "STOCK",
                    "shariah_status": "SHARIAH_COMPLIANT",
                    "rejection_reasons": [],
                }
            ),
        ),
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(autopilot_service, "SCREENER_DB", database)
    index = pd.date_range("2026-01-01", periods=3, tz="UTC")
    bars = {"A": pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=index)}
    eligibility, audit = autopilot_service._load_pit_eligibility(
        bars, family="quality_momentum"
    )
    assert eligibility["A"].tolist() == [False, True, False]
    assert audit["backprojected_rows"] == 0
    assert audit["eligible_observation_count"] == 1


def test_double_cost_profile_is_exactly_double_normal() -> None:
    assert COST_MODELS["DOUBLE"].total_bps == 2 * COST_MODELS["NORMAL"].total_bps


def test_portfolio_limits_cap_position_sector_and_total_exposure() -> None:
    index = pd.date_range("2026-01-01", periods=3, tz="UTC")
    raw = pd.DataFrame(
        {"A": [0.8] * 3, "B": [0.8] * 3, "C": [0.8] * 3},
        index=index,
    )
    metadata = {
        "A": {"sector": "TECH"},
        "B": {"sector": "TECH"},
        "C": {"sector": "HEALTH"},
    }
    limits = PortfolioRiskLimits(
        max_position_weight=0.20,
        max_sector_weight=0.30,
        max_region_weight=1.0,
        max_currency_weight=1.0,
    )
    result = enforce_portfolio_limits(raw, metadata=metadata, limits=limits)
    assert float(result.max().max()) <= 0.20
    assert float(result[["A", "B"]].sum(axis=1).max()) <= 0.30
    assert float(result.sum(axis=1).max()) <= 1.0
    assert bool((result >= 0).all().all())


def test_backtest_rejects_open_input_bar() -> None:
    strategy = generate_strategies(budget=1)[0]
    bars, eligible = deterministic_fixture(periods=700)
    bars["FIX000"].iloc[-1, bars["FIX000"].columns.get_loc("is_closed")] = False
    with pytest.raises(ValueError, match="OPEN_CANDLE_BLOCKED"):
        run_backtest(strategy, bars, eligible=eligible, fixture=True)


def test_walk_forward_uses_five_bounded_oos_segments() -> None:
    strategy = generate_strategies(budget=1, family="etf_rotation")[0]
    bars, eligible = deterministic_fixture(periods=1_200)
    folds = _walk_forward_results(strategy, bars, eligible, fixture=True)
    assert [fold_id for fold_id, _ in folds] == [
        "OOS-01",
        "OOS-02",
        "OOS-03",
        "OOS-04",
        "OOS-05",
    ]
    assert all(result.status == "COMPLETE" for _, result in folds)
    assert all(result.metrics["observations"] < 300 for _, result in folds)
    assert all(result.metrics["purged_split"] is True for _, result in folds)
    assert all(result.metrics["embargo_periods"] == 5 for _, result in folds)


def test_ledger_append_only_identity_and_forward_gate(tmp_path: Path) -> None:
    ledger = ResearchLedger(AutopilotLayout(tmp_path))
    try:
        strategy = generate_strategies(budget=1)[0]
        assert ledger.register_strategies([strategy])["inserted"] == 1
        assert ledger.register_strategies([strategy])["existing"] == 1
        assert ledger.register_forward(strategy.strategy_id)["status"] == (
            "FORWARD_REGISTRATION_BLOCKED"
        )
        ledger.append_decision(
            strategy_id=strategy.strategy_id,
            new_status=StrategyStatus.ROBUSTNESS_PASS,
            research_level=ResearchLevel.FORWARD_OBSERVER_CANDIDATE,
            reasons=["TEST_EVIDENCE"],
            evidence={"test": True},
        )
        registration = ledger.register_forward(strategy.strategy_id)
        assert registration["status"] == "GO"
        assert registration["execution_authority"] == "NONE"
        assert registration["strategy_authority"] == "NONE"
        observation = ledger.append_forward_observation(
            registration_id=registration["registration_id"],
            session_date="2026-01-02",
            payload={"target_positions": {}, "automatic_orders": 0},
        )
        assert (
            ledger.append_forward_observation(
                registration_id=registration["registration_id"],
                session_date="2026-01-02",
                payload={"target_positions": {}, "automatic_orders": 0},
            )
            == observation
        )
        with pytest.raises(
            ValueError, match="FORWARD_OBSERVATION_IMMUTABILITY_CONFLICT"
        ):
            ledger.append_forward_observation(
                registration_id=registration["registration_id"],
                session_date="2026-01-02",
                payload={"target_positions": {"A": 0.1}, "automatic_orders": 0},
            )
    finally:
        ledger.close()


def test_trial_deduplication_is_content_addressed(tmp_path: Path) -> None:
    ledger = ResearchLedger(AutopilotLayout(tmp_path))
    try:
        strategy = generate_strategies(budget=1)[0]
        ledger.register_strategies([strategy])
        campaign = ledger.register_campaign({"cadence": "TEST", "family": strategy.family})
        kwargs = {
            "campaign_id": campaign,
            "strategy_id": strategy.strategy_id,
            "stage": 1,
            "cost_profile": "NORMAL",
            "status": "COMPLETE",
            "metrics": {"return": 1.0},
            "provenance": {"fixture": True},
        }
        assert ledger.append_trial(**kwargs) == ledger.append_trial(**kwargs)
        assert ledger.counts()["trials"] == 1
    finally:
        ledger.close()


@pytest.mark.parametrize(
    "mode",
    ["confirmation", "majority", "weighted", "unanimous", "hierarchical", "sleeves"],
)
def test_transparent_ensemble_modes_are_long_only(mode: str) -> None:
    spec = build_ensemble(
        ["S1", "S2", "S3"],
        ["trend", "trend", "momentum"],
        vote_mode=mode,
    )
    index = pd.date_range("2026-01-01", periods=2, tz="UTC")
    signals = {
        "S1": pd.DataFrame({"A": [1.0, -1.0], "B": [1.0, 1.0]}, index=index),
        "S2": pd.DataFrame({"A": [1.0, 1.0], "B": [0.0, 1.0]}, index=index),
        "S3": pd.DataFrame({"A": [0.0, 1.0], "B": [1.0, -1.0]}, index=index),
    }
    regime = pd.DataFrame(True, index=index, columns=["A", "B"])
    combined = combine_signals(signals, spec, regime_gate=regime)
    assert bool((combined >= 0).all().all())
    assert float(combined.sum(axis=1).max()) <= 1.0
    assert spec.frozen_weights is True
    assert spec.execution_authority == "NONE"


def test_ensemble_requires_regime_gate_when_frozen_spec_requires_it() -> None:
    spec = build_ensemble(["S1", "S2"], ["trend", "momentum"])
    signal = pd.DataFrame({"A": [1.0]}, index=pd.date_range("2026-01-01", periods=1))
    with pytest.raises(ValueError, match="ENSEMBLE_REGIME_GATE_REQUIRED"):
        combine_signals({"S1": signal, "S2": signal}, spec)


def test_statistics_report_insufficient_sample_instead_of_false_pass() -> None:
    returns = pd.Series(
        [0.01, -0.01] * 20,
        index=pd.date_range("2026-01-01", periods=40, tz="UTC"),
    )
    result = robustness_statistics(
        returns,
        closed_episodes=3,
        trial_count=100,
        periods_per_year=252,
    )
    assert result["sample_status"] == "INSUFFICIENT_SAMPLE"
    assert result["deflated_sharpe_probability"] is None
    assert result["block_bootstrap_runs"] == 0


def test_block_bootstrap_is_deterministic_and_bounded() -> None:
    rng = pd.Series(
        [0.001 + ((index % 7) - 3) * 0.0005 for index in range(800)],
        index=pd.bdate_range("2020-01-01", periods=800, tz="UTC"),
    )
    first = robustness_statistics(
        rng,
        closed_episodes=40,
        trial_count=12,
        periods_per_year=252,
        bootstrap_runs=100,
        seed=7,
    )
    second = robustness_statistics(
        rng,
        closed_episodes=40,
        trial_count=12,
        periods_per_year=252,
        bootstrap_runs=100,
        seed=7,
    )
    assert first == second
    assert first["sample_status"] == "EVALUABLE"
    assert first["block_bootstrap_runs"] == 100


def test_pbo_requires_sufficient_fold_matrix() -> None:
    small = pd.DataFrame({"A": [1.0], "B": [0.0]})
    assert probability_of_backtest_overfitting(small, small)["status"] == (
        "INSUFFICIENT_SAMPLE"
    )


def test_neighbor_and_cohort_stability_require_broad_support() -> None:
    neighbors = [
        {"net_total_return": 0.1, "maximum_drawdown": -0.2, "Sharpe": 0.5},
        {"net_total_return": 0.2, "maximum_drawdown": -0.3, "Sharpe": 0.6},
        {"net_total_return": 0.05, "maximum_drawdown": -0.1, "Sharpe": 0.4},
    ]
    assert parameter_neighbor_stability(neighbors)["status"] == "GO"
    cohorts = {
        "A": {"sample_status": "EVALUABLE", "net_total_return": 0.1},
        "B": {"sample_status": "EVALUABLE", "net_total_return": 0.2},
        "C": {"sample_status": "EVALUABLE", "net_total_return": -0.1},
    }
    assert cohort_stability(cohorts)["status"] == "GO"
    assert parameter_neighbor_stability(neighbors[:2])["status"] == (
        "INSUFFICIENT_NEIGHBORS"
    )


def test_public_writer_cannot_overwrite_frozen_ibkr_artifact() -> None:
    with pytest.raises(ValueError, match="ARTIFACT_PATH_OUTSIDE_RESEARCH_OUTPUT"):
        _publish("../../ibkr/phase1-freeze-status.json", {"status": "BAD"})


def test_frozen_public_artifact_is_idempotent_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autopilot_service, "PROJECT_ROOT", tmp_path)
    first = _publish_immutable("frozen/test.json", {"status": "GO", "value": 1})
    second = _publish_immutable("frozen/test.json", {"status": "GO", "value": 1})
    assert first == second
    with pytest.raises(ValueError, match="FROZEN_ARTIFACT_IMMUTABILITY_CONFLICT"):
        _publish_immutable("frozen/test.json", {"status": "GO", "value": 2})


def _decision_result(
    *,
    status: str = "COMPLETE",
    net: float = 0.20,
    doubled: bool = False,
) -> SimpleNamespace:
    metrics = {
        "net_total_return": net if not doubled else net / 2,
        "trade_episodes": 50,
        "maximum_drawdown": -0.20,
        "benchmark_total_return": 0.10,
        "single_security_positive_contribution_share": 0.30,
        "single_year_positive_return_share": 0.30,
        "maximum_exposure": 1.0,
        "maximum_position_weight": 0.10,
        "average_cash": 0.10,
        "excess_total_return": 0.10,
        "period_profit_factor": 1.20,
        "robustness_statistics": {
            "sample_status": "EVALUABLE",
            "deflated_sharpe_probability": 0.75,
        },
        "parameter_neighbor_stability": {"status": "GO"},
        "cohort_stability": {"status": "GO"},
        "concentration_stress": {"status": "GO"},
        "multiple_testing": {"status": "GO", "PBO": 0.25},
    }
    result = SimpleNamespace(status=status, metrics=metrics)
    result.public_payload = lambda: {"status": status, "metrics": metrics}
    return result


def test_data_block_does_not_financially_reject_strategy(tmp_path: Path) -> None:
    ledger = ResearchLedger(AutopilotLayout(tmp_path))
    try:
        strategy = generate_strategies(budget=1)[0]
        ledger.register_strategies([strategy])
        ledger.append_decision(
            strategy_id=strategy.strategy_id,
            new_status=StrategyStatus.SMOKE_PASS,
            research_level=ResearchLevel.NO_CLASSIFICATION,
            reasons=["SMOKE"],
            evidence={"fixture": True},
        )
        blocked = _decision_result(status="BLOCKED:PIT_ELIGIBILITY_UNAVAILABLE")
        _decide(ledger, strategy, [blocked, blocked], [])
        decision = ledger.latest_decision(strategy.strategy_id)
        assert decision is not None
        assert decision["new_status"] == StrategyStatus.SMOKE_PASS
    finally:
        ledger.close()


def test_full_gate_can_only_promote_to_forward_candidate_not_authority(
    tmp_path: Path,
) -> None:
    ledger = ResearchLedger(AutopilotLayout(tmp_path))
    try:
        strategy = generate_strategies(budget=1)[0]
        ledger.register_strategies([strategy])
        normal = _decision_result()
        doubled = _decision_result(doubled=True)
        folds = [
            (f"OOS-{index:02d}", _decision_result(net=0.05))
            for index in range(1, 6)
        ]
        _decide(ledger, strategy, [normal, doubled], folds)
        decision = ledger.latest_decision(strategy.strategy_id)
        assert decision is not None
        assert decision["new_status"] == StrategyStatus.FROZEN_SHADOW
        assert decision["research_level"] == (
            ResearchLevel.FORWARD_OBSERVER_CANDIDATE
        )
        registration = ledger.register_forward(strategy.strategy_id)
        assert registration["execution_authority"] == "NONE"
        assert registration["strategy_authority"] == "NONE"
    finally:
        ledger.close()


def test_hierarchical_sector_weights_allocate_group_first() -> None:
    index = pd.date_range("2026-01-01", periods=1, tz="UTC")
    scores = pd.DataFrame(
        {"A": [3.0], "B": [1.0], "C": [2.0]},
        index=index,
    )
    selected = scores.notna()
    metadata = {
        "A": {"sector": "TECH"},
        "B": {"sector": "TECH"},
        "C": {"sector": "HEALTH"},
    }
    weights = hierarchical_group_weights(
        scores,
        selected,
        metadata=metadata,
        group_field="sector",
    )
    assert weights.loc[index[0], ["A", "B"]].sum() == pytest.approx(0.5)
    assert weights.loc[index[0], "C"] == pytest.approx(0.5)
    assert weights.loc[index[0]].sum() == pytest.approx(1.0)


def test_hierarchical_weights_fail_closed_without_metadata() -> None:
    index = pd.date_range("2026-01-01", periods=1, tz="UTC")
    scores = pd.DataFrame({"A": [1.0]}, index=index)
    with pytest.raises(ValueError, match="HIERARCHICAL_METADATA_REQUIRED"):
        hierarchical_group_weights(
            scores,
            scores.notna(),
            metadata={},
            group_field="region",
        )


def test_simple_benchmarks_are_costed_and_report_all_required_models() -> None:
    bars, eligible = deterministic_fixture(symbols=4, periods=400)
    close = pd.DataFrame(
        {symbol: frame["close"] for symbol, frame in bars.items()}
    )
    returns = close.pct_change(fill_method=None).fillna(0.0)
    _, normal = run_simple_benchmarks(
        close,
        returns,
        eligible,
        cost_profile="NORMAL",
        periods_per_year=252,
    )
    _, stress = run_simple_benchmarks(
        close,
        returns,
        eligible,
        cost_profile="DOUBLE",
        periods_per_year=252,
    )
    assert set(normal["results"]) == {
        "equal_weight",
        "inverse_volatility",
        "trend_200d",
        "momentum_rotation",
        "world_buy_and_hold",
    }
    assert normal["champion"] == "equal_weight"
    assert normal["results"]["world_buy_and_hold"]["available"] is False
    assert stress["results"]["equal_weight"]["total_return"] <= (
        normal["results"]["equal_weight"]["total_return"]
    )
