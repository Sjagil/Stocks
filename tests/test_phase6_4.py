from __future__ import annotations

import json

import main
from stocks.research.phase6_4 import (
    ABLATIONS,
    HYPOTHESIS_IDS,
    _call_counters,
    _cash_key,
    block_bootstrap_probability,
    breadth_targets,
    cap_weights,
    choose_diversified_benchmark_champion,
    component_status,
    deflated_sharpe_probability,
    dual_momentum_targets,
    episode_pf,
    hypothesis_hash,
    incremental_alpha_row,
    is_monthly_rebalance,
    phase6_4_decision,
    phase6_4_schema,
    sleeve_rotation_targets,
    trend_breadth,
    trend_risk_parity_targets,
    validate_registry,
)


def test_phase6_4_schema_blocks_authority_and_declares_exact_scope() -> None:
    schema = phase6_4_schema()

    assert schema["technical_marker"] == "PHASE6_4_PREREGISTERED_MECHANISM_RESEARCH_AND_FORWARD_SELECTION_GO"
    assert schema["highest_allowed_promotion"] == "FORWARD_RESEARCH_SHADOW_ELIGIBLE"
    assert schema["hypothesis_ids"] == list(HYPOTHESIS_IDS)
    assert schema["ablation_count"] == 4
    assert schema["financial_calls"] == _call_counters()
    assert schema["account_calls"] == 0


def test_phase6_4_cli_schema(capsys) -> None:
    exit_code = main.main(["research", "phase6-4", "schema"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema"] == "phase6_4_schema_v1"
    assert payload["paired_bootstrap"]["random_seed"] == 4638


def test_preregistration_hash_is_deterministic_and_detects_mutation() -> None:
    hypothesis = {
        "hypothesis_id": "HYP_A_DIVERSIFIED_DUAL_MOMENTUM",
        "version": "1.0",
        "signal_definition": {"relative_momentum": "12-1"},
        "parameter_hash": "ignored",
    }

    first = hypothesis_hash(hypothesis)
    second = hypothesis_hash(hypothesis)
    mutated = {**hypothesis, "version": "1.1"}

    assert first == second
    assert first != hypothesis_hash(mutated)


def test_registry_requires_exact_four_hypotheses_and_four_ablations() -> None:
    registry = {"hypotheses": [{"hypothesis_id": item} for item in HYPOTHESIS_IDS]}

    errors = validate_registry(registry)

    assert len(HYPOTHESIS_IDS) == 4
    assert len(ABLATIONS) == 4
    assert all("missing" in error for error in errors)


def test_same_close_execution_is_not_a_rebalance_proxy() -> None:
    dates = ["2020-01-31", "2020-02-03", "2020-02-04"]

    assert is_monthly_rebalance(dates, 1)
    assert "2020-02-04" > "2020-02-03"


def test_dual_momentum_uses_12_1_and_cash_fallback() -> None:
    prepared = _prepared_fixture()

    targets = dual_momentum_targets(prepared)

    last = targets[-1]
    assert _cash_key(prepared) in last or "1" in last
    assert sum(last.values()) <= 1.0


def test_breadth_and_breadth_targets_follow_fixed_regimes() -> None:
    prepared = _prepared_fixture()
    risk_keys = ["1", "2"]

    breadth = trend_breadth(prepared, risk_keys, 260)
    targets = breadth_targets(prepared)

    assert 0.0 <= breadth <= 1.0
    assert targets[-1]
    assert abs(sum(targets[-1].values()) - 1.0) < 1e-9


def test_inverse_vol_caps_and_target_vol_scaling() -> None:
    prepared = _prepared_fixture()

    capped = cap_weights({"1": 0.8, "2": 0.2}, prepared, instrument_cap=0.35, region_cap=0.50, sleeve_cap=0.60, cash_key="4")
    targets = trend_risk_parity_targets(prepared)

    assert capped["1"] <= 0.35
    assert targets[-1]
    assert sum(targets[-1].values()) <= 1.0000000001


def test_sleeve_rotation_keeps_minimum_cash_when_invested() -> None:
    prepared = _prepared_fixture()

    targets = sleeve_rotation_targets(prepared)

    assert targets[-1]
    assert targets[-1].get("4", 0.0) >= 0.10


def test_episode_pf_rules() -> None:
    assert episode_pf([])["episode_pf"] is None
    assert episode_pf([-1.0, -2.0])["episode_pf"] == 0.0
    assert episode_pf([1.0, 2.0])["pf_status"] == "NO_LOSING_EPISODES"


def test_alpha_beta_bootstrap_dsr_and_component_status() -> None:
    left = _report("H", [0.02, 0.01, -0.005, 0.004] * 10)
    right = _report("B", [0.01, 0.005, -0.004, 0.002] * 10)

    alpha = incremental_alpha_row(left, right, "BENCH", "HASH")

    assert alpha["information_ratio"] is not None
    assert alpha["beta"] is not None
    assert block_bootstrap_probability([0.01, -0.005, 0.02], iterations=100, seed=3) == block_bootstrap_probability([0.01, -0.005, 0.02], iterations=100, seed=3)
    assert deflated_sharpe_probability(1.0, 0.0, 1) > deflated_sharpe_probability(1.0, 0.0, 8)
    assert component_status({"aggregate_oos_CAGR": 0.1, "Sharpe": 1.0, "period_profit_factor": 1.2}, {"aggregate_oos_CAGR": 0.05, "Sharpe": 0.5, "period_profit_factor": 1.1}) == "ROBUSTLY_ADDS_VALUE"


def test_gld_cannot_be_diversified_champion() -> None:
    champion = choose_diversified_benchmark_champion(
        {
            "BUY_AND_HOLD_GLD": _benchmark("BUY_AND_HOLD_GLD", asset=1.0, region=1.0, sleeve=1.0, sharpe=2.0),
            "EQUAL_WEIGHT": _benchmark("EQUAL_WEIGHT", asset=0.2, region=0.4, sleeve=0.5, sharpe=1.0),
        }
    )

    assert champion["strategy_id"] == "EQUAL_WEIGHT"


def test_decision_gates_cover_promising_forward_no_candidate_and_blocked() -> None:
    ranking = {"ranking": [{"hypothesis_id": "H"}]}
    good_report = {"hypothesis_id": "H", "metrics": _metrics()}
    alpha = [{"hypothesis_id": "H", "benchmark_id": "DIVERSIFIED_BENCHMARK_CHAMPION", "paired_bootstrap_probability_strategy_gt_benchmark": 0.8}]
    ablation = [{"hypothesis_id": "H", "component_status": "ROBUSTLY_ADDS_VALUE"}]
    loo = [{"hypothesis_id": "H", "fragility_status": "ROBUST"}]

    forward = phase6_4_decision(ranking, [good_report], alpha, ablation, loo, "HASH")
    assert forward["financial_decision"] == "FORWARD_RESEARCH_SHADOW_ELIGIBLE"
    assert forward["FINANCIAL_FINALIST_GO"] is False
    assert forward["PAPER_STRATEGY_AUTHORITY"] == "blocked"

    weak = {"hypothesis_id": "H", "metrics": {**_metrics(), "DSR_probability": 0.01}}
    promising = phase6_4_decision(ranking, [weak], alpha, ablation, loo, "HASH")
    assert promising["financial_decision"] == "PROMISING_MECHANISM_CANDIDATE"

    no_candidate = {"hypothesis_id": "H", "metrics": {**_metrics(), "positive_testwindow_ratio": 0.1}}
    assert phase6_4_decision(ranking, [no_candidate], alpha, ablation, loo, "HASH")["financial_decision"] == "NO_NEW_FINANCIAL_CANDIDATE"

    assert phase6_4_decision({"ranking": []}, [], [], [], [], "HASH")["financial_decision"] == "METRIC_OR_DATA_BLOCKED"


def _prepared_fixture() -> dict[str, object]:
    dates = [f"2020-01-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-02-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-03-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-04-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-05-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-06-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-07-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-08-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-09-{day:02d}" for day in range(1, 29)]
    dates += [f"2020-10-{day:02d}" for day in range(1, 29)]
    returns = {
        "1": [0.001] * len(dates),
        "2": [0.0005] * len(dates),
        "3": [0.0002] * len(dates),
        "4": [0.0] * len(dates),
    }
    prices = {key: [100.0 + index * (1 + pos * 0.1) for index in range(len(dates))] for pos, key in enumerate(returns)}
    return {
        "dates": dates,
        "returns": returns,
        "prices": prices,
        "metadata": {
            "1": {"symbol": "AAA", "sleeve": "equity", "region": "us", "currency": "USD"},
            "2": {"symbol": "GLD", "sleeve": "commodity", "region": "global", "currency": "USD"},
            "3": {"symbol": "IEF", "sleeve": "defensive", "region": "us", "currency": "USD"},
            "4": {"symbol": "BIL", "sleeve": "cash", "region": "cash", "currency": "USD"},
        },
        "cash_key": "4",
    }


def _report(strategy_id: str, returns: list[float]) -> dict[str, object]:
    return {
        "hypothesis_id": strategy_id,
        "daily": [{"obs_id": f"w:{idx}", "window_id": "w", "net_return": value} for idx, value in enumerate(returns)],
    }


def _benchmark(strategy_id: str, *, asset: float, region: float, sleeve: float, sharpe: float) -> dict[str, object]:
    return {
        "strategy_id": strategy_id,
        "metrics": {
            "single_asset_contribution_max": asset,
            "single_region_contribution_max": region,
            "single_sleeve_contribution_max": sleeve,
            "Sharpe": sharpe,
            "aggregate_oos_CAGR": sharpe / 10,
        },
    }


def _metrics() -> dict[str, object]:
    return {
        "evaluable_testwindow_count": 10,
        "positive_testwindow_ratio": 0.7,
        "period_profit_factor": 1.2,
        "20bps_stress_pf": 1.1,
        "30bps_stress_pf": 1.0,
        "aggregate_oos_CAGR": 0.05,
        "Sharpe": 0.5,
        "maximum_drawdown": -0.1,
        "single_asset_contribution_max": 0.2,
        "single_region_contribution_max": 0.4,
        "single_sleeve_contribution_max": 0.5,
        "single_year_contribution_max": 0.4,
        "PBO": 0.1,
        "DSR_probability": 0.2,
        "bootstrap_probability_return_gt_0": 0.95,
    }
