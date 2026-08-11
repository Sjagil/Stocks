from __future__ import annotations

import json

import main
from stocks.research.phase6_3 import (
    _paired_comparison,
    alpha_beta_regression,
    block_bootstrap_probability,
    deflated_sharpe_probability,
    forward_shadow_spec,
    fragility_classification,
    incremental_alpha_status,
    paired_benchmark_comparisons,
    parameter_plateau,
    phase6_3_decision,
    phase6_3_schema,
    plateau_classification,
    rank_benchmarks,
    upside_downside_capture,
)


def test_phase6_3_schema_is_offline_and_non_authoritative() -> None:
    schema = phase6_3_schema()

    assert schema["schema"] == "phase6_3_benchmark_champion_incremental_alpha_schema_v1"
    assert schema["strategy_status"] == "REJECTED_NO_INCREMENTAL_FINANCIAL_EDGE"
    assert schema["forbidden_calls"]["orders"] is True
    assert schema["financial_calls"] == {
        "financial_calls": 0,
        "order_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def test_phase6_3_schema_cli_reports_contract(capsys) -> None:
    exit_code = main.main(["research", "phase6-3", "schema"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["schema"] == "phase6_3_benchmark_champion_incremental_alpha_schema_v1"
    assert payload["paired_bootstrap"]["seed"] == 4638


def test_ranking_uses_score_and_deterministic_simplicity_tie_break() -> None:
    rows = [
        _ranking_row("EQUAL_WEIGHT_MONTHLY", "EQUAL_WEIGHT"),
        _ranking_row("BUY_AND_HOLD_AAA", "BUY_AND_HOLD"),
    ]

    ranking = rank_benchmarks(rows)

    assert ranking["ranking"][0]["benchmark_id"] == "BUY_AND_HOLD_AAA"
    assert ranking["ranking"][0]["rank"] == 1


def test_ranking_fails_closed_on_missing_metric() -> None:
    good = _ranking_row("GOOD", "EQUAL_WEIGHT", cagr=0.05)
    bad = _ranking_row("BAD", "BUY_AND_HOLD", cagr=None)

    ranking = rank_benchmarks([bad, good])

    assert ranking["ranking"][0]["benchmark_id"] == "GOOD"
    assert ranking["ranking"][1]["fail_closed"] is True
    assert "oos_CAGR" in ranking["ranking"][1]["missing_metrics"]


def test_paired_comparison_aligns_on_observation_id_and_reports_ir() -> None:
    left = _candidate_report("A", [0.02, 0.015, -0.01])
    right = _candidate_report("B", [0.01, 0.00, -0.02])

    row = _paired_comparison(left, right)

    assert row["overlapping_observations"] == 3
    assert row["annualized_active_return"] > 0
    assert row["information_ratio"] is not None
    assert row["paired_bootstrap_probability_a_gt_b"] == block_bootstrap_probability([0.01, 0.015, 0.01])
    assert len(paired_benchmark_comparisons([left, right])) == 1


def test_alpha_beta_regression_and_incremental_statuses() -> None:
    regression = alpha_beta_regression([0.01, 0.02, 0.03, 0.04], [0.0, 0.01, 0.02, 0.03])

    assert regression["beta"] is not None
    assert regression["alpha_daily"] is not None
    assert incremental_alpha_status(
        annualized_active_return=0.04,
        information_ratio=0.4,
        bootstrap_probability=0.96,
        window_win_rate=0.7,
        overlapping_observations=100,
    ) == "POSITIVE_INCREMENTAL_ALPHA"
    assert incremental_alpha_status(
        annualized_active_return=-0.01,
        information_ratio=-0.1,
        bootstrap_probability=0.1,
        window_win_rate=0.2,
        overlapping_observations=100,
    ) == "NEGATIVE_INCREMENTAL_ALPHA"


def test_block_bootstrap_is_deterministic_and_capture_splits_upside_downside() -> None:
    values = [0.01, -0.02, 0.03, 0.01, -0.01]

    first = block_bootstrap_probability(values, iterations=100, block_length=2, seed=7)
    second = block_bootstrap_probability(values, iterations=100, block_length=2, seed=7)
    capture = upside_downside_capture([0.02, -0.01, 0.03], [0.01, -0.02, 0.02])

    assert first == second
    assert capture["upside_capture"] > 1.0
    assert capture["downside_capture"] < 1.0


def test_fragility_and_plateau_classifications() -> None:
    assert fragility_classification(0.10, -0.01) == "DOMINATED_BY_ENTITY"
    assert fragility_classification(0.10, 0.02) == "FRAGILE"
    assert fragility_classification(0.10, 0.09) == "ROBUST"
    assert (
        plateau_classification(
            variant_count=5,
            positive_variant_ratio=0.8,
            dispersion=0.01,
            champion_is_positive=True,
        )
        == "BROAD_PLATEAU"
    )
    assert (
        plateau_classification(
            variant_count=5,
            positive_variant_ratio=0.1,
            dispersion=0.01,
            champion_is_positive=True,
        )
        == "ISOLATED_WINNER"
    )


def test_parameter_plateau_marks_champion_family() -> None:
    reports = [
        {"benchmark_id": "EW1", "family": "EQUAL_WEIGHT", "metrics": _metrics(cagr=0.05, pf=1.2, sharpe=0.6)},
        {"benchmark_id": "EW2", "family": "EQUAL_WEIGHT", "metrics": _metrics(cagr=0.04, pf=1.1, sharpe=0.5)},
    ]

    plateau = parameter_plateau(reports, "EW1")

    assert plateau["champion_family"] == "EQUAL_WEIGHT"
    assert plateau["champion_family_classification"] == "BROAD_PLATEAU"


def test_deflated_sharpe_penalizes_selection_count() -> None:
    low_selection = deflated_sharpe_probability(1.0, 0.0, 1)
    high_selection = deflated_sharpe_probability(1.0, 0.0, 108)

    assert low_selection > high_selection


def test_decision_gates_finalist_promising_no_edge_and_metric_blocked() -> None:
    finalist = _champion_analysis()
    incremental = {"status": "NO_INCREMENTAL_ALPHA"}

    assert phase6_3_decision(finalist, incremental)["decision_status"] == "BENCHMARK_FINANCIAL_FINALIST_GO"

    promising = _champion_analysis(dsr=0.1)
    assert phase6_3_decision(promising, incremental)["decision_status"] == "PROMISING_SIMPLE_CANDIDATE"

    no_edge = _champion_analysis(cagr=-0.01, sharpe=-0.2)
    assert phase6_3_decision(no_edge, incremental)["decision_status"] == "NO_EXISTING_FINANCIAL_EDGE"

    blocked = _champion_analysis(cagr=-0.01, sharpe=-0.2)
    assert phase6_3_decision(blocked, {"status": "METRIC_BLOCKED"})["decision_status"] == "METRIC_OR_DATA_BLOCKED"


def test_forward_shadow_spec_keeps_authority_none(tmp_path) -> None:
    spec = forward_shadow_spec(
        tmp_path,
        {
            "benchmark_id": "EQUAL_WEIGHT_MONTHLY",
            "configuration": {"rebalance": "monthly"},
            "weight_calculation": "fixture",
        },
        {"decision_status": "PROMISING_SIMPLE_CANDIDATE"},
    )

    assert spec["authority"] == "NONE"
    assert spec["orders_allowed"] is False
    assert spec["paper_orders_allowed"] is False
    assert spec["live_orders_allowed"] is False


def _ranking_row(benchmark_id: str, family: str, cagr: float | None = 0.05) -> dict[str, object]:
    return {
        "benchmark_id": benchmark_id,
        "family": family,
        "configuration": "{}",
        "oos_CAGR": cagr,
        "Sharpe": 0.5,
        "Calmar": 0.3,
        "positive_window_ratio": 0.7,
        "stress_20bps_period_pf": 1.1,
        "DSR_probability": 0.2,
        "turnover": 1.0,
        "abs_maximum_drawdown": 0.1,
    }


def _candidate_report(benchmark_id: str, returns: list[float]) -> dict[str, object]:
    return {
        "benchmark_id": benchmark_id,
        "daily": [
            {
                "obs_id": f"w:{index}",
                "window_id": "w",
                "net_return": value,
                "turnover": 0.0,
                "cost": 0.0,
            }
            for index, value in enumerate(returns)
        ],
        "metrics": {
            "maximum_drawdown": -0.1,
            "turnover": 0.0,
            "transaction_costs": 0.0,
        },
    }


def _metrics(cagr: float, pf: float, sharpe: float) -> dict[str, object]:
    return {
        "oos_CAGR": cagr,
        "Sharpe": sharpe,
        "period_profit_factor": pf,
        "maximum_drawdown": -0.1,
        "stress_20bps_period_pf": pf,
    }


def _champion_analysis(cagr: float = 0.08, sharpe: float = 0.7, dsr: float = 0.96) -> dict[str, object]:
    return {
        "champion_benchmark_id": "EQUAL_WEIGHT_MONTHLY",
        "metrics": {
            "testwindow_count": 10,
            "positive_window_ratio": 0.7,
            "period_profit_factor": 1.1,
            "stress_20bps_period_pf": 1.05,
            "stress_30bps_period_pf": 1.0,
            "oos_CAGR": cagr,
            "Sharpe": sharpe,
            "maximum_drawdown": -0.15,
        },
        "dominance": {
            "dominance_blocked": False,
            "dominance_classification": "ROBUST",
        },
        "parameter_plateau": {"classification": "NARROW_PLATEAU"},
        "statistical_validation": {
            "PBO": 0.1,
            "DSR_probability": dsr,
            "bootstrap_probability_total_return_gt_0": 0.96,
        },
        "paired_superiority": {"superiority_ratio": 0.7},
    }
