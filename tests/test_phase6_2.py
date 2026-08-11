from __future__ import annotations

import json

import main
from stocks.research.phase6_2 import (
    _no_trades_reason,
    _phase6_2_decision,
    cost_reliability_report,
    phase6_2_schema,
    point_in_time_history_report,
)


def test_phase6_2_schema_is_non_authoritative() -> None:
    schema = phase6_2_schema()

    assert schema["schema"] == "phase6_2_sample_sufficiency_forward_oos_schema_v1"
    assert "FINANCIAL_FINALIST_GO" in schema["decision_statuses"]
    assert schema["tracks"]["D_forward_shadow"]["orders_enabled"] is False
    assert schema["tracks"]["D_forward_shadow"]["paper_authority"] is False
    assert schema["history_policy"] == "POINT_IN_TIME_ONLY_NO_RETROACTIVE_UNIVERSE_EXTENSION"
    assert schema["financial_calls"] == {"place_order": 0, "cancel_order": 0, "global_cancel": 0}


def test_no_trades_reason_distinguishes_carry_from_blocked_trend() -> None:
    assert (
        _no_trades_reason(
            entry_count=0,
            exit_count=0,
            carry_in=1,
            carry_out=1,
            signal_opportunities=2,
            blocked_reasons={},
        )
        == "POSITION_CARRIED_INTO_NEXT_FOLD"
    )
    assert (
        _no_trades_reason(
            entry_count=0,
            exit_count=0,
            carry_in=0,
            carry_out=0,
            signal_opportunities=3,
            blocked_reasons={"ALL_SIGNALS_BLOCKED_BY_TREND": 3},
        )
        == "ALL_SIGNALS_BLOCKED_BY_TREND"
    )


def test_phase6_2_decision_blocks_on_sample_gates_before_promising() -> None:
    aggregate = {
        "gates": {
            "aggregate_closed_OOS_episodes_ge_30": False,
            "effective_sample_size_ge_20": True,
            "evaluable_testwindows_ge_8": True,
            "unevaluable_no_trade_windows_le_1": True,
            "aggregate_oos_episode_pf_gt_1_10": True,
            "median_oos_period_pf_gt_1_05": True,
            "stress_20bps_pf_gt_1": True,
            "PBO_lte_0_20": True,
        }
    }

    assert _phase6_2_decision(aggregate, {"status": "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_GO"}) == "INSUFFICIENT_SAMPLE"


def test_phase6_2_schema_cli_reports_contract(capsys) -> None:
    exit_code = main.main(["research", "phase6-2", "schema"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["schema"] == "phase6_2_sample_sufficiency_forward_oos_schema_v1"
    assert payload["tracks"]["B_more_causal_windows"]["parameter_family_size"] == 108


def test_point_in_time_history_report_blocks_retroactive_extension() -> None:
    dataset = {
        "series": {"1": [{"session_date": "2020-01-01"}, {"session_date": "2020-01-02"}]},
        "metadata": {"1": {"symbol": "AAA"}},
    }

    report = point_in_time_history_report(dataset)

    assert report["status"] == "GO"
    assert report["history_expansion_performed"] is False
    assert report["instruments"][0]["retroactive_use_allowed"] is False


def test_cost_reliability_report_contains_conservative_break_even_fields() -> None:
    prepared = {
        "dates": ["2020-01-01", "2020-02-03", "2020-03-02", "2020-04-01"],
        "returns": {"1": [0.0, 0.05, -0.01, 0.03], "2": [0.0, 0.0, 0.0, 0.0]},
        "prices": {"1": [100.0, 105.0, 104.0, 107.0], "2": [100.0, 100.0, 100.0, 100.0]},
        "metadata": {
            "1": {"symbol": "AAA", "sleeve": "equity", "region": "test", "currency": "USD", "instrument_id": "AAA"},
            "2": {"symbol": "BIL", "sleeve": "cash", "region": "cash", "currency": "USD", "instrument_id": "BIL"},
        },
        "cash_key": "2",
    }
    grid = {"best_by_calmar": {"parameters": {"momentum_lookback": 1, "trend_lookback": 1, "rebalance": "monthly", "target_vol": 0.12, "cost_bps": 5.0}}}

    report = cost_reliability_report(prepared, grid)

    assert report["status"] == "GO"
    assert "cumulative_turnover" in report
    assert "closed_episode_count" in report
    assert "P5_break_even_cost_bps_bootstrap" in report
