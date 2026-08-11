from __future__ import annotations

import json

import main
from stocks.research.phase6 import (
    METRIC_FIELDS,
    _equal_weight_targets,
    _portfolio_result,
    dataset_audit,
    phase6_schema,
)


def test_phase6_schema_requires_108_configs_and_no_financial_authority() -> None:
    schema = phase6_schema()

    assert schema["strategy_grid_size"] == 108
    assert schema["execution"]["orders_enabled"] is False
    assert schema["execution"]["provider_calls_enabled"] is False
    assert schema["financial_calls"] == {"place_order": 0, "cancel_order": 0, "global_cancel": 0}
    assert set(METRIC_FIELDS).issubset(set(schema["metric_fields"]))


def test_dataset_audit_reports_common_history_and_point_in_time_universe() -> None:
    dataset = {
        "schema": "phase6_dataset_v1",
        "series": {
            "1": [_record("2020-01-01"), _record("2020-01-02"), _record("2020-01-03")],
            "2": [_record("2020-01-02"), _record("2020-01-03"), _record("2020-01-04")],
        },
        "metadata": {
            "1": _meta("AAA", "USD", "equity", "united_states"),
            "2": _meta("BBB", "EUR", "bond", "europe"),
        },
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
    }

    audit = dataset_audit(dataset)

    assert audit["status"] == "GO"
    assert audit["common_history_universe"]["common_start"] == "2020-01-02"
    assert audit["common_history_universe"]["common_end"] == "2020-01-03"
    assert audit["point_in_time_universe"]["min_count"] == 1
    assert audit["point_in_time_universe"]["max_count"] == 2


def test_equal_weight_targets_keep_previous_weights_between_rebalances() -> None:
    prepared = {
        "dates": ["2020-01-31", "2020-02-03", "2020-02-04"],
        "returns": {"1": [0.0, 0.0, 0.0], "2": [0.0, 0.0, 0.0]},
    }

    targets = _equal_weight_targets(prepared, "monthly")

    assert targets[0] == {"1": 0.5, "2": 0.5}
    assert targets[1] == {"1": 0.5, "2": 0.5}
    assert targets[2] == {"1": 0.5, "2": 0.5}


def test_portfolio_result_reports_common_metrics_and_separate_profit_factors() -> None:
    prepared = {
        "dates": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        "returns": {
            "1": [0.0, 0.02, -0.01, 0.03, -0.02],
            "2": [0.0, 0.0, 0.0001, 0.0001, 0.0001],
        },
        "prices": {
            "1": [100.0, 102.0, 101.0, 104.0, 102.0],
            "2": [100.0, 100.01, 100.02, 100.03, 100.04],
        },
        "metadata": {
            "1": _meta("AAA", "USD", "equity", "united_states"),
            "2": _meta("BIL", "USD", "cash", "united_states"),
        },
        "cash_key": "2",
    }
    weights = [{"1": 1.0}, {"1": 1.0}, {"1": 1.0}, {"2": 1.0}, {"2": 1.0}]

    result = _portfolio_result("fixture", prepared, weights, cost_bps=10.0)

    for field in METRIC_FIELDS:
        assert field in result
    assert result["period_profit_factor"] is not None
    assert "trade_profit_factor" in result
    assert result["closed_position_episodes"] >= 1
    assert result["transaction_costs"] > 0
    assert result["financial_calls"] == {"place_order": 0, "cancel_order": 0, "global_cancel": 0}


def test_phase6_schema_cli_reports_offline_contract(capsys) -> None:
    exit_code = main.main(["research", "phase6", "schema"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["schema"] == "phase6_baselines_strategy_schema_v1"
    assert payload["strategy_grid_size"] == 108


def _record(day: str) -> dict[str, object]:
    return {
        "session_date": day,
        "fx_source": "FIXTURE",
        "cash_distribution": "0",
        "raw_close": "100",
        "split_adjusted_close": "100",
    }


def _meta(symbol: str, currency: str, sleeve: str, region: str) -> dict[str, object]:
    return {
        "instrument_id": symbol,
        "symbol": symbol,
        "con_id": 1,
        "sleeve": sleeve,
        "region": region,
        "currency": currency,
        "base_currency": "EUR",
        "path": "fixture",
    }
