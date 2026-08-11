from __future__ import annotations

import json

import main
from stocks.research.phase6_diagnostics import (
    _pf_zero_reason,
    _pf_diagnostics,
    phase6_1_schema,
    phase6_1_status,
)


def test_phase6_1_schema_blocks_optimizer_and_broker_authority() -> None:
    schema = phase6_1_schema()

    assert schema["schema"] == "phase6_1_robustness_failure_attribution_schema_v1"
    assert "PROMISING_RESEARCH_CANDIDATE" in schema["decision_statuses"]
    assert schema["authority"]["optimizer_enabled"] is False
    assert schema["authority"]["paper_orders_enabled"] is False
    assert schema["authority"]["broker_calls_enabled"] is False
    assert schema["financial_calls"] == {"place_order": 0, "cancel_order": 0, "global_cancel": 0}


def test_pf_diagnostics_distinguishes_no_trades_from_all_losses() -> None:
    no_trades = _pf_diagnostics([], sample_name="episode")
    all_losses = _pf_diagnostics([-1.0, -2.0, -0.5] * 4, sample_name="episode")
    no_losses = _pf_diagnostics([1.0, 0.5, 0.25] * 4, sample_name="episode")

    assert no_trades["profit_factor"] is None
    assert no_trades["diagnostic_reason"] == "NO_TRADES"
    assert all_losses["profit_factor"] == 0
    assert all_losses["diagnostic_reason"] == "ALL_TRADES_LOSS"
    assert no_losses["profit_factor"] is None
    assert no_losses["diagnostic_reason"] == "ZERO_DENOMINATOR"


def test_pf_zero_reason_prioritizes_no_trades_over_period_loss() -> None:
    detailed = {
        "daily": [{"net_return": -0.001}, {"net_return": 0.0}],
        "episodes": [],
    }

    assert _pf_zero_reason(detailed) == "NO_TRADES"


def test_phase6_1_status_reports_no_go_without_artifacts(tmp_path) -> None:
    status = phase6_1_status(tmp_path)

    assert status["status"] == "NO_GO"
    assert status["decision_status"] == "METRIC_IMPLEMENTATION_BLOCKED"
    assert status["financial_calls"] == {"place_order": 0, "cancel_order": 0, "global_cancel": 0}


def test_phase6_1_schema_cli_reports_offline_contract(capsys) -> None:
    exit_code = main.main(["research", "phase6-1", "schema"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["schema"] == "phase6_1_robustness_failure_attribution_schema_v1"
    assert payload["authority"]["provider_calls_enabled"] is False
