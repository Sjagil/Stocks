from __future__ import annotations

from pathlib import Path

from stocks.operations.paper_runtime import (
    PaperRuntimeStore,
    database_path,
    plan_paper_cycle,
)


def _dynamic(quantity: int, proposal_status: str) -> dict:
    return {
        "signals": {
            "signals": [
                {
                    "signal_id": "SIG-1",
                    "strategy_id": "STRATEGY-1",
                    "ticker": "TEST",
                    "contract_identity": {
                        "con_id": 123,
                        "security_type": "STK",
                    },
                    "signal_timestamp": "2099-01-01T12:00:00+00:00",
                    "data_timestamp": "2099-01-01T11:00:00+00:00",
                    "expiration_timestamp": "2099-01-02T12:00:00+00:00",
                    "data_freshness": "FRESH",
                    "risk_blockers": [],
                    "limit_entry_price": "50",
                    "stop_loss": "45",
                    "take_profit_1": "57.5",
                    "take_profit_2": "62.5",
                }
            ]
        },
        "portfolio": {
            "candidates": [
                {
                    "ticker": "TEST",
                    "target_quantity": quantity,
                    "proposal_status": proposal_status,
                    "required_cash_eur": "50",
                    "required_risk_eur": "5",
                    "available_risk_eur": "5",
                    "sizing_mode": "RISK_SIZED_WHOLE_SHARE",
                }
            ]
        },
    }


def _lifecycle() -> dict:
    return {
        "rows": [
            {
                "strategy_id": "STRATEGY-1",
                "ticker": "TEST",
                "lifecycle_status": "FRESH_ENTRY",
            }
        ]
    }


def test_executable_plan_is_persistent_and_idempotent(
    tmp_path: Path,
) -> None:
    first = plan_paper_cycle(
        tmp_path,
        dynamic=_dynamic(1, "VALID_SIGNAL_EXECUTABLE"),
        lifecycle=_lifecycle(),
        execution_authority="NONE",
    )
    second = plan_paper_cycle(
        tmp_path,
        dynamic=_dynamic(1, "VALID_SIGNAL_EXECUTABLE"),
        lifecycle=_lifecycle(),
        execution_authority="NONE",
    )
    assert first["executable_plan_count"] == 1
    assert first["plans"][0]["write_status"] == "RECORDED"
    assert second["plans"][0]["write_status"] == "IDEMPOTENT_REPLAY"
    assert PaperRuntimeStore(database_path(tmp_path)).counts()["entry_plans"] == 1
    assert first["paper_place_order_calls"] == 0


def test_below_whole_share_budget_never_becomes_executable(
    tmp_path: Path,
) -> None:
    result = plan_paper_cycle(
        tmp_path,
        dynamic=_dynamic(
            0,
            "VALID_SIGNAL_BELOW_WHOLE_SHARE_BUDGET",
        ),
        lifecycle=_lifecycle(),
        execution_authority="AUTOMATIC_BOUNDED_PAPER",
    )
    assert result["below_whole_share_budget_count"] == 1
    assert result["executable_plan_count"] == 0
    assert result["plans"][0]["execution_authority"] == "NONE"
    assert (
        result["plans"][0]["broker_submission_status"]
        == "NOT_EXECUTABLE"
    )


def test_nonfresh_signal_creates_no_plan(tmp_path: Path) -> None:
    lifecycle = _lifecycle()
    lifecycle["rows"][0]["lifecycle_status"] = (
        "ACTIVE_STATE_NO_NEW_ENTRY"
    )
    result = plan_paper_cycle(
        tmp_path,
        dynamic=_dynamic(1, "VALID_SIGNAL_EXECUTABLE"),
        lifecycle=lifecycle,
        execution_authority="AUTOMATIC_BOUNDED_PAPER",
    )
    assert result["plan_count"] == 0
    assert result["paper_place_order_calls"] == 0


def test_exit_plan_is_idempotent_with_reconciled_position(
    tmp_path: Path,
) -> None:
    dynamic = _dynamic(1, "VALID_SIGNAL_EXECUTABLE")
    dynamic["signals"]["signals"][0]["action"] = "SELL"
    lifecycle = _lifecycle()
    lifecycle["rows"][0]["lifecycle_status"] = "EXIT"
    first = plan_paper_cycle(
        tmp_path,
        dynamic=dynamic,
        lifecycle=lifecycle,
        execution_authority="AUTOMATIC_BOUNDED_PAPER",
        position_quantities={123: 1},
    )
    second = plan_paper_cycle(
        tmp_path,
        dynamic=dynamic,
        lifecycle=lifecycle,
        execution_authority="AUTOMATIC_BOUNDED_PAPER",
        position_quantities={123: 1},
    )

    assert first["plans"][0]["write_status"] == "RECORDED"
    assert second["plans"][0]["write_status"] == "IDEMPOTENT_REPLAY"
    assert first["plans"][0]["side"] == "SELL"
