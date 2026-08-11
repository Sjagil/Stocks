from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from stocks.operations.paper_runtime import (
    PaperRuntimeStore,
    database_path,
    plan_paper_cycle,
)
from stocks.operations.paper_writer import (
    AUTO_PAPER_AUTHORITY,
    execute_automatic_paper_cycle,
    validate_automatic_paper_plan,
)


def test_entry_plan_requires_authority_and_allowlist(
    tmp_path: Path,
) -> None:
    plan_paper_cycle(
        tmp_path,
        dynamic=_dynamic("BUY"),
        lifecycle=_lifecycle("FRESH_ENTRY"),
        execution_authority=AUTO_PAPER_AUTHORITY,
    )
    calls = []

    def submitter(
        _root: Path, _env: str | Path, plan: dict
    ) -> dict:
        calls.append(plan)
        return {"status": "GO", "paper_place_order_calls": 1}

    blocked = execute_automatic_paper_cycle(
        tmp_path,
        execution_authority="NONE",
        preflight={"status": "GO", "blockers": []},
        submitter=submitter,
    )
    assert blocked["status"] == "NO_GO"
    assert blocked["blocker"] == "AUTHORITY_NOT_GRANTED"
    assert calls == []

    _allowlist(tmp_path, "STRATEGY-1")
    accepted = execute_automatic_paper_cycle(
        tmp_path,
        execution_authority=AUTO_PAPER_AUTHORITY,
        preflight={"status": "GO", "blockers": []},
        submitter=submitter,
    )
    assert accepted["status"] == "GO"
    assert accepted["paper_place_order_calls"] == 1
    assert len(calls) == 1


def test_blocked_preflight_never_calls_submitter(tmp_path: Path) -> None:
    calls = []

    result = execute_automatic_paper_cycle(
        tmp_path,
        execution_authority=AUTO_PAPER_AUTHORITY,
        preflight={"status": "NO_GO", "blockers": ["PIT_BLOCKED"]},
        submitter=lambda *_args: calls.append(True),
    )

    assert result["status"] == "NO_GO"
    assert result["blocker"] == "AUTOMATIC_PAPER_PREFLIGHT_BLOCKED"
    assert result["paper_place_order_calls"] == 0
    assert calls == []


def test_exit_plan_is_created_only_for_reconciled_long(
    tmp_path: Path,
) -> None:
    no_position = plan_paper_cycle(
        tmp_path,
        dynamic=_dynamic("SELL"),
        lifecycle=_lifecycle("EXIT"),
        execution_authority=AUTO_PAPER_AUTHORITY,
        position_quantities={},
    )
    assert no_position["plans"][0]["blocker"] == (
        "NO_RECONCILED_LONG_POSITION"
    )

    with_position = plan_paper_cycle(
        tmp_path,
        dynamic=_dynamic("SELL", signal_id="SIG-EXIT-2"),
        lifecycle=_lifecycle("EXIT"),
        execution_authority=AUTO_PAPER_AUTHORITY,
        position_quantities={123: 1},
    )
    exit_plan = with_position["plans"][0]
    assert exit_plan["side"] == "SELL"
    assert exit_plan["proposal_status"] == "VALID_RISK_REDUCING_EXIT"
    assert exit_plan["target_quantity"] == 1
    assert (
        PaperRuntimeStore(database_path(tmp_path)).counts()["exit_plans"]
        == 1
    )


def test_plan_validation_blocks_fractional_short_and_unlisted() -> None:
    plan = _plan()
    assert validate_automatic_paper_plan(
        plan,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={},
        now=datetime.now(UTC),
    )["status"] == "GO"

    fractional = {**plan, "target_quantity": "0.5"}
    assert validate_automatic_paper_plan(
        fractional,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={},
        now=datetime.now(UTC),
    )["reason"] == "WHOLE_SHARE_QUANTITY_BLOCKED"

    unlisted = validate_automatic_paper_plan(
        plan,
        allowed_strategies=set(),
        position_quantities={},
        now=datetime.now(UTC),
    )
    assert unlisted["reason"] == "STRATEGY_NOT_ALLOWLISTED"

    sell = {
        **plan,
        "side": "SELL",
        "proposal_status": "VALID_RISK_REDUCING_EXIT",
    }
    assert validate_automatic_paper_plan(
        sell,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={},
        now=datetime.now(UTC),
    )["reason"] == "SELL_WITHOUT_RECONCILED_POSITION"


def test_plan_validation_enforces_config_cap_freshness_and_position() -> None:
    plan = _plan()
    now = datetime.now(UTC)
    over_cap = {**plan, "required_cash_eur": "51"}
    assert validate_automatic_paper_plan(
        over_cap,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={},
        max_order_notional_eur=Decimal("50"),
        now=now,
    )["reason"] == "ORDER_NOTIONAL_EXCEEDED"

    stale = {**plan, "data_freshness": "STALE"}
    assert validate_automatic_paper_plan(
        stale,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={},
        now=now,
    )["reason"] == "STALE_DATA_BLOCKED"

    expired = {
        **plan,
        "expiration_timestamp": (now - timedelta(seconds=1)).isoformat(),
    }
    assert validate_automatic_paper_plan(
        expired,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={},
        now=now,
    )["reason"] == "PLAN_EXPIRED"

    assert validate_automatic_paper_plan(
        plan,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={999: 1},
        now=now,
    )["reason"] == "MAX_OPEN_POSITIONS_REACHED"


def test_daily_profit_target_blocks_buy_but_not_risk_reducing_sell() -> None:
    plan = _plan()
    now = datetime.now(UTC)
    assert validate_automatic_paper_plan(
        plan,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={},
        new_entries_allowed=False,
        now=now,
    )["reason"] == "DAILY_PROFIT_TARGET_REACHED"
    sell = {
        **plan,
        "side": "SELL",
        "proposal_status": "VALID_RISK_REDUCING_EXIT",
    }
    assert validate_automatic_paper_plan(
        sell,
        allowed_strategies={"STRATEGY-1"},
        position_quantities={123: 1},
        new_entries_allowed=False,
        now=now,
    )["status"] == "GO"


def test_writer_has_no_live_submission_path() -> None:
    import stocks.operations.paper_writer as module

    source = inspect.getsource(module)
    assert ".placeOrder(" not in source
    assert "connect_phase9_writer" in source
    assert "submit_place_order_once" in source
    assert "live_place_order_calls" in source


def _dynamic(action: str, *, signal_id: str = "SIG-1") -> dict:
    return {
        "signals": {
            "signals": [
                {
                    "signal_id": signal_id,
                    "strategy_id": "STRATEGY-1",
                    "ticker": "TEST",
                    "contract_identity": {
                        "con_id": 123,
                        "security_type": "STK",
                    },
                    "asset_class": "STK",
                    "currency": "EUR",
                    "exchange": "SMART",
                    "action": action,
                    "signal_timestamp": datetime.now(UTC).isoformat(),
                    "data_timestamp": datetime.now(UTC).isoformat(),
                    "expiration_timestamp": (
                        datetime.now(UTC) + timedelta(days=1)
                    ).isoformat(),
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
                    "target_quantity": 1,
                    "proposal_status": "VALID_SIGNAL_EXECUTABLE",
                    "required_cash_eur": "50",
                    "required_risk_eur": "5",
                    "available_risk_eur": "5",
                    "sizing_mode": "RISK_SIZED_WHOLE_SHARE",
                }
            ]
        },
    }


def _lifecycle(status: str) -> dict:
    return {
        "rows": [
            {
                "strategy_id": "STRATEGY-1",
                "ticker": "TEST",
                "lifecycle_status": status,
            }
        ]
    }


def _allowlist(project_root: Path, strategy_id: str) -> None:
    path = (
        project_root
        / "output"
        / "operations"
        / "paper-strategy-allowlist.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "GO",
                "strategies": [{"strategy_id": strategy_id}],
            }
        ),
        encoding="utf-8",
    )


def _plan() -> dict:
    return {
        "strategy_id": "STRATEGY-1",
        "broker_submission_status": "READY_AFTER_AUTHORITY",
        "execution_authority": AUTO_PAPER_AUTHORITY,
        "security_type": "STK",
        "order_type": "LIMIT",
        "time_in_force": "DAY",
        "outside_rth": False,
        "data_freshness": "FRESH",
        "expiration_timestamp": (
            datetime.now(UTC) + timedelta(days=1)
        ).isoformat(),
        "side": "BUY",
        "target_quantity": 1,
        "con_id": 123,
        "limit_price": "50",
        "required_cash_eur": "50",
        "proposal_status": "VALID_SIGNAL_EXECUTABLE",
    }
