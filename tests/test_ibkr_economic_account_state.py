from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from stocks.ibkr.reconciliation.account_state import (
    derive_economic_account_state,
)


def test_complete_ledger_backed_eur_account_is_execution_ready() -> None:
    now = datetime.now(UTC)
    snapshot = _snapshot(now)

    state = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=now,
    )

    assert state["lifecycle_state"] == "ACCOUNT_READY"
    assert state["research_status"] == "RESEARCH_READY"
    assert state["execution_status"] == "EXECUTION_ACCOUNT_READY"
    assert state["cash_by_currency"] == {"EUR": "1870", "USD": "500"}
    assert state["reporting_value_eur"] == "2370"
    assert state["spendable_eur"] == "1870"
    assert state["eur_available_for_new_longs"] == "1500"
    assert state["buying_power_is_cash"] is False
    assert state["implicit_fx_conversion_assumed"] is False


def test_aggregate_reporting_eur_never_becomes_fictitious_spendable_eur() -> None:
    now = datetime.now(UTC)
    snapshot = _snapshot(now)
    snapshot["account"]["values"] = [
        row
        for row in snapshot["account"]["values"]
        if row["tag"] not in {"CashBalance", "TotalCashBalance"}
    ]

    state = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=now,
    )

    assert state["research_status"] == "RESEARCH_READY"
    assert state["reporting_cash_eur"] == "2000"
    assert state["spendable_eur"] is None
    assert state["execution_status"] == "NO_GO"
    assert "DIRECT_SPENDABLE_EUR_NOT_PROVEN" in state["execution_blockers"]


def test_account_download_must_complete_before_ready() -> None:
    now = datetime.now(UTC)
    snapshot = _snapshot(now)
    snapshot["account"]["status"] = "CALLBACK_TIMEOUT"

    state = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=now,
    )

    assert state["lifecycle_state"] == "ACCOUNT_PARTIAL"
    assert state["execution_status"] == "NO_GO"
    assert "ACCOUNT_DOWNLOAD_INCOMPLETE" in state["execution_blockers"]


def test_complete_account_becomes_stale_independently_of_research() -> None:
    observed = datetime.now(UTC) - timedelta(minutes=5)
    snapshot = _snapshot(observed)

    state = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=observed + timedelta(minutes=5),
        execution_max_age=timedelta(minutes=2),
    )

    assert state["lifecycle_state"] == "ACCOUNT_STALE"
    assert state["research_status"] == "RESEARCH_READY"
    assert state["execution_status"] == "NO_GO"
    assert "ACCOUNT_STATE_STALE" in state["execution_blockers"]
    assert "POSITION_STATE_STALE" in state["execution_blockers"]
    assert "ORDER_STATE_STALE" in state["execution_blockers"]


def test_open_buy_orders_reduce_only_execution_sizing_capacity() -> None:
    now = datetime.now(UTC)
    snapshot = _snapshot(now)
    snapshot["all_api_open_orders"]["open_orders"] = [
        {
            "perm_id": "PERM-1",
            "broker_order_id": "ORDER-1",
            "action": "BUY",
            "order_status": "Submitted",
            "remaining_quantity": "2",
            "limit_price": "100",
            "currency": "EUR",
        }
    ]

    state = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=now,
        fx_to_eur={"EUR": Decimal("1")},
    )

    assert state["open_order_reserved_capital_eur"] == "200"
    assert state["research_sizing_capacity_eur"] == "1500"
    assert state["execution_sizing_capacity_eur"] == "1300"


def test_ibkr_ledger_group_is_not_misclassified_as_second_account() -> None:
    now = datetime.now(UTC)
    snapshot = _snapshot(now)
    snapshot["account"]["values"] = [
        row
        for row in snapshot["account"]["values"]
        if row["tag"] != "CashBalance"
    ]
    snapshot["account"]["values"].extend(
        [
            {
                "account_fingerprint": "HASHED-ALL-LEDGER-GROUP",
                "tag": "$LEDGER-CashBalance",
                "value": "1870",
                "currency": "EUR",
                "observed_at": now.isoformat(),
            },
            {
                "account_fingerprint": "HASHED-ALL-LEDGER-GROUP",
                "tag": "$LEDGER-CashBalance",
                "value": "500",
                "currency": "USD",
                "observed_at": now.isoformat(),
            },
        ]
    )

    state = derive_economic_account_state(
        snapshot,
        expected_base_currency="EUR",
        snapshot_hash_verified=True,
        now=now,
    )

    assert state["account_fingerprint_count"] == 1
    assert state["cash_by_currency"] == {"EUR": "1870", "USD": "500"}
    assert state["execution_status"] == "EXECUTION_ACCOUNT_READY"


def _snapshot(observed: datetime) -> dict:
    timestamp = observed.isoformat()
    values = []
    aggregate = {
        "NetLiquidation": "2370",
        "TotalCashValue": "2000",
        "SettledCash": "1870",
        "AvailableFunds": "1500",
        "BuyingPower": "3000",
        "GrossPositionValue": "370",
        "InitMarginReq": "100",
        "MaintMarginReq": "80",
        "ExcessLiquidity": "1600",
    }
    for tag, value in aggregate.items():
        values.append(_value(tag, value, "EUR", timestamp))
    values.extend(
        [
            _value("CashBalance", "1870", "EUR", timestamp),
            _value("CashBalance", "500", "USD", timestamp),
        ]
    )
    component = {
        "started_at": timestamp,
        "completed_at": timestamp,
    }
    return {
        "server_version": "188",
        "account": {"status": "COMPLETE", "values": values},
        "positions": {"status": "EMPTY_COMPLETE", "positions": []},
        "same_client_open_orders": {
            "status": "EMPTY_COMPLETE",
            "open_orders": [],
        },
        "all_api_open_orders": {
            "status": "EMPTY_COMPLETE",
            "open_orders": [],
        },
        "executions": {"status": "EMPTY_COMPLETE", "executions": []},
        "component_timestamps": {
            "accountsummary": component,
            "positions": component,
            "same_client_open_orders": component,
            "all_api_open_orders": component,
            "executions": component,
        },
    }


def _value(tag: str, value: str, currency: str, observed_at: str) -> dict:
    return {
        "account_fingerprint": "HASHED-ACCOUNT",
        "tag": tag,
        "value": value,
        "currency": currency,
        "observed_at": observed_at,
        "request_id": 8101,
        "source": "IBKR_ACCOUNT_SUMMARY",
    }
