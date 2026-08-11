from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.reconciliation.storage import BrokerObservationStore
from stocks.live.evidence import live_level_two_evidence
from stocks.live.models import ManualLiveBracketIntent
from stocks.live.store import LiveExecutionStore


def test_five_ledger_bound_level_one_round_trips_unlock_evidence(
    tmp_path: Path,
) -> None:
    _write_level_one_round_trips(tmp_path, count=5)

    report = live_level_two_evidence(tmp_path)

    assert report["status"] == "GO"
    assert report["verified_round_trip_count"] == 5
    assert report["verified_execution_count"] == 10
    assert report["blockers"] == []
    assert report["broker_calls"] == 0
    assert report["changes_authority"] is False


def test_four_round_trips_remain_below_level_two_threshold(
    tmp_path: Path,
) -> None:
    _write_level_one_round_trips(tmp_path, count=4)

    report = live_level_two_evidence(tmp_path)

    assert report["status"] == "NO_GO"
    assert report["verified_round_trip_count"] == 4
    assert (
        "MINIMUM_LIVE_LEVEL_ONE_ROUND_TRIPS_NOT_REACHED"
        in report["blockers"]
    )


def test_unallocated_broker_execution_cannot_be_used_as_promotion_evidence(
    tmp_path: Path,
) -> None:
    _write_level_one_round_trips(tmp_path, count=5, unknown_order=True)

    report = live_level_two_evidence(tmp_path)

    assert report["status"] == "NO_GO"
    assert "UNKNOWN_LIVE_EXECUTION_BLOCKED" in report["blockers"]
    assert report["verified_round_trip_count"] == 4


def _write_level_one_round_trips(
    root: Path,
    *,
    count: int,
    unknown_order: bool = False,
) -> None:
    live_store = LiveExecutionStore.from_project_root(root)
    live_store.initialize()
    base = datetime(2026, 8, 1, 14, 0, tzinfo=UTC)
    authority = root / "data/execution/live/private/authority-state.json"
    authority.write_text(
        json.dumps({"activated_at": (base - timedelta(minutes=1)).isoformat()}),
        encoding="utf-8",
    )
    executions = []
    commissions = []
    for index in range(count):
        intent = ManualLiveBracketIntent(
            intent_id=f"LIVE-CANARY-{index}",
            economic_order_key=f"KEY-{index}",
            created_at=(base + timedelta(minutes=index * 2)).isoformat(),
            expires_at=(base + timedelta(hours=1)).isoformat(),
            account_fingerprint="FINGERPRINT",
            con_id=1000 + index,
            symbol=f"T{index}",
            security_type="STK",
            currency="EUR",
            exchange="SMART",
            quantity=Decimal("1"),
            entry_limit_price=Decimal("9"),
            stop_price=Decimal("8"),
            take_profit_price=Decimal("10"),
            fx_rate_to_eur=Decimal("1"),
            estimated_notional_eur=Decimal("9"),
            maximum_planned_loss_eur=Decimal("1"),
            session_date="2026-08-01",
            operator_reason="level-one canary",
            contract_hash=f"CONTRACT-{index}",
            strategy_id="TEST-L1",
            target_id=f"TARGET-{index}",
            asset_class="STOCK",
            desired_qty=Decimal("2"),
            normal_allowed_qty=Decimal("2"),
            canary_qty=Decimal("1"),
            risk_per_share_eur=Decimal("1"),
            planned_total_risk_eur=Decimal("1"),
            portfolio_weight=Decimal("0.01"),
            cash_before_eur=Decimal("1870"),
            cash_after_eur=Decimal("1860"),
            estimated_total_cost_eur=Decimal("0.2"),
            expected_net_opportunity_eur=Decimal("0.8"),
            canary_notional_hard_cap_eur=Decimal("250"),
            sizing_reason="CANARY_DOWNSCALED_TO_ONE_SHARE",
            downscaled_for_canary=True,
            fractional_allowed=False,
        )
        assert live_store.register_intent(intent.jsonable()) == (
            "INTENT_REGISTERED"
        )
        parent = 100 + index * 3
        for order_id in (parent, parent + 1, parent + 2):
            assert live_store.allocate_order_id(order_id, intent.intent_id)[0] == (
                "ORDER_ID_READY"
            )
            assert live_store.mark_order_id_used(order_id) == "ORDER_ID_READY"
        live_store.append_event(
            intent.intent_id,
            "LIVE_BRACKET_PLACE_ORDER_CALLED_ONCE",
            {"order_count": 3},
        )
        buy_id = f"EXEC-BUY-{index}"
        sell_id = f"EXEC-SELL-{index}"
        buy_order = parent
        if unknown_order and index == count - 1:
            buy_order = 999999
        executions.extend(
            (
                _execution(
                    buy_id,
                    buy_order,
                    intent,
                    "BOT",
                    base + timedelta(minutes=index * 2),
                ),
                _execution(
                    sell_id,
                    parent + 1,
                    intent,
                    "SLD",
                    base + timedelta(minutes=index * 2 + 1),
                ),
            )
        )
        commissions.extend(
            (
                {"execution_id": buy_id, "commission": "0.1"},
                {"execution_id": sell_id, "commission": "0.1"},
            )
        )

    payload = {
        "snapshot_id": "SNAP-L2-EVIDENCE",
        "snapshot_completed_at": (base + timedelta(hours=1)).isoformat(),
        "executions": {
            "status": "COMPLETE",
            "execution_history_complete": True,
            "executions": executions,
            "commissions": commissions,
        },
    }
    snapshot_hash = stable_hash(payload)
    observation_path = (
        root / "data/execution/live/private/broker_observation.sqlite3"
    )
    BrokerObservationStore(observation_path).initialize()
    with sqlite3.connect(observation_path) as connection:
        connection.execute(
            "INSERT INTO snapshots(snapshot_id, snapshot_hash, payload_json, "
            "created_at) VALUES (?, ?, ?, ?)",
            (
                "SNAP-L2-EVIDENCE",
                snapshot_hash,
                json.dumps(payload),
                payload["snapshot_completed_at"],
            ),
        )
        connection.commit()
    reconciliation = {
        "schema": "ibkr_live_reconciliation_v1",
        "status": "GO",
        "reconciliation_status": "LIVE_RECONCILED_EMPTY",
        "unknown_orders": 0,
        "unknown_positions": 0,
        "private_snapshot_hash": snapshot_hash,
    }
    reconciliation["content_hash"] = stable_hash(reconciliation)
    path = root / "output/ibkr/live/reconciliation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reconciliation), encoding="utf-8")


def _execution(
    execution_id: str,
    order_id: int,
    intent: ManualLiveBracketIntent,
    side: str,
    timestamp: datetime,
) -> dict:
    return {
        "execution_id": execution_id,
        "broker_order_id": stable_hash({"broker_order_id": order_id})[:24],
        "account_fingerprint": "ACCOUNT-HASH",
        "con_id": intent.con_id,
        "symbol": intent.symbol,
        "security_type": "STK",
        "side": side,
        "quantity": "1",
        "price": "9",
        "execution_time": timestamp.isoformat(),
        "observed_at": timestamp.isoformat(),
    }
