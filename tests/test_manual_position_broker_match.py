from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import main

from stocks.domain.assets import IbkrSecurityType
from stocks.ibkr.contract_cache import (
    ContractCacheLayout,
    ContractCacheRow,
    write_contract_cache_rows,
)
from stocks.ibkr.contracts import ResolvedContract
from stocks.operations.service import positions_command
from stocks.signals.storage import SignalStore


def test_manual_position_broker_match_is_exact_private_and_read_only(
    tmp_path: Path,
) -> None:
    position_id = _registered_position(tmp_path, quantity="2")
    _write_snapshot(
        tmp_path,
        environment="paper",
        positions=[
            {
                "con_id": 265598,
                "position_quantity": "2",
                "average_cost": "100.25",
                "account_fingerprint": "private-account-fingerprint",
            }
        ],
    )

    result = positions_command(
        tmp_path,
        "broker-match",
        position_id=position_id,
        environment="paper",
    )

    assert result["status"] == "GO"
    assert result["broker_match_status"] == "MATCHED"
    assert result["quantity_match"] is True
    assert result["automatic_execution_eligible"] is False
    assert result["execution_authority"] == "NONE"
    assert result["broker_write_calls"] == 0
    public = json.loads(
        (
            tmp_path
            / "output"
            / "operations"
            / "manual-position-broker-match.json"
        ).read_text(encoding="utf-8")
    )
    forbidden = {
        "quantity",
        "position_quantity",
        "fill_price",
        "average_cost",
        "con_id",
        "account_fingerprint",
    }
    assert forbidden.isdisjoint(_all_keys(public))
    assert "private-account-fingerprint" not in json.dumps(public)
    with SignalStore(tmp_path) as store:
        stored = store.manual_position(position_id)
    assert stored is not None
    assert stored["broker_match_status"] == "MATCHED"
    assert stored["automatic_execution_eligible"] == 0
    assert stored["broker_snapshot_hash"] == "snapshot-hash"


def test_manual_position_quantity_mismatch_is_not_adopted(
    tmp_path: Path,
) -> None:
    position_id = _registered_position(tmp_path, quantity="2")
    _write_snapshot(
        tmp_path,
        environment="paper",
        positions=[
            {
                "con_id": 265598,
                "position_quantity": "3",
            }
        ],
    )

    result = positions_command(
        tmp_path,
        "broker-match",
        position_id=position_id,
        environment="paper",
    )

    assert result["status"] == "NO_GO"
    assert result["broker_match_status"] == "QUANTITY_MISMATCH"
    assert result["quantity_match"] is False
    assert result["automatic_execution_eligible"] is False


def test_manual_position_match_blocks_missing_stale_and_incomplete_state(
    tmp_path: Path,
) -> None:
    missing_identity = _registered_position(
        tmp_path,
        quantity="1",
        signal_id="SIG-MISSING-CONTRACT",
        con_id=None,
    )
    missing = positions_command(
        tmp_path,
        "broker-match",
        position_id=missing_identity,
        environment="paper",
    )
    assert missing["broker_match_status"] == "MISSING_CONTRACT_IDENTITY"

    stale_root = tmp_path / "stale"
    stale_position = _registered_position(stale_root, quantity="1")
    _write_snapshot(
        stale_root,
        environment="paper",
        positions=[{"con_id": 265598, "position_quantity": "1"}],
        completed_at=datetime.now(UTC) - timedelta(minutes=10),
    )
    stale = positions_command(
        stale_root,
        "broker-match",
        position_id=stale_position,
        environment="paper",
    )
    assert stale["broker_match_status"] == "BROKER_SNAPSHOT_STALE"

    incomplete_root = tmp_path / "incomplete"
    incomplete_position = _registered_position(
        incomplete_root,
        quantity="1",
    )
    _write_snapshot(
        incomplete_root,
        environment="paper",
        positions=[{"con_id": 265598, "position_quantity": "1"}],
        component_status="TIMEOUT",
    )
    incomplete = positions_command(
        incomplete_root,
        "broker-match",
        position_id=incomplete_position,
        environment="paper",
    )
    assert incomplete["broker_match_status"] == (
        "BROKER_POSITION_SNAPSHOT_INCOMPLETE"
    )


def test_live_manual_position_never_falls_back_to_paper_snapshot(
    tmp_path: Path,
) -> None:
    position_id = _registered_position(
        tmp_path,
        quantity="1",
        environment="live",
    )
    _write_snapshot(
        tmp_path,
        environment="paper",
        positions=[{"con_id": 265598, "position_quantity": "1"}],
    )

    result = positions_command(
        tmp_path,
        "broker-match",
        position_id=position_id,
        environment="live",
    )

    assert result["status"] == "NO_GO"
    assert result["broker_match_status"] == "BROKER_SNAPSHOT_UNAVAILABLE"


def test_manual_position_match_blocks_absent_ambiguous_and_malformed_rows(
    tmp_path: Path,
) -> None:
    absent_root = tmp_path / "absent"
    absent_position = _registered_position(absent_root, quantity="1")
    _write_snapshot(
        absent_root,
        environment="paper",
        positions=[
            {"con_id": "invalid", "position_quantity": "1"},
            {"con_id": 999, "position_quantity": "1"},
        ],
    )
    absent = positions_command(
        absent_root,
        "broker-match",
        position_id=absent_position,
        environment="paper",
    )
    assert absent["broker_match_status"] == "BROKER_POSITION_NOT_FOUND"

    ambiguous_root = tmp_path / "ambiguous"
    ambiguous_position = _registered_position(
        ambiguous_root,
        quantity="1",
    )
    _write_snapshot(
        ambiguous_root,
        environment="paper",
        positions=[
            {"con_id": 265598, "position_quantity": "1"},
            {"con_id": 265598, "position_quantity": "1"},
        ],
    )
    ambiguous = positions_command(
        ambiguous_root,
        "broker-match",
        position_id=ambiguous_position,
        environment="paper",
    )
    assert ambiguous["broker_match_status"] == (
        "AMBIGUOUS_BROKER_POSITION_BLOCKED"
    )
    assert ambiguous["automatic_execution_eligible"] is False


def test_claimed_matched_position_still_requires_execution_authority(
    tmp_path: Path,
) -> None:
    position_id = _registered_position(tmp_path, quantity="1")
    _write_snapshot(
        tmp_path,
        environment="paper",
        positions=[{"con_id": 265598, "position_quantity": "1"}],
    )
    matched = positions_command(
        tmp_path,
        "broker-match",
        position_id=position_id,
        environment="paper",
    )
    claimed = positions_command(
        tmp_path,
        "claim",
        position_id=position_id,
        ownership_mode="bot-managed",
        confirmed=True,
    )

    assert matched["broker_match_status"] == "MATCHED"
    assert claimed["management_status"] == (
        "BOT_MANAGED_BROKER_MATCHED_AUTHORITY_REQUIRED"
    )
    assert claimed["automatic_execution_eligible"] is False
    assert claimed["execution_authority"] == "NONE"
    assert claimed["orders_generated"] == 0


def test_registration_resolves_exact_contract_cache_identity(
    tmp_path: Path,
) -> None:
    write_contract_cache_rows(
        ContractCacheLayout.from_project_root(tmp_path),
        [_cached_stock_row()],
    )
    _write_signal(
        tmp_path,
        signal_id="SIG-RESOLVED",
        contract_identity={"con_id": 265598},
    )

    result = positions_command(
        tmp_path,
        "register-manual",
        signal_id="SIG-RESOLVED",
        quantity=Decimal("1"),
        fill_price=Decimal("100"),
        environment="paper",
    )

    assert result["status"] == "GO"
    assert "con_id" not in result
    assert "contract_hash" not in result
    with SignalStore(tmp_path) as store:
        stored = store.manual_position(str(result["position_id"]))
    assert stored is not None
    assert stored["con_id"] == 265598
    assert stored["contract_hash"]
    assert stored["automatic_execution_eligible"] == 0


def test_manual_position_broker_match_cli_requires_identity_and_environment() -> None:
    args = main.build_parser().parse_args(
        [
            "positions",
            "broker-match",
            "--position-id",
            "MPOS-TEST",
            "--environment",
            "live",
        ]
    )

    assert args.positions_command == "broker-match"
    assert args.position_id == "MPOS-TEST"
    assert args.environment == "live"


def _registered_position(
    root: Path,
    *,
    quantity: str,
    signal_id: str = "SIG-BROKER-MATCH",
    environment: str = "paper",
    con_id: int | None = 265598,
) -> str:
    _write_signal(root, signal_id=signal_id)
    with SignalStore(root) as store:
        position = store.register_manual_position(
            signal_id=signal_id,
            environment=environment,
            con_id=con_id,
            contract_hash="contract-hash" if con_id else None,
            quantity=quantity,
            fill_price="100",
            payload={"private": True},
        )
    return str(position["position_id"])


def _write_signal(
    root: Path,
    *,
    signal_id: str,
    contract_identity: dict[str, Any] | None = None,
) -> None:
    with SignalStore(root) as store:
        store.append_signal(
            {
                "signal_id": signal_id,
                "strategy_id": "STRATEGY-1",
                "ticker": "AAPL",
                "action": "BUY",
                "lifecycle_status": "ACTIVE",
                "preferred_entry": "100",
                "contract_identity": contract_identity or {},
            }
        )


def _write_snapshot(
    root: Path,
    *,
    environment: str,
    positions: list[dict[str, Any]],
    completed_at: datetime | None = None,
    component_status: str = "COMPLETE",
) -> None:
    if environment == "paper":
        path = (
            root
            / "data"
            / "broker"
            / "phase8"
            / "private"
            / "broker_observation.sqlite3"
        )
    else:
        path = (
            root
            / "data"
            / "execution"
            / "live"
            / "private"
            / "broker_observation.sqlite3"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "snapshot_completed_at": (
            completed_at or datetime.now(UTC)
        ).isoformat(),
        "positions": {
            "status": component_status,
            "positions": positions,
        },
    }
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE snapshots (
                snapshot_id TEXT PRIMARY KEY,
                snapshot_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO snapshots
            (snapshot_id, snapshot_hash, payload_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                "snapshot-1",
                "snapshot-hash",
                json.dumps(payload),
                datetime.now(UTC).isoformat(),
            ),
        )


def _cached_stock_row() -> ContractCacheRow:
    return ContractCacheRow(
        contract=ResolvedContract(
            con_id=265598,
            symbol="AAPL",
            local_symbol="AAPL",
            security_type=IbkrSecurityType.STK,
            exchange="SMART",
            primary_exchange="NASDAQ",
            currency="USD",
            trading_class="NMS",
            min_tick=Decimal("0.01"),
            valid_exchanges="SMART,NASDAQ",
            market_rule_ids="26",
            long_name="Apple Inc.",
            time_zone_id="US/Eastern",
            trading_hours="20260730:0930-1600",
            liquid_hours="20260730:0930-1600",
        ),
        resolved_at=datetime(2099, 1, 1, tzinfo=UTC),
        server_version=225,
    )


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()
