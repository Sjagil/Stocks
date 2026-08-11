from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from stocks.operations.service import positions_command
from stocks.signals.storage import SignalStore


def test_manual_position_register_claim_and_unclaim_are_fail_closed(
    tmp_path: Path,
) -> None:
    _write_signal(tmp_path)

    registered = positions_command(
        tmp_path,
        "register-manual",
        signal_id="SIG-MANUAL-1",
        quantity=Decimal("2"),
        fill_price=Decimal("101.25"),
    )
    position_id = str(registered["position_id"])

    assert registered["status"] == "GO"
    assert registered["ownership_status"] == "MANUAL_TRACKED"
    assert registered["broker_match_status"] == "UNVERIFIED"
    assert registered["automatic_execution_eligible"] is False
    assert registered["execution_authority"] == "NONE"
    public = json.loads(
        (
            tmp_path
            / "output"
            / "operations"
            / "manual-position-registration.json"
        ).read_text(encoding="utf-8")
    )
    assert "quantity" not in public
    assert "fill_price" not in public

    blocked = positions_command(
        tmp_path,
        "claim",
        position_id=position_id,
        ownership_mode="bot-managed",
        confirmed=False,
    )
    assert blocked["reason"] == "EXPLICIT_YES_CONFIRMATION_REQUIRED"

    claimed = positions_command(
        tmp_path,
        "claim",
        position_id=position_id,
        ownership_mode="bot-managed",
        confirmed=True,
    )
    assert claimed["ownership_status"] == "BOT_MANAGED"
    assert claimed["management_status"] == (
        "BOT_MANAGED_PENDING_BROKER_MATCH"
    )
    assert claimed["automatic_execution_eligible"] is False
    assert claimed["automatic_position_adoption"] is False
    assert claimed["execution_authority"] == "NONE"

    unclaimed = positions_command(
        tmp_path,
        "unclaim",
        position_id=position_id,
        confirmed=True,
    )
    assert unclaimed["ownership_status"] == "MANUAL_TRACKED"
    assert unclaimed["automatic_execution_eligible"] is False

    with SignalStore(tmp_path) as store:
        signal = store.signal("SIG-MANUAL-1")
        positions = store.manual_positions()
    assert signal is not None
    assert signal["lifecycle_status"] == "EXECUTED_MANUALLY"
    assert positions[0]["quantity"] == "2"
    assert positions[0]["fill_price"] == "101.25"


def test_manual_position_registration_is_idempotent_but_conflicts_block(
    tmp_path: Path,
) -> None:
    _write_signal(tmp_path)

    first = positions_command(
        tmp_path,
        "register-manual",
        signal_id="SIG-MANUAL-1",
        quantity=Decimal("1"),
        fill_price=Decimal("100"),
    )
    repeated = positions_command(
        tmp_path,
        "register-manual",
        signal_id="SIG-MANUAL-1",
        quantity=Decimal("1"),
        fill_price=Decimal("100"),
    )
    conflict = positions_command(
        tmp_path,
        "register-manual",
        signal_id="SIG-MANUAL-1",
        quantity=Decimal("2"),
        fill_price=Decimal("100"),
    )

    assert first["status"] == "GO"
    assert repeated["status"] == "GO"
    assert repeated["transition_status"] == (
        "MANUAL_POSITION_ALREADY_REGISTERED"
    )
    assert repeated["position_id"] == first["position_id"]
    assert conflict["status"] == "BLOCKED"
    assert conflict["reason"] == "MANUAL_POSITION_REGISTRATION_CONFLICT"


def test_manual_position_rejects_unknown_signal_and_invalid_mode(
    tmp_path: Path,
) -> None:
    missing = positions_command(
        tmp_path,
        "register-manual",
        signal_id="MISSING",
        quantity=Decimal("1"),
        fill_price=Decimal("100"),
    )
    assert missing["reason"] == "SIGNAL_NOT_FOUND"

    _write_signal(tmp_path)
    registered = positions_command(
        tmp_path,
        "register-manual",
        signal_id="SIG-MANUAL-1",
        quantity=Decimal("1"),
        fill_price=Decimal("100"),
    )
    invalid = positions_command(
        tmp_path,
        "claim",
        position_id=str(registered["position_id"]),
        ownership_mode="manual",
        confirmed=True,
    )
    assert invalid["reason"] == "BOT_MANAGED_MODE_REQUIRED"


def test_manual_position_rejects_non_entry_signal(tmp_path: Path) -> None:
    with SignalStore(tmp_path) as store:
        store.append_signal(
            {
                "signal_id": "SIG-EXIT",
                "strategy_id": "STRATEGY-1",
                "ticker": "AAPL",
                "action": "SELL",
                "lifecycle_status": "ACTIVE",
                "preferred_entry": "100",
            }
        )

    result = positions_command(
        tmp_path,
        "register-manual",
        signal_id="SIG-EXIT",
        quantity=Decimal("1"),
        fill_price=Decimal("100"),
    )

    assert result["status"] == "BLOCKED"
    assert result["reason"] == "MANUAL_LONG_ENTRY_SIGNAL_REQUIRED"


def _write_signal(root: Path) -> None:
    with SignalStore(root) as store:
        store.append_signal(
            {
                "signal_id": "SIG-MANUAL-1",
                "strategy_id": "STRATEGY-1",
                "ticker": "AAPL",
                "action": "BUY",
                "lifecycle_status": "ACTIVE",
                "preferred_entry": "100",
                "initial_stop": "95",
                "take_profit_1": "110",
                "take_profit_2": "120",
            }
        )
