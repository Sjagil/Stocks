from __future__ import annotations

from pathlib import Path

import stocks.operations.service as service
from stocks.ibkr.paper_execution import PHASE9_MARKER


def test_capability_command_without_yes_fails_before_broker_reads(
    tmp_path: Path, monkeypatch
) -> None:
    broker_reads = []
    monkeypatch.setattr(
        service,
        "live_reconcile",
        lambda *_args, **_kwargs: broker_reads.append(True),
    )

    report = service.execution_command(
        tmp_path,
        "create-live-capability",
        env_file=".env.ibkr.live",
        confirmed=False,
    )

    assert report["status"] == "NO_GO"
    assert report["execution_authority"] == "NONE"
    assert report["broker_writes"] == 0
    assert report["blockers"] == ["EXPLICIT_YES_CONFIRMATION_REQUIRED"]
    assert broker_reads == []


def test_unknown_live_profile_fails_before_broker_reads(
    tmp_path: Path, monkeypatch
) -> None:
    broker_reads = []
    monkeypatch.setattr(
        service,
        "live_reconcile",
        lambda *_args, **_kwargs: broker_reads.append(True),
    )

    report = service.execution_command(
        tmp_path,
        "create-live-capability",
        env_file=".env.ibkr.live",
        confirmed=True,
        profile="unregistered-profile",
    )

    assert report["status"] == "NO_GO"
    assert report["blockers"] == ["UNKNOWN_LIVE_PROFILE"]
    assert report["execution_authority"] == "NONE"
    assert broker_reads == []


def test_create_capability_reconciles_and_stays_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    monkeypatch.setattr(
        service,
        "live_reconcile",
        lambda *_args, **_kwargs: {
            "status": "GO",
            "reconciliation_status": "LIVE_RECONCILED_EMPTY",
        },
    )
    monkeypatch.setattr(
        service,
        "live_preflight",
        lambda *_args, **_kwargs: {
            "status": "GO",
            "blockers": [],
            "safe_config": {},
        },
    )

    def create(_root, **kwargs):
        calls.append(kwargs)
        return {
            "status": "GO",
            "capability_status": "READY",
            "execution_authority": "NONE",
            "broker_writes": 0,
        }

    monkeypatch.setattr(service, "create_live_capability", create)

    report = service.execution_command(
        tmp_path,
        "create-live-capability",
        env_file=".env.ibkr.live",
        confirmed=True,
        profile="autonomous_multi_asset_v1",
    )

    assert report["status"] == "GO"
    assert report["execution_authority"] == "NONE"
    assert report["broker_writes"] == 0
    assert calls[0]["confirmed"] is True


def test_activate_capability_reconciles_then_consumes(
    tmp_path: Path, monkeypatch
) -> None:
    sequence = []
    monkeypatch.setattr(
        service,
        "live_reconcile",
        lambda *_args, **_kwargs: (
            sequence.append("reconcile")
            or {
                "status": "GO",
                "reconciliation_status": "LIVE_RECONCILED_EMPTY",
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "live_preflight",
        lambda *_args, **_kwargs: (
            sequence.append("preflight")
            or {"status": "GO", "blockers": [], "safe_config": {}}
        ),
    )
    monkeypatch.setattr(
        service,
        "activate_live_capability",
        lambda *_args, **_kwargs: (
            sequence.append("consume")
            or {
                "status": "GO",
                "execution_authority": "LIVE_LEVEL_ONE",
                "capability_status": "CONSUMED",
            }
        ),
    )
    monkeypatch.setattr(
        service,
        "_execution_status",
        lambda _root, state: {
            "status": "GO",
            "mode": state["mode"],
            "live_enabled": state["live_enabled"],
            "execution_authority": (
                "LIVE_LEVEL_ONE" if state["live_enabled"] else "NONE"
            ),
        },
    )

    report = service.execution_command(
        tmp_path,
        "activate-live-capability",
        approval="exact private phrase",
        env_file=".env.ibkr.live",
        confirmed=True,
    )

    assert sequence == ["reconcile", "preflight", "consume"]
    assert report["execution_authority"] == "LIVE_LEVEL_ONE"
    state = service.execution_command(tmp_path, "status")
    assert state["mode"] == "CONTROLLED_LIVE"
    assert state["live_enabled"] is True


def test_execution_status_requires_complete_phase9_round_trip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    state = {
        "mode": "PAPER_AUTOMATIC",
        "paper_enabled": True,
        "live_enabled": False,
    }
    monkeypatch.setattr(
        service,
        "authority_status",
        lambda _root: {"execution_authority": "NONE"},
    )
    monkeypatch.setattr(
        service,
        "phase9_status",
        lambda _root: {
            "status": "NO_GO",
            "checks": {
                "reconciliation": True,
                "submit_cancel_canary": True,
                "fill_canary": True,
                "closing_sell_canary": False,
            },
        },
    )

    incomplete = service._execution_status(tmp_path, state)

    assert incomplete["paper_fill_close_canary_go"] is False
    assert incomplete["execution_authority"] == "NONE"
    assert incomplete["live_limits"] == {
        "max_order_eur": 250,
        "max_total_exposure_eur": 250,
        "max_open_positions": 1,
        "max_new_orders_per_day": 1,
        "max_risk_eur": 10,
        "max_daily_loss_eur": 5,
        "max_drawdown_eur": 10,
    }

    monkeypatch.setattr(
        service,
        "phase9_status",
        lambda _root: {
            "status": PHASE9_MARKER,
            "checks": {
                "reconciliation": True,
                "submit_cancel_canary": True,
                "fill_canary": True,
                "closing_sell_canary": True,
            },
        },
    )

    complete = service._execution_status(tmp_path, state)

    assert complete["paper_fill_close_canary_go"] is True
    assert complete["execution_authority"] == (
        "AUTOMATIC_BOUNDED_PAPER"
    )
