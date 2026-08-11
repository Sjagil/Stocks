from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from stocks.ibkr.p0_safety import write_p0_safety_report
from stocks.ibkr.p0_readiness import write_p0_execution_readiness
from stocks.live.authority import (
    LIVE_LEVEL_ONE,
    LIVE_LEVEL_TWO,
    LEVEL_TWO_ACTIVATION_APPROVAL,
    activate_live_capability,
    activate_level_one,
    activate_level_two,
    authority_status,
    create_live_capability,
    kill_level_one,
    pause_level_one,
    resume_level_one,
)
from stocks.research.autopilot.contracts import stable_hash


def test_activation_requires_complete_preflight(tmp_path: Path) -> None:
    result = activate_level_one(
        tmp_path,
        preflight={
            "status": "NO_GO",
            "blockers": ["LIVE_TWS_SOCKET_UNREACHABLE"],
        },
    )

    assert result["status"] == "NO_GO"
    assert result["execution_authority"] == "NONE"
    assert result["state_changed"] is False
    assert not _state_path(tmp_path).exists()


def test_activation_rejects_forged_go_without_p0_attestation(
    tmp_path: Path,
) -> None:
    _write_allowlist(tmp_path)
    path = tmp_path / "output" / "ibkr" / "live" / "reconciliation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "GO",
                "reconciliation_status": "LIVE_RECONCILED_EMPTY",
                "account_fingerprints": ["PRIVATE-FINGERPRINT"],
            }
        ),
        encoding="utf-8",
    )

    result = activate_level_one(
        tmp_path,
        preflight={"status": "GO", "blockers": []},
    )

    assert result["status"] == "NO_GO"
    assert result["transition_status"] == (
        "P0_EXECUTION_INFRASTRUCTURE_NOT_READY"
    )
    assert result["execution_authority"] == "NONE"
    assert not _state_path(tmp_path).exists()


def test_level_one_activation_is_pinned_pauseable_and_killable(
    tmp_path: Path,
) -> None:
    allowlist = _write_allowlist(tmp_path)
    _write_reconciliation(tmp_path)
    preflight = {"status": "GO", "blockers": [], "checks": {"all": True}}

    activated = activate_level_one(tmp_path, preflight=preflight)

    assert activated["execution_authority"] == LIVE_LEVEL_ONE
    assert activated["current_scaling_level"] == "LEVEL_1"
    assert activated["active_strategies"] == [
        "WEEKLY_CROSS_SECTIONAL_MOMENTUM"
    ]
    assert activated["active_symbols"] == ["AAPL"]
    assert activated["submission_mode"] == "MANUAL_APPROVAL_ONLY"
    assert activated["manual_approval_required"] is True
    assert activated["automatic_order_submission"] is False
    assert activated["limits"]["maximum_quantity"] == 100
    private_text = _state_path(tmp_path).read_text(encoding="utf-8")
    assert "EXACT APPROVAL" not in private_text

    allowlist["generated_at"] = "2026-07-30T12:00:00+00:00"
    _allowlist_path(tmp_path).write_text(
        json.dumps(allowlist), encoding="utf-8"
    )
    refreshed = authority_status(tmp_path)
    assert refreshed["allowlist_hash_matches"] is True
    assert refreshed["execution_authority"] == LIVE_LEVEL_ONE

    paused = pause_level_one(tmp_path, reason="operator maintenance")
    assert paused["lifecycle_status"] == "PAUSED"
    assert paused["execution_authority"] == "NONE"

    resumed = resume_level_one(tmp_path, preflight=preflight)
    assert resumed["lifecycle_status"] == "ACTIVE"
    assert resumed["execution_authority"] == LIVE_LEVEL_ONE

    allowlist["strategies"][0]["version"] = "MUTATED"
    _allowlist_path(tmp_path).write_text(
        json.dumps(allowlist), encoding="utf-8"
    )
    invalidated = authority_status(tmp_path)
    assert invalidated["allowlist_hash_matches"] is False
    assert invalidated["execution_authority"] == "NONE"

    killed = kill_level_one(tmp_path, reason="risk incident")
    assert killed["lifecycle_status"] == "KILLED"
    assert killed["execution_authority"] == "NONE"
    events = _state_path(tmp_path).with_name(
        "authority-events.jsonl"
    ).read_text(encoding="utf-8")
    assert "LIVE_LEVEL_ONE_ACTIVATED" in events
    assert "LIVE_LEVEL_ONE_PAUSED" in events
    assert "LIVE_LEVEL_ONE_RESUMED" in events
    assert "LIVE_LEVEL_ONE_KILLED" in events


def test_level_two_activation_changes_authority_without_broker_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from stocks.capital import service as capital_service
    from stocks.live import evidence, portfolio_targets

    _write_allowlist(tmp_path)
    _write_reconciliation(tmp_path)
    activated = activate_level_one(
        tmp_path,
        preflight={"status": "GO", "blockers": []},
    )
    assert activated["execution_authority"] == LIVE_LEVEL_ONE
    capital_path = tmp_path / "output/capital/current_level.json"
    capital_path.parent.mkdir(parents=True, exist_ok=True)
    capital_path.write_text(
        json.dumps({"CURRENT_CAPITAL_LEVEL": 2}), encoding="utf-8"
    )
    monkeypatch.setattr(
        capital_service,
        "capital_command",
        lambda _root, _command: {"CURRENT_CAPITAL_LEVEL": 2},
    )
    monkeypatch.setattr(
        evidence,
        "live_level_two_evidence",
        lambda _root: {
            "status": "GO",
            "verified_round_trip_count": 5,
            "content_hash": "EVIDENCE-HASH",
            "blockers": [],
        },
    )
    monkeypatch.setattr(
        portfolio_targets,
        "controlled_live_preflight",
        lambda _root, **_kwargs: {
            "status": "GO",
            "blockers": [],
            "safe_config": {
                "max_order_eur": "250",
                "max_total_exposure_eur": "1122",
                "max_risk_eur": "28.05",
                "max_open_positions": 4,
                "max_new_orders_per_day": 1,
                "maximum_quantity": "100",
            },
        },
    )

    result = activate_level_two(
        tmp_path,
        symbol="ON",
        approval=LEVEL_TWO_ACTIVATION_APPROVAL,
    )

    assert result["transition_status"] == "LIVE_LEVEL_TWO_ACTIVATED"
    assert result["execution_authority"] == LIVE_LEVEL_TWO
    assert result["state_changed"] is True
    assert result["broker_writes"] == 0
    state = json.loads(_state_path(tmp_path).read_text(encoding="utf-8"))
    assert state["execution_authority"] == LIVE_LEVEL_TWO
    assert state["limits"]["max_order_value_eur"] == "250"

    paused = pause_level_one(tmp_path, reason="level two maintenance")
    assert paused["execution_authority"] == "NONE"
    blocked_resume = resume_level_one(
        tmp_path, preflight={"status": "GO", "blockers": []}
    )
    assert blocked_resume["transition_status"] == (
        "LIVE_LEVEL_TWO_REACTIVATION_REQUIRES_EXPLICIT_APPROVAL"
    )
    reactivated = activate_level_two(
        tmp_path,
        symbol="ON",
        approval=LEVEL_TWO_ACTIVATION_APPROVAL,
    )
    assert reactivated["execution_authority"] == LIVE_LEVEL_TWO


def test_resume_rejects_changed_frozen_allowlist(tmp_path: Path) -> None:
    allowlist = _write_allowlist(tmp_path)
    _write_reconciliation(tmp_path)
    preflight = {"status": "GO", "blockers": []}
    activate_level_one(tmp_path, preflight=preflight)
    pause_level_one(tmp_path, reason="test")
    allowlist["qualification_hash"] = "CHANGED"
    _allowlist_path(tmp_path).write_text(
        json.dumps(allowlist), encoding="utf-8"
    )

    result = resume_level_one(tmp_path, preflight=preflight)

    assert result["status"] == "NO_GO"
    assert result["transition_status"] == "FROZEN_LIVE_ALLOWLIST_CHANGED"
    assert result["execution_authority"] == "NONE"


def test_active_authority_drops_on_current_reconciliation_failure(
    tmp_path: Path,
) -> None:
    _write_allowlist(tmp_path)
    _write_reconciliation(tmp_path)
    activated = activate_level_one(
        tmp_path,
        preflight={"status": "GO", "blockers": []},
    )
    assert activated["execution_authority"] == LIVE_LEVEL_ONE
    path = tmp_path / "output/ibkr/live/reconciliation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(
        status="NO_GO",
        reconciliation_status="LIVE_TWS_SOCKET_UNREACHABLE",
    )
    payload["content_hash"] = stable_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")

    current = authority_status(tmp_path)

    assert current["lifecycle_status"] == "ACTIVE"
    assert current["live_reconciliation_gate_go"] is False
    assert current["execution_authority"] == "NONE"


def test_live_capability_is_expiring_bound_and_single_use(
    tmp_path: Path,
) -> None:
    _write_allowlist(tmp_path)
    _write_reconciliation(tmp_path)
    _write_profile(tmp_path)
    now = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
    preflight = {
        "status": "GO",
        "blockers": [],
        "safe_config": {
            "environment": "LIVE",
            "host_local_only": True,
            "port_class": "LIVE",
            "max_order_eur": "10",
            "max_total_exposure_eur": "25",
            "max_open_positions": 1,
            "max_new_orders_per_day": 1,
            "autoscaling_enabled": False,
            "futures_allowed": False,
        },
    }

    created = create_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now,
    )
    activated = activate_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now + timedelta(minutes=1),
    )
    replay = activate_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now + timedelta(minutes=2),
    )

    assert created["status"] == "GO"
    assert created["execution_authority"] == "NONE"
    assert activated["execution_authority"] == LIVE_LEVEL_ONE
    assert activated["capability_status"] == "CONSUMED"
    assert replay["status"] == "NO_GO"
    assert "LIVE_CAPABILITY_ALREADY_CONSUMED" in replay["blockers"]


def test_live_capability_expires_and_detects_allowlist_change(
    tmp_path: Path,
) -> None:
    allowlist = _write_allowlist(tmp_path)
    _write_reconciliation(tmp_path)
    _write_profile(tmp_path)
    now = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
    preflight = {"status": "GO", "blockers": [], "safe_config": {}}
    create_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now,
    )
    expired = activate_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now + timedelta(minutes=16),
    )

    assert "LIVE_CAPABILITY_EXPIRED" in expired["blockers"]

    create_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now,
    )
    allowlist["strategies"][0]["allowed_symbols"] = ["MSFT"]
    _allowlist_path(tmp_path).write_text(
        json.dumps(allowlist), encoding="utf-8"
    )
    changed = activate_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now + timedelta(minutes=1),
    )

    assert "LIVE_CAPABILITY_BINDING_CHANGED" in changed["blockers"]


def test_live_capability_rejects_missing_profile_and_state_change(
    tmp_path: Path,
) -> None:
    _write_allowlist(tmp_path)
    _write_reconciliation(tmp_path)
    now = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
    preflight = {"status": "GO", "blockers": [], "safe_config": {}}

    missing_profile = create_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now,
    )
    assert "LIVE_PROFILE_CONFIG_INVALID" in missing_profile["blockers"]

    _write_profile(tmp_path)
    created = create_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now,
    )
    assert created["status"] == "GO"
    reconciliation = (
        tmp_path / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    payload = json.loads(reconciliation.read_text(encoding="utf-8"))
    payload["position_count"] = 1
    reconciliation.write_text(json.dumps(payload), encoding="utf-8")

    changed = activate_live_capability(
        tmp_path,
        preflight=preflight,
        confirmed=True,
        now=now + timedelta(minutes=1),
    )
    assert "LIVE_CAPABILITY_BINDING_CHANGED" in changed["blockers"]


def _write_allowlist(root: Path) -> dict:
    strategy = {
        "strategy_id": "WEEKLY_CROSS_SECTIONAL_MOMENTUM",
        "version": "PHASE11_13_V1",
        "qualification_hash": "QUALIFICATION",
        "allowed_symbols": ["AAPL"],
    }
    payload = {
        "schema": "ibkr_live_pit_strategy_allowlist_v1",
        "status": "GO",
        "qualification_hash": "QUALIFICATION",
        "strategy_count": 1,
        "strategies": [strategy],
    }
    path = _allowlist_path(root)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert stable_hash(payload)
    return payload


def _write_reconciliation(root: Path) -> None:
    path = root / "output" / "ibkr" / "live" / "reconciliation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _execution_ready_snapshot()
    snapshot_hash = stable_hash(snapshot)
    database = (
        root
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE snapshots(snapshot_id TEXT, snapshot_hash TEXT, "
            "payload_json TEXT, created_at TEXT)"
        )
        connection.execute(
            "INSERT INTO snapshots VALUES(?,?,?,?)",
            (
                "S1",
                snapshot_hash,
                json.dumps(snapshot),
                snapshot["snapshot_completed_at"],
            ),
        )
    payload = {
        "status": "GO",
        "reconciliation_status": "LIVE_RECONCILED_EMPTY",
        "account_fingerprints": ["PRIVATE-FINGERPRINT"],
        "unknown_orders": 0,
        "unknown_positions": 0,
        "position_count": 0,
        "open_order_count": 0,
        "private_snapshot_hash": snapshot_hash,
        "blockers": [],
    }
    payload["content_hash"] = stable_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    p0 = write_p0_safety_report(root)
    assert p0["status"] == "GO"
    readiness = write_p0_execution_readiness(root)
    assert readiness["status"] == "GO"


def _execution_ready_snapshot() -> dict:
    observed = datetime.now(UTC).isoformat()
    values = []
    for tag, value in {
        "NetLiquidation": "2000",
        "TotalCashValue": "2000",
        "SettledCash": "2000",
        "AvailableFunds": "1800",
        "BuyingPower": "3600",
        "GrossPositionValue": "0",
        "InitMarginReq": "0",
        "MaintMarginReq": "0",
        "ExcessLiquidity": "1800",
        "CashBalance": "2000",
    }.items():
        values.append(
            {
                "account_fingerprint": "PRIVATE-FINGERPRINT",
                "tag": tag,
                "value": value,
                "currency": "EUR",
                "observed_at": observed,
            }
        )
    component = {"started_at": observed, "completed_at": observed}
    return {
        "snapshot_completed_at": observed,
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
            name: component
            for name in (
                "accountsummary",
                "positions",
                "same_client_open_orders",
                "all_api_open_orders",
                "executions",
            )
        },
    }


def _allowlist_path(root: Path) -> Path:
    return root / "output" / "ibkr" / "live" / "strategy-allowlist.json"


def _state_path(root: Path) -> Path:
    return (
        root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "authority-state.json"
    )


def _write_profile(root: Path) -> None:
    path = (
        root
        / "config"
        / "operations"
        / "autonomous_multi_asset_v1.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "stocks_operations_profile_v1",
                "profile_id": "autonomous_multi_asset_v1",
                "execution_mode": "CONTROLLED_LIVE",
                "authority": {
                    "initial_operator_activation_required": True,
                    "per_order_approval_required": False,
                },
                "constraints": {
                    "margin_enabled": False,
                    "leverage_enabled": False,
                    "shorting_enabled": False,
                    "futures_enabled": False,
                },
            }
        ),
        encoding="utf-8",
    )
