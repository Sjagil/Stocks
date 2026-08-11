from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from datetime import UTC, datetime
from pathlib import Path

from stocks.live.service import (
    live_reconcile,
    live_status,
    publish_live_capital_safety,
    publish_live_daily_profit_target,
)
from stocks.live import service as live_service


def test_successful_live_reconcile_refreshes_p0_input_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = SimpleNamespace(
        component_audits=[
            SimpleNamespace(
                name="accountsummary",
                request_status="COMPLETE",
            )
        ],
        positions=SimpleNamespace(positions=[]),
        same_client_open_orders=SimpleNamespace(open_orders=[]),
        all_api_open_orders=SimpleNamespace(open_orders=[]),
        executions=SimpleNamespace(executions=[], commissions=[]),
        account=SimpleNamespace(
            values=[
                SimpleNamespace(
                    account_fingerprint="FINGERPRINT",
                    tag="NetLiquidation",
                )
            ]
        ),
    )
    config = SimpleNamespace(host="127.0.0.1", port=7496)
    counters = {
        "place_order_calls": 0,
        "cancel_order_calls": 0,
    }
    refresh_calls: list[Path] = []
    monkeypatch.setattr(
        live_service,
        "_load_live_observer_config",
        lambda _root, _env: (config, []),
    )
    monkeypatch.setattr(live_service, "_socket_reachable", lambda *_: True)
    monkeypatch.setattr(
        live_service,
        "capture_snapshot",
        lambda _config: (snapshot, {}, counters),
    )
    monkeypatch.setattr(
        live_service.BrokerObservationStore,
        "write_snapshot",
        lambda _self, _snapshot: "SNAPSHOT-HASH",
    )
    monkeypatch.setattr(
        live_service,
        "public_snapshot_summary",
        lambda *_: {"private_snapshot_hash": "SNAPSHOT-HASH"},
    )
    monkeypatch.setattr(
        live_service,
        "publish_live_capital_safety",
        lambda *_args, **_kwargs: {"status": "GO"},
    )
    monkeypatch.setattr(
        live_service,
        "publish_live_daily_profit_target",
        lambda *_args, **_kwargs: {"status": "GO"},
    )
    monkeypatch.setattr(
        live_service,
        "write_p0_execution_readiness",
        lambda root: refresh_calls.append(root) or {"status": "GO"},
    )

    report = live_reconcile(tmp_path, env_file=".env.ibkr.live")

    assert report["status"] == "GO"
    assert report["reconciliation_status"] == "LIVE_RECONCILED_EMPTY"
    assert report["broker_write_calls"] == 0
    assert refresh_calls == [tmp_path]


def test_capital_safety_uses_private_values_and_publishes_no_amounts(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env.ibkr.live"
    env.write_text(
        "\n".join(
            [
                "IBKR_ACCOUNT_BASE_CURRENCY=EUR",
                "IBKR_MAX_TOTAL_EXPOSURE_EUR=25",
            ]
        ),
        encoding="utf-8",
    )
    private_db = (
        tmp_path
        / "data"
        / "execution"
        / "live"
        / "private"
        / "broker_observation.sqlite3"
    )
    private_db.parent.mkdir(parents=True)
    observed = datetime.now(UTC).isoformat()
    values = [
        {
            "account_fingerprint": "MASKED-FINGERPRINT",
            "tag": tag,
            "value": value,
            "currency": "EUR",
            "observed_at": observed,
        }
        for tag, value in {
            "NetLiquidation": "100.25",
            "TotalCashValue": "100.25",
            "SettledCash": "100.25",
            "AvailableFunds": "100.25",
            "BuyingPower": "100.25",
            "GrossPositionValue": "0",
            "InitMarginReq": "0",
            "MaintMarginReq": "0",
            "ExcessLiquidity": "100.25",
            "CashBalance": "100.25",
        }.items()
    ]
    component = {"started_at": observed, "completed_at": observed}
    payload = {
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
    with sqlite3.connect(private_db) as connection:
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
            "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
            ("SNAPSHOT", "HASH", json.dumps(payload), observed),
        )
    reconciliation = (
        tmp_path / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    reconciliation.parent.mkdir(parents=True)
    reconciliation.write_text(
        json.dumps(
            {
                "reconciliation_status": "LIVE_RECONCILED_EMPTY",
                "private_snapshot_hash": "HASH",
            }
        ),
        encoding="utf-8",
    )

    report = publish_live_capital_safety(tmp_path, env_file=env)

    assert report["status"] == "GO"
    assert report["buying_power_sufficient"] is True
    public = (
        tmp_path / "output" / "ibkr" / "live" / "capital-safety.json"
    ).read_text(encoding="utf-8")
    assert "100.25" not in public
    assert "AvailableFunds" not in public
    assert "BuyingPower" not in public


def test_capital_safety_requires_explicit_eur_base_currency(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env.ibkr.live"
    env.write_text("IBKR_MAX_TOTAL_EXPOSURE_EUR=25\n", encoding="utf-8")

    report = publish_live_capital_safety(tmp_path, env_file=env)

    assert report["status"] == "NO_GO"
    assert "EXPLICIT_EUR_ACCOUNT_BASE_CURRENCY_REQUIRED" in report["blockers"]
    assert report["financial_values_public"] is False


def test_live_daily_target_uses_prior_session_equity_and_scales(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env.ibkr.live"
    env.write_text(
        "IBKR_ACCOUNT_BASE_CURRENCY=EUR\n",
        encoding="utf-8",
    )
    policy = (
        tmp_path / "config" / "capital_scaling" / "levels_v1.json"
    )
    policy.parent.mkdir(parents=True)
    policy.write_text(
        json.dumps(
            {
                "daily_profit_target": {
                    "target_pct_of_equity": 0.005,
                    "minimum_target_eur": 0,
                    "maximum_target_eur": None,
                    "timezone": "Europe/Amsterdam",
                }
            }
        ),
        encoding="utf-8",
    )
    private_db = (
        tmp_path
        / "data"
        / "execution"
        / "live"
        / "private"
        / "broker_observation.sqlite3"
    )
    private_db.parent.mkdir(parents=True)
    with sqlite3.connect(private_db) as connection:
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
        for snapshot_id, snapshot_hash, value, gross, created_at in (
            (
                "PRIOR",
                "PRIOR-HASH",
                "49750",
                "48750",
                "2026-07-28T20:00:00+00:00",
            ),
            (
                "CURRENT",
                "CURRENT-HASH",
                "50000",
                "49000",
                "2026-07-29T11:59:00+00:00",
            ),
        ):
            payload = {
                "account": {
                    "values": [
                        {
                            "account_fingerprint": "MASKED-FINGERPRINT",
                            "tag": "NetLiquidation",
                            "value": value,
                            "currency": "BASE",
                        },
                        {
                            "account_fingerprint": "MASKED-FINGERPRINT",
                            "tag": "TotalCashValue",
                            "value": "1000",
                            "currency": "BASE",
                        },
                        {
                            "account_fingerprint": "MASKED-FINGERPRINT",
                            "tag": "GrossPositionValue",
                            "value": gross,
                            "currency": "BASE",
                        },
                    ]
                }
            }
            connection.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
                (
                    snapshot_id,
                    snapshot_hash,
                    json.dumps(payload),
                    created_at,
                ),
            )
    reconciliation = (
        tmp_path / "output" / "ibkr" / "live" / "reconciliation.json"
    )
    reconciliation.parent.mkdir(parents=True)
    reconciliation.write_text(
        json.dumps(
            {
                "private_snapshot_hash": "CURRENT-HASH",
                "position_count": 1,
                "open_order_count": 0,
                "execution_count": 0,
            }
        ),
        encoding="utf-8",
    )

    report = publish_live_daily_profit_target(
        tmp_path,
        env_file=env,
        now=datetime.fromisoformat("2026-07-29T12:00:00+00:00"),
    )

    assert report["status"] == "GO"
    assert report["daily_profit_target_eur"] == 250.0
    assert report["net_daily_pnl_eur"] == 250.0
    assert report["target_reached"] is True
    assert report["new_entries_allowed"] is False
    assert report["risk_reducing_exits_allowed"] is True
    assert report["force_liquidation"] is False
    assert report["risk_chasing_allowed"] is False
    assert report["input_source"] == (
        "BROKER_RECONCILED_EQUITY_AND_CASH_CHANGE"
    )

    status = live_status(tmp_path)
    assert status["daily_pnl_eur"] == 250.0
    assert status["daily_profit_target_eur"] == 250.0
    assert status["daily_profit_target_reached"] is True
    assert status["new_entries_allowed_by_daily_target"] is False


def test_live_daily_target_treats_empty_account_deposit_as_cash_flow(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env.ibkr.live"
    env.write_text("IBKR_ACCOUNT_BASE_CURRENCY=EUR\n", encoding="utf-8")
    policy = tmp_path / "config/capital_scaling/levels_v1.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        json.dumps(
            {
                "daily_profit_target": {
                    "target_pct_of_equity": 0.005,
                    "minimum_target_eur": 0,
                    "maximum_target_eur": None,
                    "timezone": "Europe/Amsterdam",
                }
            }
        ),
        encoding="utf-8",
    )
    private_db = (
        tmp_path
        / "data/execution/live/private/broker_observation.sqlite3"
    )
    private_db.parent.mkdir(parents=True)
    with sqlite3.connect(private_db) as connection:
        connection.execute(
            "CREATE TABLE snapshots (snapshot_id TEXT PRIMARY KEY, "
            "snapshot_hash TEXT NOT NULL, payload_json TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        for snapshot_id, snapshot_hash, value, created_at in (
            ("PRIOR", "PRIOR-HASH", "20", "2026-07-28T20:00:00+00:00"),
            ("CURRENT", "CURRENT-HASH", "1870", "2026-07-29T11:59:00+00:00"),
        ):
            payload = {
                "account": {
                    "values": [
                        {
                            "account_fingerprint": "MASKED-FINGERPRINT",
                            "tag": tag,
                            "value": value if tag != "GrossPositionValue" else "0",
                            "currency": "BASE",
                        }
                        for tag in (
                            "NetLiquidation",
                            "TotalCashValue",
                            "GrossPositionValue",
                        )
                    ]
                }
            }
            connection.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
                (snapshot_id, snapshot_hash, json.dumps(payload), created_at),
            )
    reconciliation = tmp_path / "output/ibkr/live/reconciliation.json"
    reconciliation.parent.mkdir(parents=True)
    reconciliation.write_text(
        json.dumps(
            {
                "private_snapshot_hash": "CURRENT-HASH",
                "position_count": 0,
                "open_order_count": 0,
                "execution_count": 0,
            }
        ),
        encoding="utf-8",
    )

    report = publish_live_daily_profit_target(
        tmp_path,
        env_file=env,
        now=datetime.fromisoformat("2026-07-29T12:00:00+00:00"),
    )

    assert report["status"] == "GO"
    assert report["net_daily_pnl_eur"] == 0.0
    assert report["target_reached"] is False
    assert report["new_entries_allowed"] is True
    assert report["cash_flow_detected"] is True
    assert report["cash_flow_adjustment_status"] == (
        "INFERRED_EMPTY_ACCOUNT_CASH_FLOW"
    )


def test_live_status_merges_current_reconciliation_and_allowlist_blockers(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "output" / "reports"
    reports.mkdir(parents=True)
    (reports / "live_preflight.json").write_text(
        json.dumps({"blockers": ["EXACT_OPERATOR_APPROVAL_REQUIRED"]}),
        encoding="utf-8",
    )
    live_output = tmp_path / "output" / "ibkr" / "live"
    live_output.mkdir(parents=True)
    (live_output / "reconciliation.json").write_text(
        json.dumps(
            {
                "status": "NO_GO",
                "reconciliation_status": "LIVE_TWS_SOCKET_UNREACHABLE",
                "blockers": ["LIVE_TWS_SOCKET_UNREACHABLE"],
            }
        ),
        encoding="utf-8",
    )
    (live_output / "strategy-allowlist.json").write_text(
        json.dumps({"status": "NO_GO", "strategy_count": 0}),
        encoding="utf-8",
    )

    status = live_status(tmp_path)

    assert "EXACT_OPERATOR_APPROVAL_REQUIRED" in status["open_blockers"]
    assert "LIVE_TWS_SOCKET_UNREACHABLE" in status["open_blockers"]
    assert "PIT_STRATEGY_ALLOWLIST_REQUIRED" in status["open_blockers"]
    assert "LIVE_EXECUTION_WRITER_NOT_FROZEN" in status["open_blockers"]


def test_live_status_drops_stale_preflight_allowlist_blocker(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "output" / "reports"
    reports.mkdir(parents=True)
    (reports / "live_preflight.json").write_text(
        json.dumps(
            {
                "blockers": [
                    "EXACT_OPERATOR_APPROVAL_REQUIRED",
                    "PIT_STRATEGY_ALLOWLIST_REQUIRED",
                ]
            }
        ),
        encoding="utf-8",
    )
    live_output = tmp_path / "output" / "ibkr" / "live"
    live_output.mkdir(parents=True)
    (live_output / "strategy-allowlist.json").write_text(
        json.dumps({"status": "GO", "strategy_count": 1}),
        encoding="utf-8",
    )

    status = live_status(tmp_path)

    assert "EXACT_OPERATOR_APPROVAL_REQUIRED" in status["open_blockers"]
    assert "PIT_STRATEGY_ALLOWLIST_REQUIRED" not in (
        status["open_blockers"]
    )


def test_live_status_recomputes_stale_writer_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    reports = tmp_path / "output" / "reports"
    reports.mkdir(parents=True)
    (reports / "live_preflight.json").write_text(
        json.dumps(
            {
                "blockers": [
                    "EXACT_OPERATOR_APPROVAL_REQUIRED",
                    "LIVE_EXECUTION_WRITER_NOT_FROZEN",
                ]
            }
        ),
        encoding="utf-8",
    )
    live_output = tmp_path / "output" / "ibkr" / "live"
    live_output.mkdir(parents=True)
    (live_output / "strategy-allowlist.json").write_text(
        json.dumps({"status": "GO", "strategy_count": 1}),
        encoding="utf-8",
    )
    monkeypatch.setattr(live_service, "_writer_frozen", lambda _root: True)

    status = live_status(tmp_path)

    assert status["writer_hash_integrity"] is True
    assert "LIVE_EXECUTION_WRITER_NOT_FROZEN" not in (
        status["open_blockers"]
    )
    assert "EXACT_OPERATOR_APPROVAL_REQUIRED" in status["open_blockers"]
