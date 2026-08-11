from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import main
import pytest
from filelock import FileLock
from stocks.ibkr.reconciliation import audit as reconciliation_audit
from stocks.ibkr.reconciliation.adapter import (
    enforce_method_allowed,
    zero_read_counters,
    zero_write_counters,
)
from stocks.ibkr.reconciliation.callbacks import BrokerObserverState
from stocks.ibkr.reconciliation.comparator import compare_phase7_to_broker
from stocks.ibkr.reconciliation.errors import BROKER_OBSERVATION_AUTHORITY, EXECUTION_AUTHORITY, Phase8Blocked
from stocks.ibkr.reconciliation.masking import account_fingerprint, contains_raw_account
from stocks.ibkr.reconciliation.requests import load_phase8_config
from stocks.ibkr.reconciliation.snapshots import (
    capture_snapshot,
    snapshot_components_complete,
    snapshot_from_state,
    stability_status,
)
from stocks.ibkr.reconciliation.storage import BrokerObservationStore, Phase8Layout, public_snapshot_summary, write_json


def test_phase8_schema_cli_preserves_authorities(capsys) -> None:
    assert main.main(["ibkr", "phase8", "schema"]) == 0
    out = capsys.readouterr().out
    assert "PHASE8_IBKR_READ_ONLY_RECONCILIATION_ADAPTER_GO" in out
    assert f'"broker_observation_authority": "{BROKER_OBSERVATION_AUTHORITY}"' in out
    assert f'"execution_authority": "{EXECUTION_AUTHORITY}"' in out


def test_recon_config_blocks_zero_collision_and_missing_key(tmp_path) -> None:
    env = _env(tmp_path, recon_client_id=0, key="")
    _, errors = load_phase8_config(tmp_path, env)
    assert "CLIENT_ID_ZERO_BLOCKED" in errors
    assert "ACCOUNT_FINGERPRINT_KEY_MISSING" in errors

    env = _env(tmp_path, recon_client_id=17, key="fixture-key")
    _, errors = load_phase8_config(tmp_path, env)
    assert "CLIENT_ID_COLLISION_BLOCKED" in errors


def test_fingerprint_is_deterministic_and_raw_account_removed_at_callback_edge() -> None:
    key = "fixture-key"
    first = account_fingerprint("DU_TEST_ACCOUNT_001", key)
    second = account_fingerprint("DU_TEST_ACCOUNT_001", key)
    changed = account_fingerprint("DU_TEST_ACCOUNT_002", key)
    assert first == second
    assert first != changed

    state = BrokerObserverState(key)
    state.record_account_summary(1, "DU_TEST_ACCOUNT_001", "NetLiquidation", "1000.00", "EUR")
    state.record_position("DU_TEST_ACCOUNT_001", _contract(), Decimal("2"), Decimal("100"))
    payload = {
        "account_values": [item.__dict__ for item in state.account_values],
        "positions": [item.__dict__ for item in state.positions],
    }
    assert contains_raw_account(payload) is False
    assert state.account_values[0].account_fingerprint == first


def test_raw_account_detection_does_not_flag_uppercase_status_words() -> None:
    assert contains_raw_account("DU1234567") is True
    assert contains_raw_account("U1234567") is True
    assert contains_raw_account("UNCLASSIFIED_CHANGE_BLOCKED") is False
    assert contains_raw_account("UNKNOWN_PROVIDER_CONTINUITY") is False


def test_raw_account_not_written_to_private_db_or_public_artifacts(tmp_path) -> None:
    config = _config(tmp_path)
    state = _full_state(config.account_fingerprint_key)
    snapshot = snapshot_from_state(state, config, started_at="2026-07-21T10:00:00+00:00", completed_at="2026-07-21T10:00:01+00:00")
    store = BrokerObservationStore(tmp_path / "broker_observation.sqlite3")
    private_hash = store.write_snapshot(snapshot)
    summary = public_snapshot_summary(snapshot, private_hash, store.db_path)
    public = tmp_path / "public.json"
    write_json(public, summary)

    assert contains_raw_account(store.db_path.read_bytes().decode("utf-8", errors="ignore")) is False
    assert contains_raw_account(public.read_text(encoding="utf-8")) is False
    assert "1000" not in public.read_text(encoding="utf-8")


def test_accountsummary_and_positions_timeout_and_complete_sequences(tmp_path) -> None:
    config = _config(tmp_path)
    complete = _full_state(config.account_fingerprint_key)
    snapshot = snapshot_from_state(complete, config, started_at="2026-07-21T10:00:00+00:00", completed_at="2026-07-21T10:00:01+00:00")
    assert snapshot.account.status == "COMPLETE"
    assert snapshot.positions.status == "COMPLETE"
    assert snapshot.executions.status == "COMPLETE"
    assert snapshot.executions.execution_history_complete is False

    timed_out = BrokerObserverState(config.account_fingerprint_key)
    timed_out.record_account_summary(1, "DU_TEST_ACCOUNT_001", "AccountType", "INDIVIDUAL", "")
    timeout_snapshot = snapshot_from_state(timed_out, config, started_at="2026-07-21T10:00:00+00:00", completed_at="2026-07-21T10:00:01+00:00")
    assert timeout_snapshot.account.status == "CALLBACK_TIMEOUT"
    assert timeout_snapshot.positions.status == "CALLBACK_TIMEOUT"


def test_same_and_all_api_open_orders_deduplicate_and_hash_sensitive_fields() -> None:
    state = BrokerObserverState("fixture-key")
    state.record_open_order(10, _contract(), _order(client_id=817, ref="strategy-ref"), _order_state("Submitted"))
    state.record_open_order(10, _contract(), _order(client_id=817, ref="strategy-ref"), _order_state("Submitted"))
    state.record_open_order_end()
    state.current_open_order_scope = "ALL_API_CLIENTS"  # type: ignore[assignment]
    state.record_open_order(11, _contract(), _order(client_id=99, ref="other-ref"), _order_state("PreSubmitted"))

    assert len(state.same_client_open_orders) == 1
    assert state.same_client_open_orders[0].order_ref_hash is not None
    assert len(state.all_api_open_orders) == 1
    assert any(error["code"] == "DUPLICATE_CALLBACK" for error in state.errors)


def test_executions_commissions_empty_scope_and_edge_cases(tmp_path) -> None:
    key = "fixture-key"
    state = BrokerObserverState(key)
    state.record_commission_report(_commission("EXEC-A"))
    state.record_exec_details(1, _contract(), _execution("EXEC-A"))
    state.record_exec_details(1, _contract(), _execution("EXEC-B"))
    state.record_exec_details_end(1)
    state.record_commission_report(_commission("EXEC-A"))
    snapshot = snapshot_from_state(state, _config(tmp_path), started_at="2026-07-21T10:00:00+00:00", completed_at="2026-07-21T10:00:01+00:00")

    assert len(snapshot.executions.executions) == 2
    assert len(snapshot.executions.commissions) == 2
    assert snapshot.executions.completeness_status == "CURRENT_SESSION_OBSERVED"
    assert snapshot.executions.execution_history_complete is False

    empty = BrokerObserverState(key)
    empty.record_exec_details_end(1)
    empty_snapshot = snapshot_from_state(empty, _config(tmp_path), started_at="2026-07-21T10:00:00+00:00", completed_at="2026-07-21T10:00:01+00:00")
    assert empty_snapshot.executions.status == "EMPTY_COMPLETE"
    assert empty_snapshot.executions.completeness_status == "NO_EXECUTIONS_RETURNED_WITHIN_REQUEST_SCOPE"


def test_stability_check_stable_changed_and_non_atomic(tmp_path) -> None:
    config = _config(tmp_path)
    a = snapshot_from_state(_full_state(config.account_fingerprint_key), config, started_at="2026-07-21T10:00:00+00:00", completed_at="2026-07-21T10:00:01+00:00")
    b = snapshot_from_state(_full_state(config.account_fingerprint_key), config, started_at="2026-07-21T10:00:02+00:00", completed_at="2026-07-21T10:00:03+00:00")
    changed_state = _full_state(config.account_fingerprint_key)
    changed_state.positions.clear()
    c = snapshot_from_state(changed_state, config, started_at="2026-07-21T10:00:04+00:00", completed_at="2026-07-21T10:00:05+00:00")

    assert stability_status(a, b)["stability_status"] == "BROKER_SNAPSHOT_STABLE_GO"
    assert stability_status(a, c)["stability_status"] == "STATE_CHANGED_DURING_CAPTURE"
    assert a.snapshot_atomic is False


def test_incomplete_snapshots_never_report_stable(tmp_path) -> None:
    config = _config(tmp_path)
    incomplete_state = BrokerObserverState(config.account_fingerprint_key)
    first = snapshot_from_state(
        incomplete_state,
        config,
        started_at="2026-07-21T10:00:00+00:00",
        completed_at="2026-07-21T10:00:01+00:00",
    )
    second = snapshot_from_state(
        incomplete_state,
        config,
        started_at="2026-07-21T10:00:02+00:00",
        completed_at="2026-07-21T10:00:03+00:00",
    )

    result = stability_status(first, second)
    assert snapshot_components_complete(first) is False
    assert result["stable"] is False
    assert result["stability_status"] == "SNAPSHOT_INCOMPLETE_BLOCKED"


def test_phase8_snapshot_fails_closed_on_callback_timeouts(
    tmp_path, monkeypatch
) -> None:
    config = _config(tmp_path)
    incomplete = snapshot_from_state(
        BrokerObserverState(config.account_fingerprint_key),
        config,
        started_at="2026-07-21T10:00:00+00:00",
        completed_at="2026-07-21T10:00:01+00:00",
    )
    monkeypatch.setattr(
        reconciliation_audit,
        "capture_snapshot",
        lambda _config: (
            incomplete,
            zero_read_counters(),
            zero_write_counters(),
        ),
    )

    result = reconciliation_audit.phase8_snapshot(
        tmp_path, tmp_path / ".env.ibkr"
    )
    assert result["status"] == "NO_GO"
    assert result["snapshot_status"] == "BROKER_SNAPSHOT_INCOMPLETE"
    assert result["completion_errors"] == [
        "CALLBACK_SEQUENCE_INCOMPLETE",
        "READ_REQUEST_SEQUENCE_INCOMPLETE",
    ]


def test_snapshot_capture_blocks_concurrent_observer(tmp_path) -> None:
    config = _config(tmp_path)
    lock_path = (
        tmp_path
        / "data"
        / "broker"
        / "phase8"
        / "private"
        / "observer.lock"
    )
    lock_path.parent.mkdir(parents=True)

    with FileLock(str(lock_path), timeout=0):
        with pytest.raises(
            Phase8Blocked, match="OBSERVER_SINGLE_FLIGHT_BUSY"
        ):
            capture_snapshot(config)


def test_phase7_ledger_read_only_sim_ids_external_state_and_mismatches(tmp_path) -> None:
    phase7_dir = tmp_path / "data" / "execution" / "phase7"
    phase7_dir.mkdir(parents=True)
    db = phase7_dir / "execution_ledger.sqlite3"
    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE intents(intent_id TEXT, payload_json TEXT)")
        conn.execute("CREATE TABLE fills(fill_id TEXT)")
        conn.execute("INSERT INTO intents VALUES (?, ?)", ("I1", json.dumps({"broker_order_id": "SIM-LOCAL"})))
        conn.execute("INSERT INTO fills VALUES (?)", ("SIM-FILL",))
        conn.commit()
    before = db.read_bytes()
    config = _config(tmp_path)
    broker = snapshot_from_state(_full_state(config.account_fingerprint_key), config, started_at="2026-07-21T10:00:00+00:00", completed_at="2026-07-21T10:00:01+00:00")
    result = compare_phase7_to_broker(tmp_path, broker)

    assert result["phase7_ledger_unchanged"] is True
    assert db.read_bytes() == before
    assert "SIM_ORDER_IDS_IGNORED" in result["classifications"]
    assert "OBSERVED_EXTERNAL_BROKER_STATE" in result["classifications"]
    assert result["automatic_corrections"] == 0
    assert result["reconciliation_gate"] == "BLOCKED"


def test_method_allowlist_blocks_writes_and_all_counters_zero() -> None:
    assert enforce_method_allowed("cancelAccountSummary") == "READ_ONLY_ALLOWED"
    assert enforce_method_allowed("cancelPositions") == "READ_ONLY_ALLOWED"
    for method in [
        "cancel" + "Order",
        "req" + "Global" + "Cancel",
        "req" + "Ids",
        "req" + "Auto" + "Open" + "Orders",
        "place" + "Order",
        "req" + "Mkt" + "Data",
        "req" + "Historical" + "Data",
    ]:
        try:
            enforce_method_allowed(method)
        except Phase8Blocked as exc:
            assert exc.code == "BROKER_WRITE_METHOD_BLOCKED"
        else:
            raise AssertionError(method)
    assert all(value == 0 for value in zero_write_counters().values())


def test_public_artifacts_do_not_contain_financial_values(tmp_path) -> None:
    layout = Phase8Layout.from_project_root(tmp_path)
    layout.output_dir.mkdir(parents=True)
    config = _config(tmp_path)
    snapshot = snapshot_from_state(_full_state(config.account_fingerprint_key), config, started_at="2026-07-21T10:00:00+00:00", completed_at="2026-07-21T10:00:01+00:00")
    private_hash = BrokerObservationStore(layout.private_db).write_snapshot(snapshot)
    public = public_snapshot_summary(snapshot, private_hash, layout.private_db)
    path = layout.artifact("summary.json")
    write_json(path, public)
    text = path.read_text(encoding="utf-8")

    assert "NetLiquidation" not in text
    assert "TotalCashValue" not in text
    assert "1000.00" not in text
    assert "101.25" not in text
    assert contains_raw_account(text) is False


def _env(tmp_path: Path, *, recon_client_id: int, key: str) -> Path:
    env = tmp_path / ".env.ibkr"
    env.write_text(
        "\n".join(
            [
                "APP_ENV=development",
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7497",
                "IBKR_CLIENT_ID=17",
                "IBKR_ACCOUNT=",
                "IBKR_READ_ONLY=true",
                "IBKR_ORDER_AUTHORITY=NONE",
                "IBKR_LIVE_TRADING_ENABLED=false",
                "IBKR_ALLOW_ORDER_TRANSMISSION=false",
                "IBKR_MARKET_DATA_TYPE=3",
                "IBKR_ALLOWED_SECURITY_TYPES=STK,FUT",
                "IBKR_ALLOWED_CURRENCIES=EUR,USD",
                "IBKR_MAX_ORDER_NOTIONAL_EUR=0",
                "IBKR_MAX_OPEN_ORDERS=0",
                "IBKR_MAX_POSITIONS=0",
                f"IBKR_RECON_CLIENT_ID={recon_client_id}",
                f"IBKR_ACCOUNT_FINGERPRINT_KEY={key}",
                "IBKR_RECON_REQUEST_TIMEOUT_SECONDS=1",
                "IBKR_RECON_COMMISSION_GRACE_SECONDS=1",
                "IBKR_RECON_SNAPSHOT_STABILITY_DELAY_SECONDS=1",
            ]
        ),
        encoding="utf-8",
    )
    return env


def _config(root: Path):
    env = _env(root, recon_client_id=817, key="fixture-key")
    config, errors = load_phase8_config(root, env)
    assert config is not None
    assert errors == []
    return config


def _full_state(key: str) -> BrokerObserverState:
    state = BrokerObserverState(key)
    state.record_current_time(1784637600)
    state.record_account_summary(1, "DU_TEST_ACCOUNT_001", "NetLiquidation", "1000.00", "EUR")
    state.record_account_summary(1, "DU_TEST_ACCOUNT_001", "AccountType", "INDIVIDUAL", "")
    state.record_account_summary_end(1)
    state.record_position("DU_TEST_ACCOUNT_001", _contract(), Decimal("2"), Decimal("100"))
    state.record_position_end()
    state.record_open_order(10, _contract(), _order(client_id=817, ref="strategy-ref"), _order_state("Submitted"))
    state.record_open_order_end()
    state.current_open_order_scope = "ALL_API_CLIENTS"  # type: ignore[assignment]
    state.open_order_end.clear()
    state.record_open_order(11, _contract(), _order(client_id=99, ref="other-ref"), _order_state("PreSubmitted"))
    state.record_open_order_end()
    state.record_exec_details(2, _contract(), _execution("EXEC-A"))
    state.record_commission_report(_commission("EXEC-A"))
    state.record_exec_details_end(2)
    return state


def _contract() -> SimpleNamespace:
    return SimpleNamespace(conId=756733, symbol="SPY", secType="STK", currency="USD", exchange="SMART")


def _order(client_id: int, ref: str) -> SimpleNamespace:
    return SimpleNamespace(
        permId=123,
        clientId=client_id,
        orderRef=ref,
        action="BUY",
        totalQuantity=Decimal("2"),
        orderType="LMT",
        lmtPrice=Decimal("101.25"),
        auxPrice=None,
        tif="DAY",
        outsideRth=False,
        parentId=0,
    )


def _order_state(status: str) -> SimpleNamespace:
    return SimpleNamespace(status=status, filled=Decimal("0"), remaining=Decimal("2"), avgFillPrice=Decimal("0"))


def _execution(exec_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        execId=exec_id,
        orderId=10,
        permId=123,
        clientId=817,
        acctNumber="DU_TEST_ACCOUNT_001",
        side="BOT",
        shares=Decimal("2"),
        price=Decimal("101.25"),
        cumQty=Decimal("2"),
        avgPrice=Decimal("101.25"),
        exchange="ARCA",
        time="20260721 10:00:00",
        liquidation=0,
        orderRef="strategy-ref",
    )


def _commission(exec_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        execId=exec_id,
        commission=Decimal("1.00"),
        currency="USD",
        realizedPNL=Decimal("0"),
        yield_=None,
        yieldRedemptionDate=0,
    )
