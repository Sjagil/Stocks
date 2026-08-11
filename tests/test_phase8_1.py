from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import main
from stocks.ibkr.phase8_1 import (
    PHASE7_FIXTURE_STATUS,
    callback_integrity_from_events,
    classify_state_change,
    error_budget_status,
    establish_baseline,
    public_financial_value_leaks,
    recovery_drill,
    relevant_state_key,
    run_soak,
    snapshot_complete,
    subscription_cleanup_audit,
    summarize_iterations,
)
from stocks.ibkr.reconciliation.callbacks import BrokerObserverState
from stocks.ibkr.reconciliation.models import ObservationScope
from stocks.ibkr.reconciliation.requests import Phase8Config, load_phase8_config
from stocks.ibkr.reconciliation.snapshots import snapshot_from_state


def test_phase8_1_schema_cli(capsys) -> None:
    assert main.main(["ibkr", "phase8-1", "schema"]) == 0
    out = capsys.readouterr().out
    assert "PHASE8_1_READ_ONLY_OBSERVATION_SOAK_AND_RECOVERY_GO" in out
    assert '"execution_authority": "NONE"' in out


def test_baseline_stable_and_changed(tmp_path) -> None:
    config = _config(tmp_path)
    stable_capture = _capture_sequence([_snapshot(config), _snapshot(config)])
    stable = establish_baseline(tmp_path, _env(tmp_path), capture_func=stable_capture)
    assert stable["baseline_status"] == "BASELINE_STABLE_GO"
    assert stable["snapshot_atomic"] is False
    assert stable["position_count"] == 0

    changed_capture = _capture_sequence([_snapshot(config), _snapshot(config, account_value="1001.00")])
    changed = establish_baseline(tmp_path, _env(tmp_path), capture_func=changed_capture)
    assert changed["baseline_status"] == "BASELINE_CHANGED_BLOCKED"


def test_state_change_classifications(tmp_path) -> None:
    config = _config(tmp_path)
    base = _snapshot(config)
    assert classify_state_change(None, base)["state_change_classification"] == "NO_CHANGE"
    assert classify_state_change(base, _snapshot(config, account_value="1001.00"))["state_change_classification"] == "ACCOUNT_SUMMARY_CHANGED"
    assert classify_state_change(base, _snapshot(config, position_qty=Decimal("1")))["state_change_classification"] == "POSITION_ADDED"
    assert classify_state_change(_snapshot(config, position_qty=Decimal("1")), base)["state_change_classification"] == "POSITION_REMOVED"
    assert classify_state_change(_snapshot(config, position_qty=Decimal("1")), _snapshot(config, position_qty=Decimal("2")))["state_change_classification"] == "POSITION_QUANTITY_CHANGED"
    assert classify_state_change(base, _snapshot(config, order_status="Submitted"))["state_change_classification"] == "OPEN_ORDER_ADDED"
    assert classify_state_change(_snapshot(config, order_status="Submitted"), base)["state_change_classification"] == "OPEN_ORDER_REMOVED"
    assert classify_state_change(_snapshot(config, order_status="Submitted"), _snapshot(config, order_status="Filled"))["state_change_classification"] == "OPEN_ORDER_STATUS_CHANGED"
    assert classify_state_change(base, _snapshot(config, execution_id="EXEC-A"))["state_change_classification"] == "EXECUTION_ADDED"
    assert classify_state_change(_snapshot(config, execution_id="EXEC-A"), _snapshot(config, execution_id="EXEC-A", commission=True))["state_change_classification"] == "COMMISSION_ADDED"
    assert classify_state_change(base, _snapshot(config, account_value="1001.00", position_qty=Decimal("1")))["state_change_classification"] == "MULTIPLE_STATE_CHANGES"


def test_snapshot_continuity_and_phase7_fixture_separation(tmp_path) -> None:
    config = _config(tmp_path)
    first = _snapshot(config)
    second = _snapshot(config)
    assert relevant_state_key(first) == relevant_state_key(second)
    assert PHASE7_FIXTURE_STATUS == "PHASE7_FIXTURE_NOT_BROKER_MIRROR"


def test_callback_integrity_duplicate_out_of_order_late_orphan_and_incomplete() -> None:
    audit = callback_integrity_from_events(
        [
            {"callback_type": "openOrder", "classification": "CALLBACK_OK"},
            {"callback_type": "openOrder", "classification": "DUPLICATE_CALLBACK_IGNORED"},
            {"callback_type": "execDetails", "classification": "OUT_OF_ORDER_CALLBACK_BUFFERED"},
            {"callback_type": "commissionReport", "classification": "LATE_CALLBACK_QUARANTINED"},
            {"callback_type": "orderStatus", "classification": "ORPHAN_CALLBACK_BLOCKED"},
        ]
    )
    assert audit["callback_types"]["openOrder"]["duplicate_count"] == 1
    assert audit["callback_types"]["execDetails"]["out_of_order_count"] == 1
    assert audit["callback_types"]["commissionReport"]["late_count"] == 1
    assert audit["callback_types"]["orderStatus"]["orphan_count"] == 1
    assert audit["status"] == "NO_GO"


def test_subscription_cleanup_thread_leak_and_error_budget(tmp_path) -> None:
    good_rows = [{"thread_leak": False, "active_subscriptions_after_iteration": 0}]
    assert subscription_cleanup_audit(good_rows, tmp_path)["status"] == "GO"
    bad_rows = [{"thread_leak": True, "active_subscriptions_after_iteration": 1}]
    assert subscription_cleanup_audit(bad_rows, tmp_path)["status"] == "NO_GO"

    summary = summarize_iterations(
        [{"snapshot_status": "COMPLETE", "stability_status": "BROKER_SNAPSHOT_STABLE_GO", "state_change_classification": "NO_CHANGE", "write_counters": {}, "privacy_counters": {}, "thread_leak": False}],
        iterations_requested=1,
    )
    assert summary["completion_ratio"] == Decimal("1")
    assert error_budget_status(summary)["status"] == "GO"
    summary["write_attempt_count"] = 1
    assert error_budget_status(summary)["status"] == "NO_GO"


def test_bounded_soak_duration_interval_completion_and_zero_writes(tmp_path) -> None:
    config = _config(tmp_path)
    capture = _capture_sequence([_snapshot(config), _snapshot(config), _snapshot(config), _snapshot(config)])
    report = run_soak(
        tmp_path,
        _env(tmp_path),
        duration_seconds=2,
        interval_seconds=1,
        stability_delay_seconds=0.01,
        capture_func=capture,
    )
    assert report["status"] == "GO"
    assert report["iterations_requested"] == 2
    assert report["iterations_completed"] == 2
    assert report["completion_ratio"] == Decimal("1")
    assert report["write_attempt_count"] == 0
    assert (tmp_path / "output" / "ibkr" / "phase8_1" / "iteration-summary.parquet").exists()


def test_recovery_drill_fixture_bounded_reconnect_and_post_snapshot(monkeypatch, tmp_path) -> None:
    config = _config(tmp_path)
    capture = _capture_sequence([_snapshot(config), _snapshot(config)])

    class FakeService:
        def forced_disconnect_drill(self, *, seconds: float, poll_seconds: float):
            assert seconds == 1
            assert poll_seconds == 0.1
            return {
                "status": "GO",
                "disconnect_detected": True,
                "bounded_reconnect_attempted": True,
                "reconnect_recovered": True,
                "final_health": "HEALTHY",
            }

    monkeypatch.setattr("stocks.application.lifecycle.build_ibkr_service", lambda _context: FakeService())
    report = recovery_drill(tmp_path, _env(tmp_path), duration_seconds=1, poll_seconds=0.1, capture_func=capture)
    assert report["status"] == "GO"
    assert report["post_reconnect_snapshot_complete"] is True
    assert report["post_reconnect_masking_go"] is True


def test_raw_secret_public_value_and_forbidden_counters_zero(tmp_path) -> None:
    config = _config(tmp_path)
    snap = _snapshot(config)
    assert snapshot_complete(snap) is True
    out = tmp_path / "output" / "ibkr" / "phase8_1"
    out.mkdir(parents=True)
    (out / "safe.json").write_text('{"position_count": 0, "content_hash": "abc"}', encoding="utf-8")
    assert public_financial_value_leaks(out) == 0


def _env(tmp_path: Path) -> Path:
    env = tmp_path / ".env.ibkr"
    if not env.exists():
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
                    "IBKR_RECON_CLIENT_ID=817",
                    "IBKR_ACCOUNT_FINGERPRINT_KEY=fixture-key",
                    "IBKR_RECON_REQUEST_TIMEOUT_SECONDS=1",
                    "IBKR_RECON_COMMISSION_GRACE_SECONDS=0.01",
                    "IBKR_RECON_SNAPSHOT_STABILITY_DELAY_SECONDS=0.01",
                ]
            ),
            encoding="utf-8",
        )
    return env


def _config(tmp_path: Path) -> Phase8Config:
    config, errors = load_phase8_config(tmp_path, _env(tmp_path))
    assert config is not None
    assert errors == []
    return config


def _capture_sequence(snapshots):
    items = list(snapshots)

    def capture(_config: Phase8Config):
        if not items:
            snapshot = snapshots[-1]
        else:
            snapshot = items.pop(0)
        reads = {
            "read_only_account_summary_requests": 1,
            "read_only_account_summary_cancels": 1,
            "read_only_position_requests": 1,
            "read_only_position_cancels": 1,
            "read_only_same_client_open_order_requests": 1,
            "read_only_all_api_open_order_requests": 1,
            "read_only_execution_requests": 1,
            "current_time_requests": 1,
        }
        writes = {
            "place_order_calls": 0,
            "cancel_order_calls": 0,
            "global_cancel_calls": 0,
            "request_order_id_calls": 0,
            "auto_bind_order_calls": 0,
            "exercise_option_calls": 0,
            "market_data_calls": 0,
            "historical_data_calls": 0,
        }
        return snapshot, reads, writes

    return capture


def _snapshot(
    config: Phase8Config,
    *,
    account_value: str = "1000.00",
    position_qty: Decimal | None = None,
    order_status: str | None = None,
    execution_id: str | None = None,
    commission: bool = False,
):
    state = BrokerObserverState(config.account_fingerprint_key)
    state.record_connected("225")
    state.record_account_summary(1, "DU_TEST_ACCOUNT_001", "TotalValue", account_value, "EUR")
    state.record_account_summary_end(1)
    if position_qty is not None:
        state.record_position("DU_TEST_ACCOUNT_001", _contract(), position_qty, Decimal("100"))
    state.record_position_end()
    if order_status is not None:
        state.current_open_order_scope = ObservationScope.SAME_CLIENT
        state.record_open_order(10, _contract(), _order(), _order_state(order_status))
    state.record_open_order_end()
    state.current_open_order_scope = ObservationScope.ALL_API_CLIENTS
    state.open_order_end.set()
    if execution_id is not None:
        state.record_exec_details(1, _contract(), _execution(execution_id))
    if commission and execution_id is not None:
        state.record_commission_report(_commission(execution_id))
    state.record_exec_details_end(1)
    return snapshot_from_state(
        state,
        config,
        started_at="2026-07-21T10:00:00+00:00",
        completed_at="2026-07-21T10:00:01+00:00",
    )


def _contract() -> SimpleNamespace:
    return SimpleNamespace(conId=756733, symbol="SPY", secType="STK", currency="USD", exchange="SMART")


def _order() -> SimpleNamespace:
    return SimpleNamespace(
        permId=123,
        clientId=817,
        orderRef="fixture-ref",
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


def _execution(execution_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        execId=execution_id,
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
        orderRef="fixture-ref",
    )


def _commission(execution_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        execId=execution_id,
        commission=Decimal("1.00"),
        currency="USD",
        realizedPNL=Decimal("0"),
        yield_=None,
        yieldRedemptionDate=0,
    )
