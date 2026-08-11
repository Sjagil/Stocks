from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from stocks.application.config import IbkrSettings
from stocks.ibkr.callbacks import CallbackState, utc_now
from stocks.ibkr.connection import ReadOnlyIbkrConnectionService, StatusArtifactWriter


class HealthyFakeApp:
    def __init__(self, state: CallbackState, account_list: str | None = None) -> None:
        self.state = state
        self.account_list = account_list or ("DU" + "1234567")
        self.connected = False

    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.state.record_closed()

    def isConnected(self) -> bool:  # noqa: N802
        return self.connected

    def serverVersion(self) -> int:  # noqa: N802
        return 225

    def twsConnectionTime(self) -> str:  # noqa: N802
        return "20260720 00:00:00"

    def run(self) -> None:
        self.state.record_next_valid_id(1001)
        self.reqCurrentTime()
        self.state.record_managed_accounts(self.account_list)
        while self.connected:
            time.sleep(0.001)

    def reqCurrentTime(self) -> None:  # noqa: N802
        self.state.record_current_time(1784592000)

    def reqManagedAccts(self) -> None:  # noqa: N802
        self.state.record_managed_accounts(self.account_list)


class SocketFailFakeApp(HealthyFakeApp):
    def connect(self, host: str, port: int, client_id: int) -> None:
        self.connected = False


class DuplicateClientFakeApp(HealthyFakeApp):
    def run(self) -> None:
        self.state.record_error(1, (326, "Client ID is already in use."))
        self.connected = False


class DisconnectOnHeartbeatFakeApp(HealthyFakeApp):
    def reqCurrentTime(self) -> None:  # noqa: N802
        if self.state.server_time_updates == 0:
            self.state.record_current_time(1784592000)
            return
        self.connected = False
        self.state.record_closed()


def settings(tmp_path: Path) -> IbkrSettings:
    return IbkrSettings(
        output_dir=tmp_path,
        connect_timeout_seconds=0.05,
        request_timeout_seconds=0.05,
        heartbeat_interval_seconds=0.01,
        stale_after_seconds=45.0,
        reconnect_delays_seconds=(2.0, 5.0, 15.0, 30.0),
        max_reconnect_attempts=5,
    )


def test_connect_disconnect_cycle_is_healthy_and_leak_free(tmp_path: Path) -> None:
    for _ in range(25):
        service = ReadOnlyIbkrConnectionService(
            settings(tmp_path),
            app_factory=lambda state: HealthyFakeApp(state),
        )
        snapshot = service.connect_once()
        assert snapshot.status.value == "HEALTHY"
        assert snapshot.managed_account_count == 1
        assert snapshot.financial_calls == {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        }

        service.disconnect()
        after_disconnect = service.snapshot()
        assert after_disconnect.thread_leak is False
        assert after_disconnect.connected is False


def test_cycle_report_covers_requested_count(tmp_path: Path) -> None:
    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=lambda state: HealthyFakeApp(state),
    )

    report = service.run_connect_disconnect_cycles(count=25)

    assert report["status"] == "GO"
    assert report["cycles_requested"] == 25
    assert report["cycles_completed"] == 25
    assert report["healthy_cycles"] == 25
    assert report["thread_leaks"] == 0
    assert report["financial_calls"] == {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }
    assert report["failures"] == []


def test_duplicate_client_id_is_failed(tmp_path: Path) -> None:
    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=lambda state: DuplicateClientFakeApp(state),
    )

    snapshot = service.connect_once()

    assert snapshot.status.value == "FAILED"
    assert any(error["code"] == 326 for error in snapshot.errors)


def test_duplicate_client_check_reports_go_on_error_326(tmp_path: Path) -> None:
    calls = 0

    def factory(state: CallbackState) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return HealthyFakeApp(state)
        return DuplicateClientFakeApp(state)

    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=factory,
        sleeper=lambda seconds: None,
    )

    report = service.duplicate_client_check()

    assert report["status"] == "GO"
    assert report["duplicate_client_error_detected"] is True
    assert report["duplicate_client_id"] == 17
    assert report["thread_leak"] is False
    assert report["financial_calls"] == {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }


def test_reconnect_is_bounded_and_can_recover(tmp_path: Path) -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def factory(state: CallbackState) -> Any:
        attempts.append("attempt")
        if len(attempts) < 3:
            return SocketFailFakeApp(state)
        return HealthyFakeApp(state)

    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=factory,
        sleeper=sleeps.append,
    )

    snapshot = service.connect_with_retries()

    assert snapshot.status.value == "HEALTHY"
    assert len(attempts) == 3
    assert sleeps == [2.0, 5.0]
    assert snapshot.reconnect_attempts == 2
    service.disconnect()


def test_reconnect_fails_closed_after_max_attempts(tmp_path: Path) -> None:
    sleeps: list[float] = []
    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=lambda state: SocketFailFakeApp(state),
        sleeper=sleeps.append,
    )

    snapshot = service.connect_with_retries()

    assert snapshot.status.value == "FAILED"
    assert snapshot.reconnect_attempts == 5
    assert sleeps == [2.0, 5.0, 15.0, 30.0, 30.0]


def test_status_is_single_attempt_when_socket_is_unavailable(tmp_path: Path) -> None:
    sleeps: list[float] = []
    attempts = 0

    def factory(state: CallbackState) -> Any:
        nonlocal attempts
        attempts += 1
        return SocketFailFakeApp(state)

    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=factory,
        sleeper=sleeps.append,
        artifact_writer=StatusArtifactWriter(tmp_path),
    )

    snapshot = service.status()

    assert snapshot.status.value == "DISCONNECTED"
    assert attempts == 1
    assert sleeps == []
    assert snapshot.financial_calls == {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }


def test_forced_disconnect_is_detected(tmp_path: Path) -> None:
    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=lambda state: HealthyFakeApp(state),
    )
    assert service.connect_once().status.value == "HEALTHY"

    assert isinstance(service.app, HealthyFakeApp)
    service.app.disconnect()
    snapshot = service.heartbeat_once()

    assert snapshot.status.value == "DISCONNECTED"
    assert snapshot.connected is False
    assert snapshot.thread_leak is False


def test_stale_connection_is_detected(tmp_path: Path) -> None:
    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=lambda state: HealthyFakeApp(state),
    )
    assert service.connect_once().status.value == "HEALTHY"

    service.state.last_heartbeat_at = utc_now() - timedelta(seconds=90)
    snapshot = service.snapshot()

    assert snapshot.status.value == "STALE"
    service.disconnect()


def test_status_artifact_masks_account_identifiers(tmp_path: Path) -> None:
    account = "DU" + "9999999"
    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=lambda state: HealthyFakeApp(state, account_list=account),
        artifact_writer=StatusArtifactWriter(tmp_path),
    )

    snapshot = service.probe()
    files = list(tmp_path.glob("ibkr-status-*.jsonl"))

    assert snapshot.financial_calls["place_order"] == 0
    assert files
    artifact_text = files[0].read_text(encoding="utf-8")
    assert account not in artifact_text
    payload = json.loads(artifact_text.splitlines()[0])
    assert payload["managed_account_count"] == 1


def test_forced_disconnect_drill_reports_go_after_reconnect(tmp_path: Path) -> None:
    calls = 0

    def factory(state: CallbackState) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return DisconnectOnHeartbeatFakeApp(state)
        return HealthyFakeApp(state)

    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=factory,
        sleeper=lambda seconds: None,
    )

    report = service.forced_disconnect_drill(seconds=1.0, poll_seconds=0.01)

    assert report["status"] == "GO"
    assert report["host"] == "127.0.0.1"
    assert report["port"] == 7497
    assert report["client_id"] == 17
    assert report["disconnect_observed"] is True
    assert report["disconnect_detected"] is True
    assert report["disconnect_detected_at"] is not None
    assert report["disconnected_or_stale_state_seen"] is True
    assert report["bounded_reconnect_attempted"] is True
    assert report["reconnect_recovered"] is True
    assert report["reconnect_successful"] is True
    assert report["recovered_at"] is not None
    assert report["initial_health"] == "HEALTHY"
    assert report["final_health"] == "HEALTHY"
    assert report["thread_leak"] is False
    assert report["financial_calls"] == {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }
    artifact = Path(report["artifact_path"])
    assert artifact.parent == tmp_path
    assert artifact.name.startswith("phase1-disconnect-drill-")
    artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert artifact_payload["schema"] == "ibkr_forced_disconnect_drill_v1"
    assert artifact_payload["status"] == "GO"
    assert artifact_payload["disconnect_detected"] is True
    assert artifact_payload["reconnect_recovered"] is True
    assert artifact_payload["financial_calls"]["place_order"] == 0


def test_forced_disconnect_drill_reports_no_go_without_disconnect(tmp_path: Path) -> None:
    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=lambda state: HealthyFakeApp(state),
        sleeper=lambda seconds: None,
    )

    report = service.forced_disconnect_drill(seconds=0.01, poll_seconds=0.01)

    assert report["status"] == "NO_GO"
    assert report["disconnect_observed"] is False
    assert report["disconnect_detected"] is False
    assert report["disconnect_detected_at"] is None
    assert report["bounded_reconnect_attempted"] is False
    assert report["reconnect_recovered"] is False
    assert report["reconnect_successful"] is False
    assert report["final_health"] == "HEALTHY"
    assert report["failure_reason"] == "PHASE1_DISCONNECT_DRILL_NO_GO"
    assert Path(report["artifact_path"]).exists()


def test_forced_disconnect_drill_reports_reconnect_no_go_after_disconnect(
    tmp_path: Path,
) -> None:
    calls = 0

    def factory(state: CallbackState) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return DisconnectOnHeartbeatFakeApp(state)
        return SocketFailFakeApp(state)

    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=factory,
        sleeper=lambda seconds: None,
    )

    report = service.forced_disconnect_drill(seconds=1.0, poll_seconds=0.01)

    assert report["status"] == "NO_GO"
    assert report["disconnect_detected"] is True
    assert report["bounded_reconnect_attempted"] is True
    assert report["reconnect_recovered"] is False
    assert report["recovered_at"] is None
    assert report["failure_reason"] == "PHASE1_RECONNECT_NO_GO"
    assert Path(report["artifact_path"]).exists()


def test_forced_disconnect_drill_fails_fast_when_initial_session_unhealthy(
    tmp_path: Path,
) -> None:
    sleeps: list[float] = []
    service = ReadOnlyIbkrConnectionService(
        settings(tmp_path),
        app_factory=lambda state: DuplicateClientFakeApp(state),
        sleeper=sleeps.append,
    )

    report = service.forced_disconnect_drill(seconds=1.0, poll_seconds=0.01)

    assert report["status"] == "NO_GO"
    assert report["failure_reason"] == "INITIAL_SESSION_NOT_HEALTHY"
    assert report["disconnect_observed"] is False
    assert report["initial_health"] == "FAILED"
    assert report["final_health"] == "FAILED"
    assert report["reconnect_recovered"] is False
    assert sleeps == []
    assert Path(report["artifact_path"]).exists()
