from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from stocks.application.config import IbkrSettings

from .callbacks import CallbackState
from .client import make_ibkr_app
from .health import ConnectionSnapshot, HealthStatus, build_snapshot, unique_errors


class IbkrAppProtocol(Protocol):
    def connect(self, host: str, port: int, client_id: int) -> None: ...
    def disconnect(self) -> None: ...
    def isConnected(self) -> bool: ...  # noqa: N802
    def serverVersion(self) -> Any: ...  # noqa: N802
    def twsConnectionTime(self) -> Any: ...  # noqa: N802
    def run(self) -> None: ...
    def reqCurrentTime(self) -> None: ...  # noqa: N802
    def reqManagedAccts(self) -> None: ...  # noqa: N802
    def reqContractDetails(self, req_id: int, contract: Any) -> None: ...  # noqa: N802
    def reqMatchingSymbols(self, req_id: int, pattern: str) -> None: ...  # noqa: N802
    def reqMarketRule(self, market_rule_id: int) -> None: ...  # noqa: N802


AppFactory = Callable[[CallbackState], IbkrAppProtocol]
Sleeper = Callable[[float], None]


class StatusArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def append(self, command: str, snapshot: ConnectionSnapshot) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        date_key = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self.output_dir / f"ibkr-status-{date_key}.jsonl"
        payload = {
            "command": command,
            **snapshot.as_dict(),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str))
            handle.write("\n")
        return path

    def append_report(self, command: str, report: dict[str, Any]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        date_key = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = self.output_dir / f"ibkr-status-{date_key}.jsonl"
        payload = {
            "command": command,
            **report,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str))
            handle.write("\n")
        return path

    def write_phase1_disconnect_drill_report(self, report: dict[str, Any]) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        generated_at = _parse_report_timestamp(report.get("generated_at"))
        timestamp = generated_at.strftime("%Y%m%d-%H%M%S")
        path = self.output_dir / f"phase1-disconnect-drill-{timestamp}.json"
        path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return path


class ReadOnlyIbkrConnectionService:
    def __init__(
        self,
        settings: IbkrSettings,
        *,
        app_factory: AppFactory = make_ibkr_app,
        sleeper: Sleeper = time.sleep,
        artifact_writer: StatusArtifactWriter | None = None,
    ) -> None:
        self.settings = settings
        self.app_factory = app_factory
        self.sleeper = sleeper
        self.artifact_writer = artifact_writer or StatusArtifactWriter(settings.output_dir)
        self.state = CallbackState()
        self.app: IbkrAppProtocol | None = None
        self.event_thread: threading.Thread | None = None
        self.reconnect_attempts = 0
        self.reconnect_failed = False
        self._thread_leak = False

    def connect_once(self) -> ConnectionSnapshot:
        self.state = CallbackState()
        self.app = self.app_factory(self.state)
        self.reconnect_failed = False
        self._thread_leak = False

        try:
            self.app.connect(
                self.settings.host,
                self.settings.port,
                self.settings.client_id,
            )
            if not self.app.isConnected():
                self.state.errors.append(
                    {
                        "code": "SOCKET_NOT_CONNECTED",
                        "message": "IBKR socket did not become connected.",
                    }
                )
                return self.snapshot()

            self.state.record_connected(
                server_version=self.app.serverVersion(),
                connection_time=self.app.twsConnectionTime(),
            )
            self.event_thread = threading.Thread(
                target=self.app.run,
                name=f"ibkr-read-only-client-{self.settings.client_id}",
                daemon=True,
            )
            self.event_thread.start()

            if not self.state.ready_event.wait(self.settings.connect_timeout_seconds):
                self.state.errors.append(
                    {
                        "code": "API_HANDSHAKE_TIMEOUT",
                        "message": "No nextValidId received before connect timeout.",
                    }
                )
                return self.snapshot()

            self.state.current_time_event.wait(self.settings.request_timeout_seconds)
            self.state.accounts_event.wait(self.settings.request_timeout_seconds)
            if self.state.last_server_utc is None:
                self.state.errors.append(
                    {
                        "code": "CURRENT_TIME_TIMEOUT",
                        "message": "No server clock received before request timeout.",
                    }
                )
            return self.snapshot()
        except Exception as exc:
            self.state.errors.append(
                {
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return self.snapshot()

    def connect_with_retries(self) -> ConnectionSnapshot:
        final_snapshot = self.connect_once()
        if final_snapshot.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}:
            return final_snapshot

        for attempt in range(1, self.settings.max_reconnect_attempts + 1):
            self.reconnect_attempts = attempt
            self.disconnect()
            delay = self._reconnect_delay(attempt)
            self.sleeper(delay)
            final_snapshot = self.connect_once()
            self.reconnect_attempts = attempt
            if final_snapshot.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}:
                return self.snapshot()

        self.reconnect_failed = True
        return self.snapshot()

    def heartbeat_once(self) -> ConnectionSnapshot:
        if self.app is None or not self.app.isConnected():
            self.state.record_closed()
            return self.snapshot()

        before = self.state.server_time_updates
        self.state.current_time_event.clear()
        try:
            self.app.reqCurrentTime()
        except Exception as exc:
            self.state.errors.append(
                {
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
            return self.snapshot()

        self.state.current_time_event.wait(self.settings.request_timeout_seconds)
        if self.state.server_time_updates <= before:
            self.state.errors.append(
                {
                    "code": "HEARTBEAT_TIMEOUT",
                    "message": "No server clock callback received for heartbeat.",
                }
            )
        return self.snapshot()

    def probe(self) -> ConnectionSnapshot:
        snapshot = self.connect_with_retries()
        self.artifact_writer.append("ibkr probe", snapshot)
        self.disconnect()
        return snapshot

    def status(self) -> ConnectionSnapshot:
        snapshot = self.connect_once()
        self.artifact_writer.append("ibkr status", snapshot)
        self.disconnect()
        return snapshot

    def watch(self, *, seconds: float) -> ConnectionSnapshot:
        deadline = time.monotonic() + max(0.0, seconds)
        snapshot = self.connect_with_retries()
        self.artifact_writer.append("ibkr watch start", snapshot)
        if snapshot.status not in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}:
            return snapshot

        while time.monotonic() < deadline:
            sleep_for = min(
                self.settings.heartbeat_interval_seconds,
                max(0.0, deadline - time.monotonic()),
            )
            if sleep_for > 0:
                self.sleeper(sleep_for)
            snapshot = self.heartbeat_once()
            self.artifact_writer.append("ibkr watch heartbeat", snapshot)
            if snapshot.status in {
                HealthStatus.STALE,
                HealthStatus.DISCONNECTED,
                HealthStatus.FAILED,
            }:
                snapshot = self.connect_with_retries()
                self.artifact_writer.append("ibkr watch reconnect", snapshot)
                if snapshot.status == HealthStatus.FAILED:
                    return snapshot

        final_snapshot = self.snapshot()
        self.artifact_writer.append("ibkr watch stop", final_snapshot)
        self.disconnect()
        return final_snapshot

    def run_connect_disconnect_cycles(self, *, count: int) -> dict[str, Any]:
        cycle_results: list[dict[str, Any]] = []
        thread_leaks = 0
        financial_calls = {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        }

        for index in range(1, count + 1):
            snapshot = self.connect_once()
            snapshot_dict = snapshot.as_dict()
            cycle_results.append({"cycle": index, **snapshot_dict})
            self.artifact_writer.append(f"ibkr cycle {index}", snapshot)
            for key in financial_calls:
                financial_calls[key] += snapshot.financial_calls.get(key, 0)
            self.disconnect()
            if self._thread_leak:
                thread_leaks += 1

        healthy_cycles = sum(
            1
            for item in cycle_results
            if item["status"] in {HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value}
        )
        status = "GO" if healthy_cycles == count and thread_leaks == 0 else "NO_GO"
        return {
            "schema": "ibkr_connect_disconnect_cycles_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "cycles_requested": count,
            "cycles_completed": len(cycle_results),
            "healthy_cycles": healthy_cycles,
            "thread_leaks": thread_leaks,
            "financial_calls": financial_calls,
            "failures": [
                item
                for item in cycle_results
                if item["status"] not in {HealthStatus.HEALTHY.value, HealthStatus.DEGRADED.value}
            ],
        }

    def duplicate_client_check(self) -> dict[str, Any]:
        primary_snapshot = self.connect_once()
        self.artifact_writer.append("ibkr duplicate primary", primary_snapshot)
        duplicate_state = CallbackState()
        duplicate_app = self.app_factory(duplicate_state)
        duplicate_thread: threading.Thread | None = None

        try:
            duplicate_app.connect(
                self.settings.host,
                self.settings.port,
                self.settings.client_id,
            )
            if duplicate_app.isConnected():
                duplicate_state.record_connected(
                    server_version=duplicate_app.serverVersion(),
                    connection_time=duplicate_app.twsConnectionTime(),
                )
                duplicate_thread = threading.Thread(
                    target=duplicate_app.run,
                    name=f"ibkr-duplicate-client-{self.settings.client_id}",
                    daemon=True,
                )
                duplicate_thread.start()

            deadline = time.monotonic() + self.settings.connect_timeout_seconds
            while time.monotonic() < deadline:
                if self._has_duplicate_client_error(self.state, duplicate_state):
                    break
                if duplicate_state.api_ready:
                    break
                self.sleeper(0.05)
        except Exception as exc:
            duplicate_state.errors.append(
                {
                    "code": type(exc).__name__,
                    "message": str(exc),
                }
            )
        finally:
            try:
                duplicate_app.disconnect()
            except Exception as exc:
                duplicate_state.errors.append(
                    {
                        "code": "DUPLICATE_DISCONNECT_ERROR",
                        "message": str(exc),
                    }
                )
            if duplicate_thread is not None:
                duplicate_thread.join(timeout=2.0)
            self.disconnect()

        duplicate_detected = self._has_duplicate_client_error(self.state, duplicate_state)
        duplicate_thread_leak = bool(
            duplicate_thread is not None and duplicate_thread.is_alive()
        )
        financial_calls = {
            key: self.state.financial_calls.get(key, 0)
            + duplicate_state.financial_calls.get(key, 0)
            for key in self.state.financial_calls
        }
        return {
            "schema": "ibkr_duplicate_client_check_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": "GO" if duplicate_detected and not duplicate_thread_leak else "NO_GO",
            "primary_status": primary_snapshot.status.value,
            "duplicate_client_id": self.settings.client_id,
            "duplicate_client_error_detected": duplicate_detected,
            "duplicate_errors": unique_errors(
                [*duplicate_state.errors, *duplicate_state.fatal_errors]
            ),
            "primary_errors": unique_errors([*self.state.errors, *self.state.fatal_errors]),
            "thread_leak": self._thread_leak or duplicate_thread_leak,
            "financial_calls": financial_calls,
        }

    def forced_disconnect_drill(
        self,
        *,
        seconds: float,
        poll_seconds: float = 2.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.0, seconds)
        disconnect_observed = False
        reconnect_successful = False
        observed_statuses: list[dict[str, Any]] = []
        financial_calls = {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        }

        initial_snapshot = self.connect_once()
        self.artifact_writer.append("ibkr disconnect-drill start", initial_snapshot)
        observed_statuses.append({"phase": "start", **initial_snapshot.as_dict()})
        for key in financial_calls:
            financial_calls[key] += initial_snapshot.financial_calls.get(key, 0)

        if initial_snapshot.status not in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}:
            report = self._forced_disconnect_report(
                status="NO_GO",
                seconds=seconds,
                poll_seconds=poll_seconds,
                disconnect_observed=disconnect_observed,
                reconnect_successful=reconnect_successful,
                observed_statuses=observed_statuses,
                financial_calls=financial_calls,
                failure_reason="INITIAL_SESSION_NOT_HEALTHY",
            )
            self.artifact_writer.append_report("ibkr disconnect-drill result", report)
            artifact_path = self.artifact_writer.write_phase1_disconnect_drill_report(report)
            report["artifact_path"] = str(artifact_path)
            self.disconnect()
            return report

        while time.monotonic() < deadline:
            self.sleeper(min(max(0.0, poll_seconds), max(0.0, deadline - time.monotonic())))
            snapshot = self.heartbeat_once()
            self.artifact_writer.append("ibkr disconnect-drill heartbeat", snapshot)
            observed_statuses.append({"phase": "heartbeat", **snapshot.as_dict()})
            for key in financial_calls:
                financial_calls[key] += snapshot.financial_calls.get(key, 0)

            if snapshot.status in {
                HealthStatus.STALE,
                HealthStatus.DISCONNECTED,
                HealthStatus.FAILED,
            }:
                disconnect_observed = True
                reconnect_snapshot = self.connect_with_retries()
                self.artifact_writer.append(
                    "ibkr disconnect-drill reconnect",
                    reconnect_snapshot,
                )
                observed_statuses.append(
                    {"phase": "reconnect", **reconnect_snapshot.as_dict()}
                )
                for key in financial_calls:
                    financial_calls[key] += reconnect_snapshot.financial_calls.get(key, 0)
                reconnect_successful = reconnect_snapshot.status in {
                    HealthStatus.HEALTHY,
                    HealthStatus.DEGRADED,
                }
                break

        final_status = "GO" if disconnect_observed and reconnect_successful else "NO_GO"
        report = self._forced_disconnect_report(
            status=final_status,
            seconds=seconds,
            poll_seconds=poll_seconds,
            disconnect_observed=disconnect_observed,
            reconnect_successful=reconnect_successful,
            observed_statuses=observed_statuses,
            financial_calls=financial_calls,
            failure_reason=_disconnect_drill_failure_reason(
                final_status=final_status,
                disconnect_observed=disconnect_observed,
                reconnect_successful=reconnect_successful,
            ),
        )
        self.artifact_writer.append_report("ibkr disconnect-drill result", report)
        artifact_path = self.artifact_writer.write_phase1_disconnect_drill_report(report)
        report["artifact_path"] = str(artifact_path)
        self.disconnect()
        return report

    def disconnect(self) -> None:
        app = self.app
        if app is not None:
            try:
                app.disconnect()
            except Exception as exc:
                self.state.errors.append(
                    {
                        "code": "DISCONNECT_ERROR",
                        "message": str(exc),
                    }
                )
        thread = self.event_thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread_leak = thread.is_alive()
        self.state.record_closed()

    def snapshot(self) -> ConnectionSnapshot:
        app_connected = bool(self.app is not None and self.app.isConnected())
        thread_alive = bool(self.event_thread is not None and self.event_thread.is_alive())
        return build_snapshot(
            state=self.state,
            connected=app_connected,
            event_thread_alive=thread_alive,
            stale_after_seconds=self.settings.stale_after_seconds,
            reconnect_attempts=self.reconnect_attempts,
            thread_leak=self._thread_leak,
            reconnect_failed=self.reconnect_failed,
        )

    def _reconnect_delay(self, attempt: int) -> float:
        if not self.settings.reconnect_delays_seconds:
            return 0.0
        index = min(max(attempt - 1, 0), len(self.settings.reconnect_delays_seconds) - 1)
        return self.settings.reconnect_delays_seconds[index]

    @staticmethod
    def _has_duplicate_client_error(*states: CallbackState) -> bool:
        for state in states:
            errors = [*state.errors, *state.fatal_errors]
            if any(error.get("code") == 326 for error in errors):
                return True
        return False

    def _forced_disconnect_report(
        self,
        *,
        status: str,
        seconds: float,
        poll_seconds: float,
        disconnect_observed: bool,
        reconnect_successful: bool,
        observed_statuses: list[dict[str, Any]],
        financial_calls: dict[str, int],
        failure_reason: str | None,
    ) -> dict[str, Any]:
        initial_health = _phase_status(observed_statuses, "start")
        final_health = _last_observed_status(observed_statuses)
        disconnect_event = _first_disconnect_event(observed_statuses)
        reconnect_event = _first_phase_event(observed_statuses, "reconnect")
        observed_errors = _observed_errors(observed_statuses)
        informational_messages = sum(
            int(item.get("informational_message_count", 0))
            for item in observed_statuses
        )
        thread_leak = any(bool(item.get("thread_leak")) for item in observed_statuses)
        reconnect_attempts = max(
            [int(item.get("reconnect_attempts", 0)) for item in observed_statuses],
            default=self.reconnect_attempts,
        )
        return {
            "schema": "ibkr_forced_disconnect_drill_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "initial_health": initial_health,
            "host": self.settings.host,
            "port": self.settings.port,
            "client_id": self.settings.client_id,
            "seconds": seconds,
            "poll_seconds": poll_seconds,
            "disconnect_detected": disconnect_observed,
            "disconnect_observed": disconnect_observed,
            "disconnect_detected_at": None if disconnect_event is None else disconnect_event.get("generated_at"),
            "stale_detected": any(item.get("status") == HealthStatus.STALE.value for item in observed_statuses),
            "disconnected_or_stale_state_seen": any(
                item.get("status") in {HealthStatus.DISCONNECTED.value, HealthStatus.STALE.value}
                for item in observed_statuses
            ),
            "bounded_reconnect_attempted": reconnect_event is not None,
            "reconnect_attempts": reconnect_attempts,
            "reconnect_delays": list(self.settings.reconnect_delays_seconds),
            "reconnect_recovered": reconnect_successful,
            "reconnect_successful": reconnect_successful,
            "recovered_at": None if not reconnect_successful or reconnect_event is None else reconnect_event.get("generated_at"),
            "final_health": final_health,
            "thread_leak": thread_leak,
            "informational_messages": informational_messages,
            "errors": observed_errors,
            "failure_reason": failure_reason,
            "observed_statuses": observed_statuses,
            "financial_calls": financial_calls,
        }


def _parse_report_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return datetime.now(timezone.utc)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.now(timezone.utc)


def _phase_status(observed_statuses: list[dict[str, Any]], phase: str) -> str | None:
    event = _first_phase_event(observed_statuses, phase)
    return None if event is None else str(event.get("status"))


def _last_observed_status(observed_statuses: list[dict[str, Any]]) -> str | None:
    if not observed_statuses:
        return None
    return str(observed_statuses[-1].get("status"))


def _first_disconnect_event(observed_statuses: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in observed_statuses:
        if item.get("status") in {
            HealthStatus.STALE.value,
            HealthStatus.DISCONNECTED.value,
            HealthStatus.FAILED.value,
        }:
            return item
    return None


def _first_phase_event(observed_statuses: list[dict[str, Any]], phase: str) -> dict[str, Any] | None:
    return next((item for item in observed_statuses if item.get("phase") == phase), None)


def _observed_errors(observed_statuses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for item in observed_statuses:
        for error in item.get("errors", []):
            if isinstance(error, dict):
                errors.append(dict(error))
    return unique_errors(errors)


def _disconnect_drill_failure_reason(
    *,
    final_status: str,
    disconnect_observed: bool,
    reconnect_successful: bool,
) -> str | None:
    if final_status == "GO":
        return None
    if not disconnect_observed:
        return "PHASE1_DISCONNECT_DRILL_NO_GO"
    if not reconnect_successful:
        return "PHASE1_RECONNECT_NO_GO"
    return "PHASE1_DISCONNECT_DRILL_NO_GO"
