from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .callbacks import CallbackState


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    STALE = "STALE"
    DISCONNECTED = "DISCONNECTED"
    FAILED = "FAILED"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ConnectionSnapshot:
    schema: str
    generated_at: str
    status: HealthStatus
    connected: bool
    api_ready: bool
    event_thread_alive: bool
    server_version: int | str | None
    connection_time: str | None
    last_server_utc: str | None
    last_heartbeat_age_seconds: float | None
    managed_account_count: int
    reconnect_attempts: int
    errors: list[dict[str, Any]]
    informational_message_count: int
    degraded_message_count: int
    financial_calls: dict[str, int]
    thread_leak: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generated_at": self.generated_at,
            "status": self.status.value,
            "connected": self.connected,
            "api_ready": self.api_ready,
            "event_thread_alive": self.event_thread_alive,
            "server_version": self.server_version,
            "connection_time": self.connection_time,
            "last_server_utc": self.last_server_utc,
            "last_heartbeat_age_seconds": self.last_heartbeat_age_seconds,
            "managed_account_count": self.managed_account_count,
            "reconnect_attempts": self.reconnect_attempts,
            "errors": self.errors,
            "informational_message_count": self.informational_message_count,
            "degraded_message_count": self.degraded_message_count,
            "financial_calls": self.financial_calls,
            "thread_leak": self.thread_leak,
        }


def heartbeat_age_seconds(state: CallbackState, now: datetime) -> float | None:
    if state.last_heartbeat_at is None:
        return None
    return max(0.0, (now - state.last_heartbeat_at).total_seconds())


def classify_health(
    *,
    state: CallbackState,
    connected: bool,
    event_thread_alive: bool,
    stale_after_seconds: float,
    reconnect_failed: bool = False,
) -> HealthStatus:
    if reconnect_failed or state.fatal_errors:
        return HealthStatus.FAILED
    if not connected or state.connection_closed:
        return HealthStatus.DISCONNECTED
    if not event_thread_alive or not state.api_ready:
        return HealthStatus.DISCONNECTED

    age = heartbeat_age_seconds(state, utc_now())
    if age is None or age > stale_after_seconds:
        return HealthStatus.STALE
    if state.degraded_messages:
        return HealthStatus.DEGRADED
    if state.errors:
        return HealthStatus.FAILED
    return HealthStatus.HEALTHY


def build_snapshot(
    *,
    state: CallbackState,
    connected: bool,
    event_thread_alive: bool,
    stale_after_seconds: float,
    reconnect_attempts: int,
    thread_leak: bool = False,
    reconnect_failed: bool = False,
) -> ConnectionSnapshot:
    now = utc_now()
    return ConnectionSnapshot(
        schema="ibkr_connection_service_status_v1",
        generated_at=now.isoformat(),
        status=classify_health(
            state=state,
            connected=connected,
            event_thread_alive=event_thread_alive,
            stale_after_seconds=stale_after_seconds,
            reconnect_failed=reconnect_failed,
        ),
        connected=connected,
        api_ready=state.api_ready,
        event_thread_alive=event_thread_alive,
        server_version=state.server_version,
        connection_time=state.connection_time,
        last_server_utc=state.last_server_utc,
        last_heartbeat_age_seconds=heartbeat_age_seconds(state, now),
        managed_account_count=state.managed_account_count,
        reconnect_attempts=reconnect_attempts,
        errors=unique_errors([*state.errors, *state.fatal_errors]),
        informational_message_count=len(state.informational_messages),
        degraded_message_count=len(state.degraded_messages),
        financial_calls=dict(state.financial_calls),
        thread_leak=thread_leak,
    )


def unique_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, str]] = set()
    unique: list[dict[str, Any]] = []
    for error in errors:
        key = (
            error.get("req_id"),
            error.get("code"),
            str(error.get("message")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(error)
    return unique
