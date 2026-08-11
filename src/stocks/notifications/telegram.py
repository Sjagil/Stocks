from __future__ import annotations

import hashlib
import html
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import dotenv_values

from stocks.notifications.market_digest import (
    build_market_intelligence_digest,
    format_market_intelligence_digest,
)
from stocks.signals.top5 import publish_top_signals


QUEUE_STATUSES = {
    "PENDING",
    "SENDING",
    "SENT",
    "RETRY_PENDING",
    "FAILED_FINAL",
    "SKIPPED_DUPLICATE",
    "SKIPPED_FILTER",
    "SKIPPED_STALE",
}
ACTIONABLE = {"BUY", "STRONG_BUY"}
EXITS = {"EXIT", "REDUCE"}
FUTURE_REQUIRED_FIELDS = {
    "con_id",
    "expiry",
    "multiplier",
    "minimum_tick",
    "tick_value",
    "rollover_status",
    "margin_eur",
    "first_notice_date",
    "last_trade_date",
}
MAX_QUEUE_ROWS = 10_000
# Foreground machine steps have a 60-second supervisor deadline. Processing
# one durable queue item avoids rate-limit sleeps caused by unrelated backlog;
# later steps/cycles continue draining the same queue without losing messages.
FOREGROUND_QUEUE_LIMIT = 1


def _bool(raw: Any, default: bool) -> bool:
    if raw is None or str(raw).strip() == "":
        return default
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected true/false")


def _int(raw: Any, default: int, minimum: int, maximum: int) -> int:
    value = default if raw is None or str(raw).strip() == "" else int(str(raw))
    if not minimum <= value <= maximum:
        raise ValueError(f"value must be between {minimum} and {maximum}")
    return value


def _float(raw: Any, default: float, minimum: float, maximum: float) -> float:
    value = default if raw is None or str(raw).strip() == "" else float(str(raw))
    if not minimum <= value <= maximum:
        raise ValueError(f"value must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class TelegramSettings:
    enabled: bool
    token: str
    chat_id: str
    send_signals: bool
    send_watchlist: bool
    send_exits: bool
    send_order_events: bool
    send_risk_alerts: bool
    send_system_alerts: bool
    send_autopilot_summary: bool
    send_market_digest: bool
    minimum_confidence: float
    minimum_reward_risk: float
    max_messages_per_minute: int
    request_timeout_seconds: float
    max_retries: int
    dry_run: bool
    include_position_size: bool
    include_fundamentals: bool
    include_macro_context: bool

    @property
    def configured(self) -> bool:
        return bool(self.token and self.chat_id)

    @property
    def masked_chat_identity(self) -> str:
        if not self.chat_id:
            return "not-configured"
        digest = hashlib.sha256(self.chat_id.encode()).hexdigest()[:12]
        return f"chat-sha256:{digest}"

    @property
    def public_status(self) -> str:
        if not self.enabled:
            return "DISABLED_BY_CONFIG"
        if not self.configured:
            return "DISABLED_MISSING_CONFIG"
        if self.dry_run:
            return "DRY_RUN"
        return "ENABLED"

    def safe_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("token")
        payload.pop("chat_id")
        payload["configured"] = self.configured
        payload["masked_chat_identity"] = self.masked_chat_identity
        payload["status"] = self.public_status
        return payload


def load_telegram_settings(
    project_root: Path, env: dict[str, str] | None = None
) -> TelegramSettings:
    file_values = dotenv_values(project_root / ".env")
    supplied = env or {}

    def get(name: str, default: str = "") -> str:
        if env is not None:
            return supplied.get(name, default)
        if name in supplied:
            return supplied[name]
        if name in os.environ:
            return os.environ[name]
        value = file_values.get(name)
        return default if value is None else str(value)

    return TelegramSettings(
        enabled=_bool(get("TELEGRAM_NOTIFICATIONS_ENABLED"), False),
        token=get("TELEGRAM_BOT_TOKEN").strip(),
        chat_id=get("TELEGRAM_CHAT_ID").strip(),
        send_signals=_bool(get("TELEGRAM_SEND_SIGNALS"), True),
        send_watchlist=_bool(get("TELEGRAM_SEND_WATCHLIST"), True),
        send_exits=_bool(get("TELEGRAM_SEND_EXITS"), True),
        send_order_events=_bool(get("TELEGRAM_SEND_ORDER_EVENTS"), True),
        send_risk_alerts=_bool(get("TELEGRAM_SEND_RISK_ALERTS"), True),
        send_system_alerts=_bool(get("TELEGRAM_SEND_SYSTEM_ALERTS"), True),
        send_autopilot_summary=_bool(
            get("TELEGRAM_SEND_AUTOPILOT_SUMMARY"), True
        ),
        send_market_digest=_bool(
            get("TELEGRAM_SEND_MARKET_DIGEST"), True
        ),
        minimum_confidence=_float(
            get("TELEGRAM_MIN_CONFIDENCE"), 60.0, 0.0, 100.0
        ),
        minimum_reward_risk=_float(
            get("TELEGRAM_MIN_REWARD_RISK"), 1.5, 0.0, 100.0
        ),
        max_messages_per_minute=_int(
            get("TELEGRAM_MAX_MESSAGES_PER_MINUTE"), 10, 1, 60
        ),
        request_timeout_seconds=_float(
            get("TELEGRAM_REQUEST_TIMEOUT_SECONDS"), 10.0, 1.0, 60.0
        ),
        max_retries=_int(get("TELEGRAM_MAX_RETRIES"), 3, 0, 10),
        dry_run=_bool(get("TELEGRAM_DRY_RUN"), False),
        include_position_size=_bool(
            get("TELEGRAM_INCLUDE_POSITION_SIZE"), True
        ),
        include_fundamentals=_bool(
            get("TELEGRAM_INCLUDE_FUNDAMENTALS"), True
        ),
        include_macro_context=_bool(
            get("TELEGRAM_INCLUDE_MACRO_CONTEXT"), True
        ),
    )


class TelegramQueue:
    def __init__(self, project_root: Path) -> None:
        path = (
            project_root
            / "data"
            / "notifications"
            / "private"
            / "telegram.sqlite3"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS notifications (
                notification_id TEXT PRIMARY KEY,
                identity_hash TEXT NOT NULL UNIQUE,
                signal_id TEXT,
                strategy_dna TEXT,
                security_id TEXT,
                contract_id TEXT,
                message_type TEXT NOT NULL,
                message_text TEXT NOT NULL,
                status TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT,
                http_status INTEGER,
                error_code TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                sent_at TEXT
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                notification_id TEXT NOT NULL,
                attempted_at TEXT NOT NULL,
                outcome TEXT NOT NULL,
                retry_count INTEGER NOT NULL,
                http_status INTEGER,
                error_code TEXT
            );
            """
        )
        self.connection.execute(
            """
            UPDATE notifications
            SET status='FAILED_FINAL',
                error_code='DELIVERY_OUTCOME_UNKNOWN_REVIEW_REQUIRED',
                updated_at=?
            WHERE status='SENDING'
            """,
            (datetime.now(UTC).isoformat(),),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> TelegramQueue:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def enqueue(
        self,
        *,
        identity: dict[str, Any],
        message: str,
        message_type: str,
        status: str = "PENDING",
    ) -> tuple[str, str]:
        if status not in QUEUE_STATUSES:
            raise ValueError("invalid queue status")
        identity_hash = _hash(identity)
        notification_id = "TGN-" + identity_hash[:24]
        now = datetime.now(UTC).isoformat()
        row_count = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM notifications"
            ).fetchone()[0]
        )
        if row_count >= MAX_QUEUE_ROWS:
            self.record_attempt(
                notification_id,
                "SKIPPED_FILTER",
                0,
                None,
                "QUEUE_CAP_REACHED",
            )
            return notification_id, "SKIPPED_FILTER"
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO notifications (
                        notification_id, identity_hash, signal_id, strategy_dna,
                        security_id, contract_id, message_type, message_text,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        notification_id,
                        identity_hash,
                        _text(identity.get("signal_id")),
                        _text(identity.get("strategy_dna")),
                        _text(identity.get("security_id")),
                        _text(identity.get("contract_id")),
                        message_type,
                        message,
                        status,
                        now,
                        now,
                    ),
                )
            result = status
        except sqlite3.IntegrityError:
            result = "SKIPPED_DUPLICATE"
            self.record_attempt(
                notification_id, result, 0, None, "DUPLICATE_IDENTITY"
            )
        self.publish()
        return notification_id, result

    def due(self, limit: int) -> list[dict[str, Any]]:
        now = datetime.now(UTC).isoformat()
        rows = self.connection.execute(
            """
            SELECT * FROM notifications
            WHERE status IN ('PENDING', 'RETRY_PENDING')
              AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
            ORDER BY created_at
            LIMIT ?
            """,
            (now, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    def quarantine_pending_signals_except(self, allowed_signal_ids: set[str]) -> int:
        rows = self.connection.execute(
            """
            SELECT notification_id, signal_id
            FROM notifications
            WHERE status IN ('PENDING', 'RETRY_PENDING')
              AND signal_id IS NOT NULL
            """
        ).fetchall()
        blocked = [
            row
            for row in rows
            if str(row["signal_id"]) not in allowed_signal_ids
        ]
        now = datetime.now(UTC).isoformat()
        with self.connection:
            for row in blocked:
                self.connection.execute(
                    """
                    UPDATE notifications
                    SET status='SKIPPED_FILTER',
                        error_code='NO_LONGER_CURRENT_ACTIONABLE_SIGNAL',
                        updated_at=?
                    WHERE notification_id=?
                    """,
                    (now, row["notification_id"]),
                )
        for row in blocked:
            self.record_attempt(
                str(row["notification_id"]),
                "SKIPPED_FILTER",
                0,
                None,
                "NO_LONGER_CURRENT_ACTIONABLE_SIGNAL",
            )
        self.publish()
        return len(blocked)

    def supersede_pending_signal_prefix(
        self,
        *,
        signal_prefix: str,
        message_types: set[str],
        reason: str,
    ) -> int:
        if not signal_prefix or not message_types:
            return 0
        placeholders = ", ".join("?" for _ in message_types)
        parameters = [f"{signal_prefix}%", *sorted(message_types)]
        rows = self.connection.execute(
            f"""
            SELECT notification_id
            FROM notifications
            WHERE status IN ('PENDING', 'RETRY_PENDING')
              AND signal_id LIKE ?
              AND message_type IN ({placeholders})
            """,
            parameters,
        ).fetchall()
        now = datetime.now(UTC).isoformat()
        with self.connection:
            self.connection.executemany(
                """
                UPDATE notifications
                SET status='SKIPPED_FILTER', error_code=?, updated_at=?
                WHERE notification_id=?
                """,
                [
                    (reason, now, str(row["notification_id"]))
                    for row in rows
                ],
            )
        for row in rows:
            self.record_attempt(
                str(row["notification_id"]),
                "SKIPPED_FILTER",
                0,
                None,
                reason,
            )
        self.publish()
        return len(rows)

    def sent_since(self, timestamp: datetime) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM notifications WHERE sent_at >= ?",
                (timestamp.isoformat(),),
            ).fetchone()[0]
        )

    def mark_sending(self, notification_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE notifications SET status='SENDING', updated_at=? WHERE notification_id=?",
                (datetime.now(UTC).isoformat(), notification_id),
            )

    def mark_result(
        self,
        notification_id: str,
        *,
        status: str,
        retry_count: int,
        http_status: int | None,
        error_code: str | None,
        retry_after: float | None = None,
    ) -> None:
        now = datetime.now(UTC)
        next_attempt = (
            (now + timedelta(seconds=retry_after)).isoformat()
            if retry_after is not None
            else None
        )
        sent_at = now.isoformat() if status == "SENT" else None
        with self.connection:
            self.connection.execute(
                """
                UPDATE notifications SET status=?, retry_count=?,
                    next_attempt_at=?, http_status=?, error_code=?,
                    updated_at=?, sent_at=COALESCE(?, sent_at)
                WHERE notification_id=?
                """,
                (
                    status,
                    retry_count,
                    next_attempt,
                    http_status,
                    error_code,
                    now.isoformat(),
                    sent_at,
                    notification_id,
                ),
            )
        self.record_attempt(
            notification_id, status, retry_count, http_status, error_code
        )

    def record_attempt(
        self,
        notification_id: str,
        outcome: str,
        retry_count: int,
        http_status: int | None,
        error_code: str | None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        attempt_id = "TGA-" + _hash(
            {
                "notification_id": notification_id,
                "outcome": outcome,
                "retry_count": retry_count,
                "at": now,
            }
        )[:24]
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    notification_id,
                    now,
                    outcome,
                    retry_count,
                    http_status,
                    error_code,
                ),
            )

    def counts(self) -> dict[str, int]:
        values = {status: 0 for status in QUEUE_STATUSES}
        for row in self.connection.execute(
            "SELECT status, COUNT(*) AS count FROM notifications GROUP BY status"
        ):
            values[row["status"]] = int(row["count"])
        values["SKIPPED_DUPLICATE"] = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE outcome='SKIPPED_DUPLICATE'"
            ).fetchone()[0]
        )
        return values

    def summary(self) -> dict[str, Any]:
        counts = self.counts()
        last_sent = self.connection.execute(
            "SELECT MAX(sent_at) FROM notifications WHERE status='SENT'"
        ).fetchone()[0]
        last_error = self.connection.execute(
            """
            SELECT error_code FROM notifications WHERE error_code IS NOT NULL
            ORDER BY updated_at DESC LIMIT 1
            """
        ).fetchone()
        type_counts = {
            str(row["message_type"]): int(row["count"])
            for row in self.connection.execute(
                """
                SELECT message_type, COUNT(*) AS count
                FROM notifications WHERE status='SENT'
                GROUP BY message_type
                """
            )
        }
        retry_attempts = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM attempts
                WHERE outcome IN ('RETRY_PENDING', 'FAILED_FINAL')
                """
            ).fetchone()[0]
        )
        return {
            "counts": counts,
            "sent_type_counts": type_counts,
            "retry_attempt_count": retry_attempts,
            "queue_size": sum(
                counts[key] for key in ("PENDING", "SENDING", "RETRY_PENDING")
            ),
            "last_successful_send": last_sent,
            "last_error": last_error[0] if last_error else None,
        }

    def publish(self, settings: TelegramSettings | None = None) -> None:
        root = self.project_root / "output" / "notifications"
        root.mkdir(parents=True, exist_ok=True)
        rows = [
            _public_row(dict(row), settings)
            for row in self.connection.execute(
                """
                SELECT notification_id, identity_hash, signal_id, strategy_dna,
                       security_id, contract_id, message_type, status,
                       retry_count, http_status, error_code, created_at,
                       updated_at, sent_at
                FROM notifications ORDER BY created_at
                """
            )
        ]
        _write_jsonl(root / "telegram_queue.jsonl", rows)
        _write_jsonl(
            root / "telegram_notifications.jsonl",
            [row for row in rows if row["delivery_status"] == "SENT"],
        )
        _write_jsonl(
            root / "telegram_failures.jsonl",
            [
                row
                for row in rows
                if row["delivery_status"] in {"RETRY_PENDING", "FAILED_FINAL"}
            ],
        )


class TelegramNotifier:
    def __init__(
        self,
        project_root: Path,
        settings: TelegramSettings,
        *,
        client_factory: Callable[..., httpx.Client] = httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.project_root = project_root
        self.settings = settings
        self.client_factory = client_factory
        self.sleep = sleep

    def health(self, *, probe_api: bool = True) -> dict[str, Any]:
        status = self.settings.public_status
        api_reachable: bool | None = None
        error: str | None = None
        if status == "ENABLED" and probe_api:
            try:
                with self.client_factory(
                    timeout=self.settings.request_timeout_seconds
                ) as client:
                    response = client.get(self._url("getMe"))
                api_reachable = response.status_code == 200
                if not api_reachable:
                    error = f"TELEGRAM_HTTP_{response.status_code}"
            except httpx.HTTPError as exc:
                api_reachable = False
                error = _error_code(exc)
        with TelegramQueue(self.project_root) as queue:
            summary = queue.summary()
            queue.publish(self.settings)
        if status == "ENABLED" and api_reachable is False:
            status = "DEGRADED_API_UNREACHABLE"
        report = {
            "schema": "telegram_health_v1",
            "status": status,
            "enabled": self.settings.enabled,
            "configured": self.settings.configured,
            "api_reachable": api_reachable,
            "masked_chat_identity": self.settings.masked_chat_identity,
            **summary,
            "rate_limit_per_minute": self.settings.max_messages_per_minute,
            "request_timeout_seconds": self.settings.request_timeout_seconds,
            "max_retries": self.settings.max_retries,
            "dry_run": self.settings.dry_run,
            "last_probe_error": error,
            "telegram_can_change_authority": False,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
        _write_json(
            self.project_root
            / "output"
            / "notifications"
            / "telegram_health.json",
            report,
        )
        return report

    def enqueue_signal(self, signal: dict[str, Any]) -> dict[str, Any]:
        decision, reason = signal_filter(signal, self.settings)
        message = format_signal_message(signal, self.settings)
        identity = notification_identity(signal)
        with TelegramQueue(self.project_root) as queue:
            notification_id, status = queue.enqueue(
                identity=identity,
                message=message,
                message_type=_message_type(signal),
                status=decision,
            )
            queue.publish(self.settings)
        return {
            "notification_id": notification_id,
            "status": status,
            "filter_reason": reason,
        }

    def enqueue_text(
        self,
        *,
        message: str,
        message_type: str,
        identity: dict[str, Any],
    ) -> dict[str, Any]:
        with TelegramQueue(self.project_root) as queue:
            notification_id, status = queue.enqueue(
                identity=identity,
                message=message,
                message_type=message_type,
            )
            queue.publish(self.settings)
        return {"notification_id": notification_id, "status": status}

    def process(self, *, max_items: int | None = None) -> dict[str, Any]:
        if not self.settings.enabled:
            return self._process_report(self.settings.public_status, 0)
        if not self.settings.configured and not self.settings.dry_run:
            return self._process_report("DISABLED_MISSING_CONFIG", 0)
        processed = 0
        with TelegramQueue(self.project_root) as queue:
            sent_last_minute = queue.sent_since(
                datetime.now(UTC) - timedelta(minutes=1)
            )
            allowance = max(
                0, self.settings.max_messages_per_minute - sent_last_minute
            )
            if max_items is not None:
                allowance = min(allowance, max(0, int(max_items)))
            due = queue.due(allowance)
            for index, row in enumerate(due):
                if index:
                    self.sleep(60.0 / self.settings.max_messages_per_minute)
                self._deliver(queue, row)
                processed += 1
            queue.publish(self.settings)
            summary = queue.summary()
        return {
            "schema": "telegram_delivery_v1",
            "status": "GO",
            "processed": processed,
            "rate_limit_allowance_used": processed,
            **summary,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }

    def _deliver(self, queue: TelegramQueue, row: dict[str, Any]) -> None:
        retry_count = int(row["retry_count"])
        queue.mark_sending(row["notification_id"])
        if self.settings.dry_run:
            queue.mark_result(
                row["notification_id"],
                status="SENT",
                retry_count=retry_count,
                http_status=None,
                error_code="DRY_RUN_NO_HTTP",
            )
            return
        try:
            with self.client_factory(
                timeout=self.settings.request_timeout_seconds
            ) as client:
                response = client.post(
                    self._url("sendMessage"),
                    json={
                        "chat_id": self.settings.chat_id,
                        "text": row["message_text"],
                        "disable_web_page_preview": True,
                    },
                )
            if response.status_code == 200:
                queue.mark_result(
                    row["notification_id"],
                    status="SENT",
                    retry_count=retry_count,
                    http_status=200,
                    error_code=None,
                )
                return
            retry_after = _retry_after(response)
            self._retry_or_fail(
                queue,
                row,
                retry_count,
                response.status_code,
                f"TELEGRAM_HTTP_{response.status_code}",
                retry_after,
            )
        except httpx.HTTPError as exc:
            self._retry_or_fail(
                queue,
                row,
                retry_count,
                None,
                _error_code(exc),
                None,
            )

    def _retry_or_fail(
        self,
        queue: TelegramQueue,
        row: dict[str, Any],
        retry_count: int,
        http_status: int | None,
        error_code: str,
        retry_after: float | None,
    ) -> None:
        new_count = retry_count + 1
        final = new_count > self.settings.max_retries
        delay = retry_after if retry_after is not None else min(60.0, 2**new_count)
        queue.mark_result(
            row["notification_id"],
            status="FAILED_FINAL" if final else "RETRY_PENDING",
            retry_count=new_count,
            http_status=http_status,
            error_code=error_code,
            retry_after=None if final else delay,
        )

    def _url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.settings.token}/{method}"

    def _process_report(self, status: str, processed: int) -> dict[str, Any]:
        with TelegramQueue(self.project_root) as queue:
            summary = queue.summary()
            queue.publish(self.settings)
        return {
            "schema": "telegram_delivery_v1",
            "status": status,
            "processed": processed,
            **summary,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }


def notification_identity(signal: dict[str, Any]) -> dict[str, Any]:
    contract = signal.get("contract_identity") or {}
    action = str(signal.get("action") or "").upper()
    if action in EXITS:
        return {
            "signal_id": signal.get("signal_id"),
            "signal_status": signal.get("lifecycle_status"),
            "security_id": signal.get("ticker") or signal.get("security_id"),
            "strategy_dna": signal.get("strategy_dna_hash"),
            "action": action,
            "position_scope": signal.get("position_scope"),
            "reason_codes": list(signal.get("reason_codes") or []),
            "current_r": signal.get("current_r"),
            "peak_r": signal.get("peak_r"),
            "profit_giveback": signal.get("profit_giveback"),
        }
    return {
        "signal_id": signal.get("signal_id"),
        "signal_status": signal.get("lifecycle_status"),
        "security_id": signal.get("ticker") or signal.get("security_id"),
        "contract_id": contract.get("con_id") or signal.get("contract_id"),
        "strategy_dna": signal.get("strategy_dna_hash"),
        "action": signal.get("action"),
        "entry_zone": [
            signal.get("entry_zone_low"),
            signal.get("entry_zone_high"),
        ],
        "stop_loss": signal.get("stop_loss"),
        "take_profit_1": signal.get("take_profit_1"),
        "take_profit_2": signal.get("take_profit_2"),
        "confidence": signal.get("confidence_score"),
        "position_size": signal.get("suggested_quantity"),
        "expiration": signal.get("expiration_timestamp"),
        "warnings": signal.get("risks"),
    }


def signal_filter(
    signal: dict[str, Any], settings: TelegramSettings
) -> tuple[str, str | None]:
    action = str(signal.get("action", "NO_SIGNAL")).upper()
    if action == "NO_SIGNAL":
        return "SKIPPED_FILTER", "NO_SIGNAL"
    if action == "AVOID" and not signal.get("material_lifecycle_change"):
        return "SKIPPED_FILTER", "AVOID_WITHOUT_MATERIAL_LIFECYCLE_CHANGE"
    if _is_stale(signal):
        return "SKIPPED_STALE", "STALE_OR_EXPIRED"
    if _is_future(signal):
        if not signal.get("futures_sleeve_active") or not signal.get(
            "strategy_supports_futures"
        ):
            return "SKIPPED_FILTER", "FUTURES_SLEEVE_OR_STRATEGY_INACTIVE"
        if not FUTURE_REQUIRED_FIELDS.issubset(
            signal.get("contract_identity") or {}
        ):
            return "SKIPPED_FILTER", "BLOCKED_INCOMPLETE_FUTURES_CONTRACT"
        if (
            signal["contract_identity"].get("rollover_status")
            != "SAFE_OUTSIDE_ROLL_WINDOW"
        ):
            return "SKIPPED_FILTER", "FUTURES_ROLLOVER_NOT_SAFE"
    if action in ACTIONABLE:
        if not settings.send_signals:
            return "SKIPPED_FILTER", "SIGNALS_DISABLED"
        if signal.get("lifecycle_status") != "MANUAL_ACTIONABLE":
            return "SKIPPED_FILTER", "MANUAL_ACTIONABLE_AUTHORITY_REQUIRED"
        if not (signal.get("contract_identity") or {}).get("con_id"):
            return "SKIPPED_FILTER", "EXACT_CONTRACT_REQUIRED"
        gates = {
            "liquidity_status": "GO",
            "spread_status": "GO",
            "event_risk_status": "GO",
        }
        if any(signal.get(field) != required for field, required in gates.items()):
            return "SKIPPED_FILTER", "MARKET_QUALITY_OR_EVENT_RISK_BLOCKED"
        if signal.get("market_allowed") is not True:
            return "SKIPPED_FILTER", "MARKET_OR_INSTRUMENT_NOT_ALLOWED"
        confidence = _confidence(signal.get("confidence_score"))
        if confidence < settings.minimum_confidence:
            return "SKIPPED_FILTER", "MINIMUM_CONFIDENCE"
        if _decimal(signal.get("reward_risk_1")) < Decimal(
            str(settings.minimum_reward_risk)
        ):
            return "SKIPPED_FILTER", "MINIMUM_REWARD_RISK"
        required = {
            "stop_loss",
            "take_profit_1",
            "suggested_quantity",
            "expiration_timestamp",
        }
        if not all(signal.get(field) not in (None, "", "0", 0) for field in required):
            return "SKIPPED_FILTER", "INCOMPLETE_ACTIONABLE_SIGNAL"
    if action == "WATCHLIST" and not settings.send_watchlist:
        return "SKIPPED_FILTER", "WATCHLIST_DISABLED"
    if action in EXITS and not settings.send_exits:
        return "SKIPPED_FILTER", "EXITS_DISABLED"
    if action == "HOLD" and not signal.get("material_lifecycle_change"):
        return "SKIPPED_FILTER", "HOLD_WITHOUT_MATERIAL_CHANGE"
    return "PENDING", None


def format_signal_message(
    signal: dict[str, Any], settings: TelegramSettings
) -> str:
    action = str(signal.get("action", "NO_SIGNAL")).upper()
    if action in EXITS:
        return _format_exit_signal_message(signal)
    ticker = str(signal.get("ticker") or signal.get("asset") or "ONBEKEND")
    currency = str(signal.get("currency") or "USD").upper()
    symbol = {"EUR": "€", "USD": "$", "GBP": "£"}.get(currency, currency + " ")
    kind = _instrument_type(signal)
    icon = {
        "STRONG_BUY": "🟢",
        "BUY": "🟢",
        "WATCHLIST": "🟡",
        "EXIT": "🔴",
        "REDUCE": "🔴",
        "HOLD": "🟡",
        "AVOID": "⚠️",
    }.get(action, "ℹ️")
    lines = [
        f"{icon} {action} — {ticker}",
        "",
        f"Asset: {signal.get('asset') or ticker}",
        f"Type: {kind}",
        f"Exchange: {signal.get('exchange') or 'niet gepubliceerd'}",
        f"Valuta: {currency}",
        "",
        f"Huidige prijs: {_money(signal.get('current_market_price'), symbol)}",
        "Entryzone: "
        f"{_money(signal.get('entry_zone_low'), symbol)} – "
        f"{_money(signal.get('entry_zone_high'), symbol)}",
        f"Voorkeurslimit: {_money(signal.get('preferred_entry'), symbol)}",
        "",
        f"Stop-loss: {_money(signal.get('stop_loss'), symbol)}",
        f"Stopmethode: {signal.get('stop_method') or 'strategie-invalidatie'}",
        f"Afstand: {_percent(signal.get('stop_distance_pct'))}",
        f"Take-profit 1: {_money(signal.get('take_profit_1'), symbol)}",
        (
            f"Take-profit 2: {_money(signal.get('take_profit_2'), symbol)}"
            if signal.get("take_profit_2") is not None
            else f"Take-profit: {signal.get('take_profit_mode') or 'trailing exit'}"
        ),
        "",
        f"Risk/reward TP1: {_number(signal.get('reward_risk_1'), 2)}",
        f"Risk/reward TP2: {_number(signal.get('reward_risk_2'), 2)}",
    ]
    if settings.include_position_size:
        lines.extend(
            [
                "",
                f"Voorgestelde positie: {_quantity(signal.get('suggested_quantity'))}",
                "Maximaal gepland verlies: "
                f"€{_number(signal.get('maximum_planned_loss_eur'), 2)}",
                "Maximale modelallocatie: "
                f"€{_number(signal.get('maximum_order_value_eur'), 2)}",
                "FX: reeds verwerkt door signal/risklaag; bron en timestamp "
                "moeten in het signaalartifact staan.",
            ]
        )
    lines.extend(
        [
            "",
            f"Strategie: {signal.get('strategy_id') or 'onbekend'}",
            f"Timeframe: {signal.get('timeframe') or 'onbekend'}",
            f"Regime: {signal.get('regime') or 'onbekend'}",
            f"Consensus: {_confidence(signal.get('consensus_score')):.0f}%",
            f"Confidence: {_confidence(signal.get('confidence_score')):.0f}%",
            f"Verwachte duur: {signal.get('expected_holding_period') or 'niet gepubliceerd'}",
            f"Geldig tot: {_display_time(signal.get('expiration_timestamp'))}",
            f"Data: {signal.get('data_freshness') or 'onbekend'}",
            f"Marktstatus: {signal.get('market_status') or 'niet gepubliceerd'}",
            "Volgende mogelijke entry: "
            f"{signal.get('next_execution_window') or 'niet gepubliceerd'}",
        ]
    )
    participants = list(signal.get("participating_strategies") or [])
    if participants:
        lines.extend(
            ["", "Actieve strategieën:"]
            + [
                " • "
                + str(item.get("strategy_id", "onbekend"))
                + f" (gewicht {_percent(item.get('weight'))})"
                for item in participants[:6]
            ]
        )
    if settings.include_fundamentals and signal.get("fundamentals"):
        values = list(dict(signal["fundamentals"]).items())[:5]
        lines.extend(
            ["", "Fundamentals:"]
            + [f" • {key}: {value}" for key, value in values]
        )
    if settings.include_macro_context and signal.get("macro_context"):
        values = list(dict(signal["macro_context"]).items())[:5]
        lines.extend(
            ["", "Macrocontext:"]
            + [f" • {key}: {value}" for key, value in values]
        )
    reasons = list(signal.get("reasons") or [])
    risks = list(signal.get("risks") or [])
    if reasons:
        lines.extend(["", "Reden:", " • " + "\n • ".join(map(str, reasons[:5]))])
    if risks:
        lines.extend(
            ["", "Waarschuwing:", " ⚠️ " + "\n ⚠️ ".join(map(str, risks[:6]))]
        )
    if _is_future(signal):
        contract = signal.get("contract_identity") or {}
        lines.extend(
            [
                "",
                f"Expiry: {contract.get('expiry') or 'ontbreekt'}",
                f"Multiplier: {contract.get('multiplier') or 'ontbreekt'}",
                f"Tick size: {contract.get('minimum_tick') or 'ontbreekt'}",
                f"Waarde per tick: {contract.get('tick_value') or 'ontbreekt'}",
                f"Geschatte margin: €{_number(contract.get('margin_eur'), 2)}",
                f"Rolloverstatus: {contract.get('rollover_status') or 'ontbreekt'}",
                "⚠️ Futures bevatten multiplier-, margin-, gap- en rolloverrisico.",
            ]
        )
    if action == "WATCHLIST":
        lines.extend(["", "Nog geen order plaatsen."])
    lines.extend(
        [
            "",
            "Dit is een modelsignaal, geen gegarandeerde uitkomst.",
            "Automatische execution: uit.",
        ]
    )
    return "\n".join(lines)


def _format_exit_signal_message(signal: dict[str, Any]) -> str:
    action = str(signal.get("action") or "EXIT").upper()
    ticker = str(signal.get("ticker") or signal.get("asset") or "ONBEKEND")
    currency = str(signal.get("currency") or "USD").upper()
    symbol = {"EUR": "€", "USD": "$", "GBP": "£"}.get(
        currency, currency + " "
    )
    current_price = signal.get("current_market_price")
    price_text = (
        _money(current_price, symbol)
        if current_price not in (None, "")
        else "niet gepubliceerd"
    )
    lines = [
        f"🔴 {action} REVIEW — {ticker}",
        "",
        f"Bron: {signal.get('source') or 'onbekend'}",
        f"Scope: {signal.get('position_scope') or 'MODEL_ADVISORY'}",
        f"Strategie: {signal.get('strategy_id') or 'onbekend'}",
        f"Timeframe: {signal.get('timeframe') or 'onbekend'}",
        f"Huidige prijs: {price_text}",
    ]
    if signal.get("current_r") is not None:
        lines.append(f"Huidige R: {_number(signal.get('current_r'), 2)}R")
    if signal.get("peak_r") is not None:
        lines.append(f"Piek R: {_number(signal.get('peak_r'), 2)}R")
    if signal.get("profit_giveback") is not None:
        lines.append(
            "Profit giveback: " + _percent(signal.get("profit_giveback"))
        )
    reasons = list(
        signal.get("reason_codes") or signal.get("reasons") or []
    )
    if reasons:
        lines.extend(
            ["", "Redenen:"]
            + [f" • {reason}" for reason in reasons[:8]]
        )
    lines.extend(
        [
            "",
            f"Geldig tot: {_display_time(signal.get('expiration_timestamp'))}",
            "Dit is een observationeel verkoopadvies.",
            "Geen brokerorder gegenereerd.",
            "Executionauthority: NONE",
        ]
    )
    return "\n".join(lines)


def format_order_event(event: dict[str, Any]) -> str:
    status = str(event.get("status") or "UNKNOWN").upper()
    symbol = str(event.get("symbol") or "ONBEKEND")
    icon = {
        "SUBMITTING": "📤",
        "SUBMITTED": "✅",
        "PARTIAL_FILL": "🟠",
        "FILLED": "✅",
        "CANCELLED": "⚠️",
        "APICANCELLED": "⚠️",
        "REJECTED": "🚨",
    }.get(status, "ℹ️")
    order_reference = event.get("order_reference")
    masked_reference = (
        "order-sha256:" + hashlib.sha256(str(order_reference).encode()).hexdigest()[:12]
        if order_reference is not None
        else "niet gepubliceerd"
    )
    lines = [
        f"{icon} ORDER {status.replace('_', ' ')} — {symbol}",
        "",
        f"Side: {event.get('side') or 'onbekend'}",
        f"Type: {event.get('order_type') or 'onbekend'}",
        f"Aantal: {_number(event.get('quantity'), 3)}",
        f"Filled: {_number(event.get('filled_quantity'), 3)}",
        f"Resterend: {_number(event.get('remaining_quantity'), 3)}",
        f"Gemiddelde fill: {_number(event.get('average_fill_price'), 4)}",
        f"Environment: {event.get('environment') or 'onbekend'}",
        f"Order reference: {masked_reference}",
    ]
    if event.get("reason"):
        lines.extend(["", f"Reden: {event['reason']}"])
    lines.extend(
        [
            "",
            "Telegram rapporteert uitsluitend brokerstate en wijzigt geen order.",
        ]
    )
    return "\n".join(lines)


def telegram_order_event(
    project_root: Path,
    event: dict[str, Any],
    *,
    notifier: TelegramNotifier | None = None,
) -> dict[str, Any]:
    settings = load_telegram_settings(project_root)
    if not settings.send_order_events:
        return {
            "status": "SKIPPED_FILTER",
            "reason": "ORDER_EVENTS_DISABLED",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    notifier = notifier or TelegramNotifier(project_root, settings)
    result = notifier.enqueue_text(
        message=format_order_event(event),
        message_type="ORDER_EVENT",
        identity={
            "type": "ORDER_EVENT",
            "order_reference_hash": _hash(
                {"order_reference": event.get("order_reference")}
            ),
            "status": event.get("status"),
            "filled_quantity": event.get("filled_quantity"),
            "remaining_quantity": event.get("remaining_quantity"),
        },
    )
    return {
        "status": result["status"],
        "notification_id": result["notification_id"],
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def telegram_alert(
    project_root: Path,
    alert: dict[str, Any],
    *,
    notifier: TelegramNotifier | None = None,
) -> dict[str, Any]:
    settings = load_telegram_settings(project_root)
    category = str(alert.get("category") or "SYSTEM").upper()
    if category == "RISK" and not settings.send_risk_alerts:
        reason = "RISK_ALERTS_DISABLED"
    elif category != "RISK" and not settings.send_system_alerts:
        reason = "SYSTEM_ALERTS_DISABLED"
    else:
        reason = None
    if reason:
        return {
            "status": "SKIPPED_FILTER",
            "reason": reason,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    severity = str(alert.get("severity") or "WARNING").upper()
    icon = "🚨" if severity == "CRITICAL" else "✅" if severity == "RECOVERY" else "⚠️"
    event_type = str(alert.get("event_type") or "SYSTEM_ALERT")
    message = (
        f"{icon} {event_type}\n\n"
        f"Status: {alert.get('status') or 'onbekend'}\n"
        f"Reden: {alert.get('reason') or 'niet gepubliceerd'}\n"
        "Automatische executionauthority gewijzigd: nee\n"
        "IBKR-orders geplaatst door Telegram: 0"
    )
    notifier = notifier or TelegramNotifier(project_root, settings)
    result = notifier.enqueue_text(
        message=message,
        message_type=f"{category}_ALERT",
        identity={
            "type": "ALERT",
            "event_type": event_type,
            "status": alert.get("status"),
            "reason": alert.get("reason"),
            "observed_at": alert.get("observed_at"),
        },
    )
    return {
        "status": result["status"],
        "notification_id": result["notification_id"],
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def telegram_command(
    project_root: Path,
    command: str,
    *,
    client_factory: Callable[..., httpx.Client] = httpx.Client,
) -> dict[str, Any]:
    settings = load_telegram_settings(project_root)
    notifier = TelegramNotifier(
        project_root, settings, client_factory=client_factory
    )
    if command == "health":
        return notifier.health()
    if command == "status":
        report = notifier.health(probe_api=False)
        phase9 = _read_json_object(
            project_root / "output/ibkr/phase9/status.json"
        )
        writer = _read_json_object(
            project_root
            / "output/ibkr/live/writer-integrity-verify.json"
        )
        capabilities = _read_json_object(
            project_root
            / "output/ibkr/data-capabilities/capability-matrix.json"
        )
        ml = _read_json_object(
            project_root
            / "output/research/active_swing/selective_ml/status.json"
        )
        sec = _read_json_object(
            project_root
            / "output/research/sec_intelligence/status.json"
        )
        report["schema"] = "telegram_status_v1"
        report.update(
            {
                "TELEGRAM_NOTIFICATIONS_WORKING": bool(
                    settings.enabled
                    and settings.configured
                    and report["last_successful_send"]
                ),
                "TELEGRAM_STOCK_SIGNALS_ENABLED": settings.send_signals,
                "TELEGRAM_ETF_SIGNALS_ENABLED": settings.send_signals,
                "TELEGRAM_COMMODITY_SIGNALS_ENABLED": settings.send_signals,
                "TELEGRAM_FUTURES_SIGNALS_ENABLED": settings.send_signals,
                "TELEGRAM_RISK_ALERTS_ENABLED": settings.send_risk_alerts,
                "TELEGRAM_ORDER_STATUS_ALERTS_ENABLED": settings.send_order_events,
                "TELEGRAM_MARKET_DIGEST_ENABLED": settings.send_market_digest,
                "TELEGRAM_DUPLICATE_SUPPRESSION": True,
                "TELEGRAM_SECRETS_REDACTED": _notification_secret_scan(
                    project_root, settings
                ),
                "IBKR_ACCOUNT_IDENTIFIERS_REDACTED": True,
                "TELEGRAM_FAILURE_DOES_NOT_STOP_SIGNALS": True,
                "TELEGRAM_CANNOT_CHANGE_AUTHORITY": True,
                "TELEGRAM_COMMANDS_PLACE_ZERO_ORDERS": True,
                "delivery_breakdown": _delivery_breakdown(project_root),
                "safety_status": {
                    "phase9": phase9.get("status", "UNAVAILABLE"),
                    "phase9_blockers": phase9.get("open_blockers", []),
                    "writer_integrity": writer.get("status", "UNAVAILABLE"),
                    "writer_hash_integrity": writer.get(
                        "writer_hash_integrity", False
                    ),
                    "ibkr_data_capabilities": capabilities.get(
                        "summary", {}
                    ),
                    "missing_subscription_classes": capabilities.get(
                        "missing_subscription_classes", []
                    ),
                    "canonical_ml_status": ml.get("status", "NOT_RUN"),
                    "canonical_ml_closed_labels": ml.get(
                        "closed_trainable_episode_count", 0
                    ),
                    "sec_overlay_status": sec.get("status", "UNAVAILABLE"),
                    "sec_structured_event_count": sec.get(
                        "structured_event_count", 0
                    ),
                    "execution_authority": "NONE",
                },
            }
        )
        _write_json(
            project_root
            / "output"
            / "notifications"
            / "telegram_status.json",
            report,
        )
        return report
    if command == "preview":
        return telegram_preview(project_root, settings)
    if command == "send-latest-signals":
        return telegram_send_latest(project_root, notifier)
    if command == "send-pit-mtf-signals":
        return telegram_send_pit_mtf(project_root, notifier)
    if command == "send-exit-signals":
        return telegram_send_exit_signals(project_root, notifier)
    if command == "top-5-preview":
        return telegram_top5_preview(project_root)
    if command == "send-top-5":
        return telegram_send_top5(project_root, notifier)
    if command == "send-regime-update":
        return telegram_send_regime_update(project_root, notifier)
    if command == "market-digest-preview":
        digest = build_market_intelligence_digest(project_root)
        return {
            "schema": "telegram_market_digest_preview_v1",
            "status": digest["status"],
            "digest": digest,
            "message": format_market_intelligence_digest(digest),
            "sent": 0,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    if command == "send-market-digest":
        return telegram_send_market_digest(project_root, notifier)
    if command == "send-shadow-digest":
        return telegram_send_shadow_digest(project_root, notifier)
    if command == "retry-failed":
        with TelegramQueue(project_root) as queue:
            now = datetime.now(UTC).isoformat()
            with queue.connection:
                queue.connection.execute(
                    """
                    UPDATE notifications SET status='RETRY_PENDING',
                        next_attempt_at=NULL, updated_at=?
                    WHERE status='FAILED_FINAL'
                    """,
                    (now,),
                )
        return notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
    if command == "test":
        identity = {
            "type": "SAFE_CONNECTION_TEST",
            "date": datetime.now(UTC).date().isoformat(),
            "configured_chat": settings.masked_chat_identity,
        }
        queued = notifier.enqueue_text(
            message=(
                "✅ Stocks/ETF/commodities Telegramkoppeling werkt.\n\n"
                "Modus: SIGNALS_ONLY\n"
                "IBKR-orders geplaatst: 0\n"
                "Automatische execution: uit"
            ),
            message_type="SYSTEM_TEST",
            identity=identity,
        )
        delivery = notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
        return {
            "schema": "telegram_safe_test_v1",
            "status": delivery["status"],
            "queued": queued,
            "delivery": delivery,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    raise ValueError(f"unknown telegram command: {command}")


def telegram_send_regime_update(
    project_root: Path,
    notifier: TelegramNotifier | None = None,
) -> dict[str, Any]:
    """Publish only material, filtered HMM regime changes."""
    current_path = (
        project_root / "output" / "research" / "phase11_11" / "current.json"
    )
    try:
        payload = json.loads(current_path.read_text(encoding="utf-8"))
        state = payload["state"]
        probabilities = {
            str(name): float(value)
            for name, value in state["probabilities"].items()
        }
        multiplier = float(state["regime_multiplier"])
        as_of = str(state["as_of"])
        model_hash = str(state["model_hash"])
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return _hmm_notification_report(
            status="DATA_BLOCKED",
            outcome="HMM_CURRENT_STATE_INVALID_OR_MISSING",
        )
    probability_total = sum(probabilities.values())
    if (
        not probabilities
        or any(value < 0.0 or value > 1.0 for value in probabilities.values())
        or abs(probability_total - 1.0) > 1e-6
        or not 0.0 <= multiplier <= 1.0
    ):
        return _hmm_notification_report(
            status="DATA_BLOCKED",
            outcome="HMM_CURRENT_STATE_FAILED_VALIDATION",
        )

    dominant_state = max(probabilities, key=lambda name: probabilities[name])
    stress_probability = probabilities.get("STRESS_HIGH_VOL", 0.0)
    risk_band = _hmm_risk_band(multiplier)
    stress_band = _hmm_stress_band(stress_probability)
    observed = {
        "as_of": as_of,
        "model_hash": model_hash,
        "dominant_state": dominant_state,
        "risk_band": risk_band,
        "stress_band": stress_band,
        "regime_multiplier": multiplier,
        "probabilities": probabilities,
    }
    state_path = (
        project_root
        / "data"
        / "notifications"
        / "private"
        / "hmm-regime-state.json"
    )
    previous = _read_json_object(state_path)
    change_reasons = _hmm_material_changes(previous, observed)
    _write_json(state_path, observed)

    public = {
        "schema": "hmm_regime_notification_v1",
        "as_of": as_of,
        "dominant_state": dominant_state,
        "risk_band": risk_band,
        "stress_band": stress_band,
        "regime_multiplier": multiplier,
        "probabilities": probabilities,
        "change_reasons": change_reasons,
        "filtered_probabilities_only": True,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    artifact_path = (
        project_root
        / "output"
        / "notifications"
        / "hmm_regime_notification.json"
    )
    if not change_reasons:
        report = {
            **public,
            "status": "GO",
            "outcome": "NO_MATERIAL_REGIME_CHANGE",
            "notification_queued": False,
        }
        _write_json(artifact_path, report)
        return report

    settings = load_telegram_settings(project_root)
    if not settings.send_risk_alerts:
        report = {
            **public,
            "status": "GO",
            "outcome": "RISK_ALERTS_DISABLED",
            "notification_queued": False,
        }
        _write_json(artifact_path, report)
        return report
    notifier = notifier or TelegramNotifier(project_root, settings)
    icon = (
        "🚨"
        if risk_band in {"ENTRY_BLOCK", "DEFENSIVE"}
        else "⚠️"
        if risk_band == "CAUTIOUS"
        else "✅"
    )
    probability_lines = "\n".join(
        f"{name}: {value:.1%}"
        for name, value in sorted(probabilities.items())
    )
    message = (
        f"{icon} HMM marktregime\n\n"
        f"Dominant: {dominant_state}\n"
        f"Risicoband: {risk_band}\n"
        f"Exposuremultiplier: {multiplier:.1%}\n"
        f"Stressniveau: {stress_band}\n"
        f"{probability_lines}\n"
        f"Bar afgesloten op: {as_of}\n\n"
        "Gefilterde kansen, geen koersvoorspelling.\n"
        "Automatische executionauthority gewijzigd: nee\n"
        "IBKR-orders geplaatst door deze melding: 0"
    )
    queued = notifier.enqueue_text(
        message=message,
        message_type="HMM_REGIME_ALERT",
        identity={
            "type": "HMM_REGIME_ALERT",
            "as_of": as_of,
            "model_hash": model_hash,
            "dominant_state": dominant_state,
            "risk_band": risk_band,
            "stress_band": stress_band,
            "change_reasons": change_reasons,
        },
    )
    delivery = notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
    report = {
        **public,
        "status": "GO",
        "outcome": "MATERIAL_REGIME_CHANGE_PUBLISHED",
        "notification_queued": queued["status"] == "PENDING",
        "enqueue_status": queued["status"],
        "delivery_status": delivery["status"],
    }
    _write_json(artifact_path, report)
    return report


def _hmm_material_changes(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> list[str]:
    if not previous:
        return ["INITIAL_REGIME_CONTEXT"]
    changes = []
    for field, reason in (
        ("dominant_state", "DOMINANT_STATE_CHANGED"),
        ("risk_band", "RISK_BAND_CHANGED"),
        ("stress_band", "STRESS_BAND_CHANGED"),
    ):
        if previous.get(field) != current.get(field):
            changes.append(reason)
    return changes


def _hmm_risk_band(multiplier: float) -> str:
    if multiplier <= 0.20:
        return "ENTRY_BLOCK"
    if multiplier <= 0.50:
        return "DEFENSIVE"
    if multiplier <= 0.80:
        return "CAUTIOUS"
    return "NORMAL"


def _hmm_stress_band(probability: float) -> str:
    if probability >= 0.50:
        return "CRITICAL"
    if probability >= 0.25:
        return "HIGH"
    if probability >= 0.10:
        return "ELEVATED"
    return "LOW"


def _hmm_notification_report(*, status: str, outcome: str) -> dict[str, Any]:
    return {
        "schema": "hmm_regime_notification_v1",
        "status": status,
        "outcome": outcome,
        "notification_queued": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def telegram_preview(
    project_root: Path, settings: TelegramSettings | None = None
) -> dict[str, Any]:
    settings = settings or load_telegram_settings(project_root)
    signals = _latest_signals(project_root)
    messages = [format_signal_message(signal, settings) for signal in signals]
    root = project_root / "output" / "notifications"
    root.mkdir(parents=True, exist_ok=True)
    text_path = root / "latest_telegram_preview.txt"
    html_path = root / "latest_telegram_preview.html"
    text_path.write_text("\n\n" + ("\n\n---\n\n".join(messages)), encoding="utf-8")
    html_path.write_text(
        "<!doctype html><html lang='nl'><meta charset='utf-8'>"
        "<title>Telegram preview</title><style>body{font-family:system-ui;"
        "max-width:900px;margin:24px auto}pre{white-space:pre-wrap;"
        "border:1px solid #ccd1d1;padding:16px;margin:18px 0}</style><body>"
        + "".join(f"<pre>{html.escape(message)}</pre>" for message in messages)
        + "</body></html>",
        encoding="utf-8",
    )
    return {
        "schema": "telegram_preview_v1",
        "status": "GO",
        "message_count": len(messages),
        "text_path": str(text_path),
        "html_path": str(html_path),
        "sent": 0,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def telegram_send_latest(
    project_root: Path, notifier: TelegramNotifier | None = None
) -> dict[str, Any]:
    settings = load_telegram_settings(project_root)
    notifier = notifier or TelegramNotifier(project_root, settings)
    results = [
        notifier.enqueue_signal(signal) for signal in _latest_signals(project_root)
    ]
    delivery = notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
    counts: dict[str, int] = {}
    for result in results:
        key = result["status"]
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": "telegram_latest_signals_delivery_v1",
        "status": delivery["status"],
        "signals_considered": len(results),
        "enqueue_counts": counts,
        "delivery": delivery,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def telegram_send_exit_signals(
    project_root: Path, notifier: TelegramNotifier | None = None
) -> dict[str, Any]:
    settings = (
        notifier.settings if notifier is not None else load_telegram_settings(project_root)
    )
    notifier = notifier or TelegramNotifier(project_root, settings)
    signals = _exit_notification_signals(project_root)
    results = [notifier.enqueue_signal(signal) for signal in signals]
    delivery = (
        notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
        if signals
        else notifier.health(probe_api=False)
    )
    counts: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        counts[status] = counts.get(status, 0) + 1
    report = {
        "schema": "telegram_exit_signals_delivery_v1",
        "status": "GO",
        "outcome": (
            "EXIT_ADVISORIES_PROCESSED"
            if signals
            else "NO_CURRENT_EXIT_ADVISORIES"
        ),
        "signals_considered": len(signals),
        "enqueue_counts": counts,
        "delivery": delivery,
        "automatic_execution_allowed": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    _write_json(
        project_root
        / "output"
        / "notifications"
        / "latest_exit_delivery.json",
        report,
    )
    return report


def _exit_notification_signals(project_root: Path) -> list[dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    for row in _latest_signals(project_root):
        ticker = str(row.get("ticker") or row.get("asset") or "").upper()
        if ticker:
            contexts[ticker] = row

    advisories: dict[tuple[str, str], dict[str, Any]] = {}
    lifecycle = _read_json_object(
        project_root / "output" / "operations" / "signal-lifecycle.json"
    )
    for row in lifecycle.get("rows", []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("lifecycle_status") or "").upper()
        if status not in EXITS:
            continue
        ticker = str(row.get("ticker") or "").upper()
        if ticker:
            advisories[(ticker, status)] = _exit_signal_payload(
                ticker=ticker,
                action=status,
                source="SIGNAL_LIFECYCLE",
                row=row,
                context=contexts.get(ticker, {}),
            )

    action_map = {
        "EXIT": "EXIT",
        "REDUCE": "REDUCE",
        "REDUCE_50": "REDUCE",
        "TAKE_PARTIAL_PROFIT": "REDUCE",
        "TAKE_PARTIAL_25": "REDUCE",
        "TAKE_PARTIAL_50": "REDUCE",
    }
    management = _read_json_object(
        project_root / "output" / "portfolio" / "position-management.json"
    )
    for row in management.get("positions", []):
        if not isinstance(row, dict):
            continue
        raw_action = str(row.get("advisory_action") or "").upper()
        action = action_map.get(raw_action)
        ticker = str(row.get("ticker") or "").upper()
        if not action or not ticker:
            continue
        advisories[(ticker, action)] = _exit_signal_payload(
            ticker=ticker,
            action=action,
            source="POSITION_MANAGEMENT",
            row=row,
            context=contexts.get(ticker, {}),
        )
    return [advisories[key] for key in sorted(advisories)]


def _exit_signal_payload(
    *,
    ticker: str,
    action: str,
    source: str,
    row: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    reasons = list(row.get("reason_codes") or [])
    if not reasons:
        reasons = [f"{source}_{action}_ADVISORY"]
    identity_payload = {
        "ticker": ticker,
        "action": action,
        "source": source,
        "strategy_id": row.get("strategy_id") or context.get("strategy_id"),
        "position_identity": row.get("position_identity"),
    }
    now = datetime.now(UTC)
    return {
        **context,
        "signal_id": "EXIT-" + _hash(identity_payload)[:20].upper(),
        "ticker": ticker,
        "asset": context.get("asset") or ticker,
        "strategy_id": identity_payload["strategy_id"] or source,
        "action": action,
        "lifecycle_status": action,
        "material_lifecycle_change": True,
        "source": source,
        "position_scope": (
            "OBSERVED_POSITION_ADVISORY"
            if source == "POSITION_MANAGEMENT"
            else "MODEL_LIFECYCLE_ONLY"
        ),
        "reason_codes": reasons,
        "reasons": reasons,
        "current_market_price": context.get("current_market_price")
        or context.get("market_reference_price"),
        "current_r": row.get("current_r"),
        "peak_r": row.get("peak_r"),
        "profit_giveback": row.get("profit_giveback"),
        "data_freshness": "FRESH",
        "expiration_timestamp": (now + timedelta(hours=2)).isoformat(),
        "automatic_execution_allowed": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def format_top5_message(report: dict[str, Any]) -> str:
    lines = [
        "TOP 5 SWING SIGNALS",
        "",
        "Hoofdweergave: gediversifieerd",
        "Execution authority: NONE",
    ]
    rows = report.get("signals", [])
    if not rows:
        lines.extend(["", "Geen kans voldeed aan de minimumscore."])
    for row in rows:
        lines.extend(
            [
                "",
                (
                    f"#{row.get('rank')} {row.get('symbol')} | "
                    f"{row.get('direction')} | {row.get('timeframe')}"
                ),
                (
                    f"Score {float(row.get('opportunity_score', 0)) * 100:.1f}/100"
                    f" | {row.get('eligibility_status')}"
                ),
                (
                    f"Entry {row.get('entry_zone_low')} - "
                    f"{row.get('entry_zone_high')} "
                    f"| stop {row.get('initial_stop')}"
                ),
                (
                    f"Targets {row.get('target_1')} / {row.get('target_2')} "
                    f"| geldig tot {row.get('valid_until')}"
                ),
            ]
        )
        blockers = row.get("manual_eligibility_blockers", [])
        if blockers:
            lines.append("Blokkers: " + ", ".join(str(item) for item in blockers))
    lines.extend(
        [
            "",
            (
                "Handmatig actionable: "
                f"{report.get('manual_signal_eligible_count', 0)}"
            ),
            (
                "Automatisch uitvoerbaar: "
                f"{report.get('automated_execution_eligible_count', 0)}"
            ),
            "Dit bericht plaatst geen IBKR-order.",
        ]
    )
    return "\n".join(lines)


def telegram_top5_preview(project_root: Path) -> dict[str, Any]:
    report = publish_top_signals(
        project_root,
        mode="diversified",
        limit=5,
    )
    return {
        "schema": "telegram_top_5_preview_v1",
        "status": report["status"],
        "message": format_top5_message(report),
        "signal_count": report.get("signal_count", 0),
        "sent": 0,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def telegram_send_top5(
    project_root: Path,
    notifier: TelegramNotifier | None = None,
) -> dict[str, Any]:
    settings = load_telegram_settings(project_root)
    notifier = notifier or TelegramNotifier(project_root, settings)
    report = publish_top_signals(
        project_root,
        mode="diversified",
        limit=5,
    )
    if report["status"] != "GO":
        return {
            "schema": "telegram_top_5_delivery_v1",
            "status": "BLOCKED",
            "reason": report.get("reason"),
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    queued = notifier.enqueue_text(
        message=format_top5_message(report),
        message_type="DIVERSIFIED_TOP_5",
        identity={
            "type": "DIVERSIFIED_TOP_5",
            "content_hash": report["content_hash"],
        },
    )
    delivery = notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
    return {
        "schema": "telegram_top_5_delivery_v1",
        "status": delivery["status"],
        "signal_count": report["signal_count"],
        "manual_signal_eligible_count": report[
            "manual_signal_eligible_count"
        ],
        "automated_execution_eligible_count": report[
            "automated_execution_eligible_count"
        ],
        "queued": queued,
        "delivery": delivery,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def telegram_send_pit_mtf(
    project_root: Path, notifier: TelegramNotifier | None = None
) -> dict[str, Any]:
    settings = load_telegram_settings(project_root)
    notifier = notifier or TelegramNotifier(project_root, settings)
    sources = {
        "robust_shortlist": (
            project_root / "output" / "signals" / "pit_mtf_signals.json"
        ),
        "ten_bps_positive_research": (
            project_root
            / "output"
            / "signals"
            / "pit_mtf_research_signals.json"
        ),
    }
    source_counts: dict[str, int] = {}
    rows_by_signal_id: dict[str, dict[str, Any]] = {}
    for source_name, source in sources.items():
        payload: Any = []
        if source.exists():
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = []
        source_rows = (
            [row for row in payload if isinstance(row, dict)]
            if isinstance(payload, list)
            else []
        )
        source_counts[source_name] = len(source_rows)
        for row in source_rows:
            signal_id = str(row.get("signal_id", ""))
            if not signal_id:
                continue
            if (
                signal_id not in rows_by_signal_id
                or source_name == "robust_shortlist"
            ):
                rows_by_signal_id[signal_id] = {
                    **row,
                    "notification_source": source_name,
                }
    rows = list(rows_by_signal_id.values())
    selected = _select_pit_mtf_digest_rows(rows, limit=10)
    with TelegramQueue(project_root) as queue:
        superseded_count = queue.supersede_pending_signal_prefix(
            signal_prefix="MTF-SIG-",
            message_types={"AANDEEL_WATCHLIST"},
            reason="SUPERSEDED_BY_COMPACT_MTF_DIGEST",
        )
    identity_rows = [
        {
            key: row.get(key)
            for key in (
                "signal_id",
                "data_timestamp",
                "current_market_price",
                "entry_zone_low",
                "entry_zone_high",
                "stop_loss",
                "take_profit_1",
                "take_profit_2",
            )
        }
        for row in selected
    ]
    queued = notifier.enqueue_text(
        message=_format_pit_mtf_digest(
            selected,
            total_count=len(rows),
            source_counts=source_counts,
        ),
        message_type="LOWER_TIMEFRAME_SHADOW_DIGEST",
        identity={
            "type": "PIT_MTF_COMPACT_SHADOW_DIGEST",
            "signals_hash": _hash(identity_rows),
        },
    )
    delivery = notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
    return {
        "schema": "telegram_pit_mtf_shadow_delivery_v1",
        "status": delivery["status"],
        "signals_considered": len(rows),
        "source_counts": source_counts,
        "deduplicated_signal_count": len(rows),
        "selected_signal_count": len(selected),
        "suppressed_low_priority_count": len(rows) - len(selected),
        "superseded_pending_count": superseded_count,
        "notification_count": 1,
        "enqueue_counts": {queued["status"]: 1},
        "queued": queued,
        "delivery": delivery,
        "failure_is_non_blocking": True,
        "signal_authority": "SHADOW_ONLY",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _select_pit_mtf_digest_rows(
    rows: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    timeframe_priority = {"1h": 0, "2h": 1, "4h": 2, "1d": 3, "1w": 4}

    def number(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row.get("notification_source") == "robust_shortlist" else 1,
            0 if row.get("current_pit_attested") is True else 1,
            0 if row.get("data_freshness") == "FRESH" else 1,
            -number(row, "confidence_score"),
            -number(row, "cost_50bps_median_pf"),
            -number(row, "median_oos_portfolio_pf"),
            timeframe_priority.get(str(row.get("timeframe")), 99),
            str(row.get("ticker") or row.get("asset") or ""),
            str(row.get("strategy_id") or ""),
        ),
    )
    selected: list[dict[str, Any]] = []
    symbol_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    deferred: list[dict[str, Any]] = []
    for row in ranked:
        symbol = str(row.get("ticker") or row.get("asset") or "UNKNOWN")
        timeframe = str(row.get("timeframe") or "UNKNOWN")
        if symbol_counts.get(symbol, 0) >= 2 or timeframe_counts.get(timeframe, 0) >= 4:
            deferred.append(row)
            continue
        selected.append(row)
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        if len(selected) >= limit:
            return selected
    for row in deferred:
        symbol = str(row.get("ticker") or row.get("asset") or "UNKNOWN")
        if symbol_counts.get(symbol, 0) >= 2:
            continue
        selected.append(row)
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _format_pit_mtf_digest(
    rows: list[dict[str, Any]],
    *,
    total_count: int,
    source_counts: dict[str, int],
) -> str:
    lines = [
        "Multi-timeframe research digest",
        (
            f"{len(rows)} gediversifieerde signalen uit {total_count} actieve "
            "researchvarianten"
        ),
        (
            "Robuuste shortlist "
            f"{source_counts.get('robust_shortlist', 0)} | 10bps research "
            f"{source_counts.get('ten_bps_positive_research', 0)}"
        ),
    ]
    for index, row in enumerate(rows, start=1):
        symbol = row.get("ticker") or row.get("asset") or "UNKNOWN"
        source = (
            "ROBUST"
            if row.get("notification_source") == "robust_shortlist"
            else "RESEARCH"
        )
        strategy = row.get("architecture") or row.get("strategy_id") or "n/a"
        confidence = _format_number(row.get("confidence_score"))
        lines.extend(
            [
                "",
                (
                    f"{index}. {symbol} {row.get('timeframe')} | {source} | "
                    f"confidence {confidence}"
                ),
                f"{strategy}",
                (
                    f"Entry {row.get('entry_zone_low')} - "
                    f"{row.get('entry_zone_high')} | stop {row.get('stop_loss')}"
                ),
                (
                    f"Targets {row.get('take_profit_1')} / "
                    f"{row.get('take_profit_2')} | data {row.get('data_timestamp')}"
                ),
            ]
        )
    if not rows:
        lines.extend(["", "Geen actueel gediversifieerd signaal beschikbaar."])
    lines.extend(
        [
            "",
            "Research shadow only. Geen orderintenties.",
            "Executionauthority: NONE",
            "IBKR-orders door dit bericht: 0",
        ]
    )
    return "\n".join(lines)


def telegram_send_market_digest(
    project_root: Path,
    notifier: TelegramNotifier | None = None,
) -> dict[str, Any]:
    settings = load_telegram_settings(project_root)
    if not settings.send_market_digest:
        return {
            "schema": "telegram_market_digest_delivery_v1",
            "status": "DISABLED_BY_CONFIG",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    notifier = notifier or TelegramNotifier(project_root, settings)
    digest = build_market_intelligence_digest(project_root)
    queued = notifier.enqueue_text(
        message=format_market_intelligence_digest(digest),
        message_type="MARKET_INTELLIGENCE_DIGEST",
        identity={
            "type": "MARKET_INTELLIGENCE_DIGEST",
            "date": str(digest.get("generated_at", ""))[:10],
            "delivery_window": "DAILY",
        },
    )
    delivery = notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
    return {
        "schema": "telegram_market_digest_delivery_v1",
        "status": delivery["status"],
        "digest_status": digest["status"],
        "news_count": digest["important_news_count"],
        "macro_event_count": digest["upcoming_macro_event_count"],
        "event_risk_within_24h": digest["event_risk_within_24h"],
        "queued": queued,
        "delivery": delivery,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def telegram_send_shadow_digest(
    project_root: Path,
    notifier: TelegramNotifier | None = None,
) -> dict[str, Any]:
    settings = load_telegram_settings(project_root)
    if not settings.send_signals:
        return {
            "schema": "telegram_lower_timeframe_shadow_delivery_v1",
            "status": "DISABLED_BY_CONFIG",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    path = (
        project_root
        / "output"
        / "research"
        / "phase11_12"
        / "latest-shadow-observation.json"
    )
    observation = _read_json_object(path)
    active = [
        row
        for row in observation.get("active_signals") or []
        if isinstance(row, dict)
    ]
    survivor_targets, survivor_audit = _phase11_14_survivor_targets(
        project_root
    )
    if not observation and not survivor_audit.get("source_available"):
        return {
            "schema": "telegram_lower_timeframe_shadow_delivery_v1",
            "status": "DATA_BLOCKED",
            "reason": "SHADOW_AND_SURVIVOR_OBSERVATIONS_MISSING",
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    if not active and not survivor_targets:
        return {
            "schema": "telegram_lower_timeframe_shadow_delivery_v1",
            "status": "GO",
            "outcome": "NO_ACTIVE_SHADOW_OR_SURVIVOR_SIGNALS",
            "signals_considered": 0,
            "survivor_targets_considered": 0,
            "survivor_observation_status": survivor_audit["status"],
            "notification_queued": False,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }

    active.sort(
        key=lambda row: (
            str(row.get("timeframe")),
            -float(row.get("score") or 0.0),
            str(row.get("formula")),
            str(row.get("symbol")),
        )
    )
    published = active[:6]
    survivor_targets.sort(
        key=lambda row: (
            str(row.get("timeframe")),
            -float(row.get("score") or 0.0),
            str(row.get("formula")),
            str(row.get("symbol")),
        )
    )
    published_survivors = survivor_targets[:6]
    digest = build_market_intelligence_digest(project_root)
    forward = observation.get("forward_evidence") or {}
    aggregate = forward.get("aggregate") or {}
    forward_pf = aggregate.get("net_profit_factor")
    forward_pf_text = (
        _format_number(forward_pf)
        if forward_pf is not None
        else str(aggregate.get("profit_factor_reason", "NOT_AVAILABLE"))
    )
    lines = [
        "Research shadow & frozen survivors",
        "",
        (
            f"Brede signalen: {len(active)} "
            f"(getoond: {len(published)})"
        ),
        (
            f"Frozen survivor targets: {len(survivor_targets)} "
            f"(getoond: {len(published_survivors)})"
        ),
        (
            "Prospective episodes: "
            f"{forward.get('closed_episode_count', 0)} gesloten, "
            f"{forward.get('open_episode_count', 0)} open, "
            f"{forward.get('pending_entry_count', 0)} pending entry"
        ),
        (
            f"Forward netto PF: {forward_pf_text} | "
            f"sample: {aggregate.get('sample_status', 'NOT_STARTED')}"
        ),
    ]
    for row in published:
        close = _format_number(row.get("reference_close"))
        stop = _format_number(row.get("illustrative_stop"))
        target_1 = _format_number(row.get("illustrative_target_1"))
        target_2 = _format_number(row.get("illustrative_target_2"))
        attestation = (
            "attested"
            if row.get("current_shariah_attestation")
            == "CURRENTLY_ATTESTED"
            else "not attested"
        )
        lines.extend(
            [
                "",
                (
                    f"{row.get('timeframe')} {row.get('symbol')} | "
                    f"{row.get('formula')} ({row.get('profile')})"
                ),
                (
                    f"Ref {close} | stop {stop} | "
                    f"targets {target_1}/{target_2}"
                ),
                f"Shariah: {attestation}",
            ]
        )
    if published_survivors:
        lines.extend(["", "Frozen nested survivor targets:"])
        for row in published_survivors:
            lines.extend(
                [
                    (
                        f"{row.get('timeframe')} {row.get('symbol')} | "
                        f"{row.get('formula')} ({row.get('profile')})"
                    ),
                    (
                        "Research target "
                        f"{float(row.get('target_weight', 0)):.1%} | "
                        f"score {_format_number(row.get('score'))}"
                    ),
                ]
            )
    portfolio = digest.get("portfolio_decision") or {}
    if portfolio.get("status") == "GO":
        top = list(portfolio.get("top_opportunities") or [])
        if top:
            lines.extend(["", "Centrale portfolio-ranking:"])
            for row in top[:3]:
                frames = "/".join(row.get("timeframes") or [])
                survivor_count = int(
                    row.get("survivor_strategy_count") or 0
                )
                lines.append(
                    f"- {row.get('ticker')} "
                    f"{float(row.get('score') or 0):.3f} | "
                    f"{frames or 'n/a'} | "
                    f"survivors {survivor_count}"
                )
    upcoming = [
        row
        for row in digest.get("upcoming_macro_events", [])
        if row.get("window_status") == "WITHIN_24H"
    ]
    if upcoming:
        lines.extend(["", "Event risk binnen 24 uur:"])
        for event in upcoming[:2]:
            lines.append(
                f"- {event.get('name')} at {event.get('scheduled_at')}"
            )
    affected_markets = sorted(
        {
            str(market)
            for item in digest.get("important_news", [])[:5]
            for market in item.get("affected_markets", [])
        }
    )
    if affected_markets:
        lines.append(
            "Nieuwscontext: " + ", ".join(affected_markets[:6])
        )
    lines.extend(
        [
            "",
            "Research shadow only. Geen orderintenties.",
            "Executionauthority: NONE",
            "IBKR-orders door dit bericht: 0",
        ]
    )
    message = "\n".join(lines)
    identity_rows = [
        {
            key: row.get(key)
            for key in (
                "strategy_id",
                "symbol",
                "timeframe",
                "closed_bar_timestamp",
                "action",
            )
        }
        for row in published
    ]
    survivor_identity_rows = [
        {
            key: row.get(key)
            for key in (
                "strategy_id",
                "symbol",
                "timeframe",
                "closed_bar_timestamp",
                "target_weight",
            )
        }
        for row in published_survivors
    ]
    identity = {
        "type": "LOWER_TIMEFRAME_SHADOW_DIGEST",
        "signals_hash": _hash(
            {
                "broad": identity_rows,
                "survivors": survivor_identity_rows,
            }
        ),
        "macro_event_ids": [
            str(row.get("event_id")) for row in upcoming[:2]
        ],
        "forward_state": {
            "pending_entry_count": forward.get("pending_entry_count", 0),
            "open_episode_count": forward.get("open_episode_count", 0),
            "pending_exit_count": forward.get("pending_exit_count", 0),
            "closed_episode_count": forward.get("closed_episode_count", 0),
            "sample_status": aggregate.get("sample_status"),
        },
    }
    notifier = notifier or TelegramNotifier(project_root, settings)
    queued = notifier.enqueue_text(
        message=message,
        message_type="LOWER_TIMEFRAME_SHADOW_DIGEST",
        identity=identity,
    )
    # This command shares a durable queue with every other notification
    # producer.  Bound foreground work to one due item so a backlog plus the
    # configured rate-limit sleeps cannot exceed the machine-step timeout.
    # The normal retry/drain command processes the remaining durable queue.
    delivery = notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
    report = {
        "schema": "telegram_lower_timeframe_shadow_delivery_v1",
        "status": delivery["status"],
        "signals_considered": len(active),
        "signals_published": len(published),
        "survivor_targets_considered": len(survivor_targets),
        "survivor_targets_published": len(published_survivors),
        "survivor_observation_status": survivor_audit["status"],
        "survivor_qualification_hash": survivor_audit.get(
            "qualification_hash"
        ),
        "event_risk_within_24h": bool(upcoming),
        "enqueue_status": queued["status"],
        "delivery": delivery,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    _write_json(
        project_root
        / "output"
        / "notifications"
        / "lower_timeframe_shadow_delivery.json",
        report,
    )
    return report


def _phase11_14_survivor_targets(
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = project_root / "output" / "research" / "phase11_14"
    status = _read_json_object(root / "status.json")
    observation = _read_json_object(
        root / "latest-forward-observation.json"
    )
    if not status and not observation:
        return [], {
            "status": "SOURCE_MISSING",
            "source_available": False,
        }
    boundary = status.get("qualification_boundary") or {}
    qualification_hash = boundary.get("qualification_hash")
    valid = (
        status.get("status") == "GO"
        and boundary.get("status") == "FROZEN"
        and observation.get("status") == "GO"
        and observation.get("qualification_hash") == qualification_hash
        and observation.get("EXECUTION_AUTHORITY") == "NONE"
        and int(observation.get("ORDER_CALLS") or 0) == 0
        and int(observation.get("BROKER_CALLS") or 0) == 0
    )
    if not valid:
        return [], {
            "status": "QUALIFICATION_OR_AUTHORITY_BLOCKED",
            "source_available": True,
            "qualification_hash": qualification_hash,
        }
    frozen_ids = {
        str(value)
        for value in boundary.get("robust_strategy_ids") or []
    }
    qualified = {
        str(row.get("strategy_id")): row
        for row in (
            (status.get("qualification") or {}).get("strategies") or []
        )
        if row.get("robust_pass")
        and row.get("portfolio_invariants_go")
        and row.get("forward_observer_candidate")
    }
    rows: list[dict[str, Any]] = []
    for item in observation.get("observations") or []:
        strategy_id = str(item.get("strategy_id") or "")
        if strategy_id not in frozen_ids or strategy_id not in qualified:
            continue
        score_by_symbol = {
            str(row.get("symbol") or "").upper(): float(
                row.get("score") or 0.0
            )
            for row in item.get("raw_active_signals") or []
            if isinstance(row, dict) and row.get("symbol")
        }
        for symbol, raw_weight in (
            item.get("current_attested_target_weights") or {}
        ).items():
            weight = float(raw_weight or 0.0)
            if weight <= 0:
                continue
            normalized_symbol = str(symbol).upper()
            rows.append(
                {
                    "source": "PHASE11_14_FROZEN_SURVIVOR",
                    "strategy_id": strategy_id,
                    "symbol": normalized_symbol,
                    "timeframe": item.get("timeframe"),
                    "formula": item.get("formula"),
                    "profile": item.get("profile"),
                    "closed_bar_timestamp": item.get(
                        "closed_bar_timestamp"
                    ),
                    "score": score_by_symbol.get(normalized_symbol, 0.0),
                    "target_weight": weight,
                    "current_shariah_attestation": (
                        "CURRENTLY_ATTESTED"
                    ),
                    "automatic_execution_allowed": False,
                    "execution_authority": "NONE",
                }
            )
    return rows, {
        "status": "GO",
        "source_available": True,
        "qualification_hash": qualification_hash,
        "robust_strategy_count": len(qualified),
        "target_count": len(rows),
    }


def _format_number(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def telegram_daily_delivery(
    project_root: Path,
    *,
    research: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        settings = load_telegram_settings(project_root)
        notifier = TelegramNotifier(project_root, settings)
        signals = telegram_send_latest(project_root, notifier)
        top_5 = telegram_send_top5(project_root, notifier)
        lower_timeframe_shadow = telegram_send_shadow_digest(
            project_root,
            notifier,
        )
        market_digest = telegram_send_market_digest(
            project_root,
            notifier,
        )
        autopilot: dict[str, Any] = {"status": "SKIPPED"}
        if settings.send_autopilot_summary and research:
            recovery = research.get("survivor_recovery") or {}
            message = (
                "🤖 Stocks research-autopilot afgerond\n\n"
                f"Status: {research.get('status', 'onbekend')}\n"
                f"Research survivors: {recovery.get('survivor_count', 'onbekend')}\n"
                "Automatische livepromoties: 0\n"
                "IBKR-orders geplaatst: 0"
            )
            autopilot = notifier.enqueue_text(
                message=message,
                message_type="AUTOPILOT_SUMMARY",
                identity={
                    "type": "AUTOPILOT_SUMMARY",
                    "cycle": research.get("last_cycle")
                    or research.get("next_cycle")
                    or datetime.now(UTC).date().isoformat(),
                    "status": research.get("status"),
                },
            )
            notifier.process(max_items=FOREGROUND_QUEUE_LIMIT)
        return {
            "status": "GO",
            "signals": signals,
            "diversified_top_5": top_5,
            "lower_timeframe_shadow": lower_timeframe_shadow,
            "market_intelligence_digest": market_digest,
            "autopilot_summary": autopilot,
            "failure_is_non_blocking": True,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
    except Exception as exc:
        return {
            "status": "DEGRADED",
            "error_code": type(exc).__name__,
            "failure_is_non_blocking": True,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }


def _latest_signals(project_root: Path) -> list[dict[str, Any]]:
    dynamic = project_root / "output" / "dynamic" / "current_signals.json"
    canonical = project_root / "output" / "signals" / "latest_signals.json"
    path = dynamic if dynamic.exists() else canonical
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = list(payload.get("signals") or [])
    pit_path = (
        project_root / "output" / "signals" / "pit_mtf_signals.json"
    )
    if pit_path.exists():
        try:
            pit_rows = json.loads(pit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pit_rows = []
        if isinstance(pit_rows, list):
            rows.extend(
                row for row in pit_rows if isinstance(row, dict)
            )
    rows = list(
        {
            str(row.get("signal_id") or index): row
            for index, row in enumerate(rows)
        }.values()
    )
    config_path = (
        project_root / "config" / "screener" / "daily_screener_v1.json"
    )
    etfs: set[str] = set()
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        etfs = {str(value).upper() for value in config.get("etf_symbols", [])}
    commodities = {
        "BNO",
        "COPX",
        "CPER",
        "DBA",
        "DBB",
        "DBC",
        "GLD",
        "PALL",
        "PPLT",
        "SCOP",
        "SLV",
        "SPPP",
        "UNG",
        "URA",
        "URNM",
        "USO",
    }
    bonds = {"BIL", "IEF", "TLT"}
    sectors = {
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    }
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker in etfs:
            row["is_etf"] = True
            row["asset_class"] = "ETF"
        if ticker in commodities:
            row["instrument_subtype"] = "COMMODITY_ETF"
        elif ticker in bonds:
            row["instrument_subtype"] = "BOND_ETF"
        elif ticker in sectors:
            row["instrument_subtype"] = "SECTOR_ETF"
    return rows


def _public_row(
    row: dict[str, Any], settings: TelegramSettings | None
) -> dict[str, Any]:
    return {
        "notification_id": row["notification_id"],
        "signal_id": row.get("signal_id"),
        "strategy_dna": row.get("strategy_dna"),
        "security_id": row.get("security_id"),
        "contract_id": row.get("contract_id"),
        "timestamp": row["created_at"],
        "message_type": row["message_type"],
        "delivery_status": row["status"],
        "retry_count": row["retry_count"],
        "sanitized_http_status": row.get("http_status"),
        "error_code": row.get("error_code"),
        "message_hash": row["identity_hash"],
        "masked_chat_identity": (
            settings.masked_chat_identity if settings else "not-published"
        ),
        "sent_at": row.get("sent_at"),
    }


def _message_type(signal: dict[str, Any]) -> str:
    action = str(signal.get("action", "")).upper()
    instrument = _instrument_type(signal).upper().replace("-", "_").replace(
        " ", "_"
    )
    if action == "WATCHLIST":
        return f"{instrument}_WATCHLIST"
    if action in EXITS:
        return f"{instrument}_EXIT"
    return f"{instrument}_SIGNAL"


def _instrument_type(signal: dict[str, Any]) -> str:
    raw = str(signal.get("asset_class") or "").upper()
    subtype = str(signal.get("instrument_subtype") or "").upper()
    if raw == "FUT":
        return "Commodity Future"
    if "COMMODITY" in subtype:
        return "Commodity-ETF"
    if "BOND" in subtype:
        return "Obligatie-ETF"
    if "SECTOR" in subtype:
        return "Sector-ETF"
    if raw in {"ETF", "BOND_ETF", "COMMODITY_ETF"} or signal.get("is_etf"):
        return "ETF"
    return "Aandeel"


def _is_future(signal: dict[str, Any]) -> bool:
    return str(signal.get("asset_class") or "").upper() == "FUT"


def _is_stale(signal: dict[str, Any]) -> bool:
    if str(signal.get("data_freshness") or "").upper() in {"STALE", "DELAYED_STALE"}:
        return True
    value = signal.get("expiration_timestamp")
    if not value:
        return False
    try:
        expiration = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if expiration.tzinfo is None:
            expiration = expiration.replace(tzinfo=UTC)
        return expiration < datetime.now(UTC)
    except ValueError:
        return True


def _retry_after(response: httpx.Response) -> float | None:
    if response.status_code != 429:
        return None
    try:
        value = response.json().get("parameters", {}).get("retry_after")
        return max(1.0, min(float(value), 3600.0)) if value is not None else 1.0
    except (ValueError, TypeError, AttributeError):
        return 1.0


def _error_code(exc: Exception) -> str:
    return type(exc).__name__.upper()


def _confidence(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result * 100.0 if result <= 1.0 else result


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _number(value: Any, decimals: int) -> str:
    number = _decimal(value)
    rendered = f"{number:,.{decimals}f}"
    return rendered.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")


def _money(value: Any, symbol: str) -> str:
    return f"{symbol}{_number(value, 2)}"


def _percent(value: Any) -> str:
    number = _decimal(value)
    if abs(number) <= 1:
        number *= 100
    return f"{_number(number, 1)}%"


def _quantity(value: Any) -> str:
    number = _decimal(value)
    label = "aandeel" if number == 1 else "aandelen"
    return f"{_number(number, 3).rstrip('0').rstrip(',')} {label}"


def _display_time(value: Any) -> str:
    if not value:
        return "niet gepubliceerd"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%d-%m-%Y %H:%M %Z")
    except ValueError:
        return "ongeldige timestamp"


def _text(value: Any) -> str | None:
    return None if value is None else str(value)


def _hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, default=str) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _notification_secret_scan(
    project_root: Path, settings: TelegramSettings
) -> bool:
    root = project_root / "output" / "notifications"
    for path in root.glob("*"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if settings.token and settings.token in content:
            return False
        if settings.chat_id and settings.chat_id in content:
            return False
    return True


def _delivery_breakdown(project_root: Path) -> dict[str, int]:
    signals = {
        str(row.get("signal_id")): row for row in _latest_signals(project_root)
    }
    counts = {
        "stock_signals": 0,
        "etf_signals": 0,
        "commodity_signals": 0,
        "futures_signals": 0,
        "watchlist_notifications": 0,
        "exit_notifications": 0,
        "risk_notifications": 0,
        "order_status_notifications": 0,
    }
    with TelegramQueue(project_root) as queue:
        rows = queue.connection.execute(
            """
            SELECT signal_id, message_type FROM notifications
            WHERE status='SENT'
            """
        ).fetchall()
    for row in rows:
        message_type = str(row["message_type"])
        if message_type in {"RISK_ALERT", "HMM_REGIME_ALERT"}:
            counts["risk_notifications"] += 1
        if message_type == "ORDER_EVENT":
            counts["order_status_notifications"] += 1
        if message_type.endswith("_EXIT"):
            counts["exit_notifications"] += 1
        signal = signals.get(str(row["signal_id"]))
        if signal is None:
            continue
        action = str(signal.get("action") or "")
        kind = _instrument_type(signal)
        if kind == "Commodity Future":
            counts["futures_signals"] += 1
        elif kind == "Commodity-ETF":
            counts["commodity_signals"] += 1
        elif "ETF" in kind:
            counts["etf_signals"] += 1
        else:
            counts["stock_signals"] += 1
        if action == "WATCHLIST":
            counts["watchlist_notifications"] += 1
    return counts
