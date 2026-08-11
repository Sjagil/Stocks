from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.research.autopilot.contracts import stable_hash


class SignalStore:
    def __init__(self, project_root: Path) -> None:
        path = project_root / "data" / "signals" / "private" / "signals.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS authorities (
                strategy_id TEXT PRIMARY KEY,
                strategy_hash TEXT NOT NULL,
                authority TEXT NOT NULL,
                promoted_at TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                strategy_id TEXT NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                lifecycle_status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS signal_events (
                event_id TEXT PRIMARY KEY,
                signal_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS manual_executions (
                execution_id TEXT PRIMARY KEY,
                signal_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                quantity TEXT NOT NULL,
                fill_price TEXT NOT NULL,
                reason TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS manual_positions (
                position_id TEXT PRIMARY KEY,
                signal_id TEXT NOT NULL UNIQUE,
                ticker TEXT NOT NULL,
                environment TEXT NOT NULL DEFAULT 'paper',
                con_id INTEGER,
                contract_hash TEXT,
                quantity TEXT NOT NULL,
                fill_price TEXT NOT NULL,
                ownership_status TEXT NOT NULL,
                lifecycle_status TEXT NOT NULL,
                broker_match_status TEXT NOT NULL,
                automatic_execution_eligible INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                broker_snapshot_hash TEXT,
                broker_matched_at TEXT,
                opened_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS manual_position_events (
                event_id TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                from_ownership TEXT,
                to_ownership TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._ensure_column(
            "manual_positions",
            "environment",
            "TEXT NOT NULL DEFAULT 'paper'",
        )
        self._ensure_column("manual_positions", "con_id", "INTEGER")
        self._ensure_column("manual_positions", "contract_hash", "TEXT")
        self._ensure_column(
            "manual_positions",
            "broker_snapshot_hash",
            "TEXT",
        )
        self._ensure_column(
            "manual_positions",
            "broker_matched_at",
            "TEXT",
        )
        self.connection.commit()

    def _ensure_column(
        self,
        table: str,
        column: str,
        definition: str,
    ) -> None:
        columns = {
            str(row["name"])
            for row in self.connection.execute(
                f'PRAGMA table_info("{table}")'
            )
        }
        if column not in columns:
            self.connection.execute(
                f'ALTER TABLE "{table}" ADD COLUMN "{column}" {definition}'
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SignalStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def promote(self, strategy_id: str, strategy_hash: str, evidence: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO authorities
                (strategy_id, strategy_hash, authority, promoted_at, evidence_json)
                VALUES (?, ?, 'MANUAL_ACTIONABLE', ?, ?)
                ON CONFLICT(strategy_id) DO UPDATE SET
                    strategy_hash=excluded.strategy_hash,
                    authority=excluded.authority,
                    promoted_at=excluded.promoted_at,
                    evidence_json=excluded.evidence_json
                """,
                (strategy_id, strategy_hash, now, json.dumps(evidence, default=str)),
            )

    def authorities(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute("SELECT * FROM authorities")]

    def append_signal(self, payload: dict[str, Any]) -> bool:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        digest = stable_hash(payload)
        existing = self.connection.execute(
            "SELECT content_hash, lifecycle_status FROM signals WHERE signal_id=?",
            (payload["signal_id"],),
        ).fetchone()
        terminal = {
            "CLOSED",
            "CANCELLED",
            "EXECUTED_MANUALLY",
            "PARTIALLY_EXECUTED_MANUALLY",
            "STOPPED_OUT",
            "TP1_REACHED",
            "TP2_REACHED",
        }
        if existing is not None and existing["lifecycle_status"] in terminal:
            return False
        if (
            existing is not None
            and existing["content_hash"] == digest
            and existing["lifecycle_status"] == payload["lifecycle_status"]
        ):
            return False
        now = datetime.now(UTC).isoformat()
        with self.connection:
            if existing is None:
                cursor = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO signals
                    (signal_id, strategy_id, ticker, action, lifecycle_status,
                     payload_json, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["signal_id"],
                        payload["strategy_id"],
                        payload["ticker"],
                        payload["action"],
                        payload["lifecycle_status"],
                        encoded,
                        digest,
                        now,
                    ),
                )
                changed = bool(cursor.rowcount)
                event_type = "SIGNAL_CREATED"
            else:
                self.connection.execute(
                    """
                    UPDATE signals
                    SET strategy_id=?, ticker=?, action=?, lifecycle_status=?,
                        payload_json=?, content_hash=?, created_at=?
                    WHERE signal_id=?
                    """,
                    (
                        payload["strategy_id"],
                        payload["ticker"],
                        payload["action"],
                        payload["lifecycle_status"],
                        encoded,
                        digest,
                        now,
                        payload["signal_id"],
                    ),
                )
                changed = True
                event_type = "SIGNAL_STATE_REFRESHED"
            if changed:
                event_id = "SEV-" + stable_hash(
                    {
                        "signal_id": payload["signal_id"],
                        "content_hash": digest,
                        "event_type": event_type,
                    }
                )[:24]
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO signal_events
                    (event_id, signal_id, event_type, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        payload["signal_id"],
                        event_type,
                        json.dumps(
                            {
                                "content_hash": digest,
                                "lifecycle_status": payload[
                                    "lifecycle_status"
                                ],
                            },
                            sort_keys=True,
                        ),
                        now,
                    ),
                )
        return changed

    def signals(self, status: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT payload_json, lifecycle_status FROM signals"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE lifecycle_status=?"
            params = (status,)
        query += " ORDER BY created_at DESC"
        rows = []
        for row in self.connection.execute(query, params):
            payload = json.loads(row["payload_json"])
            payload["lifecycle_status"] = row["lifecycle_status"]
            rows.append(payload)
        return rows

    def signal(self, signal_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload_json, lifecycle_status FROM signals WHERE signal_id=?",
            (signal_id,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["lifecycle_status"] = row["lifecycle_status"]
        return payload

    def append_manual_execution(
        self,
        *,
        signal_id: str,
        event_type: str,
        quantity: str,
        fill_price: str,
        reason: str | None,
        payload: dict[str, Any],
        new_status: str,
    ) -> str:
        now = datetime.now(UTC).isoformat()
        execution_id = "MAN-" + stable_hash(
            {
                "signal_id": signal_id,
                "event_type": event_type,
                "quantity": quantity,
                "fill_price": fill_price,
                "reason": reason,
                "created_at": now,
            }
        )[:24]
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO manual_executions
                (execution_id, signal_id, event_type, quantity, fill_price,
                 reason, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    execution_id,
                    signal_id,
                    event_type,
                    quantity,
                    fill_price,
                    reason,
                    json.dumps(payload, default=str),
                    now,
                ),
            )
            self.connection.execute(
                "UPDATE signals SET lifecycle_status=? WHERE signal_id=?",
                (new_status, signal_id),
            )
        return execution_id

    def register_manual_position(
        self,
        *,
        signal_id: str,
        environment: str,
        con_id: int | None,
        contract_hash: str | None,
        quantity: str,
        fill_price: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        signal = self.signal(signal_id)
        if signal is None:
            raise ValueError("SIGNAL_NOT_FOUND")
        existing = self.connection.execute(
            """
            SELECT * FROM manual_positions WHERE signal_id=?
            """,
            (signal_id,),
        ).fetchone()
        if existing is not None:
            row = dict(existing)
            if (
                row["quantity"] == quantity
                and row["fill_price"] == fill_price
                and row["environment"] == environment
                and row["con_id"] == con_id
                and row["lifecycle_status"] == "OPEN"
            ):
                row["registration_status"] = (
                    "MANUAL_POSITION_ALREADY_REGISTERED"
                )
                return row
            raise ValueError("MANUAL_POSITION_REGISTRATION_CONFLICT")
        now = datetime.now(UTC).isoformat()
        position_id = "MPOS-" + stable_hash(
            {
                "signal_id": signal_id,
                "ticker": signal["ticker"],
            }
        )[:24]
        execution_id = "MAN-" + stable_hash(
            {
                "position_id": position_id,
                "event_type": "MANUAL_ENTRY",
                "quantity": quantity,
                "fill_price": fill_price,
                "created_at": now,
            }
        )[:24]
        event_id = "MPE-" + stable_hash(
            {
                "position_id": position_id,
                "event_type": "MANUAL_POSITION_REGISTERED",
                "created_at": now,
            }
        )[:24]
        encoded = json.dumps(payload, sort_keys=True, default=str)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO manual_executions
                (execution_id, signal_id, event_type, quantity, fill_price,
                 reason, payload_json, created_at)
                VALUES (?, ?, 'MANUAL_ENTRY', ?, ?, NULL, ?, ?)
                """,
                (
                    execution_id,
                    signal_id,
                    quantity,
                    fill_price,
                    encoded,
                    now,
                ),
            )
            self.connection.execute(
                """
                UPDATE signals SET lifecycle_status='EXECUTED_MANUALLY'
                WHERE signal_id=?
                """,
                (signal_id,),
            )
            self.connection.execute(
                """
                INSERT INTO manual_positions
                (position_id, signal_id, ticker, environment, con_id,
                 contract_hash, quantity, fill_price,
                 ownership_status, lifecycle_status, broker_match_status,
                 automatic_execution_eligible, payload_json, opened_at,
                 updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'MANUAL_TRACKED', 'OPEN',
                        'UNVERIFIED', 0, ?, ?, ?)
                """,
                (
                    position_id,
                    signal_id,
                    signal["ticker"],
                    environment,
                    con_id,
                    contract_hash,
                    quantity,
                    fill_price,
                    encoded,
                    now,
                    now,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO manual_position_events
                (event_id, position_id, event_type, from_ownership,
                 to_ownership, payload_json, created_at)
                VALUES (?, ?, 'MANUAL_POSITION_REGISTERED', NULL,
                        'MANUAL_TRACKED', ?, ?)
                """,
                (event_id, position_id, encoded, now),
            )
        return {
            "position_id": position_id,
            "signal_id": signal_id,
            "ticker": signal["ticker"],
            "environment": environment,
            "con_id": con_id,
            "contract_hash": contract_hash,
            "ownership_status": "MANUAL_TRACKED",
            "lifecycle_status": "OPEN",
            "broker_match_status": "UNVERIFIED",
            "automatic_execution_eligible": 0,
            "registration_status": "MANUAL_POSITION_REGISTERED",
        }

    def set_manual_position_broker_match(
        self,
        *,
        position_id: str,
        broker_match_status: str,
        snapshot_hash: str,
        detail: dict[str, Any],
    ) -> dict[str, Any] | None:
        existing = self.connection.execute(
            "SELECT * FROM manual_positions WHERE position_id=?",
            (position_id,),
        ).fetchone()
        if existing is None:
            return None
        row = dict(existing)
        if row["lifecycle_status"] != "OPEN":
            raise ValueError("MANUAL_POSITION_NOT_OPEN")
        now = datetime.now(UTC).isoformat()
        event_id = "MPE-" + stable_hash(
            {
                "position_id": position_id,
                "event_type": "BROKER_MATCH_AUDITED",
                "broker_match_status": broker_match_status,
                "snapshot_hash": snapshot_hash,
                "created_at": now,
            }
        )[:24]
        with self.connection:
            self.connection.execute(
                """
                UPDATE manual_positions
                SET broker_match_status=?, broker_snapshot_hash=?,
                    broker_matched_at=?, automatic_execution_eligible=0,
                    updated_at=?
                WHERE position_id=?
                """,
                (
                    broker_match_status,
                    snapshot_hash,
                    now,
                    now,
                    position_id,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO manual_position_events
                (event_id, position_id, event_type, from_ownership,
                 to_ownership, payload_json, created_at)
                VALUES (?, ?, 'BROKER_MATCH_AUDITED', ?, ?, ?, ?)
                """,
                (
                    event_id,
                    position_id,
                    row["ownership_status"],
                    row["ownership_status"],
                    json.dumps(detail, sort_keys=True, default=str),
                    now,
                ),
            )
        return {
            **row,
            "broker_match_status": broker_match_status,
            "broker_snapshot_hash": snapshot_hash,
            "broker_matched_at": now,
            "automatic_execution_eligible": 0,
            "transition_status": "BROKER_MATCH_AUDITED",
        }

    def manual_position(
        self,
        position_id: str,
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM manual_positions WHERE position_id=?",
            (position_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def transition_manual_position(
        self,
        *,
        position_id: str,
        to_ownership: str,
        event_type: str,
    ) -> dict[str, Any] | None:
        existing = self.connection.execute(
            "SELECT * FROM manual_positions WHERE position_id=?",
            (position_id,),
        ).fetchone()
        if existing is None:
            return None
        row = dict(existing)
        if row["lifecycle_status"] != "OPEN":
            raise ValueError("MANUAL_POSITION_NOT_OPEN")
        if row["ownership_status"] == to_ownership:
            row["transition_status"] = "IDEMPOTENT_NO_CHANGE"
            return row
        allowed = {
            ("MANUAL_TRACKED", "BOT_MANAGED"),
            ("BOT_MANAGED", "MANUAL_TRACKED"),
        }
        transition = (row["ownership_status"], to_ownership)
        if transition not in allowed:
            raise ValueError("MANUAL_POSITION_OWNERSHIP_TRANSITION_BLOCKED")
        now = datetime.now(UTC).isoformat()
        event_id = "MPE-" + stable_hash(
            {
                "position_id": position_id,
                "event_type": event_type,
                "from": row["ownership_status"],
                "to": to_ownership,
                "created_at": now,
            }
        )[:24]
        payload = {
            "broker_match_status": row["broker_match_status"],
            "automatic_execution_eligible": False,
        }
        with self.connection:
            self.connection.execute(
                """
                UPDATE manual_positions
                SET ownership_status=?, automatic_execution_eligible=0,
                    updated_at=?
                WHERE position_id=?
                """,
                (to_ownership, now, position_id),
            )
            self.connection.execute(
                """
                INSERT INTO manual_position_events
                (event_id, position_id, event_type, from_ownership,
                 to_ownership, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    position_id,
                    event_type,
                    row["ownership_status"],
                    to_ownership,
                    json.dumps(payload, sort_keys=True),
                    now,
                ),
            )
        return {
            **row,
            "ownership_status": to_ownership,
            "automatic_execution_eligible": 0,
            "transition_status": event_type,
        }

    def manual_positions(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT * FROM manual_positions
                ORDER BY opened_at, position_id
                """
            )
        ]
