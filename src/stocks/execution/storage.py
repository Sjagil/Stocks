from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS intents (
  intent_id TEXT PRIMARY KEY,
  economic_order_key TEXT NOT NULL UNIQUE,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  aggregate_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
  fill_id TEXT PRIMARY KEY,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


class ExecutionLedgerStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> dict[str, object]:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        return {"status": "GO", "database_path": str(self.path)}

    def register_intent(self, payload: dict[str, Any]) -> str:
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = stable_hash(payload)
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT payload_hash FROM intents WHERE economic_order_key = ?",
                (payload["economic_order_key"],),
            ).fetchone()
            if existing:
                return "IDEMPOTENT_REPLAY" if existing["payload_hash"] == payload_hash else "IDEMPOTENCY_CONFLICT_BLOCKED"
            conn.execute(
                "INSERT INTO intents(intent_id, economic_order_key, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (payload["intent_id"], payload["economic_order_key"], payload_hash, payload_json, payload["created_at"]),
            )
            conn.commit()
        return "INTENT_REGISTERED"

    def append_event(self, aggregate_id: str, event_type: str, payload: dict[str, Any], created_at: str) -> None:
        payload_hash = stable_hash({"aggregate_id": aggregate_id, "event_type": event_type, "payload": payload, "created_at": created_at})
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO events(aggregate_id, event_type, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (aggregate_id, event_type, payload_hash, json.dumps(payload, sort_keys=True, default=str), created_at),
            )
            conn.commit()

    def append_fill_once(self, fill_id: str, payload: dict[str, Any], created_at: str) -> str:
        payload_hash = stable_hash(payload)
        with self.connect() as conn:
            existing = conn.execute("SELECT payload_hash FROM fills WHERE fill_id = ?", (fill_id,)).fetchone()
            if existing:
                return "DUPLICATE_FILL_BLOCKED"
            conn.execute(
                "INSERT INTO fills(fill_id, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (fill_id, payload_hash, json.dumps(payload, sort_keys=True, default=str), created_at),
            )
            conn.commit()
        return "FILL_RECORDED"

    def read_events(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM events ORDER BY event_id").fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                "intent_count": int(conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]),
                "event_count": int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
                "fill_count": int(conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]),
            }

