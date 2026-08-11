from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash


TABLES = (
    "signals",
    "strategy_decisions",
    "shariah_decisions",
    "risk_decisions",
    "shadow_intents",
    "hypothetical_orders",
    "hypothetical_fills",
    "positions",
    "commissions",
    "pnl",
    "kill_switches",
    "reconciliation_episodes",
)


class AutoPaperStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            for table in TABLES:
                connection.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                      record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      economic_key TEXT NOT NULL,
                      payload_hash TEXT NOT NULL,
                      payload_json TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      UNIQUE(economic_key)
                    )
                    """
                )
            connection.commit()

    def append_once(self, table: str, economic_key: str, payload: dict[str, Any]) -> str:
        if table not in TABLES:
            raise ValueError(f"Unknown Phase 10 table: {table}")
        payload_hash = stable_hash(payload)
        with self.connect() as connection:
            existing = connection.execute(
                f"SELECT payload_hash FROM {table} WHERE economic_key = ?",
                (economic_key,),
            ).fetchone()
            if existing is not None:
                return "IDEMPOTENT_REPLAY" if existing["payload_hash"] == payload_hash else "ECONOMIC_KEY_CONFLICT"
            connection.execute(
                f"INSERT INTO {table}(economic_key,payload_hash,payload_json,created_at) VALUES (?,?,?,?)",
                (economic_key, payload_hash, json.dumps(payload, sort_keys=True, default=str), datetime.now(UTC).isoformat()),
            )
            connection.commit()
        return "RECORDED"

    def count(self, table: str) -> int:
        with self.connect() as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    def counts(self) -> dict[str, int]:
        return {table: self.count(table) for table in TABLES}

    def records(self, table: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT economic_key,payload_hash,payload_json,created_at FROM {table} ORDER BY record_id"
            ).fetchall()
        return [
            {
                "economic_key": row["economic_key"],
                "payload_hash": row["payload_hash"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]


def phase10_database_path(project_root: Path) -> Path:
    return project_root / "data" / "execution" / "phase10" / "private" / "auto_paper.sqlite3"
