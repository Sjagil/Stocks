from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash


COUNTERS = {
    "financial_calls": 0,
    "order_calls": 0,
    "cancel_calls": 0,
    "account_calls": 0,
    "position_calls": 0,
    "market_data_calls": 0,
    "historical_data_calls": 0,
}


@dataclass(frozen=True)
class Phase82Layout:
    project_root: Path
    output_dir: Path
    data_dir: Path
    db_path: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "Phase82Layout":
        return cls(
            project_root=project_root,
            output_dir=project_root / "output" / "shadow" / "phase8_2",
            data_dir=project_root / "data" / "shadow" / "phase8_2",
            db_path=project_root / "data" / "shadow" / "phase8_2" / "shadow_ledger.sqlite3",
        )

    def artifact(self, name: str) -> Path:
        return self.output_dir / name


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS strategies (
  strategy_id TEXT NOT NULL,
  strategy_version TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(strategy_id, strategy_version)
);
CREATE TABLE IF NOT EXISTS decisions (
  decision_id TEXT PRIMARY KEY,
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
CREATE TABLE IF NOT EXISTS signals (
  signal_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS target_portfolios (
  decision_id TEXT PRIMARY KEY,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fills (
  fill_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_id TEXT PRIMARY KEY,
  decision_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evaluations (
  decision_id TEXT PRIMARY KEY,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


class ShadowLedgerStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self, *, reset: bool = False) -> dict[str, Any]:
        if reset and self.path.exists():
            self.path.unlink()
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        return {"status": "GO", "database_path": str(self.path), **self.counts()}

    def register_strategy(self, payload: dict[str, Any]) -> str:
        payload_hash = stable_hash(payload)
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT payload_hash FROM strategies WHERE strategy_id = ? AND strategy_version = ?",
                (payload["strategy_id"], payload["strategy_version"]),
            ).fetchone()
            if existing:
                return "STRATEGY_IDEMPOTENT" if existing["payload_hash"] == payload_hash else "STRATEGY_CONFLICT_BLOCKED"
            conn.execute(
                "INSERT INTO strategies(strategy_id, strategy_version, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (payload["strategy_id"], payload["strategy_version"], payload_hash, _json(payload), now_iso()),
            )
            conn.commit()
        return "STRATEGY_REGISTERED"

    def append_decision(self, payload: dict[str, Any]) -> str:
        return self._insert_once("decisions", "decision_id", payload["decision_id"], payload, "DECISION_CREATED", "DUPLICATE_DECISION", "DECISION_ID_CONFLICT")

    def append_signal(self, signal_id: str, decision_id: str, payload: dict[str, Any]) -> str:
        return self._insert_child_once("signals", "signal_id", signal_id, decision_id, payload, "SIGNAL_RECORDED", "DUPLICATE_SIGNAL_BLOCKED")

    def append_target(self, decision_id: str, payload: dict[str, Any]) -> str:
        return self._insert_once("target_portfolios", "decision_id", decision_id, payload, "TARGET_RECORDED", "DUPLICATE_TARGET", "TARGET_CONFLICT_BLOCKED")

    def append_fill(self, fill_id: str, decision_id: str, payload: dict[str, Any]) -> str:
        return self._insert_child_once("fills", "fill_id", fill_id, decision_id, payload, "FILL_RECORDED", "DUPLICATE_FILL_BLOCKED")

    def append_snapshot(self, snapshot_id: str, decision_id: str, payload: dict[str, Any]) -> str:
        return self._insert_child_once("snapshots", "snapshot_id", snapshot_id, decision_id, payload, "SNAPSHOT_RECORDED", "DUPLICATE_SNAPSHOT_BLOCKED")

    def append_evaluation(self, decision_id: str, payload: dict[str, Any]) -> str:
        return self._insert_once("evaluations", "decision_id", decision_id, payload, "EVALUATION_RECORDED", "DUPLICATE_EVALUATION", "EVALUATION_CONFLICT_BLOCKED")

    def append_event(self, aggregate_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO events(aggregate_id, event_type, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (aggregate_id, event_type, stable_hash(payload), _json(payload), now_iso()),
            )
            conn.commit()

    def _insert_once(self, table: str, key_name: str, key: str, payload: dict[str, Any], inserted: str, duplicate: str, conflict: str) -> str:
        payload_hash = stable_hash(payload)
        with self.connect() as conn:
            existing = conn.execute(f"SELECT payload_hash FROM {table} WHERE {key_name} = ?", (key,)).fetchone()
            if existing:
                return duplicate if existing["payload_hash"] == payload_hash else conflict
            conn.execute(
                f"INSERT INTO {table}({key_name}, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (key, payload_hash, _json(payload), now_iso()),
            )
            conn.commit()
        self.append_event(key, inserted, {"table": table, key_name: key, "payload_hash": payload_hash})
        return inserted

    def _insert_child_once(self, table: str, key_name: str, key: str, decision_id: str, payload: dict[str, Any], inserted: str, duplicate: str) -> str:
        payload_hash = stable_hash(payload)
        with self.connect() as conn:
            existing = conn.execute(f"SELECT payload_hash FROM {table} WHERE {key_name} = ?", (key,)).fetchone()
            if existing:
                return duplicate
            conn.execute(
                f"INSERT INTO {table}({key_name}, decision_id, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (key, decision_id, payload_hash, _json(payload), now_iso()),
            )
            conn.commit()
        self.append_event(decision_id, inserted, {"table": table, key_name: key, "payload_hash": payload_hash})
        return inserted

    def read_table(self, table: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
        return [{**dict(row), "payload": json.loads(row["payload_json"])} for row in rows]

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                "strategy_count": int(conn.execute("SELECT COUNT(*) FROM strategies").fetchone()[0]),
                "decision_count": int(conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]),
                "signal_count": int(conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]),
                "target_portfolio_count": int(conn.execute("SELECT COUNT(*) FROM target_portfolios").fetchone()[0]),
                "fill_count": int(conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]),
                "snapshot_count": int(conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]),
                "evaluation_count": int(conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]),
                "event_count": int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
            }


def artifact(schema: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
    body = {
        "schema": schema,
        "generated_at": now_iso(),
        **payload,
        **COUNTERS,
    }
    body["content_hash"] = stable_hash({key: value for key, value in body.items() if key != "content_hash"})
    return body


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def file_hashes(project_root: Path, paths: list[str]) -> dict[str, str | None]:
    return {path: sha256_file(project_root / path) for path in paths}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
