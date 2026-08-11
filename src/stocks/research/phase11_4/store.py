from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


TABLES = (
    "preregistration",
    "universe_snapshots",
    "signals",
    "trades",
    "portfolio_orders",
    "portfolio_fills",
    "daily_equity",
    "cost_scenarios",
    "Shariah_cohorts",
    "robustness_runs",
    "bootstrap_runs",
    "concentration_results",
    "decisions",
    "provenance",
)


class Phase114Store:
    """Append-only private evidence store; public artifacts contain aggregates only."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            for table in TABLES:
                db.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS [{table}](
                      row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      run_id TEXT NOT NULL,
                      economic_key TEXT NOT NULL,
                      payload_hash TEXT NOT NULL,
                      payload_json TEXT NOT NULL,
                      created_at TEXT NOT NULL,
                      UNIQUE(run_id, economic_key, payload_hash)
                    )
                    """
                )
            db.commit()

    def append(self, table: str, run_id: str, rows: Iterable[tuple[str, dict[str, Any]]]) -> int:
        if table not in TABLES:
            raise ValueError(f"unknown Phase 11.4 table: {table}")
        from stocks.execution.idempotency import stable_hash

        self.initialize()
        now = datetime.now(UTC).isoformat()
        count = 0
        with sqlite3.connect(self.path) as db:
            for key, payload in rows:
                cursor = db.execute(
                    f"INSERT OR IGNORE INTO [{table}](run_id,economic_key,payload_hash,payload_json,created_at) VALUES(?,?,?,?,?)",
                    (run_id, key, stable_hash(payload), json.dumps(payload, sort_keys=True, default=str), now),
                )
                count += max(0, int(cursor.rowcount))
            db.commit()
        return count

    def latest(self, table: str) -> list[dict[str, Any]]:
        if table not in TABLES:
            raise ValueError(f"unknown Phase 11.4 table: {table}")
        self.initialize()
        with sqlite3.connect(self.path) as db:
            run = db.execute(f"SELECT run_id FROM [{table}] ORDER BY row_id DESC LIMIT 1").fetchone()
            if not run:
                return []
            rows = db.execute(
                f"SELECT payload_json FROM [{table}] WHERE run_id=? ORDER BY row_id", (run[0],)
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def counts(self) -> dict[str, int]:
        self.initialize()
        with sqlite3.connect(self.path) as db:
            return {table: int(db.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]) for table in TABLES}
