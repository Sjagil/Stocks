from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash


TABLES = (
    "providers",
    "provider_capabilities",
    "datasets",
    "ingestion_runs",
    "universe_memberships",
    "symbols",
    "prices",
    "corporate_actions",
    "fundamental_raw",
    "fundamental_versions",
    "earnings_raw",
    "earnings_versions",
    "news_raw",
    "news_versions",
    "filings",
    "filing_facts",
    "fx_rates",
    "provenance",
    "data_quality_events",
)


class PitFoundationStore:
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
                      row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                      economic_key TEXT NOT NULL,
                      version_number INTEGER NOT NULL,
                      payload_hash TEXT NOT NULL,
                      payload_json TEXT NOT NULL,
                      ingested_at TEXT NOT NULL,
                      UNIQUE(economic_key, version_number),
                      UNIQUE(economic_key, payload_hash)
                    )
                    """
                )
            connection.commit()

    def append_version(self, table: str, economic_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        if table not in TABLES:
            raise ValueError(f"Unknown Phase 11.2 table: {table}")
        payload_hash = stable_hash(payload)
        with self.connect() as connection:
            same = connection.execute(
                f"SELECT version_number FROM {table} WHERE economic_key=? AND payload_hash=?",
                (economic_key, payload_hash),
            ).fetchone()
            if same:
                return {"status": "IDEMPOTENT_REPLAY", "version_number": int(same["version_number"]), "payload_hash": payload_hash}
            latest = connection.execute(
                f"SELECT COALESCE(MAX(version_number),0) FROM {table} WHERE economic_key=?",
                (economic_key,),
            ).fetchone()[0]
            version = int(latest) + 1
            connection.execute(
                f"INSERT INTO {table}(economic_key,version_number,payload_hash,payload_json,ingested_at) VALUES (?,?,?,?,?)",
                (economic_key, version, payload_hash, json.dumps(payload, sort_keys=True, default=str), datetime.now(UTC).isoformat()),
            )
            connection.commit()
        return {"status": "VERSION_APPENDED", "version_number": version, "payload_hash": payload_hash}

    def records(self, table: str, economic_key: str | None = None) -> list[dict[str, Any]]:
        if table not in TABLES:
            raise ValueError(f"Unknown Phase 11.2 table: {table}")
        query = f"SELECT economic_key,version_number,payload_hash,payload_json,ingested_at FROM {table}"
        params: tuple[Any, ...] = ()
        if economic_key is not None:
            query += " WHERE economic_key=?"
            params = (economic_key,)
        query += " ORDER BY row_id"
        with self.connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "economic_key": row["economic_key"],
                "version_number": row["version_number"],
                "payload_hash": row["payload_hash"],
                "payload": json.loads(row["payload_json"]),
                "ingested_at": row["ingested_at"],
            }
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in TABLES}


def phase11_2_private_root(project_root: Path) -> Path:
    return project_root / "data" / "research" / "phase11_2" / "private"


def phase11_2_database_path(project_root: Path) -> Path:
    return phase11_2_private_root(project_root) / "pit_foundation.sqlite3"


def ensure_private_layout(project_root: Path) -> dict[str, Path]:
    root = phase11_2_private_root(project_root)
    paths = {name: root / name for name in ("raw", "normalized", "manifests")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
