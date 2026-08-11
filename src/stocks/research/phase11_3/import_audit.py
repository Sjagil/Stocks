from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .datascraper_adapter import DatascraperAdapter
from .export_manifest import stable_hash


def database_path(root: Path) -> Path:
    return root / "data" / "research" / "phase11_3" / "private" / "causal_research.sqlite3"


class Phase113Store:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS exports(
                  export_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  batch_id TEXT NOT NULL UNIQUE, dataset TEXT NOT NULL,
                  manifest_hash TEXT NOT NULL, manifest_json TEXT NOT NULL,
                  imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS records(
                  record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  batch_id TEXT NOT NULL, dataset TEXT NOT NULL,
                  economic_key TEXT NOT NULL, payload_hash TEXT NOT NULL,
                  payload_json TEXT NOT NULL, imported_at TEXT NOT NULL,
                  UNIQUE(dataset,economic_key,payload_hash)
                );
                CREATE TABLE IF NOT EXISTS research_events(
                  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_type TEXT NOT NULL, economic_key TEXT NOT NULL,
                  payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  UNIQUE(event_type,economic_key,payload_hash)
                );
                """
            )

    def import_from(self, adapter: DatascraperAdapter) -> dict[str, Any]:
        self.initialize()
        statuses: list[dict[str, Any]] = []
        accepted_rows = 0
        with sqlite3.connect(self.path) as db:
            for validation in adapter.manifests():
                manifest = validation.manifest
                batch_id = str(manifest.get("batch_id") or validation.path.parent.name)
                if not validation.valid:
                    statuses.append({"batch_id": batch_id, "dataset": manifest.get("dataset"), "status": validation.status, "record_count": 0})
                    continue
                existing = db.execute("SELECT 1 FROM exports WHERE batch_id=?", (batch_id,)).fetchone()
                if existing:
                    statuses.append({"batch_id": batch_id, "dataset": manifest["dataset"], "status": "DATASCRAPER_EXPORT_DUPLICATE", "record_count": 0})
                    continue
                now = datetime.now(UTC).isoformat()
                db.execute("INSERT INTO exports(batch_id,dataset,manifest_hash,manifest_json,imported_at) VALUES(?,?,?,?,?)", (batch_id, manifest["dataset"], manifest["manifest_hash"], json.dumps(manifest, sort_keys=True), now))
                count = 0
                for row in adapter.rows(validation):
                    payload_hash = stable_hash(row)
                    key = _economic_key(manifest["dataset"], row, payload_hash)
                    cursor = db.execute("INSERT OR IGNORE INTO records(batch_id,dataset,economic_key,payload_hash,payload_json,imported_at) VALUES(?,?,?,?,?,?)", (batch_id, manifest["dataset"], key, payload_hash, json.dumps(row, sort_keys=True, default=str), now))
                    count += max(0, int(cursor.rowcount))
                accepted_rows += count
                statuses.append({"batch_id": batch_id, "dataset": manifest["dataset"], "status": "DATASCRAPER_EXPORT_ACCEPTED", "record_count": count})
            db.commit()
        return {"status": "GO" if statuses and all(row["status"] in {"DATASCRAPER_EXPORT_ACCEPTED", "DATASCRAPER_EXPORT_DUPLICATE"} for row in statuses) else "PARTIAL", "batches": statuses, "accepted_record_count": accepted_rows}

    def records(self, dataset: str | None = None) -> list[dict[str, Any]]:
        return list(self.iter_records(dataset))

    def iter_records(self, dataset: str | None = None):
        self.initialize()
        query = "SELECT dataset,economic_key,payload_hash,payload_json,imported_at FROM records"
        params: tuple[Any, ...] = ()
        if dataset:
            query += " WHERE dataset=?"
            params = (dataset,)
        query += " ORDER BY record_id"
        with sqlite3.connect(self.path) as db:
            for row in db.execute(query, params):
                yield {"dataset": row[0], "economic_key": row[1], "payload_hash": row[2], "payload": json.loads(row[3]), "imported_at": row[4]}

    def counts(self) -> dict[str, int]:
        self.initialize()
        with sqlite3.connect(self.path) as db:
            return {row[0]: int(row[1]) for row in db.execute("SELECT dataset,COUNT(*) FROM records GROUP BY dataset")}

    def append_event(self, event_type: str, economic_key: str, payload: dict[str, Any]) -> None:
        self.append_events(event_type, [(economic_key, payload)])

    def append_events(self, event_type: str, events: list[tuple[str, dict[str, Any]]]) -> None:
        self.initialize()
        with sqlite3.connect(self.path) as db:
            now = datetime.now(UTC).isoformat()
            db.executemany(
                "INSERT OR IGNORE INTO research_events(event_type,economic_key,payload_hash,payload_json,created_at) VALUES(?,?,?,?,?)",
                [(event_type, key, stable_hash(payload), json.dumps(payload, sort_keys=True, default=str), now) for key, payload in events],
            )
            db.commit()


def _economic_key(dataset: str, row: dict[str, Any], fallback: str) -> str:
    symbol = str(row.get("symbol") or row.get("feed_id") or "UNKNOWN")
    if dataset == "filings" and row.get("record_type") == "COMPANYFACT":
        parts = (
            symbol, "COMPANYFACT", row.get("taxonomy"), row.get("concept"), row.get("unit"),
            row.get("period_start"), row.get("period_end"), row.get("filed_at"), row.get("form"),
            row.get("fiscal_year"), row.get("fiscal_period"), row.get("accession_hash"), row.get("value"),
        )
        return ":".join(str(part) for part in parts)
    timestamp = str(row.get("timestamp") or row.get("date") or row.get("published_at") or row.get("accepted_at") or row.get("entry_id_hash") or fallback)
    kind = str(row.get("action_type") or row.get("form") or row.get("record_type") or "ROW")
    return f"{symbol}:{timestamp}:{kind}"
