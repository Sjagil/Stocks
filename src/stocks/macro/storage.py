from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.macro.contracts import MacroObservation, stable_hash


@dataclass(frozen=True)
class MacroLayout:
    private_db: Path
    output_root: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> MacroLayout:
        return cls(
            private_db=project_root / "data" / "macro" / "private" / "macro.sqlite3",
            output_root=project_root / "output" / "macro",
        )


class MacroStore:
    def __init__(self, layout: MacroLayout):
        layout.private_db.parent.mkdir(parents=True, exist_ok=True)
        self.layout = layout
        self.connection = sqlite3.connect(layout.private_db)
        self.connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                series_id TEXT NOT NULL,
                observation_date TEXT NOT NULL,
                publication_at TEXT NOT NULL,
                available_at TEXT NOT NULL,
                vintage TEXT,
                revision_status TEXT NOT NULL,
                source TEXT NOT NULL,
                provider TEXT NOT NULL,
                value REAL NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_macro_observations_pit
                ON observations(series_id, available_at, observation_date);
            CREATE TABLE IF NOT EXISTS score_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                as_of TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS regimes (
                regime_id TEXT PRIMARY KEY,
                as_of TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                report_id TEXT PRIMARY KEY,
                period TEXT NOT NULL,
                as_of TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                event_observation_id TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                scheduled_at TEXT,
                available_at TEXT,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def append_observations(
        self,
        observations: list[MacroObservation],
        *,
        quarantine_conflicts: bool = False,
    ) -> dict[str, Any]:
        inserted = 0
        existing = 0
        conflicts: Counter[str] = Counter()
        with self.connection:
            for observation in observations:
                payload = observation.payload()
                payload_hash = stable_hash(payload)
                row = self.connection.execute(
                    "SELECT payload_hash FROM observations WHERE observation_id=?",
                    (observation.observation_id,),
                ).fetchone()
                if row is not None:
                    if row["payload_hash"] != payload_hash:
                        if not quarantine_conflicts:
                            raise ValueError(
                                "MACRO_OBSERVATION_IMMUTABILITY_CONFLICT"
                            )
                        conflicts[observation.series_id] += 1
                        continue
                    existing += 1
                    continue
                self.connection.execute(
                    """
                    INSERT INTO observations
                    (observation_id, series_id, observation_date, publication_at,
                     available_at, vintage, revision_status, source, provider,
                     value, payload_json, payload_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.observation_id,
                        observation.series_id,
                        observation.observation_date.isoformat(),
                        observation.publication_at.isoformat(),
                        observation.available_at.isoformat(),
                        observation.vintage,
                        observation.revision_status,
                        observation.source,
                        observation.provider,
                        observation.original_value,
                        _json(payload),
                        payload_hash,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                inserted += 1
        return {
            "inserted": inserted,
            "existing": existing,
            "conflict_count": sum(conflicts.values()),
            "quarantined_conflicts_by_series": dict(sorted(conflicts.items())),
        }

    def observations(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT payload_json FROM observations
            ORDER BY available_at, series_id, observation_date, observation_id
            """
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def append_snapshot(
        self,
        table: str,
        prefix: str,
        *,
        as_of: str,
        payload: dict[str, Any],
        period: str | None = None,
    ) -> str:
        if table not in {"score_snapshots", "regimes", "reports"}:
            raise ValueError("UNSUPPORTED_MACRO_SNAPSHOT_TABLE")
        digest = stable_hash(payload)
        identity = stable_hash(
            {"table": table, "as_of": as_of, "period": period, "payload": digest}
        )
        record_id = f"{prefix}-{identity[:24]}"
        id_column = {
            "score_snapshots": "snapshot_id",
            "regimes": "regime_id",
            "reports": "report_id",
        }[table]
        columns = (
            f"{id_column}, period, as_of, payload_json, payload_hash, created_at"
            if table == "reports"
            else f"{id_column}, as_of, payload_json, payload_hash, created_at"
        )
        placeholders = "?, ?, ?, ?, ?, ?" if table == "reports" else "?, ?, ?, ?, ?"
        values = (
            (
                record_id,
                period,
                as_of,
                _json(payload),
                digest,
                datetime.now(UTC).isoformat(),
            )
            if table == "reports"
            else (
                record_id,
                as_of,
                _json(payload),
                digest,
                datetime.now(UTC).isoformat(),
            )
        )
        with self.connection:
            self.connection.execute(
                f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})",
                values,
            )
        return record_id

    def latest_payload(self, table: str) -> dict[str, Any] | None:
        if table not in {"score_snapshots", "regimes", "reports"}:
            raise ValueError("UNSUPPORTED_MACRO_SNAPSHOT_TABLE")
        row = self.connection.execute(
            f"SELECT payload_json FROM {table} ORDER BY as_of DESC, created_at DESC LIMIT 1"
        ).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def append_event(self, payload: dict[str, Any]) -> str:
        digest = stable_hash(payload)
        event_observation_id = f"MACRO-EVENT-{digest[:24]}"
        with self.connection:
            row = self.connection.execute(
                "SELECT payload_hash FROM events WHERE event_observation_id=?",
                (event_observation_id,),
            ).fetchone()
            if row is not None and row["payload_hash"] != digest:
                raise ValueError("MACRO_EVENT_IMMUTABILITY_CONFLICT")
            self.connection.execute(
                """
                INSERT OR IGNORE INTO events
                (event_observation_id, event_id, scheduled_at, available_at,
                 payload_json, payload_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_observation_id,
                    str(payload["event_id"]),
                    payload.get("scheduled_at"),
                    payload.get("available_at"),
                    _json(payload),
                    digest,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return event_observation_id

    def regime_history(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT payload_json FROM regimes ORDER BY as_of, created_at"
        ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def counts(self) -> dict[str, int]:
        return {
            table: int(
                self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            )
            for table in (
                "observations",
                "score_snapshots",
                "regimes",
                "reports",
                "events",
            )
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> MacroStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
