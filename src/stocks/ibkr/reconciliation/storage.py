from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.reconciliation.masking import contains_raw_account
from stocks.ibkr.reconciliation.models import BrokerObservationSnapshot, model_to_jsonable


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_id TEXT PRIMARY KEY,
  snapshot_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS component_audits (
  snapshot_id TEXT NOT NULL,
  component_name TEXT NOT NULL,
  request_status TEXT NOT NULL,
  content_hash TEXT NOT NULL
);
"""


class Phase8Layout:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.output_dir = project_root / "output" / "ibkr" / "phase8"
        self.private_dir = project_root / "data" / "broker" / "phase8" / "private"
        self.private_db = self.private_dir / "broker_observation.sqlite3"

    @classmethod
    def from_project_root(cls, project_root: Path) -> "Phase8Layout":
        return cls(project_root)

    def artifact(self, name: str) -> Path:
        return self.output_dir / name


class BrokerObservationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def write_snapshot(self, snapshot: BrokerObservationSnapshot) -> str:
        payload = model_to_jsonable(snapshot)
        if contains_raw_account(payload):
            raise ValueError("RAW_ACCOUNT_LEAK_BLOCKED")
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        snapshot_hash = stable_hash(payload)
        self.initialize()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots(snapshot_id, snapshot_hash, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (snapshot.snapshot_id, snapshot_hash, payload_json, snapshot.snapshot_completed_at),
            )
            conn.execute("DELETE FROM component_audits WHERE snapshot_id = ?", (snapshot.snapshot_id,))
            for component in snapshot.component_audits:
                conn.execute(
                    "INSERT INTO component_audits(snapshot_id, component_name, request_status, content_hash) VALUES (?, ?, ?, ?)",
                    (snapshot.snapshot_id, component.name, component.request_status, component.content_hash),
                )
            conn.commit()
        return snapshot_hash

    def latest_snapshot(self) -> dict[str, Any] | None:
        self.initialize()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM snapshots ORDER BY created_at DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return dict(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if contains_raw_account(payload):
        raise ValueError("RAW_ACCOUNT_LEAK_BLOCKED")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def public_snapshot_summary(snapshot: BrokerObservationSnapshot, private_hash: str, private_db: Path) -> dict[str, Any]:
    account_fingerprints = sorted(
        {
            item.account_fingerprint
            for item in snapshot.account.values
            if not item.tag.startswith("$LEDGER-")
        }
    )
    return {
        "public_summary": {
            "snapshot_id": snapshot.snapshot_id,
            "account_fingerprint_count": len(account_fingerprints),
            "account_fingerprints": account_fingerprints,
            "accountsummary_tag_count": len({item.tag for item in snapshot.account.values}),
            "position_count": len(snapshot.positions.positions),
            "same_client_open_order_count": len(snapshot.same_client_open_orders.open_orders),
            "all_api_open_order_count": len(snapshot.all_api_open_orders.open_orders),
            "execution_count": len(snapshot.executions.executions),
            "commission_count": len(snapshot.executions.commissions),
            "execution_scope": snapshot.executions.execution_scope,
            "execution_history_complete": snapshot.executions.execution_history_complete,
            "snapshot_duration": str(snapshot.snapshot_span_seconds),
            "snapshot_atomic": snapshot.snapshot_atomic,
            "server_version": snapshot.server_version,
        },
        "private_snapshot_reference": str(private_db),
        "private_snapshot_hash": private_hash,
    }
