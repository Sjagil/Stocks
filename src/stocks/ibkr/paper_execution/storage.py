from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.state_machine import transition_metadata


COUNTERS = {
    "paper_place_order_calls": 0,
    "paper_cancel_order_calls": 0,
    "request_order_id_calls": 0,
    "same_client_open_order_requests": 0,
    "execution_requests": 0,
    "reconciliation_snapshot_requests": 0,
    "live_place_order_calls": 0,
    "global_cancel_calls": 0,
    "auto_bind_order_calls": 0,
    "exercise_option_calls": 0,
    "market_data_calls": 0,
    "historical_data_calls": 0,
    "strategy_generated_intents": 0,
    "automatic_submissions": 0,
    "automatic_cancellations": 0,
}


@dataclass(frozen=True)
class Phase9Layout:
    project_root: Path
    output_dir: Path
    private_dir: Path
    db_path: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "Phase9Layout":
        return cls(
            project_root=project_root,
            output_dir=project_root / "output" / "ibkr" / "phase9",
            private_dir=project_root / "data" / "execution" / "phase9" / "private",
            db_path=project_root / "data" / "execution" / "phase9" / "private" / "paper_execution.sqlite3",
        )

    def artifact(self, name: str) -> Path:
        return self.output_dir / name


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS intents (
  intent_id TEXT PRIMARY KEY,
  economic_order_key TEXT NOT NULL UNIQUE,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
  approval_id TEXT PRIMARY KEY,
  intent_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS order_ids (
  order_id INTEGER PRIMARY KEY,
  intent_id TEXT,
  allocated_at TEXT NOT NULL,
  used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  aggregate_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS executions (
  exec_identity TEXT PRIMARY KEY,
  intent_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS commissions (
  commission_identity TEXT PRIMARY KEY,
  exec_identity TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capital_reservation_events (
  reservation_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  intent_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  amount_eur TEXT NOT NULL,
  con_id INTEGER NOT NULL,
  reason TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


class PaperExecutionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> dict[str, Any]:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        return {"status": "GO", "database_path": str(self.path), **self.counts()}

    def register_intent(self, payload: dict[str, Any]) -> str:
        payload_hash = stable_hash(payload)
        with self.connect() as conn:
            existing = conn.execute("SELECT payload_hash FROM intents WHERE economic_order_key = ?", (payload["economic_order_key"],)).fetchone()
            if existing:
                return "INTENT_IDEMPOTENT" if existing["payload_hash"] == payload_hash else "DUPLICATE_INTENT"
            conn.execute(
                "INSERT INTO intents(intent_id, economic_order_key, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (payload["intent_id"], payload["economic_order_key"], payload_hash, _json(payload), payload["created_at"]),
            )
            conn.commit()
        self.append_event(payload["intent_id"], "MANUAL_OPERATOR_INTENT", {"payload_hash": payload_hash})
        return "INTENT_REGISTERED"

    def economic_order_key_exists(self, economic_order_key: str) -> bool:
        return self.economic_order_key_owner(economic_order_key) is not None

    def economic_order_key_owner(self, economic_order_key: str) -> str | None:
        if not self.path.exists():
            return None
        with self.connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'intents'",
            ).fetchone()
            if table is None:
                return None
            row = conn.execute("SELECT intent_id FROM intents WHERE economic_order_key = ? LIMIT 1", (economic_order_key,)).fetchone()
        return None if row is None else str(row["intent_id"])

    def get_intent(self, intent_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT payload_json FROM intents WHERE intent_id = ?", (intent_id,)).fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def list_intents(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'intents'",
            ).fetchone()
            if table is None:
                return []
            rows = conn.execute(
                "SELECT payload_json FROM intents ORDER BY created_at, intent_id",
            ).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]

    def append_approval(self, payload: dict[str, Any]) -> str:
        payload_hash = stable_hash(payload)
        with self.connect() as conn:
            existing = conn.execute("SELECT payload_hash, used FROM approvals WHERE approval_id = ?", (payload["approval_id"],)).fetchone()
            if existing:
                return "APPROVAL_ALREADY_USED" if existing["used"] else "APPROVAL_IDEMPOTENT"
            conn.execute(
                "INSERT INTO approvals(approval_id, intent_id, payload_hash, payload_json, created_at, used) VALUES (?, ?, ?, ?, ?, 0)",
                (payload["approval_id"], payload["intent_id"], payload_hash, _json(payload), now_iso()),
            )
            conn.commit()
        self.append_event(payload["intent_id"], "APPROVAL_RECORDED", {"approval_id_hash": stable_hash(payload["approval_id"])})
        return "APPROVAL_RECORDED"

    def find_unconsumed_approval(self, intent_id: str, approval_type: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT approval_id, payload_json
                FROM approvals
                WHERE intent_id = ? AND used = 0
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (intent_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        if payload.get("approval_type") != approval_type:
            return None
        payload["approval_id"] = row["approval_id"]
        return payload

    def consume_approval(self, approval_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT used FROM approvals WHERE approval_id = ?", (approval_id,)).fetchone()
            if row is None:
                return "APPROVAL_REQUIRED"
            if row["used"]:
                return "APPROVAL_ALREADY_USED"
            conn.execute("UPDATE approvals SET used = 1 WHERE approval_id = ?", (approval_id,))
            conn.commit()
        return "APPROVED_FOR_SINGLE_SUBMISSION"

    def allocate_order_id(self, broker_next_id: int, intent_id: str) -> tuple[str, int | None]:
        with self.connect() as conn:
            max_local = conn.execute("SELECT COALESCE(MAX(order_id), 0) FROM order_ids").fetchone()[0]
            if broker_next_id <= max_local:
                return "ORDER_ID_REGRESSION_BLOCKED", None
            conn.execute(
                "INSERT INTO order_ids(order_id, intent_id, allocated_at, used) VALUES (?, ?, ?, 0)",
                (broker_next_id, intent_id, now_iso()),
            )
            conn.commit()
        self.append_event(intent_id, "ORDER_ID_ALLOCATED", {"order_id_hash": stable_hash(str(broker_next_id))})
        return "ORDER_ID_READY", broker_next_id

    def max_order_id(self) -> int:
        with self.connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'order_ids'",
            ).fetchone()
            if table is None:
                return 0
            return int(conn.execute("SELECT COALESCE(MAX(order_id), 0) FROM order_ids").fetchone()[0])

    def mark_order_id_used(self, order_id: int) -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT used FROM order_ids WHERE order_id = ?", (order_id,)).fetchone()
            if row is None:
                return "ORDER_ID_STATE_CONFLICT"
            if row["used"]:
                return "ORDER_ID_ALREADY_USED"
            conn.execute("UPDATE order_ids SET used = 1 WHERE order_id = ?", (order_id,))
            conn.commit()
        return "ORDER_ID_READY"

    def latest_order_id_for_intent(self, intent_id: str) -> int | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT order_id
                FROM order_ids
                WHERE intent_id = ?
                ORDER BY allocated_at DESC
                LIMIT 1
                """,
                (intent_id,),
            ).fetchone()
        return None if row is None else int(row["order_id"])

    def append_event(self, aggregate_id: str, event_type: str, payload: dict[str, Any]) -> None:
        created_at = now_iso()
        enriched = dict(payload)
        enriched.setdefault(
            "transition_meta",
            transition_metadata(
                aggregate_id,
                event_type,
                enriched,
                created_at,
            ),
        )
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO events(aggregate_id, event_type, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    aggregate_id,
                    event_type,
                    stable_hash(enriched),
                    _json(enriched),
                    created_at,
                ),
            )
            conn.commit()

    def event_type_count(self, event_type: str) -> int:
        with self.connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'events'",
            ).fetchone()
            if table is None:
                return 0
            return int(conn.execute("SELECT COUNT(*) FROM events WHERE event_type = ?", (event_type,)).fetchone()[0])

    def submitted_order_count_for_session(self, *, side: str, session_date: str) -> int:
        """Count broker submissions by intent side and intended trading session."""
        with self.connect() as conn:
            tables = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('events', 'intents')"
                ).fetchall()
            }
            if tables != {"events", "intents"}:
                return 0
            row = conn.execute(
                """
                SELECT COUNT(*) AS submitted_count
                FROM events AS event
                JOIN intents AS intent ON intent.intent_id = event.aggregate_id
                WHERE event.event_type = 'PLACE_ORDER_CALLED_ONCE'
                  AND UPPER(json_extract(intent.payload_json, '$.side')) = ?
                  AND json_extract(intent.payload_json, '$.session_date') = ?
                  AND NOT EXISTS (
                    SELECT 1
                    FROM events AS failed
                    WHERE failed.aggregate_id = event.aggregate_id
                      AND failed.event_type IN (
                        'BROKER_SUBMISSION_ACK_INVALIDATED',
                        'BROKER_SUBMISSION_ACK_TIMEOUT',
                        'BROKER_SUBMISSION_REJECTED'
                      )
                      AND NOT EXISTS (
                        SELECT 1
                        FROM events AS correction
                        WHERE correction.aggregate_id = failed.aggregate_id
                          AND correction.event_type =
                            'BROKER_SUBMISSION_WARNING_RECLASSIFIED'
                          AND correction.event_id > failed.event_id
                      )
                  )
                """,
                (side.upper(), session_date),
            ).fetchone()
        return int(row["submitted_count"])

    def active_local_order_count(self) -> int:
        return len(self.active_local_order_intents())

    def active_local_order_intents(self) -> list[dict[str, Any]]:
        """Return private intents until the broker proves a terminal state."""
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT intent.payload_json
                FROM intents AS intent
                WHERE EXISTS (
                    SELECT 1 FROM events
                    WHERE aggregate_id = intent.intent_id
                      AND event_type = 'PLACE_ORDER_CALLED_ONCE'
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM events
                    WHERE aggregate_id = intent.intent_id
                      AND event_type IN (
                        'BROKER_ORDER_CANCELLED',
                        'BROKER_SUBMISSION_REJECTED',
                        'BROKER_SUBMISSION_ACK_INVALIDATED'
                      )
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM events AS failed
                    WHERE failed.aggregate_id = intent.intent_id
                      AND failed.event_type IN (
                        'BROKER_SUBMISSION_ACK_INVALIDATED',
                        'BROKER_SUBMISSION_ACK_TIMEOUT',
                        'BROKER_SUBMISSION_REJECTED'
                      )
                      AND NOT EXISTS (
                        SELECT 1
                        FROM events AS correction
                        WHERE correction.aggregate_id = failed.aggregate_id
                          AND correction.event_type =
                            'BROKER_SUBMISSION_WARNING_RECLASSIFIED'
                          AND correction.event_id > failed.event_id
                      )
                  )
                """
            ).fetchall()
        intents = [
            json.loads(str(row["payload_json"])) for row in rows
        ]
        filled_by_intent: dict[str, Decimal] = {}
        for row in self.list_executions():
            payload = row["payload"]
            intent_id = str(payload.get("intent_id", ""))
            if intent_id:
                filled_by_intent[intent_id] = (
                    filled_by_intent.get(intent_id, Decimal("0"))
                    + Decimal(str(payload.get("quantity", "0")))
                )
        return [
            intent
            for intent in intents
            if filled_by_intent.get(
                str(intent["intent_id"]), Decimal("0")
            )
            < Decimal(str(intent.get("quantity", "Infinity")))
        ]

    def reserve_capital_once(
        self, *, intent_id: str, amount_eur: Decimal, con_id: int
    ) -> str:
        if amount_eur <= 0:
            return "CAPITAL_RESERVATION_INVALID"
        current = self.capital_reservation_state(intent_id)
        if current is not None:
            if (
                current["status"] in {
                    "RESERVED",
                    "PARTIAL_FILL_RESERVED",
                    "DEPLOYED",
                }
                and Decimal(str(current["amount_eur"])) == amount_eur
                and int(current["con_id"]) == con_id
            ):
                return "CAPITAL_RESERVATION_IDEMPOTENT"
            return "CAPITAL_RESERVATION_CONFLICT_BLOCKED"
        self._append_capital_event(
            intent_id=intent_id,
            event_type="RESERVED",
            amount_eur=amount_eur,
            con_id=con_id,
            reason="PRE_TRADE_RISK_APPROVED",
        )
        self.append_event(
            intent_id,
            "CAPITAL_RESERVED",
            {"amount_eur": str(amount_eur), "con_id": con_id},
        )
        return "CAPITAL_RESERVED"

    def mark_capital_deployed(self, intent_id: str) -> str:
        current = self.capital_reservation_state(intent_id)
        if current is None:
            return "CAPITAL_RESERVATION_MISSING_BLOCKED"
        if current["status"] == "DEPLOYED":
            return "CAPITAL_DEPLOYMENT_IDEMPOTENT"
        if current["status"] not in {"RESERVED", "PARTIAL_FILL_RESERVED"}:
            return "CAPITAL_DEPLOYMENT_STATE_BLOCKED"
        original = self._capital_reservation_original_amount(intent_id)
        self._append_capital_event(
            intent_id=intent_id,
            event_type="DEPLOYED",
            amount_eur=original,
            con_id=int(current["con_id"]),
            reason="CANONICAL_BUY_FILL",
        )
        self.append_event(
            intent_id,
            "CAPITAL_DEPLOYED",
            {"con_id": int(current["con_id"])},
        )
        return "CAPITAL_DEPLOYED"

    def apply_fill_to_capital_reservation(
        self,
        intent_id: str,
        *,
        filled_notional_eur: Decimal,
        order_complete: bool,
    ) -> str:
        current = self.capital_reservation_state(intent_id)
        if current is None:
            return "CAPITAL_RESERVATION_MISSING_BLOCKED"
        if current["status"] == "DEPLOYED":
            return "CAPITAL_DEPLOYMENT_IDEMPOTENT"
        if current["status"] not in {"RESERVED", "PARTIAL_FILL_RESERVED"}:
            return "CAPITAL_DEPLOYMENT_STATE_BLOCKED"
        if filled_notional_eur <= 0:
            return "CAPITAL_FILL_NOTIONAL_INVALID"
        remaining = max(
            Decimal("0"),
            Decimal(str(current["amount_eur"])) - filled_notional_eur,
        )
        if order_complete:
            return self.mark_capital_deployed(intent_id)
        self._append_capital_event(
            intent_id=intent_id,
            event_type="PARTIAL_FILL_RESERVED",
            amount_eur=remaining,
            con_id=int(current["con_id"]),
            reason="PARTIAL_FILL_REMAINDER",
        )
        self.append_event(
            intent_id,
            "CAPITAL_PARTIALLY_DEPLOYED",
            {
                "remaining_reserved_eur": str(remaining),
                "filled_notional_eur": str(filled_notional_eur),
            },
        )
        return "CAPITAL_PARTIAL_FILL_RETAINED"

    def release_capital_once(
        self,
        intent_id: str,
        *,
        reason: str,
        allow_deployed: bool = True,
    ) -> str:
        current = self.capital_reservation_state(intent_id)
        if current is None:
            return "CAPITAL_RESERVATION_NOT_FOUND"
        if current["status"] == "RELEASED":
            return "CAPITAL_RELEASE_IDEMPOTENT"
        if current["status"] == "DEPLOYED" and not allow_deployed:
            return "CAPITAL_ALREADY_DEPLOYED"
        self._append_capital_event(
            intent_id=intent_id,
            event_type="RELEASED",
            amount_eur=Decimal(str(current["amount_eur"])),
            con_id=int(current["con_id"]),
            reason=reason,
        )
        self.append_event(
            intent_id,
            "CAPITAL_RELEASED",
            {"con_id": int(current["con_id"]), "reason": reason},
        )
        return "CAPITAL_RELEASED"

    def release_capital_for_con_id(self, con_id: int, *, reason: str) -> int:
        released = 0
        for intent in self.list_intents():
            if str(intent.get("side", "")).upper() != "BUY":
                continue
            if int(intent.get("con_id", -1)) != con_id:
                continue
            if self.release_capital_once(
                str(intent["intent_id"]), reason=reason
            ) == "CAPITAL_RELEASED":
                released += 1
        return released

    def capital_reservation_state(
        self, intent_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'capital_reservation_events'"
            ).fetchone()
            if table is None:
                return None
            row = conn.execute(
                "SELECT event_type, amount_eur, con_id, reason, created_at "
                "FROM capital_reservation_events WHERE intent_id = ? "
                "ORDER BY reservation_event_id DESC LIMIT 1",
                (intent_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "intent_id": intent_id,
            "status": str(row["event_type"]),
            "amount_eur": str(row["amount_eur"]),
            "con_id": int(row["con_id"]),
            "reason": str(row["reason"]),
            "created_at": str(row["created_at"]),
        }

    def capital_summary(self) -> dict[str, Any]:
        states = [
            self.capital_reservation_breakdown(str(intent["intent_id"]))
            for intent in self.list_intents()
            if self.capital_reservation_state(str(intent["intent_id"]))
            is not None
        ]
        reserved = sum(
            (
                Decimal(str(state["reserved_eur"]))
                for state in states
                if Decimal(str(state["reserved_eur"])) > 0
            ),
            Decimal("0"),
        )
        deployed = sum(
            (
                Decimal(str(state["deployed_eur"]))
                for state in states
                if Decimal(str(state["deployed_eur"])) > 0
            ),
            Decimal("0"),
        )
        return {
            "active_reservation_count": sum(
                Decimal(str(state["reserved_eur"])) > 0 for state in states
            ),
            "deployed_reservation_count": sum(
                Decimal(str(state["deployed_eur"])) > 0 for state in states
            ),
            "reserved_capital_eur": str(reserved),
            "deployed_capital_eur": str(deployed),
        }

    def capital_reservation_breakdown(
        self, intent_id: str
    ) -> dict[str, Any]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT event_type, amount_eur FROM capital_reservation_events "
                "WHERE intent_id = ? ORDER BY reservation_event_id",
                (intent_id,),
            ).fetchall()
        if not rows:
            return {
                "status": "NONE",
                "reserved_eur": "0",
                "deployed_eur": "0",
            }
        original = Decimal(str(rows[0]["amount_eur"]))
        latest_status = str(rows[-1]["event_type"])
        last_remaining = original
        for row in rows:
            if str(row["event_type"]) in {
                "RESERVED",
                "PARTIAL_FILL_RESERVED",
            }:
                last_remaining = Decimal(str(row["amount_eur"]))
        if latest_status in {"RESERVED", "PARTIAL_FILL_RESERVED"}:
            reserved = last_remaining
            deployed = original - last_remaining
        elif latest_status == "DEPLOYED":
            reserved = Decimal("0")
            deployed = original
        else:
            reserved = Decimal("0")
            deployed = max(Decimal("0"), original - last_remaining)
        return {
            "status": latest_status,
            "reserved_eur": str(reserved),
            "deployed_eur": str(deployed),
        }

    def _capital_reservation_original_amount(
        self, intent_id: str
    ) -> Decimal:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT amount_eur FROM capital_reservation_events "
                "WHERE intent_id = ? ORDER BY reservation_event_id LIMIT 1",
                (intent_id,),
            ).fetchone()
        return Decimal("0") if row is None else Decimal(str(row["amount_eur"]))

    def _append_capital_event(
        self,
        *,
        intent_id: str,
        event_type: str,
        amount_eur: Decimal,
        con_id: int,
        reason: str,
    ) -> None:
        payload = {
            "intent_id": intent_id,
            "event_type": event_type,
            "amount_eur": str(amount_eur),
            "con_id": con_id,
            "reason": reason,
        }
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO capital_reservation_events"
                "(intent_id, event_type, amount_eur, con_id, reason, "
                "payload_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    intent_id,
                    event_type,
                    str(amount_eur),
                    con_id,
                    reason,
                    stable_hash(payload),
                    now_iso(),
                ),
            )
            conn.commit()

    def append_execution_once(self, exec_identity: str, intent_id: str, payload: dict[str, Any]) -> str:
        return self._append_unique("executions", "exec_identity", exec_identity, intent_id, payload, "EXECUTION_RECORDED", "DUPLICATE_EXECUTION_IGNORED")

    def append_commission_once(self, commission_identity: str, exec_identity: str, payload: dict[str, Any]) -> str:
        data = {**payload, "exec_identity": exec_identity}
        return self._append_unique("commissions", "commission_identity", commission_identity, exec_identity, data, "COMMISSION_RECORDED", "DUPLICATE_COMMISSION_IGNORED")

    def _append_unique(self, table: str, key_name: str, key: str, aggregate_id: str, payload: dict[str, Any], recorded: str, duplicate: str) -> str:
        payload_hash = stable_hash(payload)
        with self.connect() as conn:
            row = conn.execute(f"SELECT payload_hash FROM {table} WHERE {key_name} = ?", (key,)).fetchone()
            if row:
                if row["payload_hash"] == payload_hash:
                    return duplicate
                return "EXECUTION_CONFLICT_BLOCKED" if table == "executions" else "COMMISSION_CONFLICT_BLOCKED"
            if table == "commissions":
                conn.execute(
                    "INSERT INTO commissions(commission_identity, exec_identity, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (key, aggregate_id, payload_hash, _json(payload), now_iso()),
                )
            else:
                conn.execute(
                    "INSERT INTO executions(exec_identity, intent_id, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (key, aggregate_id, payload_hash, _json(payload), now_iso()),
                )
            conn.commit()
        self.append_event(aggregate_id, recorded, {"identity_hash": stable_hash(key)})
        return recorded

    def list_executions(self) -> list[dict[str, Any]]:
        return self._list_payload_table("executions", "exec_identity")

    def list_commissions(self) -> list[dict[str, Any]]:
        return self._list_payload_table("commissions", "commission_identity")

    def list_events(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'events'",
            ).fetchone()
            if table is None:
                return []
            rows = conn.execute(
                "SELECT event_id, aggregate_id, event_type, payload_hash, payload_json, created_at FROM events ORDER BY event_id ASC",
            ).fetchall()
        return [
            {
                "event_id": int(row["event_id"]),
                "aggregate_id": row["aggregate_id"],
                "event_type": row["event_type"],
                "payload_hash": row["payload_hash"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _list_payload_table(self, table: str, key_name: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            if exists is None:
                return []
            rows = conn.execute(
                f"SELECT {key_name}, payload_hash, payload_json, created_at FROM {table} ORDER BY created_at ASC",
            ).fetchall()
        return [
            {
                key_name: row[key_name],
                "payload_hash": row["payload_hash"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            return {
                "intent_count": int(conn.execute("SELECT COUNT(*) FROM intents").fetchone()[0]),
                "approval_count": int(conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0]),
                "order_id_count": int(conn.execute("SELECT COUNT(*) FROM order_ids").fetchone()[0]),
                "event_count": int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
                "execution_count": int(conn.execute("SELECT COUNT(*) FROM executions").fetchone()[0]),
                "commission_count": int(conn.execute("SELECT COUNT(*) FROM commissions").fetchone()[0]),
                "capital_reservation_event_count": int(
                    conn.execute(
                        "SELECT COUNT(*) FROM capital_reservation_events"
                    ).fetchone()[0]
                ),
            }


def artifact(schema: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = {"schema": schema, "generated_at": now_iso(), **COUNTERS, **payload}
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
