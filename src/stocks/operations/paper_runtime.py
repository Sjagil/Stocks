from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.storage import (
    PaperExecutionStore,
    Phase9Layout,
)


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS entry_plans (
  plan_id TEXT PRIMARY KEY,
  economic_key TEXT NOT NULL UNIQUE,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exit_plans (
  plan_id TEXT PRIMARY KEY,
  economic_key TEXT NOT NULL UNIQUE,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lifecycle_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  aggregate_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


class PaperRuntimeStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            connection.commit()

    def append_plan(
        self, table: str, plan: dict[str, Any]
    ) -> str:
        if table not in {"entry_plans", "exit_plans"}:
            raise ValueError(f"Unknown paper runtime plan table: {table}")
        payload_hash = stable_hash(plan)
        with self.connect() as connection:
            existing = connection.execute(
                f"SELECT payload_hash FROM {table} WHERE economic_key = ?",
                (plan["economic_key"],),
            ).fetchone()
            if existing is not None:
                return (
                    "IDEMPOTENT_REPLAY"
                    if existing["payload_hash"] == payload_hash
                    else "ECONOMIC_KEY_CONFLICT"
                )
            connection.execute(
                f"""
                INSERT INTO {table}
                  (plan_id,economic_key,payload_hash,payload_json,created_at)
                VALUES (?,?,?,?,?)
                """,
                (
                    plan["plan_id"],
                    plan["economic_key"],
                    payload_hash,
                    json.dumps(plan, sort_keys=True, default=str),
                    _now(),
                ),
            )
            connection.commit()
        self.append_event(
            plan["plan_id"],
            "PLAN_RECORDED",
            {
                "plan_type": table,
                "proposal_status": plan["proposal_status"],
            },
        )
        return "RECORDED"

    def append_event(
        self, aggregate_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO lifecycle_events
                  (aggregate_id,event_type,payload_hash,payload_json,created_at)
                VALUES (?,?,?,?,?)
                """,
                (
                    aggregate_id,
                    event_type,
                    stable_hash(payload),
                    json.dumps(payload, sort_keys=True, default=str),
                    _now(),
                ),
            )
            connection.commit()

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            return {
                table: int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                )
                for table in (
                    "entry_plans",
                    "exit_plans",
                    "lifecycle_events",
                )
            }

    def plans(self, table: str) -> list[dict[str, Any]]:
        if table not in {"entry_plans", "exit_plans"}:
            raise ValueError(f"Unknown paper runtime plan table: {table}")
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT payload_json FROM {table} ORDER BY rowid"
            ).fetchall()
        return [json.loads(str(row["payload_json"])) for row in rows]


def plan_paper_cycle(
    project_root: Path,
    *,
    dynamic: dict[str, Any],
    lifecycle: dict[str, Any],
    execution_authority: str,
    position_quantities: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    store = PaperRuntimeStore(database_path(project_root))
    store.initialize()
    signals = _signal_map(dynamic)
    candidates = _candidate_map(dynamic)
    positions = (
        dict(position_quantities)
        if position_quantities is not None
        else phase9_position_quantities(project_root)
    )
    rows: list[dict[str, Any]] = []
    for transition in lifecycle.get("rows", []):
        lifecycle_status = transition.get("lifecycle_status")
        if lifecycle_status not in {"FRESH_ENTRY", "EXIT"}:
            continue
        key = (
            str(transition.get("strategy_id", "")),
            str(transition.get("ticker", "")).upper(),
        )
        signal = signals.get(key)
        candidate = candidates.get(key[1])
        if signal is None:
            rows.append(_blocked_row(key, "SIGNAL_PAYLOAD_MISSING"))
            continue
        con_id = int(
            signal.get("contract_identity", {}).get("con_id") or 0
        )
        if lifecycle_status == "EXIT":
            quantity = int(positions.get(con_id, 0))
            if con_id <= 0:
                rows.append(
                    _blocked_row(key, "CONTRACT_IDENTITY_REQUIRED")
                )
                continue
            if quantity <= 0:
                rows.append(
                    _blocked_row(key, "NO_RECONCILED_LONG_POSITION")
                )
                continue
            plan = _exit_plan(
                signal=signal,
                quantity=quantity,
                execution_authority=execution_authority,
            )
            write_status = store.append_plan("exit_plans", plan)
            rows.append({**plan, "write_status": write_status})
            continue
        if signal.get("risk_blockers"):
            rows.append(_blocked_row(key, "SIGNAL_RISK_BLOCKED"))
            continue
        if not signal.get("contract_identity", {}).get("con_id"):
            rows.append(_blocked_row(key, "CONTRACT_IDENTITY_REQUIRED"))
            continue
        if candidate is None:
            rows.append(_blocked_row(key, "PORTFOLIO_CANDIDATE_MISSING"))
            continue
        quantity = int(candidate.get("target_quantity", 0))
        proposal_status = str(
            candidate.get("proposal_status", "VALID_SIGNAL_BLOCKED_BY_RISK")
        )
        plan = _entry_plan(
            signal=signal,
            candidate=candidate,
            quantity=quantity,
            proposal_status=proposal_status,
            execution_authority=execution_authority,
        )
        write_status = store.append_plan("entry_plans", plan)
        rows.append({**plan, "write_status": write_status})
    report = {
        "schema": "bounded_automatic_paper_plan_cycle_v1",
        "status": "GO",
        "generated_at": _now(),
        "execution_authority": execution_authority,
        "automatic_submission_allowed": (
            execution_authority == "AUTOMATIC_BOUNDED_PAPER"
        ),
        "fresh_entry_count": sum(
            row.get("lifecycle_status") == "FRESH_ENTRY"
            for row in lifecycle.get("rows", [])
        ),
        "exit_signal_count": sum(
            row.get("lifecycle_status") == "EXIT"
            for row in lifecycle.get("rows", [])
        ),
        "plan_count": len(rows),
        "executable_plan_count": sum(
            row.get("proposal_status") == "VALID_SIGNAL_EXECUTABLE"
            and int(row.get("target_quantity", 0)) > 0
            for row in rows
        ),
        "below_whole_share_budget_count": sum(
            row.get("proposal_status")
            == "VALID_SIGNAL_BELOW_WHOLE_SHARE_BUDGET"
            for row in rows
        ),
        "plans": rows,
        "store_counts": store.counts(),
        "database": str(database_path(project_root)),
        "paper_place_order_calls": 0,
        "paper_cancel_order_calls": 0,
    }
    _publish(project_root, "paper-plan-cycle.json", report)
    return report


def paper_runtime_status(project_root: Path) -> dict[str, Any]:
    store = PaperRuntimeStore(database_path(project_root))
    store.initialize()
    report = {
        "schema": "bounded_automatic_paper_runtime_status_v1",
        "status": "GO",
        "store_counts": store.counts(),
        "entry_plans": store.plans("entry_plans"),
        "exit_plans": store.plans("exit_plans"),
        "database": str(database_path(project_root)),
        "broker_submission_implemented": True,
        "broker_submission_module": (
            "stocks.operations.paper_writer"
        ),
        "execution_authority": "NONE",
        "paper_place_order_calls": 0,
        "paper_cancel_order_calls": 0,
    }
    _publish(project_root, "paper-runtime-status.json", report)
    return report


def database_path(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "operations"
        / "private"
        / "paper-runtime.sqlite3"
    )


def _entry_plan(
    *,
    signal: dict[str, Any],
    candidate: dict[str, Any],
    quantity: int,
    proposal_status: str,
    execution_authority: str,
) -> dict[str, Any]:
    contract = signal["contract_identity"]
    economic_basis = {
        "signal_id": signal["signal_id"],
        "con_id": int(contract["con_id"]),
        "side": "BUY",
        "quantity": quantity,
        "limit_price": str(signal["limit_entry_price"]),
    }
    economic_key = stable_hash(economic_basis)
    executable = (
        proposal_status == "VALID_SIGNAL_EXECUTABLE" and quantity > 0
    )
    return {
        "plan_id": f"AUTO-PAPER-{economic_key[:24]}",
        "economic_key": economic_key,
        "signal_id": signal["signal_id"],
        "strategy_id": signal["strategy_id"],
        "ticker": signal["ticker"],
        "con_id": int(contract["con_id"]),
        "security_type": str(
            contract.get(
                "security_type",
                signal.get("asset_class", "STK"),
            )
        ),
        "currency": str(signal.get("currency", "EUR")),
        "exchange": str(signal.get("exchange", "SMART")),
        "signal_timestamp": str(signal.get("signal_timestamp", "")),
        "data_timestamp": str(signal.get("data_timestamp", "")),
        "expiration_timestamp": str(
            signal.get("expiration_timestamp", "")
        ),
        "data_freshness": str(signal.get("data_freshness", "")),
        "side": "BUY",
        "target_quantity": quantity,
        "order_type": "LIMIT",
        "limit_price": str(signal["limit_entry_price"]),
        "outside_rth": False,
        "time_in_force": "DAY",
        "stop_loss": str(signal["stop_loss"]),
        "take_profit_1": str(signal["take_profit_1"]),
        "take_profit_2": str(signal["take_profit_2"]),
        "proposal_status": proposal_status,
        "required_cash_eur": candidate.get("required_cash_eur"),
        "required_risk_eur": candidate.get("required_risk_eur"),
        "available_risk_eur": candidate.get("available_risk_eur"),
        "sizing_mode": candidate.get("sizing_mode"),
        "broker_submission_status": (
            "READY_AFTER_AUTHORITY"
            if executable
            else "NOT_EXECUTABLE"
        ),
        "execution_authority": (
            execution_authority if executable else "NONE"
        ),
    }


def _exit_plan(
    *,
    signal: dict[str, Any],
    quantity: int,
    execution_authority: str,
) -> dict[str, Any]:
    contract = signal["contract_identity"]
    economic_basis = {
        "signal_id": signal["signal_id"],
        "con_id": int(contract["con_id"]),
        "side": "SELL",
        "quantity": quantity,
        "limit_price": str(signal["limit_entry_price"]),
    }
    economic_key = stable_hash(economic_basis)
    return {
        "plan_id": f"AUTO-PAPER-{economic_key[:24]}",
        "economic_key": economic_key,
        "signal_id": signal["signal_id"],
        "strategy_id": signal["strategy_id"],
        "ticker": signal["ticker"],
        "con_id": int(contract["con_id"]),
        "security_type": str(
            contract.get(
                "security_type",
                signal.get("asset_class", "STK"),
            )
        ),
        "currency": str(signal.get("currency", "EUR")),
        "exchange": str(signal.get("exchange", "SMART")),
        "signal_timestamp": str(signal.get("signal_timestamp", "")),
        "data_timestamp": str(signal.get("data_timestamp", "")),
        "expiration_timestamp": str(
            signal.get("expiration_timestamp", "")
        ),
        "data_freshness": str(signal.get("data_freshness", "")),
        "side": "SELL",
        "target_quantity": quantity,
        "order_type": "LIMIT",
        "limit_price": str(signal["limit_entry_price"]),
        "outside_rth": False,
        "time_in_force": "DAY",
        "proposal_status": "VALID_RISK_REDUCING_EXIT",
        "broker_submission_status": "READY_AFTER_AUTHORITY",
        "execution_authority": execution_authority,
    }


def phase9_position_quantities(project_root: Path) -> dict[int, int]:
    store = PaperExecutionStore(
        Phase9Layout.from_project_root(project_root).db_path
    )
    store.initialize()
    positions: dict[int, Decimal] = {}
    for row in store.list_executions():
        payload = row["payload"]
        con_id = int(payload.get("con_id", 0))
        if con_id <= 0:
            continue
        quantity = Decimal(str(payload.get("quantity", "0")))
        side = str(payload.get("side", "")).upper()
        signed = quantity if side == "BUY" else -quantity
        positions[con_id] = positions.get(con_id, Decimal("0")) + signed
    return {
        con_id: int(quantity)
        for con_id, quantity in positions.items()
        if quantity > 0 and quantity == quantity.to_integral_value()
    }


def _blocked_row(
    key: tuple[str, str], reason: str
) -> dict[str, Any]:
    return {
        "strategy_id": key[0],
        "ticker": key[1],
        "target_quantity": 0,
        "proposal_status": "VALID_SIGNAL_BLOCKED_BY_RISK",
        "blocker": reason,
        "execution_authority": "NONE",
    }


def _signal_map(
    dynamic: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    payload = dynamic.get("signals", {})
    rows = payload.get("signals", []) if isinstance(payload, dict) else []
    return {
        (
            str(row.get("strategy_id", "")),
            str(row.get("ticker", "")).upper(),
        ): row
        for row in rows
    }


def _candidate_map(dynamic: dict[str, Any]) -> dict[str, dict[str, Any]]:
    portfolio = dynamic.get("portfolio", {})
    rows = portfolio.get("candidates", []) if isinstance(portfolio, dict) else []
    return {str(row.get("ticker", "")).upper(): row for row in rows}


def _publish(
    project_root: Path, name: str, payload: dict[str, Any]
) -> None:
    path = project_root / "output" / "operations" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    temp.replace(path)


def _now() -> str:
    return datetime.now(UTC).isoformat()
