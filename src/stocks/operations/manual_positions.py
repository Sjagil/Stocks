from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


SNAPSHOT_MAX_AGE_SECONDS = 300


def audit_manual_position_broker_match(
    project_root: Path,
    *,
    position: dict[str, Any],
    environment: str,
    now: datetime | None = None,
    maximum_age_seconds: int = SNAPSHOT_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    normalized_environment = environment.lower()
    if normalized_environment not in {"paper", "live"}:
        return _blocked("UNKNOWN_BROKER_ENVIRONMENT")
    if str(position.get("environment", "")).lower() != normalized_environment:
        return _blocked("MANUAL_POSITION_ENVIRONMENT_MISMATCH")
    con_id = position.get("con_id")
    if con_id is None:
        return _blocked("MISSING_CONTRACT_IDENTITY")
    snapshot = _latest_snapshot(
        _snapshot_path(project_root, normalized_environment)
    )
    if snapshot is None:
        return _blocked("BROKER_SNAPSHOT_UNAVAILABLE")
    if snapshot.get("broker_observation_authority") != "READ_ONLY":
        return _blocked("BROKER_SNAPSHOT_AUTHORITY_INVALID")
    if snapshot.get("positions", {}).get("status") != "COMPLETE":
        return _blocked("BROKER_POSITION_SNAPSHOT_INCOMPLETE")
    completed_at = _parse_datetime(snapshot.get("snapshot_completed_at"))
    effective_now = now or datetime.now(UTC)
    if completed_at is None:
        return _blocked("BROKER_SNAPSHOT_TIMESTAMP_INVALID")
    age_seconds = (effective_now - completed_at).total_seconds()
    if age_seconds < -1 or age_seconds > maximum_age_seconds:
        return _blocked(
            "BROKER_SNAPSHOT_STALE",
            snapshot_age_seconds=age_seconds,
        )
    positions = list(snapshot.get("positions", {}).get("positions", []))
    matches = [
        row for row in positions if _matches_con_id(row, int(con_id))
    ]
    if not matches:
        return _result(
            snapshot,
            "BROKER_POSITION_NOT_FOUND",
            age_seconds=age_seconds,
            position_count=len(positions),
            match_count=0,
            quantity_match=False,
        )
    if len(matches) != 1:
        return _result(
            snapshot,
            "AMBIGUOUS_BROKER_POSITION_BLOCKED",
            age_seconds=age_seconds,
            position_count=len(positions),
            match_count=len(matches),
            quantity_match=False,
        )
    try:
        expected_quantity = Decimal(str(position["quantity"]))
        observed_quantity = Decimal(str(matches[0]["position_quantity"]))
    except (InvalidOperation, KeyError):
        return _blocked("BROKER_POSITION_QUANTITY_INVALID")
    quantity_match = expected_quantity == observed_quantity
    return _result(
        snapshot,
        "MATCHED" if quantity_match else "QUANTITY_MISMATCH",
        age_seconds=age_seconds,
        position_count=len(positions),
        match_count=1,
        quantity_match=quantity_match,
        private_detail={
            "expected_quantity": str(expected_quantity),
            "observed_quantity": str(observed_quantity),
            "con_id": int(con_id),
        },
    )


def _latest_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        connection = sqlite3.connect(
            f"file:{path.resolve().as_posix()}?mode=ro",
            uri=True,
        )
        try:
            row = connection.execute(
                """
                SELECT snapshot_hash, payload_json
                FROM snapshots
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            connection.close()
    except (sqlite3.Error, OSError):
        return None
    if row is None:
        return None
    try:
        payload = json.loads(str(row[1]))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    payload["_stored_snapshot_hash"] = str(row[0])
    return payload


def _snapshot_path(project_root: Path, environment: str) -> Path:
    if environment == "paper":
        return (
            project_root
            / "data"
            / "broker"
            / "phase8"
            / "private"
            / "broker_observation.sqlite3"
        )
    return (
        project_root
        / "data"
        / "execution"
        / "live"
        / "private"
        / "broker_observation.sqlite3"
    )


def _result(
    snapshot: dict[str, Any],
    match_status: str,
    *,
    age_seconds: float,
    position_count: int,
    match_count: int,
    quantity_match: bool,
    private_detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "GO" if match_status == "MATCHED" else "NO_GO",
        "broker_match_status": match_status,
        "snapshot_hash": snapshot.get(
            "_stored_snapshot_hash",
            snapshot.get("content_hash"),
        ),
        "snapshot_age_seconds": max(0.0, age_seconds),
        "broker_position_count": position_count,
        "identity_match_count": match_count,
        "quantity_match": quantity_match,
        "private_detail": private_detail or {},
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }


def _blocked(
    reason: str,
    *,
    snapshot_age_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "status": "NO_GO",
        "broker_match_status": reason,
        "snapshot_hash": None,
        "snapshot_age_seconds": snapshot_age_seconds,
        "broker_position_count": None,
        "identity_match_count": 0,
        "quantity_match": False,
        "private_detail": {},
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }


def _parse_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _matches_con_id(row: Any, con_id: int) -> bool:
    if not isinstance(row, dict):
        return False
    try:
        return int(row.get("con_id", -1)) == con_id
    except (TypeError, ValueError):
        return False
