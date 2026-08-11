from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.paper_execution.storage import (
    PaperExecutionStore,
    Phase9Layout,
    artifact,
    write_json,
)


EVIDENCE_SCHEMA = "phase9_operator_attested_manual_completion_v1"
EVIDENCE_STATUS = "OPERATOR_ATTESTED_MANUAL_PAPER_ROUND_TRIP_GO"
PUBLIC_ARTIFACT = "operator-attested-manual-completion.json"
PRIVATE_ARTIFACT = "operator-attested-manual-completion.private.json"
MAX_TRANSITION_SECONDS = 300.0


def accept_operator_attested_manual_completion(
    project_root: Path,
    *,
    symbol: str,
    con_id: int,
    reason: str,
) -> dict[str, Any]:
    normalized_symbol = symbol.strip().upper()
    normalized_reason = reason.strip()
    if not normalized_symbol or con_id <= 0 or len(normalized_reason) < 12:
        return _result(
            status="NO_GO",
            classification="INVALID_OPERATOR_ATTESTATION",
            symbol=normalized_symbol,
            con_id=con_id,
        )

    evidence = reconstruct_manual_completion_evidence(
        project_root,
        symbol=normalized_symbol,
        con_id=con_id,
    )
    if evidence["status"] != "EVIDENCE_READY":
        return _result(
            status="NO_GO",
            classification=str(evidence["status"]),
            symbol=normalized_symbol,
            con_id=con_id,
            evidence=evidence,
        )

    result = _result(
        status=EVIDENCE_STATUS,
        classification="MANUAL_TWS_CLOSE_CONFIRMED_BY_BROKER_STATE_CONTINUITY",
        symbol=normalized_symbol,
        con_id=con_id,
        evidence=evidence,
        operator_attestation_hash=stable_hash(normalized_reason),
        paper_round_trip_operationally_accepted=True,
        api_closing_sell_path_proven=False,
        external_manual_close=True,
        phase9_ledger_mutated=False,
    )
    layout = Phase9Layout.from_project_root(project_root)
    write_json(layout.artifact(PUBLIC_ARTIFACT), result)
    private_path = layout.db_path.parent / PRIVATE_ARTIFACT
    write_json(
        private_path,
        artifact(
            "phase9_operator_attestation_private_v1",
            {
                "status": "GO",
                "reason": normalized_reason,
                "public_evidence_hash": result["content_hash"],
                "symbol": normalized_symbol,
                "con_id": con_id,
                "execution_authority": "NONE",
                "live_authority": "NONE",
            },
        ),
    )
    return result


def load_operator_completion_evidence(project_root: Path) -> dict[str, Any] | None:
    path = Phase9Layout.from_project_root(project_root).artifact(PUBLIC_ARTIFACT)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if operator_completion_evidence_valid(payload) else None


def operator_completion_evidence_valid(payload: dict[str, Any]) -> bool:
    expected_hash = stable_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )
    evidence = payload.get("evidence")
    return bool(
        payload.get("schema") == EVIDENCE_SCHEMA
        and payload.get("status") == EVIDENCE_STATUS
        and payload.get("content_hash") == expected_hash
        and isinstance(evidence, dict)
        and evidence.get("status") == "EVIDENCE_READY"
        and evidence.get("account_continuity_status")
        in {
            "SAME_ACCOUNT_FINGERPRINT",
            "PRE_MATCH_POST_ACCOUNT_TIMEOUT_SAME_OBSERVER",
            "PRE_MATCH_POST_CALLBACK_TIMEOUT_OPERATOR_ATTESTED",
        }
        and evidence.get("pre_position_count") == 1
        and _integer(evidence.get("pre_sell_order_count")) >= 1
        and evidence.get("post_position_count") == 0
        and evidence.get("post_open_order_count") == 0
        and _integer(evidence.get("phase9_buy_execution_count")) >= 1
        and evidence.get("canary_a_frozen_go") is True
        and payload.get("paper_round_trip_operationally_accepted") is True
        and payload.get("api_closing_sell_path_proven") is False
        and payload.get("phase9_ledger_mutated") is False
        and payload.get("execution_authority") == "NONE"
        and payload.get("live_authority") == "NONE"
        and payload.get("broker_write_calls") == 0
    )


def reconstruct_manual_completion_evidence(
    project_root: Path,
    *,
    symbol: str,
    con_id: int,
) -> dict[str, Any]:
    layout = Phase9Layout.from_project_root(project_root)
    store = PaperExecutionStore(layout.db_path)
    store.initialize()
    buy_executions = [
        row
        for row in store.list_executions()
        if _matches_instrument(row.get("payload", {}), symbol, con_id)
        and str(row.get("payload", {}).get("side", "")).upper() in {"BUY", "BOT"}
        and _positive(row.get("payload", {}).get("quantity"))
    ]
    if not buy_executions:
        return {"status": "PHASE9_BUY_EXECUTION_NOT_FOUND"}

    freeze = _read_json(layout.artifact("canary-a-evidence-freeze-status.json"))
    canary_a_frozen_go = bool(
        freeze
        and _content_hash_valid(freeze)
        and freeze.get("freeze_status")
        == "PHASE9_CANARY_A_EVIDENCE_ADOPTION_FROZEN_GO"
    )
    if not canary_a_frozen_go:
        return {"status": "CANARY_A_FROZEN_EVIDENCE_REQUIRED"}
    assert freeze is not None

    observation_db = (
        project_root
        / "data"
        / "broker"
        / "phase8"
        / "private"
        / "broker_observation.sqlite3"
    )
    transition = _find_broker_empty_transition(
        observation_db,
        symbol=symbol,
        con_id=con_id,
        expected_account_fingerprint_hash=str(
            buy_executions[0]
            .get("payload", {})
            .get("account_fingerprint_hash", "")
        ),
    )
    if transition is None:
        return {"status": "BROKER_EMPTY_CONTINUITY_NOT_FOUND"}

    return {
        "status": "EVIDENCE_READY",
        "evidence_basis": (
            "PHASE9_BUY_EXECUTION_PLUS_READ_ONLY_PRE_POST_BROKER_CONTINUITY"
        ),
        "symbol": symbol,
        "con_id": con_id,
        "phase9_buy_execution_count": len(buy_executions),
        "phase9_buy_execution_hashes": sorted(
            stable_hash(str(row.get("payload_hash", ""))) for row in buy_executions
        ),
        "canary_a_frozen_go": True,
        "canary_a_freeze_hash": freeze.get("content_hash"),
        **transition,
        "exact_sell_execution_observed": False,
        "sell_completion_classification": (
            "INFERRED_EXTERNAL_MANUAL_CLOSE_FROM_BROKER_STATE_CONTINUITY"
        ),
        "automatic_state_adoptions": 0,
        "broker_write_calls": 0,
    }


def _find_broker_empty_transition(
    db_path: Path,
    *,
    symbol: str,
    con_id: int,
    expected_account_fingerprint_hash: str,
) -> dict[str, Any] | None:
    if not db_path.is_file():
        return None
    connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT snapshot_hash, payload_json, created_at "
            "FROM snapshots ORDER BY created_at ASC"
        ).fetchall()
    finally:
        connection.close()
    parsed = [
        (str(snapshot_hash), json.loads(payload_json), str(created_at))
        for snapshot_hash, payload_json, created_at in rows
    ]
    for index, (pre_hash, pre, pre_time) in enumerate(parsed[:-1]):
        pre_positions = _instrument_positions(pre, symbol, con_id)
        pre_orders = _instrument_sell_orders(pre, symbol, con_id)
        if len(pre_positions) != 1 or not pre_orders:
            continue
        if _decimal(pre_positions[0].get("position_quantity")) != Decimal("1"):
            continue
        if not any(
            _decimal(order.get("total_quantity")) == Decimal("1")
            and str(order.get("order_status", "")).upper()
            in {"PRESUBMITTED", "SUBMITTED", "PENDINGSUBMIT"}
            for order in pre_orders
        ):
            continue
        pre_fingerprints = _account_fingerprints(pre)
        if expected_account_fingerprint_hash not in {
            stable_hash(fingerprint) for fingerprint in pre_fingerprints
        }:
            continue
        for post_hash, post, post_time in parsed[index + 1 :]:
            elapsed = (_parse_time(post_time) - _parse_time(pre_time)).total_seconds()
            if elapsed > MAX_TRANSITION_SECONDS:
                break
            if _instrument_positions(post, symbol, con_id):
                continue
            if _instrument_orders(post, symbol, con_id):
                continue
            post_fingerprints = _account_fingerprints(post)
            same_fingerprint = pre_fingerprints == post_fingerprints
            timeout_continuity = (
                not post_fingerprints
                and str(post.get("account", {}).get("status", "")).upper()
                == "CALLBACK_TIMEOUT"
                and _same_read_only_observer(pre, post)
            )
            callback_timeout_attestation = (
                not post_fingerprints
                and _all_observation_components_timed_out(post)
                and _same_read_only_observer_sequence(pre, post)
            )
            if (
                not same_fingerprint
                and not timeout_continuity
                and not callback_timeout_attestation
            ):
                continue
            if not _snapshot_components_complete(pre, require_account=True):
                continue
            post_verified = _snapshot_components_complete(
                post, require_account=False
            )
            if not post_verified and not callback_timeout_attestation:
                continue
            return {
                "pre_snapshot_hash": pre_hash,
                "post_snapshot_hash": post_hash,
                "pre_snapshot_at": pre_time,
                "post_snapshot_at": post_time,
                "transition_seconds": elapsed,
                "same_account_fingerprint": same_fingerprint,
                "same_observer_connection": same_fingerprint or timeout_continuity,
                "same_observer_sequence": True,
                "account_continuity_status": (
                    "SAME_ACCOUNT_FINGERPRINT"
                    if same_fingerprint
                    else (
                        "PRE_MATCH_POST_ACCOUNT_TIMEOUT_SAME_OBSERVER"
                        if timeout_continuity
                        else "PRE_MATCH_POST_CALLBACK_TIMEOUT_OPERATOR_ATTESTED"
                    )
                ),
                "post_account_component_status": str(
                    post.get("account", {}).get("status", "UNKNOWN")
                ),
                "account_fingerprint_count": len(pre_fingerprints),
                "pre_position_count": len(pre_positions),
                "pre_sell_order_count": len(pre_orders),
                "post_position_count": 0,
                "post_open_order_count": 0,
                "post_broker_state_verified": post_verified,
                "post_callback_timeout": callback_timeout_attestation,
                "snapshot_atomic": False,
                "continuity_status": (
                    "BROKER_RECONCILED_EMPTY_AFTER_MANUAL_SELL"
                    if post_verified
                    else "OPERATOR_ATTESTED_EMPTY_POST_CALLBACK_TIMEOUT"
                ),
            }
    return None


def _snapshot_components_complete(
    snapshot: dict[str, Any], *, require_account: bool
) -> bool:
    allowed = {"COMPLETE", "EMPTY_COMPLETE"}
    components = ["positions", "all_api_open_orders"]
    if require_account:
        components.append("account")
    return all(
        str(snapshot.get(component, {}).get("status", "")).upper() in allowed
        for component in components
    )


def _same_read_only_observer(
    pre: dict[str, Any], post: dict[str, Any]
) -> bool:
    pre_server_version = pre.get("server_version")
    return bool(
        pre.get("broker_observation_authority") == "READ_ONLY"
        and post.get("broker_observation_authority") == "READ_ONLY"
        and pre.get("execution_authority") == "NONE"
        and post.get("execution_authority") == "NONE"
        and pre_server_version is not None
        and pre_server_version == post.get("server_version")
    )


def _same_read_only_observer_sequence(
    pre: dict[str, Any], post: dict[str, Any]
) -> bool:
    return bool(
        pre.get("broker_observation_authority") == "READ_ONLY"
        and post.get("broker_observation_authority") == "READ_ONLY"
        and pre.get("execution_authority") == "NONE"
        and post.get("execution_authority") == "NONE"
    )


def _all_observation_components_timed_out(snapshot: dict[str, Any]) -> bool:
    return all(
        str(snapshot.get(component, {}).get("status", "")).upper()
        == "CALLBACK_TIMEOUT"
        for component in ("account", "positions", "all_api_open_orders")
    )


def _instrument_positions(
    snapshot: dict[str, Any], symbol: str, con_id: int
) -> list[dict[str, Any]]:
    rows = snapshot.get("positions", {}).get("positions", [])
    return [row for row in rows if _matches_instrument(row, symbol, con_id)]


def _instrument_orders(
    snapshot: dict[str, Any], symbol: str, con_id: int
) -> list[dict[str, Any]]:
    rows = snapshot.get("all_api_open_orders", {}).get("open_orders", [])
    return [row for row in rows if _matches_instrument(row, symbol, con_id)]


def _instrument_sell_orders(
    snapshot: dict[str, Any], symbol: str, con_id: int
) -> list[dict[str, Any]]:
    return [
        row
        for row in _instrument_orders(snapshot, symbol, con_id)
        if str(row.get("action", "")).upper() in {"SELL", "SLD"}
    ]


def _account_fingerprints(snapshot: dict[str, Any]) -> set[str]:
    values = snapshot.get("account", {}).get("values", [])
    return {
        str(row.get("account_fingerprint"))
        for row in values
        if isinstance(row, dict) and row.get("account_fingerprint")
    }


def _matches_instrument(row: dict[str, Any], symbol: str, con_id: int) -> bool:
    return int(row.get("con_id", -1)) == con_id and str(
        row.get("symbol", "")
    ).upper() == symbol


def _positive(value: Any) -> bool:
    return _decimal(value) > 0


def _integer(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _content_hash_valid(payload: dict[str, Any]) -> bool:
    return payload.get("content_hash") == stable_hash(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )


def _result(status: str, **fields: Any) -> dict[str, Any]:
    return artifact(
        EVIDENCE_SCHEMA,
        {
            "status": status,
            **fields,
            "execution_authority": "NONE",
            "strategy_authority": "NONE",
            "live_authority": "NONE",
            "broker_write_calls": 0,
            "automatic_submission": False,
        },
    )
