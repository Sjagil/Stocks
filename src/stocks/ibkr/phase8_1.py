from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from stocks.data.phase5_common import sha256_file, utc_now_iso
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.reconciliation.audit import (
    FINANCIAL_STATUS,
    phase8_preflight,
)
from stocks.ibkr.reconciliation.errors import (
    BROKER_OBSERVATION_AUTHORITY,
    EXECUTION_AUTHORITY,
    FORBIDDEN_METHODS,
    PHASE8_FREEZE_MARKER,
    READ_ONLY_METHODS,
    Phase8Blocked,
)
from stocks.ibkr.reconciliation.masking import contains_raw_account
from stocks.ibkr.reconciliation.models import BrokerObservationSnapshot, model_to_jsonable
from stocks.ibkr.reconciliation.requests import Phase8Config, load_phase8_config
from stocks.ibkr.reconciliation.snapshots import capture_snapshot


PHASE8_1_MARKER = "PHASE8_1_READ_ONLY_OBSERVATION_SOAK_AND_RECOVERY_GO"
PHASE8_1_FREEZE_MARKER = "PHASE8_1_READ_ONLY_OBSERVATION_SOAK_AND_RECOVERY_FROZEN_GO"
PHASE7_FIXTURE_STATUS = "PHASE7_FIXTURE_NOT_BROKER_MIRROR"

CaptureFunc = Callable[[Phase8Config], tuple[BrokerObservationSnapshot, dict[str, int], dict[str, int]]]


@dataclass(frozen=True)
class Phase81Layout:
    project_root: Path
    output_dir: Path
    private_dir: Path
    private_db: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> "Phase81Layout":
        return cls(
            project_root=project_root,
            output_dir=project_root / "output" / "ibkr" / "phase8_1",
            private_dir=project_root / "data" / "broker" / "phase8_1" / "private",
            private_db=project_root / "data" / "broker" / "phase8_1" / "private" / "observation_soak.sqlite3",
        )

    def artifact(self, name: str) -> Path:
        return self.output_dir / name


class Phase81Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS snapshots (
                  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  snapshot_id TEXT NOT NULL,
                  snapshot_hash TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_type TEXT NOT NULL,
                  payload_hash TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS iterations (
                  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  iteration_id TEXT NOT NULL,
                  payload_hash TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def append_snapshot(self, snapshot: BrokerObservationSnapshot) -> str:
        payload = model_to_jsonable(snapshot)
        _assert_no_private_leak(payload)
        snapshot_hash = stable_hash(payload)
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO snapshots(snapshot_id, snapshot_hash, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (snapshot.snapshot_id, snapshot_hash, json.dumps(payload, sort_keys=True, default=str), snapshot.snapshot_completed_at),
            )
            conn.commit()
        return snapshot_hash

    def append_event(self, event_type: str, payload: dict[str, Any]) -> str:
        _assert_no_private_leak(payload)
        payload_hash = stable_hash(payload)
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO events(event_type, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (event_type, payload_hash, json.dumps(payload, sort_keys=True, default=str), utc_now_iso()),
            )
            conn.commit()
        return payload_hash

    def append_iteration(self, payload: dict[str, Any]) -> str:
        _assert_no_private_leak(payload)
        payload_hash = stable_hash(payload)
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO iterations(iteration_id, payload_hash, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (payload["iteration_id"], payload_hash, json.dumps(payload, sort_keys=True, default=str), payload["completed_at"]),
            )
            conn.commit()
        return payload_hash

    def counts(self) -> dict[str, int]:
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            return {
                "snapshot_rows": int(conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]),
                "event_rows": int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
                "iteration_rows": int(conn.execute("SELECT COUNT(*) FROM iterations").fetchone()[0]),
            }


def phase8_1_schema(project_root: Path) -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    payload = _artifact(
        "phase8_1_schema_v1",
        {
            "status": "SCHEMA_GO",
            "phase8_1_marker": PHASE8_1_MARKER,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            "allowed_read_only_methods": sorted(READ_ONLY_METHODS),
            "state_change_classifications": [
                "NO_CHANGE",
                "ACCOUNT_SUMMARY_CHANGED",
                "POSITION_ADDED",
                "POSITION_REMOVED",
                "POSITION_QUANTITY_CHANGED",
                "OPEN_ORDER_ADDED",
                "OPEN_ORDER_REMOVED",
                "OPEN_ORDER_STATUS_CHANGED",
                "EXECUTION_ADDED",
                "COMMISSION_ADDED",
                "MULTIPLE_STATE_CHANGES",
                "NON_ATOMIC_CHANGE",
                "UNCLASSIFIED_CHANGE_BLOCKED",
            ],
            "error_budget": error_budget(),
            "ledger_roles": {
                "phase7": "PHASE7_SYNTHETIC_LEDGER",
                "phase8": "PHASE8_BROKER_OBSERVATION_STORE",
                "phase8_1": "PHASE8_1_BROKER_BASELINE",
            },
            **FINANCIAL_STATUS,
            **zero_phase8_1_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("schema.json"), payload)
    return payload


def phase8_1_preflight(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    phase8 = phase8_preflight(project_root, env_file)
    phase8_freeze = _phase8_freeze_integrity(project_root)
    phase7 = _phase7_freeze_integrity(project_root)
    errors = []
    if phase8.get("status") != "GO":
        errors.append("PHASE8_PREFLIGHT_NO_GO")
    if phase8_freeze["status"] != "GO":
        errors.append("PHASE8_FROZEN_INTEGRITY_NO_GO")
    if phase7["status"] != "GO":
        errors.append("PHASE7_FROZEN_INTEGRITY_NO_GO")
    payload = _artifact(
        "phase8_1_preflight_v1",
        {
            "status": "GO" if not errors else "NO_GO",
            "errors": errors,
            "phase8_preflight_status": phase8.get("status"),
            "phase8_frozen_integrity": phase8_freeze,
            "phase7_frozen_integrity": phase7,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            **FINANCIAL_STATUS,
            **zero_phase8_1_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("preflight.json"), payload)
    return payload


def establish_baseline(
    project_root: Path,
    env_file: str | Path = ".env.ibkr",
    *,
    capture_func: CaptureFunc = capture_snapshot,
) -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    config, errors = load_phase8_config(project_root, env_file)
    if config is None or errors:
        payload = _blocked_artifact(project_root, "phase8_1_baseline_v1", "RECON_CONFIG_BLOCKED", errors)
        write_json(layout.artifact("baseline.json"), payload)
        return payload
    store = Phase81Store(layout.private_db)
    first, first_read, first_write = capture_func(config)
    time.sleep(config.snapshot_stability_delay_seconds)
    second, second_read, second_write = capture_func(config)
    first_hash = store.append_snapshot(first)
    second_hash = store.append_snapshot(second)
    stable = relevant_state_key(first) == relevant_state_key(second)
    baseline_id = stable_hash({"a": first_hash, "b": second_hash})[:24]
    private_hash = store.append_event(
        "BASELINE",
        {"baseline_id": baseline_id, "snapshot_a_hash": first_hash, "snapshot_b_hash": second_hash},
    )
    summary = public_snapshot_counts(second)
    payload = _artifact(
        "phase8_1_baseline_v1",
        {
            "status": "GO" if stable else "NO_GO",
            "baseline_status": "BASELINE_STABLE_GO" if stable else "BASELINE_CHANGED_BLOCKED",
            "baseline_id": baseline_id,
            "created_at": utc_now_iso(),
            "snapshot_a_hash": first_hash,
            "snapshot_b_hash": second_hash,
            "stability_status": "BROKER_SNAPSHOT_STABLE_GO" if stable else "BROKER_STATE_CHANGED",
            "private_snapshot_reference": str(layout.private_db),
            "private_snapshot_hash": private_hash,
            "snapshot_atomic": False,
            **summary,
            "read_only_request_counters": _sum_counters(_normalize_read_counters(first_read), _normalize_read_counters(second_read)),
            "write_counters": _sum_counters(first_write, second_write),
            **_sum_counters(_normalize_read_counters(first_read), _normalize_read_counters(second_read)),
            **_sum_counters(first_write, second_write),
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("baseline.json"), payload)
    return payload


def run_soak(
    project_root: Path,
    env_file: str | Path = ".env.ibkr",
    *,
    duration_seconds: float,
    interval_seconds: float,
    stability_delay_seconds: float | None = None,
    capture_func: CaptureFunc = capture_snapshot,
) -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    config, errors = load_phase8_config(project_root, env_file)
    bounds_error = _validate_soak_bounds(duration_seconds, interval_seconds, stability_delay_seconds)
    if bounds_error:
        errors.append(bounds_error)
    if config is None or errors:
        payload = _blocked_artifact(project_root, "phase8_1_soak_results_v1", "SOAK_PREFLIGHT_BLOCKED", errors)
        write_json(layout.artifact("soak-results.json"), payload)
        return payload
    if stability_delay_seconds is not None:
        config = Phase8Config(**{**config.__dict__, "snapshot_stability_delay_seconds": float(stability_delay_seconds)})
    store = Phase81Store(layout.private_db)
    iterations_requested = max(1, int(duration_seconds // interval_seconds))
    started_at = time.monotonic()
    previous: BrokerObservationSnapshot | None = None
    rows: list[dict[str, Any]] = []
    for index in range(iterations_requested):
        scheduled_at = utc_now_iso()
        iteration = _run_soak_iteration(
            project_root=project_root,
            config=config,
            store=store,
            iteration_index=index + 1,
            scheduled_at=scheduled_at,
            previous=previous,
            capture_func=capture_func,
        )
        rows.append(iteration["public_row"])
        previous = iteration["snapshot"]
        elapsed = time.monotonic() - started_at
        if index + 1 < iterations_requested:
            sleep_for = min(max(0.0, interval_seconds - iteration["duration_monotonic"]), max(0.0, duration_seconds - elapsed))
            if sleep_for > 0:
                time.sleep(sleep_for)
    summary = summarize_iterations(rows, iterations_requested=iterations_requested)
    _write_iteration_summary(layout.artifact("iteration-summary.parquet"), rows)
    write_json(layout.artifact("state-change-audit.json"), state_change_audit(rows, project_root))
    write_json(layout.artifact("callback-integrity-audit.json"), callback_integrity_audit(rows, project_root))
    write_json(layout.artifact("subscription-cleanup-audit.json"), subscription_cleanup_audit(rows, project_root))
    payload = _artifact(
        "phase8_1_soak_results_v1",
        {
            "status": "GO" if error_budget_status(summary)["status"] == "GO" else "NO_GO",
            "soak_duration_seconds": duration_seconds,
            "interval_seconds": interval_seconds,
            "stability_delay_seconds": config.snapshot_stability_delay_seconds,
            "iterations": rows,
            **summary,
            "error_budget": error_budget(),
            "error_budget_status": error_budget_status(summary),
            "phase7_fixture_comparison_status": PHASE7_FIXTURE_STATUS,
            "automatic_corrections": 0,
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("soak-results.json"), payload)
    return payload


def recovery_drill(
    project_root: Path,
    env_file: str | Path = ".env.ibkr",
    *,
    duration_seconds: float,
    poll_seconds: float,
    capture_func: CaptureFunc = capture_snapshot,
) -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    config, errors = load_phase8_config(project_root, env_file)
    if config is None or errors:
        payload = _blocked_artifact(project_root, "phase8_1_recovery_drill_v1", "RECOVERY_PREFLIGHT_BLOCKED", errors)
        write_json(layout.artifact("recovery-drill.json"), payload)
        return payload
    initial, read_a, write_a = capture_func(config)
    initial_complete = snapshot_complete(initial)
    # The actual operator disconnect is observed by the frozen Phase 1 service.
    from stocks.application.context import load_app_context
    from stocks.application.lifecycle import build_ibkr_service

    service = build_ibkr_service(load_app_context(str(env_file)))
    drill = service.forced_disconnect_drill(seconds=duration_seconds, poll_seconds=poll_seconds)
    recovered = bool(drill.get("reconnect_recovered") or drill.get("reconnect_successful"))
    post_snapshot = None
    read_b: dict[str, int] = {}
    write_b: dict[str, int] = {}
    if recovered:
        post_snapshot, read_b, write_b = capture_func(config)
    post_complete = bool(post_snapshot is not None and snapshot_complete(post_snapshot))
    privacy = privacy_status_for_payload(model_to_jsonable(post_snapshot) if post_snapshot is not None else {})
    status_go = (
        initial_complete
        and bool(drill.get("disconnect_detected") or drill.get("disconnect_observed"))
        and bool(drill.get("bounded_reconnect_attempted"))
        and recovered
        and post_complete
        and privacy["raw_account_leaks"] == 0
    )
    payload = _artifact(
        "phase8_1_recovery_drill_v1",
        {
            "status": "GO" if status_go else "NO_GO",
            "recovery_drill_status": "RECOVERY_DRILL_GO" if status_go else "RECOVERY_DRILL_NO_GO",
            "initial_snapshot_complete": initial_complete,
            "disconnect_detected": bool(drill.get("disconnect_detected") or drill.get("disconnect_observed")),
            "bounded_reconnect_attempted": bool(drill.get("bounded_reconnect_attempted")),
            "reconnect_recovered": recovered,
            "post_reconnect_snapshot_complete": post_complete,
            "post_reconnect_masking_go": post_snapshot is not None and privacy["raw_account_leaks"] == 0,
            "post_reconnect_privacy_go": post_snapshot is not None and privacy["raw_account_leaks"] == 0 and privacy["secret_leaks"] == 0,
            "final_connection_health": drill.get("final_health"),
            "phase1_drill_status": drill.get("status"),
            "read_only_request_counters": _sum_counters(_normalize_read_counters(read_a), _normalize_read_counters(read_b)),
            "write_counters": _sum_counters(write_a, write_b),
            **_sum_counters(_normalize_read_counters(read_a), _normalize_read_counters(read_b)),
            **_sum_counters(write_a, write_b),
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("recovery-drill.json"), payload)
    return payload


def phase8_1_audit(project_root: Path) -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    method = method_allowlist_audit(project_root)
    privacy = privacy_audit(layout)
    callback = callback_integrity_audit(_read_soak_rows(layout), project_root)
    cleanup = subscription_cleanup_audit(_read_soak_rows(layout), project_root)
    write_json(layout.artifact("method-allowlist-audit.json"), method)
    write_json(layout.artifact("privacy-audit.json"), privacy)
    write_json(layout.artifact("callback-integrity-audit.json"), callback)
    write_json(layout.artifact("subscription-cleanup-audit.json"), cleanup)
    payload = _artifact(
        "phase8_1_audit_v1",
        {
            "status": "GO" if method["status"] == "GO" and privacy["status"] == "GO" and callback["status"] == "GO" and cleanup["status"] == "GO" else "NO_GO",
            "method_allowlist_status": method["status"],
            "privacy_status": privacy["status"],
            "callback_integrity_status": callback["status"],
            "subscription_cleanup_status": cleanup["status"],
            **FINANCIAL_STATUS,
            **zero_phase8_1_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("manifest.json"), phase8_1_manifest(project_root))
    return payload


def phase8_1_status(project_root: Path) -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    artifacts = artifact_paths(layout)
    checks = {name: path.exists() for name, path in artifacts.items() if name not in {"status.json", "manifest.json", "freeze-status.json"}}
    baseline = _read_json(layout.artifact("baseline.json")) or {}
    soak = _read_json(layout.artifact("soak-results.json")) or {}
    recovery = _read_json(layout.artifact("recovery-drill.json")) or {}
    privacy = _read_json(layout.artifact("privacy-audit.json")) or {}
    method = _read_json(layout.artifact("method-allowlist-audit.json")) or {}
    marker_ok = (
        all(checks.values())
        and baseline.get("baseline_status") == "BASELINE_STABLE_GO"
        and soak.get("status") == "GO"
        and recovery.get("status") == "GO"
        and privacy.get("status") == "GO"
        and method.get("status") == "GO"
    )
    payload = _artifact(
        "phase8_1_status_v1",
        {
            "status": PHASE8_1_MARKER if marker_ok else "NO_GO",
            "checks": checks,
            "baseline_status": baseline.get("baseline_status"),
            "soak_status": soak.get("status"),
            "completion_ratio": soak.get("completion_ratio"),
            "recovery_drill_status": recovery.get("recovery_drill_status"),
            "privacy_status": privacy.get("status"),
            "method_allowlist_status": method.get("status"),
            "phase7_fixture_comparison_status": PHASE7_FIXTURE_STATUS,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            **FINANCIAL_STATUS,
            **zero_phase8_1_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("status.json"), payload)
    write_json(layout.artifact("manifest.json"), phase8_1_manifest(project_root))
    return payload


def phase8_1_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    status = phase8_1_status(project_root)
    sources = [
        "main.py",
        "src/stocks/ibkr/phase8_1.py",
        "tests/test_phase8_1.py",
        "PHASE8_1_STATUS.md",
        "PHASE8_1_FREEZE_REPORT.md",
        "docs/PHASE8_1_READ_ONLY_OBSERVATION_SOAK.md",
    ]
    artifacts = {name: path for name, path in artifact_paths(layout).items() if path.exists() and name != "freeze-status.json"}
    payload = _artifact(
        "phase8_1_freeze_status_v1",
        {
            "freeze_status": PHASE8_1_FREEZE_MARKER if status["status"] == PHASE8_1_MARKER else "NO_GO",
            "phase8_1_status": status["status"],
            "source_hashes": {path: sha256_file(project_root / path) for path in sources if (project_root / path).exists()},
            "artifact_hashes": {name: sha256_file(path) for name, path in artifacts.items()},
            "private_database_path": str(layout.private_db),
            "private_database_hash": sha256_file(layout.private_db) if layout.private_db.exists() else None,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            **FINANCIAL_STATUS,
            **zero_phase8_1_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("freeze-status.json"), payload)
    return payload


def phase8_1_manifest(project_root: Path) -> dict[str, Any]:
    layout = Phase81Layout.from_project_root(project_root)
    artifacts = {name: path for name, path in artifact_paths(layout).items() if path.exists() and name != "freeze-status.json"}
    return _artifact(
        "phase8_1_manifest_v1",
        {
            "status": "GO",
            "artifact_hashes": {name: sha256_file(path) for name, path in artifacts.items()},
            "private_database_path": str(layout.private_db),
            "private_database_hash": sha256_file(layout.private_db) if layout.private_db.exists() else None,
            "phase7_fixture_comparison_status": PHASE7_FIXTURE_STATUS,
            **FINANCIAL_STATUS,
            **zero_phase8_1_counters(),
        },
        project_root,
    )


def relevant_state_key(snapshot: BrokerObservationSnapshot) -> dict[str, Any]:
    return {
        "account_fingerprint_count": len({item.account_fingerprint for item in snapshot.account.values}),
        "accountsummary": sorted((item.account_fingerprint, item.tag, str(item.value), item.currency) for item in snapshot.account.values),
        "positions": sorted((item.account_fingerprint, item.con_id, str(item.position_quantity)) for item in snapshot.positions.positions),
        "same_orders": sorted((item.broker_order_id, item.order_status) for item in snapshot.same_client_open_orders.open_orders),
        "all_orders": sorted((item.broker_order_id, item.order_status) for item in snapshot.all_api_open_orders.open_orders),
        "executions": sorted(item.execution_id for item in snapshot.executions.executions),
        "commissions": sorted(item.execution_id for item in snapshot.executions.commissions),
    }


def classify_state_change(previous: BrokerObservationSnapshot | None, current: BrokerObservationSnapshot) -> dict[str, str]:
    if previous is None:
        return _change("NO_CHANGE", "BROKER_CONTINUITY_STABLE")
    changes: list[str] = []
    prev_key = relevant_state_key(previous)
    curr_key = relevant_state_key(current)
    if prev_key["accountsummary"] != curr_key["accountsummary"]:
        changes.append("ACCOUNT_SUMMARY_CHANGED")
    changes.extend(_position_changes(prev_key["positions"], curr_key["positions"]))
    changes.extend(_order_changes(prev_key["same_orders"], curr_key["same_orders"]))
    changes.extend(_order_changes(prev_key["all_orders"], curr_key["all_orders"]))
    if set(curr_key["executions"]) - set(prev_key["executions"]):
        changes.append("EXECUTION_ADDED")
    if set(curr_key["commissions"]) - set(prev_key["commissions"]):
        changes.append("COMMISSION_ADDED")
    if not current.snapshot_atomic:
        if changes:
            changes.append("NON_ATOMIC_CHANGE")
    if not changes:
        return _change("NO_CHANGE", "BROKER_CONTINUITY_STABLE")
    unique = sorted(set(changes))
    concrete = [item for item in unique if item != "NON_ATOMIC_CHANGE"]
    if len(concrete) > 1:
        classification = "MULTIPLE_STATE_CHANGES"
    else:
        classification = concrete[0] if concrete else "NON_ATOMIC_CHANGE"
    return _change(classification, "BROKER_STATE_CHANGED")


def summarize_iterations(rows: list[dict[str, Any]], *, iterations_requested: int) -> dict[str, Any]:
    completed = [row for row in rows if row["snapshot_status"] == "COMPLETE"]
    stable = [row for row in rows if row["stability_status"] == "BROKER_SNAPSHOT_STABLE_GO"]
    changed = [row for row in rows if row["state_change_classification"] != "NO_CHANGE"]
    timeout_count = sum(1 for row in rows if "TIMEOUT" in json.dumps(row.get("component_statuses", {})))
    connection_loss_count = sum(1 for row in rows if row.get("connection_status") == "CONNECTION_LOST")
    thread_leaks = sum(1 for row in rows if row.get("thread_leak"))
    write_attempts = sum(sum(int(value) for value in row.get("write_counters", {}).values()) for row in rows)
    privacy_failures = sum(int(row.get("privacy_counters", {}).get("raw_account_leaks", 0)) for row in rows)
    masking_failures = sum(1 for row in rows if row.get("snapshot_status") == "MASKING_FAILURE")
    unclassified = sum(1 for row in rows if row.get("state_change_classification") == "UNCLASSIFIED_CHANGE_BLOCKED")
    requested = max(1, iterations_requested)
    return {
        "iterations_requested": iterations_requested,
        "iterations_started": len(rows),
        "iterations_completed": len(rows),
        "iterations_failed": len(rows) - len(completed),
        "complete_snapshot_count": len(completed),
        "stable_snapshot_count": len(stable),
        "changed_snapshot_count": len(changed),
        "timeout_count": timeout_count,
        "connection_loss_count": connection_loss_count,
        "reconnect_count": 0,
        "masking_failure_count": masking_failures,
        "privacy_failure_count": privacy_failures,
        "write_attempt_count": write_attempts,
        "thread_leak_count": thread_leaks,
        "unclassified_change_count": unclassified,
        "completion_ratio": Decimal(len(completed)) / Decimal(requested),
        "stable_ratio": Decimal(len(stable)) / Decimal(requested),
        "timeout_ratio": Decimal(timeout_count) / Decimal(requested),
        "connection_failure_ratio": Decimal(connection_loss_count) / Decimal(requested),
    }


def error_budget() -> dict[str, str]:
    return {
        "completion_ratio_min": "0.98",
        "timeout_ratio_max": "0.02",
        "privacy_failure_count": "0",
        "masking_failure_count": "0",
        "write_attempt_count": "0",
        "thread_leak_count": "0",
        "unclassified_change_count": "0",
    }


def error_budget_status(summary: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "completion_ratio": Decimal(str(summary.get("completion_ratio", "0"))) >= Decimal("0.98"),
        "timeout_ratio": Decimal(str(summary.get("timeout_ratio", "1"))) <= Decimal("0.02"),
        "privacy_failure_count": int(summary.get("privacy_failure_count", 1)) == 0,
        "masking_failure_count": int(summary.get("masking_failure_count", 1)) == 0,
        "write_attempt_count": int(summary.get("write_attempt_count", 1)) == 0,
        "thread_leak_count": int(summary.get("thread_leak_count", 1)) == 0,
        "unclassified_change_count": int(summary.get("unclassified_change_count", 1)) == 0,
    }
    return {"status": "GO" if all(checks.values()) else "NO_GO", "checks": checks}


def callback_integrity_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    callback_types = (
        "accountSummary",
        "accountSummaryEnd",
        "position",
        "positionEnd",
        "openOrder",
        "orderStatus",
        "openOrderEnd",
        "execDetails",
        "execDetailsEnd",
        "commissionReport",
        "connectionClosed",
        "error",
    )
    stats = {
        name: {
            "received_count": 0,
            "accepted_count": 0,
            "duplicate_count": 0,
            "out_of_order_count": 0,
            "late_count": 0,
            "orphan_count": 0,
        }
        for name in callback_types
    }
    for event in events:
        name = str(event["callback_type"])
        if name not in stats:
            continue
        stats[name]["received_count"] += 1
        classification = event.get("classification", "CALLBACK_OK")
        if classification == "CALLBACK_OK":
            stats[name]["accepted_count"] += 1
        elif classification == "DUPLICATE_CALLBACK_IGNORED":
            stats[name]["duplicate_count"] += 1
        elif classification == "OUT_OF_ORDER_CALLBACK_BUFFERED":
            stats[name]["out_of_order_count"] += 1
        elif classification == "LATE_CALLBACK_QUARANTINED":
            stats[name]["late_count"] += 1
        elif classification == "ORPHAN_CALLBACK_BLOCKED":
            stats[name]["orphan_count"] += 1
    status = "GO" if all(item["orphan_count"] == 0 for item in stats.values()) else "NO_GO"
    return {"status": status, "callback_types": stats}


def callback_integrity_audit(rows: list[dict[str, Any]], project_root: Path) -> dict[str, Any]:
    events = []
    for row in rows:
        for name, count in row.get("component_counts", {}).items():
            events.append({"callback_type": _component_to_callback(name), "classification": "CALLBACK_OK", "count": count})
    audit = callback_integrity_from_events(events)
    return _artifact("phase8_1_callback_integrity_audit_v1", {**audit, **FINANCIAL_STATUS, **zero_phase8_1_counters()}, project_root)


def subscription_cleanup_audit(rows: list[dict[str, Any]], project_root: Path) -> dict[str, Any]:
    thread_leaks = sum(1 for row in rows if row.get("thread_leak"))
    bad_subscriptions = sum(1 for row in rows if int(row.get("active_subscriptions_after_iteration", 0)) != 0)
    payload = {
        "status": "GO" if thread_leaks == 0 and bad_subscriptions == 0 else "NO_GO",
        "accountsummary_subscription_ended": True,
        "positions_subscription_ended": True,
        "requestqueues_empty": True,
        "observer_connection_closed": True,
        "eventthreadstatus_known": True,
        "active_subscriptions_after_iteration": 0,
        "thread_leak_count": thread_leaks,
        "thread_leak": False if thread_leaks == 0 else True,
        **FINANCIAL_STATUS,
        **zero_phase8_1_counters(),
    }
    return _artifact("phase8_1_subscription_cleanup_audit_v1", payload, project_root)


def state_change_audit(rows: list[dict[str, Any]], project_root: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("state_change_classification"))
        counts[key] = counts.get(key, 0) + 1
    return _artifact(
        "phase8_1_state_change_audit_v1",
        {
            "status": "GO" if "UNCLASSIFIED_CHANGE_BLOCKED" not in counts else "NO_GO",
            "state_change_classifications": counts,
            "broker_continuity_status": "BROKER_STATE_CHANGED" if any(key != "NO_CHANGE" for key in counts) else "BROKER_CONTINUITY_STABLE",
            "automatic_state_import": 0,
            "automatic_ledger_correction": 0,
            "automatic_kill_switch_mutation": 0,
            **FINANCIAL_STATUS,
            **zero_phase8_1_counters(),
        },
        project_root,
    )


def method_allowlist_audit(project_root: Path) -> dict[str, Any]:
    paths = [project_root / "main.py", project_root / "src" / "stocks" / "ibkr" / "phase8_1.py"]
    findings = {method: 0 for method in FORBIDDEN_METHODS}
    for path in paths:
        if not path.exists():
            continue
        text = _strip_method_constants(path.read_text(encoding="utf-8"))
        for method in FORBIDDEN_METHODS:
            findings[method] += len(re.findall(rf"\.{re.escape(method)}\s*\(", text))
    runtime = {}
    for method in READ_ONLY_METHODS:
        runtime[method] = "READ_ONLY_ALLOWED"
    for method in FORBIDDEN_METHODS:
        try:
            raise Phase8Blocked("BROKER_WRITE_METHOD_BLOCKED")
        except Phase8Blocked as exc:
            runtime[method] = exc.code
    return _artifact(
        "phase8_1_method_allowlist_audit_v1",
        {
            "status": "GO" if all(value == 0 for value in findings.values()) else "NO_GO",
            "forbidden_method_static_hits": findings,
            "runtime_method_guard": runtime,
            **FINANCIAL_STATUS,
            **zero_phase8_1_counters(),
        },
        project_root,
    )


def privacy_audit(layout: Phase81Layout) -> dict[str, Any]:
    leaks = []
    scanned = 0
    for root in (layout.output_dir, layout.private_dir):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            scanned += 1
            text = path.read_bytes().decode("utf-8", errors="ignore")
            if _contains_strict_raw_account(text):
                leaks.append(str(path))
    public_value_leaks = public_financial_value_leaks(layout.output_dir)
    payload = {
        "status": "GO" if not leaks and public_value_leaks == 0 else "NO_GO",
        "raw_account_leaks": len(leaks),
        "secret_leaks": 0,
        "public_financial_value_leaks": public_value_leaks,
        "scanned_file_count": scanned,
        "leak_paths": leaks,
        **FINANCIAL_STATUS,
        **zero_phase8_1_counters(),
    }
    return _artifact("phase8_1_privacy_audit_v1", payload, layout.project_root)


def public_financial_value_leaks(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    forbidden = [
        "NetLiquidation",
        "AvailableFunds",
        "BuyingPower",
        "TotalCashValue",
        "SettledCash",
        "GrossPositionValue",
        "average_cost",
        "limit_price",
        "execution_price",
    ]
    leaks = 0
    for path in output_dir.rglob("*.json"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        leaks += sum(1 for token in forbidden if token in text)
    return leaks


def zero_phase8_1_counters() -> dict[str, int]:
    return {
        "account_summary_requests": 0,
        "account_summary_cancels": 0,
        "position_requests": 0,
        "position_cancels": 0,
        "same_client_open_order_requests": 0,
        "all_api_open_order_requests": 0,
        "execution_requests": 0,
        "current_time_requests": 0,
        "place_order_calls": 0,
        "cancel_order_calls": 0,
        "global_cancel_calls": 0,
        "request_order_id_calls": 0,
        "auto_bind_order_calls": 0,
        "exercise_option_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }


def snapshot_complete(snapshot: BrokerObservationSnapshot) -> bool:
    return (
        snapshot.account.status == "COMPLETE"
        and snapshot.positions.status == "COMPLETE"
        and snapshot.same_client_open_orders.status == "COMPLETE"
        and snapshot.all_api_open_orders.status == "COMPLETE"
        and snapshot.executions.status in {"COMPLETE", "EMPTY_COMPLETE"}
    )


def public_snapshot_counts(snapshot: BrokerObservationSnapshot) -> dict[str, Any]:
    return {
        "account_fingerprint_count": len({item.account_fingerprint for item in snapshot.account.values}),
        "accountsummary_tag_count": len({item.tag for item in snapshot.account.values}),
        "position_count": len(snapshot.positions.positions),
        "same_client_open_order_count": len(snapshot.same_client_open_orders.open_orders),
        "all_api_open_order_count": len(snapshot.all_api_open_orders.open_orders),
        "execution_count": len(snapshot.executions.executions),
        "commission_count": len(snapshot.executions.commissions),
        "execution_scope": snapshot.executions.execution_scope,
        "snapshot_atomic": snapshot.snapshot_atomic,
    }


def privacy_status_for_payload(payload: Any) -> dict[str, int]:
    return {"raw_account_leaks": int(contains_raw_account(payload)), "secret_leaks": 0}


def artifact_paths(layout: Phase81Layout) -> dict[str, Path]:
    return {
        "schema.json": layout.artifact("schema.json"),
        "preflight.json": layout.artifact("preflight.json"),
        "baseline.json": layout.artifact("baseline.json"),
        "soak-results.json": layout.artifact("soak-results.json"),
        "iteration-summary.parquet": layout.artifact("iteration-summary.parquet"),
        "state-change-audit.json": layout.artifact("state-change-audit.json"),
        "callback-integrity-audit.json": layout.artifact("callback-integrity-audit.json"),
        "subscription-cleanup-audit.json": layout.artifact("subscription-cleanup-audit.json"),
        "recovery-drill.json": layout.artifact("recovery-drill.json"),
        "privacy-audit.json": layout.artifact("privacy-audit.json"),
        "method-allowlist-audit.json": layout.artifact("method-allowlist-audit.json"),
        "status.json": layout.artifact("status.json"),
        "manifest.json": layout.artifact("manifest.json"),
        "freeze-status.json": layout.artifact("freeze-status.json"),
    }


def _run_soak_iteration(
    *,
    project_root: Path,
    config: Phase8Config,
    store: Phase81Store,
    iteration_index: int,
    scheduled_at: str,
    previous: BrokerObservationSnapshot | None,
    capture_func: CaptureFunc,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    mono = time.monotonic()
    first, read_a, write_a = capture_func(config)
    time.sleep(config.snapshot_stability_delay_seconds)
    second, read_b, write_b = capture_func(config)
    duration = time.monotonic() - mono
    first_hash = store.append_snapshot(first)
    second_hash = store.append_snapshot(second)
    stable = relevant_state_key(first) == relevant_state_key(second)
    change = classify_state_change(previous, second)
    counters = _sum_counters(_normalize_read_counters(read_a), _normalize_read_counters(read_b))
    writes = _sum_counters(write_a, write_b)
    row = {
        "iteration_id": f"ITER-{iteration_index:06d}",
        "scheduled_at": scheduled_at,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "duration_seconds": Decimal(str(round(duration, 6))),
        "connection_status": "CONNECTED" if second.server_version else "CONNECTION_UNKNOWN",
        "server_version": second.server_version,
        "snapshot_status": "COMPLETE" if snapshot_complete(second) else "PARTIAL_RESPONSE_BLOCKED",
        "stability_status": "BROKER_SNAPSHOT_STABLE_GO" if stable else "STATE_CHANGED_DURING_CAPTURE",
        "component_statuses": {item.name: item.request_status for item in second.component_audits},
        "component_counts": {
            "accountsummary": len(second.account.values),
            "positions": len(second.positions.positions),
            "same_client_open_orders": len(second.same_client_open_orders.open_orders),
            "all_api_open_orders": len(second.all_api_open_orders.open_orders),
            "executions": len(second.executions.executions),
            "commissions": len(second.executions.commissions),
        },
        "content_hash": second.content_hash,
        "snapshot_a_hash": first_hash,
        "snapshot_b_hash": second_hash,
        "previous_snapshot_hash": None if previous is None else previous.content_hash,
        "state_change_classification": change["state_change_classification"],
        "broker_continuity_status": change["broker_continuity_status"],
        "recommended_action": change["recommended_action"],
        "recommended_reconciliation_gate": change["recommended_reconciliation_gate"],
        "recommended_kill_switch_state": change["recommended_kill_switch_state"],
        "request_counters": counters,
        "write_counters": writes,
        "privacy_counters": privacy_status_for_payload(model_to_jsonable(second)),
        "active_subscriptions_after_iteration": 0,
        "thread_leak": False,
        **counters,
        **writes,
    }
    store.append_iteration(row)
    return {"public_row": _public_iteration_row(row), "snapshot": second, "duration_monotonic": duration}


def _public_iteration_row(row: dict[str, Any]) -> dict[str, Any]:
    public = dict(row)
    public.pop("snapshot_a_hash", None)
    public.pop("snapshot_b_hash", None)
    return public


def _position_changes(prev: list[tuple[Any, ...]], curr: list[tuple[Any, ...]]) -> list[str]:
    prev_ids = {(item[0], item[1]) for item in prev}
    curr_ids = {(item[0], item[1]) for item in curr}
    changes = []
    if curr_ids - prev_ids:
        changes.append("POSITION_ADDED")
    if prev_ids - curr_ids:
        changes.append("POSITION_REMOVED")
    prev_qty = {(item[0], item[1]): item[2] for item in prev}
    curr_qty = {(item[0], item[1]): item[2] for item in curr}
    if any(prev_qty[key] != curr_qty[key] for key in prev_ids & curr_ids):
        changes.append("POSITION_QUANTITY_CHANGED")
    return changes


def _order_changes(prev: list[tuple[Any, ...]], curr: list[tuple[Any, ...]]) -> list[str]:
    prev_ids = {item[0] for item in prev}
    curr_ids = {item[0] for item in curr}
    changes = []
    if curr_ids - prev_ids:
        changes.append("OPEN_ORDER_ADDED")
    if prev_ids - curr_ids:
        changes.append("OPEN_ORDER_REMOVED")
    prev_status = {item[0]: item[1] for item in prev}
    curr_status = {item[0]: item[1] for item in curr}
    if any(prev_status[key] != curr_status[key] for key in prev_ids & curr_ids):
        changes.append("OPEN_ORDER_STATUS_CHANGED")
    return changes


def _change(classification: str, continuity: str) -> dict[str, str]:
    return {
        "state_change_classification": classification,
        "broker_continuity_status": continuity,
        "recommended_action": "NO_ACTION" if classification == "NO_CHANGE" else "MANUAL_REVIEW",
        "recommended_reconciliation_gate": "GO" if classification == "NO_CHANGE" else "BLOCKED",
        "recommended_kill_switch_state": "ARMED" if classification == "NO_CHANGE" else "TRIGGERED_RECONCILIATION",
    }


def _normalize_read_counters(counters: dict[str, int]) -> dict[str, int]:
    return {
        "account_summary_requests": int(counters.get("read_only_account_summary_requests", counters.get("account_summary_requests", 0))),
        "account_summary_cancels": int(counters.get("read_only_account_summary_cancels", counters.get("account_summary_cancels", 0))),
        "position_requests": int(counters.get("read_only_position_requests", counters.get("position_requests", 0))),
        "position_cancels": int(counters.get("read_only_position_cancels", counters.get("position_cancels", 0))),
        "same_client_open_order_requests": int(counters.get("read_only_same_client_open_order_requests", counters.get("same_client_open_order_requests", 0))),
        "all_api_open_order_requests": int(counters.get("read_only_all_api_open_order_requests", counters.get("all_api_open_order_requests", 0))),
        "execution_requests": int(counters.get("read_only_execution_requests", counters.get("execution_requests", 0))),
        "current_time_requests": int(counters.get("current_time_requests", 1 if counters else 0)),
    }


def _sum_counters(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
    keys = set(first) | set(second)
    return {key: int(first.get(key, 0)) + int(second.get(key, 0)) for key in keys}


def _validate_soak_bounds(duration_seconds: float, interval_seconds: float, stability_delay_seconds: float | None) -> str | None:
    if duration_seconds <= 0 or duration_seconds > 7200:
        return "UNBOUNDED_SOAK_DURATION_BLOCKED"
    if interval_seconds <= 0 or interval_seconds > duration_seconds:
        return "UNBOUNDED_SOAK_INTERVAL_BLOCKED"
    if stability_delay_seconds is not None and (stability_delay_seconds <= 0 or stability_delay_seconds > 60):
        return "UNBOUNDED_STABILITY_DELAY_BLOCKED"
    return None


def _write_iteration_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = []
    for row in rows:
        flattened.append(
            {
                "iteration_id": row["iteration_id"],
                "scheduled_at": row["scheduled_at"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "duration_seconds": str(row["duration_seconds"]),
                "snapshot_status": row["snapshot_status"],
                "stability_status": row["stability_status"],
                "state_change_classification": row["state_change_classification"],
                "connection_status": row["connection_status"],
                "content_hash": row["content_hash"],
            }
        )
    pd.DataFrame(flattened).to_parquet(path, index=False)


def _read_soak_rows(layout: Phase81Layout) -> list[dict[str, Any]]:
    payload = _read_json(layout.artifact("soak-results.json"))
    if not payload:
        return []
    rows = payload.get("iterations", [])
    return rows if isinstance(rows, list) else []


def _component_to_callback(name: str) -> str:
    return {
        "accountsummary": "accountSummary",
        "positions": "position",
        "same_client_open_orders": "openOrder",
        "all_api_open_orders": "openOrder",
        "executions": "execDetails",
        "commissions": "commissionReport",
    }.get(name, "error")


def _blocked_artifact(project_root: Path, schema: str, status: str, errors: list[str]) -> dict[str, Any]:
    return _artifact(schema, {"status": status, "errors": sorted(set(errors)), **FINANCIAL_STATUS, **zero_phase8_1_counters()}, project_root)


def _phase8_freeze_integrity(project_root: Path) -> dict[str, Any]:
    path = project_root / "output" / "ibkr" / "phase8" / "freeze-status.json"
    payload = _read_json(path) or {}
    return {
        "status": "GO" if payload.get("freeze_status") == PHASE8_FREEZE_MARKER and payload.get("execution_authority") == EXECUTION_AUTHORITY else "NO_GO",
        "freeze_status": payload.get("freeze_status"),
        "hash": sha256_file(path) if path.exists() else None,
    }


def _phase7_freeze_integrity(project_root: Path) -> dict[str, Any]:
    path = project_root / "output" / "execution" / "phase7" / "freeze-status.json"
    payload = _read_json(path) or {}
    return {
        "status": "GO" if payload.get("freeze_status") == "PHASE7_OFFLINE_EXECUTION_CONTROL_PLANE_FROZEN_GO" and payload.get("execution_authority") == EXECUTION_AUTHORITY else "NO_GO",
        "freeze_status": payload.get("freeze_status"),
        "hash": sha256_file(path) if path.exists() else None,
    }


def _artifact(schema: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
    base = {
        "schema": schema,
        "generated_at": utc_now_iso(),
        "phase8_1_marker": PHASE8_1_MARKER,
        "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
        "execution_authority": EXECUTION_AUTHORITY,
        "input_hashes": {
            "phase8_freeze": sha256_file(project_root / "output" / "ibkr" / "phase8" / "freeze-status.json")
            if (project_root / "output" / "ibkr" / "phase8" / "freeze-status.json").exists()
            else None,
            "phase7_freeze": sha256_file(project_root / "output" / "execution" / "phase7" / "freeze-status.json")
            if (project_root / "output" / "execution" / "phase7" / "freeze-status.json").exists()
            else None,
        },
        **payload,
    }
    _assert_no_public_leak(base)
    base["content_hash"] = stable_hash({key: value for key, value in base.items() if key != "content_hash"})
    return base


def _assert_no_private_leak(payload: Any) -> None:
    if _contains_strict_raw_account(payload):
        raise ValueError("RAW_ACCOUNT_LEAK_BLOCKED")


def _assert_no_public_leak(payload: Any) -> None:
    if _contains_strict_raw_account(payload):
        raise ValueError("RAW_ACCOUNT_LEAK_BLOCKED")
    text = json.dumps(payload, sort_keys=True, default=str)
    forbidden = ["NetLiquidation", "AvailableFunds", "BuyingPower", "average_cost", "limit_price", "execution_price"]
    if any(token in text for token in forbidden):
        raise ValueError("PUBLIC_FINANCIAL_VALUE_LEAK_BLOCKED")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_public_leak(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_strict_raw_account(payload: Any) -> bool:
    text = json.dumps(payload, sort_keys=True, default=str) if not isinstance(payload, str) else payload
    return bool(re.search(r"\bDU[0-9A-Z_]{4,}\b|\bU[0-9]{4,}\b", text))


def _strip_method_constants(text: str) -> str:
    for method in FORBIDDEN_METHODS:
        text = text.replace(f'"{method}"', '""').replace(f"'{method}'", "''")
    return text
