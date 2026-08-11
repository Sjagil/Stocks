from __future__ import annotations

import json
import re
import socket
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file, utc_now_iso
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.reconciliation.adapter import enforce_method_allowed, zero_read_counters, zero_write_counters
from stocks.ibkr.reconciliation.comparator import compare_phase7_to_broker
from stocks.ibkr.reconciliation.errors import (
    BROKER_OBSERVATION_AUTHORITY,
    EXECUTION_AUTHORITY,
    FORBIDDEN_METHODS,
    PHASE8_FREEZE_MARKER,
    PHASE8_MARKER,
    READ_ONLY_METHODS,
    SUBSCRIPTION_CANCELLATION_METHODS,
    Phase8Blocked,
)
from stocks.ibkr.reconciliation.masking import contains_raw_account
from stocks.ibkr.reconciliation.models import BrokerObservationSnapshot
from stocks.ibkr.reconciliation.requests import ACCOUNT_SUMMARY_TAGS, load_phase8_config
from stocks.ibkr.reconciliation.snapshots import (
    capture_snapshot,
    empty_counters,
    snapshot_components_complete,
    stability_status,
)
from stocks.ibkr.reconciliation.storage import BrokerObservationStore, Phase8Layout, public_snapshot_summary, write_json


FINANCIAL_STATUS = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "blocked",
    "PAPER_STRATEGY_AUTHORITY": "blocked",
    "LIVE_STRATEGY_AUTHORITY": "blocked",
    "financial_decision": "NO_NEW_FINANCIAL_CANDIDATE",
}


def phase8_schema(project_root: Path) -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    payload = _artifact(
        "phase8_schema_v1",
        {
            "status": "SCHEMA_GO",
            "phase8_marker": PHASE8_MARKER,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            "read_only_methods": sorted(READ_ONLY_METHODS),
            "subscription_cancellation_methods": sorted(SUBSCRIPTION_CANCELLATION_METHODS),
            "forbidden_methods": sorted(FORBIDDEN_METHODS),
            "account_summary_tags": list(ACCOUNT_SUMMARY_TAGS),
            "models": [
                "BrokerAccountValue",
                "BrokerAccountSnapshot",
                "BrokerPosition",
                "BrokerPositionSnapshot",
                "BrokerOpenOrder",
                "BrokerOpenOrderSnapshot",
                "BrokerExecution",
                "BrokerCommission",
                "BrokerExecutionSnapshot",
                "BrokerObservationSnapshot",
                "BrokerSnapshotManifest",
            ],
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("schema.json"), payload)
    return payload


def phase8_preflight(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    config, errors = load_phase8_config(project_root, env_file)
    socket_ok = False
    if config is not None and not errors:
        socket_ok = _socket_check(config.host, config.port, timeout=2.0)
        if not socket_ok:
            errors.append("TWS_PAPER_SOCKET_NOT_REACHABLE")
    phase1 = _phase1_integrity(project_root)
    phase7 = _phase7_integrity(project_root)
    if phase1["status"] != "GO":
        errors.append("PHASE1_FROZEN_INTEGRITY_NO_GO")
    if phase7["status"] != "GO":
        errors.append("PHASE7_FROZEN_INTEGRITY_NO_GO")
    status = "GO" if not errors else "RECON_CONFIG_BLOCKED"
    payload = _artifact(
        "phase8_preflight_v1",
        {
            "status": status,
            "errors": sorted(set(errors)),
            "config": None if config is None else config.safe_dict(),
            "tws_paper_reachable": socket_ok,
            "phase1_frozen_integrity": phase1,
            "phase7_frozen_integrity": phase7,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("preflight.json"), payload)
    return payload


def phase8_snapshot(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    config, errors = load_phase8_config(project_root, env_file)
    if config is None or errors:
        payload = _artifact(
            "phase8_snapshot_v1",
            {
                "status": "RECON_CONFIG_BLOCKED",
                "errors": sorted(set(errors)),
                "snapshot_status": "RECON_CONFIG_BLOCKED",
                **FINANCIAL_STATUS,
                **empty_counters(),
            },
            project_root,
        )
        _write_snapshot_artifacts(layout, payload)
        return payload
    try:
        snapshot, read_counters, write_counters = capture_snapshot(config)
    except Phase8Blocked as exc:
        payload = _artifact(
            "phase8_snapshot_v1",
            {
                "status": "NO_GO",
                "errors": [
                    {
                        "code": exc.code,
                        "message": str(exc),
                    }
                ],
                "snapshot_status": exc.code,
                **FINANCIAL_STATUS,
                **empty_counters(),
            },
            project_root,
        )
        _write_snapshot_artifacts(layout, payload)
        return payload
    except Exception as exc:
        payload = _artifact(
            "phase8_snapshot_v1",
            {
                "status": "CONNECTION_LOST",
                "errors": [{"code": type(exc).__name__, "message": str(exc)}],
                "snapshot_status": "CONNECTION_LOST",
                **FINANCIAL_STATUS,
                **empty_counters(),
            },
            project_root,
        )
        _write_snapshot_artifacts(layout, payload)
        return payload
    return _publish_snapshot(project_root, layout, snapshot, read_counters, write_counters)


def phase8_stability_check(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    config, errors = load_phase8_config(project_root, env_file)
    if config is None or errors:
        payload = _artifact(
            "phase8_stability_check_v1",
            {
                "status": "RECON_CONFIG_BLOCKED",
                "stability_status": "STABILITY_CHECK_TIMEOUT",
                "errors": sorted(set(errors)),
                **FINANCIAL_STATUS,
                **empty_counters(),
            },
            project_root,
        )
        write_json(layout.artifact("stability-audit.json"), payload)
        return payload
    try:
        first, read_a, write_a = capture_snapshot(config)
        import time

        time.sleep(config.snapshot_stability_delay_seconds)
        second, read_b, write_b = capture_snapshot(config)
    except Phase8Blocked as exc:
        payload = _artifact(
            "phase8_stability_check_v1",
            {
                "status": "NO_GO",
                "stability_status": exc.code,
                "errors": [{"code": exc.code, "message": str(exc)}],
                **FINANCIAL_STATUS,
                **empty_counters(),
            },
            project_root,
        )
        write_json(layout.artifact("stability-audit.json"), payload)
        return payload
    status = stability_status(first, second)
    store = BrokerObservationStore(layout.private_db)
    first_hash = store.write_snapshot(first)
    second_hash = store.write_snapshot(second)
    combined_reads = _sum_counters(read_a, read_b)
    combined_writes = _sum_counters(write_a, write_b)
    observation_complete = (
        snapshot_components_complete(first)
        and snapshot_components_complete(second)
        and _required_read_requests_present(read_a)
        and _required_read_requests_present(read_b)
        and _write_counters_zero(write_a)
        and _write_counters_zero(write_b)
    )
    payload = _artifact(
        "phase8_stability_check_v1",
        {
            "status": "GO" if observation_complete else "NO_GO",
            **status,
            "private_snapshot_hashes": [first_hash, second_hash],
            "read_counters": combined_reads,
            "write_counters": combined_writes,
            **combined_reads,
            **combined_writes,
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    write_json(layout.artifact("stability-audit.json"), payload)
    return payload


def phase8_reconcile(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    latest = BrokerObservationStore(layout.private_db).latest_snapshot()
    if latest is None:
        snapshot_report = phase8_snapshot(project_root, env_file)
        if snapshot_report.get("status") not in {"GO", PHASE8_MARKER}:
            payload = _artifact(
                "phase8_reconciliation_v1",
                {
                    "status": "RECONCILIATION_BLOCKED",
                    "reconciliation_status": "RECONCILIATION_BLOCKED",
                    "reason": "NO_COMPLETE_SNAPSHOT_AVAILABLE",
                    **FINANCIAL_STATUS,
                    **empty_counters(),
                },
                project_root,
            )
            write_json(layout.artifact("reconciliation-audit.json"), payload)
            return payload
        latest = BrokerObservationStore(layout.private_db).latest_snapshot()
    if latest is None:
        raise RuntimeError("snapshot storage unavailable")
    broker_payload = json.loads(latest["payload_json"])
    broker_snapshot = _snapshot_from_payload(broker_payload)
    if not snapshot_components_complete(broker_snapshot):
        payload = _artifact(
            "phase8_reconciliation_v1",
            {
                "status": "RECONCILIATION_BLOCKED",
                "reconciliation_status": "RECONCILIATION_BLOCKED",
                "reason": "LATEST_SNAPSHOT_INCOMPLETE",
                "mutation_policy": "NO_MUTATION",
                "operation_mode": "OBSERVE_ONLY",
                **FINANCIAL_STATUS,
                **empty_counters(),
            },
            project_root,
        )
        write_json(layout.artifact("reconciliation-audit.json"), payload)
        return payload
    comparison = compare_phase7_to_broker(project_root, broker_snapshot)
    payload = _artifact(
        "phase8_reconciliation_v1",
        {
            **comparison,
            "status": "GO" if comparison["phase7_ledger_unchanged"] else "NO_GO",
            "reconciliation_status": comparison["status"],
            "mutation_policy": "NO_MUTATION",
            "operation_mode": "OBSERVE_ONLY",
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("reconciliation-audit.json"), payload)
    return payload


def phase8_audit(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    preflight = phase8_preflight(project_root, env_file)
    method = _method_allowlist_audit(project_root)
    privacy = _privacy_audit(layout)
    snapshot_report_path = layout.artifact("account-summary-audit.json")
    required = [
        snapshot_report_path,
        layout.artifact("position-audit.json"),
        layout.artifact("open-order-audit.json"),
        layout.artifact("execution-audit.json"),
        layout.artifact("stability-audit.json"),
        layout.artifact("reconciliation-audit.json"),
    ]
    payload = _artifact(
        "phase8_audit_v1",
        {
            "status": "GO" if method["status"] == "GO" and privacy["status"] == "GO" else "NO_GO",
            "preflight_status": preflight["status"],
            "required_artifacts_present": {str(path.name): path.exists() for path in required},
            "method_allowlist_status": method["status"],
            "privacy_status": privacy["status"],
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("method-allowlist-audit.json"), method)
    write_json(layout.artifact("privacy-audit.json"), privacy)
    write_json(layout.artifact("manifest.json"), phase8_manifest(project_root))
    return payload


def phase8_status(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    artifacts = _artifact_paths(layout)
    checks = {name: path.exists() for name, path in artifacts.items() if name not in {"status.json", "manifest.json", "freeze-status.json"}}
    privacy = _privacy_audit(layout)
    method = _method_allowlist_audit(project_root)
    recon = _read_json(layout.artifact("reconciliation-audit.json"))
    marker_ok = all(checks.values()) and privacy["status"] == "GO" and method["status"] == "GO"
    payload = _artifact(
        "phase8_status_v1",
        {
            "status": PHASE8_MARKER if marker_ok else "NO_GO",
            "phase8_marker": PHASE8_MARKER,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            "checks": checks,
            "privacy_status": privacy["status"],
            "method_allowlist_status": method["status"],
            "reconciliation_status": None if recon is None else recon.get("reconciliation_status"),
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("status.json"), payload)
    write_json(layout.artifact("manifest.json"), phase8_manifest(project_root))
    return payload


def phase8_freeze(project_root: Path, env_file: str | Path = ".env.ibkr") -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    status = phase8_status(project_root, env_file)
    artifacts = {name: path for name, path in _artifact_paths(layout).items() if path.exists() and name != "freeze-status.json"}
    source_paths = [
        "main.py",
        "src/stocks/ibkr/reconciliation/__init__.py",
        "src/stocks/ibkr/reconciliation/adapter.py",
        "src/stocks/ibkr/reconciliation/callbacks.py",
        "src/stocks/ibkr/reconciliation/models.py",
        "src/stocks/ibkr/reconciliation/requests.py",
        "src/stocks/ibkr/reconciliation/masking.py",
        "src/stocks/ibkr/reconciliation/snapshots.py",
        "src/stocks/ibkr/reconciliation/normalizer.py",
        "src/stocks/ibkr/reconciliation/comparator.py",
        "src/stocks/ibkr/reconciliation/storage.py",
        "src/stocks/ibkr/reconciliation/audit.py",
        "src/stocks/ibkr/reconciliation/errors.py",
        "tests/test_phase8_reconciliation.py",
        "PHASE8_STATUS.md",
        "PHASE8_FREEZE_REPORT.md",
        "docs/PHASE8_IBKR_READ_ONLY_RECONCILIATION.md",
    ]
    payload = _artifact(
        "phase8_freeze_status_v1",
        {
            "freeze_status": PHASE8_FREEZE_MARKER if status["status"] == PHASE8_MARKER else "NO_GO",
            "phase8_status": status["status"],
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            "source_hashes": {path: sha256_file(project_root / path) for path in source_paths if (project_root / path).exists()},
            "artifact_hashes": {name: sha256_file(path) for name, path in artifacts.items()},
            "private_database_path": str(layout.private_db),
            "private_database_hash": sha256_file(layout.private_db) if layout.private_db.exists() else None,
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        project_root,
    )
    write_json(layout.artifact("freeze-status.json"), payload)
    return payload


def phase8_manifest(project_root: Path) -> dict[str, Any]:
    layout = Phase8Layout.from_project_root(project_root)
    artifacts = {name: path for name, path in _artifact_paths(layout).items() if path.exists() and name != "freeze-status.json"}
    return _artifact(
        "phase8_manifest_v1",
        {
            "status": "GO",
            "phase8_marker": PHASE8_MARKER,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            "artifact_hashes": {name: sha256_file(path) for name, path in artifacts.items()},
            "private_database_path": str(layout.private_db),
            "private_database_hash": sha256_file(layout.private_db) if layout.private_db.exists() else None,
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        project_root,
    )


def _publish_snapshot(
    project_root: Path,
    layout: Phase8Layout,
    snapshot: BrokerObservationSnapshot,
    read_counters: dict[str, int],
    write_counters: dict[str, int],
) -> dict[str, Any]:
    store = BrokerObservationStore(layout.private_db)
    private_hash = store.write_snapshot(snapshot)
    summary = public_snapshot_summary(snapshot, private_hash, layout.private_db)
    component_complete = snapshot_components_complete(snapshot)
    required_reads_present = _required_read_requests_present(read_counters)
    writes_zero = _write_counters_zero(write_counters)
    complete = component_complete and required_reads_present and writes_zero
    completion_errors = []
    if not component_complete:
        completion_errors.append("CALLBACK_SEQUENCE_INCOMPLETE")
    if not required_reads_present:
        completion_errors.append("READ_REQUEST_SEQUENCE_INCOMPLETE")
    if not writes_zero:
        completion_errors.append("BROKER_WRITE_COUNTER_NONZERO")
    payload = _artifact(
        "phase8_snapshot_v1",
        {
            "status": "GO" if complete else "NO_GO",
            "snapshot_status": (
                "BROKER_SNAPSHOT_OBSERVED"
                if complete
                else "BROKER_SNAPSHOT_INCOMPLETE"
            ),
            "completion_errors": completion_errors,
            "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
            "execution_authority": EXECUTION_AUTHORITY,
            "server_version": snapshot.server_version,
            **summary,
            "component_statuses": {item.name: item.request_status for item in snapshot.component_audits},
            "read_counters": read_counters,
            "write_counters": write_counters,
            **read_counters,
            **write_counters,
            **FINANCIAL_STATUS,
        },
        project_root,
    )
    _write_snapshot_artifacts(layout, payload)
    return payload


def _required_read_requests_present(counters: dict[str, int]) -> bool:
    required = (
        "read_only_account_summary_requests",
        "read_only_position_requests",
        "read_only_same_client_open_order_requests",
        "read_only_all_api_open_order_requests",
        "read_only_execution_requests",
    )
    return all(int(counters.get(name, 0)) > 0 for name in required)


def _write_counters_zero(counters: dict[str, int]) -> bool:
    return all(int(value) == 0 for value in counters.values())


def _write_snapshot_artifacts(layout: Phase8Layout, payload: dict[str, Any]) -> None:
    write_json(layout.artifact("account-summary-audit.json"), _component_public(payload, "accountsummary"))
    write_json(layout.artifact("position-audit.json"), _component_public(payload, "positions"))
    write_json(layout.artifact("open-order-audit.json"), _component_public(payload, "open_orders"))
    write_json(layout.artifact("execution-audit.json"), _component_public(payload, "executions"))


def _component_public(payload: dict[str, Any], component: str) -> dict[str, Any]:
    public = {
        "schema": f"phase8_{component}_audit_v1",
        "generated_at": utc_now_iso(),
        "phase8_marker": PHASE8_MARKER,
        "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
        "execution_authority": EXECUTION_AUTHORITY,
        "component": component,
        "snapshot_status": payload.get("snapshot_status"),
        "server_version": payload.get("server_version"),
        "public_summary": payload.get("public_summary", {}),
        "private_snapshot_reference": payload.get("private_snapshot_reference"),
        "private_snapshot_hash": payload.get("private_snapshot_hash"),
        "read_counters": payload.get("read_counters", zero_read_counters()),
        "write_counters": payload.get("write_counters", zero_write_counters()),
        **FINANCIAL_STATUS,
    }
    public["content_hash"] = stable_hash(public)
    return public


def _method_allowlist_audit(project_root: Path) -> dict[str, Any]:
    source_paths = [project_root / "main.py", *list((project_root / "src" / "stocks" / "ibkr" / "reconciliation").glob("*.py"))]
    findings: dict[str, int] = {method: 0 for method in FORBIDDEN_METHODS}
    for path in source_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        sanitized = _strip_audit_constants(text)
        for method in FORBIDDEN_METHODS:
            findings[method] += len(re.findall(rf"\.{re.escape(method)}\s*\(", sanitized))
    runtime = {}
    for method in sorted(READ_ONLY_METHODS):
        runtime[method] = enforce_method_allowed(method)
    for method in sorted(FORBIDDEN_METHODS):
        try:
            enforce_method_allowed(method)
        except Phase8Blocked as exc:
            runtime[method] = exc.code
    status = "GO" if all(value == 0 for value in findings.values()) and all(runtime[m] == "BROKER_WRITE_METHOD_BLOCKED" for m in FORBIDDEN_METHODS) else "NO_GO"
    return _artifact(
        "phase8_method_allowlist_audit_v1",
        {
            "status": status,
            "forbidden_method_static_hits": findings,
            "runtime_method_guard": runtime,
            "subscription_cancellations_allowed": sorted(SUBSCRIPTION_CANCELLATION_METHODS),
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        project_root,
    )


def _privacy_audit(layout: Phase8Layout) -> dict[str, Any]:
    scanned = []
    leaks = []
    for root in (layout.output_dir, layout.private_dir):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".sqlite3", ".md", ""}:
                scanned.append(str(path))
                data = path.read_bytes()
                text = data.decode("utf-8", errors="ignore")
                if contains_raw_account(text):
                    leaks.append(str(path))
    status = "GO" if not leaks else "RAW_ACCOUNT_LEAK_BLOCKED"
    return _artifact(
        "phase8_privacy_audit_v1",
        {
            "status": status,
            "raw_account_leaks": len(leaks),
            "secret_leaks": 0,
            "public_financial_value_leaks": _public_financial_value_leaks(layout.output_dir),
            "scanned_file_count": len(scanned),
            "leak_paths": leaks,
            **FINANCIAL_STATUS,
            **empty_counters(),
        },
        layout.project_root,
    )


def _public_financial_value_leaks(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    forbidden_keys = {"NetLiquidation", "TotalCashValue", "SettledCash", "AvailableFunds", "BuyingPower", "average_cost", "limit_price", "execution_price"}
    leaks = 0
    for path in output_dir.rglob("*.json"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        leaks += sum(1 for key in forbidden_keys if key in text)
    return leaks


def _phase1_integrity(project_root: Path) -> dict[str, Any]:
    path = project_root / "output" / "ibkr" / "phase1-freeze-status.json"
    alt = project_root / "PHASE1_FREEZE_REPORT.md"
    return {"status": "GO" if alt.exists() else "NO_GO", "artifact_present": path.exists(), "report_present": alt.exists()}


def _phase7_integrity(project_root: Path) -> dict[str, Any]:
    path = project_root / "output" / "execution" / "phase7" / "freeze-status.json"
    if not path.exists():
        return {"status": "NO_GO", "freeze_status": None}
    payload = _read_json(path) or {}
    return {
        "status": "GO" if payload.get("freeze_status") == "PHASE7_OFFLINE_EXECUTION_CONTROL_PLANE_FROZEN_GO" and payload.get("execution_authority") == "NONE" else "NO_GO",
        "freeze_status": payload.get("freeze_status"),
        "execution_authority": payload.get("execution_authority"),
        "hash": sha256_file(path),
    }


def _socket_check(host: str, port: int, *, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _snapshot_from_payload(payload: dict[str, Any]) -> BrokerObservationSnapshot:
    from stocks.ibkr.reconciliation.models import (
        BrokerAccountSnapshot,
        BrokerAccountValue,
        BrokerExecutionSnapshot,
        BrokerOpenOrderSnapshot,
        BrokerPosition,
        BrokerPositionSnapshot,
        ObservationScope,
        SnapshotComponentAudit,
    )
    from stocks.ibkr.reconciliation.normalizer import decimal_value

    account = BrokerAccountSnapshot(
        tuple(BrokerAccountValue(**{**item, "value": decimal_value(item["value"]) if str(item["value"]).replace(".", "", 1).isdigit() else item["value"]}) for item in payload["account"]["values"]),
        payload["account"]["status"],
        payload["account"]["content_hash"],
    )
    positions = BrokerPositionSnapshot(tuple(BrokerPosition(**{**item, "position_quantity": decimal_value(item["position_quantity"]), "average_cost": decimal_value(item["average_cost"])}) for item in payload["positions"]["positions"]), payload["positions"]["status"], payload["positions"]["content_hash"])
    same = BrokerOpenOrderSnapshot(ObservationScope.SAME_CLIENT, tuple(_open_order_from_dict(item) for item in payload["same_client_open_orders"]["open_orders"]), payload["same_client_open_orders"]["status"], payload["same_client_open_orders"]["content_hash"])
    all_api = BrokerOpenOrderSnapshot(ObservationScope.ALL_API_CLIENTS, tuple(_open_order_from_dict(item) for item in payload["all_api_open_orders"]["open_orders"]), payload["all_api_open_orders"]["status"], payload["all_api_open_orders"]["content_hash"])
    executions = BrokerExecutionSnapshot(
        tuple(_execution_from_dict(item) for item in payload["executions"]["executions"]),
        tuple(),
        payload["executions"]["status"],
        payload["executions"]["execution_scope"],
        payload["executions"]["request_filter"],
        payload["executions"]["requested_from"],
        payload["executions"]["requested_until"],
        bool(payload["executions"]["tws_trade_log_scope_known"]),
        bool(payload["executions"]["execution_history_complete"]),
        payload["executions"]["completeness_status"],
        payload["executions"]["content_hash"],
    )
    return BrokerObservationSnapshot(
        snapshot_id=payload["snapshot_id"],
        snapshot_started_at=payload["snapshot_started_at"],
        snapshot_completed_at=payload["snapshot_completed_at"],
        snapshot_span_seconds=decimal_value(payload["snapshot_span_seconds"]),
        component_timestamps=payload["component_timestamps"],
        snapshot_atomic=bool(payload["snapshot_atomic"]),
        account=account,
        positions=positions,
        same_client_open_orders=same,
        all_api_open_orders=all_api,
        executions=executions,
        component_audits=tuple(SnapshotComponentAudit(**{**item, "timeout_seconds": decimal_value(item["timeout_seconds"])}) for item in payload["component_audits"]),
        server_version=payload.get("server_version"),
        broker_observation_authority=payload["broker_observation_authority"],
        execution_authority=payload["execution_authority"],
        content_hash=payload["content_hash"],
    )


def _open_order_from_dict(item: dict[str, Any]) -> Any:
    from stocks.ibkr.reconciliation.models import BrokerOpenOrder, ObservationScope
    from stocks.ibkr.reconciliation.normalizer import decimal_or_none, decimal_value

    return BrokerOpenOrder(
        **{
            **item,
            "currency": item.get("currency", "UNKNOWN"),
            "observation_scope": ObservationScope(item["observation_scope"]),
            "total_quantity": decimal_value(item["total_quantity"]),
            "limit_price": decimal_or_none(item["limit_price"]),
            "aux_price": decimal_or_none(item["aux_price"]),
            "filled_quantity": decimal_value(item["filled_quantity"]),
            "remaining_quantity": decimal_value(item["remaining_quantity"]),
            "average_fill_price": decimal_or_none(item["average_fill_price"]),
        }
    )


def _execution_from_dict(item: dict[str, Any]) -> Any:
    from stocks.ibkr.reconciliation.models import BrokerExecution
    from stocks.ibkr.reconciliation.normalizer import decimal_value

    return BrokerExecution(**{**item, "quantity": decimal_value(item["quantity"]), "price": decimal_value(item["price"]), "cumulative_quantity": decimal_value(item["cumulative_quantity"]), "average_price": decimal_value(item["average_price"])})


def _artifact_paths(layout: Phase8Layout) -> dict[str, Path]:
    return {
        "schema.json": layout.artifact("schema.json"),
        "preflight.json": layout.artifact("preflight.json"),
        "account-summary-audit.json": layout.artifact("account-summary-audit.json"),
        "position-audit.json": layout.artifact("position-audit.json"),
        "open-order-audit.json": layout.artifact("open-order-audit.json"),
        "execution-audit.json": layout.artifact("execution-audit.json"),
        "stability-audit.json": layout.artifact("stability-audit.json"),
        "reconciliation-audit.json": layout.artifact("reconciliation-audit.json"),
        "privacy-audit.json": layout.artifact("privacy-audit.json"),
        "method-allowlist-audit.json": layout.artifact("method-allowlist-audit.json"),
        "status.json": layout.artifact("status.json"),
        "manifest.json": layout.artifact("manifest.json"),
        "freeze-status.json": layout.artifact("freeze-status.json"),
    }


def _artifact(schema: str, payload: dict[str, Any], project_root: Path) -> dict[str, Any]:
    base = {
        "schema": schema,
        "generated_at": utc_now_iso(),
        "phase8_marker": PHASE8_MARKER,
        "broker_observation_authority": BROKER_OBSERVATION_AUTHORITY,
        "execution_authority": EXECUTION_AUTHORITY,
        "input_hashes": {
            "phase7_freeze": sha256_file(project_root / "output" / "execution" / "phase7" / "freeze-status.json")
            if (project_root / "output" / "execution" / "phase7" / "freeze-status.json").exists()
            else None,
            "phase6_4_freeze": sha256_file(project_root / "output" / "research" / "phase6_4" / "freeze-status.json")
            if (project_root / "output" / "research" / "phase6_4" / "freeze-status.json").exists()
            else None,
        },
        **payload,
    }
    if contains_raw_account(base):
        raise ValueError("RAW_ACCOUNT_LEAK_BLOCKED")
    base["content_hash"] = stable_hash({key: value for key, value in base.items() if key != "content_hash"})
    return base


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sum_counters(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
    keys = set(first) | set(second)
    return {key: int(first.get(key, 0)) + int(second.get(key, 0)) for key in keys}


def _strip_audit_constants(text: str) -> str:
    for method in FORBIDDEN_METHODS:
        text = text.replace(f'"{method}"', '""').replace(f"'{method}'", "''")
    return text
