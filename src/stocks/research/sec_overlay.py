from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from stocks.execution.idempotency import stable_hash


MAX_SEC_OVERLAY_POINTS = 4.0


def sec_intelligence_status(project_root: Path) -> dict[str, Any]:
    module = _load_engine(project_root)
    db_path = _database_path(project_root)
    connection = module.connect_database(db_path)
    try:
        module.ensure_schema(connection)
        filing_count = _count(connection, "sec_intel_filings")
        event_count = _count(connection, "sec_intel_events")
        symbols = int(
            connection.execute(
                "SELECT COUNT(DISTINCT symbol) FROM sec_intel_events "
                "WHERE symbol IS NOT NULL AND symbol <> ''"
            ).fetchone()[0]
        )
        relevant_managers = int(
            connection.execute(
                "SELECT COUNT(DISTINCT "
                "json_extract(payload_json, '$.manager_cik')) "
                "FROM sec_intel_events "
                "WHERE event_type='13f_position_snapshot' "
                "AND COALESCE(json_extract(payload_json, '$.put_call'), '')='' "
                "AND symbol IS NOT NULL AND symbol <> ''"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    metadata = _read_json(
        project_root / "output" / "ibkr" / "phase11_3" / "sec-events.json"
    )
    status = "GO" if event_count > 0 else "DEGRADED"
    blockers = [] if event_count > 0 else [
        "STRUCTURED_SEC_EVENT_STORE_EMPTY",
        "RAW_FILING_CONTENT_INGEST_REQUIRED",
    ]
    report = {
        "schema": "sec_ranking_overlay_status_v1",
        "status": status,
        "generated_at": _now(),
        "private_database": str(db_path),
        "structured_filing_count": filing_count,
        "structured_event_count": event_count,
        "structured_symbol_count": symbols,
        "relevant_13f_manager_count": relevant_managers,
        "descriptive_manager_rule": (
            "EQUITY_13F_EVENTS_WITH_RESOLVED_SYMBOL_ONLY"
        ),
        "metadata_event_count": int(metadata.get("event_count", 0) or 0),
        "metadata_is_not_substantive_event_content": True,
        "supported_forms": [
            "FORM_4_5",
            "SCHEDULE_13D_G",
            "FORM_13F",
            "FORM_144",
            "FORM_8_K",
        ],
        "max_overlay_points": MAX_SEC_OVERLAY_POINTS,
        "authority": "RANKING_OVERLAY_ONLY",
        "standalone_entry_allowed": False,
        "delayed_context_only": True,
        "blockers": blockers,
        "execution_authority": "NONE",
        "broker_calls": 0,
    }
    report["content_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    _write_public(project_root, "status.json", report)
    return report


def sec_overlay_for_signal(
    project_root: Path,
    *,
    symbol: str,
    as_of: str,
    base_score: float,
    base_signal_authorized: bool,
) -> dict[str, Any]:
    module = _load_engine(project_root)
    db_path = _database_path(project_root)
    connection = module.connect_database(db_path)
    try:
        module.ensure_schema(connection)
        result = _overlay_from_connection(
            connection,
            module,
            symbol=symbol,
            as_of=as_of,
            base_score=base_score,
            base_signal_authorized=base_signal_authorized,
        )
    finally:
        connection.close()
    report = {
        "schema": "sec_signal_ranking_overlay_v1",
        "status": result["status"],
        "generated_at": _now(),
        "symbol": symbol.strip().upper(),
        "as_of": as_of,
        "features": result["features"],
        "overlay": result["overlay"],
        "max_overlay_points": MAX_SEC_OVERLAY_POINTS,
        "authority": "RANKING_OVERLAY_ONLY",
        "standalone_entry_allowed": False,
        "delayed_context_only": True,
        "blockers": result["blockers"],
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    _write_public(project_root, "latest-overlay.json", report)
    return report


def sec_overlays_for_signals(
    project_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute causal SEC ranking context in one bounded database session."""
    if not rows:
        return {}
    module = _load_engine(project_root)
    connection = module.connect_database(_database_path(project_root))
    reports: dict[str, dict[str, Any]] = {}
    try:
        module.ensure_schema(connection)
        for row in rows:
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            result = _overlay_from_connection(
                connection,
                module,
                symbol=symbol,
                as_of=str(row["as_of"]),
                base_score=float(row.get("base_score") or 0.0),
                base_signal_authorized=bool(
                    row.get("base_signal_authorized")
                ),
            )
            reports[symbol] = {
                "status": result["status"],
                "as_of": str(row["as_of"]),
                "causal_event_count": int(
                    result["features"].get("causal_event_count", 0)
                ),
                "sec_intelligence_score": float(
                    result["features"].get("sec_intelligence_score", 0.0)
                ),
                "overlay": result["overlay"],
                "authority": "RANKING_OVERLAY_ONLY",
                "standalone_entry_allowed": False,
                "delayed_context_only": True,
            }
    finally:
        connection.close()
    return reports


def _overlay_from_connection(
    connection: sqlite3.Connection,
    module: ModuleType,
    *,
    symbol: str,
    as_of: str,
    base_score: float,
    base_signal_authorized: bool,
) -> dict[str, Any]:
    clean_symbol = symbol.strip().upper()
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM sec_intel_events "
            "WHERE symbol=? AND accepted_at<=?",
            (clean_symbol, as_of),
        ).fetchone()[0]
    )
    if count:
        features = module.compute_symbol_features(
            connection, clean_symbol, as_of
        )
        status = "GO"
        blockers: list[str] = []
    else:
        features = {
            "symbol": clean_symbol,
            "as_of": as_of,
            "causal_event_count": 0,
            "sec_intelligence_score": 0.0,
            "authority": "RANKING_OVERLAY_ONLY",
            "standalone_entry_allowed": False,
        }
        status = "DEGRADED_NO_CAUSAL_SEC_EVENTS"
        blockers = ["NO_STRUCTURED_CAUSAL_SEC_EVENTS_FOR_SYMBOL"]
    overlay = module.apply_ranking_overlay(
        base_score,
        features,
        base_signal_authorized=base_signal_authorized,
        max_abs_points=MAX_SEC_OVERLAY_POINTS,
    )
    return {
        "status": status,
        "features": features,
        "overlay": overlay,
        "blockers": blockers,
    }


def sec_intelligence_audit(project_root: Path) -> dict[str, Any]:
    module = _load_engine(project_root)
    self_test = module._self_test()
    status = sec_intelligence_status(project_root)
    ablation = {
        "schema": "sec_overlay_ablation_status_v1",
        "status": "NOT_RUN" if status["structured_event_count"] == 0 else "PENDING",
        "required_horizons": ["5d", "20d", "60d"],
        "required_metrics": [
            "forward_return",
            "hit_rate",
            "profit_factor",
            "Sharpe",
            "maximum_drawdown",
            "turnover",
            "false_positive_rate",
        ],
        "promotion_allowed": False,
        "execution_authority": "NONE",
    }
    _write_public(project_root, "ablation-status.json", ablation)
    report = {
        "schema": "sec_ranking_overlay_audit_v1",
        "status": "GO" if self_test.get("status") == "PASS" else "NO_GO",
        "parser_self_test": self_test.get("status"),
        "structured_data_status": status["status"],
        "ablation_status": ablation["status"],
        "overlay_bounded": MAX_SEC_OVERLAY_POINTS == 4.0,
        "standalone_entry_allowed": False,
        "authority": "RANKING_OVERLAY_ONLY",
        "execution_authority": "NONE",
    }
    _write_public(project_root, "audit.json", report)
    return report


def _load_engine(project_root: Path) -> ModuleType:
    path = project_root / "sec_ownership_and_event_intelligence_v1.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("stocks_sec_intelligence_v1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SEC intelligence engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _database_path(project_root: Path) -> Path:
    path = (
        project_root
        / "data"
        / "sec_intelligence"
        / "sec_intelligence.sqlite3"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _write_public(project_root: Path, name: str, payload: dict[str, Any]) -> None:
    path = project_root / "output" / "research" / "sec_intelligence" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "MAX_SEC_OVERLAY_POINTS",
    "sec_intelligence_audit",
    "sec_intelligence_status",
    "sec_overlay_for_signal",
    "sec_overlays_for_signals",
]
