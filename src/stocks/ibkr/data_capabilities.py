from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from stocks.execution.idempotency import stable_hash


CAPABILITIES = (
    "delayed_quotes",
    "realtime_top_of_book",
    "tick_by_tick_trades",
    "historical_bars",
    "market_depth_l2",
    "exchange_attribution",
    "halt_status_events",
    "realtime_volume",
)


ENTITLEMENT_ERROR_CODES = {10089, 10189}


def capability_schema() -> dict[str, Any]:
    return {
        "schema": "ibkr_market_data_capability_schema_v1",
        "status": "GO",
        "capabilities": list(CAPABILITIES),
        "statuses": [
            "AVAILABLE",
            "AVAILABLE_PARTIAL",
            "UNAVAILABLE",
            "UNAVAILABLE_ENTITLEMENT",
            "UNPROVEN",
            "STALE",
        ],
        "provenance_fields": [
            "asset_reference",
            "capability",
            "status",
            "source",
            "market_data_type",
            "observed_at",
            "evidence",
        ],
        "fallback_policy": "NO_SILENT_REALTIME_TO_DELAYED_FALLBACK",
        "strategy_gate_policy": "ALL_REQUIRED_CAPABILITIES_MUST_BE_AVAILABLE",
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
    }


def build_capability_matrix(project_root: Path) -> dict[str, Any]:
    observation_path = (
        project_root
        / "output"
        / "market_context"
        / "realtime-equity-collection.json"
    )
    observation = _read_json(observation_path)
    observed_at = observation.get("generated_at")
    market_data_type = observation.get("market_data_type")
    errors = {
        int(code): int(count)
        for code, count in dict(observation.get("error_code_counts", {})).items()
        if str(code).isdigit()
    }
    entitlement_blocked = any(
        errors.get(code, 0) > 0 for code in ENTITLEMENT_ERROR_CODES
    )
    quote_rows = int(observation.get("quote_row_count", 0) or 0)
    trade_rows = int(observation.get("trade_row_count", 0) or 0)
    depth_rows = int(observation.get("depth_row_count", 0) or 0)
    observed_symbols = int(observation.get("observed_symbol_count", 0) or 0)
    depth_symbols = int(
        observation.get("depth_observed_symbol_count", 0) or 0
    )
    historical_files = sorted(
        (project_root / "data" / "bars").rglob("*.parquet")
    ) if (project_root / "data" / "bars").exists() else []
    historical_assets = sorted(
        {
            stable_hash(path.parent.parent.parent.parent.name)[:20]
            for path in historical_files
        }
    )
    requested_assets = [
        str(value)
        for value in observation.get("requested_symbols_masked", [])
        if value
    ]
    assets = requested_assets or historical_assets or ["UNSCOPED_STK"]

    statuses = {
        "delayed_quotes": "UNPROVEN",
        "realtime_top_of_book": (
            "AVAILABLE"
            if quote_rows > 0 and observed_symbols > 0 and not entitlement_blocked
            else "AVAILABLE_PARTIAL"
            if quote_rows > 0 and not entitlement_blocked
            else "UNAVAILABLE_ENTITLEMENT"
            if entitlement_blocked
            else "UNPROVEN"
        ),
        "tick_by_tick_trades": (
            "AVAILABLE"
            if trade_rows > 0
            else "UNAVAILABLE_ENTITLEMENT"
            if entitlement_blocked
            else "UNPROVEN"
        ),
        "historical_bars": "AVAILABLE" if historical_files else "UNPROVEN",
        "market_depth_l2": (
            "AVAILABLE"
            if depth_rows > 0 and depth_symbols > 0
            else "UNAVAILABLE_ENTITLEMENT"
            if entitlement_blocked
            else "UNPROVEN"
        ),
        "exchange_attribution": (
            "AVAILABLE_PARTIAL" if trade_rows > 0 else "UNPROVEN"
        ),
        "halt_status_events": "UNPROVEN",
        "realtime_volume": (
            "AVAILABLE" if trade_rows > 0 else "UNPROVEN"
        ),
    }
    rows = []
    for asset in assets:
        for capability in CAPABILITIES:
            status = statuses[capability]
            rows.append(
                {
                    "asset_reference": asset,
                    "security_type": "STK",
                    "capability": capability,
                    "status": status,
                    "source": (
                        "IBKR_HISTORICAL_CACHE"
                        if capability == "historical_bars"
                        else "IBKR_BOUNDED_REALTIME_OBSERVATION"
                    ),
                    "market_data_type": (
                        None
                        if capability == "historical_bars"
                        else market_data_type
                    ),
                    "observed_at": (
                        _latest_mtime(historical_files)
                        if capability == "historical_bars"
                        else observed_at
                    ),
                    "evidence": _evidence(
                        capability,
                        quote_rows=quote_rows,
                        trade_rows=trade_rows,
                        depth_rows=depth_rows,
                        historical_file_count=len(historical_files),
                        entitlement_errors=errors,
                    ),
                    "silent_fallback_used": False,
                }
            )
    missing_subscriptions = []
    if statuses["realtime_top_of_book"] == "UNAVAILABLE_ENTITLEMENT":
        missing_subscriptions.append(
            "US_STOCK_REALTIME_TOP_OF_BOOK_FOR_REQUESTED_EXCHANGES"
        )
    if statuses["tick_by_tick_trades"] == "UNAVAILABLE_ENTITLEMENT":
        missing_subscriptions.append(
            "US_STOCK_TICK_BY_TICK_AND_UNDERLYING_REALTIME_ENTITLEMENT"
        )
    if statuses["market_depth_l2"] == "UNAVAILABLE_ENTITLEMENT":
        missing_subscriptions.append(
            "US_STOCK_MARKET_DEPTH_FOR_REQUESTED_EXCHANGES"
        )
    report = {
        "schema": "ibkr_market_data_capability_matrix_v1",
        "status": "GO_DEGRADED" if any(
            value not in {"AVAILABLE", "AVAILABLE_PARTIAL"}
            for value in statuses.values()
        ) else "GO",
        "generated_at": _now(),
        "asset_count": len(assets),
        "row_count": len(rows),
        "summary": statuses,
        "rows": rows,
        "source_observation": str(observation_path),
        "source_observation_status": observation.get("status", "MISSING"),
        "historical_bar_file_count": len(historical_files),
        "entitlement_error_codes": {
            str(code): count
            for code, count in errors.items()
            if code in ENTITLEMENT_ERROR_CODES
        },
        "missing_subscription_classes": missing_subscriptions,
        "subscriptions_automatically_purchased": False,
        "realtime_quality_claimed": statuses["realtime_top_of_book"] == "AVAILABLE",
        "tape_features_allowed": statuses["tick_by_tick_trades"] == "AVAILABLE",
        "depth_features_allowed": statuses["market_depth_l2"] == "AVAILABLE",
        "bar_proxy_labeled_as_orderflow": False,
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }
    report["content_hash"] = stable_hash(
        {key: value for key, value in report.items() if key != "content_hash"}
    )
    path = (
        project_root
        / "output"
        / "ibkr"
        / "data-capabilities"
        / "capability-matrix.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def strategy_capability_gate(
    matrix: Mapping[str, Any],
    required_capabilities: Iterable[str],
    *,
    asset_reference: str | None = None,
) -> dict[str, Any]:
    required = sorted(set(required_capabilities))
    unknown = sorted(set(required) - set(CAPABILITIES))
    rows = [
        row
        for row in matrix.get("rows", [])
        if not asset_reference or row.get("asset_reference") == asset_reference
    ]
    available = {
        str(row.get("capability"))
        for row in rows
        if row.get("status") == "AVAILABLE"
    }
    missing = sorted(set(required) - available)
    blockers = [f"UNKNOWN_CAPABILITY:{item}" for item in unknown]
    blockers.extend(f"CAPABILITY_NOT_AVAILABLE:{item}" for item in missing)
    return {
        "schema": "ibkr_strategy_data_capability_gate_v1",
        "status": "GO" if not blockers else "NO_GO",
        "asset_reference": asset_reference,
        "required_capabilities": required,
        "available_capabilities": sorted(available),
        "blockers": blockers,
        "silent_fallback_used": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }


def _evidence(
    capability: str,
    *,
    quote_rows: int,
    trade_rows: int,
    depth_rows: int,
    historical_file_count: int,
    entitlement_errors: Mapping[int, int],
) -> dict[str, Any]:
    values: dict[str, dict[str, Any]] = {
        "delayed_quotes": {"explicit_delayed_probe_completed": False},
        "realtime_top_of_book": {"quote_rows": quote_rows},
        "tick_by_tick_trades": {"trade_rows": trade_rows},
        "historical_bars": {"parquet_file_count": historical_file_count},
        "market_depth_l2": {"depth_rows": depth_rows},
        "exchange_attribution": {"attributed_trade_rows": 0},
        "halt_status_events": {"halt_event_rows": 0},
        "realtime_volume": {"observed_trade_rows": trade_rows},
    }
    return {
        **values[capability],
        "entitlement_error_codes": {
            str(code): count
            for code, count in entitlement_errors.items()
            if code in ENTITLEMENT_ERROR_CODES
        },
    }


def _latest_mtime(paths: list[Path]) -> str | None:
    if not paths:
        return None
    timestamp = max(path.stat().st_mtime for path in paths)
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CAPABILITIES",
    "build_capability_matrix",
    "capability_schema",
    "strategy_capability_gate",
]
