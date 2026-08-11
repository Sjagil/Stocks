from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from stocks.execution.idempotency import stable_hash


REGISTRY_PATH = Path("config/portfolio/strategy_authority_registry_v1.json")
LIVE_STATUS = "LIVE_AUTHORIZED_LEVEL_ONE"
SEMANTIC_FIELDS = (
    "strategy_id",
    "version",
    "source_hash",
    "parameter_hash",
    "qualification_hash",
    "allowed_timeframes",
    "allowed_asset_classes",
    "allowed_symbols",
)
OPERATIONAL_FIELDS = (
    "strategy_id",
    "version",
    "source_hash",
    "parameter_hash",
    "qualification_hash",
    "training_data_end",
    "allowed_timeframes",
    "allowed_asset_classes",
    "allowed_symbols",
    "maximum_position_weight",
    "canary_notional_hard_cap_eur",
    "primary_sizing_authority",
    "maximum_daily_orders",
    "maximum_daily_loss_eur",
    "status",
)


def operational_strategy_allowlist_hash(allowlist: dict[str, Any]) -> str:
    strategies = [
        {key: row.get(key) for key in OPERATIONAL_FIELDS}
        for row in allowlist.get("strategies", [])
        if isinstance(row, dict)
    ]
    strategies.sort(key=lambda row: str(row.get("strategy_id", "")))
    return stable_hash(
        {
            "schema": allowlist.get("schema"),
            "status": allowlist.get("status"),
            "qualification_hash": allowlist.get("qualification_hash"),
            "strategies": strategies,
        }
    )


def load_strategy_authority_registry(
    project_root: Path,
    *,
    allowlist: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load one canonical exact strategy registry and verify the live boundary."""

    configured = _read_json(project_root / REGISTRY_PATH)
    current = allowlist or _read_json(
        project_root / "output/ibkr/live/strategy-allowlist.json"
    )
    blockers: list[str] = []
    rows = [
        row
        for row in configured.get("strategies", [])
        if isinstance(row, dict)
    ]
    ids = [str(row.get("strategy_id") or "") for row in rows]
    if configured.get("schema") != "portfolio_strategy_authority_registry_v1":
        blockers.append("STRATEGY_AUTHORITY_REGISTRY_SCHEMA_INVALID")
    if not rows or any(not value for value in ids) or len(ids) != len(set(ids)):
        blockers.append("EXACT_UNIQUE_STRATEGY_IDS_REQUIRED")
    actual_hash = (
        operational_strategy_allowlist_hash(current) if current else None
    )
    expected_hash = configured.get("operational_allowlist_hash")
    if actual_hash != expected_hash:
        blockers.append("FROZEN_OPERATIONAL_ALLOWLIST_HASH_MISMATCH")
    if current.get("qualification_hash") != configured.get("qualification_hash"):
        blockers.append("FROZEN_QUALIFICATION_HASH_MISMATCH")

    current_by_id = {
        str(row.get("strategy_id") or ""): row
        for row in current.get("strategies", [])
        if isinstance(row, dict)
    }
    for row in rows:
        strategy_id = str(row.get("strategy_id") or "")
        live = current_by_id.get(strategy_id)
        if live is None:
            blockers.append(f"FROZEN_STRATEGY_MISSING:{strategy_id}")
            continue
        for field in SEMANTIC_FIELDS:
            if _normalized(row.get(field)) != _normalized(live.get(field)):
                blockers.append(f"FROZEN_STRATEGY_FIELD_MISMATCH:{strategy_id}:{field}")
        if live.get("status") != "PIT_LIVE_ALLOWLISTED":
            blockers.append(f"STRATEGY_NOT_LIVE_ALLOWLISTED:{strategy_id}")

    report: dict[str, Any] = {
        "schema": "portfolio_strategy_authority_status_v1",
        "status": "GO" if not blockers else "NO_GO",
        "authority_mode": configured.get("authority_mode"),
        "expected_operational_allowlist_hash": expected_hash,
        "actual_operational_allowlist_hash": actual_hash,
        "qualification_hash": configured.get("qualification_hash"),
        "strategy_count": len(rows),
        "strategies": rows,
        "risk_policy": configured.get("risk_policy", {}),
        "blockers": sorted(set(blockers)),
        "automatic_capital_promotion": False,
        "capability_34_added": False,
    }
    report["content_hash"] = stable_hash(report)
    return report


def bind_exact_strategy(
    candidate: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Bind only an exact contributing strategy; never infer or promote one."""

    symbol = str(candidate.get("symbol") or candidate.get("ticker") or "").upper()
    asset_class = str(candidate.get("asset_class") or "").upper()
    explicit = str(candidate.get("strategy_id") or "")
    contributing = {
        str(value)
        for value in candidate.get("strategy_ids", [])
        if str(value)
    }
    if explicit:
        contributing.add(explicit)
    authorized: list[dict[str, Any]] = []
    for row in registry.get("strategies", []):
        if row.get("deployment_status") != LIVE_STATUS:
            continue
        if str(row.get("strategy_id") or "") not in contributing:
            continue
        symbols = {str(value).upper() for value in row.get("allowed_symbols", [])}
        classes = {str(value).upper() for value in row.get("allowed_asset_classes", [])}
        if symbol not in symbols:
            continue
        if asset_class and asset_class not in classes and not (
            asset_class in {"STOCK", "EQUITY"} and "STK" in classes
        ):
            continue
        authorized.append(row)
    authorized.sort(key=lambda row: str(row.get("strategy_id") or ""))
    blockers: list[str] = []
    if registry.get("status") != "GO":
        blockers.append("STRATEGY_AUTHORITY_REGISTRY_NOT_GO")
    if not contributing:
        blockers.append("CONTRIBUTING_STRATEGY_ID_REQUIRED")
    if not authorized:
        blockers.append("NO_EXACT_LIVE_AUTHORIZED_CONTRIBUTING_STRATEGY")
    if len(authorized) > 1:
        blockers.append("AMBIGUOUS_LIVE_AUTHORIZED_CONTRIBUTING_STRATEGY")
    selected = authorized[0] if len(authorized) == 1 and not blockers else None
    return {
        "schema": "exact_portfolio_strategy_binding_v1",
        "status": "GO" if selected else "NO_GO",
        "symbol": symbol,
        "contributing_strategy_ids": sorted(contributing),
        "authorized_contributing_strategy_ids": [
            str(row["strategy_id"]) for row in authorized
        ],
        "strategy_id": selected.get("strategy_id") if selected else None,
        "strategy_version": selected.get("version") if selected else None,
        "strategy_source_hash": selected.get("source_hash") if selected else None,
        "strategy_parameter_hash": selected.get("parameter_hash") if selected else None,
        "blockers": sorted(set(blockers)),
        "inference_used": False,
        "automatic_promotion_used": False,
    }


def bind_opportunities(
    opportunities: Iterable[dict[str, Any]],
    registry: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for candidate in opportunities:
        binding = bind_exact_strategy(candidate, registry)
        result.append({**candidate, "strategy_binding": binding})
    return result


def _normalized(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


__all__ = [
    "LIVE_STATUS",
    "REGISTRY_PATH",
    "bind_exact_strategy",
    "bind_opportunities",
    "load_strategy_authority_registry",
    "operational_strategy_allowlist_hash",
]
