from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.quant_platform.capabilities import CAPABILITIES


AUTHORITY_PATH = Path("config/portfolio/quant_capability_authority_v1.json")
VALID_AUTHORITIES = {
    "DATA_ONLY",
    "CONTEXT_ONLY",
    "RANKING_ALLOWED",
    "RISK_ALLOWED",
    "PORTFOLIO_ALLOWED",
    "EXECUTION_GATING_ALLOWED",
    "SHADOW_ONLY",
}
MONEY_MODEL_CAPABILITIES = {17, 18, 19, 31, 32}


def load_quant_authority_map(project_root: Path) -> dict[str, Any]:
    """Load and fail-closed validate authority for exactly 33 capabilities."""

    path = project_root / AUTHORITY_PATH
    try:
        configured = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        configured = {}
    registry = {int(row["id"]): row for row in CAPABILITIES}
    configured_rows = configured.get("capabilities", [])
    by_id = {
        int(row.get("id", -1)): row
        for row in configured_rows
        if isinstance(row, dict)
    }
    blockers: list[str] = []
    if set(by_id) != set(registry):
        blockers.append("EXACT_33_CAPABILITY_COVERAGE_REQUIRED")
    rows: list[dict[str, Any]] = []
    for capability_id, capability in sorted(registry.items()):
        configured_row = by_id.get(capability_id, {})
        authority = str(configured_row.get("authority") or "SHADOW_ONLY")
        if authority not in VALID_AUTHORITIES:
            blockers.append(f"INVALID_AUTHORITY:{capability_id}")
            authority = "SHADOW_ONLY"
        money_control = bool(configured_row.get("money_control", False))
        if money_control:
            blockers.append(f"DIRECT_MONEY_CONTROL_FORBIDDEN:{capability_id}")
        if capability_id in MONEY_MODEL_CAPABILITIES and authority != "SHADOW_ONLY":
            blockers.append(f"UNVALIDATED_MODEL_NOT_SHADOW:{capability_id}")
        rows.append(
            {
                **capability,
                "authority": authority,
                "money_control": False,
                "direct_broker_access": False,
            }
        )
    if configured.get("direct_broker_access") is not False:
        blockers.append("DIRECT_BROKER_ACCESS_MUST_BE_FALSE")
    if configured.get("automatic_live_promotion") is not False:
        blockers.append("AUTOMATIC_LIVE_PROMOTION_MUST_BE_FALSE")
    report: dict[str, Any] = {
        "schema": "quant_capability_authority_status_v1",
        "status": "GO" if not blockers else "NO_GO",
        "capability_count": len(rows),
        "capabilities": rows,
        "shadow_capability_ids": [
            row["id"] for row in rows if row["authority"] == "SHADOW_ONLY"
        ],
        "execution_boundary": "CANONICAL_P0_BRIDGE_ONLY",
        "automatic_live_promotion": False,
        "direct_broker_access": False,
        "blockers": sorted(set(blockers)),
    }
    report["content_hash"] = stable_hash(report)
    return report


def capability_can(capability_id: int, role: str, authority_map: dict[str, Any]) -> bool:
    return any(
        int(row.get("id", -1)) == capability_id
        and row.get("authority") == role
        for row in authority_map.get("capabilities", [])
    )


__all__ = [
    "AUTHORITY_PATH",
    "MONEY_MODEL_CAPABILITIES",
    "VALID_AUTHORITIES",
    "capability_can",
    "load_quant_authority_map",
]
