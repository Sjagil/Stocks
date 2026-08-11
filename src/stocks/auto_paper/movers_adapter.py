from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stocks.data.phase5_common import sha256_file


MOVER_TYPES = (
    "TOP_GAINERS",
    "TOP_LOSERS",
    "RELATIVE_VOLUME_LEADERS",
    "GAP_UP",
    "GAP_DOWN",
    "NEW_20D_HIGH",
    "NEW_52W_HIGH",
    "PERSISTENT_LEADERS",
)


def phase11_1_adoption(project_root: Path) -> dict[str, object]:
    freeze_path = project_root / "output" / "research" / "phase11_1" / "freeze-status.json"
    if not freeze_path.exists():
        return {"status": "NO_GO", "adoption_status": "PHASE11_1_FREEZE_MISSING"}
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    marker = payload.get("freeze_status")
    expected = "PHASE11_1_ORTHOGONAL_PIT_DATA_AND_ALPHA_STRATEGY_FOUNDATION_FROZEN_GO"
    return {
        "status": "GO" if marker == expected else "NO_GO",
        "adoption_status": "PHASE11_1_IMPORTED_READ_ONLY" if marker == expected else "PHASE11_1_FREEZE_MISMATCH",
        "freeze_marker": marker,
        "freeze_artifact_hash": sha256_file(freeze_path),
        "source_modified": False,
    }


def classify_candidate(candidate: dict[str, Any]) -> dict[str, object]:
    mover_type = str(candidate.get("mover_type", ""))
    gates = {
        "known_mover_type": mover_type in MOVER_TYPES,
        "shariah_gate": bool(candidate.get("shariah_eligible", False)),
        "liquidity_gate": bool(candidate.get("liquid", False)),
        "news_attribution": bool(candidate.get("news_attributed", False)),
        "fundamental_attribution": bool(candidate.get("fundamentals_available", False)),
        "technical_acceptance": bool(candidate.get("technical_acceptance", False)),
        "event_clustering": bool(candidate.get("event_cluster_known", False)),
        "pump_rejection": not bool(candidate.get("unexplained_pump", False)),
        "value_trap_rejection": not bool(candidate.get("value_trap", False)),
        "permanent_impairment_rejection": not bool(candidate.get("permanent_impairment", False)),
    }
    return {
        "status": "MOVER_CANDIDATE_ACCEPTED" if all(gates.values()) else "MOVER_CANDIDATE_REJECTED",
        "mover_type": mover_type,
        "gates": gates,
        "order_authority": "NONE",
    }
