from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest().upper()


def economic_order_key(
    *,
    strategy_id: str,
    strategy_version: str,
    decision_id: str,
    con_id: int,
    side: str,
    target_position: Decimal | str,
    session_date: str,
) -> str:
    return stable_hash(
        {
            "strategy_id": strategy_id,
            "strategy_version": strategy_version,
            "decision_id": decision_id,
            "con_id": int(con_id),
            "side": str(side),
            "target_position": str(target_position),
            "session_date": session_date,
        }
    )

