from __future__ import annotations

from stocks.shadow.clock import _parse
from stocks.shadow.models import ShadowSignal


def validate_signal(signal: ShadowSignal, cutoff_timestamp: str) -> dict[str, str]:
    if not signal.source_content_hash:
        return {"status": "NO_GO", "signal_status": "BLOCKED", "decision_code": "MISSING_DATASET_HASH"}
    if _parse(signal.available_at) > _parse(cutoff_timestamp):
        return {"status": "NO_GO", "signal_status": "NON_CAUSAL_BLOCKED", "decision_code": "FUTURE_DATA_BLOCKED"}
    if signal.signal_status not in {"VALID", "MISSING", "STALE", "BLOCKED", "NON_CAUSAL_BLOCKED"}:
        return {"status": "NO_GO", "signal_status": "BLOCKED", "decision_code": "INVALID_SIGNAL_STATUS"}
    return {"status": "GO" if signal.signal_status == "VALID" else "NO_GO", "signal_status": signal.signal_status}


def provenance_audit(signals: list[ShadowSignal], cutoff_timestamp: str) -> dict[str, object]:
    rows = [validate_signal(signal, cutoff_timestamp) for signal in signals]
    return {
        "status": "GO" if rows and all(row["status"] == "GO" for row in rows) else "NO_GO",
        "signal_count": len(signals),
        "valid_signal_count": sum(row["signal_status"] == "VALID" for row in rows),
        "non_causal_blocked_count": sum(row["signal_status"] == "NON_CAUSAL_BLOCKED" for row in rows),
        "missing_source_hash_count": sum(row.get("decision_code") == "MISSING_DATASET_HASH" for row in rows),
        "rows": rows,
    }
