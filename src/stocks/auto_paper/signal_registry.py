from __future__ import annotations

from stocks.auto_paper.contracts import AutoSignal, model_to_jsonable
from stocks.auto_paper.storage import AutoPaperStore
from stocks.execution.idempotency import stable_hash


def register_signal(store: AutoPaperStore, signal: AutoSignal, account_fingerprint: str) -> dict[str, object]:
    if signal.source_provenance_hash != stable_hash(signal.source_provenance):
        return {
            "status": "SIGNAL_DATA_STALE",
            "economic_key_hash": signal.economic_key(account_fingerprint),
            "registry_status": "PROVENANCE_HASH_MISMATCH",
        }
    economic_key = signal.economic_key(account_fingerprint)
    status = store.append_once("signals", economic_key, model_to_jsonable(signal))
    return {
        "status": "SIGNAL_VALIDATED" if status == "RECORDED" else "SIGNAL_DUPLICATE" if status == "IDEMPOTENT_REPLAY" else "SIGNAL_ECONOMIC_KEY_CONFLICT",
        "economic_key_hash": economic_key,
        "registry_status": status,
    }
