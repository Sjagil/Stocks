from __future__ import annotations

from stocks.execution.idempotency import stable_hash
from stocks.shadow.storage import ShadowLedgerStore


def replay_state(store: ShadowLedgerStore) -> dict[str, object]:
    counts = store.counts()
    decisions = store.read_table("decisions")
    targets = store.read_table("target_portfolios")
    fills = store.read_table("fills")
    snapshots = store.read_table("snapshots")
    evaluations = store.read_table("evaluations")
    state = {
        "decision_hashes": [row["payload_hash"] for row in decisions],
        "target_hashes": [row["payload_hash"] for row in targets],
        "fill_hashes": [row["payload_hash"] for row in fills],
        "snapshot_hashes": [row["payload_hash"] for row in snapshots],
        "evaluation_hashes": [row["payload_hash"] for row in evaluations],
        "counts": counts,
    }
    return {**state, "state_hash": stable_hash(state)}
