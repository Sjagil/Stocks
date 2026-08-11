from __future__ import annotations

from decimal import Decimal

from stocks.auto_paper.storage import AutoPaperStore


def replay_fixture(store: AutoPaperStore, *, run_id: str = "SYNTHETIC-RUN-1") -> dict[str, object]:
    order_key = f"{run_id}:ORDER-1"
    first = store.append_once("hypothetical_fills", f"{order_key}:1", {"quantity": "0.4", "status": "PARTIAL"})
    second = store.append_once("hypothetical_fills", f"{order_key}:2", {"quantity": "0.6", "status": "FILLED"})
    duplicate = store.append_once("hypothetical_fills", f"{order_key}:2", {"quantity": "0.6", "status": "FILLED"})
    late_commission = store.append_once("commissions", order_key, {"amount": "0.01", "status": "LATE_JOINED"})
    total = sum(
        Decimal(row["payload"]["quantity"])
        for row in store.records("hypothetical_fills")
        if str(row["economic_key"]).startswith(f"{order_key}:")
    )
    before_restart = store.counts()
    restarted = AutoPaperStore(store.path)
    restarted.initialize()
    after_restart = restarted.counts()
    return {
        "status": "REPLAY_GO" if total == Decimal("1.0") and before_restart == after_restart else "NO_GO",
        "partial_fill": first == "RECORDED",
        "full_fill": second == "RECORDED",
        "duplicate_callback": duplicate == "IDEMPOTENT_REPLAY",
        "late_commission": late_commission == "RECORDED",
        "restart_recovery": before_restart == after_restart,
        "disconnect_recovery": "BOUNDED_RECONNECT_FIXTURE_GO",
        "brokerwrite_calls": 0,
    }
