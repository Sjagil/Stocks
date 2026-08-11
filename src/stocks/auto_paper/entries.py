from __future__ import annotations

from stocks.auto_paper.contracts import AutoDecision, AutoSignal, model_to_jsonable
from stocks.auto_paper.signal_registry import register_signal
from stocks.auto_paper.storage import AutoPaperStore
from stocks.execution.idempotency import stable_hash


def prepare_shadow_entry(
    store: AutoPaperStore,
    signal: AutoSignal,
    *,
    account_fingerprint: str,
    risk: dict[str, object],
) -> AutoDecision:
    signal_hash = stable_hash(model_to_jsonable(signal))
    if risk.get("status") != "ENTRY_RISK_GO":
        return AutoDecision("BLOCKED", "SIGNAL_RISK_BLOCKED", "NONE", signal_hash, stable_hash(risk))
    registry = register_signal(store, signal, account_fingerprint)
    if registry["status"] != "SIGNAL_VALIDATED":
        return AutoDecision("BLOCKED", str(registry["status"]), "NONE", signal_hash, stable_hash(risk))
    order = {
        "con_id": signal.con_id,
        "side": "BUY",
        "quantity": str(signal.target_quantity),
        "order_type": "LIMIT",
        "limit_price": str(signal.maximum_limit_price),
        "time_in_force": "DAY",
        "outside_rth": False,
        "transmit": False,
        "broker_submission": False,
    }
    key = str(registry["economic_key_hash"])
    store.append_once("risk_decisions", key, risk)
    store.append_once("shadow_intents", key, {"signal_hash": signal_hash, "status": "SIGNAL_SHADOW_ONLY"})
    store.append_once("hypothetical_orders", key, order)
    return AutoDecision(
        "GO",
        "SIGNAL_SHADOW_ONLY",
        "NONE",
        signal_hash,
        stable_hash(risk),
        hypothetical_order=order,
    )
