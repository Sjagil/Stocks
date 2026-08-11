from __future__ import annotations

from decimal import Decimal

from stocks.auto_paper.authority import risk_reducing_exit_authority
from stocks.execution.idempotency import stable_hash


EXIT_REASONS = (
    "HARD_STOP_LOSS",
    "ATR_STOP",
    "MAXIMUM_HOLDING_PERIOD",
    "THESIS_INVALIDATED",
    "NEGATIVE_NEWS",
    "SHARIAH_STATUS_LOST",
    "PORTFOLIO_KILL_SWITCH",
    "DATA_INTEGRITY_EMERGENCY",
)
EXIT_STATUSES = (
    "EXIT_NOT_REQUIRED",
    "EXIT_SIGNAL_GENERATED",
    "EXIT_RISK_REDUCING_ALLOWED",
    "EXIT_SUBMITTED",
    "EXIT_PARTIALLY_FILLED",
    "EXIT_FILLED",
    "EXIT_RECONCILED",
    "EXIT_BLOCKED_POSITION_MISMATCH",
    "EXIT_BLOCKED_SNAPSHOT_INCOMPLETE",
)


def evaluate_risk_reducing_exit(
    *,
    con_id: int,
    local_con_id: int,
    broker_con_id: int,
    sell_quantity: Decimal,
    local_quantity: Decimal,
    broker_quantity: Decimal,
    account_match: bool,
    snapshot_complete: bool,
    reason: str,
    limit_price: Decimal,
    entries_today: int = 0,
    runtime_enabled: bool = False,
) -> dict[str, object]:
    if reason not in EXIT_REASONS:
        return _blocked("EXIT_NOT_REQUIRED")
    if not snapshot_complete:
        return _blocked("EXIT_BLOCKED_SNAPSHOT_INCOMPLETE")
    if not account_match or con_id != local_con_id or con_id != broker_con_id or local_quantity != broker_quantity:
        return _blocked("EXIT_BLOCKED_POSITION_MISMATCH")
    if local_quantity <= 0 or broker_quantity <= 0:
        return _blocked("SELL_WITHOUT_POSITION_BLOCKED")
    if sell_quantity <= 0 or sell_quantity > local_quantity or sell_quantity > broker_quantity:
        return _blocked("SELL_EXCEEDS_RECONCILED_POSITION")
    remaining = local_quantity - sell_quantity
    if remaining < 0:
        return _blocked("SHORT_POSITION_BLOCKED")
    order = {
        "con_id": con_id,
        "side": "SELL",
        "quantity": str(sell_quantity),
        "limit_price": str(limit_price),
        "order_type": "LIMIT",
        "time_in_force": "DAY",
        "outside_rth": False,
        "broker_submission": False,
    }
    authority = risk_reducing_exit_authority(
        existing_long_position=True,
        quantities_match=True,
        account_match=True,
        con_id_match=True,
        sell_within_position=True,
        limit_day_rth=True,
        runtime_enabled=runtime_enabled,
    )
    return {
        "status": "EXIT_RISK_REDUCING_ALLOWED",
        "authority_type": authority["authority_type"],
        "exit_authority": authority["runtime_authority"],
        "execution_authority": authority["runtime_authority"],
        "automatic_submission": authority["automatic_submission"],
        "entry_count_ignored": entries_today,
        "remaining_quantity": str(remaining),
        "hypothetical_order": order,
        "decision_hash": stable_hash(order | {"reason": reason}),
    }


def _blocked(status: str) -> dict[str, object]:
    return {"status": status, "exit_authority": "NONE", "execution_authority": "NONE"}
