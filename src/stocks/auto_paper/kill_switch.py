from __future__ import annotations


KILL_SWITCHES = (
    "AUTO_PAPER_DISABLED",
    "PHASE9_NOT_FROZEN",
    "FINANCIAL_FINALIST_MISSING",
    "FORWARD_SHADOW_NOT_GO",
    "ACCOUNT_FINGERPRINT_MISMATCH",
    "BROKER_RECONCILIATION_MISMATCH",
    "UNKNOWN_BROKER_ORDER",
    "UNKNOWN_EXECUTION",
    "COMMISSION_SCOPE_INCOMPLETE",
    "STALE_SIGNAL",
    "STALE_QUOTE",
    "WIDE_SPREAD",
    "DAILY_LOSS_LIMIT_REACHED",
    "POSITION_LIMIT_REACHED",
    "PORTFOLIO_EXPOSURE_REACHED",
    "SECTOR_EXPOSURE_REACHED",
    "EVENT_CLUSTER_LIMIT_REACHED",
    "SHARIAH_STATUS_STALE",
    "SHARIAH_STATUS_LOST",
    "HEARTBEAT_STALE",
)


def evaluate_kill_switches(conditions: dict[str, bool]) -> dict[str, object]:
    active = sorted(name for name in KILL_SWITCHES if conditions.get(name, False))
    return {
        "status": "KILL_SWITCH_CLEAR" if not active else "KILL_SWITCH_ACTIVE",
        "active_kill_switches": active,
        "new_entries_allowed": not active,
        "automatic_state_corrections": 0,
        "risk_reducing_exit_requires_exact_reconciliation": True,
    }
