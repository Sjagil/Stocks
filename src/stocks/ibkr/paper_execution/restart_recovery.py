from __future__ import annotations

from typing import Mapping


def restart_recovery_audit(
    scenario_statuses: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Derive restart readiness from executed P0 scenarios.

    Missing evidence is deliberately NO_GO; this function never manufactures
    a positive restart attestation.
    """
    statuses = dict(scenario_statuses or {})

    def proven(*scenario_ids: str) -> str:
        return (
            "GO"
            if all(statuses.get(scenario_id) == "GO" for scenario_id in scenario_ids)
            else "NO_GO"
        )

    report: dict[str, object] = {
        "disconnect_before_submission": proven(
            "CONNECTION_LOSS_FAIL_CLOSED"
        ),
        "disconnect_after_submission": proven(
            "CONNECTION_LOSS_FAIL_CLOSED",
            "MANUAL_TWS_ORDER_FAILS_RECONCILIATION",
        ),
        "disconnect_with_submitted_order": proven(
            "MANUAL_TWS_ORDER_FAILS_RECONCILIATION"
        ),
        "disconnect_with_partial_fill": proven(
            "PARTIAL_FILL_ACCUMULATION",
            "CONNECTION_LOSS_FAIL_CLOSED",
        ),
        "restart_with_open_order": proven(
            "RESTART_REPLAY_IDEMPOTENT",
            "MANUAL_TWS_ORDER_FAILS_RECONCILIATION",
        ),
        "restart_after_fill_before_commission": proven(
            "FILL_BEFORE_COMMISSION_RECONCILES",
            "RESTART_REPLAY_IDEMPOTENT",
        ),
        "restart_after_commission_before_portfolio_update": proven(
            "COMMISSION_BEFORE_EXECUTION_JOIN",
            "RESTART_REPLAY_IDEMPOTENT",
        ),
        "new_submission_blocked_until_reconciliation": proven(
            "CONNECTION_LOSS_FAIL_CLOSED",
            "EXTERNAL_POSITION_CHANGE_BLOCKS",
        )
        == "GO",
        "evidence_source": "IBKR_EXECUTION_P0_SAFETY_MATRIX",
    }
    report["status"] = (
        "GO"
        if all(
            value == "GO" or value is True
            for key, value in report.items()
            if key != "evidence_source"
        )
        else "NO_GO"
    )
    return report
