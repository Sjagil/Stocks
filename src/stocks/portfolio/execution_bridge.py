from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from stocks.execution.idempotency import stable_hash


RISK_REDUCING_ACTIONS = {"EXIT", "REDUCE", "SELL_DELTA"}
RISK_INCREASING_ACTIONS = {"OPEN", "ADD", "BUY_DELTA", "ROTATE"}


def build_p0_execution_bridge(
    project_root: Path,
    *,
    targets: Iterable[dict[str, Any]],
    sizing_rows: Iterable[dict[str, Any]],
    opportunities: Iterable[dict[str, Any]],
    reconciliation: dict[str, Any],
    live_authority: dict[str, Any],
    writer_integrity: dict[str, Any],
    p02_integrity: dict[str, Any],
    strategy_allowlist: dict[str, Any],
    dynamic_risk: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Translate target deltas to the existing P0 boundary without calling it.

    The bridge is deliberately a pure planner.  It names the canonical P0
    prepare/submit functions but never imports or invokes a broker adapter.
    """

    del project_root  # retained in the public interface for canonical routing
    sizing = {
        str(row.get("ticker") or row.get("symbol") or "").upper(): row
        for row in sizing_rows
    }
    opportunity = {
        str(row.get("symbol") or row.get("ticker") or "").upper(): row
        for row in opportunities
    }
    allowed_by_symbol: dict[str, set[str]] = {}
    for row in strategy_allowlist.get("strategies", []):
        strategy_id = str(row.get("strategy_id") or "")
        if row.get("status") != "PIT_LIVE_ALLOWLISTED" or not strategy_id:
            continue
        for symbol in row.get("allowed_symbols", []):
            allowed_by_symbol.setdefault(str(symbol).upper(), set()).add(strategy_id)

    proposals = []
    for target in targets:
        symbol = str(target.get("symbol") or "").upper()
        if not symbol:
            continue
        row = sizing.get(symbol, {})
        candidate = opportunity.get(symbol, {})
        action = _canonical_action(target)
        quantity = abs(_integer(target.get("quantity_delta")))
        canary_quantity = _integer(row.get("level1_canary_qty"))
        if action in RISK_INCREASING_ACTIONS:
            quantity = min(quantity, max(0, canary_quantity))
        strategy_binding = candidate.get("strategy_binding", {})
        strategy_id = str(
            target.get("strategy_id")
            or strategy_binding.get("strategy_id")
            or ""
        )
        contributing_strategy_ids = sorted(
            {
                str(value)
                for value in candidate.get("strategy_ids", [])
                if str(value)
            }
        )
        exact_strategy_authorized = bool(
            strategy_id and strategy_id in allowed_by_symbol.get(symbol, set())
        )
        expected_net = _optional_decimal(target.get("expected_net_return"))
        expected_loss = _optional_decimal(target.get("expected_loss"))
        fees = _optional_decimal(
            candidate.get("expected_cost_eur", candidate.get("fees_eur"))
        )
        event_risk = _decimal(candidate.get("event_risk"))
        max_event_risk = _decimal(
            policy.get("risk", {}).get("maximum_event_risk", 0.75)
        )
        shariah_allowed = candidate.get("shariah_status") == "SHARIAH_ALLOWED"
        data_freshness = candidate.get(
            "data_freshness", candidate.get("data_quality")
        )
        is_reduction = action in RISK_REDUCING_ACTIONS
        gates = {
            "BROKER_CONNECTED": reconciliation.get("status") == "GO",
            "ACCOUNT_FRESH": reconciliation.get("status") == "GO"
            and str(reconciliation.get("reconciliation_status", "")).startswith(
                "LIVE_RECONCILED"
            ),
            "CASH_FRESH": reconciliation.get("status") == "GO",
            "RECONCILIATION_GO": reconciliation.get("status") == "GO"
            and int(reconciliation.get("unknown_positions", 0)) == 0
            and int(reconciliation.get("unknown_orders", 0)) == 0,
            "WRITER_INTEGRITY_GO": writer_integrity.get("status") == "GO"
            and p02_integrity.get("status") == "GO",
            "STRATEGY_LIVE_AUTHORIZED": exact_strategy_authorized or is_reduction,
            "SHARIAH_ALLOWED": shariah_allowed or is_reduction,
            "SIGNAL_FRESH": float(data_freshness or 0) > 0 or is_reduction,
            "MARKET_DATA_FRESH": float(data_freshness or 0) > 0 or is_reduction,
            "PORTFOLIO_RISK_GO": dynamic_risk.get("status") == "GO"
            and (dynamic_risk.get("new_entries_allowed") is True or is_reduction),
            "CORRELATION_GO": not bool(
                candidate.get("correlation_blocked", False)
            ),
            "CONCENTRATION_GO": not bool(
                candidate.get("concentration_blocked", False)
            ),
            "LIQUIDITY_GO": (
                candidate.get("liquidity") is not None
                and float(candidate.get("liquidity") or 0) > 0
            )
            or is_reduction,
            "TCA_GO": fees is not None and fees >= 0 or is_reduction,
            "EXPECTED_NET_EDGE_GO": expected_net is not None
            and expected_net > 0
            or is_reduction,
            "EVENT_RISK_GO": event_risk <= max_event_risk or is_reduction,
            "DAILY_LOSS_GO": dynamic_risk.get("loss_guard", {}).get(
                "new_entries_allowed"
            )
            is True
            or is_reduction,
            "DRAWDOWN_GO": float(
                dynamic_risk.get("multipliers", {}).get("drawdown", 0)
            )
            > 0
            or is_reduction,
        }
        blockers = [name for name, passed in gates.items() if not passed]
        if quantity < 1 and action not in {"HOLD", "NO_TRADE"}:
            blockers.append("WHOLE_SHARE_DELTA_NOT_EXECUTABLE")
        if target.get("quantity_delta") is not None and not _is_whole(
            target.get("quantity_delta")
        ):
            blockers.append("FRACTIONAL_DELTA_FORBIDDEN")
        current_authority = str(live_authority.get("execution_authority") or "NONE")
        if current_authority not in {
            "LIVE_LEVEL_ONE",
            "AUTONOMOUS_LEVEL_ONE",
            "LIVE_LEVEL_TWO",
        }:
            blockers.append("CURRENT_LIVE_AUTHORITY_NOT_ACTIVE")
        if action in RISK_INCREASING_ACTIONS and not strategy_id:
            blockers.append("EXACT_STRATEGY_ID_REQUIRED")
        blockers = sorted(set(blockers))
        proposals.append(
            {
                "symbol": symbol,
                "action": action,
                "requested_quantity": abs(_integer(target.get("quantity_delta"))),
                "current_authority_quantity": quantity,
                "whole_share": True,
                "strategy_id": strategy_id or None,
                "strategy_binding": strategy_binding,
                "contributing_strategy_ids": contributing_strategy_ids,
                "allowed_strategy_ids_for_symbol": sorted(
                    allowed_by_symbol.get(symbol, set())
                ),
                "expected_net_return": (
                    None if expected_net is None else float(expected_net)
                ),
                "expected_loss": (
                    None if expected_loss is None else float(expected_loss)
                ),
                "expected_transaction_cost_eur": (
                    None if fees is None else float(fees)
                ),
                "gates": gates,
                "machine_policy_approved": not blockers,
                "blockers": blockers,
                "canonical_prepare_function": "stocks.live.service.live_prepare",
                "canonical_submit_function": (
                    "stocks.live.service.live_submit_authorized"
                ),
                "bridge_invoked_prepare": False,
                "bridge_invoked_submit": False,
                "broker_writes": 0,
            }
        )
    proposals.sort(
        key=lambda row: (
            0 if row["action"] in RISK_REDUCING_ACTIONS else 1,
            0 if row["machine_policy_approved"] else 1,
            -(row.get("expected_net_return") or 0),
            row["symbol"],
        )
    )
    executable = [row for row in proposals if row["machine_policy_approved"]]
    selected = executable[0] if executable else None
    report: dict[str, Any] = {
        "schema": "canonical_p0_execution_bridge_plan_v1",
        "status": "GO",
        "boundary": "QUANT_TARGETS_TO_CANONICAL_P0_ONLY",
        "proposal_count": len(proposals),
        "machine_approved_count": len(executable),
        "proposals": proposals,
        "selected_action": selected,
        "current_execution_authority": live_authority.get(
            "execution_authority", "NONE"
        ),
        "current_manual_approval_required": live_authority.get(
            "manual_approval_required", True
        ),
        "current_automatic_submission": live_authority.get(
            "automatic_order_submission", False
        ),
        "autonomous_bounded_per_trade_approval_required": False,
        "autonomous_bounded_separately_activated": (
            live_authority.get("execution_authority")
            == "AUTONOMOUS_LEVEL_ONE"
        ),
        "direct_ibkr_calls": 0,
        "broker_writes": 0,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _canonical_action(target: dict[str, Any]) -> str:
    raw = str(target.get("action") or "NO_TRADE").upper()
    current = _decimal(target.get("current_quantity"))
    desired = _decimal(target.get("desired_quantity"))
    if raw == "BUY_DELTA":
        return "ADD" if current > 0 else "OPEN"
    if raw == "SELL_DELTA":
        return "EXIT" if desired <= 0 else "REDUCE"
    if raw in {"HOLD", "EXIT", "REDUCE", "OPEN", "ADD", "ROTATE"}:
        return raw
    return "NO_TRADE"


def _decimal(value: Any) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")
    return parsed if parsed.is_finite() else Decimal("0")


def _optional_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _integer(value: Any) -> int:
    parsed = _decimal(value)
    if parsed != parsed.to_integral_value():
        return 0
    return int(parsed)


def _is_whole(value: Any) -> bool:
    parsed = _optional_decimal(value)
    return parsed is not None and parsed == parsed.to_integral_value()


__all__ = [
    "RISK_INCREASING_ACTIONS",
    "RISK_REDUCING_ACTIONS",
    "build_p0_execution_bridge",
]
