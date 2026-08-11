from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from stocks.execution.authority import ExecutionAuthority
from stocks.execution.models import KillSwitchState, KillSwitchStatus, OrderIntent, OrderSide, RiskDecision
from stocks.execution.portfolio import PortfolioState


@dataclass(frozen=True)
class RiskLimits:
    max_order_notional: Decimal = Decimal("25000")
    max_position_weight: Decimal = Decimal("0.35")
    max_region_weight: Decimal = Decimal("0.60")
    max_sleeve_weight: Decimal = Decimal("0.70")
    max_currency_weight: Decimal = Decimal("0.80")
    max_open_orders: int = 5
    max_trades_per_day: int = 10
    max_daily_turnover: Decimal = Decimal("0.50")
    max_daily_loss: Decimal = Decimal("0.03")
    max_drawdown: Decimal = Decimal("0.20")


def evaluate_risk(
    intent: OrderIntent,
    *,
    portfolio: PortfolioState,
    kill_switch: KillSwitchState,
    limits: RiskLimits,
    resolved: bool = True,
    session_ready: bool = True,
    data_stale: bool = False,
    instrument_blocked: bool = False,
    duplicate: bool = False,
    open_orders: int = 0,
    trades_today: int = 0,
    turnover_today: Decimal = Decimal("0"),
    daily_loss: Decimal = Decimal("0"),
    drawdown: Decimal = Decimal("0"),
    region_weight_after: Decimal = Decimal("0"),
    sleeve_weight_after: Decimal = Decimal("0"),
    currency_weight_after: Decimal = Decimal("0"),
) -> RiskDecision:
    reasons: list[str] = []
    if intent.authority is not ExecutionAuthority.NONE:
        reasons.append("AUTHORITY_NOT_GRANTED")
    if not resolved:
        reasons.append("UNRESOLVED_CONTRACT")
    if intent.security_type != "STK" or instrument_blocked:
        reasons.append("UNRESOLVED_CONTRACT")
    if not session_ready:
        reasons.append("SESSION_BLOCKED")
    if data_stale:
        reasons.append("STALE_DATA")
    if intent.quantity <= 0:
        reasons.append("INVALID_QUANTITY")
    if intent.side is OrderSide.SELL and intent.quantity > portfolio.positions.get(intent.con_id, Decimal("0")):
        reasons.append("SHORT_POSITION_BLOCKED")
    if intent.notional_eur < 0 or intent.notional_eur > limits.max_order_notional:
        reasons.append("ORDER_NOTIONAL_EXCEEDED")
    if portfolio.equity() > 0 and intent.notional_eur / portfolio.equity() > limits.max_position_weight:
        reasons.append("POSITION_LIMIT_EXCEEDED")
    if region_weight_after > limits.max_region_weight:
        reasons.append("REGION_LIMIT_EXCEEDED")
    if sleeve_weight_after > limits.max_sleeve_weight:
        reasons.append("SLEEVE_LIMIT_EXCEEDED")
    if currency_weight_after > limits.max_currency_weight:
        reasons.append("CURRENCY_LIMIT_EXCEEDED")
    if open_orders >= limits.max_open_orders:
        reasons.append("OPEN_ORDER_LIMIT_EXCEEDED")
    if trades_today >= limits.max_trades_per_day:
        reasons.append("DAILY_TRADE_LIMIT_EXCEEDED")
    if turnover_today > limits.max_daily_turnover:
        reasons.append("DAILY_TURNOVER_LIMIT_EXCEEDED")
    if daily_loss > limits.max_daily_loss:
        reasons.append("DAILY_LOSS_LIMIT_EXCEEDED")
    if drawdown > limits.max_drawdown:
        reasons.append("DRAWDOWN_LIMIT_EXCEEDED")
    if kill_switch.status is not KillSwitchStatus.ARMED:
        reasons.append("KILL_SWITCH_ACTIVE")
    if duplicate:
        reasons.append("DUPLICATE_INTENT")
    approved = not reasons
    return RiskDecision(
        intent_id=intent.intent_id,
        approved=approved,
        decision_code="RISK_APPROVED_SIMULATION_ONLY" if approved else reasons[0],
        reasons=tuple(reasons),
        evaluated_at=intent.created_at,
    )
