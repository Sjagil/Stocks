from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from stocks.execution.authority import ExecutionAuthority


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(StrEnum):
    DAY = "DAY"
    GTC = "GTC"


class OrderState(StrEnum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    RISK_REJECTED = "RISK_REJECTED"
    RISK_APPROVED = "RISK_APPROVED"
    QUEUED = "QUEUED"
    SUBMITTED_SIMULATED = "SUBMITTED_SIMULATED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    INACTIVE = "INACTIVE"
    UNKNOWN_BLOCKED = "UNKNOWN_BLOCKED"


class KillSwitchStatus(StrEnum):
    ARMED = "ARMED"
    TRIGGERED_MANUAL = "TRIGGERED_MANUAL"
    TRIGGERED_DAILY_LOSS = "TRIGGERED_DAILY_LOSS"
    TRIGGERED_DRAWDOWN = "TRIGGERED_DRAWDOWN"
    TRIGGERED_RECONCILIATION = "TRIGGERED_RECONCILIATION"
    TRIGGERED_STALE_DATA = "TRIGGERED_STALE_DATA"
    RESET_PENDING = "RESET_PENDING"


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    economic_order_key: str
    strategy_id: str
    strategy_version: str
    decision_id: str
    decision_timestamp: str
    dataset_hash: str
    parameter_hash: str
    con_id: int
    symbol: str
    security_type: str
    side: OrderSide
    quantity: Decimal
    notional_eur: Decimal
    order_type: OrderType
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: TimeInForce
    outside_rth: bool
    session_date: str
    created_at: str
    expires_at: str
    authority: ExecutionAuthority
    target_position: Decimal


@dataclass(frozen=True)
class RiskDecision:
    intent_id: str
    approved: bool
    decision_code: str
    reasons: tuple[str, ...]
    evaluated_at: str


@dataclass(frozen=True)
class BrokerOrder:
    broker_order_id: str
    intent_id: str
    state: OrderState
    submitted_quantity: Decimal
    remaining_quantity: Decimal


@dataclass(frozen=True)
class OrderEvent:
    event_id: str
    intent_id: str
    state: OrderState
    event_type: str
    created_at: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class FillEvent:
    fill_id: str
    broker_order_id: str
    intent_id: str
    con_id: int
    side: OrderSide
    quantity: Decimal
    price: Decimal
    created_at: str


@dataclass(frozen=True)
class CommissionEvent:
    commission_id: str
    fill_id: str
    amount_eur: Decimal
    created_at: str


@dataclass(frozen=True)
class PositionSnapshot:
    con_id: int
    symbol: str
    quantity: Decimal
    average_cost: Decimal
    market_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal


@dataclass(frozen=True)
class CashSnapshot:
    currency: str
    cash: Decimal
    reserved_cash: Decimal


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: Decimal
    reserved_cash: Decimal
    position_value: Decimal
    equity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    commissions: Decimal
    high_water_mark: Decimal
    drawdown: Decimal


@dataclass(frozen=True)
class ReconciliationResult:
    status: str
    mismatches: tuple[str, ...]
    kill_switch_triggered: bool


@dataclass(frozen=True)
class KillSwitchState:
    status: KillSwitchStatus
    reason: str | None
    updated_at: str


def model_to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: model_to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {key: model_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [model_to_jsonable(item) for item in value]
    return value

