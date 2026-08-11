from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any


class SignalAction(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCHLIST = "WATCHLIST"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    AVOID = "AVOID"
    NO_SIGNAL = "NO_SIGNAL"


class SignalLifecycle(StrEnum):
    GENERATED = "GENERATED"
    VALIDATED = "VALIDATED"
    MANUAL_ACTIONABLE = "MANUAL_ACTIONABLE"
    WATCHLIST = "WATCHLIST"
    TRIGGERED = "TRIGGERED"
    PARTIALLY_EXECUTED_MANUALLY = "PARTIALLY_EXECUTED_MANUALLY"
    EXECUTED_MANUALLY = "EXECUTED_MANUALLY"
    INVALIDATED = "INVALIDATED"
    STOPPED_OUT = "STOPPED_OUT"
    TP1_REACHED = "TP1_REACHED"
    TP2_REACHED = "TP2_REACHED"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class SignalPlan:
    signal_id: str
    asset: str
    ticker: str
    contract_identity: dict[str, Any]
    asset_class: str
    exchange: str
    currency: str
    signal_timestamp: datetime
    data_timestamp: datetime
    data_freshness: str
    strategy_id: str
    strategy_dna_hash: str
    timeframe: str
    higher_timeframe_context: str
    action: SignalAction
    current_market_price: Decimal
    preferred_entry: Decimal
    entry_zone_low: Decimal
    entry_zone_high: Decimal
    limit_entry_price: Decimal
    invalidation_level: Decimal
    stop_loss: Decimal
    stop_method: str
    stop_distance_pct: Decimal
    stop_distance_atr: Decimal
    take_profit_1: Decimal
    take_profit_2: Decimal
    take_profit_mode: str
    reward_risk_1: Decimal
    reward_risk_2: Decimal
    suggested_quantity: Decimal
    maximum_order_value_eur: Decimal
    maximum_planned_loss_eur: Decimal
    estimated_transaction_costs_eur: Decimal
    expected_holding_period: str
    confidence_score: Decimal
    regime: str
    reasons: tuple[str, ...]
    risks: tuple[str, ...]
    expiration_timestamp: datetime
    lifecycle_status: SignalLifecycle
    automatic_execution_allowed: bool
    source_provider: str
    source_interval: str
    bar_origin: str
    bar_closed: bool
    exchange_timezone: str
    signal_freshness_basis: str
    broker_calls: int = 0
    orders_generated: int = 0

    def public_payload(self) -> dict[str, Any]:
        return asdict(self)
