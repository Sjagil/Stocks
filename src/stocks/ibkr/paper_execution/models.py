from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class PaperWriterConfig:
    host: str
    port: int
    phase1_client_id: int
    observer_client_id: int
    writer_client_id: int
    writer_enabled: bool
    approved_account_fingerprint: str
    observed_account_fingerprint: str
    max_order_notional_eur: Decimal
    max_quantity: Decimal
    max_open_orders: int
    max_positions: int
    max_new_orders_per_day: int
    max_closing_orders_per_day: int
    approval_ttl_seconds: int
    callback_timeout_seconds: float
    reconciliation_timeout_seconds: float
    live_trading_enabled: bool
    allow_order_transmission: bool

    def safe_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "phase1_client_id": self.phase1_client_id,
            "observer_client_id_masked": _masked_int(self.observer_client_id),
            "writer_client_id_masked": _masked_int(self.writer_client_id),
            "writer_client_id_nonzero": self.writer_client_id != 0,
            "writer_enabled": self.writer_enabled,
            "account_fingerprint_match": self.approved_account_fingerprint == self.observed_account_fingerprint,
            "max_order_notional_eur": str(self.max_order_notional_eur),
            "max_quantity": str(self.max_quantity),
            "max_open_orders": self.max_open_orders,
            "max_positions": self.max_positions,
            "max_new_orders_per_day": self.max_new_orders_per_day,
            "max_closing_orders_per_day": self.max_closing_orders_per_day,
            "approval_ttl_seconds": self.approval_ttl_seconds,
            "callback_timeout_seconds": self.callback_timeout_seconds,
            "reconciliation_timeout_seconds": self.reconciliation_timeout_seconds,
            "live_trading_enabled": self.live_trading_enabled,
            "allow_order_transmission": self.allow_order_transmission,
        }


@dataclass(frozen=True)
class ManualPaperIntent:
    intent_id: str
    economic_order_key: str
    intent_source: str
    created_at: str
    expires_at: str
    account_fingerprint: str
    con_id: int
    symbol: str
    security_type: str
    currency: str
    exchange: str
    side: str
    quantity: Decimal
    order_type: str
    limit_price: Decimal
    estimated_notional_local: Decimal
    estimated_notional_eur: Decimal
    fx_rate: Decimal
    fx_rate_timestamp: str
    session_date: str
    outside_rth: bool
    time_in_force: str
    contract_hash: str
    operator_reason: str


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    intent_id: str
    intent_hash: str
    challenge_hash: str
    expires_at: str
    used: bool
    approval_type: str


@dataclass(frozen=True)
class PaperOrderEvent:
    intent_id: str
    event_type: str
    status: str
    created_at: str
    payload_hash: str


def model_to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return model_to_jsonable(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): model_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [model_to_jsonable(item) for item in value]
    return value


def _masked_int(value: int) -> str:
    text = str(value)
    return f"ID-{text[-2:].rjust(len(text), '*')}"
