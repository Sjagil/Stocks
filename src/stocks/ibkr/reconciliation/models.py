from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any


class ObservationScope(StrEnum):
    SAME_CLIENT = "SAME_CLIENT"
    ALL_API_CLIENTS = "ALL_API_CLIENTS"


class RequestStatus(StrEnum):
    COMPLETE = "COMPLETE"
    EMPTY_COMPLETE = "EMPTY_COMPLETE"
    CALLBACK_TIMEOUT = "CALLBACK_TIMEOUT"
    CONNECTION_LOST = "CONNECTION_LOST"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    MASKING_FAILURE = "MASKING_FAILURE"
    PARTIAL_RESPONSE_BLOCKED = "PARTIAL_RESPONSE_BLOCKED"


@dataclass(frozen=True)
class BrokerAccountValue:
    account_fingerprint: str
    tag: str
    value: Decimal | str
    currency: str
    observed_at: str
    request_id: int
    source: str


@dataclass(frozen=True)
class BrokerAccountSnapshot:
    values: tuple[BrokerAccountValue, ...]
    status: str
    content_hash: str


@dataclass(frozen=True)
class BrokerPosition:
    account_fingerprint: str
    con_id: int
    symbol: str
    security_type: str
    currency: str
    exchange: str
    position_quantity: Decimal
    average_cost: Decimal
    observed_at: str
    contract_hash: str


@dataclass(frozen=True)
class BrokerPositionSnapshot:
    positions: tuple[BrokerPosition, ...]
    status: str
    content_hash: str


@dataclass(frozen=True)
class BrokerOpenOrder:
    observation_scope: ObservationScope
    broker_order_id: str
    perm_id: str
    client_id: int
    order_ref_hash: str | None
    con_id: int
    symbol: str
    security_type: str
    currency: str
    action: str
    total_quantity: Decimal
    order_type: str
    limit_price: Decimal | None
    aux_price: Decimal | None
    time_in_force: str
    outside_rth: bool
    order_status: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    average_fill_price: Decimal | None
    parent_id: str | None
    observed_at: str


@dataclass(frozen=True)
class BrokerOpenOrderSnapshot:
    scope: ObservationScope
    open_orders: tuple[BrokerOpenOrder, ...]
    status: str
    content_hash: str


@dataclass(frozen=True)
class BrokerExecution:
    execution_id: str
    execution_revision_key: str
    broker_order_id: str
    perm_id: str
    client_id: int
    account_fingerprint: str
    con_id: int
    symbol: str
    security_type: str
    side: str
    quantity: Decimal
    price: Decimal
    cumulative_quantity: Decimal
    average_price: Decimal
    exchange: str
    execution_time: str
    liquidation: int
    order_ref_hash: str | None
    observed_at: str


@dataclass(frozen=True)
class BrokerCommission:
    execution_id: str
    commission: Decimal
    currency: str
    realized_pnl: Decimal | None
    yield_value: Decimal | None
    yield_redemption_date: int | None
    observed_at: str


@dataclass(frozen=True)
class BrokerExecutionSnapshot:
    executions: tuple[BrokerExecution, ...]
    commissions: tuple[BrokerCommission, ...]
    status: str
    execution_scope: str
    request_filter: str
    requested_from: str | None
    requested_until: str
    tws_trade_log_scope_known: bool
    execution_history_complete: bool
    completeness_status: str
    content_hash: str


@dataclass(frozen=True)
class SnapshotComponentAudit:
    name: str
    started_at: str
    completed_at: str
    callback_count: int
    timeout_seconds: Decimal
    request_status: str
    content_hash: str


@dataclass(frozen=True)
class BrokerObservationSnapshot:
    snapshot_id: str
    snapshot_started_at: str
    snapshot_completed_at: str
    snapshot_span_seconds: Decimal
    component_timestamps: dict[str, dict[str, str]]
    snapshot_atomic: bool
    account: BrokerAccountSnapshot
    positions: BrokerPositionSnapshot
    same_client_open_orders: BrokerOpenOrderSnapshot
    all_api_open_orders: BrokerOpenOrderSnapshot
    executions: BrokerExecutionSnapshot
    component_audits: tuple[SnapshotComponentAudit, ...]
    server_version: str | None
    broker_observation_authority: str
    execution_authority: str
    content_hash: str


@dataclass(frozen=True)
class BrokerSnapshotManifest:
    snapshot_id: str
    public_summary: dict[str, Any]
    private_snapshot_reference: str
    private_snapshot_hash: str


def model_to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: model_to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(key): model_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [model_to_jsonable(item) for item in value]
    return value
