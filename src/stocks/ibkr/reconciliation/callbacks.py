from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from stocks.execution.idempotency import stable_hash
from stocks.ibkr.reconciliation.errors import Phase8Blocked
from stocks.ibkr.reconciliation.masking import account_fingerprint, contains_raw_account, hash_optional_text
from stocks.ibkr.reconciliation.models import (
    BrokerAccountValue,
    BrokerCommission,
    BrokerExecution,
    BrokerOpenOrder,
    BrokerPosition,
    ObservationScope,
)
from stocks.ibkr.reconciliation.normalizer import contract_hash, decimal_or_none, decimal_value, safe_attr


@dataclass
class BrokerObserverState:
    fingerprint_key: str
    current_open_order_scope: ObservationScope = ObservationScope.SAME_CLIENT
    account_values: list[BrokerAccountValue] = field(default_factory=list)
    positions: list[BrokerPosition] = field(default_factory=list)
    same_client_open_orders: list[BrokerOpenOrder] = field(default_factory=list)
    all_api_open_orders: list[BrokerOpenOrder] = field(default_factory=list)
    executions: list[BrokerExecution] = field(default_factory=list)
    commissions: list[BrokerCommission] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    server_time: str | None = None
    server_version: str | None = None
    connected: bool = False
    account_summary_end: threading.Event = field(default_factory=threading.Event)
    position_end: threading.Event = field(default_factory=threading.Event)
    same_client_open_order_end: threading.Event = field(
        default_factory=threading.Event
    )
    all_api_open_order_end: threading.Event = field(
        default_factory=threading.Event
    )
    exec_details_end: threading.Event = field(default_factory=threading.Event)
    current_time_event: threading.Event = field(default_factory=threading.Event)

    @property
    def open_order_end(self) -> threading.Event:
        """Compatibility view; storage remains split by request scope."""
        if self.current_open_order_scope == ObservationScope.SAME_CLIENT:
            return self.same_client_open_order_end
        return self.all_api_open_order_end

    def observed_at(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def record_connected(self, server_version: Any) -> None:
        self.server_version = None if server_version is None else str(server_version)
        self.connected = True

    def record_current_time(self, unix_time: int) -> None:
        self.server_time = datetime.fromtimestamp(int(unix_time), tz=timezone.utc).isoformat()
        self.current_time_event.set()

    def record_account_summary(self, req_id: int, account: str, tag: str, value: str, currency: str) -> None:
        try:
            fp = account_fingerprint(account, self.fingerprint_key)
            payload_value = decimal_or_none(value)
            self.account_values.append(
                BrokerAccountValue(
                    account_fingerprint=fp,
                    tag=str(tag),
                    value=value if payload_value is None else payload_value,
                    currency=str(currency or ""),
                    observed_at=self.observed_at(),
                    request_id=int(req_id),
                    source=(
                        "IBKR_ACCOUNT_SUMMARY_LEDGER"
                        if str(tag).startswith("$LEDGER-")
                        else "IBKR_ACCOUNT_SUMMARY"
                    ),
                )
            )
        except Exception as exc:
            self.errors.append({"code": "ACCOUNT_MASKING_FAILURE", "message": str(exc)})
            raise Phase8Blocked("ACCOUNT_MASKING_FAILURE") from exc
        self._assert_no_raw_account()

    def record_account_summary_end(self, _req_id: int) -> None:
        self.account_summary_end.set()

    def record_position(self, account: str, contract: Any, position: Any, average_cost: Any) -> None:
        try:
            fp = account_fingerprint(account, self.fingerprint_key)
            con_id = int(safe_attr(contract, "conId", "con_id", default=0) or 0)
            symbol = str(safe_attr(contract, "symbol", default=""))
            security_type = str(safe_attr(contract, "secType", "security_type", default=""))
            currency = str(safe_attr(contract, "currency", default=""))
            exchange = str(safe_attr(contract, "exchange", default=""))
            self.positions.append(
                BrokerPosition(
                    account_fingerprint=fp,
                    con_id=con_id,
                    symbol=symbol,
                    security_type=security_type,
                    currency=currency,
                    exchange=exchange,
                    position_quantity=decimal_value(position),
                    average_cost=decimal_value(average_cost),
                    observed_at=self.observed_at(),
                    contract_hash=contract_hash(
                        con_id=con_id,
                        symbol=symbol,
                        security_type=security_type,
                        currency=currency,
                        exchange=exchange,
                    ),
                )
            )
        except Exception as exc:
            self.errors.append({"code": "ACCOUNT_MASKING_FAILURE", "message": str(exc)})
            raise Phase8Blocked("ACCOUNT_MASKING_FAILURE") from exc
        self._assert_no_raw_account()

    def record_position_end(self) -> None:
        self.position_end.set()

    def record_open_order(self, order_id: int, contract: Any, order: Any, order_state: Any) -> None:
        scope = (
            self.current_open_order_scope
            if isinstance(self.current_open_order_scope, ObservationScope)
            else ObservationScope(str(self.current_open_order_scope))
        )
        con_id = int(safe_attr(contract, "conId", "con_id", default=0) or 0)
        symbol = str(safe_attr(contract, "symbol", default=""))
        order_ref = safe_attr(order, "orderRef", default=None)
        open_order = BrokerOpenOrder(
            observation_scope=scope,
            broker_order_id=stable_hash({"broker_order_id": int(order_id), "scope": scope.value})[:24],
            perm_id=str(safe_attr(order, "permId", default="")),
            client_id=int(safe_attr(order, "clientId", default=0) or 0),
            order_ref_hash=hash_optional_text(order_ref, self.fingerprint_key),
            con_id=con_id,
            symbol=symbol,
            security_type=str(safe_attr(contract, "secType", default="")),
            currency=str(safe_attr(contract, "currency", default="")),
            action=str(safe_attr(order, "action", default="")),
            total_quantity=decimal_value(safe_attr(order, "totalQuantity", default="0")),
            order_type=str(safe_attr(order, "orderType", default="")),
            limit_price=decimal_or_none(safe_attr(order, "lmtPrice", default=None)),
            aux_price=decimal_or_none(safe_attr(order, "auxPrice", default=None)),
            time_in_force=str(safe_attr(order, "tif", default="")),
            outside_rth=bool(safe_attr(order, "outsideRth", default=False)),
            order_status=str(safe_attr(order_state, "status", default="")),
            filled_quantity=decimal_value(safe_attr(order_state, "filled", default="0")),
            remaining_quantity=decimal_value(safe_attr(order_state, "remaining", default="0")),
            average_fill_price=decimal_or_none(safe_attr(order_state, "avgFillPrice", default=None)),
            parent_id=None if safe_attr(order, "parentId", default=0) in {0, "", None} else str(safe_attr(order, "parentId", default="")),
            observed_at=self.observed_at(),
        )
        target = self.same_client_open_orders if scope == ObservationScope.SAME_CLIENT else self.all_api_open_orders
        if open_order.broker_order_id not in {item.broker_order_id for item in target}:
            target.append(open_order)
        else:
            self.errors.append({"code": "DUPLICATE_CALLBACK", "message": "duplicate openOrder callback"})
        self._assert_no_raw_account()

    def record_order_status(self, order_id: int, status: str, filled: Any, remaining: Any, avg_fill_price: Any) -> None:
        target = self.same_client_open_orders if self.current_open_order_scope == ObservationScope.SAME_CLIENT else self.all_api_open_orders
        hashed = stable_hash({"broker_order_id": int(order_id), "scope": self.current_open_order_scope.value})[:24]
        for index, item in enumerate(target):
            if item.broker_order_id == hashed:
                target[index] = BrokerOpenOrder(
                    **{
                        **item.__dict__,
                        "order_status": str(status),
                        "filled_quantity": decimal_value(filled),
                        "remaining_quantity": decimal_value(remaining),
                        "average_fill_price": decimal_or_none(avg_fill_price),
                    }
                )
                return

    def record_open_order_end(self) -> None:
        if self.current_open_order_scope == ObservationScope.SAME_CLIENT:
            self.same_client_open_order_end.set()
        else:
            self.all_api_open_order_end.set()

    def record_exec_details(self, req_id: int, contract: Any, execution: Any) -> None:
        raw_account = str(safe_attr(execution, "acctNumber", "acct_number", default=""))
        try:
            fp = account_fingerprint(raw_account, self.fingerprint_key)
        except Exception as exc:
            self.errors.append({"code": "ACCOUNT_MASKING_FAILURE", "message": str(exc)})
            raise Phase8Blocked("ACCOUNT_MASKING_FAILURE") from exc
        exec_id = str(safe_attr(execution, "execId", default=""))
        con_id = int(safe_attr(contract, "conId", default=0) or 0)
        self.executions.append(
            BrokerExecution(
                execution_id=stable_hash({"exec_id": exec_id})[:32],
                execution_revision_key=stable_hash({"exec_id": exec_id, "req_id": req_id}),
                broker_order_id=stable_hash({"broker_order_id": safe_attr(execution, "orderId", default="")})[:24],
                perm_id=str(safe_attr(execution, "permId", default="")),
                client_id=int(safe_attr(execution, "clientId", default=0) or 0),
                account_fingerprint=fp,
                con_id=con_id,
                symbol=str(safe_attr(contract, "symbol", default="")),
                security_type=str(safe_attr(contract, "secType", default="")),
                side=str(safe_attr(execution, "side", default="")),
                quantity=decimal_value(safe_attr(execution, "shares", default="0")),
                price=decimal_value(safe_attr(execution, "price", default="0")),
                cumulative_quantity=decimal_value(safe_attr(execution, "cumQty", default="0")),
                average_price=decimal_value(safe_attr(execution, "avgPrice", default="0")),
                exchange=str(safe_attr(execution, "exchange", default="")),
                execution_time=str(safe_attr(execution, "time", default="")),
                liquidation=int(safe_attr(execution, "liquidation", default=0) or 0),
                order_ref_hash=hash_optional_text(safe_attr(execution, "orderRef", default=None), self.fingerprint_key),
                observed_at=self.observed_at(),
            )
        )
        self._assert_no_raw_account()

    def record_exec_details_end(self, _req_id: int) -> None:
        self.exec_details_end.set()

    def record_commission_report(self, commission_report: Any) -> None:
        exec_id = str(safe_attr(commission_report, "execId", default=""))
        self.commissions.append(
            BrokerCommission(
                execution_id=stable_hash({"exec_id": exec_id})[:32],
                commission=decimal_value(safe_attr(commission_report, "commission", default="0")),
                currency=str(safe_attr(commission_report, "currency", default="")),
                realized_pnl=decimal_or_none(safe_attr(commission_report, "realizedPNL", default=None)),
                yield_value=decimal_or_none(safe_attr(commission_report, "yield_", "yield", default=None)),
                yield_redemption_date=None
                if safe_attr(commission_report, "yieldRedemptionDate", default=0) in {0, "", None}
                else int(safe_attr(commission_report, "yieldRedemptionDate", default=0)),
                observed_at=self.observed_at(),
            )
        )

    def record_error(self, req_id: Any, args: tuple[Any, ...]) -> None:
        code = args[0] if args else req_id
        message = args[1] if len(args) > 1 else ""
        severity = "INFO" if code in {2104, 2106, 2158} else "ERROR"
        self.errors.append({"req_id": req_id, "code": code, "message": str(message), "severity": severity})

    def record_closed(self) -> None:
        self.connected = False

    def _assert_no_raw_account(self) -> None:
        if contains_raw_account(
            {
                "account_values": self.account_values,
                "positions": self.positions,
                "executions": self.executions,
                "commissions": self.commissions,
            }
        ):
            self.errors.append({"code": "RAW_ACCOUNT_LEAK_BLOCKED", "message": "raw account detected past callback boundary"})
            raise Phase8Blocked("RAW_ACCOUNT_LEAK_BLOCKED")
