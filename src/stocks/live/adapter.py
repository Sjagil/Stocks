from __future__ import annotations

import threading
from decimal import Decimal, InvalidOperation
from typing import Any

from stocks.ibkr.callbacks import CallbackState
from stocks.live.models import LiveCanaryConfig, ManualLiveBracketIntent


class LiveCanaryApp:
    def __init__(self, state: CallbackState) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        outer = self

        class _App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                outer.next_valid_order_id = int(orderId)
                outer.ready.set()
                state.record_next_valid_id(orderId)

            def currentTime(self, unix_time: int) -> None:  # noqa: N802
                state.record_current_time(unix_time)

            def managedAccounts(self, accountsList: str | bytes) -> None:  # noqa: N802
                state.record_managed_accounts(accountsList)

            def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
                state.record_error(reqId, args)

            def connectionClosed(self) -> None:  # noqa: N802
                state.record_closed()

        self._app = _App()
        self.next_valid_order_id: int | None = None
        self.ready = threading.Event()
        self.thread: threading.Thread | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)

    def connect_and_wait(
        self, config: LiveCanaryConfig
    ) -> dict[str, object]:
        self._app.connect(
            config.host, config.port, config.writer_client_id
        )
        if not self._app.isConnected():
            return {
                "status": "NO_GO",
                "connection_status": "DISCONNECTED",
            }
        self.thread = threading.Thread(
            target=self._app.run,
            name=f"ibkr-live-canary-{config.writer_client_id}",
            daemon=True,
        )
        self.thread.start()
        self._app.reqCurrentTime()
        self._app.reqManagedAccts()
        if not self.ready.wait(config.callback_timeout_seconds):
            return {
                "status": "NO_GO",
                "connection_status": "ORDER_ID_TIMEOUT",
            }
        return {
            "status": "GO",
            "connection_status": "HEALTHY",
            "server_version": self._app.serverVersion(),
        }

    def disconnect(self) -> None:
        if self._app.isConnected():
            self._app.disconnect()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def build_stock_contract(intent: ManualLiveBracketIntent) -> object:
    from ibapi.contract import Contract

    contract = Contract()
    contract.conId = intent.con_id
    contract.symbol = intent.symbol
    contract.secType = "STK"
    contract.exchange = intent.exchange
    contract.currency = intent.currency
    return contract


def build_bracket_orders(
    intent: ManualLiveBracketIntent,
    *,
    parent_order_id: int,
) -> tuple[object, object, object]:
    from ibapi.order import Order

    validate_whole_share_intent(intent)
    quantity = _ib_decimal(intent.quantity)
    parent = Order()
    parent.orderId = parent_order_id
    parent.action = "BUY"
    parent.orderType = "LMT"
    parent.totalQuantity = quantity
    parent.lmtPrice = float(intent.entry_limit_price)
    parent.tif = "DAY"
    parent.outsideRth = False
    parent.transmit = False
    parent.orderRef = intent.intent_id

    stop = Order()
    stop.orderId = parent_order_id + 1
    stop.action = "SELL"
    stop.orderType = "STP"
    stop.totalQuantity = quantity
    stop.auxPrice = float(intent.stop_price)
    stop.tif = "GTC"
    stop.outsideRth = False
    stop.parentId = parent_order_id
    stop.transmit = False
    stop.orderRef = intent.intent_id

    target = Order()
    target.orderId = parent_order_id + 2
    target.action = "SELL"
    target.orderType = "LMT"
    target.totalQuantity = quantity
    target.lmtPrice = float(intent.take_profit_price)
    target.tif = "GTC"
    target.outsideRth = False
    target.parentId = parent_order_id
    target.transmit = True
    target.orderRef = intent.intent_id
    return parent, stop, target


def validate_whole_share_intent(intent: ManualLiveBracketIntent) -> None:
    """Fail before order construction when whole-share evidence is absent."""

    quantity = intent.quantity
    if (
        not quantity.is_finite()
        or quantity < 1
        or quantity != quantity.to_integral_value()
        or intent.fractional_allowed
    ):
        raise ValueError("FRACTIONAL_QUANTITY_FORBIDDEN")
    if intent.asset_class not in {"STOCK", "ETF", "COMMODITY_VEHICLE"}:
        raise ValueError("ASSET_CLASS_POLICY_BLOCKED")
    if intent.capital_level == 1:
        if (
            intent.canary_qty != quantity
            or intent.desired_qty < intent.normal_allowed_qty
            or intent.normal_allowed_qty < intent.canary_qty
            or intent.risk_per_share_eur <= 0
            or intent.planned_total_risk_eur <= 0
            or intent.canary_notional_hard_cap_eur <= 0
        ):
            raise ValueError("WHOLE_SHARE_CANARY_EVIDENCE_REQUIRED")


def order_quantity_is_whole(value: object) -> bool:
    try:
        quantity = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return False
    return bool(
        quantity.is_finite()
        and quantity >= 1
        and quantity == quantity.to_integral_value()
    )


def _ib_decimal(value: object) -> object:
    try:
        from ibapi.common import Decimal as IbDecimal

        return IbDecimal(str(value))
    except Exception:
        return float(str(value))
