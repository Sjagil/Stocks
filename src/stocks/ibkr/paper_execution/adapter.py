from __future__ import annotations

import time
from decimal import Decimal
from typing import Any

from stocks.application.config import IbkrSettings
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.callbacks import CallbackState
from stocks.ibkr.connection import ReadOnlyIbkrConnectionService
from stocks.ibkr.paper_execution.models import ManualPaperIntent, PaperWriterConfig


class Phase9IbkrApp:
    """Writer-capable app used only by Phase 9 through the frozen connection service."""

    def __init__(self, state: CallbackState) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        outer = self

        class _App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self, callback_state: CallbackState) -> None:
                self.callback_state = callback_state
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                outer.next_valid_order_id = int(orderId)
                self.callback_state.record_next_valid_id(orderId)
                self.reqCurrentTime()
                self.reqManagedAccts()

            def currentTime(self, unix_time: int) -> None:  # noqa: N802
                self.callback_state.record_current_time(unix_time)

            def managedAccounts(self, accountsList: str | bytes) -> None:  # noqa: N802
                self.callback_state.record_managed_accounts(accountsList)

            def openOrder(self, orderId: int, contract: Any, order: Any, orderState: Any) -> None:  # noqa: N802
                self.callback_state.record_event(
                    "phase9_open_order",
                    order_id_hash="ORDER-ID-OBSERVED",
                    symbol=getattr(contract, "symbol", None),
                    status=getattr(orderState, "status", None),
                )

            def orderStatus(self, orderId: int, status: str, *args: Any) -> None:  # noqa: N802
                self.callback_state.record_event(
                    "phase9_order_status",
                    order_id_hash="ORDER-ID-OBSERVED",
                    status=status,
                )

            def execDetails(self, reqId: int, contract: Any, execution: Any) -> None:  # noqa: N802
                self.callback_state.record_event(
                    "phase9_exec_details",
                    req_id=reqId,
                    exec_id_hash="EXEC-ID-OBSERVED",
                    symbol=getattr(contract, "symbol", None),
                )

            def commissionReport(self, commissionReport: Any) -> None:  # noqa: N802
                self.callback_state.record_event(
                    "phase9_commission_report",
                    exec_id_hash="EXEC-ID-OBSERVED",
                )

            def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
                self.callback_state.record_error(reqId, args)

            def connectionClosed(self) -> None:  # noqa: N802
                self.callback_state.record_closed()

        self._app = _App(state)
        self.callback_state = state
        self.next_valid_order_id: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._app, name)

    def connect(self, host: str, port: int, client_id: int) -> None:
        self._app.connect(host, port, client_id)

    def disconnect(self) -> None:
        self._app.disconnect()

    def isConnected(self) -> bool:  # noqa: N802
        return bool(self._app.isConnected())

    def serverVersion(self) -> Any:  # noqa: N802
        return self._app.serverVersion()

    def twsConnectionTime(self) -> Any:  # noqa: N802
        return self._app.twsConnectionTime()

    def run(self) -> None:
        self._app.run()

    def reqCurrentTime(self) -> None:  # noqa: N802
        self._app.reqCurrentTime()

    def reqManagedAccts(self) -> None:  # noqa: N802
        self._app.reqManagedAccts()

    def reqContractDetails(self, req_id: int, contract: Any) -> None:  # noqa: N802
        self._app.reqContractDetails(req_id, contract)

    def reqMatchingSymbols(self, req_id: int, pattern: str) -> None:  # noqa: N802
        self._app.reqMatchingSymbols(req_id, pattern)

    def reqMarketRule(self, market_rule_id: int) -> None:  # noqa: N802
        self._app.reqMarketRule(market_rule_id)


class PaperExecutionAdapter:
    """Thin marker adapter; real calls are confined to submission/cancellation/order_ids."""

    def __init__(self, app: object) -> None:
        self.app = app


def connect_phase9_writer(config: PaperWriterConfig) -> tuple[ReadOnlyIbkrConnectionService, Phase9IbkrApp | None, dict[str, Any]]:
    writer_app: Phase9IbkrApp | None = None

    def factory(state: CallbackState) -> Phase9IbkrApp:
        nonlocal writer_app
        writer_app = Phase9IbkrApp(state)
        return writer_app

    settings = IbkrSettings(
        host=config.host,
        port=config.port,
        client_id=config.writer_client_id,
        read_only=True,
        order_authority="NONE",
        live_trading_enabled=False,
        allow_order_transmission=False,
        max_order_notional_eur=0,
        max_open_orders=0,
        max_positions=0,
    )
    service = ReadOnlyIbkrConnectionService(settings, app_factory=factory, sleeper=time.sleep)
    snapshot = service.connect_with_retries()
    return service, writer_app, {
        "connection_status": snapshot.status.value,
        "server_version": snapshot.server_version,
        "thread_leak": snapshot.thread_leak,
    }


def build_stock_contract(intent: ManualPaperIntent) -> object:
    from ibapi.contract import Contract

    contract = Contract()
    contract.conId = int(intent.con_id)
    contract.symbol = intent.symbol
    contract.secType = intent.security_type
    contract.exchange = intent.exchange
    contract.currency = intent.currency
    return contract


def build_limit_day_order(intent: ManualPaperIntent) -> object:
    from ibapi.order import Order

    order = Order()
    order.action = intent.side
    order.orderType = "LMT"
    order.totalQuantity = _ib_decimal(intent.quantity)
    order.lmtPrice = float(intent.limit_price)
    order.tif = "DAY"
    order.outsideRth = False
    order.orderRef = "P9-" + stable_hash(
        {"intent_id": intent.intent_id}
    )[:20]
    order.transmit = True
    return order


def _ib_decimal(value: Decimal) -> Decimal | float:
    try:
        from ibapi.common import Decimal as IbDecimal

        return IbDecimal(str(value))
    except Exception:
        return float(value)
