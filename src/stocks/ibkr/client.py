from __future__ import annotations

from typing import Any

from .callbacks import CallbackState


class ReadOnlyIbkrApp:
    def __init__(self, state: CallbackState) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        class _App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self, callback_state: CallbackState) -> None:
                self.callback_state = callback_state
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                self.callback_state.record_next_valid_id(orderId)
                self.reqCurrentTime()
                self.reqManagedAccts()

            def currentTime(self, unix_time: int) -> None:  # noqa: N802
                self.callback_state.record_current_time(unix_time)

            def managedAccounts(self, accountsList: str | bytes) -> None:  # noqa: N802
                self.callback_state.record_managed_accounts(accountsList)

            def contractDetails(self, reqId: int, contractDetails: Any) -> None:  # noqa: N802
                self.callback_state.record_contract_details(reqId, contractDetails)

            def contractDetailsEnd(self, reqId: int) -> None:  # noqa: N802
                self.callback_state.record_contract_details_end(reqId)

            def symbolSamples(self, reqId: int, contractDescriptions: Any) -> None:  # noqa: N802
                self.callback_state.record_symbol_samples(reqId, contractDescriptions)

            def marketRule(self, marketRuleId: int, priceIncrements: Any) -> None:  # noqa: N802
                self.callback_state.record_market_rule(marketRuleId, priceIncrements)

            def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
                self.callback_state.record_error(reqId, args)

            def connectionClosed(self) -> None:  # noqa: N802
                self.callback_state.record_closed()

        self._app = _App(state)

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


def make_ibkr_app(state: CallbackState) -> ReadOnlyIbkrApp:
    return ReadOnlyIbkrApp(state)
