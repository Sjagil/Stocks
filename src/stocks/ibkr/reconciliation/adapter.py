from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from stocks.ibkr.reconciliation.callbacks import BrokerObserverState
from stocks.ibkr.reconciliation.errors import FORBIDDEN_METHODS, READ_ONLY_METHODS, Phase8Blocked
from stocks.ibkr.reconciliation.models import ObservationScope
from stocks.ibkr.reconciliation.requests import ACCOUNT_SUMMARY_TAGS, Phase8Config


class BrokerObservationAppProtocol(Protocol):
    def connect(self, host: str, port: int, client_id: int) -> None: ...
    def disconnect(self) -> None: ...
    def isConnected(self) -> bool: ...  # noqa: N802
    def serverVersion(self) -> Any: ...  # noqa: N802
    def run(self) -> None: ...
    def reqCurrentTime(self) -> None: ...  # noqa: N802
    def reqAccountSummary(self, req_id: int, group: str, tags: str) -> None: ...  # noqa: N802
    def cancelAccountSummary(self, req_id: int) -> None: ...  # noqa: N802
    def reqPositions(self) -> None: ...  # noqa: N802
    def cancelPositions(self) -> None: ...  # noqa: N802
    def reqOpenOrders(self) -> None: ...  # noqa: N802
    def reqAllOpenOrders(self) -> None: ...  # noqa: N802
    def reqExecutions(self, req_id: int, execution_filter: Any) -> None: ...  # noqa: N802


class ReadOnlyBrokerObservationApp:
    def __init__(self, state: BrokerObserverState) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        class _App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self, callback_state: BrokerObserverState) -> None:
                self.callback_state = callback_state
                EClient.__init__(self, self)

            def nextValidId(self, _orderId: int) -> None:  # noqa: N802
                self.callback_state.record_connected(self.serverVersion())
                self.reqCurrentTime()

            def currentTime(self, unix_time: int) -> None:  # noqa: N802
                self.callback_state.record_current_time(unix_time)

            def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str) -> None:  # noqa: N802
                self.callback_state.record_account_summary(reqId, account, tag, value, currency)

            def accountSummaryEnd(self, reqId: int) -> None:  # noqa: N802
                self.callback_state.record_account_summary_end(reqId)

            def position(self, account: str, contract: Any, position: Any, avgCost: Any) -> None:  # noqa: N802
                self.callback_state.record_position(account, contract, position, avgCost)

            def positionEnd(self) -> None:  # noqa: N802
                self.callback_state.record_position_end()

            def openOrder(self, orderId: int, contract: Any, order: Any, orderState: Any) -> None:  # noqa: N802
                self.callback_state.record_open_order(orderId, contract, order, orderState)

            def orderStatus(self, orderId: int, status: str, filled: Any, remaining: Any, avgFillPrice: Any, *_args: Any) -> None:  # noqa: N802
                self.callback_state.record_order_status(orderId, status, filled, remaining, avgFillPrice)

            def openOrderEnd(self) -> None:  # noqa: N802
                self.callback_state.record_open_order_end()

            def execDetails(self, reqId: int, contract: Any, execution: Any) -> None:  # noqa: N802
                self.callback_state.record_exec_details(reqId, contract, execution)

            def execDetailsEnd(self, reqId: int) -> None:  # noqa: N802
                self.callback_state.record_exec_details_end(reqId)

            def commissionReport(self, commissionReport: Any) -> None:  # noqa: N802
                self.callback_state.record_commission_report(commissionReport)

            def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
                self.callback_state.record_error(reqId, args)

            def connectionClosed(self) -> None:  # noqa: N802
                self.callback_state.record_closed()

        self._app = _App(state)

    def __getattr__(self, name: str) -> Any:
        if name in FORBIDDEN_METHODS:
            raise Phase8Blocked("BROKER_WRITE_METHOD_BLOCKED", f"{name} is forbidden in Phase 8")
        return getattr(self._app, name)

    def connect(self, host: str, port: int, client_id: int) -> None:
        self._app.connect(host, port, client_id)

    def disconnect(self) -> None:
        self._app.disconnect()

    def isConnected(self) -> bool:  # noqa: N802
        return bool(self._app.isConnected())

    def serverVersion(self) -> Any:  # noqa: N802
        return self._app.serverVersion()

    def run(self) -> None:
        self._app.run()

    def reqCurrentTime(self) -> None:  # noqa: N802
        self._app.reqCurrentTime()

    def reqAccountSummary(self, req_id: int, group: str, tags: str) -> None:  # noqa: N802
        self._app.reqAccountSummary(req_id, group, tags)

    def cancelAccountSummary(self, req_id: int) -> None:  # noqa: N802
        self._app.cancelAccountSummary(req_id)

    def reqPositions(self) -> None:  # noqa: N802
        self._app.reqPositions()

    def cancelPositions(self) -> None:  # noqa: N802
        self._app.cancelPositions()

    def reqOpenOrders(self) -> None:  # noqa: N802
        self._app.reqOpenOrders()

    def reqAllOpenOrders(self) -> None:  # noqa: N802
        self._app.reqAllOpenOrders()

    def reqExecutions(self, req_id: int, execution_filter: Any) -> None:  # noqa: N802
        self._app.reqExecutions(req_id, execution_filter)


AppFactory = Callable[[BrokerObserverState], BrokerObservationAppProtocol]
Sleeper = Callable[[float], None]


@dataclass(frozen=True)
class RequestRun:
    name: str
    started_at: str
    completed_at: str
    callback_count: int
    status: str
    content_hash: str


class BrokerObservationAdapter:
    def __init__(
        self,
        config: Phase8Config,
        *,
        app_factory: AppFactory = ReadOnlyBrokerObservationApp,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.config = config
        self.app_factory = app_factory
        self.sleeper = sleeper
        self.state = BrokerObserverState(config.account_fingerprint_key)
        self.app: BrokerObservationAppProtocol | None = None
        self.thread: threading.Thread | None = None
        self.read_counters = zero_read_counters()
        self.write_counters = zero_write_counters()

    def capture(self) -> BrokerObserverState:
        self.state = BrokerObserverState(self.config.account_fingerprint_key)
        self.app = self.app_factory(self.state)
        started_account = False
        started_positions = False
        try:
            self.app.connect(self.config.host, self.config.port, self.config.recon_client_id)
            if not self.app.isConnected():
                self.state.errors.append({"code": "CONNECTION_LOST", "message": "observer socket not connected"})
                return self.state
            self.state.record_connected(self.app.serverVersion())
            self.thread = threading.Thread(target=self.app.run, name=f"ibkr-phase8-observer-{self.config.recon_client_id}", daemon=True)
            self.thread.start()
            self.app.reqCurrentTime()
            self.state.current_time_event.wait(self.config.request_timeout_seconds)

            req_id = 8101
            started_account = True
            self.read_counters["read_only_account_summary_requests"] += 1
            self.app.reqAccountSummary(req_id, "All", ",".join(ACCOUNT_SUMMARY_TAGS))
            self._wait(self.state.account_summary_end, "accountsummary")

            started_positions = True
            self.read_counters["read_only_position_requests"] += 1
            self.app.reqPositions()
            self._wait(self.state.position_end, "positions")

            self.state.current_open_order_scope = ObservationScope.SAME_CLIENT
            self.state.same_client_open_order_end.clear()
            self.read_counters["read_only_same_client_open_order_requests"] += 1
            self.app.reqOpenOrders()
            self._wait(
                self.state.same_client_open_order_end,
                "same-client open orders",
            )

            self.state.current_open_order_scope = ObservationScope.ALL_API_CLIENTS
            self.state.all_api_open_order_end.clear()
            self.read_counters["read_only_all_api_open_order_requests"] += 1
            self.app.reqAllOpenOrders()
            self._wait(
                self.state.all_api_open_order_end,
                "all-api open orders",
            )

            self.read_counters["read_only_execution_requests"] += 1
            self._req_executions(8201)
            self._wait(self.state.exec_details_end, "executions")
            self.sleeper(self.config.commission_grace_seconds)
        except Phase8Blocked:
            raise
        except Exception as exc:
            self.state.errors.append({"code": type(exc).__name__, "message": str(exc)})
        finally:
            if self.app is not None:
                if started_account:
                    try:
                        self.read_counters["read_only_account_summary_cancels"] += 1
                        self.app.cancelAccountSummary(8101)
                    except Exception as exc:
                        self.state.errors.append({"code": "PROVIDER_ERROR", "message": str(exc)})
                if started_positions:
                    try:
                        self.read_counters["read_only_position_cancels"] += 1
                        self.app.cancelPositions()
                    except Exception as exc:
                        self.state.errors.append({"code": "PROVIDER_ERROR", "message": str(exc)})
                try:
                    self.app.disconnect()
                except Exception as exc:
                    self.state.errors.append({"code": "CONNECTION_LOST", "message": str(exc)})
            if self.thread is not None:
                self.thread.join(timeout=2.0)
            self.state.record_closed()
        return self.state

    def _wait(self, event: threading.Event, name: str) -> None:
        if not event.wait(self.config.request_timeout_seconds):
            self.state.errors.append({"code": "CALLBACK_TIMEOUT", "message": f"{name} did not complete"})

    def _req_executions(self, req_id: int) -> None:
        try:
            from ibapi.execution import ExecutionFilter
        except Exception:
            class ExecutionFilter:  # type: ignore[no-redef]
                pass

        self.app.reqExecutions(req_id, ExecutionFilter())  # type: ignore[union-attr]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def enforce_method_allowed(method_name: str) -> str:
    if method_name in FORBIDDEN_METHODS:
        raise Phase8Blocked("BROKER_WRITE_METHOD_BLOCKED", f"{method_name} is forbidden")
    if method_name not in READ_ONLY_METHODS:
        raise Phase8Blocked("READ_ONLY_METHOD_ALLOWLIST_VIOLATION", f"{method_name} is not in Phase 8 allowlist")
    return "READ_ONLY_ALLOWED"


def zero_read_counters() -> dict[str, int]:
    return {
        "read_only_account_summary_requests": 0,
        "read_only_account_summary_cancels": 0,
        "read_only_position_requests": 0,
        "read_only_position_cancels": 0,
        "read_only_same_client_open_order_requests": 0,
        "read_only_all_api_open_order_requests": 0,
        "read_only_execution_requests": 0,
    }


def zero_write_counters() -> dict[str, int]:
    return {
        "place_order_calls": 0,
        "cancel_order_calls": 0,
        "global_cancel_calls": 0,
        "request_order_id_calls": 0,
        "auto_bind_order_calls": 0,
        "exercise_option_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
    }
