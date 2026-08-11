from __future__ import annotations

import threading
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from stocks.live.models import LiveCanaryConfig


_BID = 1
_ASK = 2
_LAST = 4
_CLOSE = 9
_US_EXCHANGES = {"NASDAQ", "NYSE", "ARCA", "BATS", "IEX", "AMEX"}


class LiveQuoteApp:
    def __init__(self) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        outer = self

        class _App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                outer.ready.set()

            def tickPrice(  # noqa: N802
                self,
                reqId: int,
                tickType: int,
                price: float,
                attrib: Any,
            ) -> None:
                if price > 0:
                    outer.prices.setdefault(int(reqId), {})[
                        int(tickType)
                    ] = Decimal(str(price))

            def tickSnapshotEnd(self, reqId: int) -> None:  # noqa: N802
                outer.completed.setdefault(int(reqId), threading.Event()).set()

            def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
                outer.errors.append(
                    {"request_id": str(reqId), "detail": str(args)[:300]}
                )

            def connectionClosed(self) -> None:  # noqa: N802
                outer.closed = True

        self._app = _App()
        self.ready = threading.Event()
        self.prices: dict[int, dict[int, Decimal]] = {}
        self.completed: dict[int, threading.Event] = {}
        self.errors: list[dict[str, str]] = []
        self.closed = False
        self.thread: threading.Thread | None = None

    def connect(self, config: LiveCanaryConfig) -> dict[str, Any]:
        self._app.connect(config.host, config.port, config.quote_client_id)
        if not self._app.isConnected():
            return {"status": "NO_GO", "reason": "QUOTE_CLIENT_DISCONNECTED"}
        self.thread = threading.Thread(
            target=self._app.run,
            name=f"ibkr-live-quote-{config.quote_client_id}",
            daemon=True,
        )
        self.thread.start()
        if not self.ready.wait(config.callback_timeout_seconds):
            return {"status": "NO_GO", "reason": "QUOTE_CLIENT_READY_TIMEOUT"}
        return {"status": "GO"}

    def snapshot(
        self,
        request_id: int,
        contract: object,
        *,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.completed[request_id] = threading.Event()
        self._app.reqMarketDataType(1)
        self._app.reqMktData(
            request_id,
            contract,
            "",
            True,
            False,
            [],
        )
        complete = self.completed[request_id].wait(timeout_seconds)
        prices = self.prices.get(request_id, {})
        return {
            "status": "GO" if complete else "NO_GO",
            "complete": complete,
            "bid": prices.get(_BID),
            "ask": prices.get(_ASK),
            "last": prices.get(_LAST),
            "close": prices.get(_CLOSE),
            "market_data_calls": 1,
        }

    def disconnect(self) -> None:
        if self._app.isConnected():
            self._app.disconnect()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def capture_live_quote(
    config: LiveCanaryConfig,
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    app = LiveQuoteApp()
    connection = app.connect(config)
    if connection["status"] != "GO":
        return {
            **connection,
            "market_data_calls": 0,
            "broker_write_calls": 0,
        }
    calls = 0
    try:
        stock = app.snapshot(
            9_100_001,
            _stock_contract(contract_data),
            timeout_seconds=config.callback_timeout_seconds,
        )
        calls += int(stock["market_data_calls"])
        fx_rate = Decimal("1")
        currency = str(contract_data.get("currency", "")).upper()
        fx_status = "NOT_REQUIRED"
        if currency == "USD":
            fx = app.snapshot(
                9_100_002,
                _eur_usd_contract(),
                timeout_seconds=config.callback_timeout_seconds,
            )
            calls += int(fx["market_data_calls"])
            eur_usd_bid = fx.get("bid")
            if fx["status"] != "GO" or not _positive(eur_usd_bid):
                fx_status = "FX_QUOTE_INCOMPLETE"
            else:
                fx_rate = Decimal("1") / Decimal(str(eur_usd_bid))
                fx_status = "GO"
        elif currency != "EUR":
            fx_status = "UNSUPPORTED_LIVE_CURRENCY"
        quote = {
            "status": (
                "GO"
                if stock["status"] == "GO"
                and _positive(stock.get("bid"))
                and _positive(stock.get("ask"))
                and fx_status in {"GO", "NOT_REQUIRED"}
                else "NO_GO"
            ),
            "bid": stock.get("bid"),
            "ask": stock.get("ask"),
            "last": stock.get("last"),
            "close": stock.get("close"),
            "fx_rate_to_eur": fx_rate,
            "fx_status": fx_status,
            "captured_at": datetime.now(UTC).isoformat(),
            "market_data_calls": calls,
            "broker_write_calls": 0,
            "errors_masked": len(app.errors),
        }
        return {**quote, **validate_quote(quote)}
    finally:
        app.disconnect()


def validate_quote(
    quote: dict[str, Any],
    *,
    maximum_spread_fraction: Decimal = Decimal("0.005"),
) -> dict[str, Any]:
    bid = _decimal(quote.get("bid"))
    ask = _decimal(quote.get("ask"))
    blockers = []
    spread_fraction: Decimal | None = None
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        blockers.append("INCOMPLETE_OR_CROSSED_LIVE_QUOTE")
    else:
        midpoint = (bid + ask) / Decimal("2")
        spread_fraction = (ask - bid) / midpoint
        if spread_fraction > maximum_spread_fraction:
            blockers.append("LIVE_SPREAD_TOO_WIDE")
    return {
        "quote_validation_status": "GO" if not blockers else "NO_GO",
        "quote_blockers": blockers,
        "spread_fraction": (
            None if spread_fraction is None else str(spread_fraction)
        ),
    }


def regular_session_open(
    primary_exchange: str,
    at: datetime | None = None,
) -> dict[str, Any]:
    exchange = primary_exchange.upper()
    if exchange not in _US_EXCHANGES:
        return {
            "status": "NO_GO",
            "session_status": "UNSUPPORTED_LIVE_SESSION_CALENDAR",
        }
    import exchange_calendars as exchange_calendars
    import pandas as pd

    now = at or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    calendar = exchange_calendars.get_calendar("XNYS")
    minute = pd.Timestamp(now)
    if not calendar.is_open_on_minute(minute):
        return {"status": "NO_GO", "session_status": "MARKET_CLOSED"}
    session = calendar.minute_to_session(minute, direction="none")
    market_open = calendar.session_open(session).to_pydatetime()
    market_close = calendar.session_close(session).to_pydatetime()
    return {
        "status": "GO",
        "session_status": "REGULAR_SESSION_OPEN",
        "market_open_utc": market_open.isoformat(),
        "market_close_utc": market_close.isoformat(),
    }


def _stock_contract(data: dict[str, Any]) -> object:
    from ibapi.contract import Contract

    contract = Contract()
    contract.conId = int(data["con_id"])
    contract.symbol = str(data["symbol"])
    contract.secType = "STK"
    contract.exchange = str(data.get("exchange", "SMART"))
    contract.currency = str(data["currency"])
    return contract


def _eur_usd_contract() -> object:
    from ibapi.contract import Contract

    contract = Contract()
    contract.symbol = "EUR"
    contract.secType = "CASH"
    contract.exchange = "IDEALPRO"
    contract.currency = "USD"
    return contract


def _positive(value: Any) -> bool:
    parsed = _decimal(value)
    return parsed is not None and parsed > 0


def _decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value))
        return parsed if parsed.is_finite() else None
    except Exception:
        return None
