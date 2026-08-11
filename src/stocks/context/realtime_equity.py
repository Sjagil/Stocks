from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import dotenv_values

from stocks.execution.idempotency import stable_hash


PRIVATE_ROOT = Path("data/market_context/private")
OUTPUT_ROOT = Path("output/market_context")


@dataclass(frozen=True)
class RealtimeEquityConfig:
    duration_seconds: float = 15.0
    max_symbols: int = 10
    depth_symbols: int = 5
    depth_levels: int = 5

    def validate(self) -> None:
        if not 2.0 <= float(self.duration_seconds) <= 60.0:
            raise ValueError("duration_seconds must be between 2 and 60")
        if not 1 <= int(self.max_symbols) <= 10:
            raise ValueError("max_symbols must be between 1 and 10")
        if not 0 <= int(self.depth_symbols) <= min(5, int(self.max_symbols)):
            raise ValueError("depth_symbols must be between 0 and min(5, max_symbols)")
        if not 1 <= int(self.depth_levels) <= 10:
            raise ValueError("depth_levels must be between 1 and 10")


@dataclass(frozen=True)
class MarketDataEndpoint:
    host: str
    port: int
    client_id: int
    market_data_type: int
    connect_timeout_seconds: float

    @property
    def environment(self) -> str:
        return "LIVE_MARKET_DATA_ONLY" if self.port in {7496, 4001} else "PAPER_MARKET_DATA"


class EquityMarketDataApp:
    def __init__(self) -> None:
        from ibapi.client import EClient
        from ibapi.wrapper import EWrapper

        outer = self

        class _App(EWrapper, EClient):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                EClient.__init__(self, self)

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                outer.ready.set()

            def tickPrice(self, reqId: int, tickType: int, price: float, attrib: Any) -> None:  # noqa: N802
                if price <= 0:
                    return
                symbol = outer.request_symbols.get(int(reqId))
                if symbol and int(tickType) in {1, 2, 4, 9}:
                    outer.quotes.setdefault(symbol, {})[int(tickType)] = float(price)

            def tickSize(self, reqId: int, tickType: int, size: Any) -> None:  # noqa: N802
                symbol = outer.request_symbols.get(int(reqId))
                if symbol and int(tickType) in {0, 3, 5}:
                    outer.quote_sizes.setdefault(symbol, {})[int(tickType)] = _float(size)

            def tickByTickAllLast(  # noqa: N802
                self,
                reqId: int,
                tickType: int,
                epoch: int,
                price: float,
                size: Any,
                tickAttribLast: Any,
                exchange: str,
                specialConditions: str,
            ) -> None:
                symbol = outer.request_symbols.get(int(reqId))
                if not symbol or price <= 0 or _float(size) <= 0:
                    return
                quote = outer.quotes.get(symbol, {})
                outer.trades.append(
                    {
                        "timestamp": datetime.fromtimestamp(int(epoch), UTC).isoformat(),
                        "symbol": symbol,
                        "price": float(price),
                        "size": _float(size),
                        "bid": quote.get(1),
                        "ask": quote.get(2),
                        "exchange": str(exchange or "")[:32],
                        "source": "IBKR_TICK_BY_TICK_ALL_LAST",
                    }
                )

            def updateMktDepth(  # noqa: N802
                self,
                reqId: int,
                position: int,
                operation: int,
                side: int,
                price: float,
                size: Any,
            ) -> None:
                outer.update_depth(reqId, position, operation, side, price, size)

            def updateMktDepthL2(  # noqa: N802
                self,
                reqId: int,
                position: int,
                marketMaker: str,
                operation: int,
                side: int,
                price: float,
                size: Any,
                isSmartDepth: bool,
            ) -> None:
                outer.update_depth(reqId, position, operation, side, price, size)

            def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
                code = next(
                    (
                        int(item)
                        for item in args
                        if isinstance(item, int) and abs(int(item)) <= 1_000_000
                    ),
                    None,
                )
                outer.errors.append({"request_id": str(reqId), "code": code})

            def connectionClosed(self) -> None:  # noqa: N802
                outer.connection_closed = True

        self.client = _App()
        self.ready = threading.Event()
        self.thread: threading.Thread | None = None
        self.request_symbols: dict[int, str] = {}
        self.quotes: dict[str, dict[int, float]] = {}
        self.quote_sizes: dict[str, dict[int, float]] = {}
        self.trades: list[dict[str, Any]] = []
        self.books: dict[str, dict[tuple[int, int], tuple[float, float]]] = {}
        self.errors: list[dict[str, Any]] = []
        self.connection_closed = False
        self.market_data_requests = 0
        self.tick_by_tick_requests = 0
        self.depth_requests = 0

    def connect(self, host: str, port: int, client_id: int, timeout: float) -> bool:
        self.client.connect(host, port, client_id)
        if not self.client.isConnected():
            return False
        self.thread = threading.Thread(
            target=self.client.run,
            name=f"ibkr-equity-context-{client_id}",
            daemon=True,
        )
        self.thread.start()
        return self.ready.wait(timeout)

    def update_depth(
        self,
        request_id: int,
        position: int,
        operation: int,
        side: int,
        price: float,
        size: Any,
    ) -> None:
        symbol = self.request_symbols.get(int(request_id))
        if not symbol or int(side) not in {0, 1} or int(position) < 0:
            return
        book = self.books.setdefault(symbol, {})
        key = (int(side), int(position))
        if int(operation) == 2:
            book.pop(key, None)
        elif price > 0 and _float(size) >= 0:
            book[key] = (float(price), _float(size))

    def depth_snapshot(self, symbols: list[str], levels: int) -> list[dict[str, Any]]:
        timestamp = datetime.now(UTC).isoformat()
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            book = self.books.get(symbol, {})
            for position in range(int(levels)):
                ask = book.get((0, position))
                bid = book.get((1, position))
                if not ask or not bid:
                    continue
                rows.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "level": position + 1,
                        "bid_price": bid[0],
                        "bid_size": bid[1],
                        "ask_price": ask[0],
                        "ask_size": ask[1],
                        "source": "IBKR_SMART_DEPTH",
                    }
                )
        return rows

    def close(self, request_ids: dict[str, list[int]]) -> bool:
        for request_id in request_ids.get("market_data", []):
            self.client.cancelMktData(request_id)
        for request_id in request_ids.get("tick_by_tick", []):
            self.client.cancelTickByTickData(request_id)
        for request_id in request_ids.get("depth", []):
            self.client.cancelMktDepth(request_id, True)
        if self.client.isConnected():
            self.client.disconnect()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        return bool(self.thread and self.thread.is_alive())


def collect_realtime_equity_context(
    project_root: Path,
    *,
    env_file: str | Path = ".env.ibkr",
    config: RealtimeEquityConfig | None = None,
) -> dict[str, Any]:
    bounded = config or RealtimeEquityConfig()
    bounded.validate()
    settings = _load_market_data_endpoint(project_root / env_file)
    signals = _read_json(project_root / "output/signals/latest_signals.json")
    candidates = _candidate_contracts(signals.get("signals", []), limit=bounded.max_symbols)
    output_path = project_root / OUTPUT_ROOT / "realtime-equity-collection.json"
    if not candidates:
        return _publish(output_path, _report("NO_CURRENT_SETUPS", bounded, [], 0, 0, 0))

    app = EquityMarketDataApp()
    client_id = _observer_client_id(settings.client_id)
    if not app.connect(settings.host, settings.port, client_id, settings.connect_timeout_seconds):
        app.close({})
        return _publish(
            output_path,
            _report("IBKR_MARKET_DATA_CONNECTION_BLOCKED", bounded, candidates, 0, 0, 0),
        )

    request_ids: dict[str, list[int]] = {"market_data": [], "tick_by_tick": [], "depth": []}
    depth_symbols = [item["symbol"] for item in candidates[: bounded.depth_symbols]]
    depth_rows: list[dict[str, Any]] = []
    thread_leak = False
    try:
        app.client.reqMarketDataType(settings.market_data_type)
        for index, item in enumerate(candidates):
            contract = _native_contract(item)
            market_id = 8_200_000 + index
            tape_id = 8_210_000 + index
            app.request_symbols[market_id] = item["symbol"]
            app.request_symbols[tape_id] = item["symbol"]
            app.client.reqMktData(market_id, contract, "", False, False, [])
            request_ids["market_data"].append(market_id)
            app.market_data_requests += 1
            app.client.reqTickByTickData(tape_id, contract, "AllLast", 0, False)
            request_ids["tick_by_tick"].append(tape_id)
            app.tick_by_tick_requests += 1
            if index < bounded.depth_symbols:
                depth_id = 8_220_000 + index
                app.request_symbols[depth_id] = item["symbol"]
                app.client.reqMktDepth(depth_id, contract, bounded.depth_levels, True, [])
                request_ids["depth"].append(depth_id)
                app.depth_requests += 1
        deadline = time.monotonic() + float(bounded.duration_seconds)
        while time.monotonic() < deadline:
            time.sleep(min(1.0, max(0.05, deadline - time.monotonic())))
            depth_rows.extend(app.depth_snapshot(depth_symbols, bounded.depth_levels))
    finally:
        thread_leak = app.close(request_ids)

    quote_rows = _quote_rows(app, candidates)
    _append_private(project_root / PRIVATE_ROOT / "equity-trades.parquet", app.trades, days=7)
    _append_private(project_root / PRIVATE_ROOT / "equity-orderbook.parquet", depth_rows, days=2)
    _append_private(project_root / PRIVATE_ROOT / "equity-level1.parquet", quote_rows, days=7)
    observed_symbols = {str(row["symbol"]) for row in app.trades}
    depth_observed_symbols = {str(row["symbol"]) for row in depth_rows}
    status = (
        "GO"
        if observed_symbols and not thread_leak
        else "GO_DEGRADED_NO_TAPE_OR_DEPTH"
        if not thread_leak
        else "THREAD_LEAK_BLOCKED"
    )
    report = _report(
        status,
        bounded,
        candidates,
        len(app.trades),
        len(depth_rows),
        len(quote_rows),
        error_count=len(app.errors),
        error_code_counts={
            str(code): count
            for code, count in sorted(
                Counter(item.get("code") for item in app.errors if item.get("code") is not None).items()
            )
        },
        observed_symbol_count=len(observed_symbols),
        depth_observed_symbol_count=len(depth_observed_symbols),
        market_data_requests=app.market_data_requests,
        tick_by_tick_requests=app.tick_by_tick_requests,
        depth_requests=app.depth_requests,
        thread_leak=thread_leak,
        connection_closed=app.connection_closed,
            market_data_type=settings.market_data_type,
            endpoint_environment=settings.environment,
    )
    return _publish(output_path, report)


def _candidate_contracts(signals: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for signal in signals:
        symbol = str(signal.get("ticker") or signal.get("asset") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        raw_identity = signal.get("contract_identity")
        identity = raw_identity if isinstance(raw_identity, dict) else {}
        selected.append(
            {
                "symbol": symbol,
                "con_id": int(identity.get("con_id") or 0),
                "currency": str(identity.get("currency") or "USD").upper(),
                "exchange": str(identity.get("exchange") or "SMART").upper(),
                "primary_exchange": str(identity.get("primary_exchange") or "").upper(),
            }
        )
        seen.add(symbol)
        if len(selected) >= int(limit):
            break
    return selected


def _native_contract(item: dict[str, Any]) -> Any:
    from ibapi.contract import Contract

    contract = Contract()
    if int(item.get("con_id") or 0) > 0:
        contract.conId = int(item["con_id"])
    contract.symbol = str(item["symbol"])
    contract.secType = "STK"
    contract.exchange = str(item.get("exchange") or "SMART")
    contract.currency = str(item.get("currency") or "USD")
    if item.get("primary_exchange"):
        contract.primaryExchange = str(item["primary_exchange"])
    return contract


def _quote_rows(app: EquityMarketDataApp, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    timestamp = datetime.now(UTC).isoformat()
    rows = []
    for item in candidates:
        symbol = item["symbol"]
        prices = app.quotes.get(symbol, {})
        sizes = app.quote_sizes.get(symbol, {})
        if not prices:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "bid": prices.get(1),
                "ask": prices.get(2),
                "last": prices.get(4),
                "close": prices.get(9),
                "bid_size": sizes.get(0),
                "ask_size": sizes.get(3),
                "last_size": sizes.get(5),
                "source": "IBKR_LEVEL1",
            }
        )
    return rows


def _append_private(path: Path, rows: list[dict[str, Any]], *, days: int) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.exists():
        frame = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    timestamp_column = "timestamp" if "timestamp" in frame else "timestamp_utc"
    timestamp = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
    cutoff = pd.Timestamp(datetime.now(UTC) - timedelta(days=int(days)))
    frame = frame.loc[timestamp.ge(cutoff)].copy()
    frame[timestamp_column] = timestamp.loc[frame.index].map(lambda item: item.isoformat())
    frame = frame.drop_duplicates().sort_values([timestamp_column, "symbol"], kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _report(
    status: str,
    config: RealtimeEquityConfig,
    candidates: list[dict[str, Any]],
    trade_rows: int,
    depth_rows: int,
    quote_rows: int,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "schema": "ibkr_realtime_equity_context_collection_v1",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "requested_symbol_count": len(candidates),
        "requested_symbols_masked": [stable_hash(item["symbol"])[:20] for item in candidates],
        "duration_seconds": float(config.duration_seconds),
        "depth_symbol_budget": int(config.depth_symbols),
        "depth_levels": int(config.depth_levels),
        "trade_row_count": int(trade_rows),
        "depth_row_count": int(depth_rows),
        "quote_row_count": int(quote_rows),
        "market_data_authority": "READ_ONLY_CONTEXT",
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_write_calls": 0,
        "historical_data_calls": 0,
        **extra,
    }
    payload["content_hash"] = stable_hash(payload)
    return payload


def _publish(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return report


def _observer_client_id(base: int) -> int:
    raw = os.environ.get("IBKR_EQUITY_CONTEXT_CLIENT_ID", "")
    value = int(raw) if raw else int(base) + 420
    if value <= 0:
        raise ValueError("IBKR_EQUITY_CONTEXT_CLIENT_ID must be positive")
    return value


def _load_market_data_endpoint(path: Path) -> MarketDataEndpoint:
    values = dotenv_values(path) if path.exists() else {}

    def get(name: str, default: str) -> str:
        return str(os.environ.get(name) or values.get(name) or default).strip()

    host = get("IBKR_HOST", "127.0.0.1")
    port = int(get("IBKR_PORT", "7497"))
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("market-data observation requires localhost")
    if port not in {7497, 7496, 4002, 4001}:
        raise ValueError("market-data observation requires a known TWS/Gateway port")
    if get("IBKR_READ_ONLY", "true").lower() not in {"1", "true", "yes", "on"}:
        raise ValueError("IBKR_READ_ONLY=true is required for market-data observation")
    if get("IBKR_ORDER_AUTHORITY", "NONE").upper() != "NONE":
        raise ValueError("IBKR_ORDER_AUTHORITY=NONE is required for market-data observation")
    if get("IBKR_LIVE_TRADING_ENABLED", "false").lower() not in {"0", "false", "no", "off"}:
        raise ValueError("IBKR_LIVE_TRADING_ENABLED=false is required for market-data observation")
    if get("IBKR_ALLOW_ORDER_TRANSMISSION", "false").lower() not in {"0", "false", "no", "off"}:
        raise ValueError("IBKR_ALLOW_ORDER_TRANSMISSION=false is required for market-data observation")
    default_type = "1" if port in {7496, 4001} else "3"
    market_data_type = int(get("IBKR_MARKET_DATA_TYPE", default_type))
    if market_data_type not in {1, 2, 3, 4}:
        raise ValueError("IBKR_MARKET_DATA_TYPE must be 1, 2, 3, or 4")
    return MarketDataEndpoint(
        host="127.0.0.1",
        port=port,
        client_id=int(get("IBKR_CLIENT_ID", "17")),
        market_data_type=market_data_type,
        connect_timeout_seconds=float(get("IBKR_CONNECT_TIMEOUT_SECONDS", "12")),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = [
    "MarketDataEndpoint",
    "RealtimeEquityConfig",
    "collect_realtime_equity_context",
]
