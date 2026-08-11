from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .errors import is_degraded_code, is_fatal_code, is_info_code, normalize_error, to_text


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CallbackState:
    api_ready: bool = False
    connected: bool = False
    connection_closed: bool = False
    next_valid_id_seen: bool = False
    server_version: int | str | None = None
    connection_time: str | None = None
    last_server_unix_time: int | None = None
    last_server_utc: str | None = None
    last_heartbeat_at: datetime | None = None
    server_time_updates: int = 0
    managed_account_count: int = 0
    informational_messages: list[dict[str, Any]] = field(default_factory=list)
    degraded_messages: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    fatal_errors: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    financial_calls: dict[str, int] = field(
        default_factory=lambda: {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        }
    )
    ready_event: threading.Event = field(default_factory=threading.Event)
    current_time_event: threading.Event = field(default_factory=threading.Event)
    accounts_event: threading.Event = field(default_factory=threading.Event)
    contract_details: dict[int, list[Any]] = field(default_factory=dict)
    contract_details_end_events: dict[int, threading.Event] = field(default_factory=dict)
    symbol_samples: dict[int, list[Any]] = field(default_factory=dict)
    symbol_sample_events: dict[int, threading.Event] = field(default_factory=dict)
    market_rules: dict[int, list[Any]] = field(default_factory=dict)
    market_rule_events: dict[int, threading.Event] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def record_event(self, event: str, **fields: Any) -> None:
        with self.lock:
            self.events.append(
                {
                    "at": utc_now().isoformat(),
                    "event": event,
                    **fields,
                }
            )

    def record_connected(self, *, server_version: Any, connection_time: Any) -> None:
        with self.lock:
            self.connected = True
            self.connection_closed = False
            self.server_version = server_version
            self.connection_time = to_text(connection_time)
        self.record_event("connected")

    def record_next_valid_id(self, order_id: int) -> None:
        with self.lock:
            self.api_ready = True
            self.next_valid_id_seen = True
            self.ready_event.set()
        self.record_event("api_ready", next_valid_id_present=True)

    def record_current_time(self, unix_time: int) -> None:
        with self.lock:
            self.last_server_unix_time = int(unix_time)
            self.last_server_utc = datetime.fromtimestamp(
                int(unix_time),
                tz=timezone.utc,
            ).isoformat()
            self.last_heartbeat_at = utc_now()
            self.server_time_updates += 1
            self.current_time_event.set()
        self.record_event("server_time")

    def record_managed_accounts(self, accounts_list: str | bytes) -> None:
        text = to_text(accounts_list)
        count = len([item for item in text.split(",") if item.strip()])
        with self.lock:
            self.managed_account_count = count
            self.accounts_event.set()
        self.record_event("managed_accounts", managed_account_count=count)

    def record_error(self, req_id: Any, args: tuple[Any, ...]) -> None:
        normalized = normalize_error(req_id, args)
        code = normalized["code"]
        with self.lock:
            if is_info_code(code):
                self.informational_messages.append(normalized)
            elif is_degraded_code(code):
                self.degraded_messages.append(normalized)
            else:
                self.errors.append(normalized)
                if is_fatal_code(code):
                    self.fatal_errors.append(normalized)
        self.record_event("ibkr_error", code=code)

    def record_closed(self) -> None:
        with self.lock:
            self.connected = False
            self.connection_closed = True
        self.record_event("connection_closed")

    def contract_details_event(self, req_id: int) -> threading.Event:
        with self.lock:
            return self.contract_details_end_events.setdefault(req_id, threading.Event())

    def record_contract_details(self, req_id: int, contract_details: Any) -> None:
        with self.lock:
            self.contract_details.setdefault(req_id, []).append(contract_details)
        self.record_event("contract_details", req_id=req_id)

    def record_contract_details_end(self, req_id: int) -> None:
        with self.lock:
            self.contract_details_end_events.setdefault(req_id, threading.Event()).set()
        self.record_event("contract_details_end", req_id=req_id)

    def contract_details_for(self, req_id: int) -> list[Any]:
        with self.lock:
            return list(self.contract_details.get(req_id, []))

    def symbol_sample_event(self, req_id: int) -> threading.Event:
        with self.lock:
            return self.symbol_sample_events.setdefault(req_id, threading.Event())

    def record_symbol_samples(self, req_id: int, contract_descriptions: Any) -> None:
        samples = list(contract_descriptions or [])
        with self.lock:
            self.symbol_samples[req_id] = samples
            self.symbol_sample_events.setdefault(req_id, threading.Event()).set()
        self.record_event("symbol_samples", req_id=req_id, sample_count=len(samples))

    def symbol_samples_for(self, req_id: int) -> list[Any]:
        with self.lock:
            return list(self.symbol_samples.get(req_id, []))

    def market_rule_event(self, market_rule_id: int) -> threading.Event:
        with self.lock:
            return self.market_rule_events.setdefault(market_rule_id, threading.Event())

    def record_market_rule(self, market_rule_id: int, price_increments: Any) -> None:
        increments = list(price_increments or [])
        with self.lock:
            self.market_rules[market_rule_id] = increments
            self.market_rule_events.setdefault(market_rule_id, threading.Event()).set()
        self.record_event("market_rule", market_rule_id=market_rule_id, increment_count=len(increments))

    def market_rule_for(self, market_rule_id: int) -> list[Any]:
        with self.lock:
            return list(self.market_rules.get(market_rule_id, []))
