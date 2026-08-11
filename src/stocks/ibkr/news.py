from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from dotenv import dotenv_values

from stocks.application.config import IbkrSettings, load_ibkr_settings
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.errors import normalize_error


PRIVATE_HEADLINES = Path("data/news/ibkr/private/headlines.jsonl")
PUBLIC_OUTPUT = Path("output/ibkr/news")
CONTRACTS_PATH = Path("output/ibkr/contracts/stocks.parquet")
ENTITLEMENT_TERMS = (
    "not subscribed",
    "no market data permissions",
    "subscription",
    "news provider",
)
LIVE_PORTS = {4001, 7496}


@dataclass
class NewsObservation:
    connected: bool = False
    connection_closed_count: int = 0
    api_ready: bool = False
    providers: list[dict[str, str]] = field(default_factory=list)
    headlines: list[dict[str, Any]] = field(default_factory=list)
    completed_request_ids: set[int] = field(default_factory=set)
    error_codes: list[int] = field(default_factory=list)
    entitlement_error_count: int = 0
    thread_leak: bool = False
    provider_request_count: int = 0
    historical_news_request_count: int = 0
    requested_symbol_count: int = 0
    completed_symbol_count: int = 0


@dataclass(frozen=True)
class ReadOnlyNewsSettings:
    host: str
    port: int
    client_id: int
    connect_timeout_seconds: float
    request_timeout_seconds: float
    environment: str


NewsObserver = Callable[
    [ReadOnlyNewsSettings | IbkrSettings, list[dict[str, Any]], int, int],
    NewsObservation,
]


def ibkr_news_schema() -> dict[str, Any]:
    return {
        "schema": "ibkr_read_only_news_schema_v1",
        "status": "GO",
        "allowed_calls": ["reqNewsProviders", "reqHistoricalNews"],
        "forbidden_calls": [
            "place" + "Order",
            "cancel" + "Order",
            "reqGlobal" + "Cancel",
            "req" + "Ids",
            "reqAutoOpen" + "Orders",
            "exercise" + "Options",
        ],
        "article_body_requests_allowed": False,
        "raw_account_identifiers_stored": False,
        "private_headline_store": str(PRIVATE_HEADLINES),
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
    }


def probe_ibkr_news(
    project_root: Path,
    *,
    env_file: str | Path = ".env.ibkr",
    observer: NewsObserver | None = None,
) -> dict[str, Any]:
    settings = _settings(project_root, env_file)
    observation = (observer or _observe_ibkr_news)(settings, [], 1, 1)
    status = _capability_status(observation)
    report = {
        "schema": "ibkr_read_only_news_capability_v1",
        "status": status,
        "generated_at": _now(),
        "tws_connected": observation.connected,
        "api_ready": observation.api_ready,
        "provider_count": len(observation.providers),
        "providers": observation.providers,
        "provider_request_count": observation.provider_request_count,
        "entitlement_error_count": observation.entitlement_error_count,
        "error_codes": sorted(set(observation.error_codes)),
        "historical_headlines_capability": (
            "AVAILABLE"
            if observation.providers
            else "UNAVAILABLE_ENTITLEMENT"
            if observation.entitlement_error_count
            else "UNAVAILABLE_NO_PROVIDER_SUBSCRIPTION"
            if observation.connected
            else "UNPROVEN_TWS_UNAVAILABLE"
        ),
        "connection_environment": getattr(
            settings, "environment", "PAPER_READ_ONLY_OBSERVATION"
        ),
        "subscription_purchase_automatic": False,
        "article_body_requests": 0,
        "thread_leak": observation.thread_leak,
        "connection_closed_count": observation.connection_closed_count,
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }
    report["content_hash"] = stable_hash(report)
    _write_public(project_root, "capabilities.json", report)
    return report


def collect_ibkr_news(
    project_root: Path,
    *,
    symbols: list[str],
    lookback_hours: int = 72,
    max_results_per_symbol: int = 50,
    env_file: str | Path = ".env.ibkr",
    observer: NewsObserver | None = None,
) -> dict[str, Any]:
    bounded_hours = max(1, min(int(lookback_hours), 24 * 30))
    bounded_results = max(1, min(int(max_results_per_symbol), 100))
    contracts = _resolved_contracts(project_root, symbols[:20])
    settings = _settings(project_root, env_file)
    observation = (observer or _observe_ibkr_news)(
        settings,
        contracts,
        bounded_hours,
        bounded_results,
    )
    stored = _store_headlines(project_root, observation.headlines)
    capability = _capability_status(observation)
    if not contracts:
        status = "NO_RESOLVED_CONTRACTS"
    elif capability not in {"AVAILABLE", "AVAILABLE_NO_CURRENT_HEADLINES"}:
        status = capability
    elif observation.completed_symbol_count < observation.requested_symbol_count:
        status = "PARTIAL"
    else:
        status = "GO" if observation.headlines else "GO_NO_CURRENT_HEADLINES"
    report = {
        "schema": "ibkr_read_only_historical_news_collection_v1",
        "status": status,
        "generated_at": _now(),
        "lookback_hours": bounded_hours,
        "max_results_per_symbol": bounded_results,
        "requested_symbol_count": observation.requested_symbol_count,
        "completed_symbol_count": observation.completed_symbol_count,
        "unresolved_symbol_count": max(0, len(symbols[:20]) - len(contracts)),
        "provider_count": len(observation.providers),
        "providers": observation.providers,
        "connection_environment": getattr(
            settings, "environment", "PAPER_READ_ONLY_OBSERVATION"
        ),
        "received_headline_count": len(observation.headlines),
        "new_private_headline_count": stored["new_count"],
        "duplicate_private_headline_count": stored["duplicate_count"],
        "private_store": str(project_root / PRIVATE_HEADLINES),
        "provider_request_count": observation.provider_request_count,
        "historical_news_request_count": observation.historical_news_request_count,
        "entitlement_error_count": observation.entitlement_error_count,
        "error_codes": sorted(set(observation.error_codes)),
        "thread_leak": observation.thread_leak,
        "connection_closed_count": observation.connection_closed_count,
        "raw_article_ids_published": False,
        "raw_account_identifiers_stored": False,
        "article_body_requests": 0,
        "broker_observation_authority": "READ_ONLY",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }
    report["content_hash"] = stable_hash(report)
    _write_public(project_root, "collection.json", report)
    return report


def _observe_ibkr_news(
    settings: ReadOnlyNewsSettings | IbkrSettings,
    contracts: list[dict[str, Any]],
    lookback_hours: int,
    max_results: int,
) -> NewsObservation:
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper

    observation = NewsObservation()
    ready = threading.Event()
    providers_done = threading.Event()
    request_events: dict[int, threading.Event] = {}
    request_symbols: dict[int, str] = {}

    class App(EWrapper, EClient):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            EClient.__init__(self, self)

        def nextValidId(self, orderId: int) -> None:  # noqa: N802
            del orderId
            observation.api_ready = True
            ready.set()

        def newsProviders(self, newsProviders: list[Any]) -> None:  # noqa: N802
            observation.providers = [
                {
                    "code": str(getattr(provider, "code", "")),
                    "name": str(getattr(provider, "name", "")),
                }
                for provider in newsProviders
                if getattr(provider, "code", None)
            ]
            providers_done.set()

        def historicalNews(  # noqa: N802
            self,
            requestId: int,
            time: str,
            providerCode: str,
            articleId: str,
            headline: str,
        ) -> None:
            symbol = request_symbols.get(requestId, "UNKNOWN")
            observation.headlines.append(
                {
                    "published_at": _ibkr_timestamp(time),
                    "title": _clean_headline(headline),
                    "source": f"IBKR_TWS:{providerCode}",
                    "provider_code": providerCode,
                    "symbols": [symbol],
                    "article_reference_hash": stable_hash(
                        {"provider": providerCode, "article_id": articleId}
                    ),
                    "collected_at": _now(),
                    "sentiment_polarity": 0.0,
                    "sentiment_status": "NOT_SCORED",
                    "execution_authority": "NONE",
                }
            )

        def historicalNewsEnd(  # noqa: N802
            self, requestId: int, hasMore: bool
        ) -> None:
            del hasMore
            observation.completed_request_ids.add(requestId)
            request_events.setdefault(requestId, threading.Event()).set()

        def error(self, reqId: Any, *args: Any) -> None:  # type: ignore[override]
            normalized = normalize_error(reqId, args)
            code = normalized["code"]
            message = str(normalized["message"])
            if isinstance(code, int):
                observation.error_codes.append(code)
            if any(term in message.lower() for term in ENTITLEMENT_TERMS):
                observation.entitlement_error_count += 1

        def connectionClosed(self) -> None:  # noqa: N802
            # Preserve proof that the bounded observation connected. A normal
            # disconnect in the finally block is cleanup, not a failed probe.
            observation.connection_closed_count += 1

    app = App()
    thread: threading.Thread | None = None
    try:
        app.connect(settings.host, settings.port, settings.client_id + 2000)
        observation.connected = bool(app.isConnected())
        thread = threading.Thread(
            target=app.run,
            name="ibkr-read-only-news",
            daemon=True,
        )
        thread.start()
        if not ready.wait(settings.connect_timeout_seconds):
            return observation
        app.reqNewsProviders()
        observation.provider_request_count = 1
        providers_done.wait(settings.request_timeout_seconds)
        provider_codes = "+".join(
            row["code"] for row in observation.providers if row["code"]
        )
        if not provider_codes or not contracts:
            return observation
        now = datetime.now(UTC)
        start = now - timedelta(hours=lookback_hours)
        observation.requested_symbol_count = len(contracts)
        for index, contract in enumerate(contracts):
            request_id = 920_000 + index
            request_symbols[request_id] = str(contract["symbol"])
            event = request_events.setdefault(request_id, threading.Event())
            app.reqHistoricalNews(
                request_id,
                int(contract["con_id"]),
                provider_codes,
                _ibkr_request_timestamp(start),
                _ibkr_request_timestamp(now),
                max_results,
                [],
            )
            observation.historical_news_request_count += 1
            if event.wait(settings.request_timeout_seconds):
                observation.completed_symbol_count += 1
        return observation
    except (OSError, ConnectionError):
        return observation
    finally:
        if app.isConnected():
            app.disconnect()
        if thread is not None:
            thread.join(timeout=2.0)
            observation.thread_leak = thread.is_alive()


def _resolved_contracts(
    project_root: Path, symbols: list[str]
) -> list[dict[str, Any]]:
    path = project_root / CONTRACTS_PATH
    if not path.is_file():
        return []
    frame = pd.read_parquet(path)
    if not {"symbol", "con_id", "security_type"}.issubset(frame.columns):
        return []
    allowed = {str(symbol).upper() for symbol in symbols}
    frame = frame.loc[
        frame["symbol"].astype(str).str.upper().isin(allowed)
        & frame["security_type"].astype(str).eq("STK")
        & frame["con_id"].notna()
    ]
    return [
        {"symbol": str(row["symbol"]).upper(), "con_id": int(row["con_id"])}
        for _, row in frame.drop_duplicates("symbol").iterrows()
    ]


def _store_headlines(
    project_root: Path, headlines: list[dict[str, Any]]
) -> dict[str, int]:
    path = project_root / PRIVATE_HEADLINES
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing.add(_headline_identity(row))
    new_rows = []
    duplicate_count = 0
    for row in headlines:
        identity = _headline_identity(row)
        if identity in existing:
            duplicate_count += 1
            continue
        existing.add(identity)
        new_rows.append(row)
    if new_rows:
        with path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return {"new_count": len(new_rows), "duplicate_count": duplicate_count}


def _headline_identity(row: dict[str, Any]) -> str:
    return stable_hash(
        {
            "published_at": row.get("published_at"),
            "title": row.get("title"),
            "source": row.get("source"),
            "symbols": row.get("symbols"),
        }
    )


def _capability_status(observation: NewsObservation) -> str:
    if not observation.connected or not observation.api_ready:
        return "TWS_UNAVAILABLE"
    if observation.providers:
        return (
            "AVAILABLE"
            if observation.headlines
            else "AVAILABLE_NO_CURRENT_HEADLINES"
        )
    if observation.entitlement_error_count:
        return "UNAVAILABLE_ENTITLEMENT"
    return "UNAVAILABLE_NO_PROVIDER_SUBSCRIPTION"


def _settings(
    project_root: Path, env_file: str | Path
) -> ReadOnlyNewsSettings | IbkrSettings:
    path = Path(env_file)
    if not path.is_absolute():
        path = project_root / path
    if path.name == ".env.ibkr.live":
        values = {
            key: str(value).strip()
            for key, value in dotenv_values(path).items()
            if value is not None
        }
        host = values.get("IBKR_HOST", "")
        port = _integer(values.get("IBKR_PORT"), -1)
        client_id = _integer(
            values.get("IBKR_NEWS_CLIENT_ID")
            or values.get("IBKR_QUOTE_CLIENT_ID")
            or values.get("IBKR_RECON_CLIENT_ID"),
            -1,
        )
        if (
            values.get("IBKR_ENVIRONMENT") != "LIVE"
            or host not in {"127.0.0.1", "localhost"}
            or port not in LIVE_PORTS
            or client_id <= 0
        ):
            raise ValueError("LIVE_READ_ONLY_NEWS_CONFIG_BLOCKED")
        timeout = _positive_float(
            values.get("IBKR_LIVE_CALLBACK_TIMEOUT_SECONDS"), 15.0
        )
        return ReadOnlyNewsSettings(
            host=host,
            port=port,
            client_id=client_id,
            connect_timeout_seconds=timeout,
            request_timeout_seconds=timeout,
            environment="LIVE_READ_ONLY_OBSERVATION",
        )
    return load_ibkr_settings(path)


def _integer(value: Any, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _ibkr_request_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%d %H:%M:%S.0 UTC")


def _ibkr_timestamp(value: str) -> str:
    text = str(value).strip()
    for pattern in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y%m%d %H:%M:%S.%f",
        "%Y%m%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=UTC).isoformat()
        except ValueError:
            continue
    return text


def _clean_headline(value: Any) -> str:
    return re.sub(r"^\{[^}]+\}\s*", "", str(value or "")).strip()


def _write_public(project_root: Path, name: str, payload: dict[str, Any]) -> None:
    path = project_root / PUBLIC_OUTPUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "NewsObservation",
    "ReadOnlyNewsSettings",
    "collect_ibkr_news",
    "ibkr_news_schema",
    "probe_ibkr_news",
]
