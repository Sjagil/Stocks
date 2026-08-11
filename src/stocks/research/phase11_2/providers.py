from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from stocks.execution.idempotency import stable_hash


EODHD_BASE = "https://eodhd.com/api"
SEC_BASE = "https://data.sec.gov"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
OXR_BASE = "https://openexchangerates.org/api"


@dataclass(frozen=True)
class ProbeResult:
    provider: str
    dataset: str
    status: str
    http_status: int | None
    schema_shape: str
    record_count: int
    earliest_timestamp: str | None
    latest_timestamp: str | None
    timestamp_precision: str
    null_rate: float
    duplicate_rate: float
    revision_fields: tuple[str, ...]
    payload_hash: str | None
    attempt_count: int
    error_class: str | None

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


class SafeJsonClient:
    def __init__(self, *, user_agent: str, timeout_seconds: int = 20, max_attempts: int = 2, minimum_interval: float = 0.12) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, min(max_attempts, 3))
        self.minimum_interval = max(0.0, minimum_interval)
        self._last_request = 0.0

    def get_json(self, url: str, *, headers: dict[str, str] | None = None) -> tuple[Any | None, int | None, int, str | None]:
        request_headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        request_headers.update(headers or {})
        for attempt in range(1, self.max_attempts + 1):
            elapsed = time.monotonic() - self._last_request
            if elapsed < self.minimum_interval:
                time.sleep(self.minimum_interval - elapsed)
            request = urllib.request.Request(url, headers=request_headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    self._last_request = time.monotonic()
                    return json.loads(response.read().decode("utf-8")), int(response.status), attempt, None
            except urllib.error.HTTPError as exc:
                self._last_request = time.monotonic()
                error_class = "PLAN_NOT_ENTITLED" if exc.code in {401, 402, 403} else "RATE_LIMITED" if exc.code == 429 else "HTTP_ERROR"
                if attempt == self.max_attempts or exc.code not in {429, 500, 502, 503, 504}:
                    return None, int(exc.code), attempt, error_class
                time.sleep(float(attempt))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                self._last_request = time.monotonic()
                if attempt == self.max_attempts:
                    return None, None, attempt, "PROVIDER_UNAVAILABLE"
                time.sleep(float(attempt))
        return None, None, self.max_attempts, "PROVIDER_UNAVAILABLE"


def load_provider_secrets(project_root: Path) -> dict[str, str | None]:
    values = dotenv_values(project_root / ".env") if (project_root / ".env").exists() else {}

    def first(*names: str) -> str | None:
        for name in names:
            value = os.environ.get(name) or values.get(name)
            if value:
                return str(value)
        return None

    return {
        "EODHD": first("EODHD_API_KEY", "EOD_API_KEY", "EODHISTORICALDATA_API_KEY"),
        "OPENEXCHANGERATES": first("OPENEXCHANGERATES_APP_ID", "OPENEXCHANGE_API_KEY"),
    }


def safe_secret_status(project_root: Path) -> dict[str, bool]:
    secrets = load_provider_secrets(project_root)
    return {provider: bool(secret) for provider, secret in secrets.items()} | {"SEC_EDGAR": True}


def eodhd_probe(client: SafeJsonClient, key: str | None, dataset: str, path: str, params: dict[str, str]) -> tuple[ProbeResult, Any | None]:
    if not key:
        return _blocked_probe("EODHD", dataset, "MISSING_PROVIDER_KEY"), None
    query = urllib.parse.urlencode(params | {"api_token": key, "fmt": "json"})
    payload, status, attempts, error = client.get_json(f"{EODHD_BASE}/{path}?{query}")
    return summarize_probe("EODHD", dataset, payload, status, attempts, error), payload


def sec_probe(client: SafeJsonClient, dataset: str, url: str) -> tuple[ProbeResult, Any | None]:
    payload, status, attempts, error = client.get_json(url)
    return summarize_probe("SEC_EDGAR", dataset, payload, status, attempts, error), payload


def oxr_probe(client: SafeJsonClient, key: str | None, date: str) -> tuple[ProbeResult, Any | None]:
    if not key:
        return _blocked_probe("OPENEXCHANGERATES", "fx_rates", "MISSING_PROVIDER_KEY"), None
    query = urllib.parse.urlencode({"app_id": key, "symbols": "EUR,USD"})
    payload, status, attempts, error = client.get_json(f"{OXR_BASE}/historical/{date}.json?{query}")
    return summarize_probe("OPENEXCHANGERATES", "fx_rates", payload, status, attempts, error), payload


def summarize_probe(provider: str, dataset: str, payload: Any, http_status: int | None, attempts: int, error: str | None) -> ProbeResult:
    records = _records(payload)
    timestamps = sorted(value for row in records for value in _timestamp_values(row))
    total_values = sum(len(row) for row in records)
    null_values = sum(value is None for row in records for value in row.values())
    fingerprints = [stable_hash(row) for row in records]
    duplicates = len(fingerprints) - len(set(fingerprints))
    revision_fields = sorted({key for row in records for key in row if any(token in key.lower() for token in ("revision", "updated", "amend", "accepted"))})
    ok = http_status == 200 and payload is not None
    return ProbeResult(
        provider=provider,
        dataset=dataset,
        status="PROBE_GO" if ok else error or "PROBE_BLOCKED",
        http_status=http_status,
        schema_shape=_shape(payload),
        record_count=len(records),
        earliest_timestamp=timestamps[0] if timestamps else None,
        latest_timestamp=timestamps[-1] if timestamps else None,
        timestamp_precision=_precision(timestamps),
        null_rate=round(null_values / total_values, 6) if total_values else 0.0,
        duplicate_rate=round(duplicates / len(fingerprints), 6) if fingerprints else 0.0,
        revision_fields=tuple(revision_fields),
        payload_hash=stable_hash(payload) if payload is not None else None,
        attempt_count=attempts,
        error_class=error,
    )


def _blocked_probe(provider: str, dataset: str, error: str) -> ProbeResult:
    return ProbeResult(provider, dataset, error, None, "NONE", 0, None, None, "NONE", 0.0, 0.0, (), None, 0, error)


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "results", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return [payload]


def _timestamp_values(row: dict[str, Any]) -> list[str]:
    values = []
    for key, value in row.items():
        if value is None or not any(token in key.lower() for token in ("date", "time", "filed", "accepted", "period")):
            continue
        if isinstance(value, (str, int, float)):
            values.append(str(value))
    return values


def _shape(payload: Any) -> str:
    if isinstance(payload, list):
        keys = sorted({key for row in payload[:10] if isinstance(row, dict) for key in row})
        return f"LIST[{','.join(keys[:25])}]"
    if isinstance(payload, dict):
        return f"OBJECT[{','.join(sorted(payload)[:25])}]"
    return type(payload).__name__.upper() if payload is not None else "NONE"


def _precision(values: list[str]) -> str:
    if not values:
        return "NONE"
    if any("T" in value or ":" in value for value in values):
        return "SECOND"
    return "DATE"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
