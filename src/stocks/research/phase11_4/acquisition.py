from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dotenv import dotenv_values

from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_2.providers import SafeJsonClient, eodhd_probe, load_provider_secrets


NASDAQ_TABLES = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
EODHD_BASE = "https://eodhd.com/api"
EXCHANGES = {"NASDAQ", "NYSE", "NYSEMKT"}
EODHD_EXCHANGE_MAP = {"AMEX": "NYSEMKT", "NYSE MKT": "NYSEMKT"}


@dataclass(frozen=True)
class SecurityIdentity:
    security_id: str
    ticker: str
    name: str
    exchange: str
    category: str
    first_price_date: str
    last_price_date: str
    is_delisted: bool
    sector: str
    industry: str
    currency: str
    figi_hash: str | None
    source_hash: str
    provider_symbol: str | None = None
    provider_identity_status: str = "UNRESOLVED"
    provider_identity_method: str | None = None


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval = 1.0 / max(requests_per_second, 0.1)
        self.lock = threading.Lock()
        self.last_request = 0.0

    def wait(self) -> None:
        with self.lock:
            delay = self.interval - (time.monotonic() - self.last_request)
            if delay > 0:
                time.sleep(delay)
            self.last_request = time.monotonic()


def private_root(project_root: Path) -> Path:
    return project_root / "data" / "research" / "phase11_4" / "private"


def security_master_path(project_root: Path) -> Path:
    return private_root(project_root) / "security-master.parquet"


def identity_map_path(project_root: Path) -> Path:
    return private_root(project_root) / "provider-identity-map.parquet"


def acquire_security_master(project_root: Path, *, refresh: bool = False) -> dict[str, Any]:
    destination = security_master_path(project_root)
    mapping_destination = identity_map_path(project_root)
    if destination.is_file() and mapping_destination.is_file() and not refresh:
        frame = pd.read_parquet(mapping_destination)
        return _master_summary(frame, cache_status="CACHE_REUSED")

    values = dotenv_values(project_root / ".env")
    nasdaq_key = str(values.get("Nasdaq_API_KEY") or values.get("NASDAQ_API_KEY") or "")
    eodhd_key = load_provider_secrets(project_root).get("EODHD") or ""
    if not nasdaq_key or not eodhd_key:
        return {
            "status": "PROVIDER_CREDENTIALS_MISSING",
            "nasdaq_key_present": bool(nasdaq_key),
            "eodhd_key_present": bool(eodhd_key),
        }

    sharadar_rows, sharadar_snapshot = _download_sharadar_tickers(nasdaq_key)
    identities = _normalize_sharadar(sharadar_rows)
    eodhd_rows, eodhd_hashes = _download_eodhd_identities(eodhd_key)
    resolved = resolve_provider_identities(identities, eodhd_rows)

    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(asdict(row) for row in identities).to_parquet(destination, index=False)
    frame = pd.DataFrame(asdict(row) for row in resolved)
    frame.to_parquet(mapping_destination, index=False)
    summary = _master_summary(frame, cache_status="PROVIDER_REFRESHED")
    summary.update(
        {
            "sharadar_snapshot": sharadar_snapshot,
            "sharadar_normalized_hash": _file_hash(destination),
            "provider_identity_map_hash": _file_hash(mapping_destination),
            "eodhd_active_payload_hash": eodhd_hashes["active"],
            "eodhd_delisted_payload_hash": eodhd_hashes["delisted"],
            "raw_provider_payloads_stored": False,
        }
    )
    return summary


def load_security_master(project_root: Path, *, resolved: bool = True) -> pd.DataFrame:
    path = identity_map_path(project_root) if resolved else security_master_path(project_root)
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def resolve_provider_identities(
    identities: list[SecurityIdentity], provider_rows: list[dict[str, Any]]
) -> list[SecurityIdentity]:
    by_mode_code: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_name: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_code: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in provider_rows:
        mode = str(row["provider_status"])
        code = str(row["code"]).upper()
        exchange = _exchange(str(row["exchange"]))
        by_mode_code[(mode, code)].append(row)
        by_name[(_name(str(row["name"])), exchange)].append(row)
        simplified = re.sub(r"(_OLD\d*|[-._](OLD|UN|U|WS|WT))$", "", code)
        by_code[(_code(simplified), exchange)].append(row)

    resolved: list[SecurityIdentity] = []
    for identity in identities:
        mode = "delisted" if identity.is_delisted else "active"
        direct = by_mode_code.get((mode, identity.ticker.upper()), [])
        candidate, method = _unique(direct), "EXACT_TICKER_STATUS"
        if candidate is None:
            alternatives: list[dict[str, Any]] = []
            ticker = identity.ticker.upper()
            for value in {ticker.replace(".", "-"), ticker.replace("-", "."), ticker.replace(".", ""), ticker.replace("-", "")}:
                alternatives.extend(by_mode_code.get((mode, value), []))
            candidate = _unique(alternatives)
            method = "PUNCTUATION_TRANSFORM"
        if candidate is None:
            candidate = _unique(by_name.get((_name(identity.name), identity.exchange), []))
            method = "UNIQUE_NORMALIZED_NAME_EXCHANGE"
        if candidate is None:
            candidate = _unique(by_code.get((_code(identity.ticker), identity.exchange), []))
            method = "UNIQUE_NORMALIZED_CODE_EXCHANGE"
        payload = asdict(identity)
        if candidate is not None:
            payload.update(
                {
                    "provider_symbol": f"{candidate['code']}.US",
                    "provider_identity_status": "RESOLVED",
                    "provider_identity_method": method,
                }
            )
        else:
            payload.update(
                {
                    "provider_identity_status": "AMBIGUOUS_OR_NOT_FOUND_BLOCKED",
                    "provider_identity_method": None,
                }
            )
        resolved.append(SecurityIdentity(**payload))
    return resolved


def acquire_price_histories(
    project_root: Path,
    *,
    max_symbols: int | None = None,
    workers: int = 12,
    requests_per_second: float = 10.0,
) -> dict[str, Any]:
    master_status = acquire_security_master(project_root)
    if master_status.get("status") != "GO":
        return master_status
    frame = load_security_master(project_root)
    eligible = frame.loc[frame["provider_identity_status"].eq("RESOLVED")].sort_values("security_id")
    if max_symbols is not None:
        eligible = eligible.head(max(0, max_symbols))
    root = private_root(project_root) / "pit_prices"
    root.mkdir(parents=True, exist_ok=True)
    key = load_provider_secrets(project_root).get("EODHD") or ""
    limiter = RateLimiter(requests_per_second)
    tasks = [row for row in eligible.to_dict("records") if not _price_complete(root, str(row["security_id"]))]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 24))) as pool:
        futures = {pool.submit(_collect_one, root, row, key, limiter): row for row in tasks}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - provider/network boundary
                row = futures[future]
                results.append(
                    {
                        "security_id": row["security_id"],
                        "status": "PROVIDER_ERROR",
                        "error_class": type(exc).__name__,
                    }
                )
    counts = Counter(row["status"] for row in results)
    records = eligible.to_dict("records")
    completed = sum(_price_complete(root, str(row["security_id"])) for row in records)
    valid = sum(_price_valid(root, str(row["security_id"])) for row in records)
    payload = {
        "schema": "phase11_4_pit_data_acquisition_v1",
        "status": "GO" if completed == len(eligible) else "PARTIAL",
        "eligible_security_count": len(eligible),
        "scheduled_count": len(tasks),
        "completed_security_count": completed,
        "valid_security_count": valid,
        "remaining_security_count": len(eligible) - completed,
        "run_status_counts": dict(counts),
        "requests_per_second_limit": requests_per_second,
        "worker_count": max(1, min(workers, 24)),
        "private_price_root": str(root),
        "provider_calls_read_only": sum(int(row.get("provider_calls", 0)) for row in results),
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "raw_provider_payloads_stored": False,
    }
    payload["content_hash"] = stable_hash(payload)
    output = project_root / "output" / "research" / "rsi_pit" / "pit-data-acquisition.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def compact_price_histories(project_root: Path) -> dict[str, Any]:
    import duckdb

    source_root = private_root(project_root) / "pit_prices"
    destination = private_root(project_root) / "pit-bars.parquet"
    invalid_files_removed = 0
    for manifest_path in source_root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bars_path = manifest_path.parent / "bars.parquet"
        if manifest.get("status") != "EXECUTION_PRICE_GO" and bars_path.is_file():
            bars_path.unlink()
            invalid_files_removed += 1
    pattern = str((source_root / "**" / "bars.parquet").resolve()).replace("\\", "/")
    if not list(source_root.rglob("bars.parquet")):
        return {"status": "NO_PRICE_FILES"}
    temporary = destination.with_suffix(".tmp.parquet")
    pattern_sql = pattern.replace("'", "''")
    temporary_sql = str(temporary.resolve()).replace("'", "''")
    with duckdb.connect() as db:
        db.execute(
            f"COPY (SELECT * FROM read_parquet('{pattern_sql}') ORDER BY security_id,date) "
            f"TO '{temporary_sql}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
        )
        summary = db.execute(
            "SELECT COUNT(*),COUNT(DISTINCT security_id),MIN(date),MAX(date) FROM read_parquet(?)",
            [str(temporary.resolve())],
        ).fetchone()
    if summary is None:
        temporary.unlink(missing_ok=True)
        return {"status": "NO_PRICE_ROWS"}
    row_count, securities, first_date, last_date = summary
    temporary.replace(destination)
    payload = {
        "schema": "phase11_4_pit_price_compaction_v1",
        "status": "GO",
        "row_count": int(row_count),
        "security_count": int(securities),
        "first_date": str(first_date),
        "last_date": str(last_date),
        "content_hash": _file_hash(destination),
        "private_path": str(destination),
        "invalid_price_files_removed": invalid_files_removed,
    }
    output = project_root / "output" / "research" / "rsi_pit" / "pit-price-compaction.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _download_sharadar_tickers(key: str) -> tuple[list[dict[str, str]], str | None]:
    query = urllib.parse.urlencode({"table": "SEP", "qopts.export": "true", "api_key": key})
    request = urllib.request.Request(f"{NASDAQ_TABLES}/TICKERS.json?{query}", headers={"User-Agent": "Stocks Phase11.4"})
    with urllib.request.urlopen(request, timeout=120) as response:
        manifest = json.loads(response.read())
    bulk = manifest["datatable_bulk_download"]["file"]
    with urllib.request.urlopen(bulk["link"], timeout=180) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        name = bundle.namelist()[0]
        with bundle.open(name) as handle:
            rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
    return rows, bulk.get("data_snapshot_time")


def _normalize_sharadar(rows: Iterable[dict[str, str]]) -> list[SecurityIdentity]:
    result: list[SecurityIdentity] = []
    for row in rows:
        category = str(row.get("category") or "")
        first = str(row.get("firstpricedate") or "")
        last = str(row.get("lastpricedate") or "")
        exchange = str(row.get("exchange") or "")
        if (
            exchange not in EXCHANGES
            or not first
            or not last
            or last < "2000-01-01"
            or "Common Stock" not in category
            or "Warrant" in category
        ):
            continue
        figi = str(row.get("figi") or "")
        normalized: dict[str, Any] = {
            "security_id": f"SHARADAR:{row.get('permaticker')}",
            "ticker": str(row.get("ticker") or "").upper(),
            "name": str(row.get("name") or ""),
            "exchange": exchange,
            "category": category,
            "first_price_date": first,
            "last_price_date": last,
            "is_delisted": str(row.get("isdelisted") or "").upper() == "Y",
            "sector": str(row.get("sector") or "UNKNOWN") or "UNKNOWN",
            "industry": str(row.get("industry") or "UNKNOWN") or "UNKNOWN",
            "currency": str(row.get("currency") or "USD") or "USD",
            "figi_hash": stable_hash(figi) if figi else None,
            "source_hash": stable_hash(row),
        }
        result.append(SecurityIdentity(**normalized))
    return sorted(result, key=lambda row: row.security_id)


def _download_eodhd_identities(key: str) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    client = SafeJsonClient(user_agent="Stocks Phase11.4 identity resolver", timeout_seconds=60, max_attempts=2)
    result: list[dict[str, Any]] = []
    hashes: dict[str, str | None] = {}
    for status, params in (("active", {}), ("delisted", {"delisted": "1"})):
        audit, payload = eodhd_probe(client, key, status, "exchange-symbol-list/US", params)
        if audit.status != "PROBE_GO":
            raise RuntimeError(f"EODHD_IDENTITY_{status.upper()}_{audit.status}")
        hashes[status] = audit.payload_hash
        for row in payload or []:
            if not isinstance(row, dict) or "COMMON" not in str(row.get("Type") or "").upper():
                continue
            exchange = _exchange(str(row.get("Exchange") or ""))
            if exchange not in EXCHANGES:
                continue
            result.append(
                {
                    "code": str(row.get("Code") or "").upper(),
                    "name": str(row.get("Name") or ""),
                    "exchange": exchange,
                    "provider_status": status,
                    "source_hash": stable_hash(row),
                }
            )
    return result, hashes


def _collect_one(root: Path, row: dict[str, Any], key: str, limiter: RateLimiter) -> dict[str, Any]:
    security_id = str(row["security_id"])
    target = _security_directory(root, security_id)
    target.mkdir(parents=True, exist_ok=True)
    prices, price_calls = _request_json(
        f"eod/{row['provider_symbol']}",
        {"from": max("2000-01-01", str(row["first_price_date"])), "to": str(row["last_price_date"]), "period": "d", "order": "a"},
        key,
        limiter,
    )
    splits, split_calls = _request_json(
        f"splits/{row['provider_symbol']}",
        {"from": max("2000-01-01", str(row["first_price_date"])), "to": str(row["last_price_date"])},
        key,
        limiter,
    )
    bars, validation = _normalize_split_adjusted_prices(row, prices, splits)
    manifest = {
        "security_id": security_id,
        "provider_symbol_hash": stable_hash(row["provider_symbol"]),
        "status": validation,
        "row_count": len(bars),
        "first_date": bars[0]["date"] if bars else None,
        "last_date": bars[-1]["date"] if bars else None,
        "content_hash": stable_hash(bars),
        "provider_calls": price_calls + split_calls,
        "raw_provider_payload_stored": False,
    }
    if bars and validation == "EXECUTION_PRICE_GO":
        pd.DataFrame(bars).to_parquet(target / "bars.parquet", index=False)
    (target / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def _request_json(
    path: str, params: dict[str, str], key: str, limiter: RateLimiter, max_attempts: int = 3
) -> tuple[Any, int]:
    query = urllib.parse.urlencode(params | {"api_token": key, "fmt": "json"})
    calls = 0
    for attempt in range(1, max_attempts + 1):
        limiter.wait()
        calls += 1
        request = urllib.request.Request(f"{EODHD_BASE}/{path}?{query}", headers={"User-Agent": "Stocks Phase11.4"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read()), calls
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == max_attempts:
                raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == max_attempts:
                raise
        time.sleep(float(attempt * 2))
    raise RuntimeError("PROVIDER_RETRY_EXHAUSTED")


def _normalize_split_adjusted_prices(
    identity: dict[str, Any], prices: Any, splits: Any
) -> tuple[list[dict[str, Any]], str]:
    price_rows = [row for row in prices if isinstance(row, dict)] if isinstance(prices, list) else []
    split_rows = [row for row in splits if isinstance(row, dict)] if isinstance(splits, list) else []
    factors: dict[str, float] = {}
    for row in split_rows:
        factor = _split_factor(row.get("split") or row.get("value"))
        when = str(row.get("date") or "")[:10]
        if when and factor is not None:
            factors[when] = factors.get(when, 1.0) * factor
    cumulative = 1.0
    adjusted_factors: dict[str, float] = {}
    for when in sorted(factors, reverse=True):
        cumulative *= factors[when]
        adjusted_factors[when] = cumulative
    bars: list[dict[str, Any]] = []
    for row in price_rows:
        when = str(row.get("date") or "")[:10]
        if not when or when < max("2000-01-01", str(identity["first_price_date"])) or when > str(identity["last_price_date"]):
            continue
        future_factor = 1.0
        for split_date, value in adjusted_factors.items():
            if when < split_date:
                future_factor = value
        try:
            values = {name: float(row[name]) / future_factor for name in ("open", "high", "low", "close")}
            volume = float(row.get("volume") or 0) * future_factor
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if (
            min(values.values()) <= 0
            or values["high"] < max(values["open"], values["close"], values["low"])
            or values["low"] > min(values["open"], values["close"], values["high"])
            or volume < 0
        ):
            continue
        bars.append(
            {
                "security_id": identity["security_id"],
                "ticker": identity["ticker"],
                "date": when,
                **values,
                "volume": volume,
                "sector": identity["sector"],
                "currency": identity["currency"],
                "source": "EODHD",
                "price_basis": "SPLIT_ADJUSTED_DIVIDENDS_EXCLUDED",
            }
        )
    bars.sort(key=lambda row: row["date"])
    if not bars:
        return [], "PRICE_HISTORY_INCOMPLETE"
    first = bars[0]["date"]
    last = bars[-1]["date"]
    start_gap = (pd.Timestamp(first) - pd.Timestamp(max("2000-01-01", str(identity["first_price_date"])))).days
    end_gap = (pd.Timestamp(str(identity["last_price_date"])) - pd.Timestamp(last)).days
    status = "EXECUTION_PRICE_GO" if start_gap <= 10 and end_gap <= 10 else "PRICE_HISTORY_BOUNDARY_CONFLICT"
    return bars, status


def _split_factor(value: Any) -> float | None:
    text = str(value or "").strip()
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            factor = float(numerator) / float(denominator)
        elif ":" in text:
            numerator, denominator = text.split(":", 1)
            factor = float(numerator) / float(denominator)
        else:
            factor = float(text)
        return factor if factor > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _price_complete(root: Path, security_id: str) -> bool:
    path = _security_directory(root, security_id)
    return (path / "manifest.json").is_file()


def _price_valid(root: Path, security_id: str) -> bool:
    path = _security_directory(root, security_id)
    manifest = path / "manifest.json"
    if not manifest.is_file() or not (path / "bars.parquet").is_file():
        return False
    return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "EXECUTION_PRICE_GO"


def _security_directory(root: Path, security_id: str) -> Path:
    digest = hashlib.sha256(security_id.encode()).hexdigest().upper()
    return root / digest[:2] / digest


def _master_summary(frame: pd.DataFrame, *, cache_status: str) -> dict[str, Any]:
    statuses = frame["provider_identity_status"].value_counts().to_dict() if not frame.empty else {}
    return {
        "status": "GO" if not frame.empty else "NO_DATA",
        "cache_status": cache_status,
        "security_count": len(frame),
        "active_security_count": int((~frame["is_delisted"]).sum()) if not frame.empty else 0,
        "delisted_security_count": int(frame["is_delisted"].sum()) if not frame.empty else 0,
        "provider_identity_status_counts": {str(key): int(value) for key, value in statuses.items()},
        "resolved_security_count": int(frame["provider_identity_status"].eq("RESOLVED").sum()) if not frame.empty else 0,
        "sector_coverage_ratio": float(frame["sector"].ne("UNKNOWN").mean()) if not frame.empty else 0.0,
        "listing_window_coverage_ratio": float((frame["first_price_date"].notna() & frame["last_price_date"].notna()).mean()) if not frame.empty else 0.0,
        "raw_provider_payloads_stored": False,
    }


def _unique(rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    unique = {str(row["code"]): row for row in rows}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _exchange(value: str) -> str:
    return EODHD_EXCHANGE_MAP.get(value, value)


def _name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().upper().replace("&", " AND ")
    normalized = re.sub(
        r"\b(INCORPORATED|INC|CORPORATION|CORP|COMPANY|CO|LIMITED|LTD|PLC|SA|NV|AG|HOLDINGS?|GROUP|THE|COMMON STOCK|ADR)\b",
        " ",
        normalized,
    )
    return re.sub(r"[^A-Z0-9]", "", normalized)


def _code(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()
