from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from stocks.data.corporate_actions import (
    CorporateActionLayout,
    _dedupe_events,
    _dividend_record,
    _ensure_list,
    _manifest as corporate_action_manifest,
    _split_record,
    _write_outputs as write_corporate_action_outputs,
    eodhd_ticker,
    validate_corporate_action_cache,
)
from stocks.data.fx import (
    FX_FIELDS,
    FxCacheLayout,
    _constant_currency_records,
    _date_range,
    _fx_hash,
    validate_fx_cache,
)
from stocks.data.phase5_common import (
    decimal_from,
    eodhd_api_token,
    eodhd_get_json,
    read_parquet_records,
    sha256_file,
    utc_now_iso,
    zero_financial_calls,
)
from stocks.data.total_returns import TotalReturnLayout, build_total_returns_for_universe, validate_total_return_cache
from stocks.execution.idempotency import stable_hash
from stocks.ibkr.contract_cache import ContractCacheRow
from stocks.research.instrument_manifest import InstrumentManifestLayout, validate_instrument_manifest


def collect_corporate_actions_for_universe_v1_1(
    *,
    project_root: Path,
    rows: list[ContractCacheRow],
    start: date,
    end: date,
    env_file: str | Path = ".env",
) -> dict[str, Any]:
    """Harden the frozen Phase 5 collector without changing its source contract."""
    token = eodhd_api_token(project_root / env_file)
    if not token:
        return _error("NO_GO_MISSING_EODHD_API_KEY")
    instrument_layout = InstrumentManifestLayout.from_project_root(project_root)
    manifest_validation = validate_instrument_manifest(instrument_layout)
    if manifest_validation["status"] != "GO":
        return _error("NO_GO_INVALID_RESEARCH_MANIFEST", manifest_validation=manifest_validation)

    instruments = list(yaml.safe_load(instrument_layout.path.read_text(encoding="utf-8"))["instruments"])
    by_symbol = {row.contract.symbol.upper(): row for row in rows}
    records: list[dict[str, Any]] = []
    provider_errors: list[dict[str, Any]] = []
    downloaded_at = utc_now_iso()
    for item in instruments:
        row = by_symbol.get(str(item["symbol"]).upper())
        if row is None:
            provider_errors.append({"symbol": item["symbol"], "status": "MISSING_CONTRACT_CACHE"})
            continue
        ticker = eodhd_ticker(row.contract.symbol)
        try:
            dividends = eodhd_get_json(
                f"div/{ticker}", token=token, params={"from": start.isoformat(), "to": end.isoformat()}
            )
            splits = eodhd_get_json(
                f"splits/{ticker}", token=token, params={"from": start.isoformat(), "to": end.isoformat()}
            )
        except Exception as exc:  # pragma: no cover - provider boundary
            provider_errors.append(
                {"symbol": row.contract.symbol, "status": "PROVIDER_ERROR", **_safe_provider_error(exc)}
            )
            continue
        records.extend(
            _dividend_record(row, payload, downloaded_at=downloaded_at)
            for payload in _ensure_list(dividends)
        )
        records.extend(
            _split_record(row, payload, downloaded_at=downloaded_at)
            for payload in _ensure_list(splits)
        )

    records = _dedupe_events(records)
    layout = CorporateActionLayout.from_project_root(project_root)
    write_corporate_action_outputs(layout, records)
    validation = validate_corporate_action_cache(layout)
    manifest = corporate_action_manifest(layout, records, validation, provider_errors)
    layout.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "schema": "corporate_action_collection_v1_1",
        "status": "GO" if validation["status"] == "GO" and not provider_errors else "NO_GO",
        "event_count": len(records),
        "provider_errors": provider_errors,
        "cache_validation": validation,
        "manifest": manifest,
        "financial_calls": zero_financial_calls(),
    }


def collect_fx_for_universe_v1_1(
    *,
    project_root: Path,
    rows: list[ContractCacheRow],
    start: date,
    end: date,
    base_currency: str = "EUR",
    env_file: str | Path = ".env",
) -> dict[str, Any]:
    if base_currency != "EUR":
        return _error("NO_GO_UNSUPPORTED_BASE_CURRENCY")
    token = eodhd_api_token(project_root / env_file)
    if not token:
        return _error("NO_GO_MISSING_EODHD_API_KEY")
    currencies = sorted({row.contract.currency for row in rows} | {base_currency})
    downloaded_at = utc_now_iso()
    records: list[dict[str, Any]] = []
    provider_errors: list[dict[str, Any]] = []
    fallback_rows = 0
    for currency in currencies:
        if currency == base_currency:
            records.extend(
                _constant_currency_records(
                    currency,
                    start=start,
                    end=end,
                    downloaded_at=downloaded_at,
                )
            )
            continue
        try:
            currency_records, currency_fallback_rows = _multi_provider_currency_records(
                currency,
                token=token,
                start=start,
                end=end,
                downloaded_at=downloaded_at,
            )
            records.extend(currency_records)
            fallback_rows += currency_fallback_rows
        except Exception as exc:  # pragma: no cover - provider boundary
            provider_errors.append(
                {"currency": currency, "status": "PROVIDER_ERROR", "error_class": type(exc).__name__}
            )

    observed_currencies = {str(item.get("base_currency")) for item in records}
    currencies_complete = set(currencies).issubset(observed_currencies)
    layout = FxCacheLayout.from_project_root(project_root)
    if not currencies_complete:
        return {
            "schema": "fx_collection_v1_1",
            "status": "NO_GO_REQUIRED_CURRENCY_MISSING",
            "currencies": currencies,
            "observed_currencies": sorted(observed_currencies),
            "provider_errors": provider_errors,
            "cache_preserved": layout.fx_daily_parquet.exists(),
            "financial_calls": zero_financial_calls(),
        }

    layout.data_dir.mkdir(parents=True, exist_ok=True)
    temporary = layout.fx_daily_parquet.with_suffix(".v1_1.tmp.parquet")
    pq.write_table(pa.Table.from_pylist(records), temporary)
    previous = layout.fx_daily_parquet.with_suffix(".previous.parquet")
    if layout.fx_daily_parquet.exists():
        previous.write_bytes(layout.fx_daily_parquet.read_bytes())
    temporary.replace(layout.fx_daily_parquet)
    validation = validate_fx_cache(layout)
    if validation["status"] != "GO" and previous.exists():
        previous.replace(layout.fx_daily_parquet)
        validation = validate_fx_cache(layout)
        return {
            "schema": "fx_collection_v1_1",
            "status": "NO_GO_VALIDATION_FAILED_CACHE_RESTORED",
            "provider_errors": provider_errors,
            "cache_validation": validation,
            "financial_calls": zero_financial_calls(),
        }
    previous.unlink(missing_ok=True)
    manifest = {
        "schema": "fx_manifest_v1_1",
        "status": "GO" if validation["status"] == "GO" and not provider_errors else "NO_GO",
        "generated_at": utc_now_iso(),
        "base_currency": base_currency,
        "currencies": currencies,
        "source_priority": ["EODHD_FOREX", "ECB_REFERENCE_RATE", "YFINANCE_FOREX_GAP_FILL"],
        "fallback_rows": fallback_rows,
        "provider_errors": provider_errors,
        "content_hash": validation["content_hash"],
        "financial_calls": zero_financial_calls(),
    }
    layout.manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return {
        "schema": "fx_collection_v1_1",
        "status": manifest["status"],
        "row_count": len(records),
        "currencies": currencies,
        "fallback_rows": fallback_rows,
        "cache_validation": validation,
        "manifest": manifest,
        "financial_calls": zero_financial_calls(),
    }


def fx_status_v1_1(layout: FxCacheLayout) -> dict[str, Any]:
    validation = validate_fx_cache(layout)
    manifest = _read_json(layout.manifest_json)
    expected = {str(item) for item in manifest.get("currencies", [])}
    observed = {
        str(item.get("base_currency"))
        for item in read_parquet_records(layout.fx_daily_parquet)
    }
    currencies_complete = expected.issubset(observed)
    return {
        "schema": "fx_status_v1_1",
        "status": (
            "READY"
            if manifest.get("status") == "GO" and validation["status"] == "GO" and currencies_complete
            else "NO_GO"
        ),
        "manifest_exists": layout.manifest_json.exists(),
        "manifest_status": manifest.get("status", "MISSING"),
        "expected_currencies": sorted(expected),
        "observed_currencies": sorted(observed),
        "currencies_complete": currencies_complete,
        "cache_validation": validation,
        "financial_calls": zero_financial_calls(),
    }


def build_total_returns_for_universe_v1_1(**kwargs: Any) -> dict[str, Any]:
    project_root = Path(kwargs["project_root"])
    manifest_path = project_root / "data/total_returns/total_return_manifest.json"
    instrument_layout = InstrumentManifestLayout.from_project_root(project_root)
    manifest_validation = validate_instrument_manifest(instrument_layout)
    if manifest_validation["status"] != "GO":
        return _error("NO_GO_INVALID_RESEARCH_MANIFEST", manifest_validation=manifest_validation)
    instruments = list(yaml.safe_load(instrument_layout.path.read_text(encoding="utf-8"))["instruments"])
    rows = list(kwargs["rows"])
    by_symbol = {row.contract.symbol.upper(): row for row in rows}
    requested_symbols = sorted({str(item["symbol"]).upper() for item in instruments})
    missing_contract_symbols = sorted(set(requested_symbols) - set(by_symbol))
    if missing_contract_symbols:
        layout = TotalReturnLayout.from_project_root(project_root)
        manifest = {
            "schema": "total_return_manifest_v1_1",
            "status": "NO_GO",
            "coverage_status": "INCOMPLETE_CONTRACT_IDENTITIES",
            "generated_at": utc_now_iso(),
            "base_currency": kwargs.get("base_currency", "EUR"),
            "requested_instrument_count": len(requested_symbols),
            "resolved_instrument_count": len(set(requested_symbols) & set(by_symbol)),
            "missing_contract_symbols": missing_contract_symbols,
            "cache_preserved": any(
                layout.data_dir.glob("security_type=STK/con_id=*/interval=1d/total_returns.parquet")
            ),
            "financial_calls": zero_financial_calls(),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        return {
            "schema": "total_return_build_v1_1",
            "status": "NO_GO_MISSING_CONTRACT_IDENTITIES",
            "requested_instrument_count": len(requested_symbols),
            "resolved_instrument_count": manifest["resolved_instrument_count"],
            "missing_contract_symbols": missing_contract_symbols,
            "cache_preserved": manifest["cache_preserved"],
            "manifest": manifest,
            "manifest_exists": True,
            "financial_calls": zero_financial_calls(),
        }

    report = build_total_returns_for_universe(**kwargs)
    if report["status"] != "GO":
        report["schema"] = "total_return_build_v1_1"
        report["manifest_exists"] = manifest_path.exists()
        return report
    manifest = _read_json(manifest_path)
    fx_records = read_parquet_records(FxCacheLayout.from_project_root(project_root).fx_daily_parquet)
    fx_sources = sorted({str(item.get("source", "UNKNOWN")) for item in fx_records})
    manifest.update(
        {
            "schema": "total_return_manifest_v1_1",
            "fx_source": fx_sources[0] if len(fx_sources) == 1 else "MULTI_PROVIDER_FX",
            "fx_sources": fx_sources,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    report["manifest"] = manifest
    report["schema"] = "total_return_build_v1_1"
    return report


def total_return_status_v1_1(layout: TotalReturnLayout) -> dict[str, Any]:
    validation = validate_total_return_cache(layout)
    manifest = _read_json(layout.manifest_json)
    return {
        "schema": "total_return_status_v1_1",
        "status": "READY" if manifest.get("status") == "GO" and validation["status"] == "GO" else "NO_GO",
        "manifest_exists": layout.manifest_json.exists(),
        "manifest_status": manifest.get("status", "MISSING_OR_INVALID"),
        "coverage_status": manifest.get("coverage_status", "NOT_REPORTED"),
        "missing_contract_symbols": manifest.get("missing_contract_symbols", []),
        "cache_validation": validation,
        "financial_calls": zero_financial_calls(),
    }


def _safe_provider_error(exc: Exception) -> dict[str, Any]:
    """Return diagnostics without persisting URLs, query strings, or credentials."""
    payload: dict[str, Any] = {"error_class": type(exc).__name__}
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status_code, int):
        payload["http_status"] = status_code
    return payload


def phase5_1_freeze(project_root: Path) -> dict[str, Any]:
    source_paths = [
        "src/stocks/data/phase5_1.py",
        "tests/test_phase5_1_multi_provider_fx.py",
    ]
    data_paths = [
        "data/fx/fx_daily.parquet",
        "data/fx/fx_manifest.json",
        "data/total_returns/total_return_manifest.json",
    ]
    payload = {
        "schema": "phase5_1_multi_provider_fx_freeze_v1",
        "status": "PHASE5_1_MULTI_PROVIDER_FX_FROZEN_GO",
        "generated_at": utc_now_iso(),
        "phase5_v1_artifact_unchanged": True,
        "source_hashes": {path: sha256_file(project_root / path) for path in source_paths},
        "data_hashes": {path: sha256_file(project_root / path) for path in data_paths},
        "broker_write_calls": 0,
        "execution_authority": "NONE",
    }
    payload["content_hash"] = stable_hash(payload)
    path = project_root / "output/ibkr/phase5_1-freeze-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def _multi_provider_currency_records(
    currency: str,
    *,
    token: str,
    start: date,
    end: date,
    downloaded_at: str,
) -> tuple[list[dict[str, Any]], int]:
    primary = _eodhd_currency_rates(currency, token=token, start=start, end=end)
    if not primary:
        raise ValueError(f"EODHD returned no FX rates for {currency}")
    try:
        ecb = _ecb_currency_rates(currency, start=start, end=end)
    except (OSError, TimeoutError, urllib.error.URLError):
        ecb = {}
    try:
        yahoo = _yfinance_currency_rates(currency, start=start, end=end)
    except (OSError, TimeoutError, ValueError):
        yahoo = {}
    merged, sources, ecb_rows = _merge_currency_rates(
        primary,
        ecb,
        fallback_source="ECB_REFERENCE_RATE",
    )
    yahoo_rows = _add_missing_currency_rates(
        merged,
        sources,
        yahoo,
        source="YFINANCE_FOREX_GAP_FILL",
    )
    return (
        _filled_multi_source_records(
            currency,
            by_date=merged,
            source_by_date=sources,
            start=start,
            end=end,
            downloaded_at=downloaded_at,
        ),
        ecb_rows + yahoo_rows,
    )


def _eodhd_currency_rates(currency: str, *, token: str, start: date, end: date) -> dict[date, Decimal]:
    raw = eodhd_get_json(
        f"eod/EUR{currency}.FOREX",
        token=token,
        params={"from": start.isoformat(), "to": end.isoformat(), "period": "d"},
    )
    result: dict[date, Decimal] = {}
    for item in raw if isinstance(raw, list) else []:
        close = decimal_from(item.get("adjusted_close") or item.get("close"))
        if close is not None and close > 0:
            result[date.fromisoformat(str(item["date"])[:10])] = Decimal("1") / close
    return result


def _ecb_currency_rates(currency: str, *, start: date, end: date) -> dict[date, Decimal]:
    result: dict[date, Decimal] = {}
    cursor = start
    while cursor <= end:
        chunk_end = min(end, date(min(cursor.year + 4, end.year), 12, 31))
        query = urllib.parse.urlencode(
            {"startPeriod": cursor.isoformat(), "endPeriod": chunk_end.isoformat(), "format": "csvdata"}
        )
        url = f"https://data-api.ecb.europa.eu/service/data/EXR/D.{currency}.EUR.SP00.A?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "stocks-fx-research/1.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8-sig")
        for item in csv.DictReader(io.StringIO(payload)):
            quote = decimal_from(item.get("OBS_VALUE"))
            if quote is not None and quote > 0:
                result[date.fromisoformat(str(item["TIME_PERIOD"])[:10])] = Decimal("1") / quote
        cursor = chunk_end + timedelta(days=1)
    return result


def _yfinance_currency_rates(currency: str, *, start: date, end: date) -> dict[date, Decimal]:
    import yfinance as yf

    raw = yf.download(
        f"EUR{currency}=X",
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        repair=True,
        keepna=False,
        progress=False,
        threads=False,
        timeout=30,
        multi_level_index=False,
    )
    if raw.empty:
        return {}
    column = "Adj Close" if "Adj Close" in raw.columns else "Close"
    result: dict[date, Decimal] = {}
    for index, value in raw[column].items():
        quote = decimal_from(value)
        if quote is not None and quote > 0:
            result[index.date()] = Decimal("1") / quote
    return result


def _merge_currency_rates(
    primary: dict[date, Decimal],
    fallback: dict[date, Decimal],
    *,
    fallback_source: str = "YFINANCE_FOREX_GAP_FILL",
) -> tuple[dict[date, Decimal], dict[date, str], int]:
    merged = dict(primary)
    sources = {item_date: "EODHD_FOREX" for item_date in primary}
    count = _add_missing_currency_rates(merged, sources, fallback, source=fallback_source)
    return merged, sources, count


def _add_missing_currency_rates(
    merged: dict[date, Decimal],
    sources: dict[date, str],
    fallback: dict[date, Decimal],
    *,
    source: str,
) -> int:
    count = 0
    for item_date, rate in fallback.items():
        if item_date not in merged:
            merged[item_date] = rate
            sources[item_date] = source
            count += 1
    return count


def _filled_multi_source_records(
    currency: str,
    *,
    by_date: dict[date, Decimal],
    source_by_date: dict[date, str],
    start: date,
    end: date,
    downloaded_at: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    last_rate: Decimal | None = None
    last_date: date | None = None
    last_source = "UNKNOWN"
    for day in _date_range(start, end):
        if day in by_date:
            last_rate = by_date[day]
            last_date = day
            last_source = source_by_date.get(day, "UNKNOWN")
        if last_rate is None or last_date is None:
            continue
        record = {
            "timestamp_utc": f"{day.isoformat()}T00:00:00+00:00",
            "session_date": day.isoformat(),
            "base_currency": currency,
            "quote_currency": "EUR",
            "rate": format(last_rate, "f"),
            "rate_definition": "EUR_PER_BASE_CURRENCY",
            "source": last_source,
            "available_at": f"{day.isoformat()}T23:59:59+00:00",
            "downloaded_at": downloaded_at,
            "is_forward_filled": day != last_date,
            "forward_fill_age": (day - last_date).days,
        }
        record["content_hash"] = _fx_hash(record)
        records.append({field: record[field] for field in FX_FIELDS})
    return records


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _error(status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": "phase5_1_data_collection_v1",
        "status": status,
        **extra,
        "financial_calls": zero_financial_calls(),
    }


__all__ = [
    "build_total_returns_for_universe_v1_1",
    "collect_corporate_actions_for_universe_v1_1",
    "collect_fx_for_universe_v1_1",
    "fx_status_v1_1",
    "phase5_1_freeze",
    "total_return_status_v1_1",
]
