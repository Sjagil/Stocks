from __future__ import annotations

import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd
import pyarrow.dataset as ds
from dotenv import dotenv_values

from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_3.datascraper_adapter import DEFAULT_DATASCRAPER_ROOT, DatascraperAdapter


CANONICAL_INTERVALS = (
    "15m", "1h", "2h", "4h", "6h", "12h", "1d", "1w", "1mo"
)
FORBIDDEN_SWING_INTERVALS = frozenset({"1min", "5m", "30m", "tick", "ticks"})
INTERVAL_ALIASES = {
    "1m": "1mo",
    "month": "1mo",
    "1month": "1mo",
    "1wk": "1w",
    "60m": "1h",
}
INTRADAY_INTERVALS = {"15m", "1h", "2h", "4h", "6h", "12h"}
DEFAULT_SYMBOLS = ("AAPL", "ASML", "ON", "SPY")
EODHD_BASE = "https://eodhd.com/api"
FRESHNESS_MAX_AGE_HOURS = {
    "15m": 4.0,
    "1h": 12.0,
    "2h": 24.0,
    "4h": 72.0,
    "6h": 72.0,
    "12h": 96.0,
    "1d": 240.0,
    "1w": 24.0 * 42.0,
    "1mo": 24.0 * 62.0,
}
EXCHANGE_TIMEZONE_CALENDARS = {
    "America/New_York": "XNYS",
    "US/Eastern": "XNYS",
    "Europe/Amsterdam": "XAMS",
    "Europe/London": "XLON",
    "Europe/Paris": "XPAR",
    "Europe/Berlin": "XETR",
    "Europe/Zurich": "XSWX",
    "Europe/Madrid": "XMAD",
    "Asia/Hong_Kong": "XHKG",
    "Asia/Tokyo": "XTKS",
    "Asia/Singapore": "XSES",
    "Asia/Kolkata": "XBOM",
    "Asia/Seoul": "XKRX",
    "Australia/Sydney": "XASX",
}


@dataclass(frozen=True)
class MultiTimeframeLayout:
    project_root: Path

    @property
    def private_root(self) -> Path:
        return self.project_root / "data" / "research" / "multitimeframe" / "private"

    @property
    def output_root(self) -> Path:
        return self.project_root / "output" / "research" / "multitimeframe"

    def bars_path(
        self,
        *,
        provider: str,
        symbol: str,
        interval: str,
        source_interval: str,
    ) -> Path:
        return (
            self.private_root
            / f"provider={_safe_partition(provider)}"
            / f"symbol={_safe_partition(symbol)}"
            / f"interval={canonical_interval(interval)}"
            / f"source_interval={canonical_interval(source_interval)}"
            / "bars.parquet"
        )


def canonical_interval(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in FORBIDDEN_SWING_INTERVALS:
        raise ValueError(f"FORBIDDEN_SWING_TIMEFRAME:{normalized}")
    normalized = INTERVAL_ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_INTERVALS:
        raise ValueError(f"unsupported interval: {value}")
    return normalized


def parse_intervals(value: str | Iterable[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    result: list[str] = []
    for item in raw:
        interval = canonical_interval(str(item))
        if interval not in result:
            result.append(interval)
    if not result:
        raise ValueError("at least one interval is required")
    return result


def parse_symbols(value: str | Iterable[str]) -> list[str]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    symbols = sorted({str(item).strip().upper() for item in raw if str(item).strip()})
    if not symbols:
        raise ValueError("at least one symbol is required")
    if len(symbols) > 50:
        raise ValueError("a bounded collection run supports at most 50 symbols")
    return symbols


def multitimeframe_schema(project_root: Path) -> dict[str, Any]:
    layout = MultiTimeframeLayout(project_root)
    payload = {
        "schema": "multi_timeframe_market_data_v1",
        "status": "GO",
        "canonical_intervals": list(CANONICAL_INTERVALS),
        "interval_aliases": {
            "1m": "1mo (one month)",
            "1M": "1mo (one month)",
            "month": "1mo (one month)",
            "1min": "forbidden",
        },
        "forbidden_intervals": sorted(FORBIDDEN_SWING_INTERVALS),
        "required_fields": [
            "timestamp_utc",
            "session_date",
            "symbol",
            "provider",
            "source_provider",
            "interval",
            "target_interval",
            "source_interval",
            "derivation",
            "bar_origin",
            "aggregation_rule",
            "partial_bucket",
            "session",
            "timezone",
            "quality_status",
            "fetched_at",
            "ingested_at",
            "open",
            "high",
            "low",
            "close",
            "adjusted_close",
            "volume",
            "source_bar_count",
            "is_partial",
            "row_hash",
        ],
        "native_provider_intervals": {
            "YFINANCE": ["15m", "1h", "1d", "1w", "1mo"],
            "EODHD": ["1h", "1d"],
            "IBKR_PHASE4_LOCAL": ["1d"],
            "STOCKS_PIT_EODHD_LOCAL": ["1d"],
            "DATASCRAPER_EODHD_EXPORT": ["1d"],
            "DATASCRAPER_YFINANCE_LOCAL": ["1d"],
        },
        "derivation_policy": {
            "2h": "session-anchored aggregation from native 1h",
            "4h": "session-anchored aggregation from native 1h; final RTH bucket may be partial",
            "6h": "session-anchored aggregation from native 1h; final RTH bucket may be partial",
            "12h": "session-anchored aggregation from native 1h; final RTH bucket may be partial",
            "1w": "calendar-week aggregation from daily bars",
            "1mo": "calendar-month aggregation from daily bars",
            "no_upsampling": True,
            "closed_bars_only": True,
        },
        "storage_root": str(layout.private_root),
        "public_artifact_root": str(layout.output_root),
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
    }
    _write_public(layout.output_root / "schema.json", payload)
    return payload


def provider_inventory(project_root: Path, *, datascraper_root: Path = DEFAULT_DATASCRAPER_ROOT) -> dict[str, Any]:
    layout = MultiTimeframeLayout(project_root)
    env = _env_values(project_root)
    yfinance_version: str | None
    try:
        import yfinance as yf

        yfinance_version = str(getattr(yf, "__version__", "unknown"))
    except ImportError:
        yfinance_version = None
    adapter = DatascraperAdapter(datascraper_root)
    manifests = adapter.manifests() if adapter.export_root.is_dir() else []
    phase4_files = list((project_root / "data" / "bars").rglob("bars.parquet"))
    sources = [
        _source(
            "YFINANCE",
            yfinance_version is not None,
            ["15m", "1h", "1d", "1w", "1mo"],
            "NETWORK_API",
        ),
        _source("EODHD", bool(_first_secret(env, "EODHD_API_KEY", "EOD_API_KEY", "EODHISTORICALDATA_API_KEY")), ["1h", "1d"], "NETWORK_API"),
        _source("IBKR_PHASE4_LOCAL", bool(phase4_files), ["1d"], "LOCAL_FROZEN_CACHE", file_count=len(phase4_files)),
        _source(
            "STOCKS_YFINANCE_LOCAL",
            (project_root / "data" / "research" / "critical_trading" / "yfinance" / "manifest.json").is_file(),
            ["1d"],
            "LOCAL_CACHE",
        ),
        _source(
            "STOCKS_PIT_EODHD_LOCAL",
            (project_root / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet").is_file(),
            ["1d"],
            "LOCAL_PRIVATE_CACHE",
        ),
        _source(
            "DATASCRAPER_EODHD_EXPORT",
            any(item.valid and item.manifest.get("dataset") == "prices" for item in manifests),
            ["1d"],
            "VALIDATED_EXPORT",
            file_count=sum(item.valid and item.manifest.get("dataset") == "prices" for item in manifests),
        ),
        _source(
            "DATASCRAPER_YFINANCE_LOCAL",
            (datascraper_root / "data" / "yahoo_finance_daily_bars" / "yahoo_daily_bars_combined.parquet").is_file(),
            ["1d"],
            "LOCAL_CACHE",
        ),
        _source(
            "DATASCRAPER_EODHD_SILVER",
            (datascraper_root / "output" / "apex_data_lake" / "silver" / "eodhd_ohlcv_normalized.csv").is_file(),
            ["1d"],
            "LOCAL_CACHE",
        ),
        _source(
            "DATASCRAPER_EODHD_HISTORICAL_LAKE",
            (datascraper_root / "data" / "historical" / "eodhd" / "eodhd_raw_ohlcv_v7.parquet").is_file(),
            ["1d"],
            "LOCAL_HISTORICAL_LAKE",
        ),
        _source(
            "DATASCRAPER_EODHD_STOCK_1999",
            (datascraper_root / "output" / "apex_data_lake" / "silver" / "v4621_stock_1999_ohlcv.csv").is_file(),
            ["1d"],
            "LOCAL_HISTORICAL_LAKE",
        ),
        {
            "provider": "DATASCRAPER_WEBSITE_AND_RSS_SCRAPERS",
            "status": "CONTEXT_ONLY_NOT_OHLCV",
            "available": True,
            "native_intervals": [],
            "source_type": "FORWARD_CONTEXT",
        },
        {
            "provider": "DATASCRAPER_HISTORICAL_CANDLE_SERVICE_V1",
            "status": "BLOCKED_DETERMINISTIC_FIXTURE_STUB",
            "available": False,
            "native_intervals": [],
            "source_type": "SYNTHETIC_FIXTURE",
        },
    ]
    payload = {
        "schema": "multi_timeframe_provider_inventory_v1",
        "status": "GO",
        "generated_at": _utc_now(),
        "sources": sources,
        "available_source_count": sum(bool(row.get("available")) for row in sources),
        "datascraper_root": str(datascraper_root.resolve()),
        "secrets_read": False,
        "secret_presence_only": True,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }
    payload["content_hash"] = stable_hash(payload)
    _write_public(layout.output_root / "provider-inventory.json", payload)
    return payload


def collect_multitimeframe_data(
    project_root: Path,
    *,
    symbols: Iterable[str],
    intervals: Iterable[str],
    providers: Iterable[str] = ("all",),
    start: str | None = None,
    end: str | None = None,
    lookback_days: int = 60,
    datascraper_root: Path = DEFAULT_DATASCRAPER_ROOT,
) -> dict[str, Any]:
    selected_symbols = parse_symbols(symbols)
    selected_intervals = parse_intervals(intervals)
    selected_providers = _providers(providers)
    if not 1 <= int(lookback_days) <= 3650:
        raise ValueError("lookback_days must be between 1 and 3650")
    start_date = _parse_date(start) if start else date.today() - timedelta(days=int(lookback_days))
    end_date = _parse_date(end) if end else date.today()
    if start_date > end_date:
        raise ValueError("start must be on or before end")
    layout = MultiTimeframeLayout(project_root)
    layout.private_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    provider_calls = 0

    if "local" in selected_providers or "ibkr" in selected_providers:
        results.extend(_import_ibkr_phase4(layout, selected_symbols, selected_intervals))
    if "local" in selected_providers or "yfinance" in selected_providers:
        results.extend(_import_stocks_yfinance(layout, selected_symbols, selected_intervals))
    if "local" in selected_providers or "eodhd" in selected_providers:
        results.extend(_import_stocks_pit(layout, selected_symbols, selected_intervals))
    if "datascraper" in selected_providers:
        results.extend(_import_datascraper(layout, datascraper_root, selected_symbols, selected_intervals))
    if "yfinance" in selected_providers:
        network, calls = _collect_yfinance(
            layout,
            selected_symbols,
            selected_intervals,
            start_date=start_date,
            end_date=end_date,
        )
        results.extend(network)
        provider_calls += calls
    if "eodhd" in selected_providers:
        network, calls = _collect_eodhd(
            layout,
            selected_symbols,
            selected_intervals,
            start_date=start_date,
            end_date=end_date,
        )
        results.extend(network)
        provider_calls += calls

    validation = validate_multitimeframe_cache(project_root)
    payload = {
        "schema": "multi_timeframe_collection_run_v1",
        "status": "GO" if any(int(row.get("rows_written", 0)) > 0 for row in results) else "NO_DATA",
        "generated_at": _utc_now(),
        "symbols": selected_symbols,
        "intervals": selected_intervals,
        "providers": sorted(selected_providers),
        "requested_start": start_date.isoformat(),
        "requested_end": end_date.isoformat(),
        "results": results,
        "provider_calls_read_only": provider_calls,
        "broker_calls": 0,
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "cache_validation_status": validation["status"],
    }
    payload["content_hash"] = stable_hash(payload)
    _write_public(layout.output_root / "collection-manifest.json", payload)
    return payload


def validate_multitimeframe_cache(
    project_root: Path,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    layout = MultiTimeframeLayout(project_root)
    observed_at = as_of or datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    quarantined_files = 0
    total_rows = duplicates = invalid_ohlc = timezone_errors = 0
    symbols: set[str] = set()
    intervals: set[str] = set()
    providers: set[str] = set()
    for path in sorted(layout.private_root.rglob("bars.parquet")):
        path_text = str(path).lower()
        if any(f"interval={item}" in path_text for item in ("5m", "30m")):
            quarantined_files += 1
            continue
        try:
            frame = pd.read_parquet(path)
            result = _validate_frame(frame)
            exchange_timezone = result.get("exchange_timezone")
            result["freshness"] = bar_freshness(
                result.get("last_timestamp"),
                interval=str(result.get("interval") or ""),
                observed_at=observed_at,
                exchange_timezone=(
                    str(exchange_timezone) if exchange_timezone else None
                ),
            )
            result.update({"path_hash": _file_hash(path), "relative_path": str(path.relative_to(layout.private_root))})
            rows.append(result)
            total_rows += result["row_count"]
            duplicates += result["duplicate_rows"]
            invalid_ohlc += result["invalid_ohlc_rows"]
            timezone_errors += result["timezone_errors"]
            symbols.update(str(value) for value in frame.get("symbol", []))
            intervals.update(str(value) for value in frame.get("interval", []))
            providers.update(str(value) for value in frame.get("provider", []))
        except (OSError, ValueError, KeyError) as exc:
            rows.append(
                {
                    "status": "INVALID_FILE",
                    "relative_path": str(path.relative_to(layout.private_root)),
                    "error_class": type(exc).__name__,
                    "row_count": 0,
                    "duplicate_rows": 0,
                    "invalid_ohlc_rows": 0,
                    "timezone_errors": 0,
                }
            )
    invalid_files = sum(row["status"] != "GO" for row in rows)
    coverage = _coverage(rows)
    fresh_files = sum(
        row.get("freshness", {}).get("status") == "FRESH_CLOSED_BAR"
        for row in rows
    )
    stale_files = sum(
        row.get("freshness", {}).get("status") == "STALE_BAR_BLOCKED"
        for row in rows
    )
    payload = {
        "schema": "multi_timeframe_cache_validation_v1",
        "status": "GO" if rows and invalid_files == 0 and duplicates == 0 and invalid_ohlc == 0 and timezone_errors == 0 else "NO_GO",
        "generated_at": _utc_now(),
        "freshness_observed_at": observed_at.isoformat(),
        "freshness_policy": "EXCHANGE_SESSION_AWARE_CLOSED_BAR_V1",
        "file_count": len(rows),
        "row_count": total_rows,
        "symbol_count": len(symbols),
        "provider_count": len(providers),
        "interval_count": len(intervals),
        "intervals_present": sorted(intervals, key=_interval_order),
        "duplicate_rows": duplicates,
        "invalid_ohlc_rows": invalid_ohlc,
        "timezone_errors": timezone_errors,
        "invalid_file_count": invalid_files,
        "fresh_file_count": fresh_files,
        "stale_file_count": stale_files,
        "forbidden_interval_quarantined_file_count": quarantined_files,
        "legacy_forbidden_quarantined_file_count": quarantined_files,
        "coverage": coverage,
        "files": rows,
        "broker_calls": 0,
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
    }
    payload["content_hash"] = stable_hash(payload)
    _write_public(layout.output_root / "cache-validation.json", payload)
    return payload


def multitimeframe_status(project_root: Path) -> dict[str, Any]:
    layout = MultiTimeframeLayout(project_root)
    validation_path = layout.output_root / "cache-validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else validate_multitimeframe_cache(project_root)
    collection_path = layout.output_root / "collection-manifest.json"
    collection = json.loads(collection_path.read_text(encoding="utf-8")) if collection_path.is_file() else {}
    expected = {
        (str(symbol), canonical_interval(str(interval)))
        for symbol in collection.get("symbols", [])
        for interval in collection.get("intervals", [])
    }
    actual = {
        (str(row.get("symbol")), canonical_interval(str(row.get("interval"))))
        for row in validation.get("coverage", [])
        if row.get("symbol") and row.get("interval")
    }
    covered = expected & actual
    coverage_ratio = len(covered) / len(expected) if expected else 0.0
    current_pairs = {
        (str(row.get("symbol")), canonical_interval(str(row.get("interval"))))
        for row in validation.get("coverage", [])
        if row.get("symbol")
        and row.get("interval")
        and row.get("current_data_status") == "FRESH_CLOSED_BAR"
    }
    requested_current_pairs = {
        pair for pair in expected if pair[1] in INTRADAY_INTERVALS
    }
    fresh_requested_pairs = requested_current_pairs & current_pairs
    current_ratio = (
        len(fresh_requested_pairs) / len(requested_current_pairs)
        if requested_current_pairs
        else 1.0
    )
    current_data_status = (
        "CURRENT_DATA_GO"
        if current_ratio == 1.0
        else "CURRENT_DATA_DEGRADED"
        if fresh_requested_pairs
        else "CURRENT_DATA_NOT_READY"
    )
    payload = {
        "schema": "multi_timeframe_status_v1",
        "status": "MULTI_TIMEFRAME_DATA_GO"
        if validation.get("status") == "GO" and (not expected or coverage_ratio == 1.0)
        else "MULTI_TIMEFRAME_DATA_NOT_READY",
        "canonical_intervals": list(CANONICAL_INTERVALS),
        "intervals_present": validation.get("intervals_present", []),
        "file_count": validation.get("file_count", 0),
        "row_count": validation.get("row_count", 0),
        "provider_count": validation.get("provider_count", 0),
        "freshness_policy": validation.get(
            "freshness_policy",
            "WALL_CLOCK_LEGACY",
        ),
        "requested_symbol_interval_pairs": len(expected),
        "covered_symbol_interval_pairs": len(covered),
        "coverage_ratio": round(coverage_ratio, 8),
        "current_data_status": current_data_status,
        "requested_current_symbol_interval_pairs": len(
            requested_current_pairs
        ),
        "fresh_current_symbol_interval_pairs": len(
            fresh_requested_pairs
        ),
        "current_data_ratio": round(current_ratio, 8),
        "stale_current_symbol_interval_pairs": [
            {"symbol": symbol, "interval": interval}
            for symbol, interval in sorted(
                requested_current_pairs - current_pairs,
                key=lambda item: (item[0], _interval_order(item[1])),
            )
        ],
        "missing_symbol_interval_pairs": [
            {"symbol": symbol, "interval": interval}
            for symbol, interval in sorted(expected - actual, key=lambda item: (item[0], _interval_order(item[1])))
        ],
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
    }
    _write_public(layout.output_root / "status.json", payload)
    return payload


def audit_multitimeframe_sources(project_root: Path) -> dict[str, Any]:
    layout = MultiTimeframeLayout(project_root)
    variants: dict[tuple[str, str], list[tuple[str, pd.DataFrame]]] = {}
    for path in sorted(layout.private_root.rglob("bars.parquet")):
        try:
            frame = pd.read_parquet(
                path,
                columns=[
                    "timestamp_utc",
                    "symbol",
                    "provider",
                    "interval",
                    "source_interval",
                    "adjustment_mode",
                    "close",
                ],
            )
        except (OSError, ValueError, KeyError):
            continue
        if frame.empty:
            continue
        symbol = str(frame["symbol"].iloc[0])
        try:
            interval = canonical_interval(str(frame["interval"].iloc[0]))
        except ValueError:
            continue
        adjustment_mode = str(frame["adjustment_mode"].iloc[0])
        variant = f"{frame['provider'].iloc[0]}:{frame['source_interval'].iloc[0]}:{adjustment_mode}"
        normalized = frame[["timestamp_utc", "close"]].copy()
        normalized["timestamp_utc"] = pd.to_datetime(normalized["timestamp_utc"], utc=True)
        normalized.attrs["adjustment_mode"] = adjustment_mode
        variants.setdefault((symbol, interval), []).append((variant, normalized))
    comparisons: list[dict[str, Any]] = []
    incompatible_adjustment_pairs = 0
    for (symbol, interval), items in variants.items():
        for left_index, (left_name, left) in enumerate(items):
            for right_name, right in items[left_index + 1 :]:
                if left.attrs.get("adjustment_mode") != right.attrs.get("adjustment_mode"):
                    incompatible_adjustment_pairs += 1
                    continue
                joined = left.merge(right, on="timestamp_utc", suffixes=("_left", "_right"))
                if joined.empty:
                    continue
                denominator = joined[["close_left", "close_right"]].abs().mean(axis=1).replace(0, math.nan)
                difference = ((joined["close_left"] - joined["close_right"]).abs() / denominator * 10_000).dropna()
                if difference.empty:
                    continue
                median = float(difference.median())
                p95 = float(difference.quantile(0.95))
                classification = (
                    "INSUFFICIENT_OVERLAP"
                    if len(difference) < 5
                    else "MATERIAL_PROVIDER_DIVERGENCE"
                    if median > 50.0
                    else "PROVIDER_ALIGNMENT_GO"
                )
                comparisons.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "left_variant": left_name,
                        "right_variant": right_name,
                        "overlap_rows": len(difference),
                        "median_absolute_close_difference_bps": round(median, 6),
                        "p95_absolute_close_difference_bps": round(p95, 6),
                        "max_absolute_close_difference_bps": round(float(difference.max()), 6),
                        "classification": classification,
                    }
                )
    privacy = _privacy_audit(layout)
    payload = {
        "schema": "multi_timeframe_cross_provider_audit_v1",
        "status": "GO",
        "generated_at": _utc_now(),
        "comparison_count": len(comparisons),
        "incompatible_adjustment_pair_count": incompatible_adjustment_pairs,
        "material_divergence_count": sum(row["classification"] == "MATERIAL_PROVIDER_DIVERGENCE" for row in comparisons),
        "insufficient_overlap_count": sum(row["classification"] == "INSUFFICIENT_OVERLAP" for row in comparisons),
        "comparisons": comparisons,
        "selection_policy": {
            "silent_provider_blending": False,
            "native_preferred_over_derived": True,
            "provider_specific_partitions_preserved": True,
            "material_divergence_is_research_warning_not_automatic_repair": True,
        },
        "broker_calls": 0,
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
        "privacy_audit_status": privacy["status"],
    }
    payload["content_hash"] = stable_hash(payload)
    _write_public(layout.output_root / "cross-provider-audit.json", payload)
    _write_public(layout.output_root / "privacy-audit.json", privacy)
    return payload


def aggregate_bars(frame: pd.DataFrame, *, target_interval: str) -> pd.DataFrame:
    target = canonical_interval(target_interval)
    if frame.empty:
        return frame.copy()
    source = canonical_interval(str(frame["interval"].iloc[0]))
    allowed = {
        ("1h", "2h"),
        ("1h", "4h"),
        ("1h", "6h"),
        ("1h", "12h"),
        ("1d", "1w"),
        ("1d", "1mo"),
    }
    if (source, target) not in allowed:
        raise ValueError(f"cannot aggregate {source} to {target}")
    work = frame.copy().sort_values("timestamp_utc")
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True)
    if target in {"1w", "1mo"}:
        dates = pd.to_datetime(work["session_date"])
        periods = dates.dt.to_period("W-FRI" if target == "1w" else "M")
        work["_bucket"] = periods.astype(str)
        final_source_date = dates.max().normalize()
        work["_partial"] = periods.dt.end_time.dt.normalize().gt(final_source_date)
    else:
        minutes = {"2h": 120, "4h": 240, "6h": 360, "12h": 720}[target]
        first = work.groupby("session_date")["timestamp_utc"].transform("min")
        elapsed = (work["timestamp_utc"] - first).dt.total_seconds().div(60)
        work["_bucket"] = work["session_date"].astype(str) + ":" + (elapsed // minutes).astype(int).astype(str)
        expected = minutes // 60
        work["_partial"] = work.groupby("_bucket")["timestamp_utc"].transform("size") < expected
    grouped = work.groupby("_bucket", sort=True, observed=True)
    out = grouped.agg(
        timestamp_utc=("timestamp_utc", "min"),
        session_date=("session_date", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        adjusted_close=("adjusted_close", "last"),
        volume=("volume", "sum"),
        dividends=("dividends", "sum"),
        stock_splits=("stock_splits", "max"),
        source_bar_count=("timestamp_utc", "size"),
        is_partial=("_partial", "max"),
    ).reset_index(drop=True)
    first_row = work.iloc[0]
    out["symbol"] = first_row["symbol"]
    out["provider"] = first_row["provider"]
    out["interval"] = target
    out["source_interval"] = source
    out["derivation"] = f"AGGREGATED_{source}_TO_{target}"
    out["exchange_timezone"] = first_row.get("exchange_timezone", "UTC")
    out["adjustment_mode"] = first_row.get("adjustment_mode", "RAW_WITH_OPTIONAL_ADJUSTED_CLOSE")
    out["received_at"] = first_row.get("received_at", _utc_now())
    finalized = _finalize(out)
    if target in {"1w", "1mo"}:
        finalized = finalized.loc[~finalized["is_partial"].astype(bool)].reset_index(drop=True)
    return finalized


def _collect_yfinance(
    layout: MultiTimeframeLayout,
    symbols: list[str],
    intervals: list[str],
    *,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], int]:
    try:
        import yfinance as yf
    except ImportError:
        return [{"provider": "YFINANCE", "status": "PACKAGE_MISSING", "rows_written": 0}], 0
    required_native: list[str] = []
    for interval in intervals:
        native = "1h" if interval in {"2h", "4h", "6h", "12h"} else "1d" if interval in {"1w", "1mo"} else interval
        if native not in required_native:
            required_native.append(native)
    results: list[dict[str, Any]] = []
    calls = 0
    for symbol in symbols:
        native_frames: dict[str, pd.DataFrame] = {}
        for native in required_native:
            if native == "1h":
                safe_start = max(
                    start_date, date.today() - timedelta(days=729)
                )
            elif native in INTRADAY_INTERVALS:
                safe_start = max(
                    start_date, date.today() - timedelta(days=59)
                )
            else:
                safe_start = start_date
            try:
                calls += 1
                raw = yf.download(
                    symbol,
                    start=safe_start.isoformat(),
                    end=(end_date + timedelta(days=1)).isoformat(),
                    interval={"1w": "1wk"}.get(native, native),
                    auto_adjust=False,
                    actions=True,
                    repair=True,
                    keepna=False,
                    prepost=False,
                    progress=False,
                    threads=False,
                    timeout=30,
                    multi_level_index=False,
                )
                frame = _normalize(raw, symbol=symbol, provider="YFINANCE", source_interval=native)
                if frame.empty:
                    results.append(_result("YFINANCE", symbol, native, native, "EMPTY_RESPONSE", 0))
                    continue
                native_frames[native] = frame
                results.append(_store(layout, frame, provider="YFINANCE", symbol=symbol, interval=native, source_interval=native))
            except Exception as exc:  # pragma: no cover - network/provider boundary
                results.append(_result("YFINANCE", symbol, native, native, type(exc).__name__, 0))
        results.extend(_derive_requested(layout, native_frames, symbol, "YFINANCE", intervals))
    return results, calls


def _collect_eodhd(
    layout: MultiTimeframeLayout,
    symbols: list[str],
    intervals: list[str],
    *,
    start_date: date,
    end_date: date,
) -> tuple[list[dict[str, Any]], int]:
    env = _env_values(layout.project_root)
    key = _first_secret(env, "EODHD_API_KEY", "EOD_API_KEY", "EODHISTORICALDATA_API_KEY")
    if not key:
        return [{"provider": "EODHD", "status": "MISSING_PROVIDER_KEY", "rows_written": 0}], 0
    required_native: list[str] = []
    for interval in intervals:
        native = "1h" if interval in {"2h", "4h", "6h", "12h"} else "1d" if interval in {"1w", "1mo"} else interval
        if native not in required_native:
            required_native.append(native)
    results: list[dict[str, Any]] = []
    calls = 0
    for symbol in symbols:
        provider_symbol = symbol if "." in symbol else f"{symbol}.US"
        native_frames: dict[str, pd.DataFrame] = {}
        for native in required_native:
            params = {"api_token": key, "fmt": "json"}
            if native == "1d":
                path = f"eod/{urllib.parse.quote(provider_symbol)}"
                params.update({"period": "d", "from": start_date.isoformat(), "to": end_date.isoformat()})
            else:
                path = f"intraday/{urllib.parse.quote(provider_symbol)}"
                params.update(
                    {
                        "interval": native,
                        "from": str(int(datetime.combine(start_date, datetime.min.time(), UTC).timestamp())),
                        "to": str(int(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), UTC).timestamp())),
                    }
                )
            url = f"{EODHD_BASE}/{path}?{urllib.parse.urlencode(params)}"
            calls += 1
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "stocks-multitimeframe-research/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                raw = pd.DataFrame(payload if isinstance(payload, list) else payload.get("data", []))
                frame = _normalize(raw, symbol=symbol, provider="EODHD", source_interval=native)
                if frame.empty:
                    results.append(_result("EODHD", symbol, native, native, "EMPTY_RESPONSE", 0))
                    continue
                native_frames[native] = frame
                results.append(_store(layout, frame, provider="EODHD", symbol=symbol, interval=native, source_interval=native))
            except urllib.error.HTTPError as exc:  # pragma: no cover - network/provider boundary
                status = "PLAN_NOT_ENTITLED" if exc.code in {401, 402, 403} else "RATE_LIMITED" if exc.code == 429 else "HTTP_ERROR"
                results.append(_result("EODHD", symbol, native, native, status, 0, http_status=exc.code))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:  # pragma: no cover
                results.append(_result("EODHD", symbol, native, native, type(exc).__name__, 0))
        results.extend(_derive_requested(layout, native_frames, symbol, "EODHD", intervals))
    return results, calls


def _import_ibkr_phase4(layout: MultiTimeframeLayout, symbols: list[str], intervals: list[str]) -> list[dict[str, Any]]:
    if not any(interval in {"1d", "1w", "1mo"} for interval in intervals):
        return []
    frames: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    for path in layout.project_root.joinpath("data", "bars").rglob("bars.parquet"):
        try:
            raw = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        if "symbol" not in raw:
            continue
        for symbol in symbols:
            selected = raw.loc[raw["symbol"].astype(str).str.upper().eq(symbol)]
            if not selected.empty:
                frames[symbol].append(selected)
    return _store_daily_sources(layout, frames, intervals, "IBKR_PHASE4_LOCAL")


def _import_stocks_yfinance(layout: MultiTimeframeLayout, symbols: list[str], intervals: list[str]) -> list[dict[str, Any]]:
    root = layout.project_root / "data" / "research" / "critical_trading" / "yfinance"
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or not any(interval in {"1d", "1w", "1mo"} for interval in intervals):
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    for item in manifest.get("instruments", []):
        symbol = str(item.get("symbol", "")).upper()
        path = Path(str(item.get("path", "")))
        if symbol in frames and path.is_file():
            frames[symbol].append(pd.read_parquet(path))
    return _store_daily_sources(layout, frames, intervals, "STOCKS_YFINANCE_LOCAL")


def _import_stocks_pit(layout: MultiTimeframeLayout, symbols: list[str], intervals: list[str]) -> list[dict[str, Any]]:
    path = layout.project_root / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet"
    if not path.is_file() or not any(interval in {"1d", "1w", "1mo"} for interval in intervals):
        return []
    table = ds.dataset(path, format="parquet").to_table(filter=ds.field("ticker").isin(symbols))
    raw = table.to_pandas()
    frames = {symbol: [raw.loc[raw["ticker"].astype(str).str.upper().eq(symbol)]] for symbol in symbols}
    return _store_daily_sources(layout, frames, intervals, "STOCKS_PIT_EODHD_LOCAL")


def _import_datascraper(
    layout: MultiTimeframeLayout,
    root: Path,
    symbols: list[str],
    intervals: list[str],
) -> list[dict[str, Any]]:
    if not any(interval in {"1d", "1w", "1mo"} for interval in intervals):
        return []
    results: list[dict[str, Any]] = []
    adapter = DatascraperAdapter(root)
    frames: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    for validation in adapter.manifests() if adapter.export_root.is_dir() else []:
        if not validation.valid or validation.manifest.get("dataset") != "prices":
            continue
        selected = [row for row in adapter.rows(validation) if _base_symbol(row.get("symbol")) in frames]
        if selected:
            raw = pd.DataFrame(selected)
            for symbol in symbols:
                part = raw.loc[raw["symbol"].map(_base_symbol).eq(symbol)]
                if not part.empty:
                    frames[symbol].append(part)
    results.extend(_store_daily_sources(layout, frames, intervals, "DATASCRAPER_EODHD_EXPORT"))

    yahoo = root / "data" / "yahoo_finance_daily_bars" / "yahoo_daily_bars_combined.parquet"
    if yahoo.is_file():
        table = ds.dataset(yahoo, format="parquet").to_table(filter=ds.field("symbol").isin(symbols))
        raw = table.to_pandas()
        local = {symbol: [raw.loc[raw["symbol"].astype(str).str.upper().eq(symbol)]] for symbol in symbols}
        results.extend(_store_daily_sources(layout, local, intervals, "DATASCRAPER_YFINANCE_LOCAL"))

    silver = root / "output" / "apex_data_lake" / "silver" / "eodhd_ohlcv_normalized.csv"
    if silver.is_file():
        raw = pd.read_csv(silver)
        raw["_base_symbol"] = raw["symbol"].map(_base_symbol)
        local = {symbol: [raw.loc[raw["_base_symbol"].eq(symbol)]] for symbol in symbols}
        results.extend(_store_daily_sources(layout, local, intervals, "DATASCRAPER_EODHD_SILVER"))

    lake_root = root / "data" / "historical" / "eodhd"
    raw_path = lake_root / "eodhd_raw_ohlcv_v7.parquet"
    adjusted_path = lake_root / "eodhd_adjusted_ohlcv_v7.parquet"
    if raw_path.is_file():
        raw_table = ds.dataset(raw_path, format="parquet").to_table(filter=ds.field("symbol").isin([f"{symbol}.US" for symbol in symbols]))
        raw = raw_table.to_pandas().rename(columns={"raw_close": "close"})
        if adjusted_path.is_file():
            adjusted_table = ds.dataset(adjusted_path, format="parquet").to_table(
                columns=["symbol", "date", "adjusted_close"],
                filter=ds.field("symbol").isin([f"{symbol}.US" for symbol in symbols]),
            )
            adjusted = adjusted_table.to_pandas()
            raw = raw.merge(adjusted, on=["symbol", "date"], how="left")
        local = {symbol: [raw.loc[raw["symbol"].map(_base_symbol).eq(symbol)]] for symbol in symbols}
        results.extend(_store_daily_sources(layout, local, intervals, "DATASCRAPER_EODHD_HISTORICAL_LAKE"))

    stock_1999 = root / "output" / "apex_data_lake" / "silver" / "v4621_stock_1999_ohlcv.csv"
    if stock_1999.is_file():
        raw = pd.read_csv(stock_1999)
        local = {symbol: [raw.loc[raw["symbol"].astype(str).str.upper().eq(symbol)]] for symbol in symbols}
        results.extend(_store_daily_sources(layout, local, intervals, "DATASCRAPER_EODHD_STOCK_1999"))
    return results


def _store_daily_sources(
    layout: MultiTimeframeLayout,
    frames: dict[str, list[pd.DataFrame]],
    intervals: list[str],
    provider: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for symbol, parts in frames.items():
        if not parts:
            continue
        raw = pd.concat(parts, ignore_index=False)
        daily = _normalize(raw, symbol=symbol, provider=provider, source_interval="1d")
        if daily.empty:
            results.append(_result(provider, symbol, "1d", "1d", "NO_VALID_ROWS", 0))
            continue
        if "1d" in intervals:
            results.append(_store(layout, daily, provider=provider, symbol=symbol, interval="1d", source_interval="1d"))
        frames_by_interval = {"1d": daily}
        results.extend(_derive_requested(layout, frames_by_interval, symbol, provider, intervals))
    return results


def _derive_requested(
    layout: MultiTimeframeLayout,
    native_frames: dict[str, pd.DataFrame],
    symbol: str,
    provider: str,
    intervals: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    sources = {
        "2h": "1h",
        "4h": "1h",
        "6h": "1h",
        "12h": "1h",
        "1w": "1d",
        "1mo": "1d",
    }
    for target, source in sources.items():
        if target not in intervals or source not in native_frames:
            continue
        derived = aggregate_bars(native_frames[source], target_interval=target)
        results.append(_store(layout, derived, provider=provider, symbol=symbol, interval=target, source_interval=source))
    return results


def _normalize(raw: pd.DataFrame, *, symbol: str, provider: str, source_interval: str) -> pd.DataFrame:
    interval = canonical_interval(source_interval)
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [str(item[0]) for item in frame.columns]
    frame.columns = [_column_name(column) for column in frame.columns]
    timestamp_column = next(
        (column for column in ("timestamp_utc", "datetime", "timestamp", "date", "session_date", "index") if column in frame),
        None,
    )
    if timestamp_column is None:
        frame = frame.reset_index()
        frame.columns = [_column_name(column) for column in frame.columns]
        timestamp_column = frame.columns[0]
    values = frame[timestamp_column]
    if pd.api.types.is_numeric_dtype(values) and interval in INTRADAY_INTERVALS:
        timestamps = pd.to_datetime(values, unit="s", utc=True, errors="coerce")
    else:
        timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    output = pd.DataFrame({"timestamp_utc": timestamps})
    for column in ("open", "high", "low", "close", "volume", "dividends", "stock_splits"):
        source = frame[column] if column in frame else 0.0 if column in {"volume", "dividends", "stock_splits"} else math.nan
        output[column] = pd.to_numeric(source, errors="coerce") if not isinstance(source, float) else source
    adjusted = next((column for column in ("adjusted_close", "adj_close", "adjclose") if column in frame), None)
    output["adjusted_close"] = pd.to_numeric(frame[adjusted], errors="coerce") if adjusted else output["close"]
    output = output.dropna(subset=["timestamp_utc", "open", "high", "low", "close"])
    output = output.loc[
        output["high"].ge(output[["open", "close", "low"]].max(axis=1))
        & output["low"].le(output[["open", "close", "high"]].min(axis=1))
        & output[["open", "high", "low", "close"]].gt(0).all(axis=1)
    ].copy()
    output["volume"] = output["volume"].fillna(0.0).clip(lower=0.0)
    output["dividends"] = output["dividends"].fillna(0.0)
    output["stock_splits"] = output["stock_splits"].fillna(0.0)
    output["adjusted_close"] = output["adjusted_close"].fillna(output["close"])
    output["session_date"] = output["timestamp_utc"].dt.date.astype(str)
    output["symbol"] = symbol
    output["provider"] = provider
    output["interval"] = interval
    output["source_interval"] = interval
    output["derivation"] = "NATIVE_OR_SOURCE_CACHE"
    output["exchange_timezone"] = _timezone_name(values)
    output["adjustment_mode"] = _adjustment_mode(provider, frame)
    output["source_bar_count"] = 1
    output["is_partial"] = False
    output["received_at"] = _utc_now()
    return _finalize(output)


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    now = _utc_now()
    output["source_provider"] = output.get("source_provider", output["provider"])
    output["target_interval"] = output.get("target_interval", output["interval"])
    output["bar_origin"] = output.get(
        "bar_origin",
        output["derivation"].astype(str).map(
            lambda value: "DERIVED" if value.startswith("AGGREGATED_") else "NATIVE"
        ),
    )
    output["aggregation_rule"] = output.get("aggregation_rule", output["derivation"])
    output["partial_bucket"] = output.get("partial_bucket", output["is_partial"])
    output["session"] = output.get("session", "RTH")
    output["timezone"] = output.get("timezone", output["exchange_timezone"])
    output["quality_status"] = output.get("quality_status", "VALIDATED_OHLC")
    output["fetched_at"] = output.get("fetched_at", output.get("received_at", now))
    output["ingested_at"] = output.get("ingested_at", now)
    output["timestamp_utc"] = pd.to_datetime(output["timestamp_utc"], utc=True)
    output = output.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    output["row_hash"] = output.apply(
        lambda row: stable_hash(
            {
                "timestamp_utc": row["timestamp_utc"].isoformat(),
                "symbol": row["symbol"],
                "provider": row["provider"],
                "interval": row["interval"],
                "source_interval": row["source_interval"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        ),
        axis=1,
    )
    columns = [
        "timestamp_utc",
        "session_date",
        "symbol",
        "provider",
        "source_provider",
        "interval",
        "target_interval",
        "source_interval",
        "derivation",
        "bar_origin",
        "aggregation_rule",
        "partial_bucket",
        "session",
        "timezone",
        "quality_status",
        "fetched_at",
        "ingested_at",
        "exchange_timezone",
        "adjustment_mode",
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
        "dividends",
        "stock_splits",
        "source_bar_count",
        "is_partial",
        "received_at",
        "row_hash",
    ]
    return output[columns].reset_index(drop=True)


def _store(
    layout: MultiTimeframeLayout,
    frame: pd.DataFrame,
    *,
    provider: str,
    symbol: str,
    interval: str,
    source_interval: str,
) -> dict[str, Any]:
    if frame.empty:
        return _result(provider, symbol, interval, source_interval, "NO_VALID_ROWS", 0)
    if provider == "YFINANCE" and canonical_interval(interval) in INTRADAY_INTERVALS:
        frame = _closed_intraday_bars(
            frame,
            interval=interval,
            observed_at=datetime.now(UTC),
        )
        if frame.empty:
            return _result(
                provider,
                symbol,
                interval,
                source_interval,
                "NO_CLOSED_BARS",
                0,
            )
    path = layout.bars_path(provider=provider, symbol=symbol, interval=interval, source_interval=source_interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = frame
    if path.is_file():
        existing = pd.read_parquet(path)
        merged = _finalize(pd.concat([existing, frame], ignore_index=True))
    if provider == "YFINANCE" and canonical_interval(interval) in INTRADAY_INTERVALS:
        merged = _closed_intraday_bars(
            merged,
            interval=interval,
            observed_at=datetime.now(UTC),
        )
    temporary = path.with_suffix(".tmp.parquet")
    merged.to_parquet(temporary, index=False)
    temporary.replace(path)
    validation = _validate_frame(merged)
    return _result(
        provider,
        symbol,
        interval,
        source_interval,
        validation["status"],
        len(merged),
        content_hash=_file_hash(path),
        first_timestamp=validation.get("first_timestamp"),
        last_timestamp=validation.get("last_timestamp"),
    )


def _validate_frame(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "timestamp_utc",
        "session_date",
        "symbol",
        "provider",
        "interval",
        "source_interval",
        "open",
        "high",
        "low",
        "close",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        return {
            "status": "MISSING_FIELDS",
            "missing_fields": missing,
            "row_count": len(frame),
            "duplicate_rows": 0,
            "invalid_ohlc_rows": 0,
            "timezone_errors": 0,
        }
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    duplicate_rows = int(timestamps.duplicated().sum())
    timezone_errors = int(timestamps.isna().sum())
    numeric = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    invalid = (
        numeric.isna().any(axis=1)
        | numeric.le(0).any(axis=1)
        | numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "close", "high"]].min(axis=1))
    )
    status = "GO" if duplicate_rows == 0 and timezone_errors == 0 and not invalid.any() else "NO_GO"
    return {
        "status": status,
        "row_count": len(frame),
        "duplicate_rows": duplicate_rows,
        "invalid_ohlc_rows": int(invalid.sum()),
        "timezone_errors": timezone_errors,
        "provider": str(frame["provider"].iloc[0]) if len(frame) else None,
        "symbol": str(frame["symbol"].iloc[0]) if len(frame) else None,
        "interval": str(frame["interval"].iloc[0]) if len(frame) else None,
        "source_interval": str(frame["source_interval"].iloc[0]) if len(frame) else None,
        "exchange_timezone": (
            str(frame["exchange_timezone"].dropna().iloc[-1])
            if len(frame)
            and "exchange_timezone" in frame
            and not frame["exchange_timezone"].dropna().empty
            else None
        ),
        "first_timestamp": timestamps.min().isoformat() if len(frame) else None,
        "last_timestamp": timestamps.max().isoformat() if len(frame) else None,
    }


def _coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        symbol = row.get("symbol")
        interval = row.get("interval")
        provider = row.get("provider")
        if symbol and interval and provider and row.get("status") == "GO":
            grouped.setdefault((str(symbol), str(interval)), []).append(row)
    coverage: list[dict[str, Any]] = []
    for (symbol, interval), provider_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], _interval_order(item[0][1])),
    ):
        variants = [
            {
                "provider": str(row["provider"]),
                "last_timestamp": row.get("last_timestamp"),
                "freshness_status": row.get("freshness", {}).get("status"),
                "age_hours": row.get("freshness", {}).get("age_hours"),
                "freshness_basis": row.get("freshness", {}).get(
                    "freshness_basis"
                ),
                "calendar_code": row.get("freshness", {}).get(
                    "calendar_code"
                ),
                "market_open": row.get("freshness", {}).get("market_open"),
                "latest_completed_session": row.get("freshness", {}).get(
                    "latest_completed_session"
                ),
            }
            for row in provider_rows
        ]
        fresh = [
            row
            for row in variants
            if row["freshness_status"] == "FRESH_CLOSED_BAR"
        ]
        selected = max(
            fresh or variants,
            key=lambda row: pd.Timestamp(row["last_timestamp"]),
        )
        coverage.append(
            {
                "symbol": symbol,
                "interval": interval,
                "provider_count": len(variants),
                "providers": sorted(row["provider"] for row in variants),
                "provider_freshness": sorted(
                    variants,
                    key=lambda row: row["provider"],
                ),
                "fresh_provider_count": len(fresh),
                "current_data_status": (
                    "FRESH_CLOSED_BAR" if fresh else "STALE_BAR_BLOCKED"
                ),
                "selected_current_provider": selected["provider"],
                "selected_current_last_timestamp": selected[
                    "last_timestamp"
                ],
                "selection_policy": (
                    "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
                ),
            }
        )
    return coverage


def bar_freshness(
    last_timestamp: Any,
    *,
    interval: str,
    observed_at: datetime | None = None,
    exchange_timezone: str | None = None,
) -> dict[str, Any]:
    observed = pd.Timestamp(observed_at or datetime.now(UTC))
    observed = (
        observed.tz_localize("UTC")
        if observed.tzinfo is None
        else observed.tz_convert("UTC")
    )
    try:
        canonical = canonical_interval(interval)
        timestamp = pd.Timestamp(last_timestamp)
    except (TypeError, ValueError):
        return {
            "status": "TIMESTAMP_UNAVAILABLE_BLOCKED",
            "age_hours": None,
            "maximum_age_hours": FRESHNESS_MAX_AGE_HOURS.get(interval),
        }
    if pd.isna(timestamp):
        return {
            "status": "TIMESTAMP_UNAVAILABLE_BLOCKED",
            "age_hours": None,
            "maximum_age_hours": FRESHNESS_MAX_AGE_HOURS[canonical],
        }
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    age_hours = (observed - timestamp).total_seconds() / 3600.0
    if age_hours < 0:
        status = "FUTURE_BAR_BLOCKED"
        freshness_basis = "WALL_CLOCK"
        session_context: dict[str, Any] = {}
    else:
        session_context = _exchange_session_freshness(
            timestamp,
            observed,
            canonical=canonical,
            exchange_timezone=exchange_timezone,
        )
        maximum_age = FRESHNESS_MAX_AGE_HOURS[canonical]
        if session_context:
            status = str(session_context["status"])
            freshness_basis = str(session_context["freshness_basis"])
        else:
            weekend_bridge = (
                canonical in INTRADAY_INTERVALS
                and timestamp.weekday() == 4
                and observed.weekday() in {5, 6, 0}
                and age_hours <= 96.0
            )
            status = (
                "FRESH_CLOSED_BAR"
                if age_hours <= maximum_age or weekend_bridge
                else "STALE_BAR_BLOCKED"
            )
            freshness_basis = (
                "WEEKEND_BRIDGE_FALLBACK"
                if weekend_bridge
                else "WALL_CLOCK"
            )
    return {
        "status": status,
        "age_hours": round(max(0.0, age_hours), 4),
        "maximum_age_hours": FRESHNESS_MAX_AGE_HOURS[canonical],
        "last_timestamp": timestamp.isoformat(),
        "observed_at": observed.isoformat(),
        "exchange_timezone": exchange_timezone,
        "freshness_basis": freshness_basis,
        **session_context,
    }


def _exchange_session_freshness(
    timestamp: pd.Timestamp,
    observed: pd.Timestamp,
    *,
    canonical: str,
    exchange_timezone: str | None,
) -> dict[str, Any]:
    if canonical not in INTRADAY_INTERVALS or not exchange_timezone:
        return {}
    calendar_code = EXCHANGE_TIMEZONE_CALENDARS.get(exchange_timezone)
    if calendar_code is None:
        return {}
    try:
        calendar = _exchange_calendar(calendar_code)
        observed_minute = observed.floor("min")
        market_open = bool(
            calendar.is_open_on_minute(observed_minute, ignore_breaks=True)
        )
        if market_open:
            status = (
                "FRESH_CLOSED_BAR"
                if (observed - timestamp).total_seconds() / 3600.0
                <= FRESHNESS_MAX_AGE_HOURS[canonical]
                else "STALE_BAR_BLOCKED"
            )
            return {
                "status": status,
                "freshness_basis": "ACTIVE_EXCHANGE_SESSION_WALL_CLOCK",
                "calendar_code": calendar_code,
                "market_open": True,
                "latest_completed_session": None,
            }
        latest_session = calendar.minute_to_session(
            observed_minute,
            direction="previous",
        )
        if calendar.session_close(latest_session) > observed:
            latest_session = calendar.previous_session(latest_session)
        bar_session = calendar.minute_to_session(
            timestamp.floor("min"),
            direction="previous",
        )
    except (KeyError, TypeError, ValueError):
        return {}
    return {
        "status": (
            "FRESH_CLOSED_BAR"
            if bar_session == latest_session
            else "STALE_BAR_BLOCKED"
        ),
        "freshness_basis": "LATEST_COMPLETED_EXCHANGE_SESSION",
        "calendar_code": calendar_code,
        "market_open": False,
        "latest_completed_session": pd.Timestamp(latest_session)
        .date()
        .isoformat(),
        "bar_session": pd.Timestamp(bar_session).date().isoformat(),
    }


@lru_cache(maxsize=None)
def _exchange_calendar(calendar_code: str) -> Any:
    return xcals.get_calendar(calendar_code)


def _closed_intraday_bars(
    frame: pd.DataFrame,
    *,
    interval: str,
    observed_at: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    canonical = canonical_interval(interval)
    hours = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12}.get(
        canonical
    )
    if hours is None:
        return frame
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    observed = pd.Timestamp(observed_at)
    observed = (
        observed.tz_localize("UTC")
        if observed.tzinfo is None
        else observed.tz_convert("UTC")
    )
    closed = timestamps.add(pd.Timedelta(hours=hours)).le(observed)
    return frame.loc[closed].reset_index(drop=True)


def _providers(values: Iterable[str]) -> set[str]:
    requested = {str(value).strip().lower() for value in values if str(value).strip()}
    if "all" in requested:
        return {"local", "datascraper", "yfinance", "eodhd", "ibkr"}
    allowed = {"local", "datascraper", "yfinance", "eodhd", "ibkr"}
    invalid = requested - allowed
    if invalid:
        raise ValueError(f"unsupported providers: {sorted(invalid)}")
    return requested or {"local"}


def _source(provider: str, available: bool, intervals: list[str], source_type: str, *, file_count: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": provider,
        "status": "AVAILABLE" if available else "UNAVAILABLE",
        "available": available,
        "native_intervals": intervals,
        "source_type": source_type,
    }
    if file_count is not None:
        payload["file_count"] = file_count
    return payload


def _result(
    provider: str,
    symbol: str,
    interval: str,
    source_interval: str,
    status: str,
    rows: int,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "symbol": symbol,
        "interval": canonical_interval(interval),
        "source_interval": canonical_interval(source_interval),
        "status": status,
        "rows_written": int(rows),
        **extra,
    }


def _env_values(project_root: Path) -> dict[str, Any]:
    values = dotenv_values(project_root / ".env") if (project_root / ".env").is_file() else {}
    for name in ("EODHD_API_KEY", "EOD_API_KEY", "EODHISTORICALDATA_API_KEY"):
        if os.environ.get(name):
            values[name] = os.environ[name]
    return values


def _first_secret(values: dict[str, Any], *names: str) -> str | None:
    return next((str(values[name]) for name in names if values.get(name)), None)


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value))


def _timezone_name(values: pd.Series) -> str:
    try:
        timezone = values.dt.tz
        return str(timezone or "UTC")
    except (AttributeError, TypeError):
        return "UTC"


def _column_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _adjustment_mode(provider: str, source: pd.DataFrame) -> str:
    if provider == "STOCKS_YFINANCE_LOCAL":
        return "YFINANCE_AUTO_ADJUST_TRUE"
    if provider == "YFINANCE":
        return "YFINANCE_AUTO_ADJUST_FALSE_PROVIDER_DEFINED"
    if provider == "STOCKS_PIT_EODHD_LOCAL":
        if "price_basis" in source and not source["price_basis"].dropna().empty:
            return str(source["price_basis"].dropna().iloc[0])
        return "SPLIT_ADJUSTED_DIVIDENDS_EXCLUDED"
    if provider == "IBKR_PHASE4_LOCAL":
        return "IBKR_TRADES_PROVIDER_DEFINED"
    if provider == "DATASCRAPER_YFINANCE_LOCAL":
        return "YFINANCE_AUTO_ADJUST_FALSE_PROVIDER_DEFINED"
    if provider in {
        "EODHD",
        "DATASCRAPER_EODHD_EXPORT",
        "DATASCRAPER_EODHD_SILVER",
        "DATASCRAPER_EODHD_HISTORICAL_LAKE",
        "DATASCRAPER_EODHD_STOCK_1999",
    }:
        return "EODHD_UNADJUSTED_OHLC_WITH_OPTIONAL_ADJUSTED_CLOSE"
    return "PROVIDER_DEFINED_UNKNOWN"


def _base_symbol(value: Any) -> str:
    return str(value or "").upper().split(".", 1)[0]


def _safe_partition(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)


def _interval_order(value: str) -> int:
    try:
        return CANONICAL_INTERVALS.index(canonical_interval(value))
    except ValueError:
        return len(CANONICAL_INTERVALS)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_public(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                temporary.unlink(missing_ok=True)
                raise
            time.sleep(0.05)


def _privacy_audit(layout: MultiTimeframeLayout) -> dict[str, Any]:
    exclusions = ("SCOPE", "STATUS", "ENABLED", "HOST", "PORT", "CLIENT_ID", "AUTHORITY")
    markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "APP_ID")
    project_values = (
        dotenv_values(layout.project_root / ".env")
        if (layout.project_root / ".env").is_file()
        else {}
    )
    process_values = dict(os.environ)
    credentials = sorted(
        {
            str(value)
            for values in (project_values, process_values)
            for name, value in values.items()
            if value
            and len(str(value)) >= 8
            and any(marker in str(name).upper() for marker in markers)
            and not any(
                exclusion in str(name).upper() for exclusion in exclusions
            )
        }
    )
    files = [path for path in layout.output_root.glob("*") if path.is_file() and path.name != "privacy-audit.json"]
    leaks = 0
    for path in files:
        content = path.read_bytes()
        leaks += sum(secret.encode("utf-8") in content for secret in credentials)
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = (
        "place" + "Order",
        "cancel" + "Order",
        "reqGlobal" + "Cancel",
        "req" + "Ids",
        "reqAutoOpen" + "Orders",
        "exercise" + "Options",
    )
    forbidden_hits = sum(token in source for token in forbidden)
    return {
        "schema": "multi_timeframe_public_privacy_audit_v1",
        "status": "GO" if leaks == 0 and forbidden_hits == 0 else "NO_GO",
        "generated_at": _utc_now(),
        "public_files_checked": len(files),
        "credential_values_checked": len(credentials),
        "project_and_process_sources_checked_independently": True,
        "credential_value_leak_count": leaks,
        "forbidden_broker_method_hits": forbidden_hits,
        "raw_credentials_published": False,
        "broker_calls": 0,
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
    }


__all__ = [
    "CANONICAL_INTERVALS",
    "DEFAULT_SYMBOLS",
    "MultiTimeframeLayout",
    "aggregate_bars",
    "audit_multitimeframe_sources",
    "bar_freshness",
    "canonical_interval",
    "collect_multitimeframe_data",
    "multitimeframe_schema",
    "multitimeframe_status",
    "parse_intervals",
    "parse_symbols",
    "provider_inventory",
    "validate_multitimeframe_cache",
]
