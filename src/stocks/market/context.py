from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import pandas as pd

from stocks.data.multitimeframe import (
    MultiTimeframeLayout,
    bar_freshness,
    parse_symbols,
)
from stocks.microstructure.orderflow import bar_flow_proxy, orderflow_schema
from stocks.options.gex import (
    adapt_external_gex_snapshot,
    calculate_gex_snapshot,
    gex_schema,
)


DEFAULT_DATASCRAPER_ROOT = Path(r"C:\Users\alhar\Documents\datascraper")
DEFAULT_CONTEXT_SYMBOLS = (
    "AAPL",
    "NVDA",
    "SPY",
    "QQQ",
    "IWM",
    "GLD",
    "SLV",
    "CPER",
    "SCOP",
    "COPX",
    "PPLT",
    "URA",
    "URNM",
    "XLB",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)
FLOW_TIMEFRAME_WEIGHTS = {"1h": 0.50, "2h": 0.30, "4h": 0.20}


@dataclass(frozen=True)
class MarketContextLayout:
    project_root: Path

    @property
    def private_root(self) -> Path:
        return self.project_root / "data" / "market_context" / "private"

    @property
    def output_root(self) -> Path:
        return self.project_root / "output" / "market_context"

    @property
    def schema_json(self) -> Path:
        return self.output_root / "schema.json"

    @property
    def source_audit_json(self) -> Path:
        return self.output_root / "source-audit.json"

    @property
    def gex_json(self) -> Path:
        return self.output_root / "gex-context.json"

    @property
    def orderflow_parquet(self) -> Path:
        return self.output_root / "orderflow-context.parquet"

    @property
    def status_json(self) -> Path:
        return self.output_root / "status.json"


def market_context_schema(project_root: Path) -> dict[str, Any]:
    layout = MarketContextLayout(project_root)
    payload = {
        "schema": "stocks_market_context_v1",
        "status": "GO",
        "layers": {
            "gex": gex_schema(),
            "orderflow": orderflow_schema(),
        },
        "timeframe_roles": {
            "1w_1d": "macro fundamentals structure allocation",
            "1d_4h": "trend factor rotation swing selection",
            "4h_2h": "setup formation",
            "2h_1h": "entry confirmation and risk management",
            "ticks_quotes_orderbook": "execution and observed flow context",
        },
        "data_classes": {
            "OBSERVED_OPTION_CHAIN": "current context only unless PIT-certified",
            "ESTIMATED_DEALER_GEX": "proxy with explicit confidence",
            "OBSERVED_TRADE_FLOW": "quote/tick classified trades",
            "OBSERVED_ORDERBOOK": "visible depth, never standalone",
            "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW": (
                "low-confidence fallback, never standalone"
            ),
        },
        "authority": "CONTEXT_ONLY",
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "storage": {
            "private": str(layout.private_root),
            "public": str(layout.output_root),
        },
    }
    _write_json(layout.schema_json, payload)
    return payload


def audit_market_context_sources(
    project_root: Path,
    *,
    datascraper_root: Path = DEFAULT_DATASCRAPER_ROOT,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    layout = MarketContextLayout(project_root)
    now = pd.Timestamp(observed_at or datetime.now(UTC))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    external = Path(datascraper_root)
    sources = [
        _source_record(
            "STOCKS_MULTITIMEFRAME_BARS",
            project_root
            / "data"
            / "research"
            / "multitimeframe"
            / "private",
            role="BAR_FLOW_PROXY_INPUT",
            asset_scope="EQUITY_ETF_COMMODITY_ETF",
            source_class="BAR_DATA",
            observed_at=now,
        ),
        _source_record(
            "STOCKS_LEGACY_SPY_GEX",
            project_root / "output" / "gex" / "SPY" / "SPY_gex_summary.json",
            role="LEGACY_CONTEXT_AUDIT_ONLY",
            asset_scope="SPY",
            source_class="CURRENT_OPTION_CHAIN_DERIVATION",
            observed_at=now,
            warnings=[
                "CURRENT_CHAIN_NOT_PIT",
                "SOURCE_PROVENANCE_INCOMPLETE",
            ],
        ),
        _source_record(
            "DATASCRAPER_YFINANCE_EQUITY_OPTIONS",
            external
            / "output"
            / "worldmonitor_data_plane_launcher"
            / "v4638_yfinance_ibkr_options_fetch.json",
            role="CURRENT_GEX_CONTEXT_FALLBACK",
            asset_scope="EQUITY_ETF",
            source_class="CURRENT_OPTION_CHAIN_ANALYTICS_NOT_PIT",
            observed_at=now,
            warnings=["CURRENT_CHAIN_NOT_PIT", "LEGACY_GEX_UNIT"],
        ),
        _source_record(
            "DATASCRAPER_SECTOR_ETF_OPTIONS",
            external
            / "output"
            / "worldmonitor_data_plane_launcher"
            / "v4638_etf_greeks_enrichment.json",
            role="CURRENT_GEX_CONTEXT_FALLBACK",
            asset_scope="US_SECTOR_ETF",
            source_class="CURRENT_OPTION_CHAIN_ANALYTICS_NOT_PIT",
            observed_at=now,
            warnings=["CURRENT_CHAIN_NOT_PIT", "LEGACY_GEX_UNIT"],
        ),
        _source_record(
            "DATASCRAPER_DERIBIT_OPTIONS_SURFACE",
            external
            / "data"
            / "forward_information"
            / "options_surface_v1.jsonl",
            role="CRYPTO_DERIVATIVES_CONTEXT_ONLY",
            asset_scope="BTC_ETH_ONLY",
            source_class="OBSERVED_PUBLIC_OPTION_SUMMARY_NO_GREEKS",
            observed_at=now,
            warnings=["OUT_OF_SCOPE_FOR_EQUITY_ORDERFLOW"],
        ),
        _source_record(
            "DATASCRAPER_BITVAVO_TRADES",
            external
            / "data"
            / "microstructure"
            / "bitvavo_trade_events_v10.jsonl",
            role="CRYPTO_OBSERVED_TRADE_FLOW_ONLY",
            asset_scope="CRYPTO_ONLY",
            source_class="OBSERVED_TRADES",
            observed_at=now,
            warnings=["MUST_NOT_BE_MAPPED_TO_EQUITIES_OR_ETFS"],
        ),
        _source_record(
            "DATASCRAPER_BITVAVO_ORDERBOOK",
            external
            / "data"
            / "microstructure"
            / "bitvavo_book_events_v10.jsonl",
            role="CRYPTO_OBSERVED_ORDERBOOK_ONLY",
            asset_scope="CRYPTO_ONLY",
            source_class="OBSERVED_ORDERBOOK",
            observed_at=now,
            warnings=["MUST_NOT_BE_MAPPED_TO_EQUITIES_OR_ETFS"],
        ),
        _source_record(
            "STOCKS_EQUITY_TRADE_QUOTE_STORE",
            layout.private_root / "equity-trades.parquet",
            role="EQUITY_OBSERVED_ORDERFLOW",
            asset_scope="EQUITY_ETF",
            source_class="OBSERVED_TRADE_AND_QUOTE",
            observed_at=now,
        ),
        _source_record(
            "STOCKS_EQUITY_ORDERBOOK_STORE",
            layout.private_root / "equity-orderbook.parquet",
            role="EQUITY_OBSERVED_ORDERBOOK",
            asset_scope="EQUITY_ETF",
            source_class="OBSERVED_ORDERBOOK",
            observed_at=now,
        ),
    ]
    equity_observed_flow = any(
        row["available"]
        and row["role"]
        in {"EQUITY_OBSERVED_ORDERFLOW", "EQUITY_OBSERVED_ORDERBOOK"}
        for row in sources
    )
    current_equity_options = any(
        row["available"]
        and row["role"] == "CURRENT_GEX_CONTEXT_FALLBACK"
        for row in sources
    )
    report = {
        "schema": "stocks_market_context_source_audit_v1",
        "status": "GO",
        "observed_at": now.isoformat(),
        "sources": sources,
        "summary": {
            "source_count": len(sources),
            "available_source_count": sum(row["available"] for row in sources),
            "current_equity_option_context_available": current_equity_options,
            "historical_equity_option_chain_pit_available": False,
            "equity_observed_trade_flow_available": equity_observed_flow,
            "equity_observed_orderbook_available": any(
                row["available"]
                and row["role"] == "EQUITY_OBSERVED_ORDERBOOK"
                for row in sources
            ),
            "crypto_observed_microstructure_available": any(
                row["available"]
                and row["asset_scope"] == "CRYPTO_ONLY"
                for row in sources
            ),
            "bar_flow_proxy_available": sources[0]["available"],
        },
        "hard_truth": {
            "gex_can_be_backtested_historically": False,
            "equity_orderflow_can_be_claimed_observed": equity_observed_flow,
            "crypto_flow_can_be_reused_for_equities": False,
            "bar_flow_proxy_is_orderflow": False,
        },
        "authority": "CONTEXT_ONLY",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
    }
    report["content_hash"] = _hash(report)
    _write_json(layout.source_audit_json, report)
    return report


def build_market_context(
    project_root: Path,
    *,
    symbols: Iterable[str] = DEFAULT_CONTEXT_SYMBOLS,
    fetch_options: bool = True,
    max_expirations: int = 4,
    datascraper_root: Path = DEFAULT_DATASCRAPER_ROOT,
    observed_at: datetime | None = None,
) -> dict[str, Any]:
    requested_symbols = parse_symbols(symbols)
    if not 1 <= int(max_expirations) <= 12:
        raise ValueError("max_expirations must be between 1 and 12")
    layout = MarketContextLayout(project_root)
    layout.private_root.mkdir(parents=True, exist_ok=True)
    layout.output_root.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp(observed_at or datetime.now(UTC))
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    market_context_schema(project_root)
    source_audit = audit_market_context_sources(
        project_root,
        datascraper_root=datascraper_root,
        observed_at=now.to_pydatetime(),
    )

    gex_rows: list[dict[str, Any]] = []
    option_errors: list[dict[str, str]] = []
    if fetch_options:
        gex_rows, option_errors = _collect_yfinance_gex(
            layout,
            symbols=requested_symbols,
            max_expirations=int(max_expirations),
            observed_at=now,
        )
    fallback_symbols = set(requested_symbols).difference(
        str(row.get("symbol", "")).upper() for row in gex_rows
    )
    if fallback_symbols:
        external_rows = _external_gex_rows(
            Path(datascraper_root), observed_at=now
        )
        gex_rows.extend(
            row
            for row in external_rows
            if row.get("symbol") in fallback_symbols
        )
    gex_rows = sorted(gex_rows, key=lambda row: str(row.get("symbol", "")))

    flow_frames: list[pd.DataFrame] = []
    flow_errors: list[dict[str, str]] = []
    for symbol in requested_symbols:
        for interval in ("1h", "2h", "4h"):
            selected = _select_current_bars(
                project_root,
                symbol=symbol,
                interval=interval,
                observed_at=now,
            )
            if selected is None:
                flow_errors.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "reason": "NO_FRESH_QUALIFIED_BAR_SOURCE",
                    }
                )
                continue
            provider, bars = selected
            try:
                flow_frames.append(
                    bar_flow_proxy(
                        bars,
                        symbol=symbol,
                        interval=interval,
                        source=provider,
                    )
                )
            except (ValueError, TypeError) as exc:
                flow_errors.append(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
                )
    flow = (
        pd.concat(flow_frames, ignore_index=True)
        if flow_frames
        else pd.DataFrame()
    )
    if not flow.empty:
        flow.to_parquet(layout.orderflow_parquet, index=False)

    gex_payload = {
        "schema": "stocks_gex_context_collection_v1",
        "status": "GO" if gex_rows else "NO_CURRENT_GEX_CONTEXT",
        "observed_at": now.isoformat(),
        "requested_symbols": requested_symbols,
        "available_symbol_count": sum(
            row.get("status") == "AVAILABLE_CONTEXT_ONLY"
            for row in gex_rows
        ),
        "contexts": gex_rows,
        "errors": option_errors,
        "historical_pit_backtest_allowed": False,
        "authority": "CONTEXT_ONLY",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
    }
    gex_payload["content_hash"] = _hash(gex_payload)
    _write_json(layout.gex_json, gex_payload)

    latest_flow = (
        flow.sort_values("timestamp_utc")
        .groupby(["symbol", "interval"], as_index=False)
        .tail(1)
        if not flow.empty
        else flow
    )
    status = {
        "schema": "stocks_market_context_status_v1",
        "status": "GO",
        "context_readiness": (
            "CONTEXT_READY_DEGRADED_NO_OBSERVED_EQUITY_ORDERFLOW"
        ),
        "observed_at": now.isoformat(),
        "requested_symbol_count": len(requested_symbols),
        "gex": {
            "status": gex_payload["status"],
            "available_symbol_count": gex_payload["available_symbol_count"],
            "historical_pit_backtest_allowed": False,
            "dealer_position_observed": False,
        },
        "orderflow": {
            "status": "BAR_PROXY_ONLY_OBSERVED_EQUITY_FLOW_UNAVAILABLE",
            "proxy_row_count": int(len(flow)),
            "latest_proxy_count": int(len(latest_flow)),
            "observed_trade_flow_available": source_audit["summary"][
                "equity_observed_trade_flow_available"
            ],
            "observed_orderbook_available": source_audit["summary"][
                "equity_observed_orderbook_available"
            ],
            "bar_proxy_is_observed_orderflow": False,
            "flow_errors": flow_errors,
        },
        "source_audit_status": source_audit["status"],
        "source_audit_hash": source_audit["content_hash"],
        "authority": "CONTEXT_ONLY",
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "automatic_orders": 0,
        "broker_calls": 0,
        "market_data_calls": 0,
        "historical_data_calls": 0,
        "artifacts": {
            "schema": str(layout.schema_json),
            "source_audit": str(layout.source_audit_json),
            "gex_context": str(layout.gex_json),
            "orderflow_context": str(layout.orderflow_parquet),
            "status": str(layout.status_json),
        },
    }
    status["content_hash"] = _hash(status)
    _write_json(layout.status_json, status)
    return status


def market_context_status(project_root: Path) -> dict[str, Any]:
    layout = MarketContextLayout(project_root)
    if not layout.status_json.exists():
        return {
            "schema": "stocks_market_context_status_v1",
            "status": "NOT_BUILT",
            "authority": "CONTEXT_ONLY",
            "execution_authority": "NONE",
            "automatic_orders": 0,
            "broker_calls": 0,
        }
    payload = json.loads(layout.status_json.read_text(encoding="utf-8"))
    payload["artifact_integrity"] = {
        "schema_exists": layout.schema_json.exists(),
        "source_audit_exists": layout.source_audit_json.exists(),
        "gex_context_exists": layout.gex_json.exists(),
        "orderflow_context_exists": layout.orderflow_parquet.exists(),
    }
    payload["execution_authority"] = "NONE"
    payload["automatic_orders"] = 0
    payload["broker_calls"] = 0
    return payload


def load_market_context_map(project_root: Path) -> dict[str, dict[str, Any]]:
    """Load public, context-only GEX and bar-flow observations by symbol."""
    layout = MarketContextLayout(project_root)
    gex_payload = _read_json(layout.gex_json)
    gex_by_symbol = {
        str(row.get("symbol", "")).upper(): row
        for row in gex_payload.get("contexts", [])
        if isinstance(row, dict) and row.get("symbol")
    }
    flow = _read_parquet(layout.orderflow_parquet)
    latest_flow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not flow.empty and {"symbol", "interval", "timestamp_utc"}.issubset(
        flow.columns
    ):
        current = (
            flow.assign(
                timestamp_utc=pd.to_datetime(
                    flow["timestamp_utc"], utc=True, errors="coerce"
                )
            )
            .dropna(subset=["timestamp_utc"])
            .sort_values("timestamp_utc")
            .groupby(["symbol", "interval"], as_index=False)
            .tail(1)
        )
        for row in current.to_dict(orient="records"):
            latest_flow[str(row.get("symbol", "")).upper()].append(row)

    result: dict[str, dict[str, Any]] = {}
    for symbol in sorted(set(gex_by_symbol) | set(latest_flow)):
        flow_rows = latest_flow.get(symbol, [])
        flow_context = _summarize_flow_context(flow_rows)
        gex_context = _summarize_gex_context(
            gex_by_symbol.get(symbol),
            flow_raw_score=flow_context["raw_score"],
        )
        advisories = sorted(
            set(flow_context["advisories"] + gex_context["advisories"])
        )
        result[symbol] = {
            "schema": "stocks_symbol_market_context_v1",
            "status": (
                "CONTEXT_AVAILABLE"
                if flow_rows or symbol in gex_by_symbol
                else "NO_CONTEXT"
            ),
            "symbol": symbol,
            "orderflow": flow_context,
            "gex": gex_context,
            "ranking_components": {
                "orderflow_context": flow_context["ranking_score"],
                "gex_context": gex_context["ranking_score"],
            },
            "advisories": advisories,
            "standalone_entry_authority": False,
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
            "automatic_orders": 0,
        }
    return result


def _collect_yfinance_gex(
    layout: MarketContextLayout,
    *,
    symbols: list[str],
    max_expirations: int,
    observed_at: pd.Timestamp,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    try:
        import yfinance as yf
    except ImportError:
        return [], [{"symbol": "*", "reason": "YFINANCE_NOT_INSTALLED"}]
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    snapshot_id = f"OPT-{observed_at.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8].upper()}"
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="5d", interval="1d", auto_adjust=False)
            if history.empty or history["Close"].dropna().empty:
                raise ValueError("SPOT_UNAVAILABLE")
            spot = float(history["Close"].dropna().iloc[-1])
            expirations = list(ticker.options or [])[:max_expirations]
            if not expirations:
                raise ValueError("NO_LISTED_OPTION_EXPIRATIONS")
            chain_parts: list[pd.DataFrame] = []
            for expiration in expirations:
                option_chain = ticker.option_chain(expiration)
                for option_type, frame in (
                    ("call", option_chain.calls),
                    ("put", option_chain.puts),
                ):
                    if frame.empty:
                        continue
                    part = frame.copy()
                    part["expiration"] = expiration
                    part["option_type"] = option_type
                    part["contract_multiplier"] = 100.0
                    chain_parts.append(part)
            if not chain_parts:
                raise ValueError("EMPTY_OPTION_CHAIN")
            chain = pd.concat(chain_parts, ignore_index=True)
            symbol_root = (
                layout.private_root
                / "options"
                / f"snapshot_id={snapshot_id}"
                / f"symbol={symbol}"
            )
            symbol_root.mkdir(parents=True, exist_ok=True)
            chain.to_parquet(symbol_root / "chain.parquet", index=False)
            summary, profile, scenario = calculate_gex_snapshot(
                chain,
                symbol=symbol,
                spot=spot,
                as_of=observed_at,
                observed_at=observed_at,
                source="YFINANCE_PUBLIC_CURRENT_CHAIN",
                source_mode="CURRENT_CHAIN_NOT_PIT",
            )
            profile.to_parquet(symbol_root / "profile.parquet", index=False)
            scenario.to_parquet(symbol_root / "scenario.parquet", index=False)
            summary["snapshot_id"] = snapshot_id
            summary["private_snapshot_hash"] = _hash(
                {"snapshot_id": snapshot_id, "symbol": symbol}
            )
            rows.append(summary)
        except Exception as exc:  # provider errors are isolated per symbol
            errors.append(
                {
                    "symbol": symbol,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    manifest = {
        "schema": "stocks_option_chain_private_manifest_v1",
        "snapshot_id": snapshot_id,
        "observed_at": observed_at.isoformat(),
        "requested_symbols": symbols,
        "successful_symbols": [row["symbol"] for row in rows],
        "errors": errors,
        "source": "YFINANCE_PUBLIC_CURRENT_CHAIN",
        "historical_pit": False,
        "execution_authority": "NONE",
    }
    manifest["content_hash"] = _hash(manifest)
    manifest_path = layout.private_root / "options" / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, default=str) + "\n")
    return rows, errors


def _external_gex_rows(
    datascraper_root: Path, *, observed_at: pd.Timestamp
) -> list[dict[str, Any]]:
    paths = [
        datascraper_root
        / "output"
        / "worldmonitor_data_plane_launcher"
        / "v4638_yfinance_ibkr_options_fetch.json",
        datascraper_root
        / "output"
        / "worldmonitor_data_plane_launcher"
        / "v4638_etf_greeks_enrichment.json",
    ]
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        computed = payload.get("computed_greeks_snapshot")
        if (
            not isinstance(payload.get("equity_etf_underlyings"), dict)
            and isinstance(computed, dict)
            and isinstance(computed.get("sector_etf_underlyings"), dict)
        ):
            payload = {
                "generated_at": payload.get("generated_at"),
                "equity_etf_underlyings": computed[
                    "sector_etf_underlyings"
                ],
            }
        for row in adapt_external_gex_snapshot(
            payload, observed_at=observed_at
        ):
            rows[str(row["symbol"])] = row
    return list(rows.values())


def _select_current_bars(
    project_root: Path,
    *,
    symbol: str,
    interval: str,
    observed_at: pd.Timestamp,
) -> tuple[str, pd.DataFrame] | None:
    layout = MultiTimeframeLayout(project_root)
    candidates: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
    for provider in ("YFINANCE", "EODHD"):
        path = layout.bars_path(
            provider=provider,
            symbol=symbol,
            interval=interval,
            source_interval="1h",
        )
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        if frame.empty:
            continue
        timestamp = pd.to_datetime(frame["timestamp_utc"], utc=True).max()
        freshness = bar_freshness(
            timestamp,
            interval=interval,
            observed_at=observed_at.to_pydatetime(),
        )
        if freshness["status"] != "FRESH_CLOSED_BAR":
            continue
        candidates.append((timestamp, provider, frame))
    if not candidates:
        return None
    _, provider, frame = max(candidates, key=lambda item: item[0])
    return provider, frame


def _source_record(
    source_id: str,
    path: Path,
    *,
    role: str,
    asset_scope: str,
    source_class: str,
    observed_at: pd.Timestamp,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    exists = path.exists()
    modified_at: str | None = None
    age_hours: float | None = None
    bytes_value = 0
    file_count = 0
    if exists:
        if path.is_dir():
            file_stats = []
            for child in path.rglob("*"):
                try:
                    if child.is_file() and not child.is_symlink():
                        file_stats.append(child.stat())
                except OSError:
                    continue
            stat = (
                max(file_stats, key=lambda value: value.st_mtime)
                if file_stats
                else path.stat()
            )
            file_count = len(file_stats)
            bytes_value = sum(value.st_size for value in file_stats)
        else:
            stat = path.stat()
            file_count = 1
            bytes_value = int(stat.st_size)
        modified = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC")
        modified_at = modified.isoformat()
        age_hours = max(
            0.0, (observed_at - modified).total_seconds() / 3600.0
        )
    return {
        "source_id": source_id,
        "path": str(path),
        "available": exists,
        "role": role,
        "asset_scope": asset_scope,
        "source_class": source_class,
        "modified_at": modified_at,
        "age_hours": round(age_hours, 6) if age_hours is not None else None,
        "bytes": bytes_value,
        "file_count": file_count,
        "warnings": warnings or [],
        "read_only": True,
    }


def _summarize_flow_context(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_interval: dict[str, dict[str, Any]] = {}
    weighted_score = 0.0
    used_weight = 0.0
    weighted_confidence = 0.0
    observed_flow = False
    observed_book = False
    for row in rows:
        interval = str(row.get("interval", ""))
        weight = FLOW_TIMEFRAME_WEIGHTS.get(interval, 0.0)
        raw_score = _clamp(_number(row.get("bar_flow_score")), -1.0, 1.0)
        confidence = _clamp(_number(row.get("confidence")), 0.0, 1.0)
        if weight <= 0:
            continue
        weighted_score += raw_score * weight
        weighted_confidence += confidence * weight
        used_weight += weight
        observed_flow = observed_flow or bool(
            row.get("observed_aggressor_volume", False)
        )
        observed_book = observed_book or bool(row.get("observed_orderbook", False))
        timestamp = row.get("timestamp_utc")
        by_interval[interval] = {
            "timestamp_utc": (
                timestamp.isoformat()
                if isinstance(timestamp, pd.Timestamp)
                else str(timestamp)
            ),
            "source": row.get("source"),
            "data_class": row.get("data_class"),
            "raw_score": round(raw_score, 6),
            "confidence": round(confidence, 6),
            "normalized_delta_proxy": _optional_number(
                row.get("normalized_delta_proxy")
            ),
            "rvol_proxy": _optional_number(row.get("rvol_proxy")),
            "close_location": _optional_number(row.get("close_location")),
            "flow_efficiency_proxy": _optional_number(
                row.get("flow_efficiency_proxy")
            ),
        }
    raw = weighted_score / used_weight if used_weight else 0.0
    confidence = weighted_confidence / used_weight if used_weight else 0.0
    ranking_score = _clamp(0.5 + 0.5 * raw * confidence, 0.0, 1.0)
    advisories = []
    if rows and not observed_flow:
        advisories.append("BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW")
    if not rows:
        advisories.append("EQUITY_ORDERFLOW_CONTEXT_UNAVAILABLE")
    if raw <= -0.35:
        advisories.append("FLOW_PROXY_HEADWIND")
    elif raw >= 0.35:
        advisories.append("FLOW_PROXY_SUPPORTIVE")
    return {
        "status": (
            "OBSERVED_ORDERFLOW_AVAILABLE"
            if observed_flow
            else "BAR_FLOW_PROXY_ONLY"
            if rows
            else "UNAVAILABLE"
        ),
        "raw_score": round(raw, 6),
        "ranking_score": round(ranking_score, 6),
        "confidence": round(confidence, 6),
        "observed_aggressor_volume": observed_flow,
        "observed_orderbook": observed_book,
        "timeframes": by_interval,
        "advisories": advisories,
        "standalone_entry_authority": False,
        "execution_authority": "NONE",
    }


def _summarize_gex_context(
    row: dict[str, Any] | None,
    *,
    flow_raw_score: float,
) -> dict[str, Any]:
    if not row:
        return {
            "status": "UNAVAILABLE",
            "ranking_score": 0.5,
            "confidence": 0.0,
            "advisories": ["GEX_CONTEXT_UNAVAILABLE_OR_NOT_APPLICABLE"],
            "dealer_position_observed": False,
            "standalone_entry_authority": False,
            "execution_authority": "NONE",
        }
    confidence = _clamp(_number(row.get("confidence")), 0.0, 0.70)
    spot = _number(row.get("spot"))
    call_wall = _number(row.get("call_wall"))
    regime = str(row.get("regime_proxy", "UNKNOWN"))
    concentration = _clamp(
        _number(row.get("gex_concentration_top3")), 0.0, 1.0
    )
    near_expiry = _clamp(
        _number(row.get("near_expiry_gex_concentration")), 0.0, 1.0
    )
    base_score = 0.5
    advisories = ["DEALER_POSITION_DIRECTION_IS_A_PROXY"]
    wall_distance_pct: float | None = None
    if spot > 0 and call_wall > 0:
        wall_distance_pct = (call_wall - spot) / spot
    overhead_wall = bool(
        wall_distance_pct is not None
        and 0.0 <= wall_distance_pct <= 0.02
        and concentration >= 0.35
    )
    if overhead_wall:
        base_score = 0.25
        advisories.append("CONCENTRATED_CALL_WALL_NEAR_SPOT")
    elif regime == "NEGATIVE_GEX_PROXY" and flow_raw_score >= 0.25:
        base_score = 0.65
        advisories.append("NEGATIVE_GEX_WITH_SUPPORTIVE_FLOW_PROXY")
    elif regime == "NEGATIVE_GEX_PROXY" and flow_raw_score <= -0.25:
        base_score = 0.4
        advisories.append("NEGATIVE_GEX_WITH_WEAK_FLOW_PROXY")
    elif regime == "POSITIVE_GEX_PROXY":
        base_score = 0.55
        advisories.append("POSITIVE_GEX_MEAN_REVERSION_CONTEXT")
    if near_expiry >= 0.6:
        base_score = 0.5 + (base_score - 0.5) * 0.7
        advisories.append("HIGH_NEAR_EXPIRY_GEX_CONCENTRATION")
    ranking_score = _clamp(
        0.5 + (base_score - 0.5) * confidence,
        0.0,
        1.0,
    )
    return {
        "status": row.get("status", "AVAILABLE_CONTEXT_ONLY"),
        "source": row.get("source"),
        "source_mode": row.get("source_mode"),
        "as_of": row.get("as_of"),
        "snapshot_id": row.get("snapshot_id"),
        "regime_proxy": regime,
        "net_gex_1pct": _optional_number(row.get("net_gex_1pct")),
        "net_gex_legacy": _optional_number(row.get("net_gex_legacy")),
        "call_wall": _optional_number(row.get("call_wall")),
        "put_wall": _optional_number(row.get("put_wall")),
        "gamma_flip": _optional_number(row.get("gamma_flip")),
        "spot": _optional_number(row.get("spot")),
        "call_wall_distance_pct": (
            round(wall_distance_pct, 6)
            if wall_distance_pct is not None
            else None
        ),
        "gex_concentration_top3": round(concentration, 6),
        "near_expiry_gex_concentration": round(near_expiry, 6),
        "ranking_score": round(ranking_score, 6),
        "confidence": round(confidence, 6),
        "dealer_position_observed": False,
        "historical_pit_backtest_allowed": False,
        "advisories": sorted(set(advisories)),
        "standalone_entry_authority": False,
        "execution_authority": "NONE",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame()


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _optional_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return round(result, 8) if math.isfinite(result) else None


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest().upper()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "DEFAULT_CONTEXT_SYMBOLS",
    "MarketContextLayout",
    "audit_market_context_sources",
    "build_market_context",
    "load_market_context_map",
    "market_context_schema",
    "market_context_status",
]
