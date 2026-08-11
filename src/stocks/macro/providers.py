from __future__ import annotations

import csv
import io
import json
import os
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.macro.config import MacroConfig
from stocks.macro.contracts import MacroObservation, stable_hash


def collect_configured_sources(
    project_root: Path,
    config: MacroConfig,
    *,
    start: date,
    end: date,
) -> tuple[list[MacroObservation], dict[str, Any]]:
    observations: list[MacroObservation] = []
    errors: list[dict[str, str]] = []
    provider_calls = {
        "FRED": 0,
        "FRED_VINTAGE": 0,
        "ECB": 0,
        "EUROSTAT": 0,
        "YAHOO": 0,
        "LOCAL_MARKET_CACHE": 0,
        "DATASCRAPER_PIT_FUNDAMENTALS": 0,
    }
    fred_key = _secret(project_root, "FRED_API_KEY")
    fred_specs = [
        spec
        for spec in config.series.values()
        if spec.primary_source == "FRED" and spec.provider_id
    ]
    if fred_key:
        for spec in fred_specs:
            try:
                use_vintages = (
                    spec.vintage_capable
                    and spec.revision_sensitive
                    and spec.frequency in {"monthly", "quarterly"}
                )
                rows = (
                    _fred_vintage_observations(
                        spec,
                        api_key=fred_key,
                        start=start,
                        end=end,
                    )
                    if use_vintages
                    else _fred_observations(
                        spec,
                        api_key=fred_key,
                        start=start,
                        end=end,
                    )
                )
                observations.extend(rows)
                provider_calls[
                    "FRED_VINTAGE" if use_vintages else "FRED"
                ] += 1
            except Exception as exc:
                errors.append(
                    {
                        "series_id": spec.canonical_id,
                        "provider": "FRED",
                        "error": type(exc).__name__,
                    }
                )
    else:
        errors.append(
            {
                "series_id": "*",
                "provider": "FRED",
                "error": "FRED_API_KEY_UNAVAILABLE",
            }
        )

    for provider_name, collector in (
        ("ECB", _ecb_observations),
        ("EUROSTAT", _eurostat_observations),
    ):
        specs = [
            spec
            for spec in config.series.values()
            if spec.primary_source == provider_name and spec.provider_id
        ]
        for spec in specs:
            try:
                observations.extend(collector(spec, start=start, end=end))
                provider_calls[provider_name] += 1
            except Exception as exc:
                errors.append(
                    {
                        "series_id": spec.canonical_id,
                        "provider": provider_name,
                        "error": type(exc).__name__,
                    }
                )

    yahoo_specs = [
        spec
        for spec in config.series.values()
        if spec.primary_source == "YAHOO" and spec.provider_id
    ]
    if yahoo_specs:
        try:
            market_rows = _yahoo_observations(yahoo_specs, start=start, end=end)
            observations.extend(market_rows)
            provider_calls["YAHOO"] = 1
        except Exception as exc:
            errors.append(
                {
                    "series_id": "*",
                    "provider": "YAHOO",
                    "error": type(exc).__name__,
                }
            )

    breadth_rows = _local_breadth_observations(project_root, config)
    observations.extend(breadth_rows)
    provider_calls["LOCAL_MARKET_CACHE"] = int(bool(breadth_rows))
    pit_rows = _datascraper_pit_fundamental_observations(
        project_root,
        config,
        start=start,
        end=end,
    )
    observations.extend(pit_rows)
    provider_calls["DATASCRAPER_PIT_FUNDAMENTALS"] = int(bool(pit_rows))
    observations.extend(_derived_observations(observations, config))
    inventory = {
        "provider_calls": provider_calls,
        "observation_count": len(observations),
        "errors": errors,
        "provider_conflicts": [],
        "secret_values_logged": False,
        "broker_calls": 0,
        "order_calls": 0,
    }
    return observations, inventory


def _fred_observations(
    spec: Any,
    *,
    api_key: str,
    start: date,
    end: date,
) -> list[MacroObservation]:
    params = urllib.parse.urlencode(
        {
            "series_id": spec.provider_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
        }
    )
    request = urllib.request.Request(
        f"https://api.stlouisfed.org/fred/series/observations?{params}",
        headers={"User-Agent": "Stocks-Macro-Research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result: list[MacroObservation] = []
    for item in payload.get("observations", []):
        if str(item.get("value")) in {".", "", "None"}:
            continue
        observation_date = date.fromisoformat(str(item["date"])[:10])
        publication_date = _period_end(
            observation_date,
            spec.frequency,
        ) + timedelta(days=spec.release_lag_days)
        publication_at = datetime.combine(
            publication_date,
            time(14, 0),
            tzinfo=UTC,
        )
        result.append(
            MacroObservation(
                series_id=spec.canonical_id,
                observation_date=observation_date,
                publication_at=publication_at,
                available_at=publication_at,
                revision_status="LATEST_VINTAGE_CONSERVATIVE_RELEASE_V1",
                source="FRED",
                provider="Federal Reserve Bank of St. Louis",
                original_value=float(item["value"]),
                transformed_value=None,
                frequency=spec.frequency,
                region=spec.region,
                vintage=(
                    None
                    if not item.get("realtime_start")
                    else str(item["realtime_start"])
                ),
                quality_status="CONSERVATIVE_PERIOD_END_RELEASE_LAG_V1",
                stale_status="NOT_EVALUATED_AT_INGEST",
                provider_payload_hash=stable_hash(item),
            )
        )
    return result


def _fred_vintage_observations(
    spec: Any,
    *,
    api_key: str,
    start: date,
    end: date,
) -> list[MacroObservation]:
    utc_today = datetime.now(UTC).date()
    realtime_end = min(end, utc_today)
    params = urllib.parse.urlencode(
        {
            "series_id": spec.provider_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "realtime_start": start.isoformat(),
            "realtime_end": realtime_end.isoformat(),
            "output_type": 3,
        }
    )
    request = urllib.request.Request(
        f"https://api.stlouisfed.org/fred/series/observations?{params}",
        headers={"User-Agent": "Stocks-Macro-Research/2.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    result: list[MacroObservation] = []
    prefix = f"{spec.provider_id}_"
    for item in payload.get("observations", []):
        observation_date = date.fromisoformat(str(item["date"])[:10])
        for key, raw_value in item.items():
            if not str(key).startswith(prefix):
                continue
            if str(raw_value) in {".", "", "None"}:
                continue
            vintage_text = str(key)[len(prefix) :]
            if len(vintage_text) != 8 or not vintage_text.isdigit():
                continue
            vintage_date = datetime.strptime(vintage_text, "%Y%m%d").date()
            publication_at = datetime.combine(
                vintage_date,
                time(14, 0),
                tzinfo=UTC,
            )
            result.append(
                MacroObservation(
                    series_id=spec.canonical_id,
                    observation_date=observation_date,
                    publication_at=publication_at,
                    available_at=publication_at,
                    revision_status="HISTORICAL_VINTAGE",
                    source="FRED_ALFRED",
                    provider="Federal Reserve Bank of St. Louis",
                    original_value=float(raw_value),
                    transformed_value=None,
                    frequency=spec.frequency,
                    region=spec.region,
                    vintage=vintage_date.isoformat(),
                    quality_status="FRED_ALFRED_HISTORICAL_VINTAGE",
                    stale_status="NOT_EVALUATED_AT_INGEST",
                    provider_payload_hash=stable_hash(
                        {
                            "series_id": spec.provider_id,
                            "date": item["date"],
                            "vintage": vintage_text,
                            "value": raw_value,
                        }
                    ),
                )
            )
    return result


def _ecb_observations(
    spec: Any,
    *,
    start: date,
    end: date,
) -> list[MacroObservation]:
    params = urllib.parse.urlencode(
        {
            "startPeriod": start.isoformat(),
            "endPeriod": end.isoformat(),
            "format": "csvdata",
        }
    )
    request = urllib.request.Request(
        f"https://data-api.ecb.europa.eu/service/data/{spec.provider_id}?{params}",
        headers={
            "Accept": "text/csv",
            "User-Agent": "Stocks-Macro-Research/2.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("utf-8-sig")
    result: list[MacroObservation] = []
    for item in csv.DictReader(io.StringIO(text)):
        raw_value = item.get("OBS_VALUE")
        period = item.get("TIME_PERIOD")
        if not period or raw_value is None or raw_value in {"", "NaN"}:
            continue
        observation_date = _provider_period_date(period)
        publication_date = _period_end(
            observation_date,
            spec.frequency,
        ) + timedelta(days=spec.release_lag_days)
        available_at = datetime.combine(
            publication_date,
            time(10, 0),
            tzinfo=UTC,
        )
        result.append(
            MacroObservation(
                series_id=spec.canonical_id,
                observation_date=observation_date,
                publication_at=available_at,
                available_at=available_at,
                revision_status="LATEST_OFFICIAL_NO_HISTORICAL_VINTAGES",
                source="ECB_DATA_API",
                provider="European Central Bank",
                original_value=float(raw_value),
                transformed_value=None,
                frequency=spec.frequency,
                region=spec.region,
                vintage=None,
                quality_status="ECB_OFFICIAL_LATEST_ONLY",
                stale_status="NOT_EVALUATED_AT_INGEST",
                provider_payload_hash=stable_hash(item),
            )
        )
    return result


def _eurostat_observations(
    spec: Any,
    *,
    start: date,
    end: date,
) -> list[MacroObservation]:
    dataset, query = str(spec.provider_id).split("?", 1)
    parameters = urllib.parse.parse_qsl(query, keep_blank_values=True)
    parameters.extend(
        [
            ("lang", "en"),
            ("sinceTimePeriod", start.isoformat()),
            ("untilTimePeriod", end.isoformat()),
        ]
    )
    request = urllib.request.Request(
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        f"{dataset}?{urllib.parse.urlencode(parameters)}",
        headers={
            "Accept": "application/json",
            "User-Agent": "Stocks-Macro-Research/2.0",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    time_dimension = payload["dimension"]["time"]["category"]["index"]
    values = payload.get("value", {})
    result: list[MacroObservation] = []
    for period, position in sorted(
        time_dimension.items(),
        key=lambda item: int(item[1]),
    ):
        raw_value = (
            values.get(str(position))
            if isinstance(values, dict)
            else values[int(position)]
        )
        if raw_value is None:
            continue
        observation_date = _provider_period_date(period)
        publication_date = _period_end(
            observation_date,
            spec.frequency,
        ) + timedelta(days=spec.release_lag_days)
        available_at = datetime.combine(
            publication_date,
            time(23, 0),
            tzinfo=UTC,
        )
        result.append(
            MacroObservation(
                series_id=spec.canonical_id,
                observation_date=observation_date,
                publication_at=available_at,
                available_at=available_at,
                revision_status="LATEST_OFFICIAL_NO_HISTORICAL_VINTAGES",
                source="EUROSTAT_DATA_API",
                provider="Eurostat",
                original_value=float(raw_value),
                transformed_value=None,
                frequency=spec.frequency,
                region=spec.region,
                vintage=None,
                quality_status="EUROSTAT_OFFICIAL_LATEST_ONLY",
                stale_status="NOT_EVALUATED_AT_INGEST",
                provider_payload_hash=stable_hash(
                    {
                        "dataset": dataset,
                        "period": period,
                        "value": raw_value,
                        "query": parameters,
                    }
                ),
            )
        )
    return result


def _yahoo_observations(
    specs: list[Any],
    *,
    start: date,
    end: date,
) -> list[MacroObservation]:
    import yfinance as yf

    symbols = [spec.provider_id for spec in specs]
    raw = yf.download(
        symbols,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    result: list[MacroObservation] = []
    collection_cutoff = datetime.now(UTC)
    for spec in specs:
        if len(symbols) == 1:
            frame = raw
        elif spec.provider_id in raw:
            frame = raw[spec.provider_id]
        else:
            continue
        if "Close" not in frame:
            continue
        column = "Close"
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        for timestamp, value in values.items():
            observation_date = pd.Timestamp(timestamp).date()
            available_at = datetime.combine(
                observation_date + timedelta(days=spec.release_lag_days),
                time(0, 0),
                tzinfo=UTC,
            )
            if available_at > collection_cutoff:
                continue
            result.append(
                MacroObservation(
                    series_id=spec.canonical_id,
                    observation_date=observation_date,
                    publication_at=available_at,
                    available_at=available_at,
                    revision_status="MARKET_CLOSE_RAW_V1",
                    source="YAHOO",
                    provider="Yahoo Finance",
                    original_value=float(value),
                    transformed_value=None,
                    frequency=spec.frequency,
                    region=spec.region,
                    vintage=observation_date.isoformat(),
                    quality_status="MARKET_CLOSE_RAW_UNADJUSTED_V1",
                    stale_status="NOT_EVALUATED_AT_INGEST",
                    provider_payload_hash=stable_hash(
                        {
                            "series": spec.canonical_id,
                            "date": observation_date.isoformat(),
                            "value": float(value),
                        }
                    ),
                )
            )
    return result


def _datascraper_pit_fundamental_observations(
    project_root: Path,
    config: MacroConfig,
    *,
    start: date,
    end: date,
) -> list[MacroObservation]:
    requested = {
        spec.canonical_id
        for spec in config.series.values()
        if spec.primary_source == "DATASCRAPER_PIT_FUNDAMENTALS"
    }
    if not requested:
        return []
    source_path = (
        project_root.parent
        / "datascraper"
        / "output"
        / "worldmonitor_data_plane_launcher"
        / "v4638_pit_fundamentals_feasibility_sample_ledger.parquet"
    )
    if not source_path.exists():
        return []
    frame = pd.read_parquet(source_path)
    frame = frame.loc[
        frame["metric"].isin({"net_income", "shares_outstanding"})
        & frame["fiscal_period"].eq("FY")
    ].copy()
    frame["period_end"] = pd.to_datetime(frame["period_end"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["normalized_value"] = pd.to_numeric(
        frame["normalized_value"],
        errors="coerce",
    )
    frame = frame.dropna(
        subset=["symbol", "period_end", "available_at", "normalized_value"]
    )
    frame = frame.loc[
        frame["available_at"].dt.date.between(start, end)
    ].sort_values(["available_at", "symbol", "metric", "period_end"])
    if frame.empty:
        return []
    price_cache = _load_symbol_closes(project_root, set(frame["symbol"]))
    result: list[MacroObservation] = []
    for available_at in sorted(frame["available_at"].unique()):
        available = frame.loc[frame["available_at"] <= available_at]
        latest = available.drop_duplicates(
            ["symbol", "metric", "period_end"],
            keep="last",
        )
        net_income = latest.loc[latest["metric"].eq("net_income")].copy()
        growth_values: list[float] = []
        earnings_yield_parts: list[tuple[float, float]] = []
        for symbol, symbol_rows in net_income.groupby("symbol"):
            symbol_rows = symbol_rows.sort_values("period_end").drop_duplicates(
                "period_end",
                keep="last",
            )
            if len(symbol_rows) >= 2:
                current = float(symbol_rows.iloc[-1]["normalized_value"])
                previous = float(symbol_rows.iloc[-2]["normalized_value"])
                if previous != 0:
                    growth_values.append(current / abs(previous) - 1.0)
            share_rows = latest.loc[
                latest["symbol"].eq(symbol)
                & latest["metric"].eq("shares_outstanding")
            ].sort_values("period_end")
            closes = price_cache.get(str(symbol))
            if share_rows.empty or closes is None or closes.empty:
                continue
            price_rows = closes.loc[closes.index <= pd.Timestamp(available_at)]
            if price_rows.empty:
                continue
            earnings = float(symbol_rows.iloc[-1]["normalized_value"])
            shares = float(share_rows.iloc[-1]["normalized_value"])
            market_cap = float(price_rows.iloc[-1]) * shares
            if market_cap > 0:
                earnings_yield_parts.append((earnings, market_cap))
        timestamp = pd.Timestamp(available_at).to_pydatetime()
        observation_date = timestamp.date()
        common = {
            "observation_date": observation_date,
            "publication_at": timestamp,
            "available_at": timestamp,
            "revision_status": "PIT_FILING_VINTAGE",
            "source": "DATASCRAPER_PIT_FUNDAMENTALS",
            "provider": "SEC-derived datascraper feasibility ledger",
            "transformed_value": None,
            "frequency": "monthly",
            "region": "US",
            "vintage": timestamp.date().isoformat(),
            "quality_status": "LIMITED_FIVE_SYMBOL_PIT_AGGREGATE",
            "stale_status": "NOT_EVALUATED_AT_INGEST",
        }
        if (
            "US_REPORTED_EARNINGS_GROWTH_BREADTH" in requested
            and len(growth_values) >= 2
        ):
            value = 100.0 * sum(item > 0 for item in growth_values) / len(
                growth_values
            )
            result.append(
                MacroObservation(
                    series_id="US_REPORTED_EARNINGS_GROWTH_BREADTH",
                    original_value=value,
                    provider_payload_hash=stable_hash(
                        {
                            "available_at": timestamp.isoformat(),
                            "growth_values": growth_values,
                        }
                    ),
                    **common,
                )
            )
        if (
            "US_AGGREGATE_EARNINGS_YIELD" in requested
            and len(earnings_yield_parts) >= 2
        ):
            total_earnings = sum(item[0] for item in earnings_yield_parts)
            total_market_cap = sum(item[1] for item in earnings_yield_parts)
            value = 100.0 * total_earnings / total_market_cap
            result.append(
                MacroObservation(
                    series_id="US_AGGREGATE_EARNINGS_YIELD",
                    original_value=value,
                    provider_payload_hash=stable_hash(
                        {
                            "available_at": timestamp.isoformat(),
                            "parts": earnings_yield_parts,
                        }
                    ),
                    **common,
                )
            )
    return result


def _load_symbol_closes(
    project_root: Path,
    symbols: set[str],
) -> dict[str, pd.Series]:
    root = project_root / "data" / "research" / "critical_trading" / "yfinance"
    result: dict[str, pd.Series] = {}
    for symbol in sorted(symbols):
        candidates = list(root.glob(f"{symbol}*.parquet"))
        if not candidates:
            continue
        frame = pd.read_parquet(candidates[0])
        columns = {str(column).lower(): column for column in frame.columns}
        if "close" not in columns:
            continue
        if "session_date" in columns:
            index = pd.to_datetime(frame[columns["session_date"]], utc=True)
        else:
            index = pd.to_datetime(frame.index, utc=True)
        result[symbol] = pd.Series(
            pd.to_numeric(frame[columns["close"]], errors="coerce").to_numpy(),
            index=index,
        ).dropna().sort_index()
    return result


def _local_breadth_observations(
    project_root: Path,
    config: MacroConfig,
) -> list[MacroObservation]:
    root = project_root / "data" / "research" / "critical_trading" / "yfinance"
    closes: dict[str, pd.Series] = {}
    for path in sorted(root.glob("*.parquet")):
        frame = pd.read_parquet(path)
        columns = {str(column).lower(): column for column in frame.columns}
        if "close" not in columns:
            continue
        if "session_date" in columns:
            index = pd.to_datetime(frame[columns["session_date"]], utc=True)
        else:
            index = pd.to_datetime(frame.index, utc=True)
        closes[path.stem.upper()] = pd.Series(
            pd.to_numeric(frame[columns["close"]], errors="coerce").to_numpy(),
            index=index,
        )
    if not closes:
        return []
    close = pd.DataFrame(closes).sort_index()
    above = close > close.rolling(200, min_periods=200).mean()
    breadth = above.sum(axis=1).div(close.notna().sum(axis=1).replace(0, pd.NA)) * 100
    spec = config.series["EQUITY_BREADTH_GLOBAL"]
    result = []
    for timestamp, value in breadth.dropna().items():
        observation_date = pd.Timestamp(timestamp).date()
        available_at = datetime.combine(
            observation_date + timedelta(days=1),
            time(0, 0),
            tzinfo=UTC,
        )
        result.append(
            MacroObservation(
                series_id=spec.canonical_id,
                observation_date=observation_date,
                publication_at=available_at,
                available_at=available_at,
                revision_status="LOCAL_CACHE_FINAL",
                source="LOCAL_MARKET_CACHE",
                provider="Stocks canonical yfinance cache",
                original_value=float(value),
                transformed_value=None,
                frequency="daily",
                region="GLOBAL",
                vintage=observation_date.isoformat(),
                quality_status="DERIVED_TRANSPARENT_BREADTH",
                stale_status="NOT_EVALUATED_AT_INGEST",
                provider_payload_hash=stable_hash(
                    {
                        "date": observation_date.isoformat(),
                        "value": float(value),
                        "symbols": sorted(closes),
                    }
                ),
            )
        )
    return result


def _derived_observations(
    observations: list[MacroObservation],
    config: MacroConfig,
) -> list[MacroObservation]:
    by_series: dict[str, dict[date, MacroObservation]] = {}
    for observation in observations:
        by_series.setdefault(observation.series_id, {})[
            observation.observation_date
        ] = observation
    result: list[MacroObservation] = []
    copper = by_series.get("COPPER", {})
    gold = by_series.get("GOLD", {})
    for observation_date in sorted(set(copper) & set(gold)):
        denominator = gold[observation_date].original_value
        if denominator <= 0:
            continue
        available_at = max(
            copper[observation_date].available_at,
            gold[observation_date].available_at,
        )
        value = copper[observation_date].original_value / denominator
        result.append(
            MacroObservation(
                series_id="COPPER_GOLD_RATIO",
                observation_date=observation_date,
                publication_at=available_at,
                available_at=available_at,
                revision_status="DERIVED_FROM_MARKET_CLOSES_RAW_V1",
                source="DERIVED_EXPLICIT",
                provider="Stocks Macro Engine",
                original_value=value,
                transformed_value=None,
                frequency="daily",
                region="GLOBAL",
                vintage=observation_date.isoformat(),
                quality_status="DERIVED_TRANSPARENT_RAW_CLOSE_V1",
                stale_status="NOT_EVALUATED_AT_INGEST",
                provider_payload_hash=stable_hash(
                    {
                        "copper": copper[observation_date].provider_payload_hash,
                        "gold": gold[observation_date].provider_payload_hash,
                    }
                ),
            )
        )
    return result


def _secret(project_root: Path, name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip()
    path = project_root / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        if key.strip() == name:
            return raw.strip().strip("\"'") or None
    return None


def _period_end(value: date, frequency: str) -> date:
    timestamp = pd.Timestamp(value)
    if frequency == "monthly":
        return timestamp.to_period("M").end_time.date()
    if frequency == "quarterly":
        return timestamp.to_period("Q").end_time.date()
    return value


def _provider_period_date(value: str) -> date:
    text = str(value).strip()
    if len(text) == 7 and text[4] == "-":
        return date.fromisoformat(f"{text}-01")
    if "Q" in text:
        year, quarter = text.replace("-", "").split("Q", 1)
        month = (int(quarter) - 1) * 3 + 1
        return date(int(year), month, 1)
    return date.fromisoformat(text[:10])
