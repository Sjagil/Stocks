from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable, Mapping

import httpx
import pandas as pd

from stocks.quant_platform.data import AssetClass, clean_market_data


def _received_at(value: Any | None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value if value is not None else datetime.now(UTC))
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _daily_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp


@dataclass(frozen=True)
class EodhdAdapter:
    source: str = "EODHD"

    def fetch(
        self,
        *,
        api_key: str,
        ticker: str,
        start: str,
        end: str,
        client: httpx.Client | None = None,
    ) -> list[dict[str, Any]]:
        return _get_json(
            f"https://eodhd.com/api/eod/{ticker}",
            params={"api_token": _secret(api_key), "fmt": "json", "period": "d", "order": "a", "from": start, "to": end},
            client=client,
        )

    def normalize(
        self,
        payload: Iterable[Mapping[str, Any]],
        *,
        symbol: str,
        asset_class: AssetClass,
        currency: str,
        received_at: Any | None = None,
    ) -> pd.DataFrame:
        received = _received_at(received_at)
        rows = []
        for item in payload:
            timestamp = _daily_timestamp(item.get("datetime", item.get("date")))
            close = item.get("adjusted_close", item.get("close"))
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": item.get("open", close),
                    "high": item.get("high", close),
                    "low": item.get("low", close),
                    "close": close,
                    "volume": item.get("volume"),
                    "asset_class": asset_class,
                    "currency": currency,
                    "source": self.source,
                    "available_at": max(received, timestamp),
                    "market_cap": None,
                }
            )
        return clean_market_data(rows)


@dataclass(frozen=True)
class FredAdapter:
    source: str = "FRED"

    def fetch(
        self,
        *,
        api_key: str,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        realtime_start: str | None = None,
        realtime_end: str | None = None,
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        params = {
            "api_key": _secret(api_key),
            "file_type": "json",
            "series_id": series_id,
            "observation_start": observation_start,
            "observation_end": observation_end,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
        }
        return _get_json(
            "https://api.stlouisfed.org/fred/series/observations",
            params={key: value for key, value in params.items() if value is not None},
            client=client,
        )

    def normalize(
        self,
        payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        series_id: str,
        received_at: Any | None = None,
    ) -> pd.DataFrame:
        observations = payload.get("observations", []) if isinstance(payload, Mapping) else payload
        received = _received_at(received_at)
        rows = []
        for item in observations:
            raw = item.get("value")
            if raw in {None, ".", ""}:
                continue
            timestamp = _daily_timestamp(item.get("date"))
            available = _daily_timestamp(item.get("realtime_start", received))
            available = max(available, timestamp)
            value = float(raw)
            rows.append(
                {
                    "symbol": series_id,
                    "timestamp": timestamp,
                    "open": value,
                    "high": value,
                    "low": value,
                    "close": value,
                    "volume": None,
                    "asset_class": AssetClass.MACRO,
                    "currency": "INDEX",
                    "source": self.source,
                    "available_at": min(received, available) if received >= available else available,
                    "market_cap": None,
                }
            )
        return clean_market_data(rows)


@dataclass(frozen=True)
class OpenExchangeRatesAdapter:
    source: str = "OPENEXCHANGERATES"

    def fetch_historical(
        self,
        *,
        app_id: str,
        date: str,
        base: str = "USD",
        symbols: Iterable[str] = (),
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        params = {"app_id": _secret(app_id), "base": base.upper()}
        requested = [str(symbol).upper() for symbol in symbols]
        if requested:
            params["symbols"] = ",".join(requested)
        return _get_json(
            f"https://openexchangerates.org/api/historical/{date}.json",
            params=params,
            client=client,
        )

    def normalize(
        self,
        payload: Mapping[str, Any],
        *,
        quote_currency: str,
        received_at: Any | None = None,
    ) -> pd.DataFrame:
        base = str(payload.get("base", "USD")).upper()
        quote = str(quote_currency).upper()
        rate = float(payload["rates"][quote])
        timestamp = pd.Timestamp(int(payload["timestamp"]), unit="s", tz="UTC")
        return clean_market_data(
            [
                {
                    "symbol": f"{base}{quote}",
                    "timestamp": timestamp,
                    "open": rate,
                    "high": rate,
                    "low": rate,
                    "close": rate,
                    "volume": None,
                    "asset_class": AssetClass.FX,
                    "currency": quote,
                    "source": self.source,
                    "available_at": max(_received_at(received_at), timestamp),
                    "market_cap": None,
                }
            ]
        )


@dataclass(frozen=True)
class BitvavoAdapter:
    source: str = "BITVAVO"

    def fetch(
        self,
        *,
        market: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int = 1_440,
        client: httpx.Client | None = None,
    ) -> list[list[Any]]:
        if not 1 <= limit <= 1_440:
            raise ValueError("Bitvavo candle limit must be between 1 and 1440")
        params = {"interval": interval, "limit": limit, "start": start, "end": end}
        return _get_json(
            f"https://api.bitvavo.com/v2/{market.upper()}/candles",
            params={key: value for key, value in params.items() if value is not None},
            client=client,
        )

    def normalize(
        self,
        payload: Iterable[Iterable[Any]],
        *,
        market: str,
        received_at: Any | None = None,
    ) -> pd.DataFrame:
        received = _received_at(received_at)
        quote = market.split("-")[-1].upper()
        rows = []
        for candle in payload:
            values = list(candle)
            if len(values) < 6:
                raise ValueError("Bitvavo candles require timestamp, OHLC and volume")
            timestamp = pd.Timestamp(int(values[0]), unit="ms", tz="UTC")
            rows.append(
                {
                    "symbol": market.replace("-", "").upper(),
                    "timestamp": timestamp,
                    "open": values[1],
                    "high": values[2],
                    "low": values[3],
                    "close": values[4],
                    "volume": values[5],
                    "asset_class": AssetClass.CRYPTO,
                    "currency": quote,
                    "source": self.source,
                    "available_at": max(received, timestamp),
                    "market_cap": None,
                }
            )
        return clean_market_data(rows)


@dataclass(frozen=True)
class CoinMarketCapAdapter:
    source: str = "COINMARKETCAP"

    def fetch_latest(
        self,
        *,
        api_key: str,
        symbols: Iterable[str],
        quote_currency: str = "USD",
        client: httpx.Client | None = None,
    ) -> dict[str, Any]:
        requested = [str(symbol).upper() for symbol in symbols]
        if not requested:
            raise ValueError("at least one CoinMarketCap symbol is required")
        return _get_json(
            "https://pro-api.coinmarketcap.com/v3/cryptocurrency/quotes/latest",
            params={"symbol": ",".join(requested), "convert": quote_currency.upper()},
            headers={"Accept": "application/json", "X-CMC_PRO_API_KEY": _secret(api_key)},
            client=client,
        )

    def normalize(
        self,
        payload: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        *,
        quote_currency: str = "USD",
        received_at: Any | None = None,
    ) -> pd.DataFrame:
        items = payload.get("data", []) if isinstance(payload, Mapping) else payload
        if isinstance(items, Mapping):
            items = list(items.values())
        received = _received_at(received_at)
        quote_currency = quote_currency.upper()
        rows = []
        for item in items:
            quote = item.get("quote", {}).get(quote_currency, {})
            price = quote.get("price")
            timestamp = _daily_timestamp(quote.get("last_updated", item.get("last_updated", received)))
            rows.append(
                {
                    "symbol": str(item.get("symbol", "")).upper(),
                    "timestamp": timestamp,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": quote.get("volume_24h"),
                    "asset_class": AssetClass.CRYPTO,
                    "currency": quote_currency,
                    "source": self.source,
                    "available_at": max(received, timestamp),
                    "market_cap": quote.get("market_cap"),
                }
            )
        return clean_market_data(rows)


def _secret(value: str) -> str:
    secret = str(value).strip()
    if not secret:
        raise ValueError("provider credential is required")
    return secret


def _get_json(
    url: str,
    *,
    params: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    client: httpx.Client | None = None,
) -> Any:
    if client is not None:
        response = client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    with httpx.Client(timeout=30.0, follow_redirects=False) as owned:
        response = owned.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
