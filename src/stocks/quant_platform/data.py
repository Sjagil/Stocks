from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    COMMODITY = "commodity"
    FX = "fx"
    CRYPTO = "crypto"
    MACRO = "macro"


CANONICAL_MARKET_DATA_COLUMNS = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "asset_class",
    "currency",
    "source",
    "available_at",
    "market_cap",
)

PRICE_ASSET_CLASSES = {
    AssetClass.EQUITY.value,
    AssetClass.ETF.value,
    AssetClass.COMMODITY.value,
    AssetClass.FX.value,
    AssetClass.CRYPTO.value,
}


@dataclass(frozen=True)
class CanonicalMarketData:
    """Validated canonical observations with point-in-time availability."""

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", validate_market_data(self.frame))

    def for_symbol(self, symbol: str) -> pd.DataFrame:
        normalized = str(symbol).strip().upper()
        return self.frame.loc[self.frame["symbol"] == normalized].copy()

    def as_of(self, timestamp: Any) -> CanonicalMarketData:
        cutoff = pd.Timestamp(timestamp)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        return CanonicalMarketData(self.frame.loc[self.frame["available_at"] <= cutoff].copy())


def clean_market_data(
    records: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    drop_invalid: bool = False,
) -> pd.DataFrame:
    """Normalize records, remove deterministic duplicates and validate OHLC.

    Duplicate observations are resolved by keeping the last-received record.
    This makes revisions explicit through ``available_at`` and prevents future
    revisions from leaking into an as-of research view.
    """

    frame = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
    for column in CANONICAL_MARKET_DATA_COLUMNS:
        if column not in frame:
            frame[column] = np.nan if column in {"volume", "market_cap"} else pd.NA
    frame = frame.loc[:, list(CANONICAL_MARKET_DATA_COLUMNS)].copy()
    if frame.empty:
        return _typed_empty_frame()

    frame["symbol"] = frame["symbol"].astype("string").str.strip().str.upper()
    frame["asset_class"] = frame["asset_class"].map(_asset_class_value).astype("string")
    frame["currency"] = frame["currency"].astype("string").str.strip().str.upper()
    frame["source"] = frame["source"].astype("string").str.strip().str.upper()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "market_cap"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid = _invalid_mask(frame)
    if invalid.any() and not drop_invalid:
        rows = ", ".join(str(value) for value in frame.index[invalid].tolist()[:10])
        raise ValueError(f"invalid canonical market-data rows: {rows}")
    frame = frame.loc[~invalid].copy()
    frame = frame.sort_values(["symbol", "timestamp", "source", "available_at"], kind="stable")
    # Keep distinct vintages.  Revisions with a later availability timestamp
    # must remain queryable without overwriting the observation that was known
    # at an earlier as-of time.
    frame = frame.drop_duplicates(
        ["symbol", "timestamp", "source", "available_at"], keep="last"
    )
    frame = frame.reset_index(drop=True)
    return validate_market_data(frame)


def validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in CANONICAL_MARKET_DATA_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"missing canonical market-data columns: {', '.join(missing)}")
    candidate = frame.loc[:, list(CANONICAL_MARKET_DATA_COLUMNS)].copy()
    if candidate.empty:
        return _typed_empty_frame()
    normalized = clean_market_data(candidate, drop_invalid=False) if not _already_normalized(candidate) else candidate
    if _invalid_mask(normalized).any():
        raise ValueError("canonical market data failed validation")
    return normalized.reset_index(drop=True)


def resample_market_data(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    data = validate_market_data(frame)
    if data.empty:
        return data
    rows: list[pd.DataFrame] = []
    for (_, _, _, _, _), group in data.groupby(
        ["symbol", "asset_class", "currency", "source", "market_cap"],
        dropna=False,
        sort=True,
    ):
        ordered = (
            group.sort_values(["timestamp", "available_at"])
            .drop_duplicates("timestamp", keep="last")
            .set_index("timestamp")
        )
        aggregated = ordered.resample(frequency).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "symbol": "last",
                "asset_class": "last",
                "currency": "last",
                "source": "last",
                "available_at": "max",
                "market_cap": "last",
            }
        )
        aggregated = aggregated.dropna(subset=["close"]).reset_index()
        # A completed aggregate cannot be known before its period-end label,
        # even when every constituent bar arrived earlier.
        aggregated["available_at"] = aggregated[["available_at", "timestamp"]].max(axis=1)
        no_volume = ordered["volume"].isna().resample(frequency).agg(
            lambda values: bool(values.all())
        )
        if not aggregated.empty:
            aggregated.loc[no_volume.reindex(aggregated["timestamp"], fill_value=False).to_numpy(), "volume"] = np.nan
        rows.append(aggregated)
    return clean_market_data(pd.concat(rows, ignore_index=True) if rows else _typed_empty_frame())


def _invalid_mask(frame: pd.DataFrame) -> pd.Series:
    required_text = (
        frame["symbol"].isna()
        | frame["symbol"].astype("string").str.strip().eq("")
        | frame["currency"].isna()
        | frame["currency"].astype("string").str.strip().eq("")
        | frame["source"].isna()
        | frame["source"].astype("string").str.strip().eq("")
    )
    known_assets = frame["asset_class"].isin([item.value for item in AssetClass])
    times = frame["timestamp"].isna() | frame["available_at"].isna()
    causality = frame["available_at"] < frame["timestamp"]
    ohlc_missing = frame[["open", "high", "low", "close"]].isna().any(axis=1)
    finite = ~np.isfinite(frame[["open", "high", "low", "close"]].to_numpy(dtype=float)).all(axis=1)
    bad_high = frame["high"] < frame[["open", "low", "close"]].max(axis=1)
    bad_low = frame["low"] > frame[["open", "high", "close"]].min(axis=1)
    negative_volume = frame["volume"].notna() & (frame["volume"] < 0)
    negative_market_cap = frame["market_cap"].notna() & (frame["market_cap"] < 0)
    price_rows = frame["asset_class"].isin(PRICE_ASSET_CLASSES)
    nonpositive_prices = price_rows & (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
    return (
        required_text
        | ~known_assets
        | times
        | causality
        | ohlc_missing
        | pd.Series(finite, index=frame.index)
        | bad_high
        | bad_low
        | negative_volume
        | negative_market_cap
        | nonpositive_prices
    )


def _already_normalized(frame: pd.DataFrame) -> bool:
    return (
        isinstance(frame["timestamp"].dtype, pd.DatetimeTZDtype)
        and isinstance(frame["available_at"].dtype, pd.DatetimeTZDtype)
        and set(frame["asset_class"].dropna().astype(str)).issubset({item.value for item in AssetClass})
    )


def _asset_class_value(value: Any) -> str:
    if isinstance(value, AssetClass):
        return value.value
    text = str(value).strip().lower()
    aliases = {
        "stock": AssetClass.EQUITY.value,
        "stocks": AssetClass.EQUITY.value,
        "equities": AssetClass.EQUITY.value,
        "commodity_proxy": AssetClass.COMMODITY.value,
        "commodity_vehicle": AssetClass.COMMODITY.value,
        "forex": AssetClass.FX.value,
    }
    return aliases.get(text, text)


def _typed_empty_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(CANONICAL_MARKET_DATA_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    for column in ("open", "high", "low", "close", "volume", "market_cap"):
        frame[column] = frame[column].astype("float64")
    for column in ("symbol", "asset_class", "currency", "source"):
        frame[column] = frame[column].astype("string")
    return frame
