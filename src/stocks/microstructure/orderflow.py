from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


def orderflow_schema() -> dict[str, Any]:
    return {
        "schema": "stocks_orderflow_context_v1",
        "status": "GO",
        "authority": "CONTEXT_ONLY",
        "execution_authority": "NONE",
        "observed_trade_inputs": [
            "timestamp",
            "price",
            "size",
            "bid",
            "ask",
        ],
        "observed_orderbook_inputs": [
            "bid_price",
            "bid_size",
            "ask_price",
            "ask_size",
            "level",
        ],
        "aggregations": ["1h", "2h", "4h"],
        "bar_proxy": {
            "status": "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW",
            "maximum_confidence": 0.35,
            "standalone_entry_authority": False,
        },
        "limitations": [
            "quote and tick rules estimate aggressor side",
            "visible orderbook liquidity can be cancelled",
            "OHLCV cannot prove aggressor volume, CVD or absorption",
        ],
    }


def classify_trades(trades: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "price", "size"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"trade records missing fields: {missing}")
    frame = trades.copy()
    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"], utc=True, errors="coerce"
    )
    for column in ("price", "size", "bid", "ask"):
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = (
        frame.dropna(subset=["timestamp", "price", "size"])
        .loc[lambda row: row["price"].gt(0) & row["size"].gt(0)]
        .sort_values("timestamp", kind="stable")
        .reset_index(drop=True)
    )
    if frame.empty:
        frame["aggressor_side"] = pd.Series(dtype="int8")
        frame["classification_source"] = pd.Series(dtype="object")
        return frame

    sides: list[int] = []
    sources: list[str] = []
    previous_price: float | None = None
    previous_side = 0
    for row in frame.itertuples(index=False):
        price = float(row.price)
        bid = float(row.bid) if math.isfinite(float(row.bid)) else None
        ask = float(row.ask) if math.isfinite(float(row.ask)) else None
        if ask is not None and price >= ask:
            side, source = 1, "QUOTE_RULE_ASK"
        elif bid is not None and price <= bid:
            side, source = -1, "QUOTE_RULE_BID"
        elif previous_price is not None and price > previous_price:
            side, source = 1, "TICK_RULE_UP"
        elif previous_price is not None and price < previous_price:
            side, source = -1, "TICK_RULE_DOWN"
        elif previous_side:
            side, source = previous_side, "TICK_RULE_CARRY"
        else:
            side, source = 0, "UNCLASSIFIED_MIDPOINT"
        sides.append(side)
        sources.append(source)
        previous_price = price
        if side:
            previous_side = side
    frame["aggressor_side"] = np.asarray(sides, dtype="int8")
    frame["classification_source"] = sources
    frame["signed_volume"] = frame["size"] * frame["aggressor_side"]
    return frame


def aggregate_trade_flow(
    trades: pd.DataFrame,
    *,
    interval: str,
    session_timezone: str = "UTC",
) -> pd.DataFrame:
    frequency = _frequency(interval)
    frame = (
        trades
        if "aggressor_side" in trades.columns
        else classify_trades(trades)
    ).copy()
    if frame.empty:
        return _empty_flow_frame()
    frame = frame.set_index("timestamp").sort_index()
    frame["buy_volume"] = np.where(
        frame["aggressor_side"].gt(0), frame["size"], 0.0
    )
    frame["sell_volume"] = np.where(
        frame["aggressor_side"].lt(0), frame["size"], 0.0
    )
    frame["classified_volume"] = frame["buy_volume"] + frame["sell_volume"]
    grouped = frame.resample(frequency, label="left", closed="left")
    result = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        trade_count=("price", "size"),
        total_volume=("size", "sum"),
        classified_volume=("classified_volume", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
    ).dropna(subset=["close"])
    result["delta"] = result["buy_volume"] - result["sell_volume"]
    denominator = result["buy_volume"] + result["sell_volume"]
    result["normalized_delta"] = result["delta"].div(
        denominator.replace(0, np.nan)
    )
    result["cvd"] = result["delta"].cumsum()
    result["classification_ratio"] = result["classified_volume"].div(
        result["total_volume"].replace(0, np.nan)
    )
    result["price_change"] = result["close"] - result["open"]
    result["flow_efficiency"] = result["price_change"].abs().div(
        result["classified_volume"].replace(0, np.nan)
    )
    result["price_impact"] = result["price_change"].div(
        result["delta"].replace(0, np.nan)
    )
    result["session_timezone"] = session_timezone
    result["data_class"] = "OBSERVED_TRADE_FLOW"
    result["confidence"] = (
        0.45
        + 0.45 * result["classification_ratio"].clip(0, 1).fillna(0)
        + 0.10 * result["trade_count"].div(100).clip(0, 1)
    ).clip(0, 1)
    result["standalone_entry_authority"] = False
    result["execution_authority"] = "NONE"
    return result.reset_index().rename(columns={"timestamp": "timestamp_utc"})


def orderbook_metrics(
    levels: pd.DataFrame | Iterable[dict[str, Any]],
    *,
    decay: float = 0.7,
) -> dict[str, Any]:
    frame = (
        levels.copy()
        if isinstance(levels, pd.DataFrame)
        else pd.DataFrame(list(levels))
    )
    required = {"bid_price", "bid_size", "ask_price", "ask_size"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return {
            "status": "ORDERBOOK_UNAVAILABLE",
            "missing_fields": missing,
            "obi": None,
            "microprice": None,
            "microprice_edge": None,
            "confidence": 0.0,
            "authority": "CONTEXT_ONLY",
            "execution_authority": "NONE",
        }
    if "level" not in frame:
        frame["level"] = np.arange(1, len(frame) + 1)
    for column in (*required, "level"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required)).loc[
        lambda row: row["bid_price"].gt(0)
        & row["ask_price"].gt(row["bid_price"])
        & row["bid_size"].ge(0)
        & row["ask_size"].ge(0)
    ]
    if frame.empty:
        return {
            "status": "ORDERBOOK_UNAVAILABLE",
            "missing_fields": [],
            "obi": None,
            "microprice": None,
            "microprice_edge": None,
            "confidence": 0.0,
            "authority": "CONTEXT_ONLY",
            "execution_authority": "NONE",
        }
    weights = np.exp(-float(decay) * (frame["level"].to_numpy() - 1.0))
    weighted_bid = float((weights * frame["bid_size"].to_numpy()).sum())
    weighted_ask = float((weights * frame["ask_size"].to_numpy()).sum())
    total = weighted_bid + weighted_ask
    obi = (weighted_bid - weighted_ask) / total if total > 0 else 0.0
    best = frame.sort_values("level").iloc[0]
    best_total = float(best["bid_size"] + best["ask_size"])
    microprice = (
        (
            float(best["ask_price"]) * float(best["bid_size"])
            + float(best["bid_price"]) * float(best["ask_size"])
        )
        / best_total
        if best_total > 0
        else None
    )
    midpoint = (float(best["bid_price"]) + float(best["ask_price"])) / 2.0
    spread = float(best["ask_price"]) - float(best["bid_price"])
    edge = (microprice - midpoint) / spread if microprice is not None else None
    return {
        "status": "ORDERBOOK_CONTEXT_AVAILABLE",
        "level_count": int(len(frame)),
        "obi": round(float(np.clip(obi, -1.0, 1.0)), 8),
        "microprice": round(microprice, 10) if microprice is not None else None,
        "midpoint": round(midpoint, 10),
        "spread": round(spread, 10),
        "microprice_edge": round(float(edge), 8) if edge is not None else None,
        "confidence": round(min(0.9, 0.5 + 0.08 * len(frame)), 8),
        "visible_liquidity_warning": True,
        "standalone_entry_authority": False,
        "authority": "CONTEXT_ONLY",
        "execution_authority": "NONE",
    }


def bar_flow_proxy(
    bars: pd.DataFrame,
    *,
    symbol: str,
    interval: str,
    source: str,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"bar records missing fields: {missing}")
    frame = bars.copy().sort_index()
    if "timestamp_utc" in frame.columns:
        timestamp = pd.to_datetime(
            frame.pop("timestamp_utc"), utc=True, errors="coerce"
        )
        frame.index = timestamp
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("bars need timestamp_utc or DatetimeIndex")
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.loc[~frame.index.isna()].copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required)).loc[
        lambda row: row["close"].gt(0)
        & row["high"].ge(row[["open", "close", "low"]].max(axis=1))
        & row["low"].le(row[["open", "close", "high"]].min(axis=1))
        & row["volume"].ge(0)
    ]
    if frame.empty:
        return pd.DataFrame()
    direction = np.sign(frame["close"] - frame["open"])
    carry = np.sign(frame["close"].diff()).fillna(0)
    direction = pd.Series(direction, index=frame.index).replace(0, np.nan)
    direction = direction.fillna(carry).fillna(0)
    frame["signed_volume_proxy"] = frame["volume"] * direction
    frame["delta_proxy"] = frame["signed_volume_proxy"]
    frame["normalized_delta_proxy"] = frame["signed_volume_proxy"].div(
        frame["volume"].replace(0, np.nan)
    )
    frame["cvd_proxy"] = frame["delta_proxy"].cumsum()
    median_volume = frame["volume"].rolling(20, min_periods=5).median()
    frame["rvol_proxy"] = frame["volume"].div(median_volume.replace(0, np.nan))
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["session_date"] = frame.index.date
    volume_cumulative = frame["volume"].groupby(frame["session_date"]).cumsum()
    frame["vwap_proxy"] = (
        (typical * frame["volume"])
        .groupby(frame["session_date"])
        .cumsum()
        .div(volume_cumulative.replace(0, np.nan))
    )
    range_value = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["close_location"] = (
        (2.0 * frame["close"] - frame["high"] - frame["low"])
        .div(range_value)
        .clip(-1, 1)
    )
    frame["flow_efficiency_proxy"] = (
        frame["close"]
        .pct_change()
        .abs()
        .div(frame["volume"].replace(0, np.nan))
    )
    delta_component = frame["normalized_delta_proxy"].fillna(0).clip(-1, 1)
    location_component = frame["close_location"].fillna(0).clip(-1, 1)
    rvol_component = (
        (frame["rvol_proxy"].fillna(1.0) - 1.0).div(2.0).clip(-1, 1)
    )
    frame["bar_flow_score"] = (
        0.45 * delta_component
        + 0.35 * location_component
        + 0.20 * rvol_component
    ).clip(-1, 1)
    frame["timestamp_utc"] = frame.index
    frame["symbol"] = str(symbol).upper()
    frame["interval"] = interval
    frame["source"] = source
    frame["data_class"] = "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW"
    frame["confidence"] = 0.30
    frame["observed_aggressor_volume"] = False
    frame["observed_orderbook"] = False
    frame["standalone_entry_authority"] = False
    frame["execution_authority"] = "NONE"
    columns = [
        "timestamp_utc",
        "session_date",
        "symbol",
        "interval",
        "source",
        "data_class",
        "delta_proxy",
        "normalized_delta_proxy",
        "cvd_proxy",
        "rvol_proxy",
        "vwap_proxy",
        "close_location",
        "flow_efficiency_proxy",
        "bar_flow_score",
        "confidence",
        "observed_aggressor_volume",
        "observed_orderbook",
        "standalone_entry_authority",
        "execution_authority",
    ]
    return frame[columns].reset_index(drop=True)


def _frequency(interval: str) -> str:
    mapping = {"1h": "1h", "2h": "2h", "4h": "4h"}
    try:
        return mapping[str(interval).lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported orderflow interval: {interval}") from exc


def _empty_flow_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "trade_count",
            "total_volume",
            "classified_volume",
            "buy_volume",
            "sell_volume",
            "delta",
            "normalized_delta",
            "cvd",
            "classification_ratio",
            "flow_efficiency",
            "price_impact",
            "data_class",
            "confidence",
            "standalone_entry_authority",
            "execution_authority",
        ]
    )


__all__ = [
    "aggregate_trade_flow",
    "bar_flow_proxy",
    "classify_trades",
    "orderbook_metrics",
    "orderflow_schema",
]
