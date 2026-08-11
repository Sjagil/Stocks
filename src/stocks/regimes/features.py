from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


DAILY_MACRO_SERIES = (
    "USD_INDEX",
    "US_HIGH_YIELD_SPREAD",
    "US_YIELD_CURVE_10Y2Y",
    "US_FINANCIAL_CONDITIONS",
    "VIX",
)
REQUIRED_DAILY_FEATURES = (
    "world_index_ret",
    "realized_vol_20",
    "realized_vol_60",
    "downside_vol_20",
    "equity_bond_corr_20",
    "bond_return",
    "commodity_return",
)


def engineer_daily_cross_asset_features(
    raw: pd.DataFrame,
    *,
    periods_per_year: float = 252.0,
) -> pd.DataFrame:
    if periods_per_year <= 0:
        raise ValueError("PERIODS_PER_YEAR_MUST_BE_POSITIVE")
    required = {
        "world_index",
        "bond_index",
        "commodity_index",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"MISSING_DAILY_HMM_INPUTS:{','.join(missing)}")
    prices = raw.sort_index().astype(float)
    world_return = np.log(prices["world_index"]).diff()
    bond_return = np.log(prices["bond_index"]).diff()
    commodity_return = np.log(prices["commodity_index"]).diff()
    negative = world_return.where(world_return.lt(0.0), 0.0)
    features = pd.DataFrame(index=prices.index)
    features["world_index_ret"] = world_return
    annualizer = np.sqrt(periods_per_year)
    features["realized_vol_20"] = world_return.rolling(20).std() * annualizer
    features["realized_vol_60"] = world_return.rolling(60).std() * annualizer
    features["downside_vol_20"] = negative.rolling(20).std() * annualizer
    features["equity_bond_corr_20"] = world_return.rolling(20).corr(
        bond_return
    )
    features["bond_return"] = bond_return
    features["commodity_return"] = commodity_return
    if "usd_index" in prices:
        features["usd_return"] = np.log(prices["usd_index"]).diff()
    if "credit_spread" in prices:
        features["credit_spread_chg"] = prices["credit_spread"].diff()
    if "yield_curve" in prices:
        features["yield_curve_velocity"] = prices["yield_curve"].diff(5)
    if "financial_conditions" in prices:
        features["financial_conditions_chg"] = prices[
            "financial_conditions"
        ].diff()
    if "vix" in prices:
        features["vix_log_change"] = np.log(
            prices["vix"].clip(lower=1e-8)
        ).diff()
    return features.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(REQUIRED_DAILY_FEATURES)
    )


def engineer_short_term_features(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"MISSING_SHORT_HMM_INPUTS:{','.join(missing)}")
    frame = raw.sort_index().astype(float)
    previous_close = frame["close"].shift(1)
    log_return = np.log(frame["close"]).diff()
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr20 = true_range.ewm(
        alpha=1 / 20,
        adjust=False,
        min_periods=20,
    ).mean()
    features = pd.DataFrame(index=frame.index)
    features["world_index_ret"] = log_return
    features["realized_vol_20"] = log_return.rolling(20).std()
    features["range_expansion"] = true_range.div(atr20.replace(0.0, np.nan))
    features["downside_range_ratio"] = (
        previous_close.sub(frame["low"]).clip(lower=0.0)
        .div(atr20.replace(0.0, np.nan))
    )
    features["volume_surprise"] = frame["volume"].div(
        frame["volume"].rolling(20).median().replace(0.0, np.nan)
    )
    features["overnight_gap"] = frame["open"].div(previous_close).sub(1.0)
    return features.replace([np.inf, -np.inf], np.nan).dropna()


def build_cross_asset_raw(
    frames: Mapping[str, pd.DataFrame],
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = {"SPY", "TLT", "DBC"}
    missing = sorted(required - set(frames))
    if missing:
        raise ValueError(f"MISSING_CROSS_ASSET_PROXIES:{','.join(missing)}")
    raw = pd.concat(
        {
            "world_index": frames["SPY"]["close"],
            "bond_index": frames["TLT"]["close"],
            "commodity_index": frames["DBC"]["close"],
        },
        axis=1,
        join="inner",
    ).sort_index()
    if macro is not None and not macro.empty:
        rename = {
            "USD_INDEX": "usd_index",
            "US_HIGH_YIELD_SPREAD": "credit_spread",
            "US_YIELD_CURVE_10Y2Y": "yield_curve",
            "US_FINANCIAL_CONDITIONS": "financial_conditions",
            "VIX": "vix",
        }
        aligned = macro.rename(columns=rename)
        raw = raw.join(aligned, how="left").ffill()
    return raw


def load_point_in_time_macro(
    project_root: Path,
    index: pd.DatetimeIndex,
    series_ids: tuple[str, ...] = DAILY_MACRO_SERIES,
) -> pd.DataFrame:
    database = (
        project_root / "data" / "macro" / "private" / "macro.sqlite3"
    )
    if not database.exists() or index.empty:
        return pd.DataFrame(index=index)
    placeholders = ",".join("?" for _ in series_ids)
    query = f"""
        SELECT series_id, available_at, payload_json
        FROM observations
        WHERE series_id IN ({placeholders})
        ORDER BY available_at, observation_date, created_at
    """
    with sqlite3.connect(database) as connection:
        rows = connection.execute(query, series_ids).fetchall()
    records = []
    for series_id, available_at, payload_json in rows:
        payload = json.loads(payload_json)
        value = payload.get("transformed_value")
        if value is None:
            value = payload.get("original_value")
        if value is None:
            continue
        records.append(
            {
                "series_id": str(series_id),
                "available_at": pd.Timestamp(available_at).tz_localize(None),
                "value": float(value),
            }
        )
    if not records:
        return pd.DataFrame(index=index)
    observations = pd.DataFrame(records)
    pieces = []
    for series_id, group in observations.groupby("series_id"):
        values = (
            group.drop_duplicates("available_at", keep="last")
            .set_index("available_at")["value"]
            .sort_index()
        )
        aligned = values.reindex(index, method="ffill")
        pieces.append(aligned.rename(series_id))
    return pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=index)


def standardize_train_oos(
    train: pd.DataFrame,
    oos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    mean = train.mean()
    scale = train.std(ddof=0).replace(0.0, 1.0)
    return (
        train.sub(mean).div(scale),
        oos.sub(mean).div(scale),
        {
            column: {"mean": float(mean[column]), "scale": float(scale[column])}
            for column in train.columns
        },
    )
