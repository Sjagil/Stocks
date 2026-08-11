from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stocks.macro.config import MacroConfig


def point_in_time_series(
    observations: Iterable[dict[str, Any]],
    *,
    series_id: str,
    as_of: datetime,
) -> pd.DataFrame:
    cutoff = _utc(as_of)
    rows = [
        row
        for row in observations
        if row["series_id"] == series_id
        and _utc(pd.Timestamp(row["available_at"]).to_pydatetime()) <= cutoff
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["observation_date"] = pd.to_datetime(
        frame["observation_date"], utc=True
    )
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["_quality_rank"] = frame["quality_status"].map(
        _quality_rank
    ).fillna(50)
    frame = frame.sort_values(
        [
            "observation_date",
            "available_at",
            "_quality_rank",
            "observation_id",
        ]
    )
    # For every economic period, use the last vintage that was available at
    # the decision timestamp. Future revisions are therefore excluded.
    return frame.drop_duplicates("observation_date", keep="last").set_index(
        "observation_date"
    )


def build_feature_snapshot(
    observations: Iterable[dict[str, Any]],
    config: MacroConfig,
    *,
    as_of: datetime,
) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    cutoff = _utc(as_of)
    rows = list(observations)
    rows_by_series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_series.setdefault(str(row["series_id"]), []).append(row)
    for series_id, spec in config.series.items():
        frame = point_in_time_series(
            rows_by_series.get(series_id, []),
            series_id=series_id,
            as_of=cutoff,
        )
        if not frame.empty and "quality_status" in frame:
            valid_release = [
                _valid_estimated_release(row, spec)
                for _, row in frame.iterrows()
            ]
            frame = frame.loc[valid_release]
        if frame.empty:
            features[series_id] = _unavailable(spec, "NO_PIT_OBSERVATIONS")
            continue
        values = pd.to_numeric(frame["original_value"], errors="coerce").dropna()
        transformed = transform_series(
            values,
            transformation=spec.transformation,
            frequency=spec.frequency,
        )
        valid = transformed.dropna()
        latest_row = frame.iloc[-1]
        age_days = (
            cutoff - pd.Timestamp(latest_row["available_at"]).to_pydatetime()
        ).total_seconds() / 86_400
        stale = age_days > spec.stale_days
        if len(values) < spec.minimum_history or valid.empty:
            status = "INSUFFICIENT_HISTORY"
            score = None
            transformed_value = None if valid.empty else float(valid.iloc[-1])
        else:
            status = "STALE" if stale else "VALID"
            transformed_value = float(valid.iloc[-1])
            score = normalized_feature_score(
                values,
                transformed,
                transformation=spec.transformation,
                direction=spec.direction,
            )
        features[series_id] = {
            "series_id": series_id,
            "category": spec.category,
            "region": spec.region,
            "observation_date": pd.Timestamp(frame.index[-1]).date().isoformat(),
            "available_at": pd.Timestamp(latest_row["available_at"]).isoformat(),
            "original_value": float(values.iloc[-1]),
            "transformed_value": transformed_value,
            "normalized_score": score,
            "status": status,
            "stale": stale,
            "age_days": age_days,
            "observation_count": int(len(values)),
            "minimum_history": spec.minimum_history,
            "transformation": spec.transformation,
            "source": latest_row["source"],
            "provider": latest_row["provider"],
            "revision_status": latest_row["revision_status"],
            "vintage": latest_row.get("vintage"),
            "vintage_history_status": (
                "AVAILABLE"
                if spec.vintage_capable
                and latest_row.get("revision_status") == "HISTORICAL_VINTAGE"
                else "VINTAGE_HISTORY_UNAVAILABLE"
            ),
            "quality_status": latest_row["quality_status"],
            "quality_confidence_multiplier": _quality_confidence_multiplier(
                str(latest_row["quality_status"])
            ),
        }
    return features


def transform_series(
    values: pd.Series,
    *,
    transformation: str,
    frequency: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    periods = {
        "daily": 252,
        "weekly": 52,
        "monthly": 12,
        "quarterly": 4,
    }[frequency]
    if transformation == "yoy":
        return numeric.pct_change(periods, fill_method=None) * 100.0
    if transformation == "change":
        return numeric.diff()
    if transformation == "roc":
        lookback = min(periods, 63 if frequency == "daily" else periods)
        return numeric.pct_change(lookback, fill_method=None) * 100.0
    if transformation == "level_50":
        return numeric - 50.0
    if transformation == "level_20":
        return numeric - 20.0
    if transformation == "level_zero":
        return numeric
    raise ValueError(f"UNSUPPORTED_MACRO_TRANSFORMATION:{transformation}")


def normalized_feature_score(
    values: pd.Series,
    transformed: pd.Series,
    *,
    transformation: str,
    direction: int,
) -> float | None:
    valid = transformed.dropna()
    if valid.empty:
        return None
    current = float(valid.iloc[-1])
    if transformation == "level_50":
        normalized = current / 10.0
    elif transformation == "level_20":
        normalized = current / 10.0
    else:
        history = valid.tail(120)
        median = float(history.median())
        mad = float((history - median).abs().median())
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 1e-12:
            standard_deviation = float(history.std(ddof=1))
            scale = standard_deviation if standard_deviation > 1e-12 else 1.0
        normalized = (current - median) / scale
    return float(np.clip(normalized * 33.333333 * direction, -100.0, 100.0))


def fx_normalize(value: float, fx_rate: float) -> float:
    if not np.isfinite(value) or not np.isfinite(fx_rate) or fx_rate <= 0:
        raise ValueError("INVALID_FX_NORMALIZATION_INPUT")
    return float(value * fx_rate)


def inflation_adjust(value: float, inflation_rate: float) -> float:
    if inflation_rate <= -1:
        raise ValueError("INVALID_INFLATION_RATE")
    return float((1.0 + value) / (1.0 + inflation_rate) - 1.0)


def _unavailable(spec: Any, reason: str) -> dict[str, Any]:
    return {
        "series_id": spec.canonical_id,
        "category": spec.category,
        "region": spec.region,
        "normalized_score": None,
        "status": "UNAVAILABLE",
        "stale": False,
        "reason": reason,
        "vintage_history_status": "VINTAGE_HISTORY_UNAVAILABLE",
    }


def _valid_estimated_release(row: pd.Series, spec: Any) -> bool:
    if str(row["quality_status"]) != "ESTIMATED_RELEASE_LAG":
        return True
    if spec.frequency not in {"monthly", "quarterly"}:
        return True
    timestamp = pd.Timestamp(row.name)
    naive = (
        timestamp.tz_localize(None)
        if timestamp.tzinfo is not None
        else timestamp
    )
    period = naive.to_period("M" if spec.frequency == "monthly" else "Q")
    floor = pd.Timestamp(period.end_time.date(), tz="UTC") + pd.Timedelta(
        days=spec.release_lag_days
    )
    return pd.Timestamp(row["available_at"]) >= floor


def _quality_rank(value: str) -> int:
    quality = str(value)
    if quality == "MARKET_CLOSE_RAW_UNADJUSTED_V1":
        return 100
    if quality in {
        "FRED_ALFRED_HISTORICAL_VINTAGE",
        "ECB_OFFICIAL_LATEST_ONLY",
        "EUROSTAT_OFFICIAL_LATEST_ONLY",
    }:
        return 90
    if quality.startswith("LIMITED_"):
        return 60
    if quality == "MARKET_CLOSE":
        return 20
    if quality == "ESTIMATED_RELEASE_LAG":
        return 10
    return 50


def _quality_confidence_multiplier(value: str) -> float:
    quality = str(value)
    if quality.startswith("LIMITED_"):
        return 0.25
    if quality in {
        "CONSERVATIVE_PERIOD_END_RELEASE_LAG_V1",
        "ECB_OFFICIAL_LATEST_ONLY",
        "EUROSTAT_OFFICIAL_LATEST_ONLY",
    }:
        return 0.8
    return 1.0


def _utc(value: datetime) -> datetime:
    return (
        value.replace(tzinfo=UTC)
        if value.tzinfo is None
        else value.astimezone(UTC)
    )
