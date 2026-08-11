from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.execution.idempotency import stable_hash


CONFIG_PATH = Path("config/news/event_study_v1.json")
MATERIAL_EVENTS_PATH = Path(
    "output/news/intelligence/material-events.json"
)
PRIVATE_EVENT_LEDGER = Path(
    "data/news/private/intelligence/normalized-events.jsonl"
)
UNIVERSE_PATH = Path("output/universe/instruments.parquet")
DAILY_ROOT = Path("data/research/critical_trading/yfinance")
MULTITIMEFRAME_ROOT = Path("data/research/multitimeframe/private")
OUTPUT_ROOT = Path("output/news/event_study")
EVENT_TIME_MODES = {
    "OPERATIONAL_CAUSAL": "received_at",
    "HISTORICAL_DESCRIPTIVE": "published_at",
}


def build_news_event_study(
    project_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    generated_at = (now or datetime.now(UTC)).astimezone(UTC)
    config = _read_json(project_root / CONFIG_PATH)
    material_payload = _read_json(project_root / MATERIAL_EVENTS_PATH)
    material_rows = material_payload.get("rows", [])
    event_rows = _read_jsonl(project_root / PRIVATE_EVENT_LEDGER)
    receipt_map = _cluster_receipt_map(event_rows)
    universe = _universe_map(project_root)
    cache: dict[tuple[str, str], tuple[pd.DataFrame, str | None]] = {}
    labels: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    attempted = 0
    for cluster in material_rows:
        story_id = str(cluster.get("story_cluster_id") or "")
        published_at = _timestamp(cluster.get("first_published_at"))
        first_received_at = receipt_map.get(story_id)
        if not story_id or published_at is None:
            reason_counts["INVALID_EVENT_TIME"] += 1
            continue
        if first_received_at is None:
            reason_counts["RECEIVED_AT_UNAVAILABLE"] += 1
        for raw_symbol in cluster.get("symbols", []):
            symbol = str(raw_symbol).upper()
            metadata = universe.get(symbol, {})
            if not symbol:
                continue
            for mode, time_field in EVENT_TIME_MODES.items():
                event_at = (
                    first_received_at
                    if time_field == "received_at"
                    else published_at
                )
                for horizon, horizon_policy in config.get(
                    "horizons", {}
                ).items():
                    attempted += 1
                    if event_at is None:
                        labels.append(
                            _pending_label(
                                cluster,
                                symbol=symbol,
                                metadata=metadata,
                                mode=mode,
                                horizon=horizon,
                                reason="RECEIVED_AT_UNAVAILABLE",
                                published_at=published_at,
                                received_at=first_received_at,
                            )
                        )
                        reason_counts["RECEIVED_AT_UNAVAILABLE"] += 1
                        continue
                    label = _label_event_horizon(
                        project_root,
                        cluster=cluster,
                        symbol=symbol,
                        metadata=metadata,
                        mode=mode,
                        event_at=event_at,
                        published_at=published_at,
                        received_at=first_received_at,
                        horizon=horizon,
                        horizon_policy=horizon_policy,
                        config=config,
                        cache=cache,
                    )
                    labels.append(label)
                    reason_counts[str(label["label_status"])] += 1
    labels.sort(
        key=lambda row: (
            str(row["event_timestamp"] or ""),
            str(row["story_cluster_id"]),
            str(row["symbol"]),
            str(row["event_time_mode"]),
            str(row["horizon"]),
        )
    )
    report = _build_report(
        generated_at=generated_at,
        labels=labels,
        attempted=attempted,
        material_rows=material_rows,
        event_rows=event_rows,
        reason_counts=reason_counts,
        config=config,
        source_material_hash=material_payload.get("content_hash"),
    )
    _publish(project_root, report=report, labels=labels)
    return report


def news_event_study_status(project_root: Path) -> dict[str, Any]:
    payload = _read_json(project_root / OUTPUT_ROOT / "status.json")
    if payload:
        return payload
    return {
        "schema": "news_event_study_status_v1",
        "status": "NOT_RUN",
        "causal_training_eligible_label_count": 0,
        "standalone_entry_allowed": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
    }


def _label_event_horizon(
    project_root: Path,
    *,
    cluster: dict[str, Any],
    symbol: str,
    metadata: dict[str, Any],
    mode: str,
    event_at: datetime,
    published_at: datetime,
    received_at: datetime | None,
    horizon: str,
    horizon_policy: dict[str, Any],
    config: dict[str, Any],
    cache: dict[tuple[str, str], tuple[pd.DataFrame, str | None]],
) -> dict[str, Any]:
    interval = str(horizon_policy.get("source_interval"))
    horizon_bars = int(horizon_policy.get("bars", 0))
    asset, asset_hash = _load_bars(
        project_root, symbol, interval, config=config, cache=cache
    )
    market_symbol = str(config.get("market_benchmark", "SPY"))
    market, market_hash = _load_bars(
        project_root,
        market_symbol,
        interval,
        config=config,
        cache=cache,
    )
    sector_symbol = str(
        config.get("sector_benchmarks", {}).get(
            str(metadata.get("sector") or ""), ""
        )
    )
    if sector_symbol in {"", symbol, market_symbol}:
        sector_symbol = ""
        sector = pd.DataFrame()
        sector_hash = None
    else:
        sector, sector_hash = _load_bars(
            project_root,
            sector_symbol,
            interval,
            config=config,
            cache=cache,
        )
    base = _label_base(
        cluster,
        symbol=symbol,
        metadata=metadata,
        mode=mode,
        horizon=horizon,
        published_at=published_at,
        received_at=received_at,
        event_at=event_at,
    )
    if asset.empty:
        return {**base, "label_status": "ASSET_BARS_UNAVAILABLE"}
    if market.empty:
        return {**base, "label_status": "MARKET_BARS_UNAVAILABLE"}
    if interval == "1h":
        measurement = _measure_intraday(
            asset,
            market,
            sector,
            event_at=event_at,
            horizon_bars=horizon_bars,
            minimum_beta_observations=int(
                config.get("minimum_beta_observations", {}).get("1h", 60)
            ),
            maximum_beta=float(
                config.get("maximum_beta_absolute_value", 3.0)
            ),
        )
        timing_method = "FIRST_FULLY_POST_EVENT_1H_BAR"
    else:
        measurement = _measure_daily(
            asset,
            market,
            sector,
            event_at=event_at,
            horizon_bars=horizon_bars,
            minimum_beta_observations=int(
                config.get("minimum_beta_observations", {}).get("1d", 60)
            ),
            maximum_beta=float(
                config.get("maximum_beta_absolute_value", 3.0)
            ),
        )
        timing_method = "NEXT_FULL_SESSION_CLOSE_CONSERVATIVE"
    if measurement["label_status"] != "COMPLETE":
        return {
            **base,
            **measurement,
            "timing_method": timing_method,
            "asset_data_hash": asset_hash,
            "market_data_hash": market_hash,
            "sector_data_hash": sector_hash,
            "market_benchmark": market_symbol,
            "sector_benchmark": sector_symbol or None,
        }
    car = float(measurement["cumulative_abnormal_log_return"])
    raw_impact = float(cluster.get("raw_impact") or 0.0)
    latency = (
        (received_at - published_at).total_seconds()
        if received_at is not None
        else None
    )
    complete = {
        **base,
        **measurement,
        "timing_method": timing_method,
        "asset_data_hash": asset_hash,
        "market_data_hash": market_hash,
        "sector_data_hash": sector_hash,
        "market_benchmark": market_symbol,
        "sector_benchmark": sector_symbol or None,
        "receipt_lag_seconds": latency,
        "directional_label": (
            "POSITIVE" if car > 0 else "NEGATIVE" if car < 0 else "FLAT"
        ),
        "impact_direction_correct": (
            raw_impact * car > 0 if raw_impact != 0 and car != 0 else None
        ),
        "training_eligible": mode == "OPERATIONAL_CAUSAL",
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }
    complete["label_id"] = "CAR-" + stable_hash(complete)[:28]
    return complete


def _measure_intraday(
    asset: pd.DataFrame,
    market: pd.DataFrame,
    sector: pd.DataFrame,
    *,
    event_at: datetime,
    horizon_bars: int,
    minimum_beta_observations: int,
    maximum_beta: float,
) -> dict[str, Any]:
    return _measure(
        asset,
        market,
        sector,
        event_at=event_at,
        horizon_bars=horizon_bars,
        key="timestamp_utc",
        interval=timedelta(hours=1),
        minimum_beta_observations=minimum_beta_observations,
        maximum_beta=maximum_beta,
    )


def _measure_daily(
    asset: pd.DataFrame,
    market: pd.DataFrame,
    sector: pd.DataFrame,
    *,
    event_at: datetime,
    horizon_bars: int,
    minimum_beta_observations: int,
    maximum_beta: float,
) -> dict[str, Any]:
    return _measure(
        asset,
        market,
        sector,
        event_at=event_at,
        horizon_bars=horizon_bars,
        key="session_date",
        interval=None,
        minimum_beta_observations=minimum_beta_observations,
        maximum_beta=maximum_beta,
    )


def _measure(
    asset: pd.DataFrame,
    market: pd.DataFrame,
    sector: pd.DataFrame,
    *,
    event_at: datetime,
    horizon_bars: int,
    key: str,
    interval: timedelta | None,
    minimum_beta_observations: int,
    maximum_beta: float,
) -> dict[str, Any]:
    if horizon_bars < 1:
        return {"label_status": "INVALID_HORIZON"}
    prepared_asset = _prepare_bars(asset, key)
    prepared_market = _prepare_bars(market, key)
    prepared_sector = _prepare_bars(sector, key) if not sector.empty else sector
    if key == "timestamp_utc":
        event_key: Any = pd.Timestamp(event_at)
        close_time = prepared_asset[key] + pd.Timedelta(interval)
        before = prepared_asset.loc[close_time <= event_key]
        after = prepared_asset.loc[prepared_asset[key] >= event_key]
    else:
        event_key = event_at.date().isoformat()
        before = prepared_asset.loc[prepared_asset[key] < event_key]
        after = prepared_asset.loc[prepared_asset[key] > event_key]
    if before.empty:
        return {"label_status": "PRE_EVENT_BASELINE_UNAVAILABLE"}
    if len(after) < horizon_bars:
        return {
            "label_status": "FUTURE_HORIZON_PENDING",
            "available_future_bar_count": int(len(after)),
            "required_future_bar_count": horizon_bars,
        }
    baseline = before.iloc[-1]
    target = after.iloc[horizon_bars - 1]
    baseline_key = baseline[key]
    target_key = target[key]
    market_prices = _endpoint_prices(
        prepared_market, key, baseline_key, target_key
    )
    if market_prices is None:
        return {"label_status": "MARKET_ALIGNMENT_UNAVAILABLE"}
    sector_prices = (
        _endpoint_prices(prepared_sector, key, baseline_key, target_key)
        if not prepared_sector.empty
        else None
    )
    beta = _estimate_betas(
        prepared_asset,
        prepared_market,
        prepared_sector,
        key=key,
        baseline_key=baseline_key,
        minimum_observations=minimum_beta_observations,
        maximum_beta=maximum_beta,
    )
    asset_return = _log_return(baseline["price"], target["price"])
    market_return = _log_return(*market_prices)
    sector_return = (
        _log_return(*sector_prices) if sector_prices is not None else 0.0
    )
    expected = (
        float(beta["market_beta"]) * market_return
        + float(beta["sector_beta"]) * sector_return
    )
    car = asset_return - expected
    return {
        "label_status": "COMPLETE",
        "baseline_timestamp": _serializable_key(baseline_key),
        "target_timestamp": _serializable_key(target_key),
        "asset_log_return": round(asset_return, 10),
        "market_log_return": round(market_return, 10),
        "sector_log_return": round(sector_return, 10),
        "expected_log_return": round(expected, 10),
        "cumulative_abnormal_log_return": round(car, 10),
        "cumulative_abnormal_return": round(math.expm1(car), 10),
        "market_beta": beta["market_beta"],
        "sector_beta": beta["sector_beta"],
        "beta_observation_count": beta["observation_count"],
        "beta_status": beta["status"],
        "available_future_bar_count": int(len(after)),
        "required_future_bar_count": horizon_bars,
    }


def _estimate_betas(
    asset: pd.DataFrame,
    market: pd.DataFrame,
    sector: pd.DataFrame,
    *,
    key: str,
    baseline_key: Any,
    minimum_observations: int,
    maximum_beta: float,
) -> dict[str, Any]:
    asset_returns = _return_frame(asset, key, "asset")
    market_returns = _return_frame(market, key, "market")
    frames = [asset_returns, market_returns]
    has_sector = not sector.empty
    if has_sector:
        frames.append(_return_frame(sector, key, "sector"))
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=key, how="inner")
    merged = merged.loc[merged[key] <= baseline_key].tail(
        max(minimum_observations * 4, 240)
    )
    if len(merged) < minimum_observations:
        return {
            "market_beta": 1.0,
            "sector_beta": 0.0,
            "observation_count": int(len(merged)),
            "status": "FALLBACK_MARKET_BETA_ONE_INSUFFICIENT_HISTORY",
        }
    columns = ["market_return"]
    if has_sector:
        columns.append("sector_return")
    x = merged[columns].to_numpy(dtype=float)
    y = merged["asset_return"].to_numpy(dtype=float)
    try:
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        coefficients = np.array([1.0] + ([0.0] if has_sector else []))
    coefficients = np.clip(coefficients, -maximum_beta, maximum_beta)
    return {
        "market_beta": round(float(coefficients[0]), 8),
        "sector_beta": (
            round(float(coefficients[1]), 8) if has_sector else 0.0
        ),
        "observation_count": int(len(merged)),
        "status": "PRE_EVENT_OLS_NO_INTERCEPT",
    }


def _label_base(
    cluster: dict[str, Any],
    *,
    symbol: str,
    metadata: dict[str, Any],
    mode: str,
    horizon: str,
    published_at: datetime,
    received_at: datetime | None,
    event_at: datetime,
) -> dict[str, Any]:
    return {
        "schema": "news_car_label_v1",
        "label_id": None,
        "story_cluster_id": cluster.get("story_cluster_id"),
        "representative_event_id": cluster.get("representative_event_id"),
        "symbol": symbol,
        "sector": metadata.get("sector"),
        "industry": metadata.get("industry"),
        "event_classes": json.dumps(
            sorted(cluster.get("event_classes", [])), separators=(",", ":")
        ),
        "raw_impact": float(cluster.get("raw_impact") or 0.0),
        "materiality": float(cluster.get("materiality") or 0.0),
        "event_time_mode": mode,
        "event_timestamp": event_at.isoformat(),
        "published_at": published_at.isoformat(),
        "first_received_at": (
            received_at.isoformat() if received_at is not None else None
        ),
        "horizon": horizon,
        "label_status": "PENDING",
        "training_eligible": False,
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }


def _pending_label(
    cluster: dict[str, Any],
    *,
    symbol: str,
    metadata: dict[str, Any],
    mode: str,
    horizon: str,
    reason: str,
    published_at: datetime,
    received_at: datetime | None,
) -> dict[str, Any]:
    event_at = received_at or published_at
    return {
        **_label_base(
            cluster,
            symbol=symbol,
            metadata=metadata,
            mode=mode,
            horizon=horizon,
            published_at=published_at,
            received_at=received_at,
            event_at=event_at,
        ),
        "label_status": reason,
    }


def _load_bars(
    project_root: Path,
    symbol: str,
    interval: str,
    *,
    config: dict[str, Any],
    cache: dict[tuple[str, str], tuple[pd.DataFrame, str | None]],
) -> tuple[pd.DataFrame, str | None]:
    key = (symbol, interval)
    if key in cache:
        return cache[key]
    if interval == "1h":
        provider = str(config.get("intraday_source_provider", "YFINANCE"))
        path = (
            project_root
            / MULTITIMEFRAME_ROOT
            / f"provider={provider}"
            / f"symbol={symbol}"
            / "interval=1h"
            / "source_interval=1h"
            / "bars.parquet"
        )
    else:
        path = project_root / DAILY_ROOT / f"{symbol}.parquet"
    result: tuple[pd.DataFrame, str | None]
    try:
        frame = pd.read_parquet(path)
    except (FileNotFoundError, OSError, ValueError):
        result = (pd.DataFrame(), None)
    else:
        result = (frame, _file_hash(path))
    cache[key] = result
    return result


def _prepare_bars(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    if frame.empty or key not in frame.columns:
        return pd.DataFrame(columns=[key, "price"])
    price_column = (
        "adjusted_close"
        if "adjusted_close" in frame.columns
        else "close"
    )
    result = frame[[key, price_column]].copy()
    if key == "timestamp_utc":
        result[key] = pd.to_datetime(result[key], utc=True, errors="coerce")
    else:
        result[key] = pd.to_datetime(
            result[key], errors="coerce"
        ).dt.date.astype(str)
    result["price"] = pd.to_numeric(result[price_column], errors="coerce")
    result = result.dropna(subset=[key, "price"])
    result = result.loc[result["price"] > 0]
    return result.drop_duplicates(key, keep="last").sort_values(key)


def _endpoint_prices(
    frame: pd.DataFrame,
    key: str,
    baseline_key: Any,
    target_key: Any,
) -> tuple[float, float] | None:
    if frame.empty:
        return None
    indexed = frame.set_index(key)["price"]
    try:
        return float(indexed.loc[baseline_key]), float(indexed.loc[target_key])
    except KeyError:
        return None


def _return_frame(
    frame: pd.DataFrame, key: str, name: str
) -> pd.DataFrame:
    result = frame[[key, "price"]].copy()
    result[f"{name}_return"] = np.log(result["price"]).diff()
    return result[[key, f"{name}_return"]].dropna()


def _log_return(start: Any, end: Any) -> float:
    return math.log(float(end) / float(start))


def _build_report(
    *,
    generated_at: datetime,
    labels: list[dict[str, Any]],
    attempted: int,
    material_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
    reason_counts: Counter[str],
    config: dict[str, Any],
    source_material_hash: Any,
) -> dict[str, Any]:
    complete = [row for row in labels if row["label_status"] == "COMPLETE"]
    causal = [
        row
        for row in complete
        if row["event_time_mode"] == "OPERATIONAL_CAUSAL"
    ]
    descriptive = [
        row
        for row in complete
        if row["event_time_mode"] == "HISTORICAL_DESCRIPTIVE"
    ]
    gates = config.get("training_gates", {})
    smoke_minimum = int(gates.get("smoke_minimum_causal_labels", 100))
    serious_minimum = int(
        gates.get("serious_shadow_minimum_causal_labels", 500)
    )
    horizon_summary = _summary_by(complete, "horizon")
    mode_summary = _summary_by(complete, "event_time_mode")
    event_class_summary = _summary_by_event_class(complete)
    report = {
        "schema": "news_event_study_status_v1",
        "status": "GO_WITH_DOCUMENTED_GAPS" if complete else "DATA_UNAVAILABLE",
        "generated_at": generated_at.isoformat(),
        "material_story_cluster_count": len(material_rows),
        "normalized_event_count": len(event_rows),
        "attempted_label_count": attempted,
        "complete_label_count": len(complete),
        "historical_descriptive_complete_label_count": len(descriptive),
        "causal_received_at_complete_label_count": len(causal),
        "causal_training_eligible_label_count": sum(
            bool(row.get("training_eligible")) for row in causal
        ),
        "pending_or_unavailable_label_count": len(labels) - len(complete),
        "label_status_counts": dict(sorted(reason_counts.items())),
        "horizon_summary": horizon_summary,
        "event_time_mode_summary": mode_summary,
        "event_class_summary": event_class_summary,
        "intraday_15m_status": "UNAVAILABLE_NO_CANONICAL_15M_CACHE",
        "event_time_contract": {
            "operational_causal": "EARLIEST_RECEIVED_AT",
            "historical_descriptive": "FIRST_PUBLISHED_AT_DESCRIPTIVE_ONLY",
            "published_at_labels_training_eligible": False,
        },
        "car_method": (
            "PRE_EVENT_OLS_MARKET_AND_SECTOR_ADJUSTED_LOG_RETURN"
        ),
        "beta_fallback": "MARKET_BETA_ONE_WHEN_HISTORY_INSUFFICIENT",
        "daily_timing_precision": "NEXT_FULL_SESSION_CONSERVATIVE",
        "model_readiness": {
            "smoke_minimum": smoke_minimum,
            "serious_shadow_minimum": serious_minimum,
            "smoke_ready": len(causal) >= smoke_minimum,
            "serious_shadow_ready": len(causal) >= serious_minimum,
            "status": (
                "SHADOW_RESEARCH_READY"
                if len(causal) >= serious_minimum
                else "SMOKE_RESEARCH_READY"
                if len(causal) >= smoke_minimum
                else "NOT_TRAINED_INSUFFICIENT_CAUSAL_CAR_LABELS"
            ),
        },
        "source_material_hash": source_material_hash,
        "config_hash": stable_hash(config),
        "financial_authority": "DESCRIPTIVE_AND_SHADOW_RESEARCH_ONLY",
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _summary_by(
    rows: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    summaries = []
    for value, members in sorted(grouped.items()):
        cars = np.array(
            [float(row["cumulative_abnormal_return"]) for row in members]
        )
        direction_rows = [
            row
            for row in members
            if row.get("impact_direction_correct") is not None
        ]
        summaries.append(
            {
                key: value,
                "label_count": len(members),
                "median_car": round(float(np.median(cars)), 8),
                "mean_car": round(float(np.mean(cars)), 8),
                "positive_car_ratio": round(float(np.mean(cars > 0)), 8),
                "directional_accuracy": (
                    round(
                        sum(
                            bool(row["impact_direction_correct"])
                            for row in direction_rows
                        )
                        / len(direction_rows),
                        8,
                    )
                    if direction_rows
                    else None
                ),
            }
        )
    return summaries


def _publish(
    project_root: Path,
    *,
    report: dict[str, Any],
    labels: list[dict[str, Any]],
) -> None:
    root = project_root / OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "status.json", report)
    _atomic_json(
        root / "horizon-summary.json",
        {
            "schema": "news_car_horizon_summary_v1",
            "status": report["status"],
            "rows": report["horizon_summary"],
            "execution_authority": "NONE",
        },
    )
    _atomic_json(
        root / "training-readiness.json",
        {
            "schema": "news_car_training_readiness_v1",
            "status": report["model_readiness"]["status"],
            "causal_label_count": report[
                "causal_training_eligible_label_count"
            ],
            **report["model_readiness"],
            "published_at_labels_training_eligible": False,
            "model_authority": "SHADOW_ONLY",
            "execution_authority": "NONE",
        },
    )
    _atomic_json(
        root / "event-class-summary.json",
        {
            "schema": "news_car_event_class_summary_v1",
            "status": report["status"],
            "rows": report["event_class_summary"],
            "published_at_training_eligible": False,
            "standalone_entry_allowed": False,
            "execution_authority": "NONE",
        },
    )
    label_frame = pd.DataFrame(labels)
    temporary = root / "event-labels.parquet.tmp"
    target = root / "event-labels.parquet"
    label_frame.to_parquet(temporary, index=False)
    temporary.replace(target)


def _cluster_receipt_map(
    rows: list[dict[str, Any]],
) -> dict[str, datetime]:
    result: dict[str, datetime] = {}
    for row in rows:
        story_id = str(row.get("story_cluster_id") or "")
        received_at = _timestamp(row.get("received_at"))
        if not story_id or received_at is None:
            continue
        previous = result.get(story_id)
        if previous is None or received_at < previous:
            result[story_id] = received_at
    return result


def _summary_by_event_class(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            classes = json.loads(str(row.get("event_classes") or "[]"))
        except json.JSONDecodeError:
            classes = []
        for event_class in classes:
            grouped[str(event_class)].append(row)
    summaries = []
    for event_class, members in sorted(grouped.items()):
        cars = np.array(
            [float(row["cumulative_abnormal_return"]) for row in members]
        )
        causal_count = sum(
            row.get("event_time_mode") == "OPERATIONAL_CAUSAL"
            for row in members
        )
        direction_rows = [
            row
            for row in members
            if row.get("impact_direction_correct") is not None
        ]
        summaries.append(
            {
                "event_class": event_class,
                "label_count": len(members),
                "causal_label_count": causal_count,
                "descriptive_label_count": len(members) - causal_count,
                "median_car": round(float(np.median(cars)), 8),
                "mean_car": round(float(np.mean(cars)), 8),
                "positive_car_ratio": round(float(np.mean(cars > 0)), 8),
                "directional_accuracy": (
                    round(
                        sum(
                            bool(row["impact_direction_correct"])
                            for row in direction_rows
                        )
                        / len(direction_rows),
                        8,
                    )
                    if direction_rows
                    else None
                ),
                "training_eligible": causal_count > 0,
            }
        )
    return summaries


def _universe_map(project_root: Path) -> dict[str, dict[str, Any]]:
    try:
        frame = pd.read_parquet(project_root / UNIVERSE_PATH)
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return {
        str(row["symbol"]).upper(): {
            "sector": row.get("sector"),
            "industry": row.get("industry"),
        }
        for row in frame.to_dict("records")
        if row.get("symbol")
    }


def _serializable_key(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["build_news_event_study", "news_event_study_status"]
