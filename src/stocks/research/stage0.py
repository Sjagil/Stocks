from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.portfolio.coverage import (
    normalize_asset_class,
    normalize_asset_subclass,
)


OUTPUT_PATH = Path("output/research/p1/stage0-screen.json")
MINIMUM_HISTORY = 63
SURVIVOR_THRESHOLD = 0.55


def run_vectorized_stage0(project_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    universe_path = project_root / "output/universe/instruments.parquet"
    if not universe_path.is_file():
        return _blocked("DISCOVERY_UNIVERSE_MISSING")
    universe = pd.read_parquet(universe_path)
    candidates = universe.loc[universe["active_listing"].astype(bool)].copy()
    data_paths = _daily_data_paths(project_root)
    results: list[dict[str, Any]] = []
    for raw in candidates.to_dict(orient="records"):
        symbol = str(raw.get("symbol") or "").upper()
        path = data_paths.get(symbol)
        if path is None:
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        source = str(path.relative_to(project_root))
        if frame is None or len(frame) < MINIMUM_HISTORY:
            continue
        result = vectorized_stage0_score(frame)
        results.append(
            {
                "instrument_id": str(raw.get("instrument_id") or symbol),
                "symbol": symbol,
                "asset_class": normalize_asset_class(raw),
                "subclass": normalize_asset_subclass(raw),
                "underlying_commodity": str(raw.get("underlying_commodity") or "NONE"),
                "strategy_family": "MULTI_ASSET_TREND_MOMENTUM_STAGE0",
                "timeframe": "1d",
                "data_timestamp": result["data_timestamp"],
                "history_rows": len(frame),
                "score": result["score"],
                "survivor": result["score"] >= SURVIVOR_THRESHOLD,
                "components": result["components"],
                "volatility": result["volatility"],
                "momentum_20": result["momentum_20"],
                "momentum_63": result["momentum_63"],
                "above_sma_50": result["above_sma_50"],
                "last_price": result["last_price"],
                "data_source": source,
                "validation_status": (
                    "STAGE0_FALSIFICATION_SURVIVOR"
                    if result["score"] >= SURVIVOR_THRESHOLD
                    else "STAGE0_REJECTED"
                ),
                "exact_validation_required": True,
                "stage0_direct_promotion": False,
                "portfolio_eligible": False,
                "execution_eligible": False,
                "execution_authority": "NONE",
            }
        )
    results.sort(key=lambda row: (-float(row["score"]), row["symbol"]))
    survivors = [row for row in results if row["survivor"]]
    elapsed = time.perf_counter() - started
    by_class: dict[str, dict[str, int]] = {}
    for asset_class in ("EQUITY", "ETF", "COMMODITY_EXPOSURE"):
        evaluated = [row for row in results if row["asset_class"] == asset_class]
        by_class[asset_class] = {
            "evaluated": len(evaluated),
            "survivors": sum(bool(row["survivor"]) for row in evaluated),
            "rejected_before_exact_validation": sum(not bool(row["survivor"]) for row in evaluated),
        }
    breadth = _breadth_summary(results)
    leadership = sorted(
        (
            {
                "asset_class": asset_class,
                "median_score": values["median_score"],
                "survivor_ratio": values["survivor_ratio"],
                "medium_momentum_positive_ratio": values[
                    "medium_momentum_positive_ratio"
                ],
            }
            for asset_class, values in breadth.items()
        ),
        key=lambda row: (
            -float(row["median_score"]),
            -float(row["survivor_ratio"]),
            row["asset_class"],
        ),
    )
    for rank, row in enumerate(leadership, 1):
        row["leadership_rank"] = rank
    report: dict[str, Any] = {
        "schema": "multi_asset_vectorized_stage0_v1",
        "status": "GO" if results else "NO_DATA",
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": "FAST_APPROXIMATE_FALSIFICATION_ONLY",
        "hypothesis_count": 1,
        "parameter_combination_count": 1,
        "assets_tested": len(results),
        "timeframes_tested": ["1d"],
        "selection_criterion": f"COMPOSITE_SCORE_GTE_{SURVIVOR_THRESHOLD}",
        "survivor_count": len(survivors),
        "exact_validation_backlog_reduction_count": len(results) - len(survivors),
        "exact_validation_backlog_reduction_ratio": round((len(results) - len(survivors)) / len(results), 6) if results else 0.0,
        "elapsed_seconds": round(elapsed, 6),
        "assets_per_second": round(len(results) / elapsed, 3) if elapsed > 0 else None,
        "asset_class_summary": by_class,
        "breadth_by_asset_class": breadth,
        "cross_asset_leadership": leadership,
        "context_benchmarks": [
            {
                "symbol": row["symbol"],
                "asset_class": row["asset_class"],
                "data_timestamp": row["data_timestamp"],
                "momentum_20": row["momentum_20"],
                "momentum_63": row["momentum_63"],
                "volatility": row["volatility"],
            }
            for row in results
            if row["symbol"]
            in {
                "SPY", "QQQ", "IWM", "GLD", "IAU", "SLV", "CPER",
                "DBB", "XLE", "XLV", "XLY", "BIL", "USO", "URA",
                "URNM",
            }
        ],
        "survivors": survivors,
        "all_results_count": len(results),
        "stage0_direct_promotion": False,
        "exact_event_driven_validation_required": True,
        "walk_forward_required": True,
        "stress_test_required": True,
        "forward_required": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def vectorized_stage0_score(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = _ordered_frame(frame)
    close = pd.to_numeric(ordered["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(close) < MINIMUM_HISTORY:
        raise ValueError("INSUFFICIENT_STAGE0_HISTORY")
    returns = close.pct_change().dropna()
    momentum_20 = float(close.iloc[-1] / close.iloc[-21] - 1.0)
    momentum_63 = float(close.iloc[-1] / close.iloc[-63] - 1.0)
    sma_20 = float(close.iloc[-20:].mean())
    sma_50 = float(close.iloc[-50:].mean())
    prior_high = float(close.iloc[-21:-1].max())
    breakout = float(close.iloc[-1] / prior_high - 1.0) if prior_high > 0 else 0.0
    volatility = float(returns.iloc[-60:].std(ddof=0) * np.sqrt(252.0))
    trend = _bounded(0.5 + (sma_20 / sma_50 - 1.0) * 8.0) if sma_50 > 0 else 0.0
    medium_momentum = _bounded(0.5 + momentum_63 * 2.5)
    short_momentum = _bounded(0.5 + momentum_20 * 4.0)
    breakout_score = _bounded(0.5 + breakout * 10.0)
    volatility_quality = _bounded(1.0 - max(0.0, volatility - 0.15) / 0.85)
    score = (
        0.30 * trend
        + 0.25 * medium_momentum
        + 0.20 * short_momentum
        + 0.15 * breakout_score
        + 0.10 * volatility_quality
    )
    timestamp_column = next((name for name in ("timestamp_utc", "session_date", "date", "timestamp") if name in ordered), None)
    timestamp = str(ordered[timestamp_column].iloc[-1]) if timestamp_column else "UNKNOWN"
    return {
        "score": round(_bounded(score), 6),
        "components": {
            "trend": round(trend, 6),
            "medium_momentum": round(medium_momentum, 6),
            "short_momentum": round(short_momentum, 6),
            "breakout": round(breakout_score, 6),
            "volatility_quality": round(volatility_quality, 6),
        },
        "volatility": round(volatility, 8),
        "momentum_20": round(momentum_20, 8),
        "momentum_63": round(momentum_63, 8),
        "above_sma_50": bool(close.iloc[-1] > sma_50),
        "last_price": round(float(close.iloc[-1]), 8),
        "data_timestamp": timestamp,
    }


def _daily_data_paths(project_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    direct_root = project_root / "data/research/critical_trading/yfinance"
    if direct_root.exists():
        paths.update(
            {path.stem.upper(): path for path in direct_root.glob("*.parquet")}
        )
    root = project_root / "data/research/multitimeframe/private"
    if root.exists():
        for path in root.rglob("bars.parquet"):
            if "interval=1d" not in path.parts:
                continue
            symbol_part = next(
                (part for part in path.parts if part.startswith("symbol=")),
                "",
            )
            if symbol_part:
                paths.setdefault(
                    symbol_part.split("=", 1)[1].upper(), path
                )
    return paths


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp = next((name for name in ("timestamp_utc", "session_date", "date", "timestamp") if name in frame), None)
    return frame.sort_values(timestamp).reset_index(drop=True) if timestamp else frame.reset_index(drop=True)


def _bounded(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _breadth_summary(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for asset_class in ("EQUITY", "ETF", "COMMODITY_EXPOSURE"):
        selected = [
            row for row in results if row["asset_class"] == asset_class
        ]
        count = len(selected)
        summary[asset_class] = {
            "evaluated": count,
            "above_sma_50_ratio": _ratio(
                sum(bool(row["above_sma_50"]) for row in selected), count
            ),
            "short_momentum_positive_ratio": _ratio(
                sum(float(row["momentum_20"]) > 0 for row in selected), count
            ),
            "medium_momentum_positive_ratio": _ratio(
                sum(float(row["momentum_63"]) > 0 for row in selected), count
            ),
            "survivor_ratio": _ratio(
                sum(bool(row["survivor"]) for row in selected), count
            ),
            "median_score": round(
                float(np.median([row["score"] for row in selected])), 6
            )
            if selected
            else 0.0,
            "median_annualized_volatility": round(
                float(np.median([row["volatility"] for row in selected])), 8
            )
            if selected
            else None,
            "hard_veto": False,
            "incremental_validation_required": True,
        }
    return summary


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "multi_asset_vectorized_stage0_v1",
        "status": "NO_GO",
        "blockers": [reason],
        "stage0_direct_promotion": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


__all__ = ["run_vectorized_stage0", "vectorized_stage0_score"]
