from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.execution.idempotency import stable_hash


OUTPUT_PATH = Path("output/portfolio/cross-asset-intelligence.json")


def build_cross_asset_intelligence(
    project_root: Path,
    *,
    stage0: dict[str, Any],
    macro: dict[str, Any] | None = None,
) -> dict[str, Any]:
    macro_payload = macro or {}
    if not isinstance(macro_payload.get("regime"), dict):
        macro_payload = _read_json(project_root / "output/macro/regime.json")
    regime = macro_payload.get("regime", {})
    breadth = stage0.get("breadth_by_asset_class", {})
    benchmarks = {
        str(row.get("symbol")): row
        for row in stage0.get("context_benchmarks", [])
    }
    equity = breadth.get("EQUITY", {})
    commodity = breadth.get("COMMODITY_EXPOSURE", {})
    timestamps = [
        _timestamp(row.get("data_timestamp"))
        for row in stage0.get("context_benchmarks", [])
    ]
    timestamps = [value for value in timestamps if value is not None]
    latest = max(timestamps) if timestamps else None
    age_days = (
        (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 86400
        if latest is not None
        else None
    )
    dimensions = {
        "EQUITY_TREND": _direction(equity.get("above_sma_50_ratio")),
        "EQUITY_BREADTH": _direction(
            equity.get("medium_momentum_positive_ratio")
        ),
        "VOLATILITY_REGIME": _volatility_regime(
            equity.get("median_annualized_volatility")
        ),
        "RATES_REGIME": regime.get("monetary_regime", "UNKNOWN"),
        "REAL_YIELD_REGIME": "DATA_UNAVAILABLE_SEPARATE_FROM_RATES",
        "USD_REGIME": regime.get("currency_regime", "UNKNOWN"),
        "INFLATION_REGIME": regime.get("inflation_regime", "UNKNOWN"),
        "GROWTH_REGIME": regime.get("growth_regime", "UNKNOWN"),
        "LIQUIDITY_REGIME": regime.get("liquidity_regime", "UNKNOWN"),
        "CREDIT_REGIME": regime.get("credit_regime", "UNKNOWN"),
        "COMMODITY_REGIME": regime.get(
            "commodity_regime",
            _direction(commodity.get("medium_momentum_positive_ratio")),
        ),
        "RISK_ON_OFF": regime.get("market_regime", "UNKNOWN"),
    }
    comparisons = [
        _comparison(benchmarks, "GLD", "SPY", "gold_vs_equities"),
        _comparison(benchmarks, "SLV", "GLD", "silver_vs_gold"),
        _comparison(benchmarks, "CPER", "GLD", "copper_vs_gold"),
        _comparison(benchmarks, "XLE", "SPY", "energy_vs_broad_equities"),
        _comparison(benchmarks, "XLV", "XLY", "defensive_vs_cyclical"),
        _comparison(benchmarks, "IWM", "SPY", "small_vs_large_cap"),
    ]
    commodity_models = _commodity_family_models(benchmarks, dimensions)
    report: dict[str, Any] = {
        "schema": "p1_cross_asset_regime_breadth_leadership_v1",
        "status": "GO" if breadth else "NO_GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "regime_dimensions": dimensions,
        "breadth_by_asset_class": breadth,
        "cross_asset_leadership": stage0.get("cross_asset_leadership", []),
        "relative_strength_comparisons": comparisons,
        "commodity_family_models": commodity_models,
        "available_comparison_count": sum(
            row["status"] == "AVAILABLE" for row in comparisons
        ),
        "freshness": {
            "latest_benchmark_data_timestamp": (
                latest.isoformat() if latest is not None else None
            ),
            "age_calendar_days": (
                round(age_days, 4) if age_days is not None else None
            ),
            "status": (
                "CURRENT_CLOSED_DAILY_DATA"
                if age_days is not None and age_days <= 5
                else "STALE_CRITICAL_MARKET_DATA"
            ),
        },
        "breadth_is_automatic_hard_veto": False,
        "macro_is_automatic_entry_authority": False,
        "options_context_authority": "OBSERVATION_ONLY",
        "news_context_authority": "RISK_AND_CATALYST_CONTEXT_ONLY",
        "ranking_influence_requires_historical_incremental_evidence": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    path = project_root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _comparison(
    benchmarks: dict[str, dict[str, Any]],
    left: str,
    right: str,
    label: str,
) -> dict[str, Any]:
    left_row = benchmarks.get(left)
    right_row = benchmarks.get(right)
    if left_row is None or right_row is None:
        return {
            "comparison": label,
            "left": left,
            "right": right,
            "status": "DATA_UNAVAILABLE",
            "leader": None,
        }
    difference_20 = float(left_row["momentum_20"]) - float(
        right_row["momentum_20"]
    )
    difference_63 = float(left_row["momentum_63"]) - float(
        right_row["momentum_63"]
    )
    composite = 0.4 * difference_20 + 0.6 * difference_63
    return {
        "comparison": label,
        "left": left,
        "right": right,
        "status": "AVAILABLE",
        "left_minus_right_return_20": round(difference_20, 8),
        "left_minus_right_return_63": round(difference_63, 8),
        "composite_relative_strength": round(composite, 8),
        "leader": left if composite > 0 else right if composite < 0 else "TIE",
        "portfolio_authority": "CONTEXT_ONLY_PENDING_ABLATION",
    }


def _commodity_family_models(
    benchmarks: dict[str, dict[str, Any]],
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    contracts = {
        "GOLD": {
            "benchmark": "GLD",
            "model": "GOLD_TREND_USD_REAL_YIELD_CONTEXT",
            "features": ["trend", "momentum", "USD", "real_yield"],
        },
        "SILVER": {
            "benchmark": "SLV",
            "model": "SILVER_TREND_GOLD_RELATIVE_STRENGTH",
            "features": ["trend", "momentum", "silver_vs_gold", "volatility"],
        },
        "COPPER": {
            "benchmark": "CPER",
            "model": "COPPER_TREND_GROWTH_SENSITIVITY",
            "features": ["trend", "momentum", "growth", "USD"],
        },
        "URANIUM": {
            "benchmark": "URA",
            "model": "URANIUM_STRUCTURAL_TREND_FUND_BREADTH",
            "features": ["trend", "momentum", "URA_URNM_breadth", "event_risk"],
        },
        "OIL": {
            "benchmark": "USO",
            "model": "OIL_TREND_INFLATION_GROWTH_CONTEXT",
            "features": ["trend", "momentum", "inflation", "growth"],
        },
        "ENERGY": {
            "benchmark": "XLE",
            "model": "ENERGY_EQUITY_RELATIVE_STRENGTH_REGIME",
            "features": ["trend", "momentum", "relative_to_SPY", "equity_regime"],
        },
    }
    output: dict[str, Any] = {}
    for family, contract in contracts.items():
        benchmark = benchmarks.get(str(contract["benchmark"]))
        output[family] = {
            **contract,
            "status": "DATA_AVAILABLE" if benchmark else "DATA_UNAVAILABLE",
            "momentum_20": (
                benchmark.get("momentum_20") if benchmark else None
            ),
            "momentum_63": (
                benchmark.get("momentum_63") if benchmark else None
            ),
            "macro_context": {
                "growth": dimensions.get("GROWTH_REGIME"),
                "inflation": dimensions.get("INFLATION_REGIME"),
                "usd": dimensions.get("USD_REGIME"),
                "real_yield": dimensions.get("REAL_YIELD_REGIME"),
            },
            "same_model_as_other_commodities": False,
            "portfolio_authority": "CONTEXT_ONLY_PENDING_ASSET_CLASS_VALIDATION",
        }
    return output


def _direction(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if number >= 0.6:
        return "STRONG"
    if number <= 0.4:
        return "WEAK"
    return "MIXED"


def _volatility_regime(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if number >= 0.45:
        return "HIGH"
    if number <= 0.20:
        return "LOW"
    return "NORMAL"


def _timestamp(value: Any) -> pd.Timestamp | None:
    try:
        timestamp = pd.to_datetime(value, utc=True)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(timestamp) else timestamp


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


__all__ = ["build_cross_asset_intelligence"]
