from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stocks.execution.idempotency import stable_hash


OUTPUT_PATH = Path("output/research/p1/strategy-inventory.json")
CORRELATION_PATH = Path(
    "output/research/p1/strategy-pnl-correlation.parquet"
)
OOS_RETURN_PATHS = (
    Path("output/research/phase11_13/daily-oos-returns.parquet"),
    Path("output/research/phase11_14/oos-returns.parquet"),
)


def publish_strategy_inventory(project_root: Path) -> dict[str, Any]:
    path = project_root / "output/research/results/portfolio_results.csv"
    if not path.is_file():
        return _blocked("PORTFOLIO_RESULT_INVENTORY_MISSING")
    frame = pd.read_csv(path)
    pnl_returns = _strategy_pnl_returns(project_root)
    pnl_correlation = pnl_returns.corr(min_periods=30)
    strategies: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        asset_class = _asset_class(row)
        strategy_id = str(row.get("strategy_id"))
        correlation_evidence = _correlation_evidence(
            strategy_id, pnl_correlation
        )
        promotion_blockers = _split(row.get("deployment_blockers"))
        if correlation_evidence["status"] != "GO":
            promotion_blockers.append(
                "STRATEGY_PNL_CORRELATION_SERIES_REQUIRED"
            )
        standalone_expectancy = _number(row.get("combined_oos_CAGR"))
        maximum_correlation = correlation_evidence.get(
            "maximum_absolute_correlation"
        )
        incremental_expectancy = (
            standalone_expectancy * max(0.0, 1.0 - maximum_correlation)
            if standalone_expectancy is not None
            and maximum_correlation is not None
            else None
        )
        strategies.append(
            {
                "strategy_id": strategy_id,
                "strategy_family": _first_text(
                    row.get("formula"),
                    row.get("source_strategy_id"),
                    row.get("strategy_id"),
                ),
                "asset_class": asset_class,
                "instrument_scope": f"PIT_{asset_class}_UNIVERSE",
                "timeframe": str(row.get("timeframe") or "UNKNOWN"),
                "signal_count": None,
                "historical_trades": _integer(
                    row.get("normal_cost_fill_count")
                ),
                "net_expectancy": _number(
                    row.get("combined_oos_CAGR")
                ),
                "profit_factor": _number(
                    row.get("combined_period_profit_factor")
                ),
                "sharpe": _number(row.get("combined_oos_Sharpe")),
                "sortino": None,
                "max_drawdown": _number(row.get("maximum_drawdown")),
                "turnover": None,
                "average_holding_time": None,
                "cost_sensitivity": {
                    "20bps_return": _number(
                        row.get("cost_20bps_combined_return")
                    ),
                    "50bps_return": _number(
                        row.get("cost_50bps_combined_return")
                    ),
                    "100bps_return": _number(
                        row.get("cost_100bps_combined_return")
                    ),
                },
                "walk_forward": {
                    "fold_count": _integer(row.get("fold_count")),
                    "positive_fold_ratio": _number(
                        row.get("positive_fold_ratio")
                    ),
                    "standard_manifest_required": True,
                },
                "regime_performance": "SEPARATE_REGIME_EVIDENCE_REQUIRED",
                "parameter_stability": _number(
                    row.get("parameter_plateau_ratio")
                ),
                "forward_evidence": bool(
                    row.get("forward_observer_candidate", False)
                ),
                "standalone_research_pass": bool(
                    row.get("research_pass", False)
                ),
                "standalone_robust_pass": bool(
                    row.get("robust_pass", False)
                ),
                "incremental_portfolio_expectancy": _number(
                    incremental_expectancy
                ),
                "strategy_pnl_correlation": correlation_evidence,
                "promotion_status": (
                    "RESEARCH_INCREMENTAL_GO"
                    if bool(row.get("robust_pass", False))
                    and incremental_expectancy is not None
                    and incremental_expectancy > 0
                    else "NO_GO"
                ),
                "promotion_blockers": sorted(set(promotion_blockers)),
                "automatic_promotion": False,
                "execution_authority": "NONE",
            }
        )
    by_class = {
        asset_class: sum(
            row["asset_class"] == asset_class for row in strategies
        )
        for asset_class in ("EQUITY", "ETF", "COMMODITY_EXPOSURE")
    }
    report: dict[str, Any] = {
        "schema": "strategy_family_inventory_by_asset_class_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "strategy_count": len(strategies),
        "asset_class_counts": by_class,
        "strategies": strategies,
        "asset_class_evidence_is_not_transferable": True,
        "strategy_correlation_is_promotion_requirement": True,
        "strategy_pnl_series_count": len(pnl_returns.columns),
        "strategy_pnl_correlation_artifact": CORRELATION_PATH.as_posix(),
        "missing_strategy_pnl_series_fails_promotion_closed": True,
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(report)
    output = project_root / OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    correlation_output = project_root / CORRELATION_PATH
    correlation_output.parent.mkdir(parents=True, exist_ok=True)
    pnl_correlation.rename_axis("strategy_id").reset_index().to_parquet(
        correlation_output, index=False
    )
    return report


def _strategy_pnl_returns(project_root: Path) -> pd.DataFrame:
    frames = []
    for relative in OOS_RETURN_PATHS:
        path = project_root / relative
        if not path.is_file():
            continue
        try:
            candidate = pd.read_parquet(path)
        except (OSError, ValueError):
            continue
        required = {"strategy_id", "cost_bps", "date", "daily_return"}
        if required.issubset(candidate.columns):
            frames.append(candidate)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    normal = frame.loc[
        pd.to_numeric(frame["cost_bps"], errors="coerce").eq(10.0)
    ].copy()
    normal["date"] = pd.to_datetime(
        normal["date"], utc=True, errors="coerce"
    ).dt.normalize()
    normal["daily_return"] = pd.to_numeric(
        normal["daily_return"], errors="coerce"
    )
    normal = normal.dropna(subset=["date", "daily_return", "strategy_id"])
    if normal.empty:
        return pd.DataFrame()
    daily = (
        normal.groupby(["date", "strategy_id"], sort=True)["daily_return"]
        .apply(lambda values: float((1.0 + values).prod() - 1.0))
        .unstack("strategy_id")
        .sort_index()
    )
    return daily


def _correlation_evidence(
    strategy_id: str, correlation: pd.DataFrame
) -> dict[str, Any]:
    if strategy_id not in correlation.columns:
        return {
            "status": "NO_GO",
            "reason": "OOS_STRATEGY_PNL_SERIES_MISSING",
            "maximum_absolute_correlation": None,
            "top_correlations": [],
        }
    values = correlation[strategy_id].drop(labels=[strategy_id]).dropna()
    ordered = sorted(
        (
            {
                "strategy_id": str(other),
                "correlation": round(float(value), 8),
            }
            for other, value in values.items()
        ),
        key=lambda item: (
            -abs(float(item["correlation"])), item["strategy_id"]
        ),
    )
    maximum = max(
        (abs(float(item["correlation"])) for item in ordered),
        default=0.0,
    )
    return {
        "status": "GO" if ordered else "NO_GO",
        "reason": None if ordered else "NO_OVERLAPPING_STRATEGY_SERIES",
        "maximum_absolute_correlation": round(maximum, 8),
        "top_correlations": ordered[:5],
        "cost_bps": 10.0,
        "minimum_pairwise_observations": 30,
    }


def _asset_class(row: dict[str, Any]) -> str:
    raw = str(row.get("asset_class") or "").upper()
    if raw in {"STOCK", "EQUITY"}:
        return "EQUITY"
    if raw == "ETF":
        return "ETF"
    if "COMMODITY" in raw:
        return "COMMODITY_EXPOSURE"
    identity = str(row.get("strategy_id") or "").upper()
    return "ETF" if "MULTI_ASSET" in identity else "EQUITY"


def _split(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item for item in str(value).split("|") if item]


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if text:
            return text
    return "UNKNOWN"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(result) else round(result, 8)


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None else int(number)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "strategy_family_inventory_by_asset_class_v1",
        "status": "NO_GO",
        "blockers": [reason],
        "execution_authority": "NONE",
    }


__all__ = ["publish_strategy_inventory"]
