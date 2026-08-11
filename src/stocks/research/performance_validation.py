from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.execution.idempotency import stable_hash


OUTPUT_PATH = Path("output/research/p1/independent-performance-check.json")


def independent_performance_metrics(
    returns: pd.Series,
    *,
    periods_per_year: int = 252,
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    clean = pd.to_numeric(returns, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if clean.empty:
        return {"status": "NO_DATA"}
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = len(clean) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else None
    volatility = float(clean.std(ddof=0) * np.sqrt(periods_per_year))
    sharpe = float(clean.mean() / clean.std(ddof=0) * np.sqrt(periods_per_year)) if clean.std(ddof=0) > 0 else None
    downside = clean.loc[clean < 0]
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year)) if not downside.empty else 0.0
    sortino = float(clean.mean() * periods_per_year / downside_deviation) if downside_deviation > 0 else None
    maximum_drawdown = float(drawdown.min())
    calmar = float(cagr / abs(maximum_drawdown)) if cagr is not None and maximum_drawdown < 0 else None
    positive = float(clean.loc[clean > 0].sum())
    negative = float(abs(clean.loc[clean < 0].sum()))
    profit_factor = positive / negative if negative > 0 else None
    losses = clean.loc[clean < 0]
    wins = clean.loc[clean > 0]
    win_loss = float(wins.mean() / abs(losses.mean())) if not wins.empty and not losses.empty and losses.mean() != 0 else None
    rolling = clean.rolling(63)
    rolling_sharpe = (rolling.mean() / rolling.std(ddof=0) * np.sqrt(periods_per_year)).replace([np.inf, -np.inf], np.nan).dropna()
    rng = np.random.default_rng(seed)
    bootstrapped = np.empty(bootstrap_samples)
    values = clean.to_numpy(dtype=float)
    for index in range(bootstrap_samples):
        sample = rng.choice(values, size=len(values), replace=True)
        bootstrapped[index] = np.prod(1.0 + sample) - 1.0
    return {
        "status": "GO",
        "observations": len(clean),
        "total_return": round(float(equity.iloc[-1] - 1.0), 8),
        "cagr": _round(cagr),
        "annualized_volatility": round(volatility, 8),
        "sharpe": _round(sharpe),
        "sortino": _round(sortino),
        "calmar": _round(calmar),
        "maximum_drawdown": round(maximum_drawdown, 8),
        "maximum_drawdown_duration": _drawdown_duration(drawdown),
        "profit_factor": _round(profit_factor),
        "win_rate": round(float((clean > 0).mean()), 8),
        "win_loss_ratio": _round(win_loss),
        "rolling_63_sharpe_latest": _round(
            rolling_sharpe.iloc[-1] if not rolling_sharpe.empty else None
        ),
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "median_total_return": round(float(np.median(bootstrapped)), 8),
            "fifth_percentile_total_return": round(float(np.quantile(bootstrapped, 0.05)), 8),
            "loss_probability": round(float(np.mean(bootstrapped < 0)), 8),
        },
    }


def publish_independent_performance_check(project_root: Path) -> dict[str, Any]:
    candidates = sorted(
        project_root.glob(
            "output/research/**/top_equity_001_*.csv"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return _blocked("RETURN_SERIES_MISSING")
    frame = pd.read_csv(source)
    if "daily_return" not in frame:
        return _blocked("DAILY_RETURN_COLUMN_MISSING")
    metrics = independent_performance_metrics(frame["daily_return"])
    report: dict[str, Any] = {
        "schema": "independent_portfolio_performance_check_v1",
        "status": metrics["status"],
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(source.relative_to(project_root)),
        "method": "NATIVE_INDEPENDENT_RECOMPUTATION_USING_QUANTSTATS_CONCEPTS",
        "metrics": metrics,
        "second_financial_ledger_created": False,
        "source_financial_ledger_mutated": False,
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(report)
    output = project_root / OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _drawdown_duration(drawdown: pd.Series) -> int:
    maximum = 0
    current = 0
    for value in drawdown:
        if value < 0:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    return maximum


def _round(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(result) else round(result, 8)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "independent_portfolio_performance_check_v1",
        "status": "NO_GO",
        "blockers": [reason],
        "execution_authority": "NONE",
    }


__all__ = [
    "independent_performance_metrics",
    "publish_independent_performance_check",
]
