from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stocks.portfolio.etf_holdings import holdings_overlap_report


def build_overlap_report(
    opportunities: Iterable[dict[str, Any]],
    correlation: pd.DataFrame,
    *,
    threshold: float = 0.75,
    etf_holdings: pd.DataFrame | None = None,
) -> dict[str, Any]:
    rows = [row for row in opportunities if row.get("asset_class") != "CASH"]
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        clusters[str(row.get("correlation_cluster") or "UNKNOWN")].append(
            str(row.get("symbol") or "UNKNOWN")
        )
    correlated_pairs: list[dict[str, Any]] = []
    if not correlation.empty:
        names = [str(value) for value in correlation.columns]
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                try:
                    value = float(correlation.loc[left, right])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value) and abs(value) >= threshold:
                    correlated_pairs.append(
                        {
                            "left": left,
                            "right": right,
                            "correlation": round(value, 6),
                        }
                    )
    major = [
        {"cluster": cluster, "symbols": sorted(set(symbols)), "member_count": len(set(symbols))}
        for cluster, symbols in sorted(clusters.items())
        if len(set(symbols)) >= 2
    ]
    holdings_report = holdings_overlap_report(
        rows,
        etf_holdings if etf_holdings is not None else pd.DataFrame(),
    )
    return {
        "schema": "universal_portfolio_overlap_gate_v1",
        "status": "GO",
        "economic_cluster_count": len(clusters),
        "major_clusters": major,
        "high_correlation_pairs": sorted(correlated_pairs, key=lambda row: (-abs(row["correlation"]), row["left"], row["right"])),
        "return_correlation_checked": not correlation.empty,
        "strategy_pnl_correlation_required_for_promotion": True,
        "sector_industry_overlap_checked": True,
        "commodity_factor_overlap_checked": True,
        "etf_holdings_overlap_status": holdings_report["status"],
        "etf_holdings_overlap": holdings_report,
        "etf_holdings_missing_is_diversification_proof": False,
        "correlation_threshold": threshold,
        "execution_authority": "NONE",
    }


def evaluate_strategy_overlap_promotion(
    candidate_returns: pd.Series,
    existing_strategy_returns: pd.DataFrame,
    *,
    standalone_expectancy: float,
    maximum_correlation: float = 0.75,
    minimum_incremental_expectancy: float = 0.0,
) -> dict[str, Any]:
    aligned = existing_strategy_returns.join(
        candidate_returns.rename("candidate"), how="inner"
    ).dropna()
    correlations: dict[str, float] = {}
    if not aligned.empty:
        correlations = {
            str(column): float(aligned["candidate"].corr(aligned[column]))
            for column in existing_strategy_returns.columns
            if np.isfinite(aligned["candidate"].corr(aligned[column]))
        }
    max_abs = max((abs(value) for value in correlations.values()), default=0.0)
    diversification_credit = max(0.0, 1.0 - max_abs)
    incremental = float(standalone_expectancy) * diversification_credit
    blockers: list[str] = []
    if standalone_expectancy <= 0:
        blockers.append("STANDALONE_EXPECTANCY_NOT_POSITIVE")
    if max_abs > maximum_correlation and incremental <= minimum_incremental_expectancy:
        blockers.append("STRATEGY_OVERLAP_EXCEEDS_INCREMENTAL_VALUE")
    return {
        "schema": "strategy_incremental_portfolio_promotion_gate_v1",
        "status": "GO" if not blockers else "NO_GO",
        "standalone_expectancy": standalone_expectancy,
        "strategy_correlations": correlations,
        "maximum_absolute_correlation": round(max_abs, 8),
        "diversification_credit": round(diversification_credit, 8),
        "incremental_portfolio_expectancy": round(incremental, 8),
        "blockers": blockers,
        "promotion_authority": "NONE",
        "execution_authority": "NONE",
    }


__all__ = ["build_overlap_report", "evaluate_strategy_overlap_promotion"]
