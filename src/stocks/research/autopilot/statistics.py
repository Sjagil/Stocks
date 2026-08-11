from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


def sample_classification(
    *,
    observations: int,
    closed_episodes: int,
    active_months: int,
) -> str:
    if observations < 126 or active_months < 6 or closed_episodes < 10:
        return "INSUFFICIENT_SAMPLE"
    if observations < 504 or active_months < 24 or closed_episodes < 30:
        return "LOW_CONFIDENCE"
    return "EVALUABLE"


def robustness_statistics(
    returns: pd.Series,
    *,
    closed_episodes: int,
    trial_count: int,
    periods_per_year: float,
    bootstrap_runs: int = 500,
    block_size: int = 20,
    seed: int = 20260726,
) -> dict[str, Any]:
    clean = returns.dropna().astype(float)
    active_months = (
        int(clean.index.tz_localize(None).to_period("M").nunique())
        if isinstance(clean.index, pd.DatetimeIndex)
        else int(len(clean) / max(1.0, periods_per_year / 12.0))
    )
    classification = sample_classification(
        observations=len(clean),
        closed_episodes=closed_episodes,
        active_months=active_months,
    )
    base = {
        "sample_status": classification,
        "observations": len(clean),
        "closed_episodes": closed_episodes,
        "active_months": active_months,
        "trial_count": trial_count,
        "block_bootstrap_runs": 0,
        "deflated_sharpe_probability": None,
        "pbo_status": "INSUFFICIENT_CROSS_VALIDATION_MATRIX",
        "white_reality_check": "INSUFFICIENT_SAMPLE",
        "hansen_spa": "INSUFFICIENT_SAMPLE",
    }
    if classification == "INSUFFICIENT_SAMPLE" or clean.std(ddof=1) <= 0:
        return base
    runs = min(max(int(bootstrap_runs), 100), 5_000)
    block = min(max(2, int(block_size)), max(2, len(clean) // 4))
    rng = np.random.default_rng(seed)
    terminal = np.empty(runs)
    maximum_drawdowns = np.empty(runs)
    profit_factors = np.empty(runs)
    values = clean.to_numpy()
    blocks_needed = math.ceil(len(values) / block)
    starts = np.arange(max(1, len(values) - block + 1))
    for run in range(runs):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [values[start : start + block] for start in sampled_starts]
        )[: len(values)]
        nav = np.cumprod(1.0 + sample)
        terminal[run] = nav[-1]
        maximum_drawdowns[run] = np.min(nav / np.maximum.accumulate(nav) - 1.0)
        gains = sample[sample > 0].sum()
        losses = sample[sample < 0].sum()
        profit_factors[run] = gains / abs(losses) if losses < 0 else np.inf
    sharpe = float(clean.mean() / clean.std(ddof=1) * np.sqrt(periods_per_year))
    expected_max = _expected_maximum_sharpe(max(1, trial_count))
    standard_error = math.sqrt(
        max(1e-12, (1.0 + 0.5 * sharpe * sharpe) / max(1, len(clean) - 1))
    )
    dsr = NormalDist().cdf((sharpe - expected_max) / standard_error)
    return {
        **base,
        "block_bootstrap_runs": runs,
        "median_terminal_nav": float(np.median(terminal)),
        "p5_terminal_nav": float(np.quantile(terminal, 0.05)),
        "p95_terminal_nav": float(np.quantile(terminal, 0.95)),
        "probability_of_loss": float(np.mean(terminal < 1.0)),
        "median_maximum_drawdown": float(np.median(maximum_drawdowns)),
        "p95_maximum_drawdown_loss": float(np.quantile(maximum_drawdowns, 0.05)),
        "probability_period_pf_above_one": float(np.mean(profit_factors > 1.0)),
        "raw_sharpe": sharpe,
        "selection_threshold_sharpe": expected_max,
        "deflated_sharpe_probability": float(dsr),
        "white_reality_check": "NOT_RUN_REQUIRES_SHARED_BENCHMARK_MATRIX",
        "hansen_spa": "NOT_RUN_REQUIRES_SHARED_BENCHMARK_MATRIX",
    }


def probability_of_backtest_overfitting(
    in_sample_scores: pd.DataFrame,
    out_of_sample_scores: pd.DataFrame,
) -> dict[str, Any]:
    if (
        in_sample_scores.shape != out_of_sample_scores.shape
        or in_sample_scores.shape[0] < 4
        or in_sample_scores.shape[1] < 2
    ):
        return {
            "status": "INSUFFICIENT_SAMPLE",
            "PBO": None,
            "fold_count": int(in_sample_scores.shape[0]),
            "configuration_count": int(in_sample_scores.shape[1]),
        }
    below_median = 0
    evaluable = 0
    for fold in in_sample_scores.index:
        in_row = in_sample_scores.loc[fold]
        out_row = out_of_sample_scores.loc[fold]
        if in_row.isna().all() or out_row.isna().all():
            continue
        winner = in_row.idxmax()
        rank_pct = out_row.rank(pct=True).get(winner)
        if pd.isna(rank_pct):
            continue
        below_median += float(rank_pct) <= 0.5
        evaluable += 1
    return {
        "status": "GO" if evaluable >= 4 else "INSUFFICIENT_SAMPLE",
        "PBO": below_median / evaluable if evaluable else None,
        "fold_count": evaluable,
        "configuration_count": int(in_sample_scores.shape[1]),
    }


def parameter_neighbor_stability(
    metrics: list[dict[str, Any]],
    *,
    minimum_neighbors: int = 3,
) -> dict[str, Any]:
    evaluable = [
        item
        for item in metrics
        if item.get("net_total_return") is not None
        and item.get("maximum_drawdown") is not None
    ]
    if len(evaluable) < minimum_neighbors:
        return {
            "status": "INSUFFICIENT_NEIGHBORS",
            "neighbor_count": len(evaluable),
            "positive_neighbor_ratio": None,
        }
    positive_ratio = sum(
        float(item["net_total_return"]) > 0 for item in evaluable
    ) / len(evaluable)
    drawdowns = [float(item["maximum_drawdown"]) for item in evaluable]
    sharpes = [
        float(item["Sharpe"])
        for item in evaluable
        if item.get("Sharpe") is not None
    ]
    stable = (
        positive_ratio >= 2 / 3
        and np.median(drawdowns) >= -0.50
        and bool(sharpes)
        and np.median(sharpes) > 0
    )
    return {
        "status": "GO" if stable else "UNSTABLE_PARAMETER_NEIGHBORHOOD",
        "neighbor_count": len(evaluable),
        "positive_neighbor_ratio": float(positive_ratio),
        "median_sharpe": float(np.median(sharpes)) if sharpes else None,
        "median_maximum_drawdown": float(np.median(drawdowns)),
    }


def cohort_stability(
    cohort_metrics: dict[str, dict[str, Any]],
    *,
    minimum_cohorts: int = 3,
) -> dict[str, Any]:
    evaluable = {
        name: metrics
        for name, metrics in cohort_metrics.items()
        if metrics.get("sample_status") != "INSUFFICIENT_SAMPLE"
        and metrics.get("net_total_return") is not None
    }
    if len(evaluable) < minimum_cohorts:
        return {
            "status": "INSUFFICIENT_COHORTS",
            "cohort_count": len(evaluable),
            "positive_cohort_ratio": None,
        }
    positive = sum(
        float(metrics["net_total_return"]) > 0
        for metrics in evaluable.values()
    )
    ratio = positive / len(evaluable)
    return {
        "status": "GO" if ratio >= 2 / 3 else "COHORT_INSTABILITY",
        "cohort_count": len(evaluable),
        "positive_cohort_ratio": float(ratio),
    }


def _expected_maximum_sharpe(trials: int) -> float:
    if trials <= 1:
        return 0.0
    probability = min(0.999999, max(0.500001, 1.0 - 1.0 / trials))
    return max(0.0, NormalDist().inv_cdf(probability))
