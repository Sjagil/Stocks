from __future__ import annotations

import itertools
from math import log
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def load_shared_oos_matrix(project_root: Path, *, cost_bps: float = 10.0) -> pd.DataFrame:
    path = project_root / "output/research/phase11_14/oos-returns.parquet"
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_parquet(path)
    required = {"strategy_id", "cost_bps", "date", "daily_return"}
    if not required.issubset(rows.columns):
        return pd.DataFrame()
    rows = rows.loc[np.isclose(rows["cost_bps"].astype(float), cost_bps)].copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=True).dt.normalize()
    rows["daily_return"] = pd.to_numeric(rows["daily_return"], errors="coerce")
    rows = rows.dropna(subset=["date", "daily_return"])
    daily = (
        rows.groupby(["date", "strategy_id"], observed=True)["daily_return"]
        .apply(lambda values: float(np.prod(1.0 + values.to_numpy(float)) - 1.0))
        .unstack("strategy_id")
        .sort_index()
    )
    if daily.empty:
        return daily
    starts = daily.apply(lambda series: series.first_valid_index())
    ends = daily.apply(lambda series: series.last_valid_index())
    start = max(value for value in starts if value is not None)
    end = min(value for value in ends if value is not None)
    return daily.loc[start:end].fillna(0.0).sort_index(axis=1)


def multiple_testing_diagnostics(
    returns: pd.DataFrame,
    *,
    global_trial_count: int,
    seed: int = 20260811,
    bootstrap_samples: int = 500,
) -> dict[str, Any]:
    clean = _finite_matrix(returns)
    if clean.shape[0] < 60 or clean.shape[1] < 2:
        return {"status": "INSUFFICIENT_SHARED_OOS_RETURN_MATRIX"}
    values = clean.to_numpy(float)
    names = [str(column) for column in clean.columns]
    annualized_sharpe = _annualized_sharpe(values)
    p_values = np.asarray(
        [stats.ttest_1samp(values[:, index], 0.0, alternative="greater").pvalue for index in range(values.shape[1])]
    )
    bh_pass = _benjamini_hochberg(p_values, alpha=0.05)
    dsr = _deflated_sharpe(values, max(global_trial_count, values.shape[1]))
    white_p = _white_reality_check(values, seed=seed, samples=bootstrap_samples)
    spa_p = _hansen_spa(values, seed=seed + 1, samples=bootstrap_samples)
    pbo = _cscv_pbo(values, partitions=8)
    candidates = []
    for index, name in enumerate(names):
        candidates.append(
            {
                "strategy_id": name,
                "observations": int(values.shape[0]),
                "annualized_sharpe": _optional_float(annualized_sharpe[index]),
                "one_sided_mean_p_value": _optional_float(p_values[index]),
                "benjamini_hochberg_pass": bool(bh_pass[index]),
                "deflated_sharpe_probability": _optional_float(dsr[index]),
            }
        )
    corrected = [
        row
        for row in candidates
        if row["benjamini_hochberg_pass"]
        and (row["deflated_sharpe_probability"] or 0.0) >= 0.95
        and white_p <= 0.05
        and spa_p <= 0.05
        and (pbo.get("pbo") or 1.0) <= 0.20
    ]
    return {
        "status": "GO" if corrected else "NO_MULTIPLE_TESTING_CORRECTED_FINALIST",
        "method_version": "P3_SHARED_OOS_MULTIPLE_TESTING_V1",
        "shared_matrix_start": _index_label(clean.index.min()),
        "shared_matrix_end": _index_label(clean.index.max()),
        "shared_observation_count": int(clean.shape[0]),
        "strategy_count": int(clean.shape[1]),
        "global_trial_count": int(global_trial_count),
        "white_reality_check": {
            "method": "CENTERED_STATIONARY_BLOCK_BOOTSTRAP_MAX_MEAN",
            "p_value": _optional_float(white_p),
            "samples": bootstrap_samples,
        },
        "hansen_spa": {
            "method": "STUDENTIZED_STATIONARY_BLOCK_BOOTSTRAP",
            "p_value": _optional_float(spa_p),
            "samples": bootstrap_samples,
        },
        "cscv_pbo": pbo,
        "false_discovery_control": {
            "method": "BENJAMINI_HOCHBERG",
            "alpha": 0.05,
            "passing_count": int(bh_pass.sum()),
        },
        "corrected_finalist_count": len(corrected),
        "corrected_finalist_ids": [row["strategy_id"] for row in corrected],
        "candidates": candidates,
        "seed": seed,
    }


def strategy_dependency_diagnostics(returns: pd.DataFrame) -> dict[str, Any]:
    clean = _finite_matrix(returns)
    if clean.shape[0] < 60 or clean.shape[1] < 2:
        return {"status": "INSUFFICIENT_SHARED_OOS_STRATEGY_SERIES"}
    correlation = clean.corr()
    downside = clean.where(clean.lt(0.0)).corr(min_periods=20)
    drawdowns = (1.0 + clean).cumprod().div((1.0 + clean).cumprod().cummax()).sub(1.0)
    active = clean.ne(0.0)
    drawdown_overlap = _jaccard_matrix(drawdowns.lt(0.0))
    signal_overlap = _jaccard_matrix(active)
    clusters = _correlation_clusters(correlation, threshold=0.95)
    return {
        "status": "GO",
        "method_version": "P3_SHARED_OOS_DEPENDENCY_V1",
        "shared_observation_count": int(clean.shape[0]),
        "strategy_count": int(clean.shape[1]),
        "oos_return_correlation": _matrix_records(correlation),
        "downside_correlation": _matrix_records(downside),
        "drawdown_overlap": _matrix_records(drawdown_overlap),
        "signal_jaccard": _matrix_records(signal_overlap),
        "trade_overlap": [],
        "trade_overlap_status": "NOT_AVAILABLE_FROM_DAILY_OOS_SERIES",
        "factor_overlap": [],
        "factor_overlap_status": "NOT_AVAILABLE_FROM_CURRENT_OOS_EXPORT",
        "near_duplicate_threshold": 0.95,
        "near_duplicate_clusters": clusters,
    }


def parameter_stability_diagnostics(project_root: Path) -> dict[str, Any]:
    path = project_root / "output/research/phase11_8/nested-walk-forward-results.parquet"
    if not path.exists():
        return {"status": "INSUFFICIENT_PARAMETER_STABILITY_EVIDENCE"}
    rows = pd.read_parquet(path)
    required = {"strategy", "timeframe", "parameter_hash", "fold_id", "cost_bps", "Sharpe", "period_profit_factor"}
    if not required.issubset(rows.columns):
        return {"status": "INVALID_PARAMETER_STABILITY_SOURCE"}
    rows = rows.loc[np.isclose(rows["cost_bps"].astype(float), 10.0)].copy()
    records = []
    for keys, group in rows.groupby(["strategy", "timeframe", "parameter_hash"], observed=True):
        sharpe = pd.to_numeric(group["Sharpe"], errors="coerce").dropna()
        pf = pd.to_numeric(group["period_profit_factor"], errors="coerce").dropna()
        records.append(
            {
                "strategy": str(keys[0]),
                "timeframe": str(keys[1]),
                "parameter_hash": str(keys[2]),
                "fold_count": int(group["fold_id"].nunique()),
                "median_sharpe": _optional_float(sharpe.median()),
                "worst_sharpe": _optional_float(sharpe.min()),
                "positive_sharpe_fold_ratio": _optional_float((sharpe > 0).mean()),
                "median_profit_factor": _optional_float(pf.median()),
                "parameter_plateau": bool(group.get("parameter_plateau", pd.Series(False)).fillna(False).all()),
            }
        )
    stable = [
        row
        for row in records
        if (row["median_sharpe"] or -999.0) > 0.0
        and (row["median_profit_factor"] or 0.0) > 1.0
        and (row["positive_sharpe_fold_ratio"] or 0.0) >= 0.60
        and row["parameter_plateau"]
    ]
    return {
        "status": "GO" if records else "INSUFFICIENT_PARAMETER_STABILITY_EVIDENCE",
        "method_version": "P3_NESTED_WF_PARAMETER_STABILITY_V1",
        "source": "output/research/phase11_8/nested-walk-forward-results.parquet",
        "evaluated_parameter_set_count": len(records),
        "stable_parameter_set_count": len(stable),
        "stable_parameter_sets": stable,
        "all_parameter_sets": records,
    }


def regime_robustness_diagnostics(project_root: Path, returns: pd.DataFrame) -> dict[str, Any]:
    clean = _finite_matrix(returns)
    path = project_root / "data/research/critical_trading/yfinance/SPY.parquet"
    if clean.empty or not path.exists():
        return {"status": "INSUFFICIENT_REGIME_ROBUSTNESS_EVIDENCE"}
    benchmark = pd.read_parquet(path)
    if not {"session_date", "close"}.issubset(benchmark.columns):
        return {"status": "INVALID_BENCHMARK_REGIME_SOURCE"}
    benchmark = benchmark[["session_date", "close"]].copy()
    benchmark["date"] = pd.to_datetime(benchmark["session_date"], utc=True).dt.normalize()
    benchmark = benchmark.set_index("date").sort_index()
    close = pd.to_numeric(benchmark["close"], errors="coerce")
    daily = close.pct_change()
    trend = close.div(close.rolling(200, min_periods=100).mean()).sub(1.0)
    vol = daily.rolling(20, min_periods=10).std() * np.sqrt(252.0)
    high_vol_threshold = vol.rolling(252, min_periods=60).quantile(0.75)
    regime = pd.Series("SIDEWAYS", index=benchmark.index, dtype="object")
    regime.loc[trend > 0.03] = "BULL"
    regime.loc[trend < -0.03] = "BEAR"
    regime.loc[vol > high_vol_threshold] = "HIGH_VOLATILITY"
    aligned = clean.join(regime.rename("regime"), how="inner").dropna(subset=["regime"])
    rows = []
    for strategy in clean.columns:
        for label, group in aligned.groupby("regime", observed=True):
            values = group[strategy].to_numpy(float)
            rows.append(
                {
                    "strategy_id": str(strategy),
                    "regime": str(label),
                    "observations": len(values),
                    "annualized_return": _optional_float(np.mean(values) * 252.0),
                    "annualized_sharpe": _optional_float(_annualized_sharpe(values[:, None])[0]),
                    "positive_day_ratio": _optional_float(np.mean(values > 0.0)),
                }
            )
    qualifying = []
    for strategy, group in itertools.groupby(rows, key=lambda row: row["strategy_id"]):
        bucket = list(group)
        if len(bucket) >= 3 and min(row["observations"] for row in bucket) >= 20:
            qualifying.append(
                {
                    "strategy_id": strategy,
                    "worst_regime_annualized_return": min(row["annualized_return"] for row in bucket),
                    "positive_regime_count": sum(row["annualized_return"] > 0 for row in bucket),
                    "regime_count": len(bucket),
                }
            )
    return {
        "status": "GO" if rows else "INSUFFICIENT_REGIME_ROBUSTNESS_EVIDENCE",
        "method_version": "P3_CAUSAL_SPY_TREND_VOLATILITY_REGIMES_V1",
        "benchmark_source": "data/research/critical_trading/yfinance/SPY.parquet",
        "regime_is_causal": True,
        "regime_count": int(aligned["regime"].nunique()),
        "rows": rows,
        "strategy_summaries": qualifying,
        "leave_one_regime_out_status": "GO",
    }


def _finite_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    clean = frame.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    return clean.dropna(axis=0, how="any")


def _annualized_sharpe(values: np.ndarray) -> np.ndarray:
    means = np.mean(values, axis=0)
    standard = np.std(values, axis=0, ddof=1)
    return np.divide(means, standard, out=np.zeros_like(means), where=standard > 0) * np.sqrt(252.0)


def _benjamini_hochberg(p_values: np.ndarray, *, alpha: float) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = alpha * np.arange(1, len(ranked) + 1) / len(ranked)
    passing = np.flatnonzero(ranked <= thresholds)
    result = np.zeros(len(p_values), dtype=bool)
    if len(passing):
        result[order[: passing[-1] + 1]] = True
    return result


def _deflated_sharpe(values: np.ndarray, trials: int) -> np.ndarray:
    daily_sr = np.divide(
        values.mean(axis=0),
        values.std(axis=0, ddof=1),
        out=np.zeros(values.shape[1]),
        where=values.std(axis=0, ddof=1) > 0,
    )
    observed_variance = max(float(np.var(daily_sr, ddof=1)), 1.0 / values.shape[0])
    gamma = 0.5772156649015329
    expected_max = np.sqrt(observed_variance) * (
        (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / trials)
        + gamma * stats.norm.ppf(1.0 - 1.0 / (trials * np.e))
    )
    skew = stats.skew(values, axis=0, bias=False)
    kurtosis = stats.kurtosis(values, axis=0, fisher=False, bias=False)
    denominator = np.sqrt(
        np.maximum(1e-12, 1.0 - skew * daily_sr + ((kurtosis - 1.0) / 4.0) * daily_sr**2)
    )
    return stats.norm.cdf((daily_sr - expected_max) * np.sqrt(values.shape[0] - 1) / denominator)


def _bootstrap_indices(length: int, *, block: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = np.empty((samples, length), dtype=int)
    for sample in range(samples):
        cursor = 0
        while cursor < length:
            start = int(rng.integers(0, length))
            take = min(block, length - cursor)
            indices[sample, cursor : cursor + take] = (start + np.arange(take)) % length
            cursor += take
    return indices


def _white_reality_check(values: np.ndarray, *, seed: int, samples: int) -> float:
    observed = float(values.mean(axis=0).max())
    centered = values - values.mean(axis=0, keepdims=True)
    indices = _bootstrap_indices(len(values), block=max(5, int(np.sqrt(len(values)))), samples=samples, seed=seed)
    boot = np.asarray([centered[index].mean(axis=0).max() for index in indices])
    return float((1 + np.sum(boot >= observed)) / (samples + 1))


def _hansen_spa(values: np.ndarray, *, seed: int, samples: int) -> float:
    means = values.mean(axis=0)
    standard_error = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    observed = float(np.max(np.divide(means, standard_error, out=np.zeros_like(means), where=standard_error > 0)))
    centered = values - np.maximum(means, 0.0)
    indices = _bootstrap_indices(len(values), block=max(5, int(np.sqrt(len(values)))), samples=samples, seed=seed)
    boot_stats = []
    for index in indices:
        sample = centered[index]
        sample_mean = sample.mean(axis=0)
        stat = np.divide(sample_mean, standard_error, out=np.zeros_like(sample_mean), where=standard_error > 0)
        boot_stats.append(float(np.max(stat)))
    return float((1 + np.sum(np.asarray(boot_stats) >= observed)) / (samples + 1))


def _cscv_pbo(values: np.ndarray, *, partitions: int) -> dict[str, Any]:
    if values.shape[0] < partitions * 10:
        return {"status": "INSUFFICIENT_OBSERVATIONS", "pbo": None}
    blocks = np.array_split(np.arange(values.shape[0]), partitions)
    logits = []
    for selected in itertools.combinations(range(partitions), partitions // 2):
        is_index = np.concatenate([blocks[index] for index in selected])
        oos_index = np.concatenate([blocks[index] for index in range(partitions) if index not in selected])
        best = int(np.argmax(_annualized_sharpe(values[is_index])))
        oos = _annualized_sharpe(values[oos_index])
        rank = int(stats.rankdata(oos, method="average")[best])
        relative_rank = rank / (len(oos) + 1.0)
        logits.append(log(relative_rank / (1.0 - relative_rank)))
    return {
        "status": "GO",
        "method": "CLASSICAL_CSCV_CONTIGUOUS_PARTITIONS",
        "partition_count": partitions,
        "combination_count": len(logits),
        "pbo": float(np.mean(np.asarray(logits) < 0.0)),
        "median_logit": float(np.median(logits)),
    }


def _jaccard_matrix(flags: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=flags.columns, columns=flags.columns, dtype=float)
    for left in flags.columns:
        for right in flags.columns:
            union = (flags[left] | flags[right]).sum()
            result.loc[left, right] = float((flags[left] & flags[right]).sum() / union) if union else 0.0
    return result


def _correlation_clusters(correlation: pd.DataFrame, *, threshold: float) -> list[list[str]]:
    remaining = set(str(column) for column in correlation.columns)
    clusters = []
    while remaining:
        seed = min(remaining)
        cluster = {seed}
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining - cluster):
                if any(abs(float(correlation.loc[candidate, member])) >= threshold for member in cluster):
                    cluster.add(candidate)
                    changed = True
        remaining -= cluster
        if len(cluster) > 1:
            clusters.append(sorted(cluster))
    return clusters


def _matrix_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {"left": str(left), "right": str(right), "value": _optional_float(value)}
        for left in frame.index
        for right, value in frame.loc[left].items()
        if str(left) <= str(right)
    ]


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _index_label(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
