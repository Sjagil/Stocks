from __future__ import annotations

import hashlib
import json
import math
import msvcrt
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import exchange_calendars as xcals

from stocks.research.phase11_6 import nested_walk_forward_folds
from stocks.research.phase11_8 import _run_portfolio


SCHEMA = "phase11_9_accelerated_multitimeframe_discovery_v1"
TIMEFRAMES = ("1h", "2h", "4h", "1d", "1w", "1mo")
COSTS_BPS = (5.0, 10.0, 20.0, 30.0, 50.0)
PARAMETER_PROFILE_COUNT = 3
PARAMETER_GRID_VERSION = "COST_AWARE_SWING_V2"
INTRADAY_PROVIDER_PRIORITY = ("EODHD", "YFINANCE")
SYMBOLS = (
    "AAPL",
    "AMZN",
    "ASML",
    "DBC",
    "EEM",
    "EFA",
    "GLD",
    "GOOGL",
    "INTC",
    "IWM",
    "JPM",
    "META",
    "MSFT",
    "NVDA",
    "ON",
    "QQQ",
    "SLV",
    "SPY",
    "TLT",
    "XOM",
)
BENCHMARK_SYMBOLS = ("SPY", "IWM", "EEM", "TLT")
BASE_STRATEGIES = (
    "ma_crossover",
    "asymmetric_ma",
    "ma_channel",
    "triple_ma_trend",
    "donchian_breakout",
    "bollinger_breakout",
    "volatility_contraction_breakout",
    "atr_breakout",
    "range_expansion_breakout",
    "rsi2_adx_pullback",
    "rsi14_trend_pullback",
    "stochastic_trend_pullback",
    "macd_trend",
    "adx_trend",
    "keltner_breakout",
    "volume_breakout",
    "roc_trend",
    "risk_adjusted_momentum",
    "volatility_adjusted_trend",
    "ema_pullback",
    "etf_commodity_trend",
    "ppo_trend",
    "cci_trend_pullback",
    "mfi_trend_pullback",
    "cmf_accumulation",
    "obv_breakout",
    "aroon_trend",
    "vortex_trend",
    "choppiness_breakout",
    "efficiency_trend",
    "vwap_deviation_reversion",
    "nr7_breakout",
    "gap_recovery",
    "connors_rsi_pullback",
    "trend_quality_52w",
    "kama_adaptive_trend",
    "trix_trend",
    "tsi_trend",
    "ultimate_oscillator_pullback",
    "williams_r_pullback",
    "chaikin_oscillator_trend",
    "force_index_trend",
    "vpt_breakout",
    "linear_regression_trend",
    "parkinson_volatility_breakout",
    "yang_zhang_volatility_breakout",
    "variance_ratio_trend",
    "ulcer_low_risk_trend",
)
ENSEMBLES: dict[str, tuple[str, ...]] = {
    "trend_consensus": (
        "ma_crossover",
        "asymmetric_ma",
        "roc_trend",
    ),
    "breakout_consensus": (
        "donchian_breakout",
        "ma_channel",
        "bollinger_breakout",
        "volatility_contraction_breakout",
        "volume_breakout",
    ),
    "channel_consensus": (
        "ma_channel",
        "donchian_breakout",
        "keltner_breakout",
        "atr_breakout",
        "range_expansion_breakout",
    ),
    "trend_pullback_consensus": (
        "ma_crossover",
        "rsi2_adx_pullback",
        "rsi14_trend_pullback",
        "ema_pullback",
    ),
    "momentum_consensus": (
        "macd_trend",
        "roc_trend",
        "risk_adjusted_momentum",
        "keltner_breakout",
    ),
    "quality_trend_consensus": (
        "triple_ma_trend",
        "adx_trend",
        "risk_adjusted_momentum",
        "volatility_adjusted_trend",
    ),
    "pullback_consensus": (
        "rsi2_adx_pullback",
        "rsi14_trend_pullback",
        "stochastic_trend_pullback",
        "ema_pullback",
    ),
    "robust_trend_consensus": (
        "ma_crossover",
        "triple_ma_trend",
        "adx_trend",
        "risk_adjusted_momentum",
        "donchian_breakout",
    ),
    "diversified_consensus": (
        "ma_crossover",
        "donchian_breakout",
        "volatility_contraction_breakout",
        "rsi2_adx_pullback",
        "macd_trend",
        "ema_pullback",
    ),
    "flow_consensus": (
        "cmf_accumulation",
        "mfi_trend_pullback",
        "obv_breakout",
    ),
    "trend_strength_consensus": (
        "aroon_trend",
        "vortex_trend",
        "efficiency_trend",
        "trend_quality_52w",
    ),
    "structure_consensus": (
        "nr7_breakout",
        "gap_recovery",
        "choppiness_breakout",
    ),
    "adaptive_momentum_consensus": (
        "ppo_trend",
        "cci_trend_pullback",
        "connors_rsi_pullback",
        "vwap_deviation_reversion",
    ),
    "adaptive_trend_consensus": (
        "kama_adaptive_trend",
        "efficiency_trend",
        "linear_regression_trend",
        "variance_ratio_trend",
    ),
    "smoothed_momentum_consensus": (
        "trix_trend",
        "tsi_trend",
        "ppo_trend",
        "roc_trend",
    ),
    "accumulation_impulse_consensus": (
        "chaikin_oscillator_trend",
        "force_index_trend",
        "vpt_breakout",
        "cmf_accumulation",
    ),
    "volatility_estimator_consensus": (
        "parkinson_volatility_breakout",
        "yang_zhang_volatility_breakout",
        "volatility_contraction_breakout",
        "nr7_breakout",
    ),
    "oscillator_pullback_consensus": (
        "ultimate_oscillator_pullback",
        "williams_r_pullback",
        "cci_trend_pullback",
        "mfi_trend_pullback",
        "rsi14_trend_pullback",
    ),
}
AUTHORITY = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "BLOCKED_NEW_DISCOVERY",
    "STRATEGY_AUTHORITY": "NONE",
    "EXECUTION_AUTHORITY": "NONE",
    "PAPER_STRATEGY_AUTHORITY": "NONE",
    "LIVE_STRATEGY_AUTHORITY": "NONE",
    "BROKER_CALLS": 0,
    "ORDER_CALLS": 0,
}
_WORKER_FRAMES: Mapping[str, pd.DataFrame] = {}
_WORKER_TIMEFRAME = ""
_WORKER_FOLDS: list[dict[str, Any]] = []


def _single_flight(
    function: Callable[[Path], dict[str, Any]]
) -> Callable[[Path], dict[str, Any]]:
    @wraps(function)
    def guarded(project_root: Path) -> dict[str, Any]:
        lock_path = _output(project_root) / "run.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            handle.seek(0)
            if handle.read(1) == b"":
                handle.seek(0)
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            handle.close()
            return {
                "schema": SCHEMA,
                "status": "RUN_ALREADY_ACTIVE",
                "single_flight": True,
                **AUTHORITY,
            }
        try:
            return function(project_root)
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            handle.close()

    return guarded


def phase11_9_schema(project_root: Path) -> dict[str, Any]:
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "timeframes": list(TIMEFRAMES),
        "base_strategies": list(BASE_STRATEGIES),
        "ensembles": ENSEMBLES,
        "parameter_profiles_per_strategy_timeframe": (
            PARAMETER_PROFILE_COUNT
        ),
        "parameter_grid_version": PARAMETER_GRID_VERSION,
        "cost_stress_bps_per_side": list(COSTS_BPS),
        "portfolio_contract": {
            "whole_shares": True,
            "gross_exposure_maximum": 1.0,
            "global_security_netting": True,
            "selection": "PRIOR_CLOSED_BAR_SCORE_DESC_SECURITY_ID_ASC",
            "execution": "NEXT_BAR_OPEN",
            "base_currency": "EUR",
            "fx": "LAGGED_DAILY_EURUSD_WITH_1_BP_FILL_FRICTION",
            "primary_metric": "OOS_PORTFOLIO_PERIOD_PROFIT_FACTOR",
        },
        "confidence": {
            "1h": "EVALUABLE_MULTI_YEAR_NATIVE_HISTORY",
            "4h": "EVALUABLE_CLOSED_MULTI_YEAR_NATIVE_1H_AGGREGATION",
            "1d": "EVALUABLE_LONG_HISTORY",
            "1w": "EVALUABLE_CLOSED_DAILY_AGGREGATION",
            "1mo": "EVALUABLE_CLOSED_DAILY_AGGREGATION",
        },
        "multiple_testing": "GLOBAL_TRIAL_COUNT_REPORTED_NO_DECIMAL_OPTIMIZATION",
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "schema.json", report)
    return report


@_single_flight
def run_discovery(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    output.mkdir(parents=True, exist_ok=True)
    schema = phase11_9_schema(project_root)
    frames_by_timeframe = _load_frames(project_root)
    coverage = _coverage(frames_by_timeframe)
    strategies = (*BASE_STRATEGIES, *ENSEMBLES)
    run_signature = _hash(
        {
            "schema": schema,
            "strategies": strategies,
            "timeframes": TIMEFRAMES,
            "costs_bps": COSTS_BPS,
            "symbols": SYMBOLS,
            "data_fingerprint": _frames_fingerprint(
                frames_by_timeframe
            ),
        }
    )
    checkpoint = _load_discovery_checkpoint(output, run_signature)
    result_rows = checkpoint["result_rows"]
    selection_rows = checkpoint["selection_rows"]
    blocked = checkpoint["blocked"]
    completed_timeframes = set(checkpoint["completed_timeframes"])
    completed_strategy_timeframes = {
        tuple(item)
        for item in checkpoint["completed_strategy_timeframes"]
    }
    for timeframe in TIMEFRAMES:
        if timeframe in completed_timeframes:
            continue
        frames = frames_by_timeframe.get(timeframe, {})
        if len(frames) < 5:
            blocked.append(
                {
                    "timeframe": timeframe,
                    "reason": "INSUFFICIENT_REAL_INSTRUMENTS",
                    "instrument_count": len(frames),
                }
            )
            completed_timeframes.add(timeframe)
            _write_discovery_checkpoint(
                output,
                run_signature=run_signature,
                completed_timeframes=completed_timeframes,
                completed_strategy_timeframes=(
                    completed_strategy_timeframes
                ),
                result_rows=result_rows,
                selection_rows=selection_rows,
                blocked=blocked,
            )
            continue
        # Assets enter the investable universe when their own history starts.
        # A younger listing must not discard older point-in-time evidence.
        start = min(frame.index.min() for frame in frames.values())
        end = min(frame.index.max() for frame in frames.values())
        folds = nested_walk_forward_folds(start, end, timeframe)
        if folds.empty:
            blocked.append(
                {
                    "timeframe": timeframe,
                    "reason": "INSUFFICIENT_NESTED_WALK_FORWARD_FOLDS",
                    "instrument_count": len(frames),
                }
            )
            completed_timeframes.add(timeframe)
            _write_discovery_checkpoint(
                output,
                run_signature=run_signature,
                completed_timeframes=completed_timeframes,
                completed_strategy_timeframes=(
                    completed_strategy_timeframes
                ),
                result_rows=result_rows,
                selection_rows=selection_rows,
                blocked=blocked,
            )
            continue
        pending = [
            strategy
            for strategy in strategies
            if (timeframe, strategy)
            not in completed_strategy_timeframes
        ]
        fold_records = folds.to_dict("records")
        with ProcessPoolExecutor(
            max_workers=min(4, max(1, len(pending))),
            initializer=_initialize_strategy_worker,
            initargs=(str(project_root), timeframe, fold_records),
        ) as executor:
            evaluations = executor.map(
                _evaluate_strategy_worker,
                pending,
                chunksize=1,
            )
            for strategy, evaluation in zip(
                pending, evaluations, strict=True
            ):
                strategy_result_rows, strategy_selection_rows = evaluation
                strategy_timeframe = (timeframe, strategy)
                result_rows.extend(strategy_result_rows)
                selection_rows.extend(strategy_selection_rows)
                completed_strategy_timeframes.add(strategy_timeframe)
                _write_discovery_checkpoint(
                    output,
                    run_signature=run_signature,
                    completed_timeframes=completed_timeframes,
                    completed_strategy_timeframes=(
                        completed_strategy_timeframes
                    ),
                    result_rows=result_rows,
                    selection_rows=selection_rows,
                    blocked=blocked,
                )
        completed_timeframes.add(timeframe)
        _write_discovery_checkpoint(
            output,
            run_signature=run_signature,
            completed_timeframes=completed_timeframes,
            completed_strategy_timeframes=(
                completed_strategy_timeframes
            ),
            result_rows=result_rows,
            selection_rows=selection_rows,
            blocked=blocked,
        )
    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    summary = _summarize(results, selections)
    shortlist = _shortlist(summary)
    backtest_positive = _backtest_positive_registry(summary)
    cross_timeframe = _cross_timeframe(summary)
    _write_frame(output / "nested-results.parquet", results)
    _write_frame(output / "parameter-selections.csv", selections)
    _write_frame(output / "strategy-timeframe-summary.csv", summary)
    _write_frame(output / "cross-timeframe-stability.csv", cross_timeframe)
    _write_json(output / "shortlist.json", shortlist)
    _write_json(
        output / "backtest-positive-registry.json",
        backtest_positive,
    )
    _write_jsonl(output / "blocked.jsonl", blocked)
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "strategy_count": len(strategies),
        "base_strategy_count": len(BASE_STRATEGIES),
        "ensemble_count": len(ENSEMBLES),
        "strategy_timeframe_count": int(len(summary)),
        "global_trial_count": int(len(results)),
        "global_hypothesis_count": int(
            len(strategies) * len(TIMEFRAMES) * PARAMETER_PROFILE_COUNT
        ),
        "nested_selection_count": int(len(selections)),
        "coverage": coverage,
        "shortlist": shortlist,
        "backtest_positive": backtest_positive,
        "selection_bias_status": "BLOCKED_NO_NEW_INDEPENDENT_HOLDOUT",
        "historical_discovery_only": True,
        "schema_hash": _hash(schema),
        **AUTHORITY,
    }
    _write_json(output / "status.json", report)
    _write_json(output / "manifest.json", report)
    return report


def _evaluate_strategy_timeframe(
    frames: Mapping[str, pd.DataFrame],
    strategy: str,
    timeframe: str,
    fold_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    variants = _parameter_profiles(timeframe)
    signal_cache = {
        profile: _signals(frames, strategy, timeframe, profile)
        for profile in variants
    }
    for fold in fold_records:
        validation: list[tuple[float, float, str]] = []
        for profile in variants:
            run = _run_portfolio(
                frames,
                signal_cache[profile],
                start=pd.Timestamp(fold["validation_start"]),
                end=pd.Timestamp(fold["validation_end"]),
                cost_bps=10.0,
            )
            metrics = run["metrics"]
            validation.append(
                (
                    _finite(metrics["period_profit_factor"], -1.0),
                    _finite(metrics["CAGR"], -1.0),
                    profile,
                )
            )
        validation.sort(reverse=True)
        selected = validation[0]
        plateau = (
            len(validation) > 1
            and validation[1][0] > 1.0
            and selected[0] > 1.0
            and abs(selected[0] - validation[1][0])
            / max(abs(selected[0]), 1e-9)
            <= 0.20
        )
        selection_rows.append(
            {
                "fold_id": fold["fold_id"],
                "timeframe": timeframe,
                "strategy": strategy,
                "selected_profile": selected[2],
                "validation_portfolio_pf": selected[0],
                "validation_CAGR": selected[1],
                "parameter_plateau": plateau,
            }
        )
        for cost_bps in COSTS_BPS:
            run = _run_portfolio(
                frames,
                signal_cache[selected[2]],
                start=pd.Timestamp(fold["outer_test_start"]),
                end=pd.Timestamp(fold["outer_test_end"]),
                cost_bps=cost_bps,
            )
            fills = run["fills"]
            notional = (
                fills["shares"].mul(fills["price_eur"]).sum()
                if not fills.empty
                else 0.0
            )
            result_rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "timeframe": timeframe,
                    "strategy": strategy,
                    "profile": selected[2],
                    "cost_bps": cost_bps,
                    "parameter_plateau": plateau,
                    "fill_count": len(fills),
                    "turnover_initial_capital": float(
                        notional / 10_000.0
                    ),
                    "total_cost_eur": float(
                        fills["fee_eur"].sum()
                        if not fills.empty
                        else 0.0
                    ),
                    "maximum_gross_exposure": float(
                        run["ledger"]["gross_exposure"].max()
                        if not run["ledger"].empty
                        else 0.0
                    ),
                    **run["metrics"],
                }
            )
    return result_rows, selection_rows


def _initialize_strategy_worker(
    project_root: str,
    timeframe: str,
    fold_records: list[dict[str, Any]],
) -> None:
    global _WORKER_FRAMES, _WORKER_TIMEFRAME, _WORKER_FOLDS
    root = Path(project_root)
    _WORKER_FRAMES = (
        _load_intraday(root, timeframe)
        if timeframe in {"1h", "2h", "4h"}
        else _load_frames(root).get(timeframe, {})
    )
    _WORKER_TIMEFRAME = timeframe
    _WORKER_FOLDS = fold_records


def _evaluate_strategy_worker(
    strategy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _evaluate_strategy_timeframe(
        _WORKER_FRAMES,
        strategy,
        _WORKER_TIMEFRAME,
        _WORKER_FOLDS,
    )


def _load_discovery_checkpoint(
    output: Path, run_signature: str
) -> dict[str, Any]:
    metadata_path = output / "run-checkpoint.json"
    results_path = output / "run-checkpoint-results.parquet"
    selections_path = output / "run-checkpoint-selections.parquet"
    blocked_path = output / "run-checkpoint-blocked.json"
    empty: dict[str, Any] = {
        "result_rows": [],
        "selection_rows": [],
        "blocked": [],
        "completed_timeframes": [],
        "completed_strategy_timeframes": [],
    }
    if not all(
        path.exists()
        for path in (
            metadata_path,
            results_path,
            selections_path,
            blocked_path,
        )
    ):
        return empty
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_signature") != run_signature:
        return empty
    return {
        "result_rows": pd.read_parquet(results_path).to_dict("records"),
        "selection_rows": pd.read_parquet(selections_path).to_dict(
            "records"
        ),
        "blocked": json.loads(blocked_path.read_text(encoding="utf-8")),
        "completed_timeframes": list(
            metadata.get("completed_timeframes", [])
        ),
        "completed_strategy_timeframes": list(
            metadata.get("completed_strategy_timeframes", [])
        ),
    }


def _write_discovery_checkpoint(
    output: Path,
    *,
    run_signature: str,
    completed_timeframes: set[str],
    completed_strategy_timeframes: set[tuple[str, str]],
    result_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
) -> None:
    _write_frame(
        output / "run-checkpoint-results.parquet",
        pd.DataFrame(result_rows),
    )
    _write_frame(
        output / "run-checkpoint-selections.parquet",
        pd.DataFrame(selection_rows),
    )
    _write_json(
        output / "run-checkpoint-blocked.json",
        blocked,
    )
    _write_json(
        output / "run-checkpoint.json",
        {
            "schema": "phase11_9_discovery_checkpoint_v1",
            "run_signature": run_signature,
            "completed_timeframes": sorted(completed_timeframes),
            "completed_strategy_timeframes": sorted(
                [list(item) for item in completed_strategy_timeframes]
            ),
            "result_row_count": len(result_rows),
            "selection_row_count": len(selection_rows),
            "status": (
                "COMPLETE"
                if completed_timeframes.issuperset(TIMEFRAMES)
                else "IN_PROGRESS"
            ),
        },
    )


def phase11_9_status(project_root: Path) -> dict[str, Any]:
    path = _output(project_root) / "status.json"
    if not path.exists():
        return {"schema": SCHEMA, "status": "NOT_RUN", **AUTHORITY}
    return json.loads(path.read_text(encoding="utf-8"))


def run_diagnostics(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    results_path = output / "nested-results.parquet"
    selections_path = output / "parameter-selections.csv"
    summary_path = output / "strategy-timeframe-summary.csv"
    shortlist_path = output / "shortlist.json"
    if not all(
        path.exists()
        for path in (
            results_path,
            selections_path,
            summary_path,
            shortlist_path,
        )
    ):
        return {"schema": SCHEMA, "status": "DISCOVERY_NOT_RUN", **AUTHORITY}
    results = pd.read_parquet(results_path)
    selections = pd.read_csv(selections_path)
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    frames_by_timeframe = _load_frames(project_root)
    benchmark_rows = []
    for timeframe, frames in frames_by_timeframe.items():
        benchmark_frames = {
            symbol: frames[symbol]
            for symbol in BENCHMARK_SYMBOLS
            if symbol in frames
        }
        if len(benchmark_frames) != len(BENCHMARK_SYMBOLS):
            continue
        start = min(frame.index.min() for frame in frames.values())
        end = min(frame.index.max() for frame in frames.values())
        folds = nested_walk_forward_folds(start, end, timeframe)
        signals = {
            symbol: pd.DataFrame(
                {"signal": True, "score": 1.0}, index=frame.index
            )
            for symbol, frame in benchmark_frames.items()
        }
        for fold in folds.to_dict("records"):
            run = _run_portfolio(
                benchmark_frames,
                signals,
                start=pd.Timestamp(fold["outer_test_start"]),
                end=pd.Timestamp(fold["outer_test_end"]),
                cost_bps=10.0,
            )
            benchmark_rows.append(
                {
                    "fold_id": fold["fold_id"],
                    "timeframe": timeframe,
                    **{
                        f"benchmark_{key}": value
                        for key, value in run["metrics"].items()
                    },
                }
            )
    benchmarks = pd.DataFrame(benchmark_rows)
    normal = results.loc[results["cost_bps"].eq(10.0)].copy()
    comparison = normal.merge(
        benchmarks, on=["fold_id", "timeframe"], how="left"
    )
    comparison["excess_CAGR"] = (
        comparison["CAGR"] - comparison["benchmark_CAGR"]
    )
    comparison["excess_Sharpe"] = (
        comparison["Sharpe"] - comparison["benchmark_Sharpe"]
    )
    _write_frame(output / "benchmark-comparison.csv", comparison)
    formulas = _formula_specifications(shortlist, selections)
    _write_json(output / "formula-specifications.json", formulas)
    diagnostics = []
    for candidate in shortlist.get("candidates", []):
        group = comparison.loc[
            comparison["strategy"].eq(candidate["strategy"])
            & comparison["timeframe"].eq(candidate["timeframe"])
        ]
        bootstrap = _fold_bootstrap(group, seed=_seed(candidate))
        diagnostics.append(
            {
                "strategy": candidate["strategy"],
                "timeframe": candidate["timeframe"],
                "fold_count": len(group),
                "median_excess_CAGR": float(group["excess_CAGR"].median()),
                "positive_excess_CAGR_ratio": float(
                    group["excess_CAGR"].gt(0).mean()
                ),
                "median_excess_Sharpe": float(
                    group["excess_Sharpe"].median()
                ),
                "bootstrap": bootstrap,
                "benchmark_incremental_gate": (
                    "GO"
                    if group["excess_CAGR"].median() > 0
                    and group["excess_CAGR"].gt(0).mean() >= 0.60
                    and bootstrap["probability_median_pf_above_one"] >= 0.90
                    else "NO_GO"
                ),
            }
        )
    passed = [
        row
        for row in diagnostics
        if row["benchmark_incremental_gate"] == "GO"
    ]
    duplicate_audit = _candidate_duplicate_audit(shortlist, results)
    _write_json(
        output / "candidate-duplicate-dna-audit.json",
        duplicate_audit,
    )
    ten_strategy_audit = _ten_strategy_pass_audit(
        shortlist,
        diagnostics,
        duplicate_audit,
    )
    _write_json(
        output / "ten-strategy-pass-audit.json",
        ten_strategy_audit,
    )
    report = {
        "schema": "phase11_9_robustness_diagnostics_v1",
        "status": "GO",
        "benchmark": {
            "id": "BALANCED_4_ASSET_BUY_AND_HOLD",
            "symbols": list(BENCHMARK_SYMBOLS),
            "rebalance": "INITIAL_WHOLE_SHARE_EQUAL_ALLOCATION",
            "cost_bps": 10.0,
        },
        "candidate_diagnostics": diagnostics,
        "benchmark_incremental_candidate_count": len(passed),
        "independent_shortlist_candidate_count": duplicate_audit[
            "independent_candidate_count"
        ],
        "duplicate_dna_audit": duplicate_audit,
        "ten_strategy_pass_audit": ten_strategy_audit,
        "best_available_candidate": passed[0] if passed else None,
        "execution_run_count": len(results),
        "global_hypothesis_count": (
            len(BASE_STRATEGIES) + len(ENSEMBLES)
        )
        * len(TIMEFRAMES)
        * PARAMETER_PROFILE_COUNT,
        "multiple_testing_status": (
            f"BLOCKED_{len(BASE_STRATEGIES) + len(ENSEMBLES)}"
            f"X{len(TIMEFRAMES)}X{PARAMETER_PROFILE_COUNT}"
            "_GLOBAL_HYPOTHESES_NO_INDEPENDENT_CONFIRMATION"
        ),
        "formula_specifications": formulas,
        **AUTHORITY,
    }
    _write_json(output / "robustness-diagnostics.json", report)
    if report["best_available_candidate"] is not None:
        best = {
            "schema": "phase11_9_best_available_research_candidate_v1",
            "status": "PROMISING_ACCELERATED_RESEARCH_CANDIDATE",
            "candidate": report["best_available_candidate"],
            "formula": next(
                row
                for row in formulas["formulas"]
                if row["strategy"]
                == report["best_available_candidate"]["strategy"]
                and row["timeframe"]
                == report["best_available_candidate"]["timeframe"]
            ),
            "selection_bias_status": report["multiple_testing_status"],
            **AUTHORITY,
        }
        best["content_hash"] = _hash(best)
        _write_json(output / "best-available-research-candidate.json", best)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["robustness_diagnostics"] = {
        "status": report["status"],
        "benchmark_incremental_candidate_count": len(passed),
        "best_available_candidate": report["best_available_candidate"],
        "multiple_testing_status": report["multiple_testing_status"],
    }
    status["execution_run_count"] = len(results)
    status["global_hypothesis_count"] = report["global_hypothesis_count"]
    _write_json(status_path, status)
    _write_json(output / "manifest.json", status)
    return report


def _ten_strategy_pass_audit(
    shortlist: Mapping[str, Any],
    diagnostics: list[dict[str, Any]],
    duplicate_audit: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = list(shortlist.get("candidates", []))
    benchmark_status = {
        (str(row["strategy"]), str(row["timeframe"])): str(
            row["benchmark_incremental_gate"]
        )
        for row in diagnostics
    }
    rows = []
    for candidate in candidates:
        strategy = str(candidate["strategy"])
        timeframe = str(candidate["timeframe"])
        gate_checks = {
            "fold_count_at_least_10": int(candidate["fold_count"]) >= 10,
            "median_portfolio_pf_above_1": (
                float(candidate["median_oos_portfolio_pf"]) > 1.0
            ),
            "median_cagr_positive": float(candidate["median_oos_CAGR"]) > 0,
            "pf_at_50bps_above_1": (
                float(candidate["cost_50bps_median_pf"]) > 1.0
            ),
            "median_fills_at_least_10": (
                float(candidate["median_fill_count"]) >= 10
            ),
            "worst_drawdown_above_minus_50pct": (
                float(candidate["worst_oos_drawdown"]) > -0.50
            ),
            "evaluable_confidence": candidate["confidence"] == "EVALUABLE",
        }
        rows.append(
            {
                "strategy": strategy,
                "timeframe": timeframe,
                "portfolio_gate": (
                    "GO" if all(gate_checks.values()) else "NO_GO"
                ),
                "benchmark_incremental_gate": benchmark_status.get(
                    (strategy, timeframe), "NOT_EVALUATED"
                ),
                "checks": gate_checks,
            }
        )
    portfolio_pass_count = sum(
        row["portfolio_gate"] == "GO" for row in rows
    )
    benchmark_pass_count = sum(
        row["benchmark_incremental_gate"] == "GO" for row in rows
    )
    minimum_required = 10
    status = (
        "MINIMUM_TEN_BACKTESTED_STRATEGIES_GO"
        if portfolio_pass_count >= minimum_required
        else "MINIMUM_TEN_BACKTESTED_STRATEGIES_NO_GO"
    )
    return {
        "schema": "phase11_9_ten_strategy_pass_audit_v1",
        "status": status,
        "minimum_required": minimum_required,
        "portfolio_gate_pass_count": portfolio_pass_count,
        "independent_economic_outcome_count": int(
            duplicate_audit.get("independent_candidate_count", 0)
        ),
        "benchmark_incremental_pass_count": benchmark_pass_count,
        "candidates": rows,
        "interpretation": (
            "Portfolio-gate passes are backtest-positive research results; "
            "duplicate economic outcomes and benchmark gates remain separate."
        ),
        **AUTHORITY,
    }


def _candidate_duplicate_audit(
    shortlist: Mapping[str, Any], results: pd.DataFrame
) -> dict[str, Any]:
    fingerprints: dict[str, list[dict[str, str]]] = {}
    metric_columns = [
        "fold_id",
        "cost_bps",
        "period_profit_factor",
        "CAGR",
        "Sharpe",
        "maximum_drawdown",
        "fill_count",
        "turnover_initial_capital",
    ]
    for candidate in shortlist.get("candidates", []):
        strategy = str(candidate["strategy"])
        timeframe = str(candidate["timeframe"])
        group = results.loc[
            results["strategy"].eq(strategy)
            & results["timeframe"].eq(timeframe),
            metric_columns,
        ].sort_values(["fold_id", "cost_bps"])
        normalized = group.copy()
        for column in normalized.select_dtypes(include=[np.number]):
            normalized[column] = normalized[column].round(12)
        fingerprint = _hash(normalized.to_dict("records"))
        fingerprints.setdefault(fingerprint, []).append(
            {"strategy": strategy, "timeframe": timeframe}
        )
    groups = [
        {
            "fingerprint": fingerprint,
            "members": members,
            "classification": (
                "DUPLICATE_ECONOMIC_OUTCOME"
                if len(members) > 1
                else "UNIQUE_ECONOMIC_OUTCOME"
            ),
        }
        for fingerprint, members in sorted(fingerprints.items())
    ]
    return {
        "schema": "phase11_9_candidate_duplicate_dna_audit_v1",
        "status": "GO",
        "shortlist_candidate_count": sum(
            len(group["members"]) for group in groups
        ),
        "independent_candidate_count": len(groups),
        "duplicate_group_count": sum(
            len(group["members"]) > 1 for group in groups
        ),
        "groups": groups,
    }


def current_watchlist(project_root: Path) -> dict[str, Any]:
    frames = _load_frames(project_root)["1w"]
    rows = []
    for symbol, frame in frames.items():
        if len(frame) < 21:
            continue
        state = _strategy(
            frame, "donchian_breakout", "1w", "conservative"
        )
        latest = frame.iloc[-1]
        prior_high = frame["high"].rolling(20).max().shift(1).iloc[-1]
        exit_mean = frame["close"].rolling(20).mean().iloc[-1]
        active = bool(state["signal"].iloc[-1])
        previous = bool(state["signal"].iloc[-2])
        rows.append(
            {
                "symbol": symbol,
                "closed_bar_timestamp": frame.index[-1].isoformat(),
                "close_eur": float(latest["close"]),
                "prior_20_week_high_eur": float(prior_high),
                "exit_20_week_mean_eur": float(exit_mean),
                "momentum_20_week": float(
                    frame["close"].pct_change(20).iloc[-1]
                ),
                "active_signal": active,
                "fresh_entry_signal": active and not previous,
                "fresh_exit_signal": previous and not active,
            }
        )
    active_rows = sorted(
        (row for row in rows if row["active_signal"]),
        key=lambda row: (-row["momentum_20_week"], row["symbol"]),
    )
    report = {
        "schema": "phase11_9_donchian_weekly_watchlist_v1",
        "status": "GO",
        "strategy": "DONCHIAN_20_WEEK_BREAKOUT",
        "information_cutoff": max(
            row["closed_bar_timestamp"] for row in rows
        )
        if rows
        else None,
        "formula": {
            "entry": "WEEKLY_CLOSE_GT_PRIOR_20_WEEK_HIGH",
            "exit": "WEEKLY_CLOSE_LT_CURRENT_20_WEEK_MEAN",
            "ranking": "DESCENDING_20_WEEK_MOMENTUM",
            "execution_assumption": "NEXT_WEEKLY_BAR_OPEN",
        },
        "active_signal_count": len(active_rows),
        "top_four_observation_candidates": active_rows[:4],
        "all_instruments": rows,
        "phase9_route": "BLOCKED",
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "current-watchlist.json", report)
    return report


def _load_frames(
    project_root: Path,
) -> dict[str, dict[str, pd.DataFrame]]:
    daily = _load_daily(project_root)
    intraday = {
        interval: _load_intraday(
            project_root,
            interval,
            daily_reference=daily,
        )
        for interval in ("1h", "2h", "4h")
    }
    return {
        **intraday,
        "1d": daily,
        "1w": _aggregate(daily, "W-FRI"),
        "1mo": _aggregate(daily, "ME"),
    }


def _load_current_frames(
    project_root: Path,
    *,
    observed_at: datetime | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    daily = _load_current_daily(
        project_root,
        observed_at=observed_at,
    )
    frames = {
        "1d": daily,
        "1w": _aggregate(daily, "W-FRI"),
        "1mo": _aggregate(daily, "ME"),
    }
    for interval in ("1h", "2h", "4h"):
        frames[interval] = _load_intraday(
            project_root,
            interval,
            daily_reference=daily,
            selection_policy="FRESHEST_QUALIFIED_PROVIDER_NO_BLEND",
            observed_at=observed_at,
        )
    return frames


def _load_current_daily(
    project_root: Path,
    *,
    observed_at: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Append a validated current-provider tail without replacing history."""
    base = _load_daily(project_root)
    private_root = (
        project_root
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
    )
    observed = observed_at or datetime.now(UTC)
    merged = dict(base)
    for symbol in SYMBOLS:
        path = (
            private_root
            / "provider=YFINANCE"
            / f"symbol={symbol}"
            / "interval=1d"
            / "source_interval=1d"
            / "bars.parquet"
        )
        existing = merged.get(symbol)
        if existing is None or not path.exists():
            continue
        raw = _closed_daily_provider_frame(
            pd.read_parquet(path),
            observed_at=observed,
        )
        if raw.empty:
            continue
        raw["date"] = pd.to_datetime(
            raw["timestamp_utc"],
            utc=True,
        ).dt.tz_convert(None)
        candidate = _normalize(raw)
        converted = _convert_usd_to_eur_current(
            project_root,
            {symbol: candidate},
            observed_at=observed,
        ).get(symbol)
        if converted is None:
            continue
        selected = _append_daily_provider_tail(
            existing,
            converted,
            provider_name="YFINANCE",
        )
        for key in (
            "fx_source",
            "fx_latest_available_date",
            "fx_maximum_fill_days",
            "fx_maximum_observed_fill_age",
        ):
            if key in converted.attrs:
                selected.attrs[key] = converted.attrs[key]
        selected.attrs["provider_selection_policy"] = (
            "CURRENT_TAIL_APPEND_NO_HISTORICAL_REPLACEMENT"
        )
        selected.attrs["provider_candidates"] = [
            {
                "provider": str(
                    existing.attrs.get("forward_overlay_provider", "LOCAL")
                ),
                "last_timestamp": pd.Timestamp(
                    existing.index.max()
                ).isoformat(),
            },
            {
                "provider": "YFINANCE",
                "last_timestamp": pd.Timestamp(
                    converted.index.max()
                ).isoformat(),
            },
        ]
        merged[symbol] = selected
    return merged


def _closed_daily_provider_frame(
    frame: pd.DataFrame,
    *,
    observed_at: datetime,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    observed = pd.Timestamp(observed_at)
    observed = (
        observed.tz_localize("UTC")
        if observed.tzinfo is None
        else observed.tz_convert("UTC")
    )
    calendar = xcals.get_calendar("XNYS")
    latest_session = calendar.minute_to_session(
        observed.floor("min"),
        direction="previous",
    )
    if calendar.session_close(latest_session) > observed:
        latest_session = calendar.previous_session(latest_session)
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    session_timestamp = pd.Timestamp(latest_session)
    session_timestamp = (
        session_timestamp.tz_localize("UTC")
        if session_timestamp.tzinfo is None
        else session_timestamp.tz_convert("UTC")
    )
    closed = timestamps.dt.normalize().le(session_timestamp.normalize())
    return frame.loc[closed].copy()


def _load_intraday(
    project_root: Path,
    interval: str,
    *,
    daily_reference: Mapping[str, pd.DataFrame] | None = None,
    selection_policy: str = "HISTORICAL_PROVIDER_PRIORITY_NO_BLEND",
    observed_at: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    private_root = (
        project_root
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
    )
    frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        candidates: list[tuple[str, Path, pd.DataFrame]] = []
        for provider in INTRADAY_PROVIDER_PRIORITY:
            path = (
                private_root
                / f"provider={provider}"
                / f"symbol={symbol}"
                / f"interval={interval}"
                / "source_interval=1h"
                / "bars.parquet"
            )
            if path.exists():
                raw = pd.read_parquet(path)
                if (
                    selection_policy
                    == "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
                    and provider == "YFINANCE"
                ):
                    raw = _closed_yfinance_frame(
                        raw,
                        interval=interval,
                        observed_at=observed_at,
                    )
                if raw.empty:
                    continue
                candidates.append((provider, path, raw))
                if (
                    selection_policy
                    == "HISTORICAL_PROVIDER_PRIORITY_NO_BLEND"
                ):
                    break
        if not candidates:
            continue
        provider, _path, frame = (
            max(
                candidates,
                key=lambda item: pd.to_datetime(
                    item[2]["timestamp_utc"],
                    utc=True,
                ).max(),
            )
            if selection_policy
            == "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
            else candidates[0]
        )
        # A US regular session contains one four-hour block plus a shorter
        # closing block. Both are closed historical bars; is_partial records
        # source-bar count and does not mean the bar is still forming.
        frame["date"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
        frame["date"] = frame["date"].dt.tz_convert(None)
        normalized = _normalize(frame)
        normalized.attrs["provider"] = provider
        normalized.attrs["bar_origin"] = str(
            frame.get("bar_origin", pd.Series(["UNKNOWN"])).iloc[0]
        )
        normalized.attrs["content_fingerprint"] = _intraday_fingerprint(
            frame
        )
        normalized.attrs["provider_selection_policy"] = selection_policy
        normalized.attrs["provider_candidates"] = [
            {
                "provider": candidate_provider,
                "last_timestamp": pd.to_datetime(
                    candidate_frame["timestamp_utc"],
                    utc=True,
                ).max().isoformat(),
            }
            for candidate_provider, _candidate_path, candidate_frame in candidates
        ]
        normalized.attrs["selected_provider_last_timestamp"] = (
            pd.Timestamp(normalized.index.max()).isoformat()
        )
        frames[symbol] = normalized
    converted = (
        _convert_usd_to_eur_current(
            project_root,
            frames,
            observed_at=observed_at,
        )
        if selection_policy == "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
        else _convert_usd_to_eur(project_root, frames)
    )
    return _split_adjust_intraday(
        converted,
        daily_reference or {},
    )


def _closed_yfinance_frame(
    frame: pd.DataFrame,
    *,
    interval: str,
    observed_at: datetime | None,
) -> pd.DataFrame:
    hours = {"1h": 1, "2h": 2, "4h": 4}.get(interval)
    if frame.empty or hours is None:
        return frame
    observed = pd.Timestamp(observed_at or datetime.now(UTC))
    if observed.tzinfo is None:
        observed = observed.tz_localize("UTC")
    else:
        observed = observed.tz_convert("UTC")
    timestamps = pd.to_datetime(frame["timestamp_utc"], utc=True)
    return frame.loc[
        timestamps.add(pd.Timedelta(hours=hours)).le(observed)
    ].copy()


def _split_adjust_intraday(
    frames: Mapping[str, pd.DataFrame],
    daily_reference: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    adjusted: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        reference = daily_reference.get(symbol)
        work = frame.copy()
        split_events: list[dict[str, Any]] = []
        if reference is not None and not reference.empty:
            raw_daily_close = (
                frame["close"]
                .groupby(pd.DatetimeIndex(frame.index).normalize())
                .last()
            )
            reference_close = reference["close"].copy()
            reference_close.index = pd.DatetimeIndex(
                reference_close.index
            ).normalize()
            common = raw_daily_close.index.intersection(
                reference_close.index
            )
            if len(common) >= 2:
                ratio = raw_daily_close.reindex(common).div(
                    reference_close.reindex(common).replace(0, np.nan)
                )
                ratio_change = ratio.div(ratio.shift(1))
                for effective_date, observed_factor in ratio_change.items():
                    split_factor = _snap_split_factor(observed_factor)
                    if split_factor is None:
                        continue
                    prior = pd.DatetimeIndex(work.index).normalize() < (
                        pd.Timestamp(effective_date).normalize()
                    )
                    for column in ("open", "high", "low", "close"):
                        work.loc[prior, column] = (
                            work.loc[prior, column] * split_factor
                        )
                    work.loc[prior, "volume"] = (
                        work.loc[prior, "volume"] / split_factor
                    )
                    split_events.append(
                        {
                            "effective_date": pd.Timestamp(
                                effective_date
                            ).date().isoformat(),
                            "historical_price_multiplier": split_factor,
                            "historical_volume_multiplier": (
                                1.0 / split_factor
                            ),
                        }
                    )
        work.attrs.update(frame.attrs)
        work.attrs["price_basis"] = (
            "SPLIT_ADJUSTED_INTRADAY"
            if split_events
            else "NO_SPLIT_REGIME_DETECTED"
        )
        work.attrs["split_adjustment_events"] = split_events
        work.attrs["split_adjustment_event_count"] = len(split_events)
        adjusted[symbol] = work
    return adjusted


def _snap_split_factor(value: Any) -> float | None:
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(observed) or observed <= 0:
        return None
    candidates = (
        0.05,
        0.10,
        0.20,
        0.25,
        1 / 3,
        0.50,
        2 / 3,
        1.50,
        2.0,
        3.0,
        4.0,
        5.0,
        10.0,
        20.0,
    )
    nearest = min(candidates, key=lambda candidate: abs(observed - candidate))
    relative_error = abs(observed - nearest) / nearest
    return nearest if relative_error <= 0.08 else None


def _load_daily(project_root: Path) -> dict[str, pd.DataFrame]:
    root = (
        project_root
        / "data"
        / "research"
        / "critical_trading"
        / "yfinance"
    )
    frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        path = root / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path).rename(
            columns={"session_date": "date"}
        )
        normalized = _normalize(frame)
        overlay_path = (
            project_root
            / "data"
            / "research"
            / "multitimeframe"
            / "private"
            / "provider=EODHD"
            / f"symbol={symbol}"
            / "interval=1d"
            / "source_interval=1d"
            / "bars.parquet"
        )
        if overlay_path.exists():
            overlay = pd.read_parquet(overlay_path)
            overlay["date"] = pd.to_datetime(
                overlay["timestamp_utc"],
                utc=True,
            ).dt.tz_convert(None)
            normalized = _append_daily_provider_tail(
                normalized,
                _normalize(overlay),
                provider_name="EODHD",
            )
        frames[symbol] = normalized
    return _convert_usd_to_eur(project_root, frames)


def _append_daily_provider_tail(
    base: pd.DataFrame,
    provider_frame: pd.DataFrame,
    *,
    provider_name: str,
    maximum_close_difference: float = 0.03,
) -> pd.DataFrame:
    if base.empty or provider_frame.empty:
        return base
    common = base.index.intersection(provider_frame.index)
    if len(common) < 3:
        return base
    overlap = common[-min(10, len(common)) :]
    denominator = base.loc[overlap, "close"].abs().replace(0, np.nan)
    relative_difference = (
        provider_frame.loc[overlap, "close"]
        .sub(base.loc[overlap, "close"])
        .abs()
        .div(denominator)
    )
    if (
        relative_difference.isna().any()
        or relative_difference.max() > maximum_close_difference
    ):
        return base
    tail = provider_frame.loc[
        provider_frame.index > base.index.max()
    ].copy()
    if tail.empty:
        return base
    valid = (
        tail[["open", "high", "low", "close"]].gt(0).all(axis=1)
        & tail["volume"].ge(0)
        & tail["high"].ge(tail[["open", "close", "low"]].max(axis=1))
        & tail["low"].le(tail[["open", "close", "high"]].min(axis=1))
    )
    tail = tail.loc[valid]
    if tail.empty:
        return base
    combined = pd.concat([base, tail]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="first")]
    combined.attrs.update(base.attrs)
    combined.attrs["forward_overlay_provider"] = provider_name
    combined.attrs["forward_overlay_rows"] = len(tail)
    combined.attrs["provider_overlap_maximum_close_difference"] = float(
        relative_difference.max()
    )
    combined.attrs["historical_rows_replaced"] = 0
    return combined


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.tz_localize(None)
    work = work.sort_values("date").drop_duplicates("date").set_index("date")
    return work[["open", "high", "low", "close", "volume"]].astype(float)


def _convert_usd_to_eur(
    project_root: Path, frames: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    fx_path = (
        project_root
        / "data"
        / "research"
        / "phase11_4"
        / "private"
        / "eurusd.parquet"
    )
    fx = pd.read_parquet(fx_path)
    fx["date"] = pd.to_datetime(fx["date"])
    fx = fx.sort_values("date").set_index("date")["usd_per_eur"]
    eur_per_usd = (1.0 / fx).shift(1)
    converted: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        session_dates = pd.DatetimeIndex(frame.index).normalize()
        aligned = eur_per_usd.reindex(session_dates, method="ffill")
        aligned.index = frame.index
        if aligned.isna().any():
            continue
        work = frame.copy()
        for column in ("open", "high", "low", "close"):
            work[column] = work[column].mul(aligned)
        converted[symbol] = work
    return converted


def _convert_usd_to_eur_current(
    project_root: Path,
    frames: Mapping[str, pd.DataFrame],
    *,
    observed_at: datetime | None,
    maximum_fill_days: int = 5,
) -> dict[str, pd.DataFrame]:
    fx_path = project_root / "data" / "fx" / "fx_daily.parquet"
    if not fx_path.exists():
        return {}
    observed = pd.Timestamp(observed_at or datetime.now(UTC))
    observed = (
        observed.tz_localize("UTC")
        if observed.tzinfo is None
        else observed.tz_convert("UTC")
    )
    fx = pd.read_parquet(fx_path)
    required = {
        "session_date",
        "base_currency",
        "quote_currency",
        "rate",
        "available_at",
        "forward_fill_age",
    }
    if not required.issubset(fx.columns):
        return {}
    fx = fx.loc[
        fx["base_currency"].astype(str).eq("USD")
        & fx["quote_currency"].astype(str).eq("EUR")
    ].copy()
    fx["available_at"] = pd.to_datetime(
        fx["available_at"],
        utc=True,
        errors="coerce",
    )
    fx["source_date"] = pd.to_datetime(
        fx["session_date"],
        errors="coerce",
    ).dt.normalize()
    fx["rate"] = pd.to_numeric(fx["rate"], errors="coerce")
    fx["forward_fill_age"] = pd.to_numeric(
        fx["forward_fill_age"],
        errors="coerce",
    )
    fx = fx.loc[
        fx["available_at"].le(observed)
        & fx["rate"].gt(0)
        & fx["forward_fill_age"].le(maximum_fill_days)
    ].dropna(subset=["source_date", "rate"])
    fx = fx.sort_values("source_date").drop_duplicates(
        "source_date",
        keep="last",
    )
    if fx.empty:
        return {}

    converted: dict[str, pd.DataFrame] = {}
    source = fx.set_index("source_date")["rate"]
    source_dates = pd.Series(source.index, index=source.index)
    for symbol, frame in frames.items():
        session_dates = pd.DatetimeIndex(frame.index).normalize()
        aligned = source.reindex(session_dates, method="ffill")
        aligned_dates = source_dates.reindex(session_dates, method="ffill")
        aligned.index = frame.index
        aligned_dates.index = frame.index
        fill_age = (
            pd.Series(session_dates, index=frame.index)
            - pd.to_datetime(aligned_dates)
        ).dt.days
        if (
            aligned.isna().any()
            or fill_age.isna().any()
            or fill_age.gt(maximum_fill_days).any()
        ):
            continue
        work = frame.copy()
        for column in ("open", "high", "low", "close"):
            work[column] = work[column].mul(aligned)
        work.attrs.update(frame.attrs)
        work.attrs["fx_source"] = "CANONICAL_PIT_FX"
        work.attrs["fx_latest_available_date"] = pd.Timestamp(
            fx["source_date"].max()
        ).date().isoformat()
        work.attrs["fx_maximum_fill_days"] = maximum_fill_days
        work.attrs["fx_maximum_observed_fill_age"] = int(fill_age.max())
        converted[symbol] = work
    return converted


def _aggregate(
    frames: Mapping[str, pd.DataFrame], frequency: str
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        aggregate = frame.resample(
            frequency, label="right", closed="right"
        ).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        aggregate = aggregate.dropna(subset=["open", "high", "low", "close"])
        source_cutoff = pd.Timestamp(frame.index.max()).normalize()
        aggregate = aggregate.loc[aggregate.index <= source_cutoff]
        if len(aggregate) > 20:
            output[symbol] = aggregate
    return output


def _parameter_profiles(timeframe: str) -> tuple[str, str, str]:
    del timeframe
    return ("responsive", "balanced", "conservative")


def _parameters(timeframe: str, profile: str) -> dict[str, float]:
    table = {
        "1h": ((50, 200, 55), (75, 300, 80), (100, 400, 120)),
        "2h": ((36, 150, 35), (50, 200, 55), (75, 300, 80)),
        "4h": ((24, 100, 20), (36, 150, 35), (50, 200, 55)),
        "1d": ((50, 200, 20), (60, 215, 35), (70, 230, 55)),
        "1w": ((10, 40, 10), (15, 46, 15), (20, 52, 20)),
        "1mo": ((3, 10, 3), (4, 11, 4), (6, 12, 6)),
    }
    profile_index = {
        "responsive": 0,
        "balanced": 1,
        "conservative": 2,
    }[profile]
    fast, slow, channel = table[timeframe][profile_index]
    profile_values = {
        "responsive": {
            "rsi_low": 12,
            "rsi14_low": 35,
            "adx_min": 18,
            "sigma": 2.0,
            "atr_mult": 1.5,
            "volume_mult": 1.25,
            "roc_threshold": 0.0,
            "volatility_ratio": 0.75,
        },
        "balanced": {
            "rsi_low": 15,
            "rsi14_low": 39,
            "adx_min": 21,
            "sigma": 2.25,
            "atr_mult": 1.75,
            "volume_mult": 1.50,
            "roc_threshold": 0.01,
            "volatility_ratio": 0.65,
        },
        "conservative": {
            "rsi_low": 18,
            "rsi14_low": 42,
            "adx_min": 23,
            "sigma": 2.5,
            "atr_mult": 2.0,
            "volume_mult": 1.75,
            "roc_threshold": 0.02,
            "volatility_ratio": 0.55,
        },
    }[profile]
    return {
        "fast": fast,
        "slow": slow,
        "channel": channel,
        **profile_values,
    }


def _signals(
    frames: Mapping[str, pd.DataFrame],
    strategy: str,
    timeframe: str,
    profile: str,
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        work = frame.copy(deep=False)
        work.attrs["symbol"] = symbol
        output[symbol] = _strategy(
            work, strategy, timeframe, profile
        )
    return output


def _strategy(
    frame: pd.DataFrame,
    strategy: str,
    timeframe: str,
    profile: str,
) -> pd.DataFrame:
    parameters = _parameters(timeframe, profile)
    if strategy in ENSEMBLES:
        components = [
            _strategy(frame, component, timeframe, profile)
            for component in ENSEMBLES[strategy]
        ]
        votes = pd.concat(
            [component["signal"] for component in components], axis=1
        ).sum(axis=1)
        threshold = math.ceil(len(components) / 2)
        signal = votes.ge(threshold)
        score_table = pd.concat(
            [component["score"] for component in components], axis=1
        ).replace(-math.inf, np.nan)
        valid_count = score_table.notna().sum(axis=1).replace(0, np.nan)
        score = score_table.sum(axis=1, min_count=1).div(valid_count)
        return _signal_frame(signal, score)
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"]
    fast = int(parameters["fast"])
    slow = int(parameters["slow"])
    channel = int(parameters["channel"])
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    atr = _atr(frame, max(3, min(14, channel)))
    momentum = close.pct_change(max(2, channel))
    if strategy == "ma_crossover":
        signal = fast_ma.gt(slow_ma)
    elif strategy == "asymmetric_ma":
        entry = fast_ma.gt(slow_ma)
        exit_fast = close.rolling(max(fast + 2, int(fast * 1.5))).mean()
        exit_slow = close.rolling(max(slow + 2, int(slow * 1.25))).mean()
        signal = _persistent(entry, exit_fast.lt(exit_slow))
    elif strategy == "ma_channel":
        channel_high = high.rolling(channel).max().shift(1)
        channel_mean = close.rolling(channel).mean()
        signal = _persistent(close.gt(channel_high), close.lt(channel_mean))
    elif strategy == "triple_ma_trend":
        medium = close.rolling(max(fast + 1, (fast + slow) // 2)).mean()
        signal = fast_ma.gt(medium) & medium.gt(slow_ma)
    elif strategy == "donchian_breakout":
        entry = close.gt(high.rolling(channel).max().shift(1))
        signal = _persistent(entry, close.lt(close.rolling(channel).mean()))
    elif strategy == "bollinger_breakout":
        mean = close.rolling(channel).mean()
        upper = mean + parameters["sigma"] * close.rolling(channel).std()
        signal = _persistent(close.gt(upper), close.lt(mean))
    elif strategy == "volatility_contraction_breakout":
        returns = close.pct_change()
        contraction = returns.rolling(max(3, channel // 2)).std().lt(
            returns.rolling(max(6, channel * 2)).std().mul(0.7)
        )
        breakout = close.gt(high.rolling(channel).max().shift(1))
        signal = _persistent(
            contraction & breakout, close.lt(close.rolling(channel).mean())
        )
    elif strategy == "atr_breakout":
        baseline = close.ewm(span=channel, adjust=False).mean()
        upper = baseline + parameters["atr_mult"] * atr
        signal = _persistent(close.gt(upper), close.lt(baseline))
    elif strategy == "range_expansion_breakout":
        true_range = pd.concat(
            [
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        range_expansion = true_range.gt(
            true_range.rolling(channel).median().mul(
                parameters["atr_mult"]
            )
        )
        breakout = close.gt(high.rolling(channel).max().shift(1))
        signal = _persistent(
            range_expansion & breakout,
            close.lt(close.rolling(channel).mean()),
        )
    elif strategy == "rsi2_adx_pullback":
        rsi2 = _rsi(close, 2)
        adx, plus_di, minus_di = _adx(frame, max(3, min(14, channel)))
        mean = close.rolling(channel).mean()
        lower = mean - 2.0 * close.rolling(channel).std()
        entry = (
            close.gt(slow_ma)
            & adx.gt(parameters["adx_min"])
            & plus_di.gt(minus_di)
            & rsi2.lt(parameters["rsi_low"])
            & low.le(lower)
        )
        signal = _persistent(entry, rsi2.gt(65) | close.lt(slow_ma))
        momentum = -rsi2
    elif strategy == "rsi14_trend_pullback":
        rsi14 = _rsi(close, 14)
        entry = (
            close.gt(slow_ma)
            & rsi14.lt(parameters["rsi14_low"])
            & close.gt(low.rolling(channel).min())
        )
        signal = _persistent(entry, rsi14.gt(65) | close.lt(slow_ma))
        momentum = -rsi14
    elif strategy == "stochastic_trend_pullback":
        lowest = low.rolling(channel).min()
        highest = high.rolling(channel).max()
        stochastic = 100 * close.sub(lowest).div(
            highest.sub(lowest).replace(0, np.nan)
        )
        entry = (
            close.gt(slow_ma)
            & stochastic.lt(parameters["rsi14_low"])
            & stochastic.gt(stochastic.shift(1))
        )
        signal = _persistent(entry, stochastic.gt(80) | close.lt(slow_ma))
        momentum = -stochastic
    elif strategy == "macd_trend":
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(
            span=26, adjust=False
        ).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        signal = macd.gt(macd_signal) & close.gt(slow_ma)
        momentum = macd.div(close)
    elif strategy == "adx_trend":
        adx, plus_di, minus_di = _adx(frame, max(3, min(14, channel)))
        signal = (
            close.gt(slow_ma)
            & adx.gt(parameters["adx_min"])
            & plus_di.gt(minus_di)
        )
        momentum = adx.mul(plus_di.sub(minus_di)).div(10_000)
    elif strategy == "keltner_breakout":
        ema = close.ewm(span=channel, adjust=False).mean()
        upper = ema + parameters["atr_mult"] * atr
        signal = _persistent(close.gt(upper), close.lt(ema))
    elif strategy == "volume_breakout":
        entry = (
            close.gt(high.rolling(channel).max().shift(1))
            & volume.gt(
                volume.rolling(channel).mean().mul(parameters["volume_mult"])
            )
        )
        signal = _persistent(entry, close.lt(close.rolling(channel).mean()))
    elif strategy == "roc_trend":
        signal = (
            momentum.gt(parameters["roc_threshold"])
            & close.gt(slow_ma)
        )
    elif strategy == "risk_adjusted_momentum":
        returns = close.pct_change()
        realized_volatility = returns.rolling(channel).std().replace(
            0, np.nan
        )
        risk_adjusted = momentum.div(realized_volatility)
        signal = (
            close.gt(slow_ma)
            & momentum.gt(parameters["roc_threshold"])
            & risk_adjusted.gt(0)
        )
        momentum = risk_adjusted
    elif strategy == "volatility_adjusted_trend":
        returns = close.pct_change()
        short_volatility = returns.rolling(max(3, channel // 2)).std()
        long_volatility = returns.rolling(max(6, channel * 2)).std()
        controlled_volatility = short_volatility.lt(
            long_volatility.mul(parameters["volatility_ratio"])
        )
        signal = fast_ma.gt(slow_ma) & controlled_volatility
        momentum = momentum.div(
            long_volatility.replace(0, np.nan)
        )
    elif strategy == "ema_pullback":
        ema = close.ewm(span=fast, adjust=False).mean()
        entry = close.gt(slow_ma) & low.le(ema) & close.gt(ema)
        signal = _persistent(entry, close.lt(slow_ma))
    elif strategy == "etf_commodity_trend":
        eligible = str(frame.attrs.get("symbol", "")).upper() in {
            "DBC",
            "EEM",
            "EFA",
            "GLD",
            "IWM",
            "QQQ",
            "SLV",
            "SPY",
            "TLT",
        }
        signal = (
            momentum.gt(parameters["roc_threshold"])
            & close.gt(slow_ma)
            & eligible
        )
    elif strategy == "ppo_trend":
        fast_ema = close.ewm(span=max(3, fast), adjust=False).mean()
        slow_ema = close.ewm(span=max(fast + 2, slow), adjust=False).mean()
        ppo = 100 * fast_ema.sub(slow_ema).div(
            slow_ema.replace(0, np.nan)
        )
        ppo_signal = ppo.ewm(
            span=max(3, min(9, channel)), adjust=False
        ).mean()
        momentum = ppo.sub(ppo_signal)
        signal = momentum.gt(0) & ppo.gt(0) & close.gt(slow_ma)
    elif strategy == "cci_trend_pullback":
        cci = _cci(frame, max(5, channel))
        entry = (
            close.gt(slow_ma)
            & cci.shift(1).le(-100)
            & cci.gt(-100)
        )
        signal = _persistent(entry, cci.gt(100) | close.lt(slow_ma))
        momentum = -cci
    elif strategy == "mfi_trend_pullback":
        mfi = _mfi(frame, max(5, min(21, channel)))
        entry = (
            close.gt(slow_ma)
            & mfi.lt(40)
            & mfi.gt(mfi.shift(1))
        )
        signal = _persistent(entry, mfi.gt(80) | close.lt(slow_ma))
        momentum = -mfi
    elif strategy == "cmf_accumulation":
        cmf = _cmf(frame, max(5, channel))
        signal = close.gt(slow_ma) & cmf.gt(0.05)
        momentum = cmf
    elif strategy == "obv_breakout":
        obv = _obv(frame)
        entry = (
            close.gt(slow_ma)
            & obv.gt(obv.rolling(channel).max().shift(1))
        )
        signal = _persistent(
            entry,
            obv.lt(obv.rolling(channel).mean()) | close.lt(slow_ma),
        )
        momentum = obv.diff(channel).div(
            obv.abs().rolling(channel).mean().replace(0, np.nan)
        )
    elif strategy == "aroon_trend":
        aroon_up, aroon_down = _aroon(frame, max(5, channel))
        signal = (
            close.gt(slow_ma)
            & aroon_up.gt(70)
            & aroon_up.gt(aroon_down)
        )
        momentum = aroon_up.sub(aroon_down).div(100)
    elif strategy == "vortex_trend":
        vortex_plus, vortex_minus = _vortex(
            frame, max(5, min(28, channel))
        )
        signal = close.gt(slow_ma) & vortex_plus.gt(vortex_minus)
        momentum = vortex_plus.sub(vortex_minus)
    elif strategy == "choppiness_breakout":
        choppiness = _choppiness(frame, max(5, channel))
        entry = (
            choppiness.lt(45)
            & close.gt(high.rolling(channel).max().shift(1))
        )
        signal = _persistent(
            entry,
            choppiness.gt(61.8) | close.lt(close.rolling(channel).mean()),
        )
        momentum = -choppiness
    elif strategy == "efficiency_trend":
        efficiency = _efficiency_ratio(close, max(5, channel))
        signal = (
            close.gt(slow_ma)
            & momentum.gt(parameters["roc_threshold"])
            & efficiency.gt(0.30)
        )
        momentum = momentum.mul(efficiency)
    elif strategy == "vwap_deviation_reversion":
        vwap = _rolling_vwap(frame, max(5, channel))
        deviation = close.sub(vwap).div(
            close.rolling(max(5, channel)).std().replace(0, np.nan)
        )
        entry = close.gt(slow_ma) & deviation.lt(-1.0)
        signal = _persistent(entry, close.ge(vwap) | close.lt(slow_ma))
        momentum = -deviation
    elif strategy == "nr7_breakout":
        bar_range = high.sub(low)
        nr7 = bar_range.eq(bar_range.rolling(7).min())
        entry = nr7.shift(1, fill_value=False) & close.gt(high.shift(1))
        signal = _persistent(entry, close.lt(fast_ma))
        momentum = close.div(high.shift(1)).sub(1)
    elif strategy == "gap_recovery":
        previous_close = close.shift(1)
        gap = frame["open"].div(previous_close).sub(1)
        recovery = close.gt(previous_close) & gap.lt(
            -atr.div(previous_close).mul(parameters["atr_mult"])
        )
        signal = _persistent(recovery, close.lt(fast_ma))
        momentum = close.div(frame["open"]).sub(1)
    elif strategy == "connors_rsi_pullback":
        connors = _connors_rsi(close, max(20, channel))
        entry = close.gt(slow_ma) & connors.lt(25)
        signal = _persistent(
            entry, connors.gt(70) | close.lt(slow_ma)
        )
        momentum = -connors
    elif strategy == "trend_quality_52w":
        lookback = max(
            channel, 52 if timeframe in {"1w", "1mo"} else slow
        )
        rolling_high = close.rolling(lookback).max()
        high_score = close.div(rolling_high)
        trend_quality = _rolling_trend_quality(close, lookback)
        signal = (
            close.gt(slow_ma)
            & high_score.gt(0.85)
            & trend_quality.gt(0.40)
            & momentum.gt(parameters["roc_threshold"])
        )
        momentum = high_score.mul(trend_quality)
    elif strategy == "kama_adaptive_trend":
        efficiency = _efficiency_ratio(close, max(5, channel))
        kama = _kama(close, max(5, channel))
        entry = (
            close.gt(kama)
            & kama.gt(kama.shift(max(1, fast // 5)))
            & efficiency.gt(0.20)
        )
        signal = _persistent(entry, close.lt(kama))
        momentum = close.div(kama).sub(1).mul(efficiency)
    elif strategy == "trix_trend":
        span = max(3, min(channel, 30))
        ema_one = close.ewm(span=span, adjust=False).mean()
        ema_two = ema_one.ewm(span=span, adjust=False).mean()
        ema_three = ema_two.ewm(span=span, adjust=False).mean()
        trix = ema_three.pct_change().mul(100)
        trix_signal = trix.ewm(
            span=max(3, min(9, channel)), adjust=False
        ).mean()
        signal = close.gt(slow_ma) & trix.gt(trix_signal) & trix.gt(0)
        momentum = trix.sub(trix_signal)
    elif strategy == "tsi_trend":
        price_change = close.diff()
        first = price_change.ewm(span=25, adjust=False).mean()
        second = first.ewm(span=13, adjust=False).mean()
        absolute_first = price_change.abs().ewm(
            span=25, adjust=False
        ).mean()
        absolute_second = absolute_first.ewm(
            span=13, adjust=False
        ).mean()
        tsi = 100 * second.div(absolute_second.replace(0, np.nan))
        tsi_signal = tsi.ewm(
            span=max(3, min(9, channel)), adjust=False
        ).mean()
        signal = close.gt(slow_ma) & tsi.gt(tsi_signal) & tsi.gt(0)
        momentum = tsi.sub(tsi_signal).div(100)
    elif strategy == "ultimate_oscillator_pullback":
        ultimate = _ultimate_oscillator(frame)
        entry = (
            close.gt(slow_ma)
            & ultimate.lt(45)
            & ultimate.gt(ultimate.shift(1))
        )
        signal = _persistent(
            entry, ultimate.gt(65) | close.lt(slow_ma)
        )
        momentum = ultimate.rsub(50).div(100)
    elif strategy == "williams_r_pullback":
        period = max(5, min(28, channel))
        highest = high.rolling(period).max()
        lowest = low.rolling(period).min()
        williams = -100 * highest.sub(close).div(
            highest.sub(lowest).replace(0, np.nan)
        )
        entry = (
            close.gt(slow_ma)
            & williams.lt(-80)
            & williams.gt(williams.shift(1))
        )
        signal = _persistent(
            entry, williams.gt(-20) | close.lt(slow_ma)
        )
        momentum = williams.add(100).div(100)
    elif strategy == "chaikin_oscillator_trend":
        adl = _accumulation_distribution(frame)
        chaikin = adl.ewm(span=3, adjust=False).mean().sub(
            adl.ewm(span=10, adjust=False).mean()
        )
        scale = adl.abs().rolling(max(10, channel)).mean()
        momentum = chaikin.div(scale.replace(0, np.nan))
        signal = close.gt(slow_ma) & momentum.gt(0)
    elif strategy == "force_index_trend":
        force = close.diff().mul(volume.clip(lower=0))
        smoothed = force.ewm(
            span=max(3, min(13, channel)), adjust=False
        ).mean()
        scale = force.abs().rolling(max(5, channel)).mean()
        momentum = smoothed.div(scale.replace(0, np.nan))
        signal = close.gt(slow_ma) & momentum.gt(0)
    elif strategy == "vpt_breakout":
        vpt = close.pct_change().mul(volume.clip(lower=0)).cumsum()
        entry = (
            close.gt(slow_ma)
            & vpt.gt(vpt.rolling(channel).max().shift(1))
        )
        signal = _persistent(
            entry, vpt.lt(vpt.rolling(channel).mean()) | close.lt(slow_ma)
        )
        momentum = vpt.diff(channel).div(
            vpt.abs().rolling(channel).mean().replace(0, np.nan)
        )
    elif strategy == "linear_regression_trend":
        period = max(5, channel)
        trend_quality = _rolling_trend_quality(close, period)
        slope_proxy = close.pct_change(period)
        signal = (
            close.gt(slow_ma)
            & slope_proxy.gt(parameters["roc_threshold"])
            & trend_quality.gt(0.50)
        )
        momentum = slope_proxy.mul(trend_quality)
    elif strategy == "parkinson_volatility_breakout":
        period = max(5, channel)
        variance = (
            np.log(high.div(low).where(low.gt(0)))
            .pow(2)
            .div(4 * math.log(2))
        )
        short_volatility = variance.rolling(period).mean()
        long_volatility = variance.rolling(period * 3).mean()
        compression = short_volatility.lt(
            long_volatility.mul(parameters["volatility_ratio"])
        )
        entry = compression & close.gt(
            high.rolling(period).max().shift(1)
        )
        signal = _persistent(entry, close.lt(fast_ma))
        momentum = short_volatility.div(
            long_volatility.replace(0, np.nan)
        ).mul(-1)
    elif strategy == "yang_zhang_volatility_breakout":
        period = max(5, channel)
        yz_variance = _yang_zhang_variance(frame, period)
        long_variance = yz_variance.rolling(period * 2).mean()
        compression = yz_variance.lt(
            long_variance.mul(parameters["volatility_ratio"])
        )
        entry = compression & close.gt(
            high.rolling(period).max().shift(1)
        )
        signal = _persistent(entry, close.lt(fast_ma))
        momentum = yz_variance.div(
            long_variance.replace(0, np.nan)
        ).mul(-1)
    elif strategy == "variance_ratio_trend":
        period = max(10, channel * 2)
        variance_ratio = _variance_ratio(close, period, q=5)
        signal = (
            close.gt(slow_ma)
            & momentum.gt(parameters["roc_threshold"])
            & variance_ratio.gt(1.05)
        )
        momentum = momentum.mul(variance_ratio)
    elif strategy == "ulcer_low_risk_trend":
        period = max(10, channel)
        rolling_peak = close.rolling(period).max()
        drawdown = close.div(rolling_peak).sub(1).mul(100)
        ulcer = drawdown.pow(2).rolling(period).mean().pow(0.5)
        long_ulcer = ulcer.rolling(period * 3).median()
        signal = (
            fast_ma.gt(slow_ma)
            & ulcer.lt(long_ulcer)
            & momentum.gt(parameters["roc_threshold"])
        )
        momentum = momentum.div(ulcer.replace(0, np.nan))
    else:
        raise ValueError(f"UNREGISTERED_PHASE11_9_STRATEGY:{strategy}")
    return _signal_frame(signal, momentum)


def _signal_frame(signal: pd.Series, score: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": signal.fillna(False).astype(bool),
            "score": score.replace([np.inf, -np.inf], np.nan).fillna(
                -math.inf
            ),
        },
        index=signal.index,
    )


def _persistent(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    active = False
    values = []
    for enter, leave in zip(entry.fillna(False), exit_.fillna(False)):
        if active and bool(leave):
            active = False
        elif not active and bool(enter):
            active = True
        values.append(active)
    return pd.Series(values, index=entry.index, dtype=bool)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    relative = gain.div(loss.replace(0, np.nan))
    return 100 - 100 / (1 + relative)


def _atr(frame: pd.DataFrame, period: int) -> pd.Series:
    previous = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()


def _adx(
    frame: pd.DataFrame, period: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = _atr(frame, period)
    plus_di = 100 * plus_dm.ewm(
        alpha=1 / period, adjust=False
    ).mean().div(atr)
    minus_di = 100 * minus_dm.ewm(
        alpha=1 / period, adjust=False
    ).mean().div(atr)
    dx = 100 * (plus_di - minus_di).abs().div(plus_di + minus_di)
    adx = dx.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    return adx, plus_di, minus_di


def _cci(frame: pd.DataFrame, period: int) -> pd.Series:
    typical = frame[["high", "low", "close"]].mean(axis=1)
    mean = typical.rolling(period).mean()
    deviation = typical.rolling(period).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    return typical.sub(mean).div(deviation.mul(0.015).replace(0, np.nan))


def _mfi(frame: pd.DataFrame, period: int) -> pd.Series:
    typical = frame[["high", "low", "close"]].mean(axis=1)
    flow = typical.mul(frame["volume"].clip(lower=0))
    positive = flow.where(typical.diff().gt(0), 0.0).rolling(period).sum()
    negative = flow.where(typical.diff().lt(0), 0.0).rolling(period).sum()
    ratio = positive.div(negative.replace(0, np.nan))
    return 100 - 100 / (1 + ratio)


def _cmf(frame: pd.DataFrame, period: int) -> pd.Series:
    spread = frame["high"].sub(frame["low"]).replace(0, np.nan)
    multiplier = (
        frame["close"].mul(2).sub(frame["high"]).sub(frame["low"])
    ).div(spread)
    flow = multiplier.mul(frame["volume"].clip(lower=0))
    return flow.rolling(period).sum().div(
        frame["volume"].clip(lower=0).rolling(period).sum().replace(0, np.nan)
    )


def _obv(frame: pd.DataFrame) -> pd.Series:
    direction = np.sign(frame["close"].diff()).fillna(0)
    return direction.mul(frame["volume"].clip(lower=0)).cumsum()


def _aroon(
    frame: pd.DataFrame, period: int
) -> tuple[pd.Series, pd.Series]:
    def recency(values: np.ndarray) -> float:
        return 100 * (int(np.argmax(values)) + 1) / len(values)

    up = frame["high"].rolling(period).apply(recency, raw=True)
    down = frame["low"].mul(-1).rolling(period).apply(
        recency, raw=True
    )
    return up, down


def _vortex(
    frame: pd.DataFrame, period: int
) -> tuple[pd.Series, pd.Series]:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"].sub(frame["low"]),
            frame["high"].sub(previous_close).abs(),
            frame["low"].sub(previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    denominator = true_range.rolling(period).sum().replace(0, np.nan)
    plus = frame["high"].sub(frame["low"].shift(1)).abs()
    minus = frame["low"].sub(frame["high"].shift(1)).abs()
    return (
        plus.rolling(period).sum().div(denominator),
        minus.rolling(period).sum().div(denominator),
    )


def _choppiness(frame: pd.DataFrame, period: int) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"].sub(frame["low"]),
            frame["high"].sub(previous_close).abs(),
            frame["low"].sub(previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    span = frame["high"].rolling(period).max().sub(
        frame["low"].rolling(period).min()
    )
    ratio = true_range.rolling(period).sum().div(
        span.replace(0, np.nan)
    )
    return 100 * np.log10(ratio) / math.log10(period)


def _efficiency_ratio(close: pd.Series, period: int) -> pd.Series:
    direction = close.diff(period).abs()
    path = close.diff().abs().rolling(period).sum()
    return direction.div(path.replace(0, np.nan))


def _rolling_vwap(frame: pd.DataFrame, period: int) -> pd.Series:
    typical = frame[["high", "low", "close"]].mean(axis=1)
    volume = frame["volume"].clip(lower=0)
    return typical.mul(volume).rolling(period).sum().div(
        volume.rolling(period).sum().replace(0, np.nan)
    )


def _connors_rsi(close: pd.Series, rank_period: int) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    streak_values: list[float] = []
    streak = 0.0
    for value in direction:
        if value == 0:
            streak = 0.0
        elif np.sign(streak) == value:
            streak += float(value)
        else:
            streak = float(value)
        streak_values.append(streak)
    streak_series = pd.Series(streak_values, index=close.index)
    streak_rsi = _rsi(streak_series, 2)
    one_bar_return = close.pct_change()
    percentile = one_bar_return.rolling(rank_period).apply(
        lambda values: 100
        * float(np.sum(values[:-1] < values[-1]))
        / max(1, len(values) - 1),
        raw=True,
    )
    return (_rsi(close, 3) + streak_rsi + percentile).div(3)


def _rolling_trend_quality(close: pd.Series, period: int) -> pd.Series:
    time_index = pd.Series(
        np.arange(len(close), dtype=float), index=close.index
    )
    return close.rolling(period).corr(time_index).pow(2)


def _kama(close: pd.Series, period: int) -> pd.Series:
    efficiency = _efficiency_ratio(close, period).fillna(0)
    fast_constant = 2 / (2 + 1)
    slow_constant = 2 / (30 + 1)
    smoothing = (
        efficiency.mul(fast_constant - slow_constant).add(slow_constant)
    ).pow(2)
    values: list[float] = [math.nan] * len(close)
    first_valid = min(period, max(0, len(close) - 1))
    if len(close):
        values[first_valid] = float(
            close.iloc[: first_valid + 1].mean()
        )
    for index in range(first_valid + 1, len(close)):
        previous = values[index - 1]
        values[index] = previous + float(smoothing.iloc[index]) * (
            float(close.iloc[index]) - previous
        )
    return pd.Series(values, index=close.index)


def _accumulation_distribution(frame: pd.DataFrame) -> pd.Series:
    spread = frame["high"].sub(frame["low"]).replace(0, np.nan)
    multiplier = (
        frame["close"].mul(2).sub(frame["high"]).sub(frame["low"])
    ).div(spread)
    return multiplier.fillna(0).mul(
        frame["volume"].clip(lower=0)
    ).cumsum()


def _ultimate_oscillator(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    minimum = pd.concat(
        [frame["low"], previous_close], axis=1
    ).min(axis=1)
    maximum = pd.concat(
        [frame["high"], previous_close], axis=1
    ).max(axis=1)
    buying_pressure = frame["close"].sub(minimum)
    true_range = maximum.sub(minimum)

    def average(period: int) -> pd.Series:
        return buying_pressure.rolling(period).sum().div(
            true_range.rolling(period).sum().replace(0, np.nan)
        )

    return 100 * (
        4 * average(7) + 2 * average(14) + average(28)
    ).div(7)


def _yang_zhang_variance(
    frame: pd.DataFrame, period: int
) -> pd.Series:
    open_ = frame["open"].where(frame["open"].gt(0))
    high = frame["high"].where(frame["high"].gt(0))
    low = frame["low"].where(frame["low"].gt(0))
    close = frame["close"].where(frame["close"].gt(0))
    overnight = np.log(open_.div(close.shift(1)))
    open_close = np.log(close.div(open_))
    rogers_satchell = (
        np.log(high.div(open_)).mul(np.log(high.div(close)))
        + np.log(low.div(open_)).mul(np.log(low.div(close)))
    )
    k = 0.34 / (1.34 + (period + 1) / max(1, period - 1))
    return (
        overnight.rolling(period).var()
        + k * open_close.rolling(period).var()
        + (1 - k) * rogers_satchell.rolling(period).mean()
    )


def _variance_ratio(
    close: pd.Series, period: int, *, q: int
) -> pd.Series:
    log_price = np.log(close.where(close.gt(0)))
    one_bar = log_price.diff()
    q_bar = log_price.diff(q)
    denominator = one_bar.rolling(period).var().mul(q)
    return q_bar.rolling(period).var().div(
        denominator.replace(0, np.nan)
    )


def _summarize(
    results: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    normal = results.loc[results["cost_bps"].eq(10.0)]
    for (strategy, timeframe), group in normal.groupby(
        ["strategy", "timeframe"]
    ):
        stress = results.loc[
            results["strategy"].eq(strategy)
            & results["timeframe"].eq(timeframe)
            & results["cost_bps"].eq(50.0)
        ]
        selected = selections.loc[
            selections["strategy"].eq(strategy)
            & selections["timeframe"].eq(timeframe)
        ]
        pfs = group["period_profit_factor"].replace(
            [np.inf, -np.inf], np.nan
        )
        stress_pfs = stress["period_profit_factor"].replace(
            [np.inf, -np.inf], np.nan
        )
        rows.append(
            {
                "strategy": strategy,
                "timeframe": timeframe,
                "confidence": "EVALUABLE",
                "fold_count": len(group),
                "positive_fold_ratio": float(group["CAGR"].gt(0).mean()),
                "median_oos_portfolio_pf": float(pfs.median()),
                "worst_oos_portfolio_pf": float(pfs.min()),
                "median_oos_CAGR": float(group["CAGR"].median()),
                "median_oos_Sharpe": float(group["Sharpe"].median()),
                "worst_oos_drawdown": float(
                    group["maximum_drawdown"].min()
                ),
                "cost_50bps_median_pf": float(stress_pfs.median()),
                "plateau_fold_ratio": float(
                    selected["parameter_plateau"].mean()
                ),
                "median_fill_count": float(group["fill_count"].median()),
                "median_turnover": float(
                    group["turnover_initial_capital"].median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "worst_oos_portfolio_pf",
            "median_oos_portfolio_pf",
            "median_oos_CAGR",
        ],
        ascending=False,
    )


def _backtest_positive_registry(
    summary: pd.DataFrame,
) -> dict[str, Any]:
    if summary.empty:
        return {
            "schema": "phase11_9_backtest_positive_registry_v1",
            "status": "NO_CANDIDATES",
            "candidate_count": 0,
            "candidates": [],
        }
    gates = (
        summary["fold_count"].ge(2)
        & summary["median_oos_portfolio_pf"].gt(1.0)
        & summary["median_oos_CAGR"].gt(0)
        & summary["cost_50bps_median_pf"].gt(1.0)
        & summary["median_fill_count"].ge(10)
        & summary["worst_oos_drawdown"].gt(-0.50)
    )
    candidates = summary.loc[gates].sort_values(
        [
            "confidence",
            "median_oos_portfolio_pf",
            "median_oos_CAGR",
        ],
        ascending=[True, False, False],
    )
    return {
        "schema": "phase11_9_backtest_positive_registry_v1",
        "status": (
            "BACKTEST_POSITIVE_CANDIDATES_AVAILABLE"
            if not candidates.empty
            else "NO_BACKTEST_POSITIVE_CANDIDATE"
        ),
        "candidate_count": len(candidates),
        "unique_strategy_count": int(candidates["strategy"].nunique()),
        "timeframe_counts": {
            str(key): int(value)
            for key, value in candidates["timeframe"].value_counts().items()
        },
        "criteria": {
            "median_oos_portfolio_pf": ">1.0",
            "median_oos_CAGR": ">0",
            "cost_50bps_median_pf": ">1.0",
            "median_fill_count": ">=10",
            "worst_oos_drawdown": ">-0.50",
        },
        "candidates": candidates.to_dict("records"),
        "financial_finalist": False,
        "automatic_paper_promotion": False,
        "authority": "NONE",
    }


def _shortlist(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"status": "NO_CANDIDATES", "candidates": []}
    gates = (
        summary["fold_count"].ge(2)
        & summary["positive_fold_ratio"].ge(0.60)
        & summary["median_oos_portfolio_pf"].gt(1.0)
        & summary["cost_50bps_median_pf"].gt(1.0)
        & summary["worst_oos_portfolio_pf"].gt(0.70)
        & summary["worst_oos_drawdown"].gt(-0.50)
        & summary["plateau_fold_ratio"].ge(0.50)
        & summary["median_fill_count"].ge(10)
    )
    eligible = summary.loc[
        gates & summary["confidence"].eq("EVALUABLE")
    ].copy()
    eligible = eligible.sort_values(
        [
            "worst_oos_portfolio_pf",
            "median_oos_portfolio_pf",
            "median_oos_CAGR",
        ],
        ascending=False,
    ).head(10)
    exploratory = summary.loc[
        gates & summary["confidence"].eq("LOW_CONFIDENCE")
    ].sort_values(
        [
            "worst_oos_portfolio_pf",
            "median_oos_portfolio_pf",
            "median_oos_CAGR",
        ],
        ascending=False,
    ).head(10)
    return {
        "status": (
            "ACCELERATED_RESEARCH_SHORTLIST"
            if not eligible.empty
            else "NO_ROBUST_RESEARCH_CANDIDATE"
        ),
        "candidate_count": len(eligible),
        "candidates": eligible.to_dict("records"),
        "low_confidence_intraday_count": len(exploratory),
        "low_confidence_intraday_watchlist": exploratory.to_dict("records"),
        "authority": "NONE",
        "financial_finalist": False,
    }


def _cross_timeframe(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for strategy, group in summary.groupby("strategy"):
        rows.append(
            {
                "strategy": strategy,
                "evaluated_timeframes": len(group),
                "median_pf_above_one_timeframes": int(
                    group["median_oos_portfolio_pf"].gt(1).sum()
                ),
                "cost_50bps_pf_above_one_timeframes": int(
                    group["cost_50bps_median_pf"].gt(1).sum()
                ),
                "positive_fold_ratio_median": float(
                    group["positive_fold_ratio"].median()
                ),
                "cross_timeframe_median_pf": float(
                    group["median_oos_portfolio_pf"].median()
                ),
                "cross_timeframe_worst_fold_pf": float(
                    group["worst_oos_portfolio_pf"].min()
                ),
                "maximum_drawdown_worst": float(
                    group["worst_oos_drawdown"].min()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "cost_50bps_pf_above_one_timeframes",
            "median_pf_above_one_timeframes",
            "cross_timeframe_median_pf",
        ],
        ascending=False,
    )


def _formula_specifications(
    shortlist: Mapping[str, Any], selections: pd.DataFrame
) -> dict[str, Any]:
    rows = []
    for candidate in shortlist.get("candidates", []):
        strategy = str(candidate["strategy"])
        timeframe = str(candidate["timeframe"])
        selected = selections.loc[
            selections["strategy"].eq(strategy)
            & selections["timeframe"].eq(timeframe)
        ]
        counts = selected["selected_profile"].value_counts()
        profile = str(counts.index[0])
        row: dict[str, Any] = {
            "strategy": strategy,
            "timeframe": timeframe,
            "dominant_profile": profile,
            "profile_selection_count": int(counts.iloc[0]),
            "fold_count": len(selected),
            "parameters": _parameters(timeframe, profile),
            "portfolio_ranking_score": (
                f"{_parameters(timeframe, profile)['channel']}-bar momentum"
            ),
            "execution": "NEXT_BAR_OPEN_AFTER_CLOSED_SIGNAL_BAR",
            "maximum_positions": 4,
            "gross_exposure_maximum": 1.0,
        }
        if strategy == "breakout_consensus":
            row["formula"] = {
                "vote_threshold": math.ceil(
                    len(ENSEMBLES["breakout_consensus"]) / 2
                ),
                "components": [
                    "DONCHIAN_CLOSE_ABOVE_PRIOR_CHANNEL_HIGH",
                    "MA_CHANNEL_CLOSE_ABOVE_PRIOR_CHANNEL_HIGH",
                    "BOLLINGER_CLOSE_ABOVE_MEAN_PLUS_SIGMA_STD",
                    "VOLATILITY_CONTRACTION_AND_DONCHIAN_BREAKOUT",
                    "DONCHIAN_BREAKOUT_AND_VOLUME_ABOVE_MULTIPLE",
                ],
                "exit": "COMPONENT_STATE_EXITS_BELOW_CHANNEL_MEAN",
            }
        elif strategy == "donchian_breakout":
            row["formula"] = {
                "entry": "CLOSE_ABOVE_PRIOR_CHANNEL_HIGH",
                "exit": "CLOSE_BELOW_CHANNEL_MEAN",
            }
        rows.append(row)
    return {
        "schema": "phase11_9_formula_specifications_v1",
        "status": "GO",
        "formulas": rows,
        "authority": "NONE",
    }


def _fold_bootstrap(
    group: pd.DataFrame, *, seed: int, runs: int = 10_000
) -> dict[str, Any]:
    clean = group[
        ["period_profit_factor", "CAGR", "maximum_drawdown"]
    ].replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna()
    if clean.empty:
        return {"status": "INSUFFICIENT_SAMPLE", "runs": 0}
    values = clean.to_numpy(dtype=float)
    random = np.random.default_rng(seed)
    indices = random.integers(0, len(values), size=(runs, len(values)))
    samples = values[indices]
    median_pf = np.median(samples[:, :, 0], axis=1)
    median_cagr = np.median(samples[:, :, 1], axis=1)
    worst_drawdown = np.min(samples[:, :, 2], axis=1)
    return {
        "status": "GO",
        "runs": runs,
        "probability_median_pf_above_one": float(
            np.mean(median_pf > 1.0)
        ),
        "median_pf_p05": float(np.quantile(median_pf, 0.05)),
        "median_pf_p50": float(np.quantile(median_pf, 0.50)),
        "median_pf_p95": float(np.quantile(median_pf, 0.95)),
        "probability_median_CAGR_positive": float(
            np.mean(median_cagr > 0)
        ),
        "median_CAGR_p05": float(np.quantile(median_cagr, 0.05)),
        "worst_drawdown_p95_loss": float(
            np.quantile(worst_drawdown, 0.05)
        ),
    }


def _seed(value: Any) -> int:
    return int(_hash(value)[:8], 16)


def _coverage(
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]]
) -> list[dict[str, Any]]:
    rows = []
    for timeframe, frames in frames_by_timeframe.items():
        counts = [len(frame) for frame in frames.values()]
        rows.append(
            {
                "timeframe": timeframe,
                "instrument_count": len(frames),
                "minimum_bars": min(counts) if counts else 0,
                "maximum_bars": max(counts) if counts else 0,
                "source": (
                    _intraday_source_label(frames, timeframe)
                    if timeframe in {"1h", "2h", "4h"}
                    else "YFINANCE_ADJUSTED_DAILY"
                    if timeframe == "1d"
                    else "CLOSED_DAILY_AGGREGATION"
                ),
            }
        )
    return rows


def _intraday_source_label(
    frames: Mapping[str, pd.DataFrame], timeframe: str
) -> str:
    providers = sorted(
        {
            str(frame.attrs.get("provider", "UNKNOWN"))
            for frame in frames.values()
        }
    )
    origin = "NATIVE_1H" if timeframe == "1h" else "CLOSED_FROM_NATIVE_1H"
    return f"{'+'.join(providers)}_{origin}"


def _intraday_fingerprint(frame: pd.DataFrame) -> str:
    if "row_hash" in frame:
        values = frame["row_hash"].astype(str).tolist()
    else:
        values = (
            pd.util.hash_pandas_object(
                frame[["date", "open", "high", "low", "close", "volume"]],
                index=False,
            )
            .astype(str)
            .tolist()
        )
    return hashlib.sha256("|".join(values).encode()).hexdigest().upper()


def _frames_fingerprint(
    frames_by_timeframe: Mapping[
        str, Mapping[str, pd.DataFrame]
    ],
) -> str:
    evidence = []
    for timeframe, frames in sorted(frames_by_timeframe.items()):
        for symbol, frame in sorted(frames.items()):
            evidence.append(
                {
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "provider": frame.attrs.get("provider"),
                    "rows": len(frame),
                    "start": frame.index.min(),
                    "end": frame.index.max(),
                    "content_fingerprint": frame.attrs.get(
                        "content_fingerprint"
                    ),
                }
            )
    return _hash(evidence)


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _output(project_root: Path) -> Path:
    return project_root / "output" / "research" / "phase11_9"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest().upper()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, default=str) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


__all__ = [
    "current_watchlist",
    "phase11_9_schema",
    "phase11_9_status",
    "run_diagnostics",
    "run_discovery",
]
