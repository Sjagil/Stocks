from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.data.multitimeframe import bar_freshness
from stocks.research.phase11_6 import nested_walk_forward_folds
from stocks.research.phase11_8 import _run_portfolio
from stocks.research.phase11_9 import (
    BASE_STRATEGIES,
    BENCHMARK_SYMBOLS,
    COSTS_BPS,
    ENSEMBLES,
    _aggregate,
    _load_current_frames,
    _load_frames,
    _parameters,
    _persistent,
    _signal_frame,
    _strategy,
)
from stocks.signals.freshness import evaluate_signal_freshness


SCHEMA = "phase11_10_causal_multitimeframe_swing_v1"
PROFILES = ("responsive", "balanced", "conservative")
PIT_PRIMARY_PROVIDERS = frozenset({"EODHD", "YFINANCE"})
ARCHITECTURES: dict[str, dict[str, str]] = {
    "monthly_weekly_pullback": {
        "higher": "1mo",
        "lower": "1w",
        "entry": "ema_pullback",
    },
    "monthly_weekly_breakout": {
        "higher": "1mo",
        "lower": "1w",
        "entry": "volatility_contraction_breakout",
    },
    "weekly_daily_pullback": {
        "higher": "1w",
        "lower": "1d",
        "entry": "ema_pullback",
    },
    "weekly_daily_breakout": {
        "higher": "1w",
        "lower": "1d",
        "entry": "volatility_contraction_breakout",
    },
    "weekly_4h_pullback": {
        "higher": "1w",
        "lower": "4h",
        "entry": "ema_pullback",
    },
    "weekly_4h_breakout": {
        "higher": "1w",
        "lower": "4h",
        "entry": "volatility_contraction_breakout",
    },
    "daily_4h_pullback": {
        "higher": "1d",
        "lower": "4h",
        "entry": "ema_pullback",
    },
    "daily_4h_breakout": {
        "higher": "1d",
        "lower": "4h",
        "entry": "volatility_contraction_breakout",
    },
    "daily_1h_pullback": {
        "higher": "1d",
        "lower": "1h",
        "entry": "ema_pullback",
    },
    "daily_1h_breakout": {
        "higher": "1d",
        "lower": "1h",
        "entry": "volatility_contraction_breakout",
    },
    "four_hour_1h_pullback": {
        "higher": "4h",
        "lower": "1h",
        "entry": "ema_pullback",
    },
    "four_hour_1h_breakout": {
        "higher": "4h",
        "lower": "1h",
        "entry": "volatility_contraction_breakout",
    },
    "four_hour_1h_ma_trend": {
        "higher": "4h",
        "lower": "1h",
        "entry": "ma_crossover",
    },
    "four_hour_1h_donchian": {
        "higher": "4h",
        "lower": "1h",
        "entry": "donchian_breakout",
    },
    "four_hour_1h_rsi_pullback": {
        "higher": "4h",
        "lower": "1h",
        "entry": "rsi2_adx_pullback",
    },
    "four_hour_1h_ma_channel": {
        "higher": "4h",
        "lower": "1h",
        "entry": "ma_channel",
    },
    "four_hour_1h_bollinger": {
        "higher": "4h",
        "lower": "1h",
        "entry": "bollinger_breakout",
    },
    "daily_1h_ma_trend": {
        "higher": "1d",
        "lower": "1h",
        "entry": "ma_crossover",
    },
    "daily_1h_donchian": {
        "higher": "1d",
        "lower": "1h",
        "entry": "donchian_breakout",
    },
    "daily_1h_rsi_pullback": {
        "higher": "1d",
        "lower": "1h",
        "entry": "rsi2_adx_pullback",
    },
    "daily_1h_ma_channel": {
        "higher": "1d",
        "lower": "1h",
        "entry": "ma_channel",
    },
    "daily_1h_bollinger": {
        "higher": "1d",
        "lower": "1h",
        "entry": "bollinger_breakout",
    },
    "daily_4h_ma_trend": {
        "higher": "1d",
        "lower": "4h",
        "entry": "ma_crossover",
    },
    "daily_4h_donchian": {
        "higher": "1d",
        "lower": "4h",
        "entry": "donchian_breakout",
    },
    "daily_4h_rsi_pullback": {
        "higher": "1d",
        "lower": "4h",
        "entry": "rsi2_adx_pullback",
    },
    "daily_4h_ma_channel": {
        "higher": "1d",
        "lower": "4h",
        "entry": "ma_channel",
    },
    "daily_4h_bollinger": {
        "higher": "1d",
        "lower": "4h",
        "entry": "bollinger_breakout",
    },
    "daily_4h_1h_pullback": {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "ema_pullback",
    },
    "daily_4h_1h_breakout": {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "volatility_contraction_breakout",
    },
    "daily_4h_1h_ma_trend": {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "ma_crossover",
    },
    "daily_4h_1h_donchian": {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "donchian_breakout",
    },
    "daily_4h_1h_rsi_pullback": {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "rsi2_adx_pullback",
    },
    "weekly_daily_4h_pullback": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "ema_pullback",
    },
    "weekly_daily_4h_breakout": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "volatility_contraction_breakout",
    },
    "weekly_daily_4h_ma_trend": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "ma_crossover",
    },
    "weekly_daily_4h_donchian": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "donchian_breakout",
    },
    "weekly_daily_4h_ma_channel": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "ma_channel",
    },
    "weekly_daily_4h_bollinger": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "bollinger_breakout",
    },
    "four_hour_1h_adaptive_pullback": {
        "higher": "4h",
        "lower": "1h",
        "entry": "adaptive_percentile_pullback",
    },
    "daily_1h_adaptive_pullback": {
        "higher": "1d",
        "lower": "1h",
        "entry": "adaptive_percentile_pullback",
    },
    "daily_4h_adaptive_pullback": {
        "higher": "1d",
        "lower": "4h",
        "entry": "adaptive_percentile_pullback",
    },
    "daily_4h_1h_adaptive_pullback": {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "adaptive_percentile_pullback",
    },
    "weekly_daily_4h_adaptive_pullback": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "adaptive_percentile_pullback",
    },
    "four_hour_1h_adaptive_breakout": {
        "higher": "4h",
        "lower": "1h",
        "entry": "adaptive_volatility_breakout",
    },
    "daily_1h_adaptive_breakout": {
        "higher": "1d",
        "lower": "1h",
        "entry": "adaptive_volatility_breakout",
    },
    "daily_4h_adaptive_breakout": {
        "higher": "1d",
        "lower": "4h",
        "entry": "adaptive_volatility_breakout",
    },
    "daily_4h_1h_adaptive_breakout": {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "adaptive_volatility_breakout",
    },
    "weekly_daily_4h_adaptive_breakout": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "adaptive_volatility_breakout",
    },
    "four_hour_1h_beta_residual": {
        "higher": "4h",
        "lower": "1h",
        "entry": "beta_residual_pullback",
    },
    "daily_1h_beta_residual": {
        "higher": "1d",
        "lower": "1h",
        "entry": "beta_residual_pullback",
    },
    "daily_4h_beta_residual": {
        "higher": "1d",
        "lower": "4h",
        "entry": "beta_residual_pullback",
    },
    "daily_4h_1h_beta_residual": {
        "higher": "1d",
        "middle": "4h",
        "lower": "1h",
        "entry": "beta_residual_pullback",
    },
    "weekly_daily_4h_beta_residual": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "beta_residual_pullback",
    },
    "daily_4h_risk_adjusted_momentum": {
        "higher": "1d",
        "lower": "4h",
        "entry": "risk_adjusted_momentum",
    },
    "weekly_daily_4h_risk_adjusted_momentum": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "risk_adjusted_momentum",
    },
    "daily_4h_etf_commodity_trend": {
        "higher": "1d",
        "lower": "4h",
        "entry": "etf_commodity_trend",
    },
    "weekly_daily_4h_etf_commodity_trend": {
        "higher": "1w",
        "middle": "1d",
        "lower": "4h",
        "entry": "etf_commodity_trend",
    },
}
_ALL_IMPLEMENTED_ENTRIES = (*BASE_STRATEGIES, *ENSEMBLES)
for _prefix, _higher, _middle, _lower in (
    ("four_hour_1h", "4h", None, "1h"),
    ("daily_4h", "1d", None, "4h"),
    ("daily_4h_1h", "1d", "4h", "1h"),
    ("four_hour_2h", "4h", None, "2h"),
    ("daily_2h", "1d", None, "2h"),
    ("daily_4h_2h", "1d", "4h", "2h"),
    ("weekly_daily_2h", "1w", "1d", "2h"),
):
    for _entry in _ALL_IMPLEMENTED_ENTRIES:
        _specification = {
            "higher": _higher,
            "lower": _lower,
            "entry": _entry,
        }
        if _middle is not None:
            _specification["middle"] = _middle
        if _specification in ARCHITECTURES.values():
            continue
        ARCHITECTURES[f"{_prefix}_{_entry}"] = _specification
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
_MTF_WORKER_FRAMES: Mapping[str, Mapping[str, pd.DataFrame]] = {}
_RESEARCH_RESULT_ARTIFACTS = (
    "architecture-summary.csv",
    "coverage.csv",
    "nested-results.parquet",
    "parameter-selections.csv",
    "shortlist.json",
)
_RESEARCH_RESULT_HASH_POLICY = "IMMUTABLE_FINANCIAL_ARTIFACTS_V2"


def phase11_10_schema(project_root: Path) -> dict[str, Any]:
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "architectures": ARCHITECTURES,
        "profiles": list(PROFILES),
        "research_source_hash": _research_source_hash(project_root),
        "cost_stress_bps_per_side": list(COSTS_BPS),
        "signal_contract": {
            "higher_timeframe_gate": "TRIPLE_MA_TREND",
            "gate_availability": "SHIFTED_ONE_COMPLETE_HIGHER_BAR",
            "lower_timeframe_entries": sorted(
                {
                    specification["entry"].upper()
                    for specification in ARCHITECTURES.values()
                }
            ),
            "adaptive_threshold_policy": (
                "ROLLING_DISTRIBUTION_SHIFTED_ONE_LOWER_BAR"
            ),
            "beta_residual_anchor": "SPY",
            "execution": "NEXT_LOWER_TIMEFRAME_BAR_OPEN",
            "same_source_lineage": True,
            "synthetic_intraday": False,
        },
        "portfolio_contract": {
            "whole_shares": True,
            "base_currency": "EUR",
            "gross_exposure_maximum": 1.0,
            "global_security_netting": True,
            "primary_metric": "OOS_PORTFOLIO_PERIOD_PROFIT_FACTOR",
        },
        "trial_contract": {
            "global_hypothesis_count": len(ARCHITECTURES) * len(PROFILES),
            "decimal_optimization": False,
            "nested_validation_selection": True,
            "independent_future_holdout": False,
        },
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "schema.json", report)
    return report


def _phase11_10_worker_limit(pending_count: int) -> int:
    raw_limit = os.environ.get("STOCKS_PHASE11_10_MAX_WORKERS", "4")
    try:
        configured_limit = int(raw_limit)
    except ValueError:
        configured_limit = 4
    return min(
        12,
        max(1, configured_limit),
        max(1, pending_count),
        max(1, (os.cpu_count() or 2) - 2),
    )


def run_phase11_10(
    project_root: Path,
    *,
    historical_cutoff: str | None = None,
) -> dict[str, Any]:
    output = _output(project_root)
    output.mkdir(parents=True, exist_ok=True)
    if historical_cutoff is not None:
        _archive_current_research_evidence(project_root)
    schema = phase11_10_schema(project_root)
    all_frames = _load_frames(project_root)
    cutoff = _historical_cutoff(historical_cutoff)
    if cutoff is not None:
        all_frames = _truncate_frames(all_frames, cutoff=cutoff)
    historical_data_hash = _multitimeframe_frames_fingerprint(all_frames)
    run_signature = _hash(
        {
            "schema": schema,
            "architectures": ARCHITECTURES,
            "data_fingerprint": historical_data_hash,
            "historical_cutoff": cutoff.isoformat() if cutoff else None,
        }
    )
    checkpoint = _load_mtf_checkpoint(output, run_signature)
    result_rows = checkpoint["result_rows"]
    selection_rows = checkpoint["selection_rows"]
    coverage_rows = checkpoint["coverage_rows"]
    blocked = checkpoint["blocked"]
    completed = set(checkpoint["completed_architectures"])
    pending = [
        architecture
        for architecture in ARCHITECTURES
        if architecture not in completed
    ]
    worker_limit = _phase11_10_worker_limit(len(pending))
    checkpoint_batch_size = 10
    with ProcessPoolExecutor(
        max_workers=worker_limit,
        initializer=_initialize_mtf_worker,
        initargs=(
            str(project_root),
            cutoff.isoformat() if cutoff is not None else None,
        ),
    ) as executor:
        evaluations = executor.map(
            _evaluate_mtf_worker,
            pending,
            chunksize=1,
        )
        for index, (architecture, evaluation) in enumerate(
            zip(pending, evaluations, strict=True),
            start=1,
        ):
            (
                architecture_results,
                architecture_selections,
                architecture_coverage,
                architecture_blocked,
            ) = evaluation
            result_rows.extend(architecture_results)
            selection_rows.extend(architecture_selections)
            if architecture_coverage is not None:
                coverage_rows.append(architecture_coverage)
            blocked.extend(architecture_blocked)
            completed.add(architecture)
            if (
                index % checkpoint_batch_size == 0
                or index == len(pending)
            ):
                _write_mtf_checkpoint(
                    output,
                    run_signature=run_signature,
                    completed_architectures=completed,
                    result_rows=result_rows,
                    selection_rows=selection_rows,
                    coverage_rows=coverage_rows,
                    blocked=blocked,
                )
    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    cutoff_breaches = _cutoff_coverage_breaches(
        coverage_rows,
        cutoff=cutoff,
    )
    if cutoff_breaches:
        raise RuntimeError(
            "historical cutoff breached by worker coverage: "
            + json.dumps(cutoff_breaches[:10], default=str)
        )
    summary = _summarize(results, selections)
    shortlist = _shortlist(summary, selections)
    _write_frame(output / "nested-results.parquet", results)
    _write_frame(output / "parameter-selections.csv", selections)
    _write_frame(output / "architecture-summary.csv", summary)
    _write_frame(output / "coverage.csv", pd.DataFrame(coverage_rows))
    _write_json(output / "shortlist.json", shortlist)
    _write_json(output / "blocked.json", blocked)
    report = {
        "schema": SCHEMA,
        "status": "GO" if not results.empty else "NO_GO",
        "architecture_count": len(ARCHITECTURES),
        "evaluated_architecture_count": int(
            results["architecture"].nunique()
        )
        if not results.empty
        else 0,
        "global_hypothesis_count": len(ARCHITECTURES) * len(PROFILES),
        "execution_run_count": len(results),
        "selection_count": len(selections),
        "coverage": coverage_rows,
        "shortlist": shortlist,
        "selection_bias_status": "BLOCKED_NO_NEW_INDEPENDENT_HOLDOUT",
        "historical_discovery_only": True,
        "parallel_worker_limit": worker_limit,
        "checkpoint_batch_size": checkpoint_batch_size,
        "resumable_checkpoint": True,
        "research_source_hash": schema["research_source_hash"],
        "historical_data_cutoff": (
            cutoff.isoformat() if cutoff is not None else None
        ),
        "historical_data_hash": historical_data_hash,
        "run_signature": run_signature,
        "schema_hash": _hash(schema),
        **AUTHORITY,
    }
    _write_json(output / "status.json", report)
    _write_json(output / "manifest.json", report)
    return report


def _evaluate_architecture(
    all_frames: Mapping[str, Mapping[str, pd.DataFrame]],
    architecture: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    specification = ARCHITECTURES[architecture]
    lower_timeframe = specification["lower"]
    lower_frames = all_frames.get(lower_timeframe, {})
    if len(lower_frames) < 5:
        return (
            [],
            [],
            None,
            [
                {
                    "architecture": architecture,
                    "reason": "INSUFFICIENT_REAL_LOWER_TIMEFRAME_ASSETS",
                    "instrument_count": len(lower_frames),
                }
            ],
        )
    signals_by_profile = {
        profile: _architecture_signals(
            lower_frames,
            higher_timeframe=specification["higher"],
            intermediate_timeframes=_intermediate_timeframes(
                specification
            ),
            lower_timeframe=lower_timeframe,
            entry_strategy=specification["entry"],
            profile=profile,
        )
        for profile in PROFILES
    }
    start = min(frame.index.min() for frame in lower_frames.values())
    end = min(frame.index.max() for frame in lower_frames.values())
    folds = nested_walk_forward_folds(start, end, lower_timeframe)
    coverage = {
        "architecture": architecture,
        "higher_timeframe": specification["higher"],
        "middle_timeframe": specification.get("middle"),
        "lower_timeframe": lower_timeframe,
        "instrument_count": len(lower_frames),
        "start": start,
        "end": end,
        "fold_count": len(folds),
    }
    if folds.empty:
        return (
            [],
            [],
            coverage,
            [
                {
                    "architecture": architecture,
                    "reason": "INSUFFICIENT_NESTED_WALK_FORWARD_FOLDS",
                }
            ],
        )
    benchmark_frames = {
        symbol: lower_frames[symbol]
        for symbol in BENCHMARK_SYMBOLS
        if symbol in lower_frames
    }
    benchmark_signals = {
        symbol: pd.DataFrame(
            {"signal": True, "score": 1.0},
            index=frame.index,
        )
        for symbol, frame in benchmark_frames.items()
    }
    benchmark_cache: dict[str, Mapping[str, Any]] = {}
    result_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for fold in folds.to_dict("records"):
        validation = []
        for profile in PROFILES:
            validation_run = _run_portfolio(
                lower_frames,
                signals_by_profile[profile],
                start=pd.Timestamp(fold["validation_start"]),
                end=pd.Timestamp(fold["validation_end"]),
                cost_bps=10.0,
            )
            metrics = validation_run["metrics"]
            validation.append(
                (
                    _finite(metrics["period_profit_factor"], -1.0),
                    _finite(metrics["CAGR"], -1.0),
                    profile,
                )
            )
        validation.sort(reverse=True)
        selected = validation[0]
        plateau = _is_plateau(validation)
        selection_rows.append(
            {
                "architecture": architecture,
                "fold_id": fold["fold_id"],
                "higher_timeframe": specification["higher"],
                "middle_timeframe": specification.get("middle"),
                "lower_timeframe": lower_timeframe,
                "selected_profile": selected[2],
                "validation_portfolio_pf": selected[0],
                "validation_CAGR": selected[1],
                "parameter_plateau": plateau,
            }
        )
        fold_id = str(fold["fold_id"])
        benchmark_metrics: Mapping[str, Any] = {}
        if len(benchmark_frames) == len(BENCHMARK_SYMBOLS):
            if fold_id not in benchmark_cache:
                benchmark_cache[fold_id] = _run_portfolio(
                    benchmark_frames,
                    benchmark_signals,
                    start=pd.Timestamp(fold["outer_test_start"]),
                    end=pd.Timestamp(fold["outer_test_end"]),
                    cost_bps=10.0,
                )["metrics"]
            benchmark_metrics = benchmark_cache[fold_id]
        for cost_bps in COSTS_BPS:
            run = _run_portfolio(
                lower_frames,
                signals_by_profile[selected[2]],
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
            metrics = run["metrics"]
            result_rows.append(
                {
                    "architecture": architecture,
                    "fold_id": fold_id,
                    "higher_timeframe": specification["higher"],
                    "middle_timeframe": specification.get("middle"),
                    "lower_timeframe": lower_timeframe,
                    "entry_strategy": specification["entry"],
                    "profile": selected[2],
                    "cost_bps": cost_bps,
                    "parameter_plateau": plateau,
                    "fill_count": len(fills),
                    "turnover_initial_capital": float(
                        notional / 10_000.0
                    ),
                    "excess_CAGR": (
                        float(metrics["CAGR"])
                        - float(benchmark_metrics["CAGR"])
                        if cost_bps == 10.0 and benchmark_metrics
                        else math.nan
                    ),
                    **metrics,
                }
            )
    return result_rows, selection_rows, coverage, []


def _initialize_mtf_worker(
    project_root: str,
    historical_cutoff: str | None,
) -> None:
    global _MTF_WORKER_FRAMES
    frames = _load_frames(Path(project_root))
    cutoff = _historical_cutoff(historical_cutoff)
    _MTF_WORKER_FRAMES = (
        _truncate_frames(frames, cutoff=cutoff)
        if cutoff is not None
        else frames
    )


def _evaluate_mtf_worker(
    architecture: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any] | None,
    list[dict[str, Any]],
]:
    return _evaluate_architecture(_MTF_WORKER_FRAMES, architecture)


def _multitimeframe_frames_fingerprint(
    all_frames: Mapping[str, Mapping[str, pd.DataFrame]],
) -> str:
    rows = []
    for timeframe, frames in sorted(all_frames.items()):
        for symbol, frame in sorted(frames.items()):
            rows.append(
                {
                    "timeframe": timeframe,
                    "symbol": symbol,
                    "row_count": len(frame),
                    "start": (
                        pd.Timestamp(frame.index.min()).isoformat()
                        if not frame.empty
                        else None
                    ),
                    "end": (
                        pd.Timestamp(frame.index.max()).isoformat()
                        if not frame.empty
                        else None
                    ),
                    "content_fingerprint": _frame_content_hash(frame),
                    "provider": frame.attrs.get("provider"),
                }
            )
    return _hash(rows)


def _frame_content_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest().upper()
    canonical = frame.sort_index().sort_index(axis=1)
    try:
        values = pd.util.hash_pandas_object(
            canonical,
            index=True,
            categorize=True,
        ).values.tobytes()
    except TypeError:
        values = canonical.to_json(
            orient="split",
            date_format="iso",
            default_handler=str,
        ).encode("utf-8")
    return hashlib.sha256(values).hexdigest().upper()


def _cutoff_coverage_breaches(
    coverage: list[dict[str, Any]],
    *,
    cutoff: pd.Timestamp | None,
) -> list[dict[str, Any]]:
    if cutoff is None:
        return []
    breaches: list[dict[str, Any]] = []
    for row in coverage:
        try:
            end = pd.Timestamp(row.get("end"))
        except (TypeError, ValueError):
            breaches.append(
                {
                    "architecture": row.get("architecture"),
                    "end": row.get("end"),
                    "reason": "UNPARSEABLE_COVERAGE_END",
                }
            )
            continue
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")
        if end > cutoff:
            breaches.append(
                {
                    "architecture": row.get("architecture"),
                    "end": end.isoformat(),
                    "cutoff": cutoff.isoformat(),
                    "reason": "HISTORICAL_CUTOFF_BREACH",
                }
            )
    return breaches


def _load_mtf_checkpoint(
    output: Path, run_signature: str
) -> dict[str, Any]:
    metadata_path = output / "run-checkpoint.json"
    results_path = output / "run-checkpoint-results.parquet"
    selections_path = output / "run-checkpoint-selections.parquet"
    coverage_path = output / "run-checkpoint-coverage.json"
    blocked_path = output / "run-checkpoint-blocked.json"
    empty: dict[str, Any] = {
        "result_rows": [],
        "selection_rows": [],
        "coverage_rows": [],
        "blocked": [],
        "completed_architectures": [],
    }
    if not all(
        path.exists()
        for path in (
            metadata_path,
            results_path,
            selections_path,
            coverage_path,
            blocked_path,
        )
    ):
        return empty
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if metadata.get("run_signature") != run_signature:
        return empty
    return {
        "result_rows": pd.read_parquet(results_path).to_dict("records"),
        "selection_rows": pd.read_parquet(selections_path).to_dict(
            "records"
        ),
        "coverage_rows": json.loads(
            coverage_path.read_text(encoding="utf-8")
        ),
        "blocked": json.loads(blocked_path.read_text(encoding="utf-8")),
        "completed_architectures": list(
            metadata.get("completed_architectures", [])
        ),
    }


def _write_mtf_checkpoint(
    output: Path,
    *,
    run_signature: str,
    completed_architectures: set[str],
    result_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
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
    _write_json(output / "run-checkpoint-coverage.json", coverage_rows)
    _write_json(output / "run-checkpoint-blocked.json", blocked)
    _write_json(
        output / "run-checkpoint.json",
        {
            "schema": "phase11_10_checkpoint_v1",
            "run_signature": run_signature,
            "completed_architectures": sorted(completed_architectures),
            "architecture_count": len(ARCHITECTURES),
            "result_row_count": len(result_rows),
            "selection_row_count": len(selection_rows),
            "status": (
                "COMPLETE"
                if completed_architectures.issuperset(ARCHITECTURES)
                else "IN_PROGRESS"
            ),
        },
    )


def phase11_10_watchlist(project_root: Path) -> dict[str, Any]:
    status = phase11_10_status(project_root)
    candidates = status.get("shortlist", {}).get("candidates", [])
    if not candidates:
        return {
            "schema": "phase11_10_watchlist_v1",
            "status": "NO_SHORTLIST_CANDIDATE",
            **AUTHORITY,
        }
    selections_path = _output(project_root) / "parameter-selections.csv"
    selections = pd.read_csv(selections_path)
    observed_at = datetime.now(UTC)
    frames = _load_current_frames(
        project_root,
        observed_at=observed_at,
    )
    architecture_watchlists = [
        _current_architecture_watchlist(
            candidate,
            selections=selections,
            frames=frames,
        )
        for candidate in candidates[:5]
        if str(candidate.get("architecture")) in ARCHITECTURES
    ]
    primary = architecture_watchlists[0]
    active_swing = sorted(
        (
            {
                **row,
                "architecture": item["architecture"],
                "higher_timeframe": item["higher_timeframe"],
                "middle_timeframe": item["middle_timeframe"],
                "lower_timeframe": item["lower_timeframe"],
                "entry_strategy": item["entry_strategy"],
                "profile": item["profile"],
                "strategy_id": item["strategy_id"],
            }
            for item in architecture_watchlists
            if item["lower_timeframe"] in {"1h", "2h", "4h"}
            for row in item["observation_candidates"]
        ),
        key=lambda row: (-row["score"], row["symbol"]),
    )
    report = {
        "schema": "phase11_10_watchlist_v1",
        "status": "GO",
        "observed_at": observed_at.isoformat(),
        "frame_selection_policy": (
            "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
        ),
        "architecture": primary["architecture"],
        "higher_timeframe": primary["higher_timeframe"],
        "middle_timeframe": primary["middle_timeframe"],
        "lower_timeframe": primary["lower_timeframe"],
        "entry_strategy": primary["entry_strategy"],
        "profile": primary["profile"],
        "strategy_dna_hash": primary["strategy_dna_hash"],
        "strategy_id": primary["strategy_id"],
        "active_signal_count": primary["active_signal_count"],
        "observation_candidates": primary["observation_candidates"],
        "all_instruments": primary["all_instruments"],
        "architecture_watchlists": architecture_watchlists,
        "architecture_watchlist_count": len(architecture_watchlists),
        "active_swing_observation_candidates": active_swing[:20],
        "active_swing_observation_candidate_count": len(active_swing),
        "phase9_route": "BLOCKED",
        "automatic_submission": False,
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "current-watchlist.json", report)
    return report


def _current_architecture_watchlist(
    candidate: dict[str, Any],
    *,
    selections: pd.DataFrame,
    frames: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Any]:
    architecture = str(candidate["architecture"])
    specification = ARCHITECTURES[architecture]
    selected = selections.loc[
        selections["architecture"].eq(architecture),
        "selected_profile",
    ]
    profile = str(selected.value_counts().index[0])
    signals = _architecture_signals(
        frames[specification["lower"]],
        higher_timeframe=specification["higher"],
        intermediate_timeframes=_intermediate_timeframes(specification),
        lower_timeframe=specification["lower"],
        entry_strategy=specification["entry"],
        profile=profile,
    )
    rows = []
    for symbol, signal in signals.items():
        if signal.empty:
            continue
        latest = signal.iloc[-1]
        previous = signal.iloc[-2] if len(signal) > 1 else latest
        rows.append(
            {
                "symbol": symbol,
                "closed_bar_timestamp": signal.index[-1].isoformat(),
                "active_signal": bool(latest["signal"]),
                "fresh_entry_signal": bool(
                    latest["signal"] and not previous["signal"]
                ),
                "fresh_exit_signal": bool(
                    previous["signal"] and not latest["signal"]
                ),
                "score": _finite(latest["score"], -math.inf),
            }
        )
    active = sorted(
        (row for row in rows if row["active_signal"]),
        key=lambda row: (-row["score"], row["symbol"]),
    )
    return {
        "architecture": architecture,
        "higher_timeframe": specification["higher"],
        "middle_timeframe": specification.get("middle"),
        "lower_timeframe": specification["lower"],
        "entry_strategy": specification["entry"],
        "profile": profile,
        **_deployment_identity(
            architecture=architecture,
            profile=profile,
            specification=specification,
        ),
        "active_signal_count": len(active),
        "observation_candidates": active[:4],
        "all_instruments": rows,
    }


def phase11_10_pit_observe(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    status_path = output / "status.json"
    shortlist_path = output / "shortlist.json"
    selections_path = output / "parameter-selections.csv"
    summary_path = output / "architecture-summary.csv"
    if any(
        not path.exists()
        for path in (
            status_path,
            shortlist_path,
            selections_path,
            summary_path,
        )
    ):
        return {
            "schema": "phase11_10_pit_forward_observation_v1",
            "status": "BLOCKED_QUALIFICATION_MISSING",
            **AUTHORITY,
        }
    status = json.loads(status_path.read_text(encoding="utf-8"))
    qualification_path = output / "qualification" / "current.json"
    qualification = phase11_10_qualification_audit(project_root)
    qualified_source_hash = status.get("research_source_hash")
    current_source_hash = _research_source_hash(project_root)
    if qualification_path.exists() and qualification.get("status") != "GO":
        return {
            "schema": "phase11_10_pit_forward_observation_v1",
            "status": "BLOCKED_QUALIFICATION_MANIFEST_INVALID",
            "qualification_audit": qualification,
            **AUTHORITY,
        }
    if not qualification_path.exists() and (
        not qualified_source_hash
        or qualified_source_hash != current_source_hash
    ):
        return {
            "schema": "phase11_10_pit_forward_observation_v1",
            "status": "BLOCKED_QUALIFICATION_SOURCE_CHANGED",
            "qualified_source_hash": qualified_source_hash,
            "current_source_hash": current_source_hash,
            **AUTHORITY,
        }
    shortlist = json.loads(shortlist_path.read_text(encoding="utf-8"))
    candidates = list(shortlist.get("candidates", []))
    if not candidates:
        return {
            "schema": "phase11_10_pit_forward_observation_v1",
            "status": "NO_ROBUST_SHORTLIST_CANDIDATE",
            **AUTHORITY,
        }
    coverage = {
        str(row["architecture"]): row
        for row in status.get("coverage", [])
        if row.get("architecture")
    }
    summary = pd.read_csv(summary_path)
    selections = pd.read_csv(selections_path)
    observed_at = datetime.now(UTC)
    frames_by_timeframe = _load_pit_frames(
        project_root,
        observed_at=observed_at,
    )
    attestations = _current_pit_attestations(project_root, observed_at)
    qualified_at = datetime.fromtimestamp(
        summary_path.stat().st_mtime,
        UTC,
    ).isoformat()
    source_hash = current_source_hash
    observations = []
    signal_rows = []
    stale_frame_exclusions: list[dict[str, Any]] = []
    for candidate in candidates:
        architecture = str(candidate["architecture"])
        specification = ARCHITECTURES.get(architecture)
        architecture_coverage = coverage.get(architecture)
        if not specification or not architecture_coverage:
            continue
        selected_profiles = selections.loc[
            selections["architecture"].eq(architecture),
            "selected_profile",
        ]
        lower_frames = frames_by_timeframe.get(
            specification["lower"],
            {},
        )
        frame_freshness = {
            symbol: bar_freshness(
                frame.index.max(),
                interval=specification["lower"],
                observed_at=observed_at,
            )
            for symbol, frame in lower_frames.items()
            if not frame.empty
        }
        blocked_symbols = sorted(
            symbol
            for symbol, freshness in frame_freshness.items()
            if freshness["status"] != "FRESH_CLOSED_BAR"
        )
        stale_frame_exclusions.extend(
            {
                "architecture": architecture,
                "symbol": symbol,
                "timeframe": specification["lower"],
                "provider": lower_frames[symbol].attrs.get("provider"),
                **frame_freshness[symbol],
            }
            for symbol in blocked_symbols
        )
        lower_frames = {
            symbol: frame
            for symbol, frame in lower_frames.items()
            if frame_freshness.get(symbol, {}).get("status")
            == "FRESH_CLOSED_BAR"
        }
        if selected_profiles.empty or not lower_frames:
            continue
        profile = str(selected_profiles.mode().iloc[0])
        common_close = min(
            frame.index.max() for frame in lower_frames.values()
        )
        training_data_end = pd.Timestamp(
            architecture_coverage["end"]
        )
        signals = _architecture_signals(
            lower_frames,
            higher_timeframe=specification["higher"],
            intermediate_timeframes=_intermediate_timeframes(
                specification
            ),
            lower_timeframe=specification["lower"],
            entry_strategy=specification["entry"],
            profile=profile,
        )
        active: list[dict[str, Any]] = []
        for symbol, signal in signals.items():
            closed = signal.loc[signal.index <= common_close]
            if closed.empty:
                continue
            latest = closed.iloc[-1]
            if bool(latest["signal"]):
                active.append(
                    {
                        "symbol": symbol,
                        "score": _finite(latest["score"], -math.inf),
                    }
                )
        active = sorted(
            active,
            key=lambda row: (-float(str(row["score"])), str(row["symbol"])),
        )[:4]
        weight = 1.0 / len(active) if active else 0.0
        raw_targets = {
            str(row["symbol"]): weight for row in active
        }
        attested_targets = {
            symbol: target_weight
            for symbol, target_weight in raw_targets.items()
            if symbol in attestations
        }
        active_scores = {
            str(row["symbol"]): float(str(row["score"])) for row in active
        }
        providers = sorted(
            {
                str(frame.attrs.get("provider", "UNKNOWN"))
                for frame in lower_frames.values()
            }
        )
        provider_selections = {
            symbol: {
                "selected_provider": frame.attrs.get("provider"),
                "selection_policy": frame.attrs.get(
                    "provider_selection_policy",
                    "DAILY_OR_DERIVED_PROVIDER_POLICY",
                ),
                "candidate_providers": frame.attrs.get(
                    "provider_candidates",
                    [],
                ),
                "last_timestamp": pd.Timestamp(
                    frame.index.max()
                ).isoformat(),
            }
            for symbol, frame in sorted(lower_frames.items())
        }
        data_fingerprints = sorted(
            {
                str(frame.attrs.get("content_fingerprint", "UNKNOWN"))
                for frame in lower_frames.values()
            }
        )
        identity = _deployment_identity(
            architecture=architecture,
            profile=profile,
            specification=specification,
        )
        qualification_contract = {
            "schema": SCHEMA,
            "architecture": architecture,
            "profile": profile,
            **identity,
            "source_hash": source_hash,
            "schema_hash": status.get("schema_hash"),
            "training_data_end": training_data_end.isoformat(),
            "summary_evidence": candidate,
        }
        observations.append(
            {
                **identity,
                "version": "PHASE11_10_MTF_V1",
                "architecture": architecture,
                "entry_strategy": specification["entry"],
                "profile": profile,
                "higher_timeframe": specification["higher"],
                "middle_timeframe": specification.get("middle"),
                "lower_timeframe": specification["lower"],
                "allowed_timeframes": [
                    value
                    for value in (
                        specification.get("lower"),
                        specification.get("middle"),
                        specification.get("higher"),
                    )
                    if value
                ],
                "source_hash": source_hash,
                "parameter_hash": _hash(
                    {
                        "profile": profile,
                        "specification": specification,
                    }
                ),
                "qualification_hash": _hash(qualification_contract),
                "qualified_at": qualified_at,
                "training_data_end": training_data_end.isoformat(),
                "closed_bar_timestamp": pd.Timestamp(
                    common_close
                ).isoformat(),
                "independent_forward_session": (
                    pd.Timestamp(common_close) > training_data_end
                ),
                "raw_target_weights": raw_targets,
                "current_attested_target_weights": attested_targets,
                "current_attestation_count": len(attested_targets),
                "provider_names": providers,
                "provider_selection_policy": (
                    "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
                    if specification["lower"] in {"1h", "2h", "4h"}
                    else "DAILY_OR_DERIVED_PROVIDER_POLICY"
                ),
                "provider_selections": provider_selections,
                "data_freshness_status": "FRESH_CLOSED_BAR",
                "stale_symbol_count": len(blocked_symbols),
                "provider_continuity_status": (
                    _provider_continuity_status(providers)
                ),
                "data_fingerprint_hash": _hash(data_fingerprints),
                "qualification_status": "ROBUST_SHORTLIST_FROZEN",
                "observation_status": "PIT_OBSERVATION_COMPLETE",
                "execution_route": "BLOCKED",
                "automatic_orders": 0,
                "broker_calls": 0,
            }
        )
        for symbol in sorted(attested_targets):
            signal_rows.append(
                _pit_signal_record(
                    identity=identity,
                    specification=specification,
                    profile=profile,
                    symbol=symbol,
                    frame=lower_frames[symbol],
                    bar_timestamp=pd.Timestamp(common_close),
                    observed_at=observed_at,
                    score=active_scores[symbol],
                )
            )
    payload = {
        "schema": "phase11_10_pit_forward_observation_v1",
        "status": "GO" if observations else "NO_GO",
        "observed_at": observed_at.isoformat(),
        "attested_symbols": sorted(attestations),
        "candidate_count": len(candidates),
        "observation_count": len(observations),
        "independent_observation_count": sum(
            bool(row["independent_forward_session"])
            for row in observations
        ),
        "pit_eligible_observation_count": sum(
            bool(row["independent_forward_session"])
            and bool(row["current_attested_target_weights"])
            and row["provider_continuity_status"]
            == "SAME_PRIMARY_PROVIDER_GO"
            for row in observations
        ),
        "signal_count": len(signal_rows),
        "freshness_gate_status": (
            "GO" if observations else "NO_FRESH_OBSERVATIONS"
        ),
        "stale_frame_exclusion_count": len(stale_frame_exclusions),
        "stale_frame_exclusions": stale_frame_exclusions,
        "observations": observations,
        **AUTHORITY,
    }
    broad_shadow = _ten_bps_positive_shadow_observation(
        summary=summary,
        selections=selections,
        frames_by_timeframe=frames_by_timeframe,
        observed_at=observed_at,
        attestations=attestations,
    )
    payload["ten_bps_positive_candidate_count"] = broad_shadow[
        "candidate_count"
    ]
    payload["ten_bps_positive_active_signal_count"] = broad_shadow[
        "active_signal_count"
    ]
    payload["ten_bps_positive_evaluated_asset_count"] = broad_shadow[
        "evaluated_asset_count"
    ]
    content_hash = _hash(payload)
    _write_json(
        output / "pit-forward-observations" / f"{content_hash}.json",
        payload,
    )
    _write_json(output / "latest-pit-forward-observation.json", payload)
    signals_path = (
        project_root / "output" / "signals" / "pit_mtf_signals.json"
    )
    _write_json(signals_path, signal_rows)
    _write_json(
        output / "ten-bps-positive-shadow-observation.json",
        broad_shadow,
    )
    _write_json(
        project_root
        / "output"
        / "signals"
        / "pit_mtf_research_signals.json",
        broad_shadow["signals"],
    )
    return payload


def _load_pit_frames(
    project_root: Path,
    *,
    observed_at: datetime,
) -> dict[str, dict[str, pd.DataFrame]]:
    private_root = (
        project_root
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
    )
    if not private_root.is_dir():
        return _load_frames(project_root)
    return _load_current_frames(
        project_root,
        observed_at=observed_at,
    )


def _provider_continuity_status(providers: list[str]) -> str:
    return (
        "SAME_PRIMARY_PROVIDER_GO"
        if len(providers) == 1
        and providers[0] in PIT_PRIMARY_PROVIDERS
        else "PROVIDER_CONTINUITY_BLOCKED"
    )


def _ten_bps_positive_shadow_observation(
    *,
    summary: pd.DataFrame,
    selections: pd.DataFrame,
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
    observed_at: datetime,
    attestations: set[str],
) -> dict[str, Any]:
    required = {
        "architecture",
        "median_oos_CAGR",
        "median_oos_portfolio_pf",
    }
    if not required.issubset(summary.columns):
        candidates = summary.iloc[0:0]
    else:
        candidates = summary.loc[
            summary["median_oos_CAGR"].gt(0)
            & summary["median_oos_portfolio_pf"].gt(1)
        ].sort_values(
            ["median_oos_portfolio_pf", "median_oos_CAGR"],
            ascending=False,
        )
    signal_rows: list[dict[str, Any]] = []
    architecture_rows: list[dict[str, Any]] = []
    stale_exclusions: list[dict[str, Any]] = []
    evaluated_assets = 0
    for _, candidate in candidates.iterrows():
        architecture = str(candidate["architecture"])
        specification = ARCHITECTURES.get(architecture)
        if not specification:
            continue
        selected_profiles = selections.loc[
            selections["architecture"].eq(architecture),
            "selected_profile",
        ]
        lower_frames = frames_by_timeframe.get(
            specification["lower"],
            {},
        )
        freshness = {
            symbol: bar_freshness(
                frame.index.max(),
                interval=specification["lower"],
                observed_at=observed_at,
            )
            for symbol, frame in lower_frames.items()
            if not frame.empty
        }
        stale_exclusions.extend(
            {
                "architecture": architecture,
                "symbol": symbol,
                "timeframe": specification["lower"],
                "provider": lower_frames[symbol].attrs.get("provider"),
                **details,
            }
            for symbol, details in freshness.items()
            if details["status"] != "FRESH_CLOSED_BAR"
        )
        lower_frames = {
            symbol: frame
            for symbol, frame in lower_frames.items()
            if freshness.get(symbol, {}).get("status")
            == "FRESH_CLOSED_BAR"
        }
        if selected_profiles.empty or not lower_frames:
            continue
        profile = str(selected_profiles.mode().iloc[0])
        common_close = min(
            frame.index.max() for frame in lower_frames.values()
        )
        signals = _architecture_signals(
            lower_frames,
            higher_timeframe=specification["higher"],
            intermediate_timeframes=_intermediate_timeframes(
                specification
            ),
            lower_timeframe=specification["lower"],
            entry_strategy=specification["entry"],
            profile=profile,
        )
        identity = _deployment_identity(
            architecture=architecture,
            profile=profile,
            specification=specification,
        )
        active_symbols: list[str] = []
        evaluated_assets += len(signals)
        for symbol, signal in sorted(signals.items()):
            closed = signal.loc[signal.index <= common_close]
            if closed.empty or not bool(closed.iloc[-1]["signal"]):
                continue
            active_symbols.append(symbol)
            record = _pit_signal_record(
                identity=identity,
                specification=specification,
                profile=profile,
                symbol=symbol,
                frame=lower_frames[symbol],
                bar_timestamp=pd.Timestamp(common_close),
                observed_at=observed_at,
                score=_finite(
                    closed.iloc[-1]["score"],
                    -math.inf,
                ),
            )
            attested = symbol in attestations
            record.update(
                {
                    "architecture": architecture,
                    "entry_strategy": specification["entry"],
                    "qualification_status": (
                        "TEN_BPS_POSITIVE_RESEARCH"
                    ),
                    "research_only": True,
                    "current_pit_attested": attested,
                    "execution_route": "BLOCKED",
                    "median_oos_CAGR": _finite(
                        candidate["median_oos_CAGR"],
                        0.0,
                    ),
                    "median_oos_portfolio_pf": _finite(
                        candidate["median_oos_portfolio_pf"],
                        0.0,
                    ),
                    "cost_50bps_median_pf": _finite(
                        candidate.get("cost_50bps_median_pf"),
                        0.0,
                    ),
                }
            )
            record["reasons"] = [
                "POSITIVE_MEDIAN_OOS_CAGR_AT_10BPS",
                "POSITIVE_MEDIAN_PORTFOLIO_PF_AT_10BPS",
                "HIGHER_TIMEFRAME_GATE_CONFIRMED",
                (
                    "CURRENT_PIT_ATTESTATION_GO"
                    if attested
                    else "CURRENT_PIT_ATTESTATION_NOT_AVAILABLE"
                ),
            ]
            signal_rows.append(record)
        architecture_rows.append(
            {
                "architecture": architecture,
                **identity,
                "entry_strategy": specification["entry"],
                "higher_timeframe": specification["higher"],
                "middle_timeframe": specification.get("middle"),
                "lower_timeframe": specification["lower"],
                "profile": profile,
                "evaluated_asset_count": len(signals),
                "active_asset_count": len(active_symbols),
                "active_assets": active_symbols,
                "median_oos_CAGR": _finite(
                    candidate["median_oos_CAGR"],
                    0.0,
                ),
                "median_oos_portfolio_pf": _finite(
                    candidate["median_oos_portfolio_pf"],
                    0.0,
                ),
                "cost_50bps_median_pf": _finite(
                    candidate.get("cost_50bps_median_pf"),
                    0.0,
                ),
            }
        )
    signal_rows.sort(
        key=lambda row: (
            str(row["architecture"]),
            str(row["ticker"]),
        )
    )
    return {
        "schema": "phase11_10_ten_bps_positive_shadow_v1",
        "status": "GO",
        "observed_at": observed_at.isoformat(),
        "selection_rule": (
            "MEDIAN_OOS_CAGR_GT_0_AND_MEDIAN_OOS_PORTFOLIO_PF_GT_1"
            "_AT_10BPS"
        ),
        "candidate_count": len(architecture_rows),
        "evaluated_asset_count": evaluated_assets,
        "active_signal_count": len(signal_rows),
        "freshness_gate_status": "GO",
        "stale_frame_exclusion_count": len(stale_exclusions),
        "stale_frame_exclusions": stale_exclusions,
        "architectures": architecture_rows,
        "signals": signal_rows,
        "research_only": True,
        "automatic_execution_allowed": False,
        **AUTHORITY,
    }


def phase11_10_status(project_root: Path) -> dict[str, Any]:
    path = _output(project_root) / "status.json"
    if not path.exists():
        return {"schema": SCHEMA, "status": "NOT_RUN", **AUTHORITY}
    return json.loads(path.read_text(encoding="utf-8"))


def phase11_10_qualification_audit(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    current_path = output / "qualification" / "current.json"
    status = phase11_10_status(project_root)
    current_source_hash = _research_source_hash(project_root)
    if not current_path.exists():
        return {
            "schema": "phase11_10_qualification_audit_v1",
            "status": "BLOCKED_QUALIFICATION_MANIFEST_MISSING",
            "qualified_source_hash": status.get("research_source_hash"),
            "current_source_hash": current_source_hash,
            "change_classification": (
                "RESEARCH_SOURCE_CHANGED_REPLAY_REQUIRED"
                if status.get("research_source_hash") != current_source_hash
                else "LEGACY_RESEARCH_OUTPUT_REQUIRES_MANIFEST"
            ),
            **AUTHORITY,
        }
    manifest = json.loads(current_path.read_text(encoding="utf-8"))
    immutable_path = output / str(manifest.get("immutable_manifest_path", ""))
    failures: list[str] = []
    if not immutable_path.is_file():
        failures.append("IMMUTABLE_MANIFEST_MISSING")
    elif _file_sha256(immutable_path) != manifest.get("immutable_manifest_hash"):
        failures.append("IMMUTABLE_MANIFEST_HASH_CHANGED")
    if manifest.get("code_commit_hash") != current_source_hash:
        failures.append("QUALIFICATION_SOURCE_CHANGED")
    result_hash = _research_result_hash(output)
    if manifest.get("research_result_hash") != result_hash:
        failures.append("RESEARCH_RESULT_HASH_CHANGED")
    if status.get("historical_data_hash") != manifest.get(
        "historical_data_hash"
    ):
        failures.append("HISTORICAL_DATA_HASH_CHANGED")
    report = {
        "schema": "phase11_10_qualification_audit_v1",
        "status": "GO" if not failures else "BLOCKED",
        "qualification_id": manifest.get("qualification_id"),
        "qualification_manifest_reproducible": not failures,
        "old_freeze_immutable": not any(
            failure.startswith("IMMUTABLE_MANIFEST") for failure in failures
        ),
        "runtime_bars_part_of_research_hash": False,
        "historical_data_cutoff": manifest.get("historical_data_cutoff"),
        "historical_data_hash": manifest.get("historical_data_hash"),
        "research_result_hash": result_hash,
        "failures": failures,
        **AUTHORITY,
    }
    _write_json(output / "qualification" / "audit.json", report)
    return report


def phase11_10_qualification_freeze(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    status = phase11_10_status(project_root)
    current_source_hash = _research_source_hash(project_root)
    required = (
        "architecture-summary.csv",
        "parameter-selections.csv",
        "nested-results.parquet",
        "shortlist.json",
        "status.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    blockers: list[str] = []
    if missing:
        blockers.append("RESEARCH_ARTIFACTS_MISSING")
    if status.get("status") != "GO":
        blockers.append("RESEARCH_STATUS_NOT_GO")
    if status.get("research_source_hash") != current_source_hash:
        blockers.append("RESEARCH_SOURCE_REPLAY_REQUIRED")
    if not status.get("historical_data_hash"):
        blockers.append("HISTORICAL_DATA_HASH_MISSING_REPLAY_REQUIRED")
    if not status.get("historical_data_cutoff"):
        blockers.append("HISTORICAL_DATA_CUTOFF_MISSING_REPLAY_REQUIRED")
    checkpoint = _read_json(output / "run-checkpoint.json")
    if checkpoint.get("status") != "COMPLETE":
        blockers.append("REPRODUCIBLE_CHECKPOINT_INCOMPLETE")
    if blockers:
        return {
            "schema": "phase11_10_qualification_manifest_v2",
            "status": "BLOCKED",
            "blockers": blockers,
            "missing_artifacts": missing,
            "required_replay_command": (
                "python main.py research phase11-10 run "
                "--historical-cutoff <FROZEN_CUTOFF>"
            ),
            **AUTHORITY,
        }
    source_components = _research_source_components(project_root)
    qualified_ids = sorted(
        {
            str(row.get("strategy_id"))
            for row in status.get("shortlist", {}).get("candidates", [])
            if row.get("strategy_id")
        }
    )
    semantic = {
        "schema": "phase11_10_qualification_manifest_v2",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit_hash": current_source_hash,
        "code_identity_type": "RESEARCH_SOURCE_CONTENT_HASH",
        "source_component_hashes": source_components,
        "strategy_catalog_hash": _hash(ARCHITECTURES),
        "parameter_hash": _file_sha256(output / "parameter-selections.csv"),
        "universe_hash": _hash(status.get("coverage", [])),
        "historical_data_cutoff": status["historical_data_cutoff"],
        "historical_data_hash": status["historical_data_hash"],
        "cost_model_version": "PHASE11_8_EUR_WHOLE_SHARE_COST_MODEL_V1",
        "cost_stress_bps_per_side": list(COSTS_BPS),
        "research_result_hash": _research_result_hash(output),
        "research_result_hash_policy": _RESEARCH_RESULT_HASH_POLICY,
        "research_result_artifacts": list(_RESEARCH_RESULT_ARTIFACTS),
        "qualified_strategy_ids": qualified_ids,
        "runtime_state_excluded": True,
        "new_market_bars_change_research_hash": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
    }
    qualification_id = "QUAL-" + _hash(semantic)[:24]
    immutable_relative = (
        Path("qualification") / "freezes" / f"{qualification_id}.json"
    )
    immutable_path = output / immutable_relative
    manifest = {
        **semantic,
        "qualification_id": qualification_id,
        "status": "QUALIFICATION_FROZEN_GO",
    }
    immutable_path.parent.mkdir(parents=True, exist_ok=True)
    if immutable_path.exists():
        existing = json.loads(immutable_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("immutable qualification manifest collision")
    else:
        _write_json(immutable_path, manifest)
    pointer = {
        **manifest,
        "immutable_manifest_path": immutable_relative.as_posix(),
        "immutable_manifest_hash": _file_sha256(immutable_path),
    }
    _write_json(output / "qualification" / "current.json", pointer)
    return phase11_10_qualification_audit(project_root)


def phase11_10_reclassify(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    summary_path = output / "architecture-summary.csv"
    status_path = output / "status.json"
    if not summary_path.exists() or not status_path.exists():
        return {
            "schema": SCHEMA,
            "status": "NO_RESEARCH_RESULTS",
            **AUTHORITY,
        }
    summary = pd.read_csv(summary_path)
    report = json.loads(status_path.read_text(encoding="utf-8"))
    selections_path = output / "parameter-selections.csv"
    selections = (
        pd.read_csv(selections_path)
        if selections_path.exists()
        else None
    )
    report["shortlist"] = _shortlist(summary, selections)
    report["classification_policy"] = (
        "TWO_TIER_RESEARCH_AND_ROBUST_SHORTLIST"
    )
    report["status"] = "GO"
    _write_json(output / "shortlist.json", report["shortlist"])
    _write_json(status_path, report)
    _write_json(output / "manifest.json", report)
    return report


def phase11_10_top20(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    summary_path = output / "architecture-summary.csv"
    selections_path = output / "parameter-selections.csv"
    if not summary_path.exists():
        return {
            "schema": "phase11_10_top20_v1",
            "status": "NO_RESEARCH_RESULTS",
            **AUTHORITY,
        }
    summary = _score_summary(pd.read_csv(summary_path))
    if selections_path.exists():
        selections = pd.read_csv(selections_path)
        modal_profiles = {
            str(architecture): str(group["selected_profile"].mode().iloc[0])
            for architecture, group in selections.groupby("architecture")
            if not group["selected_profile"].mode().empty
        }
        summary["deployment_profile"] = [
            modal_profiles.get(
                str(row["architecture"]),
                "balanced",
            )
            for _, row in summary.iterrows()
        ]
        exact_identities = [
            _deployment_identity(
                architecture=str(row["architecture"]),
                profile=str(row["deployment_profile"]),
                specification=ARCHITECTURES[str(row["architecture"])],
            )
            for _, row in summary.iterrows()
        ]
        summary["strategy_id"] = [
            item["strategy_id"] for item in exact_identities
        ]
        summary["strategy_dna_hash"] = [
            item["strategy_dna_hash"] for item in exact_identities
        ]
    else:
        summary["deployment_profile"] = "UNAVAILABLE"
    duplicate_audit = _economic_outcome_audit(output)
    fingerprints = {
        str(row["architecture"]): str(row["outcome_fingerprint"])
        for row in duplicate_audit["architectures"]
    }
    summary["economic_outcome_fingerprint"] = [
        fingerprints.get(str(row["architecture"]))
        for _, row in summary.iterrows()
    ]
    ranked = _rank_viable_strategies(summary, limit=20)
    report = {
        "schema": "phase11_10_top20_v1",
        "status": "GO",
        "ranking_method": (
            "HARD_VETO_EXCLUDED_THEN_WEIGHTED_EVIDENCE_THEN_OOS_SHARPE"
            "_THEN_PORTFOLIO_PF"
        ),
        "hard_vetoed_strategy_count": int(
            summary["hard_veto_reasons"].map(len).gt(0).sum()
        ),
        "strategy_count": len(ranked),
        "strategies": _json_records(ranked),
        "contains_1h_execution_timeframe": bool(
            ranked["lower_timeframe"].eq("1h").any()
        ),
        "contains_4h_execution_timeframe": bool(
            ranked["lower_timeframe"].eq("4h").any()
        ),
        "independent_economic_outcome_count": duplicate_audit[
            "independent_economic_outcome_count"
        ],
        "duplicate_economic_groups": duplicate_audit[
            "duplicate_groups"
        ],
        "authority_granted": False,
        **AUTHORITY,
    }
    _write_json(output / "top20-strategies.json", report)
    _write_json(
        output / "economic-outcome-duplicate-audit.json",
        duplicate_audit,
    )
    _write_json(
        output / "deployment-map.json",
        {
            "schema": "phase11_10_deployment_map_v1",
            "status": "GO",
            "strategy_count": len(ranked),
            "strategies": [
                {
                    "strategy_id": row["strategy_id"],
                    "strategy_dna_hash": row["strategy_dna_hash"],
                    "architecture": row["architecture"],
                    "higher_timeframe": row["higher_timeframe"],
                    "middle_timeframe": row["middle_timeframe"],
                    "lower_timeframe": row["lower_timeframe"],
                    "entry_strategy": row["entry_strategy"],
                    "evidence_tier": row["evidence_tier"],
                    "paper_authority": False,
                    "live_authority": False,
                }
                for row in _json_records(ranked)
            ],
            **AUTHORITY,
        },
    )
    return report


def _economic_outcome_audit(output: Path) -> dict[str, Any]:
    results_path = output / "nested-results.parquet"
    if not results_path.exists():
        return {
            "schema": "phase11_10_economic_outcome_duplicate_audit_v1",
            "status": "NO_RESULTS",
            "architecture_count": 0,
            "independent_economic_outcome_count": 0,
            "duplicate_groups": [],
            "architectures": [],
        }
    results = pd.read_parquet(results_path)
    metric_columns = [
        "fold_id",
        "cost_bps",
        "CAGR",
        "Sharpe",
        "maximum_drawdown",
        "period_profit_factor",
        "terminal_nav",
        "fill_count",
        "turnover_initial_capital",
    ]
    rows = []
    groups: dict[str, list[str]] = {}
    for architecture, group in results.groupby("architecture"):
        normalized = group[metric_columns].sort_values(
            ["fold_id", "cost_bps"]
        )
        for column in normalized.select_dtypes(include=[np.number]):
            normalized[column] = normalized[column].round(10)
        fingerprint = _hash(_json_records(normalized))
        architecture_name = str(architecture)
        groups.setdefault(fingerprint, []).append(architecture_name)
        rows.append(
            {
                "architecture": architecture_name,
                "outcome_fingerprint": fingerprint,
            }
        )
    duplicates = [
        {
            "outcome_fingerprint": fingerprint,
            "architectures": sorted(architectures),
            "duplicate_count": len(architectures),
        }
        for fingerprint, architectures in groups.items()
        if len(architectures) > 1
    ]
    return {
        "schema": "phase11_10_economic_outcome_duplicate_audit_v1",
        "status": "GO",
        "architecture_count": len(rows),
        "independent_economic_outcome_count": len(groups),
        "duplicate_group_count": len(duplicates),
        "duplicate_groups": sorted(
            duplicates,
            key=lambda row: row["architectures"],
        ),
        "architectures": sorted(
            rows,
            key=lambda row: row["architecture"],
        ),
        **AUTHORITY,
    }


def _architecture_signals(
    lower_frames: Mapping[str, pd.DataFrame],
    *,
    higher_timeframe: str,
    intermediate_timeframes: tuple[str, ...] = (),
    lower_timeframe: str,
    entry_strategy: str,
    profile: str,
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for symbol, lower in lower_frames.items():
        lower_work = lower.copy(deep=False)
        lower_work.attrs["symbol"] = symbol
        entry = _entry_signals(
            lower_work,
            lower_frames=lower_frames,
            symbol=symbol,
            entry_strategy=entry_strategy,
            lower_timeframe=lower_timeframe,
            profile=profile,
        )
        signal = entry["signal"].copy()
        momentum_contexts = []
        unavailable = False
        for gate_timeframe in (
            *intermediate_timeframes,
            higher_timeframe,
        ):
            higher = _resample_from_lower(lower, gate_timeframe)
            if higher.empty:
                unavailable = True
                break
            higher_work = higher.copy(deep=False)
            higher_work.attrs["symbol"] = symbol
            gate = _strategy(
                higher_work,
                "triple_ma_trend",
                gate_timeframe,
                profile,
            )["signal"].shift(1)
            available_gate = (
                gate.reindex(lower.index, method="ffill")
                .astype("boolean")
                .fillna(False)
                .astype(bool)
            )
            signal &= available_gate
            higher_momentum = higher["close"].pct_change(
                max(
                    2,
                    int(
                        _parameters(gate_timeframe, profile)[
                            "channel"
                        ]
                    ),
                )
            )
            momentum_contexts.append(
                higher_momentum.shift(1).reindex(
                    lower.index,
                    method="ffill",
                )
            )
        if unavailable:
            continue
        available_momentum = pd.concat(
            momentum_contexts,
            axis=1,
        ).mean(axis=1)
        score = entry["score"].replace(-math.inf, np.nan).add(
            available_momentum,
            fill_value=0.0,
        )
        output[symbol] = pd.DataFrame(
            {
                "signal": signal.fillna(False).astype(bool),
                "score": score.replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(-math.inf),
            },
            index=lower.index,
        )
    return output


def _entry_signals(
    frame: pd.DataFrame,
    *,
    lower_frames: Mapping[str, pd.DataFrame],
    symbol: str,
    entry_strategy: str,
    lower_timeframe: str,
    profile: str,
) -> pd.DataFrame:
    if entry_strategy == "adaptive_percentile_pullback":
        return _adaptive_percentile_pullback(
            frame,
            lower_timeframe=lower_timeframe,
            profile=profile,
        )
    if entry_strategy == "adaptive_volatility_breakout":
        return _adaptive_volatility_breakout(
            frame,
            lower_timeframe=lower_timeframe,
            profile=profile,
        )
    if entry_strategy == "beta_residual_pullback":
        return _beta_residual_pullback(
            frame,
            market=lower_frames.get("SPY"),
            symbol=symbol,
            lower_timeframe=lower_timeframe,
            profile=profile,
        )
    return _strategy(
        frame,
        entry_strategy,
        lower_timeframe,
        profile,
    )


def _adaptive_percentile_pullback(
    frame: pd.DataFrame,
    *,
    lower_timeframe: str,
    profile: str,
) -> pd.DataFrame:
    parameters = _parameters(lower_timeframe, profile)
    close = frame["close"]
    slow = int(parameters["slow"])
    channel = int(parameters["channel"])
    quantile = {
        "responsive": 0.20,
        "balanced": 0.15,
        "conservative": 0.10,
    }[profile]
    shock_lookback = max(1, channel // 10)
    distribution_lookback = max(slow, channel * 4)
    shock = close.pct_change(shock_lookback)
    minimum = max(30, distribution_lookback // 2)
    downside_threshold = (
        shock.rolling(distribution_lookback, min_periods=minimum)
        .quantile(quantile)
        .shift(1)
    )
    center = (
        shock.rolling(distribution_lookback, min_periods=minimum)
        .median()
        .shift(1)
    )
    trend = close.gt(close.rolling(slow).mean())
    recovery = close.gt(close.shift(1))
    entry = trend & shock.le(downside_threshold) & recovery
    state = _persistent(entry, shock.ge(center) | ~trend)
    scale = (
        shock.rolling(distribution_lookback, min_periods=minimum)
        .std()
        .shift(1)
        .replace(0, np.nan)
    )
    score = downside_threshold.sub(shock).div(scale)
    return _signal_frame(state, score)


def _adaptive_volatility_breakout(
    frame: pd.DataFrame,
    *,
    lower_timeframe: str,
    profile: str,
) -> pd.DataFrame:
    parameters = _parameters(lower_timeframe, profile)
    close = frame["close"]
    high = frame["high"]
    channel = int(parameters["channel"])
    slow = int(parameters["slow"])
    quantile = {
        "responsive": 0.25,
        "balanced": 0.20,
        "conservative": 0.15,
    }[profile]
    returns = close.pct_change()
    short_volatility = returns.rolling(max(3, channel // 2)).std()
    long_volatility = returns.rolling(max(10, channel * 2)).std()
    ratio = short_volatility.div(long_volatility.replace(0, np.nan))
    distribution_lookback = max(slow, channel * 4)
    minimum = max(30, distribution_lookback // 2)
    contraction_threshold = (
        ratio.rolling(distribution_lookback, min_periods=minimum)
        .quantile(quantile)
        .shift(1)
    )
    contraction = ratio.le(contraction_threshold)
    breakout = close.gt(high.rolling(channel).max().shift(1))
    mean = close.rolling(channel).mean()
    state = _persistent(contraction & breakout, close.lt(mean))
    score = close.div(high.rolling(channel).max().shift(1)).sub(1).add(
        contraction_threshold.sub(ratio),
        fill_value=0.0,
    )
    return _signal_frame(state, score)


def _beta_residual_pullback(
    frame: pd.DataFrame,
    *,
    market: pd.DataFrame | None,
    symbol: str,
    lower_timeframe: str,
    profile: str,
) -> pd.DataFrame:
    equity_symbols = {
        "AAPL",
        "AMZN",
        "ASML",
        "EEM",
        "EFA",
        "GOOGL",
        "INTC",
        "IWM",
        "JPM",
        "META",
        "MSFT",
        "NVDA",
        "ON",
        "QQQ",
        "XOM",
    }
    if market is None or symbol.upper() not in equity_symbols:
        return _signal_frame(
            pd.Series(False, index=frame.index),
            pd.Series(-math.inf, index=frame.index),
        )
    parameters = _parameters(lower_timeframe, profile)
    close = frame["close"]
    market_close = market["close"].reindex(frame.index)
    asset_return = close.pct_change()
    market_return = market_close.pct_change(fill_method=None)
    slow = int(parameters["slow"])
    channel = int(parameters["channel"])
    distribution_lookback = max(slow, channel * 4)
    minimum = max(30, distribution_lookback // 2)
    beta = asset_return.rolling(
        distribution_lookback,
        min_periods=minimum,
    ).cov(market_return).div(
        market_return.rolling(
            distribution_lookback,
            min_periods=minimum,
        )
        .var()
        .replace(0, np.nan)
    )
    residual = asset_return.sub(beta.mul(market_return))
    quantile = {
        "responsive": 0.20,
        "balanced": 0.15,
        "conservative": 0.10,
    }[profile]
    downside_threshold = (
        residual.rolling(
            distribution_lookback,
            min_periods=minimum,
        )
        .quantile(quantile)
        .shift(1)
    )
    center = (
        residual.rolling(
            distribution_lookback,
            min_periods=minimum,
        )
        .median()
        .shift(1)
    )
    trend = close.gt(close.rolling(slow).mean())
    entry = trend & residual.le(downside_threshold)
    state = _persistent(entry, residual.ge(center) | ~trend)
    residual_scale = (
        residual.rolling(
            distribution_lookback,
            min_periods=minimum,
        )
        .std()
        .shift(1)
        .replace(0, np.nan)
    )
    score = downside_threshold.sub(residual).div(residual_scale)
    return _signal_frame(state, score)


def _resample_from_lower(
    frame: pd.DataFrame,
    higher_timeframe: str,
) -> pd.DataFrame:
    if higher_timeframe == "4h":
        frequency = "4h"
    elif higher_timeframe == "1d":
        frequency = "D"
    elif higher_timeframe == "1w":
        frequency = "W-FRI"
    elif higher_timeframe == "1mo":
        frequency = "ME"
    else:
        raise ValueError(
            f"UNSUPPORTED_HIGHER_TIMEFRAME:{higher_timeframe}"
        )
    return _aggregate({"ASSET": frame}, frequency).get(
        "ASSET",
        pd.DataFrame(columns=frame.columns),
    )


def _intermediate_timeframes(
    specification: Mapping[str, str],
) -> tuple[str, ...]:
    middle = specification.get("middle")
    return () if middle is None else (middle,)


def _is_plateau(validation: list[tuple[float, float, str]]) -> bool:
    if len(validation) < 2:
        return False
    best, neighbor = validation[0], validation[1]
    return (
        best[0] > 1.0
        and neighbor[0] > 1.0
        and abs(best[0] - neighbor[0]) / max(abs(best[0]), 1e-9) <= 0.20
    )


def _summarize(
    results: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    normal = results.loc[results["cost_bps"].eq(10.0)]
    for architecture, group in normal.groupby("architecture"):
        stress = results.loc[
            results["architecture"].eq(architecture)
            & results["cost_bps"].eq(50.0)
        ]
        selected = selections.loc[
            selections["architecture"].eq(architecture)
        ]
        pfs = group["period_profit_factor"].replace(
            [np.inf, -np.inf], np.nan
        )
        stress_pfs = stress["period_profit_factor"].replace(
            [np.inf, -np.inf], np.nan
        )
        rows.append(
            {
                "architecture": architecture,
                "higher_timeframe": group["higher_timeframe"].iloc[0],
                "middle_timeframe": group["middle_timeframe"].iloc[0],
                "lower_timeframe": group["lower_timeframe"].iloc[0],
                "entry_strategy": group["entry_strategy"].iloc[0],
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
                "median_excess_CAGR": float(
                    group["excess_CAGR"].median()
                ),
                "positive_excess_CAGR_ratio": float(
                    group["excess_CAGR"].gt(0).mean()
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


def _shortlist(
    summary: pd.DataFrame,
    selections: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if summary.empty:
        return {
            "status": "NO_CANDIDATES",
            "candidate_count": 0,
            "candidates": [],
        }
    scored = _score_summary(summary)
    if selections is not None and not selections.empty:
        modal_profiles = {
            str(architecture): str(group["selected_profile"].mode().iloc[0])
            for architecture, group in selections.groupby("architecture")
            if not group["selected_profile"].mode().empty
        }
        exact_identities = [
            _deployment_identity(
                architecture=str(row["architecture"]),
                profile=modal_profiles.get(
                    str(row["architecture"]),
                    "balanced",
                ),
                specification=ARCHITECTURES[str(row["architecture"])],
            )
            for _, row in scored.iterrows()
        ]
        scored["strategy_id"] = [
            item["strategy_id"] for item in exact_identities
        ]
        scored["strategy_dna_hash"] = [
            item["strategy_dna_hash"] for item in exact_identities
        ]

    robust_gates = (
        scored["fold_count"].ge(10)
        & scored["positive_fold_ratio"].ge(0.60)
        & scored["median_oos_portfolio_pf"].gt(1.0)
        & scored["cost_50bps_median_pf"].gt(1.0)
        & scored["worst_oos_portfolio_pf"].gt(0.70)
        & scored["worst_oos_drawdown"].gt(-0.50)
        & scored["plateau_fold_ratio"].ge(0.50)
        & scored["median_fill_count"].ge(10)
        & scored["median_excess_CAGR"].gt(0)
        & scored["positive_excess_CAGR_ratio"].ge(0.60)
    )
    research_gates = (
        scored["fold_count"].ge(10)
        & scored["positive_fold_ratio"].ge(0.50)
        & scored["median_oos_portfolio_pf"].ge(1.05)
        & scored["cost_50bps_median_pf"].ge(0.90)
        & scored["median_oos_Sharpe"].ge(0.30)
        & scored["worst_oos_drawdown"].gt(-0.80)
        & scored["median_fill_count"].ge(10)
    )
    candidates = scored.loc[robust_gates].head(10)
    research_candidates = scored.loc[research_gates].head(15).copy()
    weighted_candidates = (
        scored.loc[
            scored["weighted_evidence_score"].ge(60)
            & scored["hard_veto_reasons"].map(len).eq(0)
        ]
        .sort_values("weighted_evidence_score", ascending=False)
        .head(20)
    )
    if not research_candidates.empty:
        research_candidates["benchmark_excess_status"] = np.where(
            research_candidates["median_excess_CAGR"].gt(0)
            & research_candidates["positive_excess_CAGR_ratio"].ge(0.50),
            "POSITIVE",
            "NOT_YET_PROVEN",
        )
        research_candidates["stress_status"] = np.where(
            research_candidates["cost_50bps_median_pf"].gt(1.0),
            "SURVIVES_50BPS",
            "MARGINAL_AT_50BPS",
        )
        research_candidates["research_classification"] = (
            "PROMISING_RESEARCH"
        )
    return {
        "status": (
            "MULTITIMEFRAME_RESEARCH_SHORTLIST"
            if not candidates.empty
            else "NO_ROBUST_MULTITIMEFRAME_CANDIDATE"
        ),
        "candidate_count": len(candidates),
        "candidates": _json_records(candidates),
        "promising_research_candidate_count": len(
            research_candidates
        ),
        "promising_research_candidates": _json_records(
            research_candidates
        ),
        "weighted_evidence_candidate_count": len(weighted_candidates),
        "weighted_evidence_candidates": _json_records(
            weighted_candidates
        ),
        "weighted_evidence_policy": {
            "economic_quality_weight": 35,
            "oos_forward_weight": 30,
            "risk_quality_weight": 20,
            "statistical_support_weight": 15,
            "research_lead_threshold": 60,
            "frozen_shadow_research_ready_threshold": 70,
            "paper_candidate_research_ready_threshold": 78,
            "paper_or_live_authority_granted": False,
            "independent_point_in_time_holdout_required": True,
        },
        "research_gate": {
            "minimum_folds": 10,
            "minimum_positive_fold_ratio": 0.50,
            "minimum_median_oos_pf": 1.05,
            "minimum_50bps_stress_pf": 0.90,
            "minimum_median_oos_sharpe": 0.30,
            "maximum_worst_drawdown": -0.80,
            "minimum_median_fill_count": 10,
        },
        "research_gate_can_grant_authority": False,
        "financial_finalist": False,
        "authority": "NONE",
    }


def _score_summary(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    identities = [
        _deployment_identity(
            architecture=str(row["architecture"]),
            profile="NESTED_WALK_FORWARD_SELECTED",
            specification={
                "higher": str(row["higher_timeframe"]),
                **(
                    {}
                    if pd.isna(row.get("middle_timeframe"))
                    else {"middle": str(row["middle_timeframe"])}
                ),
                "lower": str(row["lower_timeframe"]),
                "entry": str(row["entry_strategy"]),
            },
        )
        for _, row in scored.iterrows()
    ]
    scored["strategy_id"] = [item["strategy_id"] for item in identities]
    scored["strategy_dna_hash"] = [
        item["strategy_dna_hash"] for item in identities
    ]
    score_rows = scored.apply(_weighted_evidence, axis=1)
    scored["economic_quality_score"] = [
        item["economic_quality_score"] for item in score_rows
    ]
    scored["oos_evidence_score"] = [
        item["oos_evidence_score"] for item in score_rows
    ]
    scored["risk_quality_score"] = [
        item["risk_quality_score"] for item in score_rows
    ]
    scored["statistical_support_score"] = [
        item["statistical_support_score"] for item in score_rows
    ]
    scored["weighted_evidence_score"] = [
        item["weighted_evidence_score"] for item in score_rows
    ]
    scored["hard_veto_reasons"] = [
        item["hard_veto_reasons"] for item in score_rows
    ]
    scored["evidence_tier"] = [
        item["evidence_tier"] for item in score_rows
    ]
    return scored


def _rank_viable_strategies(
    scored: pd.DataFrame,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    viable = scored.loc[scored["hard_veto_reasons"].map(len).eq(0)]
    return viable.sort_values(
        [
            "weighted_evidence_score",
            "median_oos_Sharpe",
            "median_oos_portfolio_pf",
        ],
        ascending=False,
    ).head(limit)


def _pit_signal_record(
    *,
    identity: Mapping[str, str],
    specification: Mapping[str, str],
    profile: str,
    symbol: str,
    frame: pd.DataFrame,
    bar_timestamp: pd.Timestamp,
    observed_at: datetime,
    score: float,
) -> dict[str, Any]:
    closed = frame.loc[frame.index <= bar_timestamp]
    latest = closed.iloc[-1]
    previous_close = closed["close"].shift(1)
    true_range = pd.concat(
        (
            closed["high"].sub(closed["low"]),
            closed["high"].sub(previous_close).abs(),
            closed["low"].sub(previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    close = float(latest["close"])
    atr = _finite(
        true_range.rolling(14, min_periods=5).mean().iloc[-1],
        close * 0.02,
    )
    risk_distance = max(atr * 2.0, close * 0.02)
    stop = max(0.01, close - risk_distance)
    target_one = close + risk_distance * 1.5
    target_two = close + risk_distance * 2.5
    timestamp = bar_timestamp.to_pydatetime()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    freshness_window = {
        "1h": timedelta(hours=30),
        "2h": timedelta(hours=32),
        "4h": timedelta(hours=36),
        "1d": timedelta(days=3),
        "1w": timedelta(days=10),
    }.get(str(specification["lower"]), timedelta(days=3))
    declared_expiration = timestamp + freshness_window
    exchange_timezone = str(
        latest.get("exchange_timezone") or ""
    ).strip()
    freshness_evaluation = evaluate_signal_freshness(
        {
            "timeframe": specification["lower"],
            "data_timestamp": timestamp,
            "expiration_timestamp": declared_expiration,
            "exchange_timezone": exchange_timezone,
        },
        now=observed_at,
    )
    expiration = freshness_evaluation["effective_expiration"]
    confidence = min(0.99, max(0.0, 0.5 + 0.5 * math.tanh(score)))
    signal_key = {
        "strategy_id": identity["strategy_id"],
        "symbol": symbol,
        "bar_timestamp": timestamp.isoformat(),
        "action": "BUY",
    }
    risk_pct = risk_distance / close if close else 0.0
    return {
        "signal_id": f"MTF-SIG-{_hash(signal_key)[:24]}",
        "strategy_id": identity["strategy_id"],
        "strategy_version": "PHASE11_10_MTF_V1",
        "strategy_dna_hash": identity["strategy_dna_hash"],
        "asset": symbol,
        "ticker": symbol,
        "asset_class": "STK",
        "timeframe": specification["lower"],
        "higher_timeframe_context": specification["higher"],
        "bar_timestamp": timestamp.isoformat(),
        "data_timestamp": timestamp.isoformat(),
        "signal_timestamp": observed_at.isoformat(),
        "generated_at": observed_at.isoformat(),
        "data_freshness": freshness_evaluation["status"],
        "signal_freshness": {
            key: (
                value.isoformat()
                if isinstance(value, datetime)
                else value
            )
            for key, value in freshness_evaluation.items()
            if key != "is_current"
        },
        "exchange_timezone": exchange_timezone,
        "signal_direction": "LONG",
        "action": "BUY",
        "entry_type": "LIMIT",
        "entry_price_reference": f"{close:.6f}",
        "current_market_price": f"{close:.6f}",
        "preferred_entry": f"{close:.6f}",
        "limit_entry_price": f"{close:.6f}",
        "entry_zone_low": f"{max(0.01, close - atr * 0.25):.6f}",
        "entry_zone_high": f"{close + atr * 0.25:.6f}",
        "stop_loss": f"{stop:.6f}",
        "stop_method": "2ATR_OR_2PCT_BELOW_REFERENCE",
        "stop_distance_pct": f"{risk_pct:.6f}",
        "trailing_stop": f"{stop:.6f}",
        "take_profit_1": f"{target_one:.6f}",
        "take_profit_2": f"{target_two:.6f}",
        "expected_holding_period": "2-20 lower-timeframe bars",
        "expected_return": f"{risk_pct * 1.5:.6f}",
        "expected_risk": f"{risk_pct:.6f}",
        "risk_reward_ratio": "1.500000",
        "reward_risk_1": "1.500000",
        "reward_risk_2": "2.500000",
        "confidence_score": f"{confidence:.6f}",
        "regime": "HIGHER_TIMEFRAME_TREND_CONFIRMED",
        "macro_context": {
            "policy": "CONTEXT_ONLY_NOT_AUTHORITY",
        },
        "position_state": "UNOBSERVED",
        "execution_state": "SHADOW",
        "expiration_time": expiration.isoformat(),
        "expiration_timestamp": expiration.isoformat(),
        "signal_freshness_basis": freshness_evaluation[
            "freshness_basis"
        ],
        "lifecycle_status": "SHADOW",
        "profile": profile,
        "automatic_execution_allowed": False,
        "suggested_quantity": "0",
        "execution_sizing_authority": (
            "LIVE_PREFLIGHT_WHOLE_SHARE_RISK_FIRST"
        ),
        "legacy_fixed_notional_authoritative": False,
        "maximum_planned_loss_eur": "0",
        "reasons": [
            "FROZEN_MTF_SIGNAL_ACTIVE",
            "HIGHER_TIMEFRAME_GATE_CONFIRMED",
            "CURRENT_PIT_ATTESTATION_GO",
        ],
        "risks": [
            "MODEL_SIGNAL_NOT_GUARANTEE",
            "EXECUTION_AUTHORITY_NONE",
            "LIVE_QUOTE_REVALIDATION_REQUIRED",
        ],
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _current_pit_attestations(
    project_root: Path,
    observed_at: datetime,
) -> set[str]:
    path = (
        project_root
        / "config"
        / "screener"
        / "shariah_attestations_v1.json"
    )
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    active = set()
    for row in payload.get("attestations", []):
        try:
            screened_at = datetime.fromisoformat(
                str(row["screened_at"]).replace("Z", "+00:00")
            )
            expires_at = datetime.fromisoformat(
                str(row["expires_at"]).replace("Z", "+00:00")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if (
            row.get("status") == "SHARIAH_ELIGIBLE_PIT"
            and screened_at <= observed_at <= expires_at
        ):
            active.add(str(row.get("symbol", "")).upper())
    return active


def _deployment_identity(
    *,
    architecture: str,
    profile: str,
    specification: Mapping[str, str],
) -> dict[str, str]:
    dna = {
        "schema": SCHEMA,
        "architecture": architecture,
        "profile": profile,
        "higher_timeframe": specification["higher"],
        "middle_timeframe": specification.get("middle"),
        "lower_timeframe": specification["lower"],
        "entry_strategy": specification["entry"],
        "execution_timing": "NEXT_LOWER_TIMEFRAME_BAR",
        "higher_bar_policy": "SHIFTED_ONE_COMPLETE_HIGHER_BAR",
    }
    dna_hash = _hash(dna)
    return {
        "strategy_id": f"MTF-{dna_hash[:20]}",
        "strategy_dna_hash": dna_hash,
    }


def _research_source_hash(project_root: Path) -> str | None:
    hashes = {}
    for relative in (
        "src/stocks/research/phase11_8.py",
        "src/stocks/research/phase11_9.py",
        "src/stocks/research/phase11_10.py",
    ):
        path = project_root / relative
        if path.exists():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _hash(hashes) if hashes else None


def _weighted_evidence(row: pd.Series) -> dict[str, Any]:
    cagr = float(row["median_oos_CAGR"])
    portfolio_pf = float(row["median_oos_portfolio_pf"])
    stress_pf = float(row["cost_50bps_median_pf"])
    excess_cagr = float(row["median_excess_CAGR"])
    positive_folds = float(row["positive_fold_ratio"])
    positive_excess = float(row["positive_excess_CAGR_ratio"])
    plateau = float(row["plateau_fold_ratio"])
    worst_pf = float(row["worst_oos_portfolio_pf"])
    drawdown = abs(float(row["worst_oos_drawdown"]))
    fills = float(row["median_fill_count"])
    sharpe = float(row["median_oos_Sharpe"])
    folds = float(row["fold_count"])

    economic = (
        12 * _clamp(cagr / 0.12)
        + 10 * _clamp((portfolio_pf - 1.0) / 0.25)
        + 8 * _clamp(excess_cagr / 0.05)
        + 5 * _clamp((stress_pf - 0.90) / 0.20)
    )
    oos = (
        12 * _clamp((positive_folds - 0.40) / 0.40)
        + 6 * _clamp((worst_pf - 0.50) / 0.50)
        + 6 * _clamp(plateau)
        + 3 * _clamp(folds / 20)
        + 3 * _clamp(positive_excess)
    )
    risk = (
        14 * _clamp(1 - max(0.0, drawdown - 0.20) / 0.60)
        + 3 * _clamp((stress_pf - 0.85) / 0.30)
        + 3 * _clamp(fills / 100)
    )
    statistics = (
        10 * _clamp(sharpe)
        + 5 * _clamp(fills / 100)
    )
    vetoes = _hard_veto_reasons(row)
    score = economic + oos + risk + statistics
    if vetoes:
        tier = "REJECT_HARD_VETO"
    elif score >= 78:
        tier = "PAPER_CANDIDATE_RESEARCH_READY_PIT_BLOCKED"
    elif score >= 70:
        tier = "FROZEN_SHADOW_RESEARCH_READY_PIT_BLOCKED"
    elif score >= 60:
        tier = "RESEARCH_LEAD"
    else:
        tier = "RESEARCH_CANDIDATE"
    return {
        "economic_quality_score": round(economic, 6),
        "oos_evidence_score": round(oos, 6),
        "risk_quality_score": round(risk, 6),
        "statistical_support_score": round(statistics, 6),
        "weighted_evidence_score": round(score, 6),
        "hard_veto_reasons": vetoes,
        "evidence_tier": tier,
    }


def _hard_veto_reasons(row: pd.Series) -> list[str]:
    vetoes = []
    if float(row["median_oos_CAGR"]) <= 0:
        vetoes.append("NORMAL_COST_MEDIAN_CAGR_NOT_POSITIVE")
    if float(row["median_oos_portfolio_pf"]) <= 1:
        vetoes.append("NORMAL_COST_PORTFOLIO_PF_NOT_POSITIVE")
    if float(row["worst_oos_drawdown"]) <= -0.80:
        vetoes.append("DRAWDOWN_MANDATE_BREACH")
    if float(row["median_fill_count"]) < 10:
        vetoes.append("INSUFFICIENT_PORTFOLIO_FILLS")
    return vetoes


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            str(key): _json_value(value)
            for key, value in row.items()
        }
        for row in frame.to_dict("records")
    ]


def _json_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
        }
    try:
        return None if bool(pd.isna(value)) else value
    except (TypeError, ValueError):
        return value


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _research_source_components(project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in (
        "src/stocks/research/phase11_9.py",
        "src/stocks/research/phase11_10.py",
        "src/stocks/research/phase11_8.py",
    ):
        digest = _file_sha256(project_root / relative)
        if digest:
            result[relative] = digest
    return result


def _research_result_hash(output: Path) -> str:
    artifacts = {}
    for name in _RESEARCH_RESULT_ARTIFACTS:
        digest = _file_sha256(output / name)
        if digest:
            artifacts[name] = digest
    return _hash(artifacts)


def _historical_cutoff(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    cutoff = pd.Timestamp(value)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    if len(str(value).strip()) == 10:
        cutoff = cutoff + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return cutoff


def _truncate_frames(
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    cutoff: pd.Timestamp,
) -> dict[str, dict[str, pd.DataFrame]]:
    truncated: dict[str, dict[str, pd.DataFrame]] = {}
    for timeframe, frames in frames_by_timeframe.items():
        truncated[timeframe] = {}
        for symbol, frame in frames.items():
            index = pd.to_datetime(frame.index, utc=True, errors="coerce")
            mask = index.notna() & (index <= cutoff)
            selected = frame.loc[mask].copy()
            selected.attrs.update(frame.attrs)
            if not selected.empty:
                selected.attrs["qualification_historical_cutoff"] = (
                    cutoff.isoformat()
                )
                truncated[timeframe][symbol] = selected
    return truncated


def _archive_current_research_evidence(project_root: Path) -> Path | None:
    output = _output(project_root)
    status_path = output / "status.json"
    if not status_path.is_file():
        return None
    result_hash = _research_result_hash(output)
    archive = output / "qualification" / "archive" / (
        "LEGACY-" + result_hash[:24]
    )
    if archive.exists():
        return archive
    archive.mkdir(parents=True, exist_ok=False)
    copied: dict[str, str] = {}
    for name in (
        "architecture-summary.csv",
        "blocked.json",
        "coverage.csv",
        "manifest.json",
        "nested-results.parquet",
        "parameter-selections.csv",
        "schema.json",
        "shortlist.json",
        "status.json",
    ):
        source = output / name
        if source.is_file():
            destination = archive / name
            shutil.copy2(source, destination)
            copied[name] = str(_file_sha256(destination))
    _write_json(
        archive / "archive-manifest.json",
        {
            "schema": "phase11_10_legacy_evidence_archive_v1",
            "archived_at": datetime.now(UTC).isoformat(),
            "legacy_research_result_hash": result_hash,
            "artifact_hashes": copied,
            "immutable": True,
            **AUTHORITY,
        },
    )
    return archive


def _output(project_root: Path) -> Path:
    return project_root / "output" / "research" / "phase11_10"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


__all__ = [
    "ARCHITECTURES",
    "phase11_10_pit_observe",
    "phase11_10_qualification_audit",
    "phase11_10_qualification_freeze",
    "phase11_10_schema",
    "phase11_10_status",
    "phase11_10_top20",
    "phase11_10_watchlist",
    "run_phase11_10",
]
