from __future__ import annotations

import json
import itertools
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.research.autopilot.contracts import stable_hash
from stocks.research.autopilot.ledger import AutopilotLayout, ResearchLedger
from stocks.research.phase11_12_forward import (
    update_lower_timeframe_forward,
)
from stocks.research.phase11_8 import _run_portfolio
from stocks.research.phase11_9 import (
    BASE_STRATEGIES,
    ENSEMBLES,
    TIMEFRAMES,
    _atr,
    _load_current_frames,
    _load_frames,
    _parameters,
    _signals,
)
from stocks.signals.timeframe_contracts import (
    declared_research_signal_timeframe_contract,
)


SCHEMA = "phase11_12_bulk_strategy_dna_v1"
PROFILES = ("responsive", "balanced", "conservative")
COSTS_BPS = (10.0, 50.0)
ASSET_BUCKETS: dict[str, tuple[str, ...]] = {
    "STOCK": (
        "AAPL",
        "AMZN",
        "ASML",
        "GOOGL",
        "INTC",
        "JPM",
        "META",
        "MSFT",
        "NVDA",
        "ON",
        "XOM",
    ),
    "ETF": ("EEM", "EFA", "IWM", "QQQ", "SPY", "TLT"),
    "COMMODITY_PROXY": ("DBC", "GLD", "SLV"),
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
PAIR_ENSEMBLES: dict[str, tuple[str, str]] = {
    f"pair::{left}::{right}": (left, right)
    for left, right in itertools.combinations(BASE_STRATEGIES, 2)
}
CATALOG_FORMULAS = (
    *BASE_STRATEGIES,
    *ENSEMBLES,
    *PAIR_ENSEMBLES,
)


def generate_bulk_catalog(
    *,
    component_count: int | None = None,
) -> list[dict[str, Any]]:
    if component_count is not None and not 1 <= component_count <= 5:
        raise ValueError("component_count must be in [1,5]")
    catalog: list[dict[str, Any]] = []
    for strategy in CATALOG_FORMULAS:
        components = list(
            PAIR_ENSEMBLES.get(
                strategy,
                ENSEMBLES.get(strategy, (strategy,)),
            )
        )
        if (
            component_count is not None
            and len(components) != component_count
        ):
            continue
        for timeframe in TIMEFRAMES:
            for profile in PROFILES:
                for asset_class, symbols in ASSET_BUCKETS.items():
                    payload = {
                        "schema": SCHEMA,
                        "version": "1.0.0",
                        "family": (
                            "indicator_pair"
                            if strategy in PAIR_ENSEMBLES
                            else (
                                "indicator_ensemble"
                                if strategy in ENSEMBLES
                                else "indicator_strategy"
                            )
                        ),
                        "formula": strategy,
                        "indicator_components": components,
                        "combination_rule": (
                            "UNANIMOUS_CLOSED_BAR"
                            if strategy in PAIR_ENSEMBLES
                            else (
                                "MAJORITY_CLOSED_BAR"
                                if strategy in ENSEMBLES
                                else "SINGLE_BLOCK"
                            )
                        ),
                        "timeframe": timeframe,
                        "profile": profile,
                        "parameters": _parameters(timeframe, profile),
                        "asset_class": asset_class,
                        "symbols": list(symbols),
                        "long_only": True,
                        "whole_shares": True,
                        "leverage": False,
                        "shorting": False,
                        "entry_information": "CLOSED_BAR_ONLY",
                        "execution_assumption": "NEXT_BAR_OPEN",
                        "base_currency": "EUR",
                        "global_exposure_cap": 1.0,
                        "maximum_positions": 4,
                        "selection_rule": (
                            "PRIOR_CLOSED_BAR_SCORE_DESC_SYMBOL_ASC"
                        ),
                        "research_authority": "NONE",
                    }
                    digest = stable_hash(payload)
                    catalog.append(
                        {
                            "strategy_id": f"BULK-{digest[:24]}",
                            "strategy_hash": digest,
                            **payload,
                        }
                    )
    catalog.sort(key=lambda row: str(row["strategy_id"]))
    return catalog


def phase11_12_schema(project_root: Path) -> dict[str, Any]:
    catalog = generate_bulk_catalog()
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "catalog_size": len(catalog),
        "formula_count": len(CATALOG_FORMULAS),
        "base_formula_count": len(BASE_STRATEGIES),
        "ensemble_count": len(ENSEMBLES),
        "two_block_formula_count": len(PAIR_ENSEMBLES),
        "timeframes": list(TIMEFRAMES),
        "parameter_profiles": list(PROFILES),
        "asset_buckets": {
            key: list(value) for key, value in ASSET_BUCKETS.items()
        },
        "cost_stress_bps_per_side": list(COSTS_BPS),
        "catalog_equation": (
            f"{len(CATALOG_FORMULAS)} formulas x "
            f"{len(TIMEFRAMES)} timeframes x {len(PROFILES)} profiles x "
            f"{len(ASSET_BUCKETS)} asset buckets"
        ),
        "exhaustiveness": (
            "BOUNDED_CARTESIAN_CATALOG_NOT_ALL_MATHEMATICALLY_POSSIBLE_FORMULAS"
        ),
        "evaluation": "FIXED_FINAL_30_PERCENT_OOS_STYLE_HISTORICAL_WINDOW",
        "independent_future_holdout": False,
        "automatic_promotion": False,
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "schema.json", report)
    return report


def register_phase11_12_catalog(
    project_root: Path,
    *,
    complexity: int,
    resume: bool = True,
) -> dict[str, Any]:
    catalog = generate_bulk_catalog(component_count=complexity)
    registration = _register(project_root, catalog)
    output = _output(project_root)
    output.mkdir(parents=True, exist_ok=True)
    _write_frame(
        output / f"strategy-catalog-complexity-{complexity}.parquet",
        pd.DataFrame(catalog),
    )
    report = {
        "schema": "phase11_12_complexity_registration_v1",
        "status": "GO",
        "complexity": complexity,
        "formula_count": len(
            {str(row["formula"]) for row in catalog}
        ),
        "strategy_dna_count": len(catalog),
        "registration": registration,
        "resume": resume,
        "backtests_executed": 0,
        "financial_evidence_granted": False,
        "generated_at": datetime.now(UTC).isoformat(),
        **AUTHORITY,
    }
    _write_json(
        output / f"registration-complexity-{complexity}.json",
        report,
    )
    return report


def run_phase11_12(
    project_root: Path,
    *,
    max_strategies: int | None = None,
    complexity: int | None = None,
    pending_only: bool = False,
) -> dict[str, Any]:
    output = _output(project_root)
    output.mkdir(parents=True, exist_ok=True)
    schema = phase11_12_schema(project_root)
    full_catalog = generate_bulk_catalog(component_count=complexity)
    checkpoint_path = output / "results-checkpoint.parquet"
    result_rows = _load_bulk_trial_rows(project_root)
    completed = {
        (str(row["strategy_id"]), float(row["cost_bps"]))
        for row in result_rows
    }
    pending_catalog = [
        dna
        for dna in full_catalog
        if not _dna_complete(dna, completed)
    ]
    pending_before = len(pending_catalog)
    candidate_catalog = (
        pending_catalog if pending_only else full_catalog
    )
    requested_count = max_strategies
    if max_strategies is None:
        max_strategies = (
            min(100, len(candidate_catalog))
            if pending_only
            else len(candidate_catalog)
        )
    if max_strategies < 1:
        if pending_only and not candidate_catalog:
            return _publish_queue_empty(
                project_root,
                complexity=complexity,
                full_catalog=full_catalog,
                completed=completed,
                schema_hash=stable_hash(schema),
            )
        raise ValueError("max_strategies must be positive")
    if not pending_only and max_strategies > len(candidate_catalog):
        raise ValueError(
            f"max_strategies must be in [1,{len(candidate_catalog)}]"
        )
    catalog = candidate_catalog[:max_strategies]
    if not catalog:
        return _publish_queue_empty(
            project_root,
            complexity=complexity,
            full_catalog=full_catalog,
            completed=completed,
            schema_hash=stable_hash(schema),
        )
    registration = _register(project_root, catalog)
    catalog_frame = pd.DataFrame(catalog)
    _write_frame(output / "strategy-catalog.parquet", catalog_frame)
    _write_json(
        output / "registration.json",
        {
            "schema": "phase11_12_registration_v1",
            "status": "GO",
            "requested_count": requested_count,
            "selected_count": len(catalog),
            "pending_only": pending_only,
            "pending_before": pending_before,
            "catalog_count": len(catalog),
            "full_catalog_count": len(full_catalog),
            "registration": registration,
            **AUTHORITY,
        },
    )

    frames_by_timeframe = _load_frames(project_root)
    catalog_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {
        (
            str(row["formula"]),
            str(row["timeframe"]),
            str(row["profile"]),
        ): []
        for row in catalog
    }
    for row in catalog:
        catalog_by_key[
            (
                str(row["formula"]),
                str(row["timeframe"]),
                str(row["profile"]),
            )
        ].append(row)

    processed_since_checkpoint = 0
    for (formula, timeframe, profile), dna_rows in sorted(
        catalog_by_key.items()
    ):
        pending_dna_rows = [
            dna
            for dna in dna_rows
            if not all(
                (str(dna["strategy_id"]), cost) in completed
                for cost in COSTS_BPS
            )
        ]
        if not pending_dna_rows:
            continue
        all_frames = frames_by_timeframe.get(timeframe, {})
        if not all_frames:
            for dna in pending_dna_rows:
                rows = _blocked_rows(
                    dna,
                    "TIMEFRAME_DATA_UNAVAILABLE",
                )
                result_rows.extend(rows)
                for row in rows:
                    completed.add(
                        (
                            str(row["strategy_id"]),
                            float(row["cost_bps"]),
                        )
                    )
                _register_trials(project_root, rows)
            continue
        signal_cache = _formula_signals(
            all_frames,
            formula,
            timeframe,
            profile,
        )
        for dna in pending_dna_rows:
            symbols = set(ASSET_BUCKETS[str(dna["asset_class"])])
            frames = {
                symbol: frame
                for symbol, frame in all_frames.items()
                if symbol in symbols
            }
            signals = {
                symbol: frame
                for symbol, frame in signal_cache.items()
                if symbol in frames
            }
            if not frames:
                rows = _blocked_rows(dna, "ASSET_BUCKET_DATA_UNAVAILABLE")
            else:
                rows = _evaluate_dna(dna, frames, signals)
            result_rows.extend(rows)
            for row in rows:
                completed.add(
                    (str(row["strategy_id"]), float(row["cost_bps"]))
                )
            _register_trials(project_root, rows)
            processed_since_checkpoint += 1
            if processed_since_checkpoint >= 25:
                _write_frame(
                    checkpoint_path,
                    pd.DataFrame(result_rows).drop_duplicates(
                        ["strategy_id", "cost_bps"], keep="last"
                    ),
                )
                processed_since_checkpoint = 0

    results = pd.DataFrame(result_rows).drop_duplicates(
        ["strategy_id", "cost_bps"], keep="last"
    )
    _write_frame(checkpoint_path, results)
    _write_frame(output / "results.parquet", results)
    cohort_summary = _summarize(catalog_frame, results)
    evaluated_catalog = results[
        [
            "strategy_id",
            "strategy_hash",
            "formula",
            "timeframe",
            "profile",
            "asset_class",
        ]
    ].drop_duplicates("strategy_id", keep="last")
    summary = _summarize(evaluated_catalog, results)
    _write_frame(
        output / "strategy-summary-latest-batch.parquet",
        cohort_summary,
    )
    _write_frame(output / "strategy-summary.parquet", summary)
    multiple_testing = _multiple_testing_audit(summary)
    positive = _positive_registry(summary, multiple_testing)
    _write_json(output / "multiple-testing-audit.json", multiple_testing)
    _write_json(output / "backtest-positive-registry.json", positive)

    completed_ids = set(
        results.loc[results["status"].eq("COMPLETE"), "strategy_id"]
    )
    catalog_ids = set(catalog_frame["strategy_id"].astype(str))
    completed_catalog_ids = completed_ids & catalog_ids
    cohort_results = results[
        results["strategy_id"].astype(str).isin(catalog_ids)
    ]
    remaining_after = sum(
        not _dna_complete(dna, completed) for dna in full_catalog
    )
    coverage = _coverage(catalog_frame, results)
    status = {
        "schema": SCHEMA,
        "status": (
            "GO"
            if len(completed_catalog_ids) == len(catalog)
            and len(catalog) >= min(1_000, max_strategies)
            else "PARTIAL"
        ),
        "complexity_filter": complexity,
        "pending_only": pending_only,
        "full_catalog_count": len(full_catalog),
        "pending_before": pending_before,
        "pending_after": remaining_after,
        "catalog_count": len(catalog),
        "registered_count": registration["registered_total"],
        "completed_strategy_count": len(completed_catalog_ids),
        "trial_count": len(cohort_results),
        "append_only_result_count": len(results),
        "cumulative_evaluated_strategy_count": len(summary),
        "expected_trial_count": len(catalog) * len(COSTS_BPS),
        "coverage": coverage,
        "backtest_positive": positive,
        "multiple_testing": multiple_testing,
        "phase11_9_nested_evidence_reused": True,
        "phase11_10_hierarchical_evidence_reused": True,
        "phase11_11_hmm_pair_count": _hmm_pair_count(project_root),
        "independent_future_holdout": False,
        "catalog_is_financial_finalist_evidence": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_hash": stable_hash(schema),
        **AUTHORITY,
    }
    _write_json(output / "status.json", status)
    _write_json(output / "manifest.json", status)
    return status


def phase11_12_status(project_root: Path) -> dict[str, Any]:
    path = _output(project_root) / "status.json"
    if not path.exists():
        return {
            "schema": SCHEMA,
            "status": "NOT_RUN",
            **AUTHORITY,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def phase11_12_observe(
    project_root: Path,
    *,
    max_strategies: int = 12,
) -> dict[str, Any]:
    """Observe robust discovery candidates without granting trading authority."""
    output = _output(project_root)
    summary_path = output / "strategy-summary.parquet"
    if not summary_path.exists():
        return {
            "schema": "phase11_12_lower_timeframe_shadow_observation_v1",
            "status": "BLOCKED_RESEARCH_SUMMARY_MISSING",
            **AUTHORITY,
        }
    if not 1 <= max_strategies <= 24:
        raise ValueError("max_strategies must be in [1,24]")

    summary = pd.read_parquet(summary_path)
    selected = _select_shadow_candidates(
        summary,
        max_strategies=max_strategies,
    )
    if selected.empty:
        return {
            "schema": "phase11_12_lower_timeframe_shadow_observation_v1",
            "status": "NO_STABLE_LOWER_TIMEFRAME_CANDIDATES",
            "candidate_count": 0,
            **AUTHORITY,
        }

    frames_by_timeframe = _forward_frames(project_root)
    observed_at = datetime.now(UTC)
    attestations = _current_attestations(project_root, observed_at)
    observations: list[dict[str, Any]] = []
    active_signals: list[dict[str, Any]] = []

    for candidate in selected.to_dict("records"):
        timeframe = str(candidate["timeframe"])
        formula = str(candidate["formula"])
        profile = str(candidate["profile"])
        asset_class = str(candidate["asset_class"])
        timeframe_contract = declared_research_signal_timeframe_contract(
            project_root, candidate
        )
        available = frames_by_timeframe.get(timeframe, {})
        symbols = set(ASSET_BUCKETS[asset_class])
        frames = {
            symbol: frame
            for symbol, frame in available.items()
            if symbol in symbols and not frame.empty
        }
        if not frames:
            observations.append(
                {
                    **_public_candidate(candidate),
                    "observation_status": "DATA_UNAVAILABLE",
                    "active_signal_count": 0,
                    "execution_route": "BLOCKED",
                    "order_intents": [],
                    "automatic_orders": 0,
                }
            )
            continue

        common_close = min(frame.index.max() for frame in frames.values())
        signal_frames = _formula_signals(
            frames,
            formula,
            timeframe,
            profile,
        )
        strategy_signals: list[dict[str, Any]] = []
        stale_signal_symbols: list[str] = []
        for symbol, signal_frame in signal_frames.items():
            closed_signal = signal_frame.loc[signal_frame.index <= common_close]
            closed_price = frames[symbol].loc[
                frames[symbol].index <= common_close
            ]
            if closed_signal.empty or closed_price.empty:
                continue
            latest_signal = closed_signal.iloc[-1]
            if not bool(latest_signal.get("signal", False)):
                continue
            freshness = _bar_freshness(
                closed_price.index[-1],
                observed_at,
                timeframe,
            )
            if freshness != "FRESH_CLOSED_BAR":
                stale_signal_symbols.append(symbol)
                continue
            latest_price = closed_price.iloc[-1]
            close = float(latest_price["close"])
            atr_series = _atr(closed_price, 14).dropna()
            atr_value = (
                float(atr_series.iloc[-1])
                if not atr_series.empty
                else float("nan")
            )
            risk_levels_available = math.isfinite(atr_value) and atr_value > 0
            row = {
                "strategy_id": str(candidate["strategy_id"]),
                "formula": formula,
                "timeframe": timeframe,
                "profile": profile,
                "asset_class": asset_class,
                "strategy_family": formula,
                "strategy_timeframe_contract": timeframe_contract,
                "model_version": "NO_ML_MODEL_DETERMINISTIC_SIGNAL_V1",
                "symbol": symbol,
                "forward_provider": str(
                    closed_price.attrs.get("provider", "UNKNOWN")
                ),
                "base_currency": "EUR",
                "closed_bar_timestamp": pd.Timestamp(
                    closed_price.index[-1]
                ).isoformat(),
                "data_freshness": freshness,
                "score": float(latest_signal.get("score", 0.0)),
                "reference_close": close,
                "atr_14": atr_value if risk_levels_available else None,
                "illustrative_stop": (
                    max(0.0, close - (2.0 * atr_value))
                    if risk_levels_available
                    else None
                ),
                "illustrative_target_1": (
                    close + (2.0 * atr_value)
                    if risk_levels_available
                    else None
                ),
                "illustrative_target_2": (
                    close + (3.0 * atr_value)
                    if risk_levels_available
                    else None
                ),
                "risk_level_policy": (
                    "ATR14_RESEARCH_LEVELS_NOT_BROKER_ORDERS"
                ),
                "current_shariah_attestation": (
                    "CURRENTLY_ATTESTED"
                    if symbol in attestations
                    else "NOT_CURRENTLY_ATTESTED_BLOCK_EXECUTION"
                ),
                "action": "SHADOW_ENTRY_CANDIDATE",
                "execution_route": "BLOCKED",
                "order_intents": [],
                "automatic_orders": 0,
            }
            strategy_signals.append(row)
        strategy_signals.sort(
            key=lambda row: (-float(row["score"]), str(row["symbol"]))
        )
        strategy_signals = strategy_signals[:4]
        active_signals.extend(strategy_signals)
        observations.append(
            {
                **_public_candidate(candidate),
                "closed_bar_timestamp": pd.Timestamp(common_close).isoformat(),
                "instrument_count": len(frames),
                "forward_providers": sorted(
                    {
                        str(frame.attrs.get("provider", "UNKNOWN"))
                        for frame in frames.values()
                    }
                ),
                "active_signal_count": len(strategy_signals),
                "stale_signal_count": len(stale_signal_symbols),
                "stale_signal_symbols": sorted(stale_signal_symbols),
                "active_symbols": [
                    str(row["symbol"]) for row in strategy_signals
                ],
                "observation_status": "OBSERVATION_COMPLETE",
                "execution_route": "BLOCKED",
                "order_intents": [],
                "automatic_orders": 0,
            }
        )

    active_signals.sort(
        key=lambda row: (
            str(row["timeframe"]),
            -float(row["score"]),
            str(row["formula"]),
            str(row["symbol"]),
        )
    )
    payload = {
        "schema": "phase11_12_lower_timeframe_shadow_observation_v1",
        "status": "GO",
        "observed_at": observed_at.isoformat(),
        "candidate_count": len(selected),
        "observation_count": len(observations),
        "active_signal_count": len(active_signals),
        "timeframe_counts": {
            str(key): int(value)
            for key, value in selected["timeframe"].value_counts().items()
        },
        "selection_policy": {
            "timeframes": ["1h", "4h"],
            "minimum_sharpe": 0.5,
            "minimum_period_profit_factor": 1.1,
            "minimum_50bps_profit_factor": 1.0,
            "minimum_fill_count": 75,
            "maximum_drawdown": 0.25,
            "minimum_stable_profiles": 2,
            "maximum_candidates": max_strategies,
            "economic_outcomes_deduplicated": True,
        },
        "observations": observations,
        "active_signals": active_signals,
        "attested_symbols": sorted(attestations),
        "forward_status": "COLLECTING_NEW_SHADOW_OBSERVATIONS",
        "historical_selection_bias_status": (
            "BLOCKED_NO_NEW_INDEPENDENT_HOLDOUT"
        ),
        "execution_route": "BLOCKED",
        "automatic_orders": 0,
        **AUTHORITY,
    }
    forward_evidence = update_lower_timeframe_forward(
        project_root,
        payload,
        frames_by_timeframe,
    )
    payload["forward_evidence"] = {
        "status": forward_evidence["status"],
        "observation_count": forward_evidence["observation_count"],
        "episode_count": forward_evidence["episode_count"],
        "pending_entry_count": forward_evidence["pending_entry_count"],
        "open_episode_count": forward_evidence["open_episode_count"],
        "pending_exit_count": forward_evidence["pending_exit_count"],
        "closed_episode_count": forward_evidence["closed_episode_count"],
        "aggregate": forward_evidence["aggregate"],
        "automatic_orders": 0,
        "execution_authority": "NONE",
    }
    digest = stable_hash(payload)
    observation_root = output / "forward-observations"
    _write_json(observation_root / f"{digest}.json", payload)
    _write_json(output / "latest-shadow-observation.json", payload)
    return payload


def _select_shadow_candidates(
    summary: pd.DataFrame,
    *,
    max_strategies: int,
) -> pd.DataFrame:
    work = summary.copy()
    numeric_columns = (
        "CAGR",
        "Sharpe",
        "period_profit_factor",
        "stress_50bps_profit_factor",
        "maximum_drawdown",
        "fill_count",
    )
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    lower_timeframe = work["timeframe"].isin(("1h", "4h"))
    soft_gate = (
        work["status"].eq("COMPLETE")
        & lower_timeframe
        & work["CAGR"].gt(0)
        & work["period_profit_factor"].gt(1.0)
        & work["stress_50bps_profit_factor"].gt(1.0)
        & work["maximum_drawdown"].ge(-0.30)
        & work["fill_count"].ge(50)
    )
    profile_counts = (
        work.loc[soft_gate]
        .groupby(["formula", "timeframe", "asset_class"])["profile"]
        .nunique()
    )
    work["stable_profile_count"] = [
        int(
            profile_counts.get(
                (
                    row.formula,
                    row.timeframe,
                    row.asset_class,
                ),
                0,
            )
        )
        for row in work.itertuples()
    ]
    hard_gate = (
        work["status"].eq("COMPLETE")
        & lower_timeframe
        & work["CAGR"].gt(0)
        & work["Sharpe"].ge(0.5)
        & work["period_profit_factor"].ge(1.1)
        & work["stress_50bps_profit_factor"].ge(1.0)
        & work["maximum_drawdown"].ge(-0.25)
        & work["fill_count"].ge(75)
        & work["stable_profile_count"].ge(2)
    )
    candidates = work.loc[hard_gate].sort_values(
        [
            "Sharpe",
            "stress_50bps_profit_factor",
            "period_profit_factor",
            "CAGR",
        ],
        ascending=False,
    )
    candidates = candidates.drop_duplicates(
        "economic_outcome_fingerprint",
        keep="first",
    )
    candidates = candidates.drop_duplicates(
        ["formula", "timeframe", "asset_class"],
        keep="first",
    )

    selected_indices: list[Any] = []
    timeframe_counts = {"1h": 0, "4h": 0}
    formula_counts: dict[str, int] = {}
    per_timeframe_limit = max(1, math.ceil(max_strategies / 2))
    for index, row in candidates.iterrows():
        timeframe = str(row["timeframe"])
        formula = str(row["formula"])
        if timeframe_counts[timeframe] >= per_timeframe_limit:
            continue
        if formula_counts.get(formula, 0) >= 2:
            continue
        selected_indices.append(index)
        timeframe_counts[timeframe] += 1
        formula_counts[formula] = formula_counts.get(formula, 0) + 1
        if len(selected_indices) >= max_strategies:
            break
    return work.loc[selected_indices].reset_index(drop=True)


def _forward_frames(
    project_root: Path,
) -> dict[str, dict[str, pd.DataFrame]]:
    return _load_current_frames(project_root)


def _public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": str(candidate["strategy_id"]),
        "formula": str(candidate["formula"]),
        "timeframe": str(candidate["timeframe"]),
        "profile": str(candidate["profile"]),
        "asset_class": str(candidate["asset_class"]),
        "historical_CAGR": float(candidate["CAGR"]),
        "historical_Sharpe": float(candidate["Sharpe"]),
        "historical_period_profit_factor": float(
            candidate["period_profit_factor"]
        ),
        "historical_50bps_profit_factor": float(
            candidate["stress_50bps_profit_factor"]
        ),
        "historical_maximum_drawdown": float(
            candidate["maximum_drawdown"]
        ),
        "historical_fill_count": int(candidate["fill_count"]),
        "stable_profile_count": int(candidate["stable_profile_count"]),
        "research_classification": (
            "STABLE_PROFILE_SHADOW_CANDIDATE_NOT_FINANCIAL_FINALIST"
        ),
    }


def _bar_freshness(
    bar_timestamp: Any,
    observed_at: datetime,
    timeframe: str,
) -> str:
    bar = pd.Timestamp(bar_timestamp)
    if bar.tzinfo is None:
        bar = bar.tz_localize("UTC")
    else:
        bar = bar.tz_convert("UTC")
    now = pd.Timestamp(observed_at)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    if bar > now:
        return "FUTURE_BAR_BLOCKED"
    business_days = len(
        pd.bdate_range(bar.normalize(), now.normalize())
    )
    age_hours = (now - bar).total_seconds() / 3600.0
    same_session_limit = 12.0 if timeframe == "1h" else 24.0
    if business_days <= 1 and age_hours <= same_session_limit:
        return "FRESH_CLOSED_BAR"
    if business_days == 2 and age_hours <= 96.0:
        return "FRESH_CLOSED_BAR"
    return "STALE_BAR_BLOCKED"


def _current_attestations(
    project_root: Path,
    decision_time: datetime,
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
        rows = json.loads(path.read_text(encoding="utf-8")).get(
            "attestations", []
        )
    except json.JSONDecodeError:
        return set()
    eligible: set[str] = set()
    for row in rows:
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
            and screened_at <= decision_time <= expires_at
        ):
            eligible.add(str(row["symbol"]).upper())
    return eligible


def _formula_signals(
    frames: Mapping[str, pd.DataFrame],
    formula: str,
    timeframe: str,
    profile: str,
) -> dict[str, pd.DataFrame]:
    pair = PAIR_ENSEMBLES.get(formula)
    if pair is None:
        return _signals(frames, formula, timeframe, profile)
    left = _signals(frames, pair[0], timeframe, profile)
    right = _signals(frames, pair[1], timeframe, profile)
    output: dict[str, pd.DataFrame] = {}
    for symbol in sorted(set(left) & set(right)):
        aligned = pd.concat(
            {
                "left_signal": left[symbol]["signal"],
                "right_signal": right[symbol]["signal"],
                "left_score": left[symbol]["score"],
                "right_score": right[symbol]["score"],
            },
            axis=1,
            join="inner",
        )
        signal = (
            aligned["left_signal"].fillna(False).astype(bool)
            & aligned["right_signal"].fillna(False).astype(bool)
        )
        scores = aligned[["left_score", "right_score"]].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        score = scores.mean(axis=1).where(signal, -math.inf)
        output[symbol] = pd.DataFrame(
            {
                "signal": signal,
                "score": score.fillna(-math.inf),
            },
            index=aligned.index,
        )
    return output


def _load_bulk_trial_rows(project_root: Path) -> list[dict[str, Any]]:
    ledger = ResearchLedger(AutopilotLayout(project_root))
    try:
        strategies = {
            str(row["strategy_id"]): row
            for row in ledger.bulk_strategies()
        }
        trials = ledger.bulk_trials()
    finally:
        ledger.close()
    rows: list[dict[str, Any]] = []
    for trial in trials:
        strategy = strategies.get(str(trial["strategy_id"]))
        if strategy is None:
            continue
        metrics = dict(trial.get("metrics", {}))
        rows.append(
            {
                "strategy_id": strategy["strategy_id"],
                "strategy_hash": strategy["strategy_hash"],
                "formula": strategy["formula"],
                "timeframe": strategy["timeframe"],
                "profile": strategy["profile"],
                "asset_class": strategy["asset_class"],
                "cost_bps": float(trial["cost_bps"]),
                "status": trial["status"],
                "metrics": metrics,
                "provenance": dict(trial.get("provenance", {})),
                **_flatten_metrics(metrics),
            }
        )
    return rows


def _dna_complete(
    dna: Mapping[str, Any],
    completed: set[tuple[str, float]],
) -> bool:
    strategy_id = str(dna["strategy_id"])
    return all(
        (strategy_id, float(cost)) in completed
        for cost in COSTS_BPS
    )


def _publish_queue_empty(
    project_root: Path,
    *,
    complexity: int | None,
    full_catalog: list[dict[str, Any]],
    completed: set[tuple[str, float]],
    schema_hash: str,
) -> dict[str, Any]:
    report = {
        "schema": SCHEMA,
        "status": "QUEUE_EMPTY",
        "complexity_filter": complexity,
        "pending_only": True,
        "full_catalog_count": len(full_catalog),
        "catalog_count": 0,
        "completed_strategy_count": sum(
            _dna_complete(dna, completed) for dna in full_catalog
        ),
        "pending_before": 0,
        "pending_after": 0,
        "trial_count": 0,
        "expected_trial_count": 0,
        "financial_finalist": False,
        "automatic_promotion": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "schema_hash": schema_hash,
        **AUTHORITY,
    }
    output = _output(project_root)
    _write_json(output / "status.json", report)
    _write_json(output / "manifest.json", report)
    return report


def _register(
    project_root: Path,
    catalog: list[dict[str, Any]],
) -> dict[str, int]:
    ledger = ResearchLedger(AutopilotLayout(project_root))
    try:
        result = ledger.register_bulk_strategies(catalog)
        counts = ledger.counts()
    finally:
        ledger.close()
    return {
        **result,
        "registered_total": counts["bulk_strategy_dna"],
    }


def _register_trials(
    project_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    ledger = ResearchLedger(AutopilotLayout(project_root))
    try:
        for row in rows:
            ledger.append_bulk_trial(
                strategy_id=str(row["strategy_id"]),
                cost_bps=float(row["cost_bps"]),
                status=str(row["status"]),
                metrics=dict(row.get("metrics", {})),
                provenance=dict(row.get("provenance", {})),
            )
    finally:
        ledger.close()


def _evaluate_dna(
    dna: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    signals: Mapping[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    evaluation_start, evaluation_end = _evaluation_window(frames)
    if evaluation_start is None or evaluation_end is None:
        return _blocked_rows(dna, "INSUFFICIENT_EVALUATION_HISTORY")
    rows: list[dict[str, Any]] = []
    for cost_bps in COSTS_BPS:
        try:
            run = _run_portfolio(
                frames,
                signals,
                start=evaluation_start,
                end=evaluation_end,
                cost_bps=cost_bps,
            )
            fills = run["fills"]
            metrics = {
                **run["metrics"],
                "fill_count": len(fills),
                "maximum_gross_exposure": float(
                    run["ledger"]["gross_exposure"].max()
                    if not run["ledger"].empty
                    else 0.0
                ),
                "total_cost_eur": float(
                    fills["fee_eur"].sum() if not fills.empty else 0.0
                ),
            }
            status = (
                "COMPLETE"
                if metrics["maximum_gross_exposure"] <= 1.000001
                else "EXPOSURE_CONTRACT_BLOCKED"
            )
            provenance = {
                "bar_origin": "REAL_LOCAL_PROVIDER_CACHE",
                "symbols": sorted(frames),
                "instrument_count": len(frames),
                "timeframe": dna["timeframe"],
                "asset_class": dna["asset_class"],
                "evaluation_start": evaluation_start.isoformat(),
                "evaluation_end": evaluation_end.isoformat(),
                "closed_bars_only": True,
                "next_bar_execution": True,
                "whole_shares": True,
                "base_currency": "EUR",
                "synthetic_data": False,
                "broker_calls": 0,
                "order_calls": 0,
            }
        except Exception as exc:
            status = "EVALUATION_BLOCKED"
            metrics = {}
            provenance = {
                "reason": type(exc).__name__,
                "broker_calls": 0,
                "order_calls": 0,
            }
        rows.append(
            {
                "strategy_id": dna["strategy_id"],
                "strategy_hash": dna["strategy_hash"],
                "formula": dna["formula"],
                "timeframe": dna["timeframe"],
                "profile": dna["profile"],
                "asset_class": dna["asset_class"],
                "cost_bps": cost_bps,
                "status": status,
                "metrics": metrics,
                "provenance": provenance,
                **_flatten_metrics(metrics),
            }
        )
    return rows


def _evaluation_window(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    dates = sorted(
        {
            pd.Timestamp(value)
            for frame in frames.values()
            for value in frame.index
        }
    )
    if len(dates) < 60:
        return None, None
    split = min(len(dates) - 20, max(40, int(len(dates) * 0.70)))
    return dates[split], dates[-1]


def _blocked_rows(
    dna: Mapping[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    return [
        {
            "strategy_id": dna["strategy_id"],
            "strategy_hash": dna["strategy_hash"],
            "formula": dna["formula"],
            "timeframe": dna["timeframe"],
            "profile": dna["profile"],
            "asset_class": dna["asset_class"],
            "cost_bps": cost,
            "status": reason,
            "metrics": {},
            "provenance": {
                "reason": reason,
                "broker_calls": 0,
                "order_calls": 0,
            },
        }
        for cost in COSTS_BPS
    ]


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    names = (
        "CAGR",
        "Sharpe",
        "period_profit_factor",
        "maximum_drawdown",
        "turnover",
        "fill_count",
        "maximum_gross_exposure",
        "total_cost_eur",
    )
    return {name: metrics.get(name) for name in names}


def _summarize(
    catalog: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dna in catalog.to_dict("records"):
        group = results.loc[
            results["strategy_id"].eq(dna["strategy_id"])
        ]
        normal = group.loc[group["cost_bps"].eq(10.0)]
        stress = group.loc[group["cost_bps"].eq(50.0)]
        normal_row = normal.iloc[0] if len(normal) == 1 else {}
        stress_row = stress.iloc[0] if len(stress) == 1 else {}
        complete = (
            len(normal) == 1
            and len(stress) == 1
            and normal_row.get("status") == "COMPLETE"
            and stress_row.get("status") == "COMPLETE"
        )
        fingerprint_payload = {
            "CAGR": _round(normal_row.get("CAGR")),
            "Sharpe": _round(normal_row.get("Sharpe")),
            "PF": _round(normal_row.get("period_profit_factor")),
            "DD": _round(normal_row.get("maximum_drawdown")),
            "fills": normal_row.get("fill_count"),
            "stress_PF": _round(stress_row.get("period_profit_factor")),
        }
        rows.append(
            {
                "strategy_id": dna["strategy_id"],
                "strategy_hash": dna["strategy_hash"],
                "formula": dna["formula"],
                "timeframe": dna["timeframe"],
                "profile": dna["profile"],
                "asset_class": dna["asset_class"],
                "status": "COMPLETE" if complete else "BLOCKED",
                "CAGR": normal_row.get("CAGR"),
                "Sharpe": normal_row.get("Sharpe"),
                "period_profit_factor": normal_row.get(
                    "period_profit_factor"
                ),
                "maximum_drawdown": normal_row.get("maximum_drawdown"),
                "fill_count": normal_row.get("fill_count"),
                "stress_50bps_profit_factor": stress_row.get(
                    "period_profit_factor"
                ),
                "economic_outcome_fingerprint": stable_hash(
                    fingerprint_payload
                ),
            }
        )
    return pd.DataFrame(rows)


def _positive_registry(
    summary: pd.DataFrame,
    multiple_testing: Mapping[str, Any],
) -> dict[str, Any]:
    numeric = summary.copy()
    for column in (
        "CAGR",
        "Sharpe",
        "period_profit_factor",
        "maximum_drawdown",
        "fill_count",
        "stress_50bps_profit_factor",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    gates = (
        numeric["status"].eq("COMPLETE")
        & numeric["CAGR"].gt(0)
        & numeric["period_profit_factor"].gt(1)
        & numeric["stress_50bps_profit_factor"].gt(1)
        & numeric["maximum_drawdown"].gt(-0.50)
        & numeric["fill_count"].ge(5)
    )
    positive = numeric.loc[gates].sort_values(
        ["Sharpe", "period_profit_factor", "CAGR"],
        ascending=False,
    )
    return {
        "schema": "phase11_12_backtest_positive_registry_v1",
        "status": (
            "BACKTEST_POSITIVE_RESEARCH_AVAILABLE"
            if not positive.empty
            else "NO_BACKTEST_POSITIVE_RESEARCH"
        ),
        "candidate_count": len(positive),
        "unique_economic_outcome_count": int(
            positive["economic_outcome_fingerprint"].nunique()
        ),
        "formula_count": int(positive["formula"].nunique()),
        "timeframe_counts": {
            str(key): int(value)
            for key, value in positive["timeframe"].value_counts().items()
        },
        "asset_class_counts": {
            str(key): int(value)
            for key, value in positive["asset_class"].value_counts().items()
        },
        "top_100": positive.head(100).to_dict("records"),
        "selection_bias_status": multiple_testing["status"],
        "financial_finalist": False,
        "automatic_promotion": False,
        "authority": "NONE",
    }


def _multiple_testing_audit(summary: pd.DataFrame) -> dict[str, Any]:
    complete = summary.loc[summary["status"].eq("COMPLETE")]
    unique = int(complete["economic_outcome_fingerprint"].nunique())
    return {
        "schema": "phase11_12_multiple_testing_audit_v1",
        "status": "BLOCKED_NO_NEW_INDEPENDENT_HOLDOUT",
        "registered_hypothesis_count": len(summary),
        "complete_hypothesis_count": len(complete),
        "unique_economic_outcome_count": unique,
        "duplicate_outcome_count": len(complete) - unique,
        "raw_best_sharpe": _finite_max(complete["Sharpe"]),
        "selection_bias_corrected_finalist_count": 0,
        "future_holdout_required": True,
        "interpretation": (
            "Raw positive rows are discovery evidence only; selecting the best "
            f"of {len(summary)} reused-history hypotheses is not independent "
            "proof."
        ),
    }


def _coverage(
    catalog: pd.DataFrame,
    results: pd.DataFrame,
) -> dict[str, Any]:
    completed = results.loc[results["status"].eq("COMPLETE")]
    complete_ids = set(completed["strategy_id"])
    return {
        "formula_count": int(catalog["formula"].nunique()),
        "timeframe_count": int(catalog["timeframe"].nunique()),
        "asset_class_count": int(catalog["asset_class"].nunique()),
        "profile_count": int(catalog["profile"].nunique()),
        "completed_by_timeframe": {
            timeframe: int(
                catalog.loc[
                    catalog["timeframe"].eq(timeframe), "strategy_id"
                ].isin(complete_ids).sum()
            )
            for timeframe in TIMEFRAMES
        },
        "completed_by_asset_class": {
            asset_class: int(
                catalog.loc[
                    catalog["asset_class"].eq(asset_class), "strategy_id"
                ].isin(complete_ids).sum()
            )
            for asset_class in ASSET_BUCKETS
        },
    }


def _hmm_pair_count(project_root: Path) -> int:
    path = (
        project_root
        / "output"
        / "research"
        / "phase11_11"
        / "status.json"
    )
    try:
        return int(
            json.loads(path.read_text(encoding="utf-8")).get(
                "paired_result_count", 0
            )
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return 0


def _round(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 10) if math.isfinite(number) else None


def _finite_max(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    return None if numeric.dropna().empty else float(numeric.max())


def _output(project_root: Path) -> Path:
    return project_root / "output" / "research" / "phase11_12"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        {**payload, "content_hash": stable_hash(payload)}
        if isinstance(payload, dict)
        else payload
    )
    path.write_text(
        json.dumps(body, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    work = frame.copy()
    for column in work.columns:
        if work[column].map(
            lambda value: isinstance(value, (dict, list, tuple))
        ).any():
            work[column] = work[column].map(
                lambda value: (
                    json.dumps(value, sort_keys=True, default=str)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
            )
    work.to_parquet(path, index=False)


__all__ = [
    "PAIR_ENSEMBLES",
    "generate_bulk_catalog",
    "phase11_12_observe",
    "phase11_12_schema",
    "phase11_12_status",
    "register_phase11_12_catalog",
    "run_phase11_12",
]
