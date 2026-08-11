from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pandas as pd

from stocks.research.autopilot.contracts import stable_hash
from stocks.research.phase11_6 import nested_walk_forward_folds
from stocks.research.phase11_8 import _run_portfolio
from stocks.research.phase11_12 import (
    ASSET_BUCKETS,
    PROFILES,
    _current_attestations,
    _formula_signals,
    _forward_frames,
)
from stocks.research.phase11_13 import (
    COSTS_BPS,
    _append_fills,
    _append_returns,
    _finite,
    _maximum_exposure,
    _minimum_cash,
    _summarize,
    _whole_share_violation_count,
)
from stocks.research.phase11_9 import _load_frames
from stocks.universe import (
    BROAD_UNIVERSE_PATH,
    SECURITY_MASTER_PATH,
    broad_asset_metadata,
)


SCHEMA = "phase11_14_survivor_nested_qualification_v1"
PRIMARY_TIMEFRAMES = ("1h", "4h", "1d", "1w")
ASSET_CLASS_ORDER = ("STOCK", "ETF", "COMMODITY_PROXY")
AUTHORITY = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "OBSERVATION_ONLY_AFTER_FREEZE",
    "STRATEGY_AUTHORITY": "NONE",
    "EXECUTION_AUTHORITY": "NONE",
    "PAPER_STRATEGY_AUTHORITY": "NONE",
    "LIVE_STRATEGY_AUTHORITY": "NONE",
    "BROKER_CALLS": 0,
    "ORDER_CALLS": 0,
}
FORWARD_OBSERVATION_SCHEMAS = {
    "phase11_14_forward_observation_v1",
    "phase11_14_forward_observation_v2",
    "phase11_14_forward_observation_v3",
}
FORWARD_COST_BPS_PER_SIDE = 10.0
EXPLORATORY_OBSERVER_TIMEFRAMES = frozenset({"1h", "4h"})


def _forward_asset_metadata(
    project_root: Path,
    *,
    observed_at: datetime,
) -> dict[str, dict[str, Any]]:
    """Resolve metadata available at the forward decision timestamp.

    The broad manifest remains authoritative for funds and commodity proxies.
    Listed-company classifications are added from the existing security-master
    snapshot only when one unambiguous active classification was already
    available. Nothing is inferred from a later portfolio or outcome artifact.
    """
    accepted_at = observed_at.astimezone(UTC)
    broad_path = project_root / BROAD_UNIVERSE_PATH
    broad_accepted_at = (
        datetime.fromtimestamp(broad_path.stat().st_mtime, tz=UTC)
        if broad_path.is_file()
        else None
    )
    broad_available = (
        broad_accepted_at is not None and broad_accepted_at <= accepted_at
    )
    broad_hash = (
        stable_hash(_read_json(broad_path)) if broad_available else None
    )
    broad_accepted_at_iso = (
        broad_accepted_at.isoformat()
        if broad_available and broad_accepted_at is not None
        else None
    )
    metadata: dict[str, dict[str, Any]] = {}
    if broad_available:
        metadata = {
            symbol: {
                **values,
                "industry": values.get(
                    "asset_type", "UNAVAILABLE_AT_DECISION"
                ),
                "asset_metadata_status": "AVAILABLE_AT_DECISION",
                "asset_metadata_source": "BROAD_MULTI_ASSET_MANIFEST",
                "asset_metadata_source_hash": broad_hash,
                "asset_metadata_source_accepted_at": broad_accepted_at_iso,
            }
            for symbol, values in broad_asset_metadata(project_root).items()
        }
    security_master_path = project_root / SECURITY_MASTER_PATH
    if not security_master_path.is_file():
        return metadata
    source_accepted_at = datetime.fromtimestamp(
        security_master_path.stat().st_mtime,
        tz=UTC,
    )
    if source_accepted_at > accepted_at:
        return metadata
    required = {
        "security_id",
        "ticker",
        "sector",
        "industry",
        "is_delisted",
        "source_hash",
    }
    try:
        security_master = pd.read_parquet(
            security_master_path,
            columns=sorted(required),
        )
    except (OSError, ValueError, KeyError):
        return metadata
    if not required.issubset(security_master.columns):
        return metadata
    active = security_master.loc[
        ~security_master["is_delisted"].fillna(True).astype(bool)
    ].copy()
    active["ticker"] = active["ticker"].astype(str).str.strip().str.upper()
    active["sector"] = active["sector"].fillna("").astype(str).str.strip()
    active["industry"] = active["industry"].fillna("").astype(str).str.strip()
    active = active.loc[
        active["ticker"].ne("")
        & active["sector"].ne("")
        & active["industry"].ne("")
    ]
    for symbol, rows in active.groupby("ticker", sort=True):
        if symbol in metadata:
            continue
        classifications = {
            (str(row.sector), str(row.industry))
            for row in rows.itertuples()
        }
        if len(classifications) != 1:
            metadata[str(symbol)] = {
                "sector": "UNAVAILABLE_AT_DECISION",
                "industry": "UNAVAILABLE_AT_DECISION",
                "asset_metadata_status": "AMBIGUOUS_CLASSIFICATION_BLOCKED",
                "asset_metadata_source": "SECURITY_MASTER_POINT_IN_TIME",
                "asset_metadata_source_hash": None,
                "asset_metadata_source_accepted_at": (
                    source_accepted_at.isoformat()
                ),
            }
            continue
        row = rows.sort_values("security_id", kind="stable").iloc[0]
        metadata[str(symbol)] = {
            "sector": str(row["sector"]),
            "industry": str(row["industry"]),
            "asset_metadata_status": "AVAILABLE_AT_DECISION",
            "asset_metadata_source": "SECURITY_MASTER_POINT_IN_TIME",
            "asset_metadata_source_hash": (
                str(row["source_hash"])
                if pd.notna(row["source_hash"])
                else None
            ),
            "asset_metadata_source_accepted_at": (
                source_accepted_at.isoformat()
            ),
        }
    return metadata


def phase11_14_schema(project_root: Path) -> dict[str, Any]:
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "source": "PHASE11_12_COST_STRESS_SURVIVORS",
        "primary_timeframes": list(PRIMARY_TIMEFRAMES),
        "profiles": list(PROFILES),
        "cost_stress_bps_per_side": list(COSTS_BPS),
        "selection": {
            "historical_result_positive": True,
            "minimum_sharpe": 0.40,
            "minimum_period_profit_factor": 1.05,
            "minimum_50bps_profit_factor": 1.00,
            "minimum_fill_count": 50,
            "maximum_drawdown": -0.30,
            "minimum_stable_profiles": 2,
            "economic_outcome_deduplication": True,
            "timeframe_and_asset_round_robin": True,
        },
        "exploratory_forward_observation": {
            "timeframes": sorted(EXPLORATORY_OBSERVER_TIMEFRAMES),
            "research_pass_required": True,
            "portfolio_invariants_required": True,
            "minimum_period_profit_factor": 1.05,
            "minimum_positive_fold_ratio": 0.50,
            "minimum_fill_count": 50,
            "minimum_50bps_combined_return": 0.0,
            "maximum_drawdown": -0.30,
            "portfolio_eligible": False,
            "execution_eligible": False,
            "authority": "NONE",
        },
        "nested_validation": {
            "outer_fold_count": 6,
            "profile_selection": "VALIDATION_ONLY",
            "profile_plateau_required_for_robustness_report": True,
            "entry_information": "CLOSED_BAR_ONLY",
            "execution": "NEXT_BAR_OPEN",
            "whole_shares": True,
            "base_currency": "EUR",
            "maximum_gross_exposure": 1.0,
            "benchmark": "EXPOSURE_MATCHED_SPY",
        },
        "evidence_scope": (
            "SELECTION_CONDITIONED_REUSED_HISTORY_NOT_INDEPENDENT_CONFIRMATION"
        ),
        "automatic_promotion": False,
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "schema.json", report)
    return report


def select_survivor_cohort(
    summary: pd.DataFrame,
    *,
    max_candidates: int = 16,
) -> pd.DataFrame:
    if not 4 <= max_candidates <= 32:
        raise ValueError("max_candidates must be in [4,32]")
    required = {
        "strategy_id",
        "strategy_hash",
        "formula",
        "timeframe",
        "profile",
        "asset_class",
        "status",
        "CAGR",
        "Sharpe",
        "period_profit_factor",
        "stress_50bps_profit_factor",
        "maximum_drawdown",
        "fill_count",
        "economic_outcome_fingerprint",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"PHASE11_12_SUMMARY_COLUMNS_MISSING:{missing}")

    work = summary.copy()
    numeric = (
        "CAGR",
        "Sharpe",
        "period_profit_factor",
        "stress_50bps_profit_factor",
        "maximum_drawdown",
        "fill_count",
    )
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    primary = work["timeframe"].isin(PRIMARY_TIMEFRAMES)
    soft_gate = (
        work["status"].eq("COMPLETE")
        & primary
        & work["CAGR"].gt(0)
        & work["period_profit_factor"].gt(1.0)
        & work["stress_50bps_profit_factor"].gt(1.0)
        & work["maximum_drawdown"].ge(-0.35)
        & work["fill_count"].ge(30)
    )
    profile_counts = (
        work.loc[soft_gate]
        .groupby(["formula", "timeframe", "asset_class"])["profile"]
        .nunique()
    )
    work["stable_profile_count"] = [
        int(
            profile_counts.get(
                (row.formula, row.timeframe, row.asset_class),
                0,
            )
        )
        for row in work.itertuples()
    ]
    hard_gate = (
        work["status"].eq("COMPLETE")
        & primary
        & work["CAGR"].gt(0)
        & work["Sharpe"].ge(0.40)
        & work["period_profit_factor"].ge(1.05)
        & work["stress_50bps_profit_factor"].ge(1.0)
        & work["maximum_drawdown"].ge(-0.30)
        & work["fill_count"].ge(50)
        & work["stable_profile_count"].ge(2)
    )
    candidates = work.loc[hard_gate].sort_values(
        [
            "Sharpe",
            "stress_50bps_profit_factor",
            "period_profit_factor",
            "CAGR",
            "strategy_id",
        ],
        ascending=[False, False, False, False, True],
    )
    candidates = candidates.drop_duplicates(
        "economic_outcome_fingerprint",
        keep="first",
    ).drop_duplicates(
        ["formula", "timeframe", "asset_class"],
        keep="first",
    )

    per_timeframe_limit = max(1, math.ceil(max_candidates / 4))
    selected: list[int] = []
    fingerprints: set[str] = set()
    formula_counts: Counter[str] = Counter()
    timeframe_counts: Counter[str] = Counter()

    def add_first(frame: pd.DataFrame) -> bool:
        for index, row in frame.iterrows():
            fingerprint = str(row["economic_outcome_fingerprint"])
            formula = str(row["formula"])
            timeframe = str(row["timeframe"])
            if index in selected or fingerprint in fingerprints:
                continue
            if formula_counts[formula] >= 2:
                continue
            if timeframe_counts[timeframe] >= per_timeframe_limit:
                continue
            selected.append(index)
            fingerprints.add(fingerprint)
            formula_counts[formula] += 1
            timeframe_counts[timeframe] += 1
            return True
        return False

    for timeframe in PRIMARY_TIMEFRAMES:
        for asset_class in ASSET_CLASS_ORDER:
            add_first(
                candidates.loc[
                    candidates["timeframe"].eq(timeframe)
                    & candidates["asset_class"].eq(asset_class)
                ]
            )
            if len(selected) >= max_candidates:
                break

    while len(selected) < max_candidates:
        advanced = False
        for timeframe in PRIMARY_TIMEFRAMES:
            advanced = (
                add_first(
                    candidates.loc[
                        candidates["timeframe"].eq(timeframe)
                    ]
                )
                or advanced
            )
            if len(selected) >= max_candidates:
                break
        if not advanced:
            break

    cohort = work.loc[selected].copy().reset_index(drop=True)
    cohort["source_strategy_id"] = cohort["strategy_id"].astype(str)
    cohort["qualification_strategy_id"] = [
        "P1114-"
        + stable_hash(
            {
                "formula": row.formula,
                "timeframe": row.timeframe,
                "asset_class": row.asset_class,
                "selection_contract": SCHEMA,
            }
        )[:20]
        for row in cohort.itertuples()
    ]
    cohort["selection_rank"] = np.arange(1, len(cohort) + 1)
    return cohort


def run_phase11_14(
    project_root: Path,
    *,
    max_candidates: int = 16,
) -> dict[str, Any]:
    output = _output(project_root)
    output.mkdir(parents=True, exist_ok=True)
    schema = phase11_14_schema(project_root)
    schema_hash = stable_hash(schema)
    boundary_path = output / "qualification-boundary.json"
    if boundary_path.exists():
        boundary = _read_json(boundary_path)
        if boundary.get("schema_hash") == schema_hash:
            return {
                **phase11_14_status(project_root),
                "run_status": "QUALIFICATION_FROZEN_NO_REEVALUATION",
            }

    source_path = (
        project_root
        / "output"
        / "research"
        / "phase11_12"
        / "strategy-summary.parquet"
    )
    if not source_path.exists():
        return _blocked("PHASE11_12_SUMMARY_MISSING")
    source = pd.read_parquet(source_path)
    cohort = select_survivor_cohort(
        source,
        max_candidates=max_candidates,
    )
    if cohort.empty:
        return _blocked("NO_DIVERSIFIED_COST_STRESS_SURVIVOR_COHORT")

    source_hash = stable_hash(
        source.sort_values("strategy_id").to_dict("records")
    )
    _write_frame(output / "selected-cohort.parquet", cohort)
    _write_json(
        output / "selected-cohort.json",
        {
            "schema": "phase11_14_selected_cohort_v1",
            "status": "GO",
            "source_summary_hash": source_hash,
            "candidate_count": len(cohort),
            "candidates": json.loads(cohort.to_json(orient="records")),
            **AUTHORITY,
        },
    )

    all_frames = _load_frames(project_root)
    fold_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []

    for candidate in cohort.to_dict("records"):
        qualification_id = str(candidate["qualification_strategy_id"])
        timeframe = str(candidate["timeframe"])
        formula = str(candidate["formula"])
        asset_class = str(candidate["asset_class"])
        available = all_frames.get(timeframe, {})
        symbols = set(ASSET_BUCKETS[asset_class])
        frames = {
            symbol: frame
            for symbol, frame in available.items()
            if symbol in symbols and not frame.empty
        }
        if len(frames) < 3 or "SPY" not in available:
            blocked_rows.append(
                {
                    "strategy_id": qualification_id,
                    "formula": formula,
                    "timeframe": timeframe,
                    "asset_class": asset_class,
                    "reason": "INSUFFICIENT_REAL_DATA_OR_BENCHMARK",
                    "instrument_count": len(frames),
                }
            )
            continue
        start = max(frame.index.min() for frame in frames.values())
        end = min(frame.index.max() for frame in frames.values())
        folds = nested_walk_forward_folds(start, end, timeframe).tail(6)
        coverage_rows.append(
            {
                "strategy_id": qualification_id,
                "source_strategy_id": candidate["source_strategy_id"],
                "formula": formula,
                "timeframe": timeframe,
                "asset_class": asset_class,
                "instrument_count": len(frames),
                "start": start,
                "end": end,
                "fold_count": len(folds),
            }
        )
        if len(folds) < 6:
            blocked_rows.append(
                {
                    "strategy_id": qualification_id,
                    "reason": "FEWER_THAN_SIX_OUTER_FOLDS",
                    "fold_count": len(folds),
                }
            )
            continue

        signal_cache = {
            profile: _formula_signals(
                frames,
                formula,
                timeframe,
                profile,
            )
            for profile in PROFILES
        }
        benchmark_frame = {"SPY": available["SPY"]}
        benchmark_signal = {
            "SPY": pd.DataFrame(
                {"signal": True, "score": 1.0},
                index=available["SPY"].index,
            )
        }
        for fold in folds.to_dict("records"):
            profile, plateau, validations = _select_validation_profile(
                frames,
                signal_cache,
                fold,
            )
            for validation in validations:
                profile_rows.append(
                    {
                        "strategy_id": qualification_id,
                        "fold_id": fold["fold_id"],
                        **validation,
                    }
                )
            normal_run: dict[str, Any] | None = None
            normal_row_index: int | None = None
            for cost_bps in COSTS_BPS:
                run = _run_portfolio(
                    frames,
                    signal_cache[profile],
                    start=pd.Timestamp(fold["outer_test_start"]),
                    end=pd.Timestamp(fold["outer_test_end"]),
                    cost_bps=cost_bps,
                )
                if cost_bps == 10.0:
                    normal_run = run
                    normal_row_index = len(fold_rows)
                _append_returns(
                    return_rows,
                    run,
                    strategy_id=qualification_id,
                    fold_id=str(fold["fold_id"]),
                    cost_bps=cost_bps,
                )
                _append_fills(
                    fill_rows,
                    run,
                    strategy_id=qualification_id,
                    fold_id=str(fold["fold_id"]),
                    cost_bps=cost_bps,
                )
                fold_rows.append(
                    {
                        "strategy_id": qualification_id,
                        "source_strategy_id": candidate[
                            "source_strategy_id"
                        ],
                        "formula": formula,
                        "fold_id": fold["fold_id"],
                        "timeframe": timeframe,
                        "asset_class": asset_class,
                        "profile": profile,
                        "parameter_plateau": plateau,
                        "cost_bps": cost_bps,
                        "fill_count": len(run["fills"]),
                        "maximum_gross_exposure": _maximum_exposure(run),
                        "minimum_cash_eur": _minimum_cash(run),
                        "duplicate_position_days": int(
                            run["duplicate_position_days"]
                        ),
                        "whole_share_violation_count": (
                            _whole_share_violation_count(run)
                        ),
                        **run["metrics"],
                    }
                )
            if normal_run is None or normal_row_index is None:
                continue
            exposure = normal_run["ledger"].set_index("date")[
                "gross_exposure"
            ]
            benchmark = _run_portfolio(
                benchmark_frame,
                benchmark_signal,
                start=pd.Timestamp(fold["outer_test_start"]),
                end=pd.Timestamp(fold["outer_test_end"]),
                cost_bps=10.0,
                exposure_multiplier=exposure,
            )
            strategy_cagr = _finite(normal_run["metrics"].get("CAGR"))
            benchmark_cagr = _finite(benchmark["metrics"].get("CAGR"))
            fold_rows[normal_row_index][
                "exposure_matched_benchmark_CAGR"
            ] = benchmark_cagr
            fold_rows[normal_row_index][
                "exposure_matched_alpha_CAGR"
            ] = strategy_cagr - benchmark_cagr

    folds = pd.DataFrame(fold_rows)
    returns = pd.DataFrame(return_rows)
    fills = pd.DataFrame(fill_rows)
    summary = _summarize(folds, returns)
    metadata_columns = [
        "qualification_strategy_id",
        "source_strategy_id",
        "formula",
        "timeframe",
        "asset_class",
        "stable_profile_count",
        "selection_rank",
    ]
    metadata = cohort[metadata_columns].rename(
        columns={"qualification_strategy_id": "strategy_id"}
    )
    if not summary.empty:
        summary = summary.merge(
            metadata,
            on=["strategy_id", "timeframe"],
            how="left",
        )
        modes = (
            pd.DataFrame(profile_rows)
            .groupby("strategy_id")["profile"]
            .agg(_deterministic_mode)
            .rename("frozen_profile")
        )
        summary = summary.merge(
            modes,
            left_on="strategy_id",
            right_index=True,
            how="left",
        )
        summary["forward_observer_candidate"] = (
            summary["robust_pass"]
            & summary["portfolio_invariants_go"]
        )
        summary["financial_finalist"] = False
        summary["evidence_scope"] = (
            "SELECTION_CONDITIONED_REUSED_HISTORY"
        )

    _write_frame(output / "fold-results.parquet", folds)
    _write_frame(
        output / "validation-profile-selection.parquet",
        pd.DataFrame(profile_rows),
    )
    _write_frame(output / "oos-returns.parquet", returns)
    _write_frame(output / "fills.parquet", fills)
    _write_frame(output / "strategy-summary.parquet", summary)
    _write_frame(output / "coverage.csv", pd.DataFrame(coverage_rows))
    _write_json(output / "blocked.json", blocked_rows)

    robust_ids = (
        sorted(
            summary.loc[
                summary["forward_observer_candidate"],
                "strategy_id",
            ].astype(str)
        )
        if not summary.empty
        else []
    )
    qualification = {
        "schema": "phase11_14_qualification_v1",
        "status": "GO" if not summary.empty else "NO_EVALUABLE_STRATEGIES",
        "selected_candidate_count": len(cohort),
        "evaluated_candidate_count": len(summary),
        "research_pass_count": (
            int(summary["research_pass"].sum()) if not summary.empty else 0
        ),
        "robust_pass_count": (
            int(summary["robust_pass"].sum()) if not summary.empty else 0
        ),
        "forward_observer_candidate_count": len(robust_ids),
        "financial_finalist_count": 0,
        "strategies": (
            json.loads(summary.to_json(orient="records"))
            if not summary.empty
            else []
        ),
        "automatic_promotion": False,
        **AUTHORITY,
    }
    _write_json(output / "qualification.json", qualification)
    selection_bias = {
        "schema": "phase11_14_selection_bias_audit_v1",
        "status": "INDEPENDENT_CONFIRMATION_REQUIRED",
        "source_hypothesis_count": len(source),
        "selected_candidate_count": len(cohort),
        "evaluated_candidate_count": len(summary),
        "selection_used_historical_metrics": True,
        "nested_folds_overlap_source_research_period": True,
        "multiple_testing_corrected_finalist_count": 0,
        "independent_future_observations": 0,
        "interpretation": (
            "Nested robustness can nominate frozen observers, but cannot turn "
            "selection-conditioned reused history into independent evidence."
        ),
        **AUTHORITY,
    }
    _write_json(output / "selection-bias-audit.json", selection_bias)

    ends = {
        str(row["strategy_id"]): pd.Timestamp(row["end"]).isoformat()
        for row in coverage_rows
        if row["strategy_id"] in robust_ids
    }
    boundary = {
        "schema": "phase11_14_qualification_boundary_v1",
        "status": "FROZEN",
        "schema_hash": schema_hash,
        "source_summary_hash": source_hash,
        "qualification_hash": stable_hash(qualification),
        "selected_strategy_ids": sorted(
            cohort["qualification_strategy_id"].astype(str)
        ),
        "robust_strategy_ids": robust_ids,
        "data_end_by_strategy": ends,
        "frozen_at": datetime.now(UTC).isoformat(),
        "automatic_requalification": False,
        **AUTHORITY,
    }
    _write_json(boundary_path, boundary)
    report = {
        "schema": SCHEMA,
        "status": (
            "GO"
            if len(summary) == len(cohort)
            else "PARTIAL"
        ),
        "selected_candidate_count": len(cohort),
        "evaluated_candidate_count": len(summary),
        "blocked_candidate_count": len(blocked_rows),
        "research_pass_count": qualification["research_pass_count"],
        "robust_pass_count": qualification["robust_pass_count"],
        "forward_observer_candidate_count": len(robust_ids),
        "financial_decision": (
            "ROBUST_FORWARD_OBSERVER_CANDIDATES_AVAILABLE"
            if robust_ids
            else "NO_ROBUST_FORWARD_OBSERVER_CANDIDATE"
        ),
        "qualification": qualification,
        "selection_bias_audit": selection_bias,
        "qualification_boundary": boundary,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(report)
    _write_json(output / "status.json", report)
    _write_json(output / "manifest.json", report)
    return report


def phase11_14_status(project_root: Path) -> dict[str, Any]:
    path = _output(project_root) / "status.json"
    if not path.exists():
        return {
            "schema": SCHEMA,
            "status": "NOT_RUN",
            **AUTHORITY,
        }
    return _read_json(path)


def _select_exploratory_observers(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "strategy_id",
        "timeframe",
        "research_pass",
        "robust_pass",
        "portfolio_invariants_go",
        "combined_period_profit_factor",
        "cost_50bps_combined_return",
        "normal_cost_fill_count",
        "positive_fold_count",
        "fold_count",
        "maximum_drawdown",
    }
    if summary.empty or not required.issubset(summary.columns):
        return summary.iloc[0:0].copy()
    work = summary.copy()
    numeric = (
        "combined_period_profit_factor",
        "cost_50bps_combined_return",
        "normal_cost_fill_count",
        "positive_fold_count",
        "fold_count",
        "maximum_drawdown",
    )
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    positive_fold_ratio = work["positive_fold_count"] / work[
        "fold_count"
    ].replace(0, np.nan)
    selected = work.loc[
        work["timeframe"].astype(str).isin(EXPLORATORY_OBSERVER_TIMEFRAMES)
        & work["research_pass"].fillna(False).astype(bool)
        & ~work["robust_pass"].fillna(False).astype(bool)
        & work["portfolio_invariants_go"].fillna(False).astype(bool)
        & work["combined_period_profit_factor"].ge(1.05)
        & work["cost_50bps_combined_return"].gt(0.0)
        & work["normal_cost_fill_count"].ge(50)
        & positive_fold_ratio.ge(0.50)
        & work["maximum_drawdown"].ge(-0.30)
    ].copy()
    return selected.sort_values(
        [
            "timeframe",
            "cost_50bps_combined_return",
            "combined_period_profit_factor",
            "strategy_id",
        ],
        ascending=[True, False, False, True],
    )


def _exploratory_boundary(
    output: Path,
    summary: pd.DataFrame,
    *,
    qualification_hash: str,
) -> dict[str, Any]:
    selected = _select_exploratory_observers(summary)
    strategy_ids = sorted(selected["strategy_id"].astype(str))
    coverage_path = output / "coverage.csv"
    coverage = (
        pd.read_csv(coverage_path)
        if coverage_path.exists()
        else pd.DataFrame()
    )
    data_ends: dict[str, str] = {}
    if {"strategy_id", "end"}.issubset(coverage.columns):
        data_ends = {
            str(row["strategy_id"]): pd.Timestamp(row["end"]).isoformat()
            for _, row in coverage.iterrows()
            if str(row["strategy_id"]) in set(strategy_ids)
        }
    selection_semantics = {
        "source_qualification_hash": qualification_hash,
        "strategy_ids": strategy_ids,
        "timeframes": sorted(EXPLORATORY_OBSERVER_TIMEFRAMES),
        "minimum_period_profit_factor": 1.05,
        "minimum_positive_fold_ratio": 0.50,
        "minimum_fill_count": 50,
        "minimum_50bps_combined_return": 0.0,
        "maximum_drawdown": -0.30,
        "portfolio_eligible": False,
        "execution_eligible": False,
    }
    selection_hash = stable_hash(selection_semantics)
    path = output / "exploratory-observer-boundary.json"
    if path.exists():
        existing = _read_json(path)
        if (
            existing.get("selection_hash") != selection_hash
            or existing.get("source_qualification_hash")
            != qualification_hash
        ):
            return {
                "status": "BLOCKED",
                "reason": "EXPLORATORY_OBSERVER_BOUNDARY_MISMATCH",
                **AUTHORITY,
            }
        return existing
    boundary = {
        "schema": "phase11_14_exploratory_observer_boundary_v1",
        "status": "FROZEN",
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_qualification_hash": qualification_hash,
        "selection_hash": selection_hash,
        "strategy_ids": strategy_ids,
        "data_end_by_strategy": data_ends,
        "selection": selection_semantics,
        "automatic_promotion": False,
        "portfolio_eligible": False,
        "execution_eligible": False,
        **AUTHORITY,
    }
    _write_json(path, boundary)
    return boundary


def phase11_14_observe(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    status_path = output / "status.json"
    boundary_path = output / "qualification-boundary.json"
    summary_path = output / "strategy-summary.parquet"
    if not status_path.exists() or not boundary_path.exists():
        return _blocked("PHASE11_14_QUALIFICATION_NOT_FROZEN")
    if not summary_path.exists():
        return _blocked("PHASE11_14_STRATEGY_SUMMARY_MISSING")
    status = _read_json(status_path)
    boundary = _read_json(boundary_path)
    if boundary.get("status") != "FROZEN":
        return _blocked("PHASE11_14_BOUNDARY_NOT_FROZEN")
    qualification_hash = str(boundary.get("qualification_hash", ""))
    if not qualification_hash:
        return _blocked("PHASE11_14_QUALIFICATION_HASH_MISSING")

    summary = pd.read_parquet(summary_path)
    robust_ids = {
        str(value)
        for value in boundary.get("robust_strategy_ids", [])
    }
    robust = summary.loc[
        summary["strategy_id"].astype(str).isin(robust_ids)
        & summary["forward_observer_candidate"].fillna(False)
    ].copy()
    if robust.empty:
        return _blocked("NO_FROZEN_ROBUST_OBSERVER_CANDIDATES")
    exploratory_boundary = _exploratory_boundary(
        output,
        summary,
        qualification_hash=qualification_hash,
    )
    if exploratory_boundary.get("status") != "FROZEN":
        return _blocked(
            str(
                exploratory_boundary.get(
                    "reason",
                    "EXPLORATORY_OBSERVER_BOUNDARY_NOT_FROZEN",
                )
            )
        )
    exploratory_ids = {
        str(value)
        for value in exploratory_boundary.get("strategy_ids", [])
    }
    exploratory = summary.loc[
        summary["strategy_id"].astype(str).isin(exploratory_ids)
    ].copy()
    robust["observer_tier"] = "ROBUST_FORWARD_OBSERVER"
    robust["portfolio_eligible"] = True
    exploratory["observer_tier"] = "EXPLORATORY_FORWARD_OBSERVER"
    exploratory["portfolio_eligible"] = False
    observers = pd.concat([robust, exploratory], ignore_index=True)
    data_end_by_strategy = {
        **boundary.get("data_end_by_strategy", {}),
        **exploratory_boundary.get("data_end_by_strategy", {}),
    }

    observed_at = datetime.now(UTC)
    attestations = _current_attestations(project_root, observed_at)
    frames_by_timeframe = _forward_frames(project_root)
    asset_metadata = _forward_asset_metadata(
        project_root,
        observed_at=observed_at,
    )
    dynamic_regime = _read_json(
        project_root / "output/dynamic/current_regime.json"
    )
    market_regime = str(dynamic_regime.get("regime") or "UNAVAILABLE_AT_DECISION")
    asset_context = _read_json(
        project_root / "output/market_context/asset-context.json"
    )
    context_by_symbol = {
        str(row.get("symbol", "")).upper(): row
        for row in asset_context.get("contexts", [])
        if isinstance(row, Mapping) and row.get("symbol")
    }
    observations: list[dict[str, Any]] = []
    timestamps: dict[str, str | None] = {}
    for candidate in observers.to_dict("records"):
        strategy_id = str(candidate["strategy_id"])
        timeframe = str(candidate["timeframe"])
        formula = str(candidate["formula"])
        asset_class = str(candidate["asset_class"])
        profile = str(candidate.get("frozen_profile") or "balanced")
        observer_tier = str(candidate["observer_tier"])
        portfolio_eligible = bool(candidate["portfolio_eligible"])
        available = frames_by_timeframe.get(timeframe, {})
        frames = {
            symbol: frame
            for symbol, frame in available.items()
            if symbol in set(ASSET_BUCKETS[asset_class]) and not frame.empty
        }
        if not frames:
            timestamps[strategy_id] = None
            observations.append(
                {
                    "strategy_id": strategy_id,
                    "formula": formula,
                    "timeframe": timeframe,
                    "asset_class": asset_class,
                    "profile": profile,
                    "observer_tier": observer_tier,
                    "portfolio_eligible": portfolio_eligible,
                    "execution_eligible": False,
                    "observation_status": "DATA_UNAVAILABLE",
                    "raw_active_signals": [],
                    "current_attested_target_weights": {},
                    "order_intents": [],
                    "automatic_orders": 0,
                }
            )
            continue
        common_close = min(frame.index.max() for frame in frames.values())
        common_close_utc = cast(pd.Timestamp, _as_utc(common_close))
        freshness = _forward_freshness(
            common_close_utc,
            observed_at,
            timeframe,
        )
        timestamps[strategy_id] = common_close_utc.isoformat()
        signals = _formula_signals(
            frames,
            formula,
            timeframe,
            profile,
        )
        active: list[dict[str, Any]] = []
        for symbol, signal in signals.items():
            closed = signal.loc[signal.index <= common_close]
            if closed.empty:
                continue
            latest = closed.iloc[-1]
            if not bool(latest.get("signal", False)):
                continue
            score = _finite(latest.get("score"), -math.inf)
            if not math.isfinite(score):
                continue
            envelope = _execution_envelope(
                frames[symbol],
                common_close=common_close,
                formula=formula,
                timeframe=timeframe,
            )
            metadata = asset_metadata.get(symbol, {})
            context = context_by_symbol.get(symbol, {})
            components = context.get("components", {})
            gex = components.get("gex", {}) if isinstance(components, Mapping) else {}
            active.append(
                {
                    "symbol": symbol,
                    "score": round(score, 10),
                    "currently_attested": symbol in attestations,
                    "signal_id": stable_hash(
                        {
                            "strategy_id": strategy_id,
                            "symbol": symbol,
                            "closed_bar_timestamp": (
                                common_close_utc.isoformat()
                            ),
                            "formula": formula,
                        }
                    ),
                    "action": "BUY",
                    "data_timestamp": common_close_utc.isoformat(),
                    "data_freshness": (
                        "FRESH"
                        if freshness == "FRESH_CLOSED_BAR"
                        else freshness
                    ),
                    "decision_timestamp": observed_at.isoformat(),
                    "market_regime": market_regime,
                    "sector": metadata.get("sector", "UNAVAILABLE_AT_DECISION"),
                    "industry": metadata.get(
                        "industry", "UNAVAILABLE_AT_DECISION"
                    ),
                    "asset_metadata_status": metadata.get(
                        "asset_metadata_status", "UNAVAILABLE_AT_DECISION"
                    ),
                    "asset_metadata_source": metadata.get(
                        "asset_metadata_source", "UNAVAILABLE_AT_DECISION"
                    ),
                    "asset_metadata_source_hash": metadata.get(
                        "asset_metadata_source_hash"
                    ),
                    "asset_metadata_source_accepted_at": metadata.get(
                        "asset_metadata_source_accepted_at"
                    ),
                    "earnings_distance_days": None,
                    "options_context_status": (
                        gex.get("status", "UNAVAILABLE_AT_DECISION")
                        if isinstance(gex, Mapping)
                        else "UNAVAILABLE_AT_DECISION"
                    ),
                    "context_snapshot_timestamp": asset_context.get(
                        "generated_at"
                    ),
                    "observer_tier": observer_tier,
                    "portfolio_eligible": portfolio_eligible,
                    "execution_eligible": False,
                    "automatic_execution_allowed": False,
                    **envelope,
                }
            )
        active.sort(key=lambda row: (-float(row["score"]), row["symbol"]))
        for rank, row in enumerate(active, start=1):
            row["confidence_score"] = round(1.0 / rank, 6)
        target_weights = (
            _attested_target_weights(active, attestations)
            if portfolio_eligible
            else {}
        )
        data_end = _as_utc(
            data_end_by_strategy.get(strategy_id)
        )
        independent = (
            data_end is not None
            and common_close_utc > data_end
            and common_close_utc <= pd.Timestamp(observed_at)
        )
        observations.append(
            {
                "strategy_id": strategy_id,
                "formula": formula,
                "timeframe": timeframe,
                "asset_class": asset_class,
                "profile": profile,
                "observer_tier": observer_tier,
                "portfolio_eligible": portfolio_eligible,
                "execution_eligible": False,
                "observation_status": "OBSERVATION_COMPLETE",
                "closed_bar_timestamp": common_close_utc.isoformat(),
                "data_freshness": freshness,
                "independent_forward_session": independent,
                "raw_active_signals": active,
                "current_attested_target_weights": target_weights,
                "compliance_blocked_signal_count": sum(
                    not bool(row["currently_attested"]) for row in active
                ),
                "portfolio_action": (
                    "OBSERVE_SIGNAL_ONLY_NO_ALLOCATION"
                    if not portfolio_eligible
                    else (
                        "OBSERVE_HYPOTHETICAL_NEXT_BAR_TARGET"
                        if target_weights
                        else "NO_CURRENT_ATTESTED_TARGET"
                    )
                ),
                "order_intents": [],
                "automatic_orders": 0,
            }
        )

    observation_key = stable_hash(
        {
            "observation_contract_version": (
                "V3_EXPLORATORY_FORWARD_OBSERVER_TIER"
            ),
            "qualification_hash": qualification_hash,
            "exploratory_selection_hash": exploratory_boundary.get(
                "selection_hash"
            ),
            "closed_bar_timestamps": timestamps,
        }
    )
    observation_path = (
        output / "forward-observations" / f"{observation_key}.json"
    )
    if observation_path.exists():
        existing = _read_json(observation_path)
        all_observations = _load_forward_observations(output)
        audit = _forward_session_audit(
            boundary,
            all_observations,
            observed_at=observed_at,
        )
        performance = _build_forward_performance(
            boundary,
            all_observations,
            frames_by_timeframe,
            observed_at=observed_at,
        )
        _publish_forward_evidence(
            output,
            status,
            audit,
            performance,
            observation_count=len(all_observations),
        )
        return {
            **existing,
            "run_status": "IDEMPOTENT_NO_NEW_CLOSED_BAR",
            "independent_forward_audit": audit,
            "forward_performance_summary": _performance_summary(
                performance
            ),
        }

    prior = _load_forward_observations(output)
    audit = _forward_session_audit(
        boundary,
        [*prior, {"observations": observations, "observed_at": observed_at}],
        observed_at=observed_at,
    )
    payload = {
        "schema": "phase11_14_forward_observation_v3",
        "status": "GO",
        "observation_id": observation_key,
        "observed_at": observed_at.isoformat(),
        "qualification_hash": qualification_hash,
        "candidate_count": len(observations),
        "robust_candidate_count": len(robust),
        "exploratory_candidate_count": len(exploratory),
        "active_signal_count": sum(
            len(row.get("raw_active_signals", []))
            for row in observations
        ),
        "attested_target_count": sum(
            len(row.get("current_attested_target_weights", {}))
            for row in observations
        ),
        "observations": observations,
        "independent_forward_audit": audit,
        "automatic_promotion": False,
        "order_intents": [],
        "automatic_orders": 0,
        **AUTHORITY,
    }
    _write_json(observation_path, payload)
    _write_json(output / "latest-forward-observation.json", payload)
    all_observations = [*prior, payload]
    performance = _build_forward_performance(
        boundary,
        all_observations,
        frames_by_timeframe,
        observed_at=observed_at,
    )
    _publish_forward_evidence(
        output,
        status,
        audit,
        performance,
        observation_count=len(all_observations),
    )
    status["exploratory_forward_observer_count"] = len(exploratory)
    status["observer_candidate_count"] = len(observers)
    status["exploratory_observer_boundary_hash"] = (
        exploratory_boundary.get("selection_hash")
    )
    _write_json(output / "status.json", status)
    _write_json(output / "manifest.json", status)
    payload["forward_performance_summary"] = _performance_summary(
        performance
    )
    return payload


def _attested_target_weights(
    active: list[dict[str, Any]],
    attestations: set[str],
    *,
    maximum_positions: int = 4,
    maximum_position_weight: float = 0.25,
) -> dict[str, float]:
    selected = [
        row
        for row in active
        if str(row["symbol"]).upper() in attestations
    ][:maximum_positions]
    return {
        str(row["symbol"]).upper(): maximum_position_weight
        for row in selected
    }


def _execution_envelope(
    frame: pd.DataFrame,
    *,
    common_close: Any,
    formula: str,
    timeframe: str,
) -> dict[str, Any]:
    closed = frame.loc[frame.index <= common_close].tail(40)
    required = {"high", "low", "close"}
    if closed.empty or not required.issubset(closed.columns):
        return {"execution_envelope_status": "BLOCKED_MISSING_OHLC"}
    high = pd.to_numeric(closed["high"], errors="coerce")
    low = pd.to_numeric(closed["low"], errors="coerce")
    close = pd.to_numeric(closed["close"], errors="coerce")
    if high.isna().any() or low.isna().any() or close.isna().any():
        return {"execution_envelope_status": "BLOCKED_INVALID_OHLC"}
    reference = float(close.iloc[-1])
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    atr = float(true_range.tail(14).mean())
    if not math.isfinite(reference) or reference <= 0:
        return {"execution_envelope_status": "BLOCKED_INVALID_CLOSE"}
    if not math.isfinite(atr) or atr <= 0:
        atr = reference * 0.02
    structural_low = float(low.tail(20).min())
    minimum_risk = max(atr * 0.5, reference * 0.005)
    stop = max(structural_low, reference - 2.0 * atr)
    if stop >= reference - minimum_risk:
        stop = reference - minimum_risk
    risk = reference - stop
    reward_multiple = _reward_multiple(formula)
    target_1 = reference + reward_multiple * risk
    target_2 = reference + (reward_multiple + 1.0) * risk
    maximum_holding_bars = {
        "1h": 80,
        "4h": 60,
        "1d": 40,
        "1w": 20,
    }.get(timeframe, 40)
    return {
        "execution_envelope_status": "GO",
        "entry_reference": round(reference, 6),
        "stop_loss": round(stop, 6),
        "take_profit_1": round(target_1, 6),
        "take_profit_2": round(target_2, 6),
        "initial_risk_per_share": round(risk, 6),
        "reward_multiple": reward_multiple,
        "stop_policy": "ATR_ADJUSTED_20_BAR_STRUCTURE",
        "exit_policy": (
            "PRIMARY_TARGET_WITH_ATR_TRAIL_AND_TIME_STOP"
        ),
        "maximum_holding_bars": maximum_holding_bars,
        "execution_timing": "NEXT_BAR_AFTER_CLOSED_SIGNAL",
    }


def _reward_multiple(formula: str) -> float:
    normalized = formula.lower()
    if any(
        token in normalized
        for token in ("pullback", "reversion", "rsi", "stochastic")
    ):
        return 1.5
    if any(
        token in normalized
        for token in ("breakout", "trend", "momentum", "crossover", "channel")
    ):
        return 3.0
    return 2.0


def _forward_session_audit(
    boundary: Mapping[str, Any],
    observations: list[dict[str, Any]],
    *,
    observed_at: datetime,
) -> dict[str, Any]:
    robust_ids = {
        str(value)
        for value in boundary.get("robust_strategy_ids", [])
    }
    ends = {
        strategy_id: _as_utc(value)
        for strategy_id, value in boundary.get(
            "data_end_by_strategy", {}
        ).items()
        if strategy_id in robust_ids
    }
    sessions: dict[str, set[str]] = {
        strategy_id: set() for strategy_id in robust_ids
    }
    audit_time = pd.Timestamp(observed_at)
    for payload in observations:
        payload_observed = _as_utc(payload.get("observed_at"))
        if payload_observed is None or payload_observed > audit_time:
            continue
        for row in payload.get("observations", []):
            strategy_id = str(row.get("strategy_id", ""))
            if strategy_id not in robust_ids:
                continue
            closed = _as_utc(row.get("closed_bar_timestamp"))
            boundary_end = ends.get(strategy_id)
            if (
                closed is None
                or boundary_end is None
                or closed <= boundary_end
                or closed > payload_observed
            ):
                continue
            sessions[strategy_id].add(closed.isoformat())
    completed = sum(bool(values) for values in sessions.values())
    independent_sessions = {
        value for values in sessions.values() for value in values
    }
    return {
        "schema": "phase11_14_independent_forward_audit_v1",
        "status": (
            "INDEPENDENT_FORWARD_SESSION_COMPLETE"
            if robust_ids and completed == len(robust_ids)
            else (
                "INDEPENDENT_FORWARD_SESSION_PARTIAL"
                if completed
                else "NOT_YET_OBSERVED"
            )
        ),
        "qualification_hash": boundary.get("qualification_hash"),
        "robust_strategy_count": len(robust_ids),
        "completed_strategy_count": completed,
        "independent_session_count": len(independent_sessions),
        "per_strategy": {
            strategy_id: {
                "complete": bool(sessions[strategy_id]),
                "independent_session_count": len(sessions[strategy_id]),
                "independent_sessions": sorted(sessions[strategy_id]),
                "qualification_data_end": (
                    _timestamp_iso_or_none(ends.get(strategy_id))
                ),
            }
            for strategy_id in sorted(robust_ids)
        },
        "same_or_prior_bar_counted": False,
        "automatic_orders": 0,
        **AUTHORITY,
    }


def _load_forward_observations(output: Path) -> list[dict[str, Any]]:
    root = output / "forward-observations"
    if not root.exists():
        return []
    return [
        payload
        for path in sorted(root.glob("*.json"))
        if (payload := _read_json(path)).get("schema")
        in FORWARD_OBSERVATION_SCHEMAS
    ]


def _publish_forward_evidence(
    output: Path,
    status: dict[str, Any],
    audit: dict[str, Any],
    performance: dict[str, Any],
    *,
    observation_count: int,
) -> None:
    _write_json(output / "forward-audit.json", audit)
    _write_json(output / "forward-performance.json", performance)
    history_path = (
        output
        / "forward-performance-history"
        / f"{performance['content_hash']}.json"
    )
    if not history_path.exists():
        _write_json(history_path, performance)
    status["independent_forward_audit"] = audit
    status["independent_forward_session_status"] = audit["status"]
    status["independent_forward_session_count"] = audit[
        "independent_session_count"
    ]
    status["forward_observation_count"] = observation_count
    status["independent_forward_performance_status"] = performance[
        "status"
    ]
    status["independent_forward_episode_count"] = performance["counts"][
        "episode_count"
    ]
    status["independent_forward_closed_episode_count"] = performance[
        "counts"
    ]["closed_episode_count"]
    status["independent_forward_sample_status"] = performance[
        "aggregate"
    ]["sample_status"]
    status["FORWARD_RESEARCH_SHADOW"] = "OBSERVATION_ONLY"
    status["FINANCIAL_FINALIST_GO"] = False
    status["STRATEGY_AUTHORITY"] = "NONE"
    status["EXECUTION_AUTHORITY"] = "NONE"
    status["content_hash"] = stable_hash(
        {
            key: value
            for key, value in status.items()
            if key != "content_hash"
        }
    )
    _write_json(output / "status.json", status)
    _write_json(output / "manifest.json", status)


def _build_forward_performance(
    boundary: Mapping[str, Any],
    observations: list[dict[str, Any]],
    frames_by_timeframe: Mapping[str, Mapping[str, pd.DataFrame]],
    *,
    observed_at: datetime,
    cost_bps_per_side: float = FORWARD_COST_BPS_PER_SIDE,
) -> dict[str, Any]:
    if cost_bps_per_side < 0:
        raise ValueError("cost_bps_per_side must be non-negative")
    robust_ids = {
        str(value) for value in boundary.get("robust_strategy_ids", [])
    }
    boundary_ends = {
        strategy_id: _as_utc(value)
        for strategy_id, value in boundary.get(
            "data_end_by_strategy", {}
        ).items()
        if strategy_id in robust_ids
    }
    audit_time = pd.Timestamp(observed_at)
    ordered = sorted(
        observations,
        key=lambda payload: (
            str(payload.get("observed_at", "")),
            str(payload.get("observation_id", "")),
        ),
    )
    latest_active: dict[tuple[str, str], bool] = {}
    seen_sessions: set[tuple[str, str]] = set()
    strategy_sessions: dict[str, set[str]] = {
        strategy_id: set() for strategy_id in robust_ids
    }
    episodes: list[dict[str, Any]] = []
    evidence_times: list[pd.Timestamp] = []
    for payload in ordered:
        payload_observed = _as_utc(payload.get("observed_at"))
        if payload_observed is None or payload_observed > audit_time:
            continue
        evidence_times.append(payload_observed)
        for row in payload.get("observations", []):
            strategy_id = str(row.get("strategy_id", ""))
            signal_time = _as_utc(row.get("closed_bar_timestamp"))
            boundary_end = boundary_ends.get(strategy_id)
            if (
                strategy_id not in robust_ids
                or signal_time is None
                or boundary_end is None
                or signal_time <= boundary_end
                or signal_time > payload_observed
            ):
                continue
            session_key = (strategy_id, signal_time.isoformat())
            if session_key in seen_sessions:
                continue
            seen_sessions.add(session_key)
            strategy_sessions[strategy_id].add(signal_time.isoformat())
            active_rows = {
                str(signal.get("symbol", "")).upper(): signal
                for signal in row.get("raw_active_signals", [])
                if signal.get("symbol")
            }
            previous_symbols = {
                symbol
                for candidate_id, symbol in latest_active
                if candidate_id == strategy_id
            }
            for symbol in previous_symbols - set(active_rows):
                latest_active[(strategy_id, symbol)] = False
            for symbol, signal in sorted(active_rows.items()):
                key = (strategy_id, symbol)
                is_onset = not latest_active.get(key, False)
                latest_active[key] = True
                if not is_onset:
                    continue
                episode = _evaluate_forward_episode(
                    strategy_id=strategy_id,
                    formula=str(row.get("formula", "")),
                    timeframe=str(row.get("timeframe", "")),
                    asset_class=str(row.get("asset_class", "")),
                    symbol=symbol,
                    signal_time=signal_time,
                    decision_time=payload_observed,
                    signal=signal,
                    frame=frames_by_timeframe.get(
                        str(row.get("timeframe", "")), {}
                    ).get(symbol),
                    evidence_end=audit_time,
                    qualification_hash=str(
                        boundary.get("qualification_hash", "")
                    ),
                    cost_bps_per_side=cost_bps_per_side,
                )
                episodes.append(episode)

    per_strategy = [
        {
            **_aggregate_forward_episodes(
                [
                    episode
                    for episode in episodes
                    if episode["strategy_id"] == strategy_id
                ],
                strategy_id=strategy_id,
            ),
            "independent_session_count": len(
                strategy_sessions[strategy_id]
            ),
        }
        for strategy_id in sorted(robust_ids)
    ]
    aggregate = {
        **_aggregate_forward_episodes(
            episodes,
            strategy_id="ALL_ROBUST_STRATEGIES",
        ),
        "independent_session_count": len(
            {session for values in strategy_sessions.values() for session in values}
        ),
        "completed_strategy_count": sum(
            bool(values) for values in strategy_sessions.values()
        ),
        "robust_strategy_count": len(robust_ids),
    }
    semantic = {
        "qualification_hash": boundary.get("qualification_hash"),
        "observation_ids": sorted(
            str(payload.get("observation_id", ""))
            for payload in ordered
            if payload.get("observation_id")
        ),
        "episode_hashes": [episode["episode_hash"] for episode in episodes],
        "cost_bps_per_side": cost_bps_per_side,
        "evidence_end": (
            max(evidence_times).isoformat() if evidence_times else None
        ),
    }
    report = {
        "schema": "phase11_14_forward_performance_v1",
        "status": (
            "FORWARD_OUTCOMES_EVALUABLE"
            if aggregate["sample_status"] == "EVALUABLE"
            else (
                "FORWARD_OUTCOMES_LOW_CONFIDENCE"
                if aggregate["sample_status"] == "LOW_CONFIDENCE"
                else "FORWARD_OUTCOMES_INSUFFICIENT_SAMPLE"
            )
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        "evidence_end": semantic["evidence_end"],
        "qualification_hash": boundary.get("qualification_hash"),
        "observation_count": len(ordered),
        "counts": {
            "independent_session_count": aggregate[
                "independent_session_count"
            ],
            "completed_strategy_count": aggregate[
                "completed_strategy_count"
            ],
            "robust_strategy_count": aggregate["robust_strategy_count"],
            "episode_count": len(episodes),
            "closed_episode_count": sum(
                episode["outcome_status"].startswith("CLOSED_")
                for episode in episodes
            ),
            "open_episode_count": sum(
                episode["outcome_status"] == "OPEN"
                for episode in episodes
            ),
            "awaiting_next_bar_count": sum(
                episode["outcome_status"] == "AWAITING_NEXT_BAR"
                for episode in episodes
            ),
            "blocked_episode_count": sum(
                episode["outcome_status"].startswith("BLOCKED_")
                for episode in episodes
            ),
        },
        "cost_model": {
            "cost_bps_per_side": cost_bps_per_side,
            "round_trip_cost_bps": cost_bps_per_side * 2.0,
            "spread_slippage_commission_bundle": True,
        },
        "causality": {
            "signal_source": "APPEND_ONLY_POINT_IN_TIME_OBSERVATION",
            "entry_timing": "FIRST_AVAILABLE_BAR_OPEN_AFTER_APPEND_ONLY_DECISION",
            "same_bar_stop_target_policy": "STOP_FIRST_PESSIMISTIC",
            "persistent_signal_policy": "ONE_EPISODE_UNTIL_SIGNAL_OFF",
            "closed_episode_pf_only": True,
            "future_bar_before_signal_used": False,
            "future_bar_before_decision_used": False,
        },
        "aggregate": aggregate,
        "per_strategy": per_strategy,
        "episodes": episodes,
        "evidence_scope": "RESEARCH_ONLY_RAW_SIGNAL_FORWARD_OUTCOMES",
        "automatic_promotion": False,
        "automatic_orders": 0,
        **AUTHORITY,
    }
    report["content_hash"] = stable_hash(semantic)
    return report


def _evaluate_forward_episode(
    *,
    strategy_id: str,
    formula: str,
    timeframe: str,
    asset_class: str,
    symbol: str,
    signal_time: pd.Timestamp,
    signal: Mapping[str, Any],
    frame: pd.DataFrame | None,
    evidence_end: pd.Timestamp,
    qualification_hash: str,
    cost_bps_per_side: float,
    decision_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    decision_time = decision_time if decision_time is not None else signal_time
    episode_id = stable_hash(
        {
            "qualification_hash": qualification_hash,
            "strategy_id": strategy_id,
            "symbol": symbol,
            "signal_time": signal_time.isoformat(),
            "decision_time": decision_time.isoformat(),
        }
    )
    base = {
        "episode_id": episode_id,
        "strategy_id": strategy_id,
        "formula": formula,
        "timeframe": timeframe,
        "asset_class": asset_class,
        "symbol": symbol,
        "signal_timestamp": signal_time.isoformat(),
        "decision_timestamp": decision_time.isoformat(),
        "market_regime": signal.get("market_regime", "UNAVAILABLE_AT_DECISION"),
        "sector": signal.get("sector", "UNAVAILABLE_AT_DECISION"),
        "industry": signal.get("industry", "UNAVAILABLE_AT_DECISION"),
        "asset_metadata_status": signal.get(
            "asset_metadata_status", "UNAVAILABLE_AT_DECISION"
        ),
        "asset_metadata_source": signal.get(
            "asset_metadata_source", "UNAVAILABLE_AT_DECISION"
        ),
        "asset_metadata_source_hash": signal.get(
            "asset_metadata_source_hash"
        ),
        "asset_metadata_source_accepted_at": signal.get(
            "asset_metadata_source_accepted_at"
        ),
        "earnings_distance_days": _finite_or_none(
            signal.get("earnings_distance_days")
        ),
        "options_context_status": signal.get(
            "options_context_status", "UNAVAILABLE_AT_DECISION"
        ),
        "hypothetical_spread_bps": _finite_or_none(
            signal.get("spread_bps")
        ),
        "realized_spread_bps": None,
        "spread_evidence_status": (
            "HYPOTHETICAL_SIGNAL_SNAPSHOT"
            if _finite_or_none(signal.get("spread_bps")) is not None
            else "UNAVAILABLE_INCLUDED_IN_COST_BUNDLE"
        ),
        "entry_quality": "NEXT_BAR_OPEN_CAUSAL_NO_TAPE_ATTESTATION",
        "currently_attested_at_signal": bool(
            signal.get("currently_attested", False)
        ),
        "execution_timing": "NEXT_BAR_OPEN",
        "proposed_entry_type": "NEXT_BAR_OPEN",
        "proposed_limit": None,
        "maximum_valid_entry": None,
        "would_fill": None,
        "fill_timestamp": None,
        "fill_price": None,
        "fill_fraction": None,
        "estimated_commission": None,
        "estimated_slippage": None,
        "cost_attribution_status": "BUNDLED_BPS_NOT_SEPARATELY_OBSERVED",
        "evidence_scope": "RESEARCH_ONLY_RAW_SIGNAL",
        "automatic_orders": 0,
        "EXECUTION_AUTHORITY": "NONE",
    }
    if signal.get("execution_envelope_status") != "GO":
        return _episode_result(
            base,
            outcome_status="BLOCKED_EXECUTION_ENVELOPE",
        )
    prepared = _prepare_forward_frame(frame, evidence_end=evidence_end)
    if prepared.empty:
        return _episode_result(base, outcome_status="BLOCKED_DATA_UNAVAILABLE")
    future = prepared.loc[prepared.index > decision_time]
    if future.empty:
        return _episode_result(base, outcome_status="AWAITING_NEXT_BAR")
    initial_risk = _finite(signal.get("initial_risk_per_share"), 0.0)
    reward_multiple = _finite(signal.get("reward_multiple"), 0.0)
    maximum_holding_bars = int(signal.get("maximum_holding_bars", 0) or 0)
    if initial_risk <= 0 or reward_multiple <= 0 or maximum_holding_bars <= 0:
        return _episode_result(base, outcome_status="BLOCKED_INVALID_ENVELOPE")
    entry_time = future.index[0]
    entry_price = float(future.iloc[0]["open"])
    stop = entry_price - initial_risk
    target = entry_price + reward_multiple * initial_risk
    if entry_price <= 0 or stop <= 0 or target <= entry_price:
        return _episode_result(base, outcome_status="BLOCKED_INVALID_LEVELS")

    path = future.head(maximum_holding_bars)
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    outcome_status = "OPEN"
    exit_reason: str | None = None
    evaluated: list[pd.Series] = []
    evaluated_timestamps: list[pd.Timestamp] = []
    for bar_number, (timestamp, bar) in enumerate(path.iterrows(), start=1):
        evaluated.append(bar)
        evaluated_timestamps.append(timestamp)
        bar_open = float(bar["open"])
        bar_high = float(bar["high"])
        bar_low = float(bar["low"])
        if bar_open <= stop:
            exit_time, exit_price = timestamp, bar_open
            outcome_status, exit_reason = "CLOSED_STOP", "GAP_THROUGH_STOP"
            break
        if bar_open >= target:
            exit_time, exit_price = timestamp, bar_open
            outcome_status, exit_reason = "CLOSED_TARGET", "GAP_THROUGH_TARGET"
            break
        stop_hit = bar_low <= stop
        target_hit = bar_high >= target
        if stop_hit:
            exit_time, exit_price = timestamp, stop
            outcome_status = "CLOSED_STOP"
            exit_reason = (
                "STOP_FIRST_SAME_BAR_AMBIGUITY"
                if target_hit
                else "STOP_HIT"
            )
            break
        if target_hit:
            exit_time, exit_price = timestamp, target
            outcome_status, exit_reason = "CLOSED_TARGET", "TARGET_HIT"
            break
        if bar_number == maximum_holding_bars:
            exit_time, exit_price = timestamp, float(bar["close"])
            outcome_status, exit_reason = "CLOSED_TIME", "MAX_HOLDING_BARS"
            break

    mark_time = path.index[-1]
    mark_price = float(path.iloc[-1]["close"])
    realized = outcome_status.startswith("CLOSED_")
    if realized:
        if exit_price is None:
            raise RuntimeError("closed forward episode requires an exit price")
        measured_price = exit_price
    else:
        measured_price = mark_price
    gross_return = float(measured_price / entry_price - 1.0)
    cost_rate = cost_bps_per_side / 10_000.0
    net_return = gross_return - cost_rate * (2.0 if realized else 1.0)
    lows = [float(bar["low"]) for bar in evaluated]
    highs = [float(bar["high"]) for bar in evaluated]
    mae_index = min(range(len(lows)), key=lows.__getitem__)
    mfe_index = max(range(len(highs)), key=highs.__getitem__)
    time_to_mae = (
        evaluated_timestamps[mae_index] - entry_time
    ).total_seconds()
    time_to_mfe = (
        evaluated_timestamps[mfe_index] - entry_time
    ).total_seconds()
    time_to_stop = (
        (exit_time - entry_time).total_seconds()
        if exit_time is not None and outcome_status == "CLOSED_STOP"
        else None
    )
    holding_end = exit_time if exit_time is not None else mark_time
    holding_duration = (holding_end - entry_time).total_seconds()
    gross_r = (measured_price - entry_price) / initial_risk
    net_r = net_return * entry_price / initial_risk
    maximum_adverse_r = (min(lows) - entry_price) / initial_risk
    maximum_favorable_r = (max(highs) - entry_price) / initial_risk
    first_barrier_hit = (
        "STOP_AND_TARGET_SAME_BAR_STOP_FIRST"
        if exit_reason == "STOP_FIRST_SAME_BAR_AMBIGUITY"
        else "STOP"
        if outcome_status == "CLOSED_STOP"
        else "TARGET"
        if outcome_status == "CLOSED_TARGET"
        else "TIME_EXIT"
        if outcome_status == "CLOSED_TIME"
        else "NONE"
    )
    result = {
        **base,
        "outcome_status": outcome_status,
        "exit_reason": exit_reason,
        "entry_timestamp": entry_time.isoformat(),
        "entry_price": round(entry_price, 8),
        "would_fill": True,
        "fill_timestamp": entry_time.isoformat(),
        "fill_price": round(entry_price, 8),
        "fill_fraction": 1.0,
        "effective_stop_loss": round(stop, 8),
        "effective_take_profit": round(target, 8),
        "exit_timestamp": exit_time.isoformat() if exit_time is not None else None,
        "exit_price": round(exit_price, 8) if exit_price is not None else None,
        "mark_timestamp": mark_time.isoformat(),
        "mark_price": round(mark_price, 8),
        "bars_held": len(evaluated),
        "holding_duration_seconds": holding_duration,
        "first_barrier_hit": first_barrier_hit,
        "gross_return": round(gross_return, 10),
        "net_return": round(net_return, 10),
        "gross_R": round(gross_r, 10),
        "net_R": round(net_r, 10),
        "return_is_realized": realized,
        "maximum_adverse_excursion": round(min(lows) / entry_price - 1.0, 10),
        "maximum_favorable_excursion": round(max(highs) / entry_price - 1.0, 10),
        "maximum_adverse_excursion_R": round(maximum_adverse_r, 10),
        "maximum_favorable_excursion_R": round(maximum_favorable_r, 10),
        "time_to_mae_seconds": time_to_mae,
        "time_to_mfe_seconds": time_to_mfe,
        "time_to_stop_seconds": time_to_stop,
        "exit_quality": (
            "CONSERVATIVE_SAME_BAR_PATH"
            if exit_reason == "STOP_FIRST_SAME_BAR_AMBIGUITY"
            else "BAR_BASED_CAUSAL"
            if realized
            else "OPEN_UNASSESSED"
        ),
        "cost_bps_per_side": cost_bps_per_side,
    }
    result["episode_hash"] = stable_hash(result)
    return result


def _prepare_forward_frame(
    frame: pd.DataFrame | None,
    *,
    evidence_end: pd.Timestamp,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    required = ["open", "high", "low", "close"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    prepared = frame[required].copy()
    index = pd.to_datetime(prepared.index, errors="coerce", utc=True)
    prepared.index = index
    prepared = prepared.loc[~prepared.index.isna()]
    prepared = prepared.loc[prepared.index <= evidence_end]
    prepared = prepared[~prepared.index.duplicated(keep="last")].sort_index()
    for column in required:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.dropna(subset=required)
    valid = (
        prepared["open"].gt(0)
        & prepared["high"].ge(prepared[["open", "close", "low"]].max(axis=1))
        & prepared["low"].le(prepared[["open", "close", "high"]].min(axis=1))
    )
    return prepared.loc[valid]


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _timestamp_iso_or_none(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None else None


def _episode_result(
    base: Mapping[str, Any],
    *,
    outcome_status: str,
) -> dict[str, Any]:
    result = {
        **base,
        "outcome_status": outcome_status,
        "would_fill": (
            False if outcome_status.startswith("BLOCKED_") else None
        ),
        "exit_reason": None,
        "entry_timestamp": None,
        "entry_price": None,
        "effective_stop_loss": None,
        "effective_take_profit": None,
        "exit_timestamp": None,
        "exit_price": None,
        "mark_timestamp": None,
        "mark_price": None,
        "bars_held": 0,
        "holding_duration_seconds": None,
        "first_barrier_hit": None,
        "gross_return": None,
        "net_return": None,
        "gross_R": None,
        "net_R": None,
        "return_is_realized": False,
        "maximum_adverse_excursion": None,
        "maximum_favorable_excursion": None,
        "maximum_adverse_excursion_R": None,
        "maximum_favorable_excursion_R": None,
        "time_to_mae_seconds": None,
        "time_to_mfe_seconds": None,
        "time_to_stop_seconds": None,
        "exit_quality": "NOT_APPLICABLE",
    }
    result["episode_hash"] = stable_hash(result)
    return result


def _aggregate_forward_episodes(
    episodes: list[dict[str, Any]],
    *,
    strategy_id: str,
) -> dict[str, Any]:
    closed = [
        episode
        for episode in episodes
        if episode["outcome_status"].startswith("CLOSED_")
        and episode.get("net_return") is not None
    ]
    returns = [float(episode["net_return"]) for episode in closed]
    positive = sum(value for value in returns if value > 0)
    negative = sum(value for value in returns if value < 0)
    if not closed:
        profit_factor = None
        pf_reason = "NO_CLOSED_EPISODES"
    elif positive <= 0 and negative < 0:
        profit_factor = 0.0
        pf_reason = "NO_POSITIVE_EPISODES"
    elif negative == 0 and positive > 0:
        profit_factor = None
        pf_reason = "PERFECT_NO_LOSSES"
    elif negative == 0:
        profit_factor = None
        pf_reason = "ZERO_DENOMINATOR"
    else:
        profit_factor = positive / abs(negative)
        pf_reason = "DEFINED"
    closed_count = len(closed)
    sample_status = (
        "EVALUABLE"
        if closed_count >= 30
        else "LOW_CONFIDENCE"
        if closed_count >= 10
        else "INSUFFICIENT_SAMPLE"
    )
    return {
        "strategy_id": strategy_id,
        "episode_count": len(episodes),
        "closed_episode_count": closed_count,
        "open_episode_count": sum(
            episode["outcome_status"] == "OPEN" for episode in episodes
        ),
        "positive_episode_count": sum(value > 0 for value in returns),
        "negative_episode_count": sum(value < 0 for value in returns),
        "zero_pnl_episode_count": sum(value == 0 for value in returns),
        "net_profit_factor": (
            round(profit_factor, 10) if profit_factor is not None else None
        ),
        "profit_factor_reason": pf_reason,
        "net_expectancy": (
            round(sum(returns) / closed_count, 10) if closed else None
        ),
        "net_return_sum": round(sum(returns), 10) if closed else None,
        "win_rate": (
            round(sum(value > 0 for value in returns) / closed_count, 10)
            if closed
            else None
        ),
        "sample_status": sample_status,
        "currently_attested_episode_count": sum(
            bool(episode.get("currently_attested_at_signal"))
            for episode in episodes
        ),
    }


def _performance_summary(performance: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = performance.get("aggregate", {})
    counts = performance.get("counts", {})
    return {
        "status": performance.get("status"),
        "episode_count": counts.get("episode_count", 0),
        "closed_episode_count": counts.get("closed_episode_count", 0),
        "net_profit_factor": aggregate.get("net_profit_factor"),
        "net_expectancy": aggregate.get("net_expectancy"),
        "sample_status": aggregate.get("sample_status"),
        "automatic_orders": 0,
        "EXECUTION_AUTHORITY": "NONE",
    }


def _forward_freshness(
    closed: pd.Timestamp,
    observed_at: datetime,
    timeframe: str,
) -> str:
    now = pd.Timestamp(observed_at)
    age_hours = (now - closed).total_seconds() / 3600.0
    limits = {"1h": 96.0, "4h": 120.0, "1d": 120.0, "1w": 240.0}
    return (
        "FRESH_CLOSED_BAR"
        if 0 <= age_hours <= limits[timeframe]
        else "STALE_OR_FUTURE_BAR_BLOCKED"
    )


def _as_utc(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _select_validation_profile(
    frames: Mapping[str, pd.DataFrame],
    signals: Mapping[str, Mapping[str, pd.DataFrame]],
    fold: Mapping[str, Any],
) -> tuple[str, bool, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for profile in PROFILES:
        run = _run_portfolio(
            frames,
            signals[profile],
            start=pd.Timestamp(fold["validation_start"]),
            end=pd.Timestamp(fold["validation_end"]),
            cost_bps=10.0,
        )
        rows.append(
            {
                "profile": profile,
                "validation_period_profit_factor": _finite(
                    run["metrics"].get("period_profit_factor"),
                    -1.0,
                ),
                "validation_CAGR": _finite(
                    run["metrics"].get("CAGR"),
                    -1.0,
                ),
                "validation_fill_count": len(run["fills"]),
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["validation_period_profit_factor"]),
            -float(row["validation_CAGR"]),
            str(row["profile"]),
        ),
    )
    best, neighbor = ranked[:2]
    best_pf = float(best["validation_period_profit_factor"])
    neighbor_pf = float(neighbor["validation_period_profit_factor"])
    plateau = (
        best_pf > 1.0
        and neighbor_pf > 1.0
        and abs(best_pf - neighbor_pf) / max(abs(best_pf), 1e-9)
        <= 0.20
    )
    for row in rows:
        row["selected"] = row["profile"] == best["profile"]
        row["parameter_plateau"] = plateau
    return str(best["profile"]), plateau, rows


def _deterministic_mode(values: pd.Series) -> str:
    counts = Counter(str(value) for value in values.dropna())
    return (
        sorted(counts, key=lambda value: (-counts[value], value))[0]
        if counts
        else "balanced"
    )


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "BLOCKED",
        "reason": reason,
        **AUTHORITY,
    }


def _output(project_root: Path) -> Path:
    return project_root / "output" / "research" / "phase11_14"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
