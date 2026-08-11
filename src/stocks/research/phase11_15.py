from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.microstructure.orderflow import bar_flow_proxy
from stocks.research.phase11_6 import nested_walk_forward_folds
from stocks.research.phase11_8 import _run_portfolio
from stocks.research.phase11_9 import _aggregate, _load_frames
from stocks.research.phase11_10 import (
    ARCHITECTURES,
    _architecture_signals,
    _intermediate_timeframes,
)


SCHEMA = "phase11_15_bar_flow_proxy_overlay_research_v1"
OUTPUT = Path("output/research/phase11_15")
OVERLAYS = (
    "BASELINE",
    "FLOW_ADVERSE_BLOCK",
    "FLOW_CONFIRM",
    "FLOW_STRONG_CONFIRM",
    "FLOW_SCORE_RANK",
)
SUPPORTED_LOWER_TIMEFRAMES = frozenset({"1h", "2h", "4h"})
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


def phase11_15_schema(project_root: Path) -> dict[str, Any]:
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "purpose": (
            "Nested OOS test of low-confidence OHLCV bar-flow overlays "
            "on frozen multi-timeframe swing architectures."
        ),
        "overlay_hypotheses": list(OVERLAYS),
        "selection": (
            "base profile and flow overlay selected in validation; "
            "outer test remains untouched"
        ),
        "execution": "NEXT_LOWER_BAR_OPEN",
        "cost_stress_bps": [10.0, 50.0],
        "portfolio_invariants": [
            "whole shares",
            "global maximum exposure 100 percent",
            "global security netting",
            "causal score-descending security-id tie break",
            "EUR prices, FX and explicit costs",
            "portfolio period profit factor is primary",
        ],
        "orderflow_data_class": "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW",
        "orderflow_confidence": 0.30,
        "gex_backtest_status": "BLOCKED_NO_POINT_IN_TIME_OPTION_HISTORY",
        "gex_forward_context_allowed": True,
        "post_selection_research": True,
        "can_grant_financial_authority": False,
        **AUTHORITY,
    }
    _write_json(project_root / OUTPUT / "schema.json", report)
    return report


def run_phase11_15(
    project_root: Path,
    *,
    max_architectures: int = 20,
) -> dict[str, Any]:
    schema = phase11_15_schema(project_root)
    candidates = _candidate_architectures(project_root)[: max(1, max_architectures)]
    if not candidates:
        return _publish_status(
            project_root,
            {
                "schema": SCHEMA,
                "status": "BLOCKED_NO_INTRADAY_ARCHITECTURES",
                "architecture_count": 0,
                **AUTHORITY,
            },
        )
    all_frames = _load_frames(project_root)
    selections_path = (
        project_root
        / "output/research/phase11_10/run-checkpoint-selections.parquet"
    )
    if not selections_path.is_file():
        return _publish_status(
            project_root,
            {
                "schema": SCHEMA,
                "status": "BLOCKED_PHASE11_10_SELECTIONS_UNAVAILABLE",
                "architecture_count": len(candidates),
                **AUTHORITY,
            },
        )
    prior_selections = pd.read_parquet(selections_path)
    results: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    flow_cache: dict[str, dict[str, pd.Series]] = {}
    for architecture in candidates:
        specification = ARCHITECTURES[architecture]
        lower_timeframe = specification["lower"]
        lower_frames = all_frames.get(lower_timeframe, {})
        if len(lower_frames) < 5:
            blocked.append(
                {
                    "architecture": architecture,
                    "reason": "INSUFFICIENT_REAL_LOWER_TIMEFRAME_ASSETS",
                    "instrument_count": len(lower_frames),
                }
            )
            continue
        if lower_timeframe not in flow_cache:
            flow_cache[lower_timeframe] = _flow_scores(
                lower_frames,
                lower_timeframe=lower_timeframe,
            )
        flow_scores = flow_cache[lower_timeframe]
        profile_signals = {
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
            for profile in ("responsive", "balanced", "conservative")
        }
        overlays = {
            profile: {
                overlay: _apply_overlay(
                    signals,
                    flow_scores,
                    overlay=overlay,
                )
                for overlay in OVERLAYS
            }
            for profile, signals in profile_signals.items()
        }
        start = min(frame.index.min() for frame in lower_frames.values())
        end = min(frame.index.max() for frame in lower_frames.values())
        folds = {
            str(row["fold_id"]): row
            for row in nested_walk_forward_folds(
                start, end, lower_timeframe
            ).to_dict("records")
        }
        architecture_selections = prior_selections.loc[
            prior_selections["architecture"].eq(architecture)
        ]
        for selection in architecture_selections.to_dict("records"):
            fold_id = str(selection["fold_id"])
            fold = folds.get(fold_id)
            if fold is None:
                continue
            profile = str(selection["selected_profile"])
            validation: list[tuple[float, float, str]] = []
            for overlay in OVERLAYS:
                metrics = _run_portfolio(
                    lower_frames,
                    overlays[profile][overlay],
                    start=pd.Timestamp(fold["validation_start"]),
                    end=pd.Timestamp(fold["validation_end"]),
                    cost_bps=10.0,
                )["metrics"]
                validation.append(
                    (
                        _metric(metrics, "period_profit_factor", -1.0),
                        _metric(metrics, "CAGR", -1.0),
                        overlay,
                    )
                )
            validation.sort(reverse=True)
            selected_overlay = validation[0][2]
            baseline = _run_portfolio(
                lower_frames,
                overlays[profile]["BASELINE"],
                start=pd.Timestamp(fold["outer_test_start"]),
                end=pd.Timestamp(fold["outer_test_end"]),
                cost_bps=10.0,
            )
            baseline_metrics = baseline["metrics"]
            decisions.append(
                {
                    "architecture": architecture,
                    "fold_id": fold_id,
                    "profile": profile,
                    "selected_overlay": selected_overlay,
                    "validation_portfolio_pf": validation[0][0],
                    "validation_CAGR": validation[0][1],
                    "neighbor_overlay": validation[1][2],
                    "neighbor_portfolio_pf": validation[1][0],
                    "overlay_plateau": _overlay_plateau(validation),
                }
            )
            for cost_bps in (10.0, 50.0):
                run = _run_portfolio(
                    lower_frames,
                    overlays[profile][selected_overlay],
                    start=pd.Timestamp(fold["outer_test_start"]),
                    end=pd.Timestamp(fold["outer_test_end"]),
                    cost_bps=cost_bps,
                )
                metrics = run["metrics"]
                fills = run["fills"]
                notional = (
                    float(fills["shares"].mul(fills["price_eur"]).sum())
                    if not fills.empty
                    else 0.0
                )
                baseline_pf = _metric(
                    baseline_metrics, "period_profit_factor", math.nan
                )
                baseline_cagr = _metric(
                    baseline_metrics, "CAGR", math.nan
                )
                results.append(
                    {
                        "architecture": architecture,
                        "fold_id": fold_id,
                        "higher_timeframe": specification["higher"],
                        "middle_timeframe": specification.get("middle"),
                        "lower_timeframe": lower_timeframe,
                        "entry_strategy": specification["entry"],
                        "profile": profile,
                        "selected_overlay": selected_overlay,
                        "cost_bps": cost_bps,
                        "fill_count": len(fills),
                        "turnover_initial_capital": notional / 10_000.0,
                        "baseline_10bps_CAGR": baseline_cagr,
                        "baseline_10bps_portfolio_pf": baseline_pf,
                        "incremental_CAGR_vs_baseline": (
                            _metric(metrics, "CAGR", math.nan)
                            - baseline_cagr
                            if cost_bps == 10.0
                            else math.nan
                        ),
                        "incremental_pf_vs_baseline": (
                            _metric(
                                metrics,
                                "period_profit_factor",
                                math.nan,
                            )
                            - baseline_pf
                            if cost_bps == 10.0
                            else math.nan
                        ),
                        **metrics,
                    }
                )
    result_frame = pd.DataFrame(results)
    decision_frame = pd.DataFrame(decisions)
    summary = _summarize(result_frame, decision_frame)
    output = project_root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    result_frame.to_parquet(output / "nested-results.parquet", index=False)
    decision_frame.to_parquet(
        output / "overlay-selections.parquet", index=False
    )
    summary.to_csv(output / "architecture-summary.csv", index=False)
    _write_json(output / "blocked.json", blocked)
    improved = (
        summary.loc[summary["incremental_flow_evidence_status"].eq("POSITIVE")]
        if not summary.empty
        else summary
    )
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "architecture_count": len(candidates),
        "evaluated_architecture_count": int(
            result_frame["architecture"].nunique()
        )
        if not result_frame.empty
        else 0,
        "fold_count": int(
            result_frame.loc[result_frame["cost_bps"].eq(10.0), "fold_id"].nunique()
        )
        if not result_frame.empty
        else 0,
        "outer_test_run_count": len(result_frame),
        "global_overlay_hypothesis_count": len(candidates) * len(OVERLAYS),
        "improved_architecture_count": len(improved),
        "improved_architectures": (
            improved[
                [
                    "architecture",
                    "lower_timeframe",
                    "entry_strategy",
                    "median_oos_portfolio_pf",
                    "cost_50bps_median_pf",
                    "median_incremental_CAGR",
                    "median_incremental_pf",
                    "flow_improvement_ratio",
                ]
            ].to_dict("records")
            if not improved.empty
            else []
        ),
        "selected_overlay_counts": dict(
            Counter(decision_frame.get("selected_overlay", []))
        ),
        "orderflow_data_class": "BAR_FLOW_PROXY_NOT_OBSERVED_ORDERFLOW",
        "orderflow_confidence": 0.30,
        "gex_backtest_status": "BLOCKED_NO_POINT_IN_TIME_OPTION_HISTORY",
        "current_gex_forward_context_status": "ALLOWED_CONTEXT_ONLY",
        "post_selection_research": True,
        "financial_interpretation": (
            "EXPLORATORY_INCREMENTAL_EVIDENCE_ONLY"
        ),
        "source_phase": "PHASE11_10_FROZEN_ARCHITECTURES",
        "source_schema": schema["schema"],
        "artifacts": [
            "schema.json",
            "nested-results.parquet",
            "overlay-selections.parquet",
            "architecture-summary.csv",
            "blocked.json",
            "status.json",
            "manifest.json",
        ],
        **AUTHORITY,
    }
    report["content_hash"] = _hash(report)
    return _publish_status(project_root, report)


def phase11_15_status(project_root: Path) -> dict[str, Any]:
    path = project_root / OUTPUT / "status.json"
    if not path.is_file():
        return {
            "schema": SCHEMA,
            "status": "NOT_RUN",
            **AUTHORITY,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_architectures(project_root: Path) -> list[str]:
    path = project_root / "output/research/phase11_10/top20-strategies.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[str] = []
    for row in payload.get("strategies", []):
        architecture = str(row.get("architecture") or "")
        specification = ARCHITECTURES.get(architecture)
        if (
            specification is None
            or specification["lower"] not in SUPPORTED_LOWER_TIMEFRAMES
            or architecture in candidates
        ):
            continue
        candidates.append(architecture)
    return candidates


def _flow_scores(
    frames: Mapping[str, pd.DataFrame],
    *,
    lower_timeframe: str,
) -> dict[str, pd.Series]:
    output: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        lower = _proxy_series(frame, symbol, lower_timeframe)
        if lower.empty:
            continue
        components: list[tuple[float, pd.Series]] = [(1.0, lower)]
        if lower_timeframe == "1h":
            components = [(0.50, lower)]
            for timeframe, weight in (("2h", 0.30), ("4h", 0.20)):
                aggregate = _aggregate({symbol: frame}, timeframe).get(symbol)
                if aggregate is None or aggregate.empty:
                    continue
                higher = _proxy_series(aggregate, symbol, timeframe).shift(1)
                components.append(
                    (
                        weight,
                        higher.reindex(frame.index, method="ffill"),
                    )
                )
        elif lower_timeframe == "2h":
            components = [(0.60, lower)]
            aggregate = _aggregate({symbol: frame}, "4h").get(symbol)
            if aggregate is not None and not aggregate.empty:
                higher = _proxy_series(aggregate, symbol, "4h").shift(1)
                components.append(
                    (0.40, higher.reindex(frame.index, method="ffill"))
                )
        numerator = pd.Series(0.0, index=frame.index)
        denominator = pd.Series(0.0, index=frame.index)
        for weight, component in components:
            available = component.reindex(frame.index).notna()
            numerator = numerator.add(
                component.reindex(frame.index).fillna(0.0) * weight
            )
            denominator = denominator.add(available.astype(float) * weight)
        output[symbol] = numerator.div(
            denominator.replace(0, np.nan)
        ).clip(-1, 1)
    return output


def _proxy_series(
    frame: pd.DataFrame,
    symbol: str,
    interval: str,
) -> pd.Series:
    proxy = bar_flow_proxy(
        frame,
        symbol=symbol,
        interval=interval,
        source=str(frame.attrs.get("provider") or "LOCAL_REAL_OHLCV"),
    )
    if proxy.empty:
        return pd.Series(dtype=float)
    timestamps = pd.to_datetime(proxy["timestamp_utc"], utc=True)
    if frame.index.tz is None:
        timestamps = timestamps.dt.tz_localize(None)
    return pd.Series(
        pd.to_numeric(proxy["bar_flow_score"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(timestamps),
    ).sort_index()


def _apply_overlay(
    signals: Mapping[str, pd.DataFrame],
    flow_scores: Mapping[str, pd.Series],
    *,
    overlay: str,
) -> dict[str, pd.DataFrame]:
    if overlay not in OVERLAYS:
        raise ValueError(f"UNKNOWN_FLOW_OVERLAY:{overlay}")
    output: dict[str, pd.DataFrame] = {}
    for symbol, signal in signals.items():
        work = signal.copy()
        flow = flow_scores.get(symbol, pd.Series(dtype=float)).reindex(
            work.index
        )
        if overlay == "BASELINE":
            work["flow_proxy_score"] = flow
            output[symbol] = work
            continue
        if overlay == "FLOW_SCORE_RANK":
            work["score"] = work["score"].add(
                flow.fillna(0.0).mul(0.25), fill_value=0.0
            )
            work["flow_proxy_score"] = flow
            output[symbol] = work
            continue
        threshold = {
            "FLOW_ADVERSE_BLOCK": -0.35,
            "FLOW_CONFIRM": 0.0,
            "FLOW_STRONG_CONFIRM": 0.35,
        }[overlay]
        work["signal"] = _entry_gated_state(
            work["signal"],
            flow.ge(threshold).fillna(False),
        )
        work["flow_proxy_score"] = flow
        output[symbol] = work
    return output


def _entry_gated_state(
    base_state: pd.Series,
    entry_allowed: pd.Series,
) -> pd.Series:
    active = False
    values: list[bool] = []
    for base, allowed in zip(
        base_state.fillna(False),
        entry_allowed.reindex(base_state.index).fillna(False),
    ):
        if not bool(base):
            active = False
        elif not active and bool(allowed):
            active = True
        values.append(active)
    return pd.Series(values, index=base_state.index, dtype=bool)


def _overlay_plateau(validation: list[tuple[float, float, str]]) -> bool:
    if len(validation) < 2 or validation[0][0] <= 1 or validation[1][0] <= 1:
        return False
    return abs(validation[0][0] - validation[1][0]) / max(
        abs(validation[0][0]), 1e-9
    ) <= 0.20


def _summarize(
    results: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    normal = results.loc[results["cost_bps"].eq(10.0)]
    for architecture, group in normal.groupby("architecture"):
        stress = results.loc[
            results["architecture"].eq(architecture)
            & results["cost_bps"].eq(50.0)
        ]
        selected = decisions.loc[
            decisions["architecture"].eq(architecture)
        ]
        pfs = group["period_profit_factor"].replace(
            [np.inf, -np.inf], np.nan
        )
        stress_pfs = stress["period_profit_factor"].replace(
            [np.inf, -np.inf], np.nan
        )
        median_incremental_cagr = float(
            group["incremental_CAGR_vs_baseline"].median()
        )
        median_incremental_pf = float(
            group["incremental_pf_vs_baseline"].median()
        )
        improvement_ratio = float(
            group["incremental_CAGR_vs_baseline"].gt(0).mean()
        )
        status = (
            "POSITIVE"
            if median_incremental_cagr > 0
            and median_incremental_pf > 0
            and improvement_ratio >= 0.55
            and float(pfs.median()) > 1.0
            and float(stress_pfs.median()) > 1.0
            else "NO_INCREMENTAL_EVIDENCE"
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
                "median_incremental_CAGR": median_incremental_cagr,
                "median_incremental_pf": median_incremental_pf,
                "flow_improvement_ratio": improvement_ratio,
                "overlay_plateau_ratio": float(
                    selected["overlay_plateau"].mean()
                ),
                "median_fill_count": float(group["fill_count"].median()),
                "incremental_flow_evidence_status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "incremental_flow_evidence_status",
            "median_incremental_CAGR",
            "median_oos_portfolio_pf",
        ],
        ascending=[True, False, False],
    )


def _metric(
    metrics: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    try:
        value = float(str(metrics.get(key)))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _publish_status(
    project_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    output = project_root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "status.json", report)
    _write_json(output / "manifest.json", report)
    return report


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


__all__ = [
    "phase11_15_schema",
    "phase11_15_status",
    "run_phase11_15",
]
