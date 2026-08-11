from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.research.phase11_6 import nested_walk_forward_folds
from stocks.research.phase11_8 import (
    INITIAL_CAPITAL_EUR,
    MAX_POSITIONS,
    _cost_breakdown,
    _metrics,
    _run_portfolio,
)
from stocks.research.phase11_9 import _load_frames, _signals


SCHEMA = "phase11_13_fast_track_pit_qualification_v1"
PROFILES = ("responsive", "balanced", "conservative")
COSTS_BPS = (10.0, 20.0, 50.0, 100.0)
STOCKS = {
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
}
EQUITY_ETFS = {"EEM", "EFA", "IWM", "QQQ", "SPY"}
MULTI_ASSET_ETFS = {
    "EEM",
    "EFA",
    "GLD",
    "IWM",
    "QQQ",
    "SLV",
    "SPY",
}
PROHIBITED_PRODUCT_PROXIES = {"DBC", "TLT"}
FROZEN_STRATEGIES: dict[str, dict[str, Any]] = {
    "FOUR_HOUR_STOCK_TREND_PULLBACK": {
        "timeframe": "4h",
        "strategy": "ema_pullback",
        "symbols": STOCKS,
    },
    "DAILY_DONCHIAN_BREAKOUT": {
        "timeframe": "1d",
        "strategy": "donchian_breakout",
        "symbols": STOCKS | EQUITY_ETFS,
    },
    "DAILY_UPTREND_MEAN_REVERSION": {
        "timeframe": "1d",
        "strategy": "rsi2_adx_pullback",
        "symbols": STOCKS,
    },
    "WEEKLY_CROSS_SECTIONAL_MOMENTUM": {
        "timeframe": "1w",
        "strategy": "risk_adjusted_momentum",
        "symbols": STOCKS | EQUITY_ETFS,
    },
    "MULTI_ASSET_INVERSE_VOL_TREND": {
        "timeframe": "1d",
        "strategy": "etf_commodity_trend",
        "symbols": MULTI_ASSET_ETFS,
        "allocation_note": "TRUE_CAUSAL_INVERSE_VOLATILITY_WEIGHTS",
        "implementation_status": "IMPLEMENTED",
    },
}
AUTHORITY = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "BLOCKED",
    "STRATEGY_AUTHORITY": "NONE",
    "EXECUTION_AUTHORITY": "NONE",
    "PAPER_STRATEGY_AUTHORITY": "NONE",
    "LIVE_STRATEGY_AUTHORITY": "NONE",
    "BROKER_CALLS": 0,
    "ORDER_CALLS": 0,
}


def phase11_13_schema(project_root: Path) -> dict[str, Any]:
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "strategies": _public_strategies(),
        "validation": {
            "outer_fold_count": 6,
            "profiles": list(PROFILES),
            "profile_selection": "VALIDATION_ONLY",
            "costs_bps_per_side": list(COSTS_BPS),
            "execution": "NEXT_BAR_OPEN",
            "whole_shares": True,
            "maximum_gross_exposure": 1.0,
            "base_currency": "EUR",
            "benchmark": "EXPOSURE_MATCHED_SPY",
            "inverse_volatility_lookback": 63,
            "inverse_volatility_rebalance_bars": 21,
            "inverse_volatility_target_annual_volatility": 0.10,
            "weekly_momentum_target_annual_volatility": 0.10,
        },
        "product_structure_policy": {
            "prohibited_proxies": sorted(PROHIBITED_PRODUCT_PROXIES),
            "multi_asset_universe": sorted(MULTI_ASSET_ETFS),
            "known_interest_or_derivative_proxy_count": len(
                PROHIBITED_PRODUCT_PROXIES & MULTI_ASSET_ETFS
            ),
            "etf_holdings_shariah_history": "UNVERIFIED_FAIL_CLOSED",
        },
        "research_pass": {
            "combined_oos_return": ">0",
            "positive_folds": ">=3/6",
            "combined_period_profit_factor": ">=1.02",
            "maximum_drawdown": ">=-0.35",
            "cost_20bps_combined_return": ">0",
            "fills": ">=20",
        },
        "robust_pass": {
            "combined_oos_return": ">0",
            "positive_folds": ">=4/6",
            "combined_period_profit_factor": ">=1.05",
            "maximum_drawdown": ">=-0.25",
            "exposure_matched_alpha": ">0",
            "cost_50bps_combined_return": ">0",
            "fills": ">=30",
        },
        "deployment_requires": [
            "POINT_IN_TIME_UNIVERSE_COMPLETE",
            "HISTORICAL_SHARIAH_COMPLETE",
            "INDEPENDENT_FORWARD_SESSION_COMPLETE",
            "EXPLICIT_OPERATOR_APPROVAL",
        ],
        **AUTHORITY,
    }
    _write_json(_output(project_root) / "schema.json", report)
    return report


def run_phase11_13(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    output.mkdir(parents=True, exist_ok=True)
    schema = phase11_13_schema(project_root)
    schema_hash = _hash(schema)
    boundary_path = output / "qualification-boundary.json"
    if boundary_path.exists():
        boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
        if boundary.get("schema_hash") == schema_hash:
            frozen = phase11_13_status(project_root)
            return {
                **frozen,
                "run_status": "QUALIFICATION_FROZEN_NO_REEVALUATION",
                "qualification_boundary": boundary,
            }
    all_frames = _load_frames(project_root)
    fold_rows: list[dict[str, Any]] = []
    return_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for strategy_id, specification in FROZEN_STRATEGIES.items():
        timeframe = str(specification["timeframe"])
        available = all_frames.get(timeframe, {})
        frames = {
            symbol: frame
            for symbol, frame in available.items()
            if symbol in specification["symbols"]
        }
        if len(frames) < 5 or "SPY" not in available:
            blocked.append(
                {
                    "strategy_id": strategy_id,
                    "reason": "INSUFFICIENT_REAL_DATA_OR_BENCHMARK",
                    "instrument_count": len(frames),
                }
            )
            continue
        start = min(frame.index.min() for frame in frames.values())
        end = min(frame.index.max() for frame in frames.values())
        folds = nested_walk_forward_folds(start, end, timeframe).tail(6)
        coverage_rows.append(
            {
                "strategy_id": strategy_id,
                "timeframe": timeframe,
                "instrument_count": len(frames),
                "start": start,
                "end": end,
                "fold_count": len(folds),
                "price_basis": _price_basis(frames),
            }
        )
        if len(folds) < 6:
            blocked.append(
                {
                    "strategy_id": strategy_id,
                    "reason": "FEWER_THAN_SIX_OUTER_FOLDS",
                    "fold_count": len(folds),
                }
            )
            continue
        signal_cache = {
            profile: _signals(
                frames,
                str(specification["strategy"]),
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
            selected_profile, plateau = _select_profile(
                strategy_id,
                frames,
                signal_cache,
                fold,
            )
            strategy_normal: dict[str, Any] | None = None
            first_row = len(fold_rows)
            for cost_bps in COSTS_BPS:
                run = _run_strategy_portfolio(
                    strategy_id,
                    frames,
                    signal_cache[selected_profile],
                    start=pd.Timestamp(fold["outer_test_start"]),
                    end=pd.Timestamp(fold["outer_test_end"]),
                    cost_bps=cost_bps,
                )
                if cost_bps == 10.0:
                    strategy_normal = run
                _append_returns(
                    return_rows,
                    run,
                    strategy_id=strategy_id,
                    fold_id=str(fold["fold_id"]),
                    cost_bps=cost_bps,
                )
                _append_fills(
                    fill_rows,
                    run,
                    strategy_id=strategy_id,
                    fold_id=str(fold["fold_id"]),
                    cost_bps=cost_bps,
                )
                fold_rows.append(
                    {
                        "strategy_id": strategy_id,
                        "fold_id": fold["fold_id"],
                        "timeframe": timeframe,
                        "profile": selected_profile,
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
            if strategy_normal is None:
                continue
            exposure = strategy_normal["ledger"].set_index("date")[
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
            strategy_cagr = _finite(strategy_normal["metrics"].get("CAGR"))
            benchmark_cagr = _finite(benchmark["metrics"].get("CAGR"))
            fold_rows[first_row]["exposure_matched_benchmark_CAGR"] = (
                benchmark_cagr
            )
            fold_rows[first_row]["exposure_matched_alpha_CAGR"] = (
                strategy_cagr - benchmark_cagr
            )

    folds_frame = pd.DataFrame(fold_rows)
    returns_frame = pd.DataFrame(return_rows)
    fills_frame = pd.DataFrame(fill_rows)
    summary = _summarize(folds_frame, returns_frame)
    qualification = _qualification(summary)
    _write_frame(output / "fold-results.parquet", folds_frame)
    _write_frame(output / "daily-oos-returns.parquet", returns_frame)
    _write_frame(output / "fills.parquet", fills_frame)
    _write_frame(output / "strategy-summary.parquet", summary)
    _write_frame(output / "coverage.csv", pd.DataFrame(coverage_rows))
    _write_json(output / "qualification.json", qualification)
    _write_json(output / "blocked.json", blocked)
    data_end_by_strategy = {
        str(row["strategy_id"]): pd.Timestamp(row["end"]).isoformat()
        for row in coverage_rows
    }
    boundary = {
        "schema": "phase11_13_qualification_boundary_v1",
        "status": "FROZEN",
        "qualified_at": datetime.now(UTC).isoformat(),
        "schema_hash": schema_hash,
        "qualification_hash": _hash(qualification),
        "data_end_by_strategy": data_end_by_strategy,
        "global_data_end": max(data_end_by_strategy.values())
        if data_end_by_strategy
        else None,
        "robust_strategy_ids": sorted(
            summary.loc[
                summary["robust_pass"].astype(bool),
                "strategy_id",
            ].astype(str)
        ),
        "future_observation_rule": (
            "CLOSED_BAR_TIMESTAMP_STRICTLY_AFTER_STRATEGY_DATA_END"
        ),
        "automatic_requalification": False,
        **AUTHORITY,
    }
    _write_json(boundary_path, boundary)
    report = {
        "schema": SCHEMA,
        "status": "GO" if not folds_frame.empty else "NO_GO",
        "strategy_count": len(FROZEN_STRATEGIES),
        "evaluated_strategy_count": int(
            folds_frame["strategy_id"].nunique()
        )
        if not folds_frame.empty
        else 0,
        "research_pass_count": qualification["research_pass_count"],
        "robust_pass_count": qualification["robust_pass_count"],
        "deployable_pass_count": 0,
        "qualification": qualification,
        "point_in_time_universe_status": (
            "CURRENT_SYMBOL_RESEARCH_UNIVERSE_NOT_PIT"
        ),
        "historical_shariah_status": "SHARIAH_HISTORY_PARTIAL",
        "independent_forward_session_status": "NOT_YET_OBSERVED",
        "inverse_volatility_strategy_status": "TRUE_CAUSAL_WEIGHTS_IMPLEMENTED",
        "financial_decision": (
            "ROBUST_RESEARCH_SURVIVORS_COMPLIANCE_BLOCKED"
            if qualification["robust_pass_count"]
            else (
                "PROMISING_RESEARCH_CANDIDATE"
                if qualification["research_pass_count"]
                else "NO_NEW_FINANCIAL_CANDIDATE"
            )
        ),
        "schema_hash": schema_hash,
        "qualification_boundary": boundary,
        **AUTHORITY,
    }
    _write_json(output / "status.json", report)
    _write_json(output / "manifest.json", report)
    return report


def phase11_13_status(project_root: Path) -> dict[str, Any]:
    path = _output(project_root) / "status.json"
    if not path.exists():
        return {"schema": SCHEMA, "status": "NOT_RUN", **AUTHORITY}
    return json.loads(path.read_text(encoding="utf-8"))


def phase11_13_observe(project_root: Path) -> dict[str, Any]:
    output = _output(project_root)
    summary_path = output / "strategy-summary.parquet"
    folds_path = output / "fold-results.parquet"
    boundary_path = output / "qualification-boundary.json"
    if (
        not summary_path.exists()
        or not folds_path.exists()
        or not boundary_path.exists()
    ):
        return {
            "schema": "phase11_13_forward_observation_v1",
            "status": "BLOCKED_QUALIFICATION_MISSING",
            **AUTHORITY,
        }
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    summary = pd.read_parquet(summary_path)
    survivors = summary.loc[summary["robust_pass"].astype(bool)]
    if survivors.empty:
        return {
            "schema": "phase11_13_forward_observation_v1",
            "status": "NO_ROBUST_SURVIVORS",
            **AUTHORITY,
        }
    folds = pd.read_parquet(folds_path)
    frames_by_timeframe = _load_frames(project_root)
    now = datetime.now(UTC)
    attestations = _current_attestations(project_root, now)
    observations = []
    session_dates: list[str] = []
    for strategy_id in survivors["strategy_id"].astype(str):
        specification = FROZEN_STRATEGIES[strategy_id]
        timeframe = str(specification["timeframe"])
        available = frames_by_timeframe.get(timeframe, {})
        frames = {
            symbol: frame
            for symbol, frame in available.items()
            if symbol in specification["symbols"]
        }
        if not frames:
            continue
        common_close = min(frame.index.max() for frame in frames.values())
        strategy_boundary = pd.Timestamp(
            boundary["data_end_by_strategy"][strategy_id]
        )
        independent_session = pd.Timestamp(common_close) > strategy_boundary
        session_dates.append(pd.Timestamp(common_close).date().isoformat())
        strategy_folds = folds.loc[
            folds["strategy_id"].eq(strategy_id)
            & folds["cost_bps"].eq(10.0)
        ]
        profile = str(strategy_folds["profile"].mode().iloc[0])
        signals = _signals(
            frames,
            str(specification["strategy"]),
            timeframe,
            profile,
        )
        candidates = []
        for symbol, signal in signals.items():
            closed = signal.loc[signal.index <= common_close]
            if closed.empty:
                continue
            latest = closed.iloc[-1]
            if bool(latest["signal"]):
                candidates.append((float(latest["score"]), symbol))
        desired = {
            symbol
            for _, symbol in sorted(
                candidates,
                key=lambda row: (-row[0], row[1]),
            )[:MAX_POSITIONS]
        }
        if strategy_id == "MULTI_ASSET_INVERSE_VOL_TREND":
            decision_time = pd.Timestamp(common_close) + pd.Timedelta(
                nanoseconds=1
            )
            volatilities = {
                symbol: volatility
                for symbol in desired
                if (
                    volatility := _prior_volatility(
                        frames[symbol],
                        decision_time,
                        63,
                    )
                )
                is not None
            }
            desired &= set(volatilities)
            weights = _targeted_inverse_volatility_weights(
                frames,
                desired,
                decision_time,
                volatilities,
                63,
                0.10,
            )
        else:
            weight = 1.0 / len(desired) if desired else 0.0
            weights = {symbol: weight for symbol in sorted(desired)}
        attested = {
            symbol: weight
            for symbol, weight in weights.items()
            if symbol in attestations
        }
        structure_blocked = bool(
            PROHIBITED_PRODUCT_PROXIES & set(weights)
        )
        observations.append(
            {
                "strategy_id": strategy_id,
                "timeframe": timeframe,
                "profile": profile,
                "closed_bar_timestamp": pd.Timestamp(
                    common_close
                ).isoformat(),
                "qualification_data_end": strategy_boundary.isoformat(),
                "independent_forward_session": independent_session,
                "raw_target_weights": weights,
                "current_attested_target_weights": attested,
                "current_attestation_count": len(attested),
                "product_structure_status": (
                    "BLOCKED_PROHIBITED_PRODUCT_PROXY"
                    if structure_blocked
                    else (
                        "NO_KNOWN_INTEREST_OR_DERIVATIVE_PROXY_BLOCKER"
                        "_ETF_HOLDINGS_ELIGIBILITY_UNVERIFIED"
                    )
                ),
                "observation_status": "OBSERVATION_COMPLETE",
                "execution_route": "BLOCKED",
                "order_intents": [],
                "automatic_orders": 0,
            }
        )
    payload = {
        "schema": "phase11_13_forward_observation_v1",
        "status": "GO",
        "observed_at": now.isoformat(),
        "session_date": max(session_dates) if session_dates else None,
        "robust_survivor_count": len(survivors),
        "observation_count": len(observations),
        "observations": observations,
        "attested_symbols": sorted(attestations),
        "forward_status": "COLLECTING_INDEPENDENT_OBSERVATIONS",
        "execution_route": "BLOCKED",
        "automatic_orders": 0,
        **AUTHORITY,
    }
    prior_observations = _load_forward_observations(output)
    current_audit_rows = [
        {**row, "_observed_at": now.isoformat()} for row in observations
    ]
    forward_audit = _forward_session_audit(
        boundary,
        [*prior_observations, *current_audit_rows],
    )
    payload["independent_forward_audit"] = forward_audit
    payload["independent_forward_session_status"] = forward_audit["status"]
    payload["operational_forward_session_status"] = (
        "NOT_YET_PROVEN_BY_BAR_OBSERVATION"
    )
    content_hash = _hash(payload)
    observation_root = output / "forward-observations"
    _write_json(observation_root / f"{content_hash}.json", payload)
    _write_json(output / "latest-forward-observation.json", payload)
    status = phase11_13_status(project_root)
    status["independent_forward_session_status"] = forward_audit["status"]
    status["independent_forward_session_count"] = forward_audit[
        "independent_session_count"
    ]
    status["independent_forward_audit"] = forward_audit
    status["operational_forward_session_status"] = (
        "NOT_YET_PROVEN_BY_BAR_OBSERVATION"
    )
    _write_json(output / "status.json", status)
    _write_json(output / "manifest.json", status)
    return payload


def _load_forward_observations(output: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = output / "forward-observations"
    if not root.exists():
        return rows
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for row in payload.get("observations", []):
            if isinstance(row, dict):
                rows.append(
                    {
                        **row,
                        "_observed_at": payload.get("observed_at"),
                    }
                )
    return rows


def _forward_session_audit(
    boundary: Mapping[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    robust_ids = {
        str(value) for value in boundary.get("robust_strategy_ids", [])
    }
    ends = {
        str(key): pd.Timestamp(value)
        for key, value in boundary.get("data_end_by_strategy", {}).items()
    }
    sessions: dict[str, set[str]] = {
        strategy_id: set() for strategy_id in robust_ids
    }
    for row in observations:
        strategy_id = str(row.get("strategy_id", ""))
        timestamp = row.get("closed_bar_timestamp")
        if strategy_id not in robust_ids or not timestamp:
            continue
        closed_bar = pd.Timestamp(timestamp)
        observed_at = row.get("_observed_at")
        if observed_at and closed_bar > pd.Timestamp(observed_at).tz_localize(
            None
        ):
            continue
        boundary_end = ends.get(strategy_id)
        if boundary_end is not None and closed_bar > boundary_end:
            sessions[strategy_id].add(closed_bar.isoformat())
    per_strategy = {
        strategy_id: {
            "qualification_data_end": ends[strategy_id].isoformat(),
            "independent_session_count": len(sessions[strategy_id]),
            "independent_sessions": sorted(sessions[strategy_id]),
            "complete": bool(sessions[strategy_id]),
        }
        for strategy_id in sorted(robust_ids)
    }
    completed = sum(
        int(item["complete"]) for item in per_strategy.values()
    )
    if not robust_ids:
        status = "NO_ROBUST_SURVIVORS"
    elif completed == len(robust_ids):
        status = "INDEPENDENT_FORWARD_SESSION_COMPLETE"
    elif completed:
        status = "INDEPENDENT_FORWARD_SESSION_PARTIAL"
    else:
        status = "NOT_YET_OBSERVED"
    return {
        "schema": "phase11_13_independent_forward_session_audit_v1",
        "status": status,
        "qualification_hash": boundary.get("qualification_hash"),
        "robust_strategy_count": len(robust_ids),
        "completed_strategy_count": completed,
        "independent_session_count": sum(map(len, sessions.values())),
        "per_strategy": per_strategy,
        "same_or_prior_bar_counted": False,
        "evidence_scope": (
            "CLOSED_BAR_OBSERVATION_ONLY_NOT_OPERATIONAL_SESSION"
        ),
        "automatic_orders": 0,
        "execution_authority": "NONE",
    }


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    eligible = set()
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
            and screened_at <= decision_time <= expires_at
        ):
            eligible.add(str(row["symbol"]).upper())
    return eligible


def _run_strategy_portfolio(
    strategy_id: str,
    frames: Mapping[str, pd.DataFrame],
    signals: Mapping[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
) -> dict[str, Any]:
    if strategy_id == "MULTI_ASSET_INVERSE_VOL_TREND":
        return _run_inverse_volatility_portfolio(
            frames,
            signals,
            start=start,
            end=end,
            cost_bps=cost_bps,
        )
    if strategy_id == "WEEKLY_CROSS_SECTIONAL_MOMENTUM":
        return _run_volatility_targeted_portfolio(
            frames,
            signals,
            start=start,
            end=end,
            cost_bps=cost_bps,
            periods_per_year=52.0,
        )
    return _run_portfolio(
        frames,
        signals,
        start=start,
        end=end,
        cost_bps=cost_bps,
    )


def _run_volatility_targeted_portfolio(
    frames: Mapping[str, pd.DataFrame],
    signals: Mapping[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
    periods_per_year: float,
    target_annual_volatility: float = 0.10,
    volatility_lookback: int = 26,
) -> dict[str, Any]:
    warmup_start = start - pd.Timedelta(days=550)
    pilot = _run_portfolio(
        frames,
        signals,
        start=warmup_start,
        end=end,
        cost_bps=cost_bps,
    )
    pilot_returns = pilot["ledger"].set_index("date")["daily_return"]
    realized = (
        pilot_returns.rolling(
            volatility_lookback,
            min_periods=max(8, volatility_lookback // 2),
        ).std(ddof=1)
        * math.sqrt(periods_per_year)
    )
    multiplier = (
        target_annual_volatility
        / realized.replace(0, np.nan)
    ).clip(lower=0.0, upper=1.0)
    multiplier = multiplier.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return _run_portfolio(
        frames,
        signals,
        start=start,
        end=end,
        cost_bps=cost_bps,
        exposure_multiplier=multiplier,
    )


def _run_inverse_volatility_portfolio(
    frames: Mapping[str, pd.DataFrame],
    signals: Mapping[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
    volatility_lookback: int = 63,
    rebalance_bars: int = 21,
    target_annual_volatility: float = 0.10,
) -> dict[str, Any]:
    dates = sorted(
        {
            date
            for frame in frames.values()
            for date in frame.index
            if start <= date <= end
        }
    )
    cash = INITIAL_CAPITAL_EUR
    positions: dict[str, int] = {}
    latest_signal_scores: dict[str, float] = {}
    last_close_prices: dict[str, float] = {}
    ledgers: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    previous_nav = INITIAL_CAPITAL_EUR
    duplicate_position_days = 0
    current_weights: dict[str, float] = {}

    for bar_number, date in enumerate(dates):
        available = {
            identity: frame.loc[date]
            for identity, frame in frames.items()
            if date in frame.index
        }
        for identity, signal in signals.items():
            if date not in signal.index:
                continue
            location = signal.index.get_loc(date)
            if not isinstance(location, int) or location == 0:
                continue
            previous = signal.iloc[location - 1]
            if bool(previous["signal"]):
                latest_signal_scores[identity] = float(previous["score"])
            else:
                latest_signal_scores.pop(identity, None)

        scheduled_rebalance = (
            bar_number % max(1, rebalance_bars) == 0
        )
        if scheduled_rebalance:
            volatilities = {
                identity: volatility
                for identity in latest_signal_scores
                if (
                    volatility := _prior_volatility(
                        frames[identity],
                        date,
                        volatility_lookback,
                    )
                )
                is not None
            }
            desired = {
                identity
                for _, identity in sorted(
                    (
                        (latest_signal_scores[identity], identity)
                        for identity in volatilities
                    ),
                    key=lambda row: (-row[0], row[1]),
                )[:MAX_POSITIONS]
            }
            current_weights = _targeted_inverse_volatility_weights(
                frames,
                desired,
                date,
                volatilities,
                volatility_lookback,
                target_annual_volatility,
            )
        else:
            desired = set(positions)
        weights = current_weights
        holdings_before = set(positions)

        for identity in sorted(holdings_before - desired):
            if identity not in available:
                continue
            shares = positions.pop(identity)
            price = float(available[identity]["open"])
            notional = shares * price
            transaction_cost, fx_cost, fee = _cost_breakdown(
                notional,
                cost_bps,
            )
            cash += notional - fee
            fills.append(
                _fill(
                    date,
                    identity,
                    "SELL",
                    shares,
                    price,
                    transaction_cost,
                    fx_cost,
                    fee,
                )
            )

        should_rebalance = scheduled_rebalance
        mark_value = sum(
            shares
            * (
                float(available[identity]["open"])
                if identity in available
                else last_close_prices.get(identity, 0.0)
            )
            for identity, shares in positions.items()
        )
        equity_open = cash + mark_value
        if should_rebalance:
            targets = {
                identity: max(
                    0,
                    int(
                        equity_open
                        * weights.get(identity, 0.0)
                        // float(available[identity]["open"])
                    ),
                )
                for identity in desired
                if identity in available
            }
            for identity in sorted(set(positions) & set(targets)):
                shares_to_sell = max(
                    0,
                    positions[identity] - targets[identity],
                )
                if shares_to_sell <= 0:
                    continue
                price = float(available[identity]["open"])
                notional = shares_to_sell * price
                transaction_cost, fx_cost, fee = _cost_breakdown(
                    notional,
                    cost_bps,
                )
                cash += notional - fee
                positions[identity] -= shares_to_sell
                if positions[identity] <= 0:
                    positions.pop(identity)
                fills.append(
                    _fill(
                        date,
                        identity,
                        "SELL_REBALANCE",
                        shares_to_sell,
                        price,
                        transaction_cost,
                        fx_cost,
                        fee,
                    )
                )
            for identity in sorted(targets):
                shares_to_buy = max(
                    0,
                    targets[identity] - positions.get(identity, 0),
                )
                if shares_to_buy <= 0:
                    continue
                price = float(available[identity]["open"])
                notional = shares_to_buy * price
                transaction_cost, fx_cost, fee = _cost_breakdown(
                    notional,
                    cost_bps,
                )
                while shares_to_buy > 0 and notional + fee > cash:
                    shares_to_buy -= 1
                    notional = shares_to_buy * price
                    transaction_cost, fx_cost, fee = _cost_breakdown(
                        notional,
                        cost_bps,
                    )
                if shares_to_buy <= 0:
                    continue
                cash -= notional + fee
                positions[identity] = (
                    positions.get(identity, 0) + shares_to_buy
                )
                fills.append(
                    _fill(
                        date,
                        identity,
                        "BUY_REBALANCE",
                        shares_to_buy,
                        price,
                        transaction_cost,
                        fx_cost,
                        fee,
                    )
                )

        for identity, bar in available.items():
            last_close_prices[identity] = float(bar["close"])
        close_value = sum(
            shares * last_close_prices.get(identity, 0.0)
            for identity, shares in positions.items()
        )
        nav = cash + close_value
        gross_exposure = close_value / nav if nav > 0 else math.inf
        duplicate_position_days += int(len(positions) != len(set(positions)))
        ledgers.append(
            {
                "date": date,
                "nav_eur": nav,
                "cash_eur": cash,
                "gross_exposure": gross_exposure,
                "position_count": len(positions),
                "exposure_multiplier": 1.0,
                "daily_return": nav / previous_nav - 1 if previous_nav else 0.0,
                "inverse_volatility_weights": json.dumps(
                    weights,
                    sort_keys=True,
                ),
            }
        )
        previous_nav = nav

    ledger = pd.DataFrame(ledgers)
    fill_frame = pd.DataFrame(fills)
    metrics = _metrics(
        ledger.set_index("date")["daily_return"]
        if not ledger.empty
        else pd.Series(dtype=float)
    )
    return {
        "ledger": ledger,
        "fills": fill_frame,
        "metrics": metrics,
        "duplicate_position_days": duplicate_position_days,
        "selection_rule": (
            "SCORE_DESC_SECURITY_ID_ASC_TRUE_CAUSAL_INVERSE_VOL_NEXT_BAR"
        ),
    }


def _prior_volatility(
    frame: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int,
) -> float | None:
    history = frame.loc[frame.index < date, "close"].tail(lookback + 1)
    returns = history.pct_change(fill_method=None).dropna()
    if len(returns) < min(20, lookback):
        return None
    volatility = float(returns.std(ddof=1))
    if not math.isfinite(volatility) or volatility <= 0:
        return None
    return volatility


def _targeted_inverse_volatility_weights(
    frames: Mapping[str, pd.DataFrame],
    desired: set[str],
    date: pd.Timestamp,
    volatilities: Mapping[str, float],
    lookback: int,
    target_annual_volatility: float,
) -> dict[str, float]:
    inverse = {
        identity: 1.0 / volatilities[identity]
        for identity in desired
    }
    inverse_total = sum(inverse.values())
    if inverse_total <= 0:
        return {}
    raw = {
        identity: value / inverse_total
        for identity, value in inverse.items()
    }
    ordered = sorted(raw)
    return_table = pd.concat(
        {
            identity: frames[identity]
            .loc[frames[identity].index < date, "close"]
            .tail(lookback + 1)
            .pct_change(fill_method=None)
            for identity in ordered
        },
        axis=1,
    ).dropna()
    if len(return_table) < min(20, lookback):
        return raw
    covariance = return_table.cov().to_numpy(dtype=float) * 252.0
    vector = np.array([raw[identity] for identity in ordered], dtype=float)
    variance = float(vector @ covariance @ vector)
    if not math.isfinite(variance) or variance <= 0:
        return raw
    annual_volatility = math.sqrt(variance)
    scale = min(1.0, target_annual_volatility / annual_volatility)
    return {
        identity: raw[identity] * scale
        for identity in ordered
    }


def _fill(
    date: pd.Timestamp,
    identity: str,
    side: str,
    shares: int,
    price: float,
    transaction_cost: float,
    fx_cost: float,
    fee: float,
) -> dict[str, Any]:
    return {
        "date": date,
        "security_id": identity,
        "side": side,
        "shares": shares,
        "price_eur": price,
        "transaction_cost_eur": transaction_cost,
        "fx_cost_eur": fx_cost,
        "fee_eur": fee,
    }


def _select_profile(
    strategy_id: str,
    frames: Mapping[str, pd.DataFrame],
    signals: Mapping[str, Mapping[str, pd.DataFrame]],
    fold: Mapping[str, Any],
) -> tuple[str, bool]:
    rows = []
    for profile in PROFILES:
        run = _run_strategy_portfolio(
            strategy_id,
            frames,
            signals[profile],
            start=pd.Timestamp(fold["validation_start"]),
            end=pd.Timestamp(fold["validation_end"]),
            cost_bps=10.0,
        )
        rows.append(
            (
                _finite(run["metrics"].get("period_profit_factor"), -1.0),
                _finite(run["metrics"].get("CAGR"), -1.0),
                profile,
            )
        )
    rows.sort(reverse=True)
    best, neighbor = rows[:2]
    plateau = (
        best[0] > 1.0
        and neighbor[0] > 1.0
        and abs(best[0] - neighbor[0]) / max(abs(best[0]), 1e-9)
        <= 0.20
    )
    return str(best[2]), plateau


def _summarize(
    folds: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    if folds.empty or returns.empty:
        return pd.DataFrame()
    rows = []
    for strategy_id in sorted(folds["strategy_id"].unique()):
        normal_folds = folds.loc[
            folds["strategy_id"].eq(strategy_id)
            & folds["cost_bps"].eq(10.0)
        ]
        metrics_by_cost = {}
        for cost_bps in COSTS_BPS:
            values = returns.loc[
                returns["strategy_id"].eq(strategy_id)
                & returns["cost_bps"].eq(cost_bps)
            ].copy()
            series = (
                values.sort_values(["date", "fold_id"])
                .drop_duplicates(["date", "fold_id"])
                .set_index("date")["daily_return"]
            )
            metrics_by_cost[cost_bps] = _metrics(series)
        normal = metrics_by_cost[10.0]
        alpha = normal_folds["exposure_matched_alpha_CAGR"].dropna()
        rows.append(
            {
                "strategy_id": strategy_id,
                "timeframe": normal_folds["timeframe"].iloc[0],
                "fold_count": len(normal_folds),
                "positive_fold_count": int(
                    normal_folds["CAGR"].gt(0).sum()
                ),
                "positive_fold_ratio": float(
                    normal_folds["CAGR"].gt(0).mean()
                ),
                "combined_oos_return": float(
                    normal["terminal_nav"] / 10_000.0 - 1.0
                ),
                "combined_oos_CAGR": normal["CAGR"],
                "combined_oos_Sharpe": normal["Sharpe"],
                "combined_period_profit_factor": normal[
                    "period_profit_factor"
                ],
                "maximum_drawdown": normal["maximum_drawdown"],
                "cost_20bps_combined_return": float(
                    metrics_by_cost[20.0]["terminal_nav"] / 10_000.0 - 1.0
                ),
                "cost_50bps_combined_return": float(
                    metrics_by_cost[50.0]["terminal_nav"] / 10_000.0 - 1.0
                ),
                "cost_100bps_combined_return": float(
                    metrics_by_cost[100.0]["terminal_nav"] / 10_000.0 - 1.0
                ),
                "normal_cost_fill_count": int(
                    normal_folds["fill_count"].sum()
                ),
                "maximum_gross_exposure": float(
                    normal_folds["maximum_gross_exposure"].max()
                ),
                "minimum_cash_eur": float(
                    normal_folds["minimum_cash_eur"].min()
                ),
                "duplicate_position_days": int(
                    normal_folds["duplicate_position_days"].sum()
                ),
                "whole_share_violation_count": int(
                    normal_folds["whole_share_violation_count"].sum()
                ),
                "median_exposure_matched_alpha_CAGR": (
                    float(alpha.median()) if not alpha.empty else math.nan
                ),
                "positive_alpha_fold_count": int(alpha.gt(0).sum()),
                "parameter_plateau_ratio": float(
                    normal_folds["parameter_plateau"].mean()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["research_pass"] = (
        summary["combined_oos_return"].gt(0)
        & summary["positive_fold_count"].ge(3)
        & summary["combined_period_profit_factor"].ge(1.02)
        & summary["maximum_drawdown"].ge(-0.35)
        & summary["cost_20bps_combined_return"].gt(0)
        & summary["normal_cost_fill_count"].ge(20)
        & summary["maximum_gross_exposure"].le(1.0 + 1e-9)
        & summary["minimum_cash_eur"].ge(-1e-9)
        & summary["duplicate_position_days"].eq(0)
        & summary["whole_share_violation_count"].eq(0)
    )
    summary["robust_pass"] = (
        summary["combined_oos_return"].gt(0)
        & summary["positive_fold_count"].ge(4)
        & summary["combined_period_profit_factor"].ge(1.05)
        & summary["maximum_drawdown"].ge(-0.25)
        & summary["median_exposure_matched_alpha_CAGR"].gt(0)
        & summary["cost_50bps_combined_return"].gt(0)
        & summary["normal_cost_fill_count"].ge(30)
        & summary["maximum_gross_exposure"].le(1.0 + 1e-9)
        & summary["minimum_cash_eur"].ge(-1e-9)
        & summary["duplicate_position_days"].eq(0)
        & summary["whole_share_violation_count"].eq(0)
    )
    summary["portfolio_invariants_go"] = (
        summary["maximum_gross_exposure"].le(1.0 + 1e-9)
        & summary["minimum_cash_eur"].ge(-1e-9)
        & summary["duplicate_position_days"].eq(0)
        & summary["whole_share_violation_count"].eq(0)
    )
    summary["shariah_product_structure_status"] = np.where(
        summary["strategy_id"].eq("MULTI_ASSET_INVERSE_VOL_TREND"),
        (
            "NO_KNOWN_INTEREST_OR_DERIVATIVE_PROXY_BLOCKER_"
            "ETF_HOLDINGS_ELIGIBILITY_UNVERIFIED"
        ),
        "HISTORICAL_ELIGIBILITY_UNVERIFIED",
    )
    summary["deployable_pass"] = False
    summary["deployment_blockers"] = (
        "CURRENT_SYMBOL_UNIVERSE_NOT_PIT|SHARIAH_HISTORY_PARTIAL|"
        "INDEPENDENT_FORWARD_SESSION_MISSING"
    )
    return summary.sort_values(
        [
            "robust_pass",
            "research_pass",
            "combined_oos_Sharpe",
            "combined_period_profit_factor",
        ],
        ascending=False,
    )


def _qualification(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {
            "status": "NO_EVALUABLE_STRATEGIES",
            "research_pass_count": 0,
            "robust_pass_count": 0,
            "deployable_pass_count": 0,
            "strategies": [],
        }
    return {
        "status": "FAST_TRACK_RESEARCH_COMPLETE",
        "research_pass_count": int(summary["research_pass"].sum()),
        "robust_pass_count": int(summary["robust_pass"].sum()),
        "deployable_pass_count": 0,
        "strategies": json.loads(summary.to_json(orient="records")),
        "automatic_live_activation": False,
    }


def _append_returns(
    rows: list[dict[str, Any]],
    run: Mapping[str, Any],
    *,
    strategy_id: str,
    fold_id: str,
    cost_bps: float,
) -> None:
    for item in run["ledger"].to_dict("records"):
        rows.append(
            {
                "strategy_id": strategy_id,
                "fold_id": fold_id,
                "cost_bps": cost_bps,
                "date": item["date"],
                "daily_return": item["daily_return"],
            }
        )


def _append_fills(
    rows: list[dict[str, Any]],
    run: Mapping[str, Any],
    *,
    strategy_id: str,
    fold_id: str,
    cost_bps: float,
) -> None:
    for item in run["fills"].to_dict("records"):
        rows.append(
            {
                "strategy_id": strategy_id,
                "fold_id": fold_id,
                "cost_bps": cost_bps,
                **item,
            }
        )


def _maximum_exposure(run: Mapping[str, Any]) -> float:
    ledger = run["ledger"]
    return float(ledger["gross_exposure"].max()) if not ledger.empty else 0.0


def _minimum_cash(run: Mapping[str, Any]) -> float:
    ledger = run["ledger"]
    return float(ledger["cash_eur"].min()) if not ledger.empty else 0.0


def _whole_share_violation_count(run: Mapping[str, Any]) -> int:
    fills = run["fills"]
    if fills.empty:
        return 0
    return int(
        sum(
            not float(shares).is_integer() or float(shares) <= 0
            for shares in fills["shares"]
        )
    )


def _price_basis(frames: Mapping[str, pd.DataFrame]) -> str:
    bases = {
        str(frame.attrs.get("price_basis", "ADJUSTED_DAILY_OR_UNKNOWN"))
        for frame in frames.values()
    }
    return "|".join(sorted(bases))


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _output(project_root: Path) -> Path:
    return project_root / "output" / "research" / "phase11_13"


def _public_strategies() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for strategy_id, specification in FROZEN_STRATEGIES.items():
        output[strategy_id] = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in specification.items()
        }
    return output


def _write_json(path: Path, payload: Any) -> None:
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


def _hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
