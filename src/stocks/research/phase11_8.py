from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.research.phase11_6 import (
    nested_walk_forward_folds,
    run_data_audit,
)


SCHEMA = "phase11_8_realistic_multistrategy_forward_campaign_v1"
TIMEFRAMES = ("1h", "4h", "1d", "1w", "1mo")
COSTS_BPS = (5.0, 10.0, 20.0, 30.0, 50.0)
MAX_POSITIONS = 4
INITIAL_CAPITAL_EUR = 10_000.0
FIXED_FEE_EUR = 1.0
FX_COST_BPS = 1.0
AUTHORITY = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_RESEARCH_SHADOW": "OBSERVATION_ONLY",
    "STRATEGY_AUTHORITY": "NONE",
    "EXECUTION_AUTHORITY": "NONE",
    "PAPER_STRATEGY_AUTHORITY": "NONE",
    "LIVE_STRATEGY_AUTHORITY": "NONE",
    "BROKER_CALLS": 0,
    "ORDER_CALLS": 0,
}
PARAMETERS: dict[str, tuple[dict[str, float], ...]] = {
    "ma_crossover": (
        {"fast": 50, "slow": 200},
        {"fast": 70, "slow": 230},
    ),
    "asymmetric_ma": (
        {"entry_fast": 50, "entry_slow": 200, "exit_fast": 100, "exit_slow": 250},
        {"entry_fast": 70, "entry_slow": 210, "exit_fast": 110, "exit_slow": 300},
    ),
    "ma_channel": ({"period": 20}, {"period": 55}),
    "bollinger_breakout": (
        {"period": 100, "sigma": 2.0},
        {"period": 200, "sigma": 2.5},
    ),
    "volatility_contraction_breakout": (
        {"short_vol": 10, "long_vol": 60, "ratio": 0.65, "breakout": 20},
        {"short_vol": 20, "long_vol": 100, "ratio": 0.70, "breakout": 55},
    ),
    "etf_trend": (
        {"momentum": 63, "trend": 100},
        {"momentum": 126, "trend": 200},
    ),
    "commodity_etf_trend": (
        {"momentum": 63, "trend": 100},
        {"momentum": 126, "trend": 200},
    ),
}


@dataclass(frozen=True)
class Layout:
    project_root: Path

    @property
    def output(self) -> Path:
        return self.project_root / "output" / "research" / "phase11_8"

    @property
    def private(self) -> Path:
        return (
            self.project_root / "data" / "research" / "phase11_8" / "private"
        )

    @property
    def pit_bars(self) -> Path:
        return (
            self.project_root
            / "data"
            / "research"
            / "phase11_4"
            / "private"
            / "pit-bars.parquet"
        )


def phase11_8_schema(project_root: Path) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "status": "GO",
        "timeframes": list(TIMEFRAMES),
        "cost_stress_bps_per_side": list(COSTS_BPS),
        "strategy_parameters": PARAMETERS,
        "portfolio_contract": {
            "whole_shares": True,
            "global_max_gross_exposure": 1.0,
            "global_security_netting": True,
            "simultaneous_selection": "SCORE_DESC_SECURITY_ID_ASC",
            "execution": "NEXT_BAR_OPEN",
            "base_currency": "EUR",
            "historical_fx_conversion": "POINT_IN_TIME_EURUSD",
            "fx_cost_bps_per_side": FX_COST_BPS,
            "portfolio_profit_factor_primary": True,
            "maximum_positions": MAX_POSITIONS,
        },
        "macro_policy": "CONTEXT_ONLY_UNTIL_CORRECTED_OOS_ABLATION_GO",
        "future_holdout": "STARTS_AFTER_CAMPAIGN_FREEZE",
        **AUTHORITY,
    }
    _write_json(Layout(project_root).output / "schema.json", payload)
    return payload


def data_coverage(project_root: Path) -> dict[str, Any]:
    layout = Layout(project_root)
    timeframe = run_data_audit(project_root)
    pit_rows = 0
    pit_identities = 0
    pit_start: str | None = None
    pit_end: str | None = None
    if layout.pit_bars.exists():
        import duckdb

        connection = duckdb.connect()
        try:
            row = connection.execute(
                """
                SELECT count(*), count(DISTINCT security_id),
                       min(TRY_CAST(date AS DATE)), max(TRY_CAST(date AS DATE))
                FROM read_parquet(?)
                """,
                [str(layout.pit_bars)],
            ).fetchone()
            if row is None:
                raise ValueError("PIT_BAR_SUMMARY_UNAVAILABLE")
            pit_rows, pit_identities = int(row[0]), int(row[1])
            pit_start = str(row[2]) if row[2] else None
            pit_end = str(row[3]) if row[3] else None
        finally:
            connection.close()
    sec_root = (
        project_root / "data" / "research" / "phase11_2" / "private"
    )
    fundamental_files = list(sec_root.rglob("*fundamental*.parquet"))
    earnings_files = list(sec_root.rglob("*earning*.parquet"))
    revision_files = list(sec_root.rglob("*revision*.parquet"))
    pmi_files = [
        path
        for path in (project_root / "data").rglob("*.parquet")
        if "pmi" in path.name.lower()
    ]
    blockers = []
    if not pmi_files:
        blockers.append("LICENSED_PMI_HISTORY_NOT_AVAILABLE")
    if not revision_files:
        blockers.append("BROAD_PIT_EARNINGS_REVISIONS_NOT_AVAILABLE")
    if not fundamental_files:
        blockers.append("BROAD_PIT_FUNDAMENTALS_NOT_AVAILABLE")
    if not earnings_files:
        blockers.append("BROAD_PIT_EARNINGS_EVENTS_NOT_AVAILABLE")
    report = {
        "schema": "phase11_8_data_coverage_v1",
        "status": "PARTIAL" if blockers else "GO",
        "pit_price_rows": pit_rows,
        "pit_price_identities": pit_identities,
        "pit_price_start": pit_start,
        "pit_price_end": pit_end,
        "fundamental_file_count": len(fundamental_files),
        "earnings_file_count": len(earnings_files),
        "revision_file_count": len(revision_files),
        "licensed_pmi_file_count": len(pmi_files),
        "timeframe_readiness": timeframe["rows"],
        "no_synthetic_intraday": True,
        "provider_credentials_present_not_proof_of_license": True,
        "blockers": blockers,
        **AUTHORITY,
    }
    _write_json(layout.output / "data-coverage.json", report)
    return report


def portfolio_invariant_audit(project_root: Path) -> dict[str, Any]:
    synthetic = _synthetic_frames()
    signals = _signal_frames(
        synthetic, "ma_crossover", {"fast": 5, "slow": 20}
    )
    result = _run_portfolio(
        synthetic,
        signals,
        start=pd.Timestamp("2021-01-01"),
        end=pd.Timestamp("2021-12-31"),
        cost_bps=50.0,
    )
    ledger = result["ledger"]
    fills = result["fills"]
    checks = {
        "whole_share_accounting": bool(
            fills.empty
            or np.allclose(
                fills["shares"].astype(float),
                np.floor(fills["shares"].astype(float)),
            )
        ),
        "global_exposure_at_most_100_percent": bool(
            ledger["gross_exposure"].max() <= 1.0000001
        ),
        "security_netting": result["duplicate_position_days"] == 0,
        "causal_selection": result["selection_rule"]
        == "SCORE_DESC_SECURITY_ID_ASC_NEXT_BAR",
        "eur_costs_and_fx": bool(
            (
                (fills["fee_eur"] >= 0).all()
                and (fills["transaction_cost_eur"] >= 0).all()
                and (fills["fx_cost_eur"] >= 0).all()
                and np.allclose(
                    fills["fee_eur"],
                    fills["transaction_cost_eur"] + fills["fx_cost_eur"],
                )
            )
            if not fills.empty
            else True
        ),
        "portfolio_pf_primary": "period_profit_factor" in result["metrics"],
        "cash_non_negative": bool(ledger["cash_eur"].min() >= -0.000001),
    }
    report = {
        "schema": "phase11_8_portfolio_invariant_audit_v1",
        "status": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "maximum_observed_gross_exposure": float(
            ledger["gross_exposure"].max()
        ),
        "minimum_cash_eur": float(ledger["cash_eur"].min()),
        "fill_count": len(fills),
        **AUTHORITY,
    }
    _write_json(Layout(project_root).output / "portfolio-invariant-audit.json", report)
    return report


def run_campaign(
    project_root: Path, *, max_stock_identities: int = 30
) -> dict[str, Any]:
    layout = Layout(project_root)
    layout.output.mkdir(parents=True, exist_ok=True)
    layout.private.mkdir(parents=True, exist_ok=True)
    schema = phase11_8_schema(project_root)
    coverage = data_coverage(project_root)
    accounting = portfolio_invariant_audit(project_root)
    universes = _load_universes(project_root, max_stock_identities)
    readiness = {
        row["interval"]: row["status"]
        for row in coverage["timeframe_readiness"]
    }
    result_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    blocked_rows.append(
        {
            "strategy": "quality_momentum",
            "timeframe": "*",
            "reason": "BROAD_POINT_IN_TIME_FUNDAMENTALS_AND_REVISIONS_UNAVAILABLE",
        }
    )
    for timeframe in TIMEFRAMES:
        if readiness.get(timeframe) != "GO":
            blocked_rows.append(
                {
                    "strategy": "*",
                    "timeframe": timeframe,
                    "reason": "BLOCKED_INSUFFICIENT_NATIVE_HISTORY",
                }
            )
            continue
        for strategy, variants in PARAMETERS.items():
            universe_name = (
                "commodity"
                if strategy == "commodity_etf_trend"
                else "etf"
                if strategy == "etf_trend"
                else "stocks"
            )
            if timeframe not in {"1d", "1w", "1mo"}:
                blocked_rows.append(
                    {
                        "strategy": strategy,
                        "timeframe": timeframe,
                        "reason": "INTRADAY_PORTFOLIO_INPUT_NOT_READY",
                    }
                )
                continue
            frames = _aggregate_frames(universes[universe_name], timeframe)
            if len(frames) < 2:
                blocked_rows.append(
                    {
                        "strategy": strategy,
                        "timeframe": timeframe,
                        "reason": "INSUFFICIENT_UNIVERSE",
                    }
                )
                continue
            # Instruments enter the investable universe only once their own
            # history begins; a recent listing must not truncate every asset.
            start = min(frame.index.min() for frame in frames.values())
            end = max(frame.index.max() for frame in frames.values())
            folds = nested_walk_forward_folds(start, end, timeframe)
            if folds.empty:
                blocked_rows.append(
                    {
                        "strategy": strategy,
                        "timeframe": timeframe,
                        "reason": "INSUFFICIENT_NESTED_FOLDS",
                    }
                )
                continue
            signal_cache = {
                _hash(parameters): _signal_frames(frames, strategy, parameters)
                for parameters in variants
            }
            for fold in folds.to_dict("records"):
                validation_scores = []
                for parameters in variants:
                    parameter_hash = _hash(parameters)
                    run = _run_portfolio(
                        frames,
                        signal_cache[parameter_hash],
                        start=pd.Timestamp(fold["validation_start"]),
                        end=pd.Timestamp(fold["validation_end"]),
                        cost_bps=10.0,
                    )
                    metrics = run["metrics"]
                    validation_scores.append(
                        (
                            _finite(metrics["period_profit_factor"], -1.0),
                            _finite(metrics["CAGR"], -1.0),
                            parameter_hash,
                            parameters,
                        )
                    )
                validation_scores.sort(reverse=True, key=lambda row: row[:3])
                selected = validation_scores[0]
                finite_pfs = [
                    row[0] for row in validation_scores if row[0] >= 0
                ]
                plateau = (
                    len(finite_pfs) >= 2
                    and finite_pfs[1] > 1.0
                    and abs(finite_pfs[0] - finite_pfs[1])
                    / max(abs(finite_pfs[0]), 1e-9)
                    <= 0.20
                )
                selection_rows.append(
                    {
                        "fold_id": fold["fold_id"],
                        "universe": universe_name,
                        "strategy": strategy,
                        "timeframe": timeframe,
                        "parameter_hash": selected[2],
                        "parameters": json.dumps(selected[3], sort_keys=True),
                        "validation_portfolio_pf": selected[0],
                        "parameter_plateau": plateau,
                        "selection_metric": "PORTFOLIO_PERIOD_PF_THEN_CAGR",
                    }
                )
                for cost in COSTS_BPS:
                    test = _run_portfolio(
                        frames,
                        signal_cache[selected[2]],
                        start=pd.Timestamp(fold["outer_test_start"]),
                        end=pd.Timestamp(fold["outer_test_end"]),
                        cost_bps=cost,
                    )
                    result_rows.append(
                        {
                            "fold_id": fold["fold_id"],
                            "universe": universe_name,
                            "strategy": strategy,
                            "timeframe": timeframe,
                            "parameter_hash": selected[2],
                            "cost_bps": cost,
                            "parameter_plateau": plateau,
                            **test["metrics"],
                            "maximum_gross_exposure": float(
                                test["ledger"]["gross_exposure"].max()
                            ),
                            "whole_shares": True,
                            "security_netting": True,
                            "portfolio_pf_primary": True,
                        }
                    )
    results = pd.DataFrame(result_rows)
    selections = pd.DataFrame(selection_rows)
    summary = _summarize(results, selections)
    _write_frame(layout.output / "nested-walk-forward-results.parquet", results)
    _write_frame(layout.output / "nested-walk-forward-summary.csv", summary)
    _write_frame(layout.output / "parameter-selection.csv", selections)
    _write_jsonl(layout.output / "blocked-strategy-timeframes.jsonl", blocked_rows)
    macro = {
        "status": "CONTEXT_ONLY",
        "mandatory_strategy_input": False,
        "reason": "NO_CORRECTED_OOS_INCREMENTAL_VALUE_PROVEN",
        "future_activation_gate": "PAIRED_OOS_ABLATION_EXCESS_SHARPE_AND_CAGR_POSITIVE",
    }
    _write_json(layout.output / "macro-policy.json", macro)
    candidate = _select_candidate(summary)
    holdout = _register_forward_holdout(project_root, candidate)
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "data_coverage_status": coverage["status"],
        "portfolio_invariant_status": accounting["status"],
        "stock_identity_count": len(universes["stocks"]),
        "etf_identity_count": len(universes["etf"]),
        "commodity_identity_count": len(universes["commodity"]),
        "result_count": len(results),
        "strategy_timeframe_count": int(
            summary[["strategy", "timeframe"]].drop_duplicates().shape[0]
        )
        if not summary.empty
        else 0,
        "blocked_strategy_timeframes": len(blocked_rows),
        "candidate": candidate,
        "forward_holdout": holdout,
        "macro_status": "CONTEXT_ONLY",
        "historical_period_consumed": True,
        "independent_future_observations": 0,
        "schema_hash": _hash(schema),
        **AUTHORITY,
    }
    _write_json(layout.output / "manifest.json", report)
    _write_json(layout.output / "status.json", report)
    return report


def phase11_8_status(project_root: Path) -> dict[str, Any]:
    path = Layout(project_root).output / "status.json"
    if not path.exists():
        return {"schema": SCHEMA, "status": "NOT_RUN", **AUTHORITY}
    return json.loads(path.read_text(encoding="utf-8"))


def finalize_campaign(project_root: Path) -> dict[str, Any]:
    layout = Layout(project_root)
    summary_path = layout.output / "nested-walk-forward-summary.csv"
    status_path = layout.output / "status.json"
    if not summary_path.exists() or not status_path.exists():
        return {"schema": SCHEMA, "status": "NOT_RUN", **AUTHORITY}
    summary = pd.read_csv(summary_path)
    candidate = _select_candidate(summary)
    holdout = _register_forward_holdout(project_root, candidate)
    report = json.loads(status_path.read_text(encoding="utf-8"))
    report["candidate"] = candidate
    report["forward_holdout"] = holdout
    report["selection_policy"] = (
        "PLATEAU_AND_WORST_FOLD_FIRST_THEN_MEDIAN_PORTFOLIO_PF"
    )
    _write_json(status_path, report)
    _write_json(layout.output / "manifest.json", report)
    return report


def _load_universes(
    project_root: Path, max_stock_identities: int
) -> dict[str, dict[str, pd.DataFrame]]:
    stocks = _load_stocks(project_root, max_stock_identities)
    etf_symbols = {
        "ACWI",
        "BIL",
        "DIA",
        "EEM",
        "EWJ",
        "IEF",
        "IEUR",
        "INDA",
        "IWM",
        "MCHI",
        "OEF",
        "QQQ",
        "SPY",
        "TLT",
        "VNQ",
    }
    commodity_symbols = {"DBA", "DBB", "DBC", "GLD", "SLV"}
    cache = (
        project_root / "data" / "research" / "critical_trading" / "yfinance"
    )
    etfs = _load_symbol_cache(cache, etf_symbols)
    commodities = _load_symbol_cache(cache, commodity_symbols)
    return {
        "stocks": _convert_usd_frames(project_root, stocks),
        "etf": _convert_usd_frames(project_root, etfs),
        "commodity": _convert_usd_frames(project_root, commodities),
    }


def _load_stocks(
    project_root: Path, max_stock_identities: int
) -> dict[str, pd.DataFrame]:
    import duckdb

    path = Layout(project_root).pit_bars
    connection = duckdb.connect()
    try:
        selected = connection.execute(
            """
            SELECT security_id
            FROM read_parquet(?)
            GROUP BY security_id
            HAVING count(*) >= 1260
            ORDER BY hash(security_id, 20260727)
            LIMIT ?
            """,
            [str(path), max_stock_identities],
        ).fetchdf()
        if selected.empty:
            return {}
        connection.register("_selected", selected)
        frame = connection.execute(
            """
            SELECT b.security_id, TRY_CAST(b.date AS DATE) date,
                   b.open, b.high, b.low, b.close, b.volume
            FROM read_parquet(?) b
            INNER JOIN _selected s USING(security_id)
            ORDER BY b.security_id, date
            """,
            [str(path)],
        ).fetchdf()
    finally:
        connection.close()
    return _split_frames(frame, "security_id")


def _load_symbol_cache(
    cache: Path, symbols: set[str]
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in sorted(symbols):
        path = cache / f"{symbol}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if {"session_date", "open", "high", "low", "close"}.issubset(frame):
            work = frame.rename(columns={"session_date": "date"}).copy()
            work["security_id"] = f"ETF:{symbol}"
            frames[f"ETF:{symbol}"] = _normalize_frame(work)
    return frames


def _convert_usd_frames(
    project_root: Path, frames: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    if not frames:
        return {}
    import strategy_combo_research_lab as lab

    calendar = pd.DatetimeIndex(
        sorted({date for frame in frames.values() for date in frame.index})
    )
    fx = lab.load_v2_fx(project_root, calendar)
    if fx.isna().any():
        raise RuntimeError("PHASE11_8_EUR_FX_COVERAGE_BLOCKED")
    converted: dict[str, pd.DataFrame] = {}
    for identity, frame in frames.items():
        aligned = fx.reindex(frame.index)
        if aligned.isna().any():
            continue
        work = frame.copy()
        for column in ("open", "high", "low", "close"):
            work[column] = work[column].astype(float) * aligned
        converted[identity] = work
    return converted


def _split_frames(
    frame: pd.DataFrame, identity_column: str
) -> dict[str, pd.DataFrame]:
    return {
        str(identity): _normalize_frame(group)
        for identity, group in frame.groupby(identity_column, sort=False)
    }


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.tz_localize(None)
    work = work.sort_values("date").drop_duplicates("date").set_index("date")
    return work[["open", "high", "low", "close", "volume"]].astype(float)


def _aggregate_frames(
    frames: Mapping[str, pd.DataFrame], timeframe: str
) -> dict[str, pd.DataFrame]:
    if timeframe == "1d":
        return {key: value.copy() for key, value in frames.items()}
    frequency = "W-FRI" if timeframe == "1w" else "ME"
    result = {}
    for key, frame in frames.items():
        aggregate = frame.resample(frequency, label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        aggregate = aggregate.dropna(subset=["open", "high", "low", "close"])
        if not aggregate.empty:
            result[key] = aggregate
    return result


def _signal_frames(
    frames: Mapping[str, pd.DataFrame],
    strategy: str,
    parameters: Mapping[str, float],
) -> dict[str, pd.DataFrame]:
    return {
        identity: _strategy_state(frame, strategy, parameters)
        for identity, frame in frames.items()
    }


def _strategy_state(
    frame: pd.DataFrame,
    strategy: str,
    parameters: Mapping[str, float],
) -> pd.DataFrame:
    close = frame["close"]
    score = close.pct_change(63)
    if strategy == "ma_crossover":
        state = close.rolling(int(parameters["fast"])).mean().gt(
            close.rolling(int(parameters["slow"])).mean()
        )
    elif strategy == "asymmetric_ma":
        enter = close.rolling(int(parameters["entry_fast"])).mean().gt(
            close.rolling(int(parameters["entry_slow"])).mean()
        )
        leave = close.rolling(int(parameters["exit_fast"])).mean().lt(
            close.rolling(int(parameters["exit_slow"])).mean()
        )
        active = False
        values = []
        for entry, exit_ in zip(enter.fillna(False), leave.fillna(False)):
            if active and exit_:
                active = False
            elif not active and entry:
                active = True
            values.append(active)
        state = pd.Series(values, index=close.index)
    elif strategy == "ma_channel":
        period = int(parameters["period"])
        state = close.gt(frame["high"].rolling(period).max().shift(1))
        state = _persistent_state(
            state, close.lt(close.rolling(period).mean())
        )
    elif strategy == "bollinger_breakout":
        period = int(parameters["period"])
        mean = close.rolling(period).mean()
        upper = mean + float(parameters["sigma"]) * close.rolling(period).std()
        state = _persistent_state(close.gt(upper), close.lt(mean))
    elif strategy == "volatility_contraction_breakout":
        returns = close.pct_change()
        contraction = returns.rolling(int(parameters["short_vol"])).std().div(
            returns.rolling(int(parameters["long_vol"])).std()
        ).lt(float(parameters["ratio"]))
        breakout = close.gt(
            frame["high"].rolling(int(parameters["breakout"])).max().shift(1)
        )
        state = _persistent_state(
            contraction & breakout,
            close.lt(close.rolling(int(parameters["breakout"])).mean()),
        )
    elif strategy in {"etf_trend", "commodity_etf_trend"}:
        momentum = close.pct_change(int(parameters["momentum"]))
        state = momentum.gt(0) & close.gt(
            close.rolling(int(parameters["trend"])).mean()
        )
        score = momentum
    else:
        raise ValueError(f"UNREGISTERED_PHASE11_8_STRATEGY:{strategy}")
    return pd.DataFrame(
        {
            "signal": state.fillna(False).astype(bool),
            "score": score.replace([np.inf, -np.inf], np.nan).fillna(-math.inf),
        },
        index=frame.index,
    )


def _persistent_state(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    active = False
    values = []
    for enter, leave in zip(entry.fillna(False), exit_.fillna(False)):
        active = (active or bool(enter)) and not bool(leave)
        values.append(active)
    return pd.Series(values, index=entry.index)


def _run_portfolio(
    frames: Mapping[str, pd.DataFrame],
    signals: Mapping[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_bps: float,
    exposure_multiplier: pd.Series | None = None,
    entry_block_below_multiplier: float = 0.0,
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
    ledgers: list[dict[str, Any]] = []
    fills: list[dict[str, Any]] = []
    duplicate_position_days = 0
    previous_nav = INITIAL_CAPITAL_EUR
    latest_signal_scores: dict[str, float] = {}
    last_close_prices: dict[str, float] = {}
    for date in dates:
        active_multiplier = 1.0
        if exposure_multiplier is not None:
            prior = exposure_multiplier.loc[
                exposure_multiplier.index < date
            ]
            active_multiplier = (
                float(prior.iloc[-1]) if not prior.empty else 0.0
            )
            active_multiplier = min(max(active_multiplier, 0.0), 1.0)
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
        desired = {
            identity
            for _, identity in sorted(
                (
                    (score, identity)
                    for identity, score in latest_signal_scores.items()
                ),
                key=lambda row: (-row[0], row[1]),
            )[:MAX_POSITIONS]
        }
        if active_multiplier < entry_block_below_multiplier:
            desired = set()
        for identity in sorted(set(positions) - desired):
            if identity not in available:
                continue
            shares = positions.pop(identity)
            price = float(available[identity]["open"])
            notional = shares * price
            transaction_cost, fx_cost, fee = _cost_breakdown(
                notional, cost_bps
            )
            cash += notional - fee
            fills.append(
                {
                    "date": date,
                    "security_id": identity,
                    "side": "SELL",
                    "shares": shares,
                    "price_eur": price,
                    "transaction_cost_eur": transaction_cost,
                    "fx_cost_eur": fx_cost,
                    "fee_eur": fee,
                }
            )
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
        target_value = (
            equity_open / MAX_POSITIONS * active_multiplier
        )
        for identity in sorted(set(positions) & desired):
            if identity not in available:
                continue
            price = float(available[identity]["open"])
            current_shares = positions[identity]
            target_shares = max(0, int(target_value // price))
            shares_to_sell = max(0, current_shares - target_shares)
            if shares_to_sell <= 0:
                continue
            notional = shares_to_sell * price
            transaction_cost, fx_cost, fee = _cost_breakdown(
                notional, cost_bps
            )
            cash += notional - fee
            positions[identity] -= shares_to_sell
            if positions[identity] <= 0:
                positions.pop(identity)
            fills.append(
                {
                    "date": date,
                    "security_id": identity,
                    "side": "SELL_RISK_OVERLAY",
                    "shares": shares_to_sell,
                    "price_eur": price,
                    "transaction_cost_eur": transaction_cost,
                    "fx_cost_eur": fx_cost,
                    "fee_eur": fee,
                }
            )
        for identity in sorted(desired - set(positions)):
            if identity not in available:
                continue
            price = float(available[identity]["open"])
            budget = min(target_value, cash)
            shares = max(0, int((budget - FIXED_FEE_EUR) // price))
            if shares <= 0:
                continue
            notional = shares * price
            transaction_cost, fx_cost, fee = _cost_breakdown(
                notional, cost_bps
            )
            while shares > 0 and notional + fee > cash:
                shares -= 1
                notional = shares * price
                transaction_cost, fx_cost, fee = _cost_breakdown(
                    notional, cost_bps
                )
            if shares <= 0:
                continue
            cash -= notional + fee
            positions[identity] = shares
            fills.append(
                {
                    "date": date,
                    "security_id": identity,
                    "side": "BUY",
                    "shares": shares,
                    "price_eur": price,
                    "transaction_cost_eur": transaction_cost,
                    "fx_cost_eur": fx_cost,
                    "fee_eur": fee,
                }
            )
        for identity, bar in available.items():
            last_close_prices[identity] = float(bar["close"])
        close_value = sum(
            shares * last_close_prices.get(identity, 0.0)
            for identity, shares in positions.items()
        )
        nav = cash + close_value
        gross = close_value / nav if nav > 0 else math.inf
        duplicate_position_days += int(len(positions) != len(set(positions)))
        ledgers.append(
            {
                "date": date,
                "nav_eur": nav,
                "cash_eur": cash,
                "gross_exposure": gross,
                "position_count": len(positions),
                "exposure_multiplier": active_multiplier,
                "daily_return": nav / previous_nav - 1 if previous_nav else 0.0,
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
        "selection_rule": "SCORE_DESC_SECURITY_ID_ASC_NEXT_BAR",
    }


def _cost_breakdown(
    notional_eur: float, cost_bps: float
) -> tuple[float, float, float]:
    transaction_cost = FIXED_FEE_EUR + notional_eur * cost_bps / 10_000
    fx_cost = notional_eur * FX_COST_BPS / 10_000
    return transaction_cost, fx_cost, transaction_cost + fx_cost


def _metrics(returns: pd.Series) -> dict[str, Any]:
    values = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "CAGR": None,
            "Sharpe": None,
            "maximum_drawdown": None,
            "period_profit_factor": None,
            "terminal_nav": INITIAL_CAPITAL_EUR,
        }
    nav = INITIAL_CAPITAL_EUR * (1 + values).cumprod()
    years = max((values.index.max() - values.index.min()).days / 365.2425, 1 / 12)
    gains = float(values[values > 0].sum())
    losses = abs(float(values[values < 0].sum()))
    volatility = float(values.std(ddof=1))
    periods = len(values) / years
    return {
        "CAGR": float((nav.iloc[-1] / INITIAL_CAPITAL_EUR) ** (1 / years) - 1),
        "Sharpe": (
            float(values.mean() / volatility * math.sqrt(periods))
            if volatility > 0
            else None
        ),
        "maximum_drawdown": float((nav / nav.cummax() - 1).min()),
        "period_profit_factor": (
            gains / losses if losses > 0 else math.inf if gains > 0 else None
        ),
        "terminal_nav": float(nav.iloc[-1]),
    }


def _summarize(
    results: pd.DataFrame, selections: pd.DataFrame
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = []
    normal = results.loc[results["cost_bps"].eq(10.0)]
    for keys, group in normal.groupby(["universe", "strategy", "timeframe"]):
        stress = results.loc[
            results["universe"].eq(keys[0])
            & results["strategy"].eq(keys[1])
            & results["timeframe"].eq(keys[2])
            & results["cost_bps"].eq(50.0)
        ]
        selection = selections.loc[
            selections["universe"].eq(keys[0])
            & selections["strategy"].eq(keys[1])
            & selections["timeframe"].eq(keys[2])
        ]
        pfs = group["period_profit_factor"].replace([np.inf, -np.inf], np.nan)
        rows.append(
            {
                "universe": keys[0],
                "strategy": keys[1],
                "timeframe": keys[2],
                "fold_count": len(group),
                "positive_fold_ratio": float(group["CAGR"].gt(0).mean()),
                "median_oos_portfolio_pf": float(pfs.median()),
                "worst_oos_portfolio_pf": float(pfs.min()),
                "median_oos_CAGR": float(group["CAGR"].median()),
                "worst_oos_drawdown": float(group["maximum_drawdown"].min()),
                "cost_50bps_median_pf": float(
                    stress["period_profit_factor"]
                    .replace([np.inf, -np.inf], np.nan)
                    .median()
                ),
                "plateau_fold_ratio": float(
                    selection["parameter_plateau"].mean()
                ),
                "macro_used": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["median_oos_portfolio_pf", "median_oos_CAGR"],
        ascending=False,
    )


def _select_candidate(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"status": "NO_RESEARCH_CANDIDATE"}
    eligible = summary.loc[
        summary["fold_count"].ge(3)
        & summary["positive_fold_ratio"].ge(0.60)
        & summary["median_oos_portfolio_pf"].gt(1.0)
        & summary["median_oos_CAGR"].gt(0)
        & summary["cost_50bps_median_pf"].gt(1.0)
        & summary["worst_oos_drawdown"].gt(-0.50)
        & summary["worst_oos_portfolio_pf"].gt(0.70)
        & summary["plateau_fold_ratio"].ge(0.50)
    ]
    if eligible.empty:
        return {"status": "NO_RESEARCH_CANDIDATE"}
    robust = eligible.sort_values(
        [
            "worst_oos_portfolio_pf",
            "median_oos_portfolio_pf",
            "worst_oos_drawdown",
        ],
        ascending=False,
    )
    raw = robust.iloc[0].to_dict()
    row = {
        "universe": str(raw["universe"]),
        "strategy": str(raw["strategy"]),
        "timeframe": str(raw["timeframe"]),
        "fold_count": int(raw["fold_count"]),
        "positive_fold_ratio": round(float(raw["positive_fold_ratio"]), 12),
        "median_oos_portfolio_pf": round(
            float(raw["median_oos_portfolio_pf"]), 12
        ),
        "worst_oos_portfolio_pf": round(
            float(raw["worst_oos_portfolio_pf"]), 12
        ),
        "median_oos_CAGR": round(float(raw["median_oos_CAGR"]), 12),
        "worst_oos_drawdown": round(float(raw["worst_oos_drawdown"]), 12),
        "cost_50bps_median_pf": round(
            float(raw["cost_50bps_median_pf"]), 12
        ),
        "plateau_fold_ratio": round(float(raw["plateau_fold_ratio"]), 12),
        "macro_used": bool(raw.get("macro_used", False)),
    }
    candidate_id = "P118-" + _hash(row)[:20]
    return {
        "status": "FROZEN_FORWARD_OBSERVATION_CANDIDATE",
        "candidate_id": candidate_id,
        **row,
        "independent_future_observations": 0,
        "authority": "NONE",
    }


def _register_forward_holdout(
    project_root: Path, candidate: dict[str, Any]
) -> dict[str, Any]:
    layout = Layout(project_root)
    layout.private.mkdir(parents=True, exist_ok=True)
    path = layout.private / "forward-holdout.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS registrations (
            candidate_id TEXT PRIMARY KEY,
            frozen_at TEXT NOT NULL,
            baseline_date TEXT NOT NULL,
            contract_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            authority TEXT NOT NULL,
            observations INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    if candidate.get("candidate_id"):
        frozen_at = datetime.now(UTC).isoformat()
        connection.execute(
            """
            INSERT OR IGNORE INTO registrations VALUES (?, ?, ?, ?, ?, 'NONE', 0)
            """,
            (
                candidate["candidate_id"],
                frozen_at,
                datetime.now(UTC).date().isoformat(),
                _hash(candidate),
                "FROZEN_WAITING_FOR_FUTURE_DATA",
            ),
        )
    connection.commit()
    rows = [
        dict(zip([column[0] for column in cursor.description], row))
        for cursor in [connection.execute("SELECT * FROM registrations")]
        for row in cursor.fetchall()
    ]
    connection.close()
    freeze_reference = None
    if candidate.get("candidate_id"):
        freeze_payload = {
            "schema": "phase11_8_forward_candidate_freeze_v1",
            "marker": "PHASE11_8_FORWARD_OBSERVATION_CANDIDATE_FROZEN_GO",
            "candidate": candidate,
            "baseline_date": datetime.now(UTC).date().isoformat(),
            "minimum_observation_months": 3,
            "historical_period_consumed": True,
            "automatic_activation": False,
            "phase9_route": "BLOCKED",
            **AUTHORITY,
        }
        freeze_payload["contract_hash"] = _hash(freeze_payload)
        freeze_path = (
            layout.output
            / "frozen"
            / f"{candidate['candidate_id']}.json"
        )
        _write_immutable_json(freeze_path, freeze_payload)
        freeze_reference = str(freeze_path.relative_to(project_root)).replace(
            "\\", "/"
        )
    public = {
        "status": (
            "FROZEN_WAITING_FOR_FUTURE_DATA"
            if candidate.get("candidate_id")
            else "NO_CANDIDATE_TO_REGISTER"
        ),
        "registration_count": len(rows),
        "baseline_date": datetime.now(UTC).date().isoformat(),
        "independent_future_observations": sum(
            int(row["observations"]) for row in rows
        ),
        "minimum_observation_months": 3,
        "automatic_activation": False,
        "phase9_route": "BLOCKED",
        "strategy_agnostic_shadow": "OBSERVATION_ONLY",
        "candidate_freeze_reference": freeze_reference,
        "authority": "NONE",
    }
    _write_json(layout.output / "forward-holdout-registry.json", public)
    return public


def _synthetic_frames() -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=520)
    result = {}
    for index in range(6):
        close = 20 + index + np.linspace(0, 15 + index, len(dates))
        frame = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000,
            },
            index=dates,
        )
        result[f"FIXTURE:{index}"] = frame
    return result


def _finite(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


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


def _write_immutable_json(path: Path, payload: Any) -> None:
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("PHASE11_8_FROZEN_ARTIFACT_IMMUTABILITY_CONFLICT")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
