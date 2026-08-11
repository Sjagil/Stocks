from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import exchange_calendars as xcals


SCHEMA = "phase11_7_financial_finalist_campaign_v1"
DEVELOPMENT_END = pd.Timestamp("2018-12-31")
CONFIRMATION_START = pd.Timestamp("2019-01-01")
CONFIRMATION_END = pd.Timestamp("2025-12-31")
AUTHORITY = {
    "FINANCIAL_FINALIST_GO": False,
    "FORWARD_SHADOW_GO": False,
    "STRATEGY_AUTHORITY": "NONE",
    "EXECUTION_AUTHORITY": "NONE",
    "BROKER_CALLS": 0,
}

VARIANTS: dict[str, tuple[str, dict[str, Any]]] = {
    "MA_50_200": (
        "ma_crossover",
        {"fast": 50, "slow": 200, "max_hold": 2520},
    ),
    "MA_70_230": (
        "ma_crossover",
        {"fast": 70, "slow": 230, "max_hold": 2520},
    ),
    "ASYM_50_200_100_250": (
        "asymmetric_ma_crossover",
        {
            "entry_fast": 50,
            "entry_slow": 200,
            "exit_fast": 100,
            "exit_slow": 250,
            "max_hold": 2520,
        },
    ),
    "ASYM_70_210_110_300": (
        "asymmetric_ma_crossover",
        {
            "entry_fast": 70,
            "entry_slow": 210,
            "exit_fast": 110,
            "exit_slow": 300,
            "max_hold": 2520,
        },
    ),
}

PORTFOLIOS: dict[str, tuple[str, ...]] = {
    "MA_50_200": ("MA_50_200",),
    "MA_70_230": ("MA_70_230",),
    "ASYM_50_200_100_250": ("ASYM_50_200_100_250",),
    "ASYM_70_210_110_300": ("ASYM_70_210_110_300",),
    "PAIR_50_200": ("MA_50_200", "ASYM_50_200_100_250"),
    "PAIR_70_230": ("MA_70_230", "ASYM_70_210_110_300"),
    "PAIR_MA50_ASYM70": ("MA_50_200", "ASYM_70_210_110_300"),
    "PAIR_MA70_ASYM50": ("MA_70_230", "ASYM_50_200_100_250"),
}
GROSS_EXPOSURES = (0.50, 0.75, 1.00)
REGIMES = ("NONE", "BREADTH_50")
ROTATION_LOOKBACKS = (63, 126, 252)
ROTATION_TRENDS = (100, 200)
ROTATION_REBALANCE = ("M", "Q")
ROTATION_TOP_N = (2, 4)


@dataclasses.dataclass(frozen=True)
class Layout:
    project_root: Path

    @property
    def output(self) -> Path:
        return self.project_root / "output" / "research" / "phase11_7"

    @property
    def private(self) -> Path:
        return self.project_root / "data" / "research" / "phase11_7" / "private"

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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest().upper()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _is_common_stock(metadata: Mapping[str, Any]) -> bool:
    category = str(metadata.get("category") or "").upper()
    return "COMMON STOCK" in category and not any(
        blocked in category for blocked in ("WARRANT", "UNIT", "PREFERRED", "FUND", "ETF")
    )


def phase11_7_schema(project_root: Path) -> dict[str, Any]:
    layout = Layout(project_root)
    payload = {
        "schema": SCHEMA,
        "status": "GO",
        "purpose": "bounded financial-finalist search after Phase 11.6",
        "development_period": ["2000-01-01", str(DEVELOPMENT_END.date())],
        "historical_confirmation_period": [
            str(CONFIRMATION_START.date()),
            str(CONFIRMATION_END.date()),
        ],
        "future_holdout": "UNAVAILABLE_UNTIL_FUTURE_FORWARD_DATA",
        "historical_confirmation_already_consumed": True,
        "portfolio_count": len(PORTFOLIOS),
        "gross_exposure_levels": list(GROSS_EXPOSURES),
        "regime_modes": list(REGIMES),
        "effective_trial_count": len(PORTFOLIOS)
        * len(GROSS_EXPOSURES)
        * len(REGIMES),
        "cost_stress_bps_per_side": [5, 10, 20, 30, 50],
        "whole_shares": True,
        "initial_capital_eur": 2000.0,
        "global_max_positions": 4,
        "data_quality_policy": {
            "identity_level_quarantine": True,
            "open_to_previous_close_bounds": [0.25, 4.0],
            "close_to_previous_close_bounds": [0.25, 4.0],
            "point_in_time_liquidity": "20_PRIOR_BARS_MEDIAN_CLOSE_X_VOLUME",
        },
        "financial_finalist_requires_future_holdout": True,
        "financial_finalist_requires_historical_shariah": True,
        **AUTHORITY,
    }
    _write_json(layout.output / "schema.json", payload)
    return payload


def _ledger_config(
    project_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    gross_exposure: float,
    cost_bps: float,
) -> Any:
    import strategy_combo_research_lab as lab

    return lab.V2Config(
        command="run",
        data="",
        output=str(Layout(project_root).output),
        preset="phase11_7",
        policy="shariah",
        start=str(start.date()),
        train_end=str(DEVELOPMENT_END.date()),
        validation_end=str(DEVELOPMENT_END.date()),
        end=str(end.date()),
        initial_capital=2_000.0,
        global_max_positions=4,
        max_security_weight=0.25,
        max_sector_weight=0.50,
        max_gross_exposure=gross_exposure,
        minimum_order_eur=25.0,
        whole_shares=True,
        max_order_adv_fraction=0.01,
        min_price=5.0,
        min_median_dollar_volume=5_000_000.0,
        liquidity_lookback=20,
        allowed_exchanges=("NYSE", "NASDAQ", "NYSEMKT"),
        cost_bps_per_side=cost_bps,
        slippage_bps_per_side=5.0,
        fx_cost_bps_per_side=0.0,
        fixed_fee_eur=3.0,
        min_bars=1,
        min_validation_trades=30,
        validation_max_drawdown=-0.35,
        max_symbols=None,
        corporate_action_gate=True,
        overnight_ratio_min=0.25,
        overnight_ratio_max=4.0,
        batch_size=25,
        checkpoint_every=1,
        workers=1,
        memory_budget_gb=4.0,
        full_cartesian=False,
        max_variants_per_strategy=4,
        combo_sizes=(2,),
        weight_modes=("equal",),
        allow_invalid_strategies_in_combos=False,
        bootstrap_runs=0,
        bootstrap_block_size=20,
        top_equity_curves=3,
        equity_extreme_return_threshold=0.10,
        equity_hard_fail_return_threshold=0.50,
        seed=20260723,
        include_strategies=("ma_crossover", "asymmetric_ma_crossover"),
        exclude_strategies=(),
        resume=False,
    )


def _load_clean_frames(
    project_root: Path, max_identities: int
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, dict[str, Any]],
    pd.DatetimeIndex,
    pd.Series,
    dict[str, Any],
]:
    import strategy_combo_research_lab as lab

    layout = Layout(project_root)
    layout.private.mkdir(parents=True, exist_ok=True)
    source = lab.MarketDataSource(
        layout.pit_bars,
        pd.Timestamp("2000-01-01"),
        pd.Timestamp(dt.date.today()),
        1,
        None,
        20260723,
        layout.private / "campaign.duckdb",
        corporate_action_gate=True,
        overnight_ratio_min=0.25,
        overnight_ratio_max=4.0,
    )
    try:
        metadata, metadata_audit = lab.load_v2_security_metadata(project_root)
        identities = source.identities.copy()
        identities = identities.loc[
            identities["security_id"].map(
                lambda security_id: _is_common_stock(metadata.get(str(security_id), {}))
                and str(metadata.get(str(security_id), {}).get("exchange") or "").upper()
                in {"NYSE", "NASDAQ", "NYSEMKT"}
            )
        ].copy()
        identities["sample_key"] = identities["security_id"].map(
            lambda value: _hash({"seed": 20260723, "security_id": str(value)})
        )
        selected = (
            identities.sort_values(["sample_key", "security_id"])
            .head(max_identities)["security_id"]
            .astype(str)
            .tolist()
        )
        source.con.register(
            "_phase11_7_selected", pd.DataFrame({"security_id": selected})
        )
        try:
            bars = source.con.execute(
                """
                SELECT b.*
                FROM market_bars b
                INNER JOIN _phase11_7_selected s USING(security_id)
                ORDER BY b.security_id, b.date
                """
            ).fetchdf()
        finally:
            source.con.unregister("_phase11_7_selected")
        exclusions = source.quality_exclusions()
        source_metadata = source.metadata()
    finally:
        source.con.close()

    bars["date"] = pd.to_datetime(bars["date"])
    frames = {
        str(security_id): group.sort_values("date").reset_index(drop=True)
        for security_id, group in bars.groupby("security_id", sort=False)
    }
    raw_start = pd.Timestamp(bars["date"].min()).normalize()
    raw_end = pd.Timestamp(bars["date"].max()).normalize()
    calendar = (
        xcals.get_calendar("XNYS", start=raw_start, end=raw_end)
        .sessions_in_range(raw_start, raw_end)
        .tz_localize(None)
        .normalize()
    )
    fx = lab.load_v2_fx(project_root, calendar)
    if fx.isna().any():
        raise RuntimeError("HISTORICAL_USD_EUR_NORMALIZATION_MISSING")

    _write_frame(layout.private / "data-quality-exclusions.parquet", exclusions)
    selected_rows = int(sum(len(frame) for frame in frames.values()))
    audit = {
        "schema": "phase11_7_data_quality_audit_v1",
        "status": "GO",
        "raw_identity_count": source_metadata["raw_identity_count"],
        "quarantined_identity_count": source_metadata[
            "quality_excluded_identity_count"
        ],
        "clean_identity_population": int(len(identities)),
        "selected_identity_count": len(frames),
        "selected_bar_count": selected_rows,
        "minimum_selected_bar_count": int(min(map(len, frames.values()))) if frames else 0,
        "maximum_selected_bar_count": int(max(map(len, frames.values()))) if frames else 0,
        "quarantine_bounds": [0.25, 4.0],
        "maximum_selected_absolute_close_return": float(
            max(
                (
                    frame["close"].pct_change().abs().max()
                    for frame in frames.values()
                ),
                default=0.0,
            )
        ),
        "metadata_status": metadata_audit,
        "selection_rule": "DETERMINISTIC_HASH_WITHOUT_OUTCOME_RANKING",
        "minimum_history_selection": False,
        "calendar": "XNYS_OFFICIAL_SESSIONS",
        "private_exclusion_path": str(
            layout.private / "data-quality-exclusions.parquet"
        ),
        **AUTHORITY,
    }
    _write_json(layout.output / "data-quality-audit.json", audit)
    return frames, metadata, calendar, fx, audit


def _states() -> tuple[dict[str, Any], dict[str, str]]:
    import strategy_combo_research_lab as lab

    specs = {spec.name: spec for spec in lab.strategy_registry()}
    states: dict[str, Any] = {}
    variant_ids: dict[str, str] = {}
    for label, (strategy, parameters) in VARIANTS.items():
        variant_id = lab.make_variant_id(strategy, parameters)
        states[label] = lab.VariantState(
            strategy=strategy,
            family=specs[strategy].family,
            horizon=specs[strategy].horizon,
            params=dict(parameters),
            variant_id=variant_id,
        )
        variant_ids[label] = variant_id
    return states, variant_ids


def _candidate_table(
    frames: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, Any]],
    config: Any,
) -> tuple[pd.DataFrame, dict[str, str]]:
    import strategy_combo_research_lab as lab

    specs = {spec.name: spec for spec in lab.strategy_registry()}
    states, variant_ids = _states()
    rows: list[dict[str, Any]] = []
    for security_id, frame in frames.items():
        identity = metadata.get(security_id, {})
        for state in states.values():
            rows.extend(
                lab.v2_candidate_records(
                    frame,
                    security_id,
                    identity,
                    specs,
                    (state,),
                    config,
                )
            )
    candidates = pd.DataFrame(rows)
    return candidates, variant_ids


def _portfolio_input(
    candidates: pd.DataFrame,
    labels: tuple[str, ...],
    variant_ids: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    chosen_ids = {variant_ids[label] for label in labels}
    selected = candidates.loc[candidates["variant_id"].isin(chosen_ids)].copy()
    selected["strategy"] = selected["variant_id"]
    weight = 1.0 / len(chosen_ids)
    return selected, {variant_id: weight for variant_id in chosen_ids}


def _breadth_execution_regime(
    frames: Mapping[str, pd.DataFrame],
    calendar: pd.DatetimeIndex,
    *,
    threshold: float = 0.50,
    minimum_constituents: int = 20,
) -> pd.Series:
    states = []
    for security_id, frame in frames.items():
        work = frame.set_index("date").sort_index()
        close = work["close"].astype(float)
        state = close.gt(close.rolling(200, min_periods=200).mean())
        states.append(state.rename(security_id))
    matrix = pd.concat(states, axis=1).reindex(calendar)
    constituent_count = matrix.notna().sum(axis=1)
    breadth = matrix.mean(axis=1, skipna=True)
    close_signal = breadth.gt(threshold) & constituent_count.ge(minimum_constituents)
    # A close-derived regime can only control the following session's open.
    return close_signal.shift(1, fill_value=False).astype(bool)


def _apply_execution_regime(
    candidates: pd.DataFrame,
    execution_regime: pd.Series,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    calendar = pd.DatetimeIndex(execution_regime.index)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        entry = pd.Timestamp(candidate["entry_date"]).normalize()
        exit_ = pd.Timestamp(candidate["exit_date"]).normalize()
        active_dates = calendar[(calendar >= entry) & (calendar < exit_)]
        if active_dates.empty:
            continue
        active = execution_regime.reindex(active_dates).fillna(False).astype(bool)
        groups = active.ne(active.shift(fill_value=False)).cumsum()
        for _, values in active.groupby(groups):
            if not bool(values.iloc[0]):
                continue
            run_start = pd.Timestamp(values.index[0])
            last_true = pd.Timestamp(values.index[-1])
            location = int(calendar.searchsorted(last_true, side="right"))
            run_exit = exit_ if location >= len(calendar) else min(
                exit_, pd.Timestamp(calendar[location])
            )
            if run_start >= run_exit:
                continue
            row = dict(candidate)
            row["entry_date"] = run_start
            row["exit_date"] = run_exit
            rows.append(row)
    return pd.DataFrame(rows, columns=candidates.columns)


def _period_metrics(
    ledger: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any]:
    import strategy_combo_research_lab as lab

    returns = ledger.set_index("date")["daily_return"].astype(float)
    returns = returns.loc[(returns.index >= start) & (returns.index <= end)]
    return lab.daily_metrics(returns, 2_000.0)


def _closed_episodes(fills: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame(columns=["security_id", "entry_date", "exit_date", "net_pnl"])
    rows: list[dict[str, Any]] = []
    for security_id, group in fills.sort_values(["date", "side"]).groupby(
        "security_id", sort=False
    ):
        shares = 0
        entry_date: pd.Timestamp | None = None
        cash_flow = 0.0
        for fill in group.to_dict("records"):
            quantity = int(fill["shares"])
            notional = float(fill["notional_eur"])
            fee = float(fill["fee_eur"])
            date = pd.Timestamp(fill["date"])
            if fill["side"] == "BUY":
                if shares == 0:
                    entry_date = date
                    cash_flow = 0.0
                shares += quantity
                cash_flow -= notional + fee
            else:
                shares -= quantity
                cash_flow += notional - fee
                if shares == 0 and entry_date is not None:
                    rows.append(
                        {
                            "security_id": security_id,
                            "entry_date": entry_date,
                            "exit_date": date,
                            "net_pnl": cash_flow,
                        }
                    )
                    entry_date = None
                    cash_flow = 0.0
            if shares < 0:
                raise RuntimeError("NEGATIVE_EPISODE_SHARES")
    return pd.DataFrame(rows)


def _episode_metrics(episodes: pd.DataFrame) -> dict[str, Any]:
    if episodes.empty:
        return {
            "episode_count": 0,
            "positive_episodes": 0,
            "negative_episodes": 0,
            "episode_profit_factor": None,
            "sample_status": "INSUFFICIENT_SAMPLE",
        }
    pnl = episodes["net_pnl"].astype(float)
    gains = float(pnl.loc[pnl > 0].sum())
    losses = abs(float(pnl.loc[pnl < 0].sum()))
    profit_factor: float | None
    if losses > 0:
        profit_factor = gains / losses
    elif gains > 0:
        profit_factor = float("inf")
    else:
        profit_factor = None
    return {
        "episode_count": len(episodes),
        "positive_episodes": int(pnl.gt(0).sum()),
        "negative_episodes": int(pnl.lt(0).sum()),
        "episode_profit_factor": profit_factor,
        "sample_status": (
            "EVALUABLE"
            if len(episodes) >= 30
            else "LOW_CONFIDENCE"
            if len(episodes) >= 10
            else "INSUFFICIENT_SAMPLE"
        ),
    }


def _pbo(returns_by_trial: Mapping[str, pd.Series]) -> dict[str, Any]:
    names = sorted(returns_by_trial)
    if len(names) < 2:
        return {"status": "INSUFFICIENT_TRIALS", "PBO": None, "splits": 0}
    matrix = pd.concat(
        [returns_by_trial[name].rename(name) for name in names], axis=1
    ).fillna(0.0)
    blocks = [index for index in np.array_split(np.arange(len(matrix)), 8) if len(index)]
    under_median = 0
    evaluated = 0

    def sharpe(values: pd.Series) -> float:
        std = float(values.std(ddof=1))
        return float(values.mean() / std) if std > 0 else -math.inf

    for chosen in itertools.combinations(range(len(blocks)), len(blocks) // 2):
        if 0 not in chosen:
            continue
        in_locations = np.concatenate([blocks[index] for index in chosen])
        out_locations = np.concatenate(
            [blocks[index] for index in range(len(blocks)) if index not in chosen]
        )
        in_scores = matrix.iloc[in_locations].apply(sharpe)
        winner = str(in_scores.idxmax())
        out_scores = matrix.iloc[out_locations].apply(sharpe).rank(pct=True)
        winner_rank = float(out_scores[winner])
        under_median += int(winner_rank <= 0.5)
        evaluated += 1
    return {
        "status": "GO" if evaluated else "INSUFFICIENT_SPLITS",
        "PBO": under_median / evaluated if evaluated else None,
        "splits": evaluated,
        "block_count": len(blocks),
        "trial_count": len(names),
    }


def _equal_weight_benchmark(
    frames: Mapping[str, pd.DataFrame], fx: pd.Series
) -> pd.Series:
    parts = []
    fx_return = fx.pct_change().fillna(0.0)
    for security_id, frame in frames.items():
        work = frame.set_index("date").sort_index()
        local = work["close"].astype(float).pct_change()
        eur = (1.0 + local) * (1.0 + fx_return.reindex(work.index).fillna(0.0)) - 1.0
        parts.append(eur.rename(security_id))
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True).fillna(0.0)


def _rotation_candidates(
    frames: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Mapping[str, Any]],
    calendar: pd.DatetimeIndex,
    *,
    lookback: int,
    trend_period: int,
    rebalance: str,
) -> pd.DataFrame:
    if rebalance not in {"M", "Q"}:
        raise ValueError("ROTATION_REBALANCE_UNREGISTERED")
    strategy = f"ROT_{lookback}_{trend_period}_{rebalance}"
    decision_dates = (
        pd.Series(calendar, index=calendar)
        .groupby(calendar.to_period(rebalance))
        .max()
        .tolist()
    )
    features: dict[str, pd.DataFrame] = {}
    for security_id, frame in frames.items():
        work = frame.sort_values("date").set_index("date")
        close = work["close"].astype(float)
        momentum = close.shift(21) / close.shift(lookback) - 1.0
        volatility = (
            close.pct_change(fill_method=None)
            .rolling(63, min_periods=63)
            .std(ddof=1)
        )
        features[security_id] = pd.DataFrame(
            {
                "momentum": momentum,
                "trend": close.gt(
                    close.rolling(
                        trend_period, min_periods=trend_period
                    ).mean()
                ),
                "score": momentum / volatility.replace(0.0, np.nan),
                "median_dollar_volume": (
                    close * work["volume"].astype(float)
                ).rolling(20, min_periods=20).median(),
            },
            index=work.index,
        )
    rows: list[dict[str, Any]] = []
    for index, decision_date in enumerate(decision_dates[:-1]):
        entry_location = int(calendar.searchsorted(decision_date, side="right"))
        exit_location = int(
            calendar.searchsorted(decision_dates[index + 1], side="right")
        )
        if entry_location >= len(calendar) or exit_location >= len(calendar):
            continue
        entry_date = pd.Timestamp(calendar[entry_location])
        planned_exit = pd.Timestamp(calendar[exit_location])
        for security_id, feature in features.items():
            if entry_date not in feature.index:
                continue
            available_sessions = feature.index.intersection(calendar)
            if available_sessions.empty:
                continue
            decision_location = int(
                feature.index.searchsorted(decision_date, side="right") - 1
            )
            if decision_location < 0:
                continue
            values = feature.iloc[decision_location]
            terminal_date = pd.Timestamp(available_sessions.max())
            exit_date = min(planned_exit, terminal_date)
            if entry_date >= exit_date:
                continue
            if not (
                pd.notna(values["score"])
                and float(values["momentum"]) > 0.0
                and bool(values["trend"])
                and pd.notna(values["median_dollar_volume"])
                and float(values["median_dollar_volume"]) >= 5_000_000.0
            ):
                continue
            identity = metadata.get(security_id, {})
            rows.append(
                {
                    "strategy": strategy,
                    "variant_id": strategy,
                    "family": "trend_following",
                    "security_id": security_id,
                    "symbol": identity.get("ticker", security_id),
                    "sector": identity.get("sector", "UNKNOWN"),
                    "currency": identity.get("currency", "USD"),
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "score": float(values["score"]),
                    "median_dollar_volume": float(
                        values["median_dollar_volume"]
                    ),
                    "investability_status": "INVESTABLE_GO",
                    "terminal_exit": exit_date < planned_exit,
                }
            )
    return pd.DataFrame(rows)


def _concentration(result: Any, confirmation: tuple[pd.Timestamp, pd.Timestamp]) -> dict[str, Any]:
    contributors = result.contributors.copy()
    contributors["date"] = pd.to_datetime(contributors["date"])
    contributors = contributors.loc[
        contributors["date"].between(confirmation[0], confirmation[1])
    ]
    by_security = contributors.groupby("security_id")["total_security_pnl_eur"].sum()
    positive = by_security.loc[by_security > 0]
    single_security_share = (
        float(positive.max() / positive.sum()) if not positive.empty else 1.0
    )
    ledger = result.ledger.copy()
    ledger["date"] = pd.to_datetime(ledger["date"])
    ledger = ledger.loc[ledger["date"].between(confirmation[0], confirmation[1])]
    yearly = (
        ledger.set_index("date")["daily_return"]
        .groupby(lambda value: value.year)
        .apply(lambda values: (1.0 + values).prod() - 1.0)
    )
    positive_years = yearly.loc[yearly > 0]
    single_year_share = (
        float(positive_years.max() / positive_years.sum())
        if not positive_years.empty
        else 1.0
    )
    return {
        "single_security_positive_contribution_share": single_security_share,
        "single_year_positive_return_share": single_year_share,
        "security_contribution_gate": single_security_share <= 0.40,
        "year_contribution_gate": single_year_share <= 0.50,
    }


def run_finalist_campaign(
    project_root: Path, *, max_identities: int = 500, bootstrap_runs: int = 5000
) -> dict[str, Any]:
    import strategy_combo_research_lab as lab

    if not 50 <= max_identities <= 1000:
        raise ValueError("MAX_IDENTITIES_OUT_OF_BOUNDS")
    if not 100 <= bootstrap_runs <= 20_000:
        raise ValueError("BOOTSTRAP_RUNS_OUT_OF_BOUNDS")
    layout = Layout(project_root)
    phase11_7_schema(project_root)
    frames, metadata, calendar, fx, data_audit = _load_clean_frames(
        project_root, max_identities
    )
    base_config = _ledger_config(
        project_root,
        calendar.min(),
        calendar.max(),
        gross_exposure=1.0,
        cost_bps=10.0,
    )
    candidates, variant_ids = _candidate_table(frames, metadata, base_config)
    _write_frame(layout.private / "candidate-episodes.parquet", candidates)
    candidate_audit = {
        "schema": "phase11_7_candidate_audit_v1",
        "candidate_count": len(candidates),
        "investable_candidate_count": int(
            candidates["investability_status"].eq("INVESTABLE_GO").sum()
        ),
        "candidate_security_count": int(candidates["security_id"].nunique()),
        "variant_count": int(candidates["variant_id"].nunique()),
        "point_in_time_liquidity": True,
        "causal_entry_score": True,
        "next_open_execution": True,
        "constant_score_candidates": int(
            candidates.groupby("variant_id")["score"].nunique().eq(1).sum()
        ),
        **AUTHORITY,
    }
    _write_json(layout.output / "candidate-audit.json", candidate_audit)
    breadth_regime = _breadth_execution_regime(frames, calendar)
    regime_candidates = {
        "NONE": candidates,
        "BREADTH_50": _apply_execution_regime(candidates, breadth_regime),
    }
    _write_json(
        layout.output / "regime-audit.json",
        {
            "schema": "phase11_7_regime_audit_v1",
            "decision_source": "CLOSE",
            "execution_lag": "NEXT_SESSION_OPEN",
            "threshold": 0.50,
            "minimum_constituents": 20,
            "active_session_ratio": float(breadth_regime.mean()),
            "candidate_counts": {
                name: len(frame) for name, frame in regime_candidates.items()
            },
            **AUTHORITY,
        },
    )

    trial_rows: list[dict[str, Any]] = []
    confirmation_returns: dict[str, pd.Series] = {}
    for regime_name, regime_table in regime_candidates.items():
        for portfolio_name, labels in PORTFOLIOS.items():
            portfolio_candidates, weights = _portfolio_input(
                regime_table, labels, variant_ids
            )
            for gross_exposure in GROSS_EXPOSURES:
                trial_id = (
                    f"{portfolio_name}__{regime_name}__GROSS_{gross_exposure:.2f}"
                )
                config = dataclasses.replace(
                    base_config, max_gross_exposure=gross_exposure
                )
                result = lab.run_global_ledger(
                    portfolio_candidates,
                    frames,
                    calendar,
                    fx,
                    config,
                    weights,
                    portfolio_name=trial_id,
                )
                development = _period_metrics(
                    result.ledger, calendar.min(), DEVELOPMENT_END
                )
                confirmation = _period_metrics(
                    result.ledger, CONFIRMATION_START, CONFIRMATION_END
                )
                returns = result.ledger.set_index("date")["daily_return"].astype(
                    float
                )
                confirmation_returns[trial_id] = returns.loc[
                    (returns.index >= CONFIRMATION_START)
                    & (returns.index <= CONFIRMATION_END)
                ]
                trial_rows.append(
                    {
                        "trial_id": trial_id,
                        "portfolio": portfolio_name,
                        "regime": regime_name,
                        "components": "+".join(labels),
                        "gross_exposure": gross_exposure,
                        "development_CAGR": development["CAGR"],
                        "development_Sharpe": development["Sharpe"],
                        "development_maximum_drawdown": development[
                            "maximum_drawdown"
                        ],
                        "development_period_profit_factor": development[
                            "daily_profit_factor"
                        ],
                        "confirmation_CAGR": confirmation["CAGR"],
                        "confirmation_Sharpe": confirmation["Sharpe"],
                        "confirmation_maximum_drawdown": confirmation[
                            "maximum_drawdown"
                        ],
                        "confirmation_period_profit_factor": confirmation[
                            "daily_profit_factor"
                        ],
                        "trade_count": result.metrics["trade_count"],
                        "accounting_failures": result.accounting_failures,
                        "unexplained_outliers": result.unexplained_outliers,
                    }
                )
    trials = pd.DataFrame(trial_rows)
    trials["development_gate"] = (
        trials["development_CAGR"].gt(0)
        & trials["development_Sharpe"].gt(0)
        & trials["development_maximum_drawdown"].gt(-0.35)
        & trials["accounting_failures"].eq(0)
        & trials["unexplained_outliers"].eq(0)
    )
    eligible = trials.loc[trials["development_gate"]]
    selection_pool = eligible if not eligible.empty else trials
    selected = selection_pool.sort_values(
        [
            "development_Sharpe",
            "development_CAGR",
            "development_maximum_drawdown",
            "trial_id",
        ],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected_id = str(selected["trial_id"])
    selected_portfolio = str(selected["portfolio"])
    selected_labels = PORTFOLIOS[selected_portfolio]
    selected_regime = str(selected["regime"])
    selected_candidates, selected_weights = _portfolio_input(
        regime_candidates[selected_regime], selected_labels, variant_ids
    )
    selected_config = dataclasses.replace(
        base_config, max_gross_exposure=float(selected["gross_exposure"])
    )
    selected_result = lab.run_global_ledger(
        selected_candidates,
        frames,
        calendar,
        fx,
        selected_config,
        selected_weights,
        portfolio_name=selected_id,
    )
    _write_frame(layout.output / "trial-results.csv", trials)
    _write_frame(layout.private / "selected-ledger.parquet", selected_result.ledger)
    _write_frame(layout.private / "selected-orders.parquet", selected_result.orders)
    _write_frame(layout.private / "selected-fills.parquet", selected_result.fills)
    _write_frame(
        layout.private / "selected-contributors.parquet",
        selected_result.contributors,
    )

    episodes = _closed_episodes(selected_result.fills)
    _write_frame(layout.output / "selected-closed-episodes.csv", episodes)
    confirmation_episodes = episodes.loc[
        pd.to_datetime(episodes["exit_date"]).between(
            CONFIRMATION_START, CONFIRMATION_END
        )
    ]
    aggregate_episode_metrics = _episode_metrics(confirmation_episodes)
    fold_rows: list[dict[str, Any]] = []
    for year in range(2019, 2026):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")
        period = _period_metrics(selected_result.ledger, year_start, year_end)
        fold_episode_metrics = _episode_metrics(
            confirmation_episodes.loc[
                pd.to_datetime(confirmation_episodes["exit_date"]).between(
                    year_start, year_end
                )
            ]
        )
        fold_rows.append(
            {
                "fold_id": f"HISTORICAL_CONFIRMATION_{year}",
                "start": year_start,
                "end": year_end,
                "CAGR": period["CAGR"],
                "Sharpe": period["Sharpe"],
                "maximum_drawdown": period["maximum_drawdown"],
                "period_profit_factor": period["daily_profit_factor"],
                **fold_episode_metrics,
            }
        )
    folds = pd.DataFrame(fold_rows)
    _write_frame(layout.output / "historical-confirmation-folds.csv", folds)

    stress_rows: list[dict[str, Any]] = []
    stress_results: dict[int, Any] = {}
    for cost_bps in (5, 10, 20, 30, 50):
        stress_config = dataclasses.replace(
            selected_config, cost_bps_per_side=float(cost_bps)
        )
        result = lab.run_global_ledger(
            selected_candidates,
            frames,
            calendar,
            fx,
            stress_config,
            selected_weights,
            portfolio_name=f"{selected_id}__COST_{cost_bps}",
        )
        stress_results[cost_bps] = result
        metrics = _period_metrics(
            result.ledger, CONFIRMATION_START, CONFIRMATION_END
        )
        stress_episodes = _closed_episodes(result.fills)
        stress_episodes = stress_episodes.loc[
            pd.to_datetime(stress_episodes["exit_date"]).between(
                CONFIRMATION_START, CONFIRMATION_END
            )
        ]
        stress_rows.append(
            {
                "cost_bps_per_side": cost_bps,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "period_profit_factor": metrics["daily_profit_factor"],
                **_episode_metrics(stress_episodes),
            }
        )
    stress = pd.DataFrame(stress_rows)
    _write_frame(layout.output / "cost-stress.csv", stress)

    benchmark_returns = _equal_weight_benchmark(frames, fx)
    benchmark_confirmation = benchmark_returns.loc[
        (benchmark_returns.index >= CONFIRMATION_START)
        & (benchmark_returns.index <= CONFIRMATION_END)
    ]
    benchmark_metrics = lab.daily_metrics(benchmark_confirmation, 2_000.0)
    selected_confirmation = confirmation_returns[selected_id]
    selected_confirmation_metrics = lab.daily_metrics(
        selected_confirmation, 2_000.0
    )
    excess = selected_confirmation.reindex(
        selected_confirmation.index.union(benchmark_confirmation.index)
    ).fillna(0.0) - benchmark_confirmation.reindex(
        selected_confirmation.index.union(benchmark_confirmation.index)
    ).fillna(0.0)
    excess_std = float(excess.std(ddof=1))
    information_ratio = (
        float(excess.mean() / excess_std * math.sqrt(252.0))
        if excess_std > 0
        else None
    )
    benchmark_report = {
        "schema": "phase11_7_benchmark_comparison_v1",
        "benchmark": "DYNAMIC_EQUAL_WEIGHT_CLEAN_SAMPLE_NO_COST",
        "benchmark_metrics": benchmark_metrics,
        "selected_metrics": selected_confirmation_metrics,
        "excess_CAGR": selected_confirmation_metrics["CAGR"]
        - benchmark_metrics["CAGR"],
        "excess_Sharpe": selected_confirmation_metrics["Sharpe"]
        - benchmark_metrics["Sharpe"],
        "information_ratio": information_ratio,
        "benchmark_is_not_whole_share_executable": True,
        **AUTHORITY,
    }
    _write_json(layout.output / "benchmark-comparison.json", benchmark_report)

    confirmation_sharpes = trials["confirmation_Sharpe"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    selected_series = selected_confirmation
    dsr = lab.deflated_sharpe_probability(
        float(selected_confirmation_metrics["Sharpe"]),
        len(selected_series),
        len(trials),
        float(confirmation_sharpes.std(ddof=1)),
        float(selected_series.skew()),
        float(selected_series.kurt() + 3.0),
    )
    pbo = _pbo(confirmation_returns)
    bootstrap = lab.block_bootstrap(
        selected_confirmation, bootstrap_runs, 20, 20260723
    )
    multiple_testing = {
        "schema": "phase11_7_multiple_testing_v1",
        "effective_trial_count": len(trials),
        "selected_confirmation_DSR_probability": dsr,
        "PBO": pbo,
        "bootstrap": bootstrap,
        **AUTHORITY,
    }
    _write_json(layout.output / "multiple-testing.json", multiple_testing)

    concentration = _concentration(
        selected_result, (CONFIRMATION_START, CONFIRMATION_END)
    )
    _write_json(
        layout.output / "concentration-audit.json",
        {"schema": "phase11_7_concentration_v1", **concentration, **AUTHORITY},
    )

    evaluable_folds = folds.loc[folds["sample_status"].ne("INSUFFICIENT_SAMPLE")]
    positive_folds = int(folds["CAGR"].gt(0).sum())
    median_fold_pf = (
        float(evaluable_folds["episode_profit_factor"].dropna().median())
        if not evaluable_folds.empty
        else None
    )
    worst_fold_pf = (
        float(evaluable_folds["episode_profit_factor"].dropna().min())
        if not evaluable_folds.empty
        else None
    )
    cost_20 = stress.loc[stress["cost_bps_per_side"].eq(20)].iloc[0]
    technical_gates = {
        "positive_confirmation_folds": positive_folds >= 5,
        "median_fold_episode_pf": median_fold_pf is not None
        and median_fold_pf > 1.10,
        "worst_evaluable_fold_episode_pf": worst_fold_pf is not None
        and worst_fold_pf > 0.80,
        "aggregate_episode_pf": (
            aggregate_episode_metrics["episode_profit_factor"] is not None
            and aggregate_episode_metrics["episode_profit_factor"] > 1.10
        ),
        "cost_20bps_episode_pf": (
            pd.notna(cost_20["episode_profit_factor"])
            and float(cost_20["episode_profit_factor"]) > 1.0
        ),
        "benchmark_excess_cagr": benchmark_report["excess_CAGR"] > 0,
        "drawdown_budget": selected_confirmation_metrics["maximum_drawdown"] > -0.35,
        "single_security_concentration": concentration[
            "security_contribution_gate"
        ],
        "single_year_concentration": concentration["year_contribution_gate"],
        "DSR": dsr >= 0.80,
        "PBO": pbo.get("PBO") is not None and float(pbo["PBO"]) <= 0.40,
        "sample_size": aggregate_episode_metrics["sample_status"] == "EVALUABLE",
        "accounting": selected_result.accounting_failures == 0,
        "data_quality": data_audit["status"] == "GO",
    }
    historical_candidate_go = all(technical_gates.values())
    blockers = [
        name for name, passed in technical_gates.items() if not bool(passed)
    ]
    blockers.extend(
        [
            "FUTURE_FORWARD_HOLDOUT_UNAVAILABLE",
            "HISTORICAL_SHARIAH_RECONSTRUCTION_UNAVAILABLE",
        ]
    )
    decision = (
        "PROMISING_RESEARCH_CANDIDATE"
        if historical_candidate_go
        else "NO_FINANCIAL_FINALIST"
    )
    report = {
        "schema": SCHEMA,
        "status": "GO",
        "decision": decision,
        "selected_trial": selected_id,
        "selected_regime": selected_regime,
        "selection_source": "DEVELOPMENT_2000_2018_ONLY",
        "historical_confirmation_consumed": True,
        "selected_confirmation_metrics": selected_confirmation_metrics,
        "episode_metrics": aggregate_episode_metrics,
        "fold_count": len(folds),
        "positive_fold_count": positive_folds,
        "median_fold_episode_profit_factor": median_fold_pf,
        "worst_evaluable_fold_episode_profit_factor": worst_fold_pf,
        "technical_gates": technical_gates,
        "historical_candidate_go": historical_candidate_go,
        "financial_finalist_go": False,
        "blockers": blockers,
        "future_holdout": "UNAVAILABLE_UNTIL_FUTURE_FORWARD_DATA",
        "historical_shariah_validation": "UNAVAILABLE",
        "data_quality": data_audit,
        "broker_calls": 0,
        **AUTHORITY,
    }
    _write_json(layout.output / "decision.json", report)
    _write_json(
        layout.output / "manifest.json",
        {
            "schema": "phase11_7_manifest_v1",
            "generated_at": dt.datetime.now(dt.UTC).isoformat(),
            "artifacts": sorted(
                str(path.relative_to(layout.output))
                for path in layout.output.rglob("*")
                if path.is_file()
            ),
            "private_artifacts": sorted(
                str(path.relative_to(layout.private))
                for path in layout.private.rglob("*")
                if path.is_file()
            ),
            **AUTHORITY,
        },
    )
    return report


def run_rotation_campaign(
    project_root: Path, *, max_identities: int = 500, bootstrap_runs: int = 5000
) -> dict[str, Any]:
    import strategy_combo_research_lab as lab

    if not 50 <= max_identities <= 1000:
        raise ValueError("MAX_IDENTITIES_OUT_OF_BOUNDS")
    if not 100 <= bootstrap_runs <= 20_000:
        raise ValueError("BOOTSTRAP_RUNS_OUT_OF_BOUNDS")
    layout = Layout(project_root)
    output = layout.output / "rotation"
    frames, metadata, calendar, fx, data_audit = _load_clean_frames(
        project_root, max_identities
    )
    base_config = _ledger_config(
        project_root,
        calendar.min(),
        calendar.max(),
        gross_exposure=1.0,
        cost_bps=10.0,
    )
    candidate_tables: dict[tuple[int, int, str], pd.DataFrame] = {}
    for lookback in ROTATION_LOOKBACKS:
        for trend_period in ROTATION_TRENDS:
            for rebalance in ROTATION_REBALANCE:
                key = (lookback, trend_period, rebalance)
                candidate_tables[key] = _rotation_candidates(
                    frames,
                    metadata,
                    calendar,
                    lookback=lookback,
                    trend_period=trend_period,
                    rebalance=rebalance,
                )
    candidate_manifest = pd.DataFrame(
        [
            {
                "lookback": key[0],
                "trend_period": key[1],
                "rebalance": key[2],
                "candidate_count": len(frame),
                "security_count": int(frame["security_id"].nunique()),
                "terminal_exit_count": int(frame["terminal_exit"].sum()),
            }
            for key, frame in candidate_tables.items()
        ]
    )
    _write_frame(output / "candidate-manifest.csv", candidate_manifest)

    rows: list[dict[str, Any]] = []
    confirmation_returns: dict[str, pd.Series] = {}
    for (lookback, trend_period, rebalance), candidates in candidate_tables.items():
        strategy = f"ROT_{lookback}_{trend_period}_{rebalance}"
        for top_n in ROTATION_TOP_N:
            for gross_exposure in GROSS_EXPOSURES:
                trial_id = (
                    f"{strategy}__TOP_{top_n}__GROSS_{gross_exposure:.2f}"
                )
                config = dataclasses.replace(
                    base_config,
                    global_max_positions=top_n,
                    max_security_weight=1.0 / top_n,
                    max_gross_exposure=gross_exposure,
                )
                result = lab.run_global_ledger(
                    candidates,
                    frames,
                    calendar,
                    fx,
                    config,
                    {strategy: 1.0},
                    portfolio_name=trial_id,
                )
                development = _period_metrics(
                    result.ledger, calendar.min(), DEVELOPMENT_END
                )
                confirmation = _period_metrics(
                    result.ledger, CONFIRMATION_START, CONFIRMATION_END
                )
                returns = result.ledger.set_index("date")["daily_return"].astype(
                    float
                )
                confirmation_returns[trial_id] = returns.loc[
                    (returns.index >= CONFIRMATION_START)
                    & (returns.index <= CONFIRMATION_END)
                ]
                rows.append(
                    {
                        "trial_id": trial_id,
                        "lookback": lookback,
                        "trend_period": trend_period,
                        "rebalance": rebalance,
                        "top_n": top_n,
                        "gross_exposure": gross_exposure,
                        "development_CAGR": development["CAGR"],
                        "development_Sharpe": development["Sharpe"],
                        "development_maximum_drawdown": development[
                            "maximum_drawdown"
                        ],
                        "development_period_profit_factor": development[
                            "daily_profit_factor"
                        ],
                        "confirmation_CAGR": confirmation["CAGR"],
                        "confirmation_Sharpe": confirmation["Sharpe"],
                        "confirmation_maximum_drawdown": confirmation[
                            "maximum_drawdown"
                        ],
                        "confirmation_period_profit_factor": confirmation[
                            "daily_profit_factor"
                        ],
                        "trade_count": result.metrics["trade_count"],
                        "stale_mark_days": int(
                            result.ledger["stale_mark_count"].gt(0).sum()
                        ),
                        "accounting_failures": result.accounting_failures,
                        "unexplained_outliers": result.unexplained_outliers,
                    }
                )
    trials = pd.DataFrame(rows)
    trials["development_gate"] = (
        trials["development_CAGR"].gt(0)
        & trials["development_Sharpe"].gt(0)
        & trials["development_maximum_drawdown"].gt(-0.35)
        & trials["accounting_failures"].eq(0)
        & trials["unexplained_outliers"].eq(0)
    )
    eligible = trials.loc[trials["development_gate"]]
    selection_pool = eligible if not eligible.empty else trials
    selected = selection_pool.sort_values(
        [
            "development_Sharpe",
            "development_CAGR",
            "development_maximum_drawdown",
            "trial_id",
        ],
        ascending=[False, False, False, True],
    ).iloc[0]
    _write_frame(output / "trial-results.csv", trials)

    selected_key = (
        int(selected["lookback"]),
        int(selected["trend_period"]),
        str(selected["rebalance"]),
    )
    selected_candidates = candidate_tables[selected_key]
    selected_strategy = (
        f"ROT_{selected_key[0]}_{selected_key[1]}_{selected_key[2]}"
    )
    selected_config = dataclasses.replace(
        base_config,
        global_max_positions=int(selected["top_n"]),
        max_security_weight=1.0 / int(selected["top_n"]),
        max_gross_exposure=float(selected["gross_exposure"]),
    )
    selected_result = lab.run_global_ledger(
        selected_candidates,
        frames,
        calendar,
        fx,
        selected_config,
        {selected_strategy: 1.0},
        portfolio_name=str(selected["trial_id"]),
    )
    _write_frame(output / "selected-ledger.parquet", selected_result.ledger)
    _write_frame(output / "selected-orders.parquet", selected_result.orders)
    _write_frame(output / "selected-fills.parquet", selected_result.fills)
    _write_frame(
        output / "selected-contributors.parquet", selected_result.contributors
    )

    episodes = _closed_episodes(selected_result.fills)
    confirmation_episodes = episodes.loc[
        pd.to_datetime(episodes["exit_date"]).between(
            CONFIRMATION_START, CONFIRMATION_END
        )
    ]
    aggregate_episode_metrics = _episode_metrics(confirmation_episodes)
    _write_frame(output / "selected-closed-episodes.csv", episodes)
    fold_rows: list[dict[str, Any]] = []
    for year in range(2019, 2026):
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        metrics = _period_metrics(selected_result.ledger, start, end)
        fold_episode_metrics = _episode_metrics(
            confirmation_episodes.loc[
                pd.to_datetime(confirmation_episodes["exit_date"]).between(
                    start, end
                )
            ]
        )
        fold_rows.append(
            {
                "fold_id": f"HISTORICAL_CONFIRMATION_{year}",
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "period_profit_factor": metrics["daily_profit_factor"],
                **fold_episode_metrics,
            }
        )
    folds = pd.DataFrame(fold_rows)
    _write_frame(output / "historical-confirmation-folds.csv", folds)

    stress_rows = []
    for cost_bps in (5, 10, 20, 30, 50):
        config = dataclasses.replace(
            selected_config, cost_bps_per_side=float(cost_bps)
        )
        result = lab.run_global_ledger(
            selected_candidates,
            frames,
            calendar,
            fx,
            config,
            {selected_strategy: 1.0},
            portfolio_name=f"{selected['trial_id']}__COST_{cost_bps}",
        )
        metrics = _period_metrics(
            result.ledger, CONFIRMATION_START, CONFIRMATION_END
        )
        stress_episodes = _closed_episodes(result.fills)
        stress_episodes = stress_episodes.loc[
            pd.to_datetime(stress_episodes["exit_date"]).between(
                CONFIRMATION_START, CONFIRMATION_END
            )
        ]
        stress_rows.append(
            {
                "cost_bps_per_side": cost_bps,
                "CAGR": metrics["CAGR"],
                "Sharpe": metrics["Sharpe"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "period_profit_factor": metrics["daily_profit_factor"],
                **_episode_metrics(stress_episodes),
            }
        )
    stress = pd.DataFrame(stress_rows)
    _write_frame(output / "cost-stress.csv", stress)

    selected_returns = confirmation_returns[str(selected["trial_id"])]
    selected_metrics = lab.daily_metrics(selected_returns, 2_000.0)
    benchmark_returns = _equal_weight_benchmark(frames, fx).loc[
        CONFIRMATION_START:CONFIRMATION_END
    ]
    benchmark_metrics = lab.daily_metrics(benchmark_returns, 2_000.0)
    excess_index = selected_returns.index.union(benchmark_returns.index)
    excess = selected_returns.reindex(excess_index).fillna(
        0.0
    ) - benchmark_returns.reindex(excess_index).fillna(0.0)
    excess_std = float(excess.std(ddof=1))
    benchmark_report = {
        "schema": "phase11_7_rotation_benchmark_v1",
        "benchmark": "DYNAMIC_EQUAL_WEIGHT_CLEAN_SAMPLE_NO_COST",
        "selected_metrics": selected_metrics,
        "benchmark_metrics": benchmark_metrics,
        "excess_CAGR": selected_metrics["CAGR"] - benchmark_metrics["CAGR"],
        "excess_Sharpe": selected_metrics["Sharpe"]
        - benchmark_metrics["Sharpe"],
        "information_ratio": (
            float(excess.mean() / excess_std * math.sqrt(252.0))
            if excess_std > 0
            else None
        ),
        **AUTHORITY,
    }
    _write_json(output / "benchmark-comparison.json", benchmark_report)

    trial_sharpes = trials["confirmation_Sharpe"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    dsr = lab.deflated_sharpe_probability(
        float(selected_metrics["Sharpe"]),
        len(selected_returns),
        len(trials),
        float(trial_sharpes.std(ddof=1)),
        float(selected_returns.skew()),
        float(selected_returns.kurt() + 3.0),
    )
    pbo = _pbo(confirmation_returns)
    multiple_testing = {
        "schema": "phase11_7_rotation_multiple_testing_v1",
        "effective_trial_count": len(trials),
        "selected_confirmation_DSR_probability": dsr,
        "PBO": pbo,
        "bootstrap": lab.block_bootstrap(
            selected_returns, bootstrap_runs, 20, 20260724
        ),
        **AUTHORITY,
    }
    _write_json(output / "multiple-testing.json", multiple_testing)
    concentration = _concentration(
        selected_result, (CONFIRMATION_START, CONFIRMATION_END)
    )
    _write_json(
        output / "concentration-audit.json",
        {
            "schema": "phase11_7_rotation_concentration_v1",
            **concentration,
            **AUTHORITY,
        },
    )

    evaluable_folds = folds.loc[
        folds["episode_profit_factor"].notna()
        & folds["episode_count"].ge(10)
    ]
    positive_folds = int(folds["CAGR"].gt(0).sum())
    median_fold_pf = (
        float(evaluable_folds["episode_profit_factor"].median())
        if not evaluable_folds.empty
        else None
    )
    worst_fold_pf = (
        float(evaluable_folds["episode_profit_factor"].min())
        if not evaluable_folds.empty
        else None
    )
    cost_20 = stress.loc[stress["cost_bps_per_side"].eq(20)].iloc[0]
    stale_mark_days = int(
        selected_result.ledger["stale_mark_count"].gt(0).sum()
    )
    technical_gates = {
        "positive_confirmation_folds": positive_folds >= 5,
        "median_fold_episode_pf": median_fold_pf is not None
        and median_fold_pf > 1.10,
        "worst_evaluable_fold_episode_pf": worst_fold_pf is not None
        and worst_fold_pf > 0.80,
        "aggregate_episode_pf": (
            aggregate_episode_metrics["episode_profit_factor"] is not None
            and aggregate_episode_metrics["episode_profit_factor"] > 1.10
        ),
        "cost_20bps_episode_pf": (
            pd.notna(cost_20["episode_profit_factor"])
            and float(cost_20["episode_profit_factor"]) > 1.0
        ),
        "benchmark_excess_cagr": benchmark_report["excess_CAGR"] > 0,
        "drawdown_budget": selected_metrics["maximum_drawdown"] > -0.35,
        "single_security_concentration": concentration[
            "security_contribution_gate"
        ],
        "single_year_concentration": concentration["year_contribution_gate"],
        "DSR": dsr >= 0.80,
        "PBO": pbo.get("PBO") is not None and float(pbo["PBO"]) <= 0.40,
        "sample_size": aggregate_episode_metrics["sample_status"] == "EVALUABLE",
        "accounting": selected_result.accounting_failures == 0,
        "data_quality": data_audit["status"] == "GO",
        "stale_positions": stale_mark_days == 0,
    }
    historical_candidate_go = all(technical_gates.values())
    blockers = [
        name for name, passed in technical_gates.items() if not bool(passed)
    ]
    blockers.extend(
        [
            "POST_CONFIRMATION_STRATEGY_REVISION",
            "FUTURE_FORWARD_HOLDOUT_UNAVAILABLE",
            "HISTORICAL_SHARIAH_RECONSTRUCTION_UNAVAILABLE",
            "DELISTING_CASH_SETTLEMENT_MODEL_UNAVAILABLE",
        ]
    )
    report = {
        "schema": "phase11_7_rotation_campaign_v1",
        "status": "GO",
        "decision": (
            "PROMISING_RESEARCH_CANDIDATE"
            if historical_candidate_go
            else "NO_FINANCIAL_FINALIST"
        ),
        "selected_trial": str(selected["trial_id"]),
        "selection_source": "DEVELOPMENT_2000_2018_ONLY",
        "post_confirmation_revision": True,
        "historical_confirmation_consumed": True,
        "selected_confirmation_metrics": selected_metrics,
        "episode_metrics": aggregate_episode_metrics,
        "fold_count": len(folds),
        "positive_fold_count": positive_folds,
        "median_fold_episode_profit_factor": median_fold_pf,
        "worst_evaluable_fold_episode_profit_factor": worst_fold_pf,
        "stale_mark_days": stale_mark_days,
        "terminal_exit_candidates": int(
            selected_candidates["terminal_exit"].sum()
        ),
        "technical_gates": technical_gates,
        "historical_candidate_go": historical_candidate_go,
        "financial_finalist_go": False,
        "blockers": blockers,
        **AUTHORITY,
    }
    _write_json(output / "decision.json", report)
    _write_json(
        layout.output / "final-decision.json",
        {
            "schema": "phase11_7_final_decision_v1",
            "status": "GO",
            "best_available_research_decision": report["decision"],
            "selected_trial": report["selected_trial"],
            "financial_finalist_go": False,
            "blockers": blockers,
            **AUTHORITY,
        },
    )
    return report


def phase11_7_status(project_root: Path) -> dict[str, Any]:
    layout = Layout(project_root)
    path = layout.output / "final-decision.json"
    if not path.is_file():
        path = layout.output / "decision.json"
    if not path.is_file():
        return {
            "schema": SCHEMA,
            "status": "NOT_RUN",
            "financial_finalist_go": False,
            **AUTHORITY,
        }
    return json.loads(path.read_text(encoding="utf-8"))
