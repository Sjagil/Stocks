from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stocks.research.autopilot.benchmarks import run_simple_benchmarks
from stocks.research.autopilot.components import MACRO_COMPONENT_NAMES
from stocks.research.autopilot.contracts import StrategyFamily, StrategySpec, stable_hash
from stocks.research.autopilot.risk import (
    COST_MODELS,
    PortfolioRiskLimits,
    cost_breakdown,
    enforce_portfolio_limits,
    hierarchical_group_weights,
)


COST_PROFILES_BPS = {
    profile: model.total_bps for profile, model in COST_MODELS.items()
}
ANNUAL_PERIODS = {"1h": 1_638.0, "2h": 819.0, "4h": 410.0, "6h": 273.0, "12h": 137.0, "1d": 252.0, "1w": 52.0, "1mo": 12.0}


@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    family: str
    cost_profile: str
    status: str
    metrics: dict[str, Any]
    provenance: dict[str, Any]
    returns: pd.Series
    weights: pd.DataFrame

    def public_payload(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "family": self.family,
            "cost_profile": self.cost_profile,
            "status": self.status,
            "metrics": self.metrics,
            "provenance": self.provenance,
        }


def run_backtest(
    strategy: StrategySpec,
    bars: Mapping[str, pd.DataFrame],
    *,
    eligible: pd.DataFrame | None = None,
    cost_profile: str = "NORMAL",
    fixture: bool = False,
    evaluation_start: pd.Timestamp | None = None,
    evaluation_end: pd.Timestamp | None = None,
    accounting_returns: pd.DataFrame | None = None,
    fx_returns: pd.DataFrame | None = None,
    metadata: Mapping[str, Mapping[str, str]] | None = None,
    risk_limits: PortfolioRiskLimits = PortfolioRiskLimits(),
    data_contract: Mapping[str, bool] | None = None,
    fundamental_scores: pd.DataFrame | None = None,
    initial_capital_eur: float = 100_000.0,
    macro_regime_features: Mapping[str, pd.Series] | None = None,
    macro_exposure_multiplier: pd.Series | None = None,
) -> BacktestResult:
    if cost_profile not in COST_PROFILES_BPS:
        raise ValueError("UNKNOWN_COST_PROFILE")
    close, volume, high, low = _aligned_fields(bars)
    if close.empty or len(close) < 260:
        return _blocked(strategy, cost_profile, "INSUFFICIENT_HISTORY", close, fixture)
    eligibility = (
        eligible.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
        if eligible is not None
        else pd.DataFrame(bool(fixture), index=close.index, columns=close.columns)
    )
    if not eligibility.to_numpy().any():
        return _blocked(strategy, cost_profile, "PIT_ELIGIBILITY_UNAVAILABLE", close, fixture)
    if not fixture:
        required_contracts = {
            "point_in_time_fundamentals",
            "corporate_actions",
            "delisting_settlement",
            "eur_fx_accounting",
            "market_calendar",
            "stale_data_gate",
        }
        missing = sorted(
            key
            for key in required_contracts
            if not data_contract or not bool(data_contract.get(key))
        )
        if missing:
            return _blocked(
                strategy,
                cost_profile,
                f"DATA_CONTRACT_INCOMPLETE:{','.join(missing)}",
                close,
                fixture,
            )
        if accounting_returns is None:
            return _blocked(
                strategy,
                cost_profile,
                "EUR_TOTAL_RETURN_ACCOUNTING_UNAVAILABLE",
                close,
                fixture,
            )
        if (
            StrategyFamily(strategy.family) == StrategyFamily.QUALITY_MOMENTUM
            and (
                fundamental_scores is None
                or not fundamental_scores.reindex(
                    index=close.index, columns=close.columns
                )
                .where(eligibility)
                .notna()
                .to_numpy()
                .any()
            )
        ):
            return _blocked(
                strategy,
                cost_profile,
                "PIT_QUALITY_SCORE_UNAVAILABLE",
                close,
                fixture,
            )

    scores, tradable = _family_signal(
        strategy,
        close,
        volume,
        fundamental_scores=fundamental_scores,
    )
    macro_filters = set(strategy.regime_components) & MACRO_COMPONENT_NAMES
    if macro_filters:
        available_macro_features = macro_regime_features or {}
        missing_macro = sorted(
            macro_filters - set(available_macro_features)
        )
        if missing_macro:
            return _blocked(
                strategy,
                cost_profile,
                f"MACRO_HISTORY_UNAVAILABLE:{','.join(missing_macro)}",
                close,
                fixture,
            )
        for macro_filter in sorted(macro_filters):
            gate = (
                available_macro_features[macro_filter]
                .reindex(close.index)
                .ffill()
                .fillna(False)
                .astype(bool)
            )
            tradable &= pd.DataFrame(
                {
                    symbol: gate
                    for symbol in close.columns
                },
                index=close.index,
            )
    median_dollar_volume = (close * volume).rolling(20, min_periods=20).median()
    tradable &= eligibility & (
        median_dollar_volume >= risk_limits.min_median_dollar_volume
    )
    portfolio_scores, exit_audit = _stateful_portfolio_scores(
        strategy,
        scores,
        tradable,
        eligibility,
        close,
        high,
        low,
    )
    target = _portfolio_weights(
        strategy,
        portfolio_scores,
        close,
        metadata=metadata,
    )
    target = target.where(
        target * float(initial_capital_eur) >= risk_limits.min_order_notional_eur,
        0.0,
    )
    target = enforce_portfolio_limits(target, metadata=metadata, limits=risk_limits)
    if macro_exposure_multiplier is not None:
        from stocks.macro.engine import apply_macro_exposure

        target = apply_macro_exposure(
            target,
            macro_exposure_multiplier,
            minimum=0.5,
            maximum=1.1,
        )
    # Signal at close t is executable no earlier than the next bar.
    weights = target.shift(1).fillna(0.0).clip(lower=0.0)
    asset_returns = (
        accounting_returns.reindex(index=close.index, columns=close.columns)
        if accounting_returns is not None
        else close.pct_change(fill_method=None)
    )
    asset_returns = asset_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    weights, position_loss_exits = _apply_position_loss_limit(
        weights,
        asset_returns,
        risk_limits.max_position_loss,
    )
    exit_audit["portfolio_position_loss_exit"] = position_loss_exits
    weights = _apply_drawdown_circuit_breaker(
        weights, asset_returns, risk_limits.drawdown_circuit_breaker
    )
    gross = (weights * asset_returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    cost_parts = cost_breakdown(turnover, cost_profile)
    costs = sum(cost_parts.values(), start=pd.Series(0.0, index=turnover.index))
    net = gross - costs
    periods = _empirical_periods_per_year(
        close.index, ANNUAL_PERIODS[strategy.entry_timeframe]
    )
    benchmark, benchmark_bundle = run_simple_benchmarks(
        close,
        asset_returns,
        eligibility,
        cost_profile=cost_profile,
        periods_per_year=periods,
    )
    evaluation_mask = pd.Series(True, index=net.index)
    if evaluation_start is not None:
        evaluation_mask &= net.index >= _utc_timestamp(evaluation_start)
    if evaluation_end is not None:
        evaluation_mask &= net.index <= _utc_timestamp(evaluation_end)
    metrics = _metrics(
        net.loc[evaluation_mask],
        gross.loc[evaluation_mask],
        benchmark.loc[evaluation_mask],
        turnover.loc[evaluation_mask],
        costs.loc[evaluation_mask],
        weights.loc[evaluation_mask],
        periods,
        asset_returns=asset_returns.loc[evaluation_mask],
        cost_parts={
            key: value.loc[evaluation_mask] for key, value in cost_parts.items()
        },
        fx_returns=(
            None
            if fx_returns is None
            else fx_returns.reindex(index=close.index, columns=close.columns)
            .fillna(0.0)
            .loc[evaluation_mask]
        ),
        metadata=metadata,
    )
    metrics["exit_reason_counts"] = exit_audit
    metrics["drawdown_attribution"] = _drawdown_attribution(metrics, exit_audit)
    metrics["benchmark_champion"] = benchmark_bundle["champion"]
    metrics["benchmark_results"] = benchmark_bundle["results"]
    metrics["benchmark_selection_policy"] = benchmark_bundle[
        "selection_policy"
    ]
    status = "COMPLETE" if metrics["observations"] > 0 else "NO_EVALUABLE_OBSERVATIONS"
    provenance = {
        "bar_origin": "SYNTHETIC_TEST_FIXTURE" if fixture else "LOCAL_PROVIDER_CACHE",
        "closed_candles_only": True,
        "next_bar_execution": True,
        "point_in_time_eligibility": eligible is not None,
        "fixture": fixture,
        "timeframe": strategy.entry_timeframe,
        "periods_per_year": periods,
        "cost_bps": COST_PROFILES_BPS[cost_profile],
        "cost_model": COST_MODELS[cost_profile].as_dict(),
        "risk_limits": {
            key: value
            for key, value in risk_limits.__dict__.items()
        },
        "data_contract": dict(data_contract or {}),
        "macro_filter_count": len(macro_filters),
        "macro_exposure_modifier_applied": (
            macro_exposure_multiplier is not None
        ),
        "exit_audit": exit_audit,
        "evaluation_start": (
            None if evaluation_start is None else _utc_timestamp(evaluation_start).isoformat()
        ),
        "evaluation_end": (
            None if evaluation_end is None else _utc_timestamp(evaluation_end).isoformat()
        ),
        "code_hash": _module_hash(),
        "input_hash": stable_hash(
            {
                "strategy_hash": strategy.strategy_hash,
                "symbols": list(close.columns),
                "start": str(close.index.min()),
                "end": str(close.index.max()),
                "rows": len(close),
            }
        ),
    }
    return BacktestResult(
        strategy.strategy_id,
        strategy.family,
        cost_profile,
        status,
        metrics,
        provenance,
        net,
        weights,
    )


def deterministic_fixture(
    *,
    symbols: int = 16,
    periods: int = 1_600,
    seed: int = 20260726,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=periods, tz="UTC")
    market = rng.normal(0.00025, 0.009, periods)
    bars: dict[str, pd.DataFrame] = {}
    for number in range(symbols):
        cyclic = 0.00035 * np.sin(np.arange(periods) / (35 + number))
        noise = rng.normal(0.0, 0.006 + number * 0.0001, periods)
        returns = market * (0.45 + number / 50) + cyclic + noise
        close = (45 + number * 3) * np.exp(np.cumsum(returns))
        spread = np.abs(rng.normal(0.006, 0.002, periods))
        bars[f"FIX{number:03d}"] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, periods)),
                "high": close * (1 + spread),
                "low": close * (1 - spread),
                "close": close,
                "volume": rng.integers(250_000, 4_000_000, periods),
                "is_closed": True,
            },
            index=dates,
        )
    eligible = pd.DataFrame(True, index=dates, columns=bars)
    return bars, eligible


def _aligned_fields(
    bars: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    closes: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    highs: dict[str, pd.Series] = {}
    lows: dict[str, pd.Series] = {}
    for symbol, frame in sorted(bars.items()):
        if frame.empty or "close" not in frame:
            continue
        if "is_closed" in frame and not bool(frame["is_closed"].fillna(False).all()):
            raise ValueError(f"OPEN_CANDLE_BLOCKED:{symbol}")
        index = pd.to_datetime(frame.index, utc=True)
        closes[symbol] = pd.Series(
            pd.to_numeric(frame["close"], errors="coerce").to_numpy(), index=index
        )
        volume = frame["volume"] if "volume" in frame else pd.Series(0.0, index=frame.index)
        volumes[symbol] = pd.Series(pd.to_numeric(volume, errors="coerce").to_numpy(), index=index)
        high = frame["high"] if "high" in frame else frame["close"]
        low = frame["low"] if "low" in frame else frame["close"]
        highs[symbol] = pd.Series(
            pd.to_numeric(high, errors="coerce").to_numpy(), index=index
        )
        lows[symbol] = pd.Series(
            pd.to_numeric(low, errors="coerce").to_numpy(), index=index
        )
    return (
        pd.DataFrame(closes).sort_index(),
        pd.DataFrame(volumes).sort_index().fillna(0.0),
        pd.DataFrame(highs).sort_index(),
        pd.DataFrame(lows).sort_index(),
    )


def _family_signal(
    strategy: StrategySpec,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    fundamental_scores: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = StrategyFamily(strategy.family)
    momentum_3m = close.pct_change(63)
    momentum_6m = close.pct_change(126)
    momentum_12m = close.pct_change(252)
    ma50 = close.rolling(50, min_periods=50).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    trend = (close > ma200) & (ma50 > ma200)
    if family == StrategyFamily.QUALITY_MOMENTUM:
        quality = (
            fundamental_scores.reindex(index=close.index, columns=close.columns)
            if fundamental_scores is not None
            else pd.DataFrame(0.75, index=close.index, columns=close.columns)
        )
        quality_rank = quality.rank(axis=1, pct=True)
        score = (
            0.40 * quality_rank
            + 0.25 * momentum_6m.rank(axis=1, pct=True)
            + 0.35 * momentum_12m.rank(axis=1, pct=True)
        )
        return (
            score,
            trend
            & (quality > 0)
            & (momentum_6m > 0)
            & (momentum_12m > 0),
        )
    if family == StrategyFamily.TREND_PULLBACK:
        ema_period = int(strategy.parameters["pullback_ema"])
        ema = close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
        distance = (close / ema - 1.0).abs()
        relative_volume = volume / volume.rolling(20, min_periods=20).median()
        score = (1.0 - distance.clip(upper=0.2) / 0.2) + momentum_6m.rank(axis=1, pct=True)
        return score, trend & (distance <= float(strategy.parameters["tolerance"])) & (relative_volume >= 0.7)
    if family == StrategyFamily.ETF_ROTATION:
        score = momentum_3m.rank(axis=1, pct=True) + momentum_6m.rank(axis=1, pct=True) + momentum_12m.rank(axis=1, pct=True)
        return score, (close > ma200) & (momentum_6m > 0)
    if family == StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT:
        returns = close.pct_change(fill_method=None)
        short = returns.rolling(int(strategy.parameters["vol_short"])).std()
        long = returns.rolling(int(strategy.parameters["vol_long"])).std()
        period = int(strategy.parameters["breakout_period"])
        prior_high = close.rolling(period).max().shift(1)
        relative_volume = volume / volume.rolling(20, min_periods=20).median()
        score = close / prior_high - 1.0
        return score, trend & (short / long <= float(strategy.parameters["contraction"])) & (close > prior_high) & (relative_volume >= 1.0)
    score = momentum_6m.rank(axis=1, pct=True)
    return score, trend & (momentum_6m > 0)


def _portfolio_weights(
    strategy: StrategySpec,
    scores: pd.DataFrame,
    close: pd.DataFrame,
    *,
    metadata: Mapping[str, Mapping[str, str]] | None,
) -> pd.DataFrame:
    top_n = int(strategy.parameters.get("top_n", 5))
    ranks = scores.rank(axis=1, ascending=False, method="first")
    selected = scores.notna() & (ranks <= top_n)
    if strategy.portfolio_model in {"sector_first", "regional_sleeves"}:
        if metadata is None:
            raise ValueError("HIERARCHICAL_METADATA_REQUIRED")
        return hierarchical_group_weights(
            scores,
            selected,
            metadata=metadata,
            group_field=(
                "sector"
                if strategy.portfolio_model == "sector_first"
                else "region"
            ),
        )
    if strategy.portfolio_model in {"inverse_volatility", "capped_risk_adjusted"}:
        volatility = close.pct_change(fill_method=None).rolling(63, min_periods=30).std()
        raw = selected.astype(float).div(volatility.replace(0.0, np.nan))
    elif strategy.portfolio_model in {"score_weight", "rank_weight"}:
        raw = scores.clip(lower=0.0).where(selected)
    else:
        raw = selected.astype(float)
    weights = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    cap = min(0.25, 1.0 / max(1, min(top_n, 10)))
    weights = weights.clip(upper=cap)
    return weights


def _apply_position_loss_limit(
    requested_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    maximum_loss: float,
) -> tuple[pd.DataFrame, int]:
    """Exit on the bar after an episode breaches its cumulative loss limit."""
    output = pd.DataFrame(
        0.0,
        index=requested_weights.index,
        columns=requested_weights.columns,
    )
    exit_count = 0
    for symbol in requested_weights.columns:
        episode_nav = 1.0
        locked = False
        prior_requested = 0.0
        for timestamp in requested_weights.index:
            requested = float(requested_weights.loc[timestamp, symbol])
            if requested <= 0:
                episode_nav = 1.0
                locked = False
                prior_requested = 0.0
                continue
            if prior_requested <= 0:
                episode_nav = 1.0
            if not locked:
                output.loc[timestamp, symbol] = requested
                period_return = float(asset_returns.loc[timestamp, symbol])
                episode_nav *= 1.0 + period_return
                if episode_nav - 1.0 <= maximum_loss:
                    locked = True
                    exit_count += 1
            prior_requested = requested
    return output, exit_count


def _stateful_portfolio_scores(
    strategy: StrategySpec,
    scores: pd.DataFrame,
    entries: pd.DataFrame,
    eligibility: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    family = StrategyFamily(strategy.family)
    exit_counts = {
        "ranking_exit": 0,
        "rebalance_exit": 0,
        "regime_exit": 0,
        "fundamental_eligibility_exit": 0,
        "moving_average_exit": 0,
        "atr_trailing_exit": 0,
        "time_stop": 0,
    }
    if strategy.rebalance in {"MONTHLY", "QUARTERLY"}:
        rebalance = _rebalance_mask(close.index, strategy.rebalance)
        top_n = int(strategy.parameters.get("top_n", 10))
        selected = pd.DataFrame(False, index=close.index, columns=close.columns)
        frozen_scores = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        current = pd.Series(False, index=close.columns)
        current_scores = pd.Series(np.nan, index=close.columns)
        for timestamp in close.index:
            # Point-in-time eligibility/fundamental failure always exits; it
            # does not wait until the next scheduled ranking rebalance.
            prior = current.copy()
            current &= eligibility.loc[timestamp].fillna(False)
            exit_counts["fundamental_eligibility_exit"] += int(
                (prior & ~current).sum()
            )
            if "regime_exit" in strategy.exit_components:
                prior = current.copy()
                current &= entries.loc[timestamp].fillna(False)
                exit_counts["regime_exit"] += int((prior & ~current).sum())
            if bool(rebalance.loc[timestamp]):
                prior = current.copy()
                candidates = scores.loc[timestamp].where(
                    entries.loc[timestamp].fillna(False)
                )
                names = candidates.dropna().nlargest(top_n).index
                current[:] = False
                current.loc[names] = True
                removed = int((prior & ~current).sum())
                exit_counts["ranking_exit"] += removed
                exit_counts["rebalance_exit"] += removed
                current_scores[:] = np.nan
                current_scores.loc[names] = candidates.loc[names]
            selected.loc[timestamp] = current
            frozen_scores.loc[timestamp] = current_scores
        return frozen_scores.where(selected), exit_counts

    atr = _average_true_range(high, low, close, 14)
    ema_exit = close.ewm(span=50, adjust=False, min_periods=50).mean()
    max_hold = int(strategy.parameters.get("max_hold", 126))
    atr_multiple = float(strategy.parameters.get("atr_multiple", 3.0))
    active = pd.DataFrame(False, index=close.index, columns=close.columns)
    held_scores = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for symbol in close.columns:
        holding = False
        age = 0
        peak = np.nan
        entry_score = np.nan
        for timestamp in close.index:
            price = close.loc[timestamp, symbol]
            can_enter = bool(entries.loc[timestamp, symbol]) and pd.notna(price)
            if holding:
                age += 1
                peak = max(float(peak), float(price))
                stop = peak - atr_multiple * float(atr.loc[timestamp, symbol])
                trend_exit = (
                    "moving_average_exit" in strategy.exit_components
                    and pd.notna(ema_exit.loc[timestamp, symbol])
                    and float(price) < float(ema_exit.loc[timestamp, symbol])
                )
                atr_exit = (
                    "atr_trailing_exit" in strategy.exit_components
                    and pd.notna(stop)
                    and float(price) < stop
                )
                time_exit = (
                    "time_stop" in strategy.exit_components and age >= max_hold
                )
                eligibility_exit = not bool(eligibility.loc[timestamp, symbol])
                if trend_exit or atr_exit or time_exit or eligibility_exit:
                    if eligibility_exit:
                        exit_counts["fundamental_eligibility_exit"] += 1
                    elif atr_exit:
                        exit_counts["atr_trailing_exit"] += 1
                    elif trend_exit:
                        exit_counts["moving_average_exit"] += 1
                    elif time_exit:
                        exit_counts["time_stop"] += 1
                    holding = False
            if not holding and can_enter:
                holding = True
                age = 0
                peak = float(price)
                entry_score = float(scores.loc[timestamp, symbol])
            active.loc[timestamp, symbol] = holding
            if holding:
                held_scores.loc[timestamp, symbol] = entry_score
    if family == StrategyFamily.VOLATILITY_CONTRACTION_BREAKOUT:
        return held_scores.where(active), exit_counts
    return held_scores.where(active), exit_counts


def _rebalance_mask(index: pd.DatetimeIndex, cadence: str) -> pd.Series:
    naive = index.tz_localize(None) if index.tz is not None else index
    period = naive.to_period("M" if cadence == "MONTHLY" else "Q")
    next_period = pd.Series(period, index=index).shift(-1)
    current = pd.Series(period, index=index)
    return current.ne(next_period)


def _average_true_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    period: int,
) -> pd.DataFrame:
    previous = close.shift(1)
    true_range = pd.DataFrame(
        np.maximum.reduce(
            [
                (high - low).to_numpy(),
                (high - previous).abs().to_numpy(),
                (low - previous).abs().to_numpy(),
            ]
        ),
        index=close.index,
        columns=close.columns,
    )
    return true_range.rolling(period, min_periods=period).mean()


def _metrics(
    net: pd.Series,
    gross: pd.Series,
    benchmark: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    weights: pd.DataFrame,
    periods: float,
    *,
    asset_returns: pd.DataFrame,
    cost_parts: Mapping[str, pd.Series],
    fx_returns: pd.DataFrame | None,
    metadata: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, Any]:
    active = net.dropna()
    nav = (1.0 + active).cumprod()
    years = len(active) / periods
    total = float(nav.iloc[-1] - 1.0) if len(nav) else 0.0
    cagr = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and nav.iloc[-1] > 0 else -1.0
    std = float(active.std(ddof=1))
    downside = float(active[active < 0].std(ddof=1))
    sharpe = float(active.mean() / std * np.sqrt(periods)) if std > 0 else None
    sortino = float(active.mean() / downside * np.sqrt(periods)) if downside > 0 else None
    drawdown = nav / nav.cummax() - 1.0 if len(nav) else pd.Series(dtype=float)
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    positive = float(active[active > 0].sum())
    negative = float(active[active < 0].sum())
    period_pf = positive / abs(negative) if negative < 0 else None
    active_days = weights.sum(axis=1) > 0
    episodes = _closed_episode_pnls(
        weights,
        asset_returns,
        sum(cost_parts.values(), start=pd.Series(0.0, index=weights.index)),
    )
    entries = ((weights > 0) & (weights.shift(1).fillna(0) == 0)).sum().sum()
    annual = active.groupby(active.index.year).apply(lambda value: (1 + value).prod() - 1)
    excess = active - benchmark.reindex(active.index).fillna(0.0)
    excess_std = float(excess.std(ddof=1))
    episode_gains = sum(max(item["pnl"], 0.0) for item in episodes)
    episode_losses = sum(min(item["pnl"], 0.0) for item in episodes)
    episode_pf = (
        episode_gains / abs(episode_losses)
        if episode_losses < 0
        else None
    )
    benchmark_positive = benchmark.reindex(active.index).fillna(0.0) > 0
    benchmark_negative = benchmark.reindex(active.index).fillna(0.0) < 0
    contribution = (weights * asset_returns).sum(axis=0)
    positive_contribution = contribution.clip(lower=0.0)
    positive_total = float(positive_contribution.sum())
    sector_exposure = _group_exposure(weights, metadata, "sector")
    region_exposure = _group_exposure(weights, metadata, "region")
    currency_exposure = _group_exposure(weights, metadata, "currency")
    fx_impact = (
        float((weights * fx_returns).sum(axis=1).sum())
        if fx_returns is not None
        else None
    )
    return {
        "observations": int(len(active)),
        "gross_total_return": float((1 + gross).prod() - 1),
        "net_total_return": total,
        "CAGR": cagr,
        "annualized_volatility": std * np.sqrt(periods),
        "Sharpe": sharpe,
        "Sortino": sortino,
        "maximum_drawdown": max_dd,
        "Calmar": cagr / abs(max_dd) if max_dd < 0 else None,
        "period_profit_factor": period_pf,
        "episode_profit_factor": episode_pf,
        "expectancy": (
            float(pd.Series([item["pnl"] for item in episodes]).mean())
            if episodes
            else None
        ),
        "hit_rate": (
            sum(item["pnl"] > 0 for item in episodes) / len(episodes)
            if episodes
            else None
        ),
        "turnover": float(turnover.sum()),
        "transaction_costs": float(costs.sum()),
        "commission_cost": float(cost_parts["commission"].sum()),
        "spread_cost": float(cost_parts["spread"].sum()),
        "slippage_cost": float(cost_parts["slippage"].sum()),
        "fx_trading_cost": float(cost_parts["fx"].sum()),
        "fx_return_impact": fx_impact,
        "trade_episodes": int(entries),
        "closed_episodes": len(episodes),
        "average_holding_periods": (
            float(pd.Series([item["periods"] for item in episodes]).mean())
            if episodes
            else None
        ),
        "average_exposure": float(weights.sum(axis=1).mean()),
        "maximum_exposure": float(weights.sum(axis=1).max()),
        "maximum_position_weight": float(weights.max(axis=1).max()),
        "average_weight_hhi": float(weights.pow(2).sum(axis=1).mean()),
        "average_effective_holdings": float(
            (1.0 / weights.pow(2).sum(axis=1).replace(0.0, np.nan)).mean()
        ),
        "average_cash": float((1.0 - weights.sum(axis=1)).mean()),
        "positive_years": int((annual > 0).sum()),
        "evaluated_years": int(len(annual)),
        "maximum_drawdown_duration_periods": _maximum_drawdown_duration(nav),
        "single_security_positive_contribution_share": (
            float(positive_contribution.max() / positive_total)
            if positive_total > 0
            else None
        ),
        "top_positive_contributor": (
            str(positive_contribution.idxmax())
            if positive_total > 0
            else None
        ),
        "single_year_positive_return_share": _single_year_positive_share(annual),
        "sector_exposure": sector_exposure,
        "region_exposure": region_exposure,
        "currency_exposure": currency_exposure,
        "benchmark_total_return": float((1 + benchmark).prod() - 1),
        "excess_total_return": float((1 + active).prod() - (1 + benchmark).prod()),
        "information_ratio": float(excess.mean() / excess_std * np.sqrt(periods)) if excess_std > 0 else None,
        "upside_capture": _capture_ratio(active, benchmark, benchmark_positive),
        "downside_capture": _capture_ratio(active, benchmark, benchmark_negative),
        "active_period_ratio": float(active_days.mean()),
    }


def _blocked(
    strategy: StrategySpec,
    cost_profile: str,
    reason: str,
    close: pd.DataFrame,
    fixture: bool,
) -> BacktestResult:
    return BacktestResult(
        strategy.strategy_id,
        strategy.family,
        cost_profile,
        f"BLOCKED:{reason}",
        {"observations": int(len(close)), "blocker": reason},
        {
            "fixture": fixture,
            "closed_candles_only": True,
            "next_bar_execution": True,
            "code_hash": _module_hash(),
        },
        pd.Series(dtype=float),
        pd.DataFrame(),
    )


def _module_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest().upper()


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _empirical_periods_per_year(
    index: pd.DatetimeIndex,
    fallback: float,
) -> float:
    if len(index) < 2:
        return fallback
    years = (
        (index.max() - index.min()).total_seconds()
        / (365.2425 * 24 * 60 * 60)
    )
    if years < 0.5:
        return fallback
    estimate = len(index) / years
    return float(estimate) if estimate > 0 else fallback


def _apply_drawdown_circuit_breaker(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    threshold: float,
    *,
    cooldown_periods: int = 20,
) -> pd.DataFrame:
    output = weights.copy()
    equity = peak = 1.0
    cooldown = 0
    for timestamp in output.index:
        if cooldown > 0:
            output.loc[timestamp] = 0.0
            cooldown -= 1
        period_return = float(
            (
                output.loc[timestamp]
                * asset_returns.loc[timestamp].reindex(output.columns).fillna(0.0)
            ).sum()
        )
        equity *= 1.0 + period_return
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0
        if drawdown <= threshold and cooldown == 0:
            cooldown = cooldown_periods
    return output


def _closed_episode_pnls(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    total_costs: pd.Series,
) -> list[dict[str, Any]]:
    changes = weights.diff().fillna(weights)
    absolute_changes = changes.abs().sum(axis=1).replace(0.0, np.nan)
    allocated_cost = changes.abs().div(absolute_changes, axis=0).fillna(0.0)
    allocated_cost = allocated_cost.mul(total_costs, axis=0)
    contributions = weights * asset_returns - allocated_cost
    episodes: list[dict[str, Any]] = []
    for symbol in weights.columns:
        active = weights[symbol] > 0
        start: pd.Timestamp | None = None
        pnl = 0.0
        periods = 0
        for timestamp in weights.index:
            if bool(active.loc[timestamp]):
                if start is None:
                    start = timestamp
                    pnl = 0.0
                    periods = 0
                pnl += float(contributions.loc[timestamp, symbol])
                periods += 1
            elif start is not None:
                episodes.append(
                    {
                        "symbol": symbol,
                        "start": start,
                        "end": timestamp,
                        "pnl": pnl,
                        "periods": periods,
                    }
                )
                start = None
    return episodes


def _maximum_drawdown_duration(nav: pd.Series) -> int:
    if nav.empty:
        return 0
    below_peak = nav < nav.cummax()
    longest = current = 0
    for value in below_peak:
        current = current + 1 if bool(value) else 0
        longest = max(longest, current)
    return longest


def _single_year_positive_share(annual: pd.Series) -> float | None:
    positive = annual.clip(lower=0.0)
    total = float(positive.sum())
    return float(positive.max() / total) if total > 0 else None


def _group_exposure(
    weights: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, str]] | None,
    field: str,
) -> dict[str, float]:
    if not metadata:
        return {}
    groups: dict[str, list[str]] = {}
    for symbol in weights.columns:
        value = str(metadata.get(symbol, {}).get(field) or "UNKNOWN")
        groups.setdefault(value, []).append(symbol)
    return {
        group: float(weights[symbols].sum(axis=1).mean())
        for group, symbols in sorted(groups.items())
    }


def _capture_ratio(
    strategy: pd.Series,
    benchmark: pd.Series,
    mask: pd.Series,
) -> float | None:
    selected_benchmark = benchmark.reindex(strategy.index).fillna(0.0).loc[mask]
    selected_strategy = strategy.loc[mask]
    denominator = float(selected_benchmark.mean()) if len(selected_benchmark) else 0.0
    return (
        float(selected_strategy.mean() / denominator)
        if denominator != 0
        else None
    )


def _drawdown_attribution(
    metrics: Mapping[str, Any],
    exit_counts: Mapping[str, int],
) -> dict[str, Any]:
    if float(metrics.get("maximum_drawdown") or 0.0) >= -0.05:
        return {
            "status": "NOT_MATERIAL",
            "primary_classification": None,
        }
    return {
        "status": "INSUFFICIENT_ATTRIBUTION_EVIDENCE",
        "primary_classification": "UNATTRIBUTED",
        "candidate_classes": [
            "LATE_ENTRY",
            "ASSET_SELECTION",
            "LATE_EXIT",
            "TIGHT_EXIT",
            "CONCENTRATION",
            "REGIME_FAILURE",
            "GAP",
            "VOLATILITY_EXPANSION",
        ],
        "exit_reason_counts": dict(exit_counts),
        "automatic_strategy_mutation": False,
    }
