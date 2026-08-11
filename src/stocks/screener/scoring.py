from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from stocks.features.fundamental import (
    accruals,
    dilution,
    earnings_yield,
    free_cash_flow_yield,
    margin,
    market_cap,
    roa,
)
from stocks.features.technical import exponential_moving_average, momentum, relative_strength
from stocks.screener.config import ScreenerConfig
from stocks.screener.models import AssetSnapshot, FundamentalSnapshot, ScoreResult
from stocks.screener.sources import decision_time_for_session, trading_session_distance


def classify_scores(
    *,
    total_score: float,
    fundamental_score: float,
    technical_score: float,
    hard_filter_pass: bool,
    long_term_trend_positive: bool,
    momentum_positive: bool,
    config: ScreenerConfig,
) -> str:
    if not hard_filter_pass:
        return "REJECTED"
    gates = config.classifications
    if (
        total_score >= gates["high_potential_total"]
        and fundamental_score >= gates["high_potential_fundamental"]
        and technical_score >= gates["high_potential_technical"]
        and long_term_trend_positive
        and momentum_positive
    ):
        return "HIGH_POTENTIAL"
    if total_score >= gates["watchlist_total"]:
        return "WATCHLIST"
    if total_score >= gates["neutral_total"]:
        return "NEUTRAL"
    return "REJECTED"


def score_asset(
    snapshot: AssetSnapshot,
    *,
    screening_date: date,
    config: ScreenerConfig,
    macro_context: dict[str, Any] | None = None,
    known_at: datetime | None = None,
) -> ScoreResult:
    decision_time = known_at or decision_time_for_session(screening_date)
    bars = snapshot.bars.sort_index()
    close = pd.to_numeric(bars.get("close"), errors="coerce").dropna()
    volume = pd.to_numeric(bars.get("volume"), errors="coerce").fillna(0.0)
    hard_failures: list[str] = []
    selection_reasons: list[str] = []
    warnings: list[str] = []

    if bars.empty or close.empty:
        hard_failures.append("MISSING_PRICE_DATA")
        return _empty_result(
            snapshot,
            screening_date,
            config,
            hard_failures,
            decision_time=decision_time,
        )
    latest_date = pd.Timestamp(bars.index[-1]).date()
    if latest_date > screening_date:
        hard_failures.append("LOOKAHEAD_PRICE_DATA_BLOCKED")
    stale_sessions = trading_session_distance(latest_date, screening_date)
    if stale_sessions > int(config.thresholds["maximum_stale_sessions"]):
        hard_failures.append("STALE_PRICE_DATA")
    if len(close) < int(config.thresholds["minimum_history_rows"]):
        hard_failures.append("INSUFFICIENT_PRICE_HISTORY")
    price = float(close.iloc[-1])
    daily_return = (
        None
        if len(close) < 2 or float(close.iloc[-2]) <= 0
        else float(close.iloc[-1] / close.iloc[-2] - 1.0)
    )
    if not math.isfinite(price) or price <= 0:
        hard_failures.append("INVALID_PRICE")
    if daily_return is not None and abs(daily_return) > 1.0:
        hard_failures.append("EXTREME_DAILY_RETURN_DATA_ANOMALY")
    if price < config.thresholds["minimum_price"]:
        hard_failures.append("PENNY_STOCK_BLOCKED")
    if snapshot.metadata.inactive:
        hard_failures.append("INACTIVE_OR_DELISTED")
    if snapshot.provider_conflict:
        hard_failures.append("PROVIDER_PRICE_CONFLICT")

    latest = bars.iloc[-1]
    raw_close = float(latest.get("raw_close", latest.get("close", math.nan)))
    ohlc = [float(latest.get(column, math.nan)) for column in ("open", "high", "low")]
    if not all(math.isfinite(value) and value > 0 for value in [*ohlc, raw_close]):
        hard_failures.append("INVALID_OR_INCOMPLETE_CANDLE")
    elif ohlc[1] < max(ohlc[0], ohlc[2], raw_close) or ohlc[2] > min(
        ohlc[0], ohlc[1], raw_close
    ):
        hard_failures.append("INVALID_OR_INCOMPLETE_CANDLE")

    name_and_category = " ".join(
        item
        for item in (snapshot.metadata.name, snapshot.metadata.category)
        if item is not None
    ).upper()
    if snapshot.metadata.asset_type != "STOCK" and any(
        term in name_and_category for term in config.excluded_product_terms
    ):
        hard_failures.append("LEVERAGED_OR_INVERSE_PRODUCT")
    if not snapshot.shariah.eligible_at(decision_time):
        hard_failures.append(snapshot.shariah.status)

    median_volume = float(volume.tail(20).median()) if not volume.empty else 0.0
    dollar_volume = (close * volume.reindex(close.index).fillna(0.0)).tail(20)
    median_dollar_volume = float(dollar_volume.median()) if not dollar_volume.empty else 0.0
    if median_volume < config.thresholds["minimum_median_volume_20d"]:
        hard_failures.append("INSUFFICIENT_VOLUME")
    if median_dollar_volume < config.thresholds["minimum_median_dollar_volume_20d"]:
        hard_failures.append("INSUFFICIENT_DOLLAR_VOLUME")
    if snapshot.bid_ask_spread_bps is None:
        warnings.append("BID_ASK_SPREAD_UNAVAILABLE")
    elif snapshot.bid_ask_spread_bps > config.thresholds["maximum_spread_bps"]:
        hard_failures.append("EXTREME_BID_ASK_SPREAD")

    fundamental_score, fundamental_metrics, fundamental_failures = _fundamental_score(
        snapshot.fundamental,
        price=price,
        asset_type=snapshot.metadata.asset_type,
        knowledge_date=decision_time.date(),
        config=config,
    )
    hard_failures.extend(fundamental_failures)
    if fundamental_metrics.get("applicability") == "ETF_HOLDINGS_LEVEL":
        warnings.append("ETF_HOLDINGS_FUNDAMENTALS_UNAVAILABLE")
    technical_score, technical_metrics, technical_failures = _technical_score(
        bars,
        snapshot.benchmark_bars,
    )
    hard_failures.extend(technical_failures)
    liquidity_score = _liquidity_score(median_volume, median_dollar_volume)
    risk_score, risk_metrics = _risk_score(bars, technical_metrics)
    components = {
        "fundamental": fundamental_score,
        "technical": technical_score,
        "liquidity": liquidity_score,
        "risk": risk_score,
    }
    for component, value in components.items():
        if not math.isfinite(value):
            hard_failures.append(f"INVALID_{component.upper()}_SCORE_COMPONENT")
            components[component] = 0.0
    fundamental_score = components["fundamental"]
    technical_score = components["technical"]
    liquidity_score = components["liquidity"]
    risk_score = components["risk"]
    macro_score, macro_public = _macro_context_score(
        snapshot,
        macro_context,
    )

    if risk_metrics["annualized_volatility"] is not None and (
        risk_metrics["annualized_volatility"]
        > config.thresholds["maximum_annualized_volatility"]
    ):
        hard_failures.append("EXTREME_VOLATILITY")
    if (
        technical_metrics["distance_ema50"] is not None
        and technical_metrics["distance_ema50"]
        > config.thresholds["maximum_distance_ema50"]
    ):
        hard_failures.append("EXTREME_OVEREXTENSION")

    score_values = {
        "technical": technical_score,
        "liquidity": liquidity_score,
        "risk": risk_score,
    }
    if fundamental_metrics.get("applicability") == "COMPANY_LEVEL":
        score_values["fundamental"] = fundamental_score
    active_weights = {
        key: config.weights[key]
        for key in score_values
    }
    if macro_score is not None:
        score_values["macro"] = macro_score
        active_weights["macro"] = config.weights["macro"]
    denominator = sum(active_weights.values())
    effective_weights = {
        key: value / denominator for key, value in active_weights.items()
    }
    total_score = round(
        sum(
            score_values[key] * effective_weights[key]
            for key in score_values
        ),
        4,
    )
    hard_failures = list(dict.fromkeys(hard_failures))
    long_term_trend = bool(
        technical_metrics["price_above_ema200"]
        and technical_metrics["ema50_above_ema200"]
    )
    momentum_positive = bool(
        technical_metrics["momentum_3m"] is not None
        and technical_metrics["momentum_6m"] is not None
        and technical_metrics["momentum_12m"] is not None
        and technical_metrics["momentum_3m"] > 0
        and technical_metrics["momentum_6m"] > 0
        and technical_metrics["momentum_12m"] > 0
    )
    classification = classify_scores(
        total_score=total_score,
        fundamental_score=fundamental_score,
        technical_score=technical_score,
        hard_filter_pass=not hard_failures,
        long_term_trend_positive=long_term_trend,
        momentum_positive=momentum_positive,
        config=config,
    )
    if not hard_failures:
        if long_term_trend:
            selection_reasons.append("POSITIVE_LONG_TERM_TREND")
        if momentum_positive:
            selection_reasons.append("POSITIVE_3_6_12_MONTH_MOMENTUM")
        if technical_metrics["relative_strength_6m"] is not None and (
            technical_metrics["relative_strength_6m"] > 0
        ):
            selection_reasons.append("POSITIVE_RELATIVE_STRENGTH")
        if fundamental_metrics.get("positive_earnings_or_fcf"):
            selection_reasons.append("POSITIVE_EARNINGS_OR_FCF")
        if snapshot.mover_type:
            selection_reasons.append(f"DAILY_{snapshot.mover_type}")

    public = {
        "screening_date": screening_date.isoformat(),
        "decision_time": decision_time.isoformat(),
        "asset_key": snapshot.metadata.asset_key,
        "symbol": snapshot.metadata.symbol,
        "con_id": snapshot.metadata.con_id,
        "asset_type": snapshot.metadata.asset_type,
        "exchange": snapshot.metadata.exchange,
        "currency": snapshot.metadata.currency,
        "sector": snapshot.metadata.sector,
        "industry": snapshot.metadata.industry,
        "category": snapshot.metadata.category,
        "shariah_status": snapshot.shariah.status,
        "price": round(price, 8),
        "market_cap": fundamental_metrics.get("market_cap"),
        "fundamental_coverage": fundamental_metrics.get("coverage", 0.0),
        "fundamental_applicability": fundamental_metrics.get(
            "applicability",
            "UNKNOWN",
        ),
        "median_dollar_volume_20d": median_dollar_volume,
        "bid_ask_spread_bps": snapshot.bid_ask_spread_bps,
        "fundamental_score": round(fundamental_score, 4),
        "technical_score": round(technical_score, 4),
        "liquidity_score": round(liquidity_score, 4),
        "risk_score": round(risk_score, 4),
        "macro_score": (
            None if macro_score is None else round(macro_score, 4)
        ),
        "macro_context": macro_public,
        "effective_score_weights": effective_weights,
        "total_score": total_score,
        "classification": classification,
        "selection_reasons": selection_reasons,
        "rejection_reasons": hard_failures,
        "warnings": warnings,
        "mover_type": snapshot.mover_type,
        "mover_return": snapshot.mover_return,
        "daily_return": daily_return,
        "benchmark_symbol": snapshot.benchmark_symbol,
        "price_source": snapshot.price_source,
        "data_timestamps": {
            "price_session": latest_date.isoformat(),
            "price_source_timestamp": (
                None
                if snapshot.price_source_timestamp is None
                else snapshot.price_source_timestamp.isoformat()
            ),
            "fundamental_available_at": (
                None
                if snapshot.fundamental is None
                or snapshot.fundamental.available_at is None
                else snapshot.fundamental.available_at.isoformat()
            ),
            "shariah_screened_at": (
                None
                if snapshot.shariah.screened_at is None
                else snapshot.shariah.screened_at.isoformat()
            ),
            "shariah_expires_at": (
                None
                if snapshot.shariah.expires_at is None
                else snapshot.shariah.expires_at.isoformat()
            ),
        },
        "config_version": config.version,
        "config_hash": config.config_hash,
        "screener_version": config.screener_version,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
    }
    private = {
        "fundamental_metrics": fundamental_metrics,
        "technical_metrics": technical_metrics,
        "liquidity_metrics": {
            "median_volume_20d": median_volume,
            "median_dollar_volume_20d": median_dollar_volume,
            "bid_ask_spread_bps": snapshot.bid_ask_spread_bps,
        },
        "risk_metrics": risk_metrics,
        "macro_context": macro_context,
        "provider_conflict": snapshot.provider_conflict_detail,
    }
    return ScoreResult(public=public, private=private)


def _macro_context_score(
    snapshot: AssetSnapshot,
    macro_context: dict[str, Any] | None,
) -> tuple[float | None, dict[str, Any]]:
    unavailable = {
        "status": "UNAVAILABLE",
        "macro_regime": "UNKNOWN",
        "sector_macro_tailwind": "UNKNOWN",
        "region_macro_tailwind": "UNKNOWN",
        "currency_regime": "UNKNOWN",
        "commodity_regime": "UNKNOWN",
        "market_breadth": None,
        "macro_confidence": 0.0,
        "macro_data_status": "DATA_INCOMPLETE",
        "order_signal": False,
    }
    if (
        not macro_context
        or macro_context.get("data_quality", {}).get("status") != "GO"
    ):
        return None, unavailable
    implications = macro_context["implications"]
    sector_key = str(snapshot.metadata.sector or "").strip().lower().replace(
        " ", "_"
    )
    sector = implications["sectors_and_asset_classes"].get(sector_key)
    currency = str(snapshot.metadata.currency or "").upper()
    region_key = (
        "eurozone" if currency == "EUR" else
        "united_states" if currency == "USD" else
        "global"
    )
    region = implications["regions"].get(region_key)
    values = [
        float(item["score"])
        for item in (sector, region)
        if item is not None
    ]
    if not values:
        return None, unavailable
    raw = max(-100.0, min(100.0, sum(values) / len(values)))
    score = 50.0 + raw / 2.0
    regime = macro_context["regime"]
    breadth = macro_context["scores"]["breadth"].get("value")
    return score, {
        "status": "GO",
        "macro_regime": regime["overall_macro_regime"],
        "sector_macro_tailwind": (
            "UNKNOWN" if sector is None else sector["macro_support"]
        ),
        "region_macro_tailwind": (
            "UNKNOWN" if region is None else region["macro_support"]
        ),
        "currency_regime": regime["currency_regime"],
        "commodity_regime": regime["commodity_regime"],
        "market_breadth": breadth,
        "macro_confidence": regime["confidence"],
        "macro_data_status": macro_context["data_quality"]["status"],
        "order_signal": False,
    }


def _fundamental_score(
    snapshot: FundamentalSnapshot | None,
    *,
    price: float,
    asset_type: str,
    knowledge_date: date,
    config: ScreenerConfig,
) -> tuple[float, dict[str, Any], list[str]]:
    if snapshot is None:
        if asset_type != "STOCK":
            return (
                0.0,
                {
                    "applicability": "ETF_HOLDINGS_LEVEL",
                    "coverage": 0.0,
                    "available_components": [],
                    "positive_earnings_or_fcf": False,
                },
                [],
            )
        return (
            0.0,
            {
                "applicability": "COMPANY_LEVEL",
                "coverage": 0.0,
                "available_components": [],
                "positive_earnings_or_fcf": False,
            },
            ["MISSING_FUNDAMENTAL_DATA"],
        )
    failures: list[str] = []
    if snapshot.available_at is None:
        failures.append("MISSING_FUNDAMENTAL_TIMESTAMP")
    elif snapshot.available_at.date() > knowledge_date:
        failures.append("LOOKAHEAD_FUNDAMENTAL_DATA_BLOCKED")
    elif (knowledge_date - snapshot.available_at.date()).days > 550:
        failures.append("STALE_FUNDAMENTAL_DATA")
    if snapshot.shares is None or snapshot.assets is None:
        failures.append("MISSING_CRITICAL_FUNDAMENTAL_DATA")
        return (
            0.0,
            {
                "applicability": "COMPANY_LEVEL",
                "coverage": 0.0,
                "available_components": [],
                "positive_earnings_or_fcf": False,
            },
            failures,
        )
    capitalization = market_cap(price, snapshot.shares)
    if capitalization < config.thresholds["minimum_market_cap"]:
        failures.append("MICRO_CAP_BLOCKED")

    component_weights = {
        "positive_earnings_or_fcf": 20.0,
        "earnings_yield": 15.0,
        "fcf_yield": 10.0,
        "profitability": 15.0,
        "debt": 15.0,
        "accruals": 10.0,
        "dilution": 10.0,
        "dividend_quality": 5.0,
    }
    components: dict[str, float | None] = {}
    positive = bool(
        (snapshot.net_income is not None and snapshot.net_income > 0)
        or (snapshot.free_cash_flow is not None and snapshot.free_cash_flow > 0)
    )
    components["positive_earnings_or_fcf"] = 100.0 if positive else 0.0
    ey = (
        None
        if snapshot.net_income is None
        else earnings_yield(snapshot.net_income, capitalization)
    )
    components["earnings_yield"] = None if ey is None else _linear_score(ey, -0.02, 0.08)
    fcf_yield = (
        None
        if snapshot.free_cash_flow is None
        else free_cash_flow_yield(snapshot.free_cash_flow, capitalization)
    )
    components["fcf_yield"] = (
        None if fcf_yield is None else _linear_score(fcf_yield, -0.02, 0.1)
    )
    profitability = (
        None if snapshot.net_income is None else roa(snapshot.net_income, snapshot.assets)
    )
    components["profitability"] = (
        None if profitability is None else _linear_score(profitability, -0.02, 0.2)
    )
    debt_ratio = None if snapshot.debt is None else snapshot.debt / snapshot.assets
    components["debt"] = (
        None if debt_ratio is None else _inverse_linear_score(debt_ratio, 0.0, 0.75)
    )
    accrual_ratio = (
        None
        if snapshot.net_income is None or snapshot.operating_cash_flow is None
        else accruals(snapshot.net_income, snapshot.operating_cash_flow, snapshot.assets)
    )
    components["accruals"] = (
        None if accrual_ratio is None else _inverse_linear_score(accrual_ratio, -0.1, 0.2)
    )
    dilution_ratio = (
        None
        if snapshot.previous_shares is None
        else dilution(snapshot.shares, snapshot.previous_shares)
    )
    components["dilution"] = (
        None if dilution_ratio is None else _inverse_linear_score(dilution_ratio, -0.05, 0.1)
    )
    dividend_ratio = (
        None
        if snapshot.dividends is None or snapshot.net_income is None
        else abs(snapshot.dividends) / max(snapshot.net_income, 1.0)
    )
    components["dividend_quality"] = (
        None
        if dividend_ratio is None
        else 100.0
        if 0.0 <= dividend_ratio <= 0.8
        else 20.0
    )
    available = [key for key, value in components.items() if value is not None]
    available_weight = sum(component_weights[key] for key in available)
    coverage = available_weight / sum(component_weights.values())
    if coverage < config.thresholds["minimum_fundamental_coverage"]:
        failures.append("INSUFFICIENT_FUNDAMENTAL_COVERAGE")
    if not positive:
        failures.append("NON_POSITIVE_EARNINGS_AND_FCF")
    score = (
        0.0
        if not available
        else sum(
            (components[key] or 0.0) * component_weights[key]
            for key in available
        )
        / available_weight
    )
    metrics = {
        "applicability": "COMPANY_LEVEL",
        "coverage": coverage,
        "available_components": available,
        "missing_components": [key for key in component_weights if key not in available],
        "positive_earnings_or_fcf": positive,
        "market_cap": capitalization,
        "earnings_yield": ey,
        "fcf_yield": fcf_yield,
        "profitability_roa_proxy": profitability,
        "net_margin": (
            None
            if snapshot.net_income is None or snapshot.revenue is None
            else margin(snapshot.net_income, snapshot.revenue)
        ),
        "debt_to_assets": debt_ratio,
        "accruals_to_assets": accrual_ratio,
        "dilution": dilution_ratio,
        "dividend_payout_proxy": dividend_ratio,
        "component_scores": components,
    }
    return float(np.clip(score, 0.0, 100.0)), metrics, failures


def _technical_score(
    bars: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[float, dict[str, Any], list[str]]:
    close = pd.to_numeric(bars["close"], errors="coerce").dropna()
    failures: list[str] = []
    if len(close) < 253:
        return 0.0, _empty_technical_metrics(), ["INSUFFICIENT_TECHNICAL_HISTORY"]
    values = close.astype(float).tolist()
    ema50 = exponential_moving_average(values, 50)
    ema200 = exponential_moving_average(values, 200)
    price = values[-1]
    mom3 = momentum(price, values[-64])
    mom6 = momentum(price, values[-127])
    mom12 = momentum(price, values[-253])
    returns = close.pct_change().dropna()
    volatility = float(returns.tail(63).std(ddof=0) * math.sqrt(252))
    true_ranges = _true_ranges(bars.tail(20))
    atr14 = float(true_ranges.tail(14).mean()) if not true_ranges.empty else math.nan
    trend_atr = None if not math.isfinite(atr14) or atr14 <= 0 else (ema50 - ema200) / atr14
    relative_volume = _relative_volume(bars)
    benchmark_momentum = None
    if not benchmark.empty and len(benchmark) >= 127:
        benchmark_close = pd.to_numeric(benchmark["close"], errors="coerce").dropna()
        if len(benchmark_close) >= 127:
            benchmark_momentum = momentum(
                float(benchmark_close.iloc[-1]),
                float(benchmark_close.iloc[-127]),
            )
    if benchmark_momentum is None:
        failures.append("MISSING_BENCHMARK_DATA")
    relative_6m = None if benchmark_momentum is None else relative_strength(mom6, benchmark_momentum)
    distance_ema50 = price / ema50 - 1.0
    distance_ema200 = price / ema200 - 1.0
    component_points = [
        15.0 if price > ema200 else 0.0,
        15.0 if ema50 > ema200 else 0.0,
        10.0 if mom3 > 0 else 0.0,
        10.0 if mom6 > 0 else 0.0,
        10.0 if mom12 > 0 else 0.0,
        10.0 if relative_6m is not None and relative_6m > 0 else 0.0,
        float(np.clip((trend_atr or 0.0) * 3.0, 0.0, 10.0)),
        _volatility_quality(volatility) * 10.0,
        5.0 if -0.1 <= distance_ema50 <= 0.2 else 0.0,
        float(np.clip(relative_volume / 2.0, 0.0, 1.0) * 5.0),
    ]
    metrics = {
        "ema50": ema50,
        "ema200": ema200,
        "price_above_ema200": price > ema200,
        "ema50_above_ema200": ema50 > ema200,
        "momentum_3m": mom3,
        "momentum_6m": mom6,
        "momentum_12m": mom12,
        "benchmark_momentum_6m": benchmark_momentum,
        "relative_strength_6m": relative_6m,
        "trend_strength_atr": trend_atr,
        "annualized_volatility_63d": volatility,
        "distance_ema50": distance_ema50,
        "distance_ema200": distance_ema200,
        "relative_volume_20d": relative_volume,
    }
    return float(np.clip(sum(component_points), 0.0, 100.0)), metrics, failures


def _liquidity_score(median_volume: float, median_dollar_volume: float) -> float:
    volume_score = _log_score(median_volume, 100_000.0, 5_000_000.0)
    dollar_score = _log_score(median_dollar_volume, 5_000_000.0, 250_000_000.0)
    return 0.35 * volume_score + 0.65 * dollar_score


def _risk_score(
    bars: pd.DataFrame,
    technical: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    close = pd.to_numeric(bars["close"], errors="coerce").dropna()
    returns = close.pct_change().dropna()
    volatility = (
        None if returns.empty else float(returns.tail(63).std(ddof=0) * math.sqrt(252))
    )
    rolling_peak = close.tail(252).cummax()
    drawdown = close.tail(252) / rolling_peak - 1.0
    maximum_drawdown = None if drawdown.empty else float(drawdown.min())
    true_ranges = _true_ranges(bars.tail(20))
    atr_ratio = (
        None
        if true_ranges.empty
        else float(true_ranges.tail(14).mean() / max(float(close.iloc[-1]), 1e-12))
    )
    if maximum_drawdown is not None and not math.isfinite(maximum_drawdown):
        maximum_drawdown = None
    if atr_ratio is not None and not math.isfinite(atr_ratio):
        atr_ratio = None
    distance = technical.get("distance_ema50")
    score = (
        40.0 * _volatility_quality(volatility)
        + 30.0 * _inverse_linear_score(abs(maximum_drawdown or 0.0), 0.05, 0.6) / 100.0
        + 20.0 * _inverse_linear_score(atr_ratio or 0.0, 0.01, 0.1) / 100.0
        + 10.0 * (1.0 if distance is not None and -0.15 <= distance <= 0.2 else 0.0)
    )
    return float(np.clip(score, 0.0, 100.0)), {
        "annualized_volatility": volatility,
        "maximum_drawdown_252d": maximum_drawdown,
        "atr_ratio_14d": atr_ratio,
        "overextended": distance is not None and distance > 0.3,
    }


def _true_ranges(bars: pd.DataFrame) -> pd.Series:
    required = {"high", "low", "close"}
    if not required.issubset(bars.columns):
        return pd.Series(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    raw_close = pd.to_numeric(bars.get("raw_close", bars["close"]), errors="coerce")
    previous = raw_close.shift(1)
    return pd.concat(
        [(high - low).abs(), (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)


def _relative_volume(bars: pd.DataFrame) -> float:
    volume = pd.to_numeric(bars.get("volume"), errors="coerce").dropna()
    if len(volume) < 20:
        return 0.0
    baseline = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.iloc[:-1].mean())
    return 0.0 if baseline <= 0 else float(volume.iloc[-1]) / baseline


def _volatility_quality(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    if 0.12 <= value <= 0.45:
        return 1.0
    if value < 0.12:
        return float(np.clip(value / 0.12, 0.0, 1.0))
    return float(np.clip(1.0 - (value - 0.45) / 0.75, 0.0, 1.0))


def _linear_score(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low) * 100.0, 0.0, 100.0))


def _inverse_linear_score(value: float, low: float, high: float) -> float:
    return 100.0 - _linear_score(value, low, high)


def _log_score(value: float, low: float, high: float) -> float:
    if value <= low:
        return 0.0
    if value >= high:
        return 100.0
    return float((math.log(value) - math.log(low)) / (math.log(high) - math.log(low)) * 100.0)


def _empty_technical_metrics() -> dict[str, Any]:
    return {
        "ema50": None,
        "ema200": None,
        "price_above_ema200": False,
        "ema50_above_ema200": False,
        "momentum_3m": None,
        "momentum_6m": None,
        "momentum_12m": None,
        "benchmark_momentum_6m": None,
        "relative_strength_6m": None,
        "trend_strength_atr": None,
        "annualized_volatility_63d": None,
        "distance_ema50": None,
        "distance_ema200": None,
        "relative_volume_20d": None,
    }


def _empty_result(
    snapshot: AssetSnapshot,
    screening_date: date,
    config: ScreenerConfig,
    failures: list[str],
    *,
    decision_time: datetime,
) -> ScoreResult:
    return ScoreResult(
        public={
            "screening_date": screening_date.isoformat(),
            "decision_time": decision_time.isoformat(),
            "asset_key": snapshot.metadata.asset_key,
            "symbol": snapshot.metadata.symbol,
            "con_id": snapshot.metadata.con_id,
            "asset_type": snapshot.metadata.asset_type,
            "exchange": snapshot.metadata.exchange,
            "currency": snapshot.metadata.currency,
            "sector": snapshot.metadata.sector,
            "shariah_status": snapshot.shariah.status,
            "price": None,
            "fundamental_score": 0.0,
            "technical_score": 0.0,
            "liquidity_score": 0.0,
            "risk_score": 0.0,
            "total_score": 0.0,
            "classification": "REJECTED",
            "selection_reasons": [],
            "rejection_reasons": failures,
            "warnings": [],
            "mover_type": snapshot.mover_type,
            "mover_return": snapshot.mover_return,
            "daily_return": None,
            "benchmark_symbol": snapshot.benchmark_symbol,
            "price_source": snapshot.price_source,
            "data_timestamps": {},
            "config_version": config.version,
            "config_hash": config.config_hash,
            "screener_version": config.screener_version,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "order_calls": 0,
        },
        private={},
    )
