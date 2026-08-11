from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.market import load_market_context_map
from stocks.signals.market_reference import latest_market_reference


INTERVALS = ("1h", "2h", "4h", "1d", "1w", "1mo")
PERIODS_PER_YEAR = {
    "1h": 1638.0,
    "2h": 819.0,
    "4h": 504.0,
    "1d": 252.0,
    "1w": 52.0,
    "1mo": 12.0,
}
FRESHNESS_HOURS = {
    "1h": 12.0,
    "2h": 24.0,
    "4h": 72.0,
    "1d": 240.0,
    "1w": 1008.0,
    "1mo": 1080.0,
}
INTERVAL_WEIGHTS = {
    "1h": 0.15,
    "2h": 0.20,
    "4h": 0.25,
    "1d": 0.25,
    "1w": 0.10,
    "1mo": 0.05,
}
SYMBOL_PATTERN = re.compile(r"[A-Z0-9.^=-]{1,20}")
UNIVERSE_PATH = Path("output/universe/instruments.parquet")
BAR_ROOT = Path("data/research/multitimeframe/private")
NEWS_PATH = Path("output/notifications/market-intelligence-digest.json")
OPPORTUNITY_PATH = Path("output/portfolio/opportunity_ranking.json")
SIGNALS_PATH = Path("output/signals/latest_signals.json")
ARCHITECTURE_PATH = Path(
    "output/research/phase11_10/architecture-summary.csv"
)


def analyze_asset(project_root: Path, symbol: str) -> dict[str, Any]:
    symbol = str(symbol).strip().upper()
    if not SYMBOL_PATTERN.fullmatch(symbol):
        return {
            "schema": "universe_asset_analysis_v1",
            "status": "BLOCKED",
            "reason": "INVALID_SYMBOL",
            "execution_authority": "NONE",
        }
    metadata = _metadata(project_root, symbol)
    if not metadata:
        return {
            "schema": "universe_asset_analysis_v1",
            "status": "NOT_IN_UNIVERSE",
            "symbol": symbol,
            "execution_authority": "NONE",
        }
    timeframe_rows: list[dict[str, Any]] = []
    for interval in INTERVALS:
        frame, provenance = _load_bars(project_root, symbol, interval)
        if frame.empty:
            timeframe_rows.append(
                {
                    "interval": interval,
                    "status": "NO_DATA",
                    "signal": "UNAVAILABLE",
                    "score": None,
                    "bar_count": 0,
                }
            )
            continue
        timeframe_rows.append(
            _timeframe_analysis(frame, interval, provenance)
        )
    observed = [
        row for row in timeframe_rows if row["status"] != "NO_DATA"
    ]
    available = [row for row in observed if row["status"] == "GO"]
    weighted = [
        (
            float(row["score"]),
            INTERVAL_WEIGHTS[str(row["interval"])],
        )
        for row in available
        if row.get("score") is not None
    ]
    weight_sum = sum(weight for _, weight in weighted)
    consensus_score = (
        sum(score * weight for score, weight in weighted) / weight_sum
        if weight_sum
        else None
    )
    stale_count = sum(row.get("freshness_status") == "STALE" for row in available)
    consensus = _consensus_label(consensus_score, stale_count, len(available))
    opportunity = _opportunity(project_root, symbol)
    current_market = _current_market_context(project_root, symbol)
    microstructure_context = load_market_context_map(project_root).get(
        symbol,
        {
            "status": "NO_CONTEXT_NEUTRAL_FALLBACK",
            "advisories": ["GEX_AND_ORDERFLOW_CONTEXT_UNAVAILABLE"],
            "standalone_entry_authority": False,
            "execution_authority": "NONE",
        },
    )
    recommended_action = _recommended_action(
        consensus,
        opportunity,
        stale_count,
        current_market,
    )
    report = {
        "schema": "universe_asset_analysis_v1",
        "status": "GO" if available else "DATA_UNAVAILABLE",
        "generated_at": datetime.now(UTC).isoformat(),
        "symbol": symbol,
        "metadata": metadata,
        "analysis_coverage": {
            "available_timeframe_count": len(available),
            "bar_file_timeframe_count": len(observed),
            "requested_timeframe_count": len(INTERVALS),
            "available_timeframes": [
                str(row["interval"]) for row in available
            ],
            "insufficient_history_timeframes": [
                str(row["interval"])
                for row in observed
                if row["status"] == "INSUFFICIENT_HISTORY"
            ],
            "missing_timeframes": [
                str(row["interval"])
                for row in timeframe_rows
                if row["status"] == "NO_DATA"
            ],
            "stale_timeframe_count": stale_count,
        },
        "multi_timeframe": {
            "score": _round(consensus_score),
            "classification": consensus,
            "alignment_ratio": _alignment_ratio(available),
            "fresh_entry_allowed": bool(available) and stale_count == 0,
            "timeframes": timeframe_rows,
        },
        "portfolio_context": opportunity,
        "current_market": current_market,
        "microstructure_context": microstructure_context,
        "strategy_research": _strategy_research(project_root),
        "news": _news_context(project_root, symbol),
        "external_research_links": _external_links(symbol, metadata),
        "decision": {
            "classification": "RESEARCH_AND_MANUAL_ANALYSIS_ONLY",
            "recommended_action": recommended_action,
            "reason_codes": sorted(
                set(
                    current_market.get("reason_codes", [])
                    + opportunity.get("deployment_blockers", [])
                    + microstructure_context.get("advisories", [])
                )
            ),
            "model_order_plan": _model_order_plan(
                project_root,
                symbol,
                recommended_action,
            ),
            "automatic_allocation": False,
            "automatic_submission": False,
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
        },
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    return report


def build_analysis_coverage(
    project_root: Path,
    *,
    publish: bool = True,
) -> dict[str, Any]:
    universe_path = project_root / UNIVERSE_PATH
    universe = (
        pd.read_parquet(universe_path)
        if universe_path.is_file()
        else pd.DataFrame()
    )
    symbols: dict[str, set[str]] = {}
    root = project_root / BAR_ROOT
    if root.is_dir():
        for path in root.glob(
            "provider=*/symbol=*/interval=*/source_interval=*/bars.parquet"
        ):
            partitions = {
                part.split("=", 1)[0]: part.split("=", 1)[1]
                for part in path.parts
                if "=" in part
            }
            symbol = partitions.get("symbol")
            interval = partitions.get("interval")
            if symbol and interval in INTERVALS:
                symbols.setdefault(symbol, set()).add(interval)
    universe_symbols = (
        set(universe["symbol"].astype(str).str.upper())
        if not universe.empty
        else set()
    )
    analyzable = universe_symbols.intersection(symbols)
    interval_counts = {
        interval: sum(interval in symbols[symbol] for symbol in analyzable)
        for interval in INTERVALS
    }
    report = {
        "schema": "universe_asset_analysis_coverage_v1",
        "status": "GO" if analyzable else "NO_DATA",
        "generated_at": datetime.now(UTC).isoformat(),
        "universe_instrument_count": len(universe_symbols),
        "analyzable_instrument_count": len(analyzable),
        "coverage_ratio": _round(
            len(analyzable) / len(universe_symbols)
            if universe_symbols
            else 0.0
        ),
        "interval_counts": interval_counts,
        "one_hour_instrument_count": interval_counts["1h"],
        "two_hour_instrument_count": interval_counts["2h"],
        "all_six_timeframes_count": sum(
            set(INTERVALS).issubset(symbols[symbol])
            for symbol in analyzable
        ),
        "analysis_semantics": (
            "EVERY_UNIVERSE_ASSET_RETURNS_METADATA; "
            "INDICATOR_ANALYSIS_REQUIRES_VALIDATED_LOCAL_BARS"
        ),
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    if publish:
        output = project_root / "output" / "analysis"
        output.mkdir(parents=True, exist_ok=True)
        (output / "universe-coverage.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def _metadata(project_root: Path, symbol: str) -> dict[str, Any]:
    path = project_root / UNIVERSE_PATH
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    rows = frame.loc[frame["symbol"].astype(str).str.upper().eq(symbol)]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    allowed = (
        "symbol",
        "name",
        "instrument_type",
        "asset_type",
        "exposure_type",
        "sector",
        "industry",
        "region",
        "country",
        "currency",
        "exchange",
        "active_listing",
        "signal_eligible",
        "live_executable",
        "eligibility_status",
        "compliance_status",
    )
    return {
        key: _native(row.get(key))
        for key in allowed
        if key in frame.columns
    }


def _load_bars(
    project_root: Path,
    symbol: str,
    interval: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = list(
        (project_root / BAR_ROOT).glob(
            f"provider=*/symbol={symbol}/interval={interval}/"
            "source_interval=*/bars.parquet"
        )
    )
    if not candidates:
        return pd.DataFrame(), {}
    profiles = [_bar_candidate_profile(path) for path in candidates]
    sufficient = [profile for profile in profiles if profile["row_count"] >= 30]
    pool = sufficient or profiles
    preferred = max(
        pool,
        key=lambda profile: (
            profile["last_timestamp_ns"],
            profile["row_count"],
            profile["provider_priority"],
            profile["modified_ns"],
        ),
    )["path"]
    frame = pd.read_parquet(preferred)
    if "is_partial" in frame:
        frame = frame.loc[~frame["is_partial"].fillna(False).astype(bool)]
    frame = frame.sort_values("timestamp_utc").drop_duplicates(
        "timestamp_utc", keep="last"
    )
    partitions = {
        part.split("=", 1)[0]: part.split("=", 1)[1]
        for part in preferred.parts
        if "=" in part
    }
    return frame, {
        "provider": partitions.get("provider", "LOCAL_CACHE"),
        "source_interval": partitions.get("source_interval", interval),
        "bar_origin": (
            str(frame["bar_origin"].iloc[-1])
            if not frame.empty and "bar_origin" in frame
            else "UNKNOWN"
        ),
    }


def _bar_candidate_profile(path: Path) -> dict[str, Any]:
    timestamps = pd.to_datetime(
        pd.read_parquet(path, columns=["timestamp_utc"])["timestamp_utc"],
        utc=True,
        errors="coerce",
    ).dropna()
    provider = next(
        (
            part.split("=", 1)[1]
            for part in path.parts
            if part.startswith("provider=")
        ),
        "",
    )
    return {
        "path": path,
        "row_count": len(timestamps),
        "last_timestamp_ns": (
            int(timestamps.max().value)
            if not timestamps.empty
            else -1
        ),
        "provider_priority": {
            "EODHD": 4,
            "YFINANCE": 3,
            "STOCKS_PIT_EODHD_LOCAL": 2,
            "STOCKS_YFINANCE_LOCAL": 1,
        }.get(provider, 0),
        "modified_ns": path.stat().st_mtime_ns,
    }


def _timeframe_analysis(
    frame: pd.DataFrame,
    interval: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    volume = pd.to_numeric(
        frame.get("volume", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    timestamp = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
    valid = close.notna() & high.notna() & low.notna() & timestamp.notna()
    close, high, low, volume, timestamp = (
        series.loc[valid].reset_index(drop=True)
        for series in (close, high, low, volume, timestamp)
    )
    if len(close) < 30:
        return {
            "interval": interval,
            "status": "INSUFFICIENT_HISTORY",
            "bar_count": len(close),
            "score": None,
            "signal": "UNAVAILABLE",
            **provenance,
        }
    returns = close.pct_change()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_hist = ema12 - ema26 - (ema12 - ema26).ewm(
        span=9, adjust=False
    ).mean()
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    std20 = close.rolling(20).std(ddof=0)
    atr14 = _atr(high, low, close, 14)
    adx14 = _adx(high, low, close, 14)
    rsi14 = _rsi(close, 14)
    cmf20 = _cmf(high, low, close, volume, 20)
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    obv_slope = _normalized_slope(obv.tail(20))
    donchian = high.shift(1).rolling(20).max()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    band_width = (upper - lower).replace(0.0, np.nan)
    band_position = (close - lower) / band_width
    trend_component = _bounded(
        0.45 * _sign(close.iloc[-1] - sma20.iloc[-1])
        + 0.35 * _sign(sma20.iloc[-1] - sma50.iloc[-1])
        + 0.20 * _normalized_slope(sma20.tail(20))
    )
    momentum_component = _bounded(
        0.40 * _sign(macd_hist.iloc[-1])
        + 0.35 * _bounded((rsi14.iloc[-1] - 50.0) / 20.0)
        + 0.25 * _bounded(
            (close.iloc[-1] / close.iloc[-min(21, len(close))] - 1.0)
            / max(returns.tail(60).std(), 0.005)
        )
    )
    breakout_component = _bounded(
        (close.iloc[-1] / donchian.iloc[-1] - 1.0)
        / max(float(atr14.iloc[-1] / close.iloc[-1]), 0.005)
        if pd.notna(donchian.iloc[-1])
        else 0.0
    )
    flow_component = _bounded(
        0.55 * _bounded(float(cmf20.iloc[-1]) * 4.0)
        + 0.45 * obv_slope
    )
    mean_reversion_component = _bounded(
        -(float(band_position.iloc[-1]) - 0.5) * 1.5
    )
    adx_multiplier = min(
        1.15, max(0.65, float(adx14.iloc[-1]) / 25.0)
    )
    score = _bounded(
        (
            0.35 * trend_component
            + 0.25 * momentum_component
            + 0.15 * breakout_component
            + 0.15 * flow_component
            + 0.10 * mean_reversion_component
        )
        * adx_multiplier
    )
    latest = _effective_bar_timestamp(timestamp.iloc[-1], interval)
    age_hours = max(
        0.0,
        (datetime.now(UTC) - latest).total_seconds() / 3600.0,
    )
    freshness = (
        "FRESH"
        if age_hours <= FRESHNESS_HOURS[interval]
        else "STALE"
    )
    rolling_peak = close.cummax()
    drawdown = close / rolling_peak - 1.0
    annualized_vol = (
        returns.tail(min(252, len(returns))).std(ddof=0)
        * math.sqrt(PERIODS_PER_YEAR[interval])
    )
    return {
        "interval": interval,
        "status": "GO",
        "signal": _score_label(score),
        "score": _round(score),
        "bar_count": len(close),
        "first_timestamp": timestamp.iloc[0].isoformat(),
        "last_timestamp": timestamp.iloc[-1].isoformat(),
        "age_hours": _round(age_hours),
        "freshness_status": freshness,
        "close": _round(close.iloc[-1]),
        "return_1_bar": _round(returns.iloc[-1]),
        "return_5_bars": _period_return(close, 5),
        "return_20_bars": _period_return(close, 20),
        "annualized_volatility": _round(annualized_vol),
        "maximum_drawdown": _round(drawdown.min()),
        "rsi_14": _round(rsi14.iloc[-1]),
        "adx_14": _round(adx14.iloc[-1]),
        "atr_pct": _round(atr14.iloc[-1] / close.iloc[-1]),
        "macd_histogram": _round(macd_hist.iloc[-1]),
        "bollinger_position": _round(band_position.iloc[-1]),
        "volume_ratio_20": _round(
            volume.iloc[-1] / max(volume.tail(20).mean(), 1.0)
        ),
        "cmf_20": _round(cmf20.iloc[-1]),
        "components": {
            "trend": _round(trend_component),
            "momentum": _round(momentum_component),
            "breakout": _round(breakout_component),
            "volume_flow": _round(flow_component),
            "mean_reversion": _round(mean_reversion_component),
        },
        **provenance,
    }


def _effective_bar_timestamp(
    timestamp: pd.Timestamp,
    interval: str,
) -> datetime:
    if interval == "1mo":
        return (timestamp + pd.offsets.MonthEnd(0)).to_pydatetime()
    if interval == "1w":
        return (timestamp + pd.Timedelta(days=6)).to_pydatetime()
    return timestamp.to_pydatetime()


def _opportunity(project_root: Path, symbol: str) -> dict[str, Any]:
    payload = _read_json(project_root / OPPORTUNITY_PATH)
    for row in payload.get("opportunities", []):
        if str(row.get("ticker", "")).upper() == symbol:
            return {
                "status": "RANKED",
                "opportunity_score": row.get("opportunity_score"),
                "multilayer_confluence": row.get(
                    "multilayer_confluence", {}
                ),
                "confluence_adjustment": row.get(
                    "confluence_adjustment"
                ),
                "research_allocation_eligible": row.get(
                    "research_allocation_eligible", False
                ),
                "deployment_eligible": False,
                "timeframes": row.get("timeframes", []),
                "strategy_families": row.get("strategy_families", []),
                "evidence_tiers": row.get("evidence_tiers", []),
                "signal_contract_currency_status": row.get(
                    "signal_contract_currency_status"
                ),
                "signal_currency": row.get("signal_currency"),
                "contract_currency": row.get("contract_currency"),
                "deployment_blockers": row.get(
                    "deployment_blockers", []
                ),
            }
    return {
        "status": "NOT_CURRENTLY_RANKED",
        "research_allocation_eligible": False,
        "deployment_eligible": False,
        "deployment_blockers": ["NO_CURRENT_PORTFOLIO_OPPORTUNITY"],
    }


def _current_market_context(
    project_root: Path,
    symbol: str,
) -> dict[str, Any]:
    reference = latest_market_reference(project_root, symbol)
    payload = _read_json(project_root / SIGNALS_PATH)
    rows = [
        row
        for row in payload.get("signals", [])
        if isinstance(row, dict)
        and str(row.get("ticker") or row.get("asset") or "").upper()
        == symbol
    ]
    valid = [
        row
        for row in rows
        if row.get("data_freshness") == "FRESH"
        and str(row.get("action") or "").upper()
        in {"BUY", "STRONG_BUY", "WATCHLIST"}
        and str(row.get("price_validity_status") or "")
        in {"", "CURRENT_ENTRY_REFERENCE_GO"}
    ]
    invalidated = [
        row
        for row in rows
        if str(row.get("lifecycle_status") or "").upper()
        == "INVALIDATED"
        or str(row.get("action") or "").upper() == "AVOID"
        or (
            row.get("price_validity_status")
            and row.get("price_validity_status")
            != "CURRENT_ENTRY_REFERENCE_GO"
        )
    ]
    reasons = {
        str(row.get("price_validity_status"))
        for row in invalidated
        if row.get("price_validity_status")
    }
    if reference.get("status") != "FRESH":
        state = "CURRENT_REFERENCE_STALE_OR_UNAVAILABLE"
        reasons.add(
            str(
                reference.get("reason")
                or "CURRENT_MARKET_REFERENCE_UNAVAILABLE_OR_STALE"
            )
        )
    elif valid:
        state = "CURRENT_ENTRY_SIGNAL_AVAILABLE"
    elif invalidated:
        state = "CURRENT_SIGNALS_INVALIDATED"
    else:
        state = "NO_CURRENT_ENTRY_SIGNAL"
        reasons.add("NO_CURRENT_CAUSAL_ENTRY_SIGNAL")
    return {
        "status": reference.get("status", "UNAVAILABLE"),
        "state": state,
        "price": _round(reference.get("price"), 4),
        "provider": reference.get("provider"),
        "timestamp": reference.get("timestamp"),
        "fetched_at": reference.get("fetched_at"),
        "age_minutes": _round(reference.get("age_minutes"), 2),
        "reference_kind": "INDICATIVE_INTRADAY_BAR_CLOSE",
        "executable_quote": False,
        "valid_signal_count": len(valid),
        "invalidated_signal_count": len(invalidated),
        "reason_codes": sorted(reasons),
        "entry_instruction": (
            "USE_CURRENT_CAUSAL_SIGNAL_CONDITIONALLY"
            if state == "CURRENT_ENTRY_SIGNAL_AVAILABLE"
            else "WAIT_FOR_NEW_CAUSAL_SIGNAL"
        ),
    }


def _model_order_plan(
    project_root: Path,
    symbol: str,
    recommended_action: str,
) -> dict[str, Any]:
    payload = _read_json(project_root / SIGNALS_PATH)
    candidates = [
        row
        for row in payload.get("signals", [])
        if str(row.get("ticker") or row.get("asset") or "").upper() == symbol
        and row.get("data_freshness") == "FRESH"
    ]
    candidates.sort(
        key=lambda row: float(row.get("confidence_score", 0.0)),
        reverse=True,
    )
    signal = candidates[0] if candidates else {}
    if recommended_action == "AVOID_NEW_LONG_RISK_REVIEW_EXISTING_POSITION":
        return {
            "model_action": "SELL_REVIEW_IF_HELD",
            "side": "SELL",
            "quantity": "REQUIRES_RECONCILED_LONG_POSITION",
            "status": "BLOCKED_WITHOUT_POSITION_IDENTITY",
            "execution_authority": "NONE",
        }
    if recommended_action == "AVOID_NEW_LONG_WAIT_FOR_NEW_CAUSAL_SIGNAL":
        return {
            "model_action": "AVOID_NEW_LONG",
            "side": None,
            "status": "CURRENT_PRICE_INVALIDATED_ENTRY",
            "execution_authority": "NONE",
        }
    if not signal or "CANDIDATE" not in recommended_action:
        return {
            "model_action": "NO_ORDER",
            "status": "NO_CURRENT_ACTIONABLE_SETUP",
            "execution_authority": "NONE",
        }
    return {
        "model_action": "BUY_SETUP",
        "side": "BUY",
        "order_type": "LIMIT",
        "quantity": signal.get("suggested_quantity"),
        "limit_price": signal.get("limit_entry_price"),
        "entry_zone_low": signal.get("entry_zone_low"),
        "entry_zone_high": signal.get("entry_zone_high"),
        "protective_stop": signal.get("stop_loss"),
        "target_1": signal.get("take_profit_1"),
        "target_2": signal.get("take_profit_2"),
        "reward_risk_1": signal.get("reward_risk_1"),
        "status": "DRAFT_NOT_SUBMITTABLE_AUTHORITY_NONE",
        "execution_authority": "NONE",
    }


def _strategy_research(project_root: Path) -> dict[str, Any]:
    path = project_root / ARCHITECTURE_PATH
    if not path.is_file():
        return {"status": "NO_DATA", "timeframes": {}}
    frame = pd.read_csv(path)
    result: dict[str, list[dict[str, Any]]] = {}
    for interval in ("1h", "2h", "4h", "1d", "1w"):
        rows = frame.loc[
            frame["lower_timeframe"].eq(interval)
            & frame["median_oos_CAGR"].gt(0)
            & frame["median_oos_portfolio_pf"].gt(1)
            & frame["median_fill_count"].ge(30)
        ].copy()
        rows["research_score"] = (
            rows["median_oos_Sharpe"].clip(-1, 2)
            + rows["median_oos_portfolio_pf"].clip(0, 2)
            + rows["positive_fold_ratio"]
            + rows["cost_50bps_median_pf"].clip(0, 2)
        )
        columns = [
            "architecture",
            "entry_strategy",
            "higher_timeframe",
            "middle_timeframe",
            "lower_timeframe",
            "fold_count",
            "median_oos_CAGR",
            "median_oos_Sharpe",
            "median_oos_portfolio_pf",
            "cost_50bps_median_pf",
            "worst_oos_drawdown",
            "positive_fold_ratio",
        ]
        result[interval] = (
            rows.sort_values("research_score", ascending=False)[columns]
            .head(5)
            .replace({np.nan: None})
            .to_dict(orient="records")
        )
    return {
        "status": "GO",
        "scope": "UNIVERSE_LEVEL_ARCHITECTURE_EVIDENCE_NOT_ASSET_SPECIFIC",
        "timeframes": result,
        "execution_authority": "NONE",
    }


def _news_context(project_root: Path, symbol: str) -> dict[str, Any]:
    payload = _read_json(project_root / NEWS_PATH)
    matched = []
    for row in payload.get("important_news", []):
        symbols = [str(value).upper() for value in row.get("symbols", [])]
        if symbol not in symbols:
            continue
        title = str(row.get("title", "")).strip()
        matched.append(
            {
                "title": title,
                "source": row.get("source"),
                "published_at": row.get("published_at"),
                "importance": row.get("importance"),
                "direction": row.get("direction"),
                "sentiment_polarity": row.get("sentiment_polarity"),
                "external_search_url": (
                    "https://news.google.com/search?q="
                    + quote_plus(f"{title} {symbol}")
                ),
                "link_semantics": "SEARCH_LINK_NOT_ARCHIVED_SOURCE_URL",
            }
        )
    return {
        "status": payload.get("news_freshness_status", "UNAVAILABLE"),
        "matched_count": len(matched),
        "items": matched[:20],
        "news_is_context_only": True,
        "automatic_execution": False,
    }


def _external_links(
    symbol: str,
    metadata: dict[str, Any],
) -> list[dict[str, str]]:
    links = [
        {
            "label": "Yahoo Finance news",
            "url": f"https://finance.yahoo.com/quote/{quote_plus(symbol)}/news/",
        },
        {
            "label": "Google News",
            "url": (
                "https://news.google.com/search?q="
                + quote_plus(f"{symbol} {metadata.get('name', '')}")
            ),
        },
        {
            "label": "TradingView chart",
            "url": (
                "https://www.tradingview.com/chart/?symbol="
                + quote_plus(symbol)
            ),
        },
    ]
    if metadata.get("instrument_type") == "STOCK":
        links.append(
            {
                "label": "SEC filings search",
                "url": (
                    "https://www.sec.gov/edgar/search/#/q="
                    + quote_plus(
                        str(metadata.get("name") or symbol)
                    )
                ),
            }
        )
    return links


def _recommended_action(
    consensus: str,
    opportunity: dict[str, Any],
    stale_count: int,
    current_market: dict[str, Any],
) -> str:
    market_state = str(current_market.get("state") or "")
    if opportunity.get("signal_contract_currency_status") == (
        "SIGNAL_CONTRACT_CURRENCY_MISMATCH"
    ):
        return "BLOCKED_SIGNAL_CONTRACT_CURRENCY_REVIEW"
    if market_state == "CURRENT_REFERENCE_STALE_OR_UNAVAILABLE":
        return "REFRESH_INTRADAY_REFERENCE_BEFORE_NEW_DECISION"
    if market_state == "CURRENT_SIGNALS_INVALIDATED":
        return "AVOID_NEW_LONG_WAIT_FOR_NEW_CAUSAL_SIGNAL"
    if stale_count:
        return "REFRESH_DATA_BEFORE_NEW_DECISION"
    if consensus == "BULLISH_ALIGNMENT" and opportunity.get(
        "research_allocation_eligible"
    ):
        return "RESEARCH_ALLOCATION_CANDIDATE_MANUAL_REVIEW"
    if consensus == "BULLISH_ALIGNMENT":
        return "WATCHLIST_RESEARCH_ONLY"
    if consensus == "BEARISH_ALIGNMENT":
        return "AVOID_NEW_LONG_RISK_REVIEW_EXISTING_POSITION"
    return "NO_NEW_POSITION_WAIT_FOR_ALIGNMENT"


def _consensus_label(
    score: float | None,
    stale_count: int,
    available_count: int,
) -> str:
    if score is None or not available_count:
        return "DATA_UNAVAILABLE"
    if stale_count == available_count:
        return "STALE_DATA_BLOCKED"
    if score >= 0.35:
        return "BULLISH_ALIGNMENT"
    if score <= -0.35:
        return "BEARISH_ALIGNMENT"
    return "MIXED_OR_NEUTRAL"


def _alignment_ratio(rows: list[dict[str, Any]]) -> float | None:
    scores = [
        float(row["score"])
        for row in rows
        if row.get("score") is not None
    ]
    if not scores:
        return None
    positive = sum(value > 0.15 for value in scores)
    negative = sum(value < -0.15 for value in scores)
    return _round(max(positive, negative) / len(scores))


def _score_label(score: float) -> str:
    if score >= 0.55:
        return "STRONG_BULLISH"
    if score >= 0.20:
        return "BULLISH"
    if score <= -0.55:
        return "STRONG_BEARISH"
    if score <= -0.20:
        return "BEARISH"
    return "NEUTRAL"


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(
        alpha=1.0 / period, adjust=False
    ).mean()
    loss = -delta.clip(upper=0.0).ewm(
        alpha=1.0 / period, adjust=False
    ).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, adjust=False).mean()


def _adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0.0), 0.0)
    minus_dm = down.where((down > up) & (down > 0.0), 0.0)
    atr = _atr(high, low, close, period).replace(0.0, np.nan)
    plus_di = 100.0 * plus_dm.ewm(
        alpha=1.0 / period, adjust=False
    ).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(
        alpha=1.0 / period, adjust=False
    ).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (
        plus_di + minus_di
    ).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0.0)


def _cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int,
) -> pd.Series:
    spread = (high - low).replace(0.0, np.nan)
    multiplier = ((close - low) - (high - close)) / spread
    flow = multiplier.fillna(0.0) * volume
    return flow.rolling(period).sum() / volume.rolling(period).sum().replace(
        0.0, np.nan
    )


def _normalized_slope(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope = float(np.polyfit(x, values.to_numpy(dtype=float), 1)[0])
    scale = float(np.nanstd(values.to_numpy(dtype=float)))
    return _bounded(slope * len(values) / max(scale, 1e-9) / 4.0)


def _period_return(close: pd.Series, periods: int) -> float | None:
    if len(close) <= periods:
        return None
    return _round(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def _sign(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return 1.0 if number > 0 else -1.0 if number < 0 else 0.0


def _bounded(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return max(-1.0, min(1.0, number))


def _round(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}
