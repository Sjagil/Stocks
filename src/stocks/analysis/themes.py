from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stocks.execution.idempotency import stable_hash


CONFIG_PATH = Path("config/themes/frontier_technology_energy_v1.json")
BAR_ROOT = Path("data/research/multitimeframe/private")
NEWS_PATH = Path("data/news/private/current-news.json")
THEME_NEWS_PATH = Path("output/analysis/themes/theme-news.json")
THEME_EVENT_RISK_PATH = Path(
    "output/analysis/themes/event-risk-calendar.json"
)
THEME_CONTRACTS_PATH = Path(
    "output/analysis/themes/contract-coverage.json"
)
SIGNALS_PATH = Path("output/signals/latest_signals.json")
ENTRY_SHORTLIST_PATH = Path("output/market_context/entry-shortlist.json")
ENTRY_COMPLETENESS_PATH = Path(
    "output/market_context/entry-episode-completeness.json"
)
FUNDAMENTAL_DATABASE = Path(
    "data/research/phase11_3/private/causal_research.sqlite3"
)
THEME_FUNDAMENTALS_PATH = Path(
    "output/analysis/themes/fundamental-coverage.json"
)
THEME_SHARIAH_PATH = Path("output/analysis/themes/shariah-coverage.json")
MACRO_STATUS_PATH = Path("output/macro/status.json")
MACRO_SCORE_PATH = Path("output/macro/score.json")
ASSET_TRANSMISSION_PATH = Path("config/context/asset_transmission_v1.json")
OUTPUT_ROOT = Path("output/analysis/themes")
INTERVALS = ("1h", "4h", "1d", "1w", "1mo")
PERIODS_PER_YEAR = {
    "1h": 1638.0,
    "4h": 504.0,
    "1d": 252.0,
    "1w": 52.0,
    "1mo": 12.0,
}
FRESHNESS_HOURS = {
    "1h": 72.0,
    "4h": 96.0,
    "1d": 120.0,
    "1w": 336.0,
    "1mo": 1080.0,
}
TIMEFRAME_WEIGHTS = {
    "1h": 0.15,
    "4h": 0.25,
    "1d": 0.35,
    "1w": 0.20,
    "1mo": 0.05,
}
MIN_ANALYSIS_BARS = 30


def build_frontier_theme_analysis(
    project_root: Path,
    *,
    theme: str | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Publish bounded weekend research for configured frontier themes."""
    config = _read_json(project_root / CONFIG_PATH)
    themes = config.get("themes", {})
    if theme and theme not in themes:
        return _blocked("UNKNOWN_THEME", theme=theme)
    selected = {theme: themes[theme]} if theme else themes
    if not selected:
        return _blocked("THEME_CONFIG_UNAVAILABLE", theme=theme)

    now = as_of or datetime.now(UTC)
    symbols = {
        str(row["symbol"]).upper()
        for definition in selected.values()
        for row in definition.get("instruments", [])
    }
    news = _news_by_symbol(project_root, symbols, now)
    theme_news = _theme_news_by_theme(project_root, set(selected), now)
    signals = _signals_by_symbol(project_root, symbols)
    forward_observations = _forward_observations_by_symbol(
        project_root,
        symbols,
    )
    episode_completeness = _read_json(
        project_root / ENTRY_COMPLETENESS_PATH
    )
    fundamentals = _fundamental_coverage(project_root, symbols)
    event_risk = _event_risk_coverage(project_root, symbols)
    contracts = _contract_coverage(project_root, symbols)
    shariah = _shariah_coverage(project_root, symbols)
    macro = _macro_context(project_root)
    reports: dict[str, dict[str, Any]] = {}
    all_rows: list[dict[str, Any]] = []

    for theme_id, definition in selected.items():
        current_theme_news = theme_news.get(theme_id, [])
        theme_macro = _macro_with_theme_policy_context(
            macro,
            theme_id=theme_id,
            theme_news=current_theme_news,
            as_of=now,
        )
        rows = []
        for instrument in definition.get("instruments", []):
            symbol = str(instrument["symbol"]).upper()
            timeframes = {
                interval: _analyze_timeframe(
                    project_root,
                    symbol,
                    interval,
                    now,
                )
                for interval in INTERVALS
            }
            row = _instrument_row(
                theme_id,
                instrument,
                timeframes,
                news.get(symbol, []),
                fundamentals.get(symbol),
                event_risk.get(symbol),
                contracts.get(symbol),
                shariah.get(symbol),
                signals.get(symbol, []),
                forward_observations.get(symbol, []),
                theme_macro,
            )
            rows.append(row)
            all_rows.append(row)
        rows.sort(
            key=lambda item: (
                -float(item.get("technical_score") or -2.0),
                str(item["symbol"]),
            )
        )
        report = _theme_report(
            theme_id,
            definition,
            rows,
            theme_macro,
            config,
            now,
            current_theme_news,
        )
        reports[theme_id] = report

    payload = {
        "schema": "frontier_technology_energy_theme_analysis_v1",
        "status": _overall_status(reports.values()),
        "generated_at": now.isoformat(),
        "analysis_cutoff": now.isoformat(),
        "weekend_semantics": (
            "LAST_COMPLETED_MARKET_BARS_ONLY; NO WEEKEND PRICE SYNTHESIS"
        ),
        "config_version": config.get("version"),
        "config_hash": stable_hash(config),
        "theme_count": len(reports),
        "instrument_count": len(all_rows),
        "themes": reports,
        "macro_context": macro,
        "forward_evidence": {
            key: episode_completeness.get(key)
            for key in (
                "status",
                "episode_count",
                "terminal_episode_count",
                "pending_episode_count",
                "completion_ratio",
                "feature_mutation_count",
                "duplicate_terminal_count",
            )
        },
        "data_sources": {
            "ohlcv": "VALIDATED_LOCAL_MULTI_TIMEFRAME_CACHE",
            "news": "CURRENT_NEWS_PRIVATE_STORE_CONTEXT_ONLY",
            "fundamentals": "CAUSAL_RESEARCH_SEC_FILING_COVERAGE",
            "event_risk": "POINT_IN_TIME_MULTI_SOURCE_EVENT_CALENDAR",
            "shariah": "CURRENT_EXPIRING_MANUAL_ATTESTATION_GATE",
            "contract_identity": "IBKR_TWS_LIVE_STRICT_READ_ONLY",
            "forward_observation": "IMMUTABLE_ENTRY_OBSERVER_EPISODES",
            "macro": "MACRO_ENGINE_RESEARCH_ONLY",
            "theme_policy": (
                "CURRENT_OFFICIAL_PUBLIC_RSS_CONTEXT_ONLY"
            ),
        },
        "authority": {
            "research_only": True,
            "standalone_entry_allowed": False,
            "strategy_authority": "NONE",
            "execution_authority": "NONE",
            "orders_generated": 0,
            "broker_calls": 0,
        },
    }
    payload["content_hash"] = stable_hash(payload)
    _publish(project_root, payload, reports)
    return payload


def _instrument_row(
    theme_id: str,
    instrument: dict[str, Any],
    timeframes: dict[str, dict[str, Any]],
    news: list[dict[str, Any]],
    fundamentals: dict[str, Any] | None,
    event_risk: dict[str, Any] | None,
    contract: dict[str, Any] | None,
    shariah: dict[str, Any] | None,
    signals: list[dict[str, Any]],
    forward_observations: list[dict[str, Any]],
    macro: dict[str, Any],
) -> dict[str, Any]:
    available = [
        row for row in timeframes.values() if row.get("status") == "GO"
    ]
    weighted = [
        (float(row["score"]), TIMEFRAME_WEIGHTS[row["interval"]])
        for row in available
        if row.get("score") is not None
    ]
    weight = sum(value for _, value in weighted)
    score = sum(value * part for value, part in weighted) / weight if weight else None
    daily = timeframes.get("1d", {})
    positive = sum(float(row.get("score") or 0.0) > 0.15 for row in available)
    negative = sum(float(row.get("score") or 0.0) < -0.15 for row in available)
    maturity = str(instrument.get("business_maturity") or "").upper()
    fundamentals_required = not any(
        marker in maturity for marker in ("FUND", "VEHICLE")
    )
    fundamental_context = (
        fundamentals
        if fundamentals
        else {
            "status": (
                "UNAVAILABLE"
                if fundamentals_required
                else "NOT_APPLICABLE_VEHICLE"
            ),
            "fundamentals_required": fundamentals_required,
            "filing_record_count": 0,
            "latest_filing_at": None,
        }
    )
    fundamental_quality = _fundamental_quality(fundamental_context)
    news_quality = _news_quality(news)
    macro_alignment = _macro_alignment(theme_id, macro, instrument)
    contextual_conviction = _contextual_conviction(
        technical_score=score,
        fundamental=fundamental_quality,
        news=news_quality,
        macro=macro_alignment,
        fundamentals_required=fundamentals_required,
    )
    contract_context = contract or {
        "status": "UNAVAILABLE",
        "source": "NO_RESOLVED_THEME_CONTRACT",
        "contract_identity": None,
    }
    shariah_context = shariah or {
        "status": "SHARIAH_ATTESTATION_REQUIRED",
        "currently_eligible": False,
        "screened_at": None,
        "expires_at": None,
        "methodology": None,
        "source": None,
    }
    return {
        "theme": theme_id,
        "symbol": str(instrument["symbol"]).upper(),
        "subtheme": instrument.get("subtheme"),
        "business_maturity": instrument.get("business_maturity"),
        "technical_score": _round(score),
        "technical_classification": _classification(score),
        "timeframe_alignment": {
            "classification": (
                "BULLISH_ALIGNMENT"
                if positive >= 3 and negative == 0
                else "BEARISH_ALIGNMENT"
                if negative >= 3 and positive == 0
                else "MIXED"
            ),
            "positive_timeframe_count": positive,
            "negative_timeframe_count": negative,
            "available_timeframe_count": len(available),
        },
        "daily_snapshot": {
            key: daily.get(key)
            for key in (
                "last_timestamp",
                "freshness_status",
                "close",
                "return_5_bars",
                "return_20_bars",
                "return_63_bars",
                "return_126_bars",
                "return_252_bars",
                "distance_sma_20",
                "distance_sma_50",
                "distance_sma_200",
                "rsi_14",
                "atr_pct_20",
                "realized_volatility_20",
                "drawdown_from_252_bar_high",
                "average_daily_value_20",
            )
        },
        "timeframes": timeframes,
        "news": {
            "fresh_event_count": len(news),
            "items": news[:5],
            "context_score": news_quality,
            "catalyst_summary": _news_catalyst_summary(news),
            "authority": "RANKING_CONTEXT_ONLY",
        },
        "fundamentals": fundamental_context,
        "event_risk": event_risk
        or {
            "event_risk_status": "EVENT_DATE_UNCERTAIN",
            "hard_block_recommended": True,
            "soft_penalty_recommended": False,
            "authority": "RISK_CONTEXT_ONLY",
        },
        "fundamental_quality": fundamental_quality,
        "contract": contract_context,
        "macro_alignment": macro_alignment,
        "contextual_conviction": contextual_conviction,
        "current_strategy_setups": {
            "observed_count": len(signals),
            "valid_setup_count": sum(
                row.get("lifecycle_status") not in {"INVALIDATED", "EXPIRED"}
                and row.get("action") not in {"AVOID", "BLOCKED"}
                for row in signals
            ),
            "items": signals[:5],
            "technical_theme_score_is_not_an_entry": True,
        },
        "current_forward_observations": {
            "observed_count": len(forward_observations),
            "hard_veto_pass_count": sum(
                bool(row.get("hard_veto_pass"))
                for row in forward_observations
            ),
            "state_counts": dict(
                sorted(
                    Counter(
                        str(row.get("state") or "UNKNOWN")
                        for row in forward_observations
                    ).items()
                )
            ),
            "items": forward_observations,
            "existing_episode_features_are_immutable": True,
            "authority": "OBSERVATION_ONLY",
        },
        "shariah": shariah_context,
        "shariah_status": shariah_context["status"],
        "automatic_execution_eligible": False,
    }


def _analyze_timeframe(
    project_root: Path,
    symbol: str,
    interval: str,
    now: datetime,
) -> dict[str, Any]:
    frame, provenance = _load_bars(project_root, symbol, interval)
    if frame.empty:
        return {
            "interval": interval,
            "status": "NO_DATA",
            "bar_count": 0,
            "score": None,
        }
    required = {"timestamp_utc", "high", "low", "close"}
    if not required.issubset(frame.columns):
        return {
            "interval": interval,
            "status": "INVALID_SCHEMA",
            "bar_count": len(frame),
            "score": None,
        }
    work = frame.copy()
    if "is_partial" in work:
        work = work.loc[~work["is_partial"].fillna(False).astype(bool)]
    if "partial_bucket" in work:
        work = work.loc[~work["partial_bucket"].fillna(False).astype(bool)]
    work["timestamp_utc"] = pd.to_datetime(
        work["timestamp_utc"], utc=True, errors="coerce"
    )
    for column in ("high", "low", "close", "volume"):
        if column in work:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    work = (
        work.dropna(subset=["timestamp_utc", "high", "low", "close"])
        .sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
    )
    if len(work) < MIN_ANALYSIS_BARS:
        return {
            "interval": interval,
            "status": "INSUFFICIENT_HISTORY",
            "bar_count": len(work),
            "score": None,
            **provenance,
        }
    close = work["close"]
    high = work["high"]
    low = work["low"]
    volume = work.get("volume", pd.Series(0.0, index=work.index)).fillna(0.0)
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    returns = close.pct_change()
    rsi14 = _rsi(close, 14)
    atr20 = _atr(high, low, close, 20)
    trend = np.mean(
        [
            _sign(close.iloc[-1] - sma20.iloc[-1]),
            _sign(close.iloc[-1] - sma50.iloc[-1]),
            _sign(close.iloc[-1] - sma200.iloc[-1])
            if pd.notna(sma200.iloc[-1])
            else 0.0,
        ]
    )
    momentum = np.mean(
        [
            _bounded(_return(close, 5) / 0.05),
            _bounded(_return(close, 20) / 0.15),
            _bounded((float(rsi14.iloc[-1]) - 50.0) / 25.0),
        ]
    )
    score = _bounded(0.60 * float(trend) + 0.40 * float(momentum))
    latest = work["timestamp_utc"].iloc[-1].to_pydatetime()
    age_hours = max(0.0, (now - latest).total_seconds() / 3600.0)
    annualized = returns.tail(20).std(ddof=0) * math.sqrt(
        PERIODS_PER_YEAR[interval]
    )
    rolling_high = close.tail(min(252, len(close))).max()
    adv20 = (close * volume).tail(20).mean()
    return {
        "interval": interval,
        "status": "GO",
        "score": _round(score),
        "classification": _classification(score),
        "bar_count": len(work),
        "first_timestamp": work["timestamp_utc"].iloc[0].isoformat(),
        "last_timestamp": work["timestamp_utc"].iloc[-1].isoformat(),
        "age_hours": _round(age_hours),
        "freshness_status": (
            "FRESH"
            if age_hours <= FRESHNESS_HOURS[interval]
            else "STALE"
        ),
        "close": _round(close.iloc[-1]),
        "return_5_bars": _round(_return(close, 5)),
        "return_20_bars": _round(_return(close, 20)),
        "return_63_bars": _round(_return(close, 63)),
        "return_126_bars": _round(_return(close, 126)),
        "return_252_bars": _round(_return(close, 252)),
        "distance_sma_20": _distance(close.iloc[-1], sma20.iloc[-1]),
        "distance_sma_50": _distance(close.iloc[-1], sma50.iloc[-1]),
        "distance_sma_200": _distance(close.iloc[-1], sma200.iloc[-1]),
        "rsi_14": _round(rsi14.iloc[-1]),
        "atr_pct_20": _round(atr20.iloc[-1] / close.iloc[-1]),
        "realized_volatility_20": _round(annualized),
        "drawdown_from_252_bar_high": _round(
            close.iloc[-1] / rolling_high - 1.0
        ),
        "average_daily_value_20": _round(adv20),
        **provenance,
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
    profiles: list[dict[str, Any]] = []
    for path in candidates:
        try:
            candidate = pd.read_parquet(path)
        except (OSError, ValueError, KeyError):
            continue
        required = {"timestamp_utc", "high", "low", "close"}
        schema_valid = required.issubset(candidate.columns)
        usable = candidate.copy() if schema_valid else pd.DataFrame()
        if not usable.empty:
            if "is_partial" in usable:
                usable = usable.loc[
                    ~usable["is_partial"].fillna(False).astype(bool)
                ]
            if "partial_bucket" in usable:
                usable = usable.loc[
                    ~usable["partial_bucket"].fillna(False).astype(bool)
                ]
            usable["timestamp_utc"] = pd.to_datetime(
                usable["timestamp_utc"],
                utc=True,
                errors="coerce",
            )
            for column in ("high", "low", "close"):
                usable[column] = pd.to_numeric(
                    usable[column],
                    errors="coerce",
                )
            usable = (
                usable.dropna(
                    subset=["timestamp_utc", "high", "low", "close"]
                )
                .sort_values("timestamp_utc")
                .drop_duplicates("timestamp_utc", keep="last")
            )
        provider = next(
            (
                part.split("=", 1)[1]
                for part in path.parts
                if part.startswith("provider=")
            ),
            "UNKNOWN",
        )
        profiles.append(
            {
                "path": path,
                "provider": provider,
                "rows": len(candidate),
                "usable_rows": len(usable),
                "schema_valid": schema_valid,
                "latest": (
                    usable["timestamp_utc"].max()
                    if not usable.empty
                    else None
                ),
            }
        )
    if not profiles:
        return pd.DataFrame(), {}
    profile = max(
        profiles,
        key=lambda item: (
            item["usable_rows"] >= MIN_ANALYSIS_BARS,
            item["latest"] or pd.Timestamp.min.tz_localize("UTC"),
            item["usable_rows"],
        ),
    )
    return pd.read_parquet(profile["path"]), {
        "provider": profile["provider"],
        "bar_origin": "LOCAL_VALIDATED_CACHE",
        "native_or_derived": (
            "DERIVED" if interval in {"4h", "1w", "1mo"} else "NATIVE"
        ),
        "provider_candidate_count": len(profiles),
        "selected_profile_rows": profile["usable_rows"],
        "selection_reason": (
            "SUFFICIENT_HISTORY_THEN_FRESHNESS_THEN_ROWS"
        ),
    }


def _theme_report(
    theme_id: str,
    definition: dict[str, Any],
    rows: list[dict[str, Any]],
    macro: dict[str, Any],
    config: dict[str, Any],
    now: datetime,
    theme_news: list[dict[str, Any]],
) -> dict[str, Any]:
    available = [row for row in rows if row["technical_score"] is not None]
    daily = [
        row["daily_snapshot"]
        for row in available
        if row["daily_snapshot"].get("close") is not None
    ]
    fresh = [
        row
        for row in available
        if row["daily_snapshot"].get("freshness_status") == "FRESH"
    ]
    required_fundamentals = [
        row
        for row in rows
        if row["fundamentals"].get("fundamentals_required", True)
    ]
    fundamental_count = sum(
        row["fundamentals"].get("status") == "AVAILABLE"
        for row in required_fundamentals
    )
    fundamental_decision_usable_count = sum(
        bool(row["fundamentals"].get("data_quality", {}).get("decision_usable"))
        for row in required_fundamentals
    )
    contract_count = sum(
        row["contract"].get("status") == "RESOLVED" for row in rows
    )
    shariah_count = sum(
        bool(row["shariah"].get("currently_eligible")) for row in rows
    )
    news_count = (
        sum(row["news"]["fresh_event_count"] for row in rows)
        + len(theme_news)
    )
    combined_theme_news = [
        item for row in rows for item in row["news"]["items"]
    ] + theme_news
    conviction_rows = sorted(
        rows,
        key=lambda row: (
            -float(row["contextual_conviction"].get("score") or -1.0),
            row["symbol"],
        ),
    )
    forward_rows = [
        item
        for row in rows
        for item in row["current_forward_observations"]["items"]
    ]
    report = {
        "schema": "thematic_market_analysis_v1",
        "status": (
            "GO"
            if len(fresh) == len(rows)
            and fundamental_count == len(required_fundamentals)
            and fundamental_decision_usable_count == len(required_fundamentals)
            and contract_count == len(rows)
            and shariah_count == len(rows)
            else "GO_WITH_DOCUMENTED_GAPS"
        ),
        "generated_at": now.isoformat(),
        "theme_id": theme_id,
        "label": definition.get("label"),
        "instrument_count": len(rows),
        "technical_coverage_count": len(available),
        "fresh_daily_count": len(fresh),
        "fundamental_coverage_count": fundamental_count,
        "fundamental_required_count": len(required_fundamentals),
        "fundamental_coverage_ratio": _ratio(
            fundamental_count, len(required_fundamentals)
        ),
        "fundamental_decision_usable_count": fundamental_decision_usable_count,
        "fundamental_decision_usable_ratio": _ratio(
            fundamental_decision_usable_count,
            len(required_fundamentals),
        ),
        "contract_coverage_count": contract_count,
        "contract_coverage_ratio": _ratio(contract_count, len(rows)),
        "current_shariah_eligible_count": shariah_count,
        "current_shariah_coverage_ratio": _ratio(shariah_count, len(rows)),
        "forward_observation_coverage": {
            "observed_setup_count": len(forward_rows),
            "observed_symbol_count": len(
                {row["symbol"] for row in rows if row["current_forward_observations"]["observed_count"]}
            ),
            "hard_veto_pass_count": sum(
                bool(row.get("hard_veto_pass")) for row in forward_rows
            ),
            "state_counts": dict(
                sorted(
                    Counter(
                        str(row.get("state") or "UNKNOWN")
                        for row in forward_rows
                    ).items()
                )
            ),
            "execution_authority": "NONE",
        },
        "fresh_news_event_count": news_count,
        "theme_wide_news": {
            "fresh_event_count": len(theme_news),
            "items": theme_news[:10],
            "catalyst_summary": _news_catalyst_summary(
                combined_theme_news
            ),
            "authority": "THEME_CONTEXT_ONLY",
        },
        "theme_policy_context": macro.get("theme_policy_context"),
        "breadth": {
            "above_sma_20_ratio": _ratio(
                sum(float(row.get("distance_sma_20") or -1.0) > 0 for row in daily),
                len(daily),
            ),
            "above_sma_50_ratio": _ratio(
                sum(float(row.get("distance_sma_50") or -1.0) > 0 for row in daily),
                len(daily),
            ),
            "above_sma_200_ratio": _ratio(
                sum(float(row.get("distance_sma_200") or -1.0) > 0 for row in daily),
                len(daily),
            ),
        },
        "cross_section": {
            "median_return_20d": _median(daily, "return_20_bars"),
            "median_return_63d": _median(daily, "return_63_bars"),
            "median_return_252d": _median(daily, "return_252_bars"),
            "median_realized_volatility_20d": _median(
                daily, "realized_volatility_20"
            ),
            "median_drawdown_from_252d_high": _median(
                daily, "drawdown_from_252_bar_high"
            ),
        },
        "sector_structure": _theme_sector_structure(theme_id, rows),
        "leadership": [
            {
                "rank": index,
                "symbol": row["symbol"],
                "subtheme": row["subtheme"],
                "technical_score": row["technical_score"],
                "classification": row["technical_classification"],
                "return_20d": row["daily_snapshot"].get("return_20_bars"),
                "return_63d": row["daily_snapshot"].get("return_63_bars"),
                "alignment": row["timeframe_alignment"]["classification"],
                "risk_class": row["business_maturity"],
            }
            for index, row in enumerate(rows, start=1)
        ],
        "contextual_conviction_leadership": [
            {
                "rank": index,
                "symbol": row["symbol"],
                "score": row["contextual_conviction"].get("score"),
                "classification": row["contextual_conviction"].get(
                    "classification"
                ),
                "technical_score": row["technical_score"],
                "fundamental_score": row["fundamental_quality"].get(
                    "score"
                ),
                "news_score": row["news"]["context_score"].get("score"),
                "macro_alignment_score": row["macro_alignment"].get(
                    "score"
                ),
                "not_an_entry_signal": True,
            }
            for index, row in enumerate(conviction_rows, start=1)
        ],
        "instruments": rows,
        "macro_sensitivities": definition.get("macro_sensitivities", []),
        "official_context_sources": definition.get(
            "official_context_sources", []
        ),
        "macro_context": macro,
        "documented_gaps": _gaps(rows, macro),
        "authority_contract": config.get("authority_contract", {}),
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _theme_sector_structure(
    theme_id: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Describe theme-specific confirmation without creating a trade signal."""
    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cohorts[str(row.get("subtheme") or "unclassified")].append(row)
    cohort_metrics = {
        name: _cohort_structure(group)
        for name, group in sorted(cohorts.items())
    }

    if theme_id == "nuclear_uranium":
        physical = cohort_metrics.get("physical_uranium")
        funds = cohort_metrics.get("uranium_fund")
        fuel_cycle = cohort_metrics.get("fuel_cycle")
        required = (physical, funds, fuel_cycle)
        available = all(
            item and item["structure_evaluable_count"] > 0
            for item in required
        )
        confirmations = [
            bool(physical and physical["positive_structure_ratio"] >= 1.0),
            bool(funds and funds["positive_structure_ratio"] >= 2 / 3),
            bool(
                fuel_cycle
                and fuel_cycle["positive_structure_ratio"] >= 0.6
            ),
        ]
        confirmation_count = sum(confirmations)
        status = (
            "INSUFFICIENT_DATA"
            if not available
            else "PHYSICAL_MINER_CONFIRMATION"
            if confirmation_count == 3
            else "PARTIAL_CONFIRMATION"
            if confirmation_count == 2
            else "DIVERGENT_OR_WEAK"
        )
        return {
            "schema": "nuclear_uranium_structure_v1",
            "status": status,
            "physical_proxy_positive": confirmations[0],
            "uranium_fund_breadth_positive": confirmations[1],
            "fuel_cycle_breadth_positive": confirmations[2],
            "confirmation_component_count": confirmation_count,
            "cohorts": cohort_metrics,
            "semantics": (
                "PHYSICAL_PROXY_PLUS_URANIUM_FUNDS_PLUS_FUEL_CYCLE; "
                "CONTEXT_ONLY_NOT_AN_ENTRY"
            ),
            "standalone_entry_allowed": False,
        }

    if theme_id == "quantum_computing":
        pure_play = cohort_metrics.get("pure_play")
        platform = cohort_metrics.get("platform_enabler")
        available = all(
            item and item["structure_evaluable_count"] > 0
            for item in (pure_play, platform)
        )
        pure_positive = bool(
            pure_play and pure_play["positive_structure_ratio"] >= 0.5
        )
        platform_positive = bool(
            platform and platform["positive_structure_ratio"] >= 0.5
        )
        status = (
            "INSUFFICIENT_DATA"
            if not available
            else "BROAD_CONFIRMATION"
            if pure_positive and platform_positive
            else "SPECULATIVE_PURE_PLAY_LED"
            if pure_positive
            else "PLATFORM_LED"
            if platform_positive
            else "BROAD_WEAKNESS"
        )
        return {
            "schema": "quantum_ecosystem_structure_v1",
            "status": status,
            "pure_play_breadth_positive": pure_positive,
            "platform_breadth_positive": platform_positive,
            "cohorts": cohort_metrics,
            "semantics": (
                "PURE_PLAY_VERSUS_DIVERSIFIED_PLATFORM_CONFIRMATION; "
                "CONTEXT_ONLY_NOT_AN_ENTRY"
            ),
            "standalone_entry_allowed": False,
        }

    return {
        "schema": "generic_theme_structure_v1",
        "status": "DESCRIPTIVE_ONLY",
        "cohorts": cohort_metrics,
        "standalone_entry_allowed": False,
    }


def _cohort_structure(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [
        row
        for row in rows
        if row.get("daily_snapshot", {}).get("return_63_bars") is not None
        and row.get("daily_snapshot", {}).get("distance_sma_50") is not None
    ]
    positive = [
        row
        for row in evaluable
        if float(row["daily_snapshot"]["return_63_bars"]) > 0
        and float(row["daily_snapshot"]["distance_sma_50"]) > 0
    ]
    snapshots = [row["daily_snapshot"] for row in evaluable]
    return {
        "instrument_count": len(rows),
        "structure_evaluable_count": len(evaluable),
        "positive_structure_count": len(positive),
        "positive_structure_ratio": _ratio(len(positive), len(evaluable)),
        "median_return_20d": _median(snapshots, "return_20_bars"),
        "median_return_63d": _median(snapshots, "return_63_bars"),
        "median_distance_sma_50": _median(snapshots, "distance_sma_50"),
        "positive_symbols": sorted(row["symbol"] for row in positive),
    }


def _news_by_symbol(
    project_root: Path,
    symbols: set[str],
    now: datetime,
) -> dict[str, list[dict[str, Any]]]:
    payloads = [
        _read_json(project_root / NEWS_PATH),
        _read_json(project_root / THEME_NEWS_PATH),
    ]
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for payload in payloads:
        for row in payload.get("rows", []):
            published = pd.to_datetime(
                row.get("published_at"), utc=True, errors="coerce"
            )
            if pd.isna(published):
                continue
            age_hours = (
                now - published.to_pydatetime()
            ).total_seconds() / 3600.0
            if age_hours > 168:
                continue
            for symbol in set(map(str.upper, row.get("symbols", []))) & symbols:
                event_hash = str(
                    row.get("event_hash")
                    or stable_hash(
                        {
                            "published_at": published.isoformat(),
                            "source": row.get("source"),
                            "title": row.get("title"),
                        }
                    )
                )
                if event_hash in seen[symbol]:
                    continue
                seen[symbol].add(event_hash)
                output[symbol].append(
                    {
                        "event_hash": event_hash,
                        "published_at": published.isoformat(),
                        "source": row.get("source"),
                        "source_class": row.get("source_class"),
                        "title": row.get("title"),
                        "direction": row.get("direction"),
                        "importance": row.get("importance"),
                        "event_type": row.get("event_type"),
                        "sentiment_polarity": row.get(
                            "sentiment_polarity"
                        ),
                    }
                )
    for rows in output.values():
        rows.sort(key=lambda row: str(row["published_at"]), reverse=True)
    return output


def _theme_news_by_theme(
    project_root: Path,
    themes: set[str],
    now: datetime,
) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(project_root / THEME_NEWS_PATH)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in payload.get("rows", []):
        if row.get("symbols"):
            continue
        published = pd.to_datetime(
            row.get("published_at"), utc=True, errors="coerce"
        )
        if pd.isna(published):
            continue
        age_hours = (now - published.to_pydatetime()).total_seconds() / 3600.0
        if age_hours > 168:
            continue
        for theme in set(map(str, row.get("themes", []))) & themes:
            event_hash = str(row.get("event_hash") or stable_hash(row))
            if event_hash in seen[theme]:
                continue
            seen[theme].add(event_hash)
            output[theme].append(
                {
                    "event_hash": event_hash,
                    "published_at": published.isoformat(),
                    "source": row.get("source"),
                    "source_class": row.get("source_class"),
                    "title": row.get("title"),
                    "direction": row.get("direction"),
                    "importance": row.get("importance"),
                    "event_type": row.get("event_type"),
                }
            )
    for rows in output.values():
        rows.sort(key=lambda row: str(row["published_at"]), reverse=True)
    return output


def _fundamental_coverage(
    project_root: Path,
    symbols: set[str],
) -> dict[str, dict[str, Any]]:
    current = _read_json(project_root / THEME_FUNDAMENTALS_PATH).get(
        "instruments", {}
    )
    output = {
        str(symbol).upper(): row
        for symbol, row in current.items()
        if isinstance(row, dict) and str(symbol).upper() in symbols
    }
    path = project_root / FUNDAMENTAL_DATABASE
    if not path.is_file() or not symbols:
        return output
    placeholders = ",".join("?" for _ in symbols)
    query = f"""
        SELECT UPPER(json_extract(payload_json, '$.symbol')) AS symbol,
               COUNT(*) AS record_count,
               MAX(COALESCE(
                   json_extract(payload_json, '$.accepted_at'),
                   json_extract(payload_json, '$.filing_date'),
                   json_extract(payload_json, '$.report_date')
               )) AS latest_filing_at
        FROM records
        WHERE dataset = 'filings'
          AND UPPER(json_extract(payload_json, '$.symbol')) IN ({placeholders})
        GROUP BY symbol
    """
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(query, tuple(sorted(symbols))).fetchall()
    except sqlite3.Error:
        return output
    legacy = {
        str(symbol): {
            "status": "AVAILABLE",
            "fundamentals_required": True,
            "filing_record_count": int(count),
            "latest_filing_at": latest,
            "semantics": "SEC_FILING_COVERAGE_NOT_A_VALUATION_SCORE",
        }
        for symbol, count, latest in rows
    }
    return {**legacy, **output}


def _event_risk_coverage(
    project_root: Path,
    symbols: set[str],
) -> dict[str, dict[str, Any]]:
    payload = _read_json(project_root / THEME_EVENT_RISK_PATH)
    return {
        str(row.get("symbol") or "").upper(): row
        for row in payload.get("rows", [])
        if str(row.get("symbol") or "").upper() in symbols
    }


def _contract_coverage(
    project_root: Path,
    symbols: set[str],
) -> dict[str, dict[str, Any]]:
    payload = _read_json(project_root / THEME_CONTRACTS_PATH)
    output: dict[str, dict[str, Any]] = {}
    for row in payload.get("results", []):
        symbol = str(row.get("research_symbol") or "").upper()
        if symbol not in symbols:
            continue
        output[symbol] = {
            "status": row.get("status", "UNAVAILABLE"),
            "source": row.get("source", "UNAVAILABLE"),
            "cache_hit": bool(row.get("cache_hit")),
            "returned_match_count": int(
                row.get("returned_match_count") or 0
            ),
            "contract_identity": row.get("contract_identity"),
            "authority": "IDENTITY_CONTEXT_ONLY",
        }
    return output


def _shariah_coverage(
    project_root: Path,
    symbols: set[str],
) -> dict[str, dict[str, Any]]:
    payload = _read_json(project_root / THEME_SHARIAH_PATH)
    return {
        str(row.get("symbol") or "").upper(): row
        for row in payload.get("instruments", [])
        if str(row.get("symbol") or "").upper() in symbols
    }


def _signals_by_symbol(
    project_root: Path,
    symbols: set[str],
) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(project_root / SIGNALS_PATH)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("signals", []):
        symbol = str(row.get("ticker") or row.get("symbol") or "").upper()
        if symbol not in symbols:
            continue
        output[symbol].append(
            {
                "signal_id": row.get("signal_id"),
                "strategy_id": row.get("strategy_id"),
                "timeframe": row.get("timeframe"),
                "action": row.get("action"),
                "lifecycle_status": row.get("lifecycle_status"),
                "signal_timestamp": row.get("signal_timestamp"),
                "data_timestamp": row.get("data_timestamp"),
                "price_validity_status": row.get("price_validity_status"),
                "entry_instruction": row.get("entry_instruction"),
                "market_reference_status": row.get("market_reference_status"),
                "market_reference_price": row.get("market_reference_price"),
                "market_reference_timestamp": row.get(
                    "market_reference_timestamp"
                ),
                "market_reference_fetched_at": row.get(
                    "market_reference_fetched_at"
                ),
                "market_reference_provider": row.get(
                    "market_reference_provider"
                ),
                "market_reference_kind": row.get("market_reference_kind"),
                "market_reference_is_executable_quote": bool(
                    row.get("market_reference_is_executable_quote", False)
                ),
                "preferred_entry": row.get("preferred_entry"),
                "entry_zone_low": row.get("entry_zone_low"),
                "entry_zone_high": row.get("entry_zone_high"),
                "invalidation_level": row.get("invalidation_level"),
                "stop_loss": row.get("stop_loss"),
                "stop_method": row.get("stop_method"),
                "take_profit_1": row.get("take_profit_1"),
                "take_profit_2": row.get("take_profit_2"),
                "take_profit_mode": row.get("take_profit_mode"),
                "reward_risk_1": row.get("reward_risk_1"),
                "reward_risk_2": row.get("reward_risk_2"),
                "expiration_timestamp": row.get("expiration_timestamp"),
                "source_provider": row.get("source_provider"),
                "source_interval": row.get("source_interval"),
                "bar_closed": bool(row.get("bar_closed", False)),
                "reasons": list(row.get("reasons") or []),
                "risks": list(row.get("risks") or []),
                "execution_eligible": bool(row.get("execution_eligible", False)),
            }
        )
    for rows in output.values():
        rows.sort(key=lambda row: str(row.get("signal_timestamp")), reverse=True)
    return output


def _forward_observations_by_symbol(
    project_root: Path,
    symbols: set[str],
) -> dict[str, list[dict[str, Any]]]:
    payload = _read_json(project_root / ENTRY_SHORTLIST_PATH)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("observations", []):
        symbol = str(row.get("symbol") or "").upper()
        if symbol not in symbols:
            continue
        decision = row.get("decision_contract") or {}
        gates = decision.get("gates") or {}
        contract = decision.get("contract_identity") or {}
        output[symbol].append(
            {
                "signal_id": row.get("signal_id"),
                "strategy_id": row.get("strategy_id"),
                "timeframe": row.get("timeframe"),
                "state": row.get("state"),
                "hard_veto_pass": bool(decision.get("hard_veto_pass")),
                "hard_vetoes": list(decision.get("hard_vetoes") or []),
                "soft_vetoes": list(decision.get("soft_vetoes") or []),
                "contract_status": contract.get("status"),
                "timeframe_hierarchy_ready": bool(
                    gates.get("timeframe_hierarchy_ready")
                ),
                "observed_tape_available": bool(
                    gates.get("observed_tape_available")
                ),
                "observed_depth_available": bool(
                    gates.get("observed_depth_available")
                ),
                "execution_authority": "NONE",
            }
        )
    for rows in output.values():
        rows.sort(
            key=lambda row: (
                not bool(row.get("hard_veto_pass")),
                str(row.get("strategy_id") or ""),
            )
        )
    return output


def _macro_context(project_root: Path) -> dict[str, Any]:
    status = _read_json(project_root / MACRO_STATUS_PATH)
    score = _read_json(project_root / MACRO_SCORE_PATH)
    transmission = _read_json(project_root / ASSET_TRANSMISSION_PATH)
    cycle = score.get("cycle_clock", {})
    data_quality = score.get("data_quality", {})
    regime = score.get("regime", {})
    family_scores: dict[str, dict[str, Any]] = {}
    for family, row in (score.get("scores") or {}).items():
        if not isinstance(row, dict):
            continue
        value = row.get("value")
        confidence = row.get("confidence")
        family_scores[str(family)] = {
            "value": (
                max(-1.0, min(1.0, float(value) / 100.0))
                if isinstance(value, (int, float)) and math.isfinite(value)
                else None
            ),
            "confidence": (
                max(0.0, min(1.0, float(confidence)))
                if isinstance(confidence, (int, float))
                and math.isfinite(confidence)
                else 0.0
            ),
            "status": str(row.get("status") or "UNKNOWN"),
        }
    groups = transmission.get("groups") or {}
    transmission_profiles = {
        key: dict((groups.get(key) or {}).get("sensitivities") or {})
        for key in (
            "technology_equity",
            "industrial_equity",
            "utility_equity",
            "uranium",
        )
    }
    return {
        "status": status.get("status", "UNAVAILABLE"),
        "as_of": score.get("as_of") or status.get("latest_as_of"),
        "regime": status.get("latest_regime")
        or regime.get("overall_macro_regime"),
        "market_regime": regime.get("market_regime"),
        "commodity_regime": regime.get("commodity_regime"),
        "liquidity_regime": regime.get("liquidity_regime"),
        "regime_confidence": regime.get("confidence"),
        "quadrant": cycle.get("quadrant"),
        "confidence": cycle.get("confidence"),
        "liquidity_overlay": cycle.get("liquidity_overlay"),
        "credit_overlay": cycle.get("credit_overlay"),
        "data_quality": data_quality.get("status")
        or status.get("latest_data_quality"),
        "feature_availability": status.get("feature_availability", {}),
        "family_scores": family_scores,
        "transmission_profiles": transmission_profiles,
        "transmission_source": str(ASSET_TRANSMISSION_PATH).replace("\\", "/"),
        "transmission_source_hash": (
            stable_hash(transmission) if transmission else None
        ),
        "predictive_claim": False,
        "authority": "CONTEXT_ONLY",
    }


def _fundamental_quality(context: dict[str, Any]) -> dict[str, Any]:
    if not context.get("fundamentals_required", True):
        return {
            "status": "NOT_APPLICABLE_VEHICLE",
            "score": None,
            "component_count": 0,
            "interpretation": "VEHICLE_REQUIRES_HOLDINGS_OR_NAV_ANALYSIS",
        }
    if context.get("status") != "AVAILABLE":
        return {
            "status": "UNAVAILABLE",
            "score": None,
            "component_count": 0,
        }
    components: dict[str, float] = {}
    data_quality = context.get("data_quality") or {}
    anomalous = set(data_quality.get("anomalous_metric_fields") or [])
    growth = context.get("annual_revenue_growth")
    growth_field = "annual_revenue_growth"
    if growth is None or growth_field in anomalous:
        growth = context.get("quarter_revenue_yoy_growth")
        growth_field = "quarter_revenue_yoy_growth"
    if growth is not None and growth_field not in anomalous:
        components["growth"] = _unit_scale(float(growth), -0.10, 0.50)
    margin = context.get("annual_net_margin")
    if margin is not None and "annual_net_margin" not in anomalous:
        components["net_margin"] = _unit_scale(float(margin), -0.20, 0.20)
    free_cash_flow = context.get("annual_free_cash_flow_margin")
    if (
        free_cash_flow is not None
        and "annual_free_cash_flow_margin" not in anomalous
    ):
        components["free_cash_flow_margin"] = _unit_scale(
            float(free_cash_flow), -0.20, 0.20
        )
    cash = context.get("cash_to_assets")
    debt = context.get("debt_to_assets")
    if (
        (cash is not None or debt is not None)
        and "cash_to_assets" not in anomalous
        and "debt_to_assets" not in anomalous
    ):
        components["balance_sheet"] = max(
            0.0,
            min(1.0, 0.5 + float(cash or 0.0) - float(debt or 0.0)),
        )
    raw_score = (
        float(np.mean(list(components.values()))) if components else None
    )
    quality_status = str(data_quality.get("status") or "UNKNOWN")
    shrinkage = 0.6 if quality_status == "GO" else 0.25
    score = (
        max(0.2, min(0.8, 0.5 + shrinkage * (raw_score - 0.5)))
        if raw_score is not None
        else None
    )
    return {
        "status": (
            "AVAILABLE"
            if score is not None and quality_status == "GO"
            else "LIMITED_RESEARCH_CONTEXT"
            if score is not None
            else "INSUFFICIENT_METRICS"
        ),
        "score": _round(score),
        "raw_score_before_uncertainty_shrinkage": _round(raw_score),
        "component_count": len(components),
        "components": {key: _round(value) for key, value in components.items()},
        "data_quality_status": quality_status,
        "excluded_anomalous_metric_fields": sorted(anomalous),
        "uncertainty_shrinkage_factor": shrinkage,
        "semantics": "BOUNDED_CONTEXT_SCORE_NOT_A_VALUATION_OR_ENTRY_SIGNAL",
    }


def _news_quality(news: list[dict[str, Any]]) -> dict[str, Any]:
    if not news:
        return {
            "status": "NO_FRESH_COMPANY_EVENT",
            "score": None,
            "event_count": 0,
        }
    summary = _news_catalyst_summary(news)
    return {
        "status": "AVAILABLE_CONTEXT_ONLY",
        "score": summary["bounded_context_score"],
        "event_count": len(news),
        "evidence_quality": summary["evidence_quality"],
        "directional_balance": summary["directional_balance"],
        "source_concentration_ratio": summary[
            "source_concentration_ratio"
        ],
    }


def _news_catalyst_summary(news: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize catalysts with shrinkage for sparse or concentrated news."""
    directions = Counter(str(row.get("direction") or "UNKNOWN") for row in news)
    sources = Counter(str(row.get("source") or "UNKNOWN") for row in news)
    source_classes = Counter(
        str(row.get("source_class") or "UNKNOWN") for row in news
    )
    event_types = Counter(
        str(row.get("event_type") or "UNCLASSIFIED") for row in news
    )
    weighted_positive = 0.0
    weighted_negative = 0.0
    for row in news:
        weight = 1.5 if row.get("importance") == "HIGH" else 1.0
        direction = str(row.get("direction") or "")
        if direction in {"POSITIVE", "POSITIVE_CONTEXT"}:
            weighted_positive += weight
        elif direction in {"NEGATIVE", "NEGATIVE_CONTEXT"}:
            weighted_negative += weight
    directional_weight = weighted_positive + weighted_negative
    raw_balance = (
        (weighted_positive - weighted_negative) / directional_weight
        if directional_weight
        else 0.0
    )
    directional_count = sum(
        directions.get(value, 0)
        for value in (
            "POSITIVE",
            "POSITIVE_CONTEXT",
            "NEGATIVE",
            "NEGATIVE_CONTEXT",
        )
    )
    source_count = len(sources)
    concentration = max(sources.values(), default=0) / max(len(news), 1)
    diversity_factor = min(1.0, source_count / 3.0)
    sample_factor = min(1.0, directional_count / 5.0)
    shrunk_balance = raw_balance * diversity_factor * sample_factor
    score = max(0.35, min(0.65, 0.5 + 0.15 * shrunk_balance))
    conflicting = weighted_positive > 0 and weighted_negative > 0
    evidence_quality = (
        "NO_DIRECTIONAL_EVIDENCE"
        if directional_count == 0
        else "SPARSE"
        if directional_count < 2
        else "SOURCE_CONCENTRATED"
        if concentration > 0.8 or source_count < 2
        else "DIVERSE_CONFLICTING"
        if conflicting
        else "DIVERSE_DIRECTIONAL"
    )
    classification = (
        "POSITIVE_CATALYST_BALANCE"
        if shrunk_balance >= 0.2
        else "NEGATIVE_RISK_BALANCE"
        if shrunk_balance <= -0.2
        else "MIXED_OR_INSUFFICIENT"
    )
    return {
        "status": "AVAILABLE_CONTEXT_ONLY" if news else "NO_FRESH_EVENT",
        "classification": classification,
        "evidence_quality": evidence_quality,
        "event_count": len(news),
        "directional_event_count": directional_count,
        "positive_event_count": directions.get("POSITIVE", 0)
        + directions.get("POSITIVE_CONTEXT", 0),
        "negative_event_count": directions.get("NEGATIVE", 0)
        + directions.get("NEGATIVE_CONTEXT", 0),
        "mixed_or_neutral_event_count": len(news) - directional_count,
        "distinct_source_count": source_count,
        "source_concentration_ratio": _round(concentration),
        "directional_balance_before_shrinkage": _round(raw_balance),
        "directional_balance": _round(shrunk_balance),
        "bounded_context_score": _round(score),
        "source_class_counts": dict(sorted(source_classes.items())),
        "event_type_counts": dict(sorted(event_types.items())),
        "predictive_claim": False,
        "standalone_entry_allowed": False,
    }


def _macro_alignment(
    theme_id: str,
    macro: dict[str, Any],
    instrument: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_id, sensitivities = _theme_macro_profile(
        theme_id,
        instrument or {},
        macro.get("transmission_profiles") or {},
    )
    family_scores = macro.get("family_scores") or {}
    components: list[dict[str, Any]] = []
    numerator = 0.0
    denominator = 0.0
    available_weight = 0.0
    confidence_weight = 0.0
    total_weight = sum(abs(float(value)) for value in sensitivities.values())
    for family, raw_sensitivity in sensitivities.items():
        sensitivity = float(raw_sensitivity)
        family_row = family_scores.get(family)
        if not isinstance(family_row, dict) or family_row.get("value") is None:
            components.append(
                {
                    "family": family,
                    "sensitivity": _round(sensitivity),
                    "status": "UNAVAILABLE",
                    "observed_score": None,
                    "confidence": 0.0,
                    "weighted_contribution": None,
                }
            )
            continue
        observed = max(-1.0, min(1.0, float(family_row["value"])))
        confidence = max(
            0.0,
            min(1.0, float(family_row.get("confidence") or 0.0)),
        )
        contribution = sensitivity * observed
        numerator += contribution * confidence
        denominator += abs(sensitivity) * confidence
        available_weight += abs(sensitivity)
        confidence_weight += abs(sensitivity) * confidence
        components.append(
            {
                "family": family,
                "sensitivity": _round(sensitivity),
                "status": str(family_row.get("status") or "UNKNOWN"),
                "observed_score": _round(observed),
                "confidence": _round(confidence),
                "weighted_contribution": _round(contribution),
            }
        )
    if not denominator or not total_weight:
        return {
            "status": "UNAVAILABLE",
            "profile": profile_id,
            "score": None,
            "raw_transmission_score": None,
            "confidence": 0.0,
            "coverage_ratio": 0.0,
            "components": components,
            "reasons": ["MACRO_FAMILY_SCORES_UNAVAILABLE"],
            "predictive_claim": False,
            "standalone_entry_allowed": False,
            "authority": "CONTEXT_ONLY",
        }
    raw_score = max(-1.0, min(1.0, numerator / denominator))
    coverage = available_weight / total_weight
    confidence = confidence_weight / total_weight
    shrunk_score = raw_score * coverage * confidence
    ranked = sorted(
        (
            row
            for row in components
            if row.get("weighted_contribution") is not None
        ),
        key=lambda row: abs(float(row["weighted_contribution"])),
        reverse=True,
    )
    reasons = [
        f"{str(row['family']).upper()}_"
        f"{'SUPPORTIVE' if float(row['weighted_contribution']) > 0 else 'HEADWIND'}"
        for row in ranked[:3]
        if abs(float(row["weighted_contribution"])) >= 0.01
    ]
    missing = [
        str(row["family"])
        for row in components
        if row.get("status") == "UNAVAILABLE"
    ]
    context_score = max(0.0, min(1.0, 0.5 + 0.5 * shrunk_score))
    return {
        "status": (
            "AVAILABLE_CONTEXT_ONLY"
            if coverage >= 0.75 and macro.get("data_quality") == "GO"
            else "DEGRADED_CONTEXT_ONLY"
        ),
        "profile": profile_id,
        "score": _round(context_score),
        "raw_transmission_score": _round(raw_score),
        "uncertainty_shrunk_score": _round(shrunk_score),
        "classification": (
            "SUPPORTIVE"
            if shrunk_score >= 0.10
            else "ADVERSE"
            if shrunk_score <= -0.10
            else "MIXED_OR_NEUTRAL"
        ),
        "confidence": _round(confidence),
        "coverage_ratio": _round(coverage),
        "available_component_count": len(components) - len(missing),
        "missing_component_count": len(missing),
        "missing_components": missing,
        "components": components,
        "reasons": reasons or ["MIXED_OR_LOW_MAGNITUDE_TRANSMISSION"],
        "transmission_source": macro.get("transmission_source"),
        "transmission_source_hash": macro.get("transmission_source_hash"),
        "predictive_claim": False,
        "standalone_entry_allowed": False,
        "authority": "CONTEXT_ONLY",
    }


def _macro_with_theme_policy_context(
    macro: dict[str, Any],
    *,
    theme_id: str,
    theme_news: list[dict[str, Any]],
    as_of: datetime,
) -> dict[str, Any]:
    """Add bounded official policy context to one theme's macro view."""
    output = {
        **macro,
        "family_scores": {
            str(name): dict(row)
            for name, row in (macro.get("family_scores") or {}).items()
            if isinstance(row, dict)
        },
    }
    context = _theme_policy_context(
        theme_id,
        theme_news,
        as_of=as_of,
    )
    output["theme_policy_context"] = context
    if theme_id == "nuclear_uranium" and context.get("score") is not None:
        output["family_scores"]["energy_policy"] = {
            "value": context["score"],
            "confidence": context["confidence"],
            "status": "CURRENT_OFFICIAL_CONTEXT",
            "source": "OFFICIAL_PUBLIC_RSS",
            "source_hash": context["content_hash"],
            "predictive_claim": False,
        }
    return output


def _theme_policy_context(
    theme_id: str,
    theme_news: list[dict[str, Any]],
    *,
    as_of: datetime,
) -> dict[str, Any]:
    if theme_id != "nuclear_uranium":
        return {
            "schema": "theme_policy_context_v1",
            "status": "NOT_APPLICABLE",
            "theme": theme_id,
            "as_of": as_of.isoformat(),
            "score": None,
            "confidence": 0.0,
            "event_count": 0,
            "authority": "CONTEXT_ONLY",
            "standalone_entry_allowed": False,
        }

    allowed_types = {
        "POLICY_OR_PUBLIC_FUNDING",
        "REGULATORY_OR_LICENSING",
        "TECHNOLOGY_OR_PROJECT_MILESTONE",
    }
    direction_scores = {
        "POSITIVE": 1.0,
        "POSITIVE_CONTEXT": 0.75,
        "NEGATIVE": -1.0,
        "NEGATIVE_CONTEXT": -0.75,
        "MIXED": 0.0,
        "NEUTRAL_OR_UNCERTAIN": 0.0,
    }
    importance_weights = {"HIGH": 1.0, "MEDIUM": 0.75, "LOW": 0.5}
    rows: list[dict[str, Any]] = []
    for item in theme_news:
        if item.get("source_class") != "OFFICIAL_PUBLIC_RSS":
            continue
        event_type = str(item.get("event_type") or "")
        if event_type not in allowed_types:
            continue
        published = pd.to_datetime(
            item.get("published_at"), utc=True, errors="coerce"
        )
        if pd.isna(published):
            continue
        event_time = published.to_pydatetime()
        age_hours = (as_of - event_time).total_seconds() / 3600.0
        if not 0.0 <= age_hours <= 168.0:
            continue
        direction = str(item.get("direction") or "NEUTRAL_OR_UNCERTAIN")
        importance = str(item.get("importance") or "MEDIUM")
        rows.append(
            {
                "event_hash": item.get("event_hash"),
                "published_at": event_time.isoformat(),
                "source": item.get("source"),
                "event_type": event_type,
                "direction": direction,
                "direction_score": direction_scores.get(direction, 0.0),
                "importance_weight": importance_weights.get(importance, 0.5),
            }
        )

    if not rows:
        return {
            "schema": "theme_policy_context_v1",
            "status": "UNAVAILABLE",
            "theme": theme_id,
            "as_of": as_of.isoformat(),
            "score": None,
            "confidence": 0.0,
            "event_count": 0,
            "source_count": 0,
            "reason": "NO_CURRENT_OFFICIAL_POLICY_EVENTS",
            "authority": "CONTEXT_ONLY",
            "standalone_entry_allowed": False,
        }

    total_weight = sum(float(row["importance_weight"]) for row in rows)
    score = sum(
        float(row["direction_score"]) * float(row["importance_weight"])
        for row in rows
    ) / total_weight
    sources = sorted({str(row["source"]) for row in rows if row.get("source")})
    directional_count = sum(float(row["direction_score"]) != 0.0 for row in rows)
    source_coverage = min(1.0, len(sources) / 2.0)
    event_coverage = min(1.0, len(rows) / 3.0)
    directional_coverage = directional_count / len(rows)
    confidence = min(
        0.75,
        0.50 * source_coverage
        + 0.30 * event_coverage
        + 0.20 * directional_coverage,
    )
    provenance = {
        "theme": theme_id,
        "as_of": as_of.isoformat(),
        "events": rows,
    }
    return {
        "schema": "theme_policy_context_v1",
        "status": "AVAILABLE_OFFICIAL_CURRENT_CONTEXT",
        "theme": theme_id,
        "as_of": as_of.isoformat(),
        "score": _round(max(-1.0, min(1.0, score))),
        "confidence": _round(confidence),
        "event_count": len(rows),
        "directional_event_count": directional_count,
        "source_count": len(sources),
        "sources": sources,
        "latest_published_at": max(str(row["published_at"]) for row in rows),
        "content_hash": stable_hash(provenance),
        "confidence_cap": 0.75,
        "semantics": (
            "CURRENT_OFFICIAL_HEADLINE_CONTEXT_NOT_A_POLICY_INDEX_OR_FORECAST"
        ),
        "predictive_claim": False,
        "authority": "CONTEXT_ONLY",
        "standalone_entry_allowed": False,
    }


def _theme_macro_profile(
    theme_id: str,
    instrument: dict[str, Any],
    profiles: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    subtheme = str(instrument.get("subtheme") or "").lower()
    maturity = str(instrument.get("business_maturity") or "").upper()
    if theme_id == "quantum_computing":
        base = dict(profiles.get("technology_equity") or {})
        if subtheme == "platform_enabler":
            return "QUANTUM_PLATFORM_TECHNOLOGY_EQUITY", base
        base.update(
            {
                "liquidity": 1.0,
                "credit": 0.55,
                "risk_appetite": 1.0,
                "monetary": 0.70,
                "valuation": 0.35,
            }
        )
        return "QUANTUM_EARLY_STAGE_GROWTH", base
    if theme_id == "nuclear_uranium":
        if subtheme in {"physical_uranium", "uranium_fund", "fuel_cycle"}:
            return "URANIUM_COMMODITY_CHAIN", dict(profiles.get("uranium") or {})
        if subtheme == "nuclear_power":
            base = dict(profiles.get("utility_equity") or {})
            base["energy_policy"] = 0.75
            return "NUCLEAR_POWER_UTILITY", base
        if subtheme == "nuclear_supply_chain":
            base = dict(profiles.get("industrial_equity") or {})
            base["energy_policy"] = 0.55
            return "NUCLEAR_INDUSTRIAL_SUPPLY_CHAIN", base
        if subtheme == "advanced_reactor" or "PRE_REVENUE" in maturity:
            base = dict(profiles.get("industrial_equity") or {})
            base.update(
                {
                    "liquidity": 0.70,
                    "credit": 0.85,
                    "risk_appetite": 0.80,
                    "monetary": 0.65,
                    "energy_policy": 1.0,
                }
            )
            return "ADVANCED_REACTOR_EARLY_STAGE", base
        return "NUCLEAR_INDUSTRIAL_DEFAULT", dict(
            profiles.get("industrial_equity") or {}
        )
    return "UNMAPPED_THEME", {}


def _contextual_conviction(
    *,
    technical_score: float | None,
    fundamental: dict[str, Any],
    news: dict[str, Any],
    macro: dict[str, Any],
    fundamentals_required: bool,
) -> dict[str, Any]:
    components: dict[str, tuple[float, float]] = {}
    if technical_score is not None:
        components["technical"] = (
            max(0.0, min(1.0, (technical_score + 1.0) / 2.0)),
            0.65,
        )
    if fundamentals_required and fundamental.get("score") is not None:
        components["fundamental"] = (float(fundamental["score"]), 0.25)
    if news.get("score") is not None:
        components["news"] = (float(news["score"]), 0.05)
    if macro.get("score") is not None:
        components["macro"] = (float(macro["score"]), 0.05)
    total_weight = sum(weight for _, weight in components.values())
    score = (
        sum(value * weight for value, weight in components.values())
        / total_weight
        if total_weight
        else None
    )
    return {
        "status": "AVAILABLE" if score is not None else "UNAVAILABLE",
        "score": _round(score),
        "classification": _conviction_classification(score),
        "components": {
            key: {"score": _round(value), "base_weight": weight}
            for key, (value, weight) in components.items()
        },
        "not_an_entry_signal": True,
        "standalone_entry_allowed": False,
    }


def _unit_scale(value: float, lower: float, upper: float) -> float:
    if not math.isfinite(value) or upper <= lower:
        return 0.5
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def _conviction_classification(score: float | None) -> str:
    if score is None:
        return "UNAVAILABLE"
    if score >= 0.75:
        return "HIGH_RESEARCH_CONVICTION"
    if score >= 0.60:
        return "POSITIVE_RESEARCH_CONVICTION"
    if score <= 0.35:
        return "LOW_RESEARCH_CONVICTION"
    return "MIXED_RESEARCH_CONVICTION"


def _publish(
    project_root: Path,
    payload: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> None:
    root = project_root / OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    _atomic_json(root / "frontier-technology-energy.json", payload)
    for theme_id, report in reports.items():
        _atomic_json(root / f"{theme_id.replace('_', '-')}.json", report)
    rows = [
        {
            "theme": theme_id,
            "symbol": instrument["symbol"],
            "subtheme": instrument["subtheme"],
            "business_maturity": instrument["business_maturity"],
            "technical_score": instrument["technical_score"],
            "technical_classification": instrument[
                "technical_classification"
            ],
            "contextual_conviction_score": instrument[
                "contextual_conviction"
            ].get("score"),
            "contextual_conviction_classification": instrument[
                "contextual_conviction"
            ].get("classification"),
            "fundamental_quality_score": instrument[
                "fundamental_quality"
            ].get("score"),
            "macro_alignment_score": instrument["macro_alignment"].get(
                "score"
            ),
            "alignment": instrument["timeframe_alignment"]["classification"],
            "daily_return_20": instrument["daily_snapshot"].get(
                "return_20_bars"
            ),
            "daily_return_63": instrument["daily_snapshot"].get(
                "return_63_bars"
            ),
            "fresh_news_event_count": instrument["news"][
                "fresh_event_count"
            ],
            "fundamental_status": instrument["fundamentals"]["status"],
            "shariah_status": instrument["shariah_status"],
        }
        for theme_id, report in reports.items()
        for instrument in report["instruments"]
    ]
    pd.DataFrame(rows).to_parquet(root / "instrument-summary.parquet", index=False)


def _gaps(rows: list[dict[str, Any]], macro: dict[str, Any]) -> list[str]:
    gaps = []
    missing_bars = [
        row["symbol"]
        for row in rows
        if row["timeframe_alignment"]["available_timeframe_count"] < len(INTERVALS)
    ]
    stale = [
        row["symbol"]
        for row in rows
        if row["daily_snapshot"].get("freshness_status") != "FRESH"
    ]
    missing_fundamentals = [
        row["symbol"]
        for row in rows
        if row["fundamentals"].get("fundamentals_required", True)
        and row["fundamentals"].get("status") != "AVAILABLE"
    ]
    limited_fundamentals = [
        row["symbol"]
        for row in rows
        if row["fundamentals"].get("fundamentals_required", True)
        and not row["fundamentals"].get("data_quality", {}).get(
            "decision_usable", False
        )
    ]
    missing_contracts = [
        row["symbol"]
        for row in rows
        if row["contract"].get("status") != "RESOLVED"
    ]
    missing_shariah = [
        row["symbol"]
        for row in rows
        if not row["shariah"].get("currently_eligible")
    ]
    if missing_bars:
        gaps.append("INCOMPLETE_TIMEFRAME_COVERAGE:" + ",".join(missing_bars))
    if stale:
        gaps.append("STALE_DAILY_DATA:" + ",".join(stale))
    if missing_fundamentals:
        gaps.append("FUNDAMENTALS_UNAVAILABLE:" + ",".join(missing_fundamentals))
    if limited_fundamentals:
        gaps.append(
            "FUNDAMENTAL_DECISION_QUALITY_REVIEW_REQUIRED:"
            + ",".join(limited_fundamentals)
        )
    if missing_contracts:
        gaps.append("CONTRACT_IDENTITY_UNAVAILABLE:" + ",".join(missing_contracts))
    if macro.get("data_quality") != "GO":
        gaps.append("MACRO_DATA_INCOMPLETE")
    if missing_shariah:
        gaps.append(
            "CURRENT_SHARIAH_REVIEW_REQUIRED:" + ",".join(missing_shariah)
        )
    gaps.append("REALTIME_TAPE_AND_DEPTH_UNAVAILABLE_ENTITLEMENT")
    return gaps


def _overall_status(reports: Any) -> str:
    statuses = {report.get("status") for report in reports}
    return "GO" if statuses == {"GO"} else "GO_WITH_DOCUMENTED_GAPS"


def _classification(score: float | None) -> str:
    if score is None:
        return "UNAVAILABLE"
    if score >= 0.45:
        return "STRONG_POSITIVE"
    if score >= 0.15:
        return "POSITIVE"
    if score <= -0.45:
        return "STRONG_NEGATIVE"
    if score <= -0.15:
        return "NEGATIVE"
    return "NEUTRAL_MIXED"


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1 / period, adjust=False).mean()
    relative = gain / loss.replace(0.0, np.nan)
    return (100.0 - (100.0 / (1.0 + relative))).fillna(50.0)


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    previous = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous).abs(), (low - previous).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _return(close: pd.Series, periods: int) -> float:
    if len(close) <= periods:
        return float("nan")
    return float(close.iloc[-1] / close.iloc[-periods - 1] - 1.0)


def _distance(value: Any, average: Any) -> float | None:
    if pd.isna(value) or pd.isna(average) or float(average) == 0.0:
        return None
    return _round(float(value) / float(average) - 1.0)


def _sign(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.sign(float(value)))


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return _round(float(np.median(values))) if values else None


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _round(value: Any, digits: int = 6) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _blocked(reason: str, *, theme: str | None = None) -> dict[str, Any]:
    return {
        "schema": "frontier_technology_energy_theme_analysis_v1",
        "status": "BLOCKED",
        "reason": reason,
        "theme": theme,
        "standalone_entry_allowed": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
