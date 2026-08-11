from __future__ import annotations

import html
import json
import math
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

from stocks.execution.idempotency import stable_hash
from stocks.macro.service import macro_events
from stocks.research.phase11_2.providers import (
    SafeJsonClient,
    eodhd_probe,
    load_provider_secrets,
)
from stocks.research.phase11_3.datascraper_adapter import (
    DEFAULT_DATASCRAPER_ROOT,
)


SCHEMA = "daily_market_intelligence_digest_v1"
DEFAULT_SYMBOLS = (
    "SPY",
    "QQQ",
    "GLD",
    "DBC",
    "CPER",
    "URA",
    "AAPL",
    "ASML",
)
DATASCRAPER_RSS_LEDGER = (
    DEFAULT_DATASCRAPER_ROOT
    / "data"
    / "forward_information"
    / "news_context_v1.jsonl"
)
CRYPTO_ONLY_SOURCES = {
    "binance",
    "bitcoin",
    "coinbase",
    "coindesk",
    "cointelegraph",
    "crypto",
    "kraken",
}
SYMBOL_MARKETS = {
    "AAPL": ("US_EQUITIES", "TECHNOLOGY"),
    "AMZN": ("US_EQUITIES", "CONSUMER_DISCRETIONARY"),
    "ASML": ("EUROPE_EQUITIES", "SEMICONDUCTORS"),
    "DBC": ("COMMODITIES",),
    "CPER": ("COPPER", "INDUSTRIAL_METALS", "COMMODITIES"),
    "SCOP": ("COPPER", "INDUSTRIAL_METALS", "COMMODITIES"),
    "COPX": ("COPPER", "MINING", "COMMODITIES"),
    "EEM": ("EMERGING_MARKETS",),
    "EFA": ("DEVELOPED_MARKETS_EX_US",),
    "GLD": ("GOLD", "COMMODITIES"),
    "PPLT": ("PLATINUM", "COMMODITIES"),
    "PALL": ("PALLADIUM", "COMMODITIES"),
    "GOOGL": ("US_EQUITIES", "TECHNOLOGY"),
    "INTC": ("US_EQUITIES", "SEMICONDUCTORS"),
    "IWM": ("US_EQUITIES", "US_SMALL_CAPS"),
    "JPM": ("US_EQUITIES", "FINANCIALS"),
    "META": ("US_EQUITIES", "TECHNOLOGY"),
    "MSFT": ("US_EQUITIES", "TECHNOLOGY"),
    "NVDA": ("US_EQUITIES", "SEMICONDUCTORS"),
    "ON": ("US_EQUITIES", "SEMICONDUCTORS"),
    "QQQ": ("US_EQUITIES", "TECHNOLOGY"),
    "SLV": ("SILVER", "COMMODITIES"),
    "URA": ("URANIUM", "NUCLEAR_ENERGY", "COMMODITIES"),
    "URNM": ("URANIUM", "NUCLEAR_ENERGY", "COMMODITIES"),
    "SPY": ("US_EQUITIES", "GLOBAL_EQUITIES"),
    "TLT": ("US_GOVERNMENT_BONDS", "INTEREST_RATES"),
    "XOM": ("US_EQUITIES", "ENERGY", "CRUDE_OIL"),
}
EVENT_MARKETS = {
    "FOMC": (
        "US_EQUITIES",
        "GLOBAL_EQUITIES",
        "USD",
        "US_GOVERNMENT_BONDS",
        "GOLD",
        "COMMODITIES",
    ),
    "ECB": (
        "EUROPE_EQUITIES",
        "GLOBAL_EQUITIES",
        "EUR",
        "EURO_AREA_GOVERNMENT_BONDS",
        "GOLD",
    ),
    "US_CPI": (
        "US_EQUITIES",
        "USD",
        "US_GOVERNMENT_BONDS",
        "GOLD",
    ),
    "US_PCE": (
        "US_EQUITIES",
        "USD",
        "US_GOVERNMENT_BONDS",
        "GOLD",
    ),
    "US_PAYROLLS": (
        "US_EQUITIES",
        "USD",
        "US_GOVERNMENT_BONDS",
    ),
    "US_PMI": ("US_EQUITIES", "USD", "INDUSTRIALS"),
    "EU_CPI": (
        "EUROPE_EQUITIES",
        "EUR",
        "EURO_AREA_GOVERNMENT_BONDS",
    ),
}
HIGH_IMPACT_TERMS = {
    "bankruptcy",
    "default",
    "earnings",
    "fda",
    "guidance",
    "investigation",
    "lawsuit",
    "merger",
    "acquisition",
    "opec",
    "rate decision",
    "tariff",
    "war",
}
MEDIUM_IMPACT_TERMS = {
    "analyst",
    "contract",
    "downgrade",
    "forecast",
    "launch",
    "layoff",
    "outlook",
    "recall",
    "supply",
    "upgrade",
}
NewsFetcher = Callable[
    [Path, tuple[str, ...], datetime],
    tuple[list[dict[str, Any]], dict[str, Any]],
]


def build_market_intelligence_digest(
    project_root: Path,
    *,
    now: datetime | None = None,
    news_fetcher: NewsFetcher | None = None,
    calendar_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    symbols = _watch_symbols(project_root)
    fetcher = news_fetcher or _fetch_current_news
    try:
        news_rows, news_source = fetcher(
            project_root,
            symbols,
            observed_at,
        )
    except Exception as exc:
        news_rows = []
        news_source = {
            "status": "PROVIDER_UNAVAILABLE",
            "error_class": type(exc).__name__,
            "provider_requests": 0,
        }
    if not news_rows:
        news_rows = _local_news_fallback(
            project_root,
            now=observed_at,
        )
        if news_rows:
            news_source = {
                **news_source,
                "fallback": "PHASE11_3_PRIVATE_NEWS_ARCHIVE",
            }
    calendar = calendar_payload or _calendar_with_fallback(project_root)
    events = _upcoming_events(calendar, now=observed_at)
    normalized_news = _rank_news(
        news_rows,
        watched_symbols=set(symbols),
        now=observed_at,
    )
    private_news = _write_current_news_cache(
        project_root,
        rows=normalized_news,
        source_status=news_source,
        observed_at=observed_at,
    )
    try:
        from stocks.news import (
            build_news_event_intelligence,
            build_news_event_study,
            news_event_study_status,
        )

        event_intelligence = build_news_event_intelligence(
            project_root, now=observed_at
        )
        event_study = news_event_study_status(project_root)
        material_events = _read_json_file(
            project_root
            / "output"
            / "news"
            / "intelligence"
            / "material-events.json"
        )
        event_study_refresh_reasons = _event_study_refresh_reasons(
            event_study=event_study,
            material_events=material_events,
            observed_at=observed_at,
        )
        if event_study_refresh_reasons:
            event_study = build_news_event_study(
                project_root, now=observed_at
            )
        event_study["refresh"] = {
            "status": "CURRENT",
            "reasons": [],
            "maximum_age_hours": 12,
        }
    except (OSError, ValueError, sqlite3.Error) as exc:
        event_intelligence = {
            "status": "DEGRADED",
            "error_class": type(exc).__name__,
            "standalone_entry_allowed": False,
            "execution_authority": "NONE",
        }
        event_study = {
            "status": "DEGRADED",
            "error_class": type(exc).__name__,
            "execution_authority": "NONE",
            "refresh": {
                "status": "BLOCKED",
                "reasons": [type(exc).__name__],
                "maximum_age_hours": 12,
            },
        }
    try:
        from stocks.analysis.groups import build_group_intelligence

        group_intelligence = build_group_intelligence(project_root)
    except (OSError, ValueError, sqlite3.Error) as exc:
        group_intelligence = {
            "status": "DEGRADED",
            "error_class": type(exc).__name__,
        }
    news_timestamps = [
        timestamp
        for row in normalized_news
        if (timestamp := _timestamp(row.get("published_at"))) is not None
    ]
    latest_news_at = max(news_timestamps, default=None)
    news_fresh = bool(
        latest_news_at
        and observed_at - latest_news_at <= timedelta(hours=72)
    )
    status = (
        "GO"
        if news_fresh and events
        else "PARTIAL"
        if news_fresh or events
        else "DATA_INCOMPLETE"
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "generated_at": observed_at.isoformat(),
        "watch_symbols": list(symbols),
        "important_news": normalized_news[:8],
        "important_news_count": min(8, len(normalized_news)),
        "normalized_current_news_count": len(normalized_news),
        "private_current_news_cache": private_news,
        "sector_industry_intelligence": {
            "status": group_intelligence.get("status", "UNAVAILABLE"),
            "sector_count": int(group_intelligence.get("sector_count", 0)),
            "industry_count": int(group_intelligence.get("industry_count", 0)),
            "all_sector_groups_analyzed": bool(
                group_intelligence.get("all_sector_groups_analyzed", False)
            ),
            "all_industry_groups_analyzed": bool(
                group_intelligence.get("all_industry_groups_analyzed", False)
            ),
        },
        "news_freshness_status": (
            "CURRENT_WITHIN_72H" if news_fresh else "CURRENT_NEWS_UNAVAILABLE"
        ),
        "news_source_status": news_source,
        "news_event_intelligence": {
            "status": event_intelligence.get("status", "UNAVAILABLE"),
            "raw_article_count": int(
                event_intelligence.get("raw_article_count", 0)
            ),
            "deduplicated_story_count": int(
                event_intelligence.get("deduplicated_story_count", 0)
            ),
            "material_event_count": int(
                event_intelligence.get("material_event_count", 0)
            ),
            "portfolio_impact_event_count": int(
                event_intelligence.get("portfolio_impact_event_count", 0)
            ),
            "event_classifier": event_intelligence.get(
                "event_classifier", "UNAVAILABLE"
            ),
            "standalone_entry_allowed": False,
            "execution_authority": "NONE",
        },
        "news_event_study": {
            "status": event_study.get("status", "NOT_RUN"),
            "refresh": event_study.get("refresh", {}),
            "complete_label_count": int(
                event_study.get("complete_label_count", 0)
            ),
            "causal_training_eligible_label_count": int(
                event_study.get(
                    "causal_training_eligible_label_count", 0
                )
            ),
            "historical_descriptive_complete_label_count": int(
                event_study.get(
                    "historical_descriptive_complete_label_count", 0
                )
            ),
            "model_readiness": event_study.get(
                "model_readiness",
                {"status": "NOT_TRAINED_INSUFFICIENT_CAUSAL_CAR_LABELS"},
            ),
            "published_at_training_eligible": False,
            "execution_authority": "NONE",
        },
        "upcoming_macro_events": events[:10],
        "upcoming_macro_event_count": min(10, len(events)),
        "market_context": _market_context(project_root),
        "frontier_theme_context": _frontier_theme_context(
            project_root,
            now=observed_at,
        ),
        "portfolio_decision": _portfolio_decision(project_root),
        "event_risk_within_24h": any(
            row["window_status"] == "WITHIN_24H" for row in events
        ),
        "calendar_source_status": calendar.get(
            "future_schedule_status",
            calendar.get("status", "UNAVAILABLE"),
        ),
        "impact_classification": (
            "DETERMINISTIC_KEYWORD_AND_INSTRUMENT_MAPPING; "
            "DIRECTION_IS_MODEL_INFERENCE_NOT_FACT"
        ),
        "archive_license_status": "PRIVATE_RESEARCH_ONLY",
        "automatic_execution": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
    }
    payload["content_hash"] = stable_hash(payload)
    output = (
        project_root
        / "output"
        / "notifications"
        / "market-intelligence-digest.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    preview = output.with_suffix(".txt")
    preview.write_text(format_market_intelligence_digest(payload), encoding="utf-8")
    payload["artifact_path"] = str(output)
    payload["preview_path"] = str(preview)
    return payload


def _write_current_news_cache(
    project_root: Path,
    *,
    rows: list[dict[str, Any]],
    source_status: dict[str, Any],
    observed_at: datetime,
) -> dict[str, Any]:
    path = project_root / "data" / "news" / "private" / "current-news.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "current_multi_source_news_private_v1",
        "status": "GO" if rows else "NO_CURRENT_NEWS",
        "generated_at": observed_at.isoformat(),
        "freshness_window_hours": 72,
        "record_count": len(rows),
        "source_status": source_status,
        "rows": rows,
        "execution_authority": "NONE",
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return {
        "status": payload["status"],
        "record_count": len(rows),
        "path": str(path),
        "financial_values_published": False,
    }


def format_market_intelligence_digest(payload: dict[str, Any]) -> str:
    generated_at = _timestamp(payload.get("generated_at"))
    date_text = (
        generated_at.strftime("%Y-%m-%d")
        if generated_at is not None
        else "onbekend"
    )
    lines = [
        f"📊 Dagelijkse marktintelligentie — {date_text}",
        "",
        f"Datastatus: {payload.get('status', 'onbekend')}",
        f"Actueel nieuws: {payload.get('news_freshness_status', 'onbekend')}",
    ]
    intelligence = payload.get("news_event_intelligence") or {}
    event_study = payload.get("news_event_study") or {}
    model_readiness = event_study.get("model_readiness") or {}
    lines.append(
        "News evidence: "
        f"{intelligence.get('material_event_count', 0)} material stories | "
        f"{event_study.get('causal_training_eligible_label_count', 0)} "
        "causal CAR labels | "
        f"ML {model_readiness.get('status', 'NOT_RUN')}"
    )
    context = payload.get("market_context") or {}
    lines.extend(["", "Markt- en macroregime:"])
    lines.append(
        " • "
        f"Technisch {context.get('technical_regime', 'UNAVAILABLE')} "
        f"| macro {context.get('macro_regime', 'UNAVAILABLE')} "
        f"| markt {context.get('macro_market_regime', 'UNAVAILABLE')}"
    )
    lines.append(
        " • "
        f"Macrodata {context.get('macro_data_quality', 'UNAVAILABLE')} "
        f"| confidence {context.get('macro_confidence', 0):.2f} "
        f"| benchmark drawdown "
        f"{context.get('technical_benchmark_drawdown', 0):.1%}"
    )
    blocked_frames = context.get("blocked_timeframes") or {}
    lines.append(
        " • "
        f"TF actief {','.join(context.get('active_timeframes') or []) or 'geen'} "
        f"| observe {','.join(context.get('observed_timeframes') or []) or 'geen'}"
        + (
            " | geblokkeerd "
            + ", ".join(
                f"{timeframe}:{reason}"
                for timeframe, reason in sorted(blocked_frames.items())
            )
            if blocked_frames
            else ""
        )
    )
    frontier = payload.get("frontier_theme_context") or {}
    lines.extend(["", "Quantum en nuclear/uranium:"])
    if frontier.get("status") not in {"GO", "GO_WITH_BLOCKERS"}:
        lines.append(
            " • Geen actuele frontier-themeanalyse beschikbaar; "
            "geen fallback naar oude conclusies."
        )
    else:
        for row in frontier.get("themes", [])[:2]:
            leaders = ",".join(row.get("leaders") or []) or "geen"
            lines.append(
                " • "
                f"{row.get('label', row.get('theme'))}: "
                f"{row.get('structure_status', 'UNAVAILABLE')} / "
                f"{row.get('confirmation_status', 'UNAVAILABLE')} "
                f"| leiders {leaders}"
            )
            event_risks = ", ".join(row.get("event_risk_symbols") or [])
            lines.append(
                "   "
                f"ready {row.get('ready_observation_count', 0)} "
                f"| Shariah {row.get('current_shariah_eligible_count', 0)}/"
                f"{row.get('instrument_count', 0)}"
                + (f" | eventrisico {event_risks}" if event_risks else "")
            )
    events = list(payload.get("upcoming_macro_events") or [])
    lines.extend(["", "Komende macro-events:"])
    if not events:
        lines.append(" • Geen verifieerbare komende events beschikbaar.")
    for event in events[:6]:
        lines.append(
            " • "
            f"{_display_time(event.get('scheduled_at'))} | "
            f"{event.get('importance', 'UNKNOWN')} | "
            f"{event.get('name') or event.get('event_id')}"
        )
        lines.append(
            "   Markten: "
            + ", ".join(event.get("affected_markets") or ["ONBEKEND"])
        )
    news = list(payload.get("important_news") or [])
    lines.extend(["", "Belangrijk nieuws:"])
    if not news:
        lines.append(
            " • Geen recente publiceerbare headline; stale archive is niet "
            "als actueel gebruikt."
        )
    for row in news[:6]:
        symbols = ", ".join(row.get("symbols") or ["MARKT"])
        lines.append(
            " • "
            f"[{row.get('importance', 'MEDIUM')}/"
            f"{row.get('direction', 'UNCLEAR')}] "
            f"{symbols}: {row.get('title', 'titel niet beschikbaar')}"
        )
        lines.append(
            "   Markten: "
            + ", ".join(row.get("affected_markets") or ["ONBEKEND"])
            + f" | Bron: {row.get('source', 'ONBEKEND')}"
        )
    portfolio = payload.get("portfolio_decision") or {}
    lines.extend(["", "Portefeuillebeslissing:"])
    if portfolio.get("status") != "GO":
        lines.append(" • Actuele portefeuilleplanning niet beschikbaar.")
    else:
        lines.append(
            " • "
            f"Research exposure {portfolio.get('research_exposure_pct', 0):.1%} "
            f"| approved exposure {portfolio.get('approved_exposure_pct', 0):.1%}"
        )
        lines.append(
            " • "
            f"Whole-share shadow {portfolio.get('whole_share_exposure_pct', 0):.1%} "
            f"| current observed {portfolio.get('current_exposure_pct', 0):.2%} "
            f"| heat {portfolio.get('current_heat', 0):.3%}"
        )
        lines.append(
            " • "
            f"Heat gate {portfolio.get('portfolio_heat_gate', 'UNAVAILABLE')} "
            f"| correlatie {portfolio.get('correlation_gate', 'UNAVAILABLE')} "
            f"| research cash {portfolio.get('research_cash_pct', 1):.1%} "
            f"| HWM {portfolio.get('portfolio_hwm_status', 'UNAVAILABLE')}"
        )
        for row in portfolio.get("top_opportunities", [])[:4]:
            frames = "/".join(row.get("timeframes") or [])
            blockers = row.get("blocker_count", 0)
            survivor_count = int(
                row.get("survivor_strategy_count") or 0
            )
            lines.append(
                " • "
                f"{row.get('ticker')} score {row.get('score', 0):.3f} "
                f"| {frames or 'n/a'} | blockers {blockers} "
                f"| survivor support {survivor_count}"
            )
        for row in portfolio.get("position_actions", [])[:4]:
            lines.append(
                " • "
                f"{row.get('ticker')}: {row.get('action')} "
                f"({', '.join(row.get('reasons') or [])})"
            )
    lines.extend(
        [
            "",
            "Impactrichting is modelmatige context, geen voorspelling.",
            "Automatische execution: uit.",
            "IBKR-orders door deze digest: 0.",
        ]
    )
    return "\n".join(lines)[:3900]


def _watch_symbols(project_root: Path) -> tuple[str, ...]:
    paths = (
        project_root / "output" / "dynamic" / "current_signals.json",
        project_root / "output" / "signals" / "latest_signals.json",
    )
    symbols: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        for row in payload.get("signals") or []:
            value = str(row.get("ticker") or row.get("asset") or "").upper()
            value = value.split(".", 1)[0]
            if value and value not in symbols:
                symbols.append(value)
    for symbol in DEFAULT_SYMBOLS:
        if symbol not in symbols:
            symbols.append(symbol)
    return tuple(symbols[:6])


def _portfolio_decision(project_root: Path) -> dict[str, Any]:
    plan = _read_json_file(
        project_root / "output/portfolio/active_portfolio_plan.json"
    )
    if plan.get("status") != "GO":
        return {
            "status": "UNAVAILABLE",
            "execution_authority": "NONE",
        }
    opportunities = (
        plan.get("opportunities", {}).get("opportunities", [])
    )
    actions = plan.get("position_actions", {}).get("actions", [])
    return {
        "status": "GO",
        "generated_at": plan.get("engine_status", {}).get("generated_at"),
        "research_exposure_pct": float(
            plan.get("target_allocation", {}).get(
                "research_target_exposure", 0
            )
        ),
        "approved_exposure_pct": float(
            plan.get("target_allocation", {}).get(
                "approved_target_exposure", 0
            )
        ),
        "whole_share_exposure_pct": float(
            plan.get("sizing_audit", {}).get(
                "target_gross_exposure_pct", 0
            )
        ),
        "current_exposure_pct": float(
            plan.get("sizing_audit", {}).get(
                "current_gross_exposure_pct", 0
            )
        ),
        "current_heat": float(
            plan.get("risk", {}).get(
                "observed_current_portfolio_heat", 0
            )
        ),
        "portfolio_heat_gate": plan.get("risk", {}).get(
            "portfolio_heat_gate", "UNAVAILABLE"
        ),
        "correlation_gate": plan.get("risk", {}).get(
            "correlation_gate", "UNAVAILABLE"
        ),
        "research_cash_pct": float(
            plan.get("exposures", {}).get(
                "research_cash_weight", 1
            )
        ),
        "portfolio_hwm_status": _read_json_file(
            project_root / "output/reports/portfolio_performance.json"
        ).get("status", "UNAVAILABLE"),
        "whole_share_target_position_count": int(
            plan.get("sizing_audit", {}).get(
                "target_position_count", 0
            )
        ),
        "whole_share_sizing_status": plan.get(
            "sizing_audit", {}
        ).get("status", "UNAVAILABLE"),
        "action_ledger_status": plan.get(
            "lifecycle_audit", {}
        ).get("status", "UNAVAILABLE"),
        "top_opportunities": [
            {
                "ticker": row.get("ticker"),
                "score": float(row.get("opportunity_score", 0)),
                "timeframes": list(row.get("timeframes") or []),
                "family_count": len(
                    row.get("strategy_families") or []
                ),
                "survivor_strategy_count": sum(
                    str(strategy_id).startswith(("P1113-", "P1114-"))
                    for strategy_id in row.get("strategy_ids") or []
                ),
                "blocker_count": len(
                    row.get("execution_blockers") or []
                ),
            }
            for row in opportunities[:8]
        ],
        "position_actions": [
            {
                "ticker": row.get("ticker"),
                "action": row.get("advisory_action"),
                "reasons": list(row.get("reason_codes") or [])[:3],
            }
            for row in actions[:8]
        ],
        "financial_values_included": False,
        "position_quantities_included": False,
        "automatic_execution": False,
        "execution_authority": "NONE",
    }


def _market_context(project_root: Path) -> dict[str, Any]:
    macro = _read_json_file(project_root / "output/macro/regime.json")
    technical = _read_json_file(
        project_root / "output/dynamic/current_regime.json"
    )
    dynamic_status = _read_json_file(
        project_root / "output/dynamic/status.json"
    )
    macro_regime = macro.get("regime") or {}
    technical_inputs = technical.get("inputs") or {}
    return {
        "status": (
            "GO"
            if macro.get("status") == "GO"
            and technical.get("status") == "GO"
            else "DATA_INCOMPLETE"
        ),
        "technical_regime": technical.get("regime", "UNAVAILABLE"),
        "macro_regime": macro_regime.get(
            "overall_macro_regime", "UNAVAILABLE"
        ),
        "macro_market_regime": macro_regime.get(
            "market_regime", "UNAVAILABLE"
        ),
        "macro_confidence": _bounded_number(
            (macro.get("cycle_clock") or {}).get("confidence")
        ),
        "macro_data_quality": (macro.get("data_quality") or {}).get(
            "status", "UNAVAILABLE"
        ),
        "technical_benchmark_drawdown": _bounded_number(
            technical_inputs.get("drawdown"),
            lower=-1.0,
            upper=0.0,
        ),
        "active_timeframes": list(
            dynamic_status.get("active_timeframes") or []
        ),
        "observed_timeframes": list(
            dynamic_status.get("observed_timeframes") or []
        ),
        "blocked_timeframes": dict(
            dynamic_status.get("blocked_timeframes") or {}
        ),
        "predictive_claim": False,
        "execution_authority": "NONE",
    }


def _frontier_theme_context(
    project_root: Path,
    *,
    now: datetime,
) -> dict[str, Any]:
    analysis = _read_json_file(
        project_root
        / "output/analysis/themes/frontier-technology-energy.json"
    )
    plan = _read_json_file(
        project_root
        / "output/analysis/themes/opening-session-watchplan.json"
    )
    generated_at = _timestamp(plan.get("generated_at"))
    if (
        analysis.get("status") not in {"GO", "GO_WITH_DOCUMENTED_GAPS"}
        or plan.get("status") not in {"GO", "GO_WITH_BLOCKERS"}
        or generated_at is None
        or now - generated_at > timedelta(hours=36)
    ):
        return {
            "status": "UNAVAILABLE_OR_STALE",
            "generated_at": plan.get("generated_at"),
            "themes": [],
            "standalone_entry_allowed": False,
            "execution_authority": "NONE",
        }

    plan_rows = [
        row for row in plan.get("rows", []) if isinstance(row, dict)
    ]
    matrix = plan.get("theme_decision_matrix") or {}
    themes = analysis.get("themes") or {}
    labels = {
        "quantum_computing": "Quantum",
        "nuclear_uranium": "Nuclear/uranium",
    }
    output_rows = []
    for theme_id in ("quantum_computing", "nuclear_uranium"):
        theme = themes.get(theme_id) or {}
        decision = matrix.get(theme_id) or {}
        event_risks = []
        for row in plan_rows:
            if row.get("theme") != theme_id:
                continue
            status = str(row.get("event_risk_status") or "")
            if status in {
                "",
                "EVENT_CLEAR",
                "EVENT_NOT_APPLICABLE_VEHICLE",
            }:
                continue
            event_risks.append(f"{row.get('symbol')}:{status}")
        output_rows.append(
            {
                "theme": theme_id,
                "label": labels[theme_id],
                "structure_status": (
                    (theme.get("sector_structure") or {}).get(
                        "status", "UNAVAILABLE"
                    )
                ),
                "confirmation_status": decision.get(
                    "confirmation_status", "UNAVAILABLE"
                ),
                "decision_state": decision.get(
                    "decision_state", "UNAVAILABLE"
                ),
                "leaders": list(decision.get("leadership_symbols") or [])[:3],
                "instrument_count": int(
                    decision.get("instrument_count") or 0
                ),
                "current_shariah_eligible_count": int(
                    decision.get("current_shariah_eligible_count") or 0
                ),
                "ready_observation_count": int(
                    decision.get("ready_observation_count") or 0
                ),
                "event_risk_symbols": event_risks[:3],
                "risk_flags": list(decision.get("risk_flags") or [])[:5],
                "standalone_entry_allowed": False,
            }
        )
    return {
        "status": "GO" if plan.get("status") == "GO" else "GO_WITH_BLOCKERS",
        "generated_at": generated_at.isoformat(),
        "freshness_status": "CURRENT_WITHIN_36H",
        "themes": output_rows,
        "standalone_entry_allowed": False,
        "execution_authority": "NONE",
    }


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _event_study_refresh_reasons(
    *,
    event_study: dict[str, Any],
    material_events: dict[str, Any],
    observed_at: datetime,
) -> list[str]:
    reasons: list[str] = []
    source_hash = str(event_study.get("source_material_hash") or "")
    current_hash = str(material_events.get("content_hash") or "")
    if not event_study or event_study.get("status") == "NOT_RUN":
        reasons.append("EVENT_STUDY_NOT_RUN")
    if current_hash and source_hash != current_hash:
        reasons.append("MATERIAL_EVENT_SOURCE_HASH_CHANGED")
    generated_at = _timestamp(event_study.get("generated_at"))
    if generated_at is None:
        reasons.append("EVENT_STUDY_TIMESTAMP_UNAVAILABLE")
    elif observed_at - generated_at > timedelta(hours=12):
        reasons.append("EVENT_STUDY_OLDER_THAN_12H")
    return list(dict.fromkeys(reasons))


def _fetch_current_news(
    project_root: Path,
    symbols: tuple[str, ...],
    now: datetime,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rss_rows, rss_status = _datascraper_rss_news(now=now)
    key = load_provider_secrets(project_root).get("EODHD")
    if not key:
        return rss_rows, {
            "status": "GO" if rss_rows else "MISSING_PROVIDER_KEY",
            "provider": "MULTI_SOURCE_NEWS",
            "provider_requests": 0,
            "record_count": len(rss_rows),
            "providers": [
                {
                    "provider": "EODHD",
                    "status": "MISSING_PROVIDER_KEY",
                    "provider_requests": 0,
                },
                rss_status,
            ],
        }
    client = SafeJsonClient(
        user_agent="Stocks-Market-Intelligence/1.0 read-only",
        timeout_seconds=15,
        max_attempts=2,
        minimum_interval=0.15,
    )
    eodhd_rows: list[dict[str, Any]] = []
    probes = []
    for symbol in symbols[:6]:
        provider_symbol = _provider_symbol(symbol)
        probe, payload = eodhd_probe(
            client,
            key,
            f"current_news_{symbol}",
            "news",
            {
                "s": provider_symbol,
                "from": (now - timedelta(days=3)).date().isoformat(),
                "to": now.date().isoformat(),
                "limit": "15",
            },
        )
        probes.append(probe.public_dict())
        if isinstance(payload, list):
            eodhd_rows.extend(
                {
                    **row,
                    "_requested_symbol": symbol,
                    "_provider": "EODHD",
                }
                for row in payload
                if isinstance(row, dict)
            )
    eodhd_status = {
        "status": (
            "GO"
            if any(row.get("status") == "PROBE_GO" for row in probes)
            else "PROVIDER_UNAVAILABLE"
        ),
        "provider": "EODHD",
        "provider_requests": len(probes),
        "successful_requests": sum(
            row.get("status") == "PROBE_GO" for row in probes
        ),
        "record_count": len(eodhd_rows),
        "latest_provider_timestamp": max(
            (
                str(row.get("latest_timestamp"))
                for row in probes
                if row.get("latest_timestamp")
            ),
            default=None,
        ),
    }
    rows = [*eodhd_rows, *rss_rows]
    return rows, {
        "status": (
            "GO"
            if rss_rows or eodhd_status["status"] == "GO"
            else "PROVIDER_UNAVAILABLE"
        ),
        "provider": "MULTI_SOURCE_NEWS",
        "provider_requests": len(probes),
        "successful_requests": eodhd_status["successful_requests"],
        "record_count": len(rows),
        "providers": [eodhd_status, rss_status],
        "latest_provider_timestamp": max(
            filter(
                None,
                (
                    eodhd_status["latest_provider_timestamp"],
                    rss_status.get("latest_provider_timestamp"),
                ),
            ),
            default=None,
        ),
    }


def _datascraper_rss_news(
    *,
    now: datetime,
    path: Path = DATASCRAPER_RSS_LEDGER,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        return [], {
            "provider": "DATASCRAPER_FORWARD_RSS",
            "status": "LEDGER_UNAVAILABLE",
            "record_count": 0,
        }
    cutoff = now - timedelta(hours=72)
    rows: list[dict[str, Any]] = []
    rejected_crypto = 0
    invalid_rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            source = json.loads(line)
        except json.JSONDecodeError:
            invalid_rows += 1
            continue
        if (
            source.get("authority") != "FORWARD_ONLY_PUBLIC_RSS"
            or source.get("endpoint_family") != "rss_feed"
        ):
            continue
        published = _timestamp(
            source.get("published_at") or source.get("event_time")
        )
        payload = source.get("payload")
        if (
            published is None
            or published < cutoff
            or published > now + timedelta(minutes=5)
            or not isinstance(payload, dict)
        ):
            continue
        provider = str(source.get("source") or "PUBLIC_RSS").lower()
        source_name = str(payload.get("source_name") or "").lower()
        if any(
            token in provider or token in source_name
            for token in CRYPTO_ONLY_SOURCES
        ):
            rejected_crypto += 1
            continue
        title = _clean_title(payload.get("title"))
        if not title:
            invalid_rows += 1
            continue
        rows.append(
            {
                "published_at": published.isoformat(),
                "title": title,
                "link": payload.get("url"),
                "symbols": _entity_symbols(payload.get("entities")),
                "source": str(
                    payload.get("source_name")
                    or source.get("source")
                    or "PUBLIC_RSS"
                ),
                "_provider": "DATASCRAPER_FORWARD_RSS",
            }
        )
    latest = max(
        (row["published_at"] for row in rows),
        default=None,
    )
    return rows, {
        "provider": "DATASCRAPER_FORWARD_RSS",
        "status": "GO" if rows else "NO_CURRENT_NON_CRYPTO_ROWS",
        "record_count": len(rows),
        "rejected_crypto_only_count": rejected_crypto,
        "invalid_row_count": invalid_rows,
        "latest_provider_timestamp": latest,
        "authority": "FORWARD_ONLY_PUBLIC_RSS",
        "execution_authority": "NONE",
    }


def _entity_symbols(value: Any) -> list[str]:
    if isinstance(value, list):
        values = value
    elif isinstance(value, str):
        values = re.split(r"[,;|\s]+", value)
    else:
        values = []
    symbols = []
    for item in values:
        symbol = str(item or "").upper().split(".", 1)[0]
        if re.fullmatch(r"[A-Z][A-Z0-9-]{0,7}", symbol):
            if symbol not in symbols:
                symbols.append(symbol)
    return symbols[:6]


def _provider_symbol(symbol: str) -> str:
    if symbol.upper() == "ASML":
        return "ASML.AS"
    return f"{symbol.upper()}.US"


def _local_news_fallback(
    project_root: Path,
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    path = (
        project_root
        / "data"
        / "research"
        / "phase11_3"
        / "private"
        / "causal_research.sqlite3"
    )
    if not path.exists():
        return []
    cutoff = now - timedelta(hours=72)
    rows = []
    with sqlite3.connect(path) as connection:
        records = connection.execute(
            """
            SELECT payload_json
            FROM records
            WHERE dataset='news'
            ORDER BY record_id DESC
            LIMIT 1000
            """
        )
        for (payload_json,) in records:
            try:
                row = json.loads(payload_json)
                published = _timestamp(
                    row.get("published_at") or row.get("timestamp")
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if published is None or published < cutoff or not row.get("title"):
                continue
            rows.append({**row, "_provider": "PHASE11_3_ARCHIVE"})
    return rows


def _rank_news(
    rows: Iterable[dict[str, Any]],
    *,
    watched_symbols: set[str],
    now: datetime,
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        published = _timestamp(
            row.get("date")
            or row.get("published_at")
            or row.get("timestamp")
        )
        if published is None or published > now + timedelta(minutes=5):
            continue
        title = _clean_title(row.get("title"))
        if not title:
            continue
        symbols = _news_symbols(row)
        markets = _affected_markets(title, symbols)
        polarity = _polarity(row.get("sentiment"))
        importance_score = _importance_score(
            title,
            symbols=symbols,
            watched_symbols=watched_symbols,
            polarity=polarity,
        )
        normalized = {
            "published_at": published.isoformat(),
            "title": title,
            "symbols": symbols,
            "importance": (
                "HIGH"
                if importance_score >= 4
                else "MEDIUM"
                if importance_score >= 2
                else "LOW"
            ),
            "importance_score": importance_score,
            "direction": (
                "POSITIVE_INFERENCE"
                if polarity >= 0.25
                else "NEGATIVE_INFERENCE"
                if polarity <= -0.25
                else "MIXED_OR_UNCLEAR"
            ),
            "sentiment_polarity": polarity,
            "affected_markets": markets,
            "source": str(
                row.get("source")
                or row.get("_provider")
                or "EODHD_NEWS"
            ),
            "link_hash": (
                stable_hash(row.get("link"))
                if row.get("link")
                else None
            ),
        }
        identity = stable_hash(
            {
                "published_at": normalized["published_at"],
                "title": title,
            }
        )
        unique[identity] = normalized
    return sorted(
        unique.values(),
        key=lambda row: (
            -int(row["importance_score"]),
            -(
                _timestamp(row["published_at"])
                or datetime(1970, 1, 1, tzinfo=UTC)
            ).timestamp(),
            row["title"],
        ),
    )


def _news_symbols(row: dict[str, Any]) -> list[str]:
    values = row.get("symbols")
    if not isinstance(values, list):
        values = [row.get("symbol") or row.get("_requested_symbol")]
    symbols = []
    for value in values:
        symbol = str(value or "").upper().split(".", 1)[0]
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols[:6]


def _affected_markets(title: str, symbols: list[str]) -> list[str]:
    markets = {
        market
        for symbol in symbols
        for market in SYMBOL_MARKETS.get(symbol, ("US_EQUITIES",))
    }
    lowered = title.lower()
    keyword_markets = {
        "fed": EVENT_MARKETS["FOMC"],
        "interest rate": EVENT_MARKETS["FOMC"],
        "inflation": EVENT_MARKETS["US_CPI"],
        "ecb": EVENT_MARKETS["ECB"],
        "oil": ("ENERGY", "CRUDE_OIL", "COMMODITIES"),
        "opec": ("ENERGY", "CRUDE_OIL", "COMMODITIES"),
        "gold": ("GOLD", "COMMODITIES"),
        "bank": ("FINANCIALS", "US_EQUITIES"),
        "semiconductor": ("SEMICONDUCTORS", "TECHNOLOGY"),
        "rare earth": (
            "INDUSTRIAL_METALS",
            "COMMODITIES",
            "GLOBAL_EQUITIES",
        ),
        "tariff": ("GLOBAL_EQUITIES", "USD", "COMMODITIES"),
    }
    for keyword, affected in keyword_markets.items():
        if keyword in lowered:
            markets.update(affected)
    return sorted(markets or {"GLOBAL_EQUITIES"})


def _importance_score(
    title: str,
    *,
    symbols: list[str],
    watched_symbols: set[str],
    polarity: float,
) -> int:
    lowered = title.lower()
    score = 1
    if any(term in lowered for term in HIGH_IMPACT_TERMS):
        score += 3
    elif any(term in lowered for term in MEDIUM_IMPACT_TERMS):
        score += 1
    if watched_symbols.intersection(symbols):
        score += 1
    if abs(polarity) >= 0.65:
        score += 1
    return score


def _upcoming_events(
    calendar: dict[str, Any],
    *,
    now: datetime,
) -> list[dict[str, Any]]:
    definitions = {
        str(row.get("event_id")): row
        for row in calendar.get("event_definitions") or []
    }
    events = []
    for row in calendar.get("scheduled_instances") or []:
        scheduled = _timestamp(row.get("scheduled_at"))
        if (
            scheduled is None
            or scheduled < now - timedelta(hours=6)
            or scheduled > now + timedelta(days=7)
        ):
            continue
        event_id = str(row.get("event_id") or "UNKNOWN")
        definition = definitions.get(event_id, {})
        seconds = (scheduled - now).total_seconds()
        events.append(
            {
                "event_id": event_id,
                "name": str(
                    row.get("name")
                    or definition.get("name")
                    or event_id
                ),
                "scheduled_at": scheduled.isoformat(),
                "importance": str(
                    row.get("importance")
                    or definition.get("importance")
                    or "UNKNOWN"
                ),
                "window_status": (
                    "WITHIN_24H"
                    if seconds <= 86_400
                    else "WITHIN_7D"
                ),
                "affected_markets": list(
                    row.get("affected_markets")
                    or EVENT_MARKETS.get(event_id, ("GLOBAL_MARKETS",))
                ),
                "schedule_source": row.get("schedule_source"),
                "source_url": row.get("source_url"),
            }
        )
    return sorted(
        events,
        key=lambda row: (
            row["scheduled_at"],
            row["event_id"],
        ),
    )


def _calendar_with_fallback(project_root: Path) -> dict[str, Any]:
    try:
        return macro_events(project_root)
    except Exception:
        path = project_root / "output" / "macro" / "events.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {"status": "UNAVAILABLE", "scheduled_instances": []}
        return {
            **payload,
            "future_schedule_status": "STALE_LOCAL_CALENDAR_FALLBACK",
        }


def _clean_title(value: Any) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return " ".join(text.split())[:240]


def _polarity(value: Any) -> float:
    if isinstance(value, dict):
        value = value.get("polarity")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _bounded_number(
    value: Any,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(max(number, lower), upper)


def _timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _display_time(value: Any) -> str:
    timestamp = _timestamp(value)
    return (
        timestamp.strftime("%Y-%m-%d %H:%M UTC")
        if timestamp is not None
        else "tijd onbekend"
    )
