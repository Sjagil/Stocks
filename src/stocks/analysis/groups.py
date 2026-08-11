from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from stocks.execution.idempotency import stable_hash


UNIVERSE_PATH = Path("output/universe/instruments.parquet")
OPPORTUNITY_PATH = Path("output/portfolio/opportunity_ranking.json")
CURRENT_NEWS_PATH = Path("data/news/private/current-news.json")
IBKR_NEWS_PATH = Path("data/news/ibkr/private/headlines.jsonl")
PHASE11_3_DATABASE = Path(
    "data/research/phase11_3/private/causal_research.sqlite3"
)
MACRO_SECTOR_PATH = Path("output/macro/sector-impact.json")
OUTPUT_ROOT = Path("output/analysis/groups")


def build_group_intelligence(project_root: Path) -> dict[str, Any]:
    universe = _universe(project_root)
    if universe.empty:
        return _blocked("UNIVERSE_DATA_UNAVAILABLE")
    opportunities = _opportunities(project_root)
    fundamentals = _fundamental_symbols(project_root)
    archive_news = _archive_news(project_root)
    current_news, current_news_status = _current_news(project_root)
    macro = _macro_sectors(project_root)
    reports = {
        dimension: _dimension_report(
            universe,
            dimension=dimension,
            opportunities=opportunities,
            fundamentals=fundamentals,
            archive_news=archive_news,
            current_news=current_news,
            current_news_status=current_news_status,
            macro=macro,
        )
        for dimension in ("sector", "industry")
    }
    active = universe.loc[universe["active_listing"].astype(bool)]
    eligible_stocks = active.loc[
        active["signal_eligible"].astype(bool)
        & active["instrument_type"].astype(str).eq("STOCK")
    ]
    eligible_stock_symbols = set(
        eligible_stocks["symbol"].astype(str).str.upper()
    )
    known_active = active.loc[
        ~active["sector"].astype(str).str.upper().eq("UNKNOWN")
        & ~active["industry"].astype(str).str.upper().eq("UNKNOWN")
    ]
    payload = {
        "schema": "sector_industry_intelligence_coverage_v1",
        "status": "GO_WITH_DOCUMENTED_GAPS",
        "generated_at": _now(),
        "universe_instrument_count": int(len(universe)),
        "active_listing_count": int(len(active)),
        "classified_active_listing_count": int(len(known_active)),
        "classification_coverage_ratio": _ratio(len(known_active), len(active)),
        "sector_count": reports["sector"]["group_count"],
        "industry_count": reports["industry"]["group_count"],
        "all_sector_groups_analyzed": True,
        "all_industry_groups_analyzed": True,
        "fundamental_symbol_count": len(fundamentals),
        "signal_eligible_stock_count": len(eligible_stock_symbols),
        "signal_eligible_fundamental_count": len(
            eligible_stock_symbols & fundamentals
        ),
        "signal_eligible_fundamental_coverage_ratio": _ratio(
            len(eligible_stock_symbols & fundamentals),
            len(eligible_stock_symbols),
        ),
        "signal_eligible_fundamental_missing_symbols": sorted(
            eligible_stock_symbols - fundamentals
        ),
        "archived_news_symbol_count": len(archive_news),
        "current_news_record_count": len(current_news),
        "current_news_source_status": current_news_status,
        "group_status_counts": {
            "sector": reports["sector"]["status_counts"],
            "industry": reports["industry"]["status_counts"],
        },
        "data_semantics": {
            "no_news_event_means": "NO_FRESH_EVENT_NOT_NEGATIVE_SENTIMENT",
            "fundamentals": "POINT_IN_TIME_CAUSAL_FACTS_WHERE_AVAILABLE",
            "news": "CONTEXT_AND_RANKING_ONLY",
            "macro": "CONTEXT_ONLY_REQUIRES_TECHNICAL_AND_FUNDAMENTAL_CONFIRMATION",
        },
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }
    payload["content_hash"] = stable_hash(payload)
    root = project_root / OUTPUT_ROOT
    _atomic_json(root / "coverage.json", payload)
    _atomic_json(root / "sector-analysis.json", reports["sector"])
    _atomic_json(root / "industry-analysis.json", reports["industry"])
    return payload


def group_intelligence_status(project_root: Path) -> dict[str, Any]:
    path = project_root / OUTPUT_ROOT / "coverage.json"
    report = _read_json(path)
    if not report:
        return build_group_intelligence(project_root)
    return report


def _dimension_report(
    universe: pd.DataFrame,
    *,
    dimension: str,
    opportunities: dict[str, float],
    fundamentals: set[str],
    archive_news: dict[str, dict[str, Any]],
    current_news: list[dict[str, Any]],
    current_news_status: dict[str, Any],
    macro: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    current_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in current_news:
        for symbol in row.get("symbols", []):
            current_by_symbol[str(symbol).upper()].append(row)
    rows: list[dict[str, Any]] = []
    for raw_name, group in universe.groupby(dimension, observed=True):
        name = str(raw_name or "UNKNOWN")
        symbols = sorted(set(group["symbol"].astype(str).str.upper()))
        active_symbols = sorted(
            set(
                group.loc[group["active_listing"].astype(bool), "symbol"]
                .astype(str)
                .str.upper()
            )
        )
        active_stocks = group.loc[
            group["active_listing"].astype(bool)
            & group["instrument_type"].astype(str).eq("STOCK")
        ]
        active_stock_symbols = sorted(
            set(active_stocks["symbol"].astype(str).str.upper())
        )
        eligible_stock_symbols = sorted(
            set(
                active_stocks.loc[
                    active_stocks["signal_eligible"].astype(bool), "symbol"
                ]
                .astype(str)
                .str.upper()
            )
        )
        eligible_symbols = sorted(
            set(
                group.loc[group["signal_eligible"].astype(bool), "symbol"]
                .astype(str)
                .str.upper()
            )
        )
        active_fundamental_symbols = sorted(
            set(active_stock_symbols) & fundamentals
        )
        fundamental_symbols = sorted(
            set(eligible_stock_symbols) & fundamentals
        )
        archived_symbols = sorted(set(active_symbols) & set(archive_news))
        fresh_rows = [
            row
            for symbol in active_symbols
            for row in current_by_symbol.get(symbol, [])
        ]
        fresh_rows = _deduplicate_news(fresh_rows)
        archive_count = sum(
            int(archive_news[symbol]["event_count"])
            for symbol in archived_symbols
        )
        latest_archive = max(
            (
                str(archive_news[symbol].get("latest_published_at"))
                for symbol in archived_symbols
                if archive_news[symbol].get("latest_published_at")
            ),
            default=None,
        )
        opportunity_rows = [
            (symbol, opportunities[symbol])
            for symbol in symbols
            if symbol in opportunities
        ]
        opportunity_rows.sort(key=lambda item: (-item[1], item[0]))
        status = _group_status(
            name=name,
            active_count=len(active_symbols),
            fundamental_required_count=len(eligible_stock_symbols),
            fundamental_count=len(fundamental_symbols),
            current_provider_available=current_news_status.get("status")
            in {"GO", "AVAILABLE_PARTIAL"},
        )
        macro_context = (
            macro.get(_key(name)) if dimension == "sector" else None
        )
        rows.append(
            {
                dimension: name,
                "analysis_status": status,
                "instrument_count": len(symbols),
                "active_listing_count": len(active_symbols),
                "signal_eligible_count": len(eligible_symbols),
                "asset_type_counts": {
                    str(asset_type): int(count)
                    for asset_type, count in group.loc[
                        group["active_listing"].astype(bool), "instrument_type"
                    ]
                    .astype(str)
                    .value_counts()
                    .items()
                },
                "company_fundamental_required_count": len(
                    eligible_stock_symbols
                ),
                "active_company_stock_count": len(active_stock_symbols),
                "active_company_fundamental_count": len(
                    active_fundamental_symbols
                ),
                "active_company_fundamental_coverage_ratio": _ratio(
                    len(active_fundamental_symbols), len(active_stock_symbols)
                ),
                "signal_eligible_company_stock_count": len(
                    eligible_stock_symbols
                ),
                "fundamental_symbol_count": len(fundamental_symbols),
                "fundamental_coverage_ratio": _ratio(
                    len(fundamental_symbols), len(eligible_stock_symbols)
                ),
                "fundamental_status": (
                    "NOT_REQUIRED_NO_ELIGIBLE_STOCKS"
                    if not eligible_stock_symbols
                    else
                    "AVAILABLE"
                    if len(fundamental_symbols) == len(eligible_stock_symbols)
                    else "AVAILABLE_PARTIAL"
                    if fundamental_symbols
                    else "UNAVAILABLE"
                ),
                "archived_news_event_count": archive_count,
                "archived_news_symbol_count": len(archived_symbols),
                "archived_news_coverage_ratio": _ratio(
                    len(archived_symbols), len(active_symbols)
                ),
                "latest_archived_news_at": latest_archive,
                "fresh_news_event_count": len(fresh_rows),
                "fresh_news_status": (
                    "FRESH_EVENTS" if fresh_rows else "NO_FRESH_EVENT"
                ),
                "fresh_news_sentiment": _sentiment(fresh_rows),
                "top_fresh_headlines": [
                    {
                        "published_at": row.get("published_at"),
                        "title": row.get("title"),
                        "source": row.get("source"),
                        "symbols": list(row.get("symbols") or []),
                    }
                    for row in fresh_rows[:3]
                ],
                "opportunity_symbol_count": len(opportunity_rows),
                "maximum_opportunity_score": (
                    round(opportunity_rows[0][1], 6)
                    if opportunity_rows
                    else 0.0
                ),
                "top_opportunity_symbols": [
                    symbol for symbol, _ in opportunity_rows[:5]
                ],
                "macro_context": macro_context,
                "representative_symbols": (
                    [symbol for symbol, _ in opportunity_rows[:5]]
                    or eligible_symbols[:5]
                    or active_symbols[:5]
                ),
                "standalone_entry_allowed": False,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["maximum_opportunity_score"]),
            -int(row["fresh_news_event_count"]),
            str(row[dimension]),
        )
    )
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["analysis_status"])] += 1
    report = {
        "schema": f"{dimension}_intelligence_analysis_v1",
        "status": "GO",
        "generated_at": _now(),
        "dimension": dimension,
        "group_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "current_news_source_status": current_news_status,
        "groups": [
            {"rank": index, **row}
            for index, row in enumerate(rows, start=1)
        ],
        "ranking_method": (
            "OPPORTUNITY_SCORE_THEN_FRESH_NEWS_WITH_EXPLICIT_DATA_COVERAGE"
        ),
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }
    report["content_hash"] = stable_hash(report)
    return report


def _universe(project_root: Path) -> pd.DataFrame:
    path = project_root / UNIVERSE_PATH
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    required = {
        "symbol",
        "sector",
        "industry",
        "active_listing",
        "signal_eligible",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.copy()
    for column in ("sector", "industry"):
        work[column] = work[column].fillna("UNKNOWN").astype(str)
    return work


def _opportunities(project_root: Path) -> dict[str, float]:
    rows = _read_json(project_root / OPPORTUNITY_PATH).get(
        "opportunities", []
    )
    return {
        str(row.get("ticker", "")).upper(): float(
            row.get("opportunity_score", 0.0) or 0.0
        )
        for row in rows
        if row.get("ticker")
    }


def _fundamental_symbols(project_root: Path) -> set[str]:
    path = project_root / PHASE11_3_DATABASE
    if not path.is_file():
        return set()
    try:
        with sqlite3.connect(path) as connection:
            return {
                str(row[0]).upper()
                for row in connection.execute(
                    """
                    SELECT DISTINCT json_extract(payload_json, '$.symbol')
                    FROM records
                    WHERE dataset='filings'
                      AND json_extract(payload_json, '$.record_type')='COMPANYFACT'
                      AND json_extract(payload_json, '$.accepted_at') IS NOT NULL
                    """
                )
                if row[0]
            }
    except sqlite3.Error:
        return set()


def _archive_news(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / PHASE11_3_DATABASE
    if not path.is_file():
        return {}
    try:
        with sqlite3.connect(path) as connection:
            rows = connection.execute(
                """
                SELECT
                  UPPER(json_extract(payload_json, '$.symbol')) AS symbol,
                  COUNT(*) AS event_count,
                  MAX(json_extract(payload_json, '$.published_at')) AS latest
                FROM records
                WHERE dataset='news'
                  AND json_extract(payload_json, '$.record_type') IS NULL
                  AND json_extract(payload_json, '$.published_at') IS NOT NULL
                GROUP BY symbol
                """
            )
            return {
                str(symbol): {
                    "event_count": int(count),
                    "latest_published_at": latest,
                }
                for symbol, count, latest in rows
                if symbol
            }
    except sqlite3.Error:
        return {}


def _current_news(
    project_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(project_root / CURRENT_NEWS_PATH)
    rows = [
        row for row in payload.get("rows", []) if isinstance(row, dict)
    ]
    ibkr_rows = _read_jsonl(project_root / IBKR_NEWS_PATH)
    cutoff = datetime.now(UTC) - timedelta(hours=72)
    combined = []
    for row in [*rows, *ibkr_rows]:
        published = _timestamp(row.get("published_at"))
        if published is None or published < cutoff:
            continue
        combined.append(row)
    providers = sorted(
        {
            str(row.get("source") or "UNKNOWN")
            for row in combined
        }
    )
    return _deduplicate_news(combined), {
        "status": "GO" if combined else payload.get("status", "NO_CURRENT_NEWS"),
        "provider_count": len(providers),
        "providers": providers,
        "record_count": len(combined),
        "freshness_window_hours": 72,
    }


def _macro_sectors(project_root: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(project_root / MACRO_SECTOR_PATH)
    rows = (
        payload.get("implications", {}).get(
            "sectors_and_asset_classes", {}
        )
    )
    return {
        _key(name): {
            "macro_support": row.get("macro_support"),
            "score": row.get("score"),
            "confidence": row.get("confidence"),
            "final_status": row.get("final_status"),
            "fundamental_confirmation": row.get(
                "fundamental_confirmation", "REQUIRED"
            ),
            "technical_confirmation": row.get(
                "technical_confirmation", "REQUIRED"
            ),
        }
        for name, row in rows.items()
        if isinstance(row, dict)
    }


def _group_status(
    *,
    name: str,
    active_count: int,
    fundamental_required_count: int,
    fundamental_count: int,
    current_provider_available: bool,
) -> str:
    if name.upper() == "UNKNOWN":
        return "CLASSIFICATION_REQUIRED"
    if active_count == 0:
        return "NO_ACTIVE_LISTINGS"
    if not current_provider_available:
        return "DEGRADED_CURRENT_NEWS_UNAVAILABLE"
    if fundamental_required_count and fundamental_count == 0:
        return "DEGRADED_FUNDAMENTALS_UNAVAILABLE"
    if fundamental_count < fundamental_required_count:
        return "DEGRADED_FUNDAMENTALS_PARTIAL"
    return "GO"


def _sentiment(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row.get("sentiment_polarity", 0.0) or 0.0)
        for row in rows
    ]
    if not values:
        return {"status": "NO_FRESH_EVENT", "average_polarity": None}
    average = sum(values) / len(values)
    return {
        "status": (
            "POSITIVE_INFERENCE"
            if average >= 0.25
            else "NEGATIVE_INFERENCE"
            if average <= -0.25
            else "MIXED_OR_UNCLEAR"
        ),
        "average_polarity": round(average, 6),
        "predictive_claim": False,
    }


def _deduplicate_news(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = stable_hash(
            {
                "published_at": row.get("published_at"),
                "title": row.get("title"),
                "source": row.get("source"),
            }
        )
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda row: str(row.get("published_at") or ""),
        reverse=True,
    )


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _key(value: str) -> str:
    return "_".join(str(value).strip().lower().replace("&", "and").split())


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 8) if denominator else 0.0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "sector_industry_intelligence_coverage_v1",
        "status": "NO_GO",
        "blockers": [reason],
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["build_group_intelligence", "group_intelligence_status"]
