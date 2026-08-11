from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stocks.execution.idempotency import stable_hash
from stocks.news.models import NewsStoryCluster, NormalizedNewsEvent


CONFIG_PATH = Path("config/news/event_intelligence_v1.json")
CURRENT_NEWS_PATH = Path("data/news/private/current-news.json")
IBKR_NEWS_PATH = Path("data/news/ibkr/private/headlines.jsonl")
THEME_NEWS_PATH = Path("output/analysis/themes/theme-news.json")
UNIVERSE_PATH = Path("output/universe/instruments.parquet")
PRIVATE_EVENT_LEDGER = Path(
    "data/news/private/intelligence/normalized-events.jsonl"
)
OUTPUT_ROOT = Path("output/news/intelligence")
COMMON_TICKER_WORDS = {
    "A", "AI", "ALL", "ARE", "BE", "CEO", "CFO", "FOR", "IT",
    "ON", "OR", "SO", "UK", "US", "USA",
}
POSITIVE_TERMS = {
    "approval", "award", "beat", "breakthrough", "boost", "growth",
    "higher", "launch", "record", "raise", "raised", "rally",
    "surge", "upgrade", "wins",
}
NEGATIVE_TERMS = {
    "bankruptcy", "cut", "default", "downgrade", "fraud", "halt",
    "investigation", "lawsuit", "layoff", "miss", "recall", "slump",
    "strike", "warning", "weak",
}
COUNTRY_TERMS = {
    "china": "CHINA", "taiwan": "TAIWAN", "russia": "RUSSIA",
    "ukraine": "UKRAINE", "iran": "IRAN", "israel": "ISRAEL",
    "united states": "UNITED_STATES", "europe": "EUROPE",
}
COMMODITY_TERMS = {
    "copper": "COPPER", "uranium": "URANIUM", "nuclear": "URANIUM",
    "gold": "GOLD", "silver": "SILVER", "oil": "CRUDE_OIL",
    "crude": "CRUDE_OIL", "natural gas": "NATURAL_GAS",
    "lithium": "LITHIUM", "platinum": "PLATINUM",
    "palladium": "PALLADIUM", "steel": "STEEL",
}


def build_news_event_intelligence(
    project_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    config = _read_json(project_root / CONFIG_PATH)
    lookback_days = int(config.get("lookback_days", 30))
    cutoff = observed_at - timedelta(days=lookback_days)
    universe = _universe_metadata(project_root)
    alias_index = _company_alias_index(universe)
    raw_rows, source_counts = _load_raw_rows(
        project_root, cutoff=cutoff, observed_at=observed_at
    )
    existing = _read_event_ledger(project_root / PRIVATE_EVENT_LEDGER)
    existing_by_id = {row.event_id: row for row in existing}
    recent = [row for row in existing if row.published_at >= cutoff]
    normalized_rows = []
    future_rejected = 0
    invalid_rejected = 0
    for raw in raw_rows:
        try:
            normalized = _normalize_raw_event(
                raw,
                observed_at=observed_at,
                universe=universe,
                alias_index=alias_index,
                config=config,
                recent_events=recent,
            )
        except (TypeError, ValueError):
            invalid_rejected += 1
            continue
        if normalized.published_at > observed_at:
            future_rejected += 1
            continue
        if normalized.published_at < cutoff:
            continue
        if normalized.event_id in existing_by_id:
            continue
        recent.append(normalized)
        existing_by_id[normalized.event_id] = normalized
        normalized_rows.append(normalized)
    appended = _append_events(
        project_root / PRIVATE_EVENT_LEDGER, normalized_rows
    )
    active = sorted(
        [row for row in existing_by_id.values() if row.published_at >= cutoff],
        key=lambda row: (row.published_at, row.event_id),
        reverse=True,
    )
    clusters = _build_clusters(active)
    material = sorted(
        [row for row in clusters if row.material],
        key=lambda row: (
            row.materiality,
            row.last_published_at,
            row.story_cluster_id,
        ),
        reverse=True,
    )
    portfolio_impact = _portfolio_impact(project_root, material)
    report = _status_report(
        observed_at=observed_at,
        lookback_days=lookback_days,
        raw_rows=raw_rows,
        active=active,
        clusters=clusters,
        material=material,
        source_counts=source_counts,
        appended=appended,
        invalid_rejected=invalid_rejected,
        future_rejected=future_rejected,
        portfolio_impact=portfolio_impact,
    )
    _publish(
        project_root,
        report=report,
        clusters=clusters,
        material=material,
        portfolio_impact=portfolio_impact,
    )
    return report


def news_event_intelligence_status(project_root: Path) -> dict[str, Any]:
    payload = _read_json(project_root / OUTPUT_ROOT / "status.json")
    if payload:
        return payload
    return {
        "schema": "news_event_intelligence_status_v1",
        "status": "NOT_RUN",
        "execution_authority": "NONE",
        "broker_writes": 0,
    }


def _normalize_raw_event(
    raw: dict[str, Any],
    *,
    observed_at: datetime,
    universe: dict[str, dict[str, str]],
    alias_index: dict[str, set[str]],
    config: dict[str, Any],
    recent_events: list[NormalizedNewsEvent],
) -> NormalizedNewsEvent:
    title = " ".join(str(raw.get("title") or "").split())
    if not title:
        raise ValueError("title required")
    published = _timestamp(raw.get("published_at"))
    if published is None:
        raise ValueError("published_at required")
    received = _timestamp(
        raw.get("received_at")
        or raw.get("collected_at")
        or raw.get("generated_at")
    ) or observed_at
    normalized_title = _normalized_title(title)
    direct_symbols = {
        str(symbol).upper()
        for symbol in raw.get("symbols", [])
        if str(symbol).upper() in universe
    }
    linked_symbols = direct_symbols | _link_symbols(
        title,
        universe=universe,
        alias_index=alias_index,
    )
    symbols = tuple(sorted(linked_symbols)[:50])
    sectors = tuple(
        sorted(
            {
                universe[symbol].get("sector", "UNKNOWN")
                for symbol in symbols
                if universe[symbol].get("sector")
            }
        )
    )
    industries = tuple(
        sorted(
            {
                universe[symbol].get("industry", "UNKNOWN")
                for symbol in symbols
                if universe[symbol].get("industry")
            }
        )
    )
    commodities = tuple(
        sorted(
            {
                commodity
                for term, commodity in COMMODITY_TERMS.items()
                if term in normalized_title
            }
            | {
                universe[symbol].get("underlying_commodity", "")
                for symbol in symbols
                if universe[symbol].get("underlying_commodity")
                not in {None, "", "NONE"}
            }
        )
    )
    countries = tuple(
        sorted(
            country
            for term, country in COUNTRY_TERMS.items()
            if term in normalized_title
        )
    )
    classes = _event_classes(normalized_title, config)
    sentiment, sentiment_method, sentiment_confidence = _sentiment(
        normalized_title, raw.get("sentiment_polarity"), classes
    )
    relevance = _relevance(symbols, sectors, commodities)
    severity = _severity(normalized_title, classes, config)
    source = str(raw.get("source") or "UNKNOWN")
    source_class = str(
        raw.get("source_class") or _source_class(source)
    )
    source_quality = _source_quality(source, source_class, config)
    event_id = "NEWS-" + stable_hash(
        {
            "published_at": published.isoformat(),
            "source": source,
            "title": normalized_title,
            "symbols": symbols,
        }
    )[:28]
    context = set(symbols) | set(sectors) | set(commodities)
    story_cluster_id, duplicate_similarity = _story_cluster(
        normalized_title,
        published,
        context,
        recent_events,
        threshold=float(config.get("near_duplicate_threshold", 0.72)),
        window_hours=int(config.get("story_window_hours", 72)),
    )
    if story_cluster_id is None:
        story_cluster_id = "STORY-" + stable_hash(
            {
                "event_id": event_id,
                "normalized_title": normalized_title,
            }
        )[:24]
    novelty = _novelty(
        normalized_title,
        published,
        context,
        classes,
        recent_events,
        duplicate_similarity=duplicate_similarity,
    )
    entity_confidence = 0.98 if direct_symbols else 0.85 if symbols else 0.55
    class_confidence = 0.90 if classes != ("UNCLASSIFIED",) else 0.45
    confidence = _clamp(
        0.30 * source_quality
        + 0.25 * entity_confidence
        + 0.25 * class_confidence
        + 0.20 * sentiment_confidence
    )
    raw_impact = _clamp_signed(
        sentiment
        * relevance
        * severity
        * novelty
        * source_quality
        * confidence
    )
    materiality = _clamp(
        0.70 * abs(raw_impact) + 0.30 * relevance * severity
    )
    threshold = float(config.get("materiality_threshold", 0.22))
    hard_risk_flags = tuple(
        sorted(
            {
                flag
                for event_class, flag in {
                    "BANKRUPTCY_OR_DEFAULT": "CRITICAL_SOLVENCY_EVENT",
                    "FRAUD_OR_ACCOUNTING": "CRITICAL_INTEGRITY_EVENT",
                    "TRADING_HALT": "TRADING_STATUS_EVENT",
                    "GOING_CONCERN": "CRITICAL_GOING_CONCERN_EVENT",
                }.items()
                if event_class in classes
            }
        )
    )
    return NormalizedNewsEvent(
        event_id=event_id,
        story_cluster_id=story_cluster_id,
        normalized_title_hash=stable_hash(normalized_title),
        title=title,
        source=source,
        source_class=source_class,
        published_at=published,
        received_at=received,
        link_hash=str(raw.get("link_hash") or "") or None,
        symbols=symbols,
        sectors=sectors,
        industries=industries,
        commodities=commodities,
        countries=countries,
        event_classes=classes,
        sentiment_score=round(sentiment, 8),
        sentiment_method=sentiment_method,
        relevance=round(relevance, 8),
        severity=round(severity, 8),
        novelty=round(novelty, 8),
        source_quality=round(source_quality, 8),
        confidence=round(confidence, 8),
        raw_impact=round(raw_impact, 8),
        materiality=round(materiality, 8),
        material=bool(materiality >= threshold or hard_risk_flags),
        hard_risk_flags=hard_risk_flags,
        classification_method="DETERMINISTIC_MULTILABEL_ONTOLOGY_V1",
        entity_linking_method=(
            "PROVIDER_SYMBOL_PLUS_SECURITY_MASTER_ALIAS_V1"
        ),
    )


def _load_raw_rows(
    project_root: Path,
    *,
    cutoff: datetime,
    observed_at: datetime,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    current = _read_json(project_root / CURRENT_NEWS_PATH)
    generated = current.get("generated_at")
    for row in current.get("rows", []):
        if isinstance(row, dict):
            rows.append({**row, "generated_at": generated})
            counts[str(row.get("source") or "UNKNOWN")] += 1
    for row in _read_jsonl(project_root / IBKR_NEWS_PATH):
        published = _timestamp(row.get("published_at"))
        if published is None or not cutoff <= published <= observed_at:
            continue
        rows.append({**row, "source_class": "IBKR_BROKER_NEWS"})
        counts[str(row.get("source") or "IBKR_TWS")] += 1
    theme = _read_json(project_root / THEME_NEWS_PATH)
    for row in theme.get("rows", []):
        if isinstance(row, dict):
            rows.append(row)
            counts[str(row.get("source") or "THEME_NEWS")] += 1
    return rows, dict(sorted(counts.items()))


def _universe_metadata(project_root: Path) -> dict[str, dict[str, str]]:
    path = project_root / UNIVERSE_PATH
    if not path.is_file():
        return {}
    frame = pd.read_parquet(path)
    symbol_column = "symbol" if "symbol" in frame else "ticker"
    if symbol_column not in frame:
        return {}
    wanted = [
        column
        for column in (
            symbol_column, "name", "sector", "industry", "region",
            "underlying_commodity",
        )
        if column in frame
    ]
    output: dict[str, dict[str, str]] = {}
    for _, row in frame[wanted].drop_duplicates(symbol_column).iterrows():
        symbol = str(row[symbol_column]).upper().strip()
        if not symbol:
            continue
        output[symbol] = {
            column: str(row.get(column) or "")
            for column in wanted
            if column != symbol_column
        }
    return output


def _link_symbols(
    title: str,
    *,
    universe: dict[str, dict[str, str]],
    alias_index: dict[str, set[str]],
) -> set[str]:
    upper = title.upper()
    tokens = set(re.findall(r"\$?[A-Z][A-Z0-9.\-]{1,9}", upper))
    linked = {
        token.lstrip("$")
        for token in tokens
        if token.lstrip("$") in universe
        and (
            token.startswith("$")
            or token.lstrip("$") not in COMMON_TICKER_WORDS
            and len(token.lstrip("$")) >= 3
        )
    }
    words = _normalized_title(title).split()
    for size in range(1, min(6, len(words)) + 1):
        for start in range(0, len(words) - size + 1):
            linked.update(
                alias_index.get(" ".join(words[start : start + size]), set())
            )
    return linked


def _company_alias_index(
    universe: dict[str, dict[str, str]],
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for symbol, metadata in universe.items():
        alias = _company_alias(metadata.get("name", ""))
        if len(alias) >= 5:
            index[alias].add(symbol)
    return dict(index)


def _event_classes(
    normalized_title: str, config: dict[str, Any]
) -> tuple[str, ...]:
    matched = []
    for event_class, definition in config.get("event_classes", {}).items():
        terms = [str(term).casefold() for term in definition.get("terms", [])]
        if any(term in normalized_title for term in terms):
            matched.append(str(event_class))
    return tuple(sorted(set(matched))) or ("UNCLASSIFIED",)


def _sentiment(
    text: str,
    provider_value: Any,
    classes: tuple[str, ...],
) -> tuple[float, str, float]:
    provider = _finite(provider_value)
    positive = sum(term in text for term in POSITIVE_TERMS)
    negative = sum(term in text for term in NEGATIVE_TERMS)
    lexical = (positive - negative) / max(positive + negative, 1)
    class_score = 0.0
    if {"GUIDANCE_RAISE", "ANALYST_UPGRADE", "CONTRACT_OR_AWARD"} & set(classes):
        class_score += 0.7
    if {
        "GUIDANCE_CUT", "ANALYST_DOWNGRADE", "BANKRUPTCY_OR_DEFAULT",
        "FRAUD_OR_ACCOUNTING", "GOING_CONCERN",
    } & set(classes):
        class_score -= 0.85
    if provider is not None:
        score = 0.65 * _clamp_signed(provider) + 0.20 * lexical + 0.15 * class_score
        return _clamp_signed(score), "PROVIDER_LEXICON_CLASS_CONSENSUS", 0.82
    score = 0.60 * lexical + 0.40 * class_score
    confidence = 0.68 if positive + negative or class_score else 0.35
    return _clamp_signed(score), "DETERMINISTIC_FINANCIAL_LEXICON_V1", confidence


def _relevance(
    symbols: tuple[str, ...],
    sectors: tuple[str, ...],
    commodities: tuple[str, ...],
) -> float:
    if symbols:
        return _clamp(0.98 - max(0, len(symbols) - 3) * 0.015)
    if sectors or commodities:
        return 0.72
    return 0.45


def _severity(
    text: str,
    classes: tuple[str, ...],
    config: dict[str, Any],
) -> float:
    weights = [
        float(config.get("event_classes", {}).get(name, {}).get("severity", 0.45))
        for name in classes
    ]
    severity = max(weights, default=0.45)
    if any(term in text for term in ("unexpected", "emergency", "material")):
        severity += 0.08
    return _clamp(severity)


def _story_cluster(
    title: str,
    published_at: datetime,
    context: set[str],
    recent_events: Iterable[NormalizedNewsEvent],
    *,
    threshold: float,
    window_hours: int,
) -> tuple[str | None, float]:
    tokens = _tokens(title)
    best: tuple[str | None, float] = (None, 0.0)
    for event in recent_events:
        if abs((published_at - event.published_at).total_seconds()) > window_hours * 3600:
            continue
        other_context = set(event.symbols) | set(event.sectors) | set(event.commodities)
        if context and other_context and not context.intersection(other_context):
            continue
        similarity = _jaccard(tokens, _tokens(event.title))
        if similarity >= threshold and similarity > best[1]:
            best = (event.story_cluster_id, similarity)
    return best


def _novelty(
    title: str,
    published_at: datetime,
    context: set[str],
    classes: tuple[str, ...],
    recent_events: Iterable[NormalizedNewsEvent],
    *,
    duplicate_similarity: float,
) -> float:
    if duplicate_similarity > 0:
        return _clamp(max(0.08, 1.0 - duplicate_similarity))
    tokens = _tokens(title)
    maximum = 0.0
    for event in recent_events:
        age = published_at - event.published_at
        if age.total_seconds() <= 0 or age > timedelta(days=30):
            continue
        if not set(classes).intersection(event.event_classes):
            continue
        other_context = set(event.symbols) | set(event.sectors) | set(event.commodities)
        if context and other_context and not context.intersection(other_context):
            continue
        maximum = max(maximum, _jaccard(tokens, _tokens(event.title)))
    return _clamp(max(0.15, 1.0 - 0.80 * maximum))


def _build_clusters(
    events: list[NormalizedNewsEvent],
) -> list[NewsStoryCluster]:
    grouped: dict[str, list[NormalizedNewsEvent]] = defaultdict(list)
    for event in events:
        grouped[event.story_cluster_id].append(event)
    clusters = []
    for cluster_id, rows in grouped.items():
        representative = max(
            rows,
            key=lambda row: (
                row.source_quality,
                row.materiality,
                row.published_at,
            ),
        )
        sources = tuple(sorted({row.source for row in rows}))
        confidence = _clamp(
            representative.confidence
            + min(0.12, 0.04 * (len(sources) - 1))
        )
        clusters.append(
            NewsStoryCluster(
                story_cluster_id=cluster_id,
                representative_event_id=representative.event_id,
                first_published_at=min(row.published_at for row in rows),
                last_published_at=max(row.published_at for row in rows),
                article_count=len(rows),
                independent_source_count=len(sources),
                sources=sources,
                title=representative.title,
                symbols=tuple(sorted({item for row in rows for item in row.symbols})),
                sectors=tuple(sorted({item for row in rows for item in row.sectors})),
                industries=tuple(sorted({item for row in rows for item in row.industries})),
                commodities=tuple(sorted({item for row in rows for item in row.commodities})),
                event_classes=tuple(sorted({item for row in rows for item in row.event_classes})),
                sentiment_score=representative.sentiment_score,
                relevance=max(row.relevance for row in rows),
                severity=max(row.severity for row in rows),
                novelty=max(row.novelty for row in rows),
                confidence=round(confidence, 8),
                raw_impact=representative.raw_impact,
                materiality=max(row.materiality for row in rows),
                material=any(row.material for row in rows),
                hard_risk_flags=tuple(sorted({item for row in rows for item in row.hard_risk_flags})),
            )
        )
    return clusters


def _portfolio_impact(
    project_root: Path,
    clusters: list[NewsStoryCluster],
) -> dict[str, Any]:
    plan = _read_json(
        project_root / "output/portfolio/active_portfolio_plan.json"
    )
    opportunities = {
        str(row.get("ticker") or "").upper()
        for row in plan.get("opportunities", {}).get("opportunities", [])
        if row.get("ticker")
    }
    positions = {
        str(row.get("ticker") or "").upper()
        for row in plan.get("position_actions", {}).get("actions", [])
        if row.get("ticker")
    }
    rows = []
    for cluster in clusters:
        affected = sorted(set(cluster.symbols) & (opportunities | positions))
        if not affected:
            continue
        if cluster.hard_risk_flags:
            action = "RISK_REVIEW_REQUIRED"
        elif cluster.raw_impact <= -0.12:
            action = "RANK_DOWN_CONTEXT"
        elif cluster.raw_impact >= 0.12:
            action = "RANK_UP_CONTEXT"
        else:
            action = "NO_CHANGE_CONTEXT"
        rows.append(
            {
                "story_cluster_id": cluster.story_cluster_id,
                "symbols": affected,
                "event_classes": list(cluster.event_classes),
                "hard_risk_flags": list(cluster.hard_risk_flags),
                "raw_impact": cluster.raw_impact,
                "materiality": cluster.materiality,
                "recommended_action": action,
                "standalone_entry_allowed": False,
                "execution_authority": "NONE",
            }
        )
    counts = Counter(row["recommended_action"] for row in rows)
    return {
        "schema": "news_portfolio_impact_v1",
        "status": "GO",
        "portfolio_impact_event_count": len(rows),
        "action_counts": dict(sorted(counts.items())),
        "rows": sorted(
            rows,
            key=lambda row: _finite(row["materiality"]) or 0.0,
            reverse=True,
        )[:100],
        "automatic_portfolio_mutations": 0,
        "standalone_entry_allowed": False,
        "execution_authority": "NONE",
        "broker_writes": 0,
    }


def _status_report(
    *,
    observed_at: datetime,
    lookback_days: int,
    raw_rows: list[dict[str, Any]],
    active: list[NormalizedNewsEvent],
    clusters: list[NewsStoryCluster],
    material: list[NewsStoryCluster],
    source_counts: dict[str, int],
    appended: int,
    invalid_rejected: int,
    future_rejected: int,
    portfolio_impact: dict[str, Any],
) -> dict[str, Any]:
    event_classes = Counter(
        event_class for row in clusters for event_class in row.event_classes
    )
    mapped_symbols = {symbol for row in clusters for symbol in row.symbols}
    mapped_sectors = {sector for row in clusters for sector in row.sectors}
    return {
        "schema": "news_event_intelligence_status_v1",
        "status": "GO" if clusters else "NO_CURRENT_EVENTS",
        "generated_at": observed_at.isoformat(),
        "lookback_days": lookback_days,
        "raw_article_count": len(raw_rows),
        "normalized_event_count": len(active),
        "new_event_count": appended,
        "deduplicated_story_count": len(clusters),
        "duplicate_article_count": max(0, len(active) - len(clusters)),
        "material_event_count": len(material),
        "mapped_symbol_count": len(mapped_symbols),
        "mapped_sector_count": len(mapped_sectors),
        "event_class_count": len(event_classes),
        "event_class_counts": dict(sorted(event_classes.items())),
        "source_counts": source_counts,
        "sentiment_coverage": _coverage(active, "sentiment_method"),
        "relevance_coverage": _coverage(active, "relevance"),
        "severity_coverage": _coverage(active, "severity"),
        "novelty_coverage": _coverage(active, "novelty"),
        "invalid_row_count": invalid_rejected,
        "future_timestamp_rejected_count": future_rejected,
        "portfolio_impact_event_count": portfolio_impact["portfolio_impact_event_count"],
        "event_classifier": "DETERMINISTIC_MULTILABEL_BASELINE_V1",
        "sentiment_model": "PROVIDER_PLUS_FINANCIAL_LEXICON_BASELINE_V1",
        "relevance_model": "ENTITY_AND_SCOPE_BASELINE_V1",
        "severity_model": "EVENT_ONTOLOGY_BASELINE_V1",
        "novelty_model": "STORY_CLUSTER_AND_PRIOR_EVENT_SIMILARITY_V1",
        "embedding_model": "NOT_LOADED_DETERMINISTIC_BASELINE_ACTIVE",
        "calibration": "NOT_TRAINED_NO_CANONICAL_CAR_LABELS",
        "chronological_validation": "NOT_RUN_NO_CANONICAL_CAR_LABELS",
        "financial_car_separation": "NOT_AVAILABLE",
        "model_authority": "RANKING_AND_RISK_CONTEXT_ONLY",
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "orders_generated": 0,
    }


def _publish(
    project_root: Path,
    *,
    report: dict[str, Any],
    clusters: list[NewsStoryCluster],
    material: list[NewsStoryCluster],
    portfolio_impact: dict[str, Any],
) -> None:
    root = project_root / OUTPUT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    public_clusters = [row.model_dump(mode="json") for row in clusters]
    material_rows = [row.model_dump(mode="json") for row in material[:250]]
    coverage = {
        "schema": "news_event_coverage_v1",
        "status": report["status"],
        "raw_article_count": report["raw_article_count"],
        "normalized_event_count": report["normalized_event_count"],
        "deduplicated_story_count": report["deduplicated_story_count"],
        "mapped_symbol_count": report["mapped_symbol_count"],
        "mapped_sector_count": report["mapped_sector_count"],
        "sentiment_coverage": report["sentiment_coverage"],
        "relevance_coverage": report["relevance_coverage"],
        "severity_coverage": report["severity_coverage"],
        "novelty_coverage": report["novelty_coverage"],
        "execution_authority": "NONE",
    }
    artifacts = {
        "status.json": report,
        "story-clusters.json": {
            "schema": "news_story_clusters_v1",
            "status": report["status"],
            "count": len(public_clusters),
            "rows": public_clusters[:1000],
            "execution_authority": "NONE",
        },
        "material-events.json": {
            "schema": "material_news_events_v1",
            "status": report["status"],
            "count": len(material_rows),
            "rows": material_rows,
            "standalone_entry_allowed": False,
            "execution_authority": "NONE",
        },
        "coverage.json": coverage,
        "portfolio-impact.json": portfolio_impact,
    }
    for name, payload in artifacts.items():
        payload["content_hash"] = stable_hash(payload)
        _atomic_json(root / name, payload)


def _build_event_from_json(row: dict[str, Any]) -> NormalizedNewsEvent | None:
    try:
        return NormalizedNewsEvent.model_validate(row)
    except (TypeError, ValueError):
        return None


def _read_event_ledger(path: Path) -> list[NormalizedNewsEvent]:
    rows = []
    for raw in _read_jsonl(path):
        event = _build_event_from_json(raw)
        if event is not None:
            rows.append(event)
    return rows


def _append_events(path: Path, rows: list[NormalizedNewsEvent]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n"
            )
    return len(rows)


def _normalized_title(value: str) -> str:
    value = re.sub(r"\s+[-|:]\s+(reuters|yahoo finance|cnbc)$", "", value, flags=re.I)
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _company_alias(value: str) -> str:
    normalized = _normalized_title(value)
    suffixes = {
        "inc", "incorporated", "corp", "corporation", "company", "co",
        "plc", "ltd", "limited", "holdings", "holding", "group",
    }
    return " ".join(word for word in normalized.split() if word not in suffixes)


def _tokens(value: str) -> set[str]:
    stop = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    return {word for word in _normalized_title(value).split() if word not in stop}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _source_class(source: str) -> str:
    upper = source.upper()
    if upper.startswith("IBKR_TWS"):
        return "IBKR_BROKER_NEWS"
    if "SEC" in upper:
        return "OFFICIAL_STRUCTURED_SOURCE"
    if "EODHD" in upper:
        return "LICENSED_NEWS_AGGREGATOR"
    return "PUBLIC_RSS_OR_AGGREGATOR"


def _source_quality(
    source: str, source_class: str, config: dict[str, Any]
) -> float:
    weights = config.get("source_quality", {})
    upper = source.upper()
    for key, value in weights.items():
        if str(key).upper() in upper or str(key).upper() == source_class.upper():
            return _clamp(value)
    return _clamp(weights.get("DEFAULT", 0.55))


def _coverage(rows: list[NormalizedNewsEvent], field: str) -> float:
    if not rows:
        return 0.0
    return round(
        sum(getattr(row, field, None) is not None for row in rows) / len(rows),
        8,
    )


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: Any) -> float:
    number = _finite(value)
    return min(max(number or 0.0, 0.0), 1.0)


def _clamp_signed(value: Any) -> float:
    number = _finite(value)
    return min(max(number or 0.0, -1.0), 1.0)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "build_news_event_intelligence",
    "news_event_intelligence_status",
]
