from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

from stocks.execution.idempotency import stable_hash
from stocks.research.phase11_2.providers import (
    SafeJsonClient,
    eodhd_probe,
    load_provider_secrets,
)


CONFIG_PATH = Path("config/themes/frontier_technology_energy_v1.json")
PRIVATE_LEDGER = Path("data/research/themes/private/news/events.jsonl")
OUTPUT_PATH = Path("output/analysis/themes/theme-news.json")
MAX_EODHD_SYMBOLS = 30
MAX_PUBLIC_ROWS = 250
QUANTUM_TERMS = {
    "POST-QUANTUM",
    "QUANTUM",
    "QUBIT",
}
NUCLEAR_TERMS = {
    "ADVANCED REACTOR",
    "ENRICHMENT",
    "FUEL CYCLE",
    "HALEU",
    "NUCLEAR",
    "REACTOR",
    "URANIUM",
}

FeedFetcher = Callable[[str], tuple[bytes | None, int | None, str | None]]


def collect_theme_news(
    project_root: Path,
    *,
    now: datetime | None = None,
    eod_client: SafeJsonClient | None = None,
    feed_fetcher: FeedFetcher | None = None,
) -> dict[str, Any]:
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    config = _read_json(project_root / CONFIG_PATH)
    themes = config.get("themes") or {}
    instruments = _instrument_map(themes)
    if not instruments:
        return _blocked("THEME_CONFIG_UNAVAILABLE")

    provider = eod_client or SafeJsonClient(
        user_agent="Stocks-Frontier-Theme-News/1.0 read-only",
        timeout_seconds=20,
        max_attempts=2,
        minimum_interval=0.15,
    )
    eod_key = load_provider_secrets(project_root).get("EODHD")
    rows: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    filtered_irrelevant = 0
    start = (observed_at - timedelta(days=7)).date().isoformat()
    end = observed_at.date().isoformat()
    for symbol, metadata in list(instruments.items())[:MAX_EODHD_SYMBOLS]:
        provider_symbol = _provider_symbol(symbol)
        probe, payload = eodhd_probe(
            provider,
            eod_key,
            f"frontier_theme_news_{symbol}",
            "news",
            {
                "s": provider_symbol,
                "from": start,
                "to": end,
                "limit": "20",
            },
        )
        probes.append(probe.public_dict())
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            normalized = _normalize_eodhd(
                row,
                symbol=symbol,
                metadata=metadata,
            )
            if normalized is None:
                filtered_irrelevant += 1
                continue
            rows.append(normalized)

    fetch_feed = feed_fetcher or _fetch_feed
    feed_results: list[dict[str, Any]] = []
    for theme_id, definition in themes.items():
        for source in definition.get("official_context_sources", []):
            feed_url = source.get("feed_url")
            if not feed_url:
                continue
            payload, status, error = fetch_feed(str(feed_url))
            parsed = _parse_feed(
                payload,
                source_name=str(source.get("name") or "OFFICIAL_FEED"),
                source_url=str(feed_url),
                theme=str(theme_id),
                symbols=[
                    str(symbol).upper()
                    for symbol in source.get("symbols", [])
                    if symbol
                ],
                source_scope=str(source.get("scope") or ""),
                observed_at=observed_at,
            )
            rows.extend(parsed)
            feed_results.append(
                {
                    "source": source.get("name"),
                    "feed_url": feed_url,
                    "http_status": status,
                    "status": "GO" if parsed else error or "NO_CURRENT_ROWS",
                    "record_count": len(parsed),
                }
            )

    rows = _deduplicate(rows, observed_at=observed_at)
    appended = _append_private(project_root / PRIVATE_LEDGER, rows)
    source_status = (
        "GO"
        if rows
        else "PROVIDER_UNAVAILABLE"
        if not any(probe.get("status") == "PROBE_GO" for probe in probes)
        and not any(result.get("http_status") == 200 for result in feed_results)
        else "NO_CURRENT_THEME_NEWS"
    )
    report: dict[str, Any] = {
        "schema": "frontier_theme_current_news_v1",
        "status": source_status,
        "generated_at": observed_at.isoformat(),
        "lookback_days": 7,
        "instrument_count": len(instruments),
        "current_event_count": len(rows),
        "filtered_irrelevant_event_count": filtered_irrelevant,
        "company_specific_event_count": sum(bool(row["symbols"]) for row in rows),
        "theme_wide_event_count": sum(not row["symbols"] for row in rows),
        "symbols_with_news": sorted(
            {symbol for row in rows for symbol in row["symbols"]}
        ),
        "themes_with_news": sorted(
            {theme for row in rows for theme in row["themes"]}
        ),
        "rows": rows[:MAX_PUBLIC_ROWS],
        "eodhd_probes": probes,
        "official_feed_results": feed_results,
        "private_events_appended": appended,
        "provider_calls": len(probes) + len(feed_results),
        "broker_calls": 0,
        "orders_generated": 0,
        "standalone_entry_allowed": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
    }
    report["content_hash"] = stable_hash(report)
    _atomic_json(project_root / OUTPUT_PATH, report)
    return report


def _normalize_eodhd(
    row: dict[str, Any],
    *,
    symbol: str,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    title = str(row.get("title") or "").strip()
    provider_symbols = _provider_symbols(row.get("symbols"))
    if not _relevant_title(
        title,
        symbol=symbol,
        metadata=metadata,
        provider_symbols=provider_symbols,
    ):
        return None
    published_at = _timestamp(row.get("date") or row.get("published_at"))
    polarity = _polarity(row.get("sentiment"))
    normalized = {
        "published_at": published_at.isoformat() if published_at else None,
        "source": str(row.get("source") or "EODHD"),
        "source_class": "LICENSED_NEWS_AGGREGATOR",
        "title": title,
        "link": row.get("link"),
        "symbols": [symbol],
        "themes": [metadata["theme"]],
        "provider_symbols": provider_symbols,
        "relevance": "DIRECT_TITLE_OR_EXPLICIT_THEME_TERM",
        "direction": _direction(title, polarity),
        "sentiment_polarity": polarity,
        "importance": _importance(title),
        "event_type": _event_type(title),
        "authority": "RANKING_CONTEXT_ONLY",
    }
    normalized["event_hash"] = _event_identity(normalized)
    return normalized


def _parse_feed(
    payload: bytes | None,
    *,
    source_name: str,
    source_url: str,
    theme: str,
    symbols: list[str] | None = None,
    source_scope: str = "",
    observed_at: datetime,
) -> list[dict[str, Any]]:
    if not payload:
        return []
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []
    cutoff = observed_at - timedelta(days=30)
    rows: list[dict[str, Any]] = []
    for element in root.iter():
        if _local_name(element.tag) not in {"item", "entry"}:
            continue
        title = _child_text(element, {"title"})
        published = _timestamp(
            _child_text(element, {"pubdate", "published", "updated", "date"})
        )
        if not title or published is None or not cutoff <= published <= observed_at:
            continue
        link = _entry_link(element)
        row = {
            "published_at": published.isoformat(),
            "source": source_name,
            "source_class": "OFFICIAL_PUBLIC_RSS",
            "source_feed": source_url,
            "source_scope": source_scope or None,
            "title": title,
            "link": link,
            "symbols": sorted(set(symbols or [])),
            "themes": [theme],
            "direction": _direction(title, None),
            "sentiment_polarity": None,
            "importance": _importance(title),
            "event_type": _event_type(title),
            "authority": "THEME_CONTEXT_ONLY",
        }
        row["event_hash"] = _event_identity(row)
        rows.append(row)
    return rows


def _fetch_feed(url: str) -> tuple[bytes | None, int | None, str | None]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Stocks-Frontier-Theme-News/1.0 read-only",
            "Accept": "application/rss+xml, application/atom+xml, text/xml",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read(), int(response.status), None
    except urllib.error.HTTPError as exc:
        return None, int(exc.code), "HTTP_ERROR"
    except (urllib.error.URLError, TimeoutError):
        return None, None, "PROVIDER_UNAVAILABLE"


def _deduplicate(
    rows: list[dict[str, Any]],
    *,
    observed_at: datetime,
) -> list[dict[str, Any]]:
    cutoff = observed_at - timedelta(days=30)
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        published = _timestamp(row.get("published_at"))
        title = str(row.get("title") or "").strip()
        if not title or published is None or not cutoff <= published <= observed_at:
            continue
        key = _event_identity(
            {"published_at": published.isoformat(), "title": title}
        )
        if key in selected:
            existing = selected[key]
            existing["symbols"] = sorted(
                set(existing.get("symbols", [])) | set(row.get("symbols", []))
            )
            existing["themes"] = sorted(
                set(existing.get("themes", [])) | set(row.get("themes", []))
            )
            existing["event_hash"] = key
            continue
        selected[key] = {
            **row,
            "published_at": published.isoformat(),
            "event_hash": key,
        }
    return sorted(
        selected.values(),
        key=lambda row: str(row["published_at"]),
        reverse=True,
    )


def _append_private(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    known: set[str] = set()
    known_identities: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                existing = json.loads(line)
                known.add(str(existing.get("event_hash")))
                known_identities.add(_event_identity(existing))
            except json.JSONDecodeError:
                continue
    appended = 0
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            identity = _event_identity(row)
            if str(row["event_hash"]) in known or identity in known_identities:
                continue
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            known.add(str(row["event_hash"]))
            known_identities.add(identity)
            appended += 1
    return appended


def _event_identity(row: dict[str, Any]) -> str:
    """Return an identity that is stable across classifier revisions."""
    published = _timestamp(row.get("published_at"))
    published_at = published.isoformat() if published else None
    return stable_hash(
        {
            "published_at": published_at,
            "title": str(row.get("title") or "").strip().casefold(),
        }
    )


def _instrument_map(themes: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for theme, definition in themes.items():
        for row in definition.get("instruments", []):
            symbol = str(row.get("symbol") or "").upper()
            if symbol:
                output[symbol] = {
                    "theme": str(theme),
                    "subtheme": str(row.get("subtheme") or ""),
                    "news_aliases": [
                        str(alias).upper()
                        for alias in row.get("news_aliases", [])
                        if alias
                    ],
                }
    return output


def _provider_symbols(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).upper() for item in value if item})


def _relevant_title(
    title: str,
    *,
    symbol: str,
    metadata: dict[str, Any],
    provider_symbols: list[str],
) -> bool:
    upper = title.upper()
    aliases = metadata.get("news_aliases") or [symbol.split(".", 1)[0]]
    direct = any(
        re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", upper)
        for alias in aliases
    )
    requested = _provider_symbol(symbol)
    provider_associated = requested in provider_symbols
    theme = metadata.get("theme")
    terms = QUANTUM_TERMS if theme == "quantum_computing" else NUCLEAR_TERMS
    thematic = any(term in upper for term in terms)
    if metadata.get("subtheme") == "platform_enabler":
        return provider_associated and direct and thematic
    return provider_associated and (direct or thematic)


def _provider_symbol(symbol: str) -> str:
    upper = symbol.upper()
    return upper if "." in upper else f"{upper}.US"


def _polarity(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("polarity")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(max(-1.0, min(1.0, number)), 6) if math.isfinite(number) else None


def _direction(title: str, polarity: float | None) -> str:
    text = title.casefold()
    positive = {
        "approval",
        "award",
        "beat",
        "breakthrough",
        "contract",
        "growth",
        "permit",
        "record revenue",
        "restart",
        "rises",
        "surging",
    }
    negative = {
        "crashed",
        "delay",
        "down ",
        "drop",
        "downgrade",
        "falls",
        "investigation",
        "loss",
        "loss widens",
        "miss",
        "rejection",
        "suspension",
        "withdraws",
        "wider-than-expected",
    }
    title_positive = any(term in text for term in positive)
    title_negative = any(term in text for term in negative)
    if title_positive and title_negative:
        return "MIXED_CONTEXT"
    if title_negative:
        return "NEGATIVE_CONTEXT"
    if title_positive:
        return "POSITIVE_CONTEXT"
    if polarity is not None and polarity >= 0.2:
        return "POSITIVE_CONTEXT"
    if polarity is not None and polarity <= -0.2:
        return "NEGATIVE_CONTEXT"
    return "NEUTRAL_OR_UNCERTAIN"


def _importance(title: str) -> str:
    text = title.casefold()
    high = {
        "acquisition",
        "earnings",
        "financial results",
        "guidance",
        "license",
        "permit",
        "revenue",
    }
    return "HIGH" if any(term in text for term in high) else "MEDIUM"


def _event_type(title: str) -> str:
    """Classify headline purpose without treating it as a trading signal."""
    text = title.casefold()
    categories = (
        (
            "EARNINGS_OR_GUIDANCE",
            {
                "earnings",
                "financial results",
                "guidance",
                "revenue",
                "quarter results",
            },
        ),
        (
            "REGULATORY_OR_LICENSING",
            {"approval", "license", "permit", "regulator", "nrc"},
        ),
        (
            "POLICY_OR_PUBLIC_FUNDING",
            {
                "congress",
                "department of energy",
                "doe",
                "federal funding",
                "government",
                "legislation",
            },
        ),
        (
            "CONTRACT_OR_PARTNERSHIP",
            {"award", "contract", "partnership", "strategic agreement"},
        ),
        (
            "TECHNOLOGY_OR_PROJECT_MILESTONE",
            {
                "breakthrough",
                "criticality",
                "milestone",
                "qubit",
                "reactor",
                "roadmap",
            },
        ),
        (
            "SUPPLY_OR_PRODUCTION",
            {
                "enrichment",
                "fuel cycle",
                "production",
                "restart",
                "supply",
                "uranium",
            },
        ),
        (
            "CAPITAL_OR_CORPORATE_ACTION",
            {
                "acquisition",
                "buyback",
                "capital raise",
                "financing",
                "merger",
                "offering",
            },
        ),
        (
            "LEGAL_OR_INVESTIGATION",
            {"investigation", "lawsuit", "litigation", "probe"},
        ),
        (
            "ANALYST_OR_VALUATION_COMMENTARY",
            {
                "analyst",
                "better buy",
                "price target",
                "rating",
                "valuation",
            },
        ),
    )
    for event_type, terms in categories:
        if any(term in text for term in terms):
            return event_type
    return "GENERAL_THEME_CONTEXT"


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _child_text(element: ET.Element, names: set[str]) -> str | None:
    for child in element:
        if _local_name(child.tag) in names and child.text:
            return child.text.strip()
    return None


def _entry_link(element: ET.Element) -> str | None:
    for child in element:
        if _local_name(child.tag) != "link":
            continue
        return child.attrib.get("href") or (child.text or "").strip() or None
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _blocked(reason: str) -> dict[str, Any]:
    return {
        "schema": "frontier_theme_current_news_v1",
        "status": "BLOCKED",
        "reason": reason,
        "broker_calls": 0,
        "orders_generated": 0,
        "execution_authority": "NONE",
    }
