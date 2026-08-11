from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.analysis.theme_news import collect_theme_news
from stocks.analysis.theme_news import _append_private, _event_identity


class FakeJsonClient:
    def get_json(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> tuple[Any | None, int | None, int, str | None]:
        del url, headers
        return (
            [
                {
                    "date": "2026-08-08T08:00:00Z",
                    "title": "AAA reports record revenue growth",
                    "source": "TEST_WIRE",
                    "link": "https://example.test/aaa",
                    "symbols": ["AAA.US", "BBB.US"],
                    "sentiment": {"polarity": 0.6},
                }
            ],
            200,
            1,
            None,
        )


def _write_config(root: Path) -> None:
    path = root / "config/themes/frontier_technology_energy_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "themes": {
                    "quantum_computing": {
                        "instruments": [{"symbol": "AAA"}],
                        "official_context_sources": [],
                    },
                    "nuclear_uranium": {
                        "instruments": [{"symbol": "BBB"}],
                        "official_context_sources": [
                            {
                                "name": "OFFICIAL_NUCLEAR",
                                "feed_url": "https://example.test/nuclear.xml",
                                "symbols": ["BBB"],
                                "scope": "BBB_REGULATORY_CONTEXT",
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def test_theme_news_combines_company_and_official_feed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("EODHD_API_KEY", "test-key")
    feed = b"""<?xml version='1.0'?>
    <rss><channel><item>
      <title>NRC issues construction permit for advanced reactor</title>
      <link>https://example.test/permit</link>
      <pubDate>Fri, 07 Aug 2026 15:00:00 GMT</pubDate>
    </item></channel></rss>"""

    report = collect_theme_news(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
        eod_client=FakeJsonClient(),
        feed_fetcher=lambda _url: (feed, 200, None),
    )

    assert report["status"] == "GO"
    assert report["company_specific_event_count"] == 2
    assert report["theme_wide_event_count"] == 0
    assert report["provider_calls"] == 3
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0
    assert report["execution_authority"] == "NONE"
    event_types = {row["event_type"] for row in report["rows"]}
    assert "EARNINGS_OR_GUIDANCE" in event_types
    assert "REGULATORY_OR_LICENSING" in event_types
    official = next(
        row for row in report["rows"]
        if row["source_class"] == "OFFICIAL_PUBLIC_RSS"
    )
    assert official["symbols"] == ["BBB"]
    assert official["source_scope"] == "BBB_REGULATORY_CONTEXT"
    assert (
        tmp_path / "data/research/themes/private/news/events.jsonl"
    ).is_file()


def test_theme_news_deduplicates_repeated_refreshes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.setenv("EODHD_API_KEY", "test-key")
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    first = collect_theme_news(
        tmp_path,
        now=now,
        eod_client=FakeJsonClient(),
        feed_fetcher=lambda _url: (None, 503, "HTTP_ERROR"),
    )
    second = collect_theme_news(
        tmp_path,
        now=now,
        eod_client=FakeJsonClient(),
        feed_fetcher=lambda _url: (None, 503, "HTTP_ERROR"),
    )

    assert first["private_events_appended"] == 1
    assert second["private_events_appended"] == 0


def test_classifier_revision_does_not_change_private_event_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    first = {
        "published_at": "2026-08-08T08:00:00+00:00",
        "title": "AAA reports results",
        "event_type": "GENERAL_THEME_CONTEXT",
    }
    first["event_hash"] = _event_identity(first)
    revised = {**first, "event_type": "EARNINGS_OR_GUIDANCE"}
    revised["event_hash"] = _event_identity(revised)

    assert _append_private(path, [first]) == 1
    assert _append_private(path, [revised]) == 0


def test_official_feed_without_symbol_binding_remains_theme_wide(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_config(tmp_path)
    path = tmp_path / "config/themes/frontier_technology_energy_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    source = config["themes"]["nuclear_uranium"][
        "official_context_sources"
    ][0]
    source.pop("symbols")
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("EODHD_API_KEY", "test-key")
    feed = b"""<rss><channel><item>
      <title>NRC issues construction permit for advanced reactor</title>
      <pubDate>Fri, 07 Aug 2026 15:00:00 GMT</pubDate>
    </item></channel></rss>"""

    report = collect_theme_news(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
        eod_client=FakeJsonClient(),
        feed_fetcher=lambda _url: (feed, 200, None),
    )

    assert report["theme_wide_event_count"] == 1
