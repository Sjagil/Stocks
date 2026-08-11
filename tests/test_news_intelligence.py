from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from stocks.news.intelligence import (
    _event_classes,
    build_news_event_intelligence,
)
from stocks.news.models import NormalizedNewsEvent


def _fixture(root: Path, now: datetime) -> None:
    config = (
        Path(__file__).parents[1]
        / "config"
        / "news"
        / "event_intelligence_v1.json"
    )
    target = root / "config/news"
    target.mkdir(parents=True)
    shutil.copy2(config, target / config.name)
    universe = root / "output/universe"
    universe.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "symbol": "NVDA",
                "name": "NVIDIA Corporation",
                "sector": "Technology",
                "industry": "Semiconductors",
                "underlying_commodity": "NONE",
            },
            {
                "symbol": "ON",
                "name": "ON Semiconductor Corporation",
                "sector": "Technology",
                "industry": "Semiconductors",
                "underlying_commodity": "NONE",
            },
        ]
    ).to_parquet(universe / "instruments.parquet", index=False)
    current = root / "data/news/private"
    current.mkdir(parents=True)
    rows = [
        {
            "title": "Nvidia raises full-year revenue outlook",
            "source": "EODHD",
            "published_at": (now - timedelta(hours=2)).isoformat(),
            "sentiment_polarity": 0.8,
            "symbols": ["NVDA"],
        },
        {
            "title": "Nvidia raises full year revenue outlook after earnings",
            "source": "OFFICIAL_RSS",
            "source_class": "OFFICIAL_PUBLIC_RSS",
            "published_at": (now - timedelta(hours=1)).isoformat(),
            "symbols": ["NVDA"],
        },
        {
            "title": "ON Semiconductor cuts guidance after earnings margin pressure",
            "source": "IBKR_TWS:DJNL",
            "published_at": (now - timedelta(minutes=30)).isoformat(),
            "sentiment_polarity": -0.9,
            "symbols": [],
        },
        {
            "title": "Future timestamp must be rejected",
            "source": "TEST",
            "published_at": (now + timedelta(minutes=1)).isoformat(),
            "symbols": ["NVDA"],
        },
    ]
    (current / "current-news.json").write_text(
        json.dumps(
            {
                "generated_at": now.isoformat(),
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    portfolio = root / "output/portfolio"
    portfolio.mkdir(parents=True)
    (portfolio / "active_portfolio_plan.json").write_text(
        json.dumps(
            {
                "opportunities": {
                    "opportunities": [
                        {"ticker": "NVDA"},
                        {"ticker": "ON"},
                    ]
                },
                "position_actions": {"actions": []},
            }
        ),
        encoding="utf-8",
    )


def test_news_pipeline_is_causal_deduplicated_and_append_only(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 8, 20, 0, tzinfo=UTC)
    _fixture(tmp_path, now)

    first = build_news_event_intelligence(tmp_path, now=now)
    second = build_news_event_intelligence(tmp_path, now=now)

    assert first["status"] == "GO"
    assert first["future_timestamp_rejected_count"] == 1
    assert first["normalized_event_count"] == 3
    assert first["deduplicated_story_count"] == 2
    assert first["duplicate_article_count"] == 1
    assert first["mapped_symbol_count"] == 2
    assert first["portfolio_impact_event_count"] >= 1
    assert first["new_event_count"] == 3
    assert second["new_event_count"] == 0
    assert second["normalized_event_count"] == 3
    assert second["execution_authority"] == "NONE"
    ledger = (
        tmp_path
        / "data/news/private/intelligence/normalized-events.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 3


def test_event_classifier_is_multilabel() -> None:
    config = json.loads(
        (
            Path(__file__).parents[1]
            / "config/news/event_intelligence_v1.json"
        ).read_text(encoding="utf-8")
    )

    classes = _event_classes(
        "company cuts guidance after earnings amid margin pressure",
        config,
    )

    assert {"GUIDANCE_CUT", "EARNINGS", "MARGIN_PRESSURE"}.issubset(
        classes
    )


def test_normalized_event_schema_rejects_naive_time_and_extra_fields() -> None:
    payload = {
        "event_id": "NEWS-1",
        "story_cluster_id": "STORY-1",
        "normalized_title_hash": "HASH",
        "title": "Headline",
        "source": "TEST",
        "source_class": "TEST",
        "published_at": datetime(2026, 8, 8, 12, 0),
        "received_at": datetime(2026, 8, 8, 12, 1, tzinfo=UTC),
        "event_classes": ("EARNINGS",),
        "sentiment_score": 0.0,
        "sentiment_method": "TEST",
        "relevance": 0.5,
        "severity": 0.5,
        "novelty": 1.0,
        "source_quality": 0.5,
        "confidence": 0.5,
        "raw_impact": 0.0,
        "materiality": 0.15,
        "material": False,
        "classification_method": "TEST",
        "entity_linking_method": "TEST",
        "unexpected": "blocked",
    }

    with pytest.raises(ValidationError):
        NormalizedNewsEvent.model_validate(payload)
