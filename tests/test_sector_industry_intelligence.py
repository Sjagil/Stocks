from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from stocks.analysis.groups import build_group_intelligence


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(root: Path) -> None:
    universe = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "sector": "Technology",
                "industry": "Software",
                "instrument_type": "STOCK",
                "active_listing": True,
                "signal_eligible": True,
            },
            {
                "symbol": "BBB",
                "sector": "Healthcare",
                "industry": "Biotechnology",
                "instrument_type": "STOCK",
                "active_listing": True,
                "signal_eligible": True,
            },
            {
                "symbol": "OLD",
                "sector": "Technology",
                "industry": "Hardware",
                "instrument_type": "STOCK",
                "active_listing": False,
                "signal_eligible": False,
            },
        ]
    )
    universe_path = root / "output/universe/instruments.parquet"
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(universe_path, index=False)
    _write_json(
        root / "output/portfolio/opportunity_ranking.json",
        {
            "opportunities": [
                {"ticker": "AAA", "opportunity_score": 0.8},
                {"ticker": "BBB", "opportunity_score": 0.4},
            ]
        },
    )
    _write_json(
        root / "data/news/private/current-news.json",
        {
            "status": "GO",
            "rows": [
                {
                    "published_at": "2999-01-01T00:00:00+00:00",
                    "title": "AAA launches product",
                    "source": "TEST_RSS",
                    "symbols": ["AAA"],
                    "sentiment_polarity": 0.5,
                }
            ],
        },
    )
    database = (
        root
        / "data/research/phase11_3/private/causal_research.sqlite3"
    )
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE records(dataset TEXT, payload_json TEXT)"
        )
        connection.executemany(
            "INSERT INTO records(dataset,payload_json) VALUES(?,?)",
            [
                (
                    "filings",
                    json.dumps(
                        {
                            "symbol": "AAA",
                            "record_type": "COMPANYFACT",
                            "accepted_at": "2025-01-01T00:00:00Z",
                        }
                    ),
                ),
                (
                    "news",
                    json.dumps(
                        {
                            "symbol": "AAA",
                            "published_at": "2025-01-02T00:00:00Z",
                        }
                    ),
                ),
            ],
        )


def test_every_sector_and_industry_gets_an_explicit_analysis_row(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)

    report = build_group_intelligence(tmp_path)
    sectors = json.loads(
        (
            tmp_path / "output/analysis/groups/sector-analysis.json"
        ).read_text(encoding="utf-8")
    )
    industries = json.loads(
        (
            tmp_path / "output/analysis/groups/industry-analysis.json"
        ).read_text(encoding="utf-8")
    )

    assert report["all_sector_groups_analyzed"] is True
    assert report["all_industry_groups_analyzed"] is True
    assert sectors["group_count"] == 2
    assert industries["group_count"] == 3
    technology = next(
        row for row in sectors["groups"] if row["sector"] == "Technology"
    )
    healthcare = next(
        row for row in sectors["groups"] if row["sector"] == "Healthcare"
    )
    assert technology["fresh_news_status"] == "FRESH_EVENTS"
    assert technology["fundamental_status"] == "AVAILABLE"
    assert healthcare["fresh_news_status"] == "NO_FRESH_EVENT"
    assert healthcare["analysis_status"] == "DEGRADED_FUNDAMENTALS_UNAVAILABLE"
    assert healthcare["standalone_entry_allowed"] is False
    assert report["execution_authority"] == "NONE"


def test_missing_universe_fails_closed(tmp_path: Path) -> None:
    report = build_group_intelligence(tmp_path)

    assert report["status"] == "NO_GO"
    assert report["blockers"] == ["UNIVERSE_DATA_UNAVAILABLE"]
    assert report["execution_authority"] == "NONE"
