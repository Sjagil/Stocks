from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from stocks.analysis.theme_events import collect_theme_event_risk


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(root: Path) -> None:
    _json(
        root / "config/themes/frontier_technology_energy_v1.json",
        {
            "themes": {
                "quantum": {
                    "instruments": [
                        {"symbol": "AAA", "business_maturity": "OPERATIONAL"},
                        {"symbol": "BBB", "business_maturity": "OPERATIONAL"},
                    ]
                },
                "uranium": {
                    "instruments": [
                        {"symbol": "FUND", "business_maturity": "DIVERSIFIED_FUND"}
                    ]
                },
            }
        },
    )
    _json(
        root / "output/macro/events.json",
        {
            "future_schedule_status": "GO",
            "scheduled_instances": [],
        },
    )
    _json(root / "output/analysis/themes/theme-news.json", {"rows": []})


def _eodhd(payload: object | None = None):
    def fetch(symbols, start, end):
        assert symbols
        assert start <= end
        return (
            {
                "provider": "EODHD",
                "dataset": "frontier_theme_earnings_calendar",
                "status": "PROBE_GO" if payload is not None else "PLAN_NOT_ENTITLED",
                "record_count": len(payload) if isinstance(payload, list) else 0,
                "error_class": None if payload is not None else "PLAN_NOT_ENTITLED",
            },
            payload,
        )

    return fetch


def test_event_calendar_classifies_imminent_unknown_and_vehicle(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    calendars = {
        "AAA": {"Earnings Date": ["2026-08-10"]},
        "BBB": {},
    }

    report = collect_theme_event_risk(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
        calendar_fetcher=lambda symbol: (calendars[symbol], None),
        eodhd_fetcher=_eodhd(),
    )
    rows = {row["symbol"]: row for row in report["rows"]}

    assert rows["AAA"]["event_risk_status"] == "EVENT_RISK_IMMINENT"
    assert rows["AAA"]["hard_block_recommended"] is True
    assert rows["BBB"]["event_risk_status"] == "EVENT_DATE_UNCERTAIN"
    assert rows["BBB"]["hard_block_recommended"] is True
    assert rows["FUND"]["event_risk_status"] == "EVENT_NOT_APPLICABLE_VEHICLE"
    assert report["standalone_entry_allowed"] is False
    assert report["execution_authority"] == "NONE"
    assert report["orders_generated"] == 0


def test_event_calendar_marks_provider_date_conflict(tmp_path: Path) -> None:
    _fixture(tmp_path)

    report = collect_theme_event_risk(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
        calendar_fetcher=lambda symbol: (
            {"Earnings Date": ["2026-10-01"]},
            None,
        ),
        eodhd_fetcher=_eodhd(
            [
                {"code": "AAA.US", "report_date": "2026-10-10"},
                {"code": "BBB.US", "report_date": "2026-10-10"},
            ]
        ),
    )
    rows = {row["symbol"]: row for row in report["rows"]}

    assert rows["AAA"]["event_risk_status"] == "EVENT_DATE_CONFLICT"
    assert rows["AAA"]["source_confidence"] == (
        "CONFLICTING_OR_WIDE_PROVIDER_ESTIMATES"
    )
    assert report["uncertain_company_count"] == 2


def test_recent_filing_corroborates_post_event_and_history_is_append_only(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    _json(
        tmp_path
        / "data/research/themes/private/sec/symbol=AAA/submissions-TEST.json",
        {
            "payload": {
                "filings": {
                    "recent": {
                        "form": ["10-Q"],
                        "acceptanceDateTime": ["2026-08-06T16:00:00Z"],
                        "filingDate": ["2026-08-06"],
                        "accessionNumber": ["000000-26-000001"],
                    }
                }
            }
        },
    )
    calendars = {
        "AAA": {"Earnings Date": ["2026-08-06"]},
        "BBB": {"Earnings Date": ["2026-10-01"]},
    }

    first = collect_theme_event_risk(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
        calendar_fetcher=lambda symbol: (calendars[symbol], None),
        eodhd_fetcher=_eodhd(),
    )
    second = collect_theme_event_risk(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
        calendar_fetcher=lambda symbol: (calendars[symbol], None),
        eodhd_fetcher=_eodhd(),
    )
    rows = {row["symbol"]: row for row in first["rows"]}
    history = (
        tmp_path
        / "data/research/themes/private/events/event-calendar-snapshots.jsonl"
    )

    assert rows["AAA"]["event_risk_status"] == "EVENT_RISK_POST_EVENT"
    assert rows["AAA"]["soft_penalty_recommended"] is True
    assert first["private_snapshot_appended"] is True
    assert second["private_snapshot_appended"] is False
    assert len(history.read_text(encoding="utf-8").splitlines()) == 1
