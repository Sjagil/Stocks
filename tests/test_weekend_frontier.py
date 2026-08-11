from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from stocks.analysis.weekend_frontier import _needs_refresh, run_frontier_weekend_research


def _config(root: Path) -> None:
    path = root / "config/themes/frontier_technology_energy_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "required_timeframes": ["1h", "4h", "1d", "1w", "1mo"],
                "themes": {
                    "quantum_computing": {
                        "instruments": [{"symbol": "AAA"}]
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _go(**extra) -> dict:
    return {
        "status": "GO",
        "provider_calls": 0,
        "broker_calls": 0,
        "orders_generated": 0,
        **extra,
    }


def test_weekend_run_is_bounded_and_read_only(tmp_path: Path) -> None:
    _config(tmp_path)
    calls = []

    def step(name: str):
        def run() -> dict:
            calls.append(name)
            return _go()

        return run

    report = run_frontier_weekend_research(
        tmp_path,
        now=datetime(2026, 8, 8, 12, tzinfo=UTC),
        collectors={
            name: step(name)
            for name in (
                "bars",
                "fundamentals",
                "contracts",
                "shariah",
                "news",
                "event_risk",
                "themes",
                "session_plan",
                "provisional_assessment",
                "evidence",
            )
        },
    )

    assert report["status"] == "GO"
    assert calls == [
        "bars",
        "fundamentals",
        "contracts",
        "shariah",
        "news",
        "event_risk",
        "themes",
        "session_plan",
        "provisional_assessment",
        "evidence",
    ]
    assert report["market_state"] == "WEEKEND_CLOSED"
    assert report["broker_calls"] == 0
    assert report["orders_generated"] == 0
    assert report["execution_authority"] == "NONE"
    assert not (
        tmp_path / "data/research/themes/private/weekend-research.lock"
    ).exists()


def test_weekday_run_makes_no_provider_calls(tmp_path: Path) -> None:
    _config(tmp_path)
    report = run_frontier_weekend_research(
        tmp_path,
        now=datetime(2026, 8, 10, 12, tzinfo=UTC),
        collectors={"news": lambda: (_ for _ in ()).throw(AssertionError())},
    )

    assert report["status"] == "SKIPPED_NOT_WEEKEND"
    assert report["steps"] == {}
    assert report["broker_calls"] == 0


def test_recent_lock_blocks_overlapping_run(tmp_path: Path) -> None:
    _config(tmp_path)
    lock = tmp_path / "data/research/themes/private/weekend-research.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("{}", encoding="utf-8")

    report = run_frontier_weekend_research(
        tmp_path,
        now=datetime.now(UTC),
        force=True,
    )

    assert report["status"] == "BLOCKED_SINGLE_FLIGHT"
    assert report["execution_authority"] == "NONE"


def test_fresh_fundamental_artifact_refreshes_when_theme_config_changes(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 8, 8, 12, tzinfo=UTC)
    artifact = tmp_path / "fundamental-coverage.json"
    artifact.write_text(
        json.dumps(
            {
                "generated_at": now.isoformat(),
                "config_hash": "OLD-CONFIG-HASH",
            }
        ),
        encoding="utf-8",
    )

    assert _needs_refresh(
        artifact,
        now,
        hours=18,
        expected_config_hash="NEW-CONFIG-HASH",
    ) is True
    assert _needs_refresh(
        artifact,
        now,
        hours=18,
        expected_config_hash="OLD-CONFIG-HASH",
    ) is False
