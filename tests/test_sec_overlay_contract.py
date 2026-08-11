from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from stocks.research.sec_overlay import (
    MAX_SEC_OVERLAY_POINTS,
    _database_path,
    sec_intelligence_audit,
    sec_intelligence_status,
    sec_overlay_for_signal,
    sec_overlays_for_signals,
)


ROOT = Path(__file__).resolve().parents[1]


def _engine():
    path = ROOT / "sec_ownership_and_event_intelligence_v1.py"
    spec = importlib.util.spec_from_file_location("sec_overlay_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _project(tmp_path: Path) -> Path:
    shutil.copyfile(
        ROOT / "sec_ownership_and_event_intelligence_v1.py",
        tmp_path / "sec_ownership_and_event_intelligence_v1.py",
    )
    return tmp_path


def test_overlay_uses_canonical_sec_ingest_database(tmp_path: Path) -> None:
    assert _database_path(tmp_path) == (
        tmp_path / "data/sec_intelligence/sec_intelligence.sqlite3"
    )


def test_status_counts_only_resolved_equity_13f_managers(tmp_path: Path) -> None:
    root = _project(tmp_path)
    module = _engine()
    connection = module.connect_database(_database_path(root))
    module.ensure_schema(connection)
    connection.execute(
        "INSERT INTO sec_intel_filings "
        "(accession, form_type, accepted_at, content_sha256, parser_version, "
        "metadata_json, parsed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "0001",
            "13F-HR",
            "2026-08-01T00:00:00Z",
            "A" * 64,
            "test",
            "{}",
            "2026-08-01T00:00:01Z",
        ),
    )
    required = {
        "accession": "0001",
        "event_key": "event-1",
        "form_group": "13F",
        "event_type": "13f_position_snapshot",
        "effective_at": "2026-06-30T00:00:00Z",
        "accepted_at": "2026-08-01T00:00:00Z",
        "is_derivative": 0,
        "is_planned": 0,
        "is_10b5_1": 0,
        "materiality": 0.1,
        "direction_score": 0.0,
        "confidence": 1.0,
    }
    for event_key, symbol, put_call in (
        ("event-1", "AAPL", None),
        ("event-2", "AAPL", "CALL"),
        ("event-3", None, None),
    ):
        payload = json.dumps(
            {"manager_cik": "0000123456", "put_call": put_call}
        )
        values = {
            **required,
            "event_key": event_key,
            "symbol": symbol,
            "payload_json": payload,
        }
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        connection.execute(
            f"INSERT INTO sec_intel_events ({columns}) VALUES ({placeholders})",
            tuple(values.values()),
        )
    connection.commit()
    connection.close()

    status = sec_intelligence_status(root)

    assert status["relevant_13f_manager_count"] == 1


def test_batch_overlay_never_authorizes_an_invalid_base_signal(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    reports = sec_overlays_for_signals(
        root,
        [
            {
                "symbol": "ON",
                "as_of": "2026-08-07T00:00:00Z",
                "base_score": 50.0,
                "base_signal_authorized": False,
            }
        ],
    )

    assert reports["ON"]["overlay"]["entry_authorized"] is False
    assert reports["ON"]["standalone_entry_allowed"] is False
    assert reports["ON"]["authority"] == "RANKING_OVERLAY_ONLY"


def test_overlay_is_bounded_to_four_points_and_cannot_create_entry() -> None:
    module = _engine()
    positive = module.apply_ranking_overlay(
        50.0,
        {"sec_intelligence_score": 1.0},
        base_signal_authorized=False,
    )
    negative = module.apply_ranking_overlay(
        50.0,
        {"sec_intelligence_score": -1.0},
        base_signal_authorized=True,
    )
    assert MAX_SEC_OVERLAY_POINTS == 4.0
    assert positive["sec_overlay_points"] == 4.0
    assert positive["entry_authorized"] is False
    assert negative["sec_overlay_points"] == -4.0
    assert negative["entry_authorized"] is True


def test_empty_structured_store_is_degraded_not_fabricated(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    status = sec_intelligence_status(root)
    assert status["status"] == "DEGRADED"
    assert status["structured_event_count"] == 0
    assert "RAW_FILING_CONTENT_INGEST_REQUIRED" in status["blockers"]
    assert status["standalone_entry_allowed"] is False

    overlay = sec_overlay_for_signal(
        root,
        symbol="ON",
        as_of="2026-08-06T12:00:00Z",
        base_score=70.0,
        base_signal_authorized=True,
    )
    assert overlay["status"] == "DEGRADED_NO_CAUSAL_SEC_EVENTS"
    assert overlay["overlay"]["sec_overlay_points"] == 0.0
    assert overlay["overlay"]["entry_authorized"] is True


def test_parser_audit_passes_but_ablation_stays_unpromoted(
    tmp_path: Path,
) -> None:
    report = sec_intelligence_audit(_project(tmp_path))
    assert report["status"] == "GO"
    assert report["parser_self_test"] == "PASS"
    assert report["ablation_status"] == "NOT_RUN"
    assert report["standalone_entry_allowed"] is False
    assert report["execution_authority"] == "NONE"
