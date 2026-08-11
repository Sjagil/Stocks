from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def _load_module(
    name: str,
    path: Path,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


SEC = _load_module(
    "sec_intelligence_13f_test_module",
    ROOT
    / "sec_ownership_and_event_intelligence_v1.py",
)

NORMALIZER = _load_module(
    "sec_13f_normalizer_test_module",
    ROOT
    / "tools"
    / "normalize_sec_13f_manifest.py",
)


def _connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        ":memory:"
    )

    conn.row_factory = sqlite3.Row

    conn.executescript(
        """
        CREATE TABLE sec_intel_filings (
            accession TEXT PRIMARY KEY,
            form_type TEXT NOT NULL,
            filer_cik TEXT,
            report_period TEXT,
            accepted_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE sec_intel_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            accession TEXT NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT,
            shares REAL,
            value_usd REAL,
            payload_json TEXT NOT NULL
        );
        """
    )

    return conn


def _filing(
    conn: sqlite3.Connection,
    *,
    accession: str,
    manager: str,
    period: str,
    accepted_at: str,
    form_type: str = "13F-HR",
    metadata: dict | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sec_intel_filings (
            accession,
            form_type,
            filer_cik,
            report_period,
            accepted_at,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            accession,
            form_type,
            manager,
            period,
            accepted_at,
            json.dumps(
                metadata or {}
            ),
        ),
    )


def _position(
    conn: sqlite3.Connection,
    *,
    accession: str,
    symbol: str,
    shares: float,
    value_usd: float,
    put_call: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO sec_intel_events (
            accession,
            event_type,
            symbol,
            shares,
            value_usd,
            payload_json
        )
        VALUES (
            ?,
            '13f_position_snapshot',
            ?,
            ?,
            ?,
            ?
        )
        """,
        (
            accession,
            symbol,
            shares,
            value_usd,
            json.dumps(
                {
                    "put_call": put_call,
                }
            ),
        ),
    )


def _event_context(
    *managers: str,
) -> list[dict[str, str]]:
    return [
        {
            "event_type": (
                "13f_position_snapshot"
            ),
            "person_cik": manager,
        }
        for manager in managers
    ]


def test_new_holdings_amendment_is_added_and_put_is_separate() -> None:
    conn = _connection()
    manager = "0000000001"

    _filing(
        conn,
        accession="PREVIOUS-BASE",
        manager=manager,
        period="2025-03-31",
        accepted_at="2025-05-01T12:00:00Z",
    )

    _filing(
        conn,
        accession="PREVIOUS-NEW",
        manager=manager,
        period="2025-03-31",
        accepted_at="2025-05-10T12:00:00Z",
        form_type="13F-HR/A",
        metadata={
            "is_amendment": True,
            "amendment_type": (
                "NEW HOLDINGS"
            ),
        },
    )

    _filing(
        conn,
        accession="CURRENT",
        manager=manager,
        period="2025-06-30",
        accepted_at="2025-08-01T12:00:00Z",
    )

    _position(
        conn,
        accession="PREVIOUS-BASE",
        symbol="AAPL",
        shares=100,
        value_usd=10_000,
    )

    _position(
        conn,
        accession="PREVIOUS-NEW",
        symbol="AAPL",
        shares=20,
        value_usd=2_000,
    )

    _position(
        conn,
        accession="CURRENT",
        symbol="AAPL",
        shares=150,
        value_usd=15_000,
    )

    _position(
        conn,
        accession="CURRENT",
        symbol="AAPL",
        shares=1_000,
        value_usd=5_000,
        put_call="PUT",
    )

    conn.commit()

    result = SEC._compute_13f_features(
        conn,
        "AAPL",
        _event_context(manager),
        datetime(
            2025,
            8,
            10,
            tzinfo=timezone.utc,
        ),
    )

    assert (
        result[
            "institutional_manager_count"
        ]
        == 1
    )

    assert (
        result[
            "institutional_increase_count"
        ]
        == 1
    )

    assert (
        result[
            "institutional_reported_put_value_usd"
        ]
        == 5_000
    )

    assert (
        result[
            "institutional_options_context_score"
        ]
        < 0
    )

    conn.close()


def test_restatement_replaces_original_snapshot() -> None:
    conn = _connection()
    manager = "0000000002"

    _filing(
        conn,
        accession="OLD-ORIGINAL",
        manager=manager,
        period="2025-03-31",
        accepted_at="2025-05-01T12:00:00Z",
    )

    _filing(
        conn,
        accession="OLD-RESTATEMENT",
        manager=manager,
        period="2025-03-31",
        accepted_at="2025-05-12T12:00:00Z",
        form_type="13F-HR/A",
        metadata={
            "is_amendment": True,
            "amendment_type": (
                "RESTATEMENT"
            ),
        },
    )

    _filing(
        conn,
        accession="CURRENT",
        manager=manager,
        period="2025-06-30",
        accepted_at="2025-08-01T12:00:00Z",
    )

    _position(
        conn,
        accession="OLD-ORIGINAL",
        symbol="AAPL",
        shares=100,
        value_usd=10_000,
    )

    _position(
        conn,
        accession="OLD-RESTATEMENT",
        symbol="AAPL",
        shares=60,
        value_usd=6_000,
    )

    _position(
        conn,
        accession="CURRENT",
        symbol="AAPL",
        shares=60,
        value_usd=6_000,
    )

    conn.commit()

    result = SEC._compute_13f_features(
        conn,
        "AAPL",
        _event_context(manager),
        datetime(
            2025,
            8,
            10,
            tzinfo=timezone.utc,
        ),
    )

    assert (
        result[
            "institutional_unchanged_count"
        ]
        == 1
    )

    assert (
        result[
            "institutional_decrease_count"
        ]
        == 0
    )

    conn.close()


def test_manager_count_excludes_irrelevant_historical_manager() -> None:
    conn = _connection()

    relevant = "0000000003"
    irrelevant = "0000000004"

    for manager in (
        relevant,
        irrelevant,
    ):
        _filing(
            conn,
            accession=f"{manager}-PREVIOUS",
            manager=manager,
            period="2025-03-31",
            accepted_at="2025-05-01T12:00:00Z",
        )

        _filing(
            conn,
            accession=f"{manager}-CURRENT",
            manager=manager,
            period="2025-06-30",
            accepted_at="2025-08-01T12:00:00Z",
        )

    _position(
        conn,
        accession=f"{relevant}-PREVIOUS",
        symbol="AAPL",
        shares=90,
        value_usd=9_000,
    )

    _position(
        conn,
        accession=f"{relevant}-CURRENT",
        symbol="AAPL",
        shares=100,
        value_usd=10_000,
    )

    conn.commit()

    result = SEC._compute_13f_features(
        conn,
        "AAPL",
        _event_context(
            relevant,
            irrelevant,
        ),
        datetime(
            2025,
            8,
            10,
            tzinfo=timezone.utc,
        ),
    )

    assert (
        result[
            "institutional_manager_count"
        ]
        == 1
    )

    assert (
        result[
            "institutional_equity_manager_count"
        ]
        == 1
    )

    conn.close()


def test_value_multiplier_inference_distinguishes_dollars_and_thousands() -> None:
    dollar_positions = [
        {
            "shares": 1_000,
            "value": 100_000,
            "put_call": None,
        },
        {
            "shares": 2_000,
            "value": 240_000,
            "put_call": None,
        },
    ]

    thousand_positions = [
        {
            "shares": 1_000,
            "value": 100,
            "put_call": None,
        },
        {
            "shares": 2_000,
            "value": 240,
            "put_call": None,
        },
    ]

    dollar_multiplier, _, _ = (
        NORMALIZER.infer_value_multiplier(
            dollar_positions
        )
    )

    thousand_multiplier, _, _ = (
        NORMALIZER.infer_value_multiplier(
            thousand_positions
        )
    )

    assert dollar_multiplier == 1.0
    assert thousand_multiplier == 1_000.0
