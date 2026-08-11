from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import main
from stocks import readiness
from stocks.data.multitimeframe import _write_public


def test_required_operator_commands_parse() -> None:
    parser = main.build_parser()

    assert parser.parse_args(["config", "validate"]).config_command == "validate"
    assert parser.parse_args(["self-test"]).command == "self-test"
    assert parser.parse_args(["risk", "status"]).risk_command == "status"
    assert (
        parser.parse_args(["data", "sources", "status"]).data_sources_command
        == "status"
    )


def test_config_validation_is_fail_closed_and_secret_free(
    tmp_path: Path,
) -> None:
    env = tmp_path / ".env.ibkr"
    env.write_text(
        "\n".join(
            (
                "IBKR_HOST=127.0.0.1",
                "IBKR_PORT=7497",
                "IBKR_CLIENT_ID=17",
                "IBKR_READ_ONLY=true",
                "IBKR_ORDER_AUTHORITY=NONE",
                "IBKR_LIVE_TRADING_ENABLED=false",
                "IBKR_ALLOW_ORDER_TRANSMISSION=false",
                "IBKR_MARKET_DATA_TYPE=3",
            )
        ),
        encoding="utf-8",
    )

    report = readiness.config_validation(tmp_path)

    assert report["status"] == "GO"
    assert report["execution_authority"] == "NONE"
    assert report["broker_calls"] == 0
    assert report["secrets_published"] is False
    assert "PASSWORD" not in str(report)


def test_data_source_status_is_brokerless(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        readiness,
        "provider_inventory",
        lambda _root: {
            "status": "GO",
            "available_source_count": 1,
            "secret_presence_only": True,
            "sources": [
                {
                    "provider": "LOCAL",
                    "status": "AVAILABLE",
                    "available": True,
                    "native_intervals": ["1d"],
                    "source_type": "LOCAL_CACHE",
                }
            ],
        },
    )
    monkeypatch.setattr(
        readiness,
        "multitimeframe_status",
        lambda _root: {
            "status": "MULTI_TIMEFRAME_DATA_GO",
            "current_data_status": "CURRENT_DATA_GO",
            "current_data_ratio": 1.0,
            "requested_current_symbol_interval_pairs": 1,
            "fresh_current_symbol_interval_pairs": 1,
            "stale_current_symbol_interval_pairs": [],
            "intervals_present": ["1d"],
            "file_count": 1,
            "row_count": 100,
        },
    )

    report = readiness.data_source_status(tmp_path)

    assert report["status"] == "GO"
    assert report["multitimeframe_current_data_status"] == (
        "CURRENT_DATA_GO"
    )
    assert report["current_data_ratio"] == 1.0
    assert report["broker_calls"] == 0
    assert report["financial_calls"]["place_order"] == 0
    assert report["execution_authority"] == "NONE"


def test_system_self_test_preserves_execution_blockers(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        readiness,
        "config_validation",
        lambda *_args, **_kwargs: {"status": "GO"},
    )
    monkeypatch.setattr(
        readiness,
        "data_source_status",
        lambda *_args, **_kwargs: {"status": "GO"},
    )
    monkeypatch.setattr(
        readiness,
        "portfolio_management_command",
        lambda *_args, **_kwargs: {"status": "NO_TARGET_POSITIONS"},
    )
    monkeypatch.setattr(
        readiness,
        "system_readiness",
        lambda *_args, **_kwargs: {
            "status": "TECHNICAL_RESEARCH_AND_SIGNALS_GO_EXECUTION_BLOCKED",
            "signals": {"status": "GO"},
            "telegram": {"status": "ENABLED"},
            "hard_blockers": ["LIVE_TWS_SOCKET_UNREACHABLE"],
            "broker_calls": 0,
            "orders_generated": 0,
        },
    )

    report = readiness.system_self_test(tmp_path)

    assert report["status"] == "GO"
    assert report["execution_authority"] == "NONE"
    assert report["execution_readiness"].endswith("EXECUTION_BLOCKED")
    assert report["hard_blockers"] == ["LIVE_TWS_SOCKET_UNREACHABLE"]


def test_public_artifact_writer_is_single_host_concurrency_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provider-inventory.json"

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda value: _write_public(path, {"value": value}),
                range(32),
            )
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["value"] in range(32)
    assert list(tmp_path.glob("*.tmp")) == []
