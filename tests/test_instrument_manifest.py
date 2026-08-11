from __future__ import annotations

import json

import main
from stocks.research.instrument_manifest import (
    InstrumentManifestLayout,
    default_instrument_manifest,
    initialize_instrument_manifest,
    validate_instrument_manifest,
    validate_instrument_manifest_payload,
)


def test_default_instrument_manifest_is_unvalidated_and_offline() -> None:
    report = validate_instrument_manifest_payload(default_instrument_manifest())

    assert report["status"] == "GO"
    assert report["contract_validation_status"] == "UNVALIDATED"
    assert report["instrument_count"] > 0
    assert report["benchmark_count"] > 0
    assert report["sleeve_counts"]["equity"] > 0
    assert report["sleeve_counts"]["defensive"] > 0
    assert report["sleeve_counts"]["commodity"] > 0
    assert report["financial_calls"]["place_order"] == 0


def test_instrument_manifest_blocks_duplicate_contract_requests() -> None:
    payload = default_instrument_manifest()
    payload["instruments"].append(dict(payload["instruments"][0]))

    report = validate_instrument_manifest_payload(payload)

    assert report["status"] == "NO_GO"
    assert any("duplicates contract request" in error for error in report["errors"])


def test_instrument_manifest_roundtrip(tmp_path) -> None:
    layout = InstrumentManifestLayout.from_project_root(tmp_path)

    init_report = initialize_instrument_manifest(layout)
    validation = validate_instrument_manifest(layout)

    assert init_report["status"] == "GO"
    assert layout.path.exists()
    assert validation["status"] == "GO"
    assert validation["contract_validation_status"] == "UNVALIDATED"


def test_research_universe_cli_init_and_status(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    init_exit = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "research", "universe", "init-manifest"])
    init_payload = json.loads(capsys.readouterr().out)
    status_exit = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "research", "universe", "status"])
    status_payload = json.loads(capsys.readouterr().out)

    assert init_exit == 0
    assert init_payload["status"] == "GO"
    assert status_exit == 0
    assert status_payload["status"] == "READY_FOR_CONTRACT_RESOLUTION"
    assert status_payload["manifest_validation"]["contract_validation_status"] == "UNVALIDATED"
    assert status_payload["financial_calls"]["global_cancel"] == 0
