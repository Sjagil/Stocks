from __future__ import annotations

from datetime import date
from decimal import Decimal

import yaml

from stocks.data.phase5_1 import (
    _merge_currency_rates,
    _safe_provider_error,
    build_total_returns_for_universe_v1_1,
    total_return_status_v1_1,
)
from stocks.data.total_returns import TotalReturnLayout
from stocks.research.instrument_manifest import default_instrument_manifest


def test_phase5_1_fx_merge_preserves_primary_and_labels_fallback() -> None:
    primary = {date(2026, 1, 2): Decimal("0.90")}
    fallback = {
        date(2026, 1, 2): Decimal("0.89"),
        date(2026, 1, 3): Decimal("0.905"),
    }

    merged, sources, fallback_rows = _merge_currency_rates(
        primary,
        fallback,
        fallback_source="ECB_REFERENCE_RATE",
    )

    assert merged[date(2026, 1, 2)] == Decimal("0.90")
    assert merged[date(2026, 1, 3)] == Decimal("0.905")
    assert sources[date(2026, 1, 2)] == "EODHD_FOREX"
    assert sources[date(2026, 1, 3)] == "ECB_REFERENCE_RATE"
    assert fallback_rows == 1


def test_phase5_1_provider_errors_never_persist_request_urls_or_secrets() -> None:
    class Response:
        status_code = 404

    class ProviderFailure(Exception):
        response = Response()

    error = _safe_provider_error(
        ProviderFailure("https://example.test/path?api_token=secret-value")
    )

    assert error == {"error_class": "ProviderFailure", "http_status": 404}
    assert "secret-value" not in str(error)


def test_phase5_1_total_returns_fail_closed_before_frozen_builder_on_missing_contracts(tmp_path) -> None:
    instrument_path = tmp_path / "data/instruments/research_universe.yaml"
    instrument_path.parent.mkdir(parents=True)
    instrument_path.write_text(
        yaml.safe_dump(default_instrument_manifest(), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    report = build_total_returns_for_universe_v1_1(
        project_root=tmp_path,
        rows=[],
        base_currency="EUR",
    )
    status = total_return_status_v1_1(TotalReturnLayout.from_project_root(tmp_path))

    assert report["status"] == "NO_GO_MISSING_CONTRACT_IDENTITIES"
    assert report["missing_contract_symbols"]
    assert len(report["missing_contract_symbols"]) == report["requested_instrument_count"]
    assert report["financial_calls"] == {"place_order": 0, "cancel_order": 0, "global_cancel": 0}
    assert status["status"] == "NO_GO"
    assert status["manifest_status"] == "NO_GO"
    assert status["coverage_status"] == "INCOMPLETE_CONTRACT_IDENTITIES"
