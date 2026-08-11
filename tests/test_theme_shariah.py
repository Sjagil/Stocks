from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from stocks.analysis.theme_shariah import collect_theme_shariah_coverage


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(root: Path) -> None:
    _write(
        root / "config/themes/frontier_technology_energy_v1.json",
        {
            "themes": {
                "quantum_computing": {
                    "instruments": [
                        {
                            "symbol": "IONQ",
                            "ibkr_contract": {"asset_class": "stock"},
                        },
                        {
                            "symbol": "URA",
                            "ibkr_contract": {
                                "asset_class": "commodity_etf"
                            },
                        },
                    ]
                }
            }
        },
    )
    _write(
        root / "config/screener/daily_screener_v1.json",
        {
            "shariah_attestations_path": (
                "config/screener/shariah_attestations_v1.json"
            )
        },
    )
    _write(
        root / "config/screener/shariah_attestations_v1.json",
        {
            "attestations": [
                {
                    "symbol": "IONQ",
                    "status": "SHARIAH_ELIGIBLE_PIT",
                    "screened_at": "2026-08-01T00:00:00Z",
                    "expires_at": "2026-09-01T00:00:00Z",
                    "methodology": "TEST_DUAL_SOURCE",
                    "source": "TEST",
                    "evidence": ["one", "two"],
                }
            ]
        },
    )
    _write(
        root / "output/market_context/entry-shortlist.json",
        {
            "observations": [
                {
                    "symbol": "URA",
                    "state": "DIRECTIONAL_BIAS_ONLY_4H_SETUP_PENDING",
                    "decision_contract": {"hard_veto_pass": True},
                }
            ]
        },
    )
    _write(
        root
        / "data/research/themes/private/shariah-review-evidence.json",
        {
            "observed_at": "2026-08-07T00:00:00Z",
            "valid_until": "2026-08-31T23:59:59Z",
            "rows": [
                {
                    "symbol": "URA",
                    "review_outcome": "DUAL_SOURCE_CONFLICT",
                    "sources": [
                        {
                            "provider": "ONE",
                            "status": "COMPLIANT",
                            "url": "https://one.invalid",
                        },
                        {
                            "provider": "TWO",
                            "status": "NON_COMPLIANT",
                            "url": "https://two.invalid",
                        },
                    ],
                }
            ],
        },
    )


def test_theme_shariah_uses_only_current_expiring_attestations(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)

    report = collect_theme_shariah_coverage(
        tmp_path,
        now=datetime(2026, 8, 8, tzinfo=UTC),
    )

    by_symbol = {row["symbol"]: row for row in report["instruments"]}
    assert report["status"] == "GO_WITH_REVIEW_REQUIRED"
    assert report["currently_eligible_count"] == 1
    assert by_symbol["IONQ"]["currently_eligible"] is True
    assert by_symbol["IONQ"]["evidence_reference_count"] == 2
    assert by_symbol["URA"]["status"] == "SHARIAH_PRODUCT_ATTESTATION_REQUIRED"
    assert report["review_queue"][0]["symbol"] == "URA"
    assert report["review_queue"][0]["review_priority"] == "P1_CURRENT_FORWARD_SETUP"
    assert report["review_queue"][0]["external_review_status"] == "DUAL_SOURCE_CONFLICT"
    assert report["external_review_conflict_count"] == 1
    assert by_symbol["URA"]["external_review"]["attestation_effect"] == "NONE"
    assert report["orders_generated"] == 0
    assert report["execution_authority"] == "NONE"


def test_expired_attestation_never_counts_as_current(tmp_path: Path) -> None:
    _fixture(tmp_path)

    report = collect_theme_shariah_coverage(
        tmp_path,
        now=datetime(2026, 10, 1, tzinfo=UTC),
    )

    ionq = next(row for row in report["instruments"] if row["symbol"] == "IONQ")
    assert ionq["status"] == "SHARIAH_ATTESTATION_EXPIRED"
    assert ionq["currently_eligible"] is False
