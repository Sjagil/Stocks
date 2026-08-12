from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from stocks.universe import (
    broad_asset_metadata,
    broad_commodity_symbols,
    broad_etf_symbols,
    broad_universe,
    broad_universe_status,
    commodity_product_attestations,
)


def test_repository_broad_universe_has_cross_asset_coverage() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = broad_universe(root)
    status = broad_universe_status(root)
    metadata = broad_asset_metadata(root)

    assert len(rows) >= 70
    assert status["group_count"] >= 7
    assert status["region_count"] >= 15
    assert status["sector_count"] >= 30
    assert {"SPY", "EWJ", "INDA", "MCHI"} <= broad_etf_symbols(root)
    assert {"SPUS", "HLAL", "UMMA", "SPSK"} <= broad_etf_symbols(root)
    assert {
        "COPX",
        "CPER",
        "GLD",
        "SCOP",
        "URA",
        "URNM",
        "USO",
    } <= broad_commodity_symbols(root)
    assert metadata["TLT"]["sleeve"] == "bond_defensive"
    assert metadata["SPUS"]["asset_type"] == "ETF"
    assert metadata["HLAL"]["sector"] == "SHARIAH_US_EQUITY"
    assert metadata["UMMA"]["region"] == "GLOBAL_EX_US"
    assert metadata["SPSK"]["asset_type"] == "SUKUK_ETF"
    assert metadata["SPSK"]["sleeve"] == "bond_defensive"
    assert metadata["XLE"]["sector"] == "ENERGY"
    assert metadata["SCOP"]["product_structure"] == (
        "PHYSICAL_CLOSED_END_TRUST"
    )
    assert metadata["SCOP"]["underlying_commodity"] == "COPPER"
    assert metadata["SCOP"]["physical_structure_verified"] is True
    assert metadata["SCOP"]["shariah_product_status"] == (
        "ATTESTATION_REQUIRED"
    )
    assert metadata["URA"]["commodity_exposure_type"] == (
        "PRODUCER_EQUITY_BASKET"
    )
    assert status["commodity_product_structure_count"] >= 4
    assert status["commodity_underlying_count"] >= 15
    assert status["physical_commodity_claim_count"] == 8
    assert status["current_physical_structure_verified_count"] == 8
    assert status["shariah_product_eligible_count"] == 0
    assert status["shariah_product_review_required_count"] == 8
    assert status["physical_structure_does_not_imply_shariah_eligibility"] is True
    assert status["automatic_execution_authority"] == "NONE"


def test_broad_universe_accepts_structured_product_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config" / "universes"
    path.mkdir(parents=True)
    (path / "broad_multi_asset_v1.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "group": "commodity",
                        "asset_type": "COMMODITY_CLOSED_END_TRUST",
                        "sleeve": "commodity_security",
                        "instruments": {
                            "SCOP": {
                                "sector": "COPPER",
                                "product_structure": (
                                    "PHYSICAL_CLOSED_END_TRUST"
                                ),
                                "commodity_exposure_type": (
                                    "PHYSICAL_COMMODITY"
                                ),
                                "underlying_commodity": "COPPER",
                                "primary_exchange": "ARCA",
                                "currency": "USD",
                            }
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    row = broad_universe(tmp_path)[0]

    assert row["product_structure"] == "PHYSICAL_CLOSED_END_TRUST"
    assert row["commodity_exposure_type"] == "PHYSICAL_COMMODITY"
    assert row["provider_symbol"] == "SCOP"
    assert broad_commodity_symbols(tmp_path) == {"SCOP"}


def test_broad_universe_rejects_duplicate_symbols(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config" / "universes"
    path.mkdir(parents=True)
    (path / "broad_multi_asset_v1.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "asset_type": "ETF",
                        "sleeve": "etf_core",
                        "instruments": {
                            "SPY": "BROAD",
                            "spy": "BROAD",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate broad universe symbol"):
        broad_universe(tmp_path)


def test_product_structure_attestation_expires_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "config" / "universes"
    path.mkdir(parents=True)
    (path / "commodity_product_attestations_v1.json").write_text(
        json.dumps(
            {
                "attestations": [
                    {
                        "symbol": "GLD",
                        "product_structure": "PHYSICAL_BACKED_GRANTOR_TRUST",
                        "commodity_exposure_type": "PHYSICAL_COMMODITY",
                        "structure_status": "VERIFIED_PHYSICAL_STRUCTURE",
                        "shariah_product_status": "ATTESTATION_REQUIRED",
                        "screened_at": "2026-07-01T00:00:00Z",
                        "expires_at": "2026-08-01T00:00:00Z",
                        "sources": ["https://issuer.example/official"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    row = commodity_product_attestations(
        tmp_path,
        observed_at=datetime(2026, 8, 12, tzinfo=UTC),
    )["GLD"]

    assert row["attestation_window_status"] == "EXPIRED"
    assert row["currently_verified"] is False
    assert row["shariah_product_eligible"] is False
