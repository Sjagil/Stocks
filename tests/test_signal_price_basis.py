from __future__ import annotations

from pathlib import Path

import pandas as pd

from stocks.signals.price_basis import normalize_research_signal_price_basis


def _signal() -> dict[str, str]:
    return {
        "strategy_version": "PHASE11_10_MTF_V1",
        "data_timestamp": "2026-07-31T17:30:00+00:00",
        "action": "BUY",
        "data_freshness": "FRESH",
        "current_market_price": "80",
        "preferred_entry": "80",
        "entry_zone_low": "78",
        "entry_zone_high": "82",
        "stop_loss": "72",
        "take_profit_1": "92",
        "take_profit_2": "100",
    }


def test_phase11_10_signal_prices_are_reversed_to_usd_quote_basis(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "data"
        / "research"
        / "phase11_4"
        / "private"
        / "eurusd.parquet"
    )
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-07-29", "2026-07-30", "2026-07-31"]
            ),
            "usd_per_eur": [1.20, 1.25, 1.30],
        }
    ).to_parquet(path, index=False)
    result = normalize_research_signal_price_basis(tmp_path, _signal())
    assert float(result["preferred_entry"]) == 100.0
    assert float(result["stop_loss"]) == 90.0
    assert float(result["take_profit_1"]) == 115.0
    assert result["currency"] == "USD"
    assert result["quote_to_base_fx"] == "0.80000000"
    assert result["price_basis_normalization_status"] == "GO"
    assert result["order_geometry_price_basis"] == (
        "LOCAL_QUOTE_CURRENCY_FROM_SOURCE_PIT_FX_REVERSAL"
    )


def test_missing_fx_blocks_mtf_signal_instead_of_guessing(
    tmp_path: Path,
) -> None:
    result = normalize_research_signal_price_basis(tmp_path, _signal())
    assert result["action"] == "AVOID"
    assert result["data_freshness"] == "STALE"
    assert result["order_geometry_price_basis"] == "UNVERIFIED_BLOCKED"
    assert "SIGNAL_PRICE_BASIS_NORMALIZATION_BLOCKED" in result["risks"]


def test_unrelated_signal_is_unchanged(tmp_path: Path) -> None:
    source = {"strategy_version": "OTHER", "preferred_entry": "80"}
    assert normalize_research_signal_price_basis(tmp_path, source) == source
