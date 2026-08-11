from __future__ import annotations

from datetime import date
from decimal import Decimal

from stocks.data.phase5_1 import _merge_currency_rates


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
