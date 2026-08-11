from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


PHASE11_10_STRATEGY_VERSION = "PHASE11_10_MTF_V1"
PRICE_FIELDS = (
    "current_market_price",
    "entry_price_reference",
    "preferred_entry",
    "limit_entry_price",
    "entry_zone_low",
    "entry_zone_high",
    "stop_loss",
    "trailing_stop",
    "invalidation_level",
    "take_profit_1",
    "take_profit_2",
)


def normalize_research_signal_price_basis(
    project_root: Path,
    signal: dict[str, Any],
) -> dict[str, Any]:
    """Reverse Phase 11.10 EUR research prices to the USD quote basis."""
    row = dict(signal)
    if row.get("strategy_version") != PHASE11_10_STRATEGY_VERSION:
        return row
    if row.get("order_geometry_price_basis") == (
        "LOCAL_QUOTE_CURRENCY_FROM_SOURCE_PIT_FX_REVERSAL"
    ):
        return row
    timestamp = row.get("data_timestamp") or row.get("bar_timestamp")
    factor = _lagged_eur_per_usd(project_root, timestamp)
    if factor is None or factor <= 0:
        risks = {str(value) for value in row.get("risks", [])}
        risks.add("SIGNAL_PRICE_BASIS_NORMALIZATION_BLOCKED")
        row.update(
            action="AVOID",
            data_freshness="STALE",
            lifecycle_status="INVALIDATED",
            price_validity_status="SIGNAL_PRICE_BASIS_NORMALIZATION_BLOCKED",
            risks=sorted(risks),
            order_geometry_price_basis="UNVERIFIED_BLOCKED",
        )
        return row
    for field in PRICE_FIELDS:
        value = _finite(row.get(field))
        if value is not None:
            row[field] = f"{value / factor:.6f}"
    row.update(
        currency="USD",
        portfolio_base_currency="EUR",
        quote_to_base_fx=f"{factor:.8f}",
        source_price_basis="EUR_PORTFOLIO_BASE_FROM_LAGGED_PIT_FX",
        order_geometry_price_basis=(
            "LOCAL_QUOTE_CURRENCY_FROM_SOURCE_PIT_FX_REVERSAL"
        ),
        price_basis_normalization_status="GO",
    )
    return row


def _lagged_eur_per_usd(
    project_root: Path,
    timestamp: Any,
) -> float | None:
    path = (
        project_root
        / "data"
        / "research"
        / "phase11_4"
        / "private"
        / "eurusd.parquet"
    )
    if not path.is_file():
        return None
    try:
        frame = pd.read_parquet(path)
    except (OSError, ValueError):
        return None
    if not {"date", "usd_per_eur"}.issubset(frame.columns):
        return None
    observed = pd.Timestamp(timestamp)
    if observed.tzinfo is not None:
        observed = observed.tz_convert("UTC").tz_localize(None)
    session = observed.normalize()
    fx = frame.copy()
    fx["date"] = pd.to_datetime(fx["date"], errors="coerce")
    if getattr(fx["date"].dt, "tz", None) is not None:
        fx["date"] = fx["date"].dt.tz_convert("UTC").dt.tz_localize(None)
    fx["usd_per_eur"] = pd.to_numeric(
        fx["usd_per_eur"], errors="coerce"
    )
    fx = fx.dropna(subset=["date", "usd_per_eur"]).sort_values("date")
    fx = fx.loc[fx["usd_per_eur"].gt(0)].drop_duplicates("date", keep="last")
    if fx.empty:
        return None
    eur_per_usd = (1.0 / fx.set_index("date")["usd_per_eur"]).shift(1)
    eligible = eur_per_usd.loc[eur_per_usd.index <= session].dropna()
    if eligible.empty:
        return None
    value = float(eligible.iloc[-1])
    return value if math.isfinite(value) and value > 0 else None


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


__all__ = ["normalize_research_signal_price_basis"]
