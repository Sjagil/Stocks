from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class AssetMetadata:
    asset_key: str
    symbol: str
    name: str | None
    con_id: int | None
    asset_type: str
    exchange: str | None
    currency: str | None
    sector: str | None
    industry: str | None
    category: str | None
    inactive: bool


@dataclass(frozen=True)
class FundamentalSnapshot:
    available_at: datetime | None
    net_income: float | None = None
    free_cash_flow: float | None = None
    revenue: float | None = None
    assets: float | None = None
    debt: float | None = None
    cash: float | None = None
    shares: float | None = None
    previous_shares: float | None = None
    operating_cash_flow: float | None = None
    dividends: float | None = None


@dataclass(frozen=True)
class ShariahSnapshot:
    status: str
    screened_at: datetime | None
    expires_at: datetime | None
    methodology: str | None
    source: str | None

    def eligible_at(self, decision_time: datetime) -> bool:
        return (
            self.status in {"SHARIAH_COMPLIANT", "SHARIAH_ELIGIBLE_PIT"}
            and self.screened_at is not None
            and self.screened_at <= decision_time
            and self.expires_at is not None
            and self.expires_at >= decision_time
        )


@dataclass(frozen=True)
class AssetSnapshot:
    metadata: AssetMetadata
    bars: pd.DataFrame
    price_source: str
    price_source_timestamp: datetime | None
    fundamental: FundamentalSnapshot | None
    shariah: ShariahSnapshot
    benchmark_symbol: str
    benchmark_bars: pd.DataFrame
    mover_type: str | None = None
    mover_return: float | None = None
    provider_conflict: bool = False
    provider_conflict_detail: dict[str, Any] = field(default_factory=dict)
    bid_ask_spread_bps: float | None = None


@dataclass(frozen=True)
class ScoreResult:
    public: dict[str, Any]
    private: dict[str, Any]
