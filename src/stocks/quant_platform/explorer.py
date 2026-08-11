from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stocks.quant_platform.analytics import PerformanceRiskAnalyzer
from stocks.quant_platform.data import CanonicalMarketData, clean_market_data, resample_market_data
from stocks.quant_platform.storage import MultiAssetStore


class MultiAssetMarketDataExplorer:
    """One research-only façade over ingestion, storage and Level-1 analytics."""

    def __init__(self, root: Path, *, periods_per_year: int = 252):
        self.store = MultiAssetStore(root)
        self.analyzer = PerformanceRiskAnalyzer(periods_per_year=periods_per_year)

    def ingest(self, frame: pd.DataFrame) -> dict[str, Any]:
        return self.store.write(clean_market_data(frame))

    def observations(
        self,
        *,
        symbol: str | None = None,
        asset_class: str | None = None,
        as_of: Any | None = None,
        frequency: str | None = None,
    ) -> CanonicalMarketData:
        frame = self.store.read(symbol=symbol, asset_class=asset_class, as_of=as_of)
        if frequency:
            frame = resample_market_data(frame, frequency)
        return CanonicalMarketData(frame)

    def analyze(
        self,
        symbol: str,
        *,
        benchmark_symbol: str | None = None,
        as_of: Any | None = None,
        annual_risk_free_rate: float = 0.0,
    ) -> dict[str, Any]:
        frame = self.store.read(as_of=as_of)
        report = self.analyzer.analyze_market_data(
            frame,
            symbol,
            benchmark_symbol=benchmark_symbol,
            annual_risk_free_rate=annual_risk_free_rate,
        )
        return {
            "schema": "performance_risk_report_v1",
            "research_only": True,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "broker_writes": 0,
            **report,
        }
