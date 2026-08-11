from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class OpportunityAssetClass(StrEnum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    COMMODITY_EXPOSURE = "COMMODITY_EXPOSURE"
    CASH = "CASH"


class OpportunityShariahStatus(StrEnum):
    ALLOWED = "SHARIAH_ALLOWED"
    BLOCKED = "SHARIAH_BLOCKED"
    REVIEW_REQUIRED = "SHARIAH_REVIEW_REQUIRED"
    DATA_MISSING = "SHARIAH_DATA_MISSING"


class ValidationStatus(StrEnum):
    RAW = "RAW_OPPORTUNITY"
    STAGE0_SURVIVOR = "STAGE0_FALSIFICATION_SURVIVOR"
    EXACT_VALIDATION_REQUIRED = "EXACT_VALIDATION_REQUIRED"
    VALIDATED = "VALIDATED_OPPORTUNITY"
    REJECTED = "VALIDATION_REJECTED"


@dataclass(frozen=True)
class NormalizedOpportunity:
    instrument_id: str
    symbol: str
    asset_class: str
    subclass: str
    strategy_family: str
    timeframe: str
    signal_timestamp: str
    signal_expiry: str | None
    direction: str
    expected_return: float | None
    expected_loss: float | None
    expected_net_return: float | None
    expected_r: float | None
    confidence: float
    volatility: float | None
    liquidity: float | None
    spread_bps: float | None
    estimated_slippage_bps: float
    fees_eur: float | None
    event_risk: float
    regime_fit: float
    relative_strength: float
    data_quality: float
    shariah_status: str
    broker_resolvable: bool
    whole_share_feasibility: str
    correlation_cluster: str
    validation_status: str
    research_eligible: bool
    portfolio_eligible: bool
    execution_eligible: bool
    blockers: tuple[str, ...] = ()
    source: str = "NATIVE_P1"

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blockers"] = list(self.blockers)
        payload["schema"] = "normalized_cross_asset_opportunity_v1"
        payload["execution_authority"] = "NONE"
        return payload


@dataclass(frozen=True)
class DesiredPortfolioTarget:
    instrument_id: str
    symbol: str
    asset_class: str
    desired_quantity: int
    desired_exposure_eur: float
    current_quantity: float
    quantity_delta: float
    action: str
    reason: str
    strategy_source: str
    confidence: float
    expected_net_return: float | None
    expected_loss: float | None
    priority: float
    expiry: str | None
    constraints: tuple[str, ...]
    rebalance_threshold_eur: float
    entry_reference: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    currency: str | None = None
    fx_rate_to_eur: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["constraints"] = list(self.constraints)
        payload["schema"] = "desired_portfolio_target_v1"
        payload["submits_orders"] = False
        payload["execution_authority"] = "NONE"
        return payload


__all__ = [
    "DesiredPortfolioTarget",
    "NormalizedOpportunity",
    "OpportunityAssetClass",
    "OpportunityShariahStatus",
    "ValidationStatus",
]
