from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any

from stocks.execution.idempotency import stable_hash


class AssetGroup(StrEnum):
    SHARIAH_STOCK = "SHARIAH_STOCK"
    APPROVED_SHARIAH_EQUITY_ETF = "APPROVED_SHARIAH_EQUITY_ETF"
    APPROVED_PHYSICAL_COMMODITY_PRODUCT = "APPROVED_PHYSICAL_COMMODITY_PRODUCT"


class SignalStatus(StrEnum):
    DISCOVERED = "SIGNAL_DISCOVERED"
    VALIDATED = "SIGNAL_VALIDATED"
    EXPIRED = "SIGNAL_EXPIRED"
    DUPLICATE = "SIGNAL_DUPLICATE"
    SHARIAH_BLOCKED = "SIGNAL_SHARIAH_BLOCKED"
    DATA_STALE = "SIGNAL_DATA_STALE"
    RISK_BLOCKED = "SIGNAL_RISK_BLOCKED"
    SHADOW_ONLY = "SIGNAL_SHADOW_ONLY"
    AUTO_PAPER_ELIGIBLE = "SIGNAL_AUTO_PAPER_ELIGIBLE"


class StrategyDecisionStatus(StrEnum):
    REJECTED = "REJECTED"
    WATCHLIST = "WATCHLIST"
    SHADOW_CANDIDATE = "SHADOW_CANDIDATE"


class Regime(StrEnum):
    RISK_ON = "RISK_ON"
    INFLATION_SUPPLY_SHOCK = "INFLATION_SUPPLY_SHOCK"
    GEOPOLITICAL_SUPPLY_SHOCK = "GEOPOLITICAL_SUPPLY_SHOCK"
    RECESSION_DEMAND_SHOCK = "RECESSION_DEMAND_SHOCK"
    LIQUIDITY_STRESS = "LIQUIDITY_STRESS"


@dataclass(frozen=True)
class AutoSignal:
    signal_id: str
    strategy_id: str
    strategy_version: str
    generated_at: str
    available_at: str
    expires_at: str
    session_date: str
    con_id: int
    symbol: str
    exchange: str
    currency: str
    security_type: str
    asset_group: str
    side: str
    target_quantity: Decimal
    reference_price: Decimal
    maximum_limit_price: Decimal
    entry_reason: str
    exit_reason: str | None
    confidence: Decimal
    expected_holding_period: str
    source_provenance: dict[str, str]
    source_provenance_hash: str
    feature_snapshot_hash: str
    portfolio_snapshot_hash: str
    shariah_snapshot_hash: str

    def economic_key(self, account_fingerprint: str) -> str:
        return stable_hash(
            {
                "account_fingerprint": account_fingerprint,
                "strategy_id": self.strategy_id,
                "strategy_version": self.strategy_version,
                "signal_id": self.signal_id,
                "con_id": self.con_id,
                "side": self.side,
                "session_date": self.session_date,
            }
        )


@dataclass(frozen=True)
class ShariahSnapshot:
    status: str
    methodology: str
    methodology_version: str
    screened_at: str
    financials_available_at: str
    expires_at: str
    business_activity_pass: bool
    financial_ratio_pass: bool
    non_permissible_income_pass: bool
    product_structure: str
    underlying_assets: tuple[str, ...] = ()
    physical_backing: bool = False
    derivatives_exposure: bool = False
    leverage: bool = False
    short_exposure: bool = False
    securities_lending: bool = False
    interest_bearing_cash: bool = False
    currency_hedging: bool = False
    shariah_certificate: bool = False

    @property
    def snapshot_hash(self) -> str:
        return stable_hash(model_to_jsonable(self))


@dataclass(frozen=True)
class MarketQuote:
    bid: Decimal
    ask: Decimal
    observed_at: str

    @property
    def midpoint(self) -> Decimal:
        return (self.bid + self.ask) / Decimal("2")

    @property
    def spread_bps(self) -> Decimal:
        return Decimal("0") if self.midpoint == 0 else ((self.ask - self.bid) / self.midpoint) * Decimal("10000")


@dataclass(frozen=True)
class Position:
    con_id: int
    quantity: Decimal
    market_value_eur: Decimal
    sector: str
    event_cluster: str


@dataclass(frozen=True)
class PortfolioState:
    nav_eur: Decimal
    exposure_eur: Decimal
    daily_pnl_eur: Decimal
    positions: tuple[Position, ...]
    sector_exposure_pct: dict[str, Decimal]
    event_cluster_exposure_pct: dict[str, Decimal]
    reconciliation_status: str
    snapshot_complete: bool


@dataclass(frozen=True)
class AutoDecision:
    status: str
    decision_code: str
    authority: str
    signal_hash: str
    risk_hash: str
    hypothetical_order: dict[str, Any] | None = None


def model_to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return model_to_jsonable(asdict(value))
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return [model_to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [model_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_to_jsonable(item) for key, item in value.items()}
    return value
