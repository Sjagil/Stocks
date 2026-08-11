from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class PITDataStatus(str, Enum):
    VALID = "PIT_VALID"
    FUTURE_DATED_BLOCKED = "PIT_FUTURE_DATED_BLOCKED"
    MISSING_AVAILABLE_AT = "PIT_MISSING_AVAILABLE_AT"
    REVISED_AFTER_DECISION_BLOCKED = "PIT_REVISED_AFTER_DECISION_BLOCKED"


class InstrumentType(str, Enum):
    STOCK = "STOCK"
    SHARIAH_EQUITY_ETF = "SHARIAH_EQUITY_ETF"
    APPROVED_PHYSICAL_COMMODITY_PRODUCT = "APPROVED_PHYSICAL_COMMODITY_PRODUCT"
    BOND = "BOND"
    FUTURE = "FUTURE"
    OPTION = "OPTION"
    SWAP = "SWAP"
    CFD = "CFD"
    SHORT = "SHORT"
    LEVERAGED_ETF = "LEVERAGED_ETF"
    INVERSE_ETF = "INVERSE_ETF"
    SYNTHETIC_ETF = "SYNTHETIC_ETF"


class ShariahStatus(str, Enum):
    ELIGIBLE = "SHARIAH_ELIGIBLE"
    INELIGIBLE = "SHARIAH_INELIGIBLE"
    STALE = "SHARIAH_STATUS_STALE"
    STRUCTURE_REVIEW_REQUIRED = "STRUCTURE_REVIEW_REQUIRED"
    DERIVATIVE_EXPOSURE_BLOCKED = "DERIVATIVE_EXPOSURE_BLOCKED"
    FUTURES_EXPOSURE_BLOCKED = "FUTURES_EXPOSURE_BLOCKED"
    INTEREST_EXPOSURE_BLOCKED = "INTEREST_EXPOSURE_BLOCKED"
    MISSING_CERTIFICATE = "MISSING_SHARIAH_CERTIFICATE"


class DecisionStatus(str, Enum):
    BLOCKED_DATA = "BLOCKED_DATA"
    BLOCKED_SHARIAH = "BLOCKED_SHARIAH"
    BLOCKED_NEGATIVE_EVENT = "BLOCKED_NEGATIVE_EVENT"
    BLOCKED_VALUATION = "BLOCKED_VALUATION"
    BLOCKED_BALANCE_SHEET = "BLOCKED_BALANCE_SHEET"
    WATCH_FUNDAMENTAL = "WATCH_FUNDAMENTAL"
    WATCH_CATALYST = "WATCH_CATALYST"
    WATCH_TECHNICAL_CONFIRMATION = "WATCH_TECHNICAL_CONFIRMATION"
    WATCH_INSUFFICIENT_ALPHA = "WATCH_INSUFFICIENT_ALPHA"
    ENTRY_READY = "ENTRY_READY"
    HOLD = "HOLD"
    ADD_ALLOWED = "ADD_ALLOWED"
    REDUCE = "REDUCE"
    EXIT_THESIS_BROKEN = "EXIT_THESIS_BROKEN"
    EXIT_TIME_STOP = "EXIT_TIME_STOP"
    EXIT_RISK_EVENT = "EXIT_RISK_EVENT"


class Regime(str, Enum):
    RISK_ON = "RISK_ON"
    INFLATIONARY_SUPPLY_SHOCK = "INFLATIONARY_SUPPLY_SHOCK"
    GEOPOLITICAL_ENERGY_SHOCK = "GEOPOLITICAL_ENERGY_SHOCK"
    RECESSION_DEMAND_SHOCK = "RECESSION_DEMAND_SHOCK"
    CREDIT_STRESS = "CREDIT_STRESS"
    MIXED_NEUTRAL = "MIXED_NEUTRAL"


def parse_timestamp(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class PointInTimeFact:
    fact_id: str
    entity_id: str
    field_name: str
    value: float | str | int | bool | None
    event_time: datetime | None
    published_at: datetime | None
    first_seen_at: datetime | None
    available_at: datetime | None
    ingested_at: datetime | None
    revised_at: datetime | None
    source: str
    source_hash: str
    reporting_period: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> PointInTimeFact:
        return cls(
            fact_id=str(payload["fact_id"]),
            entity_id=str(payload["entity_id"]),
            field_name=str(payload["field_name"]),
            value=payload.get("value"),
            event_time=parse_timestamp(payload.get("event_time")),
            published_at=parse_timestamp(payload.get("published_at")),
            first_seen_at=parse_timestamp(payload.get("first_seen_at")),
            available_at=parse_timestamp(payload.get("available_at")),
            ingested_at=parse_timestamp(payload.get("ingested_at")),
            revised_at=parse_timestamp(payload.get("revised_at")),
            source=str(payload.get("source", "UNKNOWN")),
            source_hash=str(payload.get("source_hash", "")),
            reporting_period=payload.get("reporting_period"),
        )


@dataclass(frozen=True)
class ShariahScreen:
    instrument_id: str
    instrument_type: InstrumentType
    compliance_status: ShariahStatus
    screening_methodology: str
    methodology_version: str
    screened_at: datetime
    financial_statement_available_at: datetime | None
    has_derivatives: bool = False
    has_futures: bool = False
    has_leverage: bool = False
    has_short_exposure: bool = False
    has_interest_bearing_cash: bool = False
    has_shariah_certificate: bool = False
    rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlphaInputs:
    instrument_id: str
    decision_timestamp: datetime
    pit_status: PITDataStatus
    shariah_status: ShariahStatus
    quality_score: float = 0.0
    value_score: float = 0.0
    revision_score: float = 0.0
    earnings_surprise_score: float = 0.0
    guidance_event_score: float = 0.0
    catalyst_score: float = 0.0
    technical_confirmation_score: float = 0.0
    negative_news_score: float = 0.0
    balance_sheet_risk: float = 0.0
    valuation_risk: float = 0.0
    macro_regime_multiplier: float = 1.0
    volatility: float = 0.20
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlphaDecision:
    strategy_id: str
    instrument_id: str
    decision_timestamp: datetime
    status: DecisionStatus
    alpha_score: float
    target_weight: float
    rejection_reasons: tuple[str, ...]
    component_scores: dict[str, float]
    authority: str = "NONE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "instrument_id": self.instrument_id,
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "status": self.status.value,
            "alpha_score": self.alpha_score,
            "target_weight": self.target_weight,
            "rejection_reasons": list(self.rejection_reasons),
            "component_scores": self.component_scores,
            "authority": self.authority,
        }
