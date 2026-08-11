from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from stocks.alpha.data_contracts import ShariahStatus


class MoverType(StrEnum):
    TOP_GAINER = "TOP_GAINER"
    TOP_LOSER = "TOP_LOSER"
    PERSISTENT_LEADER = "PERSISTENT_LEADER"


class NewsAttribution(StrEnum):
    CONFIRMED_COMPANY_EVENT = "CONFIRMED_COMPANY_EVENT"
    CONFIRMED_SECTOR_EVENT = "CONFIRMED_SECTOR_EVENT"
    CONFIRMED_MACRO_EVENT = "CONFIRMED_MACRO_EVENT"
    MULTIPLE_EVENTS = "MULTIPLE_EVENTS"
    NO_MATERIAL_NEWS_FOUND = "NO_MATERIAL_NEWS_FOUND"
    UNVERIFIED_RUMOR = "UNVERIFIED_RUMOR"
    DUPLICATE_NEWS_ONLY = "DUPLICATE_NEWS_ONLY"


class CandidateState(StrEnum):
    DISCOVERED = "DISCOVERED"
    NEWS_PENDING = "NEWS_PENDING"
    EVENT_VALIDATED = "EVENT_VALIDATED"
    SHARIAH_VALIDATED = "SHARIAH_VALIDATED"
    FUNDAMENTAL_REVIEW = "FUNDAMENTAL_REVIEW"
    TECHNICAL_CONFIRMATION_PENDING = "TECHNICAL_CONFIRMATION_PENDING"
    WATCHLIST = "WATCHLIST"
    ENTRY_READY = "ENTRY_READY"
    POSITION_ACTIVE = "POSITION_ACTIVE"
    ADD_ALLOWED = "ADD_ALLOWED"
    REDUCE = "REDUCE"
    EXIT = "EXIT"
    REJECTED_PUMP = "REJECTED_PUMP"
    REJECTED_VALUE_TRAP = "REJECTED_VALUE_TRAP"
    REJECTED_SHARIAH = "REJECTED_SHARIAH"
    REJECTED_PERMANENT_IMPAIRMENT = "REJECTED_PERMANENT_IMPAIRMENT"


class MoverClassification(StrEnum):
    G1_STRUCTURAL_BREAKOUT = "G1_STRUCTURAL_BREAKOUT"
    G2_EARNINGS_REPRICING = "G2_EARNINGS_REPRICING"
    G3_CYCLICAL_RECOVERY = "G3_CYCLICAL_RECOVERY"
    G4_SHORT_TERM_EVENT = "G4_SHORT_TERM_EVENT"
    G5_SPECULATIVE_SQUEEZE = "G5_SPECULATIVE_SQUEEZE"
    G6_LOW_QUALITY_PUMP = "G6_LOW_QUALITY_PUMP"
    L1_TRANSIENT_OVERREACTION = "L1_TRANSIENT_OVERREACTION"
    L2_FUNDAMENTAL_RESET = "L2_FUNDAMENTAL_RESET"
    L3_TURNAROUND_CANDIDATE = "L3_TURNAROUND_CANDIDATE"
    L4_VALUE_TRAP_RISK = "L4_VALUE_TRAP_RISK"
    L5_PERMANENT_IMPAIRMENT = "L5_PERMANENT_IMPAIRMENT"
    SECULAR_GROWTH_GEM = "SECULAR_GROWTH_GEM"
    CYCLICAL_RECOVERY_GEM = "CYCLICAL_RECOVERY_GEM"
    TURNAROUND_GEM = "TURNAROUND_GEM"
    PROFITABILITY_INFLECTION_GEM = "PROFITABILITY_INFLECTION_GEM"


@dataclass(frozen=True)
class MoverObservation:
    instrument_id: str
    mover_type: MoverType
    shariah_status: ShariahStatus
    news_attribution: NewsAttribution
    event_quality: float = 0.0
    earnings_revision_score: float = 0.0
    fundamental_inflection: float = 0.0
    volume_confirmation: float = 0.0
    gap_retention: float = 0.0
    sector_relative_strength: float = 0.0
    valuation_room: float = 0.0
    business_quality: float = 0.0
    balance_sheet_strength: float = 0.0
    temporary_shock_probability: float = 0.0
    valuation_reset: float = 0.0
    revision_stabilization: float = 0.0
    technical_stabilization: float = 0.0
    insider_confirmation: float = 0.0
    revenue_acceleration: float = 0.0
    margin_expansion: float = 0.0
    fcf_improvement: float = 0.0
    relative_strength: float = 0.0
    positive_weeks: float = 0.0
    higher_lows: float = 0.0
    accumulation_volume: float = 0.0
    revision_persistence: float = 0.0
    repeated_positive_events: float = 0.0
    permanent_impairment: bool = False
    low_quality_pump_risk: bool = False


def closing_location_value(high: float, low: float, close: float) -> float:
    if high <= low:
        return 0.0
    return ((close - low) - (high - close)) / (high - low)


def gainer_score(obs: MoverObservation) -> float:
    score = (
        0.20 * obs.event_quality
        + 0.20 * obs.earnings_revision_score
        + 0.15 * obs.fundamental_inflection
        + 0.15 * obs.volume_confirmation
        + 0.10 * obs.gap_retention
        + 0.10 * obs.sector_relative_strength
        + 0.10 * obs.valuation_room
    )
    if obs.news_attribution == NewsAttribution.NO_MATERIAL_NEWS_FOUND:
        score -= 0.25
    if obs.low_quality_pump_risk:
        score -= 0.35
    return _clamp(score)


def recovery_score(obs: MoverObservation) -> float:
    score = (
        0.20 * obs.business_quality
        + 0.20 * obs.balance_sheet_strength
        + 0.20 * obs.temporary_shock_probability
        + 0.15 * obs.valuation_reset
        + 0.10 * obs.revision_stabilization
        + 0.10 * obs.technical_stabilization
        + 0.05 * obs.insider_confirmation
    )
    if obs.permanent_impairment:
        score -= 1.0
    return _clamp(score)


def gem_score(obs: MoverObservation) -> float:
    inflection = (
        0.20 * obs.revenue_acceleration
        + 0.20 * obs.margin_expansion
        + 0.20 * obs.fcf_improvement
        + 0.20 * obs.earnings_revision_score
        + 0.20 * obs.relative_strength
    )
    persistence = (
        0.25 * obs.positive_weeks
        + 0.25 * obs.higher_lows
        + 0.20 * obs.accumulation_volume
        + 0.15 * obs.revision_persistence
        + 0.15 * obs.repeated_positive_events
    )
    return _clamp(0.45 * inflection + 0.30 * persistence + 0.15 * obs.business_quality + 0.10 * obs.valuation_room)


def classify_mover(obs: MoverObservation) -> dict[str, object]:
    if obs.shariah_status != ShariahStatus.ELIGIBLE:
        return _result(obs, CandidateState.REJECTED_SHARIAH, "SHARIAH_BLOCKED", 0.0)
    if obs.permanent_impairment:
        return _result(obs, CandidateState.REJECTED_PERMANENT_IMPAIRMENT, MoverClassification.L5_PERMANENT_IMPAIRMENT.value, 0.0)
    if obs.mover_type == MoverType.TOP_GAINER:
        score = gainer_score(obs)
        if obs.low_quality_pump_risk or score < 0.35:
            state = CandidateState.REJECTED_PUMP
            label = MoverClassification.G6_LOW_QUALITY_PUMP.value
        elif score >= 0.70:
            state = CandidateState.ENTRY_READY
            label = MoverClassification.G1_STRUCTURAL_BREAKOUT.value
        elif score >= 0.55:
            state = CandidateState.WATCHLIST
            label = MoverClassification.G2_EARNINGS_REPRICING.value
        else:
            state = CandidateState.TECHNICAL_CONFIRMATION_PENDING
            label = MoverClassification.G4_SHORT_TERM_EVENT.value
        return _result(obs, state, label, score)
    if obs.mover_type == MoverType.TOP_LOSER:
        score = recovery_score(obs)
        if score >= 0.65:
            return _result(obs, CandidateState.WATCHLIST, MoverClassification.L1_TRANSIENT_OVERREACTION.value, score)
        if score >= 0.50:
            return _result(obs, CandidateState.FUNDAMENTAL_REVIEW, MoverClassification.L3_TURNAROUND_CANDIDATE.value, score)
        return _result(obs, CandidateState.REJECTED_VALUE_TRAP, MoverClassification.L4_VALUE_TRAP_RISK.value, score)
    score = gem_score(obs)
    if score >= 0.70:
        return _result(obs, CandidateState.ENTRY_READY, MoverClassification.SECULAR_GROWTH_GEM.value, score)
    if score >= 0.55:
        return _result(obs, CandidateState.WATCHLIST, MoverClassification.PROFITABILITY_INFLECTION_GEM.value, score)
    return _result(obs, CandidateState.FUNDAMENTAL_REVIEW, MoverClassification.TURNAROUND_GEM.value, score)


def _result(obs: MoverObservation, state: CandidateState, classification: str, score: float) -> dict[str, object]:
    return {
        "instrument_id": obs.instrument_id,
        "mover_type": obs.mover_type.value,
        "candidate_state": state.value,
        "classification": classification,
        "score": round(score, 6),
        "news_attribution": obs.news_attribution.value,
        "shariah_status": obs.shariah_status.value,
        "automatic_buy_signal": False,
    }


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
