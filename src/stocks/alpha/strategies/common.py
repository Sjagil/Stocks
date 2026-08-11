from __future__ import annotations

from stocks.alpha.data_contracts import (
    AlphaDecision,
    AlphaInputs,
    DecisionStatus,
    PITDataStatus,
    ShariahStatus,
)
from stocks.alpha.portfolio.sizing import size_position


def blocked_or_common_decision(strategy_id: str, inputs: AlphaInputs) -> AlphaDecision | None:
    if inputs.pit_status != PITDataStatus.VALID:
        return _decision(strategy_id, inputs, DecisionStatus.BLOCKED_DATA, 0.0, 0.0, (inputs.pit_status.value,), {})
    if inputs.shariah_status != ShariahStatus.ELIGIBLE:
        return _decision(strategy_id, inputs, DecisionStatus.BLOCKED_SHARIAH, 0.0, 0.0, (inputs.shariah_status.value,), {})
    if inputs.negative_news_score >= 0.80:
        return _decision(
            strategy_id,
            inputs,
            DecisionStatus.EXIT_RISK_EVENT,
            0.0,
            0.0,
            ("NEGATIVE_NEWS_RISK_OVERLAY_BLOCK",),
            {"negative_news_score": inputs.negative_news_score},
        )
    if inputs.balance_sheet_risk > 0.70:
        return _decision(
            strategy_id,
            inputs,
            DecisionStatus.BLOCKED_BALANCE_SHEET,
            0.0,
            0.0,
            ("BALANCE_SHEET_RISK_GT_0_70",),
            {"balance_sheet_risk": inputs.balance_sheet_risk},
        )
    if inputs.valuation_risk > 0.80:
        return _decision(
            strategy_id,
            inputs,
            DecisionStatus.BLOCKED_VALUATION,
            0.0,
            0.0,
            ("VALUATION_RISK_GT_0_80",),
            {"valuation_risk": inputs.valuation_risk},
        )
    return None


def finalize_decision(strategy_id: str, inputs: AlphaInputs, alpha_score: float, components: dict[str, float]) -> AlphaDecision:
    if components.get("catalyst", 0.0) < 0.20:
        return _decision(strategy_id, inputs, DecisionStatus.WATCH_CATALYST, alpha_score, 0.0, ("CATALYST_LT_0_20",), components)
    if components.get("technical", 0.0) < 0.50:
        return _decision(
            strategy_id,
            inputs,
            DecisionStatus.WATCH_TECHNICAL_CONFIRMATION,
            alpha_score,
            0.0,
            ("TECHNICAL_CONFIRMATION_LT_0_50",),
            components,
        )
    if alpha_score < 0.60:
        return _decision(
            strategy_id,
            inputs,
            DecisionStatus.WATCH_INSUFFICIENT_ALPHA,
            alpha_score,
            0.0,
            ("ALPHA_SCORE_LT_0_60",),
            components,
        )
    target = size_position(
        alpha_confidence=alpha_score,
        regime_multiplier=inputs.macro_regime_multiplier,
        volatility=inputs.volatility,
    )
    return _decision(strategy_id, inputs, DecisionStatus.ENTRY_READY, alpha_score, target, (), components)


def _decision(
    strategy_id: str,
    inputs: AlphaInputs,
    status: DecisionStatus,
    alpha_score: float,
    target_weight: float,
    reasons: tuple[str, ...],
    components: dict[str, float],
) -> AlphaDecision:
    return AlphaDecision(
        strategy_id=strategy_id,
        instrument_id=inputs.instrument_id,
        decision_timestamp=inputs.decision_timestamp,
        status=status,
        alpha_score=round(alpha_score, 6),
        target_weight=round(target_weight, 6),
        rejection_reasons=reasons,
        component_scores={key: round(value, 6) for key, value in components.items()},
    )
