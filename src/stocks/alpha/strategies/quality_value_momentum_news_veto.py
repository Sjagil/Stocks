from __future__ import annotations

from stocks.alpha.data_contracts import AlphaDecision, AlphaInputs
from stocks.alpha.strategies.common import blocked_or_common_decision, finalize_decision

STRATEGY_ID = "QUALITY_VALUE_MOMENTUM_NEWS_VETO_V1"


def decide(inputs: AlphaInputs) -> AlphaDecision:
    blocked = blocked_or_common_decision(STRATEGY_ID, inputs)
    if blocked is not None:
        return blocked
    momentum = inputs.metadata.get("momentum_score", inputs.technical_confirmation_score)
    base = 0.40 * inputs.quality_score + 0.30 * inputs.value_score + 0.30 * float(momentum)
    components = {
        "quality": inputs.quality_score,
        "value": inputs.value_score,
        "momentum": float(momentum),
        "technical": inputs.technical_confirmation_score,
        "catalyst": max(0.20, inputs.catalyst_score),
        "negative_news": inputs.negative_news_score,
    }
    alpha_score = base * (1.0 - min(0.50, inputs.negative_news_score))
    return finalize_decision(STRATEGY_ID, inputs, alpha_score, components)
