from __future__ import annotations

from stocks.alpha.data_contracts import AlphaDecision, AlphaInputs
from stocks.alpha.strategies.common import blocked_or_common_decision, finalize_decision

STRATEGY_ID = "FUNDAMENTAL_INFLECTION_NEWS_CATALYST_V1"


def decide(inputs: AlphaInputs) -> AlphaDecision:
    blocked = blocked_or_common_decision(STRATEGY_ID, inputs)
    if blocked is not None:
        return blocked
    inflection = 0.50 * inputs.quality_score + 0.30 * inputs.revision_score + 0.20 * max(0.0, 1.0 - inputs.balance_sheet_risk)
    components = {
        "fundamental_inflection": inflection,
        "catalyst": inputs.catalyst_score,
        "technical": inputs.technical_confirmation_score,
        "negative_news": inputs.negative_news_score,
    }
    alpha_score = 0.45 * inflection + 0.35 * inputs.catalyst_score + 0.20 * inputs.technical_confirmation_score
    return finalize_decision(STRATEGY_ID, inputs, alpha_score, components)
