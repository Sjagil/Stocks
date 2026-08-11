from __future__ import annotations

from stocks.alpha.data_contracts import AlphaDecision, AlphaInputs
from stocks.alpha.strategies.common import blocked_or_common_decision, finalize_decision

STRATEGY_ID = "EARNINGS_REVISION_GUIDANCE_TREND_V1"


def decide(inputs: AlphaInputs) -> AlphaDecision:
    blocked = blocked_or_common_decision(STRATEGY_ID, inputs)
    if blocked is not None:
        return blocked
    components = {
        "revision": inputs.revision_score,
        "earnings_surprise": inputs.earnings_surprise_score,
        "guidance": inputs.guidance_event_score,
        "quality": inputs.quality_score,
        "technical": inputs.technical_confirmation_score,
        "catalyst": max(inputs.guidance_event_score, inputs.earnings_surprise_score, inputs.catalyst_score),
    }
    alpha_score = (
        0.30 * components["revision"]
        + 0.25 * components["earnings_surprise"]
        + 0.20 * components["guidance"]
        + 0.15 * components["quality"]
        + 0.10 * components["technical"]
    )
    return finalize_decision(STRATEGY_ID, inputs, alpha_score, components)
