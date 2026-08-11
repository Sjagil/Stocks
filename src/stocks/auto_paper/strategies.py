from __future__ import annotations

from typing import Any

from stocks.auto_paper.contracts import StrategyDecisionStatus


STRATEGY_IDS = (
    "QUALITY_GAINER_CONTINUATION_V1",
    "FALLEN_ANGEL_RECOVERY_V1",
    "EMERGING_COMPOUNDER_V1",
)
REQUIREMENTS = {
    "QUALITY_GAINER_CONTINUATION_V1": (
        "shariah_eligible",
        "material_event",
        "relative_volume_confirmation",
        "gap_retention",
        "positive_revision",
        "fundamental_quality",
        "sector_relative_strength",
        "no_dilution",
        "no_unexplained_pump",
    ),
    "FALLEN_ANGEL_RECOVERY_V1": (
        "shariah_eligible",
        "temporary_shock_probability",
        "strong_balance_sheet",
        "no_accounting_investigation",
        "no_permanent_impairment",
        "technical_stabilization",
        "valuation_reset",
    ),
    "EMERGING_COMPOUNDER_V1": (
        "shariah_eligible",
        "revenue_acceleration",
        "margin_expansion",
        "free_cash_flow_improvement",
        "positive_earnings_revisions",
        "persistent_relative_strength",
        "repeated_positive_events",
    ),
}


def evaluate_strategy(strategy_id: str, features: dict[str, Any]) -> dict[str, object]:
    required = REQUIREMENTS.get(strategy_id)
    if required is None:
        return {"strategy_id": strategy_id, "status": StrategyDecisionStatus.REJECTED, "failed_requirements": ["UNKNOWN_STRATEGY"]}
    passed = [name for name in required if bool(features.get(name, False))]
    failed = [name for name in required if name not in passed]
    ratio = len(passed) / len(required)
    if not failed:
        status = StrategyDecisionStatus.SHADOW_CANDIDATE
    elif ratio >= 0.6:
        status = StrategyDecisionStatus.WATCHLIST
    else:
        status = StrategyDecisionStatus.REJECTED
    return {
        "strategy_id": strategy_id,
        "status": status,
        "passed_requirements": passed,
        "failed_requirements": failed,
        "automatic_paper_eligibility": False,
        "order_authority": "NONE",
    }
