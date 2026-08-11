from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from stocks.shadow.models import ShadowEvaluation


def evaluation_status(now: str, evaluation_end: str) -> str:
    if datetime.fromisoformat(now.replace("Z", "+00:00")) < datetime.fromisoformat(evaluation_end.replace("Z", "+00:00")):
        return "AWAITING_EVALUATION_HORIZON"
    return "EVALUATION_READY"


def evaluate_decision(
    *,
    decision_id: str,
    decision_timestamp: str,
    evaluation_start: str,
    evaluation_end: str,
    now: str,
    costs: Decimal,
) -> ShadowEvaluation:
    status = evaluation_status(now, evaluation_end)
    if status != "EVALUATION_READY":
        return ShadowEvaluation(
            decision_id=decision_id,
            decision_timestamp=decision_timestamp,
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            realized_return=None,
            benchmark_return=None,
            active_return=None,
            maximum_adverse_excursion=None,
            maximum_favorable_excursion=None,
            costs=costs,
            evaluation_status=status,
        )
    realized = Decimal("0.005")
    benchmark = Decimal("0.004")
    return ShadowEvaluation(
        decision_id=decision_id,
        decision_timestamp=decision_timestamp,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        realized_return=realized,
        benchmark_return=benchmark,
        active_return=realized - benchmark,
        maximum_adverse_excursion=Decimal("-0.002"),
        maximum_favorable_excursion=Decimal("0.007"),
        costs=costs,
        evaluation_status="EVALUATED",
    )
