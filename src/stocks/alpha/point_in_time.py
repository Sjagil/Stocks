from __future__ import annotations

from datetime import datetime

from stocks.alpha.data_contracts import PITDataStatus, PointInTimeFact


def validate_fact_availability(fact: PointInTimeFact, decision_timestamp: datetime) -> PITDataStatus:
    if fact.available_at is None:
        return PITDataStatus.MISSING_AVAILABLE_AT
    if fact.available_at > decision_timestamp:
        return PITDataStatus.FUTURE_DATED_BLOCKED
    if fact.revised_at is not None and fact.revised_at <= decision_timestamp and fact.revised_at > fact.available_at:
        return PITDataStatus.REVISED_AFTER_DECISION_BLOCKED
    return PITDataStatus.VALID


def aggregate_pit_status(facts: list[PointInTimeFact], decision_timestamp: datetime) -> PITDataStatus:
    if not facts:
        return PITDataStatus.MISSING_AVAILABLE_AT
    statuses = [validate_fact_availability(fact, decision_timestamp) for fact in facts]
    for blocked in (
        PITDataStatus.MISSING_AVAILABLE_AT,
        PITDataStatus.FUTURE_DATED_BLOCKED,
        PITDataStatus.REVISED_AFTER_DECISION_BLOCKED,
    ):
        if blocked in statuses:
            return blocked
    return PITDataStatus.VALID
