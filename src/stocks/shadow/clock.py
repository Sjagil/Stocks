from __future__ import annotations

from datetime import datetime

from stocks.shadow.errors import (
    DECISION_CLOCK_INVALID,
    FUTURE_DATA_BLOCKED,
    MISSING_DATASET_HASH,
    SAME_CLOSE_EXECUTION_BLOCKED,
    STALE_DATASET,
)


DECISION_FREQUENCIES = ("DAILY_AFTER_CLOSE", "WEEKLY_AFTER_CLOSE", "MONTHLY_AFTER_CLOSE", "FIXTURE_MANUAL")


def validate_decision_clock(
    *,
    frequency: str,
    information_cutoff_timestamp: str,
    decision_timestamp: str,
    first_executable_timestamp: str,
    dataset_content_hashes: dict[str, str],
    feature_available_at: str | None = None,
    stale_dataset: bool = False,
) -> dict[str, str]:
    if frequency not in DECISION_FREQUENCIES:
        return {"status": "NO_GO", "decision_code": DECISION_CLOCK_INVALID}
    if not dataset_content_hashes or any(not value for value in dataset_content_hashes.values()):
        return {"status": "NO_GO", "decision_code": MISSING_DATASET_HASH}
    cutoff = _parse(information_cutoff_timestamp)
    decision = _parse(decision_timestamp)
    executable = _parse(first_executable_timestamp)
    if not cutoff < decision:
        return {"status": "NO_GO", "decision_code": FUTURE_DATA_BLOCKED}
    if not decision <= executable:
        return {"status": "NO_GO", "decision_code": SAME_CLOSE_EXECUTION_BLOCKED}
    if feature_available_at is not None and _parse(feature_available_at) > cutoff:
        return {"status": "NO_GO", "decision_code": FUTURE_DATA_BLOCKED}
    if stale_dataset:
        return {"status": "NO_GO", "decision_code": STALE_DATASET}
    return {"status": "GO", "decision_code": "DECISION_CLOCK_VALID"}


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
