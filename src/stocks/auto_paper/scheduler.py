from __future__ import annotations

from datetime import datetime, timedelta


PHASES = ("PRE_OPEN_REFRESH", "REGULAR_SESSION_SHADOW", "AFTER_CLOSE_RECONCILIATION")
TASKS_BY_PHASE = {
    "PRE_OPEN_REFRESH": (
        "REFRESH_UNIVERSE",
        "REFRESH_SHARIAH_STATE",
        "REFRESH_FUNDAMENTALS",
        "REFRESH_NEWS",
        "CALCULATE_MOVERS",
    ),
    "REGULAR_SESSION_SHADOW": (
        "INGEST_SIGNALS",
        "VALIDATE_FRESHNESS",
        "DEDUPLICATE_SIGNALS",
        "RUN_PORTFOLIO_RISK",
        "PRODUCE_SHADOW_INTENTS",
        "MANAGE_RISK_REDUCING_EXITS",
    ),
    "AFTER_CLOSE_RECONCILIATION": (
        "RECONCILE",
        "CALCULATE_DAILY_PNL",
        "PERSIST_EPISODES",
        "CREATE_DAILY_AUDIT_REPORT",
    ),
}


def run_bounded_scheduler(
    *,
    start_time: str,
    interval_seconds: int,
    max_iterations: int,
) -> dict[str, object]:
    interval = max(60, interval_seconds)
    iterations = max(0, min(max_iterations, 1_440))
    current = datetime.fromisoformat(start_time)
    records = []
    for index in range(iterations):
        phase = PHASES[index % len(PHASES)]
        records.append(
            {
                "iteration": index + 1,
                "scheduled_at": current.isoformat(),
                "phase": phase,
                "tasks": list(TASKS_BY_PHASE[phase]),
                "authority": "SHADOW_ONLY",
            }
        )
        current += timedelta(seconds=interval)
    return {
        "status": "SCHEDULER_BOUNDED_GO",
        "interval_seconds": interval,
        "iterations_requested": max_iterations,
        "iterations_completed": len(records),
        "bounded": True,
        "busy_loop": False,
        "records": records,
        "automatic_submission": False,
    }
