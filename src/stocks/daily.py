from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from stocks.research.autopilot.runtime import run_if_due
from stocks.research.promotion import recover_survivors
from stocks.screener.service import screener_run
from stocks.signals.service import signal_scan, signal_status
from stocks.notifications.telegram import telegram_daily_delivery
from stocks.portfolio.manager import build_active_portfolio_report


def run_daily(
    project_root: Path,
    *,
    signals_only: bool = False,
    research_only: bool = False,
    no_autopilot: bool = False,
    no_telegram: bool = False,
) -> dict[str, Any]:
    started = datetime.now(UTC)
    screener: dict[str, Any] = {"status": "SKIPPED"}
    signals: dict[str, Any] = {"status": "SKIPPED"}
    research: dict[str, Any] = {"status": "SKIPPED"}
    if not research_only:
        try:
            screener = screener_run(project_root)
        except Exception as exc:
            screener = {
                "status": "DEGRADED",
                "reason": type(exc).__name__,
                "detail": str(exc)[:300],
            }
        signals = signal_scan(project_root)
    if not signals_only:
        recovery = recover_survivors(project_root)
        research = (
            {"status": "SKIPPED_BY_OPERATOR", "survivor_recovery": recovery}
            if no_autopilot
            else {
                **run_if_due(project_root),
                "survivor_recovery": {
                    "status": recovery["status"],
                    "survivor_count": recovery["survivor_count"],
                },
            }
        )
    if research_only:
        portfolio: dict[str, Any] = {"status": "SKIPPED_RESEARCH_ONLY"}
    else:
        try:
            portfolio = build_active_portfolio_report(project_root)
        except Exception as exc:
            portfolio = {
                "status": "DEGRADED",
                "reason": type(exc).__name__,
                "detail": str(exc)[:300],
                "execution_authority": "NONE",
                "orders_generated": 0,
                "broker_write_calls": 0,
            }
    telegram: dict[str, Any] = (
        {
            "status": "SKIPPED_BY_OPERATOR",
            "failure_is_non_blocking": True,
            "execution_authority": "NONE",
            "broker_calls": 0,
            "orders_generated": 0,
        }
        if no_telegram or research_only
        else telegram_daily_delivery(project_root, research=research)
    )
    completed = datetime.now(UTC)
    return {
        "schema": "canonical_daily_workflow_v1",
        "status": "GO",
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_seconds": (completed - started).total_seconds(),
        "modes": {
            "signals_only": signals_only,
            "research_only": research_only,
            "no_autopilot": no_autopilot,
            "no_telegram": no_telegram,
        },
        "system_health": "LOCAL_WORKFLOW_GO",
        "screener": screener,
        "signals": signals,
        "signal_status": signal_status(project_root),
        "research": research,
        "portfolio": portfolio,
        "telegram": telegram,
        "automatic_orders_allowed": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }
