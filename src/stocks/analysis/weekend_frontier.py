from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from stocks.analysis.theme_fundamentals import collect_theme_fundamentals
from stocks.analysis.theme_contracts import collect_theme_contracts
from stocks.analysis.theme_events import collect_theme_event_risk
from stocks.analysis.theme_news import collect_theme_news
from stocks.analysis.theme_provisional import build_theme_provisional_assessment
from stocks.analysis.theme_session_plan import build_theme_opening_session_plan
from stocks.analysis.theme_shariah import collect_theme_shariah_coverage
from stocks.analysis.themes import build_frontier_theme_analysis
from stocks.data.multitimeframe import collect_multitimeframe_data
from stocks.execution.idempotency import stable_hash
from stocks.research.evidence_throughput import publish_evidence_throughput


CONFIG_PATH = Path("config/themes/frontier_technology_energy_v1.json")
OUTPUT_PATH = Path("output/analysis/themes/weekend-run.json")
PRIVATE_ROOT = Path("data/research/themes/private")
LOCK_PATH = PRIVATE_ROOT / "weekend-research.lock"
HISTORY_PATH = PRIVATE_ROOT / "weekend-runs.jsonl"
MAX_LOCK_AGE = timedelta(hours=3)

Collector = Callable[[], dict[str, Any]]


def run_frontier_weekend_research(
    project_root: Path,
    *,
    now: datetime | None = None,
    force: bool = False,
    refresh_bars: bool = True,
    collectors: dict[str, Collector] | None = None,
) -> dict[str, Any]:
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    if observed_at.weekday() < 5 and not force:
        report = _base_report(
            observed_at,
            status="SKIPPED_NOT_WEEKEND",
            steps={},
        )
        _atomic_json(project_root / OUTPUT_PATH, report)
        return report

    lock = project_root / LOCK_PATH
    if not _acquire_lock(lock, observed_at):
        return _base_report(
            observed_at,
            status="BLOCKED_SINGLE_FLIGHT",
            steps={},
        )
    try:
        config = _read_json(project_root / CONFIG_PATH)
        symbols = sorted(
            {
                str(row.get("symbol") or "").upper()
                for definition in (config.get("themes") or {}).values()
                for row in definition.get("instruments", [])
                if row.get("symbol")
            }
        )
        intervals = list(config.get("required_timeframes") or [])
        if not symbols or not intervals:
            report = _base_report(
                observed_at,
                status="BLOCKED_CONFIG_UNAVAILABLE",
                steps={},
            )
            _publish(project_root, report)
            return report

        supplied = collectors or {}
        steps: dict[str, dict[str, Any]] = {}
        if refresh_bars:
            steps["bars"] = _run_step(
                supplied.get("bars")
                or (
                    lambda: collect_multitimeframe_data(
                        project_root,
                        symbols=symbols,
                        intervals=intervals,
                        providers=["yfinance"],
                        lookback_days=730,
                    )
                )
            )
        else:
            steps["bars"] = {"status": "SKIPPED_BY_OPERATOR"}

        fundamentals_path = (
            project_root
            / "output"
            / "analysis"
            / "themes"
            / "fundamental-coverage.json"
        )
        if _needs_refresh(
            fundamentals_path,
            observed_at,
            hours=18,
            expected_config_hash=stable_hash(config),
        ):
            steps["fundamentals"] = _run_step(
                supplied.get("fundamentals")
                or (
                    lambda: collect_theme_fundamentals(
                        project_root, now=observed_at
                    )
                )
            )
        else:
            steps["fundamentals"] = {
                "status": "REUSED_FRESH_ARTIFACT",
                "artifact": str(fundamentals_path),
            }
        steps["contracts"] = _run_step(
            supplied.get("contracts")
            or (
                lambda: collect_theme_contracts(
                    project_root,
                    now=observed_at,
                )
            )
        )
        steps["shariah"] = _run_step(
            supplied.get("shariah")
            or (
                lambda: collect_theme_shariah_coverage(
                    project_root,
                    now=observed_at,
                )
            )
        )
        steps["news"] = _run_step(
            supplied.get("news")
            or (lambda: collect_theme_news(project_root, now=observed_at))
        )
        steps["event_risk"] = _run_step(
            supplied.get("event_risk")
            or (
                lambda: collect_theme_event_risk(
                    project_root,
                    now=observed_at,
                )
            )
        )
        steps["themes"] = _run_step(
            supplied.get("themes")
            or (
                lambda: build_frontier_theme_analysis(
                    project_root,
                    as_of=observed_at,
                )
            )
        )
        steps["session_plan"] = _run_step(
            supplied.get("session_plan")
            or (
                lambda: build_theme_opening_session_plan(
                    project_root,
                    now=observed_at,
                )
            )
        )
        steps["provisional_assessment"] = _run_step(
            supplied.get("provisional_assessment")
            or (
                lambda: build_theme_provisional_assessment(
                    project_root,
                    now=observed_at,
                )
            )
        )
        steps["evidence"] = _run_step(
            supplied.get("evidence")
            or (lambda: publish_evidence_throughput(project_root))
        )

        hard_steps = (
            "bars",
            "fundamentals",
            "contracts",
            "shariah",
            "news",
            "event_risk",
            "themes",
            "session_plan",
            "provisional_assessment",
        )
        hard_failures = [
            name
            for name in hard_steps
            if str(steps[name].get("status"))
            in {"BLOCKED", "ERROR", "PROVIDER_UNAVAILABLE"}
            or str(steps[name].get("status", "")).startswith("BLOCKED_")
        ]
        status = "GO" if not hard_failures else "DEGRADED"
        report = _base_report(observed_at, status=status, steps=steps)
        report.update(
            {
                "symbol_count": len(symbols),
                "symbols": symbols,
                "intervals": intervals,
                "hard_failures": hard_failures,
                "provider_calls_read_only": sum(
                    int(step.get("provider_calls", 0))
                    + int(step.get("provider_calls_read_only", 0))
                    for step in steps.values()
                ),
            }
        )
        report["content_hash"] = stable_hash(report)
        _publish(project_root, report)
        return report
    finally:
        lock.unlink(missing_ok=True)


def _run_step(collector: Collector) -> dict[str, Any]:
    try:
        result = collector()
    except Exception as exc:
        return {
            "status": "ERROR",
            "error_class": type(exc).__name__,
        }
    return result if isinstance(result, dict) else {"status": "ERROR"}


def _base_report(
    observed_at: datetime,
    *,
    status: str,
    steps: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "frontier_weekend_research_run_v1",
        "status": status,
        "generated_at": observed_at.isoformat(),
        "market_state": (
            "WEEKEND_CLOSED"
            if observed_at.weekday() >= 5
            else "WEEKDAY_NOT_SCHEDULED"
        ),
        "steps": steps,
        "standalone_entry_allowed": False,
        "automatic_execution": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
        "financial_calls": {
            "place_order": 0,
            "cancel_order": 0,
            "global_cancel": 0,
        },
    }


def _acquire_lock(path: Path, now: datetime) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if now - modified <= MAX_LOCK_AGE:
            return False
        path.unlink(missing_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at": now.isoformat(),
                },
                sort_keys=True,
            )
        )
    return True


def _needs_refresh(
    path: Path,
    now: datetime,
    *,
    hours: float,
    expected_config_hash: str | None = None,
) -> bool:
    payload = _read_json(path)
    if (
        expected_config_hash is not None
        and payload.get("config_hash") != expected_config_hash
    ):
        return True
    value = payload.get("generated_at")
    if not value:
        return True
    try:
        generated = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return True
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)
    return now - generated.astimezone(UTC) > timedelta(hours=hours)


def _publish(project_root: Path, report: dict[str, Any]) -> None:
    _atomic_json(project_root / OUTPUT_PATH, report)
    history = project_root / HISTORY_PATH
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, default=str) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
