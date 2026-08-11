from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from stocks.screener.config import ScreenerConfig
from stocks.macro.service import macro_context_at
from stocks.screener.scoring import score_asset
from stocks.screener.sources import (
    LocalScreenerSources,
    decision_time_for_session,
    latest_completed_session,
)
from stocks.screener.storage import (
    CANDIDATE_CLASSES,
    ScreenerLayout,
    ScreenerStore,
    write_public_artifacts,
)

UTC = timezone.utc


def screener_run(project_root: Path, *, as_of: str | None = None) -> dict[str, Any]:
    config = ScreenerConfig.load(project_root)
    screening_date = date.fromisoformat(as_of) if as_of else latest_completed_session()
    layout = ScreenerLayout.from_project_root(project_root)
    store = ScreenerStore(layout)
    try:
        existing = store.by_date(screening_date)
        if existing is not None:
            return {
                "schema": "daily_asset_screener_command_v1",
                "status": "ALREADY_REGISTERED",
                "screening_date": screening_date.isoformat(),
                "run_id": existing["run_id"],
                "content_hash": existing["content_hash"],
                "summary": existing["summary"],
                **_authority(),
            }
        with LocalScreenerSources(project_root, config) as sources:
            snapshots = sources.load(screening_date)
            macro_context = macro_context_at(
                project_root,
                as_of=decision_time_for_session(screening_date),
            )
            scored = [
                score_asset(
                    replace(snapshot),
                    screening_date=screening_date,
                    config=config,
                    macro_context=macro_context,
                )
                for snapshot in snapshots
            ]
            records = [item.public for item in scored]
            private = {
                str(item.public["asset_key"]): item.private
                for item in scored
            }
            summary = _summary(records, sources.source_inventory, config)
        registered = store.register(
            screening_date=screening_date,
            decision_time=decision_time_for_session(screening_date).isoformat(),
            config_hash=config.config_hash,
            screener_version=config.screener_version,
            records=records,
            private_records=private,
            summary_base=summary,
        )
        run = store.by_date(screening_date)
        if run is None:
            raise RuntimeError("registered screener run could not be reloaded")
        artifacts = write_public_artifacts(layout, run, store.all_public_records())
        return {
            "schema": "daily_asset_screener_command_v1",
            "status": "GO",
            "run_id": registered["run_id"],
            "screening_date": screening_date.isoformat(),
            "summary": registered["summary"],
            "artifacts": artifacts,
            "private_registry": str(layout.private_db),
            **_authority(),
        }
    finally:
        store.close()


def screener_preview(
    project_root: Path,
    *,
    as_of: str | None = None,
    known_at: datetime | None = None,
) -> dict[str, Any]:
    """Re-evaluate the latest completed session without rewriting frozen history."""
    config = ScreenerConfig.load(project_root)
    screening_date = date.fromisoformat(as_of) if as_of else latest_completed_session()
    knowledge_cutoff = (known_at or datetime.now(UTC)).astimezone(UTC)
    with LocalScreenerSources(project_root, config) as sources:
        snapshots = sources.load(screening_date, known_at=knowledge_cutoff)
        macro_context = macro_context_at(
            project_root,
            as_of=knowledge_cutoff,
        )
        scored = [
            score_asset(
                replace(snapshot),
                screening_date=screening_date,
                config=config,
                macro_context=macro_context,
                known_at=knowledge_cutoff,
            )
            for snapshot in snapshots
        ]
        records = [item.public for item in scored]
        summary = _summary(records, sources.source_inventory, config)
    payload = {
        "schema": "daily_asset_screener_runtime_preview_v1",
        "status": "GO",
        "generated_at": datetime.now(UTC).isoformat(),
        "screening_date": screening_date.isoformat(),
        "decision_time": knowledge_cutoff.isoformat(),
        "knowledge_cutoff": knowledge_cutoff.isoformat(),
        "market_data_cutoff": screening_date.isoformat(),
        "config_hash": config.config_hash,
        "screener_version": config.screener_version,
        "canonical_research_evidence": False,
        "append_only_history_mutated": False,
        "records": records,
        "summary": summary,
        **_authority(),
    }
    output = project_root / "output" / "screener" / "latest-preview.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(output)
    return {
        "schema": "daily_asset_screener_preview_command_v1",
        "status": "GO",
        "screening_date": screening_date.isoformat(),
        "knowledge_cutoff": knowledge_cutoff.isoformat(),
        "market_data_cutoff": screening_date.isoformat(),
        "record_count": len(records),
        "candidate_count": summary["candidate_count"],
        "data_quality_status": summary["data_quality_status"],
        "artifact": str(output),
        "canonical_research_evidence": False,
        "append_only_history_mutated": False,
        **_authority(),
    }


def screener_status(project_root: Path) -> dict[str, Any]:
    config = ScreenerConfig.load(project_root)
    layout = ScreenerLayout.from_project_root(project_root)
    store = ScreenerStore(layout)
    try:
        latest = store.latest()
        return {
            "schema": "daily_asset_screener_status_v1",
            "status": "GO" if latest is not None else "NOT_RUN",
            "config": config.public_dict(),
            "run_count": store.run_count(),
            "observation_count": store.observation_count(),
            "latest_config_matches_current": (
                latest is not None and latest["config_hash"] == config.config_hash
            ),
            "latest_run": (
                None
                if latest is None
                else {
                    "run_id": latest["run_id"],
                    "screening_date": latest["screening_date"],
                    "content_hash": latest["content_hash"],
                    "summary": latest["summary"],
                }
            ),
            "private_registry": str(layout.private_db),
            "output_dir": str(layout.output_dir),
            **_authority(),
        }
    finally:
        store.close()


def screener_report(project_root: Path, *, as_of: str | None = None) -> dict[str, Any]:
    layout = ScreenerLayout.from_project_root(project_root)
    store = ScreenerStore(layout)
    try:
        run = store.by_date(date.fromisoformat(as_of)) if as_of else store.latest()
        if run is None:
            return {"schema": "daily_asset_screener_report_v1", "status": "NOT_RUN", **_authority()}
        candidates = [
            item
            for item in run["records"]
            if item["classification"] in CANDIDATE_CLASSES
        ]
        return {
            "schema": "daily_asset_screener_report_v1",
            "status": "GO",
            "screening_date": run["screening_date"],
            "summary": run["summary"],
            "top_candidates": candidates[:20],
            "changes": run["summary"].get("changes", {}),
            "daily_report_path": str(
                layout.output_dir / str(run["screening_date"]) / "daily-report.md"
            ),
            **_authority(),
        }
    finally:
        store.close()


def screener_history(project_root: Path, *, symbol: str) -> dict[str, Any]:
    layout = ScreenerLayout.from_project_root(project_root)
    store = ScreenerStore(layout)
    try:
        records = store.history(symbol.upper())
        return {
            "schema": "daily_asset_screener_history_v1",
            "status": "GO" if records else "NOT_FOUND",
            "symbol": symbol.upper(),
            "observation_count": len(records),
            "history": records,
            **_authority(),
        }
    finally:
        store.close()


def screener_top(project_root: Path, *, limit: int) -> dict[str, Any]:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be in [1, 100]")
    layout = ScreenerLayout.from_project_root(project_root)
    store = ScreenerStore(layout)
    try:
        run = store.latest()
        if run is None:
            return {"schema": "daily_asset_screener_top_v1", "status": "NOT_RUN", **_authority()}
        candidates = [
            item
            for item in run["records"]
            if item["classification"] in CANDIDATE_CLASSES
        ][:limit]
        return {
            "schema": "daily_asset_screener_top_v1",
            "status": "GO",
            "screening_date": run["screening_date"],
            "limit": limit,
            "candidate_count": len(candidates),
            "candidates": candidates,
            **_authority(),
        }
    finally:
        store.close()


def screener_asset(project_root: Path, *, symbol: str) -> dict[str, Any]:
    history = screener_history(project_root, symbol=symbol)
    return {
        **history,
        "symbol": symbol.upper(),
        "screener_results_are_orders": False,
        "execution_authority": "NONE",
    }


def screener_export(project_root: Path) -> dict[str, Any]:
    report = screener_report(project_root)
    output = project_root / "output" / "screener" / "latest-export.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return {
        "status": report.get("status", "BLOCKED"),
        "path": str(output),
        "result_count": len(report.get("results", [])),
        "screener_results_are_orders": False,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
    }


def _summary(
    records: list[dict[str, Any]],
    source_inventory: dict[str, Any],
    config: ScreenerConfig,
) -> dict[str, Any]:
    classifications = Counter(str(item["classification"]) for item in records)
    reasons = Counter(
        str(reason)
        for item in records
        for reason in item.get("rejection_reasons", [])
    )
    data_reasons = {
        "MISSING_PRICE_DATA",
        "STALE_PRICE_DATA",
        "INSUFFICIENT_PRICE_HISTORY",
        "INVALID_PRICE",
        "INVALID_OR_INCOMPLETE_CANDLE",
        "PROVIDER_PRICE_CONFLICT",
        "MISSING_FUNDAMENTAL_DATA",
        "MISSING_CRITICAL_FUNDAMENTAL_DATA",
        "INSUFFICIENT_FUNDAMENTAL_COVERAGE",
        "STALE_FUNDAMENTAL_DATA",
        "ETF_HOLDINGS_FUNDAMENTALS_UNAVAILABLE",
        "MISSING_BENCHMARK_DATA",
    }
    data_exclusions = sum(
        1
        for item in records
        if any(reason in data_reasons for reason in item.get("rejection_reasons", []))
    )
    movers = [
        item
        for item in records
        if item.get("daily_return") is not None
        and item.get("data_timestamps", {}).get("price_session")
        == item.get("screening_date")
    ]
    winners = sorted(movers, key=lambda item: float(item["daily_return"]), reverse=True)[:20]
    losers = sorted(movers, key=lambda item: float(item["daily_return"]))[:20]
    data_quality_status = (
        "NO_DATA"
        if not records
        else "DEGRADED"
        if data_exclusions
        else "GO"
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "screened_count": len(records),
        "classification_counts": dict(sorted(classifications.items())),
        "candidate_count": sum(
            classifications.get(item, 0) for item in CANDIDATE_CLASSES
        ),
        "data_exclusion_count": data_exclusions,
        "rejection_reason_counts": dict(sorted(reasons.items())),
        "top_winners": [_compact_mover(item) for item in winners],
        "top_losers": [_compact_mover(item) for item in losers],
        "benchmarks": sorted(set(config.benchmarks.values())),
        "data_quality_status": data_quality_status,
        "source_inventory": source_inventory,
        "score_formula": {
            "total": (
                "fundamental*w_f + technical*w_t + liquidity*w_l + "
                "risk*w_r + available_macro*w_m"
            ),
            "weights": config.weights,
            "macro_missing_policy": (
                "REMOVE_MACRO_WEIGHT_AND_RENORMALIZE_OTHER_COMPONENTS"
            ),
            "hidden_optimization": False,
        },
        "execution_authority": "NONE",
        "strategy_authority": "NONE",
        "broker_calls": 0,
        "order_calls": 0,
    }


def _compact_mover(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": item["symbol"],
        "daily_return": item["daily_return"],
        "classification": item["classification"],
        "total_score": item["total_score"],
        "rejection_reasons": item["rejection_reasons"],
    }


def _authority() -> dict[str, Any]:
    return {
        "research_signal_only": True,
        "execution_authority": "NONE",
        "strategy_authority": "NONE",
        "broker_calls": 0,
        "broker_writes": 0,
        "order_calls": 0,
    }
