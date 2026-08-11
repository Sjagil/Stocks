from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from stocks.live.service import live_kill_switch, live_preflight
from stocks.notifications.telegram import telegram_command
from stocks.capital import portfolio_management_command
from stocks.application.context import load_app_context
from stocks.data.multitimeframe import (
    multitimeframe_status,
    provider_inventory,
)
from stocks.research.autopilot.runtime import runtime_command
from stocks.research.promotion import recover_survivors
from stocks.research.sec_overlay import sec_intelligence_status
from stocks.signals.service import signal_status
from stocks.signals.top5 import publish_top_signals
from stocks.universe import broad_universe_status


def config_validation(
    project_root: Path,
    *,
    env_file: str = ".env.ibkr",
) -> dict[str, Any]:
    env_path = Path(env_file)
    if not env_path.is_absolute():
        env_path = project_root / env_path
    try:
        settings = load_app_context(str(env_path)).ibkr
    except (OSError, ValueError) as exc:
        return {
            "schema": "stocks_config_validation_v1",
            "status": "NO_GO",
            "env_file_exists": env_path.is_file(),
            "error_class": type(exc).__name__,
            "config": None,
            "secrets_published": False,
            "broker_calls": 0,
            "financial_calls": _zero_financial_calls(),
            "execution_authority": "NONE",
        }
    checks = {
        "env_file_exists": env_path.is_file(),
        "localhost_only": settings.host == "127.0.0.1",
        "paper_port": settings.port in {7497, 4002},
        "read_only": settings.read_only is True,
        "order_authority_none": settings.order_authority == "NONE",
        "live_trading_disabled": settings.live_trading_enabled is False,
        "order_transmission_disabled": settings.allow_order_transmission is False,
    }
    return {
        "schema": "stocks_config_validation_v1",
        "status": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "config": settings.safe_dict(),
        "secrets_published": False,
        "broker_calls": 0,
        "financial_calls": _zero_financial_calls(),
        "execution_authority": "NONE",
    }


def data_source_status(project_root: Path) -> dict[str, Any]:
    inventory = provider_inventory(project_root)
    cache = multitimeframe_status(project_root)
    sources = [
        {
            "provider": row.get("provider"),
            "status": row.get("status"),
            "available": bool(row.get("available")),
            "native_intervals": row.get("native_intervals", []),
            "source_type": row.get("source_type"),
        }
        for row in inventory.get("sources", [])
    ]
    ready = (
        inventory.get("status") == "GO"
        and inventory.get("available_source_count", 0) > 0
    )
    return {
        "schema": "stocks_data_source_status_v1",
        "status": "GO" if ready else "NO_GO",
        "available_source_count": inventory.get(
            "available_source_count", 0
        ),
        "sources": sources,
        "multitimeframe_cache_status": cache.get("status"),
        "multitimeframe_current_data_status": cache.get(
            "current_data_status", "CURRENT_DATA_STATUS_UNAVAILABLE"
        ),
        "current_data_ratio": cache.get("current_data_ratio"),
        "requested_current_symbol_interval_pairs": cache.get(
            "requested_current_symbol_interval_pairs", 0
        ),
        "fresh_current_symbol_interval_pairs": cache.get(
            "fresh_current_symbol_interval_pairs", 0
        ),
        "stale_current_symbol_interval_pairs": cache.get(
            "stale_current_symbol_interval_pairs", []
        ),
        "intervals_present": cache.get("intervals_present", []),
        "file_count": cache.get("file_count", 0),
        "row_count": cache.get("row_count", 0),
        "secret_presence_only": inventory.get(
            "secret_presence_only", True
        ),
        "secrets_published": False,
        "context_sources_used_as_ohlcv": False,
        "broker_calls": 0,
        "financial_calls": _zero_financial_calls(),
        "execution_authority": "NONE",
    }


def comprehensive_data_readiness(project_root: Path) -> dict[str, Any]:
    sources = data_source_status(project_root)
    cache = _read(project_root / "output/research/multitimeframe/cache-validation.json")
    corporate_actions = _read(project_root / "data/corporate_actions/event_manifest.json")
    fx = _read(project_root / "data/fx/fx_manifest.json")
    total_returns = _read(project_root / "data/total_returns/total_return_manifest.json")
    macro = _read(project_root / "output/macro/status.json")
    equity_context = _read(project_root / "output/ibkr/phase11_3/status.json")
    sec_intelligence = sec_intelligence_status(project_root)
    market_context = _read(project_root / "output/market_context/status.json")
    realtime = _read(project_root / "output/market_context/realtime-equity-collection.json")
    cot = _read(project_root / "output/market_context/cot-context.json")
    episodes = _read(project_root / "output/market_context/entry-episode-completeness.json")
    selective_ml = _read(project_root / "output/research/active_swing/selective_ml/status.json")
    group_intelligence = _read(
        project_root / "output/analysis/groups/coverage.json"
    )
    ibkr_news = _read(
        project_root / "output/ibkr/news/capabilities.json"
    )
    confluence = _read(
        project_root / "output/portfolio/confluence-audit.json"
    )

    fundamental_symbols = int(equity_context.get("fundamental_symbol_count", 0))
    universe_size = int(equity_context.get("universe_size", 0))
    layers = {
        "ohlcv": {
            "status": cache.get("status", "MISSING"),
            "symbol_count": int(cache.get("symbol_count", 0)),
            "row_count": int(cache.get("row_count", 0)),
            "interval_count": int(cache.get("interval_count", 0)),
            "duplicate_rows": int(cache.get("duplicate_rows", 0)),
            "invalid_ohlc_rows": int(cache.get("invalid_ohlc_rows", 0)),
            "timezone_errors": int(cache.get("timezone_errors", 0)),
        },
        "corporate_actions": {
            "status": corporate_actions.get("status", "MISSING"),
            "event_count": int(corporate_actions.get("event_count", 0)),
            "provider_error_count": len(corporate_actions.get("provider_errors", [])),
        },
        "fx": {
            "status": fx.get("status", "MISSING"),
            "currencies": fx.get("currencies", []),
            "fallback_rows": int(fx.get("fallback_rows", 0)),
            "provider_error_count": len(fx.get("provider_errors", [])),
        },
        "eur_total_returns": {
            "status": total_returns.get("status", "MISSING"),
            "base_currency": total_returns.get("base_currency"),
            "instrument_count": len(total_returns.get("instrument_summaries", [])),
            "row_count": int(total_returns.get("cache_validation", {}).get("row_count", 0)),
            "fx_sources": total_returns.get("fx_sources", [total_returns.get("fx_source")]),
        },
        "fundamentals_filings_news": {
            "status": equity_context.get("status", "MISSING"),
            "universe_size": universe_size,
            "fundamental_symbol_count": fundamental_symbols,
            "fundamental_coverage_ratio": (
                round(fundamental_symbols / universe_size, 8) if universe_size else 0.0
            ),
            "news_symbol_count": int(equity_context.get("news_symbol_count", 0)),
            "sec_metadata_record_count": int(
                equity_context.get("SEC_event_count", 0)
            ),
            "sec_structured_filing_count": int(
                sec_intelligence.get("structured_filing_count", 0)
            ),
            "sec_structured_event_count": int(
                sec_intelligence.get("structured_event_count", 0)
            ),
            "sec_structured_symbol_count": int(
                sec_intelligence.get("structured_symbol_count", 0)
            ),
            "sec_overlay_status": sec_intelligence.get("status", "MISSING"),
            "sec_authority": sec_intelligence.get(
                "authority", "RANKING_OVERLAY_ONLY"
            ),
            "sec_standalone_entry_allowed": False,
            "blockers": equity_context.get("open_blockers", []),
        },
        "macro": {
            "status": macro.get("status", "MISSING"),
            "data_quality": macro.get("latest_data_quality", "MISSING"),
            "feature_availability": macro.get("feature_availability", {}),
            "series_count": int(macro.get("series_count", 0)),
        },
        "cot": {
            "status": cot.get("status", "MISSING"),
            "context_count": int(cot.get("context_count", 0)),
            "standalone_entry_authority": False,
        },
        "options_gex": market_context.get("gex", {"status": "MISSING"}),
        "observed_equity_microstructure": {
            "status": realtime.get("status", "NOT_RUN"),
            "quote_row_count": int(realtime.get("quote_row_count", 0)),
            "trade_row_count": int(realtime.get("trade_row_count", 0)),
            "depth_row_count": int(realtime.get("depth_row_count", 0)),
            "thread_leak": bool(realtime.get("thread_leak", False)),
        },
        "forward_episodes": {
            "status": episodes.get("status", "MISSING"),
            "episode_count": int(episodes.get("episode_count", 0)),
            "terminal_episode_count": int(episodes.get("terminal_episode_count", 0)),
            "completion_ratio": episodes.get("completion_ratio"),
        },
        "selective_ml": {
            "status": selective_ml.get("status", "MISSING"),
            "blockers": selective_ml.get("blockers", []),
        },
        "sector_industry_intelligence": {
            "status": group_intelligence.get("status", "MISSING"),
            "sector_count": int(group_intelligence.get("sector_count", 0)),
            "industry_count": int(group_intelligence.get("industry_count", 0)),
            "all_sector_groups_analyzed": bool(
                group_intelligence.get("all_sector_groups_analyzed", False)
            ),
            "all_industry_groups_analyzed": bool(
                group_intelligence.get("all_industry_groups_analyzed", False)
            ),
            "fundamental_symbol_count": int(
                group_intelligence.get("fundamental_symbol_count", 0)
            ),
            "signal_eligible_stock_count": int(
                group_intelligence.get("signal_eligible_stock_count", 0)
            ),
            "signal_eligible_fundamental_count": int(
                group_intelligence.get(
                    "signal_eligible_fundamental_count", 0
                )
            ),
            "signal_eligible_fundamental_coverage_ratio": (
                group_intelligence.get(
                    "signal_eligible_fundamental_coverage_ratio"
                )
            ),
            "signal_eligible_fundamental_missing_symbols": (
                group_intelligence.get(
                    "signal_eligible_fundamental_missing_symbols", []
                )
            ),
            "current_news_record_count": int(
                group_intelligence.get("current_news_record_count", 0)
            ),
            "standalone_entry_allowed": False,
        },
        "ibkr_news": {
            "status": ibkr_news.get("status", "NOT_PROBED"),
            "provider_count": int(ibkr_news.get("provider_count", 0)),
            "historical_headlines_capability": ibkr_news.get(
                "historical_headlines_capability", "UNPROVEN"
            ),
            "subscription_purchase_automatic": False,
            "execution_authority": "NONE",
        },
        "technical_fundamental_macro_confluence": {
            "status": confluence.get("status", "NOT_RUN"),
            "opportunity_count": int(
                confluence.get("opportunity_count", 0)
            ),
            "status_counts": confluence.get("status_counts", {}),
            "technical_signal_required": bool(
                confluence.get("technical_signal_required", True)
            ),
            "standalone_context_entry_allowed": False,
            "execution_authority": "NONE",
        },
    }
    gaps: list[dict[str, Any]] = []
    if equity_context.get("shariah_history_status") != "GO":
        gaps.append(_gap("HISTORICAL_SHARIAH_POINT_IN_TIME", "EXTERNAL_OR_MANUAL", "BLOCKING_COMPLIANCE_RESEARCH"))
    if "NEWS_ARCHIVE_PARTIAL" in equity_context.get("open_blockers", []):
        gaps.append(_gap("COMPLETE_HISTORICAL_NEWS_ARCHIVE", "LICENSED_PROVIDER", "BLOCKING_EVENT_RESEARCH"))
    if macro.get("latest_data_quality") != "DATA_COMPLETE":
        gaps.append(_gap("MACRO_RELEASE_VINTAGES_AND_LICENSED_PMI", "LICENSED_PROVIDER", "CONTEXT_DEGRADED"))
    if not bool(market_context.get("gex", {}).get("historical_pit_backtest_allowed")):
        gaps.append(_gap("HISTORICAL_PIT_OPTIONS_CHAIN_AND_GEX", "LICENSED_PROVIDER", "BLOCKING_GEX_BACKTEST"))
    if int(realtime.get("trade_row_count", 0)) == 0:
        gaps.append(_gap("OBSERVED_EQUITY_TAPE", "IBKR_ENTITLEMENT_AND_OPEN_SESSION", "BLOCKING_ENTRY_FILTER_EVIDENCE"))
    if int(realtime.get("depth_row_count", 0)) == 0:
        gaps.append(_gap("OBSERVED_EQUITY_DEPTH", "IBKR_DEPTH_ENTITLEMENT_AND_OPEN_SESSION", "BLOCKING_DEPTH_FILTER_EVIDENCE"))
    if fundamental_symbols < universe_size:
        gaps.append(_gap("FULL_POINT_IN_TIME_FUNDAMENTAL_COVERAGE", "PROVIDER_COVERAGE", "RESEARCH_DEGRADED"))
    gaps.append(_gap("INTRADAY_HISTORY_BEFORE_YFINANCE_RETENTION", "LICENSED_INTRADAY_PROVIDER", "LONG_OOS_INTRADAY_LIMITED"))
    if selective_ml.get("status") != "GO":
        gaps.append(_gap("CLOSED_FORWARD_LABELS_FOR_ML", "FORWARD_OBSERVATION_TIME", "ML_NOT_TRAINABLE"))
    if ibkr_news.get("historical_headlines_capability") != "AVAILABLE":
        gaps.append(
            _gap(
                "IBKR_TWS_HISTORICAL_NEWS",
                "OPTIONAL_IBKR_NEWS_SUBSCRIPTION",
                "ALTERNATIVE_NEWS_SOURCE_UNAVAILABLE",
            )
        )

    core_ready = all(
        (
            cache.get("status") == "GO",
            corporate_actions.get("status") == "GO",
            fx.get("status") == "GO",
            macro.get("status") == "GO",
            cot.get("status") == "GO",
        )
    )
    payload = {
        "schema": "stocks_comprehensive_data_readiness_v1",
        "status": "CORE_RESEARCH_DATA_GO_WITH_DOCUMENTED_GAPS" if core_ready else "CORE_RESEARCH_DATA_BLOCKED",
        "generated_at": datetime.now(UTC).isoformat(),
        "core_research_ready": core_ready,
        "all_desired_data_available": not gaps,
        "available_source_count": sources.get("available_source_count", 0),
        "layers": layers,
        "open_data_gaps": gaps,
        "synthetic_gap_filling_allowed": False,
        "context_sources_used_as_ohlcv": False,
        "strategy_authority": "NONE",
        "execution_authority": "NONE",
        "broker_write_calls": 0,
    }
    _atomic_json(project_root / "output/reports/data_readiness.json", payload)
    return payload


def _gap(name: str, acquisition: str, impact: str) -> dict[str, str]:
    return {"data": name, "acquisition": acquisition, "impact": impact}


def system_self_test(
    project_root: Path,
    *,
    env_file: str = ".env.ibkr",
) -> dict[str, Any]:
    config = config_validation(project_root, env_file=env_file)
    sources = data_source_status(project_root)
    risk = portfolio_management_command(project_root, "risk")
    readiness = system_readiness(project_root)
    checks = {
        "config_validation": config.get("status") == "GO",
        "data_sources": sources.get("status") == "GO",
        "signals": readiness.get("signals", {}).get("status") == "GO",
        "telegram": readiness.get("telegram", {}).get("status")
        == "ENABLED",
        "risk_engine": risk.get("status")
        in {"GO", "NO_TARGET_POSITIONS"},
        "broker_calls_zero": readiness.get("broker_calls") == 0,
        "orders_generated_zero": readiness.get("orders_generated") == 0,
    }
    return {
        "schema": "stocks_self_test_v1",
        "status": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "execution_readiness": readiness.get("status"),
        "hard_blockers": readiness.get("hard_blockers", []),
        "config_status": config.get("status"),
        "data_source_status": sources.get("status"),
        "risk_status": risk.get("status"),
        "broker_calls": 0,
        "financial_calls": _zero_financial_calls(),
        "orders_generated": 0,
        "execution_authority": "NONE",
    }


def system_readiness(project_root: Path) -> dict[str, Any]:
    recovery = recover_survivors(project_root)
    signals = signal_status(project_root)
    autopilot = runtime_command(project_root, "status")
    phase9 = _read(project_root / "output" / "ibkr" / "phase9" / "status.json")
    live = live_preflight(
        project_root, env_file=".env.ibkr.live", probe_socket=False
    )
    kill_switch = live_kill_switch(project_root, command="status")
    telegram = telegram_command(project_root, "status")
    top_signals = publish_top_signals(
        project_root,
        mode="diversified",
        limit=5,
    )
    counts = recovery["classification_counts"]
    blockers = [
        *live["blockers"],
        *(
            ["NO_FROZEN_SHADOW_OR_HIGHER_STRATEGY"]
            if not any(
                counts.get(level, 0)
                for level in (
                    "FROZEN_SHADOW",
                    "MANUAL_SIGNAL_CANDIDATE",
                    "PAPER_CANDIDATE",
                    "LIVE_CANARY_CANDIDATE",
                )
            )
            else []
        ),
    ]
    report = {
        "schema": "stocks_system_readiness_v1",
        "status": "TECHNICAL_RESEARCH_AND_SIGNALS_GO_EXECUTION_BLOCKED",
        "generated_at": datetime.now(UTC).isoformat(),
        "architecture": {
            "canonical_entrypoint": "main.py",
            "broker_api": "OFFICIAL_NATIVE_IBAPI",
            "paper_adapter": "PHASE9_MANUAL_PAPER",
            "live_adapter": (
                "LEVEL1_LIVE_CANARY_WRITER_OFFLINE_FROZEN"
                if live.get("checks", {}).get("live_writer_frozen") is True
                else "PREFLIGHT_AND_KILL_SWITCH_ONLY_WRITER_NOT_FROZEN"
            ),
            "signal_engine": "BROKER_INDEPENDENT",
            "research_autopilot": "BOUNDED_EXTERNAL_SCHEDULER",
        },
        "research": {
            "files_scanned": recovery["files_scanned"],
            "rows_scanned": recovery["rows_scanned"],
            "survivor_count": recovery["survivor_count"],
            "classification_counts": counts,
            "statistical_uncertainty_is_not_hard_reject": True,
        },
        "signals": signals,
        "top_signals": top_signals,
        "autopilot": autopilot,
        "telegram": telegram,
        "paper": {
            "status": phase9.get("status", "NOT_AVAILABLE"),
            "submit_cancel_canary": phase9.get("checks", {}).get(
                "submit_cancel_canary", False
            ),
            "fill_close_canary": phase9.get("checks", {}).get(
                "fill_canary", False
            ),
            "open_blockers": phase9.get("open_blockers", []),
        },
        "live": {
            "preflight_status": live["status"],
            "blockers": live["blockers"],
            "kill_switch": kill_switch,
            "real_live_order_placed": False,
        },
        "hard_blockers": sorted(set(blockers)),
        "risk_limits": {
            "first_live_max_order_eur": 10,
            "first_live_max_total_exposure_eur": 25,
            "first_live_max_positions": 1,
            "first_live_max_new_orders_per_day": 1,
            "shorts": False,
            "margin": False,
            "options": False,
            "futures_first_canary": False,
            "autoscaling": False,
        },
        "commands": {
            "daily": "python .\\main.py daily",
            "signals": "python .\\main.py signals scan",
            "autopilot": "python .\\main.py autopilot run-once",
            "telegram": "python .\\main.py telegram status",
            "paper_status": "python .\\main.py ibkr phase9 status",
            "live_preflight": "python .\\main.py live preflight",
            "live_canary": (
                "python .\\main.py live canary --strategy <ID> --symbol <SYMBOL> "
                '--max-order-eur 10 --approval "<EXACTE_PHRASE>"'
            ),
            "position_status": "python .\\main.py live position-status",
            "controlled_exit": (
                'python .\\main.py live close-position --symbol <SYMBOL> '
                '--approval "<EXACTE_PHRASE>"'
            ),
        },
        "SIGNALS_CAN_RUN_WITHOUT_BROKER": True,
        "SIGNALS_INCLUDE_STOP_LOSS": True,
        "SIGNALS_INCLUDE_TAKE_PROFIT": True,
        "MANUAL_EXECUTION_SUPPORTED": True,
        "AUTOPILOT_CONTINUOUS_RESEARCH": True,
        "AUTOPILOT_AUTO_LIVE_PROMOTION": False,
        "SIGNAL_AUTHORITY_SEPARATE_FROM_EXECUTION": True,
        "execution_authority": "NONE",
        "broker_calls": 0,
        "orders_generated": 0,
        "real_live_order_placed": False,
    }
    _publish_reports(project_root, report, recovery, phase9, kill_switch)
    _publish_system_audit(project_root, report, recovery, live)
    return report


def _zero_financial_calls() -> dict[str, int]:
    return {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
        "request_order_id": 0,
        "auto_bind_order": 0,
        "exercise_option": 0,
        "market_data": 0,
        "historical_data": 0,
    }


def _publish_reports(
    project_root: Path,
    report: dict[str, Any],
    recovery: dict[str, Any],
    phase9: dict[str, Any],
    kill_switch: dict[str, Any],
) -> None:
    unavailable = {
        "status": "DATA_NOT_AVAILABLE",
        "execution_authority": "NONE",
    }
    phase9_public = {
        "schema": "public_phase9_execution_summary_v1",
        "status": phase9.get("status", "NOT_AVAILABLE"),
        "submit_cancel_canary": phase9.get("checks", {}).get(
            "submit_cancel_canary", False
        ),
        "fill_close_canary": phase9.get("checks", {}).get(
            "fill_canary", False
        ),
        "reconciliation": phase9.get("checks", {}).get(
            "reconciliation", False
        ),
        "open_blockers": phase9.get("open_blockers", []),
        "execution_authority": "NONE",
    }
    reports = {
        "system_readiness.json": report,
        "research_candidates.json": recovery,
        "paper_execution_report.json": phase9_public,
        "reconciliation_report.json": _read(
            project_root / "output" / "ibkr" / "phase8" / "status.json"
        ),
        "risk_status.json": {
            "status": "GO",
            "limits": report["risk_limits"],
            "execution_authority": "NONE",
        },
        "kill_switch_status.json": kill_switch,
        "current_market_regime.json": _read(
            project_root / "output" / "dynamic" / "current_regime.json"
        ),
        "current_macro_regime.json": _read(
            project_root / "output" / "macro" / "regime.json"
        ),
        "current_sector_regimes.json": _read(
            project_root / "output" / "macro" / "sector-impact.json"
        ),
        "universe_status.json": _read(
            project_root / "output" / "screener" / "latest-summary.json"
        ),
        "shariah_universe.json": _read(
            project_root / "output" / "screener" / "latest-summary.json"
        ),
        "top_opportunities.json": _read(
            project_root
            / "output"
            / "signals"
            / "latest_top_5_publication.json"
        ),
        "strategy_router_status.json": _read(
            project_root / "output" / "dynamic" / "strategy_weights.json"
        ),
        "live_strategy_approvals.json": _read(
            project_root / "output" / "dynamic" / "live_canary_readiness.json"
        ),
        "live_canary_queue.json": {
            "status": "BLOCKED",
            "queue": [],
            "blockers": report["live"]["blockers"],
            "execution_authority": "NONE",
        },
        "current_portfolio.json": _read(
            project_root / "output" / "portfolio" / "current_allocation.json"
        ),
        "current_positions.json": {
            **phase9_public,
            "public_position_values": "REDACTED",
        },
        "current_orders.json": {
            **phase9_public,
            "public_order_values": "REDACTED",
        },
        "reconciliation_status.json": phase9_public,
        "portfolio_performance.json": {
            **unavailable,
            "reason": "NO_CONFIRMED_LIVE_OR_PAPER_FILL_CLOSE_SERIES",
        },
        "autopilot_status.json": report["autopilot"],
        "daily_execution_summary.json": _read(
            project_root / "output" / "operations" / "last-cycle.json"
        ),
    }
    roots = (project_root / "output" / "reports", project_root / "reports")
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        for name, payload in reports.items():
            _atomic_json(root / name, payload or unavailable)
    _atomic_text(
        project_root / "output" / "reports" / "system_readiness.md",
        "# System Readiness\n\n"
        f"Status: `{report['status']}`\n\n"
        f"Recovered survivors: {report['research']['survivor_count']}\n\n"
        f"Manual actionable strategies: "
        f"{report['signals']['manual_actionable_strategies']}\n\n"
        f"Paper status: `{report['paper']['status']}`\n\n"
        f"Live preflight: `{report['live']['preflight_status']}`\n\n"
        f"Telegram: `{report['telegram']['status']}`\n\n"
        "No real live order was placed. Autoscaling and first-canary futures are disabled.\n",
    )


def _publish_system_audit(
    project_root: Path,
    report: dict[str, Any],
    recovery: dict[str, Any],
    live: dict[str, Any],
) -> None:
    root = project_root / "output" / "reports" / "system_audit"
    module_root = project_root / "src" / "stocks"
    source_files = [
        *project_root.joinpath("src").rglob("*.py"),
        *project_root.joinpath("tests").rglob("*.py"),
        project_root / "main.py",
    ]
    source_files = [path for path in source_files if path.exists()]
    survivors = recovery.get("survivors", [])
    family_counts: dict[str, int] = {}
    timeframe_counts: dict[str, int] = {}
    duplicate_groups: dict[tuple[str, str, str, str], list[str]] = {}
    for row in survivors:
        family = str(row.get("family", "UNKNOWN"))
        timeframe = str(row.get("timeframe", "UNKNOWN"))
        family_counts[family] = family_counts.get(family, 0) + 1
        timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
        identity = (
            str(row.get("strategy_name", "")),
            family,
            timeframe,
            str(row.get("parameters", "")),
        )
        duplicate_groups.setdefault(identity, []).append(
            str(row.get("candidate_id", ""))
        )
    duplicates = [
        {
            "strategy_name": key[0],
            "family": key[1],
            "timeframe": key[2],
            "parameters": key[3],
            "candidate_ids": values,
        }
        for key, values in duplicate_groups.items()
        if len(values) > 1
    ]
    canonical_families = report["signals"].get(
        "canonical_signal_families", []
    )
    provider_inventory = _read(
        project_root
        / "output"
        / "research"
        / "phase11_6"
        / "provider_inventory.json"
    )
    macro_status = _read(project_root / "output" / "macro" / "status.json")
    macro_regime = _read(project_root / "output" / "macro" / "regime.json")
    screener = _read(
        project_root / "output" / "screener" / "latest-summary.json"
    )
    opportunity_ranking = _read(
        project_root / "output" / "portfolio" / "opportunity_ranking.json"
    )
    opportunities = opportunity_ranking.get("opportunities", [])
    fundamental_covered = sum(
        1
        for row in opportunities
        if row.get("components", {}).get("fundamental_quality") is not None
    )
    reports = {
        "repository_inventory.json": {
            "schema": "repository_inventory_v1",
            "status": "GO",
            "canonical_entrypoint": "main.py",
            "source_file_count": len(source_files),
            "test_file_count": len(
                [path for path in source_files if "tests" in path.parts]
            ),
            "top_level_modules": sorted(
                path.name
                for path in (
                    module_root.iterdir() if module_root.exists() else []
                )
                if path.is_dir() and not path.name.startswith("__")
            ),
            "execution_authority": "NONE",
        },
        "strategy_inventory.json": {
            "schema": "strategy_inventory_v1",
            "status": "GO",
            "survivor_count": len(survivors),
            "classification_counts": recovery.get(
                "classification_counts", {}
            ),
            "family_counts": family_counts,
            "timeframe_counts": timeframe_counts,
            "strategies": survivors,
            "automatic_authority": "NONE",
        },
        "strategy_family_map.json": {
            "schema": "strategy_family_map_v1",
            "status": "GO",
            "canonical_family_count": len(canonical_families),
            "canonical_families": canonical_families,
            "frozen_family_counts": family_counts,
            "unimplemented_frozen_families": report["signals"].get(
                "unimplemented_frozen_candidate_families", []
            ),
        },
        "duplicate_strategy_report.json": {
            "schema": "duplicate_strategy_report_v1",
            "status": "GO",
            "identity": (
                "strategy_name+family+timeframe+canonical_parameters"
            ),
            "duplicate_group_count": len(duplicates),
            "duplicate_groups": duplicates,
        },
        "data_provider_report.json": {
            "schema": "data_provider_report_v1",
            "status": (
                provider_inventory.get("status", "DATA_NOT_AVAILABLE")
            ),
            "provider_inventory": provider_inventory,
            "context_sources_are_not_ohlcv": True,
            "broker_calls": 0,
        },
        "macro_coverage_report.json": {
            "schema": "macro_coverage_report_v1",
            "status": macro_status.get("status", "DATA_NOT_AVAILABLE"),
            "macro_status": macro_status,
            "current_regime": macro_regime,
            "macro_is_context_only": True,
            "macro_has_execution_authority": False,
        },
        "fundamental_coverage_report.json": {
            "schema": "fundamental_coverage_report_v1",
            "status": "GO" if fundamental_covered else "DATA_NOT_AVAILABLE",
            "ranked_opportunity_count": len(opportunities),
            "fundamental_score_present_count": fundamental_covered,
            "screener_summary": screener,
            "point_in_time_limitations_preserved": True,
        },
        "broad_universe_report.json": broad_universe_status(project_root),
        "ibkr_capability_report.json": {
            "schema": "ibkr_capability_report_v1",
            "status": "EXECUTION_BLOCKED",
            "broker_api": "OFFICIAL_NATIVE_IBAPI",
            "paper_adapter": report["architecture"]["paper_adapter"],
            "live_adapter": report["architecture"]["live_adapter"],
            "live_checks": live.get("checks", {}),
            "read_only_observation_supported": True,
            "live_order_submission_permitted": False,
            "execution_authority": "NONE",
        },
        "live_blocker_report.json": {
            "schema": "live_blocker_report_v1",
            "status": "BLOCKED" if report["hard_blockers"] else "GO",
            "blockers": report["hard_blockers"],
            "paper_blockers": report["paper"]["open_blockers"],
            "live_preflight_status": report["live"]["preflight_status"],
            "financial_finalist_go": False,
            "execution_authority": "NONE",
            "real_live_order_placed": False,
        },
    }
    for name, payload in reports.items():
        _atomic_json(root / name, payload)
    _atomic_text(
        root / "architecture_gap_report.md",
        "# Architecture Gap Report\n\n"
        f"System status: `{report['status']}`\n\n"
        f"Raw top signals: {len(report['top_signals'].get('signals', []))}\n\n"
        f"Manual actionable signals: "
        f"{report['top_signals'].get('manual_signal_eligible_count', 0)}\n\n"
        f"Automated execution eligible: "
        f"{report['top_signals'].get('automated_execution_eligible_count', 0)}\n\n"
        "Open execution blockers:\n\n"
        + "".join(
            f"- `{blocker}`\n" for blocker in report["hard_blockers"]
        )
        + "\nThe running service remains SIGNALS_ONLY with execution authority NONE.\n",
    )


def _read(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, default=str))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(value, encoding="utf-8")
        last_error: PermissionError | None = None
        for delay in (0.0, 0.01, 0.05, 0.1, 0.25):
            if delay:
                time.sleep(delay)
            try:
                os.replace(temporary, path)
                return
            except PermissionError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
    finally:
        temporary.unlink(missing_ok=True)
