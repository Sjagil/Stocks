from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stocks.application.context import load_app_context  # noqa: E402
from stocks.application.lifecycle import (  # noqa: E402
    build_disconnect_drill_preflight_report,
    build_doctor_report,
    build_ibkr_service,
)
from stocks.application.phase_gates import phase1_freeze_status  # noqa: E402
from stocks.analysis import (  # noqa: E402
    analyze_asset,
    build_analysis_coverage,
    build_group_intelligence,
    build_news_event_intelligence,
    build_news_event_study,
    group_intelligence_status,
    news_event_intelligence_status,
    news_event_study_status,
)
from stocks.capital import (  # noqa: E402
    capital_command,
    portfolio_management_command,
)
from stocks.context import (  # noqa: E402
    RealtimeEquityConfig,
    build_asset_context,
    collect_realtime_equity_context,
    collect_cot_context,
    cot_status,
    entry_observer_status,
    episode_outcome_status,
    observe_shortlist,
    settle_entry_episodes,
)
from stocks.macro import (  # noqa: E402
    macro_audit,
    macro_collect,
    macro_compare,
    macro_conflicts,
    macro_events,
    macro_explain,
    macro_freeze,
    macro_history,
    macro_readiness,
    macro_regime,
    macro_report,
    macro_score,
    macro_sector_impact,
    macro_status,
    macro_strategy_impact,
    macro_update,
    macro_validate,
)
from stocks.research.autopilot.service import (  # noqa: E402
    autopilot_audit,
    autopilot_campaign,
    autopilot_candidates,
    autopilot_combine,
    autopilot_compare,
    autopilot_daily,
    autopilot_generate,
    autopilot_freeze,
    autopilot_leaderboard,
    autopilot_monthly,
    autopilot_rejected,
    autopilot_smoke,
    autopilot_status,
    autopilot_strategy,
    autopilot_taxonomy,
    autopilot_weekly,
    component_registry_report,
    forward_register,
    forward_run,
    forward_status,
    portfolio_backtest,
    portfolio_stress,
)
from stocks.research.macro_pairs import run_macro_pair_validation  # noqa: E402
from stocks.research.promotion import recover_survivors  # noqa: E402
from stocks.research.sec_overlay import (  # noqa: E402
    sec_intelligence_audit,
    sec_intelligence_status,
    sec_overlay_for_signal,
)
from stocks.research.registry_service import (  # noqa: E402
    research_registry_command,
)
from stocks.research.autopilot.runtime import runtime_command  # noqa: E402
from stocks.daily import run_daily  # noqa: E402
from stocks.dynamic import dynamic_command  # noqa: E402
from stocks.live import (  # noqa: E402
    activate_autonomous_level_one,
    activate_level_two,
    automatic_cycle,
    automatic_cycle_audit,
    automatic_cycle_freeze,
    automatic_cycle_status,
    controlled_live_approve,
    controlled_live_preflight,
    controlled_live_prepare,
    controlled_live_submit,
    live_approve,
    live_audit,
    live_canary,
    live_close_position,
    live_component_status,
    live_freeze,
    live_kill_switch,
    live_position_status,
    live_preflight,
    live_prepare,
    live_reconcile,
    live_status,
    live_strategy_allowlist,
    live_submit,
    live_writer_integrity_command,
)
from stocks.live.autonomous_policy import (  # noqa: E402
    build_p2_1_freeze,
    verify_p2_1_freeze,
)
from stocks.notifications import telegram_command  # noqa: E402
from stocks.p3.publisher import publish_p3_evidence  # noqa: E402
from stocks.operations import (  # noqa: E402
    MACHINE_MODES,
    execution_command as operations_execution_command,
    launch_command,
    machine_command,
    positions_command,
)
from stocks.readiness import (  # noqa: E402
    comprehensive_data_readiness,
    config_validation,
    data_source_status,
    system_readiness,
    system_self_test,
)
from stocks.signals import (  # noqa: E402
    promote_manual_signals,
    publish_top_signals,
    signal_asset,
    signal_explain,
    signal_export,
    signal_list,
    signal_mark_closed,
    signal_mark_executed,
    signal_order_plan,
    signal_scan,
    signal_status,
)
from stocks.universe import (  # noqa: E402
    broad_universe_status,
    discovery_universe_command,
    rank_universe_dimension,
)
from stocks.ui import ui_command  # noqa: E402
from stocks.data.bars import (  # noqa: E402
    BarCacheLayout,
    BarDataSource,
    BarDataType,
    BarInterval,
    BarRequestPolicy,
    bar_schema_manifest,
    initialize_bar_cache,
    validate_bar_cache,
    validate_bar_contract_identity_links,
)
from stocks.data.ibkr_historical import collect_ibkr_daily_bars  # noqa: E402
from stocks.data.multitimeframe import (  # noqa: E402
    CANONICAL_INTERVALS,
    DEFAULT_SYMBOLS,
    audit_multitimeframe_sources,
    collect_multitimeframe_data,
    multitimeframe_schema,
    multitimeframe_status,
    parse_intervals,
    parse_symbols,
    provider_inventory,
    validate_multitimeframe_cache,
)
from stocks.data.corporate_actions import (  # noqa: E402
    CorporateActionLayout,
    collect_corporate_actions_for_universe,
    corporate_action_schema,
    corporate_action_status,
    validate_corporate_action_cache,
)
from stocks.data.fx import (  # noqa: E402
    FxCacheLayout,
    fx_schema,
    validate_fx_cache,
)
from stocks.data.phase5_1 import (  # noqa: E402
    build_total_returns_for_universe_v1_1,
    collect_fx_for_universe_v1_1,
    fx_status_v1_1,
)
from stocks.data.total_returns import (  # noqa: E402
    TotalReturnLayout,
    total_return_schema,
    total_return_status,
    validate_total_return_cache,
)
from stocks.domain.assets import AssetClass, IbkrSecurityType  # noqa: E402
from stocks.execution.simulation import (  # noqa: E402
    Phase7Layout,
    audit_ledger,
    init_ledger,
    phase7_freeze,
    phase7_schema,
    phase7_status,
    reconcile_fixtures,
    replay_phase7,
    simulate_phase7,
)
from stocks.ibkr.contract_cache import (  # noqa: E402
    ContractCacheLayout,
    contract_schema_manifest,
    empty_contract_manifest,
    export_contract_identity,
    initialize_contract_cache,
    read_contract_cache_rows,
    validate_contract_cache,
)
from stocks.ibkr.contract_resolver import LiveContractResolver  # noqa: E402
from stocks.ibkr.contract_queue import (  # noqa: E402
    build_opportunity_contract_queue,
)
from stocks.ibkr.live_contract_refresh import (  # noqa: E402
    refresh_live_read_only_contracts,
    resolve_new_live_read_only_contracts,
)
from stocks.ibkr.contracts import (  # noqa: E402
    ContractResolutionRequest,
    build_ibkr_contract_spec,
    gated_contract_resolution_report,
)
from stocks.ibkr.data_capabilities import (  # noqa: E402
    build_capability_matrix,
    capability_schema,
    strategy_capability_gate,
)
from stocks.ibkr.news import (  # noqa: E402
    collect_ibkr_news,
    ibkr_news_schema,
    probe_ibkr_news,
)
from stocks.ibkr.reconciliation import (  # noqa: E402
    phase8_audit,
    phase8_freeze,
    phase8_preflight,
    phase8_reconcile,
    phase8_schema,
    phase8_snapshot,
    phase8_stability_check,
    phase8_status,
)
from stocks.ibkr.phase8_1 import (  # noqa: E402
    establish_baseline,
    phase8_1_audit,
    phase8_1_freeze,
    phase8_1_preflight,
    phase8_1_schema,
    phase8_1_status,
    recovery_drill,
    run_soak,
)
from stocks.ibkr.paper_execution import (  # noqa: E402
    accept_operator_attested_manual_completion,
    canary_results as phase9_canary_results,
    phase9_audit,
    phase9_canary_a_evidence,
    phase9_freeze,
    phase9_observe_known_fill,
    phase9_preflight,
    phase9_reconcile,
    phase9_schema,
    phase9_status,
)
from stocks.ibkr.paper_execution.audit import (  # noqa: E402
    approve as phase9_approve,
    approve_cancel as phase9_approve_cancel,
    cancel as phase9_cancel,
    phase9_canary_b_readiness,
    phase9_fill_close_audit,
    prepare as phase9_prepare,
    prepare_cancel as phase9_prepare_cancel,
    submit as phase9_submit,
)
from stocks.auto_paper import (  # noqa: E402
    PHASE10_FREEZE_MARKER,
    PHASE10_MARKER,
    phase10_command,
)
from stocks.research.phase11_2 import (  # noqa: E402
    PHASE11_2_FREEZE_MARKER,
    PHASE11_2_MARKER,
    phase11_2_command,
)
from stocks.research.phase11_3 import (  # noqa: E402
    PHASE11_3_FREEZE_MARKER,
    PHASE11_3_MARKER,
    phase11_3_command,
)
from stocks.research.phase11_4 import PHASE11_4_COMMANDS, phase11_4_command  # noqa: E402
from stocks.research.phase11_4.acquisition import (  # noqa: E402
    acquire_price_histories,
    compact_price_histories,
)
from stocks.research.phase11_6 import (  # noqa: E402
    completion_audit as phase11_6_completion_audit,
    phase11_6_schema,
    phase11_6_status,
    run_cohort_and_stress as phase11_6_cohorts,
    run_combinations as phase11_6_combinations,
    run_data_audit as phase11_6_data_audit,
    run_phase11_6,
    run_walk_forward as phase11_6_walk_forward,
)
from stocks.research.phase11_7 import (  # noqa: E402
    phase11_7_schema,
    phase11_7_status,
    run_finalist_campaign as phase11_7_run,
    run_rotation_campaign as phase11_7_rotation,
)
from stocks.research.phase11_8 import (  # noqa: E402
    data_coverage as phase11_8_data_coverage,
    finalize_campaign as phase11_8_finalize,
    phase11_8_schema,
    phase11_8_status,
    portfolio_invariant_audit as phase11_8_portfolio_audit,
    run_campaign as phase11_8_run,
)
from stocks.research.phase11_9 import (  # noqa: E402
    current_watchlist as phase11_9_watchlist,
    phase11_9_schema,
    phase11_9_status,
    run_diagnostics as phase11_9_diagnose,
    run_discovery as phase11_9_run,
)
from stocks.research.phase11_10 import (  # noqa: E402
    phase11_10_pit_observe,
    phase11_10_qualification_audit,
    phase11_10_qualification_freeze,
    phase11_10_reclassify,
    phase11_10_schema,
    phase11_10_status,
    phase11_10_top20,
    phase11_10_watchlist,
    run_phase11_10,
)
from stocks.research.active_swing_sprints import (  # noqa: E402
    active_swing_sprint_status,
    gate_value_attribution_status,
    publish_active_swing_leaderboards,
    publish_shortlist_coverage,
    refresh_active_swing_observation,
    run_active_swing_sprints,
    run_entry_filter_experiment,
    settle_rejected_opportunities,
    train_selective_ml,
)
from stocks.research.evidence_throughput import (  # noqa: E402
    publish_evidence_throughput,
)
from stocks.research.phase11_12 import (  # noqa: E402
    phase11_12_observe,
    phase11_12_schema,
    phase11_12_status,
    register_phase11_12_catalog,
    run_phase11_12,
)
from stocks.research.phase11_12_forward import (  # noqa: E402
    lower_timeframe_forward_status,
)
from stocks.research.phase11_13 import (  # noqa: E402
    phase11_13_observe,
    phase11_13_schema,
    phase11_13_status,
    run_phase11_13,
)
from stocks.research.phase11_14 import (  # noqa: E402
    phase11_14_observe,
    phase11_14_schema,
    phase11_14_status,
    run_phase11_14,
)
from stocks.research.phase11_15 import (  # noqa: E402
    phase11_15_schema,
    phase11_15_status,
    run_phase11_15,
)
from stocks.regimes import (  # noqa: E402
    regimes_audit,
    regimes_current,
    regimes_fit,
    regimes_schema,
    regimes_status,
    regimes_walk_forward,
)
from stocks.screener import (  # noqa: E402
    screener_asset,
    screener_export,
    screener_history,
    screener_preview,
    screener_report,
    screener_run,
    screener_status,
    screener_top,
)
from stocks.shadow import (  # noqa: E402
    activation_audit as phase8_2_activation_audit,
    audit_ledger as phase8_2_audit_ledger,
    freeze as phase8_2_freeze,
    init_ledger as phase8_2_init_ledger,
    phase8_2_schema,
    register_fixtures as phase8_2_register_fixtures,
    replay as phase8_2_replay,
    simulate as phase8_2_simulate,
    status as phase8_2_status,
)
from stocks.market import (  # noqa: E402
    DEFAULT_CONTEXT_SYMBOLS,
    audit_market_context_sources,
    build_market_context,
    market_context_schema,
    market_context_status,
)
from stocks.market.sessions import (  # noqa: E402
    SessionCacheLayout,
    build_session_cache_from_contract_rows,
    market_sessions_by_date_report,
    market_next_open_report,
    market_session_schema_manifest,
    market_sessions_report,
    market_sessions_range_report,
    market_status_report,
    read_session_cache_records,
    validate_session_cache,
)
from stocks.research.instrument_manifest import (  # noqa: E402
    InstrumentManifestLayout,
    initialize_instrument_manifest,
    instrument_manifest_schema,
    validate_instrument_manifest,
)
from stocks.research.critical_trading import (  # noqa: E402
    collect_yfinance_data,
    critical_trading_schema,
    run_critical_trading_backtests,
    run_perfection_pipeline,
)
from stocks.research.phase6 import (  # noqa: E402
    dataset_audit,
    load_phase6_dataset,
    phase6_freeze,
    phase6_schema,
    phase6_status,
    run_baselines,
    run_phase6_pipeline,
    run_strategy_grid,
    run_walk_forward,
)
from stocks.research.phase6_diagnostics import (  # noqa: E402
    Phase61Layout,
    phase6_1_freeze,
    phase6_1_schema,
    phase6_1_status,
    run_phase6_1_pipeline,
)
from stocks.research.phase6_2 import (  # noqa: E402
    Phase62Layout,
    phase6_2_freeze,
    phase6_2_schema,
    phase6_2_status,
    run_phase6_2_pipeline,
)
from stocks.research.phase6_3 import (  # noqa: E402
    Phase63Layout,
    phase6_3_freeze,
    phase6_3_schema,
    phase6_3_status,
    run_phase6_3_pipeline,
)
from stocks.research.phase6_4 import (  # noqa: E402
    Phase64Layout,
    phase6_4_freeze,
    phase6_4_schema,
    phase6_4_status,
    preregister_phase6_4,
    run_phase6_4_pipeline,
)
from stocks.research.phase11_1 import (  # noqa: E402
    PHASE11_1_FREEZE_MARKER,
    PHASE11_1_MARKER,
    Phase111Layout,
    phase11_1_freeze,
    phase11_1_schema,
    phase11_1_status,
    preregister_phase11_1,
    run_phase11_1_pipeline,
)
from stocks.strategies.multi_asset import (  # noqa: E402
    MultiAssetBacktestConfig,
    load_strategy_series_from_bar_cache,
    multi_asset_strategy_schema,
    run_multi_asset_rotation_backtest,
)


def _print_json(payload: dict[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    try:
        print(serialized)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(serialized.encode("utf-8") + b"\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical entrypoint for the stocks framework.",
    )
    parser.add_argument(
        "--env-file",
        default=".env.ibkr",
        help="Path to the read-only IBKR environment file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p3 = subparsers.add_parser(
        "p3", help="Publish the unified strategy and AI evidence contract."
    )
    p3_subparsers = p3.add_subparsers(dest="p3_command", required=True)
    p3_subparsers.add_parser("publish")

    config = subparsers.add_parser(
        "config", help="Validate the canonical fail-closed configuration."
    )
    config_subparsers = config.add_subparsers(
        dest="config_command", required=True
    )
    config_subparsers.add_parser("validate")

    sec = subparsers.add_parser(
        "sec",
        help="Causal SEC ownership/event intelligence and bounded ranking overlay.",
    )
    sec_subparsers = sec.add_subparsers(dest="sec_command", required=True)
    sec_subparsers.add_parser("status")
    sec_subparsers.add_parser("audit")
    sec_overlay = sec_subparsers.add_parser("overlay")
    sec_overlay.add_argument("--symbol", required=True)
    sec_overlay.add_argument("--as-of", required=True)
    sec_overlay.add_argument("--base-score", type=float, required=True)
    sec_overlay.add_argument("--base-authorized", action="store_true")

    subparsers.add_parser(
        "self-test",
        help="Run brokerless integrated configuration and readiness checks.",
    )

    ibkr = subparsers.add_parser("ibkr", help="IBKR read-only service commands.")
    ibkr_subparsers = ibkr.add_subparsers(dest="ibkr_command", required=True)
    ibkr_subparsers.add_parser("probe", help="Connect, prove health, disconnect.")
    ibkr_subparsers.add_parser("status", help="Connect, report current health, disconnect.")
    watch = ibkr_subparsers.add_parser("watch", help="Run heartbeat loop for a bounded time.")
    watch.add_argument("--seconds", type=float, default=60.0)
    cycle = ibkr_subparsers.add_parser(
        "cycle",
        help="Run bounded read-only connect/disconnect validation cycles.",
    )
    cycle.add_argument("--count", type=int, default=25)
    ibkr_subparsers.add_parser(
        "duplicate-client-check",
        help="Verify duplicate IBKR client ID handling without order authority.",
    )
    drill_preflight = ibkr_subparsers.add_parser(
        "disconnect-drill-preflight",
        help="Fail-closed preflight before the manual TWS paper disconnect drill.",
    )
    drill_preflight.add_argument(
        "--skip-socket-check",
        action="store_true",
        help="Validate configuration only; used by tests and offline audits.",
    )
    disconnect_drill = ibkr_subparsers.add_parser(
        "disconnect-drill",
        help="Observe a manual TWS disconnect and prove bounded reconnect.",
    )
    disconnect_drill.add_argument("--seconds", type=float, default=180.0)
    disconnect_drill.add_argument("--poll-seconds", type=float, default=2.0)
    data_capabilities = ibkr_subparsers.add_parser(
        "data-capabilities",
        help="Publish evidence-backed IBKR market-data capabilities.",
    )
    data_capability_subparsers = data_capabilities.add_subparsers(
        dest="data_capability_command", required=True
    )
    data_capability_subparsers.add_parser("schema")
    data_capability_subparsers.add_parser("status")
    data_capability_gate = data_capability_subparsers.add_parser(
        "gate",
        help="Fail closed unless all named capabilities are proven available.",
    )
    data_capability_gate.add_argument(
        "--require", action="append", required=True
    )
    data_capability_gate.add_argument("--asset-reference")
    ibkr_news = ibkr_subparsers.add_parser(
        "news",
        help="Bounded read-only TWS news provider and headline commands.",
    )
    ibkr_news_subparsers = ibkr_news.add_subparsers(
        dest="ibkr_news_command", required=True
    )
    ibkr_news_subparsers.add_parser("schema")
    ibkr_news_capabilities = ibkr_news_subparsers.add_parser(
        "capabilities"
    )
    ibkr_news_capabilities.add_argument(
        "--connection-env-file", default=".env.ibkr"
    )
    ibkr_news_collect = ibkr_news_subparsers.add_parser("collect")
    ibkr_news_collect.add_argument(
        "--connection-env-file", default=".env.ibkr"
    )
    ibkr_news_collect.add_argument(
        "--symbols", default="AAPL,SPY,ASML,ON,NVDA"
    )
    ibkr_news_collect.add_argument("--lookback-hours", type=int, default=72)
    ibkr_news_collect.add_argument(
        "--max-results-per-symbol", type=int, default=50
    )
    contract = ibkr_subparsers.add_parser(
        "contract",
        help="Phase 2 contract resolver commands, gated by Phase 1 freeze.",
    )
    contract_subparsers = contract.add_subparsers(dest="contract_command", required=True)
    contract_subparsers.add_parser("status", help="Report Phase 2 resolver gate status.")
    contract_subparsers.add_parser("schema", help="Print the offline Phase 2 contract schema.")
    contract_subparsers.add_parser(
        "init-cache",
        help="Create the local Phase 2 contract cache manifest and JSONL audit files.",
    )
    contract_subparsers.add_parser(
        "validate-cache",
        help="Validate local Phase 2 STK/FUT contract cache files without broker access.",
    )
    export_identity = contract_subparsers.add_parser(
        "export-identity",
        help="Export one resolved contract identity from the local cache by conId.",
    )
    export_identity.add_argument("--con-id", type=int, required=True)
    resolve_stock = contract_subparsers.add_parser(
        "resolve-stock",
        help="Prepare a gated STK contract resolution request.",
    )
    resolve_stock.add_argument("--symbol", required=True)
    resolve_stock.add_argument(
        "--asset-class",
        choices=("stock", "etf", "commodity_etf", "bond_etf"),
        default="stock",
    )
    resolve_stock.add_argument("--currency", default="USD")
    resolve_stock.add_argument("--exchange", default="SMART")
    resolve_stock.add_argument("--primary-exchange")
    refresh_live_contracts = contract_subparsers.add_parser(
        "refresh-live-read-only",
        help="Refresh existing STK contract identities through strict read-only live TWS.",
    )
    refresh_live_contracts.add_argument("--symbols", required=True)
    refresh_live_contracts.add_argument(
        "--connection-env-file",
        default=".env.ibkr.live",
    )
    resolve_new_live_contracts = contract_subparsers.add_parser(
        "resolve-new-live-read-only",
        help="Resolve explicit new STK identities through strict read-only live TWS.",
    )
    resolve_new_live_contracts.add_argument(
        "--manifest",
        default="config/ibkr/new_stock_contract_requests_v1.json",
    )
    resolve_new_live_contracts.add_argument(
        "--connection-env-file",
        default=".env.ibkr.live",
    )
    contract_subparsers.add_parser(
        "build-opportunity-queue",
        help="Build explicit new STK requests from current contract-only rejects.",
    )
    resolve_future = contract_subparsers.add_parser(
        "resolve-future",
        help="Prepare a gated FUT contract resolution request.",
    )
    resolve_future.add_argument("--symbol", required=True)
    resolve_future.add_argument("--exchange", required=True)
    resolve_future.add_argument("--currency", default="USD")
    resolve_future.add_argument("--expiry")
    phase8 = ibkr_subparsers.add_parser(
        "phase8",
        help="Phase 8 read-only TWS-paper reconciliation adapter.",
    )
    phase8_subparsers = phase8.add_subparsers(dest="phase8_command", required=True)
    phase8_subparsers.add_parser("schema", help="Print and write the Phase 8 schema artifact.")
    phase8_subparsers.add_parser("preflight", help="Validate read-only observer configuration and frozen gates.")
    phase8_subparsers.add_parser("snapshot", help="Capture one bounded read-only broker observation snapshot.")
    phase8_subparsers.add_parser("stability-check", help="Capture two bounded snapshots and classify stability.")
    phase8_subparsers.add_parser("reconcile", help="Compare broker observation with the frozen Phase 7 ledger without mutation.")
    phase8_subparsers.add_parser("audit", help="Run Phase 8 privacy and method allowlist audits.")
    phase8_subparsers.add_parser("status", help="Report Phase 8 readiness from current artifacts.")
    phase8_subparsers.add_parser("freeze", help="Write the Phase 8 freeze artifact.")
    phase8_1 = ibkr_subparsers.add_parser(
        "phase8-1",
        help="Phase 8.1 read-only observation soak and recovery.",
    )
    phase8_1_subparsers = phase8_1.add_subparsers(dest="phase8_1_command", required=True)
    phase8_1_subparsers.add_parser("schema", help="Print and write the Phase 8.1 schema artifact.")
    phase8_1_subparsers.add_parser("preflight", help="Validate Phase 8.1 frozen gate and observer readiness.")
    phase8_1_subparsers.add_parser("establish-baseline", help="Capture a stable A/B broker observation baseline.")
    soak = phase8_1_subparsers.add_parser("soak", help="Run bounded repeated read-only observation soak.")
    soak.add_argument("--duration-seconds", type=float, default=300.0)
    soak.add_argument("--interval-seconds", type=float, default=15.0)
    soak.add_argument("--stability-delay-seconds", type=float)
    recovery = phase8_1_subparsers.add_parser("recovery-drill", help="Run operator-assisted bounded reconnect recovery drill.")
    recovery.add_argument("--duration-seconds", type=float, default=240.0)
    recovery.add_argument("--poll-seconds", type=float, default=2.0)
    phase8_1_subparsers.add_parser("audit", help="Run Phase 8.1 privacy, method, callback and cleanup audits.")
    phase8_1_subparsers.add_parser("status", help="Report Phase 8.1 readiness from current artifacts.")
    phase8_1_subparsers.add_parser("freeze", help="Write the Phase 8.1 freeze artifact.")
    phase9 = ibkr_subparsers.add_parser(
        "phase9",
        help="Phase 9 manual-only IBKR TWS paper execution adapter.",
    )
    phase9_subparsers = phase9.add_subparsers(dest="phase9_command", required=True)
    phase9_subparsers.add_parser("schema", help="Print and write the Phase 9 schema artifact.")
    phase9_subparsers.add_parser("preflight", help="Validate Phase 9 paper writer configuration.")
    prepare9 = phase9_subparsers.add_parser("prepare", help="Prepare a manual paper canary intent.")
    prepare9.add_argument("--con-id", type=int, required=True)
    prepare9.add_argument("--side", choices=("BUY", "SELL"), required=True)
    prepare9.add_argument("--quantity", type=Decimal, required=True)
    prepare9.add_argument("--limit-price", type=Decimal, required=True)
    prepare9.add_argument("--reason", required=True)
    approve9 = phase9_subparsers.add_parser("approve", help="Approve one prepared manual paper intent.")
    approve9.add_argument("--intent-id", required=True)
    approve9.add_argument("--approval", required=True)
    submit9 = phase9_subparsers.add_parser("submit", help="Submit one approved manual paper intent.")
    submit9.add_argument("--intent-id", required=True)
    cancel_prepare9 = phase9_subparsers.add_parser("prepare-cancel", help="Prepare a manual cancellation approval challenge.")
    cancel_prepare9.add_argument("--intent-id", required=True)
    approve_cancel9 = phase9_subparsers.add_parser("approve-cancel", help="Approve one manual paper cancellation.")
    approve_cancel9.add_argument("--intent-id", required=True)
    approve_cancel9.add_argument("--approval", required=True)
    cancel9 = phase9_subparsers.add_parser("cancel", help="Cancel one known Phase 9 paper order after approval.")
    cancel9.add_argument("--intent-id", required=True)
    phase9_subparsers.add_parser("reconcile", help="Reconcile Phase 9 local ledger with paper observation.")
    observe_fill9 = phase9_subparsers.add_parser(
        "observe-known-fill",
        help="Record only a uniquely matched fill for one submitted Phase 9 intent.",
    )
    observe_fill9.add_argument("--intent-id", required=True)
    phase9_subparsers.add_parser("fill-close-audit", help="Run offline Phase 9.0.1 fill adoption and close reconciliation audits.")
    phase9_subparsers.add_parser("canary-b-readiness", help="Report offline readiness for the operator-run Phase 9 Canary B.")
    phase9_subparsers.add_parser("canary-a-evidence", help="Reconstruct immutable Canary A evidence without broker writes.")
    phase9_subparsers.add_parser(
        "canary-results",
        help="Reconstruct the durable submit/cancel and fill/close canary evidence.",
    )
    accept_manual9 = phase9_subparsers.add_parser(
        "accept-manual-completion",
        help=(
            "Accept an operator-attested external TWS paper close only when "
            "read-only pre/post broker continuity proves the account became empty."
        ),
    )
    accept_manual9.add_argument("--symbol", required=True)
    accept_manual9.add_argument("--con-id", type=int, required=True)
    accept_manual9.add_argument("--reason", required=True)
    phase9_subparsers.add_parser("audit", help="Run Phase 9 offline audits.")
    phase9_subparsers.add_parser("status", help="Report Phase 9 readiness.")
    phase9_subparsers.add_parser("freeze", help="Write Phase 9 freeze artifact.")
    phase10 = ibkr_subparsers.add_parser(
        "phase10",
        help="Offline autonomous Shariah paper-trading foundation commands.",
    )
    phase10_subparsers = phase10.add_subparsers(dest="phase10_command", required=True)
    for command, help_text in (
        ("preflight", "Validate fail-closed Phase 10 configuration and frozen dependencies."),
        ("shariah-audit", "Audit stock, ETF, commodity, and blocked-product Shariah gates."),
        ("movers-audit", "Audit read-only Phase 11.1 Daily Movers integration."),
        ("strategy-audit", "Audit the three fixed V1 strategy contracts."),
        ("portfolio-audit", "Audit portfolio sleeves, regimes, and exposure limits."),
        ("entry-audit", "Audit synthetic shadow-only automatic entry handling."),
        ("exit-audit", "Audit synthetic risk-reducing exit handling."),
        ("scheduler-audit", "Audit the bounded shadow-only scheduler."),
        ("kill-switch-audit", "Audit all Phase 10 kill switches."),
        ("replay-audit", "Audit offline fills, callbacks, recovery, and replay."),
        ("financial-evaluation", "Publish the fixed financial-evaluation contract."),
        ("status", "Report Phase 10 technical readiness and blocked runtime authority."),
        ("freeze", "Freeze the Phase 10 technical foundation without runtime authority."),
    ):
        phase10_subparsers.add_parser(command, help=help_text)

    phase11_2 = ibkr_subparsers.add_parser(
        "phase11-2",
        help="Read-only provider capability and point-in-time data foundation.",
    )
    phase11_2_subparsers = phase11_2.add_subparsers(dest="phase11_2_command", required=True)
    for command, help_text in (
        ("provider-audit", "Publish the typed provider registry and frozen dependency baseline."),
        ("capability-probe", "Run bounded read-only provider capability probes."),
        ("universe-build", "Build the active and delisted research universe foundation."),
        ("price-audit", "Audit historical price and corporate-action coverage."),
        ("fundamentals-audit", "Audit statement and fundamental coverage."),
        ("earnings-audit", "Audit earnings timestamps and historical consensus capability."),
        ("news-audit", "Audit news history, timestamps, and revisions."),
        ("sec-audit", "Audit SEC ticker mapping, submissions, acceptance times, and XBRL facts."),
        ("shariah-pit-audit", "Audit frozen Shariah methodology and historical reconstruction capability."),
        ("data-quality", "Audit PIT data quality, survivorship, and source conflicts."),
        ("status", "Report Phase 11.2 technical and data readiness."),
        ("freeze", "Freeze the Phase 11.2 provider and PIT foundation."),
    ):
        phase11_2_subparsers.add_parser(command, help=help_text)

    phase11_3 = ibkr_subparsers.add_parser(
        "phase11-3",
        help="Historical coverage and causal SEC/news attribution research.",
    )
    phase11_3_subparsers = phase11_3.add_subparsers(dest="phase11_3_command", required=True)
    for command in (
        "datascraper-inventory", "datascraper-import", "datascraper-coverage", "rss-audit",
        "price-backfill", "universe-history", "classification-audit", "sec-events",
        "fundamental-actuals", "shariah-history", "news-backfill", "movers-build",
        "attribution", "event-windows", "data-quality", "status", "freeze",
    ):
        phase11_3_subparsers.add_parser(command)

    shadow = subparsers.add_parser("shadow", help="Offline strategy-agnostic shadow infrastructure.")
    shadow_subparsers = shadow.add_subparsers(dest="shadow_command", required=True)
    phase8_2 = shadow_subparsers.add_parser(
        "phase8-2",
        help="Phase 8.2 strategy-agnostic shadow infrastructure.",
    )
    phase8_2_subparsers = phase8_2.add_subparsers(dest="phase8_2_command", required=True)
    phase8_2_subparsers.add_parser("schema", help="Print and write the Phase 8.2 schema artifact.")
    phase8_2_subparsers.add_parser("init-ledger", help="Initialize the offline shadow ledger.")
    phase8_2_subparsers.add_parser("register-fixtures", help="Register disabled and synthetic fixture strategy contracts.")
    phase8_2_subparsers.add_parser("simulate", help="Run deterministic offline Phase 8.2 shadow fixture scenarios.")
    phase8_2_subparsers.add_parser("replay", help="Replay the append-only shadow ledger deterministically.")
    phase8_2_subparsers.add_parser("audit-ledger", help="Audit the Phase 8.2 shadow ledger and security state.")
    phase8_2_subparsers.add_parser("activation-audit", help="Audit strategy activation gates without activating a strategy.")
    phase8_2_subparsers.add_parser("status", help="Report Phase 8.2 readiness from current artifacts.")
    phase8_2_subparsers.add_parser("freeze", help="Write the Phase 8.2 freeze artifact.")

    screener = subparsers.add_parser(
        "screener",
        help="Daily point-in-time research-only asset screener.",
    )
    screener_subparsers = screener.add_subparsers(dest="screener_command", required=True)
    screener_run_parser = screener_subparsers.add_parser(
        "run",
        help="Run and append one completed-session screening.",
    )
    screener_run_parser.add_argument("--as-of", help="Optional historical session date (YYYY-MM-DD).")
    screener_preview_parser = screener_subparsers.add_parser(
        "preview",
        help="Re-evaluate a completed session without mutating append-only evidence.",
    )
    screener_preview_parser.add_argument(
        "--as-of",
        help="Optional completed session date (YYYY-MM-DD).",
    )
    screener_subparsers.add_parser("status", help="Report screener configuration and latest run.")
    screener_report_parser = screener_subparsers.add_parser(
        "report",
        help="Report the latest or selected registered screening.",
    )
    screener_report_parser.add_argument("--as-of", help="Optional registered screening date (YYYY-MM-DD).")
    screener_history_parser = screener_subparsers.add_parser(
        "history",
        help="Show append-only history for one symbol.",
    )
    screener_history_parser.add_argument("--symbol", required=True)
    screener_top_parser = screener_subparsers.add_parser(
        "top",
        help="Show the latest eligible candidates.",
    )
    screener_top_parser.add_argument("--limit", type=int, default=20)
    screener_asset_parser = screener_subparsers.add_parser(
        "asset",
        help="Show the latest registered screening history for one asset.",
    )
    screener_asset_parser.add_argument("--symbol", required=True)
    screener_subparsers.add_parser(
        "export", help="Export the latest research-only screener report."
    )

    macro = subparsers.add_parser(
        "macro",
        help="Deterministic point-in-time macro engine and analyst.",
    )
    macro_subparsers = macro.add_subparsers(
        dest="macro_command",
        required=True,
    )
    macro_collect_parser = macro_subparsers.add_parser(
        "collect",
        help="Collect configured macro and cross-asset sources.",
    )
    macro_collect_parser.add_argument("--start")
    macro_collect_parser.add_argument("--end")
    macro_subparsers.add_parser("update", help="Collect, score and report.")
    macro_subparsers.add_parser("validate", help="Validate PIT macro storage.")
    macro_subparsers.add_parser("status", help="Report macro-engine status.")
    macro_subparsers.add_parser(
        "readiness",
        help="Audit bounded read-only live update readiness.",
    )
    macro_score_parser = macro_subparsers.add_parser(
        "score",
        help="Compute causally available macro scores.",
    )
    macro_score_parser.add_argument("--as-of")
    macro_regime_parser = macro_subparsers.add_parser(
        "regime",
        help="Compute the deterministic macro regime.",
    )
    macro_regime_parser.add_argument("--as-of")
    macro_history_parser = macro_subparsers.add_parser(
        "history",
        help="Build or report PIT macro-regime history.",
    )
    macro_history_parser.add_argument(
        "--rebuild",
        action="store_true",
    )
    macro_subparsers.add_parser("events", help="Report structured macro events.")
    macro_report_parser = macro_subparsers.add_parser(
        "report",
        help="Write a deterministic analyst report.",
    )
    macro_report_parser.add_argument(
        "--period",
        choices=["daily", "weekly", "monthly"],
        required=True,
    )
    macro_subparsers.add_parser("explain", help="Explain current macro state.")
    macro_compare_parser = macro_subparsers.add_parser(
        "compare",
        help="Compare two point-in-time macro snapshots.",
    )
    macro_compare_parser.add_argument("--date-a", required=True)
    macro_compare_parser.add_argument("--date-b", required=True)
    macro_subparsers.add_parser(
        "sector-impact",
        help="Report relative sector, region and asset implications.",
    )
    macro_strategy_parser = macro_subparsers.add_parser(
        "strategy-impact",
        help="Report macro context for one registered strategy.",
    )
    macro_strategy_parser.add_argument("--strategy-id", required=True)
    macro_subparsers.add_parser("audit", help="Audit macro safety and PIT state.")
    macro_subparsers.add_parser(
        "conflicts",
        help="Audit quarantined provider-conflict resolution.",
    )
    macro_subparsers.add_parser("freeze", help="Freeze Macro V1 technical state.")

    market = subparsers.add_parser("market", help="Offline market-session commands from local contract cache.")
    market_subparsers = market.add_subparsers(dest="market_command", required=True)
    market_context = market_subparsers.add_parser(
        "context",
        help="Context-only GEX and orderflow intelligence commands.",
    )
    market_context_subparsers = market_context.add_subparsers(
        dest="market_context_command",
        required=True,
    )
    market_context_subparsers.add_parser(
        "schema",
        help="Publish the GEX and orderflow data contract.",
    )
    market_context_subparsers.add_parser(
        "audit",
        help="Audit current option, tick, quote and orderbook sources.",
    )
    market_context_build = market_context_subparsers.add_parser(
        "build",
        help="Build current context and low-confidence bar-flow fallbacks.",
    )
    market_context_build.add_argument(
        "--symbols",
        default=",".join(DEFAULT_CONTEXT_SYMBOLS),
    )
    market_context_build.add_argument(
        "--max-expirations",
        type=int,
        default=4,
    )
    market_context_build.add_argument(
        "--no-network",
        action="store_true",
        help="Use only existing read-only local context artifacts.",
    )
    market_context_subparsers.add_parser(
        "status",
        help="Report current market-context readiness and limitations.",
    )
    cot_update = market_context_subparsers.add_parser(
        "cot-update",
        help="Collect official weekly CFTC COT asset-class context.",
    )
    cot_update.add_argument("--start", default="2018-01-01")
    cot_update.add_argument(
        "--no-network",
        action="store_true",
        help="Rebuild from the latest immutable local COT snapshot.",
    )
    market_context_subparsers.add_parser(
        "cot-status",
        help="Report current CFTC COT context and publication lag.",
    )
    market_context_subparsers.add_parser(
        "transmission",
        help="Build asset-specific macro, COT, GEX and flow context.",
    )
    entry_observe = market_context_subparsers.add_parser(
        "observe",
        help="Observe the setup shortlist with tape and depth authority NONE.",
    )
    entry_observe.add_argument("--max-symbols", type=int, default=20)
    entry_observe.add_argument("--depth-symbols", type=int, default=5)
    market_context_subparsers.add_parser(
        "observer-status",
        help="Report hierarchical entry-observer readiness.",
    )
    market_context_subparsers.add_parser(
        "settle-episodes",
        help="Close eligible hypothetical episodes without broker authority.",
    )
    market_context_subparsers.add_parser(
        "episode-status",
        help="Report terminal forward-episode completeness.",
    )
    realtime_equity = market_context_subparsers.add_parser(
        "collect-realtime",
        help="Collect bounded read-only IBKR Level I, tape and depth context.",
    )
    realtime_equity.add_argument("--duration-seconds", type=float, default=15.0)
    realtime_equity.add_argument("--max-symbols", type=int, default=10)
    realtime_equity.add_argument("--depth-symbols", type=int, default=5)
    realtime_equity.add_argument("--depth-levels", type=int, default=5)
    market_status = market_subparsers.add_parser(
        "status",
        help="Evaluate trading/liquid session state for one cached conId.",
    )
    market_status.add_argument("--con-id", type=int, required=True)
    market_status.add_argument("--at", help="Timezone-aware ISO datetime; defaults to now.")
    next_open = market_subparsers.add_parser(
        "next-open",
        help="Find the next known trading window for one cached conId.",
    )
    next_open.add_argument("--con-id", type=int, required=True)
    next_open.add_argument("--at", help="Timezone-aware ISO datetime; defaults to now.")
    sessions = market_subparsers.add_parser(
        "sessions",
        help="Phase 3 market-session commands from the local contract cache.",
    )
    sessions.add_argument("--con-id", type=int)
    sessions.add_argument("--date", help="Legacy session date in YYYY-MM-DD format.")
    sessions_subparsers = sessions.add_subparsers(dest="sessions_command")
    sessions_subparsers.add_parser("schema", help="Print the canonical Phase 3 market-session schema.")
    sessions_resolve = sessions_subparsers.add_parser(
        "resolve",
        help="Resolve one canonical session record for a cached conId and session date.",
    )
    sessions_resolve.add_argument("--con-id", type=int, required=True)
    sessions_resolve.add_argument("--date", required=True, help="Session date in YYYY-MM-DD format.")
    sessions_status = sessions_subparsers.add_parser(
        "status",
        help="Evaluate trading/liquid session state for one cached conId.",
    )
    sessions_status.add_argument("--con-id", type=int, required=True)
    sessions_status.add_argument("--at", help="Timezone-aware ISO datetime; defaults to now.")
    sessions_next_open = sessions_subparsers.add_parser(
        "next-open",
        help="Find the next known trading window for one cached conId.",
    )
    sessions_next_open.add_argument("--con-id", type=int, required=True)
    sessions_next_open.add_argument("--at", help="Timezone-aware ISO datetime; defaults to now.")
    sessions_range = sessions_subparsers.add_parser(
        "range",
        help="Resolve canonical sessions for one cached conId over a date range.",
    )
    sessions_range.add_argument("--con-id", type=int, required=True)
    sessions_range.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    sessions_range.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    sessions_subparsers.add_parser(
        "validate-cache",
        help="Build and validate data/sessions from the local Phase 2 contract cache.",
    )

    data = subparsers.add_parser("data", help="Offline data-layer schema and safety commands.")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    data_sources = data_subparsers.add_parser(
        "sources", help="Inventory configured and cached data providers."
    )
    data_sources_subparsers = data_sources.add_subparsers(
        dest="data_sources_command", required=True
    )
    data_sources_subparsers.add_parser("status")
    data_sources_subparsers.add_parser(
        "readiness",
        help="Publish cross-layer data readiness and explicit acquisition gaps.",
    )
    bars = data_subparsers.add_parser("bars", help="Historical bar cache planning commands.")
    bars_subparsers = bars.add_subparsers(dest="bars_command", required=True)
    bars_subparsers.add_parser("schema", help="Print the offline historical bar cache schema.")
    bars_subparsers.add_parser("status", help="Report historical bar data authority and cache layout.")
    bars_subparsers.add_parser("init-cache", help="Create the local historical bar cache manifest.")
    bars_subparsers.add_parser("validate-cache", help="Validate local historical bar cache files.")
    bars_subparsers.add_parser("request-policy", help="Print the offline historical bar request queue policy.")
    bars_collect = bars_subparsers.add_parser(
        "collect",
        help="Collect Phase 4 read-only IBKR daily STK bars into the local cache.",
    )
    bars_collect.add_argument("--con-id", type=int, required=True)
    bars_collect.add_argument("--interval", choices=["1d"], required=True)
    bars_collect.add_argument("--data-type", choices=["TRADES"], required=True)
    bars_collect.add_argument("--start", required=True, help="Start date in YYYY-MM-DD format.")
    bars_collect.add_argument("--end", required=True, help="End date in YYYY-MM-DD format.")
    multitimeframe = data_subparsers.add_parser(
        "multitimeframe",
        help="Provider-aware research OHLCV for intraday, daily, weekly and monthly intervals.",
    )
    multitimeframe_subparsers = multitimeframe.add_subparsers(dest="multitimeframe_command", required=True)
    multitimeframe_subparsers.add_parser("schema", help="Print the canonical multi-timeframe data contract.")
    multitimeframe_subparsers.add_parser("inventory", help="Inventory usable providers and local caches.")
    multitimeframe_subparsers.add_parser("import-local", help="Import all qualified local Stocks and datascraper caches read-only.")
    multitimeframe_subparsers.add_parser("normalize", help="Validate canonical normalized private partitions.")
    multitimeframe_subparsers.add_parser("coverage", help="Publish Phase 11.6 coverage and readiness artifacts.")
    multitimeframe_collect = multitimeframe_subparsers.add_parser(
        "collect",
        help="Run a bounded read-only provider and local-cache collection.",
    )
    multitimeframe_collect.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    multitimeframe_collect.add_argument("--intervals", default=",".join(CANONICAL_INTERVALS))
    multitimeframe_collect.add_argument(
        "--providers",
        default="all",
        help="Comma-separated: all, local, datascraper, yfinance, eodhd, ibkr.",
    )
    multitimeframe_collect.add_argument("--start", default=None)
    multitimeframe_collect.add_argument("--end", default=None)
    multitimeframe_collect.add_argument("--lookback-days", type=int, default=60)
    multitimeframe_subparsers.add_parser("validate-cache", help="Validate all canonical private bar partitions.")
    multitimeframe_subparsers.add_parser("audit", help="Compare overlapping provider partitions without blending them.")
    multitimeframe_subparsers.add_parser("status", help="Report multi-timeframe research data readiness.")
    corporate_actions = data_subparsers.add_parser(
        "corporate-actions",
        help="Phase 5 corporate-action and distribution ledger commands.",
    )
    corporate_actions_subparsers = corporate_actions.add_subparsers(
        dest="corporate_actions_command",
        required=True,
    )
    corporate_actions_subparsers.add_parser("schema", help="Print the corporate-action schema.")
    ca_collect = corporate_actions_subparsers.add_parser(
        "collect",
        help="Collect read-only corporate actions for the research universe.",
    )
    ca_collect.add_argument("--universe", choices=["research"], required=True)
    ca_collect.add_argument("--start", default="2015-01-01")
    ca_collect.add_argument("--end", default=date.today().isoformat())
    corporate_actions_subparsers.add_parser("validate-cache", help="Validate corporate-action cache.")
    corporate_actions_subparsers.add_parser("status", help="Report corporate-action status.")
    fx = data_subparsers.add_parser("fx", help="Phase 5 FX dataset commands.")
    fx_subparsers = fx.add_subparsers(dest="fx_command", required=True)
    fx_subparsers.add_parser("schema", help="Print the FX schema.")
    fx_collect = fx_subparsers.add_parser("collect", help="Collect read-only FX for the research universe.")
    fx_collect.add_argument("--base", default="EUR")
    fx_collect.add_argument("--universe", choices=["research"], required=True)
    fx_collect.add_argument("--start", default="2015-01-01")
    fx_collect.add_argument("--end", default=date.today().isoformat())
    fx_subparsers.add_parser("validate-cache", help="Validate FX cache.")
    fx_subparsers.add_parser("status", help="Report FX status.")
    total_returns = data_subparsers.add_parser("total-returns", help="Phase 5 EUR total-return commands.")
    total_returns_subparsers = total_returns.add_subparsers(dest="total_returns_command", required=True)
    total_returns_subparsers.add_parser("schema", help="Print the total-return schema.")
    tr_build = total_returns_subparsers.add_parser("build", help="Build EUR total-return datasets.")
    tr_build.add_argument("--universe", choices=["research"], required=True)
    tr_build.add_argument("--base-currency", default="EUR")
    tr_build.add_argument("--interval", choices=["1d"], default="1d")
    total_returns_subparsers.add_parser("validate-cache", help="Validate total-return cache.")
    total_returns_subparsers.add_parser("status", help="Report total-return status.")

    research = subparsers.add_parser("research", help="Offline research-readiness commands.")
    research_subparsers = research.add_subparsers(dest="research_command", required=True)
    universe = research_subparsers.add_parser("universe", help="Small research instrument manifest commands.")
    universe_subparsers = universe.add_subparsers(dest="universe_command", required=True)
    universe_subparsers.add_parser("schema", help="Print the instrument manifest schema.")
    universe_subparsers.add_parser("init-manifest", help="Write the default unvalidated research universe manifest.")
    universe_subparsers.add_parser("validate-manifest", help="Validate the local research universe manifest.")
    universe_subparsers.add_parser("status", help="Report local research universe manifest readiness.")
    phase6 = research_subparsers.add_parser("phase6", help="Offline Phase 6 baselines and strategy evidence.")
    phase6_subparsers = phase6.add_subparsers(dest="phase6_command", required=True)
    phase6_subparsers.add_parser("schema", help="Print the Phase 6 metrics and evidence contract.")
    phase6_subparsers.add_parser("dataset-audit", help="Audit the local EUR total-return research dataset.")
    phase6_subparsers.add_parser("benchmarks", help="Run the five simple baseline families.")
    phase6_subparsers.add_parser("strategy-grid", help="Run the fixed 108-configuration strategy matrix.")
    phase6_subparsers.add_parser("walk-forward", help="Run walk-forward evidence on fixed configurations.")
    phase6_subparsers.add_parser("run", help="Run the full Phase 6 offline research pipeline.")
    phase6_subparsers.add_parser("status", help="Report the latest Phase 6 evidence status.")
    phase6_subparsers.add_parser("freeze", help="Write the Phase 6 freeze artifact.")
    phase6_1 = research_subparsers.add_parser("phase6-1", help="Offline Phase 6.1 robustness and failure attribution.")
    phase6_1_subparsers = phase6_1.add_subparsers(dest="phase6_1_command", required=True)
    phase6_1_subparsers.add_parser("schema", help="Print the Phase 6.1 diagnostics contract.")
    phase6_1_subparsers.add_parser("run", help="Run full Phase 6.1 diagnostics.")
    phase6_1_subparsers.add_parser("status", help="Report the latest Phase 6.1 diagnostics status.")
    phase6_1_subparsers.add_parser("freeze", help="Write the Phase 6.1 freeze artifact.")
    phase6_2 = research_subparsers.add_parser("phase6-2", help="Offline Phase 6.2 sample sufficiency and forward OOS.")
    phase6_2_subparsers = phase6_2.add_subparsers(dest="phase6_2_command", required=True)
    phase6_2_subparsers.add_parser("schema", help="Print the Phase 6.2 sample sufficiency contract.")
    phase6_2_subparsers.add_parser("run", help="Run Phase 6.2 sample sufficiency and shadow diagnostics.")
    phase6_2_subparsers.add_parser("status", help="Report the latest Phase 6.2 status.")
    phase6_2_subparsers.add_parser("freeze", help="Write the Phase 6.2 freeze artifact.")
    phase6_3 = research_subparsers.add_parser(
        "phase6-3",
        help="Offline Phase 6.3 benchmark champion and incremental-alpha evidence.",
    )
    phase6_3_subparsers = phase6_3.add_subparsers(dest="phase6_3_command", required=True)
    phase6_3_subparsers.add_parser("schema", help="Print the Phase 6.3 benchmark champion contract.")
    phase6_3_subparsers.add_parser("run", help="Run Phase 6.3 benchmark champion and alpha diagnostics.")
    phase6_3_subparsers.add_parser("status", help="Report the latest Phase 6.3 status.")
    phase6_3_subparsers.add_parser("freeze", help="Write the Phase 6.3 freeze artifact.")
    phase6_4 = research_subparsers.add_parser(
        "phase6-4",
        help="Offline Phase 6.4 preregistered mechanism research and forward selection.",
    )
    phase6_4_subparsers = phase6_4.add_subparsers(dest="phase6_4_command", required=True)
    phase6_4_subparsers.add_parser("schema", help="Print the Phase 6.4 preregistered research contract.")
    phase6_4_subparsers.add_parser("preregister", help="Lock the Phase 6.4 hypothesis registry.")
    phase6_4_subparsers.add_parser("run", help="Run Phase 6.4 preregistered mechanism research.")
    phase6_4_subparsers.add_parser("status", help="Report the latest Phase 6.4 status.")
    phase6_4_subparsers.add_parser("freeze", help="Write the Phase 6.4 freeze artifact.")
    critical_trading = research_subparsers.add_parser(
        "critical-trading",
        help="Reproduce transcript strategies on local ETF and stock data.",
    )
    critical_trading_subparsers = critical_trading.add_subparsers(
        dest="critical_trading_command",
        required=True,
    )
    critical_trading_subparsers.add_parser("schema", help="Print the preregistered transcript rules.")
    critical_run = critical_trading_subparsers.add_parser("run", help="Run all offline transcript backtests.")
    critical_run.add_argument("--stock-limit", type=int, default=None)
    critical_run.add_argument("--include-yfinance", action="store_true")
    critical_collect = critical_trading_subparsers.add_parser(
        "yfinance-collect",
        help="Collect adjusted daily Yahoo Finance data into the local research cache.",
    )
    critical_collect.add_argument("--start", default="2000-01-01")
    critical_collect.add_argument("--end", default=None)
    critical_trading_subparsers.add_parser(
        "perfect",
        help="Run preregistered OOS robustness and execution-sensitivity analysis.",
    )
    phase11_1 = research_subparsers.add_parser(
        "phase11-1",
        help="Offline Phase 11.1 orthogonal PIT alpha data and strategy foundation.",
    )
    phase11_1_subparsers = phase11_1.add_subparsers(dest="phase11_1_command", required=True)
    phase11_1_subparsers.add_parser("schema", help="Print and write the Phase 11.1 alpha foundation schema.")
    phase11_1_subparsers.add_parser("preregister", help="Lock the Phase 11.1 PIT alpha foundation config.")
    phase11_1_subparsers.add_parser("run", help="Run Phase 11.1 offline foundation fixture audits.")
    phase11_1_subparsers.add_parser("status", help="Report the latest Phase 11.1 foundation status.")
    phase11_1_subparsers.add_parser("freeze", help="Write the Phase 11.1 freeze artifact.")
    rsi_pit = research_subparsers.add_parser(
        "rsi-pit",
        help="Offline Phase 11.4 point-in-time RSI mean-reversion validation.",
    )
    rsi_pit_subparsers = rsi_pit.add_subparsers(dest="rsi_pit_command", required=True)
    for command in PHASE11_4_COMMANDS:
        rsi_pit_subparsers.add_parser(command, help=f"Run Phase 11.4 {command}.")
    rsi_acquire = rsi_pit_subparsers.add_parser(
        "acquire-pit-data",
        help="Resume private read-only PIT common-stock price and split acquisition.",
    )
    rsi_acquire.add_argument("--max-symbols", type=int, default=None)
    rsi_acquire.add_argument("--workers", type=int, default=12)
    rsi_acquire.add_argument("--requests-per-second", type=float, default=10.0)
    rsi_pit_subparsers.add_parser(
        "compact-pit-data",
        help="Compact valid private PIT price partitions for offline research.",
    )
    phase11_6 = research_subparsers.add_parser(
        "phase11-6",
        help="Offline multi-timeframe nested walk-forward and combination architecture research.",
    )
    phase11_6_subparsers = phase11_6.add_subparsers(dest="phase11_6_command", required=True)
    for command in ("schema", "data-audit", "walk-forward", "cohorts", "combine", "audit", "status", "run"):
        subcommand_parser = phase11_6_subparsers.add_parser(command)
        if command in {"walk-forward", "cohorts", "run"}:
            subcommand_parser.add_argument("--max-walk-forward-identities", type=int, default=50)
        if command in {"combine", "run"}:
            subcommand_parser.add_argument("--max-combination-identities", type=int, default=100)
    phase11_7 = research_subparsers.add_parser(
        "phase11-7",
        help="Offline bounded financial-finalist campaign with corrected data and ledger gates.",
    )
    phase11_7_subparsers = phase11_7.add_subparsers(
        dest="phase11_7_command", required=True
    )
    phase11_7_subparsers.add_parser("schema")
    phase11_7_run_parser = phase11_7_subparsers.add_parser("run")
    phase11_7_run_parser.add_argument("--max-identities", type=int, default=500)
    phase11_7_run_parser.add_argument("--bootstrap-runs", type=int, default=5000)
    phase11_7_rotation_parser = phase11_7_subparsers.add_parser("rotation")
    phase11_7_rotation_parser.add_argument("--max-identities", type=int, default=500)
    phase11_7_rotation_parser.add_argument("--bootstrap-runs", type=int, default=5000)
    phase11_7_subparsers.add_parser("status")
    phase11_8 = research_subparsers.add_parser(
        "phase11-8",
        help="Realistic multi-strategy nested walk-forward and forward holdout.",
    )
    phase11_8_subparsers = phase11_8.add_subparsers(
        dest="phase11_8_command", required=True
    )
    for command in (
        "schema",
        "data-coverage",
        "portfolio-audit",
        "finalize",
        "status",
    ):
        phase11_8_subparsers.add_parser(command)
    phase11_8_run_parser = phase11_8_subparsers.add_parser("run")
    phase11_8_run_parser.add_argument(
        "--max-stock-identities", type=int, default=30
    )
    phase11_9 = research_subparsers.add_parser(
        "phase11-9",
        help="Accelerated 1h-and-longer strategy and ensemble discovery.",
    )
    phase11_9_subparsers = phase11_9.add_subparsers(
        dest="phase11_9_command", required=True
    )
    for command in ("schema", "run", "diagnose", "watchlist", "status"):
        phase11_9_subparsers.add_parser(command)
    phase11_10 = research_subparsers.add_parser(
        "phase11-10",
        help="Causal higher-timeframe gate and lower-timeframe swing research.",
    )
    phase11_10_subparsers = phase11_10.add_subparsers(
        dest="phase11_10_command",
        required=True,
    )
    for command in (
        "schema",
        "reclassify",
        "pit-observe",
        "watchlist",
        "top20",
        "status",
        "qualification-audit",
        "qualification-freeze",
    ):
        phase11_10_subparsers.add_parser(command)
    phase11_10_run = phase11_10_subparsers.add_parser("run")
    phase11_10_run.add_argument(
        "--historical-cutoff",
        help="Pin all research inputs to this UTC date or timestamp.",
    )
    phase11_12 = research_subparsers.add_parser(
        "phase11-12",
        help=(
            "Register and evaluate the bounded strategy-DNA catalog "
            "across stocks, ETFs and commodity proxies."
        ),
    )
    phase11_12_subparsers = phase11_12.add_subparsers(
        dest="phase11_12_command",
        required=True,
    )
    phase11_12_subparsers.add_parser("schema")
    phase11_12_run = phase11_12_subparsers.add_parser("run")
    phase11_12_run.add_argument(
        "--max-strategies",
        type=int,
        default=None,
    )
    phase11_12_run.add_argument(
        "--complexity",
        type=int,
        choices=range(1, 6),
        default=None,
    )
    phase11_12_run.add_argument(
        "--pending-only",
        action="store_true",
        help="Process the next bounded incomplete DNA cohort.",
    )
    phase11_12_observe_parser = phase11_12_subparsers.add_parser("observe")
    phase11_12_observe_parser.add_argument(
        "--max-strategies",
        type=int,
        default=12,
    )
    phase11_12_subparsers.add_parser("forward-status")
    phase11_12_subparsers.add_parser("status")
    phase11_13 = research_subparsers.add_parser(
        "phase11-13",
        help="Qualify the five frozen fast-track strategy DNA records.",
    )
    phase11_13_subparsers = phase11_13.add_subparsers(
        dest="phase11_13_command",
        required=True,
    )
    for command in ("schema", "run", "observe", "status"):
        phase11_13_subparsers.add_parser(command)
    phase11_14 = research_subparsers.add_parser(
        "phase11-14",
        help=(
            "Nested-qualify diversified Phase 11.12 cost-stress survivors."
        ),
    )
    phase11_14_subparsers = phase11_14.add_subparsers(
        dest="phase11_14_command",
        required=True,
    )
    phase11_14_subparsers.add_parser("schema")
    phase11_14_run = phase11_14_subparsers.add_parser("run")
    phase11_14_run.add_argument(
        "--max-candidates",
        type=int,
        default=16,
    )
    phase11_14_subparsers.add_parser("observe")
    phase11_14_subparsers.add_parser("status")
    phase11_15 = research_subparsers.add_parser(
        "phase11-15",
        help=(
            "Nested OOS research for low-confidence bar-flow overlays; "
            "GEX remains forward-only."
        ),
    )
    phase11_15_subparsers = phase11_15.add_subparsers(
        dest="phase11_15_command",
        required=True,
    )
    phase11_15_subparsers.add_parser("schema")
    phase11_15_run = phase11_15_subparsers.add_parser("run")
    phase11_15_run.add_argument(
        "--max-architectures",
        type=int,
        default=20,
    )
    phase11_15_subparsers.add_parser("status")

    regimes = subparsers.add_parser(
        "regimes",
        help="Causal HMM regime research and read-only risk context.",
    )
    regimes_subparsers = regimes.add_subparsers(
        dest="regimes_command",
        required=True,
    )
    for command in (
        "schema",
        "fit",
        "walk-forward",
        "current",
        "audit",
        "status",
    ):
        regimes_subparsers.add_parser(command)

    research_subparsers.add_parser(
        "components", help="Publish the canonical transparent component registry."
    )
    research_subparsers.add_parser(
        "taxonomy",
        help="Audit canonical indicator and strategy taxonomy coverage.",
    )
    registry = research_subparsers.add_parser(
        "registry",
        help="Publish and inspect canonical research registry artifacts.",
    )
    registry_subparsers = registry.add_subparsers(
        dest="registry_command",
        required=True,
    )
    for command in ("publish", "coverage", "status", "roles"):
        registry_subparsers.add_parser(command)
    active_swing = research_subparsers.add_parser(
        "active-swing",
        help="Run observation-only active-swing Sprints 3 through 6.",
    )
    active_swing_subparsers = active_swing.add_subparsers(
        dest="active_swing_command",
        required=True,
    )
    for command in (
        "shortlist-data",
        "rejected-shadow",
        "gate-attribution",
        "evidence-throughput",
        "entry-filter-experiment",
        "leaderboards",
        "train-ml",
        "refresh",
        "run",
        "status",
    ):
        active_swing_subparsers.add_parser(command)
    macro_pairs = research_subparsers.add_parser(
        "macro-pairs",
        help="Run strict no-macro versus PIT-macro paired validation.",
    )
    macro_pairs.add_argument("--max-identities", type=int, default=500)
    generate = research_subparsers.add_parser(
        "generate", help="Generate deterministic bounded strategy specifications."
    )
    generate.add_argument("--budget", type=int, default=100)
    generate.add_argument(
        "--family",
        choices=[
            "quality_momentum",
            "trend_pullback",
            "etf_rotation",
            "volatility_contraction_breakout",
            "commodity_etf_trend",
        ],
        default=None,
    )
    generate.add_argument("--seed", type=int, default=20260726)
    generate.add_argument(
        "--complexity",
        type=int,
        choices=range(1, 6),
        default=None,
        help=(
            "Register the complete implemented bulk DNA layer for one "
            "through five signal blocks without running backtests."
        ),
    )
    generate.add_argument(
        "--resume",
        action="store_true",
        help="Use content-addressed idempotent registration.",
    )
    smoke = research_subparsers.add_parser(
        "smoke", help="Run engine-correctness fixtures; never financial evidence."
    )
    smoke.add_argument("--family", default=None)
    campaign = research_subparsers.add_parser(
        "campaign", help="Run a bounded point-in-time research campaign."
    )
    campaign.add_argument(
        "--family",
        choices=[
            "quality_momentum",
            "trend_pullback",
            "etf_rotation",
            "volatility_contraction_breakout",
            "commodity_etf_trend",
        ],
        required=True,
    )
    campaign.add_argument("--max-trials", type=int, default=40)
    research_subparsers.add_parser("daily", help="Run bounded daily research maintenance.")
    weekly = research_subparsers.add_parser("weekly", help="Run bounded weekly campaigns.")
    weekly.add_argument("--max-trials", type=int, default=40)
    research_subparsers.add_parser("monthly", help="Publish monthly family review.")
    research_subparsers.add_parser("candidates", help="List promoted research candidates.")
    strategy_detail = research_subparsers.add_parser("strategy", help="Show one strategy.")
    strategy_detail.add_argument("--id", dest="strategy_id", required=True)
    compare = research_subparsers.add_parser("compare", help="Compare registered strategies.")
    compare.add_argument("--ids", dest="strategy_ids", nargs="+", required=True)
    combine = research_subparsers.add_parser(
        "combine", help="Register a transparent frozen-weight ensemble."
    )
    combine.add_argument("--ids", dest="strategy_ids", nargs="+", required=True)
    combine.add_argument(
        "--mode",
        choices=[
            "confirmation",
            "majority",
            "weighted",
            "unanimous",
            "hierarchical",
            "sleeves",
        ],
        default="majority",
    )
    research_subparsers.add_parser("leaderboard", help="Rank complete trials.")
    research_subparsers.add_parser("rejected", help="List rejected strategies.")
    research_subparsers.add_parser(
        "recover-survivors",
        help="Reclassify historical real-data results with the usage-specific promotion ladder.",
    )
    research_subparsers.add_parser("audit", help="Audit the canonical research lifecycle.")
    research_subparsers.add_parser("autopilot-status", help="Report research autopilot status.")
    research_subparsers.add_parser(
        "freeze", help="Freeze the technical research-autopilot manifest."
    )

    forward = subparsers.add_parser(
        "forward", help="Frozen strategy observer with authority NONE."
    )
    forward_subparsers = forward.add_subparsers(dest="forward_command", required=True)
    forward_register_parser = forward_subparsers.add_parser(
        "register", help="Register an eligible frozen observer candidate."
    )
    forward_register_parser.add_argument(
        "--strategy-id", dest="strategy_id", required=True
    )
    forward_subparsers.add_parser("run", help="Append today's observations.")
    forward_subparsers.add_parser("status", help="Report observer registrations.")
    forward_subparsers.add_parser("report", help="Report observer registrations.")

    capital = subparsers.add_parser(
        "capital",
        help="Versioned capital scaling with manual promotion and automatic demotion.",
    )
    capital_subparsers = capital.add_subparsers(
        dest="capital_command", required=True
    )
    for command in ("status", "capacity", "recommend-level"):
        capital_subparsers.add_parser(command)
    capital_daily_target = capital_subparsers.add_parser("daily-target")
    capital_daily_target.add_argument(
        "--account-equity-eur", type=Decimal, required=True
    )
    capital_daily_target.add_argument(
        "--net-daily-pnl-eur", type=Decimal, required=True
    )
    capital_daily_target.add_argument(
        "--enforce",
        action="store_true",
        help="Apply the target as a risk-reducing BUY throttle.",
    )
    capital_promote = capital_subparsers.add_parser("promote")
    capital_promote.add_argument("--level", type=int, required=True)
    capital_promote.add_argument("--approval", required=True)
    capital_demote = capital_subparsers.add_parser("demote")
    capital_demote.add_argument("--level", type=int, required=True)
    capital_demote.add_argument("--reason", required=True)

    portfolio = subparsers.add_parser(
        "portfolio", help="Point-in-time portfolio research using the canonical engine."
    )
    portfolio_subparsers = portfolio.add_subparsers(
        dest="portfolio_command", required=True
    )
    portfolio_backtest_parser = portfolio_subparsers.add_parser("backtest")
    portfolio_backtest_parser.add_argument("--strategy-id", required=True)
    portfolio_stress_parser = portfolio_subparsers.add_parser("stress")
    portfolio_stress_parser.add_argument("--strategy-id", required=True)
    for command in (
        "status",
        "plan",
        "opportunities",
        "actions",
        "sizing-audit",
        "lifecycle-audit",
        "risk",
        "dynamic-risk",
        "position-management",
        "confluence",
        "exposures",
        "capacity",
        "rebalance-preview",
        "coverage",
        "stage0",
        "intelligence",
        "attribution",
        "normalized-opportunities",
        "overlap",
        "targets",
        "p1",
    ):
        portfolio_subparsers.add_parser(command)

    risk = subparsers.add_parser(
        "risk", help="Canonical portfolio risk status."
    )
    risk_subparsers = risk.add_subparsers(
        dest="risk_command", required=True
    )
    risk_subparsers.add_parser("status")

    universe = subparsers.add_parser(
        "universe", help="Broad point-in-time discovery universe."
    )
    universe_subparsers = universe.add_subparsers(
        dest="universe_command", required=True
    )
    for command in (
        "refresh",
        "status",
        "coverage",
        "sectors",
        "industries",
        "regions",
        "stocks",
        "etfs",
        "commodities",
    ):
        universe_subparsers.add_parser(command)

    for dimension in ("sectors", "industries", "regions"):
        ranking = subparsers.add_parser(
            dimension, help=f"Rank current {dimension}."
        )
        ranking_subparsers = ranking.add_subparsers(
            dest=f"{dimension}_command", required=True
        )
        ranking_subparsers.add_parser("rank")

    analysis = subparsers.add_parser(
        "analysis",
        help="Read-only universe asset analysis and coverage.",
    )
    analysis_subparsers = analysis.add_subparsers(
        dest="analysis_command",
        required=True,
    )
    asset_analysis = analysis_subparsers.add_parser(
        "asset",
        help="Analyze one universe asset across available timeframes.",
    )
    asset_analysis.add_argument("--symbol", required=True)
    analysis_subparsers.add_parser(
        "coverage",
        help="Publish universe-wide analysis data coverage.",
    )
    analysis_subparsers.add_parser(
        "groups",
        help="Refresh sector and industry news/fundamental intelligence.",
    )
    analysis_subparsers.add_parser(
        "group-status",
        help="Report the latest sector and industry intelligence coverage.",
    )
    analysis_subparsers.add_parser(
        "news-events",
        help="Normalize, deduplicate and classify current multi-source news.",
    )
    analysis_subparsers.add_parser(
        "news-status",
        help="Report the latest causal news-event intelligence status.",
    )
    analysis_subparsers.add_parser(
        "news-event-study",
        help="Build local causal and descriptive market-adjusted CAR labels.",
    )
    analysis_subparsers.add_parser(
        "news-event-study-status",
        help="Report news CAR coverage and causal training readiness.",
    )

    strategy = subparsers.add_parser("strategy", help="Offline strategy research commands.")
    strategy_subparsers = strategy.add_subparsers(dest="strategy_command", required=True)
    multi_asset = strategy_subparsers.add_parser(
        "multi-asset",
        help="Offline multi-asset sleeve rotation and PF prequalification.",
    )
    multi_asset_subparsers = multi_asset.add_subparsers(dest="multi_asset_command", required=True)
    multi_asset_subparsers.add_parser("schema", help="Print the multi-asset strategy contract.")
    multi_asset_subparsers.add_parser("status", help="Report local strategy data readiness.")
    multi_asset_backtest = multi_asset_subparsers.add_parser(
        "backtest",
        help="Backtest the offline multi-asset sleeve rotation on local bar cache only.",
    )
    multi_asset_backtest.add_argument(
        "--interval", choices=["1d", "1h"], default="1d"
    )
    multi_asset_backtest.add_argument("--data-type", choices=[item.value for item in BarDataType], default="TRADES")
    multi_asset_backtest.add_argument(
        "--source",
        choices=["ANY", "LOCAL", *[item.value for item in BarDataSource]],
        default="ANY",
    )
    multi_asset_backtest.add_argument("--lookback-bars", type=int, default=3)
    multi_asset_backtest.add_argument("--top-n-per-sleeve", type=int, default=2)
    multi_asset_backtest.add_argument("--cost-bps", type=float, default=10.0)
    multi_asset_backtest.add_argument("--initial-nav", type=float, default=100_000.0)
    multi_asset_backtest.add_argument("--max-asset-weight", type=float, default=0.25)

    execution = subparsers.add_parser("execution", help="Offline execution-control-plane commands.")
    execution_subparsers = execution.add_subparsers(dest="execution_command", required=True)
    execution_subparsers.add_parser(
        "status", help="Report bounded paper/live execution authority."
    )
    execution_preflight = execution_subparsers.add_parser(
        "preflight", help="Run paper or live execution preflight."
    )
    execution_preflight.add_argument(
        "--environment", choices=["paper", "live"], default="paper"
    )
    execution_preflight.add_argument("--env-file", default=".env.ibkr")
    execution_activate_paper = execution_subparsers.add_parser(
        "activate-paper", help="Activate bounded automatic paper authority."
    )
    execution_activate_paper.add_argument("--approval", required=True)
    execution_activate_paper.add_argument("--env-file", default=".env.ibkr")
    execution_subparsers.add_parser("deactivate-paper")
    execution_subparsers.add_parser("paper-fill-close-canary")
    execution_subparsers.add_parser("paper-runtime-status")
    execution_subparsers.add_parser("paper-allowlist")
    paper_writer_preflight = execution_subparsers.add_parser(
        "paper-writer-preflight"
    )
    paper_writer_preflight.add_argument(
        "--env-file", default=".env.ibkr"
    )
    paper_writer_cycle = execution_subparsers.add_parser(
        "paper-writer-cycle"
    )
    paper_writer_cycle.add_argument("--env-file", default=".env.ibkr")
    execution_subparsers.add_parser("paper-kill-switch-drill")
    paper_session_audit = execution_subparsers.add_parser(
        "paper-session-audit"
    )
    paper_session_audit.add_argument("--session-date")
    execution_activate_live = execution_subparsers.add_parser(
        "activate-live-canary"
    )
    execution_activate_live.add_argument("--approval", required=True)
    execution_activate_live.add_argument(
        "--env-file", default=".env.ibkr.live"
    )
    execution_subparsers.add_parser("deactivate-live")
    phase7 = execution_subparsers.add_parser("phase7", help="Offline Phase 7 execution control plane.")
    phase7_subparsers = phase7.add_subparsers(dest="phase7_command", required=True)
    phase7_subparsers.add_parser("schema", help="Print and write the Phase 7 schema artifact.")
    phase7_subparsers.add_parser("init-ledger", help="Initialize the synthetic Phase 7 sqlite ledger.")
    phase7_subparsers.add_parser("simulate", help="Run deterministic fake-broker scenarios.")
    phase7_subparsers.add_parser("audit-ledger", help="Audit the synthetic ledger.")
    phase7_subparsers.add_parser("replay", help="Replay the synthetic event ledger.")
    phase7_subparsers.add_parser("reconcile-fixtures", help="Run fake-broker reconciliation fixtures.")
    phase7_subparsers.add_parser("status", help="Report Phase 7 status.")
    phase7_subparsers.add_parser("freeze", help="Write the Phase 7 freeze artifact.")

    positions = subparsers.add_parser(
        "positions", help="Paper/live position status, reconciliation and close gates."
    )
    positions_subparsers = positions.add_subparsers(
        dest="positions_command", required=True
    )
    positions_subparsers.add_parser("status")
    positions_reconcile = positions_subparsers.add_parser("reconcile")
    positions_reconcile.add_argument(
        "--environment", choices=["paper", "live"], default="paper"
    )
    positions_close = positions_subparsers.add_parser("close")
    positions_close.add_argument("--symbol", required=True)
    positions_close.add_argument(
        "--environment", choices=["paper", "live"], default="paper"
    )
    positions_close.add_argument("--approval", default=None)
    positions_register = positions_subparsers.add_parser(
        "register-manual",
        help="Register a manually executed signal as MANUAL_TRACKED.",
    )
    positions_register.add_argument("--signal-id", required=True)
    positions_register.add_argument(
        "--quantity",
        type=Decimal,
        required=True,
    )
    positions_register.add_argument(
        "--fill-price",
        type=Decimal,
        required=True,
    )
    positions_register.add_argument(
        "--environment",
        choices=["paper", "live"],
        default="paper",
    )
    positions_match = positions_subparsers.add_parser(
        "broker-match",
        help=(
            "Read-only match a registered manual position to one current "
            "broker snapshot."
        ),
    )
    positions_match.add_argument("--position-id", required=True)
    positions_match.add_argument(
        "--environment",
        choices=["paper", "live"],
        default="paper",
    )
    positions_claim = positions_subparsers.add_parser(
        "claim",
        help="Explicitly classify a manual position as bot-managed.",
    )
    positions_claim.add_argument("--position-id", required=True)
    positions_claim.add_argument(
        "--mode",
        choices=["bot-managed"],
        required=True,
    )
    positions_claim.add_argument("--yes", action="store_true")
    positions_unclaim = positions_subparsers.add_parser(
        "unclaim",
        help="Return a claimed position to manual tracking.",
    )
    positions_unclaim.add_argument("--position-id", required=True)
    positions_unclaim.add_argument("--yes", action="store_true")

    signals = subparsers.add_parser(
        "signals",
        help="Broker-independent manual model signals with authority separated from execution.",
    )
    signals_subparsers = signals.add_subparsers(
        dest="signals_command", required=True
    )
    signal_scan_parser = signals_subparsers.add_parser("scan")
    signal_scan_parser.add_argument(
        "--universe",
        choices=["all", "stocks", "etfs", "commodities"],
        default="all",
    )
    signal_scan_parser.add_argument("--minimum-confidence", type=float, default=0.0)
    signal_scan_parser.add_argument("--minimum-reward-risk", type=float, default=1.5)
    signal_scan_parser.add_argument("--strategy", default=None)
    signal_scan_parser.add_argument("--timeframe", default=None)
    signal_scan_parser.add_argument("--asset-class", default=None)
    signal_scan_parser.add_argument("--maximum-signals", type=int, default=5000)
    signal_asset_parser = signals_subparsers.add_parser("asset")
    signal_asset_parser.add_argument("--symbol", required=True)
    for command in ("watchlist", "active", "expired", "export", "status"):
        signals_subparsers.add_parser(command)
    signal_top_parser = signals_subparsers.add_parser("top")
    signal_top_parser.add_argument("--limit", type=int, default=5)
    signal_raw_top_parser = signals_subparsers.add_parser("raw-top")
    signal_raw_top_parser.add_argument("--limit", type=int, default=5)
    signal_diversified_top_parser = signals_subparsers.add_parser(
        "diversified-top"
    )
    signal_diversified_top_parser.add_argument("--limit", type=int, default=5)
    signal_trending_parser = signals_subparsers.add_parser("trending")
    signal_trending_parser.add_argument("--limit", type=int, default=5)
    signal_actionable_parser = signals_subparsers.add_parser("actionable")
    signal_actionable_parser.add_argument("--limit", type=int, default=5)
    for command in (
        "top-stocks",
        "top-etfs",
        "top-commodities",
        "auto-eligible",
        "dashboard",
    ):
        specialized = signals_subparsers.add_parser(command)
        specialized.add_argument("--limit", type=int, default=5)
    signal_explain_parser = signals_subparsers.add_parser("explain")
    signal_explain_parser.add_argument("--signal-id", required=True)
    signal_inspect_parser = signals_subparsers.add_parser("inspect")
    signal_inspect_parser.add_argument(
        "--id",
        dest="signal_id",
        required=True,
    )
    signal_order_plan_parser = signals_subparsers.add_parser("order-plan")
    signal_order_plan_parser.add_argument(
        "--id",
        dest="signal_id",
        required=True,
    )
    signal_order_plan_parser.add_argument(
        "--capital",
        type=Decimal,
        required=True,
    )
    signal_order_plan_parser.add_argument(
        "--risk",
        type=Decimal,
        required=True,
    )
    signal_executed_parser = signals_subparsers.add_parser("mark-executed")
    signal_executed_parser.add_argument("--signal-id", required=True)
    signal_executed_parser.add_argument("--quantity", type=Decimal, required=True)
    signal_executed_parser.add_argument("--fill-price", type=Decimal, required=True)
    signal_closed_parser = signals_subparsers.add_parser("mark-closed")
    signal_closed_parser.add_argument("--signal-id", required=True)
    signal_closed_parser.add_argument("--quantity", type=Decimal, required=True)
    signal_closed_parser.add_argument("--fill-price", type=Decimal, required=True)
    signal_closed_parser.add_argument("--reason", required=True)

    strategies = subparsers.add_parser(
        "strategies", help="Governed strategy authority transitions."
    )
    strategies_subparsers = strategies.add_subparsers(
        dest="strategies_command", required=True
    )
    promote_signals_parser = strategies_subparsers.add_parser(
        "promote-manual-signals"
    )
    promote_signals_parser.add_argument("--strategy-id", required=True)
    promote_signals_parser.add_argument("--approval", required=True)

    autopilot = subparsers.add_parser(
        "autopilot", help="Bounded externally triggered continuous research scheduler."
    )
    autopilot_subparsers = autopilot.add_subparsers(
        dest="autopilot_command", required=True
    )
    for command in (
        "start",
        "stop",
        "status",
        "pause",
        "resume",
        "leaderboard",
        "failures",
    ):
        autopilot_subparsers.add_parser(command)
    autopilot_run_once = autopilot_subparsers.add_parser("run-once")
    autopilot_run_once.add_argument(
        "--mode", choices=sorted(MACHINE_MODES), default="SIGNALS_ONLY"
    )
    autopilot_run_once.add_argument(
        "--interval-seconds", type=int, default=300
    )
    autopilot_run = autopilot_subparsers.add_parser("run")
    autopilot_run.add_argument(
        "--mode", choices=sorted(MACHINE_MODES), default="SIGNALS_ONLY"
    )
    autopilot_run.add_argument("--max-cycles", type=int, default=1)
    autopilot_run.add_argument("--interval-seconds", type=int, default=300)

    daily_parser = subparsers.add_parser(
        "daily", help="Canonical broker-independent daily research and signal workflow."
    )
    daily_parser.add_argument("--signals-only", action="store_true")
    daily_parser.add_argument("--research-only", action="store_true")
    daily_parser.add_argument("--no-autopilot", action="store_true")
    daily_parser.add_argument("--no-telegram", action="store_true")

    dynamic = subparsers.add_parser(
        "dynamic", help="Frozen multi-strategy regime, consensus and portfolio orchestration."
    )
    dynamic_subparsers = dynamic.add_subparsers(
        dest="dynamic_command", required=True
    )
    for command in (
        "status",
        "regime",
        "strategies",
        "signals",
        "portfolio",
        "daily",
        "paper-campaign",
    ):
        dynamic_subparsers.add_parser(command)
    dynamic_explain = dynamic_subparsers.add_parser("explain")
    dynamic_explain.add_argument("--symbol", required=True)

    telegram = subparsers.add_parser(
        "telegram",
        help="Outbound-only Telegram notifications; never an execution authority.",
    )
    telegram_subparsers = telegram.add_subparsers(
        dest="telegram_command", required=True
    )
    for command in (
        "health",
        "test",
        "status",
        "preview",
        "send-latest-signals",
        "send-pit-mtf-signals",
        "send-exit-signals",
        "top-5-preview",
        "send-top-5",
        "send-regime-update",
        "market-digest-preview",
        "send-market-digest",
        "send-shadow-digest",
        "retry-failed",
    ):
        telegram_subparsers.add_parser(command)

    live = subparsers.add_parser(
        "live",
        help="Fail-closed controlled IBKR live execution and runtime controls.",
    )
    live_subparsers = live.add_subparsers(dest="live_command", required=True)
    live_preflight_parser = live_subparsers.add_parser("preflight")
    live_preflight_parser.add_argument("--strategy", default=None)
    live_preflight_parser.add_argument("--symbol", default=None)
    live_preflight_parser.add_argument(
        "--max-order-eur", type=Decimal, default=Decimal("250")
    )
    live_preflight_parser.add_argument("--approval", default=None)
    live_canary_parser = live_subparsers.add_parser("canary")
    live_canary_parser.add_argument("--strategy", required=True)
    live_canary_parser.add_argument("--symbol", required=True)
    live_canary_parser.add_argument(
        "--max-order-eur", type=Decimal, default=Decimal("250")
    )
    live_canary_parser.add_argument("--approval", required=True)
    live_subparsers.add_parser(
        "reconcile",
        help="Observe live broker state read-only and require an empty baseline.",
    )
    live_prepare_parser = live_subparsers.add_parser(
        "prepare",
        help="Prepare one bounded manual Level-1 live bracket intent.",
    )
    live_prepare_parser.add_argument("--con-id", type=int, required=True)
    live_prepare_parser.add_argument("--quantity", type=Decimal, required=True)
    live_prepare_parser.add_argument(
        "--entry-limit-price", type=Decimal, required=True
    )
    live_prepare_parser.add_argument(
        "--stop-price", type=Decimal, required=True
    )
    live_prepare_parser.add_argument(
        "--take-profit-price", type=Decimal, required=True
    )
    live_prepare_parser.add_argument(
        "--fx-rate-to-eur", type=Decimal, required=True
    )
    live_prepare_parser.add_argument("--reason", required=True)
    live_prepare_parser.add_argument("--strategy", default=None)
    live_approve_parser = live_subparsers.add_parser(
        "approve",
        help="Record one exact, expiring operator approval for a live intent.",
    )
    live_approve_parser.add_argument("--intent-id", required=True)
    live_approve_parser.add_argument("--approval", required=True)
    live_submit_parser = live_subparsers.add_parser(
        "submit",
        help="Submit one separately approved live bracket after all gates pass.",
    )
    live_submit_parser.add_argument("--intent-id", required=True)
    live_submit_parser.add_argument(
        "--activation-approval", required=True
    )
    controlled_preflight = live_subparsers.add_parser(
        "controlled-preflight",
        help="Evaluate a desired whole-share target against Level-2 gates.",
    )
    controlled_preflight.add_argument("--symbol", required=True)
    controlled_prepare = live_subparsers.add_parser(
        "controlled-prepare",
        help="Prepare a post-canary whole-share intent without broker calls.",
    )
    controlled_prepare.add_argument("--symbol", required=True)
    controlled_prepare.add_argument("--strategy", required=True)
    controlled_approve = live_subparsers.add_parser(
        "controlled-approve",
        help="Approve one exact post-canary intent.",
    )
    controlled_approve.add_argument("--intent-id", required=True)
    controlled_approve.add_argument("--approval", required=True)
    controlled_submit = live_subparsers.add_parser(
        "controlled-submit",
        help="Submit an approved Level-2 intent after every gate passes.",
    )
    controlled_submit.add_argument("--intent-id", required=True)
    controlled_submit.add_argument(
        "--activation-approval", required=True
    )
    live_activate_level_two = live_subparsers.add_parser(
        "activate-level-two",
        help="Promote Level-1 to bounded manual Level-2 after verified round trips.",
    )
    live_activate_level_two.add_argument("--symbol", required=True)
    live_activate_level_two.add_argument("--approval", required=True)
    live_subparsers.add_parser(
        "audit",
        help="Run the live writer offline audit without a broker connection.",
    )
    live_subparsers.add_parser(
        "status",
        help="Report live writer freeze and private-ledger state.",
    )
    live_subparsers.add_parser(
        "allowlist",
        help="Build the frozen PIT live strategy and symbol allowlist.",
    )
    live_capability_create = live_subparsers.add_parser(
        "capability-create",
        help="Create one expiring capability after live reconciliation and preflight.",
    )
    live_capability_create.add_argument(
        "--profile",
        default="autonomous_multi_asset_v1",
    )
    live_capability_create.add_argument("--yes", action="store_true")
    live_subparsers.add_parser(
        "capability-status",
        help="Inspect the public status of the current expiring capability.",
    )
    live_activate_capability = live_subparsers.add_parser(
        "activate",
        help="Consume the capability and activate bounded Level-1 authority.",
    )
    live_activate_capability.add_argument(
        "--profile",
        default="autonomous_multi_asset_v1",
    )
    live_activate_capability.add_argument("--approval", required=True)
    live_activate_capability.add_argument("--yes", action="store_true")
    live_launch = live_subparsers.add_parser(
        "launch",
        help="Reconcile, activate and optionally start the controlled live runtime.",
    )
    live_launch.add_argument(
        "--profile",
        default="autonomous_multi_asset_v1",
    )
    live_launch.add_argument("--approval", required=True)
    live_launch.add_argument("--continuous", action="store_true")
    live_launch.add_argument("--resume", action="store_true")
    live_launch.add_argument("--yes", action="store_true")
    live_activate = live_subparsers.add_parser(
        "activate-level-one",
        help="Atomically activate bounded Level-1 authority after full preflight.",
    )
    live_activate.add_argument("--approval", required=True)
    live_autonomous = live_subparsers.add_parser(
        "activate-autonomous-level-one",
        help="Activate separately frozen autonomous Level-1 policy authority.",
    )
    live_autonomous.add_argument("--approval", required=True)
    live_autonomous.add_argument("--yes", action="store_true")
    live_pause = live_subparsers.add_parser("pause")
    live_pause.add_argument("--reason", default="OPERATOR_PAUSE")
    live_subparsers.add_parser("resume")
    live_kill_control = live_subparsers.add_parser("kill")
    live_kill_control.add_argument("--reason", required=True)
    for command in (
        "positions",
        "orders",
        "performance",
        "strategy-status",
        "risk-status",
        "runtime-status",
    ):
        live_subparsers.add_parser(command)
    live_subparsers.add_parser(
        "p2-1-freeze",
        help="Build the immutable P2.1 strategy and autonomous-policy freeze.",
    )
    live_subparsers.add_parser(
        "p2-1-status",
        help="Verify the immutable P2.1 strategy and autonomous-policy freeze.",
    )
    for command in (
        "automatic-cycle-audit",
        "automatic-cycle-freeze",
        "automatic-cycle-status",
        "automatic-cycle-run",
    ):
        live_subparsers.add_parser(command)
    live_subparsers.add_parser(
        "freeze",
        help="Freeze the offline-audited live writer source hashes.",
    )
    live_integrity = live_subparsers.add_parser(
        "writer-integrity",
        help="Inspect, verify, diff or explicitly freeze live-writer code.",
    )
    live_integrity.add_argument(
        "action",
        choices=("inspect", "verify", "diff", "freeze", "re-freeze"),
    )
    live_integrity.add_argument("--operator", default="")
    live_integrity.add_argument("--reason", default="")
    live_integrity.add_argument("--confirm", action="store_true")
    live_subparsers.add_parser("position-status")
    live_close_parser = live_subparsers.add_parser("close-position")
    live_close_parser.add_argument("--symbol", required=True)
    live_close_parser.add_argument("--approval", required=True)
    live_kill = live_subparsers.add_parser("kill-switch")
    live_kill_subparsers = live_kill.add_subparsers(
        dest="live_kill_command", required=True
    )
    live_kill_activate = live_kill_subparsers.add_parser("activate")
    live_kill_activate.add_argument("--reason", required=True)
    live_kill_subparsers.add_parser("status")

    system = subparsers.add_parser(
        "system", help="Publish integrated research, signal, paper and live readiness."
    )
    system_subparsers = system.add_subparsers(
        dest="system_command", required=True
    )
    system_subparsers.add_parser("readiness")
    system_subparsers.add_parser("audit")
    system_subparsers.add_parser("universe")

    canonical_run = subparsers.add_parser(
        "run",
        help="Run the bounded canonical 24/7 operations loop.",
    )
    canonical_run.add_argument(
        "--mode", choices=sorted(MACHINE_MODES), default="SIGNALS_ONLY"
    )
    canonical_run.add_argument("--max-cycles", type=int, default=1_440)
    canonical_run.add_argument("--interval-seconds", type=int, default=60)

    launch = subparsers.add_parser(
        "launch",
        help="Fail-closed canonical runtime and live-readiness controls.",
    )
    launch_subparsers = launch.add_subparsers(
        dest="launch_command", required=True
    )
    for command in ("preflight", "status", "stop"):
        launch_subparsers.add_parser(command)
    launch_live = launch_subparsers.add_parser("live")
    launch_live.add_argument(
        "--profile",
        default="autonomous_multi_asset_v1",
    )
    launch_live.add_argument("--approval", required=True)
    launch_live.add_argument("--continuous", action="store_true")
    launch_live.add_argument("--resume", action="store_true")
    launch_live.add_argument("--yes", action="store_true")

    ui = subparsers.add_parser(
        "ui",
        help="Local read-only operations console.",
    )
    ui_subparsers = ui.add_subparsers(dest="ui_command", required=True)
    for command in ("start", "serve"):
        ui_runtime = ui_subparsers.add_parser(command)
        ui_runtime.add_argument("--host", default="127.0.0.1")
        ui_runtime.add_argument("--port", type=int, default=8080)
    ui_subparsers.add_parser("status")
    ui_subparsers.add_parser("stop")

    subparsers.add_parser("doctor", help="Inspect local Phase 0 and Phase 1 readiness.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "p3":
        report = publish_p3_evidence(PROJECT_ROOT)
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "ui":
        report = ui_command(
            PROJECT_ROOT,
            args.ui_command,
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 8080),
        )
        if args.ui_command != "serve":
            _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "config":
        report = config_validation(PROJECT_ROOT, env_file=args.env_file)
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "self-test":
        report = system_self_test(PROJECT_ROOT, env_file=args.env_file)
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "risk":
        report = portfolio_management_command(PROJECT_ROOT, "risk")
        _print_json(report)
        return (
            0
            if report["status"] in {"GO", "NO_TARGET_POSITIONS"}
            else 2
        )

    if args.command == "analysis":
        if args.analysis_command == "asset":
            report = analyze_asset(PROJECT_ROOT, args.symbol)
        elif args.analysis_command == "coverage":
            report = build_analysis_coverage(PROJECT_ROOT)
        elif args.analysis_command == "groups":
            report = build_group_intelligence(PROJECT_ROOT)
        elif args.analysis_command == "news-events":
            report = build_news_event_intelligence(PROJECT_ROOT)
        elif args.analysis_command == "news-status":
            report = news_event_intelligence_status(PROJECT_ROOT)
        elif args.analysis_command == "news-event-study":
            report = build_news_event_study(PROJECT_ROOT)
        elif args.analysis_command == "news-event-study-status":
            report = news_event_study_status(PROJECT_ROOT)
        else:
            report = group_intelligence_status(PROJECT_ROOT)
        _print_json(report)
        return 0 if report["status"] in {
            "GO",
            "GO_WITH_DOCUMENTED_GAPS",
            "DATA_UNAVAILABLE",
            "NO_CURRENT_EVENTS",
            "NOT_RUN",
        } else 2

    if args.command == "universe":
        report = discovery_universe_command(
            PROJECT_ROOT, args.universe_command
        )
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command in {"sectors", "industries", "regions"}:
        dimensions = {
            "sectors": "sector",
            "industries": "industry",
            "regions": "region",
        }
        report = rank_universe_dimension(
            PROJECT_ROOT, dimensions[args.command]
        )
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "regimes":
        commands = {
            "schema": regimes_schema,
            "fit": regimes_fit,
            "walk-forward": regimes_walk_forward,
            "current": regimes_current,
            "audit": regimes_audit,
            "status": regimes_status,
        }
        report = commands[args.regimes_command](PROJECT_ROOT)
        _print_json(report)
        return 0 if report.get("status") in {
            "GO",
            "DEGRADED",
            "NOT_RUN",
            "MODEL_NOT_FIT",
        } else 2

    if args.command == "run":
        report = machine_command(
            PROJECT_ROOT,
            "run",
            mode=args.mode,
            max_cycles=args.max_cycles,
            interval_seconds=args.interval_seconds,
        )
        _print_json(report)
        return 0 if report["status"] in {"GO", "DEGRADED"} else 2

    if args.command == "launch":
        report = launch_command(
            PROJECT_ROOT,
            args.launch_command,
            approval=getattr(args, "approval", None),
            confirmed=getattr(args, "yes", False),
            profile=getattr(
                args,
                "profile",
                "autonomous_multi_asset_v1",
            ),
            continuous=getattr(args, "continuous", False),
            resume=getattr(args, "resume", False),
        )
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "dynamic":
        report = dynamic_command(
            PROJECT_ROOT,
            args.dynamic_command,
            symbol=getattr(args, "symbol", None),
        )
        _print_json(report)
        status = report.get("status")
        if isinstance(status, dict):
            status = status.get("status")
        return 0 if status not in {"BLOCKED", "ERROR"} else 2

    if args.command == "signals":
        if args.signals_command == "scan":
            report = signal_scan(
                PROJECT_ROOT,
                universe=args.universe,
                minimum_confidence=args.minimum_confidence,
                minimum_reward_risk=args.minimum_reward_risk,
                strategy=args.strategy,
                timeframe=args.timeframe,
                asset_class=args.asset_class,
                maximum_signals=args.maximum_signals,
            )
        elif args.signals_command == "asset":
            report = signal_asset(PROJECT_ROOT, args.symbol)
        elif args.signals_command in {"watchlist", "active", "expired"}:
            report = signal_list(PROJECT_ROOT, args.signals_command)
        elif args.signals_command in {"explain", "inspect"}:
            report = signal_explain(PROJECT_ROOT, args.signal_id)
        elif args.signals_command == "order-plan":
            report = signal_order_plan(
                PROJECT_ROOT,
                signal_id=args.signal_id,
                capital=args.capital,
                risk=args.risk,
            )
        elif args.signals_command == "mark-executed":
            report = signal_mark_executed(
                PROJECT_ROOT,
                signal_id=args.signal_id,
                quantity=args.quantity,
                fill_price=args.fill_price,
            )
        elif args.signals_command == "mark-closed":
            report = signal_mark_closed(
                PROJECT_ROOT,
                signal_id=args.signal_id,
                quantity=args.quantity,
                fill_price=args.fill_price,
                reason=args.reason,
            )
        elif args.signals_command == "export":
            report = signal_export(PROJECT_ROOT)
        elif args.signals_command in {
            "top",
            "raw-top",
            "diversified-top",
            "trending",
            "actionable",
            "top-stocks",
            "top-etfs",
            "top-commodities",
            "auto-eligible",
            "dashboard",
        }:
            mode = {
                "raw-top": "raw",
                "diversified-top": "diversified",
                "top-stocks": "stocks",
                "top-etfs": "etfs",
                "top-commodities": "commodities",
            }.get(args.signals_command, args.signals_command)
            report = publish_top_signals(
                PROJECT_ROOT,
                mode=mode,
                limit=args.limit,
            )
        else:
            report = signal_status(PROJECT_ROOT)
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "strategies":
        report = promote_manual_signals(
            PROJECT_ROOT,
            strategy_id=args.strategy_id,
            approval=args.approval,
        )
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "autopilot":
        if args.autopilot_command == "run":
            report = machine_command(
                PROJECT_ROOT,
                "run",
                mode=args.mode,
                max_cycles=args.max_cycles,
                interval_seconds=args.interval_seconds,
            )
        elif args.autopilot_command in {
            "start",
            "stop",
            "pause",
            "resume",
            "status",
        }:
            research = runtime_command(PROJECT_ROOT, args.autopilot_command)
            machine = machine_command(
                PROJECT_ROOT,
                args.autopilot_command,
            )
            report = {
                "schema": "stocks_autopilot_composite_status_v1",
                "status": (
                    "GO"
                    if research.get("status") == "GO"
                    and machine.get("status") == "GO"
                    else "DEGRADED"
                ),
                "research": research,
                "machine": machine,
                "execution_authority": machine.get(
                    "execution_authority", "NONE"
                ),
            }
        else:
            report = runtime_command(PROJECT_ROOT, args.autopilot_command)
        _print_json(report)
        return 0 if report["status"] in {"GO", "DEGRADED", "DATA_BLOCKED"} else 2

    if args.command == "daily":
        report = run_daily(
            PROJECT_ROOT,
            signals_only=args.signals_only,
            research_only=args.research_only,
            no_autopilot=args.no_autopilot,
            no_telegram=args.no_telegram,
        )
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "telegram":
        report = telegram_command(PROJECT_ROOT, args.telegram_command)
        _print_json(report)
        return (
            0
            if report["status"]
            in {
                "GO",
                "ENABLED",
                "DRY_RUN",
                "DISABLED_BY_CONFIG",
                "DISABLED_MISSING_CONFIG",
            }
            else 2
        )

    if args.command == "live":
        live_env = args.env_file if args.env_file != ".env.ibkr" else ".env.ibkr.live"
        portfolio_live_env = (
            args.env_file
            if args.env_file != ".env.ibkr"
            else ".env.ibkr.portfolio.live"
        )
        if args.live_command == "preflight":
            report = live_preflight(
                PROJECT_ROOT,
                env_file=live_env,
                strategy_id=args.strategy,
                symbol=args.symbol,
                max_order_eur=args.max_order_eur,
                approval=args.approval,
            )
        elif args.live_command == "canary":
            report = live_canary(
                PROJECT_ROOT,
                env_file=live_env,
                strategy_id=args.strategy,
                symbol=args.symbol,
                max_order_eur=args.max_order_eur,
                approval=args.approval,
            )
        elif args.live_command == "reconcile":
            report = live_reconcile(PROJECT_ROOT, env_file=live_env)
        elif args.live_command == "prepare":
            report = live_prepare(
                PROJECT_ROOT,
                env_file=live_env,
                con_id=args.con_id,
                quantity=args.quantity,
                entry_limit_price=args.entry_limit_price,
                stop_price=args.stop_price,
                take_profit_price=args.take_profit_price,
                fx_rate_to_eur=args.fx_rate_to_eur,
                reason=args.reason,
                strategy_id=args.strategy,
            )
        elif args.live_command == "approve":
            report = live_approve(
                PROJECT_ROOT,
                env_file=live_env,
                intent_id=args.intent_id,
                approval=args.approval,
            )
        elif args.live_command == "submit":
            report = live_submit(
                PROJECT_ROOT,
                env_file=live_env,
                intent_id=args.intent_id,
                activation_approval=args.activation_approval,
            )
        elif args.live_command == "controlled-preflight":
            report = controlled_live_preflight(
                PROJECT_ROOT,
                symbol=args.symbol,
                env_file=portfolio_live_env,
            )
        elif args.live_command == "controlled-prepare":
            report = controlled_live_prepare(
                PROJECT_ROOT,
                symbol=args.symbol,
                strategy_id=args.strategy,
                env_file=portfolio_live_env,
            )
        elif args.live_command == "controlled-approve":
            report = controlled_live_approve(
                PROJECT_ROOT,
                intent_id=args.intent_id,
                approval_text=args.approval,
                env_file=portfolio_live_env,
            )
        elif args.live_command == "controlled-submit":
            report = controlled_live_submit(
                PROJECT_ROOT,
                intent_id=args.intent_id,
                activation_approval=args.activation_approval,
                env_file=portfolio_live_env,
            )
        elif args.live_command == "activate-level-two":
            report = activate_level_two(
                PROJECT_ROOT,
                symbol=args.symbol,
                approval=args.approval,
                env_file=portfolio_live_env,
            )
        elif args.live_command == "audit":
            report = live_audit(PROJECT_ROOT)
        elif args.live_command == "status":
            report = live_status(PROJECT_ROOT)
        elif args.live_command == "allowlist":
            report = live_strategy_allowlist(PROJECT_ROOT)
        elif args.live_command == "capability-create":
            report = operations_execution_command(
                PROJECT_ROOT,
                "create-live-capability",
                environment="live",
                env_file=live_env,
                confirmed=args.yes,
                profile=args.profile,
            )
        elif args.live_command == "capability-status":
            report = operations_execution_command(
                PROJECT_ROOT,
                "live-capability-status",
                environment="live",
                env_file=live_env,
            )
        elif args.live_command == "activate":
            report = operations_execution_command(
                PROJECT_ROOT,
                "activate-live-capability",
                environment="live",
                approval=args.approval,
                env_file=live_env,
                confirmed=args.yes,
                profile=args.profile,
            )
        elif args.live_command == "launch":
            report = launch_command(
                PROJECT_ROOT,
                "live",
                approval=args.approval,
                confirmed=args.yes,
                profile=args.profile,
                continuous=args.continuous,
                resume=args.resume,
            )
        elif args.live_command == "activate-level-one":
            report = operations_execution_command(
                PROJECT_ROOT,
                "activate-live-canary",
                environment="live",
                approval=args.approval,
                env_file=live_env,
                confirmed=True,
            )
        elif args.live_command == "activate-autonomous-level-one":
            if (
                not args.yes
                or args.approval != "ACTIVATE AUTONOMOUS LEVEL ONE"
            ):
                report = {
                    "schema": "autonomous_level_one_activation_v1",
                    "status": "NO_GO",
                    "blockers": [
                        "EXACT_AUTONOMOUS_ACTIVATION_CONFIRMATION_REQUIRED"
                    ],
                }
            else:
                report = activate_autonomous_level_one(
                    PROJECT_ROOT,
                    preflight=live_preflight(
                        PROJECT_ROOT,
                        env_file=live_env,
                    ),
                )
        elif args.live_command == "pause":
            report = operations_execution_command(
                PROJECT_ROOT,
                "pause-live",
                approval=args.reason,
                env_file=live_env,
            )
        elif args.live_command == "resume":
            report = operations_execution_command(
                PROJECT_ROOT,
                "resume-live",
                env_file=live_env,
            )
        elif args.live_command == "kill":
            report = operations_execution_command(
                PROJECT_ROOT,
                "kill-live",
                approval=args.reason,
                env_file=live_env,
            )
        elif args.live_command in {
            "positions",
            "orders",
            "performance",
            "strategy-status",
            "risk-status",
            "runtime-status",
        }:
            report = live_component_status(
                PROJECT_ROOT, args.live_command
            )
        elif args.live_command == "automatic-cycle-audit":
            report = automatic_cycle_audit(PROJECT_ROOT)
        elif args.live_command == "automatic-cycle-freeze":
            report = automatic_cycle_freeze(PROJECT_ROOT)
        elif args.live_command == "automatic-cycle-status":
            report = automatic_cycle_status(PROJECT_ROOT)
        elif args.live_command == "automatic-cycle-run":
            report = automatic_cycle(
                PROJECT_ROOT,
                env_file=live_env,
            )
        elif args.live_command == "p2-1-freeze":
            report = build_p2_1_freeze(PROJECT_ROOT)
        elif args.live_command == "p2-1-status":
            report = verify_p2_1_freeze(PROJECT_ROOT)
        elif args.live_command == "freeze":
            report = live_freeze(PROJECT_ROOT)
        elif args.live_command == "writer-integrity":
            report = live_writer_integrity_command(
                PROJECT_ROOT,
                args.action,
                operator=args.operator,
                reason=args.reason,
                confirmed=args.confirm,
            )
        elif args.live_command == "position-status":
            report = live_position_status(PROJECT_ROOT)
        elif args.live_command == "close-position":
            report = live_close_position(
                PROJECT_ROOT, symbol=args.symbol, approval=args.approval
            )
        else:
            report = live_kill_switch(
                PROJECT_ROOT,
                command=args.live_kill_command,
                reason=getattr(args, "reason", None),
            )
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "system":
        report = (
            broad_universe_status(PROJECT_ROOT)
            if args.system_command == "universe"
            else system_readiness(PROJECT_ROOT)
        )
        _print_json(report)
        return 0

    if args.command == "forward":
        if args.forward_command == "register":
            report = forward_register(args.strategy_id)
        elif args.forward_command == "run":
            report = forward_run()
        elif args.forward_command in {"status", "report"}:
            report = forward_status()
        else:
            raise SystemExit(f"Unknown forward command: {args.forward_command}")
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "capital":
        report = capital_command(
            PROJECT_ROOT,
            args.capital_command,
            level=getattr(args, "level", None),
            approval=getattr(args, "approval", None),
            reason=getattr(args, "reason", None),
            account_equity_eur=getattr(
                args, "account_equity_eur", None
            ),
            net_daily_pnl_eur=getattr(
                args, "net_daily_pnl_eur", None
            ),
            enforce_daily_target=getattr(args, "enforce", False),
        )
        _print_json(report)
        return 0 if report["status"] in {"GO", "DATA_BLOCKED"} else 2

    if args.command == "portfolio":
        if args.portfolio_command == "backtest":
            report = portfolio_backtest(args.strategy_id)
        elif args.portfolio_command == "stress":
            report = portfolio_stress(args.strategy_id)
        else:
            report = portfolio_management_command(
                PROJECT_ROOT, args.portfolio_command
            )
        _print_json(report)
        return (
            0
            if report["status"]
            in {
                "GO",
                "GO_WITH_CONSTRAINTS",
                "DATA_BLOCKED",
                "NO_TARGET_POSITIONS",
                "ADVISORY_ONLY",
            }
            else 2
        )

    if args.command == "research" and args.research_command == "registry":
        report = research_registry_command(
            PROJECT_ROOT,
            args.registry_command,
        )
        _print_json(report)
        return (
            0
            if report["status"] in {"GO", "GO_WITH_EVIDENCE_GAPS"}
            else 2
        )

    if args.command == "research" and args.research_command == "active-swing":
        if args.active_swing_command == "shortlist-data":
            report = publish_shortlist_coverage(PROJECT_ROOT)
        elif args.active_swing_command == "rejected-shadow":
            report = settle_rejected_opportunities(PROJECT_ROOT)
        elif args.active_swing_command == "gate-attribution":
            report = gate_value_attribution_status(PROJECT_ROOT)
        elif args.active_swing_command == "evidence-throughput":
            report = publish_evidence_throughput(PROJECT_ROOT)
        elif args.active_swing_command == "entry-filter-experiment":
            report = run_entry_filter_experiment(PROJECT_ROOT)
        elif args.active_swing_command == "leaderboards":
            report = publish_active_swing_leaderboards(PROJECT_ROOT)
        elif args.active_swing_command == "train-ml":
            report = train_selective_ml(PROJECT_ROOT)
        elif args.active_swing_command == "run":
            report = run_active_swing_sprints(PROJECT_ROOT)
        elif args.active_swing_command == "refresh":
            report = refresh_active_swing_observation(PROJECT_ROOT)
        elif args.active_swing_command == "status":
            report = active_swing_sprint_status(PROJECT_ROOT)
        else:
            raise SystemExit(
                f"Unknown active-swing command: {args.active_swing_command}"
            )
        _print_json(report)
        return 0 if report.get("status") in {
            "GO",
            "PARTIAL",
            "NO_CURRENT_SETUPS",
            "INSUFFICIENT_SAMPLE",
            "NOT_TRAINED_INSUFFICIENT_FORWARD_LABELS",
            "EXPERIMENTAL_REPORT_ELIGIBLE",
            "SHADOW_COMPARISON_ELIGIBLE",
            "PAPER_RANKING_RESEARCH_ELIGIBLE",
        } else 2

    if args.command == "research" and args.research_command in {
        "components",
        "taxonomy",
        "macro-pairs",
        "generate",
        "smoke",
        "campaign",
        "daily",
        "weekly",
        "monthly",
        "candidates",
        "strategy",
        "compare",
        "combine",
        "leaderboard",
        "rejected",
        "recover-survivors",
        "audit",
        "autopilot-status",
        "freeze",
    }:
        if args.research_command == "components":
            report = component_registry_report()
        elif args.research_command == "taxonomy":
            report = autopilot_taxonomy()
        elif args.research_command == "macro-pairs":
            report = run_macro_pair_validation(
                PROJECT_ROOT,
                max_identities=args.max_identities,
            )
        elif args.research_command == "generate":
            if args.complexity is not None:
                if args.family is not None:
                    raise SystemExit(
                        "--family cannot be combined with --complexity"
                    )
                report = register_phase11_12_catalog(
                    PROJECT_ROOT,
                    complexity=args.complexity,
                    resume=args.resume,
                )
            else:
                report = autopilot_generate(
                    budget=args.budget,
                    family=args.family,
                    seed=args.seed,
                )
        elif args.research_command == "smoke":
            report = autopilot_smoke(family=args.family)
        elif args.research_command == "campaign":
            report = autopilot_campaign(
                args.family, max_trials=args.max_trials
            )
        elif args.research_command == "daily":
            report = autopilot_daily()
        elif args.research_command == "weekly":
            report = autopilot_weekly(max_trials=args.max_trials)
        elif args.research_command == "monthly":
            report = autopilot_monthly()
        elif args.research_command == "candidates":
            report = autopilot_candidates()
        elif args.research_command == "strategy":
            report = autopilot_strategy(args.strategy_id)
        elif args.research_command == "compare":
            report = autopilot_compare(args.strategy_ids)
        elif args.research_command == "combine":
            report = autopilot_combine(args.strategy_ids, mode=args.mode)
        elif args.research_command == "leaderboard":
            report = autopilot_leaderboard()
        elif args.research_command == "rejected":
            report = autopilot_rejected()
        elif args.research_command == "recover-survivors":
            report = recover_survivors(PROJECT_ROOT)
        elif args.research_command == "audit":
            report = autopilot_audit()
        elif args.research_command == "freeze":
            report = autopilot_freeze()
        else:
            report = autopilot_status()
        _print_json(report)
        return 0 if report["status"] in {"GO", "DATA_BLOCKED"} else 2

    if args.command == "shadow":
        if args.shadow_command == "phase8-2":
            if args.phase8_2_command == "schema":
                _print_json(phase8_2_schema(PROJECT_ROOT))
                return 0
            if args.phase8_2_command == "init-ledger":
                report = phase8_2_init_ledger(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_2_command == "register-fixtures":
                report = phase8_2_register_fixtures(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_2_command == "simulate":
                report = phase8_2_simulate(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_2_command == "replay":
                report = phase8_2_replay(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_2_command == "audit-ledger":
                report = phase8_2_audit_ledger(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_2_command == "activation-audit":
                report = phase8_2_activation_audit(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_2_command == "status":
                report = phase8_2_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE8_2_STRATEGY_AGNOSTIC_SHADOW_INFRASTRUCTURE_GO" else 2
            if args.phase8_2_command == "freeze":
                report = phase8_2_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE8_2_STRATEGY_AGNOSTIC_SHADOW_INFRASTRUCTURE_FROZEN_GO" else 2
            raise SystemExit(f"Unknown phase8-2 command: {args.phase8_2_command}")
        raise SystemExit(f"Unknown shadow command: {args.shadow_command}")

    if args.command == "screener":
        if args.screener_command == "run":
            report = screener_run(PROJECT_ROOT, as_of=args.as_of)
            _print_json(report)
            return 0 if report["status"] in {"GO", "ALREADY_REGISTERED"} else 2
        if args.screener_command == "preview":
            report = screener_preview(PROJECT_ROOT, as_of=args.as_of)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.screener_command == "status":
            report = screener_status(PROJECT_ROOT)
            _print_json(report)
            return 0
        if args.screener_command == "report":
            report = screener_report(PROJECT_ROOT, as_of=args.as_of)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.screener_command == "history":
            report = screener_history(PROJECT_ROOT, symbol=args.symbol)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.screener_command == "top":
            report = screener_top(PROJECT_ROOT, limit=args.limit)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.screener_command == "asset":
            report = screener_asset(PROJECT_ROOT, symbol=args.symbol)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.screener_command == "export":
            report = screener_export(PROJECT_ROOT)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        raise SystemExit(f"Unknown screener command: {args.screener_command}")

    if args.command == "macro":
        if args.macro_command == "collect":
            report = macro_collect(
                PROJECT_ROOT,
                start=args.start,
                end=args.end,
            )
        elif args.macro_command == "update":
            report = macro_update(PROJECT_ROOT)
        elif args.macro_command == "validate":
            report = macro_validate(PROJECT_ROOT)
        elif args.macro_command == "status":
            report = macro_status(PROJECT_ROOT)
        elif args.macro_command == "readiness":
            report = macro_readiness(PROJECT_ROOT)
        elif args.macro_command == "score":
            report = macro_score(PROJECT_ROOT, as_of=args.as_of)
        elif args.macro_command == "regime":
            report = macro_regime(PROJECT_ROOT, as_of=args.as_of)
        elif args.macro_command == "history":
            report = macro_history(PROJECT_ROOT, rebuild=args.rebuild)
        elif args.macro_command == "events":
            report = macro_events(PROJECT_ROOT)
        elif args.macro_command == "report":
            report = macro_report(PROJECT_ROOT, period=args.period)
        elif args.macro_command == "explain":
            report = macro_explain(PROJECT_ROOT)
        elif args.macro_command == "compare":
            report = macro_compare(
                PROJECT_ROOT,
                date_a=args.date_a,
                date_b=args.date_b,
            )
        elif args.macro_command == "sector-impact":
            report = macro_sector_impact(PROJECT_ROOT)
        elif args.macro_command == "strategy-impact":
            report = macro_strategy_impact(
                PROJECT_ROOT,
                strategy_id=args.strategy_id,
            )
        elif args.macro_command == "audit":
            report = macro_audit(PROJECT_ROOT)
        elif args.macro_command == "conflicts":
            report = macro_conflicts(PROJECT_ROOT)
        elif args.macro_command == "freeze":
            report = macro_freeze(PROJECT_ROOT)
        else:
            raise SystemExit(f"Unknown macro command: {args.macro_command}")
        _print_json(report)
        return 0 if report["status"] in {
            "GO",
            "DATA_INCOMPLETE",
            "NOT_RUN",
            "READ_ONLY_LIVE_READY_GO",
            "READ_ONLY_LIVE_READY_DEGRADED_DATA",
        } else 2

    if args.command == "sec":
        if args.sec_command == "status":
            report = sec_intelligence_status(PROJECT_ROOT)
        elif args.sec_command == "audit":
            report = sec_intelligence_audit(PROJECT_ROOT)
        else:
            report = sec_overlay_for_signal(
                PROJECT_ROOT,
                symbol=args.symbol,
                as_of=args.as_of,
                base_score=args.base_score,
                base_signal_authorized=args.base_authorized,
            )
        _print_json(report)
        return 0 if report["status"] in {
            "GO",
            "DEGRADED",
            "DEGRADED_NO_CAUSAL_SEC_EVENTS",
        } else 2

    if (
        args.command == "market"
        and args.market_command == "context"
        and args.market_context_command == "collect-realtime"
    ):
        report = collect_realtime_equity_context(
            PROJECT_ROOT,
            env_file=args.env_file,
            config=RealtimeEquityConfig(
                duration_seconds=args.duration_seconds,
                max_symbols=args.max_symbols,
                depth_symbols=args.depth_symbols,
                depth_levels=args.depth_levels,
            ),
        )
        _print_json(report)
        return 0 if report["status"] in {"GO", "GO_DEGRADED_NO_TAPE_OR_DEPTH"} else 2

    context = load_app_context(args.env_file)

    if args.command == "doctor":
        _print_json(build_doctor_report(context, PROJECT_ROOT))
        return 0

    if args.command == "ibkr":
        if args.ibkr_command == "disconnect-drill-preflight":
            report = build_disconnect_drill_preflight_report(
                context,
                require_socket=not args.skip_socket_check,
            )
            _print_json(report)
            return 0 if report["status"] == "GO" else 2

        if args.ibkr_command == "data-capabilities":
            if args.data_capability_command == "schema":
                _print_json(capability_schema())
                return 0
            matrix = build_capability_matrix(PROJECT_ROOT)
            if args.data_capability_command == "status":
                _print_json(matrix)
                return 0 if matrix["status"] in {"GO", "GO_DEGRADED"} else 2
            report = strategy_capability_gate(
                matrix,
                args.require,
                asset_reference=args.asset_reference,
            )
            _print_json(report)
            return 0 if report["status"] == "GO" else 2

        if args.ibkr_command == "news":
            if args.ibkr_news_command == "schema":
                _print_json(ibkr_news_schema())
                return 0
            if args.ibkr_news_command == "capabilities":
                report = probe_ibkr_news(
                    PROJECT_ROOT,
                    env_file=args.connection_env_file,
                )
            else:
                report = collect_ibkr_news(
                    PROJECT_ROOT,
                    symbols=[
                        item.strip().upper()
                        for item in args.symbols.split(",")
                        if item.strip()
                    ],
                    lookback_hours=args.lookback_hours,
                    max_results_per_symbol=args.max_results_per_symbol,
                    env_file=args.connection_env_file,
                )
            _print_json(report)
            return 0 if report["status"] in {
                "AVAILABLE",
                "AVAILABLE_NO_CURRENT_HEADLINES",
                "GO",
                "GO_NO_CURRENT_HEADLINES",
                "PARTIAL",
                "UNAVAILABLE_ENTITLEMENT",
                "UNAVAILABLE_NO_PROVIDER_SUBSCRIPTION",
                "TWS_UNAVAILABLE",
            } else 2

        if args.ibkr_command == "contract":
            phase1 = phase1_freeze_status(PROJECT_ROOT)
            cache_layout = ContractCacheLayout.from_project_root(PROJECT_ROOT)
            if args.contract_command == "build-opportunity-queue":
                report = build_opportunity_contract_queue(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] in {"GO", "GO_EMPTY"} else 2
            if args.contract_command == "resolve-new-live-read-only":
                if not phase1.frozen:
                    _print_json(
                        {
                            "schema": "ibkr_new_live_read_only_contract_resolution_v1",
                            "status": "PHASE1_FREEZE_REQUIRED",
                            "blockers": ["PHASE1_CONNECTION_SERVICE_FROZEN_GO_REQUIRED"],
                            "financial_calls": {
                                "place_order": 0,
                                "cancel_order": 0,
                                "global_cancel": 0,
                            },
                            "execution_authority": "NONE",
                        }
                    )
                    return 2
                report = resolve_new_live_read_only_contracts(
                    PROJECT_ROOT,
                    manifest_file=args.manifest,
                    env_file=args.connection_env_file,
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.contract_command == "refresh-live-read-only":
                if not phase1.frozen:
                    _print_json(
                        {
                            "schema": "ibkr_live_read_only_contract_refresh_v1",
                            "status": "PHASE1_FREEZE_REQUIRED",
                            "blockers": ["PHASE1_CONNECTION_SERVICE_FROZEN_GO_REQUIRED"],
                            "financial_calls": {
                                "place_order": 0,
                                "cancel_order": 0,
                                "global_cancel": 0,
                            },
                            "execution_authority": "NONE",
                        }
                    )
                    return 2
                report = refresh_live_read_only_contracts(
                    PROJECT_ROOT,
                    symbols=[
                        item.strip().upper()
                        for item in args.symbols.split(",")
                        if item.strip()
                    ],
                    env_file=args.connection_env_file,
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.contract_command == "status":
                _print_json(
                    {
                        "schema": "ibkr_contract_resolver_status_v1",
                        "status": "READY" if phase1.frozen else "PHASE1_NOT_FROZEN",
                        "phase1": phase1.as_dict(),
                        "supported_security_types": ["STK", "FUT"],
                        "supported_asset_classes": [
                            "stock",
                            "etf",
                            "bond_etf",
                            "commodity_etf",
                            "commodity_future",
                        ],
                        "data_sources": {
                            "ibkr": {
                                "enabled": phase1.frozen,
                                "authority": "read_only_contract_resolution",
                            },
                            "external_providers": {
                                "enabled": False,
                                "authority": "disabled_for_phase2_contract_identity",
                            },
                        },
                        "cache": empty_contract_manifest(cache_layout),
                        "financial_calls": {
                            "place_order": 0,
                            "cancel_order": 0,
                            "global_cancel": 0,
                        },
                    }
                )
                return 0

            if args.contract_command == "schema":
                _print_json(contract_schema_manifest())
                return 0

            if args.contract_command == "init-cache":
                manifest = initialize_contract_cache(cache_layout)
                _print_json(
                    {
                        "schema": "ibkr_contract_cache_init_v1",
                        "status": "GO",
                        "phase1": phase1.as_dict(),
                        "cache": manifest,
                        "financial_calls": {
                            "place_order": 0,
                            "cancel_order": 0,
                            "global_cancel": 0,
                        },
                    }
                )
                return 0

            if args.contract_command == "validate-cache":
                report = validate_contract_cache(cache_layout)
                _print_json(
                    {
                        "schema": "ibkr_contract_cache_validation_command_v1",
                        "status": report["status"],
                        "phase1": phase1.as_dict(),
                        "cache_validation": report,
                        "financial_calls": {
                            "place_order": 0,
                            "cancel_order": 0,
                            "global_cancel": 0,
                        },
                    }
                )
                return 0 if report["status"] == "GO" else 2

            if args.contract_command == "export-identity":
                cache_validation = validate_contract_cache(cache_layout)
                if cache_validation["status"] != "GO":
                    _print_json(
                        {
                            "schema": "ibkr_contract_identity_export_command_v1",
                            "status": "NO_GO",
                            "phase1": phase1.as_dict(),
                            "cache_validation": cache_validation,
                            "resolved_contract_identity": None,
                            "financial_calls": {
                                "place_order": 0,
                                "cancel_order": 0,
                                "global_cancel": 0,
                            },
                        }
                    )
                    return 2
                try:
                    export = export_contract_identity(cache_layout, args.con_id)
                except ValueError as exc:
                    _print_json(
                        {
                            "schema": "ibkr_contract_identity_export_command_v1",
                            "status": "VALIDATION_ERROR",
                            "phase1": phase1.as_dict(),
                            "reason": str(exc),
                            "resolved_contract_identity": None,
                            "financial_calls": {
                                "place_order": 0,
                                "cancel_order": 0,
                                "global_cancel": 0,
                            },
                        }
                    )
                    return 2
                _print_json(
                    {
                        "schema": "ibkr_contract_identity_export_command_v1",
                        "status": export["status"],
                        "phase1": phase1.as_dict(),
                        "export": export,
                        "financial_calls": {
                            "place_order": 0,
                            "cancel_order": 0,
                            "global_cancel": 0,
                        },
                    }
                )
                return 0 if export["status"] == "GO" else 2

            if args.contract_command == "resolve-stock":
                request = ContractResolutionRequest(
                    symbol=args.symbol,
                    asset_class=AssetClass(args.asset_class),
                    security_type=IbkrSecurityType.STK,
                    currency=args.currency,
                    exchange=args.exchange,
                    primary_exchange=args.primary_exchange,
                )
            elif args.contract_command == "resolve-future":
                request = ContractResolutionRequest(
                    symbol=args.symbol,
                    asset_class=AssetClass.COMMODITY_FUTURE,
                    security_type=IbkrSecurityType.FUT,
                    currency=args.currency,
                    exchange=args.exchange,
                    expiry=args.expiry,
                )
            else:
                raise SystemExit(f"Unknown contract command: {args.contract_command}")

            prepared_spec: dict[str, Any] | None
            spec_error: str | None
            try:
                prepared_spec = build_ibkr_contract_spec(request).as_dict()
            except ValueError as exc:
                prepared_spec = None
                spec_error = str(exc)
            else:
                spec_error = None

            if not phase1.frozen:
                resolution_report = gated_contract_resolution_report(request, phase1)
                payload = resolution_report.as_dict()
                payload["prepared_ibkr_contract_spec"] = prepared_spec
                if spec_error:
                    payload["contract_spec_error"] = spec_error
                _print_json(payload)
                return 0 if resolution_report.status.value == "RESOLVED" else 2

            service = build_ibkr_service(context)
            resolver = LiveContractResolver(service, cache_layout)
            payload = resolver.resolve(request).as_dict()
            payload["phase1"] = phase1.as_dict()
            payload["prepared_ibkr_contract_spec"] = prepared_spec
            if spec_error:
                payload["contract_spec_error"] = spec_error
            _print_json(payload)
            return 0 if payload["status"] == "RESOLVED" else 2

        if args.ibkr_command == "phase8":
            if args.phase8_command == "schema":
                _print_json(phase8_schema(PROJECT_ROOT))
                return 0
            if args.phase8_command == "preflight":
                report = phase8_preflight(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_command == "snapshot":
                report = phase8_snapshot(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_command == "stability-check":
                report = phase8_stability_check(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_command == "reconcile":
                report = phase8_reconcile(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_command == "audit":
                report = phase8_audit(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_command == "status":
                report = phase8_status(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["status"] == "PHASE8_IBKR_READ_ONLY_RECONCILIATION_ADAPTER_GO" else 2
            if args.phase8_command == "freeze":
                report = phase8_freeze(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE8_IBKR_READ_ONLY_RECONCILIATION_ADAPTER_FROZEN_GO" else 2
            raise SystemExit(f"Unknown phase8 command: {args.phase8_command}")

        if args.ibkr_command == "phase8-1":
            if args.phase8_1_command == "schema":
                _print_json(phase8_1_schema(PROJECT_ROOT))
                return 0
            if args.phase8_1_command == "preflight":
                report = phase8_1_preflight(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_1_command == "establish-baseline":
                report = establish_baseline(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["baseline_status"] == "BASELINE_STABLE_GO" else 2
            if args.phase8_1_command == "soak":
                report = run_soak(
                    PROJECT_ROOT,
                    args.env_file,
                    duration_seconds=args.duration_seconds,
                    interval_seconds=args.interval_seconds,
                    stability_delay_seconds=args.stability_delay_seconds,
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_1_command == "recovery-drill":
                report = recovery_drill(
                    PROJECT_ROOT,
                    args.env_file,
                    duration_seconds=args.duration_seconds,
                    poll_seconds=args.poll_seconds,
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_1_command == "audit":
                report = phase8_1_audit(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase8_1_command == "status":
                report = phase8_1_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE8_1_READ_ONLY_OBSERVATION_SOAK_AND_RECOVERY_GO" else 2
            if args.phase8_1_command == "freeze":
                report = phase8_1_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE8_1_READ_ONLY_OBSERVATION_SOAK_AND_RECOVERY_FROZEN_GO" else 2
            raise SystemExit(f"Unknown phase8-1 command: {args.phase8_1_command}")

        if args.ibkr_command == "phase9":
            if args.phase9_command == "schema":
                _print_json(phase9_schema(PROJECT_ROOT))
                return 0
            if args.phase9_command == "preflight":
                report = phase9_preflight(PROJECT_ROOT, args.env_file)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "prepare":
                report = phase9_prepare(
                    PROJECT_ROOT,
                    args.env_file,
                    con_id=args.con_id,
                    side=args.side,
                    quantity=args.quantity,
                    limit_price=args.limit_price,
                    reason=args.reason,
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "approve":
                report = phase9_approve(PROJECT_ROOT, args.env_file, intent_id=args.intent_id, approval_text=args.approval)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "submit":
                report = phase9_submit(PROJECT_ROOT, args.env_file, intent_id=args.intent_id)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "prepare-cancel":
                report = phase9_prepare_cancel(PROJECT_ROOT, args.env_file, intent_id=args.intent_id)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "approve-cancel":
                report = phase9_approve_cancel(PROJECT_ROOT, args.env_file, intent_id=args.intent_id, approval_text=args.approval)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "cancel":
                report = phase9_cancel(PROJECT_ROOT, args.env_file, intent_id=args.intent_id)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "reconcile":
                report = phase9_reconcile(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "observe-known-fill":
                report = phase9_observe_known_fill(
                    PROJECT_ROOT, intent_id=args.intent_id
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "fill-close-audit":
                report = phase9_fill_close_audit(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE9_0_1_FILL_ADOPTION_AND_CLOSE_RECONCILIATION_GO" else 2
            if args.phase9_command == "canary-b-readiness":
                report = phase9_canary_b_readiness(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE9_CANARY_B_READY" else 2
            if args.phase9_command == "canary-a-evidence":
                report = phase9_canary_a_evidence(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "CANARY_A_EVIDENCE_GO" else 2
            if args.phase9_command == "canary-results":
                report = phase9_canary_results(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "accept-manual-completion":
                report = accept_operator_attested_manual_completion(
                    PROJECT_ROOT,
                    symbol=args.symbol,
                    con_id=args.con_id,
                    reason=args.reason,
                )
                _print_json(report)
                return 0 if report["status"].endswith("_GO") else 2
            if args.phase9_command == "audit":
                report = phase9_audit(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase9_command == "status":
                report = phase9_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_GO" else 2
            if args.phase9_command == "freeze":
                report = phase9_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE9_IBKR_MANUAL_PAPER_EXECUTION_ADAPTER_FROZEN_GO" else 2
            raise SystemExit(f"Unknown phase9 command: {args.phase9_command}")

        if args.ibkr_command == "phase10":
            report = phase10_command(PROJECT_ROOT, args.phase10_command, args.env_file)
            _print_json(report)
            success = (
                report.get("status") in {"GO", PHASE10_MARKER}
                or report.get("freeze_status") == PHASE10_FREEZE_MARKER
            )
            return 0 if success else 2

        if args.ibkr_command == "phase11-2":
            report = phase11_2_command(PROJECT_ROOT, args.phase11_2_command)
            _print_json(report)
            success = (
                report.get("status") in {"GO", PHASE11_2_MARKER}
                or report.get("freeze_status") == PHASE11_2_FREEZE_MARKER
            )
            return 0 if success else 2

        if args.ibkr_command == "phase11-3":
            report = phase11_3_command(PROJECT_ROOT, args.phase11_3_command)
            _print_json(report)
            success = (
                report.get("status") in {"GO", "PARTIAL", PHASE11_3_MARKER, "PHASE11_3_DATASCRAPER_INTEGRATION_GO"}
                or report.get("freeze_status") == PHASE11_3_FREEZE_MARKER
            )
            return 0 if success else 2

        service = build_ibkr_service(context)
        if args.ibkr_command == "probe":
            snapshot = service.probe()
        elif args.ibkr_command == "status":
            snapshot = service.status()
        elif args.ibkr_command == "watch":
            snapshot = service.watch(seconds=args.seconds)
        elif args.ibkr_command == "cycle":
            report = service.run_connect_disconnect_cycles(count=args.count)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        elif args.ibkr_command == "duplicate-client-check":
            report = service.duplicate_client_check()
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        elif args.ibkr_command == "disconnect-drill":
            report = service.forced_disconnect_drill(
                seconds=args.seconds,
                poll_seconds=args.poll_seconds,
            )
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        else:
            raise SystemExit(f"Unknown ibkr command: {args.ibkr_command}")

        _print_json(snapshot.as_dict())
        return 0 if snapshot.status.value in {"HEALTHY", "DEGRADED", "DISCONNECTED"} else 2

    if args.command == "market":
        if args.market_command == "context":
            if args.market_context_command == "schema":
                report = market_context_schema(PROJECT_ROOT)
            elif args.market_context_command == "audit":
                report = audit_market_context_sources(PROJECT_ROOT)
            elif args.market_context_command == "build":
                report = build_market_context(
                    PROJECT_ROOT,
                    symbols=parse_symbols(args.symbols),
                    fetch_options=not args.no_network,
                    max_expirations=args.max_expirations,
                )
            elif args.market_context_command == "status":
                report = market_context_status(PROJECT_ROOT)
            elif args.market_context_command == "cot-update":
                report = collect_cot_context(
                    PROJECT_ROOT,
                    start=args.start,
                    fetch=not args.no_network,
                )
            elif args.market_context_command == "cot-status":
                report = cot_status(PROJECT_ROOT)
            elif args.market_context_command == "transmission":
                report = build_asset_context(PROJECT_ROOT)
            elif args.market_context_command == "observe":
                report = observe_shortlist(
                    PROJECT_ROOT,
                    max_symbols=args.max_symbols,
                    depth_symbols=args.depth_symbols,
                )
            elif args.market_context_command == "observer-status":
                report = entry_observer_status(PROJECT_ROOT)
            elif args.market_context_command == "settle-episodes":
                report = settle_entry_episodes(PROJECT_ROOT)
            elif args.market_context_command == "episode-status":
                report = episode_outcome_status(PROJECT_ROOT)
            elif args.market_context_command == "collect-realtime":
                report = collect_realtime_equity_context(
                    PROJECT_ROOT,
                    env_file=args.env_file,
                    config=RealtimeEquityConfig(
                        duration_seconds=args.duration_seconds,
                        max_symbols=args.max_symbols,
                        depth_symbols=args.depth_symbols,
                        depth_levels=args.depth_levels,
                    ),
                )
            else:
                raise SystemExit(
                    "Unknown market context command: "
                    f"{args.market_context_command}"
                )
            _print_json(report)
            return (
                0
                if report["status"]
                in {
                    "GO",
                    "GO_DEGRADED",
                    "NO_CURRENT_SETUPS",
                    "NO_INCREMENTAL_OBSERVED_FLOW_EVIDENCE",
                }
                else 2
            )

        phase1 = phase1_freeze_status(PROJECT_ROOT)
        if args.market_command == "sessions" and args.sessions_command == "schema":
            _print_json(market_session_schema_manifest())
            return 0

        cache_layout = ContractCacheLayout.from_project_root(PROJECT_ROOT)
        cache_validation = validate_contract_cache(cache_layout)
        if cache_validation["status"] != "GO":
            _print_json(
                {
                    "schema": "market_command_v1",
                    "status": "NO_GO",
                    "phase1": phase1.as_dict(),
                    "cache_validation": cache_validation,
                    "financial_calls": _zero_financial_calls(),
                }
            )
            return 2

        rows = read_contract_cache_rows(cache_layout)
        try:
            if args.market_command == "status":
                row = _cached_row_by_con_id(rows, args.con_id)
                report = market_status_report(row, at=_parse_iso_datetime(args.at) if args.at else None)
            elif args.market_command == "next-open":
                row = _cached_row_by_con_id(rows, args.con_id)
                report = market_next_open_report(row, at=_parse_iso_datetime(args.at) if args.at else None)
            elif args.market_command == "sessions":
                report = _handle_market_sessions_command(args, rows)
            else:
                raise SystemExit(f"Unknown market command: {args.market_command}")
        except LookupError as exc:
            _print_json(
                {
                    "schema": "market_command_v1",
                    "status": "NOT_FOUND",
                    "phase1": phase1.as_dict(),
                    "reason": str(exc),
                    "source": "local_contract_cache",
                    "financial_calls": _zero_financial_calls(),
                }
            )
            return 2
        except ValueError as exc:
            _print_json(
                {
                    "schema": "market_command_v1",
                    "status": "VALIDATION_ERROR",
                    "phase1": phase1.as_dict(),
                    "con_id": args.con_id,
                    "reason": str(exc),
                    "financial_calls": _zero_financial_calls(),
                }
            )
            return 2

        report["phase1"] = phase1.as_dict()
        _print_json(report)
        return 0 if report["status"] == "GO" else 2

    if args.command == "data":
        if args.data_command == "sources":
            report = (
                comprehensive_data_readiness(PROJECT_ROOT)
                if args.data_sources_command == "readiness"
                else data_source_status(PROJECT_ROOT)
            )
            _print_json(report)
            return 0 if report["status"] in {"GO", "CORE_RESEARCH_DATA_GO_WITH_DOCUMENTED_GAPS"} else 2
        phase1 = phase1_freeze_status(PROJECT_ROOT)
        if args.data_command == "corporate-actions":
            layout = CorporateActionLayout.from_project_root(PROJECT_ROOT)
            if args.corporate_actions_command == "schema":
                _print_json(corporate_action_schema())
                return 0
            if args.corporate_actions_command == "status":
                report = corporate_action_status(layout)
                report["phase1"] = phase1.as_dict()
                _print_json(report)
                return 0 if report["status"] == "READY" else 2
            if args.corporate_actions_command == "validate-cache":
                report = validate_corporate_action_cache(layout)
                _print_json(
                    {
                        "schema": "corporate_action_cache_validation_command_v1",
                        "status": report["status"],
                        "phase1": phase1.as_dict(),
                        "cache_validation": report,
                        "financial_calls": _zero_financial_calls(),
                    }
                )
                return 0 if report["status"] == "GO" else 2
            if args.corporate_actions_command == "collect":
                contract_layout = ContractCacheLayout.from_project_root(PROJECT_ROOT)
                contract_validation = validate_contract_cache(contract_layout)
                if not phase1.frozen or contract_validation["status"] != "GO":
                    _print_json(
                        {
                            "schema": "corporate_action_collection_command_v1",
                            "status": "NO_GO",
                            "phase1": phase1.as_dict(),
                            "contract_cache_validation": contract_validation,
                            "financial_calls": _zero_financial_calls(),
                        }
                    )
                    return 2
                report = collect_corporate_actions_for_universe(
                    project_root=PROJECT_ROOT,
                    rows=read_contract_cache_rows(contract_layout),
                    start=_parse_iso_date(args.start),
                    end=_parse_iso_date(args.end),
                    env_file=args.env_file,
                )
                report["phase1"] = phase1.as_dict()
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            raise SystemExit(f"Unknown corporate-actions command: {args.corporate_actions_command}")

        if args.data_command == "fx":
            fx_layout = FxCacheLayout.from_project_root(PROJECT_ROOT)
            if args.fx_command == "schema":
                _print_json(fx_schema())
                return 0
            if args.fx_command == "status":
                report = fx_status_v1_1(fx_layout)
                report["phase1"] = phase1.as_dict()
                _print_json(report)
                return 0 if report["status"] == "READY" else 2
            if args.fx_command == "validate-cache":
                report = validate_fx_cache(fx_layout)
                _print_json(
                    {
                        "schema": "fx_cache_validation_command_v1",
                        "status": report["status"],
                        "phase1": phase1.as_dict(),
                        "cache_validation": report,
                        "financial_calls": _zero_financial_calls(),
                    }
                )
                return 0 if report["status"] == "GO" else 2
            if args.fx_command == "collect":
                contract_layout = ContractCacheLayout.from_project_root(PROJECT_ROOT)
                contract_validation = validate_contract_cache(contract_layout)
                if not phase1.frozen or contract_validation["status"] != "GO":
                    _print_json(
                        {
                            "schema": "fx_collection_command_v1",
                            "status": "NO_GO",
                            "phase1": phase1.as_dict(),
                            "contract_cache_validation": contract_validation,
                            "financial_calls": _zero_financial_calls(),
                        }
                    )
                    return 2
                report = collect_fx_for_universe_v1_1(
                    project_root=PROJECT_ROOT,
                    rows=read_contract_cache_rows(contract_layout),
                    start=_parse_iso_date(args.start),
                    end=_parse_iso_date(args.end),
                    base_currency=args.base,
                    env_file=args.env_file,
                )
                report["phase1"] = phase1.as_dict()
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            raise SystemExit(f"Unknown fx command: {args.fx_command}")

        if args.data_command == "total-returns":
            total_return_layout = TotalReturnLayout.from_project_root(PROJECT_ROOT)
            if args.total_returns_command == "schema":
                _print_json(total_return_schema())
                return 0
            if args.total_returns_command == "status":
                report = total_return_status(total_return_layout)
                report["phase1"] = phase1.as_dict()
                _print_json(report)
                return 0 if report["status"] == "READY" else 2
            if args.total_returns_command == "validate-cache":
                report = validate_total_return_cache(total_return_layout)
                _print_json(
                    {
                        "schema": "total_return_cache_validation_command_v1",
                        "status": report["status"],
                        "phase1": phase1.as_dict(),
                        "cache_validation": report,
                        "financial_calls": _zero_financial_calls(),
                    }
                )
                return 0 if report["status"] == "GO" else 2
            if args.total_returns_command == "build":
                contract_layout = ContractCacheLayout.from_project_root(PROJECT_ROOT)
                contract_validation = validate_contract_cache(contract_layout)
                ca_validation = validate_corporate_action_cache(CorporateActionLayout.from_project_root(PROJECT_ROOT))
                fx_validation = validate_fx_cache(FxCacheLayout.from_project_root(PROJECT_ROOT))
                if (
                    not phase1.frozen
                    or contract_validation["status"] != "GO"
                    or ca_validation["status"] != "GO"
                    or fx_validation["status"] != "GO"
                    or args.interval != "1d"
                ):
                    _print_json(
                        {
                            "schema": "total_return_build_command_v1",
                            "status": "NO_GO",
                            "phase1": phase1.as_dict(),
                            "contract_cache_validation": contract_validation,
                            "corporate_action_validation": ca_validation,
                            "fx_validation": fx_validation,
                            "financial_calls": _zero_financial_calls(),
                        }
                    )
                    return 2
                report = build_total_returns_for_universe_v1_1(
                    project_root=PROJECT_ROOT,
                    rows=read_contract_cache_rows(contract_layout),
                    base_currency=args.base_currency,
                )
                report["phase1"] = phase1.as_dict()
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            raise SystemExit(f"Unknown total-returns command: {args.total_returns_command}")

        if args.data_command == "multitimeframe":
            if args.multitimeframe_command == "schema":
                _print_json(multitimeframe_schema(PROJECT_ROOT))
                return 0
            if args.multitimeframe_command == "inventory":
                _print_json(provider_inventory(PROJECT_ROOT))
                return 0
            if args.multitimeframe_command == "import-local":
                report = collect_multitimeframe_data(
                    PROJECT_ROOT,
                    symbols=parse_symbols(",".join(DEFAULT_SYMBOLS)),
                    intervals=parse_intervals(",".join(CANONICAL_INTERVALS)),
                    providers=["local", "datascraper", "ibkr"],
                    start="2000-01-01",
                    end=date.today().isoformat(),
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.multitimeframe_command == "normalize":
                report = validate_multitimeframe_cache(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.multitimeframe_command == "coverage":
                report = phase11_6_data_audit(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.multitimeframe_command == "collect":
                report = collect_multitimeframe_data(
                    PROJECT_ROOT,
                    symbols=parse_symbols(args.symbols),
                    intervals=parse_intervals(args.intervals),
                    providers=[item for item in args.providers.split(",") if item.strip()],
                    start=args.start,
                    end=args.end,
                    lookback_days=args.lookback_days,
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.multitimeframe_command == "validate-cache":
                report = validate_multitimeframe_cache(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.multitimeframe_command == "audit":
                _print_json(audit_multitimeframe_sources(PROJECT_ROOT))
                return 0
            if args.multitimeframe_command == "status":
                report = multitimeframe_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "MULTI_TIMEFRAME_DATA_GO" else 2
            raise SystemExit(f"Unknown multitimeframe command: {args.multitimeframe_command}")

        if args.data_command != "bars":
            raise SystemExit(f"Unknown data command: {args.data_command}")

        bar_layout = BarCacheLayout.from_project_root(PROJECT_ROOT)
        if args.bars_command == "schema":
            _print_json(bar_schema_manifest())
            return 0
        if args.bars_command == "status":
            contract_layout = ContractCacheLayout.from_project_root(PROJECT_ROOT)
            session_layout = SessionCacheLayout.from_project_root(PROJECT_ROOT)
            contract_cache_validation = validate_contract_cache(contract_layout)
            session_cache_validation = validate_session_cache(session_layout)
            _print_json(
                {
                    "schema": "historical_bar_data_status_v1",
                    "status": "READY"
                    if phase1.frozen
                    and contract_cache_validation["status"] == "GO"
                    and session_cache_validation["status"] == "GO"
                    else "NO_GO",
                    "phase1": phase1.as_dict(),
                    "phase2_contract_identity_required": True,
                    "phase3_market_sessions_required": True,
                    "contract_cache_validation": contract_cache_validation,
                    "session_cache_validation": session_cache_validation,
                    "data_sources": {
                        "ibkr": {
                            "enabled": True,
                            "authority": "read_only_historical_bars_daily_stk_only",
                        },
                        "external_providers": {
                            "enabled": False,
                            "authority": "disabled_until_explicit_data_phase",
                        },
                    },
                    "cache_layout": {
                        "data_dir": str(bar_layout.data_dir),
                        "partitioning": bar_schema_manifest()["partitioning"],
                    },
                    "financial_calls": _zero_financial_calls(),
                    "market_data_streaming_calls": 0,
                }
            )
            return 0
        if args.bars_command == "init-cache":
            manifest = initialize_bar_cache(bar_layout)
            _print_json(
                {
                    "schema": "historical_bar_cache_init_command_v1",
                    "status": "GO",
                    "phase1": phase1.as_dict(),
                    "phase4": {
                        "enabled": False,
                        "authority": "disabled_until_phase4",
                    },
                    "cache": manifest,
                    "financial_calls": _zero_financial_calls(),
                }
            )
            return 0
        if args.bars_command == "validate-cache":
            report = validate_bar_cache(bar_layout)
            bar_contract_cache_validation: dict[str, Any] | None = None
            contract_rows: list[Any] = []
            if report["status"] == "GO" and report["file_count"] > 0:
                contract_layout = ContractCacheLayout.from_project_root(PROJECT_ROOT)
                bar_contract_cache_validation = validate_contract_cache(contract_layout)
                if bar_contract_cache_validation["status"] == "GO":
                    contract_rows = read_contract_cache_rows(contract_layout)
            contract_identity_links = validate_bar_contract_identity_links(
                bar_layout,
                contract_rows,
                bar_validation=report,
            )
            status = "GO" if report["status"] == "GO" and contract_identity_links["status"] == "GO" else "NO_GO"
            payload = {
                "schema": "historical_bar_cache_validation_command_v1",
                "status": status,
                "phase1": phase1.as_dict(),
                "phase4": {
                    "enabled": True,
                    "authority": "read_only_historical_bars_daily_stk_only",
                },
                "cache_validation": report,
                "contract_cache_validation": bar_contract_cache_validation,
                "contract_identity_links": contract_identity_links,
                "financial_calls": _zero_financial_calls(),
                "market_data_streaming_calls": 0,
            }
            bars_output_dir = PROJECT_ROOT / "output" / "ibkr" / "bars"
            bars_output_dir.mkdir(parents=True, exist_ok=True)
            (bars_output_dir / "cache-validation.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n",
                encoding="utf-8",
            )
            _print_json(payload)
            return 0 if status == "GO" else 2
        if args.bars_command == "collect":
            if not phase1.frozen:
                _print_json(
                    {
                        "schema": "historical_bar_collection_command_v1",
                        "status": "PHASE1_NOT_FROZEN",
                        "phase1": phase1.as_dict(),
                        "financial_calls": _zero_financial_calls(),
                    }
                )
                return 2
            if args.interval != "1d" or args.data_type != "TRADES":
                _print_json(
                    {
                        "schema": "historical_bar_collection_command_v1",
                        "status": "VALIDATION_ERROR",
                        "reason": "Phase 4 V1 only supports --interval 1d --data-type TRADES",
                        "financial_calls": _zero_financial_calls(),
                    }
                )
                return 2
            contract_layout = ContractCacheLayout.from_project_root(PROJECT_ROOT)
            session_layout = SessionCacheLayout.from_project_root(PROJECT_ROOT)
            contract_cache_validation = validate_contract_cache(contract_layout)
            session_cache_validation = validate_session_cache(session_layout)
            if contract_cache_validation["status"] != "GO" or session_cache_validation["status"] != "GO":
                _print_json(
                    {
                        "schema": "historical_bar_collection_command_v1",
                        "status": "NO_GO",
                        "phase1": phase1.as_dict(),
                        "contract_cache_validation": contract_cache_validation,
                        "session_cache_validation": session_cache_validation,
                        "financial_calls": _zero_financial_calls(),
                    }
                )
                return 2
            contract_rows = read_contract_cache_rows(contract_layout)
            row = _cached_row_by_con_id(contract_rows, args.con_id)
            if row.contract.security_type != IbkrSecurityType.STK:
                _print_json(
                    {
                        "schema": "historical_bar_collection_command_v1",
                        "status": "VALIDATION_ERROR",
                        "reason": "Phase 4 V1 only supports STK contracts",
                        "financial_calls": _zero_financial_calls(),
                    }
                )
                return 2
            report = collect_ibkr_daily_bars(
                settings=context.ibkr,
                layout=bar_layout,
                contract_row=row,
                session_rows=read_session_cache_records(session_layout),
                start=_parse_iso_date(args.start),
                end=_parse_iso_date(args.end),
            )
            report["phase1"] = phase1.as_dict()
            report["contract_cache_validation"] = {
                "status": contract_cache_validation["status"],
                "row_count": contract_cache_validation["row_count"],
            }
            report["session_cache_validation"] = {
                "status": session_cache_validation["status"],
                "row_count": session_cache_validation["row_count"],
            }
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.bars_command == "request-policy":
            _print_json(
                {
                    "schema": "historical_bar_request_policy_command_v1",
                    "status": "GO",
                    "phase1": phase1.as_dict(),
                    "phase4": {
                        "enabled": True,
                        "authority": "read_only_historical_bars_daily_stk_only",
                    },
                    "request_policy": BarRequestPolicy().as_dict(),
                    "execution_enabled": True,
                    "financial_calls": _zero_financial_calls(),
                }
            )
            return 0
        raise SystemExit(f"Unknown bars command: {args.bars_command}")

    if args.command == "research":
        if args.research_command == "rsi-pit":
            if args.rsi_pit_command == "acquire-pit-data":
                report = acquire_price_histories(
                    PROJECT_ROOT,
                    max_symbols=args.max_symbols,
                    workers=args.workers,
                    requests_per_second=args.requests_per_second,
                )
            elif args.rsi_pit_command == "compact-pit-data":
                report = compact_price_histories(PROJECT_ROOT)
            else:
                report = phase11_4_command(PROJECT_ROOT, args.rsi_pit_command)
            _print_json(report)
            return 0

        if args.research_command == "critical-trading":
            if args.critical_trading_command == "schema":
                _print_json(critical_trading_schema())
                return 0
            if args.critical_trading_command == "run":
                report = run_critical_trading_backtests(
                    PROJECT_ROOT,
                    stock_limit=args.stock_limit,
                    include_yfinance=args.include_yfinance,
                )
                _print_json(report)
                return 0 if report["status"] == "RESEARCH_RESULTS_AVAILABLE" else 2
            if args.critical_trading_command == "yfinance-collect":
                report = collect_yfinance_data(PROJECT_ROOT, start=args.start, end=args.end)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.critical_trading_command == "perfect":
                report = run_perfection_pipeline(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "ROBUSTNESS_RESULTS_AVAILABLE" else 2
            raise SystemExit(f"Unknown critical-trading command: {args.critical_trading_command}")

        if args.research_command == "phase6":
            output_dir = PROJECT_ROOT / "output" / "research" / "phase6"
            output_dir.mkdir(parents=True, exist_ok=True)
            if args.phase6_command == "schema":
                _print_json(phase6_schema())
                return 0
            if args.phase6_command == "run":
                report = run_phase6_pipeline(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_BASELINES_AND_STRATEGY_EVIDENCE_GO" else 2
            if args.phase6_command == "status":
                report = phase6_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_BASELINES_AND_STRATEGY_EVIDENCE_GO" else 2
            if args.phase6_command == "freeze":
                report = phase6_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE6_BASELINES_AND_STRATEGY_EVIDENCE_FROZEN_GO" else 2

            dataset = load_phase6_dataset(PROJECT_ROOT)
            if args.phase6_command == "dataset-audit":
                report = dataset_audit(dataset)
                (output_dir / "dataset-audit.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=True, default=str) + "\n",
                    encoding="utf-8",
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase6_command == "benchmarks":
                report = run_baselines(dataset)
                (output_dir / "benchmarks.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=True, default=str) + "\n",
                    encoding="utf-8",
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase6_command == "strategy-grid":
                report = run_strategy_grid(dataset)
                (output_dir / "strategy-grid.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=True, default=str) + "\n",
                    encoding="utf-8",
                )
                _print_json(report)
                return 0 if report["status"] == "GO" and report["config_count"] == 108 else 2
            if args.phase6_command == "walk-forward":
                report = run_walk_forward(dataset)
                (output_dir / "walk-forward.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=True, default=str) + "\n",
                    encoding="utf-8",
                )
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            raise SystemExit(f"Unknown phase6 command: {args.phase6_command}")

        if args.research_command == "phase6-1":
            Phase61Layout.from_project_root(PROJECT_ROOT).output_dir.mkdir(parents=True, exist_ok=True)
            if args.phase6_1_command == "schema":
                _print_json(phase6_1_schema())
                return 0
            if args.phase6_1_command == "run":
                report = run_phase6_1_pipeline(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_GO" else 2
            if args.phase6_1_command == "status":
                report = phase6_1_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_GO" else 2
            if args.phase6_1_command == "freeze":
                report = phase6_1_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE6_1_ROBUSTNESS_AND_FAILURE_ATTRIBUTION_FROZEN_GO" else 2
            raise SystemExit(f"Unknown phase6-1 command: {args.phase6_1_command}")

        if args.research_command == "phase6-2":
            Phase62Layout.from_project_root(PROJECT_ROOT).output_dir.mkdir(parents=True, exist_ok=True)
            if args.phase6_2_command == "schema":
                _print_json(phase6_2_schema())
                return 0
            if args.phase6_2_command == "run":
                report = run_phase6_2_pipeline(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_2_SAMPLE_SUFFICIENCY_AND_FORWARD_OOS_GO" else 2
            if args.phase6_2_command == "status":
                report = phase6_2_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_2_SAMPLE_SUFFICIENCY_AND_FORWARD_OOS_GO" else 2
            if args.phase6_2_command == "freeze":
                report = phase6_2_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE6_2_SAMPLE_SUFFICIENCY_AND_FORWARD_OOS_FROZEN_GO" else 2
            raise SystemExit(f"Unknown phase6-2 command: {args.phase6_2_command}")

        if args.research_command == "phase6-3":
            Phase63Layout.from_project_root(PROJECT_ROOT).output_dir.mkdir(parents=True, exist_ok=True)
            if args.phase6_3_command == "schema":
                _print_json(phase6_3_schema())
                return 0
            if args.phase6_3_command == "run":
                report = run_phase6_3_pipeline(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_3_BENCHMARK_CHAMPION_AND_INCREMENTAL_ALPHA_GO" else 2
            if args.phase6_3_command == "status":
                report = phase6_3_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_3_BENCHMARK_CHAMPION_AND_INCREMENTAL_ALPHA_GO" else 2
            if args.phase6_3_command == "freeze":
                report = phase6_3_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE6_3_BENCHMARK_CHAMPION_AND_INCREMENTAL_ALPHA_FROZEN_GO" else 2
            raise SystemExit(f"Unknown phase6-3 command: {args.phase6_3_command}")

        if args.research_command == "phase6-4":
            Phase64Layout.from_project_root(PROJECT_ROOT).output_dir.mkdir(parents=True, exist_ok=True)
            if args.phase6_4_command == "schema":
                _print_json(phase6_4_schema())
                return 0
            if args.phase6_4_command == "preregister":
                report = preregister_phase6_4(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase6_4_command == "run":
                report = run_phase6_4_pipeline(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_4_PREREGISTERED_MECHANISM_RESEARCH_AND_FORWARD_SELECTION_GO" else 2
            if args.phase6_4_command == "status":
                report = phase6_4_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "PHASE6_4_PREREGISTERED_MECHANISM_RESEARCH_AND_FORWARD_SELECTION_GO" else 2
            if args.phase6_4_command == "freeze":
                report = phase6_4_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == "PHASE6_4_PREREGISTERED_MECHANISM_RESEARCH_AND_FORWARD_SELECTION_FROZEN_GO" else 2
            raise SystemExit(f"Unknown phase6-4 command: {args.phase6_4_command}")

        if args.research_command == "phase11-1":
            Phase111Layout.from_project_root(PROJECT_ROOT).output_dir.mkdir(parents=True, exist_ok=True)
            if args.phase11_1_command == "schema":
                _print_json(phase11_1_schema(PROJECT_ROOT))
                return 0
            if args.phase11_1_command == "preregister":
                report = preregister_phase11_1(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == "GO" else 2
            if args.phase11_1_command == "run":
                report = run_phase11_1_pipeline(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == PHASE11_1_MARKER else 2
            if args.phase11_1_command == "status":
                report = phase11_1_status(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["status"] == PHASE11_1_MARKER else 2
            if args.phase11_1_command == "freeze":
                report = phase11_1_freeze(PROJECT_ROOT)
                _print_json(report)
                return 0 if report["freeze_status"] == PHASE11_1_FREEZE_MARKER else 2
            raise SystemExit(f"Unknown phase11-1 command: {args.phase11_1_command}")

        if args.research_command == "phase11-6":
            if args.phase11_6_command == "schema":
                report = phase11_6_schema(PROJECT_ROOT)
            elif args.phase11_6_command == "data-audit":
                report = phase11_6_data_audit(PROJECT_ROOT)
            elif args.phase11_6_command == "walk-forward":
                report = phase11_6_walk_forward(
                    PROJECT_ROOT,
                    max_identities=args.max_walk_forward_identities,
                )
            elif args.phase11_6_command == "cohorts":
                report = phase11_6_cohorts(
                    PROJECT_ROOT,
                    max_identities=args.max_walk_forward_identities,
                )
            elif args.phase11_6_command == "combine":
                report = phase11_6_combinations(
                    PROJECT_ROOT,
                    max_identities=args.max_combination_identities,
                )
            elif args.phase11_6_command == "audit":
                report = phase11_6_completion_audit(PROJECT_ROOT)
            elif args.phase11_6_command == "status":
                report = phase11_6_status(PROJECT_ROOT)
            elif args.phase11_6_command == "run":
                report = run_phase11_6(
                    PROJECT_ROOT,
                    max_walk_forward_identities=args.max_walk_forward_identities,
                    max_combination_identities=args.max_combination_identities,
                )
            else:
                raise SystemExit(f"Unknown phase11-6 command: {args.phase11_6_command}")
            _print_json(report)
            return 0 if report.get("status") in {"GO", "PARTIAL"} or report.get("audit", {}).get("status") in {"GO", "PARTIAL"} else 2

        if args.research_command == "phase11-7":
            if args.phase11_7_command == "schema":
                report = phase11_7_schema(PROJECT_ROOT)
            elif args.phase11_7_command == "run":
                report = phase11_7_run(
                    PROJECT_ROOT,
                    max_identities=args.max_identities,
                    bootstrap_runs=args.bootstrap_runs,
                )
            elif args.phase11_7_command == "rotation":
                report = phase11_7_rotation(
                    PROJECT_ROOT,
                    max_identities=args.max_identities,
                    bootstrap_runs=args.bootstrap_runs,
                )
            elif args.phase11_7_command == "status":
                report = phase11_7_status(PROJECT_ROOT)
            else:
                raise SystemExit(
                    f"Unknown phase11-7 command: {args.phase11_7_command}"
                )
            _print_json(report)
            return 0 if report.get("status") in {"GO", "NOT_RUN"} else 2

        if args.research_command == "phase11-8":
            if args.phase11_8_command == "schema":
                report = phase11_8_schema(PROJECT_ROOT)
            elif args.phase11_8_command == "data-coverage":
                report = phase11_8_data_coverage(PROJECT_ROOT)
            elif args.phase11_8_command == "portfolio-audit":
                report = phase11_8_portfolio_audit(PROJECT_ROOT)
            elif args.phase11_8_command == "run":
                report = phase11_8_run(
                    PROJECT_ROOT,
                    max_stock_identities=args.max_stock_identities,
                )
            elif args.phase11_8_command == "finalize":
                report = phase11_8_finalize(PROJECT_ROOT)
            elif args.phase11_8_command == "status":
                report = phase11_8_status(PROJECT_ROOT)
            else:
                raise SystemExit(
                    f"Unknown phase11-8 command: {args.phase11_8_command}"
                )
            _print_json(report)
            return 0 if report.get("status") in {"GO", "PARTIAL", "NOT_RUN"} else 2

        if args.research_command == "phase11-9":
            if args.phase11_9_command == "schema":
                report = phase11_9_schema(PROJECT_ROOT)
            elif args.phase11_9_command == "run":
                report = phase11_9_run(PROJECT_ROOT)
            elif args.phase11_9_command == "diagnose":
                report = phase11_9_diagnose(PROJECT_ROOT)
            elif args.phase11_9_command == "watchlist":
                report = phase11_9_watchlist(PROJECT_ROOT)
            elif args.phase11_9_command == "status":
                report = phase11_9_status(PROJECT_ROOT)
            else:
                raise SystemExit(
                    f"Unknown phase11-9 command: {args.phase11_9_command}"
                )
            _print_json(report)
            return 0 if report.get("status") in {"GO", "NOT_RUN"} else 2

        if args.research_command == "phase11-10":
            if args.phase11_10_command == "schema":
                report = phase11_10_schema(PROJECT_ROOT)
            elif args.phase11_10_command == "run":
                report = run_phase11_10(
                    PROJECT_ROOT,
                    historical_cutoff=args.historical_cutoff,
                )
            elif args.phase11_10_command == "qualification-audit":
                report = phase11_10_qualification_audit(PROJECT_ROOT)
            elif args.phase11_10_command == "qualification-freeze":
                report = phase11_10_qualification_freeze(PROJECT_ROOT)
            elif args.phase11_10_command == "reclassify":
                report = phase11_10_reclassify(PROJECT_ROOT)
            elif args.phase11_10_command == "pit-observe":
                report = phase11_10_pit_observe(PROJECT_ROOT)
            elif args.phase11_10_command == "watchlist":
                report = phase11_10_watchlist(PROJECT_ROOT)
            elif args.phase11_10_command == "top20":
                report = phase11_10_top20(PROJECT_ROOT)
            elif args.phase11_10_command == "status":
                report = phase11_10_status(PROJECT_ROOT)
            else:
                raise SystemExit(
                    "Unknown phase11-10 command: "
                    f"{args.phase11_10_command}"
                )
            _print_json(report)
            return 0 if report.get("status") in {"GO", "NOT_RUN"} else 2

        if args.research_command == "phase11-12":
            if args.phase11_12_command == "schema":
                report = phase11_12_schema(PROJECT_ROOT)
            elif args.phase11_12_command == "run":
                report = run_phase11_12(
                    PROJECT_ROOT,
                    max_strategies=args.max_strategies,
                    complexity=args.complexity,
                    pending_only=args.pending_only,
                )
            elif args.phase11_12_command == "observe":
                report = phase11_12_observe(
                    PROJECT_ROOT,
                    max_strategies=args.max_strategies,
                )
            elif args.phase11_12_command == "forward-status":
                report = lower_timeframe_forward_status(PROJECT_ROOT)
            elif args.phase11_12_command == "status":
                report = phase11_12_status(PROJECT_ROOT)
            else:
                raise SystemExit(
                    "Unknown phase11-12 command: "
                    f"{args.phase11_12_command}"
                )
            _print_json(report)
            return 0 if report.get("status") in {
                "GO",
                "PARTIAL",
                "NOT_RUN",
                "QUEUE_EMPTY",
            } else 2

        if args.research_command == "phase11-13":
            if args.phase11_13_command == "schema":
                report = phase11_13_schema(PROJECT_ROOT)
            elif args.phase11_13_command == "run":
                report = run_phase11_13(PROJECT_ROOT)
            elif args.phase11_13_command == "observe":
                report = phase11_13_observe(PROJECT_ROOT)
            elif args.phase11_13_command == "status":
                report = phase11_13_status(PROJECT_ROOT)
            else:
                raise SystemExit(
                    "Unknown phase11-13 command: "
                    f"{args.phase11_13_command}"
                )
            _print_json(report)
            return 0 if report.get("status") in {"GO", "NOT_RUN"} else 2

        if args.research_command == "phase11-14":
            if args.phase11_14_command == "schema":
                report = phase11_14_schema(PROJECT_ROOT)
            elif args.phase11_14_command == "run":
                report = run_phase11_14(
                    PROJECT_ROOT,
                    max_candidates=args.max_candidates,
                )
            elif args.phase11_14_command == "observe":
                report = phase11_14_observe(PROJECT_ROOT)
            elif args.phase11_14_command == "status":
                report = phase11_14_status(PROJECT_ROOT)
            else:
                raise SystemExit(
                    "Unknown phase11-14 command: "
                    f"{args.phase11_14_command}"
                )
            _print_json(report)
            return 0 if report.get("status") in {
                "GO",
                "PARTIAL",
                "NOT_RUN",
            } else 2

        if args.research_command == "phase11-15":
            if args.phase11_15_command == "schema":
                report = phase11_15_schema(PROJECT_ROOT)
            elif args.phase11_15_command == "run":
                report = run_phase11_15(
                    PROJECT_ROOT,
                    max_architectures=args.max_architectures,
                )
            elif args.phase11_15_command == "status":
                report = phase11_15_status(PROJECT_ROOT)
            else:
                raise SystemExit(
                    "Unknown phase11-15 command: "
                    f"{args.phase11_15_command}"
                )
            _print_json(report)
            return 0 if report.get("status") in {"GO", "NOT_RUN"} else 2

        if args.research_command != "universe":
            raise SystemExit(f"Unknown research command: {args.research_command}")

        instrument_layout = InstrumentManifestLayout.from_project_root(PROJECT_ROOT)
        if args.universe_command == "schema":
            _print_json(instrument_manifest_schema())
            return 0
        if args.universe_command == "init-manifest":
            report = initialize_instrument_manifest(instrument_layout)
            _print_json(report)
            return 0
        if args.universe_command in {"validate-manifest", "status"}:
            validation = validate_instrument_manifest(instrument_layout)
            if args.universe_command == "validate-manifest":
                _print_json(validation)
                return 0 if validation["status"] == "GO" else 2
            _print_json(
                {
                    "schema": "research_universe_status_v1",
                    "status": (
                        "READY_FOR_CONTRACT_RESOLUTION"
                        if validation["status"] == "GO"
                        else validation["status"]
                    ),
                    "manifest_validation": validation,
                    "next_gate": "IBKR Phase 2 contract resolution after Phase 1 freeze",
                    "financial_calls": _zero_financial_calls(),
                }
            )
            return 0 if validation["status"] == "GO" else 2
        raise SystemExit(f"Unknown universe command: {args.universe_command}")

    if args.command == "strategy":
        if args.strategy_command != "multi-asset":
            raise SystemExit(f"Unknown strategy command: {args.strategy_command}")

        phase1 = phase1_freeze_status(PROJECT_ROOT)
        strategy_bar_layout = BarCacheLayout.from_project_root(PROJECT_ROOT)
        if args.multi_asset_command == "schema":
            _print_json(multi_asset_strategy_schema())
            return 0
        if args.multi_asset_command == "status":
            cache_validation = validate_bar_cache(strategy_bar_layout)
            strategy_status = (
                "READY"
                if cache_validation["status"] == "GO" and cache_validation["row_count"] > 0
                else "NO_DATA"
                if cache_validation["status"] == "GO"
                else "NO_GO"
            )
            _print_json(
                {
                    "schema": "multi_asset_strategy_status_v1",
                    "status": strategy_status,
                    "phase1": phase1.as_dict(),
                    "phase8": {
                        "enabled": True,
                        "authority": "local_prequalification_backtest_only",
                    },
                    "data": {
                        "provider_calls_enabled": False,
                        "local_bar_cache_status": cache_validation["status"],
                        "local_bar_cache_file_count": cache_validation["file_count"],
                        "local_bar_cache_row_count": cache_validation["row_count"],
                    },
                    "execution": {
                        "orders_enabled": False,
                        "order_intents_enabled": False,
                    },
                    "financial_calls": _zero_financial_calls(),
                }
            )
            return 0 if strategy_status in {"READY", "NO_DATA"} else 2
        if args.multi_asset_command == "backtest":
            interval = BarInterval(args.interval)
            data_type = BarDataType(args.data_type)
            source = None if args.source in {"ANY", "LOCAL"} else BarDataSource(args.source)
            series, cache_validation = load_strategy_series_from_bar_cache(
                strategy_bar_layout,
                interval=interval,
                data_type=data_type,
                source=source,
            )
            if cache_validation["status"] != "GO":
                _print_json(
                    {
                        "schema": "multi_asset_strategy_backtest_command_v1",
                        "status": "NO_GO",
                        "phase1": phase1.as_dict(),
                        "cache_validation": cache_validation,
                        "financial_calls": _zero_financial_calls(),
                    }
                )
                return 2
            config = MultiAssetBacktestConfig(
                lookback_bars=args.lookback_bars,
                top_n_per_sleeve=args.top_n_per_sleeve,
                cost_bps=args.cost_bps,
                initial_nav=args.initial_nav,
                max_asset_weight=args.max_asset_weight,
            )
            report = run_multi_asset_rotation_backtest(series, config=config)
            report["schema"] = "multi_asset_strategy_backtest_command_v1"
            report["phase1"] = phase1.as_dict()
            report["bar_filter"] = {
                "interval": interval.value,
                "data_type": data_type.value,
                "source": args.source if source is None else source.value,
            }
            report["cache_validation"] = {
                "status": cache_validation["status"],
                "file_count": cache_validation["file_count"],
                "row_count": cache_validation["row_count"],
            }
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        raise SystemExit(f"Unknown multi-asset strategy command: {args.multi_asset_command}")

    if args.command == "execution":
        if args.execution_command != "phase7":
            report = operations_execution_command(
                PROJECT_ROOT,
                args.execution_command,
                environment=getattr(args, "environment", "paper"),
                approval=getattr(args, "approval", None),
                env_file=getattr(args, "env_file", ".env.ibkr"),
                session_date=getattr(args, "session_date", None),
                confirmed=(
                    args.execution_command == "activate-live-canary"
                ),
            )
            _print_json(report)
            return 0 if report.get("status") in {
                "GO",
                "OPERATOR_ACTION_REQUIRED",
            } else 2
        Phase7Layout.from_project_root(PROJECT_ROOT).output_dir.mkdir(parents=True, exist_ok=True)
        if args.phase7_command == "schema":
            _print_json(phase7_schema(PROJECT_ROOT))
            return 0
        if args.phase7_command == "init-ledger":
            report = init_ledger(PROJECT_ROOT)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.phase7_command == "simulate":
            report = simulate_phase7(PROJECT_ROOT)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.phase7_command == "audit-ledger":
            report = audit_ledger(PROJECT_ROOT)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.phase7_command == "replay":
            report = replay_phase7(PROJECT_ROOT)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.phase7_command == "reconcile-fixtures":
            report = reconcile_fixtures(PROJECT_ROOT)
            _print_json(report)
            return 0 if report["status"] == "GO" else 2
        if args.phase7_command == "status":
            report = phase7_status(PROJECT_ROOT)
            _print_json(report)
            return 0 if report["status"] == "PHASE7_OFFLINE_EXECUTION_CONTROL_PLANE_GO" else 2
        if args.phase7_command == "freeze":
            report = phase7_freeze(PROJECT_ROOT)
            _print_json(report)
            return 0 if report["freeze_status"] == "PHASE7_OFFLINE_EXECUTION_CONTROL_PLANE_FROZEN_GO" else 2
        raise SystemExit(f"Unknown phase7 command: {args.phase7_command}")

    if args.command == "positions":
        report = positions_command(
            PROJECT_ROOT,
            args.positions_command,
            environment=getattr(args, "environment", "paper"),
            symbol=getattr(args, "symbol", None),
            approval=getattr(args, "approval", None),
            signal_id=getattr(args, "signal_id", None),
            quantity=getattr(args, "quantity", None),
            fill_price=getattr(args, "fill_price", None),
            position_id=getattr(args, "position_id", None),
            ownership_mode=getattr(args, "mode", None),
            confirmed=getattr(args, "yes", False),
        )
        _print_json(report)
        return 0 if report.get("status") in {
            "GO",
            "OPERATOR_ACTION_REQUIRED",
        } else 2

    raise SystemExit(f"Unknown command: {args.command}")


def _handle_market_sessions_command(args: argparse.Namespace, rows: list[Any]) -> dict[str, Any]:
    if args.sessions_command is None:
        if args.date is None:
            raise ValueError("sessions requires --date when no nested command is provided")
        session_date = _parse_iso_date(args.date)
        if args.con_id is None:
            return market_sessions_by_date_report(rows, session_date=session_date)
        row = _cached_row_by_con_id(rows, args.con_id)
        return market_sessions_report(row, session_date=session_date)

    if args.sessions_command == "resolve":
        row = _cached_row_by_con_id(rows, args.con_id)
        return market_sessions_report(row, session_date=_parse_iso_date(args.date))

    if args.sessions_command == "status":
        row = _cached_row_by_con_id(rows, args.con_id)
        return market_status_report(row, at=_parse_iso_datetime(args.at) if args.at else None)

    if args.sessions_command == "next-open":
        row = _cached_row_by_con_id(rows, args.con_id)
        return market_next_open_report(row, at=_parse_iso_datetime(args.at) if args.at else None)

    if args.sessions_command == "range":
        row = _cached_row_by_con_id(rows, args.con_id)
        return market_sessions_range_report(
            row,
            start=_parse_iso_date(args.start),
            end=_parse_iso_date(args.end),
        )

    if args.sessions_command == "validate-cache":
        layout = SessionCacheLayout.from_project_root(PROJECT_ROOT)
        return build_session_cache_from_contract_rows(layout, rows)

    raise SystemExit(f"Unknown market sessions command: {args.sessions_command}")


def _cached_row_by_con_id(rows: list[Any], con_id: int) -> Any:
    if con_id <= 0:
        raise ValueError("con_id must be positive")
    row = next((item for item in rows if item.contract.con_id == con_id), None)
    if row is None:
        raise LookupError(f"con_id {con_id} not found in local contract cache")
    return row


def _parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("at must be timezone-aware")
    return parsed


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value.strip())


def _zero_financial_calls() -> dict[str, int]:
    return {
        "place_order": 0,
        "cancel_order": 0,
        "global_cancel": 0,
    }


if __name__ == "__main__":
    raise SystemExit(main())
