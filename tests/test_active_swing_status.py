from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stocks.portfolio.swing_status import publish_active_swing_product_status
from stocks.portfolio.swing_status import _current_broker_read_only_usable
from stocks.portfolio.swing_status import _operational_runtime_usable
from stocks.portfolio.swing_status import _runtime_control_state


def test_product_status_fails_closed_and_does_not_fabricate_positions(
    tmp_path: Path,
) -> None:
    report = publish_active_swing_product_status(
        tmp_path,
        signals=[],
        ranked=[],
        normalized_opportunities={"status": "GO", "combined_ranking": []},
        opportunity_funnel={},
        current_positions=[],
        observed_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    assert report["status"] == "NO_GO"
    assert report["readiness"]["SWING_PRODUCT_READY"] is False
    assert report["readiness"]["SWING_DATA_READY"] is False
    assert report["position_management"]["status"] == ("BROKER_POSITION_STATE_UNVERIFIED")
    assert report["orders_generated"] == 0
    assert report["broker_write_calls"] == 0
    assert report["current_opportunities"]["status"] == ("CANDIDATE_ARTIFACT_INVALID")
    assert (tmp_path / "output/portfolio/active-swing-product-status.json").is_file()
    assert (tmp_path / "output/portfolio/active-swing-required-report.json").is_file()
    required = report["required_active_swing_report"]
    assert set(required) >= {
        "A_TIMEFRAME_PIPELINE",
        "B_STRATEGY_ARCHITECTURES",
        "C_CURRENT_OPPORTUNITIES",
        "D_NEAR_SETUPS",
        "E_PORTFOLIO",
        "F_ML",
        "G_RL",
        "H_EXECUTION",
        "I_ECONOMICS",
        "J_PRODUCT_STATUS",
    }
    assert required["I_ECONOMICS"]["status"] == ("UNPROVEN_NO_REALIZED_ROUND_TRIP")
    assert required["I_ECONOMICS"]["expectancy"] is None
    assert required["A_TIMEFRAME_PIPELINE"]["roles"]["15m"] == [
        "TACTICAL_ENTRY_ACTIVATION",
        "TACTICAL_POSITION_MANAGEMENT",
    ]
    assert required["C_CURRENT_OPPORTUNITIES"][
        "natural_candidate_counts_by_entry_timeframe"
    ] == {"15m": 0, "1h": 0, "2h": 0, "4h": 0, "1d": 0}
    assert required["E_PORTFOLIO"]["cash_status"] == "CASH_NO_AUTHORIZED_EDGE"
    assert required["F_ML"][
        "incremental_performance_by_timeframe_architecture_status"
    ] == "UNAVAILABLE_NO_ELIGIBLE_FIXED_OOS_ABLATION"
    assert required["G_RL"]["architecture_performance_status"] == (
        "UNAVAILABLE_NO_CLOSED_BOUND_CANDIDATE_OUTCOMES"
    )
    assert required["H_EXECUTION"]["slippage_status"] == (
        "UNPROVEN_NO_REALIZED_ROUND_TRIP"
    )
    assert required["execution_authority"] == "NONE"
    assert required["orders_generated"] == 0


def test_required_report_includes_current_persisted_near_setups(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "output/research/active_swing/candidate-generation-status.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps(
            {
                "schema": "active_swing_15m_candidate_generation_v1",
                "status": "GO",
                "near_setup_count": 1,
                "near_setup_promoted_count": 0,
                "near_setup_expired_count": 0,
                "near_setups": [
                    {
                        "setup_id": "NEAR-1",
                        "ticker": "TEST",
                        "strategy_id": "ACTIVE-SWING-15M-BREAKOUT-V1",
                        "lifecycle_state": "NEAR_SETUP",
                        "does_not_create_candidate": True,
                        "execution_authority": "NONE",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = publish_active_swing_product_status(
        tmp_path,
        signals=[],
        ranked=[],
        normalized_opportunities={"status": "GO", "combined_ranking": []},
        opportunity_funnel={},
        current_positions=[],
        observed_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    near = report["required_active_swing_report"]["D_NEAR_SETUPS"]
    assert near["count"] == 1
    assert near["persisted_observer_count"] == 1
    assert near["current_pre_trigger_observers"][0]["setup_id"] == "NEAR-1"
    assert near["near_setups_are_not_natural_candidates"] is True
    assert near["near_setups_submit_orders"] is False
    assert report["funnel"]["near_setup_count"] == 1


def test_product_status_separates_research_from_execution_readiness(
    tmp_path: Path,
) -> None:
    universe = tmp_path / "config/universes"
    universe.mkdir(parents=True)
    (universe / "broad_multi_asset_v1.json").write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "group": "shariah",
                        "asset_type": "ETF",
                        "sleeve": "etf_core",
                        "region": "GLOBAL",
                        "instruments": {
                            "SPUS": {
                                "sector": "SHARIAH_US_EQUITY",
                                "provider_symbol": "SPUS",
                            }
                        },
                    },
                    {
                        "group": "commodity",
                        "asset_type": "COMMODITY_ETF",
                        "sleeve": "commodity_security",
                        "region": "GLOBAL",
                        "instruments": {
                            "GLDM": {
                                "sector": "GOLD",
                                "commodity_exposure_type": "PHYSICAL_COMMODITY",
                                "provider_symbol": "GLDM",
                            }
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    report = publish_active_swing_product_status(
        tmp_path,
        signals=[{"action": "WATCHLIST", "timeframe": "1h"}],
        ranked=[{"deployment_blockers": ["EXECUTION_AUTHORITY_NONE"]}],
        normalized_opportunities={
            "status": "GO",
            "combined_ranking": [
                {
                    "expected_net_return": 0.01,
                    "validation_status": "EXACT_VALIDATION_REQUIRED",
                    "shariah_status": "SHARIAH_ALLOWED",
                    "whole_share_feasibility": "FEASIBLE",
                }
            ],
        },
        opportunity_funnel={"portfolio_candidate_count": 1},
        current_positions=[],
        observed_at=datetime(2026, 8, 12, tzinfo=UTC),
    )
    assert report["readiness"]["SWING_RESEARCH_READY"] is True
    assert report["readiness"]["SWING_PORTFOLIO_READY"] is True
    assert report["readiness"]["SWING_EXECUTION_READY"] is False
    assert report["readiness"]["SWING_AUTHORITY_READY"] is False
    assert report["discovery"]["shariah_fund_count"] == 1
    assert report["discovery"]["physical_commodity_claim_count"] == 1
    assert report["discovery"]["physically_backed_commodity_count"] == 0
    assert report["funnel"]["economically_positive_count"] == 0
    assert report["funnel"]["unvalidated_positive_estimate_count"] == 1
    assert report["funnel"]["submitted_count"] == 0


def test_15m_health_rejects_5m_derivation_and_partial_native_bar(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data/research/multitimeframe/private/provider=YFINANCE"
    common = pd.DataFrame(
        {
            "timestamp_utc": ["2026-08-11T19:45:00Z"],
            "quality_status": ["VALIDATED_OHLC"],
            "partial_bucket": [False],
            "exchange_timezone": ["America/New_York"],
        }
    )
    derived = root / "symbol=TEST/interval=15m/source_interval=5m"
    derived.mkdir(parents=True)
    common.assign(is_partial=False).to_parquet(derived / "bars.parquet", index=False)
    native = root / "symbol=TEST/interval=15m/source_interval=15m"
    native.mkdir(parents=True)
    common.assign(is_partial=True).to_parquet(native / "bars.parquet", index=False)

    report = publish_active_swing_product_status(
        tmp_path,
        signals=[],
        ranked=[],
        normalized_opportunities={"status": "GO", "combined_ranking": []},
        opportunity_funnel={},
        current_positions=[],
        observed_at=datetime(2026, 8, 11, 20, 1, tzinfo=UTC),
    )

    health = report["timeframe_health"]["15m"]
    assert health["dataset_count"] == 1
    assert health["row_count"] == 0
    assert health["status"] == "TIMESTAMP_UNAVAILABLE_BLOCKED"
    assert health["native_source_required"] is True


def test_product_status_uses_current_phase9_broker_truth_not_stale_live_state(
    tmp_path: Path,
) -> None:
    observed = datetime(2026, 8, 12, 2, 45, tzinfo=UTC)
    phase9 = tmp_path / "output/ibkr/phase9"
    phase9.mkdir(parents=True)
    (phase9 / "reconciliation-audit.json").write_text(
        json.dumps(
            {
                "schema": "phase9_reconciliation_audit_v1",
                "status": "NO_GO",
                "generated_at": observed.isoformat(),
                "reconciliation_status": "LOCAL_ORDER_MISSING_AT_BROKER",
                "broker_observation_status": "GO",
                "broker_snapshot_status": "BROKER_SNAPSHOT_OBSERVED",
                "operational_broker_state_status": ("CURRENT_BROKER_FLAT_READ_ONLY"),
                "canonical_execution_evidence_status": ("INCOMPLETE_HISTORICAL_EXECUTION_CHAIN"),
                "historical_orphan_quarantine_status": (
                    "HISTORICAL_ORPHANS_QUARANTINED_FAIL_CLOSED"
                ),
                "broker_position_count": 0,
                "broker_open_order_count": 0,
                "read_only_request_counters": {
                    "read_only_account_summary_requests": 1,
                    "read_only_all_api_open_order_requests": 1,
                    "read_only_execution_requests": 1,
                    "read_only_position_requests": 1,
                    "read_only_same_client_open_order_requests": 1,
                },
                "broker_write_counters": {
                    "place_order_calls": 0,
                    "cancel_order_calls": 0,
                },
                "paper_place_order_calls": 0,
                "live_place_order_calls": 0,
                "automatic_submission": False,
                "execution_authority": "NONE",
            }
        ),
        encoding="utf-8",
    )
    stale_live = tmp_path / "output/ibkr/live"
    stale_live.mkdir(parents=True)
    (stale_live / "reconciliation.json").write_text(
        json.dumps(
            {
                "status": "NO_GO",
                "reconciliation_status": "LIVE_TWS_SOCKET_UNREACHABLE",
            }
        ),
        encoding="utf-8",
    )

    report = publish_active_swing_product_status(
        tmp_path,
        signals=[],
        ranked=[],
        normalized_opportunities={"status": "GO", "combined_ranking": []},
        opportunity_funnel={},
        current_positions=[],
        observed_at=observed,
    )

    evidence = report["evidence"]
    assert evidence["reconciliation_current_read_only_usable"] is True
    assert evidence["reconciliation_operational_broker_state"] == ("CURRENT_BROKER_FLAT_READ_ONLY")
    assert evidence["reconciliation_canonical_execution_evidence"] == (
        "INCOMPLETE_HISTORICAL_EXECUTION_CHAIN"
    )
    assert evidence["reconciliation_historical_execution_blocks_trading"] is True
    assert report["readiness"]["SWING_EXECUTION_READY"] is False
    assert report["position_management"]["status"] == ("NO_OPEN_POSITION_MANAGEMENT_REQUIRED")
    assert report["position_management"]["fabricated_positions"] == 0


def test_current_broker_truth_allows_only_bounded_publication_race() -> None:
    observed = datetime(2026, 8, 12, 2, 45, tzinfo=UTC)
    payload = {
        "schema": "phase9_reconciliation_audit_v1",
        "generated_at": "2026-08-12T02:46:00+00:00",
        "broker_observation_status": "GO",
        "broker_snapshot_status": "BROKER_SNAPSHOT_OBSERVED",
        "operational_broker_state_status": ("CURRENT_BROKER_FLAT_READ_ONLY"),
        "broker_position_count": 0,
        "broker_open_order_count": 0,
        "read_only_request_counters": {
            "read_only_account_summary_requests": 1,
            "read_only_all_api_open_order_requests": 1,
            "read_only_execution_requests": 1,
            "read_only_position_requests": 1,
            "read_only_same_client_open_order_requests": 1,
        },
        "broker_write_counters": {"place_order_calls": 0},
        "paper_place_order_calls": 0,
        "live_place_order_calls": 0,
        "automatic_submission": False,
        "execution_authority": "NONE",
    }

    assert _current_broker_read_only_usable(payload, observed_at=observed) is True
    payload["generated_at"] = "2026-08-12T02:48:00+00:00"
    assert _current_broker_read_only_usable(payload, observed_at=observed) is False


def test_operational_runtime_allows_only_evidence_degradation() -> None:
    heartbeat = {
        "runtime_status": "DEGRADED",
        "runtime_state": "ENTRY_BLOCKED",
    }
    machine = {
        "status": "DEGRADED",
        "enabled": True,
        "paused": False,
        "last_cycle_blockers": [
            "OPERATIONAL_STEP_P4_EVIDENCE_NO_GO",
            "OPERATIONAL_STEP_RL_SHADOW_INSUFFICIENT_EVIDENCE",
        ],
    }

    assert _operational_runtime_usable(heartbeat, machine) is True

    machine["last_cycle_blockers"].append("OPERATIONAL_STEP_RECONCILIATION_NO_GO")
    assert _operational_runtime_usable(heartbeat, machine) is False

    machine["last_cycle_blockers"] = []
    machine["paused"] = True
    assert _operational_runtime_usable(heartbeat, machine) is False


def test_newer_heartbeat_overrides_stale_disabled_machine_snapshot() -> None:
    heartbeat = {
        "runtime_status": "RUNNING",
        "enabled": True,
        "paused": False,
        "last_heartbeat": "2026-08-12T03:29:47+00:00",
    }
    machine = {
        "status": "DEGRADED",
        "enabled": False,
        "paused": False,
        "last_heartbeat": "2026-08-12T03:25:36+00:00",
        "last_cycle_blockers": [
            "OPERATIONAL_STEP_P4_EVIDENCE_NO_GO",
        ],
    }

    assert _runtime_control_state(heartbeat, machine) == {
        "source": "NEWER_RUNTIME_HEARTBEAT",
        "enabled": True,
        "paused": False,
    }
    assert _operational_runtime_usable(heartbeat, machine) is True


def test_older_heartbeat_cannot_override_current_disabled_machine() -> None:
    heartbeat = {
        "runtime_status": "RUNNING",
        "enabled": True,
        "paused": False,
        "last_heartbeat": "2026-08-12T03:25:36+00:00",
    }
    machine = {
        "status": "GO",
        "enabled": False,
        "paused": False,
        "last_heartbeat": "2026-08-12T03:29:47+00:00",
        "last_cycle_blockers": [],
    }

    assert _runtime_control_state(heartbeat, machine)["source"] == ("MACHINE_STATUS")
    assert _operational_runtime_usable(heartbeat, machine) is False


def test_current_opportunity_report_requires_exact_candidate_bound_evidence(
    tmp_path: Path,
) -> None:
    observed = datetime(2026, 8, 12, 3, 0, tzinfo=UTC)
    candidate_path = tmp_path / "output/signals/active_swing_15m_signals.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps(
            {
                "schema": "active_swing_15m_candidate_generation_v1",
                "generated_at": observed.isoformat(),
                "candidate_count": 1,
                "signals": [
                    {
                        "ticker": "TEST",
                        "setup_id": "SETUP-1",
                        "candidate_unit": "ONE_NATURAL_STRATEGY_SETUP",
                        "strategy_id": "ACTIVE-SWING-15M-BREAKOUT-V1",
                        "strategy_family": "volatility_contraction_breakout",
                        "strategy_timeframe_contract": {
                            "entry_timeframe": "15m",
                            "setup_timeframe": "1h",
                            "required_timeframes": ["15m", "1h", "4h"],
                            "optional_timeframes": ["1d"],
                        },
                        "signal_timestamp": observed.isoformat(),
                        "preferred_entry": 105.0,
                        "stop_loss": 100.0,
                        "take_profit_1": 112.5,
                        "take_profit_2": 117.5,
                        "confidence_score": 0.7,
                        "reasons": ["NATURAL_CLOSED_BAR_TRIGGER"],
                        "risks": ["UNQUALIFIED_FORWARD_OBSERVER"],
                        "portfolio_eligible": False,
                        "qualification_status": ("UNQUALIFIED_FORWARD_OBSERVER"),
                        "strategy_authority": "NONE",
                        "execution_authority": "NONE",
                        "suggested_quantity": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = publish_active_swing_product_status(
        tmp_path,
        signals=[],
        ranked=[],
        normalized_opportunities={"status": "GO", "combined_ranking": []},
        opportunity_funnel={},
        current_positions=[],
        observed_at=observed,
    )

    current = report["current_opportunities"]
    assert current["status"] == "RESEARCH_OBSERVERS_ONLY"
    assert current["natural_candidate_count"] == 1
    row = current["highest_ranked_opportunities"][0]
    assert set(current["required_fields"]) <= set(row)
    assert row["entry_timeframe"] == "15m"
    assert row["setup_timeframe"] == "1h"
    assert row["expected_net_return"]["value"] is None
    assert row["ml_prediction"]["value"] is None
    assert row["rl_advisory"]["action"] is None
    assert row["whole_share_quantity"] == 0
    assert row["quote_status"]["quote_valid"] is False
    assert row["final_decision"]["action"] == "NO_TRADE"
    assert current["symbol_only_cross_artifact_join_allowed"] is False
    assert current["orders_generated"] == 0


def test_required_report_counts_only_explicit_timeframe_architectures(
    tmp_path: Path,
) -> None:
    report = publish_active_swing_product_status(
        tmp_path,
        signals=[
            {
                "strategy_timeframe_contract": {
                    "entry_timeframe": "15m",
                    "setup_timeframe": "1h",
                    "context_timeframes": ["1d", "4h"],
                }
            },
            {"timeframes": ["15m", "1h", "4h"]},
        ],
        ranked=[],
        normalized_opportunities={"status": "GO", "combined_ranking": []},
        opportunity_funnel={},
        current_positions=[],
        observed_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    section = report["required_active_swing_report"]["B_STRATEGY_ARCHITECTURES"]
    assert section["count"] == 1
    assert section["signals_without_explicit_contract"] == 1
    assert section["signals_with_explicit_contract"] == 1
    assert section["explicit_strategy_count"] == 1
    assert section["unique_strategies_without_explicit_contract"] == 1
    assert section["counting_unit"] == "ONE_UNIQUE_STRATEGY_DNA_NOT_SIGNAL_INSTANCE"
    assert section["counts_by_explicit_entry_setup_context"] == {
        "15m_ENTRY__1h_SETUP__1d+4h_CONTEXT": 1
    }


def test_required_report_counts_declared_candidate_architecture_without_trigger(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "output/research/active_swing/candidate-generation-status.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(
        json.dumps(
            {
                "status": "GO",
                "declared_strategy_architectures": [
                    {
                        "strategy_id": "ACTIVE-SWING-15M-BREAKOUT-V1",
                        "strategy_timeframe_contract": {
                            "entry_timeframe": "15m",
                            "setup_timeframe": "1h",
                            "context_timeframes": ["1h", "4h", "1d"],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = publish_active_swing_product_status(
        tmp_path,
        signals=[],
        ranked=[],
        normalized_opportunities={"status": "GO", "combined_ranking": []},
        opportunity_funnel={},
        current_positions=[],
        observed_at=datetime(2026, 8, 12, tzinfo=UTC),
    )

    section = report["required_active_swing_report"]["B_STRATEGY_ARCHITECTURES"]
    assert section["count"] == 1
    assert section["explicit_strategy_count"] == 1
    assert section["signals_without_explicit_contract"] == 0
