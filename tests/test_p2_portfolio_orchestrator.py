from __future__ import annotations

import json
from pathlib import Path

import pytest

from stocks.portfolio.execution_bridge import build_p0_execution_bridge
from stocks.portfolio.orchestrator import (
    build_actual_portfolio,
    build_continuous_desired_portfolio,
    build_global_risk,
    build_integer_portfolio,
    build_portfolio_drift,
    normalize_orchestrator_opportunities,
    run_autonomous_dry_run,
    supervise_positions,
)
from stocks.portfolio.quant_authority import load_quant_authority_map


ROOT = Path(__file__).resolve().parents[1]


def _account() -> dict[str, object]:
    return {
        "status": "GO",
        "net_liquidation_eur": "1870",
        "total_cash_value_eur": "1870",
        "snapshot_hash": "SNAPSHOT",
        "snapshot_completed_at": "2026-08-10T20:00:00+00:00",
    }


def _snapshot(*, positions: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "positions": {"status": "COMPLETE", "positions": positions or []},
        "all_api_open_orders": {"status": "COMPLETE", "open_orders": []},
        "executions": {"status": "EMPTY_COMPLETE", "executions": [], "commissions": []},
    }


def _opportunity(symbol: str = "ON", *, asset_class: str = "EQUITY") -> dict[str, object]:
    return {
        "instrument_id": f"ID:{symbol}",
        "symbol": symbol,
        "asset_class": asset_class,
        "strategy_id": "S1",
        "strategy_version": "V1",
        "strategy_family": "MOMENTUM",
        "validation_status": "VALIDATED_OPPORTUNITY",
        "signal_timestamp": "2026-08-10T20:00:00+00:00",
        "direction": "LONG",
        "expected_return": 0.10,
        "expected_net_return": 0.05,
        "expected_r": 1.5,
        "confidence": 0.75,
        "fees_eur": 0.50,
        "volatility": 0.20,
        "liquidity": 0.90,
        "event_risk": 0.10,
        "data_quality": 1.0,
        "shariah_status": "SHARIAH_ALLOWED",
        "broker_resolvable": True,
        "whole_share_feasibility": "WHOLE_SHARE_FEASIBLE_RISK_FIRST",
        "correlation_cluster": "SEMICONDUCTOR_FACTOR",
        "portfolio_eligible": True,
        "execution_eligible": True,
        "research_eligible": True,
        "blockers": [],
    }


def _target() -> dict[str, object]:
    return {
        "symbol": "ON",
        "action": "BUY_DELTA",
        "current_quantity": 0,
        "desired_quantity": 1,
        "quantity_delta": 1,
        "desired_exposure_eur": 70,
        "expected_net_return": 0.05,
        "expected_loss": 0.03,
        "strategy_id": "S1",
    }


def _sizing() -> dict[str, object]:
    return {
        "status": "GO",
        "positions": [
            {
                "ticker": "ON",
                "target_quantity": "1",
                "level1_canary_qty": "1",
                "actual_risk_eur": "4.9",
            }
        ],
    }


def _policy() -> dict[str, object]:
    return {"risk": {"maximum_event_risk": 0.75}}


def _dynamic_risk() -> dict[str, object]:
    return {
        "status": "GO",
        "new_entries_allowed": True,
        "maximum_portfolio_heat": 0.06,
        "multipliers": {"drawdown": 1.0, "combined": 0.85},
        "loss_guard": {"new_entries_allowed": True},
    }


def _reconciliation() -> dict[str, object]:
    return {
        "status": "GO",
        "reconciliation_status": "LIVE_RECONCILED_EMPTY",
        "unknown_positions": 0,
        "unknown_orders": 0,
    }


def _allowlist() -> dict[str, object]:
    return {
        "status": "GO",
        "strategies": [
            {
                "strategy_id": "S1",
                "status": "PIT_LIVE_ALLOWLISTED",
                "allowed_symbols": ["ON"],
            }
        ],
    }


def _strategy_authority() -> dict[str, object]:
    return {
        "status": "GO",
        "blockers": [],
        "strategies": [
            {
                "strategy_id": "S1",
                "version": "V1",
                "source_hash": "SOURCE",
                "parameter_hash": "PARAMETER",
                "allowed_symbols": ["ON"],
                "allowed_asset_classes": ["STK"],
                "deployment_status": "LIVE_AUTHORIZED_LEVEL_ONE",
            }
        ],
    }


def test_authority_map_covers_exact_33_and_keeps_ml_moe_rl_shadow() -> None:
    report = load_quant_authority_map(ROOT)
    assert report["status"] == "GO"
    assert report["capability_count"] == 33
    assert {17, 18, 19, 31, 32}.issubset(report["shadow_capability_ids"])
    assert all(not row["money_control"] for row in report["capabilities"])


def test_actual_portfolio_comes_from_reconciled_snapshot() -> None:
    actual = build_actual_portfolio(
        _snapshot(positions=[{"symbol": "ON", "position": 2, "market_value_eur": 140}]),
        _account(),
    )
    assert actual["status"] == "GO"
    assert actual["position_count"] == 1
    assert actual["positions"][0]["source"] == "CANONICAL_IBKR_RECONCILIATION"


def test_continuous_desired_portfolio_includes_cash_as_asset() -> None:
    result = build_continuous_desired_portfolio(
        {"allocations": [{"ticker": "ON", "asset_type": "STOCK", "research_target_weight": 0.1}]},
        _account(),
    )
    assert result["positions"][0]["desired_weight"] == 0.1
    assert result["cash"]["desired_weight"] == 0.9
    assert result["cash_is_first_class_asset"] is True


def test_integer_projection_is_whole_share_and_cash_secured() -> None:
    result = build_integer_portfolio(
        {"targets": [_target()]},
        _sizing(),
        _account(),
    )
    assert result["status"] == "GO"
    assert result["positions"][0]["quantity"] == 1
    assert result["fractional_positions"] == 0
    assert result["cash_eur"] == 1800


def test_integer_diagnostics_explain_ten_percent_to_one_share() -> None:
    sizing = {
        "positions": [
            {
                "ticker": "ON",
                "unit_notional_eur": "70.18590575010808",
                "desired_notional_eur": "187",
                "normal_allowed_qty": "1",
                "risk_quantity": "1",
                "level1_canary_qty": "1",
                "cash_quantity": "26",
                "capacity_quantity": "16059",
                "position_cap_quantity": "9",
                "risk_budget_eur": "9.382357545",
                "risk_per_share_eur": "4.9770670125",
                "actual_risk_eur": "4.9770670125",
            }
        ]
    }
    continuous = {
        "positions": [
            {
                "symbol": "ON",
                "desired_weight": 0.1,
                "desired_notional_eur": 187,
            }
        ]
    }
    result = build_integer_portfolio(
        {"targets": [_target()]},
        sizing,
        _account(),
        continuous_portfolio=continuous,
    )
    diagnostics = result["positions"][0]["integer_diagnostics"]

    assert diagnostics["continuous_weight"] == 0.1
    assert diagnostics["raw_fractional_quantity"] == pytest.approx(2.6643527)
    assert diagnostics["lower_integer_quantity"] == 2
    assert diagnostics["upper_integer_quantity"] == 3
    assert diagnostics["normal_risk_allowed_quantity"] == 1
    assert diagnostics["authority_allowed_quantity"] == 1
    assert diagnostics["final_quantity"] == 1
    assert set(diagnostics["binding_constraint"]) == {
        "authority_level_one_allowed_quantity",
        "normal_risk_allowed_quantity",
        "risk_based_quantity",
    }
    assert diagnostics["weight_error"] == pytest.approx(-0.06256684)


def test_integer_projection_rejects_fractional_target() -> None:
    target = {**_target(), "desired_quantity": 0.5, "quantity_delta": 0.5}
    result = build_integer_portfolio({"targets": [target]}, _sizing(), _account())
    assert result["status"] == "NO_GO"
    assert result["violations"] == ["FRACTIONAL_TARGET:ON"]


def test_drift_emits_open_add_hold_reduce_and_exit_semantics() -> None:
    actual = {
        "positions": [
            {"symbol": "HOLD", "quantity": 1, "weight": 0.1},
            {"symbol": "EXIT", "quantity": 2, "weight": 0.2},
        ]
    }
    target = {
        "positions": [
            {"symbol": "HOLD", "quantity": 1, "weight": 0.1},
            {"symbol": "OPEN", "quantity": 1, "weight": 0.1},
        ]
    }
    rows = {row["symbol"]: row for row in build_portfolio_drift(actual, target)["rows"]}
    assert rows["HOLD"]["action"] == "HOLD"
    assert rows["EXIT"]["action"] == "EXIT"
    assert rows["OPEN"]["action"] == "OPEN"


def test_cross_asset_normalization_keeps_real_asset_family() -> None:
    row = _opportunity("COPX", asset_class="COMMODITY_EXPOSURE")
    row["correlation_cluster"] = "INDUSTRIAL_METALS_COPPER_FACTOR"
    result = normalize_orchestrator_opportunities(
        [row],
        ranking_rows=[
            {
                "ticker": "COPX",
                "strategy_ids": ["COPPER-TREND-V1"],
                "sector": "COPPER_MINERS",
                "expected_holding_period": "2-20 closed weeks",
            }
        ],
    )[0]
    assert result["asset_class"] == "COMMODITY_EXPOSURE"
    assert result["commodity_family"] == "COPPER"
    assert result["strategy_ids"] == ["COPPER-TREND-V1"]
    assert result["sector"] == "COPPER_MINERS"
    assert "expected_cost_eur" in result


def test_global_risk_counts_planned_risk_and_etf_lookthrough() -> None:
    actual = {"equity_eur": 1870}
    integer = {"positions": [{"risk_eur": 4.9}]}
    result = build_global_risk(actual, integer, _dynamic_risk(), {"status": "GO"})
    assert result["status"] == "GO"
    assert result["new_planned_risk_eur"] == 4.9
    assert result["etf_lookthrough_go"] is True
    assert result["correlation_cluster_risk_modeled"] is True


def test_position_supervisor_detects_partial_protection_gap() -> None:
    snapshot = _snapshot()
    snapshot["all_api_open_orders"]["open_orders"] = [
        {"symbol": "ON", "side": "SELL", "quantity": 1}
    ]
    result = supervise_positions(
        [{"symbol": "ON", "quantity": 2}],
        [{"symbol": "ON", "quantity": 2}],
        snapshot,
        [_opportunity()],
    )
    assert result["positions"][0]["action"] == "HOLD"
    assert result["positions"][0]["protection_gap_qty"] == 1
    assert result["unprotected_position_count"] == 1


def test_bridge_uses_p0_route_and_never_calls_broker() -> None:
    result = build_p0_execution_bridge(
        ROOT,
        targets=[_target()],
        sizing_rows=_sizing()["positions"],
        opportunities=[_opportunity()],
        reconciliation=_reconciliation(),
        live_authority={"execution_authority": "LIVE_LEVEL_ONE", "manual_approval_required": False},
        writer_integrity={"status": "GO"},
        p02_integrity={"status": "GO"},
        strategy_allowlist=_allowlist(),
        dynamic_risk=_dynamic_risk(),
        policy=_policy(),
    )
    assert result["machine_approved_count"] == 1
    assert result["selected_action"]["canonical_prepare_function"] == "stocks.live.service.live_prepare"
    assert result["selected_action"]["bridge_invoked_submit"] is False
    assert result["broker_writes"] == 0


def test_bridge_fail_closed_for_unapproved_strategy_and_inactive_authority() -> None:
    target = {**_target(), "strategy_id": "UNAPPROVED"}
    result = build_p0_execution_bridge(
        ROOT,
        targets=[target],
        sizing_rows=_sizing()["positions"],
        opportunities=[_opportunity()],
        reconciliation=_reconciliation(),
        live_authority={"execution_authority": "NONE"},
        writer_integrity={"status": "GO"},
        p02_integrity={"status": "GO"},
        strategy_allowlist=_allowlist(),
        dynamic_risk=_dynamic_risk(),
        policy=_policy(),
    )
    blockers = result["proposals"][0]["blockers"]
    assert "STRATEGY_LIVE_AUTHORIZED" in blockers
    assert "CURRENT_LIVE_AUTHORITY_NOT_ACTIVE" in blockers
    assert result["selected_action"] is None


def test_full_dry_run_publishes_actual_desired_integer_and_zero_orders(tmp_path: Path) -> None:
    config = tmp_path / "config/portfolio"
    config.mkdir(parents=True)
    for name in (
        "quant_capability_authority_v1.json",
        "p2_orchestrator_v1.json",
        "p2_2_execution_feasibility_v1.json",
    ):
        (config / name).write_text((ROOT / "config/portfolio" / name).read_text(encoding="utf-8"), encoding="utf-8")
    inputs = {
        "account_state": _account(),
        "whole_share_sizing": _sizing(),
        "broker_snapshot": _snapshot(),
        "reconciliation": _reconciliation(),
        "opportunities": {"combined_ranking": [_opportunity()], "content_hash": "O"},
        "target_allocation": {
            "allocations": [
                {
                    "ticker": "ON",
                    "asset_type": "STOCK",
                    "research_target_weight": 0.1,
                    "stop_risk_pct": 0.03,
                }
            ]
        },
        "desired_targets": {"targets": [_target()], "content_hash": "D"},
        "dynamic_risk": _dynamic_risk(),
        "overlap": {"status": "GO"},
        "funnel": {
            "universe_instrument_count": 100,
            "watchlist_candidate_count": 20,
            "ranked_opportunity_count": 5,
            "portfolio_candidate_count": 1,
            "execution_candidate_count": 1,
        },
        "portfolio_status": {"technical_regime": "BULL_TREND_LOW_VOL"},
            "strategy_allowlist": _allowlist(),
            "strategy_authority": _strategy_authority(),
        "live_authority": {"execution_authority": "LIVE_LEVEL_ONE", "manual_approval_required": False},
        "writer_integrity": {"status": "GO"},
        "p02_integrity": {"status": "GO"},
        "p2_2_integrity": {"status": "GO", "freeze_hash": "P22"},
        "market_data_capabilities": {
            "summary": {"realtime_top_of_book": "AVAILABLE"}
        },
    }
    result = run_autonomous_dry_run(tmp_path, inputs=inputs)
    assert result["status"] == "GO"
    assert result["decision"] == "OPEN"
    assert result["runtime_action"] == "AUTONOMOUS_DRY_RUN_READY"
    assert result["actual_portfolio"]["position_count"] == 0
    assert result["integer_portfolio"]["positions"][0]["quantity"] == 1
    assert result["orders_created"] == 0
    assert result["orders_submitted"] == 0
    assert result["broker_write_calls"] == 0
    public = json.loads((tmp_path / "output/portfolio/orchestrator/current-cycle.json").read_text())
    assert public["actual_portfolio"]["equity_eur"] is None


def test_weak_candidates_leave_cash_and_no_trade() -> None:
    weak = _opportunity()
    weak["expected_net_return"] = -0.01
    normalized = normalize_orchestrator_opportunities([weak])
    assert normalized[0]["expected_net_return"] < 0
    desired = build_continuous_desired_portfolio({"allocations": []}, _account())
    assert desired["cash_weight"] == 1.0
