from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stocks.portfolio import execution_feasibility
from stocks.portfolio.execution_feasibility import (
    build_execution_feasibility_report,
    build_p2_2_freeze,
    verify_p2_2_freeze,
)
from stocks.portfolio.learning_integration import integrate_learning_evidence
from stocks.portfolio.quant_authority import load_quant_authority_map


ROOT = Path(__file__).resolve().parents[1]


def _root(tmp_path: Path) -> Path:
    config = tmp_path / "config/portfolio"
    config.mkdir(parents=True)
    (config / "p2_2_execution_feasibility_v1.json").write_text(
        (ROOT / "config/portfolio/p2_2_execution_feasibility_v1.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    costs = tmp_path / "config/costs"
    costs.mkdir(parents=True)
    (costs / "shared_transaction_cost_v1.json").write_text(
        (ROOT / "config/costs/shared_transaction_cost_v1.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    universe = tmp_path / "output/universe"
    universe.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "instrument_id": "ID:ON",
                "symbol": "ON",
                "asset_type": "STOCK",
                "currency": "EUR",
                "primary_exchange": "TEST",
                "compliance_status": "SHARIAH_ALLOWED",
            }
        ]
    ).to_parquet(universe / "instruments.parquet")
    return tmp_path


def _opportunity(*, entry: float = 100, stop: float = 95) -> dict[str, object]:
    return {
        "symbol": "ON",
        "instrument_id": "ID:ON",
        "asset_class": "STOCK",
        "strategy_id": "S1",
        "strategy_ids": ["S1"],
        "contributing_strategy_id": "S1",
        "holding_horizon": "1d",
        "strategy_evidence_status": "WALK_FORWARD_GO",
        "strategy_binding": {"status": "GO", "strategy_id": "S1"},
        "shariah_status": "SHARIAH_ALLOWED",
        "timestamp": "2026-08-11T12:00:00+00:00",
        "live_quote_price": entry,
        "live_quote_timestamp": "2026-08-11T12:00:01+00:00",
        "live_quote_status": "GO",
        "targets": {"entry": entry, "stop": stop, "take_profit": 112},
        "expected_gross_return": 0.05,
        "expected_net_return": 0.04,
        "liquidity": 1.0,
        "spread_bps": 2.0,
        "estimated_slippage_bps": 5.0,
        "event_risk": 0.2,
        "regime_fit": 0.8,
        "blockers": [],
        "learning_overlay": {
            "tcn_probability": 0.7,
            "execution_influence": "NONE",
        },
    }


def _report(tmp_path: Path, opportunity: dict[str, object]) -> dict[str, object]:
    return build_execution_feasibility_report(
        _root(tmp_path),
        opportunities=[opportunity],
        funnel={"watchlist_candidates": []},
        account={
            "status": "GO",
            "fresh_reconciliation": True,
            "net_liquidation_eur": "1870",
            "available_funds_eur": "1870",
        },
        live_authority={
            "execution_authority": "AUTONOMOUS_LEVEL_ONE",
            "lifecycle_status": "ACTIVE",
        },
        market_data_capabilities={
            "summary": {"realtime_top_of_book": "AVAILABLE"}
        },
        level_two_evidence={
            "status": "NO_GO",
            "verified_round_trip_count": 0,
            "minimum_round_trips": 5,
        },
    )


def test_eur9_stop_risk_uses_one_whole_share_and_never_fractional(tmp_path: Path) -> None:
    report = _report(tmp_path, _opportunity())
    row = report["candidates"][0]
    assert report["level_one_risk_budget_eur"] == 9
    assert row["risk_per_share_eur"] == 5
    assert row["maximum_quantity_by_risk"] == 1
    assert row["final_maximum_whole_quantity"] == 1
    assert row["actual_stop_risk_eur"] == 5
    assert row["fractional_shares_allowed"] is False
    assert row["feasibility_status"] == "FEASIBLE_NOW"
    assert row["learning_may_grant_feasibility"] is False


def test_one_share_risk_above_eur9_is_rejected(tmp_path: Path) -> None:
    report = _report(tmp_path, _opportunity(stop=90))
    row = report["candidates"][0]
    assert row["final_maximum_whole_quantity"] == 0
    assert "ONE_SHARE_STOP_RISK_EXCEEDS_EUR9" in row["exact_rejection_reasons"]
    assert "NO_WHOLE_SHARE_WITHIN_ALL_LEVEL_ONE_LIMITS" in row["exact_rejection_reasons"]


def test_notional_sensitivity_does_not_activate_a_larger_cap(tmp_path: Path) -> None:
    report = _report(tmp_path, _opportunity(entry=260, stop=255))
    row = report["candidates"][0]
    assert row["maximum_quantity_by_notional"] == 0
    assert row["notional_sensitivity"]["275"]["notional_quantity"] == 1
    assert report["active_notional_cap_eur"] == 250
    assert report["notional_cap_automatically_changed"] is False


def test_missing_exact_strategy_remains_blocked_even_with_positive_tcn(tmp_path: Path) -> None:
    opportunity = _opportunity()
    opportunity["strategy_binding"] = {"status": "NO_GO"}
    report = _report(tmp_path, opportunity)
    row = report["candidates"][0]
    assert "EXACT_STRATEGY_LIVE_AUTHORITY_NOT_PROVEN" in row["exact_rejection_reasons"]
    assert row["learning_overlay"]["tcn_probability"] == 0.7
    assert row["feasibility_status"] == "REJECTED"
    assert report["strategy_registry_mutated"] is False


def test_planned_entry_never_substitutes_for_a_live_quote(tmp_path: Path) -> None:
    opportunity = _opportunity()
    opportunity.pop("live_quote_price")
    opportunity.pop("live_quote_timestamp")
    opportunity.pop("live_quote_status")
    report = _report(tmp_path, opportunity)
    row = report["candidates"][0]
    assert row["price_source"] == "OPPORTUNITY_PLANNED_ENTRY"
    assert row["price_is_live_actionable"] is False
    assert "LIVE_ACTIONABLE_ENTRY_PRICE_UNAVAILABLE" in row["exact_rejection_reasons"]
    assert row["feasibility_status"] == "REJECTED"


def test_p2_2_freeze_is_immutable_and_detects_source_change(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("v1", encoding="utf-8")
    monkeypatch.setattr(execution_feasibility, "P2_2_SOURCES", ("source.txt",))
    frozen = build_p2_2_freeze(tmp_path)
    assert frozen["status"] == "GO"
    rebuilt = build_p2_2_freeze(tmp_path)
    assert rebuilt == frozen
    assert verify_p2_2_freeze(tmp_path)["status"] == "GO"
    source.write_text("v2", encoding="utf-8")
    verification = verify_p2_2_freeze(tmp_path)
    assert verification["status"] == "NO_GO"
    assert verification["changed_sources"] == ["source.txt"]
    pointer = json.loads(
        (tmp_path / execution_feasibility.FREEZE_POINTER).read_text(encoding="utf-8")
    )
    assert pointer["freeze_hash"] == frozen["freeze_hash"]


def test_learning_overlay_never_mutates_economics_or_authority() -> None:
    opportunity = _opportunity()
    evidence = {
        "status": "GO",
        "evidence_hash": "E1",
        "symbol_predictions": [
            {
                "symbol": "ON",
                "supervised": {
                    "probability_positive_after_costs": 0.8,
                    "validation_status": "SHADOW_VALIDATION_GO",
                },
                "tcn": {
                    "probability_positive_after_costs": 0.9,
                    "validation_status": "SHADOW_VALIDATION_GO",
                },
                "unsupervised": {"regime": "RISK_ON", "confidence": 0.7},
            }
        ],
        "rl_counterfactual_target_weights": {"ON": 0.25},
    }
    overlaid, report = integrate_learning_evidence(
        [opportunity], evidence, load_quant_authority_map(ROOT)
    )
    assert report["status"] == "GO"
    assert overlaid[0]["expected_gross_return"] == 0.05
    assert overlaid[0]["strategy_binding"] == opportunity["strategy_binding"]
    assert overlaid[0]["learning_overlay"]["execution_influence"] == "NONE"
    assert overlaid[0]["learning_overlay"]["may_increase_quantity"] is False
