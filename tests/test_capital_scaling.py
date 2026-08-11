from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from stocks.capital import service as capital_service
from stocks.live import evidence as live_evidence
from stocks.capital.service import (
    allowed_trade_risk,
    capital_level_limits,
    capital_command,
    daily_profit_target,
    implementation_shortfall,
    whole_share_quantity,
)


def _policy(root: Path) -> None:
    source = (
        Path(__file__).parents[1]
        / "config/capital_scaling/levels_v1.json"
    )
    target = root / "config/capital_scaling/levels_v1.json"
    target.parent.mkdir(parents=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def test_default_level_is_zero_and_promotion_is_manual(tmp_path: Path) -> None:
    _policy(tmp_path)
    report = capital_command(tmp_path, "status")
    assert report["CURRENT_CAPITAL_LEVEL"] == 0
    assert report["AUTOMATIC_CAPITAL_PROMOTION"] is False
    assert report["MARGIN_ENABLED"] is False
    assert report["LEVERAGE_ENABLED"] is False


def test_promotion_above_evidence_is_blocked_even_with_phrase(
    tmp_path: Path,
) -> None:
    _policy(tmp_path)
    report = capital_command(
        tmp_path,
        "promote",
        level=1,
        approval="PROMOTE CAPITAL LEVEL 1 WITH MANUAL APPROVAL",
    )
    assert report["status"] == "BLOCKED"
    assert "EVIDENCE_LEVEL_NOT_REACHED" in report["blockers"]


def test_level_one_requires_execution_evidence_and_exact_phrase(
    tmp_path: Path,
) -> None:
    _policy(tmp_path)
    phase9 = tmp_path / "output/ibkr/phase9/status.json"
    phase9.parent.mkdir(parents=True)
    phase9.write_text(
        json.dumps(
            {
                "checks": {
                    "fill_canary": True,
                    "closing_sell_canary": True,
                    "reconciliation": True,
                }
            }
        ),
        encoding="utf-8",
    )
    reconciliation = (
        tmp_path / "output/ibkr/phase9/reconciliation-audit.json"
    )
    reconciliation.write_text(
        json.dumps(
            {"reconciliation_status": "PAPER_RECONCILED_EMPTY"}
        ),
        encoding="utf-8",
    )
    wrong = capital_command(
        tmp_path, "promote", level=1, approval="WRONG"
    )
    assert wrong["status"] == "BLOCKED"
    approved = capital_command(
        tmp_path,
        "promote",
        level=1,
        approval="PROMOTE CAPITAL LEVEL 1 WITH MANUAL APPROVAL",
    )
    assert approved["CURRENT_CAPITAL_LEVEL"] == 1


def test_level_two_requires_verified_live_round_trips_and_sequential_promotion(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _policy(tmp_path)
    phase9 = tmp_path / "output/ibkr/phase9/status.json"
    phase9.parent.mkdir(parents=True)
    phase9.write_text(
        json.dumps(
            {
                "checks": {
                    "fill_canary": True,
                    "closing_sell_canary": True,
                    "reconciliation": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "output/ibkr/phase9/reconciliation-audit.json").write_text(
        json.dumps({"reconciliation_status": "PAPER_RECONCILED_EMPTY"}),
        encoding="utf-8",
    )
    research = tmp_path / "output/research/phase11_9/status.json"
    research.parent.mkdir(parents=True)
    research.write_text(
        json.dumps({"FINANCIAL_FINALIST_GO": True}), encoding="utf-8"
    )
    monkeypatch.setattr(
        live_evidence,
        "live_level_two_evidence",
        lambda _root, minimum_round_trips=5: {
            "status": "GO",
            "verified_round_trip_count": minimum_round_trips,
            "content_hash": "EVIDENCE-HASH",
            "blockers": [],
        },
    )

    recommendation = capital_command(tmp_path, "recommend-level")
    skipped = capital_command(
        tmp_path,
        "promote",
        level=2,
        approval="PROMOTE CAPITAL LEVEL 2 WITH MANUAL APPROVAL",
    )
    level_one = capital_command(
        tmp_path,
        "promote",
        level=1,
        approval="PROMOTE CAPITAL LEVEL 1 WITH MANUAL APPROVAL",
    )
    level_two = capital_command(
        tmp_path,
        "promote",
        level=2,
        approval="PROMOTE CAPITAL LEVEL 2 WITH MANUAL APPROVAL",
    )

    assert recommendation["recommended_level"] == 2
    assert recommendation["level_two_promotion_eligible"] is True
    assert "SEQUENTIAL_CAPITAL_PROMOTION_REQUIRED" in skipped["blockers"]
    assert level_one["CURRENT_CAPITAL_LEVEL"] == 1
    assert level_two["CURRENT_CAPITAL_LEVEL"] == 2


def test_level_one_rejects_open_position_as_incomplete_round_trip(
    tmp_path: Path,
) -> None:
    _policy(tmp_path)
    phase9 = tmp_path / "output/ibkr/phase9/status.json"
    phase9.parent.mkdir(parents=True)
    phase9.write_text(
        json.dumps(
            {
                "checks": {
                    "fill_canary": True,
                    "closing_sell_canary": False,
                    "reconciliation": True,
                }
            }
        ),
        encoding="utf-8",
    )
    reconciliation = (
        tmp_path / "output/ibkr/phase9/reconciliation-audit.json"
    )
    reconciliation.write_text(
        json.dumps(
            {"reconciliation_status": "PAPER_RECONCILED_OPEN_LONG"}
        ),
        encoding="utf-8",
    )

    report = capital_command(tmp_path, "status")

    assert report["recommendation"]["recommended_level"] == 0
    assert (
        report["recommendation"]["evidence"]["paper_fill_close_canary"]
        is False
    )
    assert report["recommendation"]["evidence"]["paper_reconciliation"] is False
    assert (
        "EXECUTION_FILL_CLOSE_CANARY_NOT_PROVEN"
        in report["recommendation"]["blockers"]
    )


def test_operator_attested_completion_removes_duplicate_phase9_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _policy(tmp_path)
    phase9 = tmp_path / "output/ibkr/phase9/status.json"
    phase9.parent.mkdir(parents=True)
    phase9.write_text(
        json.dumps(
            {
                "checks": {
                    "fill_canary": True,
                    "closing_sell_canary": True,
                    "reconciliation": True,
                }
            }
        ),
        encoding="utf-8",
    )
    reconciliation = (
        tmp_path / "output/ibkr/phase9/reconciliation-audit.json"
    )
    reconciliation.write_text(
        json.dumps({"reconciliation_status": "BROKER_OBSERVATION_BLOCKED"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        capital_service,
        "load_operator_completion_evidence",
        lambda _root: {"status": "OPERATOR_ATTESTED_MANUAL_PAPER_ROUND_TRIP_GO"},
    )

    report = capital_command(tmp_path, "status")

    assert report["recommendation"]["recommended_level"] == 1
    assert report["recommendation"]["promotion_allowed"] is True
    assert report["recommendation"]["recommendation_scope"] == (
        "WHOLE_SHARE_EXECUTION_CANARY_ONLY"
    )
    assert report["recommendation"][
        "financial_finalist_required_for_level_one"
    ] is False
    assert "FINANCIAL_FINALIST_NOT_PROVEN" not in report[
        "recommendation"
    ]["blockers"]
    assert "FINANCIAL_FINALIST_NOT_PROVEN" in report[
        "recommendation"
    ]["normal_allocation_blockers"]
    assert report["recommendation"]["evidence"]["paper_reconciliation"] is True
    assert (
        report["recommendation"]["evidence"][
            "operator_attested_manual_completion"
        ]
        is True
    )
    assert (
        "EXECUTION_FILL_CLOSE_CANARY_NOT_PROVEN"
        not in report["recommendation"]["blockers"]
    )


def test_risk_sizing_uses_stop_distance_and_whole_shares() -> None:
    risk = allowed_trade_risk(
        Decimal("50000"),
        Decimal("0.0035"),
        strategy_multiplier=Decimal("0.8"),
        regime_multiplier=Decimal("0.75"),
        drawdown_multiplier=Decimal("1"),
        liquidity_multiplier=Decimal("1"),
        data_quality_multiplier=Decimal("1"),
    )
    assert risk == Decimal("105.000000")
    quantity = whole_share_quantity(
        risk,
        Decimal("50"),
        Decimal("45"),
        cash_cap_eur=Decimal("1000"),
        position_cap_eur=Decimal("750"),
        liquidity_cap_eur=Decimal("2000"),
    )
    assert quantity == 15


def test_level_two_micro_account_limits_make_meaningful_whole_shares_possible(
    tmp_path: Path,
) -> None:
    _policy(tmp_path)

    limits = capital_level_limits(
        tmp_path,
        level=2,
        account_equity_eur=Decimal("1870"),
    )

    assert limits["maximum_stock_order_eur"] == "250"
    assert Decimal(limits["maximum_total_exposure_eur"]) == Decimal("1122")
    assert Decimal(limits["maximum_risk_per_trade_eur"]) == Decimal("28.05")
    assert Decimal(limits["maximum_portfolio_heat_eur"]) == Decimal("112.2")
    assert limits["maximum_positions"] == 4
    assert limits["automatic_orders"] is False
    assert limits["margin_enabled"] is False
    assert limits["leverage_enabled"] is False


def test_capital_level_limits_reject_unbounded_risk_configuration(
    tmp_path: Path,
) -> None:
    _policy(tmp_path)
    path = tmp_path / "config/capital_scaling/levels_v1.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    policy["levels"]["2"]["maximum_risk_per_trade_pct"] = 0.20
    path.write_text(json.dumps(policy), encoding="utf-8")

    try:
        capital_level_limits(
            tmp_path,
            level=2,
            account_equity_eur=Decimal("1870"),
        )
    except ValueError as exc:
        assert str(exc) == "maximum_risk_per_trade_pct_OUT_OF_RANGE"
    else:
        raise AssertionError("unsafe risk policy was accepted")


def test_implementation_shortfall_sign_is_side_aware() -> None:
    buy = implementation_shortfall(
        "BUY", Decimal("100"), Decimal("101"), Decimal("102")
    )
    sell = implementation_shortfall(
        "SELL", Decimal("100"), Decimal("99"), Decimal("98")
    )
    assert buy["total_implementation_shortfall_per_share"] == 2.0
    assert sell["total_implementation_shortfall_per_share"] == 2.0


def test_daily_profit_target_scales_with_equity_and_throttles_entries() -> None:
    policy = {
        "target_pct_of_equity": 0.005,
        "minimum_target_eur": 0,
        "maximum_target_eur": None,
        "timezone": "Europe/Amsterdam",
    }
    below = daily_profit_target(
        Decimal("50000"), Decimal("249"), policy
    )
    reached = daily_profit_target(
        Decimal("50000"),
        Decimal("250"),
        policy,
        enforcement_active=True,
    )
    assert below["daily_profit_target_eur"] == 250.0
    assert below["new_entries_allowed"] is True
    assert reached["target_reached"] is True
    assert reached["enforcement_active"] is True
    assert reached["entry_risk_multiplier"] == 0.0
    assert reached["risk_reducing_exits_allowed"] is True
    assert reached["force_liquidation"] is False


def test_daily_target_command_publishes_scaled_target(tmp_path: Path) -> None:
    _policy(tmp_path)
    report = capital_command(
        tmp_path,
        "daily-target",
        account_equity_eur=Decimal("50000"),
        net_daily_pnl_eur=Decimal("100"),
    )
    assert report["status"] == "GO"
    assert report["daily_profit_target_eur"] == 250.0
    assert (
        tmp_path / "output/capital/daily_profit_target.json"
    ).exists()
