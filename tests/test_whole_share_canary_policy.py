from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from stocks.capital.canary import (
    build_small_account_canary_matrix,
    evaluate_whole_share_canary,
)
from stocks.live.automatic import select_candidate


PROJECT_ROOT = Path(__file__).parents[1]


def _evaluate(**overrides):
    values = {
        "asset_class": "STOCK",
        "instrument_currency": "EUR",
        "desired_qty": Decimal("3"),
        "account_equity_eur": Decimal("1870"),
        "available_cash_eur": Decimal("1870"),
        "reserved_cash_eur": Decimal("0"),
        "entry_price_local": Decimal("70"),
        "protective_stop_local": Decimal("65.5"),
        "take_profit_local": Decimal("85"),
        "fx_rate_to_eur": Decimal("1"),
        "normal_risk_budget_eur": Decimal("10.5"),
        "normal_maximum_position_weight": Decimal("0.35"),
        "normal_maximum_portfolio_heat_pct": Decimal("0.06"),
        "liquidity_notional_eur": Decimal("100000"),
        "existing_position_notional_eur": Decimal("0"),
        "existing_total_exposure_eur": Decimal("0"),
        "existing_portfolio_risk_eur": Decimal("0"),
    }
    values.update(overrides)
    return evaluate_whole_share_canary(PROJECT_ROOT, **values)


def test_on_like_normal_two_downscales_to_one_without_legacy_cap() -> None:
    result = _evaluate(desired_qty=Decimal("2"))

    assert result["status"] == "GO"
    assert result["desired_qty"] == 2
    assert result["normal_allowed_qty"] == 2
    assert result["canary_qty"] == 1
    assert result["sizing_reason"] == "CANARY_DOWNSCALED_TO_ONE_SHARE"
    assert Decimal(result["actual_notional_eur"]) == Decimal("70")
    assert Decimal(result["actual_notional_eur"]) > Decimal("10")
    assert result["notional_cap_role"] == "SECONDARY_EMERGENCY_BACKSTOP"


def test_fractional_quantity_is_rejected_without_silent_rounding() -> None:
    result = _evaluate(desired_qty=Decimal("1.5"))

    assert result["status"] == "NO_GO"
    assert result["blocking_reason"] == "FRACTIONAL_QUANTITY_FORBIDDEN"
    assert result["canary_qty"] == 0


@pytest.mark.parametrize("asset_class", ["STOCK", "ETF", "COMMODITY_VEHICLE"])
def test_supported_asset_classes_share_whole_share_principle(
    asset_class: str,
) -> None:
    result = _evaluate(asset_class=asset_class)

    assert result["status"] == "GO"
    assert result["canary_qty"] == 1
    assert result["fractional_shares_allowed"] is False


def test_price_above_ten_can_pass_when_one_share_is_safe() -> None:
    result = _evaluate(
        desired_qty=Decimal("1"),
        entry_price_local=Decimal("250"),
        protective_stop_local=Decimal("248"),
        take_profit_local=Decimal("300"),
    )

    assert result["status"] == "GO"
    assert result["canary_qty"] == 1
    assert Decimal(result["actual_notional_eur"]) == Decimal("250")


def test_expensive_share_returns_zero_without_fractional_fallback() -> None:
    result = _evaluate(
        desired_qty=Decimal("1"),
        entry_price_local=Decimal("500"),
        protective_stop_local=Decimal("450"),
        take_profit_local=Decimal("600"),
    )

    assert result["status"] == "NO_GO"
    assert result["canary_qty"] == 0
    assert result["blocking_reason"] == "WHOLE_SHARE_RISK_INFEASIBLE"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"available_cash_eur": Decimal("50")}, "INSUFFICIENT_CASH_FOR_ONE_SHARE"),
        (
            {"existing_position_notional_eur": Decimal("280.5")},
            "CONCENTRATION_LIMIT",
        ),
        (
            {"existing_portfolio_risk_eur": Decimal("9.35")},
            "PORTFOLIO_HEAT_LIMIT",
        ),
        ({"liquidity_notional_eur": Decimal("50")}, "LIQUIDITY_LIMIT"),
    ],
)
def test_zero_share_reasons_are_explicit(overrides, reason: str) -> None:
    result = _evaluate(**overrides)

    assert result["canary_qty"] == 0
    assert result["blocking_reason"] == reason


def test_economically_bad_one_share_trade_is_blocked() -> None:
    result = _evaluate(
        desired_qty=Decimal("1"),
        take_profit_local=Decimal("70.25"),
    )

    assert result["status"] == "NO_GO"
    assert result["blocking_reason"] == "ECONOMICALLY_TOO_SMALL"


def test_foreign_currency_price_and_risk_are_normalized_to_eur() -> None:
    foreign = _evaluate(
        desired_qty=Decimal("1"),
        instrument_currency="USD",
        entry_price_local=Decimal("100"),
        protective_stop_local=Decimal("95"),
        take_profit_local=Decimal("120"),
        fx_rate_to_eur=Decimal("0.80"),
    )

    assert foreign["status"] == "GO"
    assert Decimal(foreign["share_price_eur"]) == Decimal("80.00")
    assert Decimal(foreign["risk_per_share_eur"]) > Decimal("4")
    assert foreign["fx_rate_to_eur"] == "0.80"


def test_small_account_matrix_covers_all_required_equity_price_pairs() -> None:
    matrix = build_small_account_canary_matrix(PROJECT_ROOT)

    assert len(matrix) == 55
    assert {row["account_equity_eur"] for row in matrix} == {
        "1000",
        "1870",
        "2500",
        "5000",
        "10000",
    }
    assert {row["share_price_eur"] for row in matrix} == {
        "5",
        "10",
        "25",
        "50",
        "70",
        "100",
        "150",
        "250",
        "500",
        "1000",
        "2000",
    }
    assert all(
        int(row["level1_canary_qty"] or 0) >= 0 for row in matrix
    )


def test_candidate_ranking_has_no_cheap_share_input_or_bias() -> None:
    allowlist = {
        "status": "GO",
        "strategies": [
            {
                "strategy_id": "QUALITY",
                "allowed_symbols": ["EXPENSIVE", "CHEAP"],
                "training_data_end": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    observation = {
        "observations": [
            {
                "strategy_id": "QUALITY",
                "independent_forward_session": True,
                "observation_status": "OBSERVATION_COMPLETE",
                "current_attested_target_weights": {
                    "EXPENSIVE": 0.10,
                    "CHEAP": 0.10,
                },
            }
        ]
    }
    base = {
        "strategy_id": "QUALITY",
        "action": "BUY",
        "data_freshness": "FRESH",
        "data_timestamp": "2026-02-01T00:00:00+00:00",
        "stop_loss": "1",
        "take_profit_1": "2",
    }
    selected, blockers = select_candidate(
        allowlist,
        observation,
        [
            {**base, "ticker": "CHEAP", "confidence_score": "0.6"},
            {**base, "ticker": "EXPENSIVE", "confidence_score": "0.9"},
        ],
    )

    assert blockers == []
    assert selected is not None
    assert selected["symbol"] == "EXPENSIVE"

