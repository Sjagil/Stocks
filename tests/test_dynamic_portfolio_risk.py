from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from stocks.portfolio.dynamic_risk import build_dynamic_risk_state


def _policy() -> dict[str, object]:
    return {
        "ranking": {"minimum_allocation_score": 0.55},
        "portfolio": {
            "maximum_portfolio_heat": 0.04,
            "maximum_positions": 25,
        },
    }


def _candidate(
    symbol: str,
    score: float,
    *,
    sector: str = "TECH",
    region: str = "US",
    sleeve: str = "STOCK",
) -> dict[str, object]:
    return {
        "ticker": symbol,
        "opportunity_score": score,
        "sector": sector,
        "region": region,
        "sleeve": sleeve,
        "research_allocation_blockers": [],
    }


def _state(**overrides: object) -> dict[str, object]:
    values = {
        "equity_eur": Decimal("50000"),
        "equity_history": [],
        "daily_pnl": [],
        "candidates": [
            _candidate("A", 0.85),
            _candidate("B", 0.80, sector="HEALTH", region="EU"),
            _candidate("C", 0.75, sector="ENERGY", sleeve="ETF"),
        ],
        "policy": _policy(),
        "technical_regime": {"status": "GO", "regime": "BULL_TREND_LOW_VOL"},
        "macro": {
            "regime": {"confidence": 0.8},
            "data_quality": {"status": "GO"},
        },
        "capital_level": 3,
    }
    values.update(overrides)
    return build_dynamic_risk_state(**values)  # type: ignore[arg-type]


def test_capital_tier_and_operational_level_both_cap_positions() -> None:
    report = _state(equity_eur=Decimal("9000"), capital_level=2)

    assert report["equity_band"] == "EUR_1K_TO_10K"
    assert report["tier_maximum_positions"] == 5
    assert report["operational_maximum_positions"] == 3
    assert 0 < report["dynamic_research_maximum_positions"] <= 3
    assert report["exact_equity_public"] is False


def test_micro_account_profile_uses_meaningful_but_bounded_risk() -> None:
    policy = _policy()
    policy["small_account_whole_share"] = {
        "enabled": True,
        "maximum_equity_eur": 10000,
        "maximum_positions": 4,
        "minimum_meaningful_target_weight": 0.10,
        "maximum_stock_weight": 0.35,
        "base_risk_per_trade": 0.01,
        "maximum_risk_per_trade": 0.015,
        "maximum_portfolio_heat": 0.06,
    }
    report = _state(
        equity_eur=Decimal("1870"),
        policy=policy,
        capital_level=2,
        daily_pnl=[(date(2026, 8, 7), Decimal("1"))],
    )

    assert report["small_account_whole_share_mode"] is True
    assert report["base_risk_per_trade"] == 0.01
    assert report["maximum_risk_per_trade"] == 0.015
    assert report["maximum_portfolio_heat"] == 0.06
    assert report["minimum_meaningful_target_weight"] == 0.10
    assert report["maximum_position_weight"] == 0.35
    assert report["tier_maximum_positions"] == 5
    assert report["risk_budget_maximum_positions"] == 6


def test_insufficient_history_never_invents_drawdown_or_loss_streak() -> None:
    report = _state()

    assert report["portfolio_drawdown_status"] == "INSUFFICIENT_HISTORY"
    assert report["portfolio_drawdown_pct"] is None
    assert report["drawdown_velocity_status"] == "INSUFFICIENT_HISTORY"
    assert report["loss_streak_status"] == "INSUFFICIENT_HISTORY"
    assert report["multipliers"]["drawdown"] == 1.0
    assert report["loss_guard"]["status"] == "INSUFFICIENT_HISTORY"
    assert report["new_entries_allowed"] is False


def test_fast_drawdown_and_loss_streak_reduce_risk() -> None:
    start = datetime(2026, 7, 20, tzinfo=UTC)
    equities = [Decimal("100000"), Decimal("99000"), Decimal("95000")]
    equity_history = [
        (start + timedelta(days=index), value)
        for index, value in enumerate(equities)
    ]
    daily_pnl = [
        (date(2026, 7, 20), Decimal("-100")),
        (date(2026, 7, 21), Decimal("-200")),
        (date(2026, 7, 22), Decimal("-300")),
    ]
    report = _state(
        equity_eur=equities[-1],
        equity_history=equity_history,
        daily_pnl=daily_pnl,
    )

    assert report["portfolio_drawdown_pct"] == 0.05
    assert report["drawdown_velocity_per_day"] == 0.025
    assert report["consecutive_loss_sessions"] == 3
    assert report["multipliers"]["drawdown"] == 0.9
    assert report["multipliers"]["drawdown_velocity"] == 0.25
    assert report["multipliers"]["consecutive_loss"] == 0.7
    assert report["multipliers"]["combined"] < 0.2


def test_candidate_scarcity_and_concentration_reduce_dynamic_positions() -> None:
    concentrated = [
        _candidate("A", 0.8),
        _candidate("B", 0.8),
        _candidate("C", 0.8),
        _candidate("D", 0.8),
    ]
    diversified = [
        _candidate("A", 0.8, sector="TECH", region="US", sleeve="STOCK"),
        _candidate("B", 0.8, sector="HEALTH", region="EU", sleeve="ETF"),
        _candidate("C", 0.8, sector="ENERGY", region="ASIA", sleeve="COMMODITY"),
        _candidate("D", 0.8, sector="BOND", region="GLOBAL", sleeve="BOND"),
    ]

    concentrated_report = _state(candidates=concentrated)
    diversified_report = _state(candidates=diversified)

    assert (
        diversified_report["diversification_score"]
        > concentrated_report["diversification_score"]
    )
    assert (
        diversified_report["dynamic_research_maximum_positions"]
        > concentrated_report["dynamic_research_maximum_positions"]
    )


def test_daily_loss_guard_blocks_new_entries_but_not_risk_reduction() -> None:
    report = _state(
        equity_eur=Decimal("50000"),
        daily_pnl=[(date(2026, 7, 31), Decimal("-800"))],
    )

    assert report["loss_guard"]["daily_loss_ratio"] == -0.016
    assert report["new_entries_allowed"] is False
    assert report["risk_reducing_actions_allowed"] is True
    assert report["multipliers"]["loss_guard"] == 0.0


def test_severe_daily_loss_requires_review_without_forced_liquidation() -> None:
    report = _state(
        equity_eur=Decimal("50000"),
        daily_pnl=[(date(2026, 7, 31), Decimal("-1300"))],
    )

    guard = report["loss_guard"]
    assert guard["position_review_required"] is True
    assert guard["healthy_position_forced_liquidation"] is False
    assert "DAILY_LOSS_RISK_REVIEW" in guard["reason_codes"]


def test_weekly_and_monthly_loss_throttles_are_reported() -> None:
    weekly = _state(
        equity_eur=Decimal("50000"),
        daily_pnl=[
            (date(2026, 7, day), Decimal("-450"))
            for day in range(20, 25)
        ],
    )
    monthly = _state(
        equity_eur=Decimal("50000"),
        daily_pnl=[
            (date(2026, 7, day), Decimal("-200"))
            for day in range(1, 22)
        ],
    )

    assert weekly["loss_guard"]["rolling_week_loss_ratio"] == -0.045
    assert weekly["multipliers"]["loss_guard"] == 0.5
    assert monthly["loss_guard"]["rolling_month_loss_ratio"] == -0.084
    assert monthly["multipliers"]["loss_guard"] == 0.25
