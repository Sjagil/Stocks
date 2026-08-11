from __future__ import annotations

from stocks.portfolio.position_management import evaluate_long_position


def _decision(**overrides: object) -> dict[str, object]:
    values = {
        "entry_price": 100.0,
        "current_price": 110.0,
        "initial_stop": 95.0,
        "previous_stop": 95.0,
        "peak_price": 110.0,
        "atr": 2.0,
        "structural_stop": 98.0,
        "trend_strength": 0.8,
        "volatility_regime": 0.3,
    }
    values.update(overrides)
    return evaluate_long_position(**values)  # type: ignore[arg-type]


def test_first_target_takes_partial_at_two_r() -> None:
    report = _decision()

    assert report["current_r"] == 2.0
    assert report["action"] == "TAKE_PARTIAL_25"
    assert report["first_target_taken"] is True
    assert report["automatic_execution_allowed"] is False


def test_giveback_is_not_activated_before_one_and_half_r() -> None:
    report = _decision(current_price=101.0, peak_price=106.0)

    assert report["peak_r"] == 1.2
    assert report["profit_giveback"] == 0.0
    assert not any("GIVEBACK" in reason for reason in report["reason_codes"])


def test_weak_trend_and_25_percent_giveback_reduces_position() -> None:
    report = _decision(
        current_price=109.0,
        peak_price=112.0,
        trend_strength=0.4,
        first_target_taken=True,
    )

    assert report["peak_r"] == 2.4
    assert report["profit_giveback"] == 0.25
    assert report["action"] == "TAKE_PARTIAL_25"
    assert "TREND_STRENGTH_WEAKENED" in report["reason_codes"]


def test_40_percent_giveback_reduces_half() -> None:
    report = _decision(
        current_price=109.0,
        peak_price=115.0,
        first_target_taken=True,
    )

    assert report["profit_giveback"] == 0.4
    assert report["action"] == "REDUCE_50"


def test_55_percent_giveback_exits() -> None:
    report = _decision(
        current_price=108.0,
        peak_price=118.0,
        first_target_taken=True,
        second_target_taken=True,
    )

    assert report["profit_giveback"] > 0.55
    assert report["action"] == "EXIT"


def test_trailing_stop_never_moves_down() -> None:
    report = _decision(
        current_price=109.0,
        peak_price=110.0,
        previous_stop=106.0,
        first_target_taken=True,
    )

    assert report["proposed_stop"] >= 106.0


def test_stop_breach_has_priority_over_profit_targets() -> None:
    report = _decision(
        current_price=104.0,
        peak_price=120.0,
        previous_stop=105.0,
    )

    assert report["action"] == "EXIT"
    assert report["reason_codes"] == ["PROTECTIVE_STOP_BREACHED"]


def test_invalid_initial_stop_fails_closed() -> None:
    report = _decision(initial_stop=101.0)

    assert report["status"] == "DATA_BLOCKED"
    assert report["action"] == "NO_ACTION"
    assert report["broker_write_calls"] == 0
