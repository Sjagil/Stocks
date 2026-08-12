from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from stocks.portfolio.swing import (
    ActiveSwingOpportunity,
    StrategyTimeframeContract,
    SwingLifecycleState,
    causal_timeframe_asof_join,
    resolve_signal_swing_contract,
    stable_setup_identity,
)
from stocks.signals.timeframe_contracts import (
    declared_research_signal_timeframe_contract,
)


def _contract() -> StrategyTimeframeContract:
    return StrategyTimeframeContract(
        entry_timeframe="15m",
        setup_timeframe="1h",
        context_timeframes=("4h", "1d"),
        structural_timeframe="1d",
        management_timeframe="1h",
        exit_timeframe="15m",
        required_timeframes=("15m", "1h", "4h"),
        optional_timeframes=("1d",),
    )


def test_strategy_timeframe_contract_requires_entry_and_setup_data() -> None:
    contract = _contract()
    assert contract.all_timeframes == ("15m", "1h", "4h", "1d")
    assert contract.as_dict()["all_timeframes_need_not_agree"] is True
    with pytest.raises(ValueError, match="entry_timeframe must be required"):
        StrategyTimeframeContract(
            entry_timeframe="15m",
            setup_timeframe="1h",
            context_timeframes=("4h",),
            structural_timeframe="4h",
            management_timeframe="1h",
            exit_timeframe="1h",
            required_timeframes=("1h", "4h"),
        )


def test_causal_join_uses_latest_available_closed_bar_without_crossing_identity() -> None:
    decisions = pd.DataFrame(
        {
            "security_id": ["A", "B"],
            "decision_timestamp": ["2026-08-10T14:05:00Z"] * 2,
        }
    )
    states = {
        "15m": pd.DataFrame(
            {
                "security_id": ["A", "A", "B"],
                "bar_close_time": [
                    "2026-08-10T13:45:00Z",
                    "2026-08-10T14:00:00Z",
                    "2026-08-10T14:00:00Z",
                ],
                "available_at": [
                    "2026-08-10T13:45:01Z",
                    "2026-08-10T14:06:00Z",
                    "2026-08-10T14:00:01Z",
                ],
                "momentum": [0.1, 9.9, -0.2],
            }
        ),
        "1h": pd.DataFrame(
            {
                "security_id": ["A", "B"],
                "bar_close_time": ["2026-08-10T14:00:00Z"] * 2,
                "available_at": ["2026-08-10T14:00:10Z"] * 2,
                "trend": [0.7, -0.4],
            }
        ),
        "4h": pd.DataFrame(
            {
                "security_id": ["A", "B"],
                "bar_close_time": ["2026-08-10T12:00:00Z"] * 2,
                "available_at": ["2026-08-10T12:00:10Z"] * 2,
                "trend": [0.8, -0.5],
            }
        ),
    }
    result = causal_timeframe_asof_join(decisions, states, _contract())
    rows = result.set_index("security_id")
    assert rows.loc["A", "momentum_15m"] == 0.1
    assert rows.loc["B", "momentum_15m"] == -0.2
    assert rows["timeframe_contract_valid"].all()
    assert not rows["has_1d"].any()
    assert rows["timeframe_blockers"].map(len).eq(0).all()


def test_missing_required_timeframe_is_explicit_and_fail_closed() -> None:
    decisions = pd.DataFrame(
        {
            "security_id": ["A"],
            "decision_timestamp": ["2026-08-10T14:05:00Z"],
        }
    )
    result = causal_timeframe_asof_join(decisions, {}, _contract()).iloc[0]
    assert result["timeframe_contract_valid"] == False  # noqa: E712
    assert result["has_15m"] == False  # noqa: E712
    assert result["timeframe_blockers"] == (
        "REQUIRED_TIMEFRAME_MISSING:15m",
        "REQUIRED_TIMEFRAME_MISSING:1h",
        "REQUIRED_TIMEFRAME_MISSING:4h",
    )


def test_corrupt_future_close_is_rejected_instead_of_forward_filled() -> None:
    decisions = pd.DataFrame(
        {
            "security_id": ["A"],
            "decision_timestamp": ["2026-08-10T14:05:00Z"],
        }
    )
    corrupt = pd.DataFrame(
        {
            "security_id": ["A"],
            "bar_close_time": ["2026-08-10T16:00:00Z"],
            "available_at": ["2026-08-10T14:00:00Z"],
        }
    )
    with pytest.raises(ValueError, match="CAUSALITY_VIOLATION:4h"):
        causal_timeframe_asof_join(decisions, {"4h": corrupt}, _contract())


def test_setup_identity_is_stable_across_tactical_re_evaluations() -> None:
    first = stable_setup_identity(
        symbol="asml",
        strategy_id="pullback-v1",
        setup_origin_timestamp="2026-08-10T12:00:00Z",
        setup_timeframe="1h",
    )
    second = stable_setup_identity(
        symbol="ASML",
        strategy_id="pullback-v1",
        setup_origin_timestamp="2026-08-10T14:00:00+02:00",
        setup_timeframe="1h",
    )
    assert first == second


def test_opportunity_exposes_economic_and_lifecycle_contract_without_authority() -> None:
    opportunity = ActiveSwingOpportunity(
        symbol="ASML",
        security_id="ASML.AS",
        strategy_id="pullback-v1",
        strategy_family="LONG_TREND_PULLBACK",
        setup_origin_timestamp="2026-08-10T12:00:00Z",
        signal_timestamp="2026-08-10T14:00:00Z",
        timeframe_contract=_contract(),
        lifecycle_state=SwingLifecycleState.NEAR_SETUP,
        entry_price_reference=700.0,
        stop=680.0,
        target=750.0,
        expected_holding_period="2-10 sessions",
        expected_net_return=0.03,
        expected_net_r=1.05,
        current_regime="BULL_TREND_LOW_VOL",
        multi_timeframe_alignment="1D_BULL_4H_BULL_1H_PULLBACK",
        multi_timeframe_conflict=None,
        liquidity=0.9,
        spread_bps=2.0,
        estimated_round_trip_cost=1.0,
        whole_share_feasibility="FEASIBLE",
        portfolio_fit="RESEARCH_CANDIDATE",
        shariah_status="SHARIAH_ELIGIBLE_PIT",
        authority_status="AUTHORITY_NONE",
    )
    payload = opportunity.as_dict(observed_at=datetime(2026, 8, 10, 14, 15, tzinfo=UTC))
    assert payload["risk_per_share"] == 20.0
    assert payload["signal_age_seconds"] == 900.0
    assert payload["entry_timeframe"] == "15m"
    assert payload["lifecycle_state"] == "NEAR_SETUP"
    assert payload["submits_orders"] is False
    assert payload["execution_authority"] == "NONE"


def test_signal_contract_is_never_inferred_from_bare_timeframes() -> None:
    resolved = resolve_signal_swing_contract(
        [{"strategy_id": "s1", "timeframe": "15m"}],
        symbol="ASML",
    )
    assert resolved["status"] == "UNDECLARED_RESEARCH_ONLY"
    assert resolved["lifecycle_state"] == "DISCOVERED"
    assert resolved["setup_id"] is None


def test_registry_declares_legacy_native_contract_as_research_only(
    tmp_path: Path,
) -> None:
    target = tmp_path / "config/research_contracts/stocks_strategy_timeframe_registry_v1.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(
        (
            Path(__file__).parents[1]
            / "config/research_contracts/stocks_strategy_timeframe_registry_v1.json"
        ).read_bytes()
    )

    contract = declared_research_signal_timeframe_contract(
        tmp_path,
        {
            "candidate_id": "LEGACY-DAILY",
            "classification": "FROZEN_SHADOW",
            "timeframe": "1d",
        },
    )

    assert contract is not None
    assert contract["entry_timeframe"] == "1d"
    assert contract["setup_timeframe"] == "1d"
    assert contract["context_timeframes"] == []
    assert contract["research_only"] is True
    assert contract["multi_timeframe_edge_claimed"] is False
    assert contract["strategy_authority"] == "NONE"
    assert contract["execution_authority"] == "NONE"


def test_explicit_signal_contract_preserves_setup_identity_and_state() -> None:
    contract = _contract().as_dict()
    rows = [
        {
            "strategy_id": "pullback-v1",
            "timeframe": timeframe,
            "setup_origin_timestamp": "2026-08-10T12:00:00Z",
            "lifecycle_state": "NEAR_SETUP",
            "strategy_timeframe_contract": contract,
        }
        for timeframe in ("15m", "1h", "4h")
    ]
    resolved = resolve_signal_swing_contract(rows, symbol="ASML")
    assert resolved["status"] == "EXPLICIT_VALID"
    assert resolved["lifecycle_state"] == "NEAR_SETUP"
    assert resolved["setup_id"]
    assert resolved["blockers"] == []
