from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stocks.research.phase11_13 import (
    AUTHORITY,
    FROZEN_STRATEGIES,
    MULTI_ASSET_ETFS,
    PROHIBITED_PRODUCT_PROXIES,
    _current_attestations,
    _forward_session_audit,
    _qualification,
    _run_inverse_volatility_portfolio,
    phase11_13_schema,
)


def test_schema_has_exact_five_strategies_and_no_authority(
    tmp_path: Path,
) -> None:
    report = phase11_13_schema(tmp_path)
    assert len(FROZEN_STRATEGIES) == 5
    assert set(report["strategies"]) == {
        "FOUR_HOUR_STOCK_TREND_PULLBACK",
        "DAILY_DONCHIAN_BREAKOUT",
        "DAILY_UPTREND_MEAN_REVERSION",
        "WEEKLY_CROSS_SECTIONAL_MOMENTUM",
        "MULTI_ASSET_INVERSE_VOL_TREND",
    }
    assert report["EXECUTION_AUTHORITY"] == "NONE"
    assert report["BROKER_CALLS"] == 0


def test_qualification_never_grants_deployment() -> None:
    summary = pd.DataFrame(
        [
            {
                "strategy_id": "EXAMPLE",
                "research_pass": True,
                "robust_pass": True,
                "deployable_pass": False,
            }
        ]
    )
    report = _qualification(summary)
    assert report["research_pass_count"] == 1
    assert report["robust_pass_count"] == 1
    assert report["deployable_pass_count"] == 0
    assert report["automatic_live_activation"] is False


def test_authority_contract_is_fail_closed() -> None:
    assert AUTHORITY["FINANCIAL_FINALIST_GO"] is False
    assert AUTHORITY["STRATEGY_AUTHORITY"] == "NONE"
    assert AUTHORITY["EXECUTION_AUTHORITY"] == "NONE"
    assert AUTHORITY["ORDER_CALLS"] == 0


def test_multi_asset_universe_excludes_known_prohibited_proxies() -> None:
    assert PROHIBITED_PRODUCT_PROXIES == {"DBC", "TLT"}
    assert MULTI_ASSET_ETFS.isdisjoint(PROHIBITED_PRODUCT_PROXIES)
    assert FROZEN_STRATEGIES[
        "MULTI_ASSET_INVERSE_VOL_TREND"
    ]["symbols"].isdisjoint(PROHIBITED_PRODUCT_PROXIES)


def test_forward_session_audit_only_counts_bars_after_boundary() -> None:
    boundary = {
        "qualification_hash": "ABC",
        "robust_strategy_ids": ["A", "B"],
        "data_end_by_strategy": {
            "A": "2026-07-24T00:00:00",
            "B": "2026-07-24T00:00:00",
        },
    }
    report = _forward_session_audit(
        boundary,
        [
            {
                "strategy_id": "A",
                "closed_bar_timestamp": "2026-07-24T00:00:00",
            },
            {
                "strategy_id": "A",
                "closed_bar_timestamp": "2026-07-25T00:00:00",
            },
            {
                "strategy_id": "B",
                "closed_bar_timestamp": "2026-07-23T00:00:00",
            },
        ],
    )
    assert report["status"] == "INDEPENDENT_FORWARD_SESSION_PARTIAL"
    assert report["independent_session_count"] == 1
    assert report["per_strategy"]["A"]["complete"] is True
    assert report["per_strategy"]["B"]["complete"] is False


def test_forward_session_audit_is_complete_for_every_survivor() -> None:
    boundary = {
        "qualification_hash": "ABC",
        "robust_strategy_ids": ["A", "B"],
        "data_end_by_strategy": {
            "A": "2026-07-24T00:00:00",
            "B": "2026-07-24T00:00:00",
        },
    }
    report = _forward_session_audit(
        boundary,
        [
            {
                "strategy_id": "A",
                "closed_bar_timestamp": "2026-07-25T00:00:00",
            },
            {
                "strategy_id": "B",
                "closed_bar_timestamp": "2026-07-28T00:00:00",
            },
        ],
    )
    assert report["status"] == "INDEPENDENT_FORWARD_SESSION_COMPLETE"
    assert report["completed_strategy_count"] == 2


def test_forward_session_audit_rejects_future_period_label() -> None:
    boundary = {
        "qualification_hash": "ABC",
        "robust_strategy_ids": ["A"],
        "data_end_by_strategy": {"A": "2026-07-24T00:00:00"},
    }
    report = _forward_session_audit(
        boundary,
        [
            {
                "strategy_id": "A",
                "closed_bar_timestamp": "2026-07-31T00:00:00",
                "_observed_at": "2026-07-28T20:00:00+00:00",
            }
        ],
    )
    assert report["status"] == "NOT_YET_OBSERVED"
    assert report["independent_session_count"] == 0


def test_current_attestations_are_bounded_by_observation_time(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "config"
        / "screener"
        / "shariah_attestations_v1.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "attestations": [
                    {
                        "symbol": "AAPL",
                        "status": "SHARIAH_ELIGIBLE_PIT",
                        "screened_at": "2026-07-01T00:00:00Z",
                        "expires_at": "2026-08-01T00:00:00Z",
                    },
                    {
                        "symbol": "OLD",
                        "status": "SHARIAH_ELIGIBLE_PIT",
                        "screened_at": "2026-06-01T00:00:00Z",
                        "expires_at": "2026-06-30T00:00:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    decision_time = datetime(2026, 7, 28, tzinfo=UTC)
    assert _current_attestations(tmp_path, decision_time) == {"AAPL"}


def test_inverse_volatility_portfolio_is_causal_and_whole_share() -> None:
    dates = pd.date_range("2022-01-03", periods=130, freq="B")
    slow_returns = 0.001 + 0.002 * np.sin(np.arange(len(dates)) / 5)
    fast_returns = 0.001 + 0.02 * np.sin(np.arange(len(dates)) / 3)
    frames = {
        "LOW_VOL": _frame(dates, 100 * np.cumprod(1 + slow_returns)),
        "HIGH_VOL": _frame(dates, 100 * np.cumprod(1 + fast_returns)),
    }
    signals = {
        symbol: pd.DataFrame(
            {"signal": True, "score": score},
            index=dates,
        )
        for symbol, score in (("LOW_VOL", 2.0), ("HIGH_VOL", 1.0))
    }
    start, cutoff, end = dates[70], dates[100], dates[115]
    original = _run_inverse_volatility_portfolio(
        frames,
        signals,
        start=start,
        end=end,
        cost_bps=10.0,
    )
    changed = {symbol: frame.copy() for symbol, frame in frames.items()}
    changed["HIGH_VOL"].loc[
        changed["HIGH_VOL"].index > cutoff,
        ["open", "high", "low", "close"],
    ] *= 5
    perturbed = _run_inverse_volatility_portfolio(
        changed,
        signals,
        start=start,
        end=end,
        cost_bps=10.0,
    )
    left = original["ledger"].loc[
        original["ledger"]["date"] <= cutoff,
        ["date", "nav_eur", "inverse_volatility_weights"],
    ].reset_index(drop=True)
    right = perturbed["ledger"].loc[
        perturbed["ledger"]["date"] <= cutoff,
        ["date", "nav_eur", "inverse_volatility_weights"],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)
    assert original["fills"]["shares"].map(
        lambda value: float(value).is_integer()
    ).all()
    assert original["ledger"]["gross_exposure"].max() <= 1.0
    first_weights = json.loads(
        original["ledger"]["inverse_volatility_weights"].iloc[0]
    )
    assert first_weights["LOW_VOL"] > first_weights["HIGH_VOL"]


def _frame(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=index,
    )
