from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from stocks.research.phase11_9 import (
    BASE_STRATEGIES,
    ENSEMBLES,
    _aggregate,
    _append_daily_provider_tail,
    _backtest_positive_registry,
    _candidate_duplicate_audit,
    _closed_daily_provider_frame,
    _fold_bootstrap,
    _formula_specifications,
    _frames_fingerprint,
    _load_current_daily,
    _load_daily,
    _load_intraday,
    _parameters,
    _signals,
    _strategy,
    _ten_strategy_pass_audit,
    phase11_9_schema,
)


def _frame(periods: int = 800) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=periods)
    trend = np.linspace(50, 120, periods)
    cycle = np.sin(np.arange(periods) / 12) * 4
    close = trend + cycle
    return pd.DataFrame(
        {
            "open": close * 0.998,
            "high": close * 1.015,
            "low": close * 0.985,
            "close": close,
            "volume": 1_000_000
            + np.cos(np.arange(periods) / 7) * 250_000,
        },
        index=index,
    )


def test_schema_is_portfolio_realistic_and_authority_none(
    tmp_path: Path,
) -> None:
    report = phase11_9_schema(tmp_path)
    contract = report["portfolio_contract"]
    assert contract["whole_shares"] is True
    assert contract["gross_exposure_maximum"] == 1.0
    assert contract["global_security_netting"] is True
    assert contract["base_currency"] == "EUR"
    assert report["EXECUTION_AUTHORITY"] == "NONE"
    assert report["BROKER_CALLS"] == 0


def test_every_base_strategy_and_ensemble_produces_causal_state() -> None:
    frame = _frame()
    strategies = (*BASE_STRATEGIES, *ENSEMBLES)
    for strategy in strategies:
        result = _strategy(frame, strategy, "1d", "responsive")
        assert result.index.equals(frame.index)
        assert result["signal"].dtype == bool
        assert result["score"].notna().all()


def test_future_bar_change_does_not_change_prior_signals() -> None:
    frame = _frame()
    changed = frame.copy()
    changed.loc[changed.index[-1], "close"] *= 4
    for strategy in (*BASE_STRATEGIES, *ENSEMBLES):
        original = _strategy(frame, strategy, "1d", "conservative")
        revised = _strategy(changed, strategy, "1d", "conservative")
        pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


def test_timeframe_profiles_are_bounded_and_ordered() -> None:
    for timeframe in ("1h", "2h", "4h", "1d", "1w", "1mo"):
        responsive = _parameters(timeframe, "responsive")
        balanced = _parameters(timeframe, "balanced")
        conservative = _parameters(timeframe, "conservative")
        assert 1 < responsive["fast"] < responsive["slow"]
        assert 1 < balanced["fast"] < balanced["slow"]
        assert 1 < conservative["fast"] < conservative["slow"]
        assert (
            responsive["channel"]
            <= balanced["channel"]
            <= conservative["channel"]
        )


def test_etf_commodity_trend_is_false_for_non_etf_identity() -> None:
    frame = _frame()
    signals = _signals(
        {"AAPL": frame, "GLD": frame},
        "etf_commodity_trend",
        "1d",
        "responsive",
    )
    assert not signals["AAPL"]["signal"].any()
    assert signals["GLD"]["signal"].any()


def test_closed_daily_aggregation_is_deterministic() -> None:
    frame = _frame()
    first = _aggregate({"A": frame}, "W-FRI")["A"]
    second = _aggregate({"A": frame}, "W-FRI")["A"]
    pd.testing.assert_frame_equal(first, second)
    assert len(first) < len(frame)
    assert first.index.max() <= frame.index.max().normalize()


def test_open_week_and_month_are_not_published_as_closed() -> None:
    weekly_frame = _frame(130)
    monthly_frame = _frame(520).iloc[:-10]
    weekly = _aggregate({"A": weekly_frame}, "W-FRI")["A"]
    monthly = _aggregate({"A": monthly_frame}, "ME")["A"]
    assert weekly.index.max() <= weekly_frame.index.max().normalize()
    assert monthly.index.max() <= monthly_frame.index.max().normalize()


def test_daily_provider_overlay_appends_only_strictly_new_rows() -> None:
    base = _frame(10)
    provider = base.copy()
    provider.loc[provider.index[-1], "close"] *= 1.001
    future = provider.iloc[[-1]].copy()
    future.index = pd.DatetimeIndex([provider.index[-1] + pd.Timedelta(days=1)])
    future.loc[:, ["open", "high", "low", "close"]] *= 1.01
    provider = pd.concat([provider, future])
    combined = _append_daily_provider_tail(
        base,
        provider,
        provider_name="TEST",
    )
    pd.testing.assert_frame_equal(combined.iloc[:-1], base, check_freq=False)
    assert combined.index[-1] == future.index[-1]
    assert combined.attrs["historical_rows_replaced"] == 0
    assert combined.attrs["forward_overlay_rows"] == 1


def test_daily_provider_overlay_blocks_large_overlap_conflict() -> None:
    base = _frame(10)
    provider = base.copy()
    provider.loc[:, "close"] *= 1.20
    future = provider.iloc[[-1]].copy()
    future.index = pd.DatetimeIndex([provider.index[-1] + pd.Timedelta(days=1)])
    provider = pd.concat([provider, future])
    combined = _append_daily_provider_tail(
        base,
        provider,
        provider_name="TEST",
    )
    pd.testing.assert_frame_equal(combined, base)


def test_fold_bootstrap_is_deterministic() -> None:
    folds = pd.DataFrame(
        {
            "period_profit_factor": [1.2, 1.4, 0.9, 1.3],
            "CAGR": [0.1, 0.2, -0.02, 0.15],
            "maximum_drawdown": [-0.1, -0.2, -0.15, -0.12],
        }
    )
    first = _fold_bootstrap(folds, seed=42, runs=1_000)
    second = _fold_bootstrap(folds, seed=42, runs=1_000)
    assert first == second
    assert first["probability_median_pf_above_one"] > 0.5


def test_shortlist_formula_uses_dominant_profile() -> None:
    shortlist = {
        "candidates": [
            {"strategy": "donchian_breakout", "timeframe": "1w"}
        ]
    }
    selections = pd.DataFrame(
        {
            "strategy": ["donchian_breakout"] * 3,
            "timeframe": ["1w"] * 3,
            "selected_profile": [
                "conservative",
                "responsive",
                "conservative",
            ],
        }
    )
    report = _formula_specifications(shortlist, selections)
    formula = report["formulas"][0]
    assert formula["dominant_profile"] == "conservative"
    assert formula["parameters"]["channel"] == 20


def test_candidate_duplicate_audit_collapses_identical_outcomes() -> None:
    shortlist = {
        "candidates": [
            {"strategy": "A", "timeframe": "1w"},
            {"strategy": "B", "timeframe": "1w"},
            {"strategy": "C", "timeframe": "1d"},
        ]
    }
    rows = []
    for strategy, timeframe, cagr in (
        ("A", "1w", 0.1),
        ("B", "1w", 0.1),
        ("C", "1d", 0.2),
    ):
        rows.append(
            {
                "strategy": strategy,
                "timeframe": timeframe,
                "fold_id": "F1",
                "cost_bps": 10.0,
                "period_profit_factor": 1.2,
                "CAGR": cagr,
                "Sharpe": 0.5,
                "maximum_drawdown": -0.1,
                "fill_count": 20,
                "turnover_initial_capital": 2.0,
            }
        )
    report = _candidate_duplicate_audit(
        shortlist, pd.DataFrame(rows)
    )
    assert report["shortlist_candidate_count"] == 3
    assert report["independent_candidate_count"] == 2
    assert report["duplicate_group_count"] == 1


def test_backtest_positive_registry_uses_cost_adjusted_portfolio_gates() -> None:
    summary = pd.DataFrame(
        [
            {
                "strategy": "PASS",
                "timeframe": "1d",
                "confidence": "EVALUABLE",
                "fold_count": 20,
                "median_oos_portfolio_pf": 1.2,
                "median_oos_CAGR": 0.1,
                "cost_50bps_median_pf": 1.05,
                "median_fill_count": 30,
                "worst_oos_drawdown": -0.2,
            },
            {
                "strategy": "FAIL_COST",
                "timeframe": "1d",
                "confidence": "EVALUABLE",
                "fold_count": 20,
                "median_oos_portfolio_pf": 1.2,
                "median_oos_CAGR": 0.1,
                "cost_50bps_median_pf": 0.95,
                "median_fill_count": 30,
                "worst_oos_drawdown": -0.2,
            },
        ]
    )
    report = _backtest_positive_registry(summary)
    assert report["candidate_count"] == 1
    assert report["candidates"][0]["strategy"] == "PASS"


def test_ten_strategy_audit_separates_portfolio_and_benchmark_gates() -> None:
    candidates = []
    diagnostics = []
    for index in range(10):
        strategy = f"S{index}"
        candidates.append(
            {
                "strategy": strategy,
                "timeframe": "1w",
                "confidence": "EVALUABLE",
                "fold_count": 20,
                "median_oos_portfolio_pf": 1.2,
                "median_oos_CAGR": 0.1,
                "cost_50bps_median_pf": 1.05,
                "median_fill_count": 30,
                "worst_oos_drawdown": -0.2,
            }
        )
        diagnostics.append(
            {
                "strategy": strategy,
                "timeframe": "1w",
                "benchmark_incremental_gate": (
                    "GO" if index < 7 else "NO_GO"
                ),
            }
        )
    report = _ten_strategy_pass_audit(
        {"candidates": candidates},
        diagnostics,
        {"independent_candidate_count": 8},
    )
    assert report["status"] == "MINIMUM_TEN_BACKTESTED_STRATEGIES_GO"
    assert report["portfolio_gate_pass_count"] == 10
    assert report["independent_economic_outcome_count"] == 8
    assert report["benchmark_incremental_pass_count"] == 7
    assert report["EXECUTION_AUTHORITY"] == "NONE"


def _write_intraday_fixture(
    root: Path, provider: str, close: float
) -> None:
    dates = pd.date_range("2024-01-02 14:30", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp_utc": dates,
            "date": dates,
            "bar_origin": "NATIVE",
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000,
            "row_hash": [f"{provider}-{index}" for index in range(3)],
        }
    )
    path = (
        root
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
        / f"provider={provider}"
        / "symbol=AAPL"
        / "interval=1h"
        / "source_interval=1h"
        / "bars.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _write_fx_fixture(root: Path) -> None:
    path = (
        root
        / "data"
        / "research"
        / "phase11_4"
        / "private"
        / "eurusd.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": pd.date_range("2023-12-28", periods=20, freq="D"),
            "usd_per_eur": 1.0,
        }
    ).to_parquet(path, index=False)
    canonical = root / "data" / "fx" / "fx_daily.parquet"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2023-12-28", periods=20, freq="D")
    pd.DataFrame(
        {
            "session_date": dates.strftime("%Y-%m-%d"),
            "base_currency": "USD",
            "quote_currency": "EUR",
            "rate": "1.0",
            "available_at": (
                dates.tz_localize("UTC") + pd.Timedelta(hours=23)
            ).astype(str),
            "forward_fill_age": 0,
        }
    ).to_parquet(canonical, index=False)


def _write_daily_fixtures(root: Path) -> None:
    dates = pd.bdate_range("2024-01-02", "2024-01-09")
    close = pd.Series(range(100, 100 + len(dates)), dtype=float)
    frame = pd.DataFrame(
        {
            "session_date": dates,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000.0,
        }
    )
    base_path = (
        root
        / "data"
        / "research"
        / "critical_trading"
        / "yfinance"
        / "AAPL.parquet"
    )
    base_path.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[:4].to_parquet(base_path, index=False)

    provider = frame.rename(columns={"session_date": "timestamp_utc"})
    provider["timestamp_utc"] = pd.to_datetime(
        provider["timestamp_utc"],
        utc=True,
    )
    provider_path = (
        root
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
        / "provider=YFINANCE"
        / "symbol=AAPL"
        / "interval=1d"
        / "source_interval=1d"
        / "bars.parquet"
    )
    provider_path.parent.mkdir(parents=True, exist_ok=True)
    provider.to_parquet(provider_path, index=False)


def test_current_daily_loader_appends_only_completed_provider_tail(
    tmp_path: Path,
) -> None:
    _write_fx_fixture(tmp_path)
    _write_daily_fixtures(tmp_path)

    historical = _load_daily(tmp_path)["AAPL"]
    current = _load_current_daily(
        tmp_path,
        observed_at=datetime(2024, 1, 8, 22, 0, tzinfo=UTC),
    )["AAPL"]

    assert historical.index.max() == pd.Timestamp("2024-01-05")
    assert current.index.max() == pd.Timestamp("2024-01-08")
    pd.testing.assert_frame_equal(
        current.loc[historical.index],
        historical,
        check_freq=False,
    )
    assert current.attrs["historical_rows_replaced"] == 0
    assert current.attrs["forward_overlay_provider"] == "YFINANCE"
    assert current.attrs["provider_selection_policy"] == (
        "CURRENT_TAIL_APPEND_NO_HISTORICAL_REPLACEMENT"
    )
    assert current.attrs["fx_source"] == "CANONICAL_PIT_FX"


def test_daily_provider_filter_excludes_active_session_bar() -> None:
    dates = pd.to_datetime(
        ["2024-01-05", "2024-01-08", "2024-01-09"],
        utc=True,
    )
    frame = pd.DataFrame({"timestamp_utc": dates})

    closed = _closed_daily_provider_frame(
        frame,
        observed_at=datetime(2024, 1, 8, 18, 0, tzinfo=UTC),
    )

    assert closed["timestamp_utc"].tolist() == [dates[0]]


def test_intraday_loader_uses_eodhd_then_yfinance_fallback(
    tmp_path: Path,
) -> None:
    _write_fx_fixture(tmp_path)
    _write_intraday_fixture(tmp_path, "YFINANCE", 100.0)
    _write_intraday_fixture(tmp_path, "EODHD", 200.0)

    preferred = _load_intraday(tmp_path, "1h")
    assert preferred["AAPL"].attrs["provider"] == "EODHD"
    assert preferred["AAPL"]["close"].iloc[0] == 200.0
    preferred_fingerprint = _frames_fingerprint({"1h": preferred})

    eodhd = (
        tmp_path
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
        / "provider=EODHD"
        / "symbol=AAPL"
        / "interval=1h"
        / "source_interval=1h"
        / "bars.parquet"
    )
    eodhd.unlink()
    fallback = _load_intraday(tmp_path, "1h")
    assert fallback["AAPL"].attrs["provider"] == "YFINANCE"
    assert fallback["AAPL"]["close"].iloc[0] == 100.0
    assert _frames_fingerprint({"1h": fallback}) != preferred_fingerprint


def test_current_intraday_loader_selects_freshest_closed_provider(
    tmp_path: Path,
) -> None:
    _write_fx_fixture(tmp_path)
    _write_intraday_fixture(tmp_path, "YFINANCE", 100.0)
    _write_intraday_fixture(tmp_path, "EODHD", 200.0)
    yfinance_path = (
        tmp_path
        / "data"
        / "research"
        / "multitimeframe"
        / "private"
        / "provider=YFINANCE"
        / "symbol=AAPL"
        / "interval=1h"
        / "source_interval=1h"
        / "bars.parquet"
    )
    yfinance = pd.read_parquet(yfinance_path)
    yfinance["timestamp_utc"] = pd.to_datetime(
        yfinance["timestamp_utc"],
        utc=True,
    ) + pd.Timedelta(days=1)
    yfinance.to_parquet(yfinance_path, index=False)

    selected = _load_intraday(
        tmp_path,
        "1h",
        selection_policy="FRESHEST_QUALIFIED_PROVIDER_NO_BLEND",
        observed_at=datetime(2024, 1, 3, 17, 0, tzinfo=UTC),
    )

    frame = selected["AAPL"]
    assert frame.attrs["provider"] == "YFINANCE"
    assert frame.attrs["provider_selection_policy"] == (
        "FRESHEST_QUALIFIED_PROVIDER_NO_BLEND"
    )
    assert {row["provider"] for row in frame.attrs["provider_candidates"]} == {
        "EODHD",
        "YFINANCE",
    }
    assert frame.index.max() == pd.Timestamp("2024-01-03T15:30:00")
