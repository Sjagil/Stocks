from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from stocks.research.phase11_4.acquisition import (
    SecurityIdentity,
    _normalize_split_adjusted_prices,
    _normalize_sharadar,
    resolve_provider_identities,
)
from stocks.research.phase11_4.engine import generate_trades, portfolio_simulation, rsi_wilder, trade_summary
from stocks.research.phase11_4.pipeline import (
    _cluster_bootstrap,
    _corporate_action_conflicts,
    _decision,
    _delisting_stress,
    _repriced_summary,
    _stationary_bootstrap,
    preregister,
)
from stocks.research.phase11_4.store import TABLES, Phase114Store


def _frame(closes: list[float], opens: list[float] | None = None, volume: float = 1_000_000) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    opens = opens or closes
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(left, right) * 1.01 for left, right in zip(opens, closes, strict=True)],
            "low": [min(left, right) * 0.99 for left, right in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": volume,
        },
        index=index,
    )


def test_rsi_signal_enters_and_exits_only_at_next_open() -> None:
    closes = [100.0] * 22 + [90.0, 80.0, 70.0, 71.0, 72.0]
    opens = [100.0] * 25 + [75.0, 74.0]
    trades, signals = generate_trades({"TEST": _frame(closes, opens)}, minimum_median_dollar_volume=1)
    assert signals
    trade = trades[0]
    assert trade.entry_date > trade.signal_date
    assert trade.exit_date is not None
    assert trade.exit_signal_date is not None
    assert trade.exit_date > trade.exit_signal_date


def test_missing_next_exit_open_is_delisting_uncertain() -> None:
    closes = [100.0] * 22 + [90.0, 80.0, 70.0]
    trades, _ = generate_trades({"TEST": _frame(closes)}, minimum_median_dollar_volume=1)
    assert trades[-1].status == "DELISTING_EXECUTION_UNCERTAIN"
    assert trade_summary(trades)["uncertain_delisting_exits"] == 1


def test_penny_and_illiquid_signals_are_blocked() -> None:
    closes = [4.0] * 22 + [3.0, 2.0, 1.0, 1.1]
    trades, signals = generate_trades({"TEST": _frame(closes, volume=10)}, minimum_median_dollar_volume=1_000_000)
    assert not trades
    assert not signals


def test_portfolio_capacity_is_deterministic() -> None:
    frames = {
        "A": _frame([100.0] * 22 + [90.0, 80.0, 70.0, 71.0, 72.0]),
        "B": _frame([100.0] * 22 + [90.0, 80.0, 70.0, 71.0, 72.0]),
    }
    trades, _ = generate_trades(frames, minimum_median_dollar_volume=1)
    result, _, fills = portfolio_simulation(trades, frames, max_positions=1)
    buys = [row for row in fills if row["side"] == "BUY"]
    assert result["maximum_positions"] == 1
    assert result["missed_signals_due_to_capacity"] >= 1
    assert buys[0]["symbol"] == "A"


def test_private_store_has_required_append_only_tables(tmp_path: Path) -> None:
    store = Phase114Store(tmp_path / "private.sqlite3")
    store.initialize()
    store.append("trades", "RUN", [("A", {"return": 0.01})])
    store.append("trades", "RUN", [("A", {"return": 0.01})])
    assert store.counts()["trades"] == 1
    with sqlite3.connect(store.path) as db:
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(TABLES) <= names


def test_preregistration_is_idempotent_and_authority_none(tmp_path: Path) -> None:
    first = preregister(tmp_path)
    second = preregister(tmp_path)
    assert first == second
    assert first["candidate_pre_registered"] is True
    assert first["strategy_authority"] == "NONE"
    assert first["execution_authority"] == "NONE"
    assert first["robustness_grid"]["trial_count"] == 30


def test_survivorship_gate_has_decision_precedence() -> None:
    result = _decision(
        {"survivorship_bias_blocked": False},
        {"periods": {"test": {"trade_profit_factor": 99}}},
        {"test_max_positions_4": {"CAGR": 1, "maximum_drawdown": -0.01}},
        {"per_side_bps": {"25": {"trade_profit_factor": 99}}},
        {"status": "SHARIAH_COHORT_GO"},
        {"Deflated_Sharpe_Ratio": 1, "Probability_of_Backtest_Overfitting": 0},
        {"probability_expectancy_positive": 1},
        {"warnings": {}},
    )
    assert result == "REJECTED_SURVIVORSHIP_DEPENDENT"


def test_rsi_is_causal() -> None:
    original = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0])
    changed_future = original.copy()
    changed_future.iloc[-1] = 1_000.0
    assert rsi_wilder(original, 3).iloc[4] == rsi_wilder(changed_future, 3).iloc[4]


def test_higher_cost_reduces_expectancy() -> None:
    frames = {"TEST": _frame([100.0] * 22 + [90.0, 80.0, 70.0, 71.0, 72.0])}
    trades, _ = generate_trades(frames, minimum_median_dollar_volume=1)
    low = _repriced_summary(trades, 10, "2000-01-01", "2030-01-01")
    high = _repriced_summary(trades, 50, "2000-01-01", "2030-01-01")
    assert high["expectancy"] < low["expectancy"]


def test_delisting_stress_is_monotonic() -> None:
    frames = {"TEST": _frame([100.0] * 22 + [90.0, 80.0, 70.0])}
    trades, _ = generate_trades(frames, minimum_median_dollar_volume=1)
    fifty = _delisting_stress(trades, -0.5)
    total = _delisting_stress(trades, -1.0)
    assert total["expectancy"] < fifty["expectancy"]


def test_sector_cluster_requires_multiple_sectors() -> None:
    import numpy as np

    result = _cluster_bootstrap({"UNKNOWN": [0.1, -0.1]}, np.random.default_rng(1))
    assert result["status"] == "INSUFFICIENT_CLUSTERS"


def test_stationary_bootstrap_is_seed_deterministic() -> None:
    import numpy as np

    values = np.array([0.01, -0.02, 0.03, 0.0])
    first = _stationary_bootstrap(values, np.random.default_rng(114))
    second = _stationary_bootstrap(values, np.random.default_rng(114))
    assert first == second


def test_no_trade_profit_factor_is_null() -> None:
    summary = trade_summary([])
    assert summary["trade_count"] == 0
    assert summary["trade_profit_factor"] is None


def test_security_master_excludes_warrants_and_preserves_listing_window() -> None:
    rows = [
        {
            "table": "SEP",
            "permaticker": "1",
            "ticker": "AAA",
            "name": "AAA Inc",
            "exchange": "NASDAQ",
            "category": "Domestic Common Stock",
            "firstpricedate": "2001-01-01",
            "lastpricedate": "2020-01-01",
            "isdelisted": "Y",
            "sector": "Technology",
            "industry": "Software",
            "currency": "USD",
            "figi": "FIGI",
        },
        {
            "table": "SEP",
            "permaticker": "2",
            "ticker": "AAAW",
            "name": "AAA Warrant",
            "exchange": "NASDAQ",
            "category": "Domestic Common Stock Warrant",
            "firstpricedate": "2001-01-01",
            "lastpricedate": "2020-01-01",
        },
    ]
    result = _normalize_sharadar(rows)
    assert len(result) == 1
    assert result[0].security_id == "SHARADAR:1"
    assert result[0].is_delisted is True


def test_provider_identity_uses_unique_normalized_name_when_ticker_changed() -> None:
    identity = SecurityIdentity(
        security_id="SHARADAR:1",
        ticker="OLD",
        name="Example Corporation",
        exchange="NYSE",
        category="Domestic Common Stock",
        first_price_date="2000-01-01",
        last_price_date="2010-01-01",
        is_delisted=True,
        sector="Technology",
        industry="Software",
        currency="USD",
        figi_hash=None,
        source_hash="HASH",
    )
    provider = [
        {
            "code": "NEW_OLD",
            "name": "Example Corp",
            "exchange": "NYSE",
            "provider_status": "delisted",
            "source_hash": "SOURCE",
        }
    ]
    result = resolve_provider_identities([identity], provider)[0]
    assert result.provider_symbol == "NEW_OLD.US"
    assert result.provider_identity_method == "UNIQUE_NORMALIZED_NAME_EXCHANGE"


def test_split_adjustment_changes_only_pre_split_ohlc_and_volume() -> None:
    identity = {
        "security_id": "SHARADAR:1",
        "ticker": "AAA",
        "first_price_date": "2020-01-01",
        "last_price_date": "2020-01-03",
        "sector": "Technology",
        "currency": "USD",
    }
    prices = [
        {"date": "2020-01-01", "open": 100, "high": 110, "low": 90, "close": 100, "volume": 10},
        {"date": "2020-01-02", "open": 50, "high": 55, "low": 45, "close": 50, "volume": 20},
        {"date": "2020-01-03", "open": 51, "high": 56, "low": 50, "close": 52, "volume": 20},
    ]
    bars, status = _normalize_split_adjusted_prices(identity, prices, [{"date": "2020-01-02", "split": "2/1"}])
    assert status == "EXECUTION_PRICE_GO"
    assert bars[0]["open"] == 50
    assert bars[0]["volume"] == 20
    assert bars[1]["open"] == 50


def test_unexplained_split_adjusted_jump_is_privately_blocked(tmp_path: Path) -> None:
    price_path = tmp_path / "data" / "research" / "phase11_4" / "private" / "pit-bars.parquet"
    price_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "security_id": ["OK", "OK", "BAD", "BAD"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
            "open": [10.0, 10.5, 10.0, 100.0],
            "close": [10.0, 10.5, 10.0, 100.0],
        }
    ).to_parquet(price_path, index=False)

    conflicts = _corporate_action_conflicts(tmp_path)

    assert conflicts == {"BAD"}
    public = (tmp_path / "output" / "research" / "rsi_pit" / "corporate-action-audit.json").read_text()
    assert '"conflicted_security_count": 1' in public
    assert '"BAD"' not in public
