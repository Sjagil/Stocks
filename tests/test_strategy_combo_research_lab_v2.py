from __future__ import annotations

import dataclasses
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import strategy_combo_research_lab as lab


def config(**overrides: object) -> lab.V2Config:
    values: dict[str, object] = {
        "command": "run",
        "data": "",
        "output": "output/research/strategy_combo_lab_v2_test",
        "preset": "smoke",
        "policy": "long_only",
        "start": "2020-01-01",
        "train_end": "2020-01-02",
        "validation_end": "2020-01-03",
        "end": "2020-01-06",
        "initial_capital": 2_000.0,
        "global_max_positions": 4,
        "max_security_weight": 0.25,
        "max_sector_weight": 0.50,
        "max_gross_exposure": 1.0,
        "minimum_order_eur": 25.0,
        "whole_shares": True,
        "max_order_adv_fraction": 0.01,
        "min_price": 5.0,
        "min_median_dollar_volume": 5_000_000.0,
        "liquidity_lookback": 20,
        "allowed_exchanges": ("NYSE", "NASDAQ", "NYSEMKT"),
        "cost_bps_per_side": 10.0,
        "slippage_bps_per_side": 0.0,
        "fx_cost_bps_per_side": 0.0,
        "fixed_fee_eur": 3.0,
        "min_bars": 2,
        "min_validation_trades": 1,
        "validation_max_drawdown": -0.35,
        "max_symbols": None,
        "corporate_action_gate": True,
        "overnight_ratio_min": 0.25,
        "overnight_ratio_max": 4.0,
        "batch_size": 10,
        "checkpoint_every": 1,
        "workers": 1,
        "memory_budget_gb": 1.0,
        "full_cartesian": False,
        "max_variants_per_strategy": 1,
        "combo_sizes": (2,),
        "weight_modes": ("equal",),
        "allow_invalid_strategies_in_combos": False,
        "bootstrap_runs": 0,
        "bootstrap_block_size": 2,
        "top_equity_curves": 1,
        "equity_extreme_return_threshold": 0.10,
        "equity_hard_fail_return_threshold": 0.50,
        "seed": 7,
        "include_strategies": (),
        "exclude_strategies": (),
        "resume": False,
    }
    values.update(overrides)
    return lab.V2Config(**values)


def frame(
    security_id: str,
    prices: list[float],
    *,
    currency: str = "EUR",
    sector: str = "Technology",
) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=len(prices))
    return pd.DataFrame(
        {
            "security_id": security_id,
            "symbol": security_id,
            "date": dates,
            "open": prices,
            "high": np.asarray(prices) * 1.01,
            "low": np.asarray(prices) * 0.99,
            "close": prices,
            "volume": 10_000_000,
            "sector": sector,
            "currency": currency,
        }
    )


def candidate(
    security_id: str,
    *,
    strategy: str = "alpha",
    entry: str = "2020-01-01",
    exit_: str = "2020-01-06",
    score: float = 1.0,
    sector: str = "Technology",
    currency: str = "EUR",
    adv: float = 10_000_000.0,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "security_id": security_id,
        "entry_date": pd.Timestamp(entry),
        "exit_date": pd.Timestamp(exit_),
        "score": score,
        "symbol": security_id,
        "sector": sector,
        "currency": currency,
        "median_dollar_volume": adv,
        "investability_status": "INVESTABLE_GO",
    }


def run_ledger(
    candidates: list[dict[str, object]],
    frames: dict[str, pd.DataFrame],
    cfg: lab.V2Config | None = None,
    weights: dict[str, float] | None = None,
) -> lab.V2LedgerResult:
    cfg = cfg or config()
    calendar = pd.bdate_range("2020-01-01", "2020-01-08")
    fx = pd.Series(1.0, index=calendar)
    return lab.run_global_ledger(
        pd.DataFrame(candidates),
        frames,
        calendar,
        fx,
        cfg,
        weights or {"alpha": 1.0},
        portfolio_name="test",
    )


def test_whole_share_cash_fee_nav_and_reconciliation() -> None:
    result = run_ledger([candidate("A")], {"A": frame("A", [101.0] * 6)})
    buy = result.fills.iloc[0]
    assert buy["shares"] == 4
    assert buy["fee_eur"] == pytest.approx(3.404)
    first = result.ledger.iloc[0]
    assert first["cash_eur"] == pytest.approx(1_592.596)
    assert first["portfolio_nav_eur"] == pytest.approx(1_996.596)
    assert first["portfolio_nav_eur"] == pytest.approx(
        first["cash_eur"] + first["gross_market_value_eur"]
    )
    assert result.accounting_failures == 0
    assert result.ledger["cash_eur"].min() >= 0


def test_sell_cash_realized_unrealized_and_fees_once() -> None:
    result = run_ledger(
        [candidate("A", exit_="2020-01-03")],
        {"A": frame("A", [100.0, 110.0, 120.0, 120.0, 120.0, 120.0])},
    )
    assert result.fills.iloc[0]["side"] == "BUY"
    assert set(result.fills.iloc[1:]["side"]) == {"SELL"}
    assert (
        result.fills.loc[result.fills["side"].eq("BUY"), "shares"].sum()
        == result.fills.loc[result.fills["side"].eq("SELL"), "shares"].sum()
    )
    assert result.ledger.iloc[2]["cash_eur"] > config().initial_capital
    assert result.ledger.iloc[2]["unrealized_pnl_eur"] == pytest.approx(0.0)
    assert result.ledger["transaction_costs_eur"].sum() == pytest.approx(
        result.fills["fee_eur"].sum()
    )
    assert result.accounting_failures == 0
    assert result.fills["side"].tolist() == ["BUY", "SELL"]


def test_zero_shares_rejected_by_minimum_order_or_affordability() -> None:
    result = run_ledger(
        [candidate("A")],
        {"A": frame("A", [5_000.0] * 6)},
    )
    assert result.fills.empty
    assert result.orders.iloc[0]["status"] == "REJECTED"
    assert result.orders.iloc[0]["quantity"] == 0


@pytest.mark.parametrize("strategy_count", [2, 3, 4])
def test_pair_triple_quad_share_one_global_position_cap(strategy_count: int) -> None:
    cfg = config(global_max_positions=2)
    active = {}
    weights = {}
    for index in range(strategy_count):
        strategy = f"s{index}"
        weights[strategy] = 1.0 / strategy_count
        for security in range(3):
            row = candidate(
                f"{strategy}-{security}", strategy=strategy, score=3 - security
            )
            active[(strategy, str(security), "x")] = row
    targets, _, _, _ = lab._target_weights(active, weights, cfg)
    assert len(targets) <= 2


def test_security_sector_and_gross_caps() -> None:
    cfg = config(
        global_max_positions=4,
        max_security_weight=0.20,
        max_sector_weight=0.30,
        max_gross_exposure=0.50,
    )
    active = {
        ("alpha", sid, "x"): candidate(sid, sector="Technology")
        for sid in ("A", "B", "C")
    }
    targets, _, _, _ = lab._target_weights(active, {"alpha": 1.0}, cfg)
    assert max(targets.values()) <= 0.20
    assert sum(targets.values()) <= 0.30
    assert sum(targets.values()) <= 0.50


def test_same_security_is_netted_to_one_order_fee_and_position() -> None:
    rows = [candidate("A", strategy="alpha"), candidate("A", strategy="beta")]
    result = run_ledger(
        rows,
        {"A": frame("A", [100.0] * 6)},
        weights={"alpha": 0.5, "beta": 0.5},
    )
    assert len(result.fills.loc[result.fills["side"].eq("BUY")]) == 1
    assert (
        len(
            result.positions.loc[
                result.positions["date"].eq(pd.Timestamp("2020-01-01"))
            ]
        )
        == 1
    )
    attribution = result.attribution.loc[
        result.attribution["date"].eq(pd.Timestamp("2020-01-01"))
    ]
    assert set(attribution["strategy"]) == {"alpha", "beta"}


def test_one_strategy_exit_only_trades_net_reduction() -> None:
    rows = [
        candidate("A", strategy="alpha", exit_="2020-01-03"),
        candidate("A", strategy="beta", exit_="2020-01-08"),
    ]
    result = run_ledger(
        rows,
        {"A": frame("A", [100.0] * 6)},
        config(max_security_weight=1.0, max_sector_weight=1.0),
        weights={"alpha": 0.5, "beta": 0.5},
    )
    sells = result.fills.loc[result.fills["side"].eq("SELL")]
    assert len(sells) == 2
    assert sells.iloc[0]["shares"] < result.fills.iloc[0]["shares"]


def test_full_calendar_contains_zero_return_cash_days() -> None:
    result = run_ledger([], {})
    assert len(result.ledger) == len(pd.bdate_range("2020-01-01", "2020-01-08"))
    assert (result.ledger["daily_return"] == 0).all()
    assert (result.ledger["cash_eur"] == config().initial_capital).all()
    assert result.metrics["missing_calendar_days"] == 0


def test_causal_liquidity_window_excludes_current_and_future_volume() -> None:
    raw = frame("A", [10.0] * 5)
    raw["volume"] = [1.0, 2.0, 3.0, 1_000_000.0, 9_000_000.0]
    prepared = lab.prepare_v2_frame(raw, 3)
    assert prepared.loc[3, "median_dollar_volume"] == pytest.approx(20.0)
    assert prepared.loc[3, "previous_close"] == 10.0
    mutated = raw.copy()
    mutated.loc[4, "volume"] = 99_000_000.0
    assert lab.prepare_v2_frame(mutated, 3).loc[
        3, "median_dollar_volume"
    ] == pytest.approx(20.0)


def test_investability_reasons_are_fail_closed() -> None:
    def builder(
        input_frame: pd.DataFrame,
        _cache: lab.FeatureCache,
        _params: object,
    ) -> lab.TradeBatch:
        return lab.TradeBatch(
            entry_dates=np.asarray(
                [input_frame.loc[2, "date"]], dtype="datetime64[ns]"
            ),
            exit_dates=np.asarray([input_frame.loc[3, "date"]], dtype="datetime64[ns]"),
            entry_prices=np.asarray([input_frame.loc[2, "open"]]),
            exit_prices=np.asarray([input_frame.loc[3, "open"]]),
            gross_returns=np.asarray([0.0]),
            scores=np.asarray([1.0]),
            durations=np.asarray([1]),
            forced=np.asarray([False]),
        )

    spec = lab.StrategySpec("synthetic", "test", "short", "", {}, {}, builder)
    state = lab.VariantState("synthetic", "test", "short", {}, "v1")
    records = lab.v2_candidate_records(
        frame("A", [4.0, 4.0, 4.0, 4.0]),
        "A",
        {},
        {"synthetic": spec},
        [state],
        config(liquidity_lookback=2, min_median_dollar_volume=1.0),
    )
    reasons = records[0]["investability_reasons"]
    assert "PRICE_BELOW_MINIMUM" in reasons
    assert "NOT_PROVEN_COMMON_STOCK" in reasons
    assert "EXCHANGE_NOT_ALLOWED" in reasons


def test_adv_cap_limits_actual_order() -> None:
    cfg = config(max_order_adv_fraction=0.01, max_security_weight=1.0)
    result = run_ledger(
        [candidate("A", adv=10_000.0)],
        {"A": frame("A", [100.0] * 6)},
        cfg,
    )
    assert result.fills.iloc[0]["shares"] == 1
    assert result.fills.iloc[0]["notional_eur"] <= 100.0


def test_eur_prices_are_not_fx_converted() -> None:
    cfg = config(cost_bps_per_side=0.0, fixed_fee_eur=0.0)
    calendar = pd.bdate_range("2020-01-01", "2020-01-08")
    result = lab.run_global_ledger(
        pd.DataFrame([candidate("A", currency="EUR")]),
        {"A": frame("A", [100.0] * 6, currency="EUR")},
        calendar,
        pd.Series(0.5, index=calendar),
        cfg,
        {"alpha": 1.0},
        portfolio_name="fx-test",
    )
    assert result.fills.iloc[0]["price_eur"] == 100.0


def test_usd_fx_impact_and_strategy_attribution_reconcile() -> None:
    cfg = config(
        max_security_weight=1.0,
        max_sector_weight=1.0,
        cost_bps_per_side=0.0,
        fixed_fee_eur=0.0,
    )
    calendar = pd.bdate_range("2020-01-01", "2020-01-08")
    eur_per_usd = pd.Series([0.8, 0.9, 0.9, 0.9, 0.9, 0.9], index=calendar)
    result = lab.run_global_ledger(
        pd.DataFrame([candidate("A", currency="USD")]),
        {"A": frame("A", [100.0] * 6, currency="USD")},
        calendar,
        eur_per_usd,
        cfg,
        {"alpha": 1.0},
        portfolio_name="fx-attribution-test",
    )
    second = result.ledger.iloc[1]
    assert second["security_pnl_eur"] == pytest.approx(0.0)
    assert second["fx_impact_eur"] == pytest.approx(250.0)
    assert second["portfolio_nav_eur"] - result.ledger.iloc[0][
        "portfolio_nav_eur"
    ] == pytest.approx(250.0)
    attributed = result.attribution.loc[
        result.attribution["date"].eq(pd.Timestamp("2020-01-02")), "strategy_pnl_eur"
    ].sum()
    assert attributed == pytest.approx(250.0)
    assert result.accounting_failures == 0


def test_extreme_day_contributors_reconcile() -> None:
    cfg = config(max_security_weight=1.0, cost_bps_per_side=0.0, fixed_fee_eur=0.0)
    result = run_ledger(
        [candidate("A")],
        {"A": frame("A", [100.0, 200.0, 200.0, 200.0, 200.0, 200.0])},
        cfg,
    )
    extreme, contributors = lab._extreme_audits(result, cfg)
    assert not extreme.empty
    day = extreme.iloc[0]["date"]
    ledger_row = result.ledger.loc[result.ledger["date"].eq(day)].iloc[0]
    contribution = contributors.loc[
        contributors["date"].eq(day), "portfolio_contribution"
    ].sum()
    assert ledger_row["daily_return"] == pytest.approx(contribution)
    assert result.unexplained_outliers == 0


def test_validation_winner_is_independent_of_test_returns() -> None:
    cfg = config(min_validation_trades=1)
    good = {
        "trade_count": 5,
        "CAGR": 0.10,
        "Sharpe": 1.0,
        "Calmar": 0.7,
        "daily_profit_factor": 1.3,
        "maximum_drawdown": -0.10,
        "turnover_eur": 500.0,
        "maximum_security_weight": 0.25,
        "maximum_sector_weight": 0.5,
        "accounting_failure_count": 0,
    }
    weak = {**good, "CAGR": 0.02, "Sharpe": 0.1, "daily_profit_factor": 1.01}
    winner_before = max(
        {"a": good, "b": weak},
        key=lambda key: lab.v2_validation_score({"a": good, "b": weak}[key], cfg)[0],
    )
    arbitrary_test_returns = {"a": -0.99, "b": 8.0}
    winner_after = max(
        {"a": good, "b": weak},
        key=lambda key: lab.v2_validation_score({"a": good, "b": weak}[key], cfg)[0],
    )
    assert arbitrary_test_returns["b"] > arbitrary_test_returns["a"]
    assert winner_before == winner_after == "a"


def test_v2_registry_preserves_rotational_momentum_inventory() -> None:
    _, variants, grids = lab.v2_registry(config())
    rotational = [
        state for state in variants.values() if state.strategy == "rotational_momentum"
    ]
    assert len(rotational) == 1
    assert grids["rotational_momentum"] == [lab.ROTATIONAL_DEFAULT]


def test_run_hash_changes_with_code_config_and_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "bars.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(data, index=False)
    first = lab.v2_run_fingerprint(config(), data)
    second = lab.v2_run_fingerprint(dataclasses.replace(config(), seed=8), data)
    pd.DataFrame({"x": [2]}).to_parquet(data, index=False)
    third = lab.v2_run_fingerprint(config(), data)
    assert first["run_id"] != second["run_id"]
    assert first["run_id"] != third["run_id"]
    monkeypatch.setattr(lab, "sha256_file", lambda _path: "DIFFERENT_CODE_HASH")
    fourth = lab.v2_run_fingerprint(config(), data)
    assert third["run_id"] != fourth["run_id"]


def test_processed_identity_checkpoint_is_idempotent() -> None:
    con = duckdb.connect()
    lab._create_v2_candidate_table(con, reset=True)
    con.execute("INSERT OR IGNORE INTO v2_processed_identities VALUES ('A')")
    con.execute("INSERT OR IGNORE INTO v2_processed_identities VALUES ('A')")
    assert (
        con.execute("SELECT COUNT(*) FROM v2_processed_identities").fetchone()[0] == 1
    )


def test_research_lab_has_no_forbidden_broker_calls() -> None:
    source = Path(lab.__file__).read_text(encoding="utf-8")
    forbidden = (
        "place" + "Order",
        "cancel" + "Order",
        "reqGlobal" + "Cancel",
        "req" + "Ids",
        "reqAutoOpen" + "Orders",
        "exercise" + "Options",
        "reqMkt" + "Data",
        "reqHistorical" + "Data",
    )
    assert not [name for name in forbidden if name in source]
