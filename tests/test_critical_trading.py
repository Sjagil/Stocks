from __future__ import annotations

import pandas as pd
import json

import main

from stocks.research.critical_trading import (
    _attach_yfinance_provenance,
    _backtest_trades,
    _five_bar_inverted_signals,
    _five_bar_signals,
    _bearish_engulfing_long_signals,
    _cross_provider_comparison,
    _ma_channel_signals,
    _ma_crossover_signals,
    _trade_object_summary,
    Trade,
    critical_trading_schema,
)


def _frame(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000_000
    frame.index = pd.date_range("2020-01-01", periods=len(frame), freq="B")
    return frame


def test_schema_is_offline_and_flags_discretionary_strategy() -> None:
    schema = critical_trading_schema()
    assert schema["status"] == "OFFLINE_RESEARCH_ONLY"
    assert schema["strategies"]["supply_demand_market_structure"]["status"] == "SPECIFICATION_BLOCKED"
    assert schema["financial_calls"] == {"place_order": 0, "cancel_order": 0, "global_cancel": 0}


def test_schema_cli_is_offline(capsys) -> None:
    assert main.main(["research", "critical-trading", "schema"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "OFFLINE_RESEARCH_ONLY"


def test_five_bar_uses_only_prior_bars() -> None:
    frame = _frame([(10, 11, 9, 10)] * 5 + [(9, 10, 8, 9), (10, 12, 9, 11), (11, 13, 10, 12)])
    entry, exit_signal, _ = _five_bar_signals(frame)
    assert not entry.iloc[4]
    assert entry.iloc[5]
    assert exit_signal.iloc[6]


def test_signals_execute_at_next_open_and_charge_two_sides() -> None:
    frame = _frame([(10, 11, 9, 10)] * 5 + [(9, 10, 8, 9), (20, 21, 19, 20), (30, 31, 29, 30)])
    trades = _backtest_trades("AAA", frame, _five_bar_signals, cost_bps=10)
    assert len(trades) == 1
    assert trades[0].entry_price == 20
    assert trades[0].exit_price == 30
    assert trades[0].net_return < 0.5


def test_inverted_rule_is_not_accidentally_equal_to_contrarian() -> None:
    frame = _frame([(10, 11, 9, 10)] * 5 + [(12, 13, 11, 12)])
    contrarian, _, _ = _five_bar_signals(frame)
    inverted, _, _ = _five_bar_inverted_signals(frame)
    assert not contrarian.iloc[-1]
    assert inverted.iloc[-1]


def test_ma_cross_waits_for_complete_slow_average() -> None:
    frame = _frame([(10, 10, 10, 10)] * 5 + [(20, 20, 20, 20)] * 3)
    entry, _, _ = _ma_crossover_signals(2, 5, 2, 5)(frame)
    assert not entry.iloc[:5].any()
    assert entry.iloc[5]


def test_ma_channel_requires_five_complete_bars() -> None:
    frame = _frame([(10, 10, 9, 9.5)] * 10 + [(20, 21, 19, 20)] * 5)
    entry, _, _ = _ma_channel_signals(frame)
    assert not entry.iloc[-2]
    assert entry.iloc[-1]


def test_bearish_engulfing_long_pattern_is_mechanical() -> None:
    frame = _frame([(10, 12, 9, 11), (12, 13, 8, 9)])
    entry, _, _ = _bearish_engulfing_long_signals(frame)
    assert entry.iloc[-1]


def test_cross_provider_comparison_does_not_merge_returns() -> None:
    local = {"strategy": {"trade_count": 10, "win_rate": 0.6, "trade_profit_factor": 1.2}}
    yahoo = {"strategy": {"trade_count": 12, "win_rate": 0.55, "trade_profit_factor": 1.1}}
    report = _cross_provider_comparison(local, yahoo)
    assert report["comparable_strategy_count"] == 1
    assert report["same_pf_direction_count"] == 1
    assert report["rows"][0]["local_trade_count"] == 10
    assert report["rows"][0]["yfinance_trade_count"] == 12


def test_trade_object_summary_exposes_holding_and_censoring() -> None:
    trades = [
        Trade("AAA", "2020-01-01", "2020-01-03", 10, 11, 0.1, "SIGNAL"),
        Trade("AAA", "2020-01-01", "2020-02-01", 10, 9, -0.1, "END_OF_DATA"),
    ]
    summary = _trade_object_summary(trades)
    assert summary["maximum_holding_days"] == 31
    assert summary["end_of_data_forced_exits"] == 1


def test_yfinance_cache_provenance_survives_parquet_roundtrip(
    tmp_path,
) -> None:
    frame = _attach_yfinance_provenance(
        _frame([(10, 11, 9, 10), (11, 12, 10, 11)])
    )
    stored = frame.reset_index(names="session_date")
    stored.attrs.update(frame.attrs)
    path = tmp_path / "bars.parquet"
    stored.to_parquet(path, index=False)

    loaded = pd.read_parquet(path)

    assert loaded.attrs["provider"] == "YFINANCE"
    assert loaded.attrs["bar_origin"] == "NETWORK_API"
    assert (
        loaded.attrs["adjustment_mode"]
        == "YFINANCE_AUTO_ADJUST_REPAIR"
    )
    assert len(loaded.attrs["content_fingerprint"]) == 64
