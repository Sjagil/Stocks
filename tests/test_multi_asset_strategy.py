from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import main

from stocks.data.bars import (
    BarCacheLayout,
    BarDataSource,
    BarDataType,
    BarInterval,
    HistoricalBar,
    write_bar_cache_file,
)
from stocks.domain.assets import IbkrSecurityType
from stocks.portfolio.sleeves import SleeveName
from stocks.strategies.multi_asset import (
    MultiAssetBacktestConfig,
    StrategyAssetSeries,
    run_multi_asset_rotation_backtest,
)


def test_multi_asset_rotation_can_report_positive_pf_on_causal_fixture() -> None:
    timestamps = _timestamps(8)
    winner = StrategyAssetSeries(
        con_id=1001,
        security_type=IbkrSecurityType.STK,
        sleeve=SleeveName.ETF_CORE_ROTATION,
        timestamps=timestamps,
        closes=(100.0, 101.0, 102.0, 103.0, 102.0, 104.0, 106.0, 108.0),
    )
    loser = StrategyAssetSeries(
        con_id=1002,
        security_type=IbkrSecurityType.STK,
        sleeve=SleeveName.ETF_CORE_ROTATION,
        timestamps=timestamps,
        closes=(100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0),
    )

    report = run_multi_asset_rotation_backtest(
        [winner, loser],
        config=MultiAssetBacktestConfig(lookback_bars=2, cost_bps=0.0, max_asset_weight=0.5),
    )

    assert report["status"] == "GO"
    assert report["instrument_count"] == 2
    assert report["bar_count"] == 16
    assert report["rebalance_count"] == report["period_count"]
    assert report["profit_factor"] > 1.0
    assert report["profit_factor_status"] == "POSITIVE"
    assert report["total_return"] > 0
    assert report["net_return"] == report["total_return"]
    assert report["CAGR"] is not None
    assert report["annualized_volatility"] > 0
    assert report["Sharpe"] is not None
    assert report["Sortino"] is not None
    assert report["maximum_drawdown"] == report["max_drawdown"]
    assert report["Calmar"] is not None
    assert report["average_win"] > 0
    assert report["average_loss"] < 0
    assert report["total_costs"] == 0.0
    assert report["cash_average"] >= 0.0
    assert report["cash_maximum"] >= report["cash_average"]
    assert report["sleeve_exposure"][SleeveName.ETF_CORE_ROTATION.value] > 0.0
    assert report["region_exposure"]["GLOBAL_OR_UNCLASSIFIED"] > 0.0
    assert report["negative_periods"] >= 1
    assert report["financial_calls"]["place_order"] == 0


def test_multi_asset_rotation_reports_no_data_without_bars() -> None:
    report = run_multi_asset_rotation_backtest([])

    assert report["status"] == "NO_DATA"
    assert report["period_count"] == 0
    assert report["financial_calls"]["global_cancel"] == 0


def test_strategy_multi_asset_schema_cli_is_offline(capsys) -> None:
    exit_code = main.main(["strategy", "multi-asset", "schema"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["schema"] == "multi_asset_strategy_schema_v1"
    assert payload["execution"]["live_orders_enabled"] is False
    assert payload["execution"]["provider_calls_enabled"] is False
    assert payload["financial_calls"]["place_order"] == 0


def test_strategy_multi_asset_status_cli_reports_empty_cache_as_no_data(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(["--env-file", str(tmp_path / ".env.ibkr"), "strategy", "multi-asset", "status"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "NO_DATA"
    assert payload["data"]["local_bar_cache_status"] == "GO"
    assert payload["data"]["local_bar_cache_row_count"] == 0
    assert payload["financial_calls"]["place_order"] == 0


def test_strategy_multi_asset_backtest_cli_reports_no_data_for_empty_cache(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "strategy",
            "multi-asset",
            "backtest",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "NO_DATA"
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert payload["cache_validation"]["status"] == "GO"
    assert payload["financial_calls"]["place_order"] == 0


def test_strategy_multi_asset_backtest_cli_uses_local_bar_cache_only(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    layout = BarCacheLayout.from_project_root(tmp_path)
    timestamps = _timestamps(8)
    _write_daily_stock_bars(layout, con_id=1001, timestamps=timestamps, closes=[100, 101, 102, 103, 102, 104, 106, 108])
    _write_daily_stock_bars(layout, con_id=1002, timestamps=timestamps, closes=[100, 99, 98, 97, 96, 95, 94, 93])

    exit_code = main.main(
        [
            "--env-file",
            str(tmp_path / ".env.ibkr"),
            "strategy",
            "multi-asset",
            "backtest",
            "--source",
            "LOCAL",
            "--lookback-bars",
            "2",
            "--cost-bps",
            "0",
            "--max-asset-weight",
            "0.5",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["status"] == "GO"
    assert payload["profit_factor"] > 1.0
    assert payload["profit_factor_status"] == "POSITIVE"
    assert payload["bar_filter"]["source"] == "LOCAL"
    assert payload["instrument_count"] == 2
    assert payload["maximum_drawdown"] == payload["max_drawdown"]
    assert payload["cache_validation"]["row_count"] == 16
    assert payload["phase1"]["status"] == "PHASE1_NOT_FROZEN"
    assert payload["financial_calls"]["cancel_order"] == 0


def _timestamps(count: int) -> tuple[datetime, ...]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return tuple(start + timedelta(days=index) for index in range(count))


def _write_daily_stock_bars(
    layout: BarCacheLayout,
    *,
    con_id: int,
    timestamps: tuple[datetime, ...],
    closes: list[int],
) -> None:
    bars = [
        HistoricalBar(
            con_id=con_id,
            security_type=IbkrSecurityType.STK,
            interval=BarInterval.ONE_DAY,
            data_type=BarDataType.TRADES,
            source=BarDataSource.IBKR,
            timestamp_utc=timestamp,
            open=Decimal(str(close)),
            high=Decimal(str(close + 1)),
            low=Decimal(str(close - 1)),
            close=Decimal(str(close)),
            volume=1_000_000,
            available_at=timestamp,
        )
        for timestamp, close in zip(timestamps, closes, strict=True)
    ]
    path = layout.bars_path(
        security_type=IbkrSecurityType.STK,
        con_id=con_id,
        interval=BarInterval.ONE_DAY,
        data_type=BarDataType.TRADES,
        source=BarDataSource.IBKR,
    )
    write_bar_cache_file(path, bars)
