from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow.parquet as pq

from stocks.data.phase5_common import utc_now_iso
from stocks.research.phase6 import (
    _inverse_vol_targets,
    _portfolio_result,
    _prepare_common_history,
    load_phase6_dataset,
)
from stocks.universe import broad_universe_symbols


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    net_return: float
    exit_reason: str


def critical_trading_schema() -> dict[str, Any]:
    return {
        "schema": "critical_trading_transcript_backtest_schema_v1",
        "status": "OFFLINE_RESEARCH_ONLY",
        "strategies": {
            "connors_7_day_etf": {
                "entry": "close < prior 7-bar low and close > SMA(200)",
                "exit": "close > prior 7-bar high",
                "stop": "entry - 2 * ATR(20)",
                "execution": "next session open",
            },
            "five_bar_contrarian": {
                "entry": "low < prior 5-bar low",
                "exit": "high > prior 5-bar high",
                "execution": "next session open",
            },
            "five_bar_inverted": {
                "entry": "high > prior 5-bar high",
                "exit": "low < prior 5-bar low",
                "execution": "next session open",
            },
            "structure_atr_pullback": {
                "activation": "high > prior 63-bar high",
                "buy_level": "breakout bar low - 2 * ATR(20)",
                "validity": "10 bars, reset by a new breakout",
                "exit": "close > each of the prior two closes",
                "stop": "entry - 2 * ATR(20) measured on trigger bar",
                "execution": "next session open",
            },
            "monthly_inverse_volatility": {
                "allocation": "normalized inverse rolling daily volatility",
                "rebalance": "monthly",
                "lookback_sensitivity": [20, 63, 126, 252],
            },
            "supply_demand_market_structure": {
                "status": "SPECIFICATION_BLOCKED",
                "reason": "impulse, consolidation, swing validation and zone expiry are discretionary",
            },
            "monthly_top15_momentum": {
                "rank": "252-session close return",
                "filter": "close > SMA(200)",
                "rebalance": "monthly, maximum 15 equal-weight stocks",
            },
            "moving_average_systems": {
                "traditional": "SMA(50) cross SMA(200)",
                "optimized_symmetric": "SMA(70) cross SMA(210)",
                "optimized_asymmetric": "70/210 entry and 110/230 exit",
                "channel": "five complete bars above/below SMA(10) high/low channel",
            },
            "bollinger_breakout": "close above SMA(100)+3 sigma; exit below SMA(100)-1 sigma",
            "twenty_percent_flipper": "close 20% above 50-day low; exit 20% below 50-day high",
            "engulfing_long": "bullish and contrarian bearish-engulfing long variants",
            "spy_tlt_seasonality": "SPY month-end trend gate; TLT Thursday below SMA(5)",
            "triple_screen": "prior completed weekly MACD histogram positive; daily Bear Power negative and rising",
            "rsi2_adx": "RSI(2)<15, optionally ADX(5)>35; exit close above prior high",
        },
        "cost_bps_per_side": 10.0,
        "authority": {"strategy": "NONE", "execution": "NONE"},
        "provider_calls": 0,
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
    }


def run_critical_trading_backtests(
    project_root: Path, *, stock_limit: int | None = None, include_yfinance: bool = False
) -> dict[str, Any]:
    output_dir = project_root / "output" / "research" / "critical_trading"
    output_dir.mkdir(parents=True, exist_ok=True)
    etfs = _load_ibkr_ohlc(project_root / "data" / "bars")
    stocks = _load_stock_ohlc(project_root, stock_limit=stock_limit)

    connors_symbols = [symbol for symbol in ("SPY", "TLT", "GLD", "VNQ") if symbol in etfs]
    connors = _run_universe(etfs, connors_symbols, _connors_signals)
    five_etf = _run_universe(etfs, sorted(etfs), _five_bar_signals)
    inverted_etf = _run_universe(etfs, sorted(etfs), _five_bar_inverted_signals)
    stock_symbols = sorted(stocks)
    five_stock = _run_universe(stocks, stock_symbols, _five_bar_signals)
    inverted_stock = _run_universe(stocks, stock_symbols, _five_bar_inverted_signals)
    structure_stock = _run_universe(stocks, stock_symbols, _structure_signals)
    traditional_ma = _run_universe(stocks, stock_symbols, _ma_crossover_signals(50, 200, 50, 200))
    optimized_ma = _run_universe(stocks, stock_symbols, _ma_crossover_signals(70, 210, 70, 210))
    asymmetric_ma = _run_universe(stocks, stock_symbols, _ma_crossover_signals(70, 210, 110, 230))
    ma_channel = _run_universe(stocks, stock_symbols, _ma_channel_signals)
    bollinger = _run_universe(stocks, stock_symbols, _bollinger_signals)
    flipper = _run_universe(stocks, stock_symbols, _flipper_signals)
    momentum = _monthly_momentum(stocks)
    engulfing_bearish = _run_universe(etfs, ["SPY"] if "SPY" in etfs else [], _bearish_engulfing_long_signals)
    engulfing_bullish = _run_universe(etfs, ["SPY"] if "SPY" in etfs else [], _bullish_engulfing_long_signals)
    seasonality = _seasonality_results(etfs)
    triple_screen = _run_triple_screen_universe(stocks, stock_symbols)
    rsi_symbols = [symbol for symbol in ("SPY", "QQQ", "OEF", "DIA") if symbol in etfs]
    rsi2 = _run_universe(etfs, rsi_symbols, _rsi2_signals(False))
    rsi2_adx = _run_universe(etfs, rsi_symbols, _rsi2_signals(True))
    inverse_vol = _inverse_volatility_sensitivity(project_root)

    report: dict[str, Any] = {
        "schema": "critical_trading_transcript_backtest_results_v1",
        "generated_at": utc_now_iso(),
        "status": "RESEARCH_RESULTS_AVAILABLE",
        "data": {
            "etf_symbol_count": len(etfs),
            "stock_symbol_count": len(stocks),
            "stock_universe": stock_symbols,
            "etf_start": min((str(frame.index.min().date()) for frame in etfs.values()), default=None),
            "etf_end": max((str(frame.index.max().date()) for frame in etfs.values()), default=None),
            "stock_start": min((str(frame.index.min().date()) for frame in stocks.values()), default=None),
            "stock_end": max((str(frame.index.max().date()) for frame in stocks.values()), default=None),
        },
        "results": {
            "connors_7_day_etf": connors,
            "five_bar_contrarian_etf": five_etf,
            "five_bar_inverted_etf": inverted_etf,
            "five_bar_contrarian_stocks": five_stock,
            "five_bar_inverted_stocks": inverted_stock,
            "structure_atr_pullback_stocks_without_vix": structure_stock,
            "monthly_top15_momentum_stocks": momentum,
            "ma_crossover_50_200_stocks": traditional_ma,
            "ma_crossover_70_210_stocks": optimized_ma,
            "ma_entry_70_210_exit_110_230_stocks": asymmetric_ma,
            "ma_channel_10x5_stocks": ma_channel,
            "bollinger_100_3_1_stocks": bollinger,
            "twenty_percent_flipper_stocks": flipper,
            "bearish_engulfing_bought_on_spy": engulfing_bearish,
            "bullish_engulfing_long_on_spy": engulfing_bullish,
            "spy_tlt_seasonality": seasonality,
            "triple_screen_stocks": triple_screen,
            "rsi2_etf_proxy": {**rsi2, "status": "PARTIAL_REPLICATION"},
            "rsi2_adx_etf_proxy": {**rsi2_adx, "status": "PARTIAL_REPLICATION"},
            "monthly_inverse_volatility": inverse_vol,
            "supply_demand_market_structure": {
                "status": "SPECIFICATION_BLOCKED",
                "reason": "The transcript does not define objective impulse, consolidation, swing or zone-expiry thresholds.",
            },
        },
        "yfinance_validation": run_yfinance_validation(project_root) if include_yfinance else {"status": "NOT_REQUESTED"},
        "replication_gaps": {
            "VNQ_missing": "VNQ" not in etfs,
            "VIX_missing": True,
            "exact_connors_replication": len(connors_symbols) == 4,
            "exact_structure_vix_replication": False,
            "stock_universe_point_in_time": False,
            "stock_universe_warning": "Current locally available symbols create survivorship/selection bias.",
            "ibkr_ohlc_price_basis": "raw TRADES; dividends are not embedded in entry/exit OHLC",
            "inverse_vol_price_basis": "EUR total return, split/dividend/FX adjusted",
            "position_sizing": "Trade metrics are pooled; transcript portfolio sizing was not fully specified.",
            "four_index_regime_filter_missing": True,
            "bollinger_flipper_atr_stop_omitted": "The transcript does not state the ATR multiplier.",
            "seasonality_execution": "next-open conservative variant; transcript describes close execution",
            "rsi_etf_universe_missing": [symbol for symbol in ("OEF", "DIA") if symbol not in etfs],
            "entry_setup_studies": "Ten SPY triggers and EMA-stretch material omit a complete exit/risk contract.",
        },
        "authority": {"strategy": "NONE", "execution": "NONE"},
        "provider_calls": 0,
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
    }
    if include_yfinance and report["yfinance_validation"].get("status") == "GO":
        report["cross_provider_comparison"] = _cross_provider_comparison(
            report["results"], report["yfinance_validation"]["results"]
        )
    (output_dir / "schema.json").write_text(json.dumps(critical_trading_schema(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    if include_yfinance:
        (output_dir / "yfinance-validation.json").write_text(
            json.dumps(report["yfinance_validation"], indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (output_dir / "cross-provider-comparison.json").write_text(
            json.dumps(report.get("cross_provider_comparison", {}), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    (output_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report


def collect_yfinance_data(
    project_root: Path,
    *,
    symbols: list[str] | None = None,
    start: str = "2000-01-01",
    end: str | None = None,
    batch_size: int = 40,
) -> dict[str, Any]:
    import yfinance as yf

    requested = sorted(set(symbols or _default_yfinance_symbols(project_root)))
    cache_dir = project_root / "data" / "research" / "critical_trading" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    end_exclusive = end or (date.today() + timedelta(days=1)).isoformat()
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for offset in range(0, len(requested), batch_size):
        batch = requested[offset : offset + batch_size]
        try:
            downloaded = yf.download(
                batch,
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=True,
                actions=False,
                repair=True,
                keepna=False,
                progress=False,
                threads=True,
                group_by="ticker",
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - network/provider behavior
            failures.extend({"symbol": symbol, "reason": type(exc).__name__} for symbol in batch)
            continue
        for symbol in batch:
            try:
                raw = downloaded[symbol] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded
                frame = raw.rename(columns={str(column): str(column).lower().replace(" ", "_") for column in raw.columns})
                frame = frame.rename(columns={"stock_splits": "splits"})
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                frame = _clean_frame(frame)
                if frame.empty:
                    failures.append({"symbol": symbol, "reason": "EMPTY_RESPONSE"})
                    continue
                frame = _attach_yfinance_provenance(frame)
                path = cache_dir / f"{_safe_symbol(symbol)}.parquet"
                stored = frame.reset_index(names="session_date")
                stored.attrs.update(frame.attrs)
                stored.to_parquet(path, index=False)
                summaries.append(
                    {
                        "symbol": symbol,
                        "rows": len(frame),
                        "first": str(frame.index.min().date()),
                        "last": str(frame.index.max().date()),
                        "path": str(path),
                    }
                )
            except (KeyError, ValueError, TypeError) as exc:
                failures.append({"symbol": symbol, "reason": type(exc).__name__})
    manifest = {
        "schema": "critical_trading_yfinance_cache_manifest_v1",
        "status": "GO" if summaries else "NO_DATA",
        "generated_at": utc_now_iso(),
        "source": "yfinance",
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "auto_adjust": True,
        "repair": True,
        "start": start,
        "end_exclusive": end_exclusive,
        "requested_count": len(requested),
        "collected_count": len(summaries),
        "failure_count": len(failures),
        "instruments": summaries,
        "failures": failures,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _attach_yfinance_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    digest = hashlib.sha256()
    digest.update(b"YFINANCE_AUTO_ADJUST_REPAIR_V1")
    digest.update(
        pd.util.hash_pandas_object(work, index=True).values.tobytes()
    )
    work.attrs.update(
        {
            "provider": "YFINANCE",
            "bar_origin": "NETWORK_API",
            "adjustment_mode": "YFINANCE_AUTO_ADJUST_REPAIR",
            "content_fingerprint": digest.hexdigest().upper(),
        }
    )
    return work


def load_yfinance_cache(project_root: Path) -> dict[str, pd.DataFrame]:
    cache_dir = project_root / "data" / "research" / "critical_trading" / "yfinance"
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    for item in manifest.get("instruments", []):
        path = Path(item["path"])
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame["session_date"])
        frames[str(item["symbol"])] = _clean_frame(frame)
    return frames


def run_yfinance_validation(project_root: Path) -> dict[str, Any]:
    frames = load_yfinance_cache(project_root)
    if not frames:
        return {"status": "NO_YFINANCE_CACHE"}
    etf_names = {"SPY", "TLT", "GLD", "VNQ", "QQQ", "DIA", "OEF", "^VIX"}
    etfs = {symbol: frame for symbol, frame in frames.items() if symbol in etf_names}
    stocks = {symbol: frame for symbol, frame in frames.items() if symbol not in etf_names and not symbol.startswith("^")}
    stock_symbols = sorted(stocks)
    rsi_symbols = [symbol for symbol in ("SPY", "QQQ", "OEF", "DIA") if symbol in etfs]
    return {
        "status": "GO",
        "source": "yfinance_auto_adjusted",
        "etf_count": len(etfs),
        "stock_count": len(stocks),
        "results": {
            "connors_7_day_etf": _run_universe(etfs, [s for s in ("SPY", "TLT", "GLD", "VNQ") if s in etfs], _connors_signals),
            "five_bar_contrarian_etf": _run_universe(etfs, sorted(s for s in etfs if not s.startswith("^")), _five_bar_signals),
            "five_bar_contrarian_stocks": _run_universe(stocks, stock_symbols, _five_bar_signals),
            "ma_crossover_50_200_stocks": _run_universe(stocks, stock_symbols, _ma_crossover_signals(50, 200, 50, 200)),
            "ma_crossover_70_210_stocks": _run_universe(stocks, stock_symbols, _ma_crossover_signals(70, 210, 70, 210)),
            "ma_entry_70_210_exit_110_230_stocks": _run_universe(stocks, stock_symbols, _ma_crossover_signals(70, 210, 110, 230)),
            "ma_channel_10x5_stocks": _run_universe(stocks, stock_symbols, _ma_channel_signals),
            "bollinger_100_3_1_stocks": _run_universe(stocks, stock_symbols, _bollinger_signals),
            "twenty_percent_flipper_stocks": _run_universe(stocks, stock_symbols, _flipper_signals),
            "triple_screen_stocks": _run_triple_screen_universe(stocks, stock_symbols),
            "bearish_engulfing_bought_on_spy": _run_universe(etfs, ["SPY"] if "SPY" in etfs else [], _bearish_engulfing_long_signals),
            "rsi2_etf": _run_universe(etfs, rsi_symbols, _rsi2_signals(False)),
            "rsi2_adx_etf": _run_universe(etfs, rsi_symbols, _rsi2_signals(True)),
            "spy_tlt_seasonality": _seasonality_results(etfs),
            "monthly_top15_momentum_stocks": _monthly_momentum(stocks) if stocks else {"status": "NO_STOCK_DATA"},
        },
    }


def run_perfection_pipeline(project_root: Path) -> dict[str, Any]:
    frames = load_yfinance_cache(project_root)
    etf_names = {"SPY", "TLT", "GLD", "VNQ", "QQQ", "DIA", "OEF", "^VIX"}
    stocks = {symbol: frame for symbol, frame in frames.items() if symbol not in etf_names and not symbol.startswith("^")}
    if not stocks:
        return {"status": "NO_YFINANCE_STOCK_DATA"}
    splits = {
        "train": ("2000-01-01", "2012-12-31"),
        "validation": ("2013-01-01", "2018-12-31"),
        "test": ("2019-01-01", "2026-12-31"),
    }
    rsi_grid = _rsi_parameter_grid(stocks, splits)
    selected = max(
        (row for row in rsi_grid if row["validation"]["trade_count"] >= 500),
        key=lambda row: (row["validation"]["trade_profit_factor"] or -1, row["validation"]["mean_trade_return"] or -1),
    )
    selected_exits = {}
    for mode in ("first_up_day", "first_profitable_close", "rsi_above_75"):
        trades = _rsi_trades(stocks, selected["period"], selected["trigger"], mode)
        selected_exits[mode] = _trade_object_summary(
            [trade for trade in trades if trade.entry_date >= splits["test"][0]]
        )
    ma_configs = [
        (50, 200, 50, 200),
        (60, 200, 60, 200),
        (70, 200, 70, 200),
        (70, 210, 70, 210),
        (70, 210, 100, 220),
        (70, 210, 110, 230),
        (80, 220, 120, 240),
    ]
    test_stocks = {symbol: frame.loc[frame.index >= splits["test"][0]] for symbol, frame in stocks.items()}
    ma_plateau = []
    for entry_fast, entry_slow, exit_fast, exit_slow in ma_configs:
        result = _run_universe(
            test_stocks,
            sorted(test_stocks),
            _ma_crossover_signals(entry_fast, entry_slow, exit_fast, exit_slow),
        )
        ma_plateau.append(
            {
                "entry_fast": entry_fast,
                "entry_slow": entry_slow,
                "exit_fast": exit_fast,
                "exit_slow": exit_slow,
                **{
                    key: result[key]
                    for key in (
                        "trade_count",
                        "win_rate",
                        "trade_profit_factor",
                        "mean_trade_return",
                        "average_holding_days",
                        "maximum_holding_days",
                        "end_of_data_forced_exits",
                    )
                },
            }
        )
    seasonality = _seasonality_execution_sensitivity(frames)
    positive_test = [row for row in rsi_grid if (row["test"]["trade_profit_factor"] or 0) > 1]
    report = {
        "schema": "critical_trading_perfection_robustness_v1",
        "status": "ROBUSTNESS_RESULTS_AVAILABLE",
        "generated_at": utc_now_iso(),
        "method": {
            "parameter_selection": "validation PF with minimum 500 trades",
            "test_data_used_for_selection": False,
            "splits": splits,
            "cost_bps_per_side": 10.0,
            "authority": "NONE",
        },
        "rsi2_research": {
            "grid": rsi_grid,
            "selected_on_validation": selected,
            "selected_test_exit_sensitivity": selected_exits,
            "positive_test_config_count": len(positive_test),
            "config_count": len(rsi_grid),
            "plateau_status": "PRESENT" if len(positive_test) >= int(0.7 * len(rsi_grid)) else "NARROW_OR_ABSENT",
        },
        "moving_average_oos_plateau": ma_plateau,
        "seasonality_execution_sensitivity": seasonality,
        "decision": {
            "status": "PROMISING_RESEARCH_CANDIDATE",
            "candidate": "RSI_MEAN_REVERSION_CAUSAL_V1",
            "rules": {
                "entry": f"RSI({selected['period']}) < {selected['trigger']} at close",
                "entry_execution": "next session open",
                "exit": "first close above previous close",
                "exit_execution": "next session open",
                "cost_bps_per_side": 10.0,
            },
            "oos_trade_profit_factor": selected["test"]["trade_profit_factor"],
            "oos_trade_count": selected["test"]["trade_count"],
            "financial_finalist_go": False,
            "strategy_authority": "NONE",
            "blocked_variants": {
                "first_profitable_close": "unbounded holding period and no stop contract",
                "seasonality_close": "same-close signal/execution assumption is not conservatively executable",
                "moving_average_finalist": "survivorship-biased universe and omitted portfolio/regime rules",
            },
        },
        "limitations": [
            "Yahoo stock universe is not point-in-time and remains survivorship biased.",
            "No strategy or broker authority is granted.",
            "First profitable close can create very long holding periods because no stop is defined.",
        ],
        "financial_calls": {"place_order": 0, "cancel_order": 0, "global_cancel": 0},
    }
    output_dir = project_root / "output" / "research" / "critical_trading"
    (output_dir / "perfection-robustness.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    (output_dir / "perfection-report.md").write_text(_perfection_markdown(report), encoding="utf-8")
    return report


def _perfection_markdown(report: dict[str, Any]) -> str:
    selected = report["rsi2_research"]["selected_on_validation"]
    exits = report["rsi2_research"]["selected_test_exit_sensitivity"]
    lines = [
        "# Critical Trading robustness refinement",
        "",
        f"Status: **{report['decision']['status']}**",
        "",
        "## Selected causal candidate",
        "",
        f"- Entry: RSI({selected['period']}) < {selected['trigger']} at close; execute next open.",
        "- Exit: first close above previous close; execute next open.",
        f"- OOS trades: {selected['test']['trade_count']}.",
        f"- OOS win rate: {_pct(selected['test']['win_rate'])}.",
        f"- OOS trade PF: {_num(selected['test']['trade_profit_factor'])}.",
        f"- Parameter plateau: {report['rsi2_research']['plateau_status']}.",
        "",
        "## Exit sensitivity",
        "",
    ]
    for name, result in exits.items():
        lines.append(
            f"- **{name}**: PF {_num(result['trade_profit_factor'])}, average hold "
            f"{_num(result['average_holding_days'])} days, maximum {result['maximum_holding_days']} days, "
            f"forced exits {result['end_of_data_forced_exits']}."
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The candidate remains research-only. Survivorship bias and absent point-in-time membership block financial finalist status.",
            "",
        ]
    )
    return "\n".join(lines)


def _rsi_parameter_grid(
    frames: dict[str, pd.DataFrame], splits: dict[str, tuple[str, str]]
) -> list[dict[str, Any]]:
    rows = []
    for period in (2, 3, 5, 10, 14, 20):
        for trigger in (5, 10, 20, 30, 40):
            trades = _rsi_trades(frames, period, trigger, "first_up_day")
            row: dict[str, Any] = {"period": period, "trigger": trigger}
            for name, (start, end) in splits.items():
                row[name] = _trade_return_summary(
                    [trade.net_return for trade in trades if start <= trade.entry_date <= end]
                )
            rows.append(row)
    return rows


def _rsi_trades(
    frames: dict[str, pd.DataFrame], period: int, trigger: float, exit_mode: str
) -> list[Trade]:
    trades: list[Trade] = []
    cost = 10.0 / 10_000.0
    for symbol, frame in frames.items():
        rsi = _rsi(frame["close"], period).to_numpy()
        opens = frame["open"].to_numpy()
        closes = frame["close"].to_numpy()
        dates = frame.index
        pending_entry = False
        pending_exit = False
        entry = 0.0
        entry_date = ""
        for index in range(len(frame)):
            if pending_entry:
                entry = float(opens[index])
                entry_date = str(dates[index].date())
                pending_entry = False
            if entry > 0 and pending_exit:
                trades.append(_trade(symbol, entry_date, str(dates[index].date()), entry, float(opens[index]), cost, exit_mode.upper()))
                entry = 0.0
                pending_exit = False
            if index >= len(frame) - 1:
                continue
            if entry > 0:
                if exit_mode == "first_up_day":
                    pending_exit = closes[index] > closes[index - 1]
                elif exit_mode == "first_profitable_close":
                    pending_exit = closes[index] > entry
                elif exit_mode == "rsi_above_75":
                    pending_exit = rsi[index] > 75
                else:
                    raise ValueError(f"unsupported RSI exit mode: {exit_mode}")
            elif pd.notna(rsi[index]) and rsi[index] < trigger:
                pending_entry = True
        if entry > 0:
            trades.append(_trade(symbol, entry_date, str(dates[-1].date()), entry, float(closes[-1]), cost, "END_OF_DATA"))
    return trades


def _trade_return_summary(values: list[float]) -> dict[str, Any]:
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    return {
        "trade_count": len(values),
        "win_rate": sum(value > 0 for value in values) / len(values) if values else None,
        "trade_profit_factor": None if negative == 0 else positive / negative,
        "mean_trade_return": sum(values) / len(values) if values else None,
        "median_trade_return": float(pd.Series(values).median()) if values else None,
    }


def _trade_object_summary(trades: list[Trade]) -> dict[str, Any]:
    summary = _trade_return_summary([trade.net_return for trade in trades])
    holding_days = [
        (date.fromisoformat(trade.exit_date) - date.fromisoformat(trade.entry_date)).days
        for trade in trades
    ]
    return {
        **summary,
        "average_holding_days": sum(holding_days) / len(holding_days) if holding_days else None,
        "median_holding_days": float(pd.Series(holding_days).median()) if holding_days else None,
        "maximum_holding_days": max(holding_days) if holding_days else None,
        "end_of_data_forced_exits": sum(trade.exit_reason == "END_OF_DATA" for trade in trades),
    }


def _seasonality_execution_sensitivity(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    next_open = _seasonality_results(frames)
    close_results = {}
    if "SPY" in frames:
        frame = frames["SPY"]
        periods = pd.Series(frame.index.to_period("M"), index=frame.index)
        entry = periods.ne(periods.shift(-1)) & (frame["close"] > frame["close"].rolling(200).mean())
        close_results["SPY"] = _backtest_at_close("SPY", frame, entry, frame["close"] > frame["high"].shift(1))
    if "TLT" in frames:
        frame = frames["TLT"]
        entry = pd.Series((frame.index.dayofweek == 3) & (frame["close"] < frame["close"].rolling(5).mean()), index=frame.index)
        close_results["TLT"] = _backtest_at_close("TLT", frame, entry, frame["close"] > frame["high"].shift(1))
    return {"next_open": next_open, "video_close_assumption": close_results}


def _backtest_at_close(
    symbol: str, frame: pd.DataFrame, entry_signal: pd.Series, exit_signal: pd.Series
) -> dict[str, Any]:
    values = []
    entry = 0.0
    cost = 10.0 / 10_000.0
    for index in range(len(frame)):
        close = float(frame.iloc[index]["close"])
        if entry > 0 and bool(exit_signal.iloc[index]):
            values.append(close * (1 - cost) / (entry * (1 + cost)) - 1)
            entry = 0.0
        elif entry == 0 and bool(entry_signal.iloc[index]):
            entry = close
    if entry > 0:
        close = float(frame.iloc[-1]["close"])
        values.append(close * (1 - cost) / (entry * (1 + cost)) - 1)
    return {"symbol": symbol, **_trade_return_summary(values)}


def _default_yfinance_symbols(project_root: Path) -> list[str]:
    result_path = project_root / "output" / "research" / "critical_trading" / "results.json"
    stocks: list[str] = []
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        stocks = [str(symbol) for symbol in payload.get("data", {}).get("stock_universe", [])]
    core = [
        "ACWI",
        "BIL",
        "DBA",
        "DBB",
        "DBC",
        "DIA",
        "EEM",
        "EWJ",
        "GLD",
        "IEF",
        "IEUR",
        "INDA",
        "IWM",
        "MCHI",
        "OEF",
        "QQQ",
        "SLV",
        "SPY",
        "TLT",
        "VNQ",
        "^VIX",
    ]
    return sorted(
        set(stocks + core) | broad_universe_symbols(project_root)
    )


def _safe_symbol(symbol: str) -> str:
    return symbol.replace("^", "INDEX_").replace("/", "-")


def _cross_provider_comparison(local: dict[str, Any], yahoo: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for name in sorted(set(local) & set(yahoo)):
        left = local[name]
        right = yahoo[name]
        if not isinstance(left, dict) or not isinstance(right, dict) or "trade_count" not in left or "trade_count" not in right:
            continue
        local_pf = left.get("trade_profit_factor")
        yahoo_pf = right.get("trade_profit_factor")
        rows.append(
            {
                "strategy": name,
                "local_trade_count": left["trade_count"],
                "yfinance_trade_count": right["trade_count"],
                "local_win_rate": left.get("win_rate"),
                "yfinance_win_rate": right.get("win_rate"),
                "local_trade_profit_factor": local_pf,
                "yfinance_trade_profit_factor": yahoo_pf,
                "same_pf_direction": None if local_pf is None or yahoo_pf is None else (local_pf > 1) == (yahoo_pf > 1),
            }
        )
    return {
        "schema": "critical_trading_cross_provider_comparison_v1",
        "status": "GO" if rows else "NO_COMPARABLE_RESULTS",
        "comparable_strategy_count": len(rows),
        "same_pf_direction_count": sum(row["same_pf_direction"] is True for row in rows),
        "rows": rows,
    }


def _load_ibkr_ohlc(root: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for path in root.rglob("bars.parquet"):
        frame = pq.ParquetFile(path).read().to_pandas()
        if frame.empty:
            continue
        symbol = str(frame.iloc[0]["symbol"])
        frame.index = pd.to_datetime(frame["session_date"])
        result[symbol] = _clean_frame(frame)
    return result


def _load_stock_ohlc(project_root: Path, *, stock_limit: int | None) -> dict[str, pd.DataFrame]:
    path = project_root / "data" / "research" / "phase11_3" / "private" / "causal_research.sqlite3"
    query = """
        SELECT json_extract(payload_json, '$.symbol') AS provider_symbol,
               json_extract(payload_json, '$.timestamp') AS session_date,
               CAST(json_extract(payload_json, '$.open') AS REAL) AS open,
               CAST(json_extract(payload_json, '$.high') AS REAL) AS high,
               CAST(json_extract(payload_json, '$.low') AS REAL) AS low,
               CAST(json_extract(payload_json, '$.close') AS REAL) AS close,
               CAST(json_extract(payload_json, '$.volume') AS REAL) AS volume
        FROM records WHERE dataset = 'prices'
        ORDER BY provider_symbol, session_date
    """
    with sqlite3.connect(path) as connection:
        frame = pd.read_sql_query(query, connection)
    frame["symbol"] = frame["provider_symbol"].str.replace(r"\.[A-Z]+$", "", regex=True)
    result: dict[str, pd.DataFrame] = {}
    for symbol, group in frame.groupby("symbol", sort=True):
        clean = _clean_frame(group)
        if len(clean) < 1_000 or clean["close"].median() < 5 or clean["volume"].median() < 100_000:
            continue
        result[str(symbol)] = clean
        if stock_limit is not None and len(result) >= stock_limit:
            break
    return result


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    if not isinstance(clean.index, pd.DatetimeIndex):
        clean.index = pd.to_datetime(clean["session_date"])
    clean = clean.sort_index()
    clean = clean[~clean.index.duplicated(keep="last")]
    for column in ("open", "high", "low", "close", "volume"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    return clean.dropna(subset=["open", "high", "low", "close"])


def _atr(frame: pd.DataFrame, period: int = 20) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def _connors_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    prior_low = frame["low"].shift(1).rolling(7).min()
    prior_high = frame["high"].shift(1).rolling(7).max()
    entry = (frame["close"] < prior_low) & (frame["close"] > frame["close"].rolling(200).mean())
    exit_signal = frame["close"] > prior_high
    return entry, exit_signal, 2.0 * _atr(frame)


def _five_bar_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
    return frame["low"] < frame["low"].shift(1).rolling(5).min(), frame["high"] > frame["high"].shift(1).rolling(5).max(), None


def _five_bar_inverted_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
    return frame["high"] > frame["high"].shift(1).rolling(5).max(), frame["low"] < frame["low"].shift(1).rolling(5).min(), None


def _structure_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    atr2 = 2.0 * _atr(frame)
    breakout = frame["high"] > frame["high"].shift(1).rolling(63).max()
    levels = pd.Series(math.nan, index=frame.index)
    current_level: float | None = None
    age = 0
    for index in frame.index:
        if bool(breakout.loc[index]) and pd.notna(atr2.loc[index]):
            current_level = float(frame.loc[index, "low"] - atr2.loc[index])
            age = 0
        elif current_level is not None:
            age += 1
            if age > 10:
                current_level = None
        if current_level is not None:
            levels.loc[index] = current_level
    entry = levels.notna() & (frame["low"] <= levels)
    exit_signal = (frame["close"] > frame["close"].shift(1)) & (frame["close"] > frame["close"].shift(2))
    return entry, exit_signal, atr2


def _ma_crossover_signals(
    entry_fast: int, entry_slow: int, exit_fast: int, exit_slow: int
) -> Callable[[pd.DataFrame], tuple[pd.Series, pd.Series, None]]:
    def signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
        entry_left = frame["close"].rolling(entry_fast).mean()
        entry_right = frame["close"].rolling(entry_slow).mean()
        exit_left = frame["close"].rolling(exit_fast).mean()
        exit_right = frame["close"].rolling(exit_slow).mean()
        entry = (entry_left > entry_right) & (entry_left.shift(1) <= entry_right.shift(1))
        exit_signal = (exit_left < exit_right) & (exit_left.shift(1) >= exit_right.shift(1))
        return entry, exit_signal, None

    return signals


def _ma_channel_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
    upper = frame["high"].rolling(10).mean()
    lower = frame["low"].rolling(10).mean()
    above = frame["low"] > upper
    below = frame["high"] < lower
    entry_state = above.rolling(5).sum() == 5
    exit_state = below.rolling(5).sum() == 5
    prior_entry = entry_state.shift(1).eq(True)
    prior_exit = exit_state.shift(1).eq(True)
    return entry_state & ~prior_entry, exit_state & ~prior_exit, None


def _bollinger_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
    middle = frame["close"].rolling(100).mean()
    deviation = frame["close"].rolling(100).std(ddof=0)
    return frame["close"] > middle + 3 * deviation, frame["close"] < middle - deviation, None


def _flipper_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
    return frame["close"] > 1.2 * frame["low"].rolling(50).min(), frame["close"] < 0.8 * frame["high"].rolling(50).max(), None


def _bearish_engulfing_long_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
    previous_bull = frame["close"].shift(1) > frame["open"].shift(1)
    current_bear = frame["close"] < frame["open"]
    engulf = (frame["open"] > frame["close"].shift(1)) & (frame["close"] < frame["open"].shift(1))
    return previous_bull & current_bear & engulf, frame["close"] > frame["high"].shift(1), None


def _bullish_engulfing_long_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
    previous_bear = frame["close"].shift(1) < frame["open"].shift(1)
    current_bull = frame["close"] > frame["open"]
    engulf = (frame["open"] < frame["close"].shift(1)) & (frame["close"] > frame["open"].shift(1))
    return previous_bear & current_bull & engulf, frame["close"] > frame["high"].shift(1), None


def _rsi2_signals(with_adx: bool) -> Callable[[pd.DataFrame], tuple[pd.Series, pd.Series, None]]:
    def signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
        entry = _rsi(frame["close"], 2) < 15
        if with_adx:
            entry &= _adx(frame, 5) > 35
        return entry, frame["close"] > frame["high"].shift(1), None

    return signals


def _rsi(values: pd.Series, period: int) -> pd.Series:
    delta = values.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    relative_strength = gain / loss.replace(0, math.nan)
    return 100 - 100 / (1 + relative_strength)


def _adx(frame: pd.DataFrame, period: int) -> pd.Series:
    up = frame["high"].diff()
    down = -frame["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    true_range = pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - frame["close"].shift()).abs(), (frame["low"] - frame["close"].shift()).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, math.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _triple_screen_entry(frame: pd.DataFrame) -> pd.Series:
    weekly_close = frame["close"].resample("W-FRI").last()
    macd = weekly_close.ewm(span=12, adjust=False).mean() - weekly_close.ewm(span=26, adjust=False).mean()
    histogram = macd - macd.ewm(span=9, adjust=False).mean()
    prior_completed_week = histogram.shift(1).reindex(frame.index, method="ffill")
    bear_power = frame["low"] - frame["close"].ewm(span=13, adjust=False).mean()
    return (prior_completed_week > 0) & (bear_power < 0) & (bear_power > bear_power.shift(1)) & (frame["high"] > frame["high"].shift(1))


def _run_triple_screen_universe(frames: dict[str, pd.DataFrame], symbols: list[str]) -> dict[str, Any]:
    trades: list[Trade] = []
    cost = 10.0 / 10_000.0
    for symbol in symbols:
        frame = frames[symbol]
        entry_signal = _triple_screen_entry(frame)
        pending = False
        entry = 0.0
        entry_date = ""
        high_water = 0.0
        for position, (timestamp, row) in enumerate(frame.iterrows()):
            if pending:
                entry = float(row["open"])
                entry_date = str(timestamp.date())
                high_water = entry
                pending = False
            if entry > 0:
                high_water = max(high_water, float(row["high"]))
                trailing_stop = high_water * 0.8
                target = entry * 1.3
                stop_hit = float(row["low"]) <= trailing_stop
                target_hit = float(row["high"]) >= target
                if stop_hit or target_hit:
                    if stop_hit:
                        exit_price = min(float(row["open"]), trailing_stop) if float(row["open"]) <= trailing_stop else trailing_stop
                        reason = "TRAILING_STOP"
                    else:
                        exit_price = max(float(row["open"]), target) if float(row["open"]) >= target else target
                        reason = "PROFIT_TARGET"
                    trades.append(_trade(symbol, entry_date, str(timestamp.date()), entry, exit_price, cost, reason))
                    entry = 0.0
            if position < len(frame) - 1 and entry == 0 and bool(entry_signal.iloc[position]):
                pending = True
        if entry > 0:
            trades.append(_trade(symbol, entry_date, str(frame.index[-1].date()), entry, float(frame.iloc[-1]["close"]), cost, "END_OF_DATA"))
    result = _summarize_trades(trades, symbols)
    result["status"] = "PARTIAL_REPLICATION"
    result["execution_note"] = "next-open alternative described in transcript; conservative stop priority on ambiguous daily bars"
    return result


def _seasonality_results(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    results: dict[str, Any] = {"status": "PARTIAL_REPLICATION", "execution": "next_open"}
    if "SPY" in frames:
        def spy_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
            next_month = frame.index.to_series().shift(-1).dt.month
            month_end = frame.index.to_series().dt.month != next_month
            entry = month_end.to_numpy() & (frame["close"] > frame["close"].rolling(200).mean())
            return pd.Series(entry, index=frame.index), frame["close"] > frame["high"].shift(1), None
        results["SPY"] = _run_universe(frames, ["SPY"], spy_signals)
    if "TLT" in frames:
        def tlt_signals(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, None]:
            entry = (frame.index.dayofweek == 3) & (frame["close"] < frame["close"].rolling(5).mean())
            return pd.Series(entry, index=frame.index), frame["close"] > frame["high"].shift(1), None
        results["TLT"] = _run_universe(frames, ["TLT"], tlt_signals)
    return results


def _monthly_momentum(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    closes = pd.concat({symbol: frame["close"] for symbol, frame in frames.items()}, axis=1).sort_index()
    returns = closes.pct_change(fill_method=None)
    momentum = closes / closes.shift(252) - 1.0
    trend = closes > closes.rolling(200).mean()
    month_periods = pd.Series(closes.index.to_period("M"), index=closes.index)
    monthly = month_periods.ne(month_periods.shift(1))
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    current = pd.Series(0.0, index=closes.columns)
    turnover = 0.0
    for index, day in enumerate(closes.index):
        if index > 0 and bool(monthly.iloc[index]):
            scores = momentum.iloc[index - 1].where(trend.iloc[index - 1]).dropna().sort_values(ascending=False)
            selected = list(scores.head(15).index)
            target = pd.Series(0.0, index=closes.columns)
            if selected:
                target.loc[selected] = 1.0 / len(selected)
            turnover += 0.5 * float((target - current).abs().sum())
            current = target
        weights.iloc[index] = current
    portfolio = (weights.shift(1).fillna(0.0) * returns.fillna(0.0)).sum(axis=1)
    costs = pd.Series(0.0, index=portfolio.index)
    changes = weights.diff().abs().sum(axis=1) * 0.5
    costs += changes * 10.0 / 10_000.0
    net = portfolio - costs
    nav = (1.0 + net).cumprod()
    years = (net.index[-1] - net.index[0]).days / 365.25
    positive = float(net[net > 0].sum())
    negative = abs(float(net[net < 0].sum()))
    peak = nav.cummax()
    return {
        "status": "PARTIAL_REPLICATION",
        "reason": "Risk rules omitted in transcript and stock universe is not point-in-time.",
        "start": str(net.index[0].date()),
        "end": str(net.index[-1].date()),
        "CAGR": float(nav.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
        "Sharpe": float(net.mean() / net.std(ddof=0) * math.sqrt(252)) if net.std(ddof=0) > 0 else None,
        "maximum_drawdown": float((nav / peak - 1).min()),
        "period_profit_factor": None if negative == 0 else positive / negative,
        "turnover": turnover,
    }


def _backtest_trades(
    symbol: str,
    frame: pd.DataFrame,
    signal_builder: Callable[[pd.DataFrame], tuple[pd.Series, pd.Series, pd.Series | None]],
    *,
    cost_bps: float = 10.0,
) -> list[Trade]:
    entry_signal, exit_signal, stop_distance = signal_builder(frame)
    trades: list[Trade] = []
    pending_entry = False
    pending_exit = False
    entry_price = 0.0
    entry_date = ""
    stop: float | None = None
    pending_stop_distance: float | None = None
    cost = cost_bps / 10_000.0
    for position, (timestamp, row) in enumerate(frame.iterrows()):
        if pending_entry:
            entry_price = float(row["open"])
            entry_date = str(timestamp.date())
            stop = None if pending_stop_distance is None else entry_price - pending_stop_distance
            pending_entry = False
        if entry_price > 0 and stop is not None and float(row["low"]) <= stop:
            exit_price = min(float(row["open"]), stop) if float(row["open"]) <= stop else stop
            trades.append(_trade(symbol, entry_date, str(timestamp.date()), entry_price, exit_price, cost, "STOP"))
            entry_price = 0.0
            stop = None
            pending_exit = False
        elif entry_price > 0 and pending_exit:
            trades.append(_trade(symbol, entry_date, str(timestamp.date()), entry_price, float(row["open"]), cost, "SIGNAL"))
            entry_price = 0.0
            stop = None
            pending_exit = False
        if position == len(frame) - 1:
            continue
        if entry_price > 0 and bool(exit_signal.iloc[position]):
            pending_exit = True
        elif entry_price == 0 and bool(entry_signal.iloc[position]):
            pending_entry = True
            value = None if stop_distance is None else stop_distance.iloc[position]
            pending_stop_distance = None if value is None or pd.isna(value) else float(value)
    if entry_price > 0:
        last_timestamp = frame.index[-1]
        trades.append(_trade(symbol, entry_date, str(last_timestamp.date()), entry_price, float(frame.iloc[-1]["close"]), cost, "END_OF_DATA"))
    return trades


def _trade(symbol: str, entry_date: str, exit_date: str, entry: float, exit_price: float, cost: float, reason: str) -> Trade:
    net_return = exit_price * (1.0 - cost) / (entry * (1.0 + cost)) - 1.0
    return Trade(symbol, entry_date, exit_date, entry, exit_price, net_return, reason)


def _run_universe(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    signal_builder: Callable[[pd.DataFrame], tuple[pd.Series, pd.Series, pd.Series | None]],
) -> dict[str, Any]:
    trades = [trade for symbol in symbols for trade in _backtest_trades(symbol, frames[symbol], signal_builder)]
    return _summarize_trades(trades, symbols)


def _summarize_trades(trades: list[Trade], symbols: list[str]) -> dict[str, Any]:
    values = [trade.net_return for trade in trades]
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    per_symbol: dict[str, Any] = {}
    for symbol in symbols:
        subset = [trade.net_return for trade in trades if trade.symbol == symbol]
        pos = sum(value for value in subset if value > 0)
        neg = abs(sum(value for value in subset if value < 0))
        per_symbol[symbol] = {
            "trades": len(subset),
            "win_rate": sum(value > 0 for value in subset) / len(subset) if subset else None,
            "trade_profit_factor": None if neg == 0 else pos / neg,
            "mean_trade_return": sum(subset) / len(subset) if subset else None,
        }
    holding_days = [
        (date.fromisoformat(trade.exit_date) - date.fromisoformat(trade.entry_date)).days
        for trade in trades
    ]
    return {
        "status": "GO" if trades else "NO_TRADES",
        "symbols": symbols,
        "trade_count": len(trades),
        "winning_trades": sum(value > 0 for value in values),
        "losing_trades": sum(value < 0 for value in values),
        "win_rate": sum(value > 0 for value in values) / len(values) if values else None,
        "trade_profit_factor": None if negative == 0 else positive / negative,
        "mean_trade_return": sum(values) / len(values) if values else None,
        "median_trade_return": float(pd.Series(values).median()) if values else None,
        "average_holding_days": sum(holding_days) / len(holding_days) if holding_days else None,
        "maximum_holding_days": max(holding_days) if holding_days else None,
        "end_of_data_forced_exits": sum(trade.exit_reason == "END_OF_DATA" for trade in trades),
        "cost_bps_per_side": 10.0,
        "per_symbol": per_symbol,
    }


def _inverse_volatility_sensitivity(project_root: Path) -> dict[str, Any]:
    prepared = _prepare_common_history(load_phase6_dataset(project_root))
    symbol_to_key = {item["symbol"]: key for key, item in prepared["metadata"].items()}
    transcript_symbols = [symbol for symbol in ("SPY", "TLT", "GLD", "VNQ") if symbol in symbol_to_key]
    transcript_prepared = _subset_prepared(prepared, [symbol_to_key[symbol] for symbol in transcript_symbols])
    return {
        "status": "GO",
        "transcript_proxy": {
            "symbols": transcript_symbols,
            "missing_symbols": [symbol for symbol in ("SPY", "TLT", "GLD", "VNQ") if symbol not in symbol_to_key],
            "results": _inverse_vol_results(transcript_prepared, "transcript_proxy"),
        },
        "broad_local_extension": {
            "symbols": [item["symbol"] for item in prepared["metadata"].values()],
            "results": _inverse_vol_results(prepared, "broad_local"),
        },
    }


def _subset_prepared(prepared: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {
        **prepared,
        "returns": {key: prepared["returns"][key] for key in keys},
        "prices": {key: prepared["prices"][key] for key in keys},
        "metadata": {key: prepared["metadata"][key] for key in keys},
        "cash_key": prepared["cash_key"] if prepared["cash_key"] in keys else None,
    }


def _inverse_vol_results(prepared: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    results = []
    for lookback in (20, 63, 126, 252):
        weights = _inverse_vol_targets(prepared, lookback=lookback, rebalance="monthly")
        result = _portfolio_result(
            f"inverse_volatility_monthly_{lookback}_{scope}",
            prepared,
            weights,
            cost_bps=10.0,
            parameters={"lookback": lookback, "scope": scope},
        )
        calendar_returns = _calendar_portfolio_returns(prepared, weights, cost_bps=10.0)
        compact = {key: result[key] for key in ("name", "parameters", "CAGR", "Sharpe", "maximum_drawdown", "trade_profit_factor", "period_profit_factor", "turnover", "transaction_costs")}
        compact["return_2020_1x"] = calendar_returns["1x"].get("2020")
        compact["return_2020_2x"] = calendar_returns["2x"].get("2020")
        results.append(compact)
    return results


def _calendar_portfolio_returns(
    prepared: dict[str, Any], weights: list[dict[str, float]], *, cost_bps: float
) -> dict[str, dict[str, float]]:
    annual: dict[str, dict[str, float]] = {"1x": {}, "2x": {}}
    previous: dict[str, float] = {}
    for index, day in enumerate(prepared["dates"]):
        target = weights[index]
        turnover = 0.5 * sum(abs(target.get(key, 0.0) - previous.get(key, 0.0)) for key in set(target) | set(previous))
        gross = sum(weight * prepared["returns"][key][index] for key, weight in previous.items() if key in prepared["returns"])
        year = day[:4]
        for label, leverage in (("1x", 1.0), ("2x", 2.0)):
            net = leverage * gross - leverage * turnover * cost_bps / 10_000.0
            annual[label][year] = (1.0 + annual[label].get(year, 0.0)) * (1.0 + net) - 1.0
        previous = target
    return annual


def _markdown_report(report: dict[str, Any]) -> str:
    lines = ["# Critical Trading transcript backtests", "", "Generated: " + report["generated_at"], "", "## Results", ""]
    for name, result in report["results"].items():
        if "trade_count" in result:
            lines.append(
                f"- **{name}**: {result['trade_count']} trades, win rate {_pct(result['win_rate'])}, "
                f"trade PF {_num(result['trade_profit_factor'])}, mean trade {_pct(result['mean_trade_return'])}."
            )
        elif name == "monthly_inverse_volatility":
            for row in result["transcript_proxy"]["results"]:
                lines.append(
                    f"- **inverse vol transcript proxy {row['parameters']['lookback']}d**: CAGR {_pct(row['CAGR'])}, "
                    f"Sharpe {_num(row['Sharpe'])}, max DD {_pct(row['maximum_drawdown'])}, "
                    f"2020 1x {_pct(row['return_2020_1x'])}, 2020 2x {_pct(row['return_2020_2x'])}."
                )
        else:
            lines.append(f"- **{name}**: {result['status']} - {result.get('reason', '')}")
    lines.extend(["", "## Replication limits", ""])
    for key, value in report["replication_gaps"].items():
        lines.append(f"- **{key}**: {value}")
    comparison = report.get("cross_provider_comparison")
    if comparison:
        lines.extend(["", "## Cross-provider comparison", ""])
        for row in comparison["rows"]:
            lines.append(
                f"- **{row['strategy']}**: local PF {_num(row['local_trade_profit_factor'])}; "
                f"yfinance PF {_num(row['yfinance_trade_profit_factor'])}; "
                f"same direction {row['same_pf_direction']}."
            )
    lines.extend(["", "These are offline research results, not trading authority or investment advice.", ""])
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
