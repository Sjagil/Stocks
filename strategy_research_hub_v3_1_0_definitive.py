#!/usr/bin/env python3
"""
Strategy Research Hub
=====================

A standalone launcher for an existing Stocks repository. It leaves main.py and
strategy_combo_research_lab.py unchanged.

It can:
1. Import and delegate any existing main.py command.
2. Import strategy_combo_research_lab.py and temporarily register the
   RSI(2) + ADX(5) + Bollinger + rolling-VWAP strategy in its strategy registry.
3. Import and run the standalone rsi2_adx5_vwap.py backtester and sweep tool.
4. Inspect component paths and import readiness with a doctor command.

Expected placement
------------------
Put this file in the same project root as:
    main.py
    strategy_combo_research_lab.py
    rsi2_adx5_vwap.py

The launcher also recognizes browser/download duplicate names ending in "(1)".
No source file is modified on disk. The combo-lab registry patch exists only in
this Python process.

Examples
--------
    python strategy_research_hub.py doctor
    python strategy_research_hub.py main doctor
    python strategy_research_hub.py main ibkr status
    python strategy_research_hub.py combo list
    python strategy_research_hub.py combo run --preset smoke --max-symbols 50
    python strategy_research_hub.py combo run --preset long \
        --include-strategies rsi2_adx5_vwap --combo-sizes 2,3,4
    python strategy_research_hub.py rsi --csv data/SPY_1d.csv \
        --capital 2000 --vwap-kind session --vwap-timezone UTC

Global path overrides may be supplied before the command:
    --project-root PATH
    --main-file PATH
    --combo-file PATH
    --rsi-file PATH
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROGRAM_VERSION = "3.1.0"
INJECTED_STRATEGY_NAME = "rsi2_adx5_vwap"


class HubError(RuntimeError):
    """User-facing launcher failure."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_component(
    project_root: Path,
    explicit: str | None,
    environment_name: str,
    candidates: Sequence[str],
    label: str,
) -> Path:
    selected = explicit or os.getenv(environment_name)
    if selected:
        path = Path(selected).expanduser()
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise HubError(f"{label} not found: {path}")
        return path

    for relative in candidates:
        candidate = (project_root / relative).resolve()
        if candidate.is_file():
            return candidate

    rendered = "\n  - ".join(str(project_root / item) for item in candidates)
    raise HubError(f"{label} not found. Checked:\n  - {rendered}")


def _component_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.project_root).expanduser().resolve()
    if not root.is_dir():
        raise HubError(f"Project root does not exist: {root}")
    return {
        "project_root": root,
        "main": _resolve_component(
            root,
            args.main_file,
            "STOCKS_MAIN_FILE",
            ("main.py",),
            "main.py",
        ),
        "combo": _resolve_component(
            root,
            args.combo_file,
            "STOCKS_COMBO_FILE",
            (
                "strategy_combo_research_lab.py",
                "strategy_combo_research_lab(1).py",
            ),
            "strategy_combo_research_lab.py",
        ),
        "rsi": _resolve_component(
            root,
            args.rsi_file,
            "STOCKS_RSI_FILE",
            (
                "rsi2_adx5_vwap.py",
                "rsi2_adx5_vwap(1).py",
                "src/stocks/strategies/rsi2_adx5_vwap.py",
            ),
            "rsi2_adx5_vwap.py",
        ),
    }


@contextlib.contextmanager
def _project_import_path(project_root: Path) -> Iterator[None]:
    additions = [str(project_root), str(project_root / "src")]
    original = list(sys.path)
    try:
        for item in reversed(additions):
            if item not in sys.path:
                sys.path.insert(0, item)
        yield
    finally:
        sys.path[:] = original


@contextlib.contextmanager
def _temporary_argv(program: str, argv: Sequence[str]) -> Iterator[None]:
    original = list(sys.argv)
    sys.argv = [program, *argv]
    try:
        yield
    finally:
        sys.argv = original


def _load_module(path: Path, module_name: str, project_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise HubError(f"Could not build import specification for {path}")

    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        with _project_import_path(project_root):
            spec.loader.exec_module(module)
    except SystemExit as exc:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        detail = exc.code if exc.code not in (None, 0) else "module exited during import"
        raise HubError(f"Import of {path.name} failed: {detail}") from exc
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _call_module_main(module: ModuleType, path: Path, argv: Sequence[str]) -> int:
    main_function = getattr(module, "main", None)
    if not callable(main_function):
        raise HubError(f"{path.name} has no callable main() function")

    try:
        signature = inspect.signature(main_function)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if positional:
            result = main_function(list(argv))
        else:
            with _temporary_argv(str(path), argv):
                result = main_function()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1

    return int(result or 0)


def _finite_array(values: Any) -> Any:
    import numpy as np

    return np.isfinite(np.asarray(values, dtype=float))


def _build_combo_adapter(combo: ModuleType, rsi_module: ModuleType):
    """Create a daily-data StrategySpec builder compatible with the combo lab."""

    import numpy as np
    import pandas as pd

    def builder(frame: pd.DataFrame, cache: Any, params: Any) -> Any:
        n = len(frame)
        if n < 5:
            return combo.TradeBatch.empty()

        close = cache.arr("close")
        open_ = cache.arr("open")
        high = cache.arr("high")
        volume = pd.to_numeric(frame.get("volume"), errors="coerce").fillna(0.0).to_numpy(dtype=float)

        rsi_period = int(params["rsi_period"])
        adx_period = int(params["adx_period"])
        trend_ma = int(params["trend_ma"])
        bb_period = int(params["bb_period"])
        bb_std_mult = float(params["bb_std"])
        atr_period = int(params["atr_period"])
        exit_ema_period = int(params["exit_ema"])
        vwap_window = int(params["vwap_window"])

        # Use the combo lab's cached indicators where possible. The formulas are
        # Wilder-style and match the imported standalone strategy's intent.
        rsi_values = cache.rsi(rsi_period)
        adx_values, plus_di, minus_di = cache.adx(adx_period)
        trend_values = cache.sma("close", trend_ma)
        bb_mid = cache.sma("close", bb_period)
        bb_std_values = cache.rolling_std("close", bb_period)
        bb_lower = bb_mid - bb_std_mult * bb_std_values
        atr_values = cache.atr(atr_period)
        exit_ema = cache.ema("close", exit_ema_period)

        typical = (cache.arr("high") + cache.arr("low") + close) / 3.0
        pv = pd.Series(typical * volume)
        vol = pd.Series(volume)
        rolling_pv = pv.rolling(vwap_window, min_periods=vwap_window).sum().to_numpy(dtype=float)
        rolling_volume = vol.rolling(vwap_window, min_periods=vwap_window).sum().to_numpy(dtype=float)
        vwap = np.divide(
            rolling_pv,
            rolling_volume,
            out=np.full(n, np.nan, dtype=float),
            where=rolling_volume > 0.0,
        )

        entry_threshold = float(params["entry_threshold"])
        adx_threshold = float(params["adx_threshold"])
        vwap_premium = float(params["vwap_max_premium"])
        require_positive_dmi = bool(params["require_positive_dmi"])

        finite = (
            _finite_array(close)
            & _finite_array(rsi_values)
            & _finite_array(adx_values)
            & _finite_array(plus_di)
            & _finite_array(minus_di)
            & _finite_array(trend_values)
            & _finite_array(bb_lower)
            & _finite_array(atr_values)
            & _finite_array(vwap)
        )
        direction_ok = plus_di > minus_di if require_positive_dmi else np.ones(n, dtype=bool)
        setup_condition = (
            finite
            & (close > trend_values)
            & direction_ok
            & (adx_values >= adx_threshold)
            & (rsi_values <= entry_threshold)
            & (close <= bb_lower)
            & (close <= vwap * (1.0 + vwap_premium))
        )

        entry_signal = np.zeros(n, dtype=bool)
        score = np.zeros(n, dtype=float)
        confirmation_bars = int(params["confirmation_bars"])
        require_bullish = bool(params["require_bullish_confirmation"])
        vwap_mode = str(params["vwap_mode"])

        setup_high = float("nan")
        setup_score = 0.0
        expiry_index = -1
        active_setup = False

        for i in range(n):
            if active_setup:
                if i > expiry_index:
                    active_setup = False
                else:
                    regime_ok = (
                        math.isfinite(close[i])
                        and math.isfinite(trend_values[i])
                        and close[i] > trend_values[i]
                        and (
                            not require_positive_dmi
                            or (
                                math.isfinite(plus_di[i])
                                and math.isfinite(minus_di[i])
                                and plus_di[i] > minus_di[i]
                            )
                        )
                    )
                    bullish_ok = (close[i] > open_[i]) if require_bullish else True
                    break_setup_high = math.isfinite(setup_high) and close[i] > setup_high
                    reclaim_vwap = False
                    if i > 0 and all(
                        math.isfinite(value)
                        for value in (close[i], vwap[i], close[i - 1], vwap[i - 1])
                    ):
                        reclaim_vwap = close[i] > vwap[i] and close[i - 1] <= vwap[i - 1]

                    if vwap_mode in {"discount", "off"}:
                        confirmed = break_setup_high
                    elif vwap_mode == "reclaim":
                        confirmed = reclaim_vwap
                    elif vwap_mode == "either":
                        confirmed = break_setup_high or reclaim_vwap
                    else:
                        raise ValueError(f"Unknown vwap_mode: {vwap_mode}")

                    if regime_ok and bullish_ok and confirmed:
                        entry_signal[i] = True
                        score[i] = setup_score
                        active_setup = False

            # Never confirm on the same close that creates a setup. A newer
            # setup replaces an older unconfirmed setup, matching the standalone logic.
            if not entry_signal[i] and setup_condition[i]:
                setup_high = float(high[i])
                expiry_index = i + confirmation_bars
                rsi_severity = max(entry_threshold - float(rsi_values[i]), 0.0) / max(entry_threshold, 1.0)
                adx_strength = max(float(adx_values[i]) - adx_threshold, 0.0) / 100.0
                vwap_discount = max(float(vwap[i]) - float(close[i]), 0.0) / max(float(atr_values[i]), 1e-12)
                setup_score = rsi_severity + adx_strength + vwap_discount
                active_setup = True

        previous_close = np.roll(close, 1)
        previous_ema = np.roll(exit_ema, 1)
        previous_close[0] = np.nan
        previous_ema[0] = np.nan
        ema_recovery = (
            _finite_array(close)
            & _finite_array(exit_ema)
            & _finite_array(previous_close)
            & _finite_array(previous_ema)
            & (previous_close < previous_ema)
            & (close >= exit_ema)
        )
        rsi_recovery = _finite_array(rsi_values) & (rsi_values >= float(params["exit_threshold"]))
        exit_signal = ema_recovery | rsi_recovery

        stop_distance = np.where(
            _finite_array(atr_values),
            atr_values * float(params["stop_atr"]),
            np.nan,
        )
        target_multiple = float(params["target_atr"])
        target_distance = None
        if target_multiple > 0.0:
            target_distance = np.where(
                _finite_array(atr_values),
                atr_values * target_multiple,
                np.nan,
            )

        return combo.trades_from_signals(
            frame=frame,
            entry_signal=entry_signal,
            exit_signal=exit_signal,
            score=score,
            stop_distance=stop_distance,
            target_distance=target_distance,
            max_hold=int(params["max_hold"]),
            force_close_end=True,
        )

    return builder


def _install_combo_strategy(combo: ModuleType, rsi_module: ModuleType) -> None:
    if getattr(combo, "_rsi2_adx5_vwap_hub_installed", False):
        return

    required = ("StrategySpec", "strategy_registry", "TradeBatch", "trades_from_signals")
    missing = [name for name in required if not hasattr(combo, name)]
    if missing:
        raise HubError(
            "The combo lab API is incompatible with this launcher. Missing: "
            + ", ".join(missing)
        )
    if not hasattr(rsi_module, "StrategyConfig") or not hasattr(rsi_module, "backtest"):
        raise HubError("The RSI module is incompatible: StrategyConfig/backtest missing")

    original_registry = combo.strategy_registry
    adapter_builder = _build_combo_adapter(combo, rsi_module)

    default_params = {
        "rsi_period": 2,
        "entry_threshold": 10,
        "exit_threshold": 60,
        "adx_period": 5,
        "adx_threshold": 25,
        "trend_ma": 200,
        "bb_period": 20,
        "bb_std": 2.0,
        "atr_period": 14,
        "stop_atr": 2.0,
        "target_atr": 1.5,
        "exit_ema": 5,
        "vwap_window": 20,
        "vwap_mode": "either",
        "vwap_max_premium": 0.0,
        "confirmation_bars": 3,
        "require_positive_dmi": False,
        "require_bullish_confirmation": True,
        "max_hold": 10,
    }
    choices = {
        # Put the most important dimensions first because the combo lab caps
        # one-factor variants according to --max-variants-per-strategy.
        "entry_threshold": [5, 10, 15, 20],
        "adx_threshold": [20, 25, 30, 35, 40],
        "stop_atr": [1.5, 2.0, 2.5, 3.0],
        "target_atr": [0.0, 1.0, 1.5, 2.0, 2.5],
        "vwap_mode": ["discount", "reclaim", "either"],
        "confirmation_bars": [1, 2, 3, 5],
        "max_hold": [5, 10, 20, 40],
        "trend_ma": [100, 150, 200, 250],
        "bb_std": [1.5, 2.0, 2.5],
        "vwap_window": [10, 20, 30, 50],
        "exit_threshold": [50, 60, 70, 80],
        "exit_ema": [3, 5, 8, 10],
        "require_positive_dmi": [False, True],
        "require_bullish_confirmation": [True, False],
        "rsi_period": [2],
        "adx_period": [5],
        "bb_period": [20],
        "atr_period": [14],
        "vwap_max_premium": [0.0],
    }

    def patched_registry() -> list[Any]:
        specs = list(original_registry())
        if any(getattr(spec, "name", None) == INJECTED_STRATEGY_NAME for spec in specs):
            return specs
        specs.append(
            combo.StrategySpec(
                name=INJECTED_STRATEGY_NAME,
                family="trend_filtered_mean_reversion",
                horizon="short",
                description=(
                    "RSI(2) oversold pullback above a long trend filter with ADX(5), "
                    "lower Bollinger Band, rolling daily VWAP, delayed confirmation, "
                    "ATR stop/target and EMA/RSI recovery exits."
                ),
                default_params=default_params,
                choices=choices,
                builder=adapter_builder,
                policy_status="LONG_ONLY_GO",
                notes=(
                    "Injected in memory by strategy_research_hub.py. The combo lab uses "
                    "daily data, so VWAP is rolling rather than session-reset VWAP."
                ),
            )
        )
        return specs

    combo.strategy_registry = patched_registry
    combo._rsi2_adx5_vwap_hub_installed = True
    combo._rsi2_adx5_vwap_original_registry = original_registry


def _load_all(paths: dict[str, Path]) -> tuple[ModuleType, ModuleType, ModuleType]:
    root = paths["project_root"]
    main_module = _load_module(paths["main"], "_stocks_framework_main_hub", root)
    combo_module = _load_module(paths["combo"], "_strategy_combo_lab_hub", root)
    rsi_module = _load_module(paths["rsi"], "_rsi2_adx5_vwap_hub", root)
    return main_module, combo_module, rsi_module


def _doctor(paths: dict[str, Path]) -> int:
    report: dict[str, Any] = {
        "schema": "strategy_research_hub_doctor_v1",
        "version": PROGRAM_VERSION,
        "status": "GO",
        "project_root": str(paths["project_root"]),
        "components": {},
        "features": {
            "missing_csv_auto_download": True,
            "current_or_csv_gex_snapshot": True,
            "gamma_flip_scenario_estimate": True,
            "orderflow_footprint": True,
            "stacked_imbalance_detection": True,
            "delta_and_absorption_features": True,
            "institutional_signal_evaluation": True,
            "institutional_pit_overlay_backtest": True,
        },
        "safety": {
            "source_files_modified": False,
            "combo_patch_scope": "current_process_only",
            "broker_order_methods_added": False,
            "orders_enabled": False,
        },
    }

    modules: dict[str, ModuleType] = {}
    names = {
        "main": "_stocks_framework_main_hub_doctor",
        "combo": "_strategy_combo_lab_hub_doctor",
        "rsi": "_rsi2_adx5_vwap_hub_doctor",
    }
    for key in ("main", "combo", "rsi"):
        path = paths[key]
        item: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256(path),
        }
        try:
            modules[key] = _load_module(path, names[key], paths["project_root"])
            item["import"] = "GO"
            item["main_callable"] = callable(getattr(modules[key], "main", None))
        except BaseException as exc:  # doctor must report all components
            item["import"] = "NO_GO"
            item["error"] = f"{type(exc).__name__}: {exc}"
            report["status"] = "NO_GO"
        report["components"][key] = item

    if "combo" in modules and "rsi" in modules:
        try:
            _install_combo_strategy(modules["combo"], modules["rsi"])
            names_in_registry = [
                getattr(spec, "name", "") for spec in modules["combo"].strategy_registry()
            ]
            injected = INJECTED_STRATEGY_NAME in names_in_registry
            report["combo_injection"] = {
                "status": "GO" if injected else "NO_GO",
                "strategy": INJECTED_STRATEGY_NAME,
                "registry_count": len(names_in_registry),
                "source_files_modified": False,
            }
            if not injected:
                report["status"] = "NO_GO"
        except BaseException as exc:
            report["combo_injection"] = {
                "status": "NO_GO",
                "strategy": INJECTED_STRATEGY_NAME,
                "error": f"{type(exc).__name__}: {exc}",
            }
            report["status"] = "NO_GO"

    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["status"] == "GO" else 2



# ---------------------------------------------------------------------------
# Standalone data, GEX and orderflow research features
# ---------------------------------------------------------------------------


def _safe_name(value: str) -> str:
    cleaned = []
    for char in str(value):
        if char.isalnum() or char in {"-", "_", "."}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "artifact"


def _json_safe_value(value: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        np = None  # type: ignore[assignment]
        pd = None  # type: ignore[assignment]

    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if np is not None and isinstance(value, np.integer):
        return int(value)
    if np is not None and isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _print_payload(payload: Mapping[str, Any]) -> None:
    print(json.dumps(_json_safe_value(dict(payload)), indent=2, ensure_ascii=False, default=str))


def _require_analytics() -> tuple[Any, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise HubError(
            "Missing analytics dependency. Install with: "
            "python -m pip install numpy pandas"
        ) from exc
    return np, pd


def _require_yfinance() -> Any:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise HubError(
            "yfinance is required for downloads and current option snapshots. "
            "Install with: python -m pip install yfinance"
        ) from exc
    return yf


def _resolve_output_path(project_root: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _normalize_yfinance_download(raw: Any) -> Any:
    _, pd = _require_analytics()
    if raw is None or raw.empty:
        raise HubError("The data provider returned no rows")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = {str(item).lower() for item in frame.columns.get_level_values(0)}
        if {"open", "high", "low", "close", "volume"}.issubset(level0):
            frame.columns = frame.columns.get_level_values(0)
        else:
            frame.columns = frame.columns.get_level_values(-1)
    frame = frame.reset_index()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    time_column = next(
        (column for column in ("timestamp", "datetime", "date", "time") if column in frame.columns),
        None,
    )
    if time_column is None:
        raise HubError(f"Downloaded frame has no timestamp column: {list(frame.columns)}")
    frame = frame.rename(columns={time_column: "timestamp"})
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise HubError(f"Downloaded frame is missing columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[required].dropna(subset=required).sort_values("timestamp")
    frame = frame.drop_duplicates("timestamp", keep="last")
    if frame.empty:
        raise HubError("No valid OHLCV rows remained after normalization")
    return frame


def _collect_market_data(
    project_root: Path,
    *,
    ticker: str,
    output: Path,
    period: str,
    interval: str,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    yf = _require_yfinance()
    kwargs: dict[str, Any] = {
        "tickers": ticker,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
        "threads": False,
    }
    if start or end:
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period
    raw = yf.download(**kwargs)
    frame = _normalize_yfinance_download(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return {
        "schema": "strategy_research_hub_market_data_collection_v1",
        "status": "GO",
        "provider": "yfinance",
        "ticker": ticker,
        "period": period if not (start or end) else None,
        "start": start,
        "end": end,
        "interval": interval,
        "rows": int(len(frame)),
        "first_timestamp": frame["timestamp"].iloc[0],
        "last_timestamp": frame["timestamp"].iloc[-1],
        "output": str(output),
        "authority": "research_data_only",
    }


def _data_command(project_root: Path, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="strategy_research_hub.py data")
    sub = parser.add_subparsers(dest="data_command", required=True)
    collect = sub.add_parser("collect", help="Download OHLCV data to a normalized CSV file.")
    collect.add_argument("--ticker", required=True)
    collect.add_argument("--period", default="10y")
    collect.add_argument("--interval", default="1d")
    collect.add_argument("--start")
    collect.add_argument("--end")
    collect.add_argument("--output")
    args = parser.parse_args(list(argv))
    normalized_interval = str(args.interval).strip().lower()
    if normalized_interval not in {"1h", "1d", "1wk", "1mo"}:
        raise HubError(
            f"FORBIDDEN_OR_UNSUPPORTED_SWING_TIMEFRAME:{normalized_interval}"
        )

    output = args.output or f"data/{_safe_name(args.ticker)}_{_safe_name(args.interval)}.csv"
    report = _collect_market_data(
        project_root,
        ticker=args.ticker,
        output=_resolve_output_path(project_root, output),
        period=args.period,
        interval=args.interval,
        start=args.start,
        end=args.end,
    )
    _print_payload(report)
    return 0


def _infer_download_from_csv(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    lower = stem.lower()
    interval = "1d"
    period = "10y"
    suffixes = [
        ("_daily", "1d", "10y"),
        ("_1d", "1d", "10y"),
        ("_1h", "1h", "2y"),
        ("_60m", "60m", "2y"),
    ]
    if any(lower.endswith(suffix) for suffix in ("_30m", "_15m", "_5m", "_2m", "_1min")):
        raise HubError("FORBIDDEN_SWING_TIMEFRAME_IN_FILENAME")
    ticker_text = stem
    for suffix, detected_interval, detected_period in suffixes:
        if lower.endswith(suffix):
            ticker_text = stem[: -len(suffix)]
            interval = detected_interval
            period = detected_period
            break
    ticker = ticker_text.replace("_", "-").strip("-")
    if not ticker:
        raise HubError(f"Could not infer a ticker from missing CSV filename: {path.name}")
    return ticker, interval, period


def _prepare_rsi_arguments(project_root: Path, argv: Sequence[str]) -> list[str]:
    delegated = list(argv)
    no_auto = False
    while "--no-auto-download" in delegated:
        delegated.remove("--no-auto-download")
        no_auto = True
    if "--csv" not in delegated:
        return delegated
    index = delegated.index("--csv")
    if index + 1 >= len(delegated):
        raise HubError("--csv requires a path")
    raw = delegated[index + 1]
    csv_path = _resolve_output_path(project_root, raw)
    delegated[index + 1] = str(csv_path)
    if csv_path.is_file():
        return delegated
    if no_auto:
        raise HubError(
            f"CSV does not exist: {csv_path}. Download it with the data collect command "
            "or remove --no-auto-download."
        )
    ticker, interval, period = _infer_download_from_csv(csv_path)
    print(
        f"[data] Missing CSV detected. Downloading {ticker} interval={interval} "
        f"period={period} to {csv_path}",
        file=sys.stderr,
    )
    report = _collect_market_data(
        project_root,
        ticker=ticker,
        output=csv_path,
        period=period,
        interval=interval,
    )
    print(
        f"[data] Wrote {report['rows']} rows. Continuing with the RSI backtest.",
        file=sys.stderr,
    )
    return delegated


def _normal_pdf(value: Any) -> Any:
    np, _ = _require_analytics()
    return np.exp(-0.5 * np.asarray(value, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


def _normalize_option_chain(frame: Any, as_of: Any) -> Any:
    np, pd = _require_analytics()
    out = frame.copy()
    out.columns = [str(column).strip().lower().replace(" ", "_") for column in out.columns]
    aliases = {
        "expiration": ("expiration", "expiry", "expiration_date"),
        "option_type": ("option_type", "type", "right", "put_call"),
        "strike": ("strike", "strike_price"),
        "open_interest": ("open_interest", "openinterest", "oi"),
        "implied_volatility": ("implied_volatility", "impliedvolatility", "iv"),
        "contract_multiplier": ("contract_multiplier", "multiplier", "contract_size"),
    }
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        found = next((candidate for candidate in candidates if candidate in out.columns), None)
        if found:
            rename[found] = canonical
    out = out.rename(columns=rename)
    required = ["expiration", "option_type", "strike", "open_interest", "implied_volatility"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise HubError(f"Option chain is missing required columns: {missing}")
    if "contract_multiplier" not in out.columns:
        out["contract_multiplier"] = 100.0
    out["expiration"] = pd.to_datetime(out["expiration"], utc=True, errors="coerce")
    out["option_type"] = out["option_type"].astype(str).str.lower().str.strip()
    out["option_type"] = out["option_type"].replace({"c": "call", "p": "put", "calls": "call", "puts": "put"})
    for column in ("strike", "open_interest", "implied_volatility", "contract_multiplier"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    as_of_ts = pd.Timestamp(as_of)
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    else:
        as_of_ts = as_of_ts.tz_convert("UTC")
    out["time_to_expiry_years"] = (
        (out["expiration"] + pd.Timedelta(hours=21) - as_of_ts).dt.total_seconds()
        / (365.25 * 86400.0)
    )
    out = out[
        out["option_type"].isin(["call", "put"])
        & (out["strike"] > 0)
        & (out["open_interest"] > 0)
        & (out["implied_volatility"] > 0.001)
        & (out["implied_volatility"] < 5.0)
        & (out["time_to_expiry_years"] > 0)
        & (out["contract_multiplier"] > 0)
    ].copy()
    if out.empty:
        raise HubError("No valid option rows remained after validation")
    return out


def _calculate_gex_profile(
    chain: Any,
    *,
    spot: float,
    as_of: Any,
    risk_free_rate: float,
    dividend_yield: float,
) -> tuple[dict[str, Any], Any, Any]:
    np, pd = _require_analytics()
    if spot <= 0:
        raise HubError("Spot price must be positive")
    frame = _normalize_option_chain(chain, as_of)
    strike = frame["strike"].to_numpy(dtype=float)
    sigma = frame["implied_volatility"].to_numpy(dtype=float)
    years = frame["time_to_expiry_years"].to_numpy(dtype=float)
    oi = frame["open_interest"].to_numpy(dtype=float)
    multiplier = frame["contract_multiplier"].to_numpy(dtype=float)
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * sigma**2) * years
    ) / (sigma * np.sqrt(years))
    gamma = np.exp(-dividend_yield * years) * _normal_pdf(d1) / (
        spot * sigma * np.sqrt(years)
    )
    sign = np.where(frame["option_type"].to_numpy() == "call", 1.0, -1.0)
    # Dollar gamma for an approximately 1 percent move in the underlying.
    gex = sign * oi * multiplier * gamma * spot**2 * 0.01
    frame["gamma"] = gamma
    frame["signed_gex_1pct"] = gex
    frame["call_gex_1pct"] = np.where(sign > 0, gex, 0.0)
    frame["put_gex_1pct"] = np.where(sign < 0, gex, 0.0)

    profile = (
        frame.groupby("strike", as_index=False)
        .agg(
            signed_gex_1pct=("signed_gex_1pct", "sum"),
            call_gex_1pct=("call_gex_1pct", "sum"),
            put_gex_1pct=("put_gex_1pct", "sum"),
            open_interest=("open_interest", "sum"),
        )
        .sort_values("strike")
    )
    call_wall = float(profile.loc[profile["call_gex_1pct"].idxmax(), "strike"])
    put_wall = float(profile.loc[profile["put_gex_1pct"].idxmin(), "strike"])
    net_gex = float(frame["signed_gex_1pct"].sum())

    scenario_spots = np.linspace(spot * 0.80, spot * 1.20, 161)
    scenario_values: list[float] = []
    for scenario_spot in scenario_spots:
        scenario_d1 = (
            np.log(scenario_spot / strike)
            + (risk_free_rate - dividend_yield + 0.5 * sigma**2) * years
        ) / (sigma * np.sqrt(years))
        scenario_gamma = np.exp(-dividend_yield * years) * _normal_pdf(scenario_d1) / (
            scenario_spot * sigma * np.sqrt(years)
        )
        scenario_values.append(
            float((sign * oi * multiplier * scenario_gamma * scenario_spot**2 * 0.01).sum())
        )
    scenario = pd.DataFrame({"spot": scenario_spots, "net_gex_1pct": scenario_values})
    gamma_flip = None
    signs = np.sign(scenario["net_gex_1pct"].to_numpy(dtype=float))
    changes = np.flatnonzero(signs[:-1] * signs[1:] <= 0)
    if len(changes):
        candidates: list[float] = []
        for index in changes:
            x1 = float(scenario_spots[index])
            x2 = float(scenario_spots[index + 1])
            y1 = float(scenario_values[index])
            y2 = float(scenario_values[index + 1])
            if abs(y2 - y1) <= 1e-12:
                candidates.append((x1 + x2) / 2.0)
            else:
                candidates.append(x1 - y1 * (x2 - x1) / (y2 - y1))
        gamma_flip = min(candidates, key=lambda value: abs(value - spot))

    summary = {
        "schema": "gex_snapshot_v2",
        "status": "GO",
        "as_of": pd.Timestamp(as_of),
        "spot": spot,
        "net_gex_1pct": net_gex,
        "regime_proxy": "POSITIVE_GEX" if net_gex > 0 else "NEGATIVE_GEX" if net_gex < 0 else "NEUTRAL_GEX",
        "call_wall": call_wall,
        "put_wall": put_wall,
        "gamma_flip": gamma_flip,
        "expiration_count": int(frame["expiration"].dt.date.nunique()),
        "option_rows": int(len(frame)),
        "assumptions": {
            "dealer_sign_proxy": "calls positive, puts negative",
            "gex_unit": "estimated dollar gamma for a 1 percent spot move",
            "historical_warning": "A current chain snapshot cannot be used as historical point-in-time GEX.",
        },
    }
    return summary, profile, scenario


def _download_option_chain(
    ticker_symbol: str,
    expiration_mode: str,
    max_expirations: int,
) -> tuple[Any, float, Any]:
    _, pd = _require_analytics()
    yf = _require_yfinance()
    ticker = yf.Ticker(ticker_symbol)
    history = ticker.history(period="5d", interval="1d", auto_adjust=False)
    if history.empty:
        raise HubError(f"No spot history returned for {ticker_symbol}")
    spot = float(history["Close"].dropna().iloc[-1])
    options = list(ticker.options or [])
    if not options:
        raise HubError(f"No listed option expirations returned for {ticker_symbol}")
    if expiration_mode not in {"nearest", "all"}:
        if expiration_mode not in options:
            raise HubError(
                f"Expiration {expiration_mode} is unavailable. First values: {options[:10]}"
            )
        selected = [expiration_mode]
    elif expiration_mode == "nearest":
        selected = options[:1]
    else:
        selected = options[: max(1, max_expirations)]
    rows: list[Any] = []
    for expiration in selected:
        chain = ticker.option_chain(expiration)
        for option_type, source in (("call", chain.calls), ("put", chain.puts)):
            if source is None or source.empty:
                continue
            part = source.copy()
            part["expiration"] = expiration
            part["option_type"] = option_type
            part["contract_multiplier"] = 100.0
            rows.append(part)
    if not rows:
        raise HubError("The selected option chains contained no rows")
    return pd.concat(rows, ignore_index=True), spot, pd.Timestamp.now(tz="UTC")


def _gex_command(project_root: Path, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="strategy_research_hub.py gex")
    sub = parser.add_subparsers(dest="gex_command", required=True)
    snapshot = sub.add_parser("snapshot", help="Calculate a current or CSV-backed GEX proxy.")
    source = snapshot.add_mutually_exclusive_group(required=True)
    source.add_argument("--ticker")
    source.add_argument("--chain-csv")
    snapshot.add_argument("--spot", type=float)
    snapshot.add_argument("--as-of")
    snapshot.add_argument("--expiration", default="all")
    snapshot.add_argument("--max-expirations", type=int, default=4)
    snapshot.add_argument("--risk-free-rate", type=float, default=0.04)
    snapshot.add_argument("--dividend-yield", type=float, default=0.0)
    snapshot.add_argument("--output-dir", default="output/gex")
    args = parser.parse_args(list(argv))

    _, pd = _require_analytics()
    if args.ticker:
        chain, spot, as_of = _download_option_chain(
            args.ticker,
            args.expiration,
            args.max_expirations,
        )
        source_name = args.ticker
    else:
        chain_path = _resolve_output_path(project_root, args.chain_csv)
        if not chain_path.is_file():
            raise HubError(f"Option chain CSV does not exist: {chain_path}")
        chain = pd.read_csv(chain_path)
        if args.spot is None:
            raise HubError("--spot is required with --chain-csv")
        spot = float(args.spot)
        as_of = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now(tz="UTC")
        source_name = chain_path.stem
    summary, profile, scenario = _calculate_gex_profile(
        chain,
        spot=spot,
        as_of=as_of,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
    )
    summary["source"] = args.ticker or str(_resolve_output_path(project_root, args.chain_csv))
    output_dir = _resolve_output_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_name(source_name)
    profile_path = output_dir / f"{prefix}_gex_profile.csv"
    scenario_path = output_dir / f"{prefix}_gex_scenario.csv"
    summary_path = output_dir / f"{prefix}_gex_summary.json"
    profile.to_csv(profile_path, index=False)
    scenario.to_csv(scenario_path, index=False)
    summary["artifacts"] = {
        "summary": str(summary_path),
        "profile": str(profile_path),
        "scenario": str(scenario_path),
    }
    summary_path.write_text(
        json.dumps(_json_safe_value(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_payload(summary)
    return 0


def _normalize_trade_ticks(frame: Any) -> tuple[Any, dict[str, Any]]:
    np, pd = _require_analytics()
    out = frame.copy()
    out.columns = [str(column).strip().lower().replace(" ", "_") for column in out.columns]
    aliases = {
        "timestamp": ("timestamp", "datetime", "date", "time"),
        "price": ("price", "trade_price", "last"),
        "size": ("size", "quantity", "qty", "volume", "trade_size"),
        "side": ("side", "aggressor_side", "taker_side", "direction"),
        "bid": ("bid", "best_bid", "bid_price"),
        "ask": ("ask", "best_ask", "ask_price"),
    }
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        found = next((candidate for candidate in candidates if candidate in out.columns), None)
        if found:
            rename[found] = canonical
    out = out.rename(columns=rename)
    missing = [column for column in ("timestamp", "price", "size") if column not in out.columns]
    if missing:
        raise HubError(f"Tick CSV is missing columns: {missing}")
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["size"] = pd.to_numeric(out["size"], errors="coerce")
    if "bid" in out.columns:
        out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    if "ask" in out.columns:
        out["ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out = out.dropna(subset=["timestamp", "price", "size"])
    out = out[(out["price"] > 0) & (out["size"] > 0)].sort_values("timestamp")
    if out.empty:
        raise HubError("No valid trade ticks remained")

    classified = np.zeros(len(out), dtype=int)
    method = np.full(len(out), "unknown", dtype=object)
    if "side" in out.columns:
        raw_side = out["side"].astype(str).str.lower().str.strip()
        buys = raw_side.isin(["buy", "b", "ask", "buyer", "1", "+1"])
        sells = raw_side.isin(["sell", "s", "bid", "seller", "-1"])
        classified[buys.to_numpy()] = 1
        classified[sells.to_numpy()] = -1
        method[(buys | sells).to_numpy()] = "provided_side"
    unresolved = classified == 0
    if unresolved.any() and "bid" in out.columns and "ask" in out.columns:
        price = out["price"].to_numpy(dtype=float)
        bid = out["bid"].to_numpy(dtype=float)
        ask = out["ask"].to_numpy(dtype=float)
        quote_buy = unresolved & np.isfinite(ask) & (price >= ask)
        quote_sell = unresolved & np.isfinite(bid) & (price <= bid)
        classified[quote_buy] = 1
        classified[quote_sell] = -1
        method[quote_buy | quote_sell] = "quote_test"
    unresolved = classified == 0
    if unresolved.any():
        prices = out["price"].to_numpy(dtype=float)
        changes = np.sign(np.diff(prices, prepend=prices[0]))
        last_nonzero = 1
        for index in range(len(changes)):
            if changes[index] > 0:
                last_nonzero = 1
            elif changes[index] < 0:
                last_nonzero = -1
            if unresolved[index]:
                classified[index] = last_nonzero
                method[index] = "tick_rule"
    out["aggressor"] = classified
    out["classification_method"] = method
    counts = out["classification_method"].value_counts().to_dict()
    metadata = {
        "rows": int(len(out)),
        "classification_counts": {str(key): int(value) for key, value in counts.items()},
        "low_confidence_share": float((out["classification_method"] == "tick_rule").mean()),
    }
    return out, metadata


def _max_consecutive(values: Iterable[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _build_orderflow(
    ticks: Any,
    *,
    bar_size: str,
    tick_size: float,
    imbalance_ratio: float,
    min_denominator: float,
    absorption_share: float,
) -> tuple[Any, Any, dict[str, Any]]:
    np, pd = _require_analytics()
    if tick_size <= 0:
        raise HubError("tick_size must be positive")
    work = ticks.copy()
    work["bar_time"] = work["timestamp"].dt.floor(bar_size)
    work["price_level"] = np.round(work["price"] / tick_size) * tick_size
    work["buy_volume"] = np.where(work["aggressor"] > 0, work["size"], 0.0)
    work["sell_volume"] = np.where(work["aggressor"] < 0, work["size"], 0.0)
    levels = (
        work.groupby(["bar_time", "price_level"], as_index=False)
        .agg(
            buy_volume=("buy_volume", "sum"),
            sell_volume=("sell_volume", "sum"),
            trade_count=("size", "size"),
        )
        .sort_values(["bar_time", "price_level"])
    )
    levels["delta"] = levels["buy_volume"] - levels["sell_volume"]
    level_parts: list[Any] = []
    bar_rows: list[dict[str, Any]] = []
    cumulative_delta = 0.0
    previous_delta = 0.0
    for bar_time, group in levels.groupby("bar_time", sort=True):
        group = group.sort_values("price_level").reset_index(drop=True).copy()
        buy = group["buy_volume"].to_numpy(dtype=float)
        sell = group["sell_volume"].to_numpy(dtype=float)
        buy_imbalance = np.zeros(len(group), dtype=bool)
        sell_imbalance = np.zeros(len(group), dtype=bool)
        if len(group) > 1:
            buy_imbalance[1:] = buy[1:] / np.maximum(sell[:-1], min_denominator) >= imbalance_ratio
            sell_imbalance[:-1] = sell[:-1] / np.maximum(buy[1:], min_denominator) >= imbalance_ratio
        group["buy_imbalance"] = buy_imbalance
        group["sell_imbalance"] = sell_imbalance
        level_parts.append(group)

        raw_bar = work[work["bar_time"] == bar_time].sort_values("timestamp")
        open_price = float(raw_bar["price"].iloc[0])
        high_price = float(raw_bar["price"].max())
        low_price = float(raw_bar["price"].min())
        close_price = float(raw_bar["price"].iloc[-1])
        buy_total = float(group["buy_volume"].sum())
        sell_total = float(group["sell_volume"].sum())
        total = buy_total + sell_total
        delta = buy_total - sell_total
        cumulative_delta += delta
        price_range = max(high_price - low_price, tick_size)
        bottom_cut = low_price + 0.25 * price_range
        top_cut = high_price - 0.25 * price_range
        bottom_sell = float(group.loc[group["price_level"] <= bottom_cut, "sell_volume"].sum())
        top_buy = float(group.loc[group["price_level"] >= top_cut, "buy_volume"].sum())
        buy_absorption = bool(total > 0 and bottom_sell / total >= absorption_share and close_price >= low_price + 0.50 * price_range)
        sell_absorption = bool(total > 0 and top_buy / total >= absorption_share and close_price <= high_price - 0.50 * price_range)
        bar_rows.append(
            {
                "timestamp": bar_time,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "buy_volume": buy_total,
                "sell_volume": sell_total,
                "delta": delta,
                "delta_ratio": delta / total if total > 0 else 0.0,
                "cumulative_delta": cumulative_delta,
                "delta_flip_up": previous_delta <= 0 < delta,
                "delta_flip_down": previous_delta >= 0 > delta,
                "max_stacked_buy": _max_consecutive(buy_imbalance.tolist()),
                "max_stacked_sell": _max_consecutive(sell_imbalance.tolist()),
                "buy_absorption": buy_absorption,
                "sell_absorption": sell_absorption,
                "trade_count": int(len(raw_bar)),
            }
        )
        previous_delta = delta
    level_frame = pd.concat(level_parts, ignore_index=True) if level_parts else levels
    bar_frame = pd.DataFrame(bar_rows)
    if bar_frame.empty:
        raise HubError("Orderflow aggregation produced no bars")
    summary = {
        "schema": "orderflow_footprint_summary_v2",
        "status": "GO",
        "bar_size": bar_size,
        "tick_size": tick_size,
        "imbalance_ratio": imbalance_ratio,
        "bars": int(len(bar_frame)),
        "levels": int(len(level_frame)),
        "latest": bar_frame.iloc[-1].to_dict(),
        "authority": "research_orderflow_proxy_only",
    }
    return level_frame, bar_frame, summary


def _orderflow_command(project_root: Path, argv: Sequence[str]) -> int:
    raise HubError("TICK_DATA_AND_ORDERFLOW_PATH_FORBIDDEN_FOR_SWING_RESEARCH")
    # Retained below as unreachable legacy parsing code for forensic reference.
    parser = argparse.ArgumentParser(prog="strategy_research_hub.py orderflow")
    sub = parser.add_subparsers(dest="orderflow_command", required=True)
    footprint = sub.add_parser("footprint", help="Build footprint and delta features from trade ticks.")
    footprint.add_argument("--trades-csv", required=True)
    footprint.add_argument("--bar-size", default="5min")
    footprint.add_argument("--tick-size", type=float, required=True)
    footprint.add_argument("--imbalance-ratio", type=float, default=3.0)
    footprint.add_argument("--min-denominator", type=float, default=1.0)
    footprint.add_argument("--absorption-share", type=float, default=0.30)
    footprint.add_argument("--output-dir", default="output/orderflow")
    args = parser.parse_args(list(argv))

    _, pd = _require_analytics()
    source = _resolve_output_path(project_root, args.trades_csv)
    if not source.is_file():
        raise HubError(f"Trade tick CSV does not exist: {source}")
    raw = pd.read_csv(source)
    ticks, classification = _normalize_trade_ticks(raw)
    levels, bars, summary = _build_orderflow(
        ticks,
        bar_size=args.bar_size,
        tick_size=args.tick_size,
        imbalance_ratio=args.imbalance_ratio,
        min_denominator=args.min_denominator,
        absorption_share=args.absorption_share,
    )
    summary["classification"] = classification
    output_dir = _resolve_output_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_name(source.stem)
    levels_path = output_dir / f"{prefix}_footprint_levels.csv"
    bars_path = output_dir / f"{prefix}_orderflow_bars.csv"
    summary_path = output_dir / f"{prefix}_orderflow_summary.json"
    levels.to_csv(levels_path, index=False)
    bars.to_csv(bars_path, index=False)
    summary["artifacts"] = {
        "levels": str(levels_path),
        "bars": str(bars_path),
        "summary": str(summary_path),
    }
    summary_path.write_text(
        json.dumps(_json_safe_value(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_payload(summary)
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise HubError(f"JSON file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HubError(f"Expected a JSON object in {path}")
    return payload


def _coerce_bool_series(series: Any) -> Any:
    _, pd = _require_analytics()
    if str(series.dtype) == "bool":
        return series.fillna(False)
    lowered = series.astype(str).str.lower().str.strip()
    return lowered.isin(["true", "1", "yes", "y", "go", "buy"])


def _load_overlay(project_root: Path, raw_path: str, candle_index: Any, max_age: str) -> Any:
    _, pd = _require_analytics()
    path = _resolve_output_path(project_root, raw_path)
    if not path.is_file():
        raise HubError(f"Overlay CSV does not exist: {path}")
    overlay = pd.read_csv(path)
    overlay.columns = [str(column).strip().lower().replace(" ", "_") for column in overlay.columns]
    time_column = next((column for column in ("timestamp", "datetime", "date", "time") if column in overlay.columns), None)
    if time_column is None:
        raise HubError("Overlay CSV needs timestamp/date/datetime/time")
    overlay[time_column] = pd.to_datetime(overlay[time_column], utc=True, errors="coerce")
    overlay = overlay.dropna(subset=[time_column]).sort_values(time_column)
    numeric_columns = [
        "net_gex", "net_gex_1pct", "call_wall", "put_wall", "gamma_flip",
        "delta", "delta_ratio", "max_stacked_buy", "max_stacked_sell",
    ]
    for column in numeric_columns:
        if column in overlay.columns:
            overlay[column] = pd.to_numeric(overlay[column], errors="coerce")
    for column in ("delta_flip_up", "delta_flip_down", "buy_absorption", "sell_absorption", "orderflow_buy_trigger"):
        if column in overlay.columns:
            overlay[column] = _coerce_bool_series(overlay[column])
    base = pd.DataFrame({"timestamp": candle_index}).sort_values("timestamp")
    merged = pd.merge_asof(
        base,
        overlay.rename(columns={time_column: "overlay_timestamp"}).sort_values("overlay_timestamp"),
        left_on="timestamp",
        right_on="overlay_timestamp",
        direction="backward",
        tolerance=pd.Timedelta(max_age),
    ).set_index("timestamp")
    if "net_gex" not in merged.columns and "net_gex_1pct" in merged.columns:
        merged["net_gex"] = merged["net_gex_1pct"]
    return merged


def _latest_orderflow_row(project_root: Path, raw_path: str | None, at: Any) -> dict[str, Any] | None:
    if not raw_path:
        return None
    _, pd = _require_analytics()
    path = _resolve_output_path(project_root, raw_path)
    if not path.is_file():
        raise HubError(f"Orderflow bars CSV does not exist: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    time_column = next((column for column in ("timestamp", "datetime", "date", "time") if column in frame.columns), None)
    if time_column is None:
        raise HubError("Orderflow bars need a timestamp column")
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    at_ts = pd.Timestamp(at)
    if at_ts.tzinfo is None:
        at_ts = at_ts.tz_localize("UTC")
    valid = frame[frame[time_column] <= at_ts].sort_values(time_column)
    if valid.empty:
        return None
    return valid.iloc[-1].to_dict()


def _institutional_schema() -> dict[str, Any]:
    return {
        "schema": "institutional_rsi_adx_vwap_gex_orderflow_contract_v1",
        "status": "GO",
        "authority": "offline_research_and_signal_evaluation_only",
        "long_only": True,
        "regimes": {
            "positive_gex_mean_reversion": {
                "context": ["net_gex > 0", "price above SMA trend filter"],
                "setup": ["RSI oversold", "ADX strong", "lower Bollinger touch", "VWAP discount"],
                "trigger": ["stacked buy imbalance", "or delta flip plus buy absorption"],
                "exit": ["EMA/RSI recovery", "call wall when available", "ATR stop", "time stop"],
            },
            "negative_gex_breakout": {
                "context": ["net_gex < 0", "price above SMA trend filter"],
                "setup": ["RSI momentum extreme", "ADX expansion", "call-wall or upper-band breakout"],
                "trigger": ["stacked buy imbalance and positive delta"],
                "exit": ["EMA8 close", "ATR trailing stop", "time stop"],
            },
        },
        "overlay_csv_columns": {
            "required": ["timestamp", "net_gex"],
            "recommended": [
                "call_wall", "put_wall", "gamma_flip", "delta", "delta_ratio",
                "max_stacked_buy", "delta_flip_up", "buy_absorption",
            ],
        },
        "warnings": [
            "GEX sign is a dealer-positioning proxy, not directly observed dealer inventory.",
            "Historical backtests require historical point-in-time option chains and orderflow. Current snapshots must not be backfilled into the past.",
            "Orderflow inferred with the tick rule has lower confidence than exchange-provided aggressor flags.",
        ],
        "execution": {
            "orders_enabled": False,
            "broker_calls_enabled": False,
        },
    }


def _evaluate_institutional(
    enriched: Any,
    gex: Mapping[str, Any] | None,
    flow: Mapping[str, Any] | None,
    *,
    min_stacked_buy: int,
    wall_distance_atr: float,
    breakout_rsi: float,
) -> dict[str, Any]:
    np, pd = _require_analytics()
    latest = enriched.iloc[-1]
    previous = enriched.iloc[-2] if len(enriched) > 1 else latest
    net_gex = None
    call_wall = None
    put_wall = None
    gamma_flip = None
    if gex:
        net_gex = gex.get("net_gex", gex.get("net_gex_1pct"))
        call_wall = gex.get("call_wall")
        put_wall = gex.get("put_wall")
        gamma_flip = gex.get("gamma_flip")
    net_gex = float(net_gex) if net_gex is not None and pd.notna(net_gex) else None
    call_wall = float(call_wall) if call_wall is not None and pd.notna(call_wall) else None
    put_wall = float(put_wall) if put_wall is not None and pd.notna(put_wall) else None

    stacked_buy = int(float(flow.get("max_stacked_buy", 0))) if flow else 0
    delta = float(flow.get("delta", 0.0)) if flow and pd.notna(flow.get("delta", 0.0)) else 0.0
    delta_flip = bool(flow.get("delta_flip_up", False)) if flow else False
    absorption = bool(flow.get("buy_absorption", False)) if flow else False
    orderflow_trigger = stacked_buy >= min_stacked_buy or (delta_flip and absorption)

    atr_value = float(latest["atr"]) if pd.notna(latest.get("atr")) else float("nan")
    near_put_wall = True
    if put_wall is not None and math.isfinite(atr_value) and atr_value > 0:
        near_put_wall = abs(float(latest["close"]) - put_wall) <= wall_distance_atr * atr_value
    positive_setup = bool(
        net_gex is not None
        and net_gex > 0
        and float(latest["close"]) > float(latest["trend_sma"])
        and float(latest["adx"]) >= 25.0
        and float(latest["rsi"]) <= 10.0
        and float(latest["close"]) <= float(latest["bb_lower"])
        and float(latest["close"]) <= float(latest["vwap"])
        and near_put_wall
    ) if all(pd.notna(latest.get(column)) for column in ("trend_sma", "adx", "rsi", "bb_lower", "vwap")) else False

    adx_expanding = bool(
        pd.notna(latest.get("adx"))
        and pd.notna(previous.get("adx"))
        and float(latest["adx"]) >= 25.0
        and float(latest["adx"]) > float(previous["adx"])
    )
    call_break = call_wall is not None and float(latest["close"]) > call_wall and float(previous["close"]) <= call_wall
    band_break = pd.notna(latest.get("bb_upper")) and float(latest["close"]) > float(latest["bb_upper"])
    negative_setup = bool(
        net_gex is not None
        and net_gex < 0
        and pd.notna(latest.get("trend_sma"))
        and float(latest["close"]) > float(latest["trend_sma"])
        and pd.notna(latest.get("rsi"))
        and float(latest["rsi"]) >= breakout_rsi
        and adx_expanding
        and (call_break or band_break)
    )

    if positive_setup and orderflow_trigger:
        action = "LONG_CANDIDATE_POSITIVE_GEX_MEAN_REVERSION"
    elif negative_setup and orderflow_trigger and delta > 0:
        action = "LONG_CANDIDATE_NEGATIVE_GEX_BREAKOUT"
    elif positive_setup:
        action = "WAIT_FOR_ORDERFLOW_MEAN_REVERSION"
    elif negative_setup:
        action = "WAIT_FOR_ORDERFLOW_BREAKOUT"
    elif net_gex is None:
        action = "NO_GEX_CONTEXT"
    else:
        action = "NO_SETUP"
    return {
        "schema": "institutional_signal_evaluation_v1",
        "status": "GO",
        "timestamp": enriched.index[-1],
        "action": action,
        "market": {
            "close": float(latest["close"]),
            "rsi_2": float(latest["rsi"]) if pd.notna(latest.get("rsi")) else None,
            "adx_5": float(latest["adx"]) if pd.notna(latest.get("adx")) else None,
            "vwap": float(latest["vwap"]) if pd.notna(latest.get("vwap")) else None,
            "sma": float(latest["trend_sma"]) if pd.notna(latest.get("trend_sma")) else None,
            "bb_lower": float(latest["bb_lower"]) if pd.notna(latest.get("bb_lower")) else None,
            "bb_upper": float(latest["bb_upper"]) if pd.notna(latest.get("bb_upper")) else None,
            "atr": atr_value if math.isfinite(atr_value) else None,
        },
        "gex": {
            "net_gex": net_gex,
            "call_wall": call_wall,
            "put_wall": put_wall,
            "gamma_flip": gamma_flip,
        },
        "orderflow": {
            "max_stacked_buy": stacked_buy,
            "delta": delta,
            "delta_flip_up": delta_flip,
            "buy_absorption": absorption,
            "trigger": orderflow_trigger,
        },
        "conditions": {
            "positive_gex_mean_reversion_setup": positive_setup,
            "negative_gex_breakout_setup": negative_setup,
            "near_put_wall": near_put_wall,
            "adx_expanding": adx_expanding,
        },
        "execution": {
            "orders_enabled": False,
            "broker_calls_enabled": False,
        },
    }


def _institutional_backtest(
    project_root: Path,
    rsi_module: ModuleType,
    args: argparse.Namespace,
) -> dict[str, Any]:
    np, pd = _require_analytics()
    candles_path = _resolve_output_path(project_root, args.candles)
    if not candles_path.is_file():
        raise HubError(f"Candle CSV does not exist: {candles_path}")
    raw = rsi_module.load_csv(candles_path)
    cfg = rsi_module.StrategyConfig(
        initial_capital=args.capital,
        rsi_entry=args.rsi_entry,
        rsi_exit=args.rsi_exit,
        adx_entry=args.adx_entry,
        trend_sma_period=args.trend_sma,
        vwap_kind=args.vwap_kind,
        vwap_window=args.vwap_window,
        vwap_timezone=args.vwap_timezone,
        risk_per_trade=args.risk_per_trade,
        max_position_fraction=args.max_position_fraction,
        fee_rate=args.fee_rate,
        slippage_bps=args.slippage_bps,
    )
    enriched, selected_vwap = rsi_module.add_indicators(raw, cfg)
    enriched["ema8"] = enriched["close"].ewm(span=8, adjust=False, min_periods=8).mean()
    overlay = _load_overlay(project_root, args.overlay, enriched.index, args.overlay_max_age)
    frame = enriched.join(overlay.drop(columns=["overlay_timestamp"], errors="ignore"), how="left")
    if "net_gex" not in frame.columns:
        raise HubError("Overlay must contain net_gex or net_gex_1pct")
    for column in ("call_wall", "put_wall", "delta", "delta_ratio", "max_stacked_buy"):
        if column not in frame.columns:
            frame[column] = np.nan
    for column in ("delta_flip_up", "buy_absorption", "orderflow_buy_trigger"):
        if column not in frame.columns:
            frame[column] = False
        else:
            frame[column] = _coerce_bool_series(frame[column])

    cash = float(args.capital)
    position: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    slip = args.slippage_bps / 10_000.0

    for index in range(len(frame)):
        timestamp = frame.index[index]
        row = frame.iloc[index]
        previous = frame.iloc[index - 1] if index > 0 else row

        if position is None and pending is not None:
            entry_price = float(row["open"]) * (1.0 + slip)
            atr_signal = float(pending["atr"])
            stop_mult = args.mr_stop_atr if pending["regime"] == "positive_gex_mean_reversion" else args.breakout_stop_atr
            stop_price = entry_price - stop_mult * atr_signal
            if pending["regime"] == "negative_gex_breakout" and math.isfinite(float(pending.get("call_wall", np.nan))):
                wall_stop = float(pending["call_wall"]) - args.wall_buffer_atr * atr_signal
                if wall_stop < entry_price:
                    stop_price = max(stop_price, wall_stop)
            risk_budget = cash * args.risk_per_trade
            stop_distance = max(entry_price - stop_price, entry_price * 0.001)
            quantity_risk = risk_budget / stop_distance
            quantity_cap = cash * args.max_position_fraction / (entry_price * (1.0 + args.fee_rate))
            quantity = max(0.0, min(quantity_risk, quantity_cap))
            notional = quantity * entry_price
            entry_fee = notional * args.fee_rate
            if quantity > 0 and notional + entry_fee <= cash:
                cash -= notional + entry_fee
                position = {
                    "entry_time": timestamp,
                    "entry_index": index,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "entry_fee": entry_fee,
                    "stop_price": stop_price,
                    "highest": float(row["high"]),
                    "regime": pending["regime"],
                    "call_wall": pending.get("call_wall"),
                    "signal_time": pending["signal_time"],
                }
            pending = None

        if position is not None:
            position["highest"] = max(float(position["highest"]), float(row["high"]))
            if position["regime"] == "negative_gex_breakout" and pd.notna(row.get("atr")):
                trail = float(position["highest"]) - args.breakout_trailing_atr * float(row["atr"])
                position["stop_price"] = max(float(position["stop_price"]), trail)
            exit_reason = None
            exit_price = None
            if float(row["open"]) <= float(position["stop_price"]):
                exit_price = float(row["open"]) * (1.0 - slip)
                exit_reason = "stop_gap"
            elif float(row["low"]) <= float(position["stop_price"]):
                exit_price = float(position["stop_price"]) * (1.0 - slip)
                exit_reason = "stop_loss"
            else:
                held = index - int(position["entry_index"])
                if position["regime"] == "positive_gex_mean_reversion":
                    call_wall = position.get("call_wall")
                    if call_wall is not None and math.isfinite(float(call_wall)) and float(call_wall) > float(position["entry_price"]) and float(row["high"]) >= float(call_wall):
                        exit_price = float(call_wall) * (1.0 - slip)
                        exit_reason = "call_wall_target"
                    elif held >= 1 and pd.notna(row.get("exit_ema")) and float(row["close"]) >= float(row["exit_ema"]):
                        exit_price = float(row["close"]) * (1.0 - slip)
                        exit_reason = "ema_recovery"
                    elif held >= 1 and pd.notna(row.get("rsi")) and float(row["rsi"]) >= args.rsi_exit:
                        exit_price = float(row["close"]) * (1.0 - slip)
                        exit_reason = "rsi_recovery"
                    elif held >= args.mr_max_hold:
                        exit_price = float(row["close"]) * (1.0 - slip)
                        exit_reason = "time_exit"
                else:
                    if held >= 1 and pd.notna(row.get("ema8")) and float(row["close"]) < float(row["ema8"]):
                        exit_price = float(row["close"]) * (1.0 - slip)
                        exit_reason = "ema8_trailing_exit"
                    elif held >= args.breakout_max_hold:
                        exit_price = float(row["close"]) * (1.0 - slip)
                        exit_reason = "time_exit"
            if exit_reason is not None and exit_price is not None:
                exit_notional = float(position["quantity"]) * exit_price
                exit_fee = exit_notional * args.fee_rate
                cash += exit_notional - exit_fee
                gross = float(position["quantity"]) * (exit_price - float(position["entry_price"]))
                fees = float(position["entry_fee"]) + exit_fee
                net = gross - fees
                cost_basis = float(position["quantity"]) * float(position["entry_price"]) + float(position["entry_fee"])
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "regime": position["regime"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "quantity": position["quantity"],
                        "gross_pnl": gross,
                        "fees": fees,
                        "net_pnl": net,
                        "return_pct": net / cost_basis if cost_basis > 0 else 0.0,
                        "bars_held": index - int(position["entry_index"]),
                        "exit_reason": exit_reason,
                    }
                )
                position = None

        if position is None and pending is None and index < len(frame) - 1:
            required = ("rsi", "adx", "trend_sma", "bb_lower", "bb_upper", "vwap", "atr", "net_gex")
            if all(pd.notna(row.get(column)) for column in required):
                stacked = float(row.get("max_stacked_buy", 0.0)) if pd.notna(row.get("max_stacked_buy", np.nan)) else 0.0
                delta = float(row.get("delta", 0.0)) if pd.notna(row.get("delta", np.nan)) else 0.0
                flow_trigger = bool(row.get("orderflow_buy_trigger", False)) or stacked >= args.min_stacked_buy or (bool(row.get("delta_flip_up", False)) and bool(row.get("buy_absorption", False)))
                near_put = True
                if pd.notna(row.get("put_wall", np.nan)):
                    near_put = abs(float(row["close"]) - float(row["put_wall"])) <= args.wall_distance_atr * float(row["atr"])
                positive = (
                    float(row["net_gex"]) > 0
                    and float(row["close"]) > float(row["trend_sma"])
                    and float(row["adx"]) >= args.adx_entry
                    and float(row["rsi"]) <= args.rsi_entry
                    and float(row["close"]) <= float(row["bb_lower"])
                    and float(row["close"]) <= float(row["vwap"])
                    and near_put
                    and flow_trigger
                )
                adx_expansion = float(row["adx"]) >= args.adx_entry and float(row["adx"]) > float(previous.get("adx", row["adx"]))
                call_break = pd.notna(row.get("call_wall", np.nan)) and float(row["close"]) > float(row["call_wall"]) and float(previous["close"]) <= float(row["call_wall"])
                negative = (
                    float(row["net_gex"]) < 0
                    and float(row["close"]) > float(row["trend_sma"])
                    and float(row["rsi"]) >= args.breakout_rsi
                    and adx_expansion
                    and (call_break or float(row["close"]) > float(row["bb_upper"]))
                    and flow_trigger
                    and delta > 0
                )
                if positive or negative:
                    pending = {
                        "signal_time": timestamp,
                        "regime": "positive_gex_mean_reversion" if positive else "negative_gex_breakout",
                        "atr": float(row["atr"]),
                        "call_wall": float(row["call_wall"]) if pd.notna(row.get("call_wall", np.nan)) else np.nan,
                    }

        equity = cash
        in_position = 0
        if position is not None:
            equity += float(position["quantity"]) * float(row["close"])
            in_position = 1
        equity_rows.append({"timestamp": timestamp, "close": float(row["close"]), "cash": cash, "equity": equity, "in_position": in_position})

    if position is not None:
        row = frame.iloc[-1]
        timestamp = frame.index[-1]
        exit_price = float(row["close"]) * (1.0 - slip)
        exit_notional = float(position["quantity"]) * exit_price
        exit_fee = exit_notional * args.fee_rate
        cash += exit_notional - exit_fee
        gross = float(position["quantity"]) * (exit_price - float(position["entry_price"]))
        fees = float(position["entry_fee"]) + exit_fee
        net = gross - fees
        cost_basis = float(position["quantity"]) * float(position["entry_price"]) + float(position["entry_fee"])
        trades.append({
            "entry_time": position["entry_time"], "exit_time": timestamp, "regime": position["regime"],
            "entry_price": position["entry_price"], "exit_price": exit_price, "quantity": position["quantity"],
            "gross_pnl": gross, "fees": fees, "net_pnl": net,
            "return_pct": net / cost_basis if cost_basis > 0 else 0.0,
            "bars_held": len(frame) - 1 - int(position["entry_index"]), "exit_reason": "end_of_data",
        })
        equity_rows[-1]["cash"] = cash
        equity_rows[-1]["equity"] = cash
        equity_rows[-1]["in_position"] = 0

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows).set_index("timestamp")
    ending = float(equity_df["equity"].iloc[-1])
    running = equity_df["equity"].cummax()
    max_dd = float((equity_df["equity"] / running - 1.0).min())
    if trades_df.empty:
        profit_factor = 0.0
        win_rate = 0.0
        fees_paid = 0.0
    else:
        gains = float(trades_df.loc[trades_df["net_pnl"] > 0, "net_pnl"].sum())
        losses = abs(float(trades_df.loc[trades_df["net_pnl"] < 0, "net_pnl"].sum()))
        profit_factor = gains / losses if losses > 0 else float("inf") if gains > 0 else 0.0
        win_rate = float((trades_df["net_pnl"] > 0).mean())
        fees_paid = float(trades_df["fees"].sum())
    metrics = {
        "starting_equity": args.capital,
        "ending_equity": ending,
        "total_return": ending / args.capital - 1.0,
        "max_drawdown": max_dd,
        "trade_count": int(len(trades_df)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "fees_paid": fees_paid,
        "exposure": float(equity_df["in_position"].mean()),
        "positive_gex_trades": int((trades_df["regime"] == "positive_gex_mean_reversion").sum()) if not trades_df.empty else 0,
        "negative_gex_trades": int((trades_df["regime"] == "negative_gex_breakout").sum()) if not trades_df.empty else 0,
    }
    output_dir = _resolve_output_path(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = output_dir / "institutional_trades.csv"
    equity_path = output_dir / "institutional_equity.csv"
    enriched_path = output_dir / "institutional_enriched.csv"
    summary_path = output_dir / "institutional_summary.json"
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path)
    frame.to_csv(enriched_path)
    report = {
        "schema": "institutional_gex_orderflow_backtest_v1",
        "status": "GO",
        "authority": "offline_backtest_only",
        "selected_vwap_kind": selected_vwap,
        "metrics": metrics,
        "artifacts": {
            "summary": str(summary_path),
            "trades": str(trades_path),
            "equity": str(equity_path),
            "enriched": str(enriched_path),
        },
        "validation_warning": "The overlay must contain historical point-in-time GEX and orderflow. Backfilling a current GEX snapshot would invalidate this backtest.",
        "execution": {"orders_enabled": False, "broker_calls_enabled": False},
    }
    summary_path.write_text(json.dumps(_json_safe_value(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _institutional_command(project_root: Path, rsi_module: ModuleType, argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="strategy_research_hub.py institutional")
    sub = parser.add_subparsers(dest="institutional_command", required=True)
    sub.add_parser("schema", help="Print the GEX and orderflow strategy contract.")
    evaluate = sub.add_parser("evaluate", help="Evaluate the latest closed candle with GEX and orderflow context.")
    evaluate.add_argument("--candles", required=True)
    evaluate.add_argument("--gex-json")
    evaluate.add_argument("--orderflow-bars")
    evaluate.add_argument("--vwap-kind", default="auto", choices=["auto", "session", "rolling"])
    evaluate.add_argument("--vwap-window", type=int, default=20)
    evaluate.add_argument("--vwap-timezone", default="UTC")
    evaluate.add_argument("--min-stacked-buy", type=int, default=3)
    evaluate.add_argument("--wall-distance-atr", type=float, default=1.5)
    evaluate.add_argument("--breakout-rsi", type=float, default=90.0)

    backtest = sub.add_parser("backtest", help="Backtest the two long-only GEX regimes with a PIT overlay CSV.")
    backtest.add_argument("--candles", required=True)
    backtest.add_argument("--overlay", required=True)
    backtest.add_argument("--overlay-max-age", default="1D")
    backtest.add_argument("--capital", type=float, default=10_000.0)
    backtest.add_argument("--vwap-kind", default="auto", choices=["auto", "session", "rolling"])
    backtest.add_argument("--vwap-window", type=int, default=20)
    backtest.add_argument("--vwap-timezone", default="UTC")
    backtest.add_argument("--trend-sma", type=int, default=200)
    backtest.add_argument("--rsi-entry", type=float, default=10.0)
    backtest.add_argument("--rsi-exit", type=float, default=60.0)
    backtest.add_argument("--breakout-rsi", type=float, default=90.0)
    backtest.add_argument("--adx-entry", type=float, default=25.0)
    backtest.add_argument("--min-stacked-buy", type=int, default=3)
    backtest.add_argument("--wall-distance-atr", type=float, default=1.5)
    backtest.add_argument("--wall-buffer-atr", type=float, default=0.25)
    backtest.add_argument("--mr-stop-atr", type=float, default=1.5)
    backtest.add_argument("--breakout-stop-atr", type=float, default=1.5)
    backtest.add_argument("--breakout-trailing-atr", type=float, default=2.0)
    backtest.add_argument("--mr-max-hold", type=int, default=12)
    backtest.add_argument("--breakout-max-hold", type=int, default=40)
    backtest.add_argument("--risk-per-trade", type=float, default=0.005)
    backtest.add_argument("--max-position-fraction", type=float, default=0.20)
    backtest.add_argument("--fee-rate", type=float, default=0.001)
    backtest.add_argument("--slippage-bps", type=float, default=5.0)
    backtest.add_argument("--output-dir", default="output/institutional")
    args = parser.parse_args(list(argv))

    if args.institutional_command == "schema":
        _print_payload(_institutional_schema())
        return 0
    if args.institutional_command == "evaluate":
        candles = _resolve_output_path(project_root, args.candles)
        if not candles.is_file():
            raise HubError(f"Candle CSV does not exist: {candles}")
        raw = rsi_module.load_csv(candles)
        cfg = rsi_module.StrategyConfig(
            vwap_kind=args.vwap_kind,
            vwap_window=args.vwap_window,
            vwap_timezone=args.vwap_timezone,
        )
        enriched, selected = rsi_module.add_indicators(raw, cfg)
        gex = _read_json(_resolve_output_path(project_root, args.gex_json)) if args.gex_json else None
        flow = _latest_orderflow_row(project_root, args.orderflow_bars, enriched.index[-1])
        report = _evaluate_institutional(
            enriched,
            gex,
            flow,
            min_stacked_buy=args.min_stacked_buy,
            wall_distance_atr=args.wall_distance_atr,
            breakout_rsi=args.breakout_rsi,
        )
        report["selected_vwap_kind"] = selected
        _print_payload(report)
        return 0
    report = _institutional_backtest(project_root, rsi_module, args)
    _print_payload(report)
    return 0

def _usage() -> str:
    return f"""\
Strategy Research Hub {PROGRAM_VERSION}

Usage:
  python strategy_research_hub.py [global options] version
  python strategy_research_hub.py [global options] doctor
  python strategy_research_hub.py [global options] main <main.py arguments...>
  python strategy_research_hub.py [global options] combo <combo-lab arguments...>
  python strategy_research_hub.py [global options] combo-native <combo-lab arguments...>
  python strategy_research_hub.py [global options] rsi <RSI backtester arguments...>
  python strategy_research_hub.py [global options] data collect <arguments...>
  python strategy_research_hub.py [global options] gex snapshot <arguments...>
  python strategy_research_hub.py [global options] orderflow footprint <arguments...>
  python strategy_research_hub.py [global options] institutional <schema|evaluate|backtest> <arguments...>

Commands:
  version        Print the hub version, file path and SHA-256.
  doctor         Check component imports and temporary combo strategy injection.
  main           Delegate to main.py without modifying it.
  combo          Inject rsi2_adx5_vwap in memory and run the combo lab.
  combo-native   Run the original combo lab without injection.
  rsi            Run the standalone RSI2-ADX5-VWAP backtester or sweep.
                 A missing --csv is downloaded automatically from its filename.
  data           Download normalized OHLCV CSV data.
  gex            Calculate a current or CSV-backed GEX proxy and walls.
  orderflow      Build footprint, delta, stacked imbalance and absorption features.
  institutional  Evaluate or backtest positive-GEX mean reversion and
                 negative-GEX breakout regimes. No order execution is included.

Global options must appear before the command:
  --project-root PATH   Default: directory containing this launcher.
  --main-file PATH      Override main.py path.
  --combo-file PATH     Override strategy_combo_research_lab.py path.
  --rsi-file PATH       Override rsi2_adx5_vwap.py path.
  --traceback           Print a full traceback on launcher errors.

Examples:
  python strategy_research_hub.py version
  python strategy_research_hub.py doctor
  python strategy_research_hub.py data collect --ticker SPY --period 10y --interval 1d --output data/SPY_daily.csv
  python strategy_research_hub.py rsi --csv data/SPY_daily.csv --capital 10000 --vwap-kind rolling
  python strategy_research_hub.py gex snapshot --ticker SPY --expiration all --max-expirations 4
  python strategy_research_hub.py orderflow footprint --trades-csv data/SPY_ticks.csv --tick-size 0.01
  python strategy_research_hub.py institutional schema
  python strategy_research_hub.py combo run --preset smoke --max-symbols 50
"""

def _parse_top_level(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--main-file", default=None)
    parser.add_argument("--combo-file", default=None)
    parser.add_argument("--rsi-file", default=None)
    parser.add_argument("--traceback", action="store_true")
    parser.add_argument("command", nargs="?")
    return parser.parse_known_args(list(argv))


def _version_report() -> int:
    current = Path(__file__).resolve()
    payload = {
        "schema": "strategy_research_hub_version_v1",
        "version": PROGRAM_VERSION,
        "file": str(current),
        "sha256": _sha256(current),
        "commands": [
            "version", "doctor", "main", "combo", "combo-native",
            "rsi", "data", "gex", "orderflow", "institutional",
        ],
    }
    _print_payload(payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0

    args, delegated = _parse_top_level(raw)
    if not args.command:
        print(_usage(), file=sys.stderr)
        return 2

    command = str(args.command).lower()
    if command == "framework":
        command = "main"
    elif command == "lab":
        command = "combo"

    try:
        if command == "version":
            if delegated:
                raise HubError(f"version takes no delegated arguments: {delegated}")
            return _version_report()

        root = Path(args.project_root).expanduser().resolve()
        if not root.is_dir():
            raise HubError(f"Project root does not exist: {root}")

        if command == "data":
            return _data_command(root, delegated)
        if command == "gex":
            raise HubError("OPTIONS_GEX_PATH_FORBIDDEN_FOR_SWING_RESEARCH")
        if command == "orderflow":
            return _orderflow_command(root, delegated)
        if command == "institutional":
            raise HubError(
                "TICK_ORDERFLOW_OPTIONS_OVERLAY_FORBIDDEN_FOR_SWING_RESEARCH"
            )

        paths = _component_paths(args)
        if command == "doctor":
            if delegated:
                raise HubError(f"doctor takes no delegated arguments: {delegated}")
            return _doctor(paths)

        if command == "main":
            module = _load_module(paths["main"], "_stocks_framework_main_hub_run", root)
            return _call_module_main(module, paths["main"], delegated)

        if command in {"combo", "combo-native"}:
            combo_module = _load_module(paths["combo"], "_strategy_combo_lab_hub_run", root)
            if command == "combo":
                rsi_module = _load_module(paths["rsi"], "_rsi2_adx5_vwap_hub_combo", root)
                _install_combo_strategy(combo_module, rsi_module)
            return _call_module_main(combo_module, paths["combo"], delegated)

        if command == "rsi":
            module = _load_module(paths["rsi"], "_rsi2_adx5_vwap_hub_run", root)
            prepared = _prepare_rsi_arguments(root, delegated)
            return _call_module_main(module, paths["rsi"], prepared)

        if command == "institutional":
            rsi_module = _load_module(paths["rsi"], "_rsi2_adx5_vwap_hub_institutional", root)
            return _institutional_command(root, rsi_module, delegated)

        raise HubError(f"Unknown command: {args.command}\n\n{_usage()}")
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"HUB ERROR: {exc}", file=sys.stderr)
        if args.traceback:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
