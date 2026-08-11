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
from typing import Any, Iterator, Sequence


PROGRAM_VERSION = "1.0.0"
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
        "safety": {
            "source_files_modified": False,
            "combo_patch_scope": "current_process_only",
            "broker_order_methods_added": False,
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


def _usage() -> str:
    return f"""\
Strategy Research Hub {PROGRAM_VERSION}

Usage:
  python strategy_research_hub.py [global options] doctor
  python strategy_research_hub.py [global options] main <main.py arguments...>
  python strategy_research_hub.py [global options] combo <combo-lab arguments...>
  python strategy_research_hub.py [global options] combo-native <combo-lab arguments...>
  python strategy_research_hub.py [global options] rsi <RSI backtester arguments...>

Commands:
  doctor        Check paths, imports and temporary strategy injection.
  main          Import main.py and delegate its CLI without changing it.
  combo         Import the combo lab, inject {INJECTED_STRATEGY_NAME}, then delegate.
  combo-native  Run the combo lab exactly as-is, without the injected strategy.
  rsi           Import and run the standalone RSI2-ADX5-VWAP backtester/sweep.

Global options must appear before the command:
  --project-root PATH   Default: directory containing this launcher.
  --main-file PATH      Override main.py path.
  --combo-file PATH     Override strategy_combo_research_lab.py path.
  --rsi-file PATH       Override rsi2_adx5_vwap.py path.
  --traceback           Print a full traceback on launcher errors.

Examples:
  python strategy_research_hub.py doctor
  python strategy_research_hub.py main doctor
  python strategy_research_hub.py combo list
  python strategy_research_hub.py combo run --preset smoke --max-symbols 50
  python strategy_research_hub.py combo run --preset long --include-strategies {INJECTED_STRATEGY_NAME}
  python strategy_research_hub.py rsi --csv data/SPY_1d.csv --capital 2000
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
        paths = _component_paths(args)
        if command == "doctor":
            if delegated:
                raise HubError(f"doctor takes no delegated arguments: {delegated}")
            return _doctor(paths)

        root = paths["project_root"]
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
            return _call_module_main(module, paths["rsi"], delegated)

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
