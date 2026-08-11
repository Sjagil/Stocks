from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from stocks.data.phase5_common import read_parquet_records, sha256_file, utc_now_iso
from stocks.data.total_returns import TotalReturnLayout, validate_total_return_cache
from stocks.research.instrument_manifest import InstrumentManifestLayout, validate_instrument_manifest


METRIC_FIELDS = (
    "start_date",
    "end_date",
    "instrument_count",
    "rebalance_count",
    "closed_position_episodes",
    "gross_return",
    "net_return",
    "CAGR",
    "annualized_volatility",
    "Sharpe",
    "Sortino",
    "maximum_drawdown",
    "Calmar",
    "trade_profit_factor",
    "period_profit_factor",
    "expectancy_per_episode",
    "win_rate",
    "average_win",
    "average_loss",
    "turnover",
    "transaction_costs",
    "average_cash",
    "maximum_cash",
    "worst_month",
    "worst_year",
    "best_year",
    "region_exposure",
    "sleeve_exposure",
    "currency_exposure",
)


@dataclass(frozen=True)
class Phase6Layout:
    output_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> Phase6Layout:
        return cls(output_dir=project_root / "output" / "research" / "phase6")

    @property
    def dataset_audit_json(self) -> Path:
        return self.output_dir / "dataset-audit.json"

    @property
    def benchmarks_json(self) -> Path:
        return self.output_dir / "benchmarks.json"

    @property
    def strategy_grid_json(self) -> Path:
        return self.output_dir / "strategy-grid.json"

    @property
    def walk_forward_json(self) -> Path:
        return self.output_dir / "walk-forward.json"

    @property
    def phase6_status_json(self) -> Path:
        return self.output_dir / "phase6-status.json"


def phase6_schema() -> dict[str, Any]:
    return {
        "schema": "phase6_baselines_strategy_schema_v1",
        "status": "OFFLINE_SCHEMA_ONLY",
        "metric_fields": list(METRIC_FIELDS),
        "benchmarks": [
            "buy_and_hold_per_instrument",
            "equal_weight_monthly",
            "inverse_volatility_monthly",
            "trend_200_cash_filter",
            "momentum_12_1_top4_trend_filter",
        ],
        "strategy_grid_size": 108,
        "execution": {"orders_enabled": False, "provider_calls_enabled": False},
        "financial_calls": _zero_financial_calls(),
    }


def run_phase6_pipeline(project_root: Path) -> dict[str, Any]:
    layout = Phase6Layout.from_project_root(project_root)
    layout.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_phase6_dataset(project_root)
    audit = dataset_audit(dataset)
    _write_json(layout.dataset_audit_json, audit)
    benchmarks = run_baselines(dataset)
    _write_json(layout.benchmarks_json, benchmarks)
    grid = run_strategy_grid(dataset)
    _write_json(layout.strategy_grid_json, grid)
    walk_forward = run_walk_forward(dataset, grid)
    _write_json(layout.walk_forward_json, walk_forward)
    status = phase6_status(project_root, audit=audit, benchmarks=benchmarks, grid=grid, walk_forward=walk_forward)
    _write_json(layout.phase6_status_json, status)
    return status


def phase6_status(
    project_root: Path,
    *,
    audit: dict[str, Any] | None = None,
    benchmarks: dict[str, Any] | None = None,
    grid: dict[str, Any] | None = None,
    walk_forward: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layout = Phase6Layout.from_project_root(project_root)
    if audit is None and layout.dataset_audit_json.exists():
        audit = json.loads(layout.dataset_audit_json.read_text(encoding="utf-8"))
    if benchmarks is None and layout.benchmarks_json.exists():
        benchmarks = json.loads(layout.benchmarks_json.read_text(encoding="utf-8"))
    if grid is None and layout.strategy_grid_json.exists():
        grid = json.loads(layout.strategy_grid_json.read_text(encoding="utf-8"))
    if walk_forward is None and layout.walk_forward_json.exists():
        walk_forward = json.loads(layout.walk_forward_json.read_text(encoding="utf-8"))
    checks = {
        "dataset_audit_go": audit is not None and audit.get("status") == "GO",
        "five_benchmarks": benchmarks is not None and benchmarks.get("benchmark_family_count", 0) >= 5,
        "strategy_grid_108": grid is not None and grid.get("config_count") == 108,
        "walk_forward_available": walk_forward is not None and walk_forward.get("fold_count", 0) > 0,
        "no_financial_calls": True,
    }
    return {
        "schema": "phase6_baselines_strategy_status_v1",
        "status": "PHASE6_BASELINES_AND_STRATEGY_EVIDENCE_GO" if all(checks.values()) else "NO_GO",
        "generated_at": utc_now_iso(),
        "checks": checks,
        "dataset_audit": _compact_audit(audit),
        "benchmarks": _compact_benchmarks(benchmarks),
        "strategy_grid": _compact_grid(grid),
        "walk_forward": _compact_walk_forward(walk_forward),
        "financial_finalist_status": _financial_finalist_status(grid, walk_forward),
        "artifacts": {
            "dataset_audit": str(layout.dataset_audit_json),
            "benchmarks": str(layout.benchmarks_json),
            "strategy_grid": str(layout.strategy_grid_json),
            "walk_forward": str(layout.walk_forward_json),
        },
        "financial_calls": _zero_financial_calls(),
    }


def load_phase6_dataset(project_root: Path) -> dict[str, Any]:
    manifest_layout = InstrumentManifestLayout.from_project_root(project_root)
    manifest_validation = validate_instrument_manifest(manifest_layout)
    if manifest_validation["status"] != "GO":
        raise ValueError("research manifest is not valid")
    total_return_validation = validate_total_return_cache(TotalReturnLayout.from_project_root(project_root))
    if total_return_validation["status"] != "GO":
        raise ValueError("total-return cache is not valid")
    manifest = yaml.safe_load(manifest_layout.path.read_text(encoding="utf-8"))
    meta_by_symbol = {item["symbol"]: item for item in manifest["instruments"]}
    series: dict[str, list[dict[str, Any]]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for path in sorted((project_root / "data" / "total_returns").glob("security_type=STK/con_id=*/interval=1d/total_returns.parquet")):
        records = read_parquet_records(path)
        if not records:
            continue
        symbol = str(records[0]["symbol"])
        if symbol not in meta_by_symbol:
            continue
        meta = meta_by_symbol[symbol]
        con_id = int(records[0]["con_id"])
        key = str(con_id)
        series[key] = records
        metadata[key] = {
            "instrument_id": meta["instrument_id"],
            "symbol": symbol,
            "con_id": con_id,
            "sleeve": meta["sleeve"],
            "region": meta["region"],
            "currency": records[0]["instrument_currency"],
            "base_currency": records[0]["base_currency"],
            "path": str(path),
        }
    return {
        "schema": "phase6_dataset_v1",
        "project_root": str(project_root),
        "series": series,
        "metadata": metadata,
        "financial_calls": _zero_financial_calls(),
    }


def dataset_audit(dataset: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    first_dates: list[str] = []
    last_dates: list[str] = []
    all_dates_by_key: dict[str, set[str]] = {}
    action_counts = _corporate_action_counts(Path(dataset["project_root"])) if dataset.get("project_root") else {}
    fx_forward_fill = _fx_forward_fill_by_currency(Path(dataset["project_root"])) if dataset.get("project_root") else {}
    for key, records in dataset["series"].items():
        meta = dataset["metadata"][key]
        dates = [str(record["session_date"]) for record in records]
        first_dates.append(min(dates))
        last_dates.append(max(dates))
        all_dates_by_key[key] = set(dates)
        rows.append(
            {
                **meta,
                "first_valid_date": min(dates),
                "last_valid_date": max(dates),
                "valid_total_return_rows": len(records),
                "instrument_currency": meta["currency"],
                "fx_source": _unique_value(records, "fx_source"),
                "corporate_action_count": action_counts.get(key, {}).get("corporate_action_count", 0),
                "dividend_count": action_counts.get(key, {}).get("dividend_count", 0),
                "split_count": action_counts.get(key, {}).get("split_count", 0),
                "max_fx_forward_fill": fx_forward_fill.get(meta["currency"], 0),
                "maximum_missing_sequence": _max_missing_sequence(dates),
            }
        )
    common_start = max(first_dates)
    common_end = min(last_dates)
    common_dates = sorted(set.intersection(*(dates for dates in all_dates_by_key.values())))
    all_dates = sorted(set.union(*(dates for dates in all_dates_by_key.values())))
    pit_counts = {
        day: sum(1 for records in dataset["series"].values() if str(records[0]["session_date"]) <= day <= str(records[-1]["session_date"]))
        for day in all_dates
    }
    return {
        "schema": "phase6_dataset_audit_v1",
        "status": "GO",
        "instrument_count": len(rows),
        "common_history_universe": {
            "common_start": common_start,
            "common_end": common_end,
            "common_session_count": len([day for day in common_dates if common_start <= day <= common_end]),
        },
        "point_in_time_universe": {
            "enabled": True,
            "min_count": min(pit_counts.values()) if pit_counts else 0,
            "max_count": max(pit_counts.values()) if pit_counts else 0,
        },
        "instruments": sorted(rows, key=lambda item: item["instrument_id"]),
        "financial_calls": _zero_financial_calls(),
    }


def run_baselines(dataset: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_common_history(dataset)
    cost_bps = 10.0
    results: list[dict[str, Any]] = []
    for key in sorted(prepared["returns"]):
        symbol = prepared["metadata"][key]["symbol"]
        weights = [{key: 1.0} for _ in prepared["dates"]]
        results.append(_portfolio_result(f"buy_and_hold_{symbol}", prepared, weights, cost_bps=0.0))
    results.append(_portfolio_result("equal_weight_monthly", prepared, _equal_weight_targets(prepared, "monthly"), cost_bps=cost_bps))
    results.append(
        _portfolio_result(
            "inverse_volatility_monthly",
            prepared,
            _inverse_vol_targets(prepared, lookback=63, rebalance="monthly"),
            cost_bps=cost_bps,
        )
    )
    results.append(
        _portfolio_result(
            "trend_200_cash_filter",
            prepared,
            _trend_targets(prepared, trend_lookback=200, rebalance="monthly"),
            cost_bps=cost_bps,
        )
    )
    results.append(
        _portfolio_result(
            "momentum_12_1_top4_trend_filter",
            prepared,
            _momentum_targets(prepared, momentum_lookback=252, trend_lookback=200, top_n=4, rebalance="monthly"),
            cost_bps=cost_bps,
        )
    )
    return {
        "schema": "phase6_baseline_results_v1",
        "status": "GO",
        "universe_variant": "common_history",
        "benchmark_family_count": 5,
        "result_count": len(results),
        "results": results,
        "financial_calls": _zero_financial_calls(),
    }


def run_strategy_grid(dataset: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_common_history(dataset)
    configs = list(
        itertools.product(
            (63, 126, 252),
            (100, 200),
            ("weekly", "monthly"),
            (0.08, 0.10, 0.12),
            (5.0, 10.0, 20.0),
        )
    )
    results: list[dict[str, Any]] = []
    for momentum, trend, rebalance, target_vol, cost_bps in configs:
        weights = _momentum_targets(
            prepared,
            momentum_lookback=momentum,
            trend_lookback=trend,
            top_n=4,
            rebalance=rebalance,
            target_vol=target_vol,
        )
        result = _portfolio_result(
            "fixed_momentum_trend_rotation",
            prepared,
            weights,
            cost_bps=cost_bps,
            parameters={
                "momentum_lookback": momentum,
                "trend_lookback": trend,
                "rebalance": rebalance,
                "target_vol": target_vol,
                "cost_bps": cost_bps,
            },
        )
        results.append(result)
    positive = [item for item in results if (item["period_profit_factor"] or 0) > 1.0 and item["net_return"] > 0]
    plateau = _plateau_report(results)
    return {
        "schema": "phase6_strategy_grid_v1",
        "status": "GO",
        "config_count": len(results),
        "positive_config_count": len(positive),
        "plateau": plateau,
        "best_by_calmar": max(results, key=lambda item: item["Calmar"] if item["Calmar"] is not None else -999),
        "results": results,
        "financial_calls": _zero_financial_calls(),
    }


def run_walk_forward(dataset: dict[str, Any], grid: dict[str, Any] | None = None) -> dict[str, Any]:
    prepared = _prepare_common_history(dataset)
    dates = prepared["dates"]
    folds: list[dict[str, Any]] = []
    start_year = date.fromisoformat(dates[0]).year
    end_year = date.fromisoformat(dates[-1]).year
    config_candidates = list(
        itertools.product(
            (63, 126, 252),
            (100, 200),
            ("weekly", "monthly"),
            (0.08, 0.10, 0.12),
            (5.0, 10.0, 20.0),
        )
    )
    for train_start_year in range(start_year, end_year - 4):
        train_start = f"{train_start_year}-01-01"
        train_end = f"{train_start_year + 2}-12-31"
        validation_start = f"{train_start_year + 3}-01-01"
        validation_end = f"{train_start_year + 3}-12-31"
        test_start = f"{train_start_year + 4}-01-01"
        test_end = f"{train_start_year + 4}-12-31"
        if test_end > dates[-1]:
            break
        validation_scores: list[tuple[float, tuple[int, int, str, float, float]]] = []
        for config in config_candidates:
            subset = _slice_prepared(prepared, validation_start, validation_end)
            result = _portfolio_result(
                "walk_forward_validation",
                subset,
                _momentum_targets(subset, momentum_lookback=config[0], trend_lookback=config[1], top_n=4, rebalance=config[2], target_vol=config[3]),
                cost_bps=config[4],
            )
            validation_scores.append((result["net_return"], config))
        selected = max(validation_scores, key=lambda item: item[0])[1]
        test_subset = _slice_prepared(prepared, test_start, test_end)
        test_result = _portfolio_result(
            "walk_forward_test",
            test_subset,
            _momentum_targets(test_subset, momentum_lookback=selected[0], trend_lookback=selected[1], top_n=4, rebalance=selected[2], target_vol=selected[3]),
            cost_bps=selected[4],
            parameters={
                "momentum_lookback": selected[0],
                "trend_lookback": selected[1],
                "rebalance": selected[2],
                "target_vol": selected[3],
                "cost_bps": selected[4],
            },
        )
        folds.append(
            {
                "train": {"start": train_start, "end": train_end},
                "validation": {"start": validation_start, "end": validation_end},
                "test": {"start": test_start, "end": test_end},
                "selected_parameters": test_result["parameters"],
                "test_metrics": {field: test_result[field] for field in METRIC_FIELDS},
            }
        )
    test_pfs = [fold["test_metrics"]["period_profit_factor"] for fold in folds if fold["test_metrics"]["period_profit_factor"] is not None]
    sharpes = [fold["test_metrics"]["Sharpe"] for fold in folds if fold["test_metrics"]["Sharpe"] is not None]
    return {
        "schema": "phase6_walk_forward_v1",
        "status": "GO" if folds else "NO_GO",
        "fold_count": len(folds),
        "positive_test_folds": sum(1 for fold in folds if fold["test_metrics"]["net_return"] > 0),
        "median_test_return": _median([fold["test_metrics"]["net_return"] for fold in folds]),
        "median_test_PF": _median(test_pfs),
        "worst_test_PF": min(test_pfs) if test_pfs else None,
        "median_test_Sharpe": _median(sharpes),
        "worst_drawdown": min((fold["test_metrics"]["maximum_drawdown"] for fold in folds), default=None),
        "parameter_selection_frequency": _selection_frequency(folds),
        "folds": folds,
        "financial_calls": _zero_financial_calls(),
    }


def _prepare_common_history(dataset: dict[str, Any]) -> dict[str, Any]:
    first = max(str(records[0]["session_date"]) for records in dataset["series"].values())
    last = min(str(records[-1]["session_date"]) for records in dataset["series"].values())
    date_sets = [{str(item["session_date"]) for item in records if first <= str(item["session_date"]) <= last} for records in dataset["series"].values()]
    dates = sorted(set.intersection(*date_sets))
    returns: dict[str, list[float]] = {}
    prices: dict[str, list[float]] = {}
    for key, records in dataset["series"].items():
        by_date = {str(record["session_date"]): record for record in records}
        returns[key] = [float(Decimal(str(by_date[day]["eur_total_return"]))) for day in dates]
        prices[key] = [float(Decimal(str(by_date[day]["eur_total_return_index"]))) for day in dates]
    return {
        "dates": dates,
        "returns": returns,
        "prices": prices,
        "metadata": dataset["metadata"],
        "cash_key": _cash_key(dataset["metadata"]),
    }


def _slice_prepared(prepared: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    indices = [index for index, day in enumerate(prepared["dates"]) if start <= day <= end]
    return {
        **prepared,
        "dates": [prepared["dates"][index] for index in indices],
        "returns": {key: [values[index] for index in indices] for key, values in prepared["returns"].items()},
        "prices": {key: [values[index] for index in indices] for key, values in prepared["prices"].items()},
    }


def _portfolio_result(
    name: str,
    prepared: dict[str, Any],
    weights_by_period: list[dict[str, float]],
    *,
    cost_bps: float,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dates = prepared["dates"]
    returns = prepared["returns"]
    metadata = prepared["metadata"]
    nav = 100.0
    previous_weights: dict[str, float] = {"CASH": 1.0}
    daily_returns: list[float] = []
    navs: list[float] = [nav]
    turnovers: list[float] = []
    costs: list[float] = []
    cash_weights: list[float] = []
    episodes = _EpisodeTracker()
    exposure_region: dict[str, float] = {}
    exposure_sleeve: dict[str, float] = {}
    exposure_currency: dict[str, float] = {}
    for index in range(1, len(dates)):
        target = weights_by_period[index] if index < len(weights_by_period) else previous_weights
        turnover = 0.5 * sum(abs(target.get(key, 0.0) - previous_weights.get(key, 0.0)) for key in set(target) | set(previous_weights))
        cost = turnover * cost_bps / 10_000.0
        gross = sum(weight * returns[key][index] for key, weight in previous_weights.items() if key in returns)
        net = gross - cost
        nav *= 1.0 + net
        daily_returns.append(net)
        navs.append(nav)
        turnovers.append(turnover)
        costs.append(cost)
        cash_weights.append(max(0.0, 1.0 - sum(weight for key, weight in previous_weights.items() if key in returns)))
        episodes.update(previous_weights, {key: returns[key][index] for key in returns}, nav)
        _add_exposures(exposure_region, exposure_sleeve, exposure_currency, previous_weights, metadata)
        previous_weights = target
    episodes.close_all()
    metrics = _metrics(
        dates=dates,
        navs=navs,
        returns=daily_returns,
        turnovers=turnovers,
        costs=costs,
        cash_weights=cash_weights,
        episodes=episodes.closed_pnls,
        instrument_count=len(returns),
        rebalance_count=sum(1 for value in turnovers if value > 0),
        region_exposure=_average_map(exposure_region, max(1, len(daily_returns))),
        sleeve_exposure=_average_map(exposure_sleeve, max(1, len(daily_returns))),
        currency_exposure=_average_map(exposure_currency, max(1, len(daily_returns))),
    )
    return {
        "schema": "phase6_portfolio_result_v1",
        "name": name,
        "parameters": parameters or {},
        **metrics,
        "financial_calls": _zero_financial_calls(),
    }


def _metrics(
    *,
    dates: list[str],
    navs: list[float],
    returns: list[float],
    turnovers: list[float],
    costs: list[float],
    cash_weights: list[float],
    episodes: list[float],
    instrument_count: int,
    rebalance_count: int,
    region_exposure: dict[str, float],
    sleeve_exposure: dict[str, float],
    currency_exposure: dict[str, float],
) -> dict[str, Any]:
    start = dates[0] if dates else None
    end = dates[-1] if dates else None
    net_return = navs[-1] / navs[0] - 1.0 if len(navs) > 1 else 0.0
    cagr = _cagr(start, end, net_return)
    vol = _annualized_vol(returns)
    sharpe = None if vol == 0 else _mean(returns) / _std(returns) * math.sqrt(252)
    downside = [min(0.0, value) for value in returns]
    sortino = None if _std(downside) == 0 else _mean(returns) / _std(downside) * math.sqrt(252)
    max_dd = _max_drawdown(navs)
    period_pf = _profit_factor(returns)
    episode_pf = _profit_factor(episodes)
    positive_episodes = [value for value in episodes if value > 0]
    negative_episodes = [value for value in episodes if value < 0]
    monthly = _calendar_returns(dates[1:], returns, 7)
    yearly = _calendar_returns(dates[1:], returns, 4)
    return {
        "start_date": start,
        "end_date": end,
        "instrument_count": instrument_count,
        "rebalance_count": rebalance_count,
        "closed_position_episodes": len(episodes),
        "gross_return": net_return + sum(costs),
        "net_return": net_return,
        "CAGR": cagr,
        "annualized_volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "maximum_drawdown": max_dd,
        "Calmar": None if not cagr or max_dd == 0 else cagr / abs(max_dd),
        "trade_profit_factor": episode_pf,
        "period_profit_factor": period_pf,
        "expectancy_per_episode": None if not episodes else sum(episodes) / len(episodes),
        "win_rate": None if not episodes else len(positive_episodes) / len(episodes),
        "average_win": None if not positive_episodes else sum(positive_episodes) / len(positive_episodes),
        "average_loss": None if not negative_episodes else sum(negative_episodes) / len(negative_episodes),
        "turnover": sum(turnovers),
        "transaction_costs": sum(costs),
        "average_cash": _mean(cash_weights) if cash_weights else 0.0,
        "maximum_cash": max(cash_weights) if cash_weights else 0.0,
        "worst_month": min(monthly.values()) if monthly else None,
        "worst_year": min(yearly.values()) if yearly else None,
        "best_year": max(yearly.values()) if yearly else None,
        "region_exposure": region_exposure,
        "sleeve_exposure": sleeve_exposure,
        "currency_exposure": currency_exposure,
    }


def _equal_weight_targets(prepared: dict[str, Any], rebalance: str) -> list[dict[str, float]]:
    keys = sorted(prepared["returns"])
    weight = 1.0 / len(keys)
    previous: dict[str, float] = {}
    targets: list[dict[str, float]] = []
    for index, _day in enumerate(prepared["dates"]):
        if _is_rebalance(prepared["dates"], index, rebalance):
            previous = {key: weight for key in keys}
        targets.append(previous)
    return targets


def _inverse_vol_targets(prepared: dict[str, Any], *, lookback: int, rebalance: str) -> list[dict[str, float]]:
    targets: list[dict[str, float]] = []
    previous: dict[str, float] = {}
    for index, _day in enumerate(prepared["dates"]):
        if index < lookback or not _is_rebalance(prepared["dates"], index, rebalance):
            targets.append(previous)
            continue
        inv = {key: 1.0 / max(_std(values[index - lookback : index]), 1e-6) for key, values in prepared["returns"].items()}
        total = sum(inv.values())
        previous = {key: value / total for key, value in inv.items()}
        targets.append(previous)
    return targets


def _trend_targets(prepared: dict[str, Any], *, trend_lookback: int, rebalance: str) -> list[dict[str, float]]:
    targets: list[dict[str, float]] = []
    previous: dict[str, float] = {}
    cash = prepared["cash_key"]
    for index, _day in enumerate(prepared["dates"]):
        if index < trend_lookback or not _is_rebalance(prepared["dates"], index, rebalance):
            targets.append(previous)
            continue
        eligible = [
            key
            for key, prices in prepared["prices"].items()
            if prices[index] > _mean(prices[index - trend_lookback : index])
        ]
        if not eligible:
            previous = {cash: 1.0} if cash else {}
        else:
            weight = 1.0 / len(eligible)
            previous = {key: weight for key in eligible}
        targets.append(previous)
    return targets


def _momentum_targets(
    prepared: dict[str, Any],
    *,
    momentum_lookback: int,
    trend_lookback: int,
    top_n: int,
    rebalance: str,
    target_vol: float | None = None,
) -> list[dict[str, float]]:
    targets: list[dict[str, float]] = []
    previous: dict[str, float] = {}
    cash = prepared["cash_key"]
    min_lookback = max(momentum_lookback, trend_lookback)
    for index, _day in enumerate(prepared["dates"]):
        if index < min_lookback or not _is_rebalance(prepared["dates"], index, rebalance):
            targets.append(previous)
            continue
        scored = []
        for key, prices in prepared["prices"].items():
            if key == cash:
                continue
            if prices[index] <= _mean(prices[index - trend_lookback : index]):
                continue
            momentum_start = max(0, index - momentum_lookback)
            skip_month = max(momentum_start, index - 21)
            if prices[momentum_start] <= 0:
                continue
            score = prices[skip_month] / prices[momentum_start] - 1.0
            if score > 0:
                scored.append((score, key))
        selected = [key for _score, key in sorted(scored, reverse=True)[:top_n]]
        if not selected:
            previous = {cash: 1.0} if cash else {}
            targets.append(previous)
            continue
        weight = 1.0 / len(selected)
        risky = {key: weight for key in selected}
        if target_vol is not None:
            recent = [sum(risky.get(key, 0.0) * prepared["returns"][key][j] for key in risky) for j in range(index - 63, index) if j >= 0]
            realized = _std(recent) * math.sqrt(252)
            scale = min(1.0, target_vol / realized) if realized > 0 else 1.0
            risky = {key: value * scale for key, value in risky.items()}
            if cash:
                risky[cash] = max(0.0, 1.0 - sum(risky.values()))
        previous = risky
        targets.append(previous)
    return targets


class _EpisodeTracker:
    def __init__(self) -> None:
        self.active: dict[str, float] = {}
        self.closed_pnls: list[float] = []

    def update(self, weights: dict[str, float], returns: dict[str, float], nav: float) -> None:
        for key, weight in weights.items():
            if key == "CASH" or weight <= 0:
                continue
            self.active[key] = self.active.get(key, 0.0) + weight * returns.get(key, 0.0) * nav
        for key in list(self.active):
            if weights.get(key, 0.0) <= 0:
                self.closed_pnls.append(self.active.pop(key))

    def close_all(self) -> None:
        for key in list(self.active):
            self.closed_pnls.append(self.active.pop(key))


def _is_rebalance(dates: list[str], index: int, rebalance: str) -> bool:
    if index == 0:
        return True
    current = date.fromisoformat(dates[index])
    previous = date.fromisoformat(dates[index - 1])
    if rebalance == "weekly":
        return current.isocalendar().week != previous.isocalendar().week
    if rebalance == "monthly":
        return current.month != previous.month or current.year != previous.year
    return True


def _add_exposures(
    region_acc: dict[str, float],
    sleeve_acc: dict[str, float],
    currency_acc: dict[str, float],
    weights: dict[str, float],
    metadata: dict[str, dict[str, Any]],
) -> None:
    for key, weight in weights.items():
        if key not in metadata:
            if key == "CASH":
                _inc(sleeve_acc, "cash", weight)
                _inc(region_acc, "cash", weight)
                _inc(currency_acc, "EUR", weight)
            continue
        meta = metadata[key]
        _inc(region_acc, meta["region"], weight)
        _inc(sleeve_acc, meta["sleeve"], weight)
        _inc(currency_acc, meta["currency"], weight)


def _inc(mapping: dict[str, float], key: str, value: float) -> None:
    mapping[key] = mapping.get(key, 0.0) + value


def _average_map(mapping: dict[str, float], divisor: int) -> dict[str, float]:
    return {key: value / divisor for key, value in sorted(mapping.items())}


def _profit_factor(values: list[float]) -> float | None:
    positive = sum(value for value in values if value > 0)
    negative = abs(sum(value for value in values if value < 0))
    return None if negative == 0 else positive / negative


def _cagr(start: str | None, end: str | None, net_return: float) -> float | None:
    if not start or not end or net_return <= -1:
        return None
    years = (date.fromisoformat(end) - date.fromisoformat(start)).days / 365.25
    return None if years <= 0 else (1.0 + net_return) ** (1.0 / years) - 1.0


def _annualized_vol(values: list[float]) -> float:
    return _std(values) * math.sqrt(252)


def _max_drawdown(navs: list[float]) -> float:
    peak = navs[0]
    drawdown = 0.0
    for nav in navs:
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1.0)
    return drawdown


def _calendar_returns(dates: list[str], returns: list[float], prefix_len: int) -> dict[str, float]:
    grouped: dict[str, float] = {}
    for day, value in zip(dates, returns, strict=False):
        key = day[:prefix_len]
        grouped[key] = (1.0 + grouped.get(key, 0.0)) * (1.0 + value) - 1.0
    return grouped


def _plateau_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [item for item in results if item["net_return"] > 0 and (item["period_profit_factor"] or 0) > 1.0]
    by_momentum: dict[str, int] = {}
    by_trend: dict[str, int] = {}
    by_rebalance: dict[str, int] = {}
    by_cost: dict[str, int] = {}
    for item in positive:
        params = item["parameters"]
        _inc_count(by_momentum, str(params["momentum_lookback"]))
        _inc_count(by_trend, str(params["trend_lookback"]))
        _inc_count(by_rebalance, str(params["rebalance"]))
        _inc_count(by_cost, str(params["cost_bps"]))
    return {
        "positive_config_count": len(positive),
        "has_plateau": len(by_momentum) >= 2 and len(by_trend) >= 2 and len(by_rebalance) >= 1 and len(by_cost) >= 2,
        "positive_by_momentum": by_momentum,
        "positive_by_trend": by_trend,
        "positive_by_rebalance": by_rebalance,
        "positive_by_cost_bps": by_cost,
    }


def _inc_count(mapping: dict[str, int], key: str) -> None:
    mapping[key] = mapping.get(key, 0) + 1


def _selection_frequency(folds: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for fold in folds:
        key = json.dumps(fold["selected_parameters"], sort_keys=True)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _financial_finalist_status(grid: dict[str, Any] | None, walk_forward: dict[str, Any] | None) -> str:
    if not grid or not walk_forward:
        return "NO_GO"
    best = grid.get("best_by_calmar", {})
    stress_positive = all(
        item["net_return"] > 0 and (item["period_profit_factor"] or 0) > 1.0
        for item in grid.get("results", [])
        if item.get("parameters", {}).get("cost_bps") == 20.0
    )
    if (
        (best.get("period_profit_factor") or 0) > 1.10
        and walk_forward.get("positive_test_folds", 0) == walk_forward.get("fold_count", 0)
        and (walk_forward.get("median_test_PF") or 0) > 1.0
        and (walk_forward.get("worst_test_PF") or 0) > 1.0
        and stress_positive
        and grid.get("plateau", {}).get("has_plateau") is True
    ):
        return "FINANCIAL_FINALIST_CANDIDATE_REQUIRES_REVIEW"
    return "NO_FINANCIAL_FINALIST_YET"


def _compact_audit(audit: dict[str, Any] | None) -> dict[str, Any] | None:
    if audit is None:
        return None
    return {
        "status": audit["status"],
        "instrument_count": audit["instrument_count"],
        "common_history_universe": audit["common_history_universe"],
        "point_in_time_universe": audit["point_in_time_universe"],
    }


def _compact_benchmarks(benchmarks: dict[str, Any] | None) -> dict[str, Any] | None:
    if benchmarks is None:
        return None
    return {
        "status": benchmarks["status"],
        "benchmark_family_count": benchmarks["benchmark_family_count"],
        "result_count": benchmarks["result_count"],
    }


def _compact_grid(grid: dict[str, Any] | None) -> dict[str, Any] | None:
    if grid is None:
        return None
    return {
        "status": grid["status"],
        "config_count": grid["config_count"],
        "positive_config_count": grid["positive_config_count"],
        "plateau": grid["plateau"],
    }


def _compact_walk_forward(walk_forward: dict[str, Any] | None) -> dict[str, Any] | None:
    if walk_forward is None:
        return None
    return {
        "status": walk_forward["status"],
        "fold_count": walk_forward["fold_count"],
        "positive_test_folds": walk_forward["positive_test_folds"],
        "median_test_PF": walk_forward["median_test_PF"],
        "worst_test_PF": walk_forward["worst_test_PF"],
    }


def _corporate_action_counts(project_root: Path) -> dict[str, dict[str, int]]:
    path = project_root / "data" / "corporate_actions" / "corporate_actions.parquet"
    counts: dict[str, dict[str, int]] = {}
    for record in read_parquet_records(path):
        key = str(record["con_id"])
        bucket = counts.setdefault(key, {"corporate_action_count": 0, "dividend_count": 0, "split_count": 0})
        bucket["corporate_action_count"] += 1
        event_type = str(record.get("event_type") or "")
        if "DIVIDEND" in event_type or "DISTRIBUTION" in event_type:
            bucket["dividend_count"] += 1
        if "SPLIT" in event_type:
            bucket["split_count"] += 1
    return counts


def _fx_forward_fill_by_currency(project_root: Path) -> dict[str, int]:
    path = project_root / "data" / "fx" / "fx_daily.parquet"
    maximums: dict[str, int] = {}
    for record in read_parquet_records(path):
        currency = str(record["base_currency"])
        maximums[currency] = max(maximums.get(currency, 0), int(record.get("forward_fill_age") or 0))
    return maximums


def _max_missing_sequence(dates: list[str]) -> int:
    parsed = [date.fromisoformat(day) for day in dates]
    max_gap = 0
    for previous, current in zip(parsed, parsed[1:], strict=False):
        max_gap = max(max_gap, (current - previous).days - 1)
    return max_gap


def _unique_value(records: list[dict[str, Any]], field: str) -> str:
    values = sorted({str(record[field]) for record in records})
    return values[0] if len(values) == 1 else "MIXED"


def _cash_key(metadata: dict[str, dict[str, Any]]) -> str | None:
    for key, meta in metadata.items():
        if meta["sleeve"] == "cash":
            return key
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")


def phase6_freeze(project_root: Path) -> dict[str, Any]:
    layout = Phase6Layout.from_project_root(project_root)
    status = phase6_status(project_root)
    source_paths = (
        "main.py",
        "src/stocks/research/phase6.py",
        "tests/test_phase6_research.py",
    )
    artifact_paths = (
        layout.dataset_audit_json,
        layout.benchmarks_json,
        layout.strategy_grid_json,
        layout.walk_forward_json,
        layout.phase6_status_json,
    )
    freeze = {
        "schema": "phase6_freeze_v1",
        "contract_id": "PHASE6_BASELINES_AND_STRATEGY_EVIDENCE_V1",
        "freeze_status": "PHASE6_BASELINES_AND_STRATEGY_EVIDENCE_FROZEN_GO"
        if status["status"] == "PHASE6_BASELINES_AND_STRATEGY_EVIDENCE_GO"
        else "NO_GO",
        "generated_at": utc_now_iso(),
        "source_hashes": {path: sha256_file(project_root / path) for path in source_paths},
        "artifact_hashes": {str(path): sha256_file(path) for path in artifact_paths},
        "phase6_status": status["status"],
        "financial_calls": _zero_financial_calls(),
    }
    _write_json(layout.output_dir / "phase6-freeze.json", freeze)
    return freeze


def _zero_financial_calls() -> dict[str, int]:
    return {"place_order": 0, "cancel_order": 0, "global_cancel": 0}
